#!/usr/bin/env bash
# The end-to-end scenario, against whatever cluster kubectl points at (a kind cluster in CI; never a cloud cluster).
# Assumes the CPU image `s3-search:cpu` is already on the nodes (kind load docker-image).
#
#   1. install the CRD and the operator
#   2. apply the toy SearchIndex (2 replicas) and wait until it is Ready
#   3. query it through its Service from inside the cluster
#   4. roll a new segment (spec.reindexToken) under a 50 QPS replay: zero failed queries (S4 prediction 1)
#   5. roll back to the first segment (spec.segmentVersion) and time it
set -euo pipefail
cd "$(dirname "$0")/../.."
NS=${NS:-default}
IMAGE=${IMAGE:-s3-search:cpu}
QPS=${QPS:-50}
K="kubectl -n $NS"

step() { printf '\n== %s\n' "$*"; }
phase() { $K get searchindex toy -o jsonpath='{.status.phase}' 2>/dev/null || true; }
segment() { $K get searchindex toy -o jsonpath='{.status.segmentId}'; }
wait_phase() {  # wait_phase Ready 600
  local want=$1 limit=${2:-600} t=0
  until [ "$(phase)" = "$want" ]; do
    [ "$(phase)" = "Failed" ] && { echo "phase Failed: $($K get searchindex toy -o jsonpath='{.status.message}')"; return 1; }
    t=$((t + 2)); [ $t -gt "$limit" ] && { echo "timed out waiting for phase $want (now: $(phase))"; return 1; }
    sleep 2
  done
}
cluster_run() {  # cluster_run NAME ARGS... : a one-off pod with the image; its exit status is the result
  local name=$1; shift
  $K delete pod "$name" --ignore-not-found >/dev/null
  $K run "$name" --image="$IMAGE" --image-pull-policy=IfNotPresent --restart=Never --command -- "$@" >/dev/null
  $K wait --for=jsonpath='{.status.phase}'=Succeeded "pod/$name" --timeout=300s >/dev/null || {
    $K logs "$name" || true; return 1; }
  $K logs "$name"
}

step "install the CRD and the operator"
kubectl apply -f deploy/crd/searchindex.yaml
kubectl wait --for=condition=Established crd/searchindexes.search.s3.dev --timeout=60s
kubectl apply -f deploy/manifests/operator.yaml
kubectl -n s3-search-system rollout status deployment/s3-operator --timeout=180s

step "create the toy index"
$K create configmap toy-mapping --from-file=mapping.json=deploy/samples/toy-mapping.json --dry-run=client -o yaml | $K apply -f -
$K apply -f deploy/samples/toy-searchindex.yaml
wait_phase Ready 600
$K get searchindex toy
SEG1=$(segment)
echo "serving $SEG1"

step "query through the Service"
cluster_run smoke python -m deploy.e2e.replay --url "http://toy.$NS.svc" --index toy --qps 10 --duration 5

step "roll a new segment under a ${QPS} QPS replay"
$K delete pod replay --ignore-not-found >/dev/null
$K run replay --image="$IMAGE" --image-pull-policy=IfNotPresent --restart=Never --command -- \
  python -m deploy.e2e.replay --url "http://toy.$NS.svc" --index toy --qps "$QPS" --duration 240 >/dev/null
sleep 10
$K patch searchindex toy --type merge -p '{"spec":{"reindexToken":"roll-1"}}'
until [ "$(segment)" != "$SEG1" ] && [ "$(phase)" = "Ready" ]; do
  [ "$(phase)" = "Failed" ] && { echo "roll failed: $($K get searchindex toy -o jsonpath='{.status.message}')"; exit 1; }
  sleep 2
done
SEG2=$(segment)
echo "rolled $SEG1 -> $SEG2"
$K wait --for=jsonpath='{.status.phase}'=Succeeded pod/replay --timeout=300s
$K logs replay | tee /tmp/replay.json
python3 - <<'PY'
import json, sys
out = json.loads(open("/tmp/replay.json").read().strip().splitlines()[-1])
if out["failed"] != 0 or out["sent"] == 0:
    sys.exit(f"FAILED QUERIES DURING THE ROLL: {out}")
print("zero failed queries during the roll:", out["sent"], "sent")
PY

step "roll back to $SEG1"
START=$(date +%s)
$K patch searchindex toy --type merge -p "{\"spec\":{\"segmentVersion\":\"$SEG1\"}}"
until [ "$(segment)" = "$SEG1" ] && [ "$(phase)" = "Ready" ]; do sleep 1; done
echo "rollback to $SEG1 took $(( $(date +%s) - START )) s (pods rolling one at a time, maxUnavailable 0)"
cluster_run smoke2 python -m deploy.e2e.replay --url "http://toy.$NS.svc" --index toy --qps 10 --duration 5

step "ok"
