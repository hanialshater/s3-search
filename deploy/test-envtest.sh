#!/usr/bin/env bash
# Run the operator's tests against a real kube-apiserver and etcd (envtest style). Outside `python -m referee fast`.
#   deploy/test-envtest.sh [pytest args]
# Binaries: S3_ENVTEST_BIN (default /tmp/claude-0/envtest/controller-tools/envtest) must hold `etcd` and `kube-apiserver`;
# `openssl` and PyYAML are needed (service-account key, CRD loading). Nothing here touches a cloud.
set -euo pipefail
cd "$(dirname "$0")/.."
export S3_ENVTEST=1
export S3_ENVTEST_BIN="${S3_ENVTEST_BIN:-/tmp/claude-0/envtest/controller-tools/envtest}"
for b in etcd kube-apiserver; do
  [ -x "$S3_ENVTEST_BIN/$b" ] || { echo "missing $S3_ENVTEST_BIN/$b (set S3_ENVTEST_BIN)" >&2; exit 2; }
done
exec python3 -m pytest deploy/tests/test_envtest.py -q -p no:cacheprovider "$@"
