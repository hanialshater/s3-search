"""Build ESCI-US on CPU once (single process) and save what the study needs: postings, norms, idf, ids, and each
query's planned terms (test-query texts: the 8,956; and a fit-split sample with its judgments). Development only."""
import copy, json, sys, time, pickle
import numpy as np
REPO = str(__import__("pathlib").Path(__file__).resolve().parents[4])   # the repository root
sys.path.insert(0, REPO)
from referee import loader
from engine import api
from engine.plan.plan import plan as make_plan
OUT = sys.argv[1]
t0 = time.time()
ds = loader.load_dataset("esci-us")
tmpl = json.load(open(f"{REPO}/referee/requests/bm25f-tuned.json"))
FIELDS = tmpl["query"]["multi_match"]["fields"]
m = copy.deepcopy(ds.mapping)
m["_meta"]["engine"]["bm25"] = json.load(open(f"{REPO}/referee/requests/bm25f-tuned.bm25.json"))
print("loaded", len(ds.catalogue), round(time.time() - t0), flush=True)
import os, gc
from pathlib import Path
from engine.build import build_segment
from engine.segment import write_segment
SEG = Path(OUT) / "seg"
if not (SEG / "manifest.json").exists() and not any(SEG.glob("*")):
    seg = build_segment(ds.catalogue, m)
    write_segment(seg, SEG)
    del seg
    gc.collect()
    print("segment written", round(time.time() - t0), flush=True)
os.environ["S3_SEARCH_DEVICE_BUDGET"] = str(16 << 30)
index = api.Index.from_segment(SEG, device="cpu")
st = index._st
part = st.parts[0]
print("built", st.N, "postings", int(part.host_term_ptr[-1]), round(time.time() - t0), flush=True)
F = len(st.text_fields)
bb = np.array([st.mapping.b_of(f) for f in st.text_fields], np.float32)

def planned(q):
    r = make_plan({"query": {"multi_match": {"query": q, "fields": FIELDS}}, "size": 2000}, st).root
    w = np.zeros(F, np.float32)
    for f, wt in r.fields:
        w[st.field_index[f]] = wt
    return dict(terms=[part.vocab[t] for t in r.terms if t in part.vocab], msm=int(r.msm), boost=float(r.boost),
                idf=r.idf, w=w, n_terms=len(r.terms), kind=type(r).__name__)

test = [str(t) for t in ds.test_queries["query"]]
tp = [planned(q) for q in test]
fit = ds.fit
qids = np.sort(fit["query_id"].unique())
pick = qids[:: max(1, len(qids) // 2000)][:2000]
fq = fit[fit["query_id"].isin(set(pick))]
fit_queries = fq.drop_duplicates("query_id")[["query_id", "query"]].values.tolist()
fit_judg = {}
for qid, pid, lab in fq[["query_id", "product_id", "esci_label"]].itertuples(index=False):
    fit_judg.setdefault(qid, {})[pid] = lab
fp = [planned(str(q)) for _, q in fit_queries]
print("planned", len(tp), len(fp), "kinds", {k: sum(p["kind"] == k for p in tp) for k in set(p["kind"] for p in tp)},
      round(time.time() - t0), flush=True)
np.savez(OUT + "/index.npz", term_ptr=np.asarray(part.host_term_ptr), doc_ids=np.asarray(part.doc_ids),
         tf=np.asarray(part.host_tf), rel_len=np.asarray(st.host_rel_len), idf=np.asarray(part.host_idf),
         live=np.asarray(st.live), bb=bb, k1=np.float32(st.mapping.k1))
ids = list(st.ids)
pickle.dump({"test": tp, "test_texts": test, "fit": fp, "fit_queries": fit_queries, "fit_judg": fit_judg, "ids": ids,
             "fields": list(st.text_fields), "n_fit_queries_total": int(len(qids))}, open(OUT + "/plans.pkl", "wb"))
# a few engine answers to hold the study's exact scores to
chk = {}
for i in range(0, 8956, 179)[:50]:
    r = index.search_batch([{"query": {"multi_match": {"query": test[i], "fields": FIELDS}}, "size": 2000}],
                           hits="columns", ids="bytes")[0]
    c = r["hits"]["columns"]
    chk[i] = (r["hits"]["total"], [x.decode() for x in c["_id"].tolist()], np.asarray(c["_score"], np.float32))
pickle.dump(chk, open(OUT + "/engine_check.pkl", "wb"))
print("done", round(time.time() - t0), flush=True)
