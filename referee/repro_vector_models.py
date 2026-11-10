"""Defect reproduction (judge, 2026-11-10): a segment built by the data plane from encoder-made vectors records model
"given", so a neural query on it raises UnsupportedQuery; engine.api.build_index on the same catalogue answers it.
Run: S3_SEARCH_SAMPLE=data/esci-sample python -m referee.repro_vector_models"""
import copy, json, tempfile, traceback
from pathlib import Path
import numpy as np
import engine.api as api
from dataplane import indexer, intake
from dataplane.kv import Store
from referee import loader

ds = loader.load_dataset("esci-sample")
cat = ds.catalogue.iloc[:3000].reset_index(drop=True)
m = copy.deepcopy(ds.mapping)
m["properties"]["embedding"] = {"type": "knn_vector", "dimension": 128,
                                "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}}
m["_meta"]["engine"]["embeddings"] = {"embedding": {"model": "tfidf-svd-128",
                                                    "inputs": ["product_title", "product_brand", "product_color"]}}
w = Path(tempfile.mkdtemp())
cat.to_parquet(w / "c.parquet", index=False)
st = Store(w / "s.sqlite", m)
try:
    rep = intake.ingest(w / "c.parquet", m, st, w / "rej.jsonl")
    print("ingest:", rep)
    seg = indexer.build(st, m, w / "seg")
    man = json.loads((seg / "manifest.json").read_text())
    print("segment vector_fields:", man.get("vector_fields"))
    idx = api.Index.from_segment(seg)
    v = np.ones(128, dtype=float).tolist()
    r = idx.search({"query": {"knn": {"embedding": {"vector": v, "k": 5}}}, "size": 5})
    print("knn ok:", [h["_id"] for h in r["hits"]["hits"]])
except Exception:
    traceback.print_exc()
print("store's encoder record:", {k: st.meta(k) for k in ("encoders", "encoder", "vector_models")})
for name, index in (("from_segment (data plane)", idx), ("engine.api.build_index (direct)", api.build_index(cat, m))):
    try:
        r = index.search({"query": {"neural": {"embedding": {"query_text": "red shoes", "k": 5}}}, "size": 5})
        print(name, "neural ok:", [h["_id"] for h in r["hits"]["hits"]])
    except Exception as e:
        print(name, "neural FAILED:", type(e).__name__, str(e)[:200])
