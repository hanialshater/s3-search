"""Build a zvec 0.8.1 collection from a catalogue (S6-18s). Development only.

One FTS field per text field the engine searches. Each uses tokenizer ``standard`` and filters ``["lowercase",
"stemmer"]``, zvec's documented choice for English (docs: concepts/fts-index). The document id is ``product_id``.
zvec writes in batches of at most 1,024, so the catalogue is inserted that way and then ``optimize()``-d.

    from colab.competitors.zvec_build import build
    build(catalogue_dataframe, "/content/zvec/esci", fields=[...])   # -> {"n_docs", "build_s"}
"""
from __future__ import annotations

import shutil
import time

TEXT_FIELDS = ("product_title", "product_brand", "product_color", "product_bullet_point", "product_description")
BATCH = 1024                                                    # zvec's maximum write batch


def build(catalogue, path: str, fields=TEXT_FIELDS, id_field: str = "product_id") -> dict:
    import zvec
    shutil.rmtree(path, ignore_errors=True)
    fts = zvec.FtsIndexParam(tokenizer_name="standard", filters=["lowercase", "stemmer"])
    schema = zvec.CollectionSchema(name="esci", fields=[
        zvec.FieldSchema(f, zvec.DataType.STRING, nullable=True, index_param=fts) for f in fields])
    col = zvec.create_and_open(path=path, schema=schema)
    t0 = time.time()
    ids = catalogue[id_field].astype(str).tolist()
    cols = {f: catalogue[f].fillna("").astype(str).tolist() for f in fields}
    for k in range(0, len(ids), BATCH):
        col.insert([zvec.Doc(id=ids[i], fields={f: cols[f][i] for f in fields}) for i in range(k, min(k + BATCH, len(ids)))])
    t_insert = time.time() - t0
    col.optimize()
    out = {"n_docs": len(ids), "insert_s": round(t_insert, 1), "build_s": round(time.time() - t0, 1),
           "fields": list(fields), "tokenizer": "standard", "filters": ["lowercase", "stemmer"],
           "zvec": getattr(zvec, "__version__", None)}
    col.close() if hasattr(col, "close") else None
    return out
