from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dataplane.intake import (REASONS, HashingEncoder, Rejection, apply_changes, enrich, ingest, read_batch,
                              validate)
from dataplane.mapping import InvalidMapping, parse_mapping


# ---- the mapping --------------------------------------------------------------------------------------------------

def test_mapping_parses_the_interface_example(mapping):
    m = parse_mapping(mapping)
    assert m.id_field == "product_id" and m.hot == ("price", "stock")
    assert m.vector_fields == ["embedding"] and m.fields["embedding"].dim == 16
    assert m.fields["product_brand"].subfields == (("keyword", "keyword"),)


@pytest.mark.parametrize("edit, path", [
    (lambda m: m["properties"]["price"].update(type="double"), "properties.price.type"),
    (lambda m: m["properties"]["embedding"].pop("dimension"), "properties.embedding.dimension"),
    (lambda m: m["_meta"]["engine"].update(hot=["category"]), "_meta.engine.hot[0]"),
    (lambda m: m["_meta"]["engine"].update(hot=["nope"]), "_meta.engine.hot[0]"),
    (lambda m: m["_meta"]["engine"]["embeddings"]["embedding"].update(inputs=["price"]),
     "_meta.engine.embeddings.embedding.inputs[0]"),
    (lambda m: m.update(properties={}), "properties"),
])
def test_bad_mappings_name_their_path(mapping, edit, path):
    edit(mapping)
    with pytest.raises(InvalidMapping) as e:
        parse_mapping(mapping)
    assert e.value.path == path


# ---- validation ----------------------------------------------------------------------------------------------------

def good(**kw):
    d = {"product_id": "P1", "product_title": "red shoe", "price": "9.5", "stock": 3.0, "category": "shoe"}
    d.update(kw)
    return d


def test_validate_coerces(mapping):
    d = validate(good(), mapping)
    assert d == {"product_id": "P1", "product_title": "red shoe", "price": 9.5, "stock": 3, "category": "shoe"}
    assert validate(good(product_id=17), mapping)["product_id"] == "17"
    v = validate(good(embedding=[3.0] + [0.0] * 14 + [4.0]), mapping)["embedding"]
    assert v.dtype == np.float32 and np.isclose(v[0], 0.6) and np.isclose(v[-1], 0.8)
    # missing values are dropped, unmapped ones kept (dynamic is not strict)
    d = validate(good(stock=None, price=float("nan"), extra={"a": np.int64(2)}), mapping)
    assert "stock" not in d and "price" not in d and d["extra"] == {"a": 2}


@pytest.mark.parametrize("doc, reason, field", [
    ({"product_title": "x"}, "missing_id", "product_id"),
    (good(product_id=""), "missing_id", "product_id"),
    (good(product_id="x" * 513), "too_long", "product_id"),
    (good(product_id=True), "bad_type", "product_id"),
    (good(price="cheap"), "bad_type", "price"),
    (good(price=True), "bad_type", "price"),
    (good(price=float("inf")), "bad_type", "price"),
    (good(stock=1.5), "bad_type", "stock"),
    (good(stock=2 ** 31), "bad_type", "stock"),
    (good(category=["a", "b"]), "bad_type", "category"),
    (good(embedding=[1.0, 2.0]), "dims_mismatch", "embedding"),
    (good(embedding=[0.0] * 16), "bad_type", "embedding"),
    (good(embedding="nope"), "bad_type", "embedding"),
])
def test_validate_rejects_with_closed_reasons(mapping, doc, reason, field):
    r = validate(doc, mapping)
    assert isinstance(r, Rejection) and (r.reason, r.field) == (reason, field)
    assert r.reason in REASONS


def test_strict_mapping_rejects_unknown_fields(mapping):
    mapping["dynamic"] = "strict"
    r = validate(good(colour="red"), mapping)
    assert (r.reason, r.field) == ("unknown_field", "colour")


def test_keyword_over_ignore_above_is_kept(mapping):
    assert validate(good(product_brand="b" * 300), mapping)["product_brand"] == "b" * 300


# ---- enrichment ----------------------------------------------------------------------------------------------------

def test_hashing_encoder_is_deterministic_and_normalised(mapping):
    a = enrich([validate(good(), mapping)], mapping)[0]["embedding"]
    b = enrich([validate(good(), mapping)], mapping, encoder=HashingEncoder(16))[0]["embedding"]
    assert a.shape == (16,) and np.isclose(np.linalg.norm(a), 1.0) and np.array_equal(a, b)
    other = enrich([validate(good(product_title="blue lamp"), mapping)], mapping)[0]["embedding"]
    assert not np.array_equal(a, other)


def test_enrich_keeps_given_vectors_and_uses_the_hook(mapping):
    class Ones:
        name, dim = "ones", 16

        def encode(self, texts):
            return np.ones((len(texts), 16), dtype=np.float32)

    given = validate(good(embedding=[1.0] + [0.0] * 15), mapping)
    fresh = validate(good(product_id="P2"), mapping)
    empty = {"product_id": "P3", "price": 1.0}
    enrich([given, fresh, empty], mapping, encoder=Ones())
    assert given["embedding"][0] == 1.0
    assert np.allclose(fresh["embedding"], 0.25)
    assert "embedding" not in empty
    with pytest.raises(ValueError):
        enrich([validate(good(), mapping)], mapping, encoder=HashingEncoder(8))


# ---- reading and ingest --------------------------------------------------------------------------------------------

def test_read_batch_parquet_and_jsonl(tmp_path, catalogue):
    pq_path = tmp_path / "b.parquet"
    catalogue.head(5).to_parquet(pq_path)
    rows = list(read_batch(pq_path))
    assert [r["product_id"] for r in rows] == list(catalogue.product_id[:5])
    jl = tmp_path / "b.jsonl"
    jl.write_text('{"product_id": "A"}\n\nnot json\n[1, 2]\n{"product_id": "B"}\n')
    out = list(read_batch(jl))
    assert [type(o).__name__ for o in out] == ["dict", "Rejection", "Rejection", "dict"]
    assert out[1].reason == "malformed" and out[1].raw == "not json" and "line 3" in out[1].detail


def test_ingest_writes_store_and_rejected(tmp_path, store, catalogue, mapping):
    rows = catalogue.head(20).to_dict("records")
    rows[3]["price"] = "cheap"               # parquet needs one type per column, so use JSONL
    rows.append(dict(rows[0]))               # a duplicate id
    rows.append({"product_title": "no id"})
    path = tmp_path / "batch.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "{broken\n")
    rep = ingest(path, mapping, store, tmp_path)
    assert (rep.accepted, rep.rejected) == (19, 4)
    assert rep.reasons == {"bad_type": 1, "duplicate_id_in_batch": 1, "missing_id": 1, "malformed": 1}
    assert (rep.first_offset, rep.last_offset) == (1, 19) and store.head() == 19
    lines = [json.loads(l) for l in (tmp_path / "rejected.jsonl").read_text().splitlines()]
    assert {l["rejection"]["reason"] for l in lines} == set(rep.reasons)
    assert all(set(l) == {"record", "rejection"} for l in lines)
    doc = store.get(["P00000"])[0]
    assert len(doc["embedding"]) == 16 and isinstance(doc["embedding"][0], float)
    assert store.meta("encoder.embedding") == "hash-16"


def test_ingest_parquet_round_trip(tmp_path, store, catalogue, mapping):
    path = tmp_path / "batch.parquet"
    catalogue.to_parquet(path)
    rep = ingest(path, mapping, store, tmp_path / "rej.jsonl", chunk=64)
    assert rep.accepted == len(catalogue) and rep.rejected == 0
    snap = store.snapshot()
    assert list(snap.product_id) == list(catalogue.product_id)
    assert np.allclose(snap.price.to_numpy(), catalogue.price.to_numpy())


def test_ingest_refuses_to_mix_encoders(tmp_path, store, catalogue, mapping):
    path = tmp_path / "batch.parquet"
    catalogue.head(3).to_parquet(path)
    ingest(path, mapping, store, tmp_path / "rej.jsonl")

    class Other(HashingEncoder):
        def __init__(self):
            super().__init__(16)
            self.name = "other"

    with pytest.raises(ValueError, match="mix"):
        ingest(path, mapping, store, tmp_path / "rej.jsonl", encoder=Other())


def test_change_feed(tmp_path, store, catalogue, mapping):
    from dataplane.tests.conftest import load
    load(store, catalogue.head(3))
    rep = apply_changes([
        {"op": "patch", "_id": "P00000", "fields": {"price": "1.25"}},
        {"op": "patch", "_id": "P00001", "fields": {"category": "x"}},       # not hot
        {"op": "patch", "_id": "NOPE", "fields": {"stock": 1}},              # no such key
        {"op": "upsert", "_id": "NEW", "doc": {"product_title": "new kettle", "price": 3}},
        {"op": "patch", "_id": "NEW", "fields": {"stock": 9}},               # live within the batch
        {"op": "delete", "_id": "P00002"},
        {"op": "patch", "_id": "P00002", "fields": {"stock": 1}},            # deleted within the batch
        {"op": "merge", "_id": "P00001"},
        {"op": "patch", "fields": {"stock": 1}},
        {"op": "patch", "_id": "P00001", "fields": {"stock": "many"}},
    ], mapping, store, rejected=tmp_path / "rej.jsonl")
    assert rep.accepted == 4 and (rep.first_offset, rep.last_offset) == (4, 7)
    assert [r.reason for _, r in rep.rejected] == ["not_hot", "no_such_key", "no_such_key", "bad_op",
                                                   "missing_id", "bad_type"]
    new, p0, p2 = store.get(["NEW", "P00000", "P00002"])
    assert new["stock"] == 9 and len(new["embedding"]) == 16 and p0["price"] == 1.25 and p2 is None
    assert len((tmp_path / "rej.jsonl").read_text().splitlines()) == 6
