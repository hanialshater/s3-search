"""Dense search, RRF fusion, the feature gather and the ranker, against independent numpy implementations."""
import copy

import numpy as np
import pandas as pd
import pytest

from engine.api import Index
from engine.encode import TfidfSvdEncoder, doc_texts
from engine.rank.gbdt import GBDT
from engine.rank.train import build_training, select_rows, train_ranker

INPUTS = ["product_title", "product_brand", "product_color"]


@pytest.fixture(scope="module")
def emb(segment):
    return np.asarray(segment.vectors["embedding"], np.float64)


def qvec(index, text):
    return index._st.encoders["embedding"].encode([text])[0]


def test_encoder_deterministic_and_normalised(catalogue):
    a = TfidfSvdEncoder(16).fit(doc_texts(catalogue, INPUTS))
    b = TfidfSvdEncoder(16).fit(doc_texts(catalogue, INPUTS))
    va, vb = a.encode(["black running shoes"]), b.encode(["black running shoes"])
    assert np.array_equal(va, vb)
    assert abs(np.linalg.norm(va) - 1) < 1e-5
    d = a.encode(doc_texts(catalogue, INPUTS))
    sims = d @ va[0]
    top = doc_texts(catalogue, INPUTS)[int(np.argmax(sims))]
    assert any(w in top for w in ("black", "running", "shoes"))
    assert np.allclose(np.linalg.norm(d, axis=1)[np.linalg.norm(d, axis=1) > 0], 1, atol=1e-5)


def test_knn_exact_with_filter(index, catalogue, emb):
    v = qvec(index, "blue cotton shirt")
    sims = (emb @ v.astype(np.float64)).astype(np.float32)
    want = np.lexsort((np.arange(len(sims)), -sims))[:10]
    r = index.search({"query": {"knn": {"embedding": {"vector": v.tolist(), "k": 10}}}, "size": 10})
    assert [h["_id"] for h in r["hits"]["hits"]] == catalogue["product_id"].iloc[want].tolist()
    assert np.allclose([h["_score"] for h in r["hits"]["hits"]], (1 + sims[want]) / 2, rtol=1e-5)
    assert r["hits"]["total"] == 10
    ok = (catalogue["category"] == "socks").to_numpy()
    want_f = [i for i in np.lexsort((np.arange(len(sims)), -sims)) if ok[i]][:5]
    r = index.search({"query": {"knn": {"embedding": {"vector": v.tolist(), "k": 5,
                                                      "filter": {"term": {"category": "socks"}}}}}})
    assert [h["_id"] for h in r["hits"]["hits"]] == catalogue["product_id"].iloc[want_f].tolist()


def host_rrf(lists, k=60):
    acc = {}
    for ids in lists:
        for r, d in enumerate(ids, start=1):
            acc[d] = acc.get(d, 0.0) + 1.0 / (k + r)
    return sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))


def test_rrf_against_host(index, catalogue):
    text = "black shoes"
    v = qvec(index, text).tolist()
    lex = {"match": {"product_title": text}}
    knn = {"knn": {"embedding": {"vector": v, "k": 25}}}
    pos = {p: i for i, p in enumerate(catalogue["product_id"])}
    l1 = [pos[h["_id"]] for h in index.search({"query": lex, "size": 30})["hits"]["hits"]]
    l2 = [pos[h["_id"]] for h in index.search({"query": knn, "size": 25})["hits"]["hits"]]
    want = host_rrf([l1, l2])
    r = index.search({"query": {"hybrid": {"queries": [lex, knn]}}, "rank": {"rrf": {"window_size": 30}}, "size": 100})
    got = [(pos[h["_id"]], h["_score"]) for h in r["hits"]["hits"]]
    assert [d for d, _ in got] == [d for d, _ in want]
    assert np.allclose([s for _, s in got], [s for _, s in want], rtol=1e-6)
    assert r["hits"]["total"] == len(want)


def test_features_against_host(index, catalogue, segment):
    text = "nike black running shoes"
    v = qvec(index, text)
    req = {"query": {"hybrid": {"queries": [{"match": {"product_title": text}},
                                            {"knn": {"embedding": {"vector": v.tolist(), "k": 40}}}]}}}
    ids, X = index.candidate_features(req, 50)
    names = index.feature_names()
    pos = {p: i for i, p in enumerate(catalogue["product_id"])}
    lex_full = np.zeros(len(catalogue))
    for h in index.search({"query": {"match": {"product_title": text}}, "size": 200})["hits"]["hits"]:
        lex_full[pos[h["_id"]]] = h["_score"]
    lex_list = [pos[h["_id"]] for h in index.search({"query": {"match": {"product_title": text}},
                                                    "size": 200})["hits"]["hits"]]
    sims = np.asarray(segment.vectors["embedding"]) @ (v / np.linalg.norm(v))
    dense_list = list(np.lexsort((np.arange(len(sims)), -sims))[:40])
    col = {n: X[:, i] for i, n in enumerate(names)}
    for j, pid in enumerate(ids):
        d = pos[pid]
        assert np.isclose(col["lex_score"][j], lex_full[d], rtol=1e-5)
        assert np.isclose(col["lex_norm"][j], lex_full[d] / lex_full.max(), rtol=1e-5)
        assert np.isclose(col["dense_sim"][j], sims[d], rtol=1e-4, atol=1e-6)
        lr = lex_list.index(d) + 1 if d in lex_list else 0
        dr = dense_list.index(d) + 1 if d in dense_list else 0
        assert (np.isnan(col["lex_rank_log"][j]) if lr == 0 else np.isclose(col["lex_rank_log"][j], np.log1p(lr)))
        assert (np.isnan(col["dense_rank_log"][j]) if dr == 0 else np.isclose(col["dense_rank_log"][j], np.log1p(dr)))
        assert col["in_both"][j] == float(lr > 0 and dr > 0)
        assert np.isclose(col["rrf"][j], (1 / (60 + lr) if lr else 0) + (1 / (60 + dr) if dr else 0), rtol=1e-6)
        brand = catalogue["product_brand"].iloc[d]
        assert col["match_product_brand.keyword"][j] == float(brand == "Nike")
        assert col["q_len"][j] == 4


def test_select_rows_never_takes_top_unjudged():
    rng = np.random.default_rng(0)
    cand = np.arange(1000)
    judged = np.zeros(1000, bool)
    judged[[3, 50, 400]] = True
    picks = [set(select_rows(cand, judged, rng, 20)) for _ in range(30)]
    assert all({3, 50, 400} <= p and len(p) == 23 for p in picks)
    top_unjudged = {0, 1, 2, 4, 5}
    share = np.mean([len(p & top_unjudged) for p in picks]) / 5
    assert share < 0.2                    # uniform: about 20 / 997 per row, not the top rows every time


def test_gbdt_learns_and_round_trips():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4)).astype(np.float32)
    X[rng.random(2000) < 0.1, 2] = np.nan
    y = (X[:, 0] > 0).astype(float) * 2 + np.nan_to_num(X[:, 2]) * 0.5
    m = GBDT(n_rounds=60, max_depth=3).fit(X, y)
    p = m.predict(X)
    assert np.corrcoef(p, y)[0, 1] > 0.95
    assert np.array_equal(GBDT.from_dict(m.to_dict()).predict(X), p)


def test_train_and_rescore(segment, catalogue):
    ix = Index(copy.deepcopy(segment), "cpu")
    rng = np.random.default_rng(3)
    rows = []
    for q, text in enumerate(["black shoes", "wool socks", "blue jeans", "nike running", "leather jacket"]):
        for pid in rng.choice(catalogue["product_id"], 15, replace=False):
            title = catalogue.set_index("product_id").loc[pid, "product_title"]
            lab = "E" if any(w in title for w in text.split()) else rng.choice(["S", "C", "I"])
            rows.append({"query_id": q, "query": text, "product_id": pid, "esci_label": lab})
    fit = pd.DataFrame(rows)

    def request_for(text):
        return {"query": {"hybrid": {"queries": [{"multi_match": {"query": text}},
                                                 {"neural": {"embedding": {"query_text": text, "k": 50}}}]}}}

    X, y, g = build_training(ix, fit, request_for, window=200, n_random=10)
    assert X.shape[1] == len(ix.feature_names()) and len(np.unique(g)) >= 3
    model = train_ranker(ix, fit, request_for, window=200, n_random=10, n_rounds=20, min_leaf=5)
    ix.add_ranker("gbdt", model)
    req = {**request_for("black shoes"), "rescore": {"window_size": 30, "s3_ranker": {"model": "gbdt"}}, "size": 40}
    r = ix.search(req)
    hits = r["hits"]["hits"]
    top = [h for h in hits if h["sort"][0] == 1]
    assert len(top) == 30 and hits[:30] == top                       # the window first, ordered by the ranker
    assert [h["_score"] for h in top] == sorted((h["_score"] for h in top), reverse=True)
    ids, Xc = ix.candidate_features(request_for("black shoes"), 30)
    pred = model.predict(Xc)
    order = [ids[i] for i in np.lexsort((np.arange(len(ids)), -pred))]
    pos = {p: i for i, p in enumerate(catalogue["product_id"])}
    assert [h["_id"] for h in top] == sorted(order, key=lambda p: (-pred[ids.index(p)], pos[p]))
    nxt = ix.search({**req, "size": 10, "search_after": hits[29]["sort"]})["hits"]["hits"]
    assert [h["_id"] for h in nxt] == [h["_id"] for h in hits[30:40]]
