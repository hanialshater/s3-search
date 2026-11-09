"""Hand-computed checks for the referee's metrics. Every expected value is worked out in the comment beside it."""
import math

import pytest

from referee.metrics import GAIN, score_query, score_run


def d(rank):  # discount for a 0-based rank
    return 1.0 / math.log2(rank + 2)


def test_perfect_order_scores_one():
    j = {"a": "E", "b": "S", "c": "I"}
    m = score_query(["a", "b", "c"], j)
    assert m["ndcg@10"] == pytest.approx(1.0)
    assert m["exact_recall@100"] == 1.0


def test_unjudged_items_count_as_zero_gain():
    # ranked: x (unjudged), a (E). DCG = 0*d(0) + 1*d(1) = 1/log2(3). Ideal: a first = 1.
    m = score_query(["x", "a"], {"a": "E"})
    assert m["ndcg@10"] == pytest.approx(1 / math.log2(3))


def test_esci_gains_and_ideal_over_all_judged():
    # judged: a E, b S, c C. Ranked: c, b, a.
    # DCG = 0.01*d0 + 0.1*d1 + 1.0*d2 ; IDCG = 1.0*d0 + 0.1*d1 + 0.01*d2
    j = {"a": "E", "b": "S", "c": "C"}
    dcg = 0.01 * d(0) + 0.1 * d(1) + 1.0 * d(2)
    idcg = 1.0 * d(0) + 0.1 * d(1) + 0.01 * d(2)
    assert score_query(["c", "b", "a"], j)["ndcg@10"] == pytest.approx(dcg / idcg)


def test_missed_relevant_products_still_count_against():
    # Only b (S) retrieved; a (E) is judged but missed. DCG = 0.1*d0 ; IDCG = 1*d0 + 0.1*d1.
    j = {"a": "E", "b": "S"}
    assert score_query(["b"], j)["ndcg@10"] == pytest.approx(0.1 / (1 + 0.1 * d(1)))


def test_cutoff_at_ten():
    # The Exact product sits at rank 10 (0-based), outside the top 10: nDCG@10 = 0; recall@100 = 1.
    ranked = [f"x{i}" for i in range(10)] + ["a"]
    m = score_query(ranked, {"a": "E"})
    assert m["ndcg@10"] == 0.0 and m["exact_recall@100"] == 1.0


def test_exact_recall_and_queries_without_exact():
    m = score_query(["a", "x", "b"], {"a": "E", "b": "E", "c": "E", "s": "S"}, recall_ks=(2, 100))
    assert m["exact_recall@2"] == pytest.approx(1 / 3) and m["exact_recall@100"] == pytest.approx(2 / 3)
    assert score_query(["s"], {"s": "S"})["exact_recall@100"] is None  # no Exact product: left out of the mean


def test_padding_ids_are_ignored():
    assert score_query([-1, "a"], {"a": "E"})["ndcg@10"] == pytest.approx(1.0)


def test_all_irrelevant_query_is_left_out_of_the_mean():
    r = score_run({1: ["a"], 2: ["z"]}, {1: {"a": "E"}, 2: {"z": "I"}})
    assert r["mean"]["ndcg@10"] == pytest.approx(1.0) and r["mean"]["n_ndcg@10"] == 1


def test_a_missing_run_scores_zero_never_raises_the_mean():
    r = score_run({1: ["a"]}, {1: {"a": "E"}, 2: {"b": "E"}})
    assert r["mean"]["ndcg@10"] == pytest.approx(0.5)


def test_gains_are_the_esci_papers():
    assert GAIN == {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
