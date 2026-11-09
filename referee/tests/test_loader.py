"""The loader's boundary on a tiny synthetic ESCI: the fit split holds no test rows, test queries carry no labels."""
import pandas as pd
import pytest

from referee import loader


@pytest.fixture()
def esci(tmp_path):
    products = pd.DataFrame({
        "product_id": ["p1", "p2", "p3"], "product_title": ["red shoe", "blue sock", "hat"],
        "product_brand": ["a", "b", "c"], "product_color": ["red", "blue", None],
        "product_bullet_point": [None, None, None], "product_description": [None, None, None],
        "product_locale": ["us", "us", "jp"]})
    examples = pd.DataFrame({
        "query_id": [1, 1, 2, 3, 4], "query": ["shoe", "shoe", "sock", "hat", "big"],
        "product_id": ["p1", "p2", "p2", "p3", "p1"], "esci_label": ["E", "I", "E", "E", "S"],
        "split": ["train", "train", "test", "test", "test"], "product_locale": ["us", "us", "us", "jp", "us"],
        "small_version": [1, 1, 1, 1, 0]})
    # Real files are large; the size check in fetch() is not exercised here.
    products.to_parquet(tmp_path / loader.PRODUCTS)
    examples.to_parquet(tmp_path / loader.EXAMPLES)
    return tmp_path


def test_fit_split_has_no_test_rows(esci):
    fit = loader.load_fit(esci)
    assert set(fit["query_id"]) == {1} and "split" not in fit.columns


def test_test_queries_carry_no_labels(esci):
    q = loader.load_test_queries(esci)
    assert list(q.columns) == ["query_id", "query"]
    assert list(q["query_id"]) == [2]  # query 3 is jp, query 4 is outside the reduced set


def test_catalogue_is_one_locale(esci):
    assert set(loader.load_catalogue(esci)["product_id"]) == {"p1", "p2"}


def test_scorer_judgments(esci):
    assert loader._load_test_judgments(esci) == {2: {"p2": "E"}}
