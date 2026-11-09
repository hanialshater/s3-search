"""The OpenSearch adapter without a server: its index body, the analyzer swap, and response conversion."""
import json

from referee import loader, opensearch
from referee.analysis import EN_STOP


def test_index_body_defines_s3_english(monkeypatch):
    monkeypatch.delenv("S3_SEARCH_OS_ANALYZER", raising=False)
    body = opensearch.index_settings(loader.ESCI_MAPPING)
    an = body["settings"]["analysis"]
    assert an["analyzer"]["s3_english"]["filter"][-1] == "s3_porter"
    assert set(an["filter"]["s3_stop"]["stopwords"]) == EN_STOP and len(EN_STOP) == 33
    assert body["mappings"]["properties"]["product_title"]["analyzer"] == "s3_english"
    json.dumps(body)


def test_analyzer_swap_for_the_prototype(monkeypatch):
    monkeypatch.setenv("S3_SEARCH_OS_ANALYZER", "english")
    body = opensearch.index_settings(loader.ESCI_MAPPING)
    assert body["mappings"]["properties"]["product_title"]["analyzer"] == "english"
    assert loader.ESCI_MAPPING["properties"]["product_title"]["analyzer"] == "s3_english"  # not mutated


def test_response_conversion():
    resp = {"took": 3, "hits": {"total": {"value": 2, "relation": "eq"},
                                "hits": [{"_id": "a", "_score": 2.0}, {"_id": "b", "_score": 1.0, "sort": [1.0, 5]}]},
            "aggregations": {"c": {"buckets": [{"key": "x", "doc_count": 2}], "sum_other_doc_count": 0}}}
    out = opensearch._convert(resp, 4.0)
    assert out["hits"]["total"] == 2 and [h["_id"] for h in out["hits"]["hits"]] == ["a", "b"]
    assert out["aggregations"]["c"]["buckets"] == [{"key": "x", "doc_count": 2}] and out["took_ms"]["total"] == 4.0
