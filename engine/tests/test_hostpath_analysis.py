"""R3a: the analyser oracle. s3_hostpath's Rust analyser equals engine/analysis.py token for token.

Skipped where the extension is not built (it is built on a Colab CPU runtime, colab/hostpath_ci.py). Every word and
text below must give Python's exact tokens: the stems, the stopwords, the apostrophes, and Unicode lowercasing."""
import itertools
import os

import pytest

from engine import analysis
from referee import loader

s3_hostpath = pytest.importorskip("s3_hostpath")

SUFFIXES = ["", "s", "es", "ies", "sses", "ss", "ed", "eed", "ing", "ings", "ly", "ness", "ful", "fulness", "ation",
            "ational", "tional", "izer", "ization", "ator", "alli", "alism", "aliti", "ousli", "ousness", "iveness",
            "iviti", "biliti", "enci", "anci", "abli", "entli", "eli", "icate", "ative", "alize", "iciti", "ical",
            "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "sion", "tion", "ou",
            "ism", "ate", "iti", "ous", "ive", "ize", "e", "l", "ll", "y", "yy", "ay", "at", "bl", "iz", "bb", "dd"]
STEMS = ["", "a", "y", "yo", "sky", "fly", "cat", "hop", "fil", "fail", "agre", "plaster", "contro", "rel", "relat",
         "gener", "condit", "adjust", "depend", "communic", "triplic", "formal", "hope", "electr", "rational", "rat",
         "feud", "sensit", "probat", "dog", "bowl", "nois", "say", "play", "yell", "toy", "sayy", "boy", "eye", "0x",
         "a1b2", "zz", "q", "bbb", "abl"]


def _words_of_dataset(name):
    try:
        ds = loader.load_dataset(name)
    except Exception:  # noqa: BLE001
        return None, []
    texts = []
    for col in ds.catalogue.columns:
        if ds.catalogue[col].dtype == object:
            texts += [str(x) for x in ds.catalogue[col].dropna().tolist()[:40_000]]
    texts += [str(q) for q in list(ds.fit["query"]) + list(ds.test_queries["query"])]
    return ds, texts


def test_stem_equals_snowball_on_generated_words():
    words = {a + b for a, b in itertools.product(STEMS, SUFFIXES)} - {""}
    bad = [w for w in sorted(words) if s3_hostpath.stem(w) != analysis._stemmer.stemWord(w)]
    assert not bad, [(w, s3_hostpath.stem(w), analysis._stemmer.stemWord(w)) for w in bad[:20]]


@pytest.mark.parametrize("name", ["toy", "esci-sample"] + (["esci-us"] if os.environ.get("S3_ORACLE_FULL") else []))
def test_analyze_equals_python_on_every_text(name):
    ds, texts = _words_of_dataset(name)
    if ds is None:
        pytest.skip(f"{name} not available")
    words = sorted({w for t in texts for w in analysis._WORD.findall(t.lower())})
    bad_w = [w for w in words if s3_hostpath.token(w) != analysis._token(w)]
    assert not bad_w, [(w, s3_hostpath.token(w), analysis._token(w)) for w in bad_w[:20]]
    bad_t = [t for t in texts if s3_hostpath.analyze(t) != analysis.analyze(t)]
    assert not bad_t, [(t[:80], s3_hostpath.analyze(t)[:8], analysis.analyze(t)[:8]) for t in bad_t[:10]]


def test_analyze_equals_python_on_unicode_and_edge_texts():
    texts = ["İstanbul İZMİR", "KELVIN K scale", "ÅNGSTRÖM Å", "straße STRASSE", "ΣΊΣΥΦΟΣ σοφός",
             "don't won't o'clock rock'n'roll", "it's the dog's toys'", "'quoted' ''double''", "a'b'c 'x' y'",
             "ＡＢＣ fullwidth ａｂｃ", "naïve café résumé", "日本語 テキスト 123abc", "İi̇", "ẋy",
             "éclair", "TAB\tNEW\nLINE", "Ǆ ǅ ǆ", "ﬁne ﬂow", "1,234.56 -78", "", "   ", "'s", "s's", "the of and"]
    for t in texts:
        assert s3_hostpath.words_of(t) == analysis._WORD.findall(t.lower()), t
        assert s3_hostpath.analyze(t) == analysis.analyze(t), t
