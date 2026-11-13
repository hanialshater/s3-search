"""The referee's stemmer against the examples printed in Porter (1980), and its analyzer on hand-worked texts."""
import pytest

from referee.analysis import analyze
from referee.porter import stem

# Pairs from the paper's step-by-step examples (Porter, "An algorithm for suffix stripping", 1980).
PAPER = {
    "caresses": "caress", "ponies": "poni", "ties": "ti", "caress": "caress", "cats": "cat",
    "feed": "feed", "agreed": "agre", "plastered": "plaster", "bled": "bled", "motoring": "motor", "sing": "sing",
    "conflated": "conflat", "troubled": "troubl", "sized": "size", "hopping": "hop", "tanned": "tan",
    "falling": "fall", "hissing": "hiss", "fizzed": "fizz", "failing": "fail", "filing": "file",
    "happy": "happi", "sky": "sky",
    "relational": "relat", "conditional": "condit", "rational": "ration", "valenci": "valenc",
    "hesitanci": "hesit", "digitizer": "digit", "conformabli": "conform", "radicalli": "radic",
    "differentli": "differ", "vileli": "vile", "analogousli": "analog", "vietnamization": "vietnam",
    "predication": "predic", "operator": "oper", "feudalism": "feudal", "decisiveness": "decis",
    "hopefulness": "hope", "callousness": "callous", "formaliti": "formal", "sensitiviti": "sensit",
    "sensibiliti": "sensibl",
    "triplicate": "triplic", "formative": "form", "formalize": "formal", "electriciti": "electr",
    "electrical": "electr", "hopeful": "hope", "goodness": "good",
    "revival": "reviv", "allowance": "allow", "inference": "infer", "airliner": "airlin",
    "gyroscopic": "gyroscop", "adjustable": "adjust", "defensible": "defens", "irritant": "irrit",
    "replacement": "replac", "adjustment": "adjust", "dependent": "depend", "adoption": "adopt",
    "homologou": "homolog", "communism": "commun", "activate": "activ", "angulariti": "angular",
    "homologous": "homolog", "effective": "effect", "bowdlerize": "bowdler",
    "probate": "probat", "rate": "rate", "cease": "ceas", "controll": "control", "roll": "roll",
    "generalizations": "gener", "oscillators": "oscil",
}


@pytest.mark.parametrize("word,expected", sorted(PAPER.items()))
def test_porter_paper_examples(word, expected):
    assert stem(word) == expected


def test_analyzer_lowercases_drops_stop_words_and_possessives():
    assert analyze("The Kestrel's Waterproof Boots, for Hiking!") == ["kestrel", "waterproof", "boot", "hike"]


def test_analyzer_keeps_digits_and_inner_apostrophes_once():
    assert analyze("2-pack rock'n roll 12oz") == ["2", "pack", "rockn", "roll", "12oz"]


def test_analyzer_handles_missing_text():
    assert analyze(None) == [] and analyze("") == [] and analyze("the and of") == []
