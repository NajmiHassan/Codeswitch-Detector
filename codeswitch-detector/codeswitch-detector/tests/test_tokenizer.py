import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokenizer import tokenize, token_kind


def test_keeps_mentions_hashtags_urls_whole():
    toks = tokenize("@ali kal #plan https://x.com/a")
    assert "@ali" in toks
    assert "#plan" in toks
    assert any(t.startswith("https://") for t in toks)


def test_splits_punctuation():
    assert tokenize("hi, bro") == ["hi", ",", "bro"]


def test_token_kind():
    assert token_kind("@ali") == "mention"
    assert token_kind("#plan") == "hashtag"
    assert token_kind("123") == "number"
    assert token_kind("yaar") == "word"
    assert token_kind(",") == "symbol"
