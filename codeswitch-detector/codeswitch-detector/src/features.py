import re
from tokenizer import token_kind

# Substrings that are comparatively common in romanised Urdu.
URDU_CUES = [
    "aa", "ee", "oo", "kh", "gh", "ch", "ng", "ain", "ein", "ay", "ye",
    "ki", "ka", "ke", "ho", "hai", "nahi", "kar",
]


def _char_ngrams(text: str, n: int) -> list[str]:
    text = f"^{text}$"  # boundary markers so prefixes/suffixes are captured
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def token_features(
    tokens: list[str],
    i: int,
    en_lex: set[str],
    ur_lex: set[str],
) -> dict:
    tok = tokens[i]
    low = tok.lower()
    kind = token_kind(tok)

    feats: dict[str, float | bool | str] = {
        "bias": 1.0,
        "kind": kind,
        "tok.lower": low,
        "len.bucket": min(len(low), 8),
        "is.title": tok.istitle(),
        "is.upper": tok.isupper() and len(tok) > 1,
        "in.en": low in en_lex,
        "in.ur": low in ur_lex,
        "in.both": (low in en_lex) and (low in ur_lex),
        "in.neither": (low not in en_lex) and (low not in ur_lex),
    }

    # Only build character n-grams for actual words.
    if kind == "word":
        for n in (2, 3, 4):
            for g in _char_ngrams(low, n):
                feats[f"{n}g={g}"] = True
        for cue in URDU_CUES:
            if cue in low:
                feats[f"urcue={cue}"] = True
        feats["suf3"] = low[-3:]
        feats["pre3"] = low[:3]
        feats["has.double"] = bool(re.search(r"(.)\1", low))  # doubled letter

    # --- light context features ---
    if i > 0:
        prev = tokens[i - 1].lower()
        feats["-1.in.ur"] = prev in ur_lex
        feats["-1.in.en"] = prev in en_lex
    else:
        feats["BOS"] = True
    if i < len(tokens) - 1:
        nxt = tokens[i + 1].lower()
        feats["+1.in.ur"] = nxt in ur_lex
        feats["+1.in.en"] = nxt in en_lex
    else:
        feats["EOS"] = True

    return feats


def sentence_features(tokens, en_lex, ur_lex) -> list[dict]:
    return [token_features(tokens, i, en_lex, ur_lex) for i in range(len(tokens))]
