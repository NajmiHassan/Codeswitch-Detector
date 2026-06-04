import sys
from pathlib import Path

from tokenizer import tokenize, token_kind
from dataset import load_lexicon

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def weak_label_line(text, en_lex, ur_lex, model_tag=None):
    tokens = tokenize(text)
    model_preds = None
    if model_tag is not None and tokens:
        model_preds = {i: lbl for i, (_, lbl) in enumerate(model_tag(text))}

    out = []
    for i, tok in enumerate(tokens):
        kind = token_kind(tok)
        low = tok.lower()
        if kind != "word":
            label = "other"
        elif low in en_lex and low in ur_lex:
            label = "ambig"
        elif low in en_lex:
            label = "en"
        elif low in ur_lex:
            label = "ur"
        else:
            label = model_preds[i] if model_preds else "ur"
        out.append((tok, label))
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: python src/weak_label.py <raw_text_file>", file=sys.stderr)
        sys.exit(1)

    en_lex = load_lexicon(DATA / "lexicons" / "english_common.txt")
    ur_lex = load_lexicon(DATA / "lexicons" / "urdu_roman_common.txt")

    model_tag = None
    try:
        from predict import load_tagger
        model_tag = load_tagger()
    except Exception:
        pass  # no trained model yet; fall back to lexicon + prior

    with open(sys.argv[1], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for tok, label in weak_label_line(line, en_lex, ur_lex, model_tag):
                print(f"{tok}\t{label}")
            print()  # sentence boundary


if __name__ == "__main__":
    main()
