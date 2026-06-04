import pickle
from pathlib import Path

from tokenizer import tokenize, token_kind
from features import sentence_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "baseline.pkl"

# These token kinds are deterministically 'other' — don't let the model guess.
FORCED_OTHER = {"url", "mention", "hashtag", "number", "symbol"}


def load_tagger(model_path=MODEL_PATH):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    vec, clf = bundle["vec"], bundle["clf"]
    en_lex, ur_lex = bundle["en_lex"], bundle["ur_lex"]

    def tag(text: str):
        tokens = tokenize(text)
        if not tokens:
            return []
        feats = sentence_features(tokens, en_lex, ur_lex)
        X = vec.transform(feats)
        labels = clf.predict(X)
        out = []
        for tok, lbl in zip(tokens, labels):
            lbl = "other" if token_kind(tok) in FORCED_OTHER else str(lbl)
            out.append((tok, lbl))
        return out

    def tag_with_proba(text: str):
        tokens = tokenize(text)
        if not tokens:
            return []
        feats = sentence_features(tokens, en_lex, ur_lex)
        X = vec.transform(feats)
        probs = clf.predict_proba(X)
        classes = clf.classes_
        out = []
        for tok, p in zip(tokens, probs):
            if token_kind(tok) in FORCED_OTHER:
                out.append((tok, "other", 1.0))
            else:
                j = p.argmax()
                out.append((tok, str(classes[j]), float(p[j])))
        return out

    tag.with_proba = tag_with_proba  # attach for convenience
    return tag


if __name__ == "__main__":
    tag = load_tagger()
    for line in [
        "yaar that meeting was so boring kal phir karenge",
        "main office pohanch gaya bro traffic bohot tha",
        "let me check karta hoon ek minute",
    ]:
        print(line)
        print("  ", tag(line))
