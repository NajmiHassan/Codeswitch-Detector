from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

from dataset import load_conll, load_lexicon, LABELS
from features import sentence_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def fit_predict(train_sents, test_sents, en_lex, ur_lex):
    def to_xy(sents):
        X, y = [], []
        for s in sents:
            toks = [t for t, _ in s]
            X.extend(sentence_features(toks, en_lex, ur_lex))
            y.extend([l for _, l in s])
        return X, y

    Xtr, ytr = to_xy(train_sents)
    Xte, yte = to_xy(test_sents)

    vec = DictVectorizer(sparse=True)
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)

    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    clf.fit(Xtr_v, ytr)
    return yte, list(clf.predict(Xte_v))


def main(n_splits=5):
    en_lex = load_lexicon(DATA / "lexicons" / "english_common.txt")
    ur_lex = load_lexicon(DATA / "lexicons" / "urdu_roman_common.txt")
    sentences = load_conll(DATA / "seed_labeled.tsv")
    sentences = np.array(sentences, dtype=object)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_true, all_pred = [], []
    for tr_idx, te_idx in kf.split(sentences):
        yte, ypred = fit_predict(
            list(sentences[tr_idx]), list(sentences[te_idx]), en_lex, ur_lex
        )
        all_true.extend(yte)
        all_pred.extend(ypred)

    present = [l for l in LABELS if l in set(all_true)]
    print(f"\n{n_splits}-fold sentence-level cross-validation\n" + "=" * 44)
    print(classification_report(all_true, all_pred, labels=present, digits=3,
                                zero_division=0))

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(all_true, all_pred, labels=present)
    header = "       " + " ".join(f"{l:>6}" for l in present)
    print(header)
    for lbl, row in zip(present, cm):
        print(f"{lbl:>6} " + " ".join(f"{v:>6}" for v in row))


if __name__ == "__main__":
    main()
