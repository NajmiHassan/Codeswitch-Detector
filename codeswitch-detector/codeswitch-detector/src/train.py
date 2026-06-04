import pickle
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from dataset import load_conll, load_lexicon
from features import sentence_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"


def build_xy(sentences, en_lex, ur_lex):
    X, y = [], []
    for sent in sentences:
        tokens = [t for t, _ in sent]
        labels = [l for _, l in sent]
        X.extend(sentence_features(tokens, en_lex, ur_lex))
        y.extend(labels)
    return X, y


def train(save: bool = True):
    en_lex = load_lexicon(DATA / "lexicons" / "english_common.txt")
    ur_lex = load_lexicon(DATA / "lexicons" / "urdu_roman_common.txt")
    sentences = load_conll(DATA / "seed_labeled.tsv")

    X_dicts, y = build_xy(sentences, en_lex, ur_lex)

    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dicts)

    # class_weight='balanced' is important: 'ne'/'ambig' are rare classes and
    # we don't want the model to ignore them.
    clf = LogisticRegression(
        max_iter=2000,
        C=4.0,
        class_weight="balanced",
    )
    clf.fit(X, y)

    print(f"Trained on {X.shape[0]} tokens, {X.shape[1]} features.")

    if save:
        MODELS.mkdir(exist_ok=True)
        with open(MODELS / "baseline.pkl", "wb") as f:
            pickle.dump({"vec": vec, "clf": clf,
                         "en_lex": en_lex, "ur_lex": ur_lex}, f)
        print(f"Saved model -> {MODELS / 'baseline.pkl'}")

    return vec, clf, en_lex, ur_lex


if __name__ == "__main__":
    train()
