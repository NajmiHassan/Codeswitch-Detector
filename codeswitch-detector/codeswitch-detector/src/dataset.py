from pathlib import Path

LABELS = ["en", "ur", "ne", "other", "ambig"]


def load_conll(path: str | Path) -> list[list[tuple[str, str]]]:
    """
    Returns a list of sentences; each sentence is a list of (token, label) pairs.
    """
    sentences: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Bad line (expected token<TAB>label): {line!r}")
            token, label = parts
            if label not in LABELS:
                raise ValueError(f"Unknown label {label!r} on line: {line!r}")
            current.append((token, label))
    if current:
        sentences.append(current)
    return sentences


def load_lexicon(path: str | Path) -> set[str]:
    """Load a newline-separated word list, lowercased, into a set."""
    words = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w and not w.startswith("#"):
                words.add(w)
    return words


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    sents = load_conll(here / "data" / "seed_labeled.tsv")
    n_tokens = sum(len(s) for s in sents)
    print(f"Loaded {len(sents)} sentences, {n_tokens} tokens.")
    from collections import Counter
    c = Counter(lbl for s in sents for _, lbl in s)
    print("Label distribution:", dict(c))
