from collections import Counter

LANG_LABELS = {"en", "ur"}  # 'ne', 'other', 'ambig' are excluded from CMI


def utterance_cmi(tagged: list[tuple[str, str]]) -> float:
    counts = Counter(lbl for _, lbl in tagged if lbl in LANG_LABELS)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 100.0 * (1 - max(counts.values()) / total)


def corpus_cmi(tagged_utterances: list[list[tuple[str, str]]]) -> dict:
    per = [utterance_cmi(u) for u in tagged_utterances]
    mixed = [c for c in per if c > 0]
    return {
        "n_utterances": len(per),
        "avg_cmi": round(sum(per) / len(per), 2) if per else 0.0,
        "avg_cmi_mixed_only": round(sum(mixed) / len(mixed), 2) if mixed else 0.0,
        "pct_code_mixed": round(100 * len(mixed) / len(per), 1) if per else 0.0,
    }


if __name__ == "__main__":
    from predict import load_tagger
    tag = load_tagger()
    samples = [
        "yaar that meeting was so boring kal phir karenge",
        "this is completely fine",
        "main bilkul theek hoon",
    ]
    tagged = [tag(s) for s in samples]
    for s, t in zip(samples, tagged):
        print(f"CMI {utterance_cmi(t):5.1f}  | {s}")
    print("\nCorpus stats:", corpus_cmi(tagged))
