import sys
from pathlib import Path

# make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predict import load_tagger          # noqa: E402
from cmi import utterance_cmi            # noqa: E402

COLORS = {
    "en": "\033[94m",     # blue
    "ur": "\033[92m",     # green
    "ne": "\033[95m",     # magenta
    "ambig": "\033[93m",  # yellow
    "other": "\033[90m",  # grey
}
RESET = "\033[0m"


def render_cli(tagged):
    parts = [f"{COLORS.get(lbl,'')}{tok}{RESET}" for tok, lbl in tagged]
    return " ".join(parts)


def run_cli(tag, text=None):
    print("Legend:", " ".join(f"{COLORS[k]}{k}{RESET}" for k in COLORS))
    print("-" * 50)

    def show(t):
        tagged = tag(t)
        print(render_cli(tagged))
        print(f"  tags: {tagged}")
        print(f"  CMI : {utterance_cmi(tagged):.1f}\n")

    if text:
        show(text)
        return
    print("Type a sentence (blank line to quit):")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        show(line)


def run_web(tag):
    import gradio as gr  # imported lazily

    def classify(text):
        tagged = tag(text)
        html = " ".join(
            f'<span style="padding:2px 4px;border-radius:4px;'
            f'background:{c};">{tok}</span>'
            for tok, lbl in tagged
            for c in [{"en": "#cfe2ff", "ur": "#d1f0d1", "ne": "#f0d1f0",
                       "ambig": "#fff2cc", "other": "#e0e0e0"}.get(lbl, "#fff")]
        )
        return html, f"Code-Mixing Index: {utterance_cmi(tagged):.1f}"

    gr.Interface(
        fn=classify,
        inputs=gr.Textbox(label="Roman Urdu / English text"),
        outputs=[gr.HTML(label="Tagged"), gr.Text(label="CMI")],
        title="Roman Urdu–English Code-Switching Detector",
        examples=[
            "yaar that meeting was so boring kal phir karenge",
            "main office pohanch gaya bro traffic bohot tha",
            "let me check karta hoon ek minute",
        ],
    ).launch()


if __name__ == "__main__":
    tag = load_tagger()
    if "--web" in sys.argv:
        run_web(tag)
    else:
        arg = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
        run_cli(tag, arg)
