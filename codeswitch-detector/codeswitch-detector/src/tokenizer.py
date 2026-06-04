import re

# Order matters: more specific patterns first.
TOKEN_PATTERN = re.compile(
    r"""
    (?P<url>https?://\S+|www\.\S+)        |   # URLs
    (?P<mention>@\w+)                     |   # @mentions
    (?P<hashtag>\#\w+)                    |   # #hashtags
    (?P<number>\d+(?:[.,]\d+)*)           |   # numbers like 3, 3.5, 1,000
    (?P<word>[A-Za-z]+(?:'[A-Za-z]+)?)    |   # words, incl. simple apostrophes
    (?P<emoji>[^\w\s])                        # any single non-word, non-space char (punct/emoji)
    """,
    re.VERBOSE,
)


def tokenize(text: str) -> list[str]:
    """Split raw text into tokens. Returns a flat list of token strings."""
    tokens = []
    for match in TOKEN_PATTERN.finditer(text):
        tokens.append(match.group())
    return tokens


def token_kind(token: str) -> str:
    """
    Cheap rule-based 'kind' of a token. Used both as a feature and as a
    shortcut in the weak labeller for things that are obviously 'other'.
    """
    if re.match(r"^https?://|^www\.", token):
        return "url"
    if token.startswith("@"):
        return "mention"
    if token.startswith("#"):
        return "hashtag"
    if re.match(r"^\d", token):
        return "number"
    if re.match(r"^[A-Za-z]+(?:'[A-Za-z]+)?$", token):
        return "word"
    return "symbol"  # punctuation, emoji, etc.


if __name__ == "__main__":
    demo = "yaar that meeting was so boring 🙂 @ali kal milte hain #plan https://x.com/a"
    for t in tokenize(demo):
        print(f"{t!r:20} -> {token_kind(t)}")
