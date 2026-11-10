"""Text normalization for Sukuma.

Sukuma is written in two orthographic forms: with diacritics (e.g. wandījo)
and without (wandijo). This module normalizes surface text (Unicode form,
whitespace, punctuation) and explicitly handles the diacritic / non-diacritic
distinction so downstream tools can pick one canonical form.
"""

import re
import unicodedata

# Curly quotes, dashes and other typographic punctuation mapped to ASCII.
_PUNCT_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)

_WHITESPACE_RE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    """Normalize a Sukuma sentence to a canonical surface form.

    Steps: NFC Unicode normalization, typographic punctuation mapped to
    ASCII, whitespace collapsed, surrounding whitespace stripped.
    Diacritics are preserved; use :func:`strip_diacritics` to remove them.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_PUNCT_MAP)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def strip_diacritics(text: str) -> str:
    """Convert diacritic-form Sukuma to the non-diacritic written form.

    Uses NFD decomposition and drops combining marks, then recomposes.
    E.g. "wandījo" -> "wandijo".
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    )
    return unicodedata.normalize("NFC", stripped)


def detect_diacritics(text: str) -> bool:
    """Return True if the text contains any diacritic marks."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    return unicodedata.normalize("NFC", text) != strip_diacritics(text)
