"""sukuma-toolkit: text and speech utilities for the Sukuma language."""

from sukuma_toolkit.normalize import (
    detect_diacritics,
    normalize_text,
    strip_diacritics,
)
from sukuma_toolkit.split import train_test_split_sentences, validate_split

__version__ = "0.1.0"
__all__ = [
    "normalize_text",
    "strip_diacritics",
    "detect_diacritics",
    "train_test_split_sentences",
    "validate_split",
]
