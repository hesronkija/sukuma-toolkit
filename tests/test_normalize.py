"""Tests for sukuma_toolkit.normalize, using real corpus examples."""

from sukuma_toolkit.normalize import (
    detect_diacritics,
    is_empty_after_normalization,
    normalize_text,
    strip_diacritics,
)


def test_normalize_collapses_whitespace():
    assert normalize_text("Alīyo   ū Moyo\twa ng'wa") == "Alīyo ū Moyo wa ng'wa"


def test_normalize_curly_quotes():
    assert normalize_text("\u201cwandījo\u201d") == '"wandījo"'


def test_normalize_strips_edges():
    assert normalize_text("  wandījo\n") == "wandījo"


def test_normalize_preserves_diacritics():
    s = "A ha wandījo wa kūseema"
    assert normalize_text(s) == s


def test_strip_diacritics():
    assert strip_diacritics("wandījo") == "wandijo"
    assert strip_diacritics("kūseema") == "kuseema"


def test_detect_diacritics():
    assert detect_diacritics("A ha wandījo wa kūseema") is True
    assert detect_diacritics("plain ascii words") is False


def test_is_empty_after_normalization():
    assert is_empty_after_normalization("   ") is True
    assert is_empty_after_normalization("wandijo") is False
