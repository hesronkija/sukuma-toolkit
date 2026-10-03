"""Dataset validation and train/test splitting utilities."""

from __future__ import annotations

import random
from typing import Sequence


def validate_sentences(sentences: Sequence[str]) -> dict:
    """Basic validation report for a sentence list.

    Returns counts: total, empty, duplicates, and the share with diacritics.
    """
    from sukuma_toolkit.normalize import detect_diacritics

    total = len(sentences)
    empty = sum(1 for s in sentences if not s.strip())
    seen: set[str] = set()
    dups = 0
    for s in sentences:
        if s in seen:
            dups += 1
        seen.add(s)
    with_diacritics = sum(1 for s in sentences if detect_diacritics(s))
    return {
        "total": total,
        "empty": empty,
        "duplicates": dups,
        "unique": total - dups,
        "with_diacritics": with_diacritics,
        "diacritic_share": (with_diacritics / total) if total else 0.0,
    }


def train_test_split_sentences(
    sentences: Sequence[str],
    test_size: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[list[str], list[str]]:
    """Deterministic train/test split for sentence lists.

    Shuffles with the given seed (when shuffle=True) so splits are
    reproducible across runs, then takes the last ``test_size`` fraction
    as the test set.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    items = list(sentences)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    n_test = max(1, round(len(items) * test_size))
    return items[:-n_test], items[-n_test:]


def validate_split(
    train: Sequence[str], test: Sequence[str]
) -> dict:
    """Check a split for leakage (overlap) and report sizes."""
    train_set, test_set = set(train), set(test)
    overlap = train_set & test_set
    return {
        "train_size": len(train),
        "test_size": len(test),
        "overlap": len(overlap),
        "leak_free": len(overlap) == 0,
    }
