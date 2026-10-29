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


def linked_groups(keys: Sequence[Sequence[str]]) -> list[int]:
    """Component id per item, linking items that share ANY key.

    ``keys[i]`` holds item i's keys, e.g. ``(speaker_id, normalized_text)``.
    Two items land in the same component when they share a speaker or a
    sentence, directly or through a chain of other items. Splitting whole
    components keeps every speaker and every sentence on one side.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for item_keys in keys:
        tagged = [f"{j}:{k}" for j, k in enumerate(item_keys)]
        for k in tagged[1:]:
            parent[find(k)] = find(tagged[0])
        find(tagged[0])
    roots: dict[str, int] = {}
    return [roots.setdefault(find(f"0:{item_keys[0]}"), len(roots)) for item_keys in keys]


def disjoint_split(
    keys: Sequence[Sequence[str]],
    sizes: dict[str, float] | None = None,
    seed: int = 42,
) -> list[str]:
    """Assign each item to a split so that no key (speaker, sentence) crosses splits.

    ``sizes`` maps split names to target fractions for every split except
    ``train``, which gets the rest (default ``{"test": 0.1}``). Whole linked
    components are assigned in a seeded random order, each to the held-out
    split furthest below its target; a component that would overshoot a
    target by more than half its own size goes to train. Real sizes are
    approximate when components are large; check them with ``validate_split``.
    """
    sizes = sizes or {"test": 0.1}
    if "train" in sizes or not sizes or sum(sizes.values()) >= 1.0 or min(sizes.values()) <= 0:
        raise ValueError("sizes: held-out fractions > 0 summing below 1, without 'train'")
    comp = linked_groups(keys)
    counts: dict[int, int] = {}
    for c in comp:
        counts[c] = counts.get(c, 0) + 1
    order = sorted(counts)
    random.Random(seed).shuffle(order)
    total = len(comp)
    targets = {name: frac * total for name, frac in sizes.items()}
    filled = {name: 0 for name in sizes}
    assign: dict[int, str] = {}
    for c in order:
        name = max(targets, key=lambda s: targets[s] - filled[s])
        gap = targets[name] - filled[name]
        if gap > 0 and counts[c] / 2 <= gap:
            assign[c] = name
            filled[name] += counts[c]
        else:
            assign[c] = "train"
    return [assign[c] for c in comp]


def split_leakage(keys: Sequence[str], splits: Sequence[str], held_out: str = "test") -> dict:
    """How much of a held-out split shares a key (e.g. a sentence) with train."""
    train_keys = {k for k, s in zip(keys, splits) if s == "train"}
    held = [k for k, s in zip(keys, splits) if s == held_out]
    leaked = [k for k in held if k in train_keys]
    return {
        "held_out": held_out,
        "items": len(held),
        "unique_keys": len(set(held)),
        "leaked_items": len(leaked),
        "leaked_unique_keys": len(set(leaked)),
        "leaked_share": len(leaked) / len(held) if held else 0.0,
    }


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
