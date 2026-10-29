"""Corpus statistics (EDA) for Sukuma text.

``corpus_stats`` summarizes a list of sentences: size, vocabulary, sentence
lengths, how much of the text carries diacritics, and which diacritic
characters occur. ``format_report`` turns that summary into Markdown for
blog posts and the poster. Words are counted after :func:`normalize_text`
and lowercasing; apostrophes inside words (ng'wana) are kept.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from typing import Iterable, Sequence

from sukuma_toolkit.normalize import detect_diacritics, normalize_text, strip_diacritics

_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")


def words(text: str) -> list[str]:
    """Lowercased word tokens of a sentence (letters, with inner apostrophes)."""
    return _WORD_RE.findall(normalize_text(text).lower())


def _percentile(sorted_values: Sequence[int], q: float) -> float:
    """Nearest-rank percentile of an already sorted list (q in [0, 100])."""
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1, round(q / 100 * (len(sorted_values) - 1))))
    return float(sorted_values[k])


def corpus_stats(sentences: Iterable[str], top_n: int = 20) -> dict:
    """Summary statistics for a Sukuma sentence list."""
    sents = [normalize_text(s) for s in sentences]
    sents = [s for s in sents if s]
    tokens_per_sentence = [words(s) for s in sents]
    all_tokens = [w for ws in tokens_per_sentence for w in ws]
    counts = Counter(all_tokens)
    stripped = Counter(strip_diacritics(w) for w in all_tokens)
    lengths = sorted(len(ws) for ws in tokens_per_sentence)

    diacritic_chars: Counter = Counter()
    for s in sents:
        for ch in unicodedata.normalize("NFC", s):
            if ch.isalpha() and strip_diacritics(ch) != ch:
                diacritic_chars[ch.lower()] += 1

    n = len(sents)
    return {
        "sentences": n,
        "unique_sentences": len(set(sents)),
        "tokens": len(all_tokens),
        "vocabulary": len(counts),
        "vocabulary_without_diacritics": len(stripped),
        "hapax_words": sum(1 for c in counts.values() if c == 1),
        "type_token_ratio": (len(counts) / len(all_tokens)) if all_tokens else 0.0,
        "words_per_sentence": {
            "mean": statistics.fmean(lengths) if lengths else 0.0,
            "median": float(statistics.median(lengths)) if lengths else 0.0,
            "p10": _percentile(lengths, 10),
            "p90": _percentile(lengths, 90),
            "max": float(lengths[-1]) if lengths else 0.0,
        },
        "sentences_with_diacritics": sum(1 for s in sents if detect_diacritics(s)),
        "diacritic_share": (sum(1 for s in sents if detect_diacritics(s)) / n) if n else 0.0,
        "diacritic_characters": dict(diacritic_chars.most_common()),
        "top_words": counts.most_common(top_n),
        "length_histogram": dict(sorted(Counter(lengths).items())),
    }


def format_report(stats: dict, title: str = "Sukuma corpus statistics", source: str = "") -> str:
    """Render :func:`corpus_stats` output as a Markdown report."""
    w = stats["words_per_sentence"]
    merged = stats["vocabulary"] - stats["vocabulary_without_diacritics"]
    lines = [
        f"# {title}",
        "",
        *( [f"Source: {source}", ""] if source else [] ),
        "| Measure | Value |",
        "| --- | --- |",
        f"| Sentences | {stats['sentences']:,} |",
        f"| Unique sentences | {stats['unique_sentences']:,} |",
        f"| Word tokens | {stats['tokens']:,} |",
        f"| Vocabulary (distinct words) | {stats['vocabulary']:,} |",
        f"| Vocabulary with diacritics removed | {stats['vocabulary_without_diacritics']:,} |",
        f"| Words seen only once | {stats['hapax_words']:,} |",
        f"| Type-token ratio | {stats['type_token_ratio']:.3f} |",
        f"| Words per sentence: mean / median | {w['mean']:.1f} / {w['median']:.0f} |",
        f"| Words per sentence: 10th-90th percentile | {w['p10']:.0f}-{w['p90']:.0f} (max {w['max']:.0f}) |",
        f"| Sentences with diacritics | {stats['sentences_with_diacritics']:,} ({stats['diacritic_share']:.1%}) |",
        "",
        f"Removing diacritics merges {merged:,} spellings into words that already exist, "
        "which is how much a model would see as 'different words' without normalization.",
        "",
        "## Diacritic characters",
        "",
        "| Character | Count |",
        "| --- | --- |",
        *[f"| {ch} | {c:,} |" for ch, c in stats["diacritic_characters"].items()],
        "",
        "## Most frequent words",
        "",
        "| Rank | Word | Count |",
        "| --- | --- | --- |",
        *[f"| {i} | {wd} | {c:,} |" for i, (wd, c) in enumerate(stats["top_words"], 1)],
        "",
    ]
    return "\n".join(lines)
