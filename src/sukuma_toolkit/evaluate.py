"""Evaluation helpers: WER and CER for Sukuma ASR experiments."""

from __future__ import annotations


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance between two token sequences (dynamic programming)."""
    m, n = len(ref), len(hyp)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate between a reference and hypothesis transcription.

    Computed as edit distance over whitespace-tokenized words divided by
    the number of reference words. Returns 0.0 for two empty strings.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate (same algorithm, character tokens)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(list(reference), list(hypothesis)) / len(reference)


def corpus_wer(references: list[str], hypotheses: list[str]) -> float:
    """Micro-averaged WER over parallel lists of references/hypotheses."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    total_edits = 0
    total_words = 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        total_edits += _edit_distance(ref_words, hyp.split())
        total_words += len(ref_words)
    return total_edits / total_words if total_words else 0.0
