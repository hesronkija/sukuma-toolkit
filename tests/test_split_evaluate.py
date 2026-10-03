"""Tests for sukuma_toolkit.split and .evaluate."""

from sukuma_toolkit.evaluate import cer, corpus_wer, wer
from sukuma_toolkit.split import (
    train_test_split_sentences,
    validate_sentences,
    validate_split,
)

_SENTENCES = [f"sentence number {i} wandījo" for i in range(100)]


def test_split_sizes_and_reproducibility():
    train1, test1 = train_test_split_sentences(_SENTENCES, test_size=0.1, seed=7)
    train2, test2 = train_test_split_sentences(_SENTENCES, test_size=0.1, seed=7)
    assert len(test1) == 10
    assert len(train1) == 90
    assert train1 == train2 and test1 == test2


def test_split_no_leak():
    train, test = train_test_split_sentences(_SENTENCES, test_size=0.2, seed=1)
    report = validate_split(train, test)
    assert report["leak_free"] is True
    assert report["train_size"] + report["test_size"] == 100


def test_validate_sentences_counts_duplicates():
    report = validate_sentences(["a", "b", "a", ""])
    assert report["total"] == 4
    assert report["duplicates"] == 1
    assert report["empty"] == 1


def test_wer_perfect_and_empty():
    assert wer("wandijo wa kuseema", "wandijo wa kuseema") == 0.0
    assert wer("", "") == 0.0


def test_wer_one_substitution():
    # 1 wrong word out of 3 -> 1/3
    assert abs(wer("a b c", "a x c") - 1 / 3) < 1e-9


def test_cer():
    assert cer("abc", "abc") == 0.0
    assert abs(cer("abc", "axc") - 1 / 3) < 1e-9


def test_corpus_wer_averages():
    refs = ["a b c", "d e"]
    hyps = ["a b c", "d x"]
    assert abs(corpus_wer(refs, hyps) - 1 / 5) < 1e-9
