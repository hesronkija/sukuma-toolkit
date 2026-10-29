"""Subword tokenizer experiments for Sukuma (SentencePiece).

Needs the optional ``sentencepiece`` package: ``pip install -e ".[tokenizer]"``.

``train_tokenizer`` writes ``<prefix>.model`` and ``<prefix>.vocab``.
``fertility`` measures tokens per word on held-out text: lower means the
tokenizer splits Sukuma words into fewer pieces.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable, Sequence

from sukuma_toolkit.stats import words


def _require_spm():
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError('Install the tokenizer extra: pip install -e ".[tokenizer]"') from exc
    return spm


def train_tokenizer(
    sentences: Iterable[str],
    model_prefix: str | Path,
    vocab_size: int = 8000,
    model_type: str = "unigram",
    character_coverage: float = 1.0,
    seed_sentencepiece_size: int = 1_000_000,
) -> Path:
    """Train a SentencePiece model on sentences; return the .model path.

    ``character_coverage=1.0`` keeps every diacritic character in the vocabulary.
    """
    spm = _require_spm()
    prefix = Path(model_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        for s in sentences:
            s = s.strip()
            if s:
                f.write(s.replace("\n", " ") + "\n")
        corpus = f.name
    try:
        spm.SentencePieceTrainer.train(
            input=corpus,
            model_prefix=str(prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            seed_sentencepiece_size=seed_sentencepiece_size,
            hard_vocab_limit=False,
            input_sentence_size=0,
            minloglevel=2,
        )
    finally:
        Path(corpus).unlink(missing_ok=True)
    return prefix.with_suffix(".model")


def fertility(model_path: str | Path, sentences: Sequence[str]) -> dict:
    """Tokens per word and per sentence for a trained model on given sentences."""
    spm = _require_spm()
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    n_words = sum(len(words(s)) for s in sentences)
    n_pieces = sum(len(sp.encode(s, out_type=str)) for s in sentences)
    return {
        "sentences": len(sentences),
        "words": n_words,
        "pieces": n_pieces,
        "pieces_per_word": n_pieces / n_words if n_words else 0.0,
        "vocab_size": sp.get_piece_size(),
    }
