#!/usr/bin/env python3
"""Tokenizer experiment: does normalizing diacritics change how Sukuma is split?

    pip install -e ".[tokenizer]"
    python scripts/train_tokenizer.py                       # data/sukuma_sentences.csv
    python scripts/train_tokenizer.py --vocab-size 4000

Holds out 10% of sentences (seeded), then trains two SentencePiece models on
the rest: one on the text as written, one with diacritics removed. Reports
tokens per word on the held-out sentences for each and writes
reports/tokenizer_experiment.md. Models go to models/ (gitignored: they
are trained on copyrighted Bible text, so they stay local for now).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sukuma_toolkit.corpus import load_afrispeech_text  # noqa: E402
from sukuma_toolkit.normalize import strip_diacritics  # noqa: E402
from sukuma_toolkit.split import train_test_split_sentences, validate_split  # noqa: E402
from sukuma_toolkit.tokenizer import fertility, train_tokenizer  # noqa: E402


def run(sentences: list[str], out_models: Path, out_reports: Path, vocab_size: int, model_type: str,
        seed: int = 42) -> dict:
    sentences = list(dict.fromkeys(sentences))  # dedupe, keep order, so the split cannot leak
    train, test = train_test_split_sentences(sentences, test_size=0.1, seed=seed)
    split = validate_split(train, test)
    results = {"split": split, "vocab_size": vocab_size, "model_type": model_type, "models": {}}
    variants = {
        "as_written": (train, test),
        "diacritics_removed": ([strip_diacritics(s) for s in train], [strip_diacritics(s) for s in test]),
    }
    for name, (tr, te) in variants.items():
        model = train_tokenizer(tr, out_models / f"sukuma_{model_type}_{vocab_size}_{name}",
                                vocab_size=vocab_size, model_type=model_type)
        results["models"][name] = {"path": str(model.relative_to(ROOT)) if model.is_relative_to(ROOT) else str(model),
                                   **fertility(model, te)}

    out_reports.mkdir(parents=True, exist_ok=True)
    a, b = results["models"]["as_written"], results["models"]["diacritics_removed"]
    lines = [
        "# Tokenizer experiment",
        "",
        f"SentencePiece {model_type}, target vocabulary {vocab_size:,}. "
        f"Trained on {split['train_size']:,} sentences, scored on {split['test_size']:,} held-out sentences "
        f"(overlap: {split['overlap']}).",
        "",
        "| Text | Pieces per word | Pieces | Vocabulary |",
        "| --- | --- | --- | --- |",
        f"| As written | {a['pieces_per_word']:.3f} | {a['pieces']:,} | {a['vocab_size']:,} |",
        f"| Diacritics removed | {b['pieces_per_word']:.3f} | {b['pieces']:,} | {b['vocab_size']:,} |",
        "",
        "Fewer pieces per word means the tokenizer keeps more Sukuma words whole.",
        "",
    ]
    (out_reports / "tokenizer_experiment.md").write_text("\n".join(lines), encoding="utf-8")
    (out_reports / "tokenizer_experiment.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", type=Path, default=ROOT / "data" / "sukuma_sentences.csv")
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--model-type", choices=["unigram", "bpe"], default="unigram")
    args = ap.parse_args(argv)
    if not args.input.exists():
        print(f"error: {args.input} not found; run scripts/download_afrispeech_text.py first.", file=sys.stderr)
        return 2
    r = run(load_afrispeech_text(args.input), ROOT / "models", ROOT / "reports", args.vocab_size, args.model_type)
    for name, m in r["models"].items():
        print(f"{name:20} {m['pieces_per_word']:.3f} pieces/word  ({m['path']})")
    print("Report: reports/tokenizer_experiment.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
