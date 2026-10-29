#!/usr/bin/env python3
"""Corpus statistics (EDA) report for Sukuma text.

    python scripts/run_eda.py                                   # data/sukuma_sentences.csv
    python scripts/run_eda.py data/raw/sukuma-speech-corpus/text.tsv
    python scripts/run_eda.py data/sukuma_sentences.csv --plot  # + length histogram PNG

Reads a corpus CSV (``suk`` column, from download_afrispeech_text.py) or a
``text.tsv`` written by fetch_data.py, and writes reports/eda_<name>.md (and
reports/eda_<name>_lengths.png with --plot, needs matplotlib). The report
holds counts and frequent words only, never full sentences.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sukuma_toolkit.corpus import load_afrispeech_text  # noqa: E402
from sukuma_toolkit.stats import corpus_stats, format_report  # noqa: E402

DEFAULT_INPUT = ROOT / "data" / "sukuma_sentences.csv"
REPORTS = ROOT / "reports"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def load_sentences(path: Path) -> list[str]:
    """Load sentences from a corpus CSV, a fetch_data text.tsv, or a plain text file."""
    if path.suffix == ".tsv":
        with path.open(encoding="utf-8", newline="") as f:
            return [r["text"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
    return load_afrispeech_text(path)


def report_name(path: Path) -> str:
    return path.parent.name if path.name == "text.tsv" else path.stem


def plot_lengths(stats: dict, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = stats["length_histogram"]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.bar(list(hist), list(hist.values()), width=0.9, color="#2F6B8A")
    ax.set_xlabel("Words per sentence")
    ax.set_ylabel("Sentences")
    ax.set_title(title, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=REPORTS)
    ap.add_argument("--top", type=int, default=25, help="how many frequent words to list")
    ap.add_argument("--plot", action="store_true", help="also save a sentence-length histogram (matplotlib)")
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"error: {args.input} not found. Run scripts/fetch_data.py and "
              "scripts/download_afrispeech_text.py first.", file=sys.stderr)
        return 2

    sentences = load_sentences(args.input)
    stats = corpus_stats(sentences, top_n=args.top)
    name = report_name(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = args.out_dir / f"eda_{name}.md"
    md.write_text(format_report(stats, f"Corpus statistics: {name}", source=args.input.name), encoding="utf-8")
    (args.out_dir / f"eda_{name}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.plot:
        plot_lengths(stats, args.out_dir / f"eda_{name}_lengths.png", f"Sentence lengths: {name}")

    print(f"{stats['sentences']:,} sentences, {stats['tokens']:,} words, "
          f"{stats['vocabulary']:,} distinct words ({stats['vocabulary_without_diacritics']:,} without diacritics), "
          f"{stats['diacritic_share']:.1%} of sentences carry diacritics")
    print(f"Report: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
