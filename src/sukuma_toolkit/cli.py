"""The ``sukuma`` command line tool.

    sukuma normalize notes.txt                  # clean Unicode, quotes and spacing
    sukuma normalize --strip-diacritics notes.txt
    sukuma restore plain.txt                     # wandijo -> wandījo
    echo "U Yesu ubawila" | sukuma restore
    sukuma stats sentences.csv                   # corpus report (Markdown)
    sukuma score refs.txt hyps.txt               # WER / CER at three levels

Files are read one sentence per line (``-`` or no file means standard input).
A .csv with a ``suk`` column or a .tsv with a ``text`` column also works.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from sukuma_toolkit import __version__
from sukuma_toolkit.normalize import normalize_text, strip_diacritics


def read_lines(source: str | None) -> list[str]:
    if source in (None, "-"):
        return [line.rstrip("\n") for line in sys.stdin]
    path = Path(source)
    if not path.exists():
        raise SystemExit(f"sukuma: no such file: {source}")
    if path.suffix.lower() == ".tsv":
        csv.field_size_limit(sys.maxsize)
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            if "text" not in (reader.fieldnames or []):
                raise SystemExit(f"sukuma: {source} has no 'text' column")
            return [r["text"] for r in reader]
    if path.suffix.lower() == ".csv":
        from sukuma_toolkit.corpus import load_afrispeech_text

        return load_afrispeech_text(path)
    return path.read_text(encoding="utf-8").splitlines()


def cmd_normalize(args) -> int:
    for line in read_lines(args.file):
        out = normalize_text(line)
        print(strip_diacritics(out) if args.strip_diacritics else out)
    return 0


def cmd_restore(args) -> int:
    from sukuma_toolkit.diacritics import DiacriticRestorer, load_default

    restorer = DiacriticRestorer.load(args.model) if args.model else load_default()
    for line in read_lines(args.file):
        print(restorer.restore(line))
    return 0


def cmd_stats(args) -> int:
    from sukuma_toolkit.stats import corpus_stats, format_report

    sentences = [s for s in (normalize_text(x) for x in read_lines(args.file)) if s]
    if not sentences:
        raise SystemExit("sukuma: no sentences found")
    name = Path(args.file).name if args.file not in (None, "-") else "stdin"
    print(format_report(corpus_stats(sentences, top_n=args.top), title=f"Corpus statistics: {name}", source=name))
    return 0


def cmd_score(args) -> int:
    from sukuma_toolkit.evaluate import score_levels

    refs, hyps = read_lines(args.references), read_lines(args.hypotheses)
    if len(refs) != len(hyps):
        raise SystemExit(f"sukuma: {len(refs)} references but {len(hyps)} hypotheses; need one per line, same order")
    scores = score_levels(refs, hyps)
    labels = {"strict": "Strict (exact text)", "normalized": "Lowercase, no punctuation",
              "diacritic_blind": "Diacritics also ignored"}
    for key, label in labels.items():
        print(f"{label:28} WER {scores[key]['wer']:6.1%}   CER {scores[key]['cer']:6.1%}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sukuma", description="Text tools for the Sukuma language.")
    ap.add_argument("--version", action="version", version=f"sukuma-toolkit {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("normalize", help="clean Unicode, quotes and spacing, one line at a time")
    p.add_argument("file", nargs="?")
    p.add_argument("--strip-diacritics", action="store_true", help="also convert to the plain written form")
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("restore", help="add diacritics to plain Sukuma text")
    p.add_argument("file", nargs="?")
    p.add_argument("--model", help="a restorer JSON from scripts/train_restorer.py (default: the shipped model)")
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser("stats", help="corpus statistics report in Markdown")
    p.add_argument("file", nargs="?")
    p.add_argument("--top", type=int, default=20, help="number of frequent words to list")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("score", help="WER and CER of hypotheses against references")
    p.add_argument("references")
    p.add_argument("hypotheses")
    p.set_defaults(func=cmd_score)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:  # e.g. piped into head
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
