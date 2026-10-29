#!/usr/bin/env python3
"""Audit the Sukuma Speech Corpus split and write a speaker- and sentence-disjoint one.

    python scripts/fetch_data.py --only sukuma-speech-corpus    # metadata only
    python scripts/make_disjoint_split.py

The official split keeps speakers apart but reuses the same prompt sentences
in train and test, so a model can score well by memorizing sentences. This
script measures that leak, then reassigns whole speaker+sentence groups so no
speaker and no sentence appears in more than one split.

Writes:
  data/splits/sukuma-speech-corpus_disjoint.csv   utterance_id, official split, new split
  reports/split_audit_sukuma-speech-corpus.md / .json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sukuma_toolkit.normalize import normalize_text  # noqa: E402
from sukuma_toolkit.split import disjoint_split, linked_groups, split_leakage  # noqa: E402

SPLITS = ("train", "validation", "test")


def load_rows(corpus_dir: Path) -> list[dict]:
    rows = []
    for split in SPLITS:
        path = corpus_dir / split / "metadata.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run: python scripts/fetch_data.py --only sukuma-speech-corpus")
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def sentence_key(text: str) -> str:
    return " ".join(normalize_text(text).lower().split())


def describe(rows: list[dict], splits: list[str]) -> dict:
    out = {}
    for name in SPLITS:
        part = [r for r, s in zip(rows, splits) if s == name]
        hours = sum(float(r["duration_seconds"]) for r in part) / 3600
        gender = collections.Counter(r.get("gender", "") for r in part)
        env = collections.Counter(r.get("recording_environment", "") for r in part)
        out[name] = {
            "utterances": len(part),
            "hours": round(hours, 2),
            "speakers": len({r["participant_record_id"] for r in part}),
            "sentences": len({sentence_key(r["text"]) for r in part}),
            "female_share": round(gender.get("Female", 0) / len(part), 3) if part else 0.0,
            "outdoor_share": round(env.get("Outdoor", 0) / len(part), 3) if part else 0.0,
        }
    return out


def audit(rows: list[dict], new_splits: list[str]) -> dict:
    official = [r["split"] for r in rows]
    sentences = [sentence_key(r["text"]) for r in rows]
    speakers = [r["participant_record_id"] for r in rows]
    report = {"utterances": len(rows), "groups": len(set(linked_groups(list(zip(speakers, sentences)))))}
    for label, splits in (("official", official), ("disjoint", new_splits)):
        report[label] = {
            "sizes": describe(rows, splits),
            "test_sentence_leak": split_leakage(sentences, splits, "test"),
            "test_speaker_leak": split_leakage(speakers, splits, "test"),
            "validation_sentence_leak": split_leakage(sentences, splits, "validation"),
        }
    return report


def format_report(r: dict) -> str:
    lines = [
        "# Sukuma Speech Corpus: split audit",
        "",
        f"{r['utterances']:,} utterances. Linking utterances that share a speaker or a sentence gives "
        f"{r['groups']} independent groups; each speaker reads one prompt set.",
        "",
        "## Leakage into train",
        "",
        "| Split | Test sentences also in train | Test speakers also in train | Validation sentences also in train |",
        "| --- | --- | --- | --- |",
    ]
    for label in ("official", "disjoint"):
        s, p, v = (r[label][k] for k in ("test_sentence_leak", "test_speaker_leak", "validation_sentence_leak"))
        lines.append(
            f"| {label} | {s['leaked_unique_keys']} of {s['unique_keys']} ({s['leaked_share']:.0%} of clips) "
            f"| {p['leaked_unique_keys']} of {p['unique_keys']} "
            f"| {v['leaked_unique_keys']} of {v['unique_keys']} ({v['leaked_share']:.0%} of clips) |"
        )
    for label in ("official", "disjoint"):
        lines += ["", f"## {label.capitalize()} split", "",
                  "| Split | Utterances | Hours | Speakers | Sentences | Female | Outdoor |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, d in r[label]["sizes"].items():
            lines.append(f"| {name} | {d['utterances']:,} | {d['hours']} | {d['speakers']} | {d['sentences']} "
                         f"| {d['female_share']:.0%} | {d['outdoor_share']:.0%} |")
    lines += ["", "With so few groups, held-out sizes are approximate. Use the disjoint split to measure how "
              "well a model handles new sentences, and the official split to measure new speakers only.", ""]
    return "\n".join(lines)


def run(corpus_dir: Path, out_csv: Path, report_dir: Path, test: float = 0.1, validation: float = 0.075,
        seed: int = 42) -> dict:
    rows = load_rows(corpus_dir)
    keys = [(r["participant_record_id"], sentence_key(r["text"])) for r in rows]
    new = disjoint_split(keys, {"test": test, "validation": validation}, seed=seed)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utterance_id", "official_split", "disjoint_split"])
        for r, s in sorted(zip(rows, new), key=lambda x: x[0]["utterance_id"]):
            w.writerow([r["utterance_id"], r["split"], s])
    report = audit(rows, new)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "split_audit_sukuma-speech-corpus.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_dir / "split_audit_sukuma-speech-corpus.md").write_text(format_report(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "raw" / "sukuma-speech-corpus")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "splits" / "sukuma-speech-corpus_disjoint.csv")
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--validation", type=float, default=0.075)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    r = run(args.corpus_dir, args.out, ROOT / "reports", args.test, args.validation, args.seed)
    for label in ("official", "disjoint"):
        leak = r[label]["test_sentence_leak"]
        print(f"{label:9} test sentences also in train: {leak['leaked_unique_keys']}/{leak['unique_keys']}")
    print(f"Split: {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    print("Report: reports/split_audit_sukuma-speech-corpus.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
