#!/usr/bin/env python3
"""Train and evaluate the Sukuma diacritic restorer (plain text -> diacritic form).

    python scripts/download_afrispeech_text.py               # Bible sentences, local only
    python scripts/fetch_data.py --only sukuma-speech-corpus # corpus prompts (CC BY 4.0)
    python scripts/make_disjoint_split.py                    # sentence-disjoint corpus split
    python scripts/train_restorer.py

Experiments (each one strips the diacritics from held-out sentences, restores
them, and counts words written exactly right):

  bible -> bible      train on 90% of the Bible sentences, test on the other 10%
  bible -> corpus     same model, tested on the corpus's held-out prompt sentences
  corpus -> corpus    train on the corpus's training sentences only
  both -> corpus      Bible + corpus training sentences

Writes reports/diacritic_restorer.md / .json and saves:
  models/sukuma_restorer_bible.json        (Bible-derived: local only, gitignored)
  src/sukuma_toolkit/data/restorer_speech_corpus.json  (CC BY 4.0 corpus only: ships as the default)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sukuma_toolkit.diacritics import DiacriticRestorer, evaluate_restorer  # noqa: E402
from sukuma_toolkit.normalize import detect_diacritics  # noqa: E402
from sukuma_toolkit.split import train_test_split_sentences  # noqa: E402

DEFAULT_SHIPPED = ROOT / "src" / "sukuma_toolkit" / "data" / "restorer_speech_corpus.json"


def read_bible(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [row[0] for row in csv.reader(f)][1:]


def read_corpus(corpus_dir: Path, split_csv: Path) -> dict[str, list[str]]:
    """Unique corpus prompt sentences per disjoint split (train / validation / test)."""
    if not split_csv.exists():
        return {}
    with split_csv.open(encoding="utf-8", newline="") as f:
        assign = {r["utterance_id"]: r["disjoint_split"] for r in csv.DictReader(f)}
    out: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for split in ("train", "validation", "test"):
        path = corpus_dir / split / "metadata.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["utterance_id"] in assign:
                    out[assign[r["utterance_id"]]].add(r["text"])
    return {k: sorted(v) for k, v in out.items()}


def format_report(results: dict) -> str:
    lines = [
        "# Diacritic restorer v1",
        "",
        "Plain Sukuma in, diacritic form out. Each test strips the marks from held-out sentences, restores them, "
        "and counts words written exactly right (every mark must match).",
        "",
        "| Train -> test | Test words | Leave plain | Restored | Words that need a mark | Unseen words |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['words']:,} | {r['baseline_accuracy']:.1%} | **{r['word_accuracy']:.1%}** "
                     f"| {r['marked_word_accuracy']:.1%} | {r['unseen_word_accuracy']:.1%} "
                     f"({r['unseen_words']:,}) |")
    lines += ["", "\"Leave plain\" is the score of doing nothing. \"Unseen words\" never appeared in training, so the "
              "character model restores them letter by letter.", ""]
    return "\n".join(lines)


def run(bible_csv: Path, corpus_dir: Path, split_csv: Path, out_dir: Path, models_dir: Path,
        shipped: Path = DEFAULT_SHIPPED) -> dict:
    bible = [s for s in read_bible(bible_csv) if detect_diacritics(s)]
    corpus = read_corpus(corpus_dir, split_csv)
    if not bible and not corpus:
        raise SystemExit("No training text. Run download_afrispeech_text.py and/or fetch the speech corpus.")
    results: dict[str, dict] = {}
    if bible:
        b_train, b_test = train_test_split_sentences(bible, 0.1, seed=42)
        bible_model = DiacriticRestorer().fit(b_train)
        results["bible -> bible"] = evaluate_restorer(bible_model, b_test)
        if corpus.get("test"):
            results["bible -> corpus"] = evaluate_restorer(bible_model, corpus["test"])
        models_dir.mkdir(parents=True, exist_ok=True)
        DiacriticRestorer().fit(bible).save(models_dir / "sukuma_restorer_bible.json")
    if corpus.get("train") and corpus.get("test"):
        results["corpus -> corpus"] = evaluate_restorer(DiacriticRestorer().fit(corpus["train"]), corpus["test"])
        if bible:
            both = DiacriticRestorer().fit(b_train + corpus["train"])
            results["both -> corpus"] = evaluate_restorer(both, corpus["test"])
            results["both -> bible"] = evaluate_restorer(both, b_test)
    if corpus:
        shipped.parent.mkdir(parents=True, exist_ok=True)
        DiacriticRestorer().fit(sum(corpus.values(), [])).save(shipped)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diacritic_restorer.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "diacritic_restorer.md").write_text(format_report(results), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bible", type=Path, default=ROOT / "data" / "sukuma_sentences.csv")
    ap.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "raw" / "sukuma-speech-corpus")
    ap.add_argument("--split", type=Path, default=ROOT / "data" / "splits" / "sukuma-speech-corpus_disjoint.csv")
    args = ap.parse_args(argv)
    results = run(args.bible, args.corpus_dir, args.split, ROOT / "reports", ROOT / "models")
    print(format_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
