#!/usr/bin/env python3
"""Score Sukuma -> Swahili translation on held-out Bible books and on the UDHR.

    python scripts/eval_translator.py --model models/nllb-suk-swh
    python scripts/eval_translator.py --baselines          # also score the two baselines

Metrics (sacrebleu): chrF++ (character n-grams; the fairer score for
agglutinative languages) and BLEU. Bible verses are scored against two
Swahili translations at once (SUV and Biblia Habari Njema).

Baselines, so the fine-tune's numbers mean something:
  copy        output = the Sukuma input unchanged (credit for shared names and loanwords)
  nllb-as-swh base NLLB-600M told the Sukuma input is Swahili (what you get without fine-tuning)

Writes reports/translation.md / .json (scores only), reports/translation_udhr_samples.tsv
(UDHR is public domain) and data/raw/mt/predictions_*.tsv (Bible outputs, local only).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

BASE_MODEL = "facebook/nllb-200-distilled-600M"


def read_tsv(path: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_sets(data_dir: Path) -> dict[str, dict]:
    bible = read_tsv(data_dir / "test_bible.tsv")
    udhr = read_tsv(data_dir / "test_udhr.tsv")
    return {
        "bible": {"src": [r["suk"] for r in bible], "refs": [[r["swh_suv"] for r in bible], [r["swh_bhn"] for r in bible]],
                  "keys": [r["key"] for r in bible]},
        "udhr": {"src": [r["suk"] for r in udhr], "refs": [[r["swh"] for r in udhr]], "keys": [r["key"] for r in udhr]},
    }


def score(hyps: list[str], refs: list[list[str]]) -> dict:
    import sacrebleu

    return {"chrf": sacrebleu.corpus_chrf(hyps, refs, word_order=2).score,
            "bleu": sacrebleu.corpus_bleu(hyps, refs).score}


def format_report(results: dict, sets: dict) -> str:
    names = {"copy": "Copy the Sukuma input (floor)", "nllb-as-swh": "NLLB-600M, no fine-tuning",
             "fine-tuned": "NLLB-600M fine-tuned on Sukuma (ours)"}
    lines = [
        "# Sukuma -> Swahili translation",
        "",
        f"Held-out Bible books: {len(sets['bible']['src'])} verses (Ruth, Jonah, Philippians, James, 1 John), "
        "scored against two Swahili Bibles. UDHR: the 30 articles of the Universal Declaration of Human Rights, "
        "not Bible text and in plain spelling.",
        "",
        "| System | Bible chrF++ | Bible BLEU | UDHR chrF++ | UDHR BLEU |",
        "| --- | --- | --- | --- | --- |",
    ]
    for system, r in results.items():
        lines.append(f"| {names.get(system, system)} | {r['bible']['chrf']:.1f} | {r['bible']['bleu']:.1f} "
                     f"| {r['udhr']['chrf']:.1f} | {r['udhr']['bleu']:.1f} |")
    lines += ["", "chrF++ (0-100) counts matching character and word n-grams; BLEU counts matching word n-grams. "
              "Higher is better. A rough guide for chrF++: under 30 is mostly wrong, 40-50 conveys the gist, "
              "60+ is good.", ""]
    return "\n".join(lines)


def run(model_dir: Path | None, data_dir: Path, out_dir: Path, baselines: bool = False, translators=None) -> dict:
    """``translators`` maps system name -> callable(list[str]) -> list[str] (tests inject fakes)."""
    sets = load_sets(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if translators is None:
        from sukuma_toolkit.translate import Translator

        translators = {}
        if baselines:
            translators["copy"] = lambda xs: list(xs)
            translators["nllb-as-swh"] = Translator(BASE_MODEL, src_lang="swh_Latn").translate
        if model_dir:
            translators["fine-tuned"] = Translator(model_dir).translate
    results: dict[str, dict] = {}
    for system, translate in translators.items():
        print(f"Translating with {system} ...", flush=True)
        results[system] = {}
        for name, s in sets.items():
            hyps = translate(s["src"])
            results[system][name] = score(hyps, s["refs"])
            target = out_dir / "translation_udhr_samples.tsv" if name == "udhr" else data_dir / "predictions_bible.tsv"
            if system == "fine-tuned" or (system != "copy" and "fine-tuned" not in translators):
                with target.open("w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(["key", "sukuma", "reference", f"{system}_output"])
                    for k, src, ref, h in zip(s["keys"], s["src"], s["refs"][0], hyps):
                        w.writerow([k, src, ref, h])
        print(f"  {system}: " + ", ".join(f"{n} chrF++ {r['chrf']:.1f}" for n, r in results[system].items()))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "translation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "translation.md").write_text(format_report(results, sets), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="fine-tuned model directory (from train_translator.py)")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "mt")
    ap.add_argument("--baselines", action="store_true", help="also score the copy and no-fine-tuning baselines")
    args = ap.parse_args(argv)
    if not args.model and not args.baselines:
        ap.error("give --model and/or --baselines")
    results = run(args.model, args.data, ROOT / "reports", args.baselines)
    print(format_report(results, load_sets(args.data)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
