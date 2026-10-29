#!/usr/bin/env python3
"""Audit web text tagged as Sukuma: how much is really Sukuma, and how much is Bible?

    pip install -e ".[audit]"
    python scripts/fetch_data.py                   # web sources -> data/raw/<id>/text.tsv
    python scripts/download_afrispeech_text.py     # Bible sentences (local only), for the Bible check
    python scripts/audit_web_text.py               # first run downloads GlotLID (1.7 GB)

FineWeb-2 and its relatives tag whole DOCUMENTS as Sukuma. This script
re-labels every SENTENCE with GlotLID (cis-lmu/glotlid, the language-ID
model behind GlotCC and FineWeb-2), so mixed documents show up: Sukuma
stories printed next to their Swahili translation, English front matter,
tables and OCR noise.

Labels: Sukuma (suk_Latn), Swahili (swh_Latn, swc_Latn), English
(eng_Latn), other (any other language), unclear (top score below 0.5).
A Sukuma sentence whose word 6-grams mostly appear in the local Bible
sentences is counted as Bible text.

Writes reports/web_audit.md and .json (counts only, no source text).
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sukuma_toolkit.normalize import normalize_text, strip_diacritics  # noqa: E402

SOURCES = ("fineweb2-suk", "dcad2000-suk", "finepdfs-suk", "glotcc-suk")
LABELS = ("sukuma", "swahili", "english", "other", "unclear")
NGRAM = 6

_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
_SENT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


def words(text: str) -> list[str]:
    return _WORD_RE.findall(strip_diacritics(normalize_text(text)).lower())


def sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENT_RE.split(text)) if len(words(s)) >= 3]


GROUPS = {"suk_Latn": "sukuma", "swh_Latn": "swahili", "swc_Latn": "swahili", "eng_Latn": "english"}
MIN_SCORE = 0.5


def load_lid(model_path: Path | None = None):
    """GlotLID fastText model; returns predict(text) -> (label, score)."""
    import fasttext
    from huggingface_hub import hf_hub_download

    path = model_path or hf_hub_download("cis-lmu/glotlid", "model.bin",
                                         local_dir=ROOT / "models" / "glotlid")
    model = fasttext.load_model(str(path))

    def predict(text: str) -> tuple[str, float]:
        labels, scores = model.predict(text.replace("\n", " "), k=1)
        return labels[0].removeprefix("__label__"), float(scores[0])

    return predict


def label_sentence(text: str, predict) -> str:
    code, score = predict(text)
    if score < MIN_SCORE:
        return "unclear"
    return GROUPS.get(code, "other")


def ngrams(tokens: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def is_bible(tokens: list[str], bible_ngrams: set, n: int = NGRAM) -> bool:
    grams = ngrams(tokens, n)
    return bool(grams) and sum(g in bible_ngrams for g in grams) / len(grams) >= 0.5


def audit_texts(docs: list[str], predict, bible_ngrams: set) -> dict:
    by_label = collections.Counter()
    sents_by_label = collections.Counter()
    bible_words = 0
    docs_majority = collections.Counter()
    for doc in docs:
        doc_words = collections.Counter()
        for s in sentences(doc):
            toks = words(s)
            lab = label_sentence(s, predict)
            by_label[lab] += len(toks)
            sents_by_label[lab] += 1
            doc_words[lab] += len(toks)
            if lab == "sukuma" and is_bible(toks, bible_ngrams):
                bible_words += len(toks)
        if doc_words:
            docs_majority[doc_words.most_common(1)[0][0]] += 1
    total = sum(by_label.values())
    return {
        "documents": len(docs),
        "words": total,
        "word_share": {k: (by_label[k] / total if total else 0.0) for k in LABELS},
        "sentences": {k: sents_by_label[k] for k in LABELS},
        "documents_mostly": {k: docs_majority[k] for k in LABELS},
        "sukuma_words_from_bible": (bible_words / by_label["sukuma"]) if by_label["sukuma"] else 0.0,
    }


def read_tsv_texts(path: Path) -> list[str]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as f:
        return [row["text"] for row in csv.DictReader(f, delimiter="\t") if row.get("text")]


def read_bible(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [row[0] for row in csv.reader(f)][1:]


def format_report(results: dict, have_bible: bool) -> str:
    lines = [
        "# Web text tagged as Sukuma: what is actually in it",
        "",
        "Every sentence re-labeled with GlotLID (`scripts/audit_web_text.py`). Shares are by word count.",
        "",
        "| Source | Documents | Words | Sukuma | Swahili | English | Other | Unclear | Sukuma words that are Bible text |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for src, r in results.items():
        ws = r["word_share"]
        bible = f"{r['sukuma_words_from_bible']:.0%}" if have_bible else "n/a"
        lines.append(f"| {src} | {r['documents']:,} | {r['words']:,} | {ws['sukuma']:.0%} | {ws['swahili']:.0%} "
                     f"| {ws['english']:.0%} | {ws['other']:.0%} | {ws['unclear']:.0%} | {bible} |")
    lines += ["", "Unclear: GlotLID's top score is below 0.5 (names, numbers, tables, OCR noise, very short lines). "
              "Other: any other language, often a close Bantu neighbor; spot-check before trusting it.", ""]
    return "\n".join(lines)


def run(raw_dir: Path, bible_csv: Path, out_dir: Path, predict=None, sources=SOURCES) -> dict:
    predict = predict or load_lid()
    bible_ngrams: set = set()
    bible = read_bible(bible_csv)
    for s in bible:
        bible_ngrams |= ngrams(words(s))
    results = {}
    for src in sources:
        path = raw_dir / src / "text.tsv"
        if path.exists():
            results[src] = audit_texts(read_tsv_texts(path), predict, bible_ngrams)
    if not results:
        raise SystemExit(f"No web text under {raw_dir}. Run: python scripts/fetch_data.py")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"method": "GlotLID v3 per sentence", "bible_reference": bool(bible), "sources": results}
    (out_dir / "web_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "web_audit.md").write_text(format_report(results, bool(bible)), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--bible", type=Path, default=ROOT / "data" / "sukuma_sentences.csv")
    ap.add_argument("--model", type=Path, help="path to a local GlotLID model.bin")
    args = ap.parse_args(argv)
    r = run(args.raw_dir, args.bible, ROOT / "reports", load_lid(args.model))
    for src, s in r["sources"].items():
        ws = s["word_share"]
        print(f"{src:14} {s['words']:>9,} words  Sukuma {ws['sukuma']:4.0%}  Swahili {ws['swahili']:4.0%}  "
              f"English {ws['english']:4.0%}  other {ws['other']:4.0%}  unclear {ws['unclear']:4.0%}")
    print("Report: reports/web_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
