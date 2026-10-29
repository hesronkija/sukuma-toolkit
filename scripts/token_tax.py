#!/usr/bin/env python3
"""The Sukuma "token tax": how many more tokens LLM tokenizers spend on Sukuma.

    pip install -e ".[tokentax]"
    python scripts/token_tax.py                 # downloads tokenizer files only (a few MB each)

Uses the Universal Declaration of Human Rights in data/open/udhr/: the same
30 articles in Sukuma, Swahili and English, so every language says exactly
the same thing. For each tokenizer it counts tokens on the articles all
three languages share and reports:

- tokens per word (fertility): how many pieces an average word is cut into
- token tax: tokens for the Sukuma (or Swahili) text / tokens for the English text

A tax of 2.0 means the same content costs twice as many tokens: twice the
API price, half the usable context window, and (per the AfricaNLP 2026
"Token Tax" paper) usually lower accuracy.

If a Sukuma SentencePiece model from train_tokenizer.py is in models/, it is
added as a reference point.

Writes reports/token_tax.md, .json and .png.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UDHR = ROOT / "data" / "open" / "udhr"
LANGS = {"suk": "Sukuma", "swh": "Swahili", "eng": "English"}
HEADERS = {
    "suk": re.compile(r"^\s*Luhilili\s+(\d+)\s*$", re.I),
    "swh": re.compile(r"^\s*K[iu]fungu\s+(?:ch\w*\s+)?(\d+)\.?\s*$", re.I),  # the source has typos: "Kufungu", "ch"
    "eng": re.compile(r"^\s*Article\s+(\d+)\s*$", re.I),
}
TOKENIZERS = {
    "GPT-4o / o200k": "Xenova/gpt-4o",
    "GPT-4 / cl100k": "Xenova/gpt-4",
    "Llama 3": "unsloth/Llama-3.2-1B",
    "Qwen 2.5": "Qwen/Qwen2.5-7B-Instruct",
    "Gemma 3": "unsloth/gemma-3-1b-it",
    "XLM-R": "FacebookAI/xlm-roberta-base",
}
# The UDHR Sukuma text has no diacritics, so the matching model is the diacritics-removed one
SUKUMA_SPM = ROOT / "models" / "sukuma_unigram_8000_diacritics_removed.model"
OURS = "Sukuma SentencePiece (ours)"
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def parse_articles(path: Path, header: re.Pattern) -> dict[int, str]:
    """Article number -> text (header lines removed, whitespace collapsed)."""
    articles: dict[int, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = header.match(line)
        if m:
            current = int(m.group(1))
            articles.setdefault(current, [])
        elif current is not None and line.strip():
            articles[current].append(line.strip())
    return {k: " ".join(v) for k, v in articles.items() if v}


def load_parallel(udhr_dir: Path = UDHR) -> dict[str, str]:
    """Text per language, restricted to the articles every language has."""
    parsed = {lang: parse_articles(udhr_dir / f"{lang}.txt", HEADERS[lang]) for lang in LANGS}
    shared = sorted(set.intersection(*(set(p) for p in parsed.values())))
    if not shared:
        raise SystemExit(f"No shared UDHR articles found in {udhr_dir}")
    return {lang: "\n".join(parsed[lang][a] for a in shared) for lang in LANGS} | {"_articles": shared}


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def hf_counter(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    return lambda text: sum(len(tok.encode(line, add_special_tokens=False)) for line in text.splitlines())


def spm_counter(model_path: Path):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    return lambda text: sum(len(sp.encode(line)) for line in text.splitlines())


def measure(texts: dict[str, str], counters: dict) -> dict:
    words = {lang: count_words(texts[lang]) for lang in LANGS}
    results = {}
    for name, count in counters.items():
        tokens = {lang: count(texts[lang]) for lang in LANGS}
        results[name] = {
            "tokens": tokens,
            "tokens_per_word": {lang: tokens[lang] / words[lang] for lang in LANGS},
            "tax_vs_english": {lang: tokens[lang] / tokens["eng"] for lang in LANGS},
        }
    return {"articles": texts["_articles"], "words": words, "tokenizers": results}


def format_report(r: dict) -> str:
    w = r["words"]
    lines = [
        "# The Sukuma token tax",
        "",
        f"Same content in three languages: {len(r['articles'])} articles of the Universal Declaration of Human "
        f"Rights ({w['suk']:,} Sukuma, {w['swh']:,} Swahili, {w['eng']:,} English words).",
        "",
        "| Tokenizer | Tokens per word: Sukuma | Swahili | English | Sukuma tax vs English | Swahili tax vs English |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, t in r["tokenizers"].items():
        if name == OURS:
            continue
        tpw, tax = t["tokens_per_word"], t["tax_vs_english"]
        lines.append(f"| {name} | {tpw['suk']:.2f} | {tpw['swh']:.2f} | {tpw['eng']:.2f} "
                     f"| {tax['suk']:.2f}x | {tax['swh']:.2f}x |")
    lines += ["", "Tax = tokens needed for the same articles divided by tokens for English.", ""]
    if OURS in r["tokenizers"]:
        ours = r["tokenizers"][OURS]["tokens"]["suk"]
        others = {n: t["tokens"]["suk"] for n, t in r["tokenizers"].items() if n != OURS}
        best = min(others, key=others.get)
        lines += [f"Reference: our 8,000-piece Sukuma SentencePiece model (trained on Bible text, diacritics "
                  f"removed) needs {ours:,} tokens for the Sukuma text, {1 - ours / others[best]:.0%} fewer than the "
                  f"best LLM tokenizer here ({best}, {others[best]:,} tokens), even though those vocabularies are "
                  f"12 to 33 times larger.", ""]
    return "\n".join(lines)


def plot(r: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in r["tokenizers"] if n != OURS]
    colors = {"suk": "#c97c5d", "swh": "#61a5c2", "eng": "#2a6f97"}
    fig, ax = plt.subplots(figsize=(8, 3.9), dpi=150)
    width = 0.26
    for i, lang in enumerate(("eng", "swh", "suk")):
        xs = [j + (i - 1) * width for j in range(len(names))]
        ys = [r["tokenizers"][n]["tokens_per_word"][lang] for n in names]
        bars = ax.bar(xs, ys, width * 0.92, label=LANGS[lang], color=colors[lang])
        if lang == "suk":
            labels = [f"{r['tokenizers'][n]['tax_vs_english']['suk']:.1f}x" for n in names]
            ax.bar_label(bars, labels=labels, fontsize=7, padding=2)
    ax.set_xticks(range(len(names)), [n.replace(" / ", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Tokens per word")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.suptitle("The Sukuma token tax: same text, more tokens (label = Sukuma tokens / English tokens)",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    texts = load_parallel()
    counters = {name: hf_counter(mid) for name, mid in TOKENIZERS.items()}
    if SUKUMA_SPM.exists():
        counters[OURS] = spm_counter(SUKUMA_SPM)
    r = measure(texts, counters)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "token_tax.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    (args.out / "token_tax.md").write_text(format_report(r), encoding="utf-8")
    if not args.no_plot:
        plot(r, args.out / "token_tax.png")
    print(format_report(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
