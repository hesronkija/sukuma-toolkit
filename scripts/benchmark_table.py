#!/usr/bin/env python3
"""Collect every benchmark report into one table and chart.

    python scripts/benchmark_table.py          # after omnilingual_smoke_test.py / whisper_baseline.py runs

Reads reports/omnilingual_*.json and reports/whisper_*.json (the largest run
per model and dataset) and writes reports/benchmark_summary.md and
reports/benchmark_summary.png (word error rate by model, both test sets).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_NAMES = {"sukuma-voices-acl": "Sukuma Voices (Bible)", "sukuma-speech-corpus": "Speech Corpus (prompts)"}
PAPER_WER = 0.2519  # Mgonzo et al. 2026, Whisper Large v3 fine-tune, Sukuma Voices test


def model_label(model: str) -> str:
    if model.startswith("omniASR_"):
        return "Omnilingual " + model.removeprefix("omniASR_").removesuffix("_v2").replace("_", " ")
    if model == "sartifyllc/sukuma-voices-asr":
        return "Paper's Whisper fine-tune"
    return model


def collect(report_dir: Path) -> list[dict]:
    best: dict[tuple[str, str], dict] = {}
    for path in sorted(report_dir.glob("omnilingual_*.json")) + sorted(report_dir.glob("whisper_*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if not {"model", "dataset", "clips", "normalized"} <= set(r):
            continue
        key = (r["model"], r["dataset"])
        if key not in best or r["clips"] > best[key]["clips"]:
            best[key] = r
    order = ["omniASR_CTC_300M_v2", "omniASR_CTC_1B_v2", "omniASR_LLM_7B_v2", "sartifyllc/sukuma-voices-asr"]
    rank = {m: i for i, m in enumerate(order)}
    return sorted(best.values(), key=lambda r: (r["dataset"] != "sukuma-voices-acl", rank.get(r["model"], 99), r["model"]))


def format_table(results: list[dict]) -> str:
    lines = [
        "# Sukuma ASR benchmark",
        "",
        "Zero-shot unless noted. WER/CER after lowercasing and removing punctuation (diacritics kept); "
        "strict and diacritic-blind scores in the last columns.",
        "",
        "| Test set | Model | Clips | Hours | WER | CER | Strict WER | Diacritic-blind WER | Diacritic-blind CER |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {DATASET_NAMES.get(r['dataset'], r['dataset'])} | {model_label(r['model'])} | {r['clips']} "
            f"| {r['audio_hours']:.2f} | {r['normalized']['wer']:.1%} | {r['normalized']['cer']:.1%} "
            f"| {r['strict']['wer']:.1%} | {r['diacritic_blind']['wer']:.1%} | {r['diacritic_blind']['cer']:.1%} |"
        )
    lines += ["", f"Paper baseline (Whisper Large v3 fine-tuned on Sukuma Voices, its own scoring): "
              f"{PAPER_WER:.2%} WER on Sukuma Voices. Meta reports 2.0% CER for Sukuma (7B LLM model, Meta's own "
              "test data).", ""]
    return "\n".join(lines)


def plot(results: list[dict], path: Path) -> None:
    """Two panels: word error rate and character error rate, per test set and model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [d for d in DATASET_NAMES if any(r["dataset"] == d for r in results)]
    models = list(dict.fromkeys(r["model"] for r in results))
    colors = ["#2a6f97", "#61a5c2", "#c97c5d", "#8a5a83", "#6a994e"]
    width = min(0.45, 0.8 / max(1, len(models)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    for ax, metric, label in ((axes[0], "wer", "Word error rate (%)"), (axes[1], "cer", "Character error rate (%)")):
        for i, m in enumerate(models):
            xs, ys = [], []
            for j, d in enumerate(datasets):
                r = next((r for r in results if r["model"] == m and r["dataset"] == d), None)
                if r:
                    xs.append(j + (i - (len(models) - 1) / 2) * width)
                    ys.append(r["normalized"][metric] * 100)
            bars = ax.bar(xs, ys, width * 0.92, label=model_label(m), color=colors[i % len(colors)])
            ax.bar_label(bars, fmt="%.0f%%" if metric == "wer" else "%.1f%%", fontsize=7, padding=2)
        if metric == "wer" and "sukuma-voices-acl" in datasets:
            j = datasets.index("sukuma-voices-acl")
            ax.hlines(PAPER_WER * 100, j - 0.4, j + 0.4, colors="#333333", linestyles="--", linewidth=1)
            ax.text(j, PAPER_WER * 100 + 2, "paper: 25%", ha="center", fontsize=7, color="#333333")
        if metric == "cer" and "sukuma-voices-acl" in datasets:
            j = datasets.index("sukuma-voices-acl")
            ax.hlines(2.0, j - 0.4, j + 0.4, colors="#333333", linestyles=":", linewidth=1)
            ax.text(j - 0.4, 3.0, "Meta's claim: 2%", fontsize=7, color="#333333")
        ax.set_xticks(range(len(datasets)), [DATASET_NAMES[d] for d in datasets], fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(0, 105 if metric == "wer" else 32)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper center", ncol=len(models),
               bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Sukuma speech recognition (lowercase, no punctuation; lower is better)", fontsize=10,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    results = collect(args.reports)
    if not results:
        raise SystemExit("No benchmark reports found. Run scripts/omnilingual_smoke_test.py first.")
    (args.reports / "benchmark_summary.md").write_text(format_table(results), encoding="utf-8")
    if not args.no_plot:
        plot(results, args.reports / "benchmark_summary.png")
    print(format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
