#!/usr/bin/env python3
"""Reproduce the Sukuma Voices paper baseline (Whisper Large v3 fine-tune) with the same scoring.

    pip install -e ".[whisper]"
    python scripts/whisper_baseline.py -n 362                     # full Sukuma Voices test split
    python scripts/whisper_baseline.py --dataset sukuma-speech-corpus -n 400

Runs sartifyllc/sukuma-voices-asr (6.2 GB) on the same clips as
omnilingual_smoke_test.py and scores it at the same three levels, so the
two tables line up. Use a GPU: on an RTX 3060 (12 GB) the 362 clips take a
few minutes in float16; on a laptop CPU expect hours.

Caveat: the model card lists both Sukuma-Voices-ACL and the larger Sukuma
Voices release as training data. If the larger release contains the ACL
test clips, this model has heard the test audio, and its score is a
ceiling, not a fair comparison. Say so when reporting it.

Writes reports/whisper_<dataset>_<model>_<n>.md / .tsv / .json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from omnilingual_smoke_test import DATASETS, LEVELS, load_corpus, load_voices  # noqa: E402

from sukuma_toolkit.evaluate import score_levels  # noqa: E402

DEFAULT_MODEL = "sartifyllc/sukuma-voices-asr"
TARGET_SR = 16_000


def resample(wav, sr: int, target: int = TARGET_SR):
    if sr == target:
        return wav
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, target)
    return resample_poly(wav, target // g, sr // g).astype("float32")


def make_pipeline(model_id: str, device: str | None = None):
    import torch
    from transformers import pipeline

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    return pipeline("automatic-speech-recognition", model=model_id, torch_dtype=dtype, device=device)


def transcribe(pipe, clips: list[dict], batch_size: int = 8, language: str | None = None) -> list[str]:
    inputs = [{"raw": resample(c["waveform"], c["sample_rate"]), "sampling_rate": TARGET_SR} for c in clips]
    kwargs = {"task": "transcribe", "max_new_tokens": 200}  # cap hallucination loops on hard clips
    if language:
        kwargs["language"] = language
    out = pipe(inputs, batch_size=batch_size, generate_kwargs=kwargs)
    return [o["text"].strip() for o in out]


def write_report(clips: list[dict], hyps: list[str], model_id: str, dataset: str, seconds: float,
                 out_dir: Path) -> dict:
    refs = [c["text"] for c in clips]
    scores = score_levels(refs, hyps)
    audio_hours = sum(len(c["waveform"]) / c["sample_rate"] for c in clips) / 3600
    slug = model_id.replace("/", "_")
    stem = f"whisper_{dataset}_{slug}_{len(clips)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{stem}.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["clip", "reference", "hypothesis"])
        for c, h in zip(clips, hyps):
            w.writerow([c.get("id", ""), c["text"], h])
    rows = [f"| {label} | {scores[key]['wer']:.1%} | {scores[key]['cer']:.1%} |" for key, label in LEVELS]
    md = "\n".join([
        f"# Whisper baseline on Sukuma: {model_id}",
        "",
        f"{len(clips)} clips ({audio_hours:.2f} h of audio) from the {DATASETS.get(dataset, dataset)} test split; "
        f"transcription took {seconds:.0f} s.",
        "",
        "| Scoring | WER | CER |",
        "| --- | --- | --- |",
        *rows,
        "",
        "The paper reports 25.19% WER for this model on Sukuma Voices. The model card lists the larger "
        "Sukuma Voices release as training data too, so test clips may have been seen in training.",
        "",
    ])
    (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
    result = {"model": model_id, "dataset": dataset, "clips": len(clips), "audio_hours": audio_hours,
              "seconds": seconds, **scores}
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=20, help="number of test clips (default 20)")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="sukuma-voices-acl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--language", help="force a Whisper language token, e.g. sw (default: model decides)")
    ap.add_argument("--device", help="cuda, cpu or mps (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (sukuma-speech-corpus)")
    args = ap.parse_args(argv)

    print(f"Loading {args.n} clips from {args.dataset} ...")
    clips = load_voices(args.n) if args.dataset == "sukuma-voices-acl" else load_corpus(args.n, args.seed)
    print(f"Loading {args.model} (first run downloads 6.2 GB) ...")
    pipe = make_pipeline(args.model, args.device)
    t0 = time.perf_counter()
    hyps = transcribe(pipe, clips, args.batch_size, args.language)
    r = write_report(clips, hyps, args.model, args.dataset, time.perf_counter() - t0, ROOT / "reports")
    for key, label in LEVELS:
        print(f"{label:45} WER {r[key]['wer']:6.1%}  CER {r[key]['cer']:6.1%}")
    print(f"Report: reports/whisper_{args.dataset}_{args.model.replace('/', '_')}_{r['clips']}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
