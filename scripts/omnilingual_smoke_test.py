#!/usr/bin/env python3
"""Benchmark Meta's Omnilingual ASR on Sukuma test data.

    pip install -e ".[omnilingual]"         # macOS also: brew install libsndfile
    python scripts/omnilingual_smoke_test.py                                  # 20 Sukuma Voices clips
    python scripts/omnilingual_smoke_test.py -n 362                           # full Sukuma Voices test split
    python scripts/omnilingual_smoke_test.py --dataset sukuma-speech-corpus -n 400   # seeded sample
    python scripts/omnilingual_smoke_test.py --model omniASR_CTC_1B_v2

Datasets:
  sukuma-voices-acl     sartifyllc/Sukuma-Voices-ACL test split (362 Bible-reading clips), first N
  sukuma-speech-corpus  joseph-telemala/sukuma-speech-corpus test split (2,410 clips), seeded random N;
                        only the sampled audio files are downloaded

Scores word and character error rates at three levels: strict (exact text),
normalized (lowercase, no punctuation, diacritics kept) and diacritic-blind.
Writes reports/omnilingual_<dataset>_<model>_<n>.md and a per-clip TSV.

Context: Meta's per-language table reports suk_Latn trained on 20.2 hours
with 2.0% CER (7B LLM model, Meta's own test data). Meta's training data may
overlap with these test sets; say so when reporting results.
Clips longer than 40 s are skipped (the pipeline's limit).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sukuma_toolkit.evaluate import score_both_ways  # noqa: E402

MAX_SECONDS = 40.0
TARGET_SR = 16_000  # the models' rate; resampling on load halves memory for 32 kHz audio
CHUNK = 50  # clips per transcribe() call, to bound peak memory on small machines
DATASETS = {
    "sukuma-voices-acl": "sartifyllc/Sukuma-Voices-ACL",
    "sukuma-speech-corpus": "joseph-telemala/sukuma-speech-corpus",
}
LEVELS = (
    ("strict", "Strict (exact text)"),
    ("normalized", "Lowercase, no punctuation, diacritics kept"),
    ("diacritic_blind", "Diacritics also ignored"),
)


def to_16k(wav, sr: int):
    """Mono float32 at 16 kHz (no-op when already 16 kHz)."""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr == TARGET_SR:
        return wav.astype("float32", copy=False), sr
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr), TARGET_SR)
    return resample_poly(wav, TARGET_SR // g, int(sr) // g).astype("float32"), TARGET_SR


def load_voices(n: int) -> list[dict]:
    """First n clips (<= 40 s) of the Sukuma Voices ACL test split.

    Reads the split's single parquet file directly (no `datasets` dependency,
    which conflicts with omnilingual-asr's NumPy pin) and decodes the audio bytes.
    """
    import io

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASETS["sukuma-voices-acl"], "data/test-00000-of-00001.parquet", repo_type="dataset",
                           local_dir=ROOT / "data" / "raw" / "sukuma-voices-acl")
    table = pq.read_table(path, columns=["audio", "text"]).to_pylist()
    clips = []
    for i, ex in enumerate(table):
        wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if len(wav) / sr > MAX_SECONDS:
            continue
        clips.append({"id": f"test-{i}", "waveform": wav, "sample_rate": sr, "text": ex["text"]})
        if len(clips) >= n:
            break
    return clips


def sample_corpus_rows(metadata_path: Path, n: int, seed: int = 42) -> list[dict]:
    """Seeded random sample of n test rows (<= 40 s) from the corpus metadata."""
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if float(r["duration_seconds"]) <= MAX_SECONDS]
    rows.sort(key=lambda r: r["utterance_id"])
    return random.Random(seed).sample(rows, min(n, len(rows)))


def load_corpus(n: int, seed: int = 42) -> list[dict]:
    """Seeded sample from the Sukuma Speech Corpus test split; downloads only those files."""
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    local = ROOT / "data" / "raw" / "sukuma-speech-corpus"
    meta = local / "test" / "metadata.jsonl"
    if not meta.exists():
        hf_hub_download(DATASETS["sukuma-speech-corpus"], "test/metadata.jsonl", repo_type="dataset", local_dir=local)
    clips = []
    for r in sample_corpus_rows(meta, n, seed):
        path = hf_hub_download(DATASETS["sukuma-speech-corpus"], f"test/{r['file_name']}",
                               repo_type="dataset", local_dir=local)
        wav, sr = to_16k(*sf.read(path, dtype="float32"))
        clips.append({"id": r["utterance_id"], "waveform": wav, "sample_rate": sr, "text": r["text"],
                      "gender": r.get("gender", ""), "participant": r.get("participant_record_id", "")})
    return clips


def make_pipeline(model_card: str):
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    return ASRInferencePipeline(model_card=model_card)


def run(clips: list[dict], pipeline, model_card: str, out_dir: Path, batch_size: int = 4,
        dataset: str = "sukuma-voices-acl") -> dict:
    t0 = time.perf_counter()
    hyps: list[str] = []
    for start in range(0, len(clips), CHUNK):
        part = clips[start:start + CHUNK]
        inputs = [{"waveform": c["waveform"], "sample_rate": c["sample_rate"]} for c in part]
        lang = None if "CTC" in model_card else ["suk_Latn"] * len(inputs)  # CTC ignores language codes
        hyps += list(pipeline.transcribe(inputs, lang=lang, batch_size=batch_size))
    seconds = time.perf_counter() - t0
    refs = [c["text"] for c in clips]
    scores = score_both_ways(refs, hyps)
    audio_hours = sum(len(c["waveform"]) / c["sample_rate"] for c in clips) / 3600

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"omnilingual_{dataset}_{model_card}_{len(clips)}"
    with (out_dir / f"{stem}.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["clip", "reference", "hypothesis"])
        for c, h in zip(clips, hyps):
            w.writerow([c.get("id", ""), c["text"], h])
    rows = [f"| {label} | {scores[key]['wer']:.1%} | {scores[key]['cer']:.1%} |" for key, label in LEVELS]
    md = "\n".join([
        f"# Omnilingual ASR on Sukuma: {model_card}",
        "",
        f"{len(clips)} clips ({audio_hours:.2f} h of audio) from the {DATASETS.get(dataset, dataset)} test split; "
        f"transcription took {seconds:.0f} s.",
        "",
        "| Scoring | WER | CER |",
        "| --- | --- | --- |",
        *rows,
        "",
        "Reference points: Sukuma Voices paper baseline (Whisper Large v3 fine-tune), 25.19% WER. "
        "Meta reports 2.0% CER for suk_Latn on its own test data (7B LLM model).",
        "",
    ])
    (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
    result = {"model": model_card, "dataset": dataset, "clips": len(clips), "audio_hours": audio_hours,
              "seconds": seconds, **scores}
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=20, help="number of test clips (default 20)")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="sukuma-voices-acl")
    ap.add_argument("--model", default="omniASR_CTC_300M_v2", help="Omnilingual model card")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (sukuma-speech-corpus)")
    args = ap.parse_args(argv)

    print(f"Loading {args.n} clips from {args.dataset} ...")
    clips = load_voices(args.n) if args.dataset == "sukuma-voices-acl" else load_corpus(args.n, args.seed)
    print(f"Loading {args.model} (first run downloads the weights) ...")
    r = run(clips, make_pipeline(args.model), args.model, ROOT / "reports", args.batch_size, args.dataset)
    for key, label in LEVELS:
        print(f"{label:45} WER {r[key]['wer']:6.1%}  CER {r[key]['cer']:6.1%}")
    print(f"Report: reports/omnilingual_{args.dataset}_{args.model}_{r['clips']}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
