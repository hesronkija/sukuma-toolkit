#!/usr/bin/env python3
"""Build data/sukuma_sentences.csv from the AfriSpeech Sukuma Bible text (Pillar A).

Source: the AfriSpeech/africa-corpus dataset on the Hugging Face Hub, read
through the africa-bitext-builder package (PyPI). Sukuma has three versions:

    1512  SNT00    Sukuma New Testament 2000
    1517  SUKBI15  Bibilia Ilagano Lya Kale 2015
    2684  SUK60    Ilagano Ipya Lya

License: the builder's own license table marks all three as copyrighted, NOT
public domain. Its public-domain-only default therefore returns no Sukuma
text at all, so this script names the version IDs explicitly. The output is
for local research use only; it is gitignored and must never be committed
(see data/README.md).

If `python scripts/fetch_data.py --only africa-corpus-bible` has already
downloaded the raw files to data/raw/africa-corpus-bible/, they are used
offline; otherwise the builder downloads them from the Hub.

Output: a CSV with a single ``suk`` column, one deduplicated verse per row in
canonical book/chapter/verse order (the format corpus.load_afrispeech_text reads).

Requires: pip install africa-bitext-builder huggingface_hub
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sukuma_sentences.csv"
LOCAL_RAW = ROOT / "data" / "raw" / "africa-corpus-bible"
LANG = "suk"
SUKUMA_VERSION_IDS = (1512, 1517, 2684)


def _load_builder():
    try:
        from africa_bitext_builder.builder import CorpusBuilder
    except ImportError:
        print("Installing africa-bitext-builder ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "africa-bitext-builder>=0.1.13"])
        from africa_bitext_builder.builder import CorpusBuilder
    return CorpusBuilder


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--versions",
        default=",".join(map(str, SUKUMA_VERSION_IDS)),
        help="comma-separated Bible version IDs (default: all three Sukuma versions)",
    )
    ap.add_argument("--out", type=Path, default=OUT, help="output CSV (default: data/sukuma_sentences.csv)")
    args = ap.parse_args(argv)

    version_ids = [int(v) for v in args.versions.split(",") if v.strip()]
    CorpusBuilder = _load_builder()

    print("Note: all Sukuma Bible versions are copyrighted. Local research use only; never commit the output.")
    print(f"Building Sukuma ({LANG}) monolingual corpus from versions {version_ids} ...")
    builder = CorpusBuilder(
        source_lang=LANG,
        mode="monolingual",
        source_version_ids=version_ids,
        data_root=str(LOCAL_RAW),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    path = Path(builder.download(args.out))

    with path.open(encoding="utf-8", newline="") as f:
        n = sum(1 for _ in csv.reader(f)) - 1  # minus header
    print(f"Wrote {n:,} sentences -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
