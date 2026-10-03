#!/usr/bin/env python3
"""Download the Sukuma Voices speech corpus (Pillar B).

Dataset: sartifyllc/Sukuma-Voices-ACL on the HuggingFace Hub (7.47h of
transcribed Sukuma Bible readings; the Hub card says Apache-2.0, the project's
GitHub repo says CC BY 4.0, so attribute either way). Downloads to
data/sukuma-voices/ and is never committed to git (see data/README.md).
For the other speech corpora see data/SOURCES.md and scripts/fetch_data.py.

Requires: pip install 'sukuma-toolkit[audio]'  (datasets + librosa + soundfile)
Paper: https://aclanthology.org/2026.loreslm-1.25/
"""

from __future__ import annotations

from pathlib import Path

REPO_ID = "sartifyllc/Sukuma-Voices-ACL"
OUT = Path(__file__).resolve().parent.parent / "data" / "sukuma-voices"


def main() -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "Need the 'datasets' package: pip install 'sukuma-toolkit[audio]'"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    for split in (
        "train",
        "test",
        "test_indistribution_synthesis",
        "test_outdistribution_synthesis",
    ):
        print(f"Downloading split: {split} ...")
        ds = load_dataset(REPO_ID, split=split)
        ds.save_to_disk(str(OUT / split))
        print(f"  {len(ds):,} examples -> {OUT / split}")
    print("Done. Attribute: Sukuma Voices (Mgonzo et al., LoResLM 2026).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
