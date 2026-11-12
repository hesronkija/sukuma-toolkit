#!/usr/bin/env python3
"""Build Sukuma -> Swahili translation data: Bible verse pairs plus the UDHR.

    pip install -e ".[translate]"
    python scripts/build_translation_data.py

Training data: every Sukuma Bible (3 versions) paired verse by verse with two
Swahili Bibles (SUV 1952-style Union Version and Biblia Habari Njema), from
AfriSpeech/africa-corpus via africa-bitext-builder. ALL of these Bibles are
copyrighted: the output is for local research only, is written under
data/raw/ (gitignored), and must never be committed or published, and nor
may a model trained on it.

Splits are by whole BOOK, so no test verse (in any version) is seen in
training:

  test: Ruth, Jonah, Philippians, James, 1 John
  dev:  Titus, Zephaniah
  train: everything else

Each test verse keeps both Swahili versions as references (multi-reference
scoring). The UDHR (data/open/udhr, public domain) gives a second,
non-Bible test set: the same 30 articles in Sukuma and Swahili.

Writes data/raw/mt/train.tsv, dev.tsv, test_bible.tsv, test_udhr.tsv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sukuma_toolkit.normalize import normalize_text  # noqa: E402

SUKUMA_VERSIONS = (1512, 1517, 2684)
SWAHILI_VERSIONS = (164, 74)  # SUV, Biblia Habari Njema
TEST_BOOKS = {"RUT", "JON", "PHP", "JAS", "1JN"}
DEV_BOOKS = {"TIT", "ZEP"}
TEST_SOURCE_VERSION = 1517  # the full Sukuma Bible: covers every test book


def split_of(verse_key: str) -> str:
    book = verse_key.split(".")[0]
    return "test" if book in TEST_BOOKS else "dev" if book in DEV_BOOKS else "train"


def build_pairs(out_dir: Path, data_root: Path) -> dict[tuple[int, int], Path]:
    """One CSV per (Sukuma version, Swahili version), via africa-bitext-builder."""
    from africa_bitext_builder.builder import CorpusBuilder

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for sv in SUKUMA_VERSIONS:
        for tv in SWAHILI_VERSIONS:
            path = out_dir / f"suk{sv}_swh{tv}.csv"
            if not path.exists():
                print(f"Building Sukuma {sv} x Swahili {tv} ...", flush=True)
                CorpusBuilder(source_lang="suk", target_lang="swh", source_version_ids=[sv], target_version_ids=[tv],
                              mode="parallel", data_root=str(data_root)).download(str(path))
            paths[(sv, tv)] = path
    return paths


def read_pairs(path: Path) -> list[tuple[str, str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [(r["verse_key"], normalize_text(r["suk"]), normalize_text(r["swh"])) for r in csv.DictReader(f)
                if r.get("suk", "").strip() and r.get("swh", "").strip()]


def split_pairs(pairs_by_version: dict[tuple[int, int], list[tuple[str, str, str]]]) -> dict[str, list]:
    """Train/dev: every version pairing. Test: one Sukuma source, both Swahili references."""
    out: dict[str, list] = {"train": [], "dev": []}
    seen: dict[str, set] = {"train": set(), "dev": set()}
    refs: dict[str, dict[int, str]] = defaultdict(dict)
    sources: dict[str, str] = {}
    for (sv, tv), pairs in sorted(pairs_by_version.items()):
        for key, suk, swh in pairs:
            split = split_of(key)
            if split == "test":
                if sv == TEST_SOURCE_VERSION:
                    sources[key] = suk
                    refs[key][tv] = swh
                continue
            if (suk, swh) not in seen[split]:  # identical wording across versions adds nothing
                seen[split].add((suk, swh))
                out[split].append((key, suk, swh))
    out["test"] = [(k, sources[k], *(refs[k].get(tv, "") for tv in SWAHILI_VERSIONS))
                   for k in sorted(sources) if all(refs[k].get(tv) for tv in SWAHILI_VERSIONS)]
    return out


def udhr_pairs(udhr_dir: Path) -> list[tuple[str, str, str]]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from token_tax import HEADERS, parse_articles

    suk = parse_articles(udhr_dir / "suk.txt", HEADERS["suk"])
    swh = parse_articles(udhr_dir / "swh.txt", HEADERS["swh"])
    return [(f"UDHR.{a}", normalize_text(suk[a]), normalize_text(swh[a])) for a in sorted(set(suk) & set(swh))]


def write_tsv(path: Path, header: list[str], rows: list[tuple]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(header)
        w.writerows(rows)


def run(out_dir: Path, data_root: Path, udhr_dir: Path) -> dict[str, int]:
    paths = build_pairs(out_dir / "pairs", data_root)
    splits = split_pairs({k: read_pairs(p) for k, p in paths.items()})
    write_tsv(out_dir / "train.tsv", ["key", "suk", "swh"], splits["train"])
    write_tsv(out_dir / "dev.tsv", ["key", "suk", "swh"], splits["dev"])
    write_tsv(out_dir / "test_bible.tsv", ["key", "suk", "swh_suv", "swh_bhn"], splits["test"])
    udhr = udhr_pairs(udhr_dir)
    write_tsv(out_dir / "test_udhr.tsv", ["key", "suk", "swh"], udhr)
    counts = {"train": len(splits["train"]), "dev": len(splits["dev"]), "test_bible": len(splits["test"]),
              "test_udhr": len(udhr)}
    print(counts)
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "mt")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "africa-corpus-bible")
    args = ap.parse_args(argv)
    print("Note: every Bible used here is copyrighted. Local research use only; never commit the output.")
    run(args.out, args.data_root, ROOT / "data" / "open" / "udhr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
