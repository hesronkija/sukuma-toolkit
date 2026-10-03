#!/usr/bin/env python3
"""Rebuild data/open/: the small, openly licensed Sukuma data committed to git.

Every committed file is extracted from a pinned upstream commit listed in
data/catalog.json (sources with ``"in_repo": true``), so the output is
reproducible byte for byte. Run it again after bumping a commit pin, then
commit the regenerated files and data/open/MANIFEST.json.

    python scripts/build_open_data.py            # download (cached) + rebuild
    python scripts/build_open_data.py --check    # rebuild into a temp dir and
                                                 # compare with what's committed

Needs network access to raw.githubusercontent.com only. Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog.json"
OPEN_DIR = ROOT / "data" / "open"
CACHE_DIR = ROOT / "data" / "raw" / "_upstream_cache"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

Fetch = Callable[[str], Path]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_fetcher(repo: str, commit: str, cache_dir: Path = CACHE_DIR) -> Fetch:
    """Return fetch(path) -> local Path for a file at a pinned GitHub commit."""

    def fetch(path: str) -> Path:
        dest = cache_dir / repo.replace("/", "__") / commit / path
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        print(f"  downloading {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=300) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dest)
        return dest

    return fetch


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> int:
    """Write a UTF-8, LF-terminated TSV. Tabs/newlines inside cells become spaces."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(value: object) -> str:
        return re.sub(r"[\t\r\n]+", " ", "" if value is None else str(value)).strip()

    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(clean(v) for v in row) + "\n")
    return len(rows)


def bib_subset(bib_text: str, keys: set[str]) -> str:
    """Return the BibTeX entries whose citation keys are in ``keys``."""
    entries = re.split(r"\n(?=@)", "\n" + bib_text)
    keep = []
    for entry in entries:
        m = re.match(r"\s*@\w+\s*\{\s*([^,\s]+)\s*,", entry)
        if m and m.group(1) in keys:
            keep.append(entry.strip())
    return "\n\n".join(sorted(keep)) + ("\n" if keep else "")


def split_sources(value: str) -> set[str]:
    """CLDF Source cells look like 'key1;key2[12-14]'. Return bare keys."""
    return {re.sub(r"\[.*?\]$", "", s.strip()) for s in value.split(";") if s.strip()}


# --------------------------------------------------------------------------
# extractors: one per catalog id, each returns the files it wrote
# --------------------------------------------------------------------------

def extract_udhr(fetch: Fetch, out: Path) -> list[Path]:
    zpath = fetch("packages/corpora/udhr2.zip")
    written = []
    with zipfile.ZipFile(zpath) as z:
        for code in ("suk", "swh", "eng"):
            data = z.read(f"udhr2/{code}.txt").decode("utf-8")
            data = data.replace("\r\n", "\n")
            if not data.endswith("\n"):
                data += "\n"
            dest = out / "udhr" / f"{code}.txt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(data, encoding="utf-8")
            written.append(dest)
    return written


def extract_asjp(fetch: Fetch, out: Path) -> list[Path]:
    params = {p["ID"]: p for p in read_csv(fetch("cldf/parameters.csv"))}
    forms = [r for r in read_csv(fetch("cldf/forms.csv")) if r["Language_ID"] == "F21_SUKUMA"]
    forms.sort(key=lambda r: (int(r["Parameter_ID"]), r["ID"]))
    rows = [
        [
            r["Parameter_ID"],
            params[r["Parameter_ID"]]["Concepticon_Gloss"],
            r.get("gloss_in_source", ""),
            r["Form"],
            r["Segments"],
            r["Loan"],
            r["Source"],
        ]
        for r in forms
    ]
    dest = out / "lexicon" / "asjp_sukuma.tsv"
    write_tsv(
        dest,
        ["concept_id", "concepticon_gloss", "gloss_in_source", "form_asjpcode", "segments", "loan", "source"],
        rows,
    )
    keys = set().union(*(split_sources(r["Source"]) for r in forms)) if forms else set()
    bib = out / "lexicon" / "asjp_sukuma.bib"
    bib.write_text(bib_subset(fetch("cldf/sources.bib").read_text(encoding="utf-8"), keys), encoding="utf-8")
    return [dest, bib]


def extract_grollemund(fetch: Fetch, out: Path) -> list[Path]:
    params = {p["ID"]: p for p in read_csv(fetch("cldf/parameters.csv"))}
    variety = {"f21sukuma": "Sukuma", "f21ntuzu": "Ntuzu"}
    forms = [r for r in read_csv(fetch("cldf/forms.csv")) if r["Language_ID"] in variety]
    forms.sort(key=lambda r: (r["Language_ID"] != "f21sukuma", r["Parameter_ID"], r["ID"]))
    rows = [
        [
            variety[r["Language_ID"]],
            r["Language_ID"],
            r["Parameter_ID"],
            params[r["Parameter_ID"]]["Concepticon_Gloss"],
            params[r["Parameter_ID"]]["Concepticon_ID"],
            r["Value"],
            r["Form"],
            r["Segments"],
            r["Cognacy"],
            r["Source"],
        ]
        for r in forms
    ]
    dest = out / "lexicon" / "grollemund_sukuma.tsv"
    write_tsv(
        dest,
        ["variety", "language_id", "concept_id", "concepticon_gloss", "concepticon_id",
         "value", "form", "segments", "cognate_set", "source"],
        rows,
    )
    keys = set().union(*(split_sources(r["Source"]) for r in forms)) if forms else set()
    bib = out / "lexicon" / "grollemund_sukuma.bib"
    bib.write_text(bib_subset(fetch("cldf/sources.bib").read_text(encoding="utf-8"), keys), encoding="utf-8")
    return [dest, bib]


def _structure_dataset(fetch: Fetch, out_name: str, language_id: str, out: Path) -> list[Path]:
    params = {p["ID"]: p for p in read_csv(fetch("cldf/parameters.csv"))}
    codes = {c["ID"]: c for c in read_csv(fetch("cldf/codes.csv"))}
    values = [v for v in read_csv(fetch("cldf/values.csv")) if v["Language_ID"] == language_id]

    def sort_key(v: dict) -> tuple:
        m = re.match(r"([A-Z]*)(\d+)([A-Z]*)$", v["Parameter_ID"])
        return (m.group(1), int(m.group(2)), m.group(3)) if m else (v["Parameter_ID"], 0, "")

    values.sort(key=sort_key)
    rows = []
    for v in values:
        code = codes.get(v.get("Code_ID") or "", {})
        rows.append(
            [
                v["Parameter_ID"],
                params.get(v["Parameter_ID"], {}).get("Name", ""),
                v["Value"],
                code.get("Description") or code.get("Name", ""),
                v.get("Comment", ""),
                v.get("Source", ""),
            ]
        )
    dest = out / "typology" / f"{out_name}.tsv"
    write_tsv(dest, ["feature_id", "feature", "value", "value_label", "comment", "source"], rows)
    keys = set().union(*(split_sources(v.get("Source", "")) for v in values)) if values else set()
    bib = out / "typology" / f"{out_name}.bib"
    bib.write_text(bib_subset(fetch("cldf/sources.bib").read_text(encoding="utf-8"), keys), encoding="utf-8")
    return [dest, bib]


def extract_grambank(fetch: Fetch, out: Path) -> list[Path]:
    return _structure_dataset(fetch, "grambank_sukuma", "suku1261", out)


def extract_wals(fetch: Fetch, out: Path) -> list[Path]:
    return _structure_dataset(fetch, "wals_sukuma", "skm", out)


def extract_phoible(fetch: Fetch, out: Path) -> list[Path]:
    rows_in = [r for r in read_csv(fetch("data/phoible.csv")) if r["Glottocode"] == "suku1261"]
    rows_in.sort(key=lambda r: (int(r["InventoryID"]), r["SegmentClass"], r["Phoneme"]))
    rows = [
        [r["InventoryID"], r["Phoneme"], r["Allophones"], r["Marginal"], r["SegmentClass"], r["Source"]]
        for r in rows_in
    ]
    dest = out / "phonology" / "phoible_sukuma.tsv"
    write_tsv(dest, ["inventory_id", "phoneme", "allophones", "marginal", "segment_class", "source"], rows)
    return [dest]


EXTRACTORS: dict[str, Callable[[Fetch, Path], list[Path]]] = {
    "udhr-suk": extract_udhr,
    "asjp-sukuma": extract_asjp,
    "grollemund-bantu-sukuma": extract_grollemund,
    "grambank-sukuma": extract_grambank,
    "wals-sukuma": extract_wals,
    "phoible-sukuma": extract_phoible,
}


# --------------------------------------------------------------------------
# manifest + driver
# --------------------------------------------------------------------------

def file_record(path: Path, source_id: str) -> dict:
    data = path.read_bytes()
    record = {
        "source_id": source_id,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    text = data.decode("utf-8")
    if path.suffix == ".tsv":
        record["rows"] = max(0, text.count("\n") - 1)
    elif path.suffix == ".txt":
        record["lines"] = text.count("\n")
    return record


def build(out: Path, fetcher_factory: Callable[[str, str], Fetch] = make_fetcher) -> dict:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    open_sources = [s for s in catalog["sources"] if s.get("in_repo")]
    missing = [s["id"] for s in open_sources if s["id"] not in EXTRACTORS]
    if missing:
        raise SystemExit(f"no extractor for in_repo sources: {missing}")

    files: dict[str, dict] = {}
    for src in open_sources:
        access = src["access"]
        print(f"[{src['id']}] {access['repo']}@{access['commit'][:7]}")
        fetch = fetcher_factory(access["repo"], access["commit"])
        for path in EXTRACTORS[src["id"]](fetch, out):
            files[path.relative_to(out).as_posix()] = file_record(path, src["id"])

    manifest = {
        "generated_by": "scripts/build_open_data.py",
        "catalog": "data/catalog.json",
        "pins": {
            s["id"]: {"repo": s["access"]["repo"], "commit": s["access"]["commit"], "license": s["license"]}
            for s in open_sources
        },
        "files": dict(sorted(files.items())),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def generated_paths(out: Path) -> set[str]:
    """Files the builder owns: everything in MANIFEST plus MANIFEST.json itself."""
    manifest = out / "MANIFEST.json"
    if not manifest.exists():
        return set()
    return set(json.loads(manifest.read_text(encoding="utf-8"))["files"]) | {"MANIFEST.json"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="rebuild in a temp dir and diff against data/open/")
    args = ap.parse_args(argv)

    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_out = Path(tmp)
            build(tmp_out)
            expected = generated_paths(tmp_out)
            committed = generated_paths(OPEN_DIR)
            problems = sorted(expected ^ committed)
            problems += [p for p in sorted(expected & committed)
                         if not filecmp.cmp(tmp_out / p, OPEN_DIR / p, shallow=False)]
            if problems:
                print("data/open/ is out of date:", *problems, sep="\n  ")
                return 1
            print(f"data/open/ matches upstream pins ({len(expected)} files).")
            return 0

    OPEN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build(OPEN_DIR)
    total = sum(f["bytes"] for f in manifest["files"].values())
    print(f"Wrote {len(manifest['files'])} files ({total:,} bytes) + MANIFEST.json to {OPEN_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
