#!/usr/bin/env python3
"""Download every remote Sukuma data source listed in data/catalog.json.

Everything lands in data/raw/<source-id>/ (gitignored, never committed).
Small openly licensed data is already in the repo under data/open/; rebuild it
with scripts/build_open_data.py.

    python scripts/fetch_data.py --list            # show every source + size
    python scripts/fetch_data.py                   # all text (~36 MB)
    python scripts/fetch_data.py --audio           # + speech audio (~8.3 GB+)
    python scripts/fetch_data.py --only fineweb2-suk,sukuma-speech-corpus
    python scripts/fetch_data.py --skip-restricted # leave out copyrighted text
    python scripts/fetch_data.py --dry-run         # plan only, download nothing

For each downloaded source the script also writes:
    data/raw/<id>/text.tsv     one row per text item: file, row, text
    data/raw/<id>/SOURCE.json  license, citation and access details

Requires huggingface_hub and pyarrow (both come with `pip install -e .`).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog.json"
DEFAULT_DEST = ROOT / "data" / "raw"
TEXT_FALLBACK_COLUMNS = ("text", "content", "local", "sentence", "transcription")

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


@dataclass
class Step:
    source_id: str
    action: str  # "download" | "in_repo" | "manual" | "skip"
    reason: str = ""
    repo_id: str = ""
    patterns: list[str] = field(default_factory=list)
    approx_bytes: int | None = 0
    source: dict = field(default_factory=dict, repr=False)


# --------------------------------------------------------------------------
# planning (pure; unit-tested without network)
# --------------------------------------------------------------------------

def load_catalog(path: Path = CATALOG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plan(
    catalog: dict,
    only: Iterable[str] | None = None,
    audio: bool = False,
    skip_restricted: bool = False,
) -> list[Step]:
    """Decide what to do for each catalog source. Raises on unknown ids."""
    sources = catalog["sources"]
    known = {s["id"] for s in sources}
    wanted = set(only) if only else None
    if wanted is not None:
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"unknown source id(s): {', '.join(unknown)}. Use --list to see ids.")

    steps: list[Step] = []
    for src in sources:
        sid = src["id"]
        if wanted is not None and sid not in wanted:
            continue
        access = src["access"]
        kind = access["type"]
        if kind == "github" or src.get("in_repo"):
            steps.append(Step(sid, "in_repo", "already committed under data/open/", source=src))
            continue
        if kind == "manual":
            steps.append(Step(sid, "manual", f"download by hand: {access['url']}", source=src))
            continue
        if kind != "hf":
            steps.append(Step(sid, "skip", f"unsupported access type {kind!r}", source=src))
            continue
        if skip_restricted and not src.get("redistributable", False):
            steps.append(Step(sid, "skip", f"restricted license ({src['license']}); --skip-restricted", source=src))
            continue

        patterns = list(access.get("text_patterns", []))
        size = access.get("text_bytes")
        has_audio = bool(access.get("audio_patterns"))
        if audio and has_audio:
            patterns += access["audio_patterns"]
            a = access.get("audio_bytes")
            size = None if (size is None or a is None) else size + a
            reason = "text + audio"
        elif has_audio and access.get("transcripts_in_audio"):
            reason = "dataset card only; transcripts are inside the audio files (add --audio)"
        else:
            reason = "text"
        steps.append(Step(sid, "download", reason, access["repo_id"], patterns, size, src))
    return steps


def human_bytes(n: int | None) -> str:
    if n is None:
        return "size unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} GB"


# --------------------------------------------------------------------------
# text extraction (pure file processing; unit-tested with fixtures)
# --------------------------------------------------------------------------

def _clean(text: str) -> str:
    return re.sub(r"[\t\r\n]+", " ", text).strip()


def _iter_parquet(path: Path, column: str) -> Iterator[str]:
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema_arrow.names
    col = column if column in names else next((c for c in TEXT_FALLBACK_COLUMNS if c in names), None)
    if col is None:
        raise KeyError(f"{path.name}: no text column among {names}")
    for value in pq.read_table(path, columns=[col]).column(col).to_pylist():
        yield "" if value is None else str(value)


def _iter_jsonl(path: Path, column: str) -> Iterator[str]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            value = obj.get(column)
            if value is None:
                value = next((obj[c] for c in TEXT_FALLBACK_COLUMNS if c in obj), "")
            yield str(value)


def _iter_csv(path: Path, column: str) -> Iterator[str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        col = column if column in fields else next(
            (c for c in fields if c not in ("verse_key", "id", "book", "chapter", "verse")), None
        )
        if col is None:
            raise KeyError(f"{path.name}: no text column among {fields}")
        for row in reader:
            yield row.get(col) or ""


READERS: dict[str, Callable[[Path, str], Iterator[str]]] = {
    "parquet": _iter_parquet,
    "jsonl": _iter_jsonl,
    "csv": _iter_csv,
}


def extract_text(source_dir: Path, spec: dict | None) -> int:
    """Write source_dir/text.tsv from the files matching spec['glob']. Returns rows written."""
    if not spec:
        return 0
    files = sorted(p for p in source_dir.glob(spec["glob"]) if p.is_file())
    if not files:
        return 0
    reader = READERS[spec["format"]]
    out = source_dir / "text.tsv"
    n = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write("file\trow\ttext\n")
        for path in files:
            rel = path.relative_to(source_dir).as_posix()
            for i, text in enumerate(reader(path, spec["column"])):
                text = _clean(text)
                if text:
                    f.write(f"{rel}\t{i}\t{text}\n")
                    n += 1
    return n


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def download(step: Step, dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    target = dest / step.source_id
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=step.repo_id,
        repo_type="dataset",
        allow_patterns=step.patterns,
        local_dir=str(target),
    )
    return target


def write_provenance(target: Path, src: dict) -> None:
    keep = ("id", "name", "license", "license_url", "redistributable", "homepage", "citation", "notes", "access")
    (target / "SOURCE.json").write_text(
        json.dumps({k: src[k] for k in keep if k in src}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def print_list(catalog: dict) -> None:
    print(f"{'id':26} {'kind':10} {'license':32} {'where':10} text / audio size")
    for s in catalog["sources"]:
        a = s["access"]
        where = "in repo" if s.get("in_repo") else a["type"]
        sizes = ""
        if a["type"] == "hf":
            sizes = human_bytes(a.get("text_bytes"))
            if a.get("audio_patterns"):
                sizes += f" / {human_bytes(a.get('audio_bytes'))}"
        print(f"{s['id']:26} {s['kind']:10} {s['license'][:32]:32} {where:10} {sizes}")


def main(argv: list[str] | None = None, downloader: Callable[[Step, Path], Path] = download) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list sources and exit")
    ap.add_argument("--only", default="", help="comma-separated source ids")
    ap.add_argument("--audio", action="store_true", help="also download speech audio (large)")
    ap.add_argument("--skip-restricted", action="store_true", help="skip sources that are not redistributable")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="output root (default: data/raw)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, download nothing")
    args = ap.parse_args(argv)

    catalog = load_catalog()
    if args.list:
        print_list(catalog)
        return 0

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    try:
        steps = plan(catalog, only=only, audio=args.audio, skip_restricted=args.skip_restricted)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    downloads = [s for s in steps if s.action == "download"]
    known = [s.approx_bytes for s in downloads if s.approx_bytes is not None]
    unknown = len(downloads) - len(known)
    print(f"Plan: {len(downloads)} download(s), about {human_bytes(sum(known))}"
          + (f" + {unknown} of unknown size" if unknown else ""))
    for s in steps:
        size = f" [{human_bytes(s.approx_bytes)}]" if s.action == "download" else ""
        print(f"  {s.action:9} {s.source_id}{size}: {s.reason}")
    if args.dry_run:
        return 0

    failures = []
    for s in downloads:
        print(f"\n== {s.source_id} ({s.repo_id})")
        if not s.source.get("redistributable", False):
            print(f"   note: {s.source['license']}. Local research use only; never commit this data.")
        try:
            target = downloader(s, args.dest)
            write_provenance(target, s.source)
            rows = extract_text(target, s.source["access"].get("extract_text"))
            print(f"   saved to {target}" + (f"; text.tsv has {rows:,} rows" if rows else ""))
        except Exception as exc:  # keep going; report at the end
            failures.append((s.source_id, exc))
            print(f"   FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} source(s) failed: " + ", ".join(f for f, _ in failures), file=sys.stderr)
        return 1
    print("\nDone. Cite each source you use; see data/SOURCES.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
