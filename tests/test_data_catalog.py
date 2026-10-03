"""Checks that data/catalog.json, data/SOURCES.md and the committed data/open/ agree."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CATALOG = json.loads((ROOT / "data" / "catalog.json").read_text(encoding="utf-8"))
SOURCES = CATALOG["sources"]
OPEN = ROOT / "data" / "open"
MANIFEST = json.loads((OPEN / "MANIFEST.json").read_text(encoding="utf-8"))

KINDS = {"speech", "text", "lexicon", "typology", "phonology"}
REQUIRED = {"id", "name", "kind", "description", "license", "redistributable", "in_repo", "access", "homepage", "citation"}


def test_language_identifiers():
    assert CATALOG["language"]["iso639_3"] == "suk"
    assert CATALOG["language"]["glottocode"] == "suku1261"


def test_ids_unique_and_slugs():
    ids = [s["id"] for s in SOURCES]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", i) for i in ids)


@pytest.mark.parametrize("src", SOURCES, ids=lambda s: s["id"])
def test_source_schema(src):
    assert REQUIRED <= set(src), REQUIRED - set(src)
    assert src["kind"] in KINDS
    assert src["license"].strip()
    access = src["access"]
    assert access["type"] in {"hf", "github", "manual"}
    if access["type"] == "hf":
        assert re.fullmatch(r"[\w.-]+/[\w.-]+", access["repo_id"])
        assert access["text_patterns"], "every HF source downloads at least its card or text"
    if access["type"] == "github":
        assert re.fullmatch(r"[0-9a-f]{40}", access["commit"]), "pin a full commit SHA"
        assert access["files"]
    if access["type"] == "manual":
        assert access["url"].startswith("https://")


@pytest.mark.parametrize("src", SOURCES, ids=lambda s: s["id"])
def test_nothing_restricted_is_committed(src):
    if not src["redistributable"]:
        assert not src["in_repo"], f"{src['id']} is not redistributable but marked in_repo"
    if src["in_repo"]:
        assert src["access"]["type"] == "github", "committed data must be rebuildable from a pinned commit"


def test_every_source_documented():
    doc = (ROOT / "data" / "SOURCES.md").read_text(encoding="utf-8")
    missing = [s["id"] for s in SOURCES if f"`{s['id']}`" not in doc]
    assert not missing, f"add these ids to data/SOURCES.md: {missing}"


def test_manifest_pins_match_catalog():
    in_repo = {s["id"]: s for s in SOURCES if s["in_repo"]}
    assert set(MANIFEST["pins"]) == set(in_repo)
    for sid, pin in MANIFEST["pins"].items():
        assert pin["commit"] == in_repo[sid]["access"]["commit"]
        assert pin["repo"] == in_repo[sid]["access"]["repo"]
    covered = {f["source_id"] for f in MANIFEST["files"].values()}
    assert covered == set(in_repo), "every committed source has at least one file"


@pytest.mark.parametrize("rel", sorted(MANIFEST["files"]))
def test_manifest_checksums(rel):
    rec = MANIFEST["files"][rel]
    data = (OPEN / rel).read_bytes()
    assert hashlib.sha256(data).hexdigest() == rec["sha256"], f"{rel} changed; rerun scripts/build_open_data.py"
    assert len(data) == rec["bytes"]
    data.decode("utf-8")  # must be valid UTF-8


@pytest.mark.parametrize("rel", sorted(r for r in MANIFEST["files"] if r.endswith(".tsv")))
def test_tsv_well_formed(rel):
    with (OPEN / rel).open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    header, body = rows[0], rows[1:]
    assert body, f"{rel} is empty"
    assert len(body) == MANIFEST["files"][rel]["rows"]
    bad = [i for i, r in enumerate(body, 2) if len(r) != len(header)]
    assert not bad, f"{rel}: wrong column count on lines {bad[:5]}"


def test_no_stray_files_in_open():
    on_disk = {p.relative_to(OPEN).as_posix() for p in OPEN.rglob("*")
               if p.is_file() and not p.name.startswith(".")}  # ignore .DS_Store etc.
    expected = set(MANIFEST["files"]) | {"MANIFEST.json", "README.md"}
    assert on_disk == expected, f"unexpected: {on_disk - expected}; missing: {expected - on_disk}"


def test_open_data_content():
    def tsv(rel):
        with (OPEN / rel).open(encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

    asjp = tsv("lexicon/asjp_sukuma.tsv")
    assert {r["concepticon_gloss"]: r["form_asjpcode"] for r in asjp}["TWO"] == "bili"

    groll = tsv("lexicon/grollemund_sukuma.tsv")
    assert {r["variety"] for r in groll} == {"Sukuma", "Ntuzu"}

    phoible = tsv("phonology/phoible_sukuma.tsv")
    assert {"tone", "vowel", "consonant"} <= {r["segment_class"] for r in phoible}

    wals = {r["feature_id"]: r["value_label"] for r in tsv("typology/wals_sukuma.tsv")}
    assert wals["81A"] == "Subject-verb-object (SVO)"

    udhr = (OPEN / "udhr" / "suk.txt").read_text(encoding="utf-8")
    assert "GUNGUNO" in udhr and len(udhr.split()) > 1000


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


@pytest.mark.skipif(shutil.which("git") is None or _git("rev-parse").returncode != 0, reason="not a git checkout")
def test_gitignore_keeps_open_data_and_blocks_raw():
    for rel in MANIFEST["files"]:
        assert _git("check-ignore", "-q", f"data/open/{rel}").returncode == 1, f"data/open/{rel} is gitignored"
    for rel in ("data/raw/sukuma-voices/data/train.parquet", "data/raw/fineweb2-suk/text.tsv",
                "data/sukuma_sentences.csv", "data/raw/africa-corpus-bible/Sukuma_suk_v1512.csv"):
        assert _git("check-ignore", "-q", rel).returncode == 0, f"{rel} should be gitignored"
