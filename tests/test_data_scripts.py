"""Offline tests for scripts/fetch_data.py, scripts/build_open_data.py and
scripts/download_afrispeech_text.py. No network access is needed."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    """Import scripts/<name>.py as a module (scripts/ is not a package)."""
    mod_name = f"_scripts_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # dataclasses look the module up while executing
    spec.loader.exec_module(module)
    return module


fetch_data = _load("fetch_data")
build_open_data = _load("build_open_data")
download_text = _load("download_afrispeech_text")
CATALOG = fetch_data.load_catalog()


def _by_id(steps):
    return {s.source_id: s for s in steps}


# ----------------------------------------------------------------- planning

def test_default_plan_is_text_only():
    steps = _by_id(fetch_data.plan(CATALOG))
    for s in steps.values():
        if s.action != "download":
            continue
        audio = s.source["access"].get("audio_patterns", [])
        assert not set(audio) & set(s.patterns), f"{s.source_id} downloads audio without --audio"
    assert "--audio" in steps["sukuma-voices"].reason
    assert steps["sukuma-speech-corpus"].patterns == [
        "README.md", "train/metadata.jsonl", "validation/metadata.jsonl", "test/metadata.jsonl"
    ]


def test_default_plan_routes_each_access_type():
    steps = _by_id(fetch_data.plan(CATALOG))
    assert steps["asjp-sukuma"].action == "in_repo"
    assert steps["udhr-suk"].action == "in_repo"
    assert steps["global-recordings-suk"].action == "manual"
    assert steps["fineweb2-suk"].action == "download"


def test_default_text_download_stays_small():
    steps = fetch_data.plan(CATALOG)
    known = sum(s.approx_bytes for s in steps if s.action == "download" and s.approx_bytes)
    assert known < 50_000_000


def test_audio_plan_adds_audio_and_unknown_sizes_propagate():
    steps = _by_id(fetch_data.plan(CATALOG, audio=True))
    acl = steps["sukuma-voices-acl"]
    assert "data/*.parquet" in acl.patterns
    assert acl.approx_bytes > 200_000_000
    assert steps["sukuma-speech-corpus"].approx_bytes is None  # audio size not published


def test_skip_restricted():
    steps = _by_id(fetch_data.plan(CATALOG, skip_restricted=True))
    assert steps["africa-corpus-bible"].action == "skip"
    assert steps["dcad2000-suk"].action == "skip"
    assert steps["glotcc-suk"].action == "download"


def test_only_filters_and_rejects_unknown_ids():
    steps = fetch_data.plan(CATALOG, only=["glotcc-suk"])
    assert [s.source_id for s in steps] == ["glotcc-suk"]
    with pytest.raises(ValueError, match="unknown source"):
        fetch_data.plan(CATALOG, only=["glotcc-suk", "not-a-source"])


# ----------------------------------------------------------------- extraction

def _read_tsv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def test_extract_jsonl(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    lines = [{"utterance_id": "a", "text": "Ū Yesu\tūbawīla"}, {"utterance_id": "b", "text": ""},
             {"utterance_id": "c", "text": "nzīla\nya"}]
    (split / "metadata.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n",
                                          encoding="utf-8")
    n = fetch_data.extract_text(tmp_path, {"format": "jsonl", "column": "text", "glob": "*/metadata.jsonl"})
    rows = _read_tsv(tmp_path / "text.tsv")
    assert n == 2
    assert [r["text"] for r in rows] == ["Ū Yesu ūbawīla", "nzīla ya"]
    assert rows[0]["file"] == "train/metadata.jsonl" and rows[1]["row"] == "2"


def test_extract_csv_falls_back_to_first_text_column(tmp_path):
    with (tmp_path / "Sukuma_suk_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["verse_key", "verse_text"])
        w.writerow(["MAT.1.1", "Ilagano lya ng'wa Yesu"])
    n = fetch_data.extract_text(tmp_path, {"format": "csv", "column": "local", "glob": "Sukuma_suk_v*.csv"})
    assert n == 1 and _read_tsv(tmp_path / "text.tsv")[0]["text"] == "Ilagano lya ng'wa Yesu"


def test_extract_parquet_reads_only_text_column(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    (tmp_path / "data").mkdir()
    table = pa.table({"audio": [b"\x00" * 10, b"\x01"], "content": ["Bamò babo", "Ī haha"]})
    pq.write_table(table, tmp_path / "data" / "train-00000.parquet")
    n = fetch_data.extract_text(tmp_path, {"format": "parquet", "column": "text", "glob": "data/*.parquet"})
    assert n == 2 and [r["text"] for r in _read_tsv(tmp_path / "text.tsv")] == ["Bamò babo", "Ī haha"]


def test_extract_without_spec_or_files_writes_nothing(tmp_path):
    assert fetch_data.extract_text(tmp_path, None) == 0
    assert fetch_data.extract_text(tmp_path, {"format": "jsonl", "column": "text", "glob": "*.jsonl"}) == 0
    assert not (tmp_path / "text.tsv").exists()


# ----------------------------------------------------------------- main()

def test_main_downloads_writes_provenance_and_text(tmp_path, capsys):
    def fake_download(step, dest):
        target = dest / step.source_id
        (target / "train").mkdir(parents=True)
        (target / "train" / "metadata.jsonl").write_text('{"text": "Mhola"}\n', encoding="utf-8")
        return target

    code = fetch_data.main(["--only", "sukuma-speech-corpus", "--dest", str(tmp_path)], downloader=fake_download)
    target = tmp_path / "sukuma-speech-corpus"
    assert code == 0
    assert json.loads((target / "SOURCE.json").read_text())["license"] == "CC-BY-4.0"
    assert _read_tsv(target / "text.tsv")[0]["text"] == "Mhola"


def test_main_reports_failures_and_continues(tmp_path, capsys):
    seen = []

    def flaky(step, dest):
        seen.append(step.source_id)
        if step.source_id == "fineweb2-suk":
            raise ConnectionError("blocked")
        (dest / step.source_id).mkdir(parents=True)
        return dest / step.source_id

    code = fetch_data.main(["--only", "fineweb2-suk,glotcc-suk", "--dest", str(tmp_path)], downloader=flaky)
    assert code == 1
    assert seen == ["fineweb2-suk", "glotcc-suk"]
    assert "fineweb2-suk" in capsys.readouterr().err


def test_main_restricted_source_prints_license_warning(tmp_path, capsys):
    fetch_data.main(["--only", "africa-corpus-bible", "--dest", str(tmp_path)],
                    downloader=lambda step, dest: (dest / step.source_id).mkdir(parents=True) or dest / step.source_id)
    assert "never commit" in capsys.readouterr().out


# ----------------------------------------------------------------- build_open_data helpers

def test_split_sources_and_bib_subset():
    assert build_open_data.split_sources("g_Batibo[244-249];30800") == {"g_Batibo", "30800"}
    bib = "@book{keep,\n title={A}\n}\n\n@misc{drop,\n title={B}\n}\n"
    out = build_open_data.bib_subset(bib, {"keep"})
    assert "keep" in out and "drop" not in out


def test_write_tsv_cleans_cells(tmp_path):
    p = tmp_path / "x.tsv"
    assert build_open_data.write_tsv(p, ["a", "b"], [["x\ty", "line\nbreak"]]) == 1
    assert p.read_text(encoding="utf-8") == "a\tb\nx y\tline break\n"


def test_phoible_extractor_filters_sukuma(tmp_path):
    src = tmp_path / "phoible.csv"
    with src.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["InventoryID", "Glottocode", "Phoneme", "Allophones", "Marginal", "SegmentClass", "Source"])
        w.writerow(["1568", "suku1261", "a", "a", "FALSE", "vowel", "ph"])
        w.writerow(["9", "swah1253", "a", "a", "FALSE", "vowel", "x"])
    out = tmp_path / "out"
    (path,) = build_open_data.extract_phoible(lambda _p: src, out)
    assert path.read_text(encoding="utf-8").count("\n") == 2  # header + one Sukuma row


def test_every_in_repo_source_has_an_extractor():
    in_repo = {s["id"] for s in CATALOG["sources"] if s["in_repo"]}
    assert in_repo == set(build_open_data.EXTRACTORS)


# ----------------------------------------------------------------- download_afrispeech_text

def _fake_builder_module(calls):
    class FakeCorpusBuilder:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def download(self, out):
            out = Path(out)
            out.write_text("suk\nAlīyo ū Moyo\nNīyo yalī būjikū\n", encoding="utf-8")
            return str(out)

    mod = types.ModuleType("africa_bitext_builder.builder")
    mod.CorpusBuilder = FakeCorpusBuilder
    return mod


def test_text_script_requests_all_copyrighted_versions_explicitly(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setitem(sys.modules, "africa_bitext_builder", types.ModuleType("africa_bitext_builder"))
    monkeypatch.setitem(sys.modules, "africa_bitext_builder.builder", _fake_builder_module(calls))
    out = tmp_path / "s.csv"
    assert download_text.main(["--out", str(out)]) == 0
    (kw,) = calls
    assert kw["source_lang"] == "suk" and kw["mode"] == "monolingual"
    assert kw["source_version_ids"] == [1512, 1517, 2684]
    printed = capsys.readouterr().out
    assert "copyrighted" in printed and "Wrote 2 sentences" in printed


def test_text_script_with_real_builder_offline(tmp_path, monkeypatch):
    """Runs the real africa-bitext-builder against local files shaped like the Hub CSVs."""
    pytest.importorskip("africa_bitext_builder.builder")
    raw = tmp_path / "raw"
    raw.mkdir()
    # The builder ignores local CSVs under 64 bytes (then goes to the network),
    # so the verses are long enough to look like real files.
    gen = "Mu kwandya ū Mulungu wazumba ū lūlanga nī sī."
    mat1 = "Ilagano lya ng'wa Yesu Kilisito, ng'wana wa ng'wa Daūdi."
    mat2 = "Abulahamu wamyala Isaka, nū Isaka wamyala Yakobo."
    for vid, rows in {1512: [("MAT.1.2", mat2), ("MAT.1.1", mat1)],
                      2684: [("MAT.1.1", mat1), ("GEN.1.1", gen)]}.items():
        with (raw / f"Sukuma_suk_v{vid}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["verse_key", "local"])
            w.writerows(rows)
    monkeypatch.setattr(download_text, "LOCAL_RAW", raw)
    out = tmp_path / "sukuma_sentences.csv"
    assert download_text.main(["--versions", "1512,2684", "--out", str(out)]) == 0
    with out.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # header from the builder, deduplicated, canonical Bible order (GEN before MAT)
    assert rows == [["suk"], [gen], [mat1], [mat2]]
