"""Tests for the Sukuma -> Swahili translation data, evaluation and CLI wiring (no model downloads)."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_mt_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_by_book():
    mod = _load("build_translation_data")
    assert mod.split_of("RUT.1.1") == "test" and mod.split_of("TIT.2.3") == "dev" and mod.split_of("GEN.1.1") == "train"


def test_split_pairs_keeps_test_books_out_of_training():
    mod = _load("build_translation_data")
    src = mod.TEST_SOURCE_VERSION
    pairs = {
        (src, 164): [("GEN.1.1", "a", "A"), ("RUT.1.1", "r", "R-suv"), ("TIT.1.1", "t", "T")],
        (src, 74): [("GEN.1.1", "a", "A2"), ("RUT.1.1", "r", "R-bhn")],
        (1512, 164): [("GEN.1.1", "a", "A"), ("JAS.1.1", "j-other", "J")],  # duplicate pair; test verse, wrong source
    }
    out = mod.split_pairs(pairs)
    assert sorted(out["train"]) == [("GEN.1.1", "a", "A"), ("GEN.1.1", "a", "A2")]
    assert out["dev"] == [("TIT.1.1", "t", "T")]
    assert out["test"] == [("RUT.1.1", "r", "R-suv", "R-bhn")]  # JAS has no source-version text and is dropped
    assert all(not mod.split_of(k) == "test" for k, *_ in out["train"])


def test_udhr_pairs_align_all_30_articles():
    mod = _load("build_translation_data")
    pairs = mod.udhr_pairs(ROOT / "data" / "open" / "udhr")
    assert len(pairs) == 30 and pairs[0][0].startswith("UDHR.")
    assert all(suk and swh and suk != swh for _, suk, swh in pairs)


def _write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def test_eval_translator_with_fake_systems(tmp_path):
    pytest.importorskip("sacrebleu")
    mod = _load("eval_translator")
    data = tmp_path / "mt"
    data.mkdir()
    _write(data / "test_bible.tsv", ["key", "suk", "swh_suv", "swh_bhn"],
           [["RUT.1.1", "Yesu akayomba", "Yesu akasema", "Yesu alisema"]])
    _write(data / "test_udhr.tsv", ["key", "suk", "swh"], [["UDHR.1", "Banhu bose", "Watu wote"]])
    perfect = {"Yesu akayomba": "Yesu akasema", "Banhu bose": "Watu wote"}
    res = mod.run(None, data, tmp_path / "reports",
                  translators={"copy": lambda xs: list(xs), "fine-tuned": lambda xs: [perfect[x] for x in xs]})
    assert res["fine-tuned"]["bible"]["chrf"] == pytest.approx(100.0)
    assert res["copy"]["udhr"]["chrf"] < 50
    md = (tmp_path / "reports" / "translation.md").read_text(encoding="utf-8")
    assert "fine-tuned on Sukuma (ours)" in md
    samples = list(csv.reader((tmp_path / "reports" / "translation_udhr_samples.tsv").open(encoding="utf-8"),
                              delimiter="\t"))
    assert samples[1] == ["UDHR.1", "Banhu bose", "Watu wote", "Watu wote"]
    assert (data / "predictions_bible.tsv").exists()  # Bible outputs stay under data/raw (gitignored)


def test_cli_translate_without_model_explains(monkeypatch, capsys):
    from sukuma_toolkit import cli

    monkeypatch.delenv("SUKUMA_MT_MODEL", raising=False)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("Banhu bose\n"))
    with pytest.raises(SystemExit) as e:
        cli.main(["translate"])
    assert "translate" in str(e.value) or "model" in str(e.value)
