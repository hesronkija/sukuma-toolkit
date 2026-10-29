"""Tests for speaker- and sentence-disjoint splitting."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sukuma_toolkit.split import disjoint_split, linked_groups, split_leakage

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    mod_name = f"_scripts_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def test_linked_groups_chain_through_shared_keys():
    keys = [("spk1", "a"), ("spk2", "a"), ("spk2", "b"), ("spk3", "c")]
    groups = linked_groups(keys)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def _prompt_set_keys(n_sets=20, speakers_per_set=3, sentences_per_set=5):
    keys = []
    for s in range(n_sets):
        for p in range(speakers_per_set):
            for t in range(sentences_per_set):
                keys.append((f"set{s}-spk{p}", f"set{s}-sent{t}"))
    return keys


def test_disjoint_split_has_no_shared_speaker_or_sentence():
    keys = _prompt_set_keys()
    splits = disjoint_split(keys, {"test": 0.1, "validation": 0.1}, seed=3)
    assert set(splits) == {"train", "validation", "test"}
    for i in (0, 1):
        by_key: dict[str, set[str]] = {}
        for k, s in zip(keys, splits):
            by_key.setdefault(k[i], set()).add(s)
        assert all(len(v) == 1 for v in by_key.values())
    assert splits.count("test") == pytest.approx(0.1 * len(keys), abs=15)


def test_disjoint_split_is_seeded():
    keys = _prompt_set_keys()
    assert disjoint_split(keys, seed=5) == disjoint_split(keys, seed=5)


@pytest.mark.parametrize("sizes", [{"train": 0.5}, {"test": 0.0}, {"test": 0.6, "validation": 0.5}])
def test_disjoint_split_rejects_bad_sizes(sizes):
    with pytest.raises(ValueError):
        disjoint_split([("a", "b")], sizes)


def test_split_leakage_counts():
    r = split_leakage(["x", "x", "y", "z"], ["train", "test", "test", "test"])
    assert r["items"] == 3 and r["leaked_items"] == 1 and r["leaked_unique_keys"] == 1
    assert r["leaked_share"] == pytest.approx(1 / 3)


def test_make_disjoint_split_script(tmp_path):
    mod = _load("make_disjoint_split")
    corpus = tmp_path / "corpus"
    uid = 0
    lines: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for s in range(10):  # 10 prompt sets, 2 speakers each, like the real corpus
        for p in range(2):
            # official split keeps speakers apart but not sentences: speaker 1 of sets 6-9 is held out
            split = "train" if p == 0 or s < 6 else ("validation" if s < 8 else "test")
            for t in range(3):
                uid += 1
                lines[split].append(json.dumps({
                    "utterance_id": f"u{uid:04d}", "text": f"Sentensi {t} ya seti {s}",
                    "duration_seconds": 2.0, "split": split, "participant_record_id": f"P{s}-{p}",
                    "gender": "Female" if p else "Male", "recording_environment": "Outdoor",
                }))
    for split, split_lines in lines.items():
        (corpus / split).mkdir(parents=True)
        (corpus / split / "metadata.jsonl").write_text("\n".join(split_lines), encoding="utf-8")
    out = tmp_path / "split.csv"
    r = mod.run(corpus, out, tmp_path / "reports", test=0.1, validation=0.1, seed=1)
    assert r["groups"] == 10
    assert r["official"]["test_sentence_leak"]["leaked_share"] == 1.0
    assert r["official"]["test_speaker_leak"]["leaked_items"] == 0
    assert r["disjoint"]["test_sentence_leak"]["items"] > 0
    assert r["disjoint"]["test_sentence_leak"]["leaked_items"] == 0
    assert r["disjoint"]["test_speaker_leak"]["leaked_items"] == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == uid and set(rows[0]) == {"utterance_id", "official_split", "disjoint_split"}
    md = (tmp_path / "reports" / "split_audit_sukuma-speech-corpus.md").read_text(encoding="utf-8")
    assert "Leakage into train" in md


def test_make_disjoint_split_missing_metadata(tmp_path):
    mod = _load("make_disjoint_split")
    with pytest.raises(SystemExit):
        mod.load_rows(tmp_path)


def _fake_lid(text: str) -> tuple[str, float]:
    t = text.lower()
    if "the " in t:
        return "eng_Latn", 0.9
    if "kitabu" in t:
        return "swh_Latn", 0.9
    if "itunda" in t:
        return "nim_Latn", 0.8
    if "mulungu" in t:
        return "suk_Latn", 0.99
    return "suk_Latn", 0.3


def test_audit_web_text(tmp_path):
    mod = _load("audit_web_text")
    src = tmp_path / "raw" / "fineweb2-suk"
    src.mkdir(parents=True)
    doc = ("Akayombaga na Mulungu ku nzila ya banhu. Hii ni kitabu cha hadithi nzuri. "
           "This is the front page of it. Itunda akaleta muntu ku kaya. Ndi ng'wa Mulungu wise na bose.")
    with (src / "text.tsv").open("w", encoding="utf-8") as f:
        f.write("file\trow\ttext\n")
        f.write(f"x\t0\t{doc}\n")
    bible = tmp_path / "bible.csv"
    bible.write_text("suk\nAkayombaga na Mulungu ku nzila ya banhu\n", encoding="utf-8")
    r = mod.run(tmp_path / "raw", bible, tmp_path / "reports", predict=_fake_lid)
    s = r["sources"]["fineweb2-suk"]
    assert s["documents"] == 1 and s["words"] == 31
    assert s["word_share"]["sukuma"] == pytest.approx(13 / 31)
    assert s["word_share"]["swahili"] == pytest.approx(6 / 31)
    assert s["word_share"]["english"] == pytest.approx(7 / 31)
    assert s["word_share"]["other"] == pytest.approx(5 / 31)
    assert s["sukuma_words_from_bible"] == pytest.approx(7 / 13)
    md = (tmp_path / "reports" / "web_audit.md").read_text(encoding="utf-8")
    assert "fineweb2-suk" in md and "Mulungu" not in md


def test_audit_web_text_needs_sources(tmp_path):
    mod = _load("audit_web_text")
    with pytest.raises(SystemExit):
        mod.run(tmp_path, tmp_path / "none.csv", tmp_path / "reports", predict=_fake_lid)


def test_whisper_baseline_report(tmp_path):
    np = pytest.importorskip("numpy")
    mod = _load("whisper_baseline")
    clips = [{"id": "a", "waveform": np.zeros(16000, dtype="float32"), "sample_rate": 16000,
              "text": "Ū Yesu ūbawīla"}]
    r = mod.write_report(clips, ["u yesu ubawila"], "org/model", "sukuma-voices-acl", 1.0, tmp_path)
    assert r["strict"]["wer"] == 1.0 and r["diacritic_blind"]["wer"] == 0.0
    assert (tmp_path / "whisper_sukuma-voices-acl_org_model_1.md").exists()
    assert mod.resample(clips[0]["waveform"], 16000) is clips[0]["waveform"]


def test_benchmark_table(tmp_path):
    mod = _load("benchmark_table")
    level = {"wer": 0.5, "cer": 0.1}
    for n in (20, 362):
        (tmp_path / f"omnilingual_sukuma-voices-acl_omniASR_CTC_300M_v2_{n}.json").write_text(json.dumps(
            {"model": "omniASR_CTC_300M_v2", "dataset": "sukuma-voices-acl", "clips": n, "audio_hours": 1.0,
             "strict": level, "normalized": level, "diacritic_blind": level}), encoding="utf-8")
    rows = mod.collect(tmp_path)
    assert len(rows) == 1 and rows[0]["clips"] == 362
    assert "Omnilingual CTC 300M" in mod.format_table(rows)
    assert mod.main(["--reports", str(tmp_path), "--no-plot"]) == 0


def test_omnilingual_resamples_to_16k():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    mod = _load("omnilingual_smoke_test")
    wav, sr = mod.to_16k(np.zeros((32000, 2), dtype="float32"), 32000)
    assert sr == 16000 and wav.shape == (16000,) and wav.dtype == np.float32
