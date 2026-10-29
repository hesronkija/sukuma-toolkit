"""Tests for corpus statistics, two-way scoring, the EDA and tokenizer scripts,
and the Omnilingual smoke test (with a fake pipeline; no network)."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from sukuma_toolkit.evaluate import corpus_cer, score_both_ways
from sukuma_toolkit.stats import corpus_stats, format_report, words

ROOT = Path(__file__).resolve().parent.parent

SENTS = [
    "Ū Yesu ūbawīla gīkī, “Bing'we!”",
    "U Yesu ubawila giki bing'we",
    "Nguno ng'wana wakwe alī mhola.",
    "Nguno ng'wana wakwe alī mhola.",
]


def _load(name: str):
    mod_name = f"_scripts_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, sentences):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suk"])
        w.writerows([s] for s in sentences)


def test_words_keep_inner_apostrophes_and_drop_punctuation():
    assert words("Ū ng'wana, “bing'we”!") == ["ū", "ng'wana", "bing'we"]
    assert words("Ng’wana 12") == ["ng'wana"]  # curly apostrophe normalized, digits dropped


def test_corpus_stats_counts():
    s = corpus_stats(SENTS, top_n=3)
    assert s["sentences"] == 4 and s["unique_sentences"] == 3
    assert s["tokens"] == 5 + 5 + 5 + 5
    assert s["vocabulary"] == 13  # ū, u, yesu, ūbawīla, ubawila, gīkī, giki, bing'we, nguno, ng'wana, wakwe, alī, mhola
    assert s["vocabulary_without_diacritics"] == 10
    assert s["sentences_with_diacritics"] == 3
    assert set(s["diacritic_characters"]) == {"ū", "ī"}
    assert s["top_words"][0][1] == 2
    assert s["words_per_sentence"]["max"] == 5


def test_corpus_stats_empty():
    s = corpus_stats(["", "   "])
    assert s["sentences"] == 0 and s["tokens"] == 0 and s["diacritic_share"] == 0.0


def test_format_report_has_no_full_sentences():
    md = format_report(corpus_stats(SENTS), source="x.csv")
    assert "| Sentences | 4 |" in md and "Source: x.csv" in md
    assert "Nguno ng'wana wakwe" not in md


def test_score_both_ways_separates_diacritic_errors():
    refs = ["Ū Yesu ūbawīla gīkī"]
    hyps = ["u yesu ubawila giki"]
    s = score_both_ways(refs, hyps)
    assert s["strict"]["wer"] == 1.0
    assert s["diacritic_blind"]["wer"] == 0.0 and s["diacritic_blind"]["cer"] == 0.0


def test_corpus_cer_and_length_check():
    assert corpus_cer(["ab", "cd"], ["ab", "cx"]) == 0.25
    with pytest.raises(ValueError):
        corpus_cer(["a"], [])


def test_run_eda_writes_report(tmp_path):
    run_eda = _load("run_eda")
    src = tmp_path / "sents.csv"
    _write_csv(src, SENTS)
    assert run_eda.main([str(src), "--out-dir", str(tmp_path / "r")]) == 0
    md = (tmp_path / "r" / "eda_sents.md").read_text(encoding="utf-8")
    assert "| Word tokens | 20 |" in md


def test_run_eda_reads_fetch_text_tsv(tmp_path):
    run_eda = _load("run_eda")
    d = tmp_path / "fineweb2-suk"
    d.mkdir()
    (d / "text.tsv").write_text("file\trow\ttext\nx.parquet\t0\tNguno ng'wana\n", encoding="utf-8")
    assert run_eda.main([str(d / "text.tsv"), "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "eda_fineweb2-suk.md").exists()


def test_run_eda_missing_input(tmp_path):
    assert _load("run_eda").main([str(tmp_path / "nope.csv")]) == 2


def test_tokenizer_experiment(tmp_path):
    pytest.importorskip("sentencepiece")
    train_tokenizer = _load("train_tokenizer")
    base = ["Ū Yesu ūbawīla gīkī bing'we", "Nguno ng'wana wakwe alī mhola", "Abo ikūmi aho bigwa bakabapelanīla",
            "Ū Petelo ūng'wīla gīkī yaya", "A banhū aho bamona bakakumya", "Ū munhū o nguzu ūlū adimile"]
    sents = [f"{s} {i}" for i in range(40) for s in base]  # 240 distinct sentences
    r = train_tokenizer.run(sents, tmp_path / "models", tmp_path / "reports", vocab_size=60, model_type="unigram")
    assert r["split"]["leak_free"]
    for m in r["models"].values():
        assert m["pieces_per_word"] >= 1.0
    assert "Diacritics removed" in (tmp_path / "reports" / "tokenizer_experiment.md").read_text(encoding="utf-8")


class FakePipeline:
    def __init__(self):
        self.calls = []

    def transcribe(self, inputs, lang=None, batch_size=1):
        self.calls.append((len(inputs), lang))
        return ["u yesu ubawila giki", "nguno ng'wana"]


def test_omnilingual_smoke_test_scores_and_reports(tmp_path):
    smoke = _load("omnilingual_smoke_test")
    clips = [{"waveform": [0.0] * 16, "sample_rate": 16000, "text": "Ū Yesu ūbawīla gīkī"},
             {"waveform": [0.0] * 16, "sample_rate": 16000, "text": "Nguno ng'wana"}]
    fake = FakePipeline()
    r = smoke.run(clips, fake, "omniASR_LLM_300M_v2", tmp_path)
    assert fake.calls == [(2, ["suk_Latn", "suk_Latn"])]  # LLM models get the language code
    assert r["diacritic_blind"]["wer"] == 0.0 and r["strict"]["wer"] > 0.0
    assert r["normalized"]["wer"] > 0.0  # diacritics still count at this level
    md = (tmp_path / "omnilingual_sukuma-voices-acl_omniASR_LLM_300M_v2_2.md").read_text(encoding="utf-8")
    assert "Diacritics also ignored" in md
    smoke.run(clips, fake, "omniASR_CTC_300M_v2", tmp_path)
    assert fake.calls[-1][1] is None  # CTC models get no language code


def test_corpus_sample_is_seeded_and_skips_long_clips(tmp_path):
    import json as _json
    smoke = _load("omnilingual_smoke_test")
    meta = tmp_path / "metadata.jsonl"
    rows = [{"utterance_id": f"u{i:03d}", "duration_seconds": 50.0 if i == 0 else 10.0, "text": "x",
             "file_name": f"audio/{i}.wav"} for i in range(30)]
    meta.write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")
    a = smoke.sample_corpus_rows(meta, 10, seed=1)
    assert a == smoke.sample_corpus_rows(meta, 10, seed=1)
    assert len(a) == 10 and all(r["duration_seconds"] <= 40 for r in a)


def test_score_levels_isolate_diacritics():
    s = score_both_ways(["Ū Yesu, ūbawīla."], ["ū yesu ubawila"])
    assert s["strict"]["wer"] > s["normalized"]["wer"] > s["diacritic_blind"]["wer"] == 0.0
