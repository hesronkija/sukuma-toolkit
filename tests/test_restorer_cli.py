"""Tests for the token tax script, the diacritic restorer and the CLI."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    mod_name = f"_scripts_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def test_udhr_articles_align():
    mod = _load("token_tax")
    texts = mod.load_parallel()
    assert texts["_articles"] == list(range(1, 31))
    assert all(texts[lang] for lang in ("suk", "swh", "eng"))
    assert "Luhilili" not in texts["suk"] and "Article 1" not in texts["eng"]


def test_token_tax_measure_and_report(tmp_path):
    mod = _load("token_tax")
    texts = {"suk": "aaa bbb", "swh": "cc dd", "eng": "e f", "_articles": [1]}
    counters = {"chars": lambda t: len(t.replace(" ", "")), mod.OURS: lambda t: 1}
    r = mod.measure(texts, counters)
    assert r["tokenizers"]["chars"]["tax_vs_english"]["suk"] == pytest.approx(3.0)
    assert r["tokenizers"]["chars"]["tokens_per_word"]["eng"] == pytest.approx(1.0)
    md = mod.format_report(r)
    assert "| chars |" in md and "Reference: our" in md and f"| {mod.OURS}" not in md


from sukuma_toolkit.diacritics import DiacriticRestorer, evaluate_restorer, load_default  # noqa: E402

TRAIN = [
    "Ū Yesu ūbawīla gīkī, banhū bose.",
    "Ū Yesu alīyo na banhū.",
    "Banhū bakabyalagwa na wīyabi.",
    "Plain sentence without marks na banhu",  # no diacritics: skipped
]


def test_restorer_known_words_case_and_punctuation():
    r = DiacriticRestorer().fit(TRAIN)
    assert r.restore("U Yesu ubawila giki, BANHU bose!") == "Ū Yesu ūbawīla gīkī, BANHŪ bose!"
    assert r.words["banhu"] == {"banhū": 3}  # the plain-orthography sentence was skipped


def test_restorer_unseen_word_uses_character_context():
    r = DiacriticRestorer().fit(TRAIN)
    out = r.restore("ubaliwa")
    assert out.startswith("ū")  # word-initial u is always ū in training
    assert len(out) == len("ubaliwa") and out.endswith("wa")
    assert not r.is_known("ubaliwa")


def test_restorer_bigram_disambiguates():
    r = DiacriticRestorer().fit(["Ā ka mbalī", "Ā ka mbalī", "Ā ka mbalī", "nīna kā mbali", "ū kā zo"])
    assert r.restore_word("ka", prev="nina") == "kā"
    assert r.restore_word("ka", prev="a") == "ka"


def test_restorer_save_load_roundtrip(tmp_path):
    r = DiacriticRestorer().fit(TRAIN)
    path = tmp_path / "m.json"
    r.save(path)
    r2 = DiacriticRestorer.load(path)
    assert r2.restore("u yesu ubawila") == r.restore("u yesu ubawila") == "ū yesu ūbawīla"
    with pytest.raises(ValueError):
        DiacriticRestorer.from_dict({"format": 99})


def test_evaluate_restorer_counts():
    r = DiacriticRestorer().fit(TRAIN)
    s = evaluate_restorer(r, ["Ū Yesu ūbawīla", "Banhū bose"])
    assert s["words"] == 5 and s["word_accuracy"] == 1.0
    assert s["baseline_accuracy"] == pytest.approx(2 / 5)  # only "Yesu" and "bose" are unmarked
    assert s["marked_words"] == 3 and s["unseen_words"] == 0


def test_shipped_default_model_loads():
    r = load_default()
    assert len(r.words) > 1000
    assert r.restore("Ikanza") .lower() == "ikanza"


def test_train_restorer_script(tmp_path):
    import csv as _csv
    import json as _json

    mod = _load("train_restorer")
    bible = tmp_path / "bible.csv"
    with bible.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["suk"])
        w.writerows([[s] for s in TRAIN * 10])
    corpus = tmp_path / "corpus"
    rows = []
    for i, (split, text) in enumerate([("train", "Ūmulūgendo lwise lwakūja"), ("test", "Nalīnayega iyene aho")]):
        (corpus / split).mkdir(parents=True)
        rows.append({"utterance_id": f"u{i}", "text": text})
        (corpus / split / "metadata.jsonl").write_text(_json.dumps(rows[-1]), encoding="utf-8")
    split_csv = tmp_path / "split.csv"
    split_csv.write_text("utterance_id,official_split,disjoint_split\nu0,train,train\nu1,test,test\n",
                         encoding="utf-8")
    shipped = tmp_path / "pkg" / "restorer.json"
    res = mod.run(bible, corpus, split_csv, tmp_path / "reports", tmp_path / "models", shipped)
    assert {"bible -> bible", "bible -> corpus", "corpus -> corpus", "both -> corpus"} <= set(res)
    assert (tmp_path / "models" / "sukuma_restorer_bible.json").exists() and shipped.exists()
    assert "Leave plain" in (tmp_path / "reports" / "diacritic_restorer.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- CLI
from sukuma_toolkit import cli  # noqa: E402


def test_cli_normalize_and_strip(tmp_path, capsys):
    f = tmp_path / "in.txt"
    f.write_text("  Ū  Yesu “ūbawīla”\n", encoding="utf-8")
    assert cli.main(["normalize", str(f)]) == 0
    assert capsys.readouterr().out == 'Ū Yesu "ūbawīla"\n'
    cli.main(["normalize", "--strip-diacritics", str(f)])
    assert capsys.readouterr().out == 'U Yesu "ubawila"\n'


def test_cli_restore_with_model(tmp_path, capsys, monkeypatch):
    model = tmp_path / "m.json"
    DiacriticRestorer().fit(TRAIN).save(model)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("U Yesu ubawila\n"))
    assert cli.main(["restore", "--model", str(model)]) == 0
    assert capsys.readouterr().out == "Ū Yesu ūbawīla\n"


def test_cli_stats_and_score(tmp_path, capsys):
    refs, hyps = tmp_path / "r.txt", tmp_path / "h.txt"
    refs.write_text("Ū Yesu ūbawīla\nbanhū bose\n", encoding="utf-8")
    hyps.write_text("u yesu ubawila\nbanhū bose\n", encoding="utf-8")
    assert cli.main(["stats", str(refs)]) == 0
    assert "Sentences" in capsys.readouterr().out
    assert cli.main(["score", str(refs), str(hyps)]) == 0
    out = capsys.readouterr().out
    assert "Diacritics also ignored      WER   0.0%" in out
    hyps.write_text("one line\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main(["score", str(refs), str(hyps)])


def test_cli_missing_file():
    with pytest.raises(SystemExit):
        cli.main(["normalize", "/nonexistent/file.txt"])
