# sukuma-toolkit

To our knowledge, the first installable Python package for **Sukuma** (Kisukuma), a Bantu language
spoken by ~10 million people in northern Tanzania: text normalization,
diacritic restoration, corpus statistics, dataset splitting, speech/text
evaluation, and a `sukuma` command-line tool.

Built as a Fall 2026 senior project (BYU-Idaho, DS 499) — and as the start of
real tooling for one of Africa's most under-resourced languages.

## Install

```bash
pip install -e .            # base: text utilities
pip install -e ".[audio]"   # + audio decoding for the speech corpus
pip install -e ".[dev]"     # + pytest
```

## Quickstart

```python
from sukuma_toolkit import normalize_text, strip_diacritics, detect_diacritics
from sukuma_toolkit.corpus import load_afrispeech_text
from sukuma_toolkit.split import train_test_split_sentences
from sukuma_toolkit.evaluate import wer

normalize_text("A ha wandījo wa kūseema,")   # NFC, punctuation, whitespace
strip_diacritics("wandījo")                  # -> "wandijo"
detect_diacritics("wandījo")                 # -> True

sents = load_afrispeech_text("data/sukuma_sentences.csv")
train, test = train_test_split_sentences(sents, test_size=0.1, seed=42)
wer("wandijo wa kuseema", "wandijo wa kuseema")  # -> 0.0
```

## Command line

```bash
sukuma normalize notes.txt                    # clean Unicode, quotes and spacing
sukuma normalize --strip-diacritics notes.txt # wandījo -> wandijo
echo "U Yesu ubawila" | sukuma restore        # wandijo -> wandījo (diacritic restorer v1)
sukuma stats sentences.csv                    # corpus report in Markdown
sukuma score refs.txt hyps.txt                # WER / CER: strict, normalized, diacritic-blind
```

The shipped restorer is trained on the Sukuma Speech Corpus prompts
(joseph-telemala, CC BY 4.0) and follows that corpus's spelling. For
Bible-style text, train one with `scripts/train_restorer.py` and pass
`--model models/sukuma_restorer_bible.json`. Accuracy is in
`reports/diacritic_restorer.md`.

## Data

Every public Sukuma source found (16 data sources, plus 6 models) is listed
in [`data/SOURCES.md`](data/SOURCES.md), with licenses and sizes in
[`data/catalog.json`](data/catalog.json).

- **In the repo:** small openly licensed data under `data/open/`: the UDHR in
  Sukuma (with Swahili and English), ASJP and Grollemund wordlists, Grambank and
  WALS features, and the PHOIBLE phoneme inventory.
- **Speech:** the 111.6-hour Sukuma Speech Corpus (CC BY 4.0) and Sukuma Voices
  (7.47 h and 19.56 h releases). `python scripts/fetch_data.py --audio`
- **Text:** 46,989 Bible sentences (copyrighted, local use only), plus web and
  PDF text from FineWeb-2, FinePDFs, GlotCC and DCAD-2000.
  `python scripts/fetch_data.py`, then `python scripts/download_afrispeech_text.py`

Large or restricted data is never committed; see `data/README.md`.

## Analysis

```bash
python scripts/run_eda.py --plot            # corpus statistics -> reports/
python scripts/train_tokenizer.py           # tokenizer experiment -> reports/, models/
python scripts/omnilingual_smoke_test.py -n 362                          # Omnilingual ASR, full Sukuma Voices test
python scripts/omnilingual_smoke_test.py --dataset sukuma-speech-corpus -n 400   # seeded corpus sample
python scripts/make_disjoint_split.py       # split audit + speaker- and sentence-disjoint split
python scripts/train_restorer.py            # diacritic restorer experiments -> reports/
python scripts/token_tax.py                 # tokens per word for LLM tokenizers on Sukuma vs English
python scripts/audit_web_text.py            # what share of "Sukuma" web text is Sukuma (GlotLID)
```

The Sukuma Speech Corpus's official test split shares every sentence with train
(speakers differ). `data/splits/sukuma-speech-corpus_disjoint.csv` gives a split
where neither speakers nor sentences cross over.

## Layout

```
src/sukuma_toolkit/
    normalize.py   # NFC/whitespace/punctuation + diacritic-form handling
    diacritics.py  # diacritic restorer (word table + character model)
    cli.py         # the `sukuma` command
    corpus.py      # loaders for AfriSpeech text + Sukuma Voices audio
    split.py       # validation, reproducible and speaker/sentence-disjoint splits
    evaluate.py    # WER / CER: strict, normalized, diacritic-blind
    stats.py       # corpus statistics (EDA)
    tokenizer.py   # SentencePiece tokenizer experiments
scripts/           # fetch_data.py, build_open_data.py, download scripts
tests/             # pytest suite
data/              # catalog + open/ + splits/ committed; raw/ local-only (see data/README.md)
reports/           # generated EDA, tokenizer, benchmark and audit reports
```

## License

MIT — see LICENSE.
