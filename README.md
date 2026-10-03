# sukuma-toolkit

The first installable Python package for **Sukuma** (Kisukuma), a Bantu language
spoken by ~10 million people in northern Tanzania: text normalization,
corpus loading/validation, dataset splitting, and speech/text evaluation
utilities.

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

## Layout

```
src/sukuma_toolkit/
    normalize.py   # NFC/whitespace/punctuation + diacritic-form handling
    corpus.py      # loaders for AfriSpeech text + Sukuma Voices audio
    split.py       # validation + reproducible train/test splits
    evaluate.py    # WER / CER
scripts/           # fetch_data.py, build_open_data.py, download scripts
tests/             # pytest suite
data/              # catalog + open/ committed; raw/ local-only (see data/README.md)
```

## License

MIT — see LICENSE.
