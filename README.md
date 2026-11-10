# sukuma-toolkit

Python tools for **Sukuma** (Kisukuma), a Bantu language spoken by about
10 million people in northern Tanzania: text normalization, diacritic
restoration, corpus statistics, leak-free dataset splits, speech/text scoring,
and a `sukuma` command-line tool. To our knowledge it is the first installable
Python package for the language.

Fall 2026 senior project (BYU-Idaho, DS 499).

## Findings so far

| Question | Result | Report |
| --- | --- | --- |
| Is the Sukuma Speech Corpus test split clean? | No. All 619 test sentences also appear in train (speakers differ). This repo ships a split with no shared speakers or sentences. | [split audit](reports/split_audit_sukuma-speech-corpus.md) |
| How well does zero-shot ASR work? | Omnilingual CTC 300M: 52.4% WER / 10.9% CER on Sukuma Voices (362 clips), 91.7% / 26.6% on 400 Speech Corpus clips. | [benchmarks](reports/benchmark_summary.md) |
| Can diacritics be restored without a neural model? | 91.6% of words exactly right on held-out corpus sentences (43.7% if left plain); 98.8% on Bible text. | [restorer](reports/diacritic_restorer.md) |
| What does Sukuma cost in LLM tokens? | 1.46x to 1.82x the tokens English needs for the same UDHR articles, across six tokenizers. | [token tax](reports/token_tax.md) |
| Is web text tagged "Sukuma" usable? | FineWeb-2 is 94% Sukuma, but about 88% of that is Bible text. FinePDFs is 70% Sukuma. | [web audit](reports/web_audit.md) |

## Install

Not on PyPI yet; install from a clone.

```bash
git clone https://github.com/hesronkija/sukuma-toolkit && cd sukuma-toolkit
pip install -e .            # the package and CLI (no dependencies)
pip install -e ".[data]"    # + what scripts/fetch_data.py needs
pip install -e ".[audio]"   # + audio decoding for the speech corpora
pip install -e ".[dev]"     # + pytest
```

## Quickstart

```python
from sukuma_toolkit import normalize_text, strip_diacritics, detect_diacritics
from sukuma_toolkit.diacritics import load_default
from sukuma_toolkit.evaluate import wer, cer

normalize_text("A  ha wandījo wa “kūseema” ")    # 'A ha wandījo wa "kūseema"'
strip_diacritics("wandījo")                      # 'wandijo'
detect_diacritics("wandijo")                     # False

load_default().restore("wandijo wa kuseema")     # 'wandījo wa kūseema'

# A transcript that drops the diacritics gets 2 of 3 words wrong:
wer("wandījo wa kūseema", "wandijo wa kuseema")  # 0.67
cer("wandījo wa kūseema", "wandijo wa kuseema")  # 0.11
```

## Command line

```bash
sukuma normalize notes.txt                    # clean Unicode, quotes and spacing
sukuma normalize --strip-diacritics notes.txt # wandījo -> wandijo
echo "wandijo wa kuseema" | sukuma restore    # -> wandījo wa kūseema
sukuma stats sentences.csv                    # corpus report in Markdown
sukuma score refs.txt hyps.txt                # WER / CER: strict, normalized, diacritic-blind
```

The shipped restorer is trained on the Sukuma Speech Corpus prompts
(joseph-telemala, CC BY 4.0) and follows that corpus's spelling. For
Bible-style text, train one with `scripts/train_restorer.py` and pass
`--model models/sukuma_restorer_bible.json`.

`sukuma score` reports all three levels because Sukuma has two written forms:
the gap between the last two lines is the error caused by diacritics alone.

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
