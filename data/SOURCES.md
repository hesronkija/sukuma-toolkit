# Sukuma data sources

Every public Sukuma (`suk`, Glottolog `suku1261`) data source found, as of
2026-10-02. `data/catalog.json` holds the same information in machine-readable
form and drives `scripts/fetch_data.py` and `scripts/build_open_data.py`.

**How to get each source**

- **In repo:** already committed under `data/open/` (small, openly licensed).
- **fetch:** `python scripts/fetch_data.py --only <id>` downloads it to
  `data/raw/<id>/` (gitignored). Add `--audio` for speech audio.
- **manual:** download by hand from the link.

## Speech

| id | Hours | Utterances | License | Get it | Notes |
| --- | --- | --- | --- | --- | --- |
| `sukuma-speech-corpus` | 111.6 | 21,744 | CC BY 4.0 | fetch (19 MB text; audio size not listed, likely several GB) | Read speech, 283 participant records, speaker metadata (gender, age group, region, district, SNR). Largest by far. Published 2026-09 by [joseph-telemala](https://huggingface.co/datasets/joseph-telemala/sukuma-speech-corpus); prompt-text source not documented. |
| `sukuma-voices-acl` | 7.47 | 3,260 train / 362 test (+2 synthetic test sets) | Apache-2.0 (Hub card) | fetch `--audio` (233 MB) | Bible readings. Release behind [Mgonzo et al., LoResLM 2026](https://aclanthology.org/2026.loreslm-1.25/). Baseline: Whisper Large v3, 25.19% WER. |
| `sukuma-voices` | 19.56 | 6,871 | CC BY 4.0 | fetch `--audio` (8.1 GB) | Larger [Sukuma Voices](https://github.com/sartify/sukuma-voices) release, Bible narratives. |
| `global-recordings-suk` | n/a | n/a | GRN terms (not open) | manual | [Global Recordings Network](https://globalrecordings.net/en/language/suk) audio, untranscribed. |

## Text

| id | Size | License | Get it | Notes |
| --- | --- | --- | --- | --- |
| `africa-corpus-bible` | 46,989 sentences, ~8 MB | **Copyrighted** | fetch, then `scripts/download_afrispeech_text.py` | Three Sukuma Bibles (version ids 1512, 1517, 2684) from [AfriSpeech/africa-corpus](https://huggingface.co/datasets/AfriSpeech/africa-corpus). None is public domain. Local research use only. |
| `finepdfs-suk` | 6.5 MB | ODC-By 1.0 | fetch | [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) PDF text tagged Sukuma by language ID. Audit quality. |
| `dcad2000-suk` | 1.3 MB | Other (see dataset) | fetch | [DCAD-2000](https://huggingface.co/datasets/openbmb/DCAD-2000) re-cleaning of FineWeb-2; overlaps `fineweb2-suk`. |
| `fineweb2-suk` | 0.6 MB | ODC-By 1.0 | fetch | [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) web text tagged Sukuma. Audit quality. |
| `glotcc-suk` | 0.1 MB | CC0 1.0 | fetch | [GlotCC-V1](https://huggingface.co/datasets/cis-lmu/GlotCC-V1) minority-language web text. |
| `udhr-suk` | 1,361 words | Public domain | In repo | UDHR in Sukuma ([OHCHR](https://www.ohchr.org/en/human-rights/universal-declaration/translations/sukuma), UNDP Tanzania), plus Swahili and English. The only general-domain, non-religious text found. |

## Lexicon, typology and phonology

| id | Size | License | Get it | Notes |
| --- | --- | --- | --- | --- |
| `panlex-suk` | unknown | CC0 1.0 | fetch | [PanLex](https://huggingface.co/datasets/gtak1/panlex-meanings) expressions and meanings. CC0, so it can move into `data/open/` after review. |
| `grollemund-bantu-sukuma` | 194 forms | **CC BY-NC 4.0** | In repo | 100-concept lists for Sukuma and Ntuzu varieties ([lexibank](https://github.com/lexibank/grollemundbantu)). |
| `asjp-sukuma` | 36 forms | CC BY 4.0 | In repo | [ASJP](https://asjp.clld.org/) basic vocabulary in ASJPcode. |
| `grambank-sukuma` | 195 features | CC BY 4.0 | In repo | [Grambank](https://grambank.clld.org/languages/suku1261) grammar features. |
| `phoible-sukuma` | 57 segments | CC BY 4.0 | In repo | [PHOIBLE](https://phoible.org/inventories/view/1568) inventory 1568 with tones. |
| `wals-sukuma` | 17 features | CC BY 4.0 | In repo | [WALS](https://wals.info/languoid/lect/wals_code_skm) features (SVO order, etc.). |

## Models (baselines, not data)

| Model | Task | Notes |
| --- | --- | --- |
| [sartifyllc/sukuma-voices-asr](https://huggingface.co/sartifyllc/sukuma-voices-asr) | ASR | Whisper Large v3 fine-tune, 25.19% WER on Sukuma Voices. |
| [sartifyllc/sukuma-voices-tts](https://huggingface.co/sartifyllc/sukuma-voices-tts) | TTS | Orpheus 3B fine-tune, MOS 3.9 (human 4.6). |
| [facebook/mms-tts-suk](https://huggingface.co/facebook/mms-tts-suk) | TTS | Meta MMS VITS voice, CC BY-NC 4.0. |
| [Omnilingual ASR](https://arxiv.org/html/2511.09690v1) | ASR | Lists Sukuma (`suk_Latn`) as supported; no published Sukuma scores. |
| [nsomazr/whisper-small-sukuma](https://huggingface.co/nsomazr/whisper-small-sukuma), [HMkumbo/whisper-small-sukuma](https://huggingface.co/HMkumbo/whisper-small-sukuma) | ASR | Community fine-tunes; data and scores undocumented. |

## Checked and not available

Mozilla Common Voice (v27.0 scripted, v5.0 spontaneous; Sept 2026), Google
WAXAL, the Omnilingual ASR Corpus (model supports Sukuma, corpus has no
`suk_Latn`), MADLAD-400, Glot500-c, SIL Bloom, the eBible corpus, the
Christodoulopoulos Bible corpus, the Tatoeba Challenge, AfriSpeech
`african-corpus-jw`, Lacuna Fund datasets and the Bantu Basic Vocabulary
Database have no Sukuma. Leipzig's CURL crawl has 47 Sukuma sentences but no
download. `lexFollio/sukuma` on Hugging Face is not Sukuma-language data.
