# data/

| Path | In git? | What |
| --- | --- | --- |
| `catalog.json` | yes | Machine-readable list of every Sukuma source: license, size, how to get it |
| `SOURCES.md` | yes | The same inventory for humans, plus models and sources checked and ruled out |
| `open/` | yes | Small, openly licensed data (UDHR, wordlists, typology, phonology); see `open/README.md` |
| `raw/` | no | Everything `scripts/fetch_data.py` downloads |
| `sukuma_sentences.csv` | no | Bible text built by `scripts/download_afrispeech_text.py` |
| `sukuma-voices/` | no | Speech corpus saved by `scripts/download_sukuma_voices.py` |

## Policy

Only data that is small **and** openly licensed for redistribution is
committed, and only through `scripts/build_open_data.py` from a pinned
upstream commit. Everything else stays local:

- **The Sukuma Bible text is copyrighted.** All three versions in the AfriSpeech
  corpus are marked not public domain, so the builder's public-domain-only mode
  returns no Sukuma text. Use it for research locally; never commit it.
- **Speech audio is large** (up to 8 GB per corpus) and belongs on the Hugging
  Face Hub, not in git.
- **Web text** (FineWeb-2, FinePDFs, GlotCC, DCAD-2000) is language-ID filtered
  and unaudited. Review it before any of it moves into `open/`.

## Get everything

```bash
python scripts/fetch_data.py --list         # every source, license and size
python scripts/fetch_data.py                # all text, ~36 MB -> data/raw/
python scripts/fetch_data.py --audio        # + speech audio, 8.3 GB+
python scripts/download_afrispeech_text.py  # -> data/sukuma_sentences.csv
python scripts/download_sukuma_voices.py    # -> data/sukuma-voices/
```

Each downloaded source gets `data/raw/<id>/text.tsv` (one text item per row)
and `data/raw/<id>/SOURCE.json` (license and citation).
