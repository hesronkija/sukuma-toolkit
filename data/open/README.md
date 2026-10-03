# data/open/

Small, openly licensed Sukuma data that **is** committed to git. Everything
here is extracted from a pinned upstream commit by
`scripts/build_open_data.py`, and `MANIFEST.json` records the SHA-256, size and
row count of every file. `pytest` fails if a file drifts from its manifest.

```bash
python scripts/build_open_data.py          # rebuild from the pins in data/catalog.json
python scripts/build_open_data.py --check  # verify the committed files reproduce exactly
```

| Path | Rows | What it is | License |
| --- | --- | --- | --- |
| `udhr/suk.txt` | 212 lines | Universal Declaration of Human Rights in Sukuma (UNDP Tanzania translation) | Public domain |
| `udhr/swh.txt`, `udhr/eng.txt` | 215, 213 lines | Same text in Swahili and English, for article-level parallel use | Public domain |
| `lexicon/asjp_sukuma.tsv` | 36 | ASJP basic vocabulary, forms in ASJPcode (phonetic ASCII, not standard spelling) | CC BY 4.0 |
| `lexicon/grollemund_sukuma.tsv` | 194 | 100-concept wordlists for the Sukuma and Ntuzu varieties, with cognate sets | **CC BY-NC 4.0** |
| `typology/grambank_sukuma.tsv` | 195 | Grambank grammatical features | CC BY 4.0 |
| `typology/wals_sukuma.tsv` | 17 | WALS features (word order, etc.) | CC BY 4.0 |
| `phonology/phoible_sukuma.tsv` | 57 | PHOIBLE phoneme inventory 1568, including tones | CC BY 4.0 |

Each `.bib` file holds the references cited in the matching table's `source`
column.

## Attribution

Cite the upstream source whenever you use a file (full citations are in
`data/catalog.json` and `data/SOURCES.md`):

- UDHR: United Nations; Sukuma translation by UNDP Tanzania, via the Unicode UDHR project (NLTK `udhr2`).
- ASJP: Wichmann, Holman & Brown (eds.), *The ASJP Database*, lexibank CLDF edition.
- Grollemund et al. (2015), *PNAS* 112(43), lexibank CLDF edition.
- Grambank: Skirgård et al. (2023), *Science Advances* 9(16).
- WALS: Dryer & Haspelmath (eds.) (2013), *WALS Online*.
- PHOIBLE: Moran & McCloy (eds.) (2019), *PHOIBLE 2.0*.

**Licensing note:** the repository's MIT license covers the code only. Each
file here keeps its upstream license. `grollemund_sukuma.tsv` is
NonCommercial: remove it from any commercial redistribution.
