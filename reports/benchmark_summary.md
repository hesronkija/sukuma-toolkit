# Sukuma ASR benchmark

Zero-shot unless noted. WER/CER after lowercasing and removing punctuation (diacritics kept); strict and diacritic-blind scores in the last columns.

| Test set | Model | Clips | Hours | WER | CER | Strict WER | Diacritic-blind WER | Diacritic-blind CER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sukuma Voices (Bible) | Omnilingual CTC 300M | 362 | 1.01 | 52.4% | 10.9% | 69.2% | 50.8% | 10.2% |
| Speech Corpus (prompts) | Omnilingual CTC 300M | 400 | 1.98 | 91.7% | 26.6% | 95.2% | 83.1% | 21.2% |

Paper baseline (Whisper Large v3 fine-tuned on Sukuma Voices, its own scoring): 25.19% WER on Sukuma Voices. Meta reports 2.0% CER for Sukuma (7B LLM model, Meta's own test data).
