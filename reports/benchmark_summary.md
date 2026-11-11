# Sukuma ASR benchmark

Zero-shot unless noted. WER/CER after lowercasing and removing punctuation (diacritics kept); strict and diacritic-blind scores in the last columns.

| Test set | Model | Clips | Hours | WER | CER | Strict WER | Diacritic-blind WER | Diacritic-blind CER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sukuma Voices (Bible) | Omnilingual CTC 300M | 362 | 1.01 | 52.4% | 10.9% | 69.2% | 50.8% | 10.2% |
| Sukuma Voices (Bible) | Omnilingual CTC 1B | 362 | 1.01 | 34.6% | 7.3% | 56.0% | 33.0% | 6.8% |
| Sukuma Voices (Bible) | Omnilingual LLM 7B | 362 | 1.01 | 9.9% | 2.6% | 36.7% | 9.4% | 2.5% |
| Sukuma Voices (Bible) | Paper's Whisper fine-tune | 362 | 1.01 | 9.5% | 2.9% | 16.9% | 9.3% | 2.7% |
| Speech Corpus (prompts) | Omnilingual CTC 300M | 400 | 1.98 | 91.7% | 26.6% | 95.2% | 83.1% | 21.2% |
| Speech Corpus (prompts) | Omnilingual CTC 1B | 400 | 1.98 | 89.2% | 26.4% | 92.8% | 77.0% | 19.8% |
| Speech Corpus (prompts) | Omnilingual LLM 7B | 400 | 1.98 | 90.4% | 19.8% | 95.5% | 84.3% | 17.1% |

Paper baseline (Whisper Large v3 fine-tuned on Sukuma Voices, its own scoring): 25.19% WER on Sukuma Voices. Meta reports 2.0% CER for Sukuma (7B LLM model, Meta's own test data).
