# Omnilingual ASR on Sukuma: omniASR_CTC_1B_v2

362 clips (1.01 h of audio) from the sartifyllc/Sukuma-Voices-ACL test split; transcription took 23 s.

| Scoring | WER | CER |
| --- | --- | --- |
| Strict (exact text) | 56.0% | 12.6% |
| Lowercase, no punctuation, diacritics kept | 34.6% | 7.3% |
| Diacritics also ignored | 33.0% | 6.8% |

Reference points: Sukuma Voices paper baseline (Whisper Large v3 fine-tune), 25.19% WER. Meta reports 2.0% CER for suk_Latn on its own test data (7B LLM model).
