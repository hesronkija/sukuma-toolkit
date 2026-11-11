# Omnilingual ASR on Sukuma: omniASR_LLM_7B_v2

362 clips (1.01 h of audio) from the sartifyllc/Sukuma-Voices-ACL test split; transcription took 1412 s.

| Scoring | WER | CER |
| --- | --- | --- |
| Strict (exact text) | 36.7% | 8.1% |
| Lowercase, no punctuation, diacritics kept | 9.9% | 2.6% |
| Diacritics also ignored | 9.4% | 2.5% |

Reference points: Sukuma Voices paper baseline (Whisper Large v3 fine-tune), 25.19% WER. Meta reports 2.0% CER for suk_Latn on its own test data (7B LLM model).
