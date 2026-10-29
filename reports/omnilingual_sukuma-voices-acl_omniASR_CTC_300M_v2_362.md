# Omnilingual ASR on Sukuma: omniASR_CTC_300M_v2

362 clips (1.01 h of audio) from the sartifyllc/Sukuma-Voices-ACL test split; transcription took 1035 s.

| Scoring | WER | CER |
| --- | --- | --- |
| Strict (exact text) | 69.2% | 16.0% |
| Lowercase, no punctuation, diacritics kept | 52.4% | 10.9% |
| Diacritics also ignored | 50.8% | 10.2% |

Reference points: Sukuma Voices paper baseline (Whisper Large v3 fine-tune), 25.19% WER. Meta reports 2.0% CER for suk_Latn on its own test data (7B LLM model).
