# Omnilingual ASR on Sukuma: omniASR_CTC_300M_v2

20 clips (0.06 h of audio) from the sartifyllc/Sukuma-Voices-ACL test split; transcription took 40 s.

| Scoring | WER | CER |
| --- | --- | --- |
| Strict (exact text) | 70.0% | 15.7% |
| Lowercase, no punctuation, diacritics kept | 53.9% | 10.1% |
| Diacritics also ignored | 52.7% | 9.5% |

Reference points: Sukuma Voices paper baseline (Whisper Large v3 fine-tune), 25.19% WER. Meta reports 2.0% CER for suk_Latn on its own test data (7B LLM model).
