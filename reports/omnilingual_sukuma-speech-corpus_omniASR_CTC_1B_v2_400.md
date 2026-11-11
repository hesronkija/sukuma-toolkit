# Omnilingual ASR on Sukuma: omniASR_CTC_1B_v2

400 clips (1.98 h of audio) from the joseph-telemala/sukuma-speech-corpus test split; transcription took 48 s.

| Scoring | WER | CER |
| --- | --- | --- |
| Strict (exact text) | 92.8% | 28.0% |
| Lowercase, no punctuation, diacritics kept | 89.2% | 26.4% |
| Diacritics also ignored | 77.0% | 19.8% |

Reference points: Sukuma Voices paper baseline (Whisper Large v3 fine-tune), 25.19% WER. Meta reports 2.0% CER for suk_Latn on its own test data (7B LLM model).
