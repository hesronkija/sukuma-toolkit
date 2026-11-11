# Omnilingual ASR on Sukuma: omniASR_LLM_7B_v2

400 clips (1.98 h of audio) from the joseph-telemala/sukuma-speech-corpus test split; transcription took 2400 s.

| Scoring | WER | CER |
| --- | --- | --- |
| Strict (exact text) | 95.5% | 21.6% |
| Lowercase, no punctuation, diacritics kept | 90.4% | 19.8% |
| Diacritics also ignored | 84.3% | 17.1% |

Reference points: Sukuma Voices paper baseline (Whisper Large v3 fine-tune), 25.19% WER. Meta reports 2.0% CER for suk_Latn on its own test data (7B LLM model).
