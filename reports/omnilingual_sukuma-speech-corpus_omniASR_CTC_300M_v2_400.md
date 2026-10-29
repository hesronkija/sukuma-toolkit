# Omnilingual ASR on Sukuma: omniASR_CTC_300M_v2

400 clips (1.98 h of audio) from the joseph-telemala/sukuma-speech-corpus test split; transcription took 1069 s.

| Scoring | WER | CER |
| --- | --- | --- |
| Strict (exact text) | 95.2% | 28.2% |
| Lowercase, no punctuation, diacritics kept | 91.7% | 26.6% |
| Diacritics also ignored | 83.1% | 21.2% |

Reference points: Sukuma Voices paper baseline (Whisper Large v3 fine-tune), 25.19% WER. Meta reports 2.0% CER for suk_Latn on its own test data (7B LLM model).
