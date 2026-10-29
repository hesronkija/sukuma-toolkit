# The Sukuma token tax

Same content in three languages: 30 articles of the Universal Declaration of Human Rights (1,081 Sukuma, 1,342 Swahili, 1,366 English words).

| Tokenizer | Tokens per word: Sukuma | Swahili | English | Sukuma tax vs English | Swahili tax vs English |
| --- | --- | --- | --- | --- | --- |
| GPT-4o / o200k | 2.31 | 1.62 | 1.12 | 1.64x | 1.43x |
| GPT-4 / cl100k | 2.57 | 2.25 | 1.12 | 1.82x | 1.98x |
| Llama 3 | 2.55 | 2.24 | 1.12 | 1.81x | 1.97x |
| Qwen 2.5 | 2.56 | 2.25 | 1.12 | 1.82x | 1.98x |
| Gemma 3 | 2.43 | 1.84 | 1.11 | 1.74x | 1.63x |
| XLM-R | 2.32 | 1.45 | 1.26 | 1.46x | 1.13x |

Tax = tokens needed for the same articles divided by tokens for English.

Reference: our 8,000-piece Sukuma SentencePiece model (trained on Bible text, diacritics removed) needs 2,003 tokens for the Sukuma text, 20% fewer than the best LLM tokenizer here (GPT-4o / o200k, 2,498 tokens), even though those vocabularies are 12 to 33 times larger.
