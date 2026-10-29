# Tokenizer experiment

SentencePiece unigram, target vocabulary 8,000. Trained on 38,611 sentences, scored on 4,290 held-out sentences (overlap: 0).

| Text | Pieces per word | Pieces | Vocabulary |
| --- | --- | --- | --- |
| As written | 1.532 | 147,666 | 8,000 |
| Diacritics removed | 1.503 | 144,931 | 8,000 |

Fewer pieces per word means the tokenizer keeps more Sukuma words whole.
