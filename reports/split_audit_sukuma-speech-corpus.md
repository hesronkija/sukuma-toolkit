# Sukuma Speech Corpus: split audit

21,744 utterances. Linking utterances that share a speaker or a sentence gives 10 independent groups; each speaker reads one prompt set.

## Leakage into train

| Split | Test sentences also in train | Test speakers also in train | Validation sentences also in train |
| --- | --- | --- | --- |
| official | 619 of 619 (100% of clips) | 0 of 31 | 699 of 699 (100% of clips) |
| disjoint | 0 of 80 (0% of clips) | 0 of 24 | 0 of 80 (0% of clips) |

## Official split

| Split | Utterances | Hours | Speakers | Sentences | Female | Outdoor |
| --- | --- | --- | --- | --- | --- | --- |
| train | 17,702 | 92.12 | 231 | 780 | 9% | 100% |
| validation | 1,632 | 7.25 | 21 | 699 | 15% | 100% |
| test | 2,410 | 12.26 | 31 | 619 | 7% | 100% |

## Disjoint split

| Split | Utterances | Hours | Speakers | Sentences | Female | Outdoor |
| --- | --- | --- | --- | --- | --- | --- |
| train | 17,452 | 91.86 | 226 | 620 | 10% | 100% |
| validation | 2,529 | 9.22 | 33 | 80 | 10% | 100% |
| test | 1,763 | 10.56 | 24 | 80 | 4% | 100% |

With so few groups, held-out sizes are approximate. Use the disjoint split to measure how well a model handles new sentences, and the official split to measure new speakers only.
