# Diacritic restorer v1

Plain Sukuma in, diacritic form out. Each test strips the marks from held-out sentences, restores them, and counts words written exactly right (every mark must match).

| Train -> test | Test words | Leave plain | Restored | Words that need a mark | Unseen words |
| --- | --- | --- | --- | --- | --- |
| bible -> bible | 81,821 | 50.7% | **98.8%** | 98.4% | 85.5% (3,366) |
| bible -> corpus | 2,074 | 43.7% | **83.7%** | 78.3% | 72.5% (1,056) |
| corpus -> corpus | 2,074 | 43.7% | **91.6%** | 88.3% | 77.6% (595) |
| both -> corpus | 2,074 | 43.7% | **89.8%** | 84.8% | 71.8% (483) |
| both -> bible | 81,821 | 50.7% | **98.8%** | 98.4% | 85.6% (3,340) |

"Leave plain" is the score of doing nothing. "Unseen words" never appeared in training, so the character model restores them letter by letter.
