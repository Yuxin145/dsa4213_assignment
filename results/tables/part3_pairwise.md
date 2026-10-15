| LM seed | LSTM acc | Transformer acc | Transformer − LSTM | only LSTM right (b) | only Transformer right (c) | exact McNemar p |
|---|---|---|---|---|---|---|
| 0 | 0.7297 | 0.7784 | +0.0487 | 638 | 1008 | 6.6e-20 |
| 1 | 0.7334 | 0.7732 | +0.0397 | 664 | 966 | 7.6e-14 |
| 2 | 0.7304 | 0.7797 | +0.0493 | 591 | 966 | 1.7e-21 |
| mean ± std | 0.7312 ± 0.0020 | 0.7771 ± 0.0035 | 0.0459 ± 0.0054 |  |  |  |

Mean pooling, main comparison. b/c count test articles that exactly one model classifies correctly; the exact McNemar test is a two-sided binomial test of b vs c (paired, same test set, per LM seed).
