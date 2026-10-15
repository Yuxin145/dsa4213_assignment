| representation | test acc (seed 0) | test acc mean ± std (seeds [0, 1, 2]) | per-seed acc | macro-F1 (seed 0) | macro-F1 mean ± std | LM test ppl mean ± std | C per seed |
|---|---|---|---|---|---|---|---|
| lstm_mean (main) | 0.7297 | 0.7312 ± 0.0020 | 0.7297 / 0.7334 / 0.7304 | 0.7286 | 0.7303 ± 0.0020 | 135.62 ± 2.73 | 0.1 / 0.01 / 0.01 |
| transformer_mean (main) | 0.7784 | 0.7771 ± 0.0035 | 0.7784 / 0.7732 / 0.7797 | 0.7782 | 0.7768 ± 0.0034 | 135.19 ± 1.23 | 1.0 / 10.0 / 0.1 |
| lstm_last | 0.6916 | 0.6948 ± 0.0057 | 0.6916 / 0.7014 / 0.6914 | 0.6898 | 0.6937 ± 0.0062 | 135.62 ± 2.73 | 1.0 / 0.01 / 0.01 |
| transformer_last | 0.7186 | 0.7190 ± 0.0099 | 0.7186 / 0.7093 / 0.7291 | 0.7180 | 0.7183 ± 0.0094 | 135.19 ± 1.23 | 0.01 / 0.1 / 0.01 |
| own_frozen_mean | 0.7864 | 0.7889 ± 0.0050 | 0.7864 / 0.7946 / 0.7857 | 0.7857 | 0.7883 ± 0.0049 | 126.21 ± 0.73 | 0.01 / 0.01 / 0.1 |
| glove_frozen_mean | 0.8272 | 0.8286 ± 0.0013 | 0.8272 / 0.8297 / 0.8288 | 0.8269 | 0.8284 ± 0.0014 | 136.54 ± 3.67 | 0.01 / 0.01 / 0.01 |
| glove_avg_baseline | 0.8689 | 0.8689 ± 0.0000 | 0.8689 / 0.8689 / 0.8689 | 0.8688 | 0.8688 ± 0.0000 | – | 0.01 / 0.01 / 0.01 |

Identical protocol for every LM seed (same AG News split, conversion, pooling, classifier, C grid on validation, test once). glove_avg_baseline does not depend on the LM seed. Std = sample std over LM seeds.
