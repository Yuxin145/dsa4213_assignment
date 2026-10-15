| representation | C (val) | val acc | test acc | test macro-F1 | LM test ppl (seed 0) |
|---|---|---|---|---|---|
| lstm_mean (main) | 0.1 | 0.7520 | 0.7297 | 0.7286 | 133.43 |
| transformer_mean (main) | 1.0 | 0.7930 | 0.7784 | 0.7782 | 135.51 |
| lstm_last | 1.0 | 0.6930 | 0.6916 | 0.6898 | 133.43 |
| transformer_last | 0.01 | 0.7140 | 0.7186 | 0.7180 | 135.51 |
| own_frozen_mean | 0.01 | 0.8105 | 0.7864 | 0.7857 | 125.43 |
| glove_frozen_mean | 0.01 | 0.8410 | 0.8272 | 0.8269 | 133.47 |
| glove_avg_baseline | 0.01 | 0.8855 | 0.8689 | 0.8688 | – |

AG News: 8,000 train / 2,000 val (stratified from official train, seed 0), 7,600 official test. StandardScaler + LogisticRegression; C chosen on validation accuracy.
