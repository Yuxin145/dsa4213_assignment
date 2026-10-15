| part | model | decoding | distinct-1 | distinct-2 | repeated-trigram rate | <unk> rate |
|---|---|---|---|---|---|---|
| part1 | Trigram (interp.) | greedy | 0.032 | 0.041 | 0.892 | 0.248 |
| part1 | Trigram (interp.) | sample_t0.8_k40 | 0.316 | 0.706 | 0.004 | 0.136 |
| part1 | RNN | greedy | 0.084 | 0.159 | 0.579 | 0.100 |
| part1 | RNN | sample_t0.8_k40 | 0.364 | 0.739 | 0.025 | 0.088 |
| part1 | LSTM | greedy | 0.176 | 0.327 | 0.508 | 0.128 |
| part1 | LSTM | sample_t0.8_k40 | 0.388 | 0.727 | 0.062 | 0.112 |
| part1 | Transformer | greedy | 0.204 | 0.437 | 0.287 | 0.000 |
| part1 | Transformer | sample_t0.8_k40 | 0.456 | 0.833 | 0.021 | 0.044 |
| part2 | LSTM | greedy | 0.176 | 0.327 | 0.508 | 0.128 |
| part2 | LSTM | sample_t0.8_k40 | 0.388 | 0.727 | 0.062 | 0.112 |
| part2 | LSTM + own W2V (frozen) | greedy | 0.128 | 0.245 | 0.654 | 0.220 |
| part2 | LSTM + own W2V (frozen) | sample_t0.8_k40 | 0.384 | 0.722 | 0.054 | 0.080 |
| part2 | LSTM + GloVe (frozen) | greedy | 0.124 | 0.233 | 0.654 | 0.048 |
| part2 | LSTM + GloVe (frozen) | sample_t0.8_k40 | 0.352 | 0.747 | 0.042 | 0.128 |

50 new tokens for each of the 5 prompts, pooled; seed-0 checkpoints.
