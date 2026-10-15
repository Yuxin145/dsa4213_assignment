# Requirements checklist (Assignment1_2026.pdf → evidence)

Status: **DONE** = implemented, executed on the full data, output file exists; **STUDENT** = needs the student's
own writing or completion; **MISSING** = not satisfied. Updated 2026-10-15 after the full runs.

| # | PDF requirement | Satisfied by | Status |
|---|---|---|---|
| G1 | Choose own English corpus for Parts I/II | WikiText-2 v1 (`configs/base.yaml: data`, README, `results/data_stats.json`) | DONE |
| G2 | Part III: suitable labeled dataset, representations from a Part I/II LM | AG News; `results/part3/results.json` (+ `seed1/`, `seed2/`), `lm_checkpoints` = checkpoint path + SHA-256 | DONE |
| G3 | Train required LMs yourself from scratch | 15 runs in `results/part{1,2}/*/seed*/` (log, metrics, DONE); test 10 shows Part II does not reuse Part I weights | DONE |
| G4 | Computationally reasonable dataset/model size | 2.08M training tokens, 10.5–12.0M params; 5.87 GPU-h total (README "Actual runtimes") | DONE |
| G5 | Comparisons reasonable and clearly explained | shared data/vocab/targets (tests 1, 5), identical Part II setup (test 10), DECISIONS.md | DONE / STUDENT (explain in report) |
| I1 | Same training corpus for n-gram, RNN, LSTM, Transformer | one vocab hash in every run config (test 1) | DONE |
| I2 | n-gram with smoothing/interpolation | interpolated trigram, λ on validation (`results/part1/ngram/metrics.json`, test 2) | DONE |
| I3 | RNN, LSTM, Transformer with causal self-attention | `src/models.py`; causal mask (test 3) | DONE |
| I4 | Random, trainable input embeddings for the three neural models | `make_embedding` (std 0.1, `<pad>` row 0); `params.frozen = 0` in Part I metrics | DONE |
| I5 | Document architectures, hyperparameters, context, decoding | Full settings: `configs/base.yaml`, `configs/lr_selected.yaml`, DECISIONS.md, README; report currently gives only a compressed setup | DONE / STUDENT (add architecture/decoding detail to report) |
| I6 | All models predict each target using only preceding tokens | tests 3, 15; n-gram context reset per chunk (test `context_reset_at_chunk_start`) | DONE |
| I7 | Test perplexity / CE on common evaluation setup, padding excluded | `results/tables/part1_main.md` (ppl + CE, mean ± std over 3 seeds); tests 4, 5 | DONE |
| I8 | Training time, model size, inference time/throughput | `part1_main` (param split, time to best, train time, s/epoch, GPU/CPU tok/s, latency); `results/part1/benchmark.json` | DONE |
| I9 | State hardware, training budget, timing procedure; common inference workload | README (hardware, runtimes), `benchmark.json` (`protocol`, hardware string), DECISIONS.md (budget, throttling) | DONE / STUDENT (state in report) |
| I10 | Train/val loss curves for neural models; discuss convergence/overfitting | `figures/part1_loss_curves.{png,pdf}`; per-epoch history in `metrics.json` | DONE / STUDENT (discussion) |
| I11 | Generated text from same prompts, matched decoding & lengths; discuss repetition, fluency, consistency | `results/part1/generations.json`, `results/tables/generation_metrics.md` | DONE / STUDENT (discussion) |
| I12 | Explain tradeoffs; separate observations from explanations | `report/draft.md` ([OBSERVATION]/[HYPOTHESIS] tags) | STUDENT |
| II1 | One architecture, three embedding settings | LSTM: random-trainable / own Word2Vec frozen / GloVe frozen | DONE |
| II2 | Same embedding dim, architecture, data, training procedure | dim 100 = GloVe-100d; test 10 (configs differ only in `embedding.*`, identical non-embedding init); same LR 1e-2 | DONE |
| II3 | Document vocabulary coverage and missing-word initialisation | `results/part2/embeddings/*_info.json`, `part2_embeddings` table, DECISIONS.md | DONE |
| II4 | Entire input-embedding table fixed in the fixed settings | test 8; `params.frozen = 2.89M` in Part II metrics | DONE |
| II5 | Compare test quality, convergence, stability, generated text | `part2_embeddings` (ppl, best epoch, seed variance, spikes), `figures/part2_loss_curves`, `results/part2/generations.json` | DONE / STUDENT (discussion) |
| II6 | Cost of own embeddings reported separately | Word2Vec 403 s (`own_frozen_info.json`, `part2_embeddings` column, README) | DONE |
| II7 | Interpret trainability vs additional data of public embeddings | report draft §3 hypotheses + GloVe overlap limitation | STUDENT |
| II8 | Reuse Part I model for setting 1 if matching; train two new LMs | Part II "random" row = Part I LSTM runs (not retrained) | DONE |
| III1 | Simple supervised task using Part I/II models | AG News 4-class topic classification | DONE |
| III2 | ≥ 2 representation choices on the same labeled data | main: LSTM vs Transformer (mean pooling); also last-state pooling, Part II LSTMs, GloVe-average baseline; all on LM seeds 0, 1, 2 | DONE |
| III3 | Describe how inputs become fixed-length representations | report draft §4; `src/part3.py: pool` (test 11) | DONE / STUDENT (describe) |
| III4 | Same classifier family and evaluation procedure for both | StandardScaler + LogisticRegression, C on validation, test once (test 12) | DONE |
| III5 | Held-out metric (accuracy / macro-F1) and representative errors | `part3_downstream`, `part3_seeds` (mean ± std), `part3_pairwise` (McNemar), `results/part3/errors.json`, `figures/part3_confusion` | DONE / STUDENT (examine errors) |
| III6 | Which representation works better; LM ranking vs downstream ranking | `part3_rank`, `part3_rank_seeds`, `figures/part3_ppl_vs_acc` | DONE / STUDENT (discussion) |
| F1 | Clear comparison question | report draft §0 (suggested question) | STUDENT |
| S1 | PDF report ≤ 6 pages | `report/draft.md` is a skeleton with all numbers | STUDENT |
| S2 | Report states dataset/preprocessing, models, criteria, observations, limitations | report draft §1–5 | STUDENT |
| S3 | README with data access, preprocessing, dependencies, commands, configs, seeds | `README.md` (incl. actual runtimes), `requirements.txt` | DONE |
| S4 | No API keys / credentials / unshareable data | no data or keys in the repo; `.gitignore` excludes checkpoints and caches | DONE |
| A1 | Cite external code, datasets, pretrained models, tool assistance | `AI_USE.md` (skeleton), README data section | STUDENT (complete AI_USE.md) |

No requirement is MISSING on the code/experiment side; the remaining items are the student's writing.
