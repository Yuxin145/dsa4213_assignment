# Requirements checklist (Assignment1_2026.pdf → evidence)

Status: "READY" = code + test in place, number produced by the full run; "PENDING RUN" = filled once the full runs
(`python run_all.py --stage full`) finish; "STUDENT" = needs the student's own writing. Smoke versions of every
artifact exist in `results_smoke/` (pipeline verified end to end).

| # | PDF requirement | Satisfied by | Status |
|---|---|---|---|
| G1 | Choose own English corpus for Parts I/II | WikiText-2 (`configs/base.yaml: data`, README) | READY |
| G2 | Part III: suitable labeled dataset, representations from a Part I/II LM | AG News, `src/part3.py`, `results/part3/results.json` (`lm_checkpoints`: path + SHA-256) | PENDING RUN |
| G3 | Train required LMs yourself from scratch | `src/train.py`; random init (test 10 shows no Part I weights reused in Part II) | READY |
| G4 | Computationally reasonable dataset/model size | 2M tokens, ~10–12M params; profiling in VERIFICATION.md | READY |
| G5 | Comparisons reasonable and clearly explained | shared data/vocab/targets (tests 1, 5), DECISIONS.md, report draft | READY / STUDENT |
| I1 | Same training corpus for n-gram, RNN, LSTM, Transformer | `src/data.py`, one vocab hash for all (test 1) | READY |
| I2 | n-gram with smoothing/interpolation | `src/ngram.py` interpolated trigram, λ on validation (test 2) | READY |
| I3 | RNN, LSTM, Transformer with causal self-attention | `src/models.py`; causal mask (test 3) | READY |
| I4 | Random, trainable input embeddings for the three neural models | `make_embedding`, std 0.1, `<pad>` row 0 (test `param_split_and_pad_row`) | READY |
| I5 | Document architectures, hyperparameters, context, decoding | `configs/base.yaml`, `configs/lr_selected.yaml`, DECISIONS.md, README | READY |
| I6 | All models predict each target using only preceding tokens | test 3 (causality), test 15, n-gram context reset (test `context_reset_at_chunk_start`) | READY |
| I7 | Test perplexity / CE on common evaluation setup, padding excluded | `results/tables/part1_main.md`; tests 4, 5 | PENDING RUN |
| I8 | Training time, model size, inference time/throughput | `part1_main` (param split, time to best, s/epoch, GPU/CPU tok/s, latency); `results/part1/benchmark.json` | PENDING RUN |
| I9 | State hardware, training budget, timing procedure; common inference workload | README (hardware), `benchmark.json` (`protocol`, hardware string), VERIFICATION.md (budget) | PENDING RUN |
| I10 | Train/val loss curves for neural models; discuss convergence/overfitting | `figures/part1_loss_curves.png/pdf`; per-epoch logs in `metrics.json` | PENDING RUN / STUDENT |
| I11 | Generated text from same prompts, matched decoding & lengths; discuss repetition, fluency, consistency | `results/part1/generations.json`, `results/tables/generation_metrics.md` (distinct-1/2, repeated trigrams, `<unk>` rate) | PENDING RUN / STUDENT |
| I12 | Explain tradeoffs; separate observations from explanations | `report/draft.md` with [OBSERVATION]/[HYPOTHESIS] tags | STUDENT |
| II1 | One architecture, three embedding settings (random trainable / own fixed / public fixed) | LSTM; `src/pipeline.py: VARIANTS`, `src/embeddings.py` | READY |
| II2 | Same embedding dim, architecture, data, training procedure | dim 100 = GloVe-100d; test 10 (configs differ only in `embedding.*`, identical non-embedding init) | READY |
| II3 | Document vocabulary coverage and missing-word initialisation | `results/part2/embeddings/*_info.json`, `part2_embeddings` table, DECISIONS.md | PENDING RUN |
| II4 | Entire input-embedding table fixed in the fixed settings | test 8 (`requires_grad=False`, not in optimizer, bit-identical after training) | READY |
| II5 | Compare test quality, convergence, stability, generated text | `part2_embeddings` table (ppl, best epoch, seed variance, spikes), `figures/part2_loss_curves`, `results/part2/generations.json` | PENDING RUN / STUDENT |
| II6 | Cost of own embeddings reported separately | `own_frozen_info.json: train_time_s`, column in `part2_embeddings` | PENDING RUN |
| II7 | Interpret trainability vs additional data of public embeddings | report draft §3 hypotheses + GloVe overlap limitation | STUDENT |
| II8 | Reuse Part I model for setting 1 if matching; train two new LMs | Part II "random" row = Part I LSTM runs (not retrained) | READY |
| III1 | Simple supervised task using Part I/II models | AG News topic classification | PENDING RUN |
| III2 | ≥ 2 representation choices on the same labeled data | main: LSTM vs Transformer (mean pooling); optional: last-state pooling, Part II LSTMs, GloVe-average baseline | PENDING RUN |
| III3 | Describe how inputs become fixed-length representations | report draft §4; `src/part3.py: pool` (test 11) | READY |
| III4 | Same classifier family and evaluation procedure for both | StandardScaler + LogisticRegression, C on validation, test once (test 12) | READY |
| III5 | Held-out metric (accuracy / macro-F1) and representative errors | `part3_downstream` table, `results/part3/errors.json`, `figures/part3_confusion` | PENDING RUN |
| III6 | Which representation works better; LM ranking vs downstream ranking | `part3_rank` table, `figures/part3_ppl_vs_acc` | PENDING RUN / STUDENT |
| F1 | Clear comparison question | report draft §0 | STUDENT |
| S1 | PDF report ≤ 6 pages | `report/draft.md` is a skeleton | STUDENT |
| S2 | Report states dataset/preprocessing, models, criteria, observations, limitations | report draft §1–5 | STUDENT |
| S3 | README with data access, preprocessing, dependencies, commands, configs, seeds | `README.md`, `requirements.txt` | READY (runtimes after run) |
| S4 | No API keys / credentials / unshareable data | repository contains no data or keys (`.gitignore` excludes checkpoints/caches) | READY |
| A1 | Cite external code, datasets, pretrained models, tool assistance | `AI_USE.md`, README data section | STUDENT (complete) |
