# DECISIONS.md — RULE choices, interpretations and deviations

Levels refer to CLAUDE.md §0. Entries are appended as the pipeline runs; evidence paths are given.

## Conflicts between the PDF and CLAUDE.md
None found. CLAUDE.md is a stricter instance of the PDF (the PDF leaves corpus, sizes and settings to the
student). WikiText-2 as the "own corpus", AG News for Part III and LSTM for Part II are all within the PDF.

## Data / vocabulary
- **V = 28,913** (training-split types after lowercasing, incl. `<pad>`=0, `<unk>`=1, `<eos>`=2). Ordering:
  specials, then frequency desc, ties alphabetical (deterministic). Hash in `results/vocab.json`.
- WikiText-2 v1 is a closed-vocabulary corpus (rare words already `<unk>` in all splits), so after lowercasing
  there are essentially no validation/test-only types; test 1 therefore also checks the rule on a synthetic example.
- Smoke mode uses the same full vocabulary (built from the full training split, cheap) and the first 1% of each
  split's token stream.

## Common evaluation / training mechanics (FREE unless noted)
- Training uses the same non-overlapping 128-token chunks as evaluation, reshuffled every epoch with a
  generator seeded by (seed, epoch) — resume reproduces the permutation without extra state.
- Loss per optimizer step = sum NLL over the effective batch of 32 chunks / number of non-pad targets in that
  batch (identical with or without gradient accumulation).
- **AMP only on CUDA.** CPU runs (unit tests, CPU smoke) are fp32 because fp16 autocast + GradScaler is a CUDA
  feature. All full runs and all GPU benchmarks use AMP; CPU benchmark throughput is fp32 for all models.
- Weight decay 0.01 applied to all trainable parameters (incl. biases/LayerNorm/embeddings) for every model.
- Transformer: dropout applied to `proj(embedding) + positional` (the embedding-block output); RNN/LSTM: to the
  100-d embedding output. Both then dropout before the output layer; inter-layer dropout via `nn.RNN/LSTM(dropout=)`
  and inside the encoder layers. `encode()` returns the last layer (after final LayerNorm for the Transformer).
- Model construction order is core → output layer → embedding, then the training RNG is re-seeded with
  `seed + 1`. Consequence: for a given seed, every Part II variant has exactly the same non-embedding init and
  the same dropout/shuffle stream as the Part I LSTM (test 10). Only the embedding block differs.
- Early stopping: "improved" means strictly lower validation CE; patience counter resets on improvement; the
  best checkpoint is restored before the (single) test evaluation.
- Time to best = cumulative wall time (train + per-epoch validation) up to the end of the best epoch.

## n-gram
- Trigram counts over the continuous training stream (crossing `<eos>`, like the neural models' chunks).
- Context reset per chunk: the first target of a chunk only has a bigram context (u is absent), so its λ3 is
  renormalised away — matching exactly what the neural models see.
- λ grid: all 1,771 points of step 0.05 on the 4-simplex, CE on validation; points with λ1 + λ0 = 0 are excluded
  (they could give zero probability). Uniform term is 1/V over all V ids (incl. `<pad>`; `<pad>` is masked at
  generation time for every model alike).

## Tests
- **Overfit test (7)**: with the training dropout (0.3) and lr 3e-3, the 2-layer LSTM had not memorised a
  4×32-token batch after 300 steps (eval loss 2.47), while the RNN and Transformer had. Diagnosis (scratch runs):
  identical behaviour in fp32 (not AMP); a 1-layer LSTM memorises in ~100 steps; the stacked LSTM with default
  init and small-magnitude inputs (emb std 0.1) starts slowly but its loss falls steadily. The test therefore uses,
  for all three models alike, dropout 0 and lr 1e-2 (still AdamW wd 0.01, clip 1.0, AMP on CUDA, real V), and
  keeps the criterion (eval loss < 0.5 within 300 steps). Not a code change to any model.
  Implication noted for the LR sweep: the LSTM may prefer a higher LR.
- Training-loop tests (13 determinism, 14 resume, parts of 10) use a reduced vocabulary (2,000 most frequent
  training types, others → `<unk>`) of the smoke data so that they run on CPU in seconds; the code path is the
  same `train_run`. Model-level tests (3, 4, 5, 6, 7, 9, 11, 15) use the real V.

## Part II
- Word2Vec sentences = preprocessed training lines **including `<eos>`**, so the own table covers every training
  type (min_count 1) incl. `<unk>` and `<eos>`; `<unk>` keeps its trained vector (the "mean of covered vectors"
  rule applies only when `<unk>` is missing, as for GloVe).
- The frozen tables are built once with a fixed seed (0) for missing-word rows, shared by all three LM seeds, so
  seed-to-seed variance reflects LM training only.
- GloVe loaded via `gensim.downloader` "glove-wiki-gigaword-100" (a native-format copy is cached in `.cache/` to
  avoid re-parsing the text file; same vectors).

## Part III
- Text conversion: HTML-unescape (twice; AG News has doubly-escaped entities), broken `#NN;` entities, `quot;`
  etc., backslashes → space, HTML tags removed; lowercase; `@,@` / `@.@` inside numbers, `@-@` inside words;
  leading/trailing punctuation split unless the token is in the vocab (keeps `u.s.`, `st.`); contractions split
  WikiText-style (`don 't`, `street 's`). No `<eos>` appended. Truncate to 128 tokens.
- Features extracted in fp32 (no autocast), eval mode, `no_grad`. C chosen by validation accuracy (ties → smaller C).

## RULE: micro-batch (§6.3)
- **32 for every model** (no accumulation): peak GPU memory 1.78 / 1.80 / 1.89 GB (RNN / LSTM / Transformer)
  of 4 GB in the profiling epoch (`results/profile/profile.json`).

## RULE: peak LR (§6.3) — evidence `results/lr_sweep/selected.json`, log `logs/lr_sweep.log`
Best validation CE after 2 epochs, seed 0 (grid {3e-4, 1e-3, 3e-3}; one edge extension when the best was at an edge):

| arch | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | **selected** |
|---|---|---|---|---|---|---|
| RNN | 6.715 | **6.181** | 6.206 | 6.927 | – | **3e-4** (edge → tried 1e-4, worse) |
| LSTM | – | 6.433 | 6.163 | 5.862 | **5.582** | **1e-2** (edge → tried 1e-2, better; no further extension per rule) |
| Transformer | – | 6.103 | 5.789 | **5.635** | 5.742 | **3e-3** (edge → tried 1e-2, worse) |

- Part II reuses the LSTM value (1e-2). Written to `configs/lr_selected.yaml` (generated, merged into the config).
- Caveat for the report: a 2-epoch sweep rewards fast early progress; 1e-2 (LSTM) is high for AdamW and may give
  loss spikes or earlier overfitting in long runs. This is recorded per epoch (spikes are counted in the Part II
  table) and early stopping restores the best checkpoint; the rule was applied as written, not tuned further.
- No GradScaler-skipped steps in any sweep run (AMP stable).

## RULE: epochs / budget (§9)
- Sweep GPU time 0.93 h. Sweep epoch times varied 66–174 s because the laptop GPU throttles (see
  VERIFICATION.md NEEDS HUMAN: SW thermal slowdown at ~75 °C, clocks down to 652 MHz, plus SW power cap).
- Re-estimate with the sweep's median epoch times (RNN 100 s, LSTM 163 s, Transformer 163 s; Part II ≈ LSTM):
  typical (assumed 15 epochs per run — a 2-epoch sweep cannot show epochs-to-best) 9.4 h + 0.93 h sweep > 8 h.
  **Ladder step 1 applied: max_epochs 40 → 25** (`configs/base.yaml`). The typical estimate does not change, so
  step 2 (escalate) is triggered under throttled conditions → NEEDS HUMAN.
- A short speedcheck right after the sweep (cooler GPU) measured 47 / 58 / 66 s per epoch → typical 3.8 h,
  worst (25 epochs) 6.3 h: within budget. Because speed depends on the thermal state, the full stage re-checks the
  budget **after all seed-0 runs with their real times**, and stops before seeds 1–2 if sweep + seed 0 + 2× seed 0
  would exceed 8 h (`results/budget_check.json`; override only with `--ignore-budget` = human approval).
- Consequence to report: a model that is still improving at epoch 25 stops with `stopped_reason = max_epochs`
  (visible in the tables) — a training-budget limitation, not convergence.

### Update (2026-10-11 03:12): clean re-profiling → max_epochs back to 40
- The sweep's epoch times (and the first profiling run) were measured while the machine was not idle (the student
  was using a browser; some of my own tests overlapped the first sweep runs) and the GPU was throttling. After the
  student closed other programs and improved cooling, one full-data epoch per model was re-measured
  (`results/profile/profile.json`, logs `logs/profile2.log`, GPU samples `logs/profile2_gpu.csv`; old results kept
  in `results/profile_run1/`):

  | | RNN | LSTM | Transformer |
  |---|---|---|---|
  | first profiling (not idle, throttled) | 110.3 s | 131.0 s | 145.1 s |
  | clean re-profiling | **49.2 s** | **55.9 s** | **54.6 s** |

  GPU under load: mean 33.0 W (max 34.7 W; before ~15–22 W), SM clock mean 1,522 MHz (min 1,170; before down to
  652), temperature 59–75 °C; thermal slowdown (`0x20`) still in 12/33 samples, power cap (`0x4`) in 20/33.
  Losses are identical to the first run (e.g. RNN train 7.1164 / val 6.5992): speed does not change results.
- Re-estimate: one epoch of all five variants ≈ 272 s; typical (15 epochs/run) 3 × 272 × 15 ≈ 3.4 h + 0.93 h sweep
  ≈ 4.3 h ≤ 8 h → **the §9 ladder is not triggered; max_epochs restored to 40** (worst case ≈ 9.1 h + 0.93 h, the
  same order as CLAUDE.md's reference worst case of 9.6 h; the after-seed-0 budget check still guards seeds 1–2).
  The earlier 40 → 25 reduction is withdrawn: it was based on contaminated timings, and 25 epochs risked cutting
  off the RNN (lr 3e-4), which learns slowly.
- Not affected by the speed issue: LR selection, all losses/perplexities, the n-gram results, all tests.
  Affected (timing only, to be noted in the report): n-gram fit time (42 s) and Word2Vec time (403 s) may be
  slightly inflated by concurrent use; the sweep's own epoch times are not reported as results.
