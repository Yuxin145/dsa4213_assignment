# VERIFICATION.md

> Restored on 2026-10-11 ~03:15 by the agent: this file was deleted by accident around 02:04 (confirmed by the
> student). No git commit existed, so it was rewritten from the agent's last version plus the 03:12 updates below.

## STATUS SUMMARY (2026-10-11, before the full runs)
- **Implemented:** the whole pipeline — data/vocab, interpolated trigram, RNN/LSTM/Transformer, AMP training with
  checkpoint/resume and early stopping, generation, Part II embeddings, Part III classifier, benchmarks, tables,
  figures, report draft (`src/`, `scripts/`, `run_all.py`, `configs/`).
- **Tests:** `python -m pytest -q` → **65 passed, 0 failed, 0 skipped** (`logs/pytest_final.log`), covering all 15
  required tests of CLAUDE.md §10.
- **Executed on the full data:** vocab + data stats; trigram (test ppl **248.73**, val 264.06,
  λ = (0.25, 0.45, 0.25, 0.05)); profiling (twice — see Phase 5); LR sweep (0.93 GPU-h; selected RNN 3e-4, LSTM 1e-2,
  Transformer 3e-3); own Word2Vec (403 s CPU) and the GloVe table. End-to-end smoke run of every stage.
- **Not yet executed:** the 15 full LM runs (Part I neural + Part II), generation, Part III, benchmarks on real
  checkpoints, final tables/figures/draft → **`## RUN THIS`** below (planned stop, CLAUDE.md §2).
- **Deviations / RULE outcomes:** max_epochs stays 40 (a temporary 40 → 25 reduction was withdrawn after clean
  re-profiling); the overfit test uses dropout 0 + lr 1e-2; training-loop tests use a 2,000-word vocabulary.
  Details in `DECISIONS.md`.

## NEEDS HUMAN

1. **Resolved (03:12): GPU speed / budget.** The earlier slow timings came from throttling while the machine was in
   use. After you closed other programs and improved cooling, I re-measured: **49.2 / 55.9 / 54.6 s per epoch**
   (RNN / LSTM / Transformer), mean 33 W, clock ~1,500 MHz. Typical estimate ≈ 4.3 h incl. the sweep → within the
   8 h budget; max_epochs back to 40 (DECISIONS.md). Keep these conditions during the full run (performance mode,
   cooling, no other heavy programs), especially during the final benchmark step. Mild thermal slowdown at 75 °C
   still occurs, so mention throttling as a limitation of the timings.
2. The after-seed-0 budget check stays active (`results/budget_check.json`). If it stops before seeds 1–2, the
   options are `--ignore-budget`, Kaggle, or fewer seeds (CLAUDE.md §9).
3. **Recommended: commit the code before the full runs** (`git add -A` then `git commit`), so that every run
   manifest records a real commit hash and project files are protected against accidental deletion (this file was
   lost once). The repo has no commit yet, so manifests say `no-commit`. I did not commit without your instruction.

## Phase 1 — environment, data, vocab, utils (tests 1, 13)
- Interpreter: `D:\anaconda3\envs\dsa4213\python.exe`, `torch 2.14.1+cu132`, `torch.cuda.is_available() = True`,
  GPU "NVIDIA GeForce RTX 3050 4GB Laptop GPU", driver 617.42 (CUDA UMD 13.4), new enough.
- Installed with pip (dry run first: no change to torch/numpy): datasets 5.1.0, gensim 4.4.0, scikit-learn 1.9.1,
  PyYAML, pytest, matplotlib (+ deps). Pinned in `requirements.txt`.
- Data downloaded: WikiText-2 v1, AG News (HF), GloVe 6B-100d (gensim downloader). Log: `logs/download.log`.
- `results/data_stats.json`: **V = 28,913**; tokens train 2,075,677 / val 216,347 / test 244,102;
  `<unk>` rate train 0.0263 / val 0.0542 / test 0.0623. Vocab SHA-256 `86014eda…b153f` (`results/vocab.json`).
- Tests: `tests/test_data.py` (4) + `tests/test_determinism.py` (2): **6 passed**.

## Phase 2 — n-gram, evaluator, generation (tests 2, 5)
- `src/ngram.py`, `src/evaluate.py`, `src/generate.py`.
- Tests: `tests/test_ngram.py` (5) + `tests/test_eval.py` (9, incl. test 4): **14 passed**.

## Phase 3 — neural models, training loop, checkpoint/resume (tests 3, 4, 6, 7, 9, 14, 15)
- `src/models.py`, `src/train.py`. Params (V = 28,913): RNN 10.55M (emb 2.89M / core 0.22M / out 7.43M),
  LSTM 11.21M (core 0.89M), Transformer 11.96M (core 1.64M).
- The overfit test (7) first failed for the LSTM only (eval loss 2.47 after 300 steps with dropout 0.3, lr 3e-3).
  Diagnosis: slow memorisation by the stacked LSTM, not AMP and not a bug (details in DECISIONS.md). The test now
  uses dropout 0 and lr 1e-2 for all three models; the criterion is unchanged. The real models are unchanged.
- The resume test (14) checks that an interrupted-then-resumed run reproduces the uninterrupted run's losses exactly.
- Tests: `test_models.py` (18), `test_checkpoint.py` (4), `test_generation.py` (6): **28 passed**.
- Smoke-trained all three architectures on the GPU (AMP): peak memory 1.75–1.85 GB with micro-batch 32.

## Phase 4 — Part II embeddings, Part III (tests 8, 10, 11, 12)
- `src/embeddings.py`, `src/part3.py`, `src/pipeline.py`.
- AG News conversion OOV (smoke subset): train 18.4% → 6.8%, val 18.8% → 7.5%, test 19.3% → 7.5%. The remaining OOV
  is mostly genuine domain vocabulary missing from WikiText-2 (corp, afp, oracle, darfur, peoplesoft, …). Added
  afterwards: cp1252 dashes → `—`/`–`, and `&` inside words split (`at & t`).
- Tests: `test_part2.py` (7), `test_part3.py` (9): **16 passed** (the saved-config check was skipped before any run
  existed; it ran and passed in the later full-suite runs).

## Phase 5 — end-to-end smoke run and profiling
- `python run_all.py --smoke` (logs `logs/smoke_run.log`, `results_smoke/run_all_log.txt`): data → n-gram →
  smoke LR sweep → 5 LM variants (seed 0, 1 epoch) → generation → Part III → benchmark → tables/figures/draft.
  The first attempt crashed in the benchmark (fewer test batches than warm-up batches in the 1% split). Fixed by
  clamping warm-up to `n_batches − 1` (recorded in `benchmark.json`; no effect on the full split, 60 batches).
  Re-ran `--stage benchmark` and `--stage report`: completed. Outputs in `results_smoke/`, `report/draft_smoke.md`.
- The `--stage full` code path (incl. the budget check) was also run end to end on the smoke data
  (`logs/smoke_full_stage.log`): completed.
- Full test suite: **65 passed** after the smoke run (`logs/pytest_full.log`) and again after the last code
  changes (`logs/pytest_final.log`, 332 s).

### Profiling (§9) — `python run_all.py --stage profile`
| model | run 1: s/epoch (not idle, throttled) | **run 2: s/epoch (clean)** | peak GPU mem | micro-batch | steps/epoch |
|---|---|---|---|---|---|
| RNN | 110.3 | **49.2** | 1,781 MB | 32 | 507 |
| LSTM | 131.0 | **55.9** | 1,804 MB | 32 | 507 |
| Transformer | 145.1 | **54.6** | 1,885 MB | 32 | 507 |

- Run 1: `logs/profile.log`, `results/profile_run1/`. Run 2 (after the student closed other programs and improved
  cooling): `logs/profile2.log`, `results/profile/profile.json`, GPU samples `logs/profile2_gpu.csv` (mean 33.0 W,
  clock mean 1,522 MHz / min 1,170, 59–75 °C; thermal slowdown in 12/33 samples). Losses are identical in both runs
  (e.g. RNN train 7.1164, val 6.5992), so speed does not change results.
- RULE micro-batch: **32 fits** (≤ 1.9 GB of 4 GB) for every model → no gradient accumulation, same for all.
- Per-step breakdown (scratch microbenchmark, throttled): the 256×28,913 output layer + fp32 cross-entropy
  dominate the step. Padding V to a multiple of 8 would save only ~12% and would change the LOCKED output layer, so
  it was not done.

## Phase 6 — LR sweep, n-gram, embeddings (agent); full runs (human)
- **n-gram** (`run_all.py --stage ngram`, `results/part1/ngram/`): counts in 42.1 s, λ grid on validation in 19.1 s
  (1,771 points); best λ = (0.25, 0.45, 0.25, 0.05), val CE 5.5762; **val ppl 264.06, test ppl 248.73** (sanity
  range 200–400 ✓). It scores 244,101 test targets (= test tokens − 1, the same as the neural evaluator) and stores
  1.99 M counts. (Timing measured while the machine may not have been idle.)
- **LR sweep** (`scripts/lr_sweep.py`, log `logs/lr_sweep.log`, `results/lr_sweep/selected.json`): RNN 3e-4,
  LSTM 1e-2 (edge extension), Transformer 3e-3 (edge extension tried, 1e-2 worse). Table in DECISIONS.md. No
  AMP-skipped steps and no NaN/inf. Total 0.93 GPU-h.
- **Budget:** first re-estimated with the sweep's (contaminated) epoch times → ladder step 1 (40 → 25). Then clean
  re-profiling → one epoch of all five variants ≈ 272 s, typical ≈ 3.4 h + 0.93 h sweep ≈ 4.3 h ≤ 8 h → ladder not
  triggered, **`configs/base.yaml: max_epochs = 40`** (DECISIONS.md, update 03:12).
- **Part II embedding step** (`scripts/build_embeddings.py`, log `results/part2/embeddings/log.txt`): own Word2Vec
  trained in **403.1 s** (CPU, workers 1), 28,912 vectors, coverage 100% of types / 100% of training tokens. GloVe
  coverage **96.18% of types, 94.74% of training tokens**; 1,103 missing types (1,102 random rows, `<unk>` = mean).
  Cached; the full run reuses them.

## RUN THIS

In a normal PowerShell terminal, keeping the clean conditions (performance mode, cooling, no other heavy programs):

```powershell
cd D:\Users\peter\Desktop\dsa4213_a1
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
D:\anaconda3\envs\dsa4213\python.exe run_all.py --stage full
```

- Order: data check → trigram (DONE, skipped) → seed 0 of RNN, LSTM, Transformer, LSTM+own-W2V, LSTM+GloVe →
  generation (seed 0) → Part III (seed-0 checkpoints) → budget check → seeds 1 and 2 of all five → benchmarks
  (`results/part1/benchmark.json`) → tables, figures, `report/draft.md`.
- Expected GPU time: typically ~3.5–4 h at the clean profiling speed (worst case ~9 h if most runs went to 40
  epochs); seed 0 alone is about a third. Keep the laptop plugged in and stop it from sleeping (e.g.
  `powercfg /change standby-timeout-ac 0`; restore it afterwards).
- **Interruptions are safe:** run the same command again. Finished runs (`DONE`) are skipped, and the current run
  resumes from `last.pt` at its last completed epoch.
- Progress: `results/run_all_log.txt` (overview) and `results/part*/<model>/seed*/log.txt` (per epoch).
- When it finishes (or stops at the budget check), come back to the agent. It will check the logs and the sanity
  ranges (CLAUDE.md §11), then complete Phase 7 (README runtimes, checklist, final summary).
