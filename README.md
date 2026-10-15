# DSA4213 Assignment 1 — Language models, embeddings and representations

Compares an interpolated trigram, RNN, LSTM and Transformer LM on WikiText-2 (Part I), three input-embedding
settings for the LSTM (Part II), and frozen LSTM vs Transformer features for AG News topic classification
(Part III). Specification: `CLAUDE.md`; choices made during the run: `DECISIONS.md`; what was executed and
verified: `VERIFICATION.md`; requirement → evidence map: `REQUIREMENTS_CHECKLIST.md`.

## Setup
- Windows 11, conda env `dsa4213` (Python 3.12, `torch 2.14.1+cu132`). Other packages: `requirements.txt`
  (`<env python> -m pip install -r requirements.txt` — but do **not** let pip reinstall torch on Windows).
- Hardware used for all full runs and timings: laptop, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB, Ampere),
  Intel Core (Raptor Lake, 14 torch CPU threads), 16 GB RAM. Laptop on AC power, no other heavy apps running;
  thermal throttling is possible and is a limitation of the timing numbers.
- NVIDIA "CUDA – Sysmem Fallback Policy" should be "Prefer No Sysmem Fallback" (OOM instead of silent slowdown).

## Data access (downloaded automatically on first run)
- WikiText-2: HuggingFace `Salesforce/wikitext`, config `wikitext-2-v1` (official splits).
- AG News: HuggingFace `fancyzhx/ag_news` (stratified 10k sample of official train → 8k/2k, seed 0; official test).
- GloVe 6B 100d: `gensim.downloader` `"glove-wiki-gigaword-100"` (~128 MB, cached in `~/gensim-data`).
No data is redistributed in this repository.

Preprocessing (Parts I/II): lowercase, whitespace split, drop empty lines, append `<eos>` per line; vocabulary =
all training-split types (no cutoff) + `<pad>`(0), `<unk>`, `<eos>`; saved once to `results/vocab.json` with a
SHA-256 hash that every model and Part III check. Evaluation: each split is one stream cut into non-overlapping
128-token chunks; padded targets excluded; ppl = exp(total NLL / non-pad targets).

## Commands (use the env's interpreter, e.g. `D:\anaconda3\envs\dsa4213\python.exe`)
```
python -m pytest -q                      # all required tests (CLAUDE.md §10)
python run_all.py --smoke                # end-to-end smoke run (1% data, 1 epoch, seed 0) -> results_smoke/
python run_all.py --stage profile        # one epoch per neural model (compute budget)
python scripts/lr_sweep.py               # LR sweep -> configs/lr_selected.yaml
python run_all.py --stage speedcheck     # ~2 min: current s/epoch and projected full-run hours vs the 8 h budget
python run_all.py --stage full           # all full runs, Part III, benchmarks, tables, figures, draft (resumable)
python run_all.py --stage report         # regenerate tables/figures/report draft from results/**/*.json
```
`--stage full` order: n-gram; seed 0 of RNN, LSTM, Transformer, LSTM+own-W2V, LSTM+GloVe; generation and Part III
on seed-0 checkpoints; seeds 1 and 2 of all five; benchmarks (`scripts/benchmark.py`); tables/figures/draft.
Interrupting is safe: re-running skips runs with a `DONE` marker and resumes others from `last.pt`.
After the seed-0 runs a budget check (`results/budget_check.json`) stops before seeds 1–2 if they would push the
total GPU time over 8 h (CLAUDE.md §9); `--ignore-budget` overrides it.

Selected settings: LR RNN 3e-4, LSTM 1e-2 (also Part II), Transformer 3e-3 (2-epoch sweep, `DECISIONS.md`);
max 40 epochs, early stopping patience 3, 90-min cap per run; micro-batch 32 (no accumulation).

## Configuration and seeds
All hyperparameters: `configs/base.yaml`; LRs selected by the sweep: `configs/lr_selected.yaml` (generated).
Each run saves its resolved `config.yaml`, `log.txt`, `metrics.json` (with a run manifest: timestamps, git
commit, config hash, seed, hardware, log path), `best.pt`, `last.pt`, `DONE` in
`results/part{1,2}/<model>/seed<k>/`. Seeds: 0, 1, 2 for every neural model; n-gram is deterministic;
Word2Vec seed 0 (workers 1); AG News split seed 0; generation sampling seed 1234 + prompt index.

## Outputs
- `results/tables/*.md|csv` (Part I/II/III tables), `figures/*.png|pdf`, `report/draft.md`.
- `results/part1/generations.json`, `results/part2/generations.json`, `results/part1/benchmark.json`, `results/part3/seed{1,2}/`,
  `results/part3/results.json`, `results/part3/errors.json`, `results/part2/embeddings/*_info.json`.

## Actual runtimes (from the run logs; laptop above, performance mode, otherwise idle)
| step | time | source |
|---|---|---|
| LR sweep (12 runs × 2 epochs, GPU) | 0.93 h (partly while the GPU was throttling) | `results/lr_sweep/selected.json` |
| Trigram counts + λ grid search (CPU) | 42 s + 19 s | `results/part1/ngram/log.txt` |
| Own Word2Vec (CPU, 1 worker) | 403 s | `results/part2/embeddings/own_frozen_info.json` |
| `run_all.py --stage full` (all 15 LM runs, generation, Part III seed 0, benchmarks, report) | 5 h 07 min wall clock (2026-10-11 03:20 → 08:27) | `results/run_all_log.txt` |
| — RNN per run (40 epochs, hit the limit) | 32.9–33.2 min | `results/part1/rnn/seed*/metrics.json` |
| — LSTM per run (13–16 epochs, early stopping) | 12.4–15.0 min | `results/part1/lstm/seed*/` |
| — Transformer per run (19–30 epochs, early stopping) | 17.4–27.4 min | `results/part1/transformer/seed*/` |
| — LSTM + own W2V / + GloVe per run | 12.0–14.8 / 14.7–16.8 min | `results/part2/*/seed*/` |
| — total LM training GPU time (15 runs) | 4.94 h (+ 0.93 h sweep = 5.87 h of the 8 h budget) | metrics.json `total_time_s` |
| Part III on seed-1 and seed-2 checkpoints (`--stage part3 --seeds 1 2`) | ~3.5 min per seed | `logs/part3_seeds12.log` |
| Inference benchmarks | 4 min | `results/part1/benchmark.json` |

Part III on additional seeds: `python run_all.py --stage part3 --seeds 1 2` writes `results/part3/seed1/` and
`results/part3/seed2/`; `python run_all.py --stage report` aggregates them (`results/tables/part3_seeds.md`,
`part3_pairwise.md`, `part3_rank_seeds.md`).
