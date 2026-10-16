# DSA4213 Assignment 1 — Language models, embeddings and representations

Code for three experiments:

- **Part I:** an interpolated trigram, RNN, LSTM and Transformer language model on WikiText-2.
- **Part II:** three input-embedding settings for the LSTM: random and trainable, own Word2Vec (frozen), GloVe (frozen).
- **Part III:** frozen LSTM and Transformer features for AG News topic classification.

## Repository layout
```
run_all.py            single entry point for all stages
configs/base.yaml     all hyperparameters (single source of truth)
configs/lr_selected.yaml  learning rates chosen by the LR sweep (generated)
src/                  data, n-gram, neural models, training, evaluation, generation, embeddings, Part III, pipeline
scripts/              LR sweep, embedding building, benchmarks, tables, figures, report draft
tests/                pytest suite (correctness checks: causality, padding, perplexity definition, ...)
results/              result JSONs, per-run configs/logs/metrics, tables (checkpoints *.pt are not committed)
figures/              loss curves, confusion matrices, ppl-vs-accuracy plot (PNG + PDF)
```

## Setup
- Python 3.12 and PyTorch `2.14.1+cu132` (CUDA build). Other packages are pinned in `requirements.txt`.
  ```
  pip install -r requirements.txt
  ```
  On Windows, install torch from the official CUDA index first. A plain `pip install torch` can replace the CUDA
  build with a CPU-only one.
- Hardware used for all full runs and timings: a laptop with an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB, Ampere),
  an Intel Core CPU (Raptor Lake, 14 torch CPU threads) and 16 GB RAM. The laptop ran on AC power with no other
  heavy apps open. Thermal throttling is possible and limits how precise the timing numbers are.
- Clone the repository with `git`; do not download it as a zip. Full-result runs record the git commit in every
  result manifest and refuse to start if the experiment source has uncommitted changes. Smoke runs and tests do
  not need this.

## Data access (downloaded automatically on first run)
- WikiText-2: HuggingFace `Salesforce/wikitext`, config `wikitext-2-v1` (official splits).
- AG News: HuggingFace `fancyzhx/ag_news`. A stratified 10k sample of the official train split (8k train / 2k
  validation, seed 0); the official test split (7,600) is the test set.
- GloVe 6B 100d: `gensim.downloader` `"glove-wiki-gigaword-100"` (~128 MB, cached in `~/gensim-data`).

No dataset is redistributed in this repository.

## Preprocessing and evaluation
- **Parts I/II:**
  - Lowercase, split on whitespace, drop empty lines, and append `<eos>` to each line.
  - The vocabulary is every type in the training split, with no frequency cutoff, plus `<pad>` (id 0), `<unk>`
    and `<eos>`. It is saved once to `results/vocab.json` with a SHA-256 hash, and every model and Part III check
    that hash.
- **Evaluation:**
  - Each split is one token stream, cut into non-overlapping 128-token chunks. No hidden state is carried across
    chunks, and the n-gram is scored on exactly the same targets.
  - Padded targets are excluded. ppl = exp(total NLL / number of non-pad targets).
- **Part III:**
  - AG News text is converted to WikiText tokenization conventions (punctuation split, `@-@`, `@,@`, `@.@`) and
    truncated to 128 tokens.
  - Features are the frozen LM's last-layer states, mean-pooled over non-pad positions.
  - The classifier is `StandardScaler` + `LogisticRegression`, with C chosen on validation only.

## Commands
```
python -m pytest -q                             # test suite
python run_all.py --smoke                       # end-to-end smoke run on ~1% data (CPU OK) -> results_smoke/
python scripts/lr_sweep.py                      # LR sweep -> configs/lr_selected.yaml
python run_all.py --stage full                  # all full runs, generation, Part III (seed 0), benchmarks, tables
python run_all.py --stage part3 --seeds 1 2     # Part III on the seed-1 and seed-2 checkpoints
python run_all.py --stage report                # regenerate tables and figures from results/**/*.json
```
- **Order of `--stage full`:**
  1. n-gram (λ grid search on validation);
  2. seed 0 of RNN, LSTM, Transformer, LSTM + own Word2Vec and LSTM + GloVe;
  3. generation and Part III on the seed-0 checkpoints;
  4. seeds 1 and 2 of all five neural models;
  5. inference benchmarks (`scripts/benchmark.py`);
  6. tables and figures.
- **Interrupting is safe.** Re-running skips runs that have a `DONE` marker and resumes the others from `last.pt`.
- **Budget check.** After the seed-0 runs, a budget check stops before seeds 1–2 if they would push total GPU time
  over 8 h. Pass `--ignore-budget` to run them anyway.

## Configuration and seeds
- All hyperparameters are in `configs/base.yaml`. Each run saves its resolved `config.yaml`, `log.txt` and
  `metrics.json` in `results/part{1,2}/<model>/seed<k>/`. The run manifest in `metrics.json` records timestamps,
  git commit, config hash, seed, hardware and log path.
- **Neural models:**
  - Embedding dim 100, hidden/d_model 256, 2 layers, dropout 0.3, context 128, no weight tying.
  - Transformer: 4 heads, FFN 1024, pre-LN, causal mask, learned positions.
- **Training:**
  - AdamW (weight decay 0.01), grad clip 1.0, batch 32, AMP (fp16).
  - Transformer: 200-step linear warmup.
  - At most 40 epochs, early stopping with patience 3 on validation loss (the best checkpoint is restored), and a
    90-min cap per run.
- **Learning rates**, chosen by a 2-epoch sweep (seed 0, lowest validation loss) over {3e-4, 1e-3, 3e-3}. When
  the best value was at an edge of the grid, the next value (1e-4 or 1e-2) was also tried. Selected: RNN 3e-4,
  LSTM 1e-2 (reused in Part II), Transformer 3e-3. See `results/lr_sweep/selected.json`.
- **Word2Vec:** skip-gram, dim 100, window 5, min_count 1, negative 10, 10 epochs, trained on the training split
  only.
- **Seeds:**
  - 0, 1 and 2 for every neural model; the n-gram is deterministic.
  - Word2Vec uses seed 0 with 1 worker.
  - The AG News split uses seed 0.
  - Generation sampling uses seed 1234 + prompt index.

## Outputs
- Tables: `results/tables/*.md` and `results/tables/*.csv`.
- Figures: `figures/*.png` and `figures/*.pdf`.
- Generated text: `results/part1/generations.json` and `results/part2/generations.json`.
- Benchmarks: `results/part1/benchmark.json`.
- Embedding coverage and missing-word initialization: `results/part2/embeddings/*_info.json`.
- Part III results and error examples: `results/part3/results.json` and `results/part3/errors.json` for seed 0;
  `results/part3/seed{1,2}/` for seeds 1 and 2.

## Actual runtimes (from the run logs on the laptop above)
| step | time | source |
|---|---|---|
| LR sweep (12 runs × 2 epochs, GPU) | 0.93 h | `results/lr_sweep/selected.json` |
| Trigram counts + λ grid search (CPU) | 42 s + 19 s | `results/part1/ngram/log.txt` |
| Own Word2Vec (CPU, 1 worker) | 403 s | `results/part2/embeddings/own_frozen_info.json` |
| `run_all.py --stage full` (15 LM runs, generation, Part III seed 0, benchmarks) | 5 h 07 min wall clock | `results/run_all_log.txt` |
| — RNN per run (40 epochs, hit the limit) | 32.9–33.2 min | `results/part1/rnn/seed*/metrics.json` |
| — LSTM per run (13–16 epochs, early stopping) | 12.4–15.0 min | `results/part1/lstm/seed*/` |
| — Transformer per run (19–30 epochs, early stopping) | 17.4–27.4 min | `results/part1/transformer/seed*/` |
| — LSTM + own W2V / + GloVe per run | 12.0–14.8 / 14.7–16.8 min | `results/part2/*/seed*/` |
| — total LM training GPU time (15 runs) | 4.94 h | metrics.json `total_time_s` |
| Part III on seed-1 and seed-2 checkpoints | ~3.5 min per seed | `results/part3/seed*/log.txt` |
| Inference benchmarks | 4 min | `results/part1/benchmark.json` |

## Acknowledgements
Datasets and resources:
- WikiText-2 (Merity et al., 2016)
- AG News (Zhang et al., 2015)
- GloVe (Pennington et al., 2014)

Libraries: PyTorch, gensim, scikit-learn, HuggingFace `datasets`.

The code was written with the help of an AI coding assistant (Claude Code). The report describes how tools were
used.
