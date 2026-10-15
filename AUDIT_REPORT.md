# Independent technical audit — DSA4213 Assignment 1

Audit date: 2026-10-15 (SGT)  
Audited commit: `d2e17b18f5665008d4f287ee9cc2b16a5e192e49`

> This report was created before any audit fix was applied. A preliminary file inventory incorrectly reported
> `figures/` as empty; direct Python/Git inspection showed that all eight figure files were already tracked and
> non-empty. That preliminary observation was retracted and is not classified as a finding.

## Verdict

**Not ready for final human review or submission.** No Critical defect was identified, but two Major spec-level
issues remain in the existing artifacts: AMP scaler overflows were not escalated, and validity-critical LR
selection is not tied to an exact clean source revision. The student also still needs to turn the unfinished
draft into the final six-page PDF and complete `AI_USE.md`.

No experiment was rerun because the follow-up findings are Major, not Critical, as required by the audit brief.

## Scope and method

- Read `Assignment1_2026.pdf`, `CLAUDE.md`, `DECISIONS.md`, and `VERIFICATION.md`.
- Inspected all source modules, runners, configs, tests, result JSON classes, run logs, saved configs, tables,
  report draft, README, checklist, and artifact inventory.
- Ran the complete unmodified suite using the required interpreter: **65 passed, 1 warning in 221.59 s**.
- After the documentation fix, reran the complete suite: **65 passed, 1 warning in 370.60 s**. The warning is
  only about converting an embedding-weight standard deviation tensor to a Python float in a test assertion.
- After the Major follow-up fixes and two new regression tests, reran the complete suite:
  **67 passed, 1 warning in 340.52 s**. The strengthened regression file was also rerun directly:
  **2 passed in 12.77 s**.
- Used a detached Git worktree as the mutation scratch copy. Each mutation was reverted before the next one;
  the scratch tree had no tracked changes after the six checks.
- Rebuilt generated tables and `report/draft.md` in the scratch worktree and compared them to the committed
  versions with `git diff --exit-code`.

## Verified checks

### Mutation sensitivity

All six required mutations were detected by existing tests:

| Mutation | Existing test exercised | Observed failure |
|---|---|---|
| Remove Transformer causal mask | `test_causality` | Transformer case failed; 1 failed, 2 passed |
| Include padding in LM loss/count | padding tests in `test_eval.py` | 4 failed |
| Unfreeze Part II embedding | `test_frozen_embeddings` | both frozen variants failed |
| Build saved pipeline vocab from train + test | `test_data.py` | saved/shared vocab hash test failed |
| Tie input and output weights | `test_no_weight_tying` | all three architectures failed |
| Mean-pool over padded positions | `test_pooling_excludes_pad` | all three mean-pooling cases failed |

The vocabulary mutation is detected partly through the checked-in artifact hash. The direct synthetic and real
tests verify train-only construction, but do not independently exercise `load_lm_data` with mocked split-specific
tokens. This is a residual test-design weakness, not evidence that the present vocabulary leaked.

### Validity-critical implementation

- **Splits and selection:** WikiText vocabulary construction uses training lines only; Word2Vec receives training
  lines only; the LR sweep evaluates validation only (`eval_test=False` and no test metric in all 12 sweep runs);
  Part III samples only the official AG News train split for its 8k/2k split and uses the separate official test
  split only for final prediction. Stored Part III train/validation indices are disjoint and exactly reproduce
  from `split_seed=0`.
- **Alignment and common targets:** `make_chunks` implements `x=t[i:i+128]`, `y=t[i+1:i+129]`; all evaluators
  count 216,346 validation and 244,101 test targets. The n-gram resets higher-order context at each chunk start.
- **Causality:** RNN/LSTM operate left-to-right without carried state; Transformer uses the registered upper
  triangular `-inf` mask and `is_causal=True`. Behavioral causality and autoregressive generation tests passed.
- **Loss/perplexity:** neural and n-gram paths aggregate summed NLL and divide once by the non-pad token count.
  Padding is excluded from both training and evaluation.
- **Part II fairness:** every main Part II saved config differs from the corresponding Part I config only in
  `embedding.type` and `embedding.freeze`; LSTM LR and all other settings match. Model construction initializes
  non-embedding parameters before the embedding table and never loads a Part I checkpoint. All six frozen
  checkpoints contain embedding rows bit-identical to their recorded matrix; the recorded frozen parameter count
  is exactly the matrix size.
- **Part III:** features use `model.eval()`, `no_grad`, last-layer `encode` states, and non-pad pooling.
  `select_C` has no test arguments; final classifiers are fitted on train only. Confusion matrices independently
  reproduce saved accuracy and macro-F1 for all seven representations and all three LM seeds.
- **Benchmarks:** common chunked test workload, batch 32, five warm-up batches excluded, three throughput runs,
  one generation warm-up, five latency runs, and CUDA synchronization immediately around timed regions were
  confirmed in code and artifacts.

### Config compliance

- All 15 main run configs match the current resolved base/LR config exactly, apart from the two allowed Part II
  embedding keys. Seeds, selected LRs, batch/micro-batch, AMP, model sizes, context length, optimizer settings,
  patience, wall-clock cap, and 40-epoch main-run limit comply with `CLAUDE.md`.
- The 12 LR-sweep configs correctly use seed 0 and their candidate LR, including the required single edge
  extension. The three clean profile runs have `run.max_epochs=1`; their saved base section still contains the
  temporary 25-epoch value. This does not affect profiling and the temporary value/restoration is documented in
  `DECISIONS.md`.
- Older sweep/profile configs predate the later-added budget block and selected-LR overlay. Their operative run
  values are present; this is historical config evolution rather than a main-run deviation.

Exhaustive resolved-config differences from the final current config:

| Runs | Difference | Compliance / documentation |
|---|---|---|
| 9 Part I main runs | none | compliant |
| 6 Part II main runs | only `embedding.type` and `embedding.freeze` | required by Part II; documented |
| 12 LR-sweep runs | selected-LR fields are null and `budget.*` is absent; `run.lr` is the candidate | expected because these runs produce the selected LRs and predate the later budget block; sequencing is documented, schema absence is not separately noted |
| 3 archived `profile_run1` runs | selected-LR fields are null and `budget.*` is absent | profiling uses explicit LR 1e-3/max 1; no result impact; old profiling is documented |
| 3 clean `profile` runs | base `train.max_epochs=25` instead of final 40 | `run.max_epochs=1`; temporary 25→40 history is explicitly documented |

### Artifact authenticity

- Enumerated 53 full-result JSON files: 40 contain manifests; 13 supporting/derived JSON files do not.
- Checked all 33 neural run metric artifacts: saved config hash, `DONE`, `best.pt`, `last.pt`, sequential history,
  epoch count, best epoch/value, cumulative time, and step count. No inconsistency was found.
- Loaded every one of the 33 neural `best.pt` files (15 main, 12 LR sweep, three clean profile, three archived
  profile) and reevaluated the full validation split on CUDA/AMP. Every reproduced CE matched the reported value
  exactly (observed delta 0), and every checkpoint's stored `val_ce` matched `best_val_ce`.
- Refit the trigram, reran all 1,771 validation lambda candidates, and rescored validation/test. The selected
  `(0.25, 0.45, 0.25, 0.05)` and both CEs reproduced to below `1e-12`.
- Recomputed and verified the hashes of all 12 Part III LM checkpoint references; all record
  `params_unchanged=true`.
- The 15 main LM manifests reference valid commit `c35c63c`; that commit differs from the audited code only in
  later Part III multi-seed/report-generation additions, not in LM training/evaluation code.

### Report numbers and PDF requirements

- Regenerating every table and `report/draft.md` from the checked-in result JSONs produced no Git diff. Thus the
  committed generated numbers match the source JSONs and current generators.
- All four required figures exist as non-empty PNG and PDF files. The PNGs were opened and visually checked for
  legible labels, complete plots, and agreement with the corresponding result values.
- Independently mapped the PDF requirements against `REQUIREMENTS_CHECKLIST.md`. Its requirement list is
  substantially complete and correctly leaves report authorship, six-page PDF production, interpretation, and
  AI-use completion to the student.

Checklist item-by-item outcome:

- **Verified DONE:** G1–G4; I1–I4, I6–I8; II1–II4, II6, II8; III1, III2, III4; S3, S4.
- **Evidence verified; student prose still required:** G5; I5, I9–I12; II5, II7; III3,
  III5, III6; S2.
- **Correctly marked wholly student-owned/incomplete:** F1, S1, A1. In particular, no final PDF exists and
  `AI_USE.md` still has placeholders.
- No PDF requirement sentence was found that is absent from the checklist. I5 previously overstated completion
  because full model/decoding detail is not yet in the report; its status was corrected to `DONE / STUDENT`.

## Findings

### Critical

None found.

### Major

#### M-01 — AMP scaler overflows were tolerated instead of escalated

Six main runs record 24 GradScaler-skipped optimizer steps: RNN seeds 0/1/2 have 7/6/7 skips, Transformer seeds
0/1 have 1/2, and GloVe-frozen seed 2 has 1. `train_epoch` detected scale reductions but continued training and
advanced the scheduler. Saved scalar losses remained finite and the largest rate is only 7/20,280 steps
(0.035%), so this is not evidence of general numerical divergence. Nevertheless, `CLAUDE.md` explicitly requires
escalation when AMP produces NaN/inf rather than silently continuing. Existing tests covered finite losses but
did not cover GradScaler-detected non-finite gradients.

Affected run directories:

- `results/part1/rnn/seed0`, `seed1`, `seed2`
- `results/part1/transformer/seed0`, `seed1`
- `results/part2/glove_frozen/seed2`

#### M-02 — LR selection is not tied to an exact clean source revision

All 12 LR-sweep run manifests record `git_commit: no-commit-dirty`, while
`results/lr_sweep/selected.json` has no manifest. These selected LRs feed all 15 main neural runs. The n-gram
result and profiling artifacts have the same provenance weakness, and optional Part III seeds 1–2 record
`c35c63c-dirty`. The 15 main LM runs themselves do identify clean commit `c35c63c`, and this audit exactly
reproduced every saved checkpoint metric and the n-gram result using current code, which provides strong
corroboration but cannot reconstruct the original dirty source state.

The provenance helper also omitted `run_all.py` from dirty detection. Future real-result entry points must refuse
to run with dirty experiment source.

### Minor

#### A-01 — Not every result JSON has the mandated manifest

Thirteen of 53 full-result JSONs lack a `manifest`: vocab/data stats, LR selection, budget check, two embedding
info files, three Part III error files, two profile summaries, the old speedcheck, and `tables.json`. Most are
derived/supporting artifacts whose parent run is traceable, so this does not invalidate a comparison, but it
contradicts the literal “every result JSON” rule and weakens per-file provenance. In addition, the Part I and
Part II generation JSONs have manifests but set `log_file` to null; their messages are recoverable from
`results/run_all_log.txt`, which the manifests do not reference.

#### A-02 — Archived first-profile manifests point to the replacement profile logs

`results/profile_run1/{rnn,lstm,transformer}/metrics.json` still records
`results/profile/<model>/log.txt` after the original directories were moved. The matching archived logs do exist
under `profile_run1`, and their epoch values agree with the JSON, but the manifest path is stale. These timings are
presented only as contaminated historical profiling, not as final benchmark results.

#### A-03 — README examples used bare `python` (resolved)

The README named the required interpreter immediately above the command block, but every example invoked bare
`python`, contrary to the locked rule. The command examples now use
`D:\anaconda3\envs\dsa4213\python.exe` explicitly.

#### A-04 — Submission document and AI-use disclosure remain unfinished

There is no final report PDF. `report/draft.md` still contains 21 `[OBSERVATION]`/`[HYPOTHESIS]` or “Fill in”
markers, and `AI_USE.md` retains student placeholders. The checklist labels these items `STUDENT`, so this is not
an undisclosed code-side omission, but the repository is not submission-ready.

#### A-05 — One Part III manifest ends one second before its final log line

`results/part3/seed1/results.json` records end time `18:42:54`, while its final “saved” log line is at `18:42:55`.
`stage_part3` constructs the manifest before writing the JSON and then logs the save, so second-resolution
timestamps can have this ordering. All substantive feature/classifier log lines and the artifact start are
within the manifest interval; this is a provenance timestamp defect, not evidence of a fabricated run.

#### A-06 — Generic top-k code can retain more than k candidates on an exact cutoff tie

`choose_next` masks values below the kth score, so equal scores at the cutoff all survive. Replaying every step
of the saved n-gram sampled generations showed exactly 40 candidates at all 250 steps, and neural logits are
continuous, so the reported generations were unaffected. The helper is not an exact top-k implementation for
all possible score vectors.

#### A-07 — Dataset/source revisions and embedding-cache inputs are incompletely pinned

Manifests name HuggingFace datasets but do not record dataset revisions or split-stream hashes. Frozen embedding
caches validate only the vocabulary hash, not the Word2Vec/GloVe configuration and source hash before reuse.
For the present artifacts, the saved Word2Vec parameters match the locked config and the Word2Vec, aligned
matrix, vocabulary, and checkpoint hashes all verify. The issue affects future reruns after a source/config
change, not the audited comparison.

#### A-08 — Benchmark checkpoint paths are not accompanied by checkpoint hashes

The benchmark records seed-0 checkpoint paths, the clean code commit, protocol, and hardware, but not checkpoint
SHA-256 values. Current files are the same checkpoints whose validation metrics and Part III hashes verified,
but the benchmark JSON alone cannot prove they were unchanged since timing.

#### A-09 — Saved error examples are the first ten mistakes, not a representative sample

Part III satisfies the required count and stores ten errors per representation plus ten examples in each
pairwise direction. Selection by first test index can reflect dataset ordering; the report should call them
illustrative rather than representative unless the student manually justifies the chosen cases.

## Fixes made

- Added `tests/test_audit_regressions.py`. Before implementation, both tests failed: the AMP escalation helper
  and clean-source guard did not exist. After implementation, both pass.
- Training now raises `FloatingPointError` and logs an `AMP ESCALATION` message immediately when GradScaler
  reduces its scale after a skipped optimizer step; the LR scheduler is not advanced.
- Real-result execution through `run_all.py`, `scripts/lr_sweep.py`, `scripts/benchmark.py`, and
  `scripts/build_embeddings.py` now refuses to start unless `src/`, `scripts/`, `configs/`, `run_all.py`, and
  `requirements.txt` exactly match a commit. Smoke runs remain available for development on dirty code.
- Updated README command examples to invoke
  `D:\anaconda3\envs\dsa4213\python.exe` explicitly, resolving A-03.
- Corrected checklist item I5 to `DONE / STUDENT`, and added both audit escalations under
  `VERIFICATION.md` → `NEEDS HUMAN`.
- No historical result JSON, checkpoint, generated table, report draft, or figure content was changed.

## Runs regenerated or needing regeneration

- Critical-finding reruns performed: **none** (there are no Critical findings).
- Major M-01 runs needing regeneration under corrected escalation behavior:
  `results/part1/rnn/seed{0,1,2}`, `results/part1/transformer/seed{0,1}`,
  `results/part2/glove_frozen/seed2`.
- Major M-02 direct evidence needing regeneration from a clean revision:
  - `results/lr_sweep/rnn/lr0.0001`
  - `results/lr_sweep/rnn/lr0.0003`
  - `results/lr_sweep/rnn/lr0.001`
  - `results/lr_sweep/rnn/lr0.003`
  - `results/lr_sweep/lstm/lr0.0003`
  - `results/lr_sweep/lstm/lr0.001`
  - `results/lr_sweep/lstm/lr0.003`
  - `results/lr_sweep/lstm/lr0.01`
  - `results/lr_sweep/transformer/lr0.0003`
  - `results/lr_sweep/transformer/lr0.001`
  - `results/lr_sweep/transformer/lr0.003`
  - `results/lr_sweep/transformer/lr0.01`
  - `results/part1/ngram`
  - `results/profile/rnn`
  - `results/profile/lstm`
  - `results/profile/transformer`
  - `results/part3/seed1`
  - `results/part3/seed2`
- Because LR selection is upstream of all 15 main neural runs, a clean LR-sweep rerun must be compared with the
  saved selections. If any selected LR changes, all 15 main neural run directories become transitively affected.
- Derived artifacts requiring regeneration: **none**.

## Unresolved risks

- The audit did not retrain the 15 main LMs; authenticity was checked through manifests/logs/configs, exact
  checkpoint reevaluation, checkpoint hashes, and internally consistent histories.
- The seed-0 Part III artifact predates saving raw test predictions, so its confusion-derived metrics were
  recomputed from the saved confusion matrices rather than from per-example predictions. Its representative and
  pairwise errors are present.
- Hardware-idle and thermal conditions recorded for benchmark timing cannot be independently reconstructed after
  the fact.

