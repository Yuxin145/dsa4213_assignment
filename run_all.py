"""Single entry point for the whole pipeline.

  python run_all.py --smoke                 # end-to-end smoke run (~1% data, 1 epoch, seed 0) -> results_smoke/
  python run_all.py --stage lr_sweep        # RULE §6.3 LR sweep (writes configs/lr_selected.yaml)
  python run_all.py --stage profile         # §9 one epoch per neural model
  python run_all.py --stage speedcheck      # ~2 min: current s/epoch per model + projected full-run hours
  python run_all.py --stage full            # all full runs in the required order (resumable)
  python run_all.py --stage report          # tables, figures, report draft from results/**/*.json

Every stage skips finished work (DONE markers) and resumes interrupted runs from last.pt.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from src import pipeline as P
from src.utils import ROOT, Logger, get_device, require_clean_source, results_root

STAGES = ("data", "ngram", "lr_sweep", "profile", "speedcheck", "train", "generation", "part3", "benchmark", "report", "full")


def run_report(smoke: bool, log: Logger) -> None:
    for script in ("make_tables.py", "make_figures.py", "make_report.py"):
        cmd = [sys.executable, str(ROOT / "scripts" / script)] + (["--smoke"] if smoke else [])
        log(f"running {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=None, choices=list(P.VARIANTS))
    ap.add_argument("--ignore-budget", action="store_true", help="run seeds 1-2 even if over the §9 budget")
    args = ap.parse_args()

    smoke = args.smoke
    cfg = P.get_cfg(smoke)
    device = get_device(args.device)
    variants = args.variants or list(P.VARIANTS)
    stage = args.stage or ("smoke" if smoke else None)
    if stage is None:
        ap.error("choose --stage (or --smoke for the end-to-end smoke run)")
    if not smoke:
        require_clean_source()
    root = results_root(cfg, smoke)
    root.mkdir(parents=True, exist_ok=True)
    log = Logger(root / "run_all_log.txt")
    log(f"run_all stage={args.stage} smoke={smoke} device={device} python={sys.executable}")

    if stage in ("data", "smoke", "full"):
        P.stage_data(cfg, smoke, log)
    if stage in ("ngram", "smoke", "full"):
        P.stage_ngram(cfg, smoke, log)
    if stage == "lr_sweep" or stage == "smoke":
        P.stage_lr_sweep(cfg, smoke, device, log)
    if stage == "profile":
        P.stage_profile(cfg, device, log)
    if stage == "speedcheck":
        P.stage_speedcheck(cfg, smoke, device, log)
    if stage == "train":
        P.stage_train(cfg, smoke, device, log, variants, args.seeds or cfg["train"]["seeds"])
    if stage == "smoke":
        P.stage_train(cfg, smoke, device, log, variants, [0])
        P.stage_generation(cfg, smoke, device, log)
        P.stage_part3(cfg, smoke, device, log)
        P.stage_benchmark(cfg, smoke, device, log)
        run_report(smoke, log)
    if stage == "full":
        if device.type != "cuda":
            raise RuntimeError("full runs must use the GPU (CLAUDE.md §2); CUDA is not available")
        seeds = cfg["train"]["seeds"]
        # seed 0 of everything first, then Part III on seed-0 checkpoints, then the remaining seeds
        P.stage_train(cfg, smoke, device, log, list(P.VARIANTS), [seeds[0]])
        P.stage_generation(cfg, smoke, device, log, seed=seeds[0])
        P.stage_part3(cfg, smoke, device, log, seed=seeds[0])
        ok, msg = P.budget_check_after_seed0(cfg, smoke, seeds)
        log(msg)
        if ok or args.ignore_budget:
            P.stage_train(cfg, smoke, device, log, list(P.VARIANTS), seeds[1:])
        else:
            log("BUDGET STOP: seeds " + str(seeds[1:]) + " not started (CLAUDE.md §9 step 2). To run them anyway: "
                "python run_all.py --stage full --ignore-budget")
        P.stage_benchmark(cfg, smoke, device, log, seed=seeds[0])
        run_report(smoke, log)
    if stage == "generation":
        P.stage_generation(cfg, smoke, device, log)
    if stage == "part3":
        for sd in (args.seeds or [0]):
            P.stage_part3(cfg, smoke, device, log, seed=sd)
    if stage == "benchmark":
        P.stage_benchmark(cfg, smoke, device, log)
    if stage == "report":
        run_report(smoke, log)
    log("run_all finished")


if __name__ == "__main__":
    main()
