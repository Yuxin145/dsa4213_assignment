"""RULE §6.3 learning-rate sweep (writes configs/lr_selected.yaml; smoke -> results_smoke/lr_sweep/selected.json).
  python scripts/lr_sweep.py [--smoke] [--archs rnn lstm transformer]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import pipeline as P  # noqa: E402
from src.models import ARCHS  # noqa: E402
from src.utils import Logger, get_device, require_clean_source, results_root  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--archs", nargs="*", default=list(ARCHS), choices=list(ARCHS))
    a = ap.parse_args()
    if not a.smoke:
        require_clean_source()
    cfg = P.get_cfg(a.smoke)
    log = Logger(results_root(cfg, a.smoke) / "lr_sweep" / "sweep_log.txt")
    P.stage_lr_sweep(cfg, a.smoke, get_device(a.device), log, archs=a.archs)
