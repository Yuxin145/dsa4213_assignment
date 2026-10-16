"""All inference benchmarks in one session: GPU + CPU throughput and generation latency.
  python scripts/benchmark.py [--smoke] [--device cuda]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import pipeline as P  # noqa: E402
from src.utils import Logger, get_device, require_clean_source, results_root  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if not a.smoke:
        require_clean_source()
    cfg = P.get_cfg(a.smoke)
    P.stage_benchmark(cfg, a.smoke, get_device(a.device), Logger(results_root(cfg, a.smoke) / "run_all_log.txt"))
