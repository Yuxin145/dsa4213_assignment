"""Part II embedding step (separate from LM training): own Word2Vec on the training split + GloVe alignment.
Writes results/part2/embeddings/{own_frozen,glove_frozen}_{matrix.pt,info.json} (cached; skipped if present).
  python scripts/build_embeddings.py [--smoke]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import pipeline as P  # noqa: E402
from src.utils import Logger, require_clean_source, results_root  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if not a.smoke:
        require_clean_source()
    cfg = P.get_cfg(a.smoke)
    log = Logger(results_root(cfg, a.smoke) / "part2" / "embeddings" / "log.txt")
    for v in P.PART2_NEW:
        P.pretrained_for(v, cfg, a.smoke, log)
