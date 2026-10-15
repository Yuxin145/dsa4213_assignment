"""Generate report figures (PNG + PDF, sized for half page width) from results/**/*.json.
  python scripts/make_figures.py [--smoke]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import get_cfg, run_dir  # noqa: E402
from src.utils import load_json, results_root  # noqa: E402

# reference categorical palette (fixed slot order) + recessive ink/grid
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
HALF_W = 3.4  # inches

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "pdf.fonttype": 42,
})


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


def loss_curves(root, seeds, variants, labels, out, name, title):
    runs = {}
    for v in variants:
        for s in seeds:
            d = run_dir(root, v, s)
            if (d / "DONE").exists() or (d / "metrics.json").exists():
                runs[(v, s)] = load_json(d / "metrics.json")
    if not runs:
        print(f"skip {name}: no runs")
        return
    fig, axes = plt.subplots(2, 1, figsize=(HALF_W, 4.4), sharex=True)
    for ax, key, ylabel in ((axes[0], "val_loss", "validation CE (nats/token)"),
                            (axes[1], "train_loss", "training CE (nats/token)")):
        for i, v in enumerate(variants):
            for s in seeds:
                m = runs.get((v, s))
                if not m:
                    continue
                ep = [h["epoch"] for h in m["history"]]
                y = [h[key] for h in m["history"]]
                main = s == seeds[0]
                ax.plot(ep, y, color=SERIES[i], lw=1.8 if main else 0.9, alpha=1.0 if main else 0.45,
                        label=labels[i] if main else None)
                if main and key == "val_loss" and m.get("best_epoch"):
                    be = m["best_epoch"]
                    ax.plot([be], [y[ep.index(be)]], "o", ms=4.5, color=SERIES[i], mec=SURFACE, mew=1.2)
        ax.set_ylabel(ylabel)
    axes[0].set_title(title, loc="left")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_xlabel("epoch")
    note = f"thick: seed {seeds[0]} (dot = best epoch); thin: seeds {', '.join(map(str, seeds[1:]))}" \
        if len(seeds) > 1 else f"seed {seeds[0]} (dot = best epoch)"
    fig.text(0.01, -0.01, note, fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, out, name)


def confusion(root, out):
    p = root / "part3" / "results.json"
    if not p.exists():
        print("skip confusion: no part3 results")
        return
    r = load_json(p)
    labels = r["labels"]
    names = r["main_comparison"]
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    fig, axes = plt.subplots(1, len(names), figsize=(HALF_W * 2 * 0.9, 2.9))
    for ax, n in zip(np.atleast_1d(axes), names):
        cm = np.asarray(r["representations"][n]["confusion_matrix"], dtype=float)
        rate = cm / cm.sum(1, keepdims=True)
        ax.imshow(rate, cmap=cmap, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}\n{rate[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color=SURFACE if rate[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("gold")
        acc = r["representations"][n]["test_acc"]
        ax.set_title(f"{n.replace('_mean', '').upper()} (mean pooling), acc {acc:.3f}", loc="left")
    fig.text(0.01, -0.02, "AG News test (n = %d); cell = count and row-normalised rate" % int(cm.sum()),
             fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, out, "part3_confusion")


def ppl_vs_acc(root, out):
    """LM test perplexity vs AG News accuracy (mean pooling). Uses the mean ± std over LM seeds when Part III was
    run on several seeds (tables.json 'aggregate'), otherwise the seed-0 values."""
    tj = root / "tables" / "tables.json"
    agg = load_json(tj).get("part3", {}).get("aggregate") if tj.exists() else None
    pts = []
    if agg:
        for n, a in agg.items():
            if n.endswith("_mean") and a.get("lm_test_ppl"):
                pts.append((n, a["lm_test_ppl"][0], a["lm_test_ppl"][1], a["test_acc"][0], a["test_acc"][1]))
        sub = f"mean ± std over LM seeds {load_json(tj)['part3']['seeds']}"
    else:
        p = root / "part3" / "results.json"
        if not p.exists():
            return
        r = load_json(p)
        for x in r["ppl_vs_downstream"]:
            if x["lm_test_ppl_seed0"] is not None and x["representation"].endswith("_mean"):
                pts.append((x["representation"], x["lm_test_ppl_seed0"], 0, x["test_acc"], 0))
        sub = "LM seed 0"
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(HALF_W, 2.6))
    for i, (n, px, sx, py, sy) in enumerate(pts):
        ax.errorbar(px, py, xerr=sx, yerr=sy, fmt="o", ms=6.5, color=SERIES[i % len(SERIES)], mec=SURFACE,
                    mew=1.3, elinewidth=1.2, capsize=2.5)
        ax.annotate(n.replace("_mean", ""), (px, py), xytext=(6, 4), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xlabel("LM test perplexity (WikiText-2) — lower is better")
    ax.set_ylabel("AG News test accuracy")
    ax.set_title("LM quality vs downstream accuracy (mean pooling)", loc="left")
    fig.text(0.01, -0.02, sub, fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, out, "part3_ppl_vs_acc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = get_cfg(args.smoke)
    root = results_root(cfg, args.smoke)
    out = (root / "figures") if args.smoke else (ROOT / "figures")
    seeds = cfg["train"]["seeds"]
    loss_curves(root, seeds, ["rnn", "lstm", "transformer"], ["RNN", "LSTM", "Transformer"], out,
                "part1_loss_curves", "Part I: learning curves")
    loss_curves(root, seeds, ["lstm", "own_frozen", "glove_frozen"],
                ["random (trainable)", "own W2V (frozen)", "GloVe (frozen)"], out, "part2_loss_curves",
                "Part II: LSTM embedding variants")
    confusion(root, out)
    ppl_vs_acc(root, out)


if __name__ == "__main__":
    main()
