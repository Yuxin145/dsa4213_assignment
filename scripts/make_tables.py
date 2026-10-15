"""Generate all result tables from results/**/*.json (never hand-copied numbers).

Writes <results>/tables/*.md and *.csv plus tables.json (the values the report draft inserts).
  python scripts/make_tables.py [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import PART1_NEURAL, PART2_NEW, VARIANTS, get_cfg, run_dir  # noqa: E402
from src.utils import load_json, results_root, save_json  # noqa: E402

NAMES = {"ngram": "Trigram (interp.)", "rnn": "RNN", "lstm": "LSTM", "transformer": "Transformer",
         "own_frozen": "LSTM + own W2V (frozen)", "glove_frozen": "LSTM + GloVe (frozen)",
         "random": "LSTM + random (trainable)"}


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0), len(xs)


def fmt(m, s=None, n=None, prec=2):
    if m is None:
        return "–"
    if s is None or n is None or n < 2:
        return f"{m:.{prec}f}" + (" (n=1)" if n == 1 else "")
    return f"{m:.{prec}f} ± {s:.{prec}f}"


def load_runs(root, variant, seeds):
    out = {}
    for s in seeds:
        d = run_dir(root, variant, s)
        if (d / "DONE").exists():
            out[s] = load_json(d / "metrics.json")
    return out


def spikes(history, thr=0.10):
    v = [h["val_loss"] for h in history]
    return sum(1 for a, b in zip(v[:-1], v[1:]) if b > a * (1 + thr))


def write_table(tdir: Path, name: str, header: list[str], rows: list[list], note: str = ""):
    tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if note:
        lines += ["", note]
    (tdir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = get_cfg(args.smoke)
    root = results_root(cfg, args.smoke)
    tdir = root / "tables"
    seeds = cfg["train"]["seeds"]
    T = {"seeds": seeds, "smoke": args.smoke, "max_epochs": cfg["train"]["max_epochs"]}

    # ------------------------------------------------------------------ data
    ds = load_json(root / "data_stats.json")
    T["data"] = ds
    write_table(tdir, "data_stats", ["split", "lines", "tokens", "<unk> rate"],
                [[s, v["lines"], v["tokens"], f"{v['unk_rate']:.4f}"] for s, v in ds["splits"].items()],
                f"Vocabulary size V = {ds['vocab_size']} (training split only, incl. <pad>, <unk>, <eos>).")

    # ------------------------------------------------------------------ Part I
    bench = load_json(root / "part1" / "benchmark.json") if (root / "part1" / "benchmark.json").exists() else None
    ng_path = root / "part1" / "ngram" / "metrics.json"
    p1 = {}
    rows = []
    if ng_path.exists():
        ng = load_json(ng_path)
        b = bench["models"]["ngram"] if bench else {}
        p1["ngram"] = {"test_ppl": ng["final"]["test"]["ppl"], "test_ce": ng["final"]["test"]["ce"],
                       "val_ppl": ng["final"]["val"]["ppl"], "lambdas": ng["lambdas"],
                       "train_time_s": ng["fit_time_s"] + ng["lambda_search_time_s"],
                       "cpu_tok_s": b.get("cpu_throughput", {}).get("tokens_per_s_median"),
                       "cpu_latency_ms": (b.get("cpu_latency", {}).get("latency_s_median") or 0) * 1000 or None,
                       "n_seeds": 1, "test_count": ng["final"]["test"]["count"]}
        r = p1["ngram"]
        n_counts = ng.get("n_stored_counts")
        p1["ngram"]["n_stored_counts"] = n_counts
        rows.append([NAMES["ngram"], fmt(r["test_ppl"]), f"{r['test_ce']:.4f}", fmt(r["val_ppl"]), "–", "–", "–",
                     "–" if n_counts is None else f"{n_counts / 1e6:.2f}M counts", "–",
                     f"{r['train_time_s']:.0f} (count+λ search)", "–", "–", "–",
                     fmt(r["cpu_tok_s"], prec=0), "–", fmt(r["cpu_latency_ms"], prec=1)])
    for v in PART1_NEURAL:
        runs = load_runs(root, v, seeds)
        if not runs:
            continue
        te = [m["final"]["test"]["ppl"] for m in runs.values()]
        ce = [m["final"]["test"]["ce"] for m in runs.values()]
        va = [m["final"]["val"]["ppl"] for m in runs.values()]
        ttb = [m["time_to_best_s"] for m in runs.values()]
        tot = [m["total_time_s"] for m in runs.values()]
        spe = [statistics.mean(h["train_time_s"] for h in m["history"]) for m in runs.values()]
        be = [m["best_epoch"] for m in runs.values()]
        par = next(iter(runs.values()))["params"]
        b = bench["models"].get(v, {}) if bench else {}
        g = lambda k, f: b.get(k, {}).get(f)
        p1[v] = {"test_ppl": mean_std(te), "test_ce": mean_std(ce), "val_ppl": mean_std(va),
                 "time_to_best_s": mean_std(ttb), "total_time_s": mean_std(tot), "sec_per_epoch": mean_std(spe),
                 "best_epoch": be, "params": par, "lr": next(iter(runs.values()))["lr"],
                 "gpu_tok_s": g("gpu_throughput", "tokens_per_s_median"),
                 "cpu_tok_s": g("cpu_throughput", "tokens_per_s_median"),
                 "gpu_latency_ms": (g("gpu_latency", "latency_s_median") or 0) * 1000 or None,
                 "cpu_latency_ms": (g("cpu_latency", "latency_s_median") or 0) * 1000 or None,
                 "n_seeds": len(runs), "stopped": [m["stopped_reason"] for m in runs.values()],
                 "tokens_seen": mean_std([m["tokens_seen"] for m in runs.values()])}
        r = p1[v]
        rows.append([NAMES[v], fmt(*r["test_ppl"]), fmt(*r["test_ce"], prec=4), fmt(*r["val_ppl"]),
                     f"{par['input_embedding'] / 1e6:.2f}M", f"{par['core'] / 1e6:.2f}M",
                     f"{par['output_layer'] / 1e6:.2f}M", f"{par['total'] / 1e6:.2f}M",
                     ",".join(str(x) for x in be), fmt(*r["time_to_best_s"], prec=0), fmt(*r["total_time_s"], prec=0),
                     fmt(*r["sec_per_epoch"], prec=1), fmt(r["gpu_tok_s"], prec=0), fmt(r["cpu_tok_s"], prec=0),
                     fmt(r["gpu_latency_ms"], prec=1), fmt(r["cpu_latency_ms"], prec=1)])
    T["part1"] = p1
    write_table(tdir, "part1_main",
                ["model", "test ppl", "test CE (nats)", "val ppl", "emb params", "core params", "output params",
                 "total params", "best epoch(s)", "time to best (s)", "train time (s)", "s/epoch (train)",
                 "GPU tok/s", "CPU tok/s", "GPU latency 50 tok (ms)", "CPU latency 50 tok (ms)"], rows,
                f"Mean ± std over seeds {seeds} for neural models (\"(n=k)\" marks rows with fewer finished seeds); "
                "n-gram is deterministic. "
                "Throughput: scoring every target of the test split, batch 32 chunks, median of 3 runs. "
                "Latency: greedy 50 new tokens, batch 1, median of 5 runs. Timing hardware in benchmark.json.")

    # ------------------------------------------------------------------ Part II
    p2 = {}
    rows = []
    emb_dir = root / "part2" / "embeddings"
    for v in ("lstm",) + PART2_NEW:
        runs = load_runs(root, v, seeds)
        if not runs:
            continue
        kind = VARIANTS[v][2]
        info = load_json(emb_dir / f"{kind}_info.json") if kind != "random" else None
        te = [m["final"]["test"]["ppl"] for m in runs.values()]
        bv = [m["best_val_ce"] for m in runs.values()]
        be = [m["best_epoch"] for m in runs.values()]
        sp = [spikes(m["history"]) for m in runs.values()]
        par = next(iter(runs.values()))["params"]
        p2[kind] = {"test_ppl": mean_std(te), "best_val_ce": mean_std(bv), "best_epoch": be, "spikes": sp,
                    "val_ce_var": statistics.pvariance(bv) if len(bv) > 1 else None,
                    "trainable": par["trainable"], "frozen": par["frozen"], "n_seeds": len(runs),
                    "time_to_best_s": mean_std([m["time_to_best_s"] for m in runs.values()]),
                    "coverage": info["coverage"] if info else None,
                    "emb_train_time_s": info.get("train_time_s") if info else None,
                    "emb_source": info["source"] if info else "random N(0, 0.1²), trained jointly"}
        r = p2[kind]
        cov = r["coverage"]
        rows.append([NAMES[kind], fmt(*r["test_ppl"]), fmt(*r["best_val_ce"], prec=4),
                     "–" if r["val_ce_var"] is None else f"{r['val_ce_var']:.2e}", ",".join(map(str, be)),
                     ",".join(map(str, sp)), fmt(*r["time_to_best_s"], prec=0),
                     f"{r['trainable'] / 1e6:.2f}M / {r['frozen'] / 1e6:.2f}M",
                     "–" if not cov else f"{100 * cov['type_coverage']:.1f}%",
                     "–" if not cov else f"{100 * cov['token_coverage']:.1f}%",
                     "–" if r["emb_train_time_s"] is None else f"{r['emb_train_time_s']:.0f}"])
    T["part2"] = p2
    write_table(tdir, "part2_embeddings",
                ["embedding", "test ppl", "best val CE", "var(best val CE) over seeds", "best epoch(s)",
                 "val-loss spikes (>10%)", "time to best (s)", "trainable / frozen params", "vocab type coverage",
                 "train-token coverage", "embedding training time (s, CPU)"], rows,
                "Random-trainable row reuses the Part I LSTM runs. Coverage is over non-<pad> types and training "
                "tokens. Embedding training time (own Word2Vec) is separate from LM training.")

    # ------------------------------------------------------------------ generation
    rows = []
    gen = {}
    for part in ("part1", "part2"):
        gp = root / part / "generations.json"
        if not gp.exists():
            continue
        g = load_json(gp)
        for name, mdl in g["models"].items():
            for s in g["settings"]:
                mt = mdl[s]["metrics"]
                gen[(part, name, s)] = mt
                rows.append([part, NAMES.get(name, name), s, f"{mt['distinct_1']:.3f}", f"{mt['distinct_2']:.3f}",
                             f"{mt['repeated_trigram_rate']:.3f}", f"{mt['unk_rate']:.3f}"])
    T["generation"] = {f"{a}|{b}|{c}": v for (a, b, c), v in gen.items()}
    write_table(tdir, "generation_metrics", ["part", "model", "decoding", "distinct-1", "distinct-2",
                                             "repeated-trigram rate", "<unk> rate"], rows,
                "50 new tokens for each of the 5 prompts, pooled; seed-0 checkpoints.")

    # ------------------------------------------------------------------ Part III
    p3p = root / "part3" / "results.json"
    if p3p.exists():
        r3 = load_json(p3p)
        reps = r3["representations"]
        rows = []
        pairs = {p["representation"]: p for p in r3["ppl_vs_downstream"]}
        for name, r in reps.items():
            best_val = max(g["val_acc"] for g in r["selection"]["grid"])
            ppl = pairs[name]["lm_test_ppl_seed0"]
            rows.append([name + (" (main)" if name in r3["main_comparison"] else ""), r["selection"]["best_C"],
                         f"{best_val:.4f}", f"{r['test_acc']:.4f}", f"{r['test_macro_f1']:.4f}",
                         "–" if ppl is None else f"{ppl:.2f}"])
        write_table(tdir, "part3_downstream", ["representation", "C (val)", "val acc", "test acc", "test macro-F1",
                                               "LM test ppl (seed 0)"], rows,
                    "AG News: 8,000 train / 2,000 val (stratified from official train, seed 0), 7,600 official test. "
                    "StandardScaler + LogisticRegression; C chosen on validation accuracy.")
        lm_reps = [p for p in r3["ppl_vs_downstream"] if p["lm_test_ppl_seed0"] is not None and p["representation"].endswith("_mean")]
        by_ppl = sorted(lm_reps, key=lambda p: p["lm_test_ppl_seed0"])
        by_acc = sorted(lm_reps, key=lambda p: -p["test_acc"])
        rows = [[p["representation"], f"{p['lm_test_ppl_seed0']:.2f}", by_ppl.index(p) + 1, f"{p['test_acc']:.4f}",
                 by_acc.index(p) + 1] for p in lm_reps]
        write_table(tdir, "part3_rank", ["representation (mean pooling)", "LM test ppl", "ppl rank", "test acc",
                                         "downstream rank"], rows)
        oov = r3["oov"]
        write_table(tdir, "part3_oov", ["split", "tokens (converted)", "OOV before", "OOV after"],
                    [[s, v["after"]["tokens"], f"{v['before']['oov_rate']:.4f}", f"{v['after']['oov_rate']:.4f}"]
                     for s, v in oov.items()],
                    "Before = lowercase + whitespace split; after = conversion to WikiText-2 conventions.")
        err = load_json(root / "part3" / "errors.json")
        T["part3"] = {"representations": {k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"}
                                          for k, v in reps.items()},
                      "rank": rows, "oov": oov, "pairwise_counts": {k: (v["count"] if isinstance(v, dict) else v)
                                                                    for k, v in err["pairwise"].items()},
                      "truncated_fraction": r3["truncated_fraction"]}

        # ---- Part III over LM seeds (same protocol on the seed-1/2 checkpoints; seed 0 stays the main result)
        from scipy.stats import binomtest

        from src.pipeline import part3_dir

        per_seed = {}
        for s in seeds:
            fp = part3_dir(root, s) / "results.json"
            if fp.exists():
                per_seed[s] = (load_json(fp), load_json(part3_dir(root, s) / "errors.json"))
        if len(per_seed) > 1:
            ms = sorted(per_seed)
            ppl_key = lambda p: p.get("lm_test_ppl", p.get("lm_test_ppl_seed0"))
            agg = {}
            rows = []
            for name in reps:
                accs = [per_seed[s][0]["representations"][name]["test_acc"] for s in ms]
                f1s = [per_seed[s][0]["representations"][name]["test_macro_f1"] for s in ms]
                ppls = [ppl_key({p["representation"]: p for p in per_seed[s][0]["ppl_vs_downstream"]}[name]) for s in ms]
                ppls = [p for p in ppls if p is not None]
                agg[name] = {"test_acc": mean_std(accs), "test_macro_f1": mean_std(f1s),
                             "lm_test_ppl": mean_std(ppls) if ppls else None, "per_seed_acc": dict(zip(ms, accs)),
                             "per_seed_macro_f1": dict(zip(ms, f1s)), "per_seed_C":
                                 {s: per_seed[s][0]["representations"][name]["selection"]["best_C"] for s in ms}}
                a = agg[name]
                rows.append([name + (" (main)" if name in r3["main_comparison"] else ""),
                             f"{reps[name]['test_acc']:.4f}", fmt(*a["test_acc"], prec=4),
                             " / ".join(f"{x:.4f}" for x in accs),
                             f"{reps[name]['test_macro_f1']:.4f}", fmt(*a["test_macro_f1"], prec=4),
                             "–" if not ppls else fmt(*a["lm_test_ppl"]),
                             " / ".join(str(a["per_seed_C"][s]) for s in ms)])
            write_table(tdir, "part3_seeds",
                        ["representation", "test acc (seed 0)", f"test acc mean ± std (seeds {ms})", "per-seed acc",
                         "macro-F1 (seed 0)", "macro-F1 mean ± std", "LM test ppl mean ± std", "C per seed"], rows,
                        "Identical protocol for every LM seed (same AG News split, conversion, pooling, classifier, "
                        "C grid on validation, test once). glove_avg_baseline does not depend on the LM seed. "
                        "Std = sample std over LM seeds.")
            # paired LSTM vs Transformer comparison on the same 7,600 test articles, per LM seed
            rows, pair_rec = [], []
            for s in ms:
                r, e = per_seed[s]
                la, ta = r["representations"]["lstm_mean"]["test_acc"], r["representations"]["transformer_mean"]["test_acc"]
                b = e["pairwise"]["lstm_right_transformer_wrong"]["count"]
                c = e["pairwise"]["transformer_right_lstm_wrong"]["count"]
                pval = binomtest(min(b, c), b + c, 0.5).pvalue
                pair_rec.append({"seed": s, "lstm_acc": la, "transformer_acc": ta, "diff": ta - la, "b": b, "c": c,
                                 "mcnemar_exact_p": pval})
                rows.append([s, f"{la:.4f}", f"{ta:.4f}", f"{ta - la:+.4f}", b, c, f"{pval:.1e}"])
            d = mean_std([x["diff"] for x in pair_rec])
            rows.append(["mean ± std", fmt(*agg["lstm_mean"]["test_acc"], prec=4),
                         fmt(*agg["transformer_mean"]["test_acc"], prec=4), fmt(*d, prec=4), "", "", ""])
            write_table(tdir, "part3_pairwise",
                        ["LM seed", "LSTM acc", "Transformer acc", "Transformer − LSTM", "only LSTM right (b)",
                         "only Transformer right (c)", "exact McNemar p"], rows,
                        "Mean pooling, main comparison. b/c count test articles that exactly one model classifies "
                        "correctly; the exact McNemar test is a two-sided binomial test of b vs c (paired, same "
                        "test set, per LM seed).")
            # rank agreement using seed-mean LM ppl and seed-mean accuracy
            lm_names = [n for n in agg if n.endswith("_mean") and agg[n]["lm_test_ppl"]]
            by_ppl = sorted(lm_names, key=lambda n: agg[n]["lm_test_ppl"][0])
            by_acc = sorted(lm_names, key=lambda n: -agg[n]["test_acc"][0])
            rows = [[n, fmt(*agg[n]["lm_test_ppl"]), by_ppl.index(n) + 1, fmt(*agg[n]["test_acc"], prec=4),
                     by_acc.index(n) + 1] for n in lm_names]
            write_table(tdir, "part3_rank_seeds", ["representation (mean pooling)", "LM test ppl (mean ± std)",
                                                   "ppl rank", "test acc (mean ± std)", "downstream rank"], rows)
            T["part3"]["seeds"] = ms
            T["part3"]["aggregate"] = agg
            T["part3"]["pairwise_lstm_vs_transformer"] = pair_rec
            T["part3"]["rank_seeds"] = rows
    save_json(T, tdir / "tables.json")
    print(f"tables written to {tdir}")


if __name__ == "__main__":
    main()
