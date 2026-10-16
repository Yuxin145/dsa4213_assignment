"""Experiment stages. Every stage is idempotent: finished runs (DONE markers / existing outputs) are skipped."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import yaml

from . import benchmark as bm
from .data import SPLITS, load_lm_data
from .embeddings import get_pretrained
from .generate import NeuralScorer, NgramScorer, generate, text_metrics
from .models import ARCHS, build_model, param_split
from .ngram import InterpolatedTrigram, grid_search_lambdas
from .train import train_run
from .utils import (ROOT, Logger, deep_update, get_device, load_config, load_json, make_manifest,
                    now_str, rel_path, results_root, save_json, sha256_file)

LR_FILE = ROOT / "configs" / "lr_selected.yaml"
# variant -> (part, arch, embedding kind)
VARIANTS = {
    "rnn": ("part1", "rnn", "random"),
    "lstm": ("part1", "lstm", "random"),
    "transformer": ("part1", "transformer", "random"),
    "own_frozen": ("part2", "lstm", "own_frozen"),
    "glove_frozen": ("part2", "lstm", "glove_frozen"),
}
PART1_NEURAL = ("rnn", "lstm", "transformer")
PART2_NEW = ("own_frozen", "glove_frozen")

_DATA_CACHE: dict = {}


# ----------------------------------------------------------------------------- config / data
def get_cfg(smoke: bool) -> dict:
    cfg = load_config()
    if smoke:
        sel = results_root(cfg, True) / "lr_sweep" / "selected.json"
        if sel.exists():
            cfg["train"]["lr"].update(load_json(sel)["selected_lr"])
        cfg["train"]["max_epochs"] = 1
        cfg["train"]["seeds"] = [0]
    elif LR_FILE.exists():
        with open(LR_FILE, "r", encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f))
    return cfg


def get_data(cfg: dict, smoke: bool) -> dict:
    if smoke not in _DATA_CACHE:
        root = results_root(cfg, smoke)
        root.mkdir(parents=True, exist_ok=True)
        _DATA_CACHE[smoke] = load_lm_data(cfg, root, smoke)
    return _DATA_CACHE[smoke]


def variant_cfg(cfg: dict, variant: str) -> dict:
    part, arch, kind = VARIANTS[variant]
    if kind == "random":
        return cfg
    # Part II: identical to the Part I LSTM config except the embedding keys
    return deep_update(cfg, {"embedding": {"type": kind, "freeze": True}})


def run_dir(root: Path, variant: str, seed: int) -> Path:
    part, arch, kind = VARIANTS[variant]
    name = arch if part == "part1" else kind
    return root / part / name / f"seed{seed}"


def variant_lr(cfg: dict, variant: str) -> float:
    arch = VARIANTS[variant][1]
    lr = cfg["train"]["lr"][arch]
    if lr is None:
        raise RuntimeError(f"no LR selected for {arch}: run the LR sweep first (configs/lr_selected.yaml)")
    return float(lr)


def pretrained_for(variant: str, cfg: dict, smoke: bool, log: Logger):
    kind = VARIANTS[variant][2]
    if kind == "random":
        return None, None
    data = get_data(cfg, smoke)
    root = results_root(cfg, smoke)
    lines = data["lines"]["train"]
    if smoke:  # smoke: Word2Vec on the first 1% of training lines only
        lines = lines[: max(1, int(len(lines) * cfg["data"]["smoke_fraction"]))]
    return get_pretrained(kind, cfg, data["vocab"], lines, data["streams"]["train"],
                          root / "part2" / "embeddings", log)


# ----------------------------------------------------------------------------- stages
def stage_data(cfg, smoke, log):
    data = get_data(cfg, smoke)
    root = results_root(cfg, smoke)
    stats = load_json(root / "data_stats.json")
    log(f"data: V={stats['vocab_size']} vocab_sha256={stats['vocab_sha256'][:12]} "
        + ", ".join(f"{s}={stats['splits'][s]['tokens']} tok (unk {stats['splits'][s]['unk_rate']:.4f})"
                    for s in SPLITS)
        + f"; chunks train/val/test = {[data['chunks'][s][0].shape[0] for s in SPLITS]}")
    vocab = data["vocab"]
    for p in cfg["generation"]["prompts"]:
        oov = [t for t in p.split() if t not in vocab.stoi]
        if oov:
            raise ValueError(f"generation prompt has out-of-vocab tokens {oov}: {p!r}")
    return stats


def fit_ngram(cfg, smoke) -> InterpolatedTrigram:
    data = get_data(cfg, smoke)
    ng = InterpolatedTrigram(len(data["vocab"]), data["vocab"].pad_id)
    return ng.fit(data["streams"]["train"])


def load_ngram(cfg, smoke) -> InterpolatedTrigram:
    m = load_json(results_root(cfg, smoke) / "part1" / "ngram" / "metrics.json")
    ng = fit_ngram(cfg, smoke)
    ng.set_lambdas(m["lambdas"])
    return ng


def stage_ngram(cfg, smoke, log):
    out = results_root(cfg, smoke) / "part1" / "ngram"
    if (out / "DONE").exists():
        log("ngram: DONE, skipping")
        return load_json(out / "metrics.json")
    out.mkdir(parents=True, exist_ok=True)
    rlog = Logger(out / "log.txt")
    start = now_str()
    data = get_data(cfg, smoke)
    t0 = time.perf_counter()
    ng = fit_ngram(cfg, smoke)
    fit_s = time.perf_counter() - t0
    rlog(f"fit trigram counts on {data['streams']['train'].numel()} train tokens in {fit_s:.1f}s; "
         f"{len(ng.ctx2)} bigram contexts, {len(ng.ctx3)} trigram contexts")
    t0 = time.perf_counter()
    P, A = ng.chunk_components(*data["chunks"]["validation"])
    gs = grid_search_lambdas(P, A, cfg["ngram"]["lambda_grid_step"])
    sel_s = time.perf_counter() - t0
    ng.set_lambdas(gs["best_lambdas"])
    rlog(f"lambda grid search on VALIDATION ({gs['n_grid']} points, {sel_s:.1f}s): best (l3,l2,l1,l0)="
         f"{gs['best_lambdas']} val CE {gs['best_val_ce']:.4f}")
    val = ng.score_chunks(*data["chunks"]["validation"])
    test = ng.score_chunks(*data["chunks"]["test"])
    rlog(f"val ppl {val['ppl']:.2f} | test ppl {test['ppl']:.2f} | test targets {test['count']}")
    m = {"model": "interpolated_trigram", "lambdas": list(gs["best_lambdas"]), "grid_search": gs,
         "fit_time_s": fit_s, "lambda_search_time_s": sel_s, "final": {"val": val, "test": test},
         "n_bigram_contexts": len(ng.ctx2), "n_trigram_contexts": len(ng.ctx3),
         "n_stored_counts": len(ng.uni) + sum(len(c) for c in ng.next2.values()) + sum(len(c) for c in ng.next3.values()),
         "manifest": make_manifest(cfg, None, out / "log.txt", start, None,
                                   {"vocab_sha256": data["vocab"].sha256})}
    save_json(m, out / "metrics.json")
    (out / "DONE").write_text(now_str(), encoding="utf-8")
    return m


def stage_lr_sweep(cfg, smoke, device, log, archs=ARCHS):
    """RULE: per architecture sweep {3e-4,1e-3,3e-3}, 2 epochs, seed 0, lowest val loss; if the best is at
    an edge, also try the next value once. Writes configs/lr_selected.yaml (real) or selected.json (smoke)."""
    root = results_root(cfg, smoke) / "lr_sweep"
    data = get_data(cfg, smoke)
    sc = cfg["lr_sweep"]
    epochs = 1 if smoke else sc["epochs"]
    summary = {}
    for arch in archs:
        tried = {}

        def run(lr):
            d = root / arch / f"lr{lr:g}"
            log(f"lr_sweep {arch} lr={lr:g} -> {d}")
            m = train_run(cfg, arch, sc["seed"], lr, d, data, device, max_epochs=epochs, eval_test=False,
                          extra_manifest={"purpose": "lr_sweep"})
            tried[lr] = {"best_val_ce": m["best_val_ce"], "epoch_time_s": [h["epoch_time_s"] for h in m["history"]],
                         "val_curve": [h["val_loss"] for h in m["history"]],
                         "train_curve": [h["train_loss"] for h in m["history"]],
                         "peak_mem_mb": max((h["peak_mem_mb"] or 0) for h in m["history"]),
                         "skipped_steps": sum(h["skipped_steps"] for h in m["history"])}

        grid = list(sc["grid"])
        for lr in grid:
            run(lr)
        best = min(tried, key=lambda k: tried[k]["best_val_ce"])
        edge = None
        if best == min(grid):
            edge = sc["edge_low"]
        elif best == max(grid):
            edge = sc["edge_high"]
        if edge is not None:
            log(f"lr_sweep {arch}: best {best:g} is at the grid edge -> also trying {edge:g}")
            run(edge)
            best = min(tried, key=lambda k: tried[k]["best_val_ce"])
        summary[arch] = {"selected": best, "tried": {f"{k:g}": v for k, v in sorted(tried.items())},
                         "edge_extension": edge}
        log(f"lr_sweep {arch}: selected lr={best:g} "
            + ", ".join(f"{k:g}:{v['best_val_ce']:.4f}" for k, v in sorted(tried.items())))
    sel = {a: summary[a]["selected"] for a in summary}
    out = {"selected_lr": sel, "summary": summary, "epochs": epochs, "seed": sc["seed"], "time": now_str()}
    prev = root / "selected.json"
    if prev.exists():
        old = load_json(prev)
        old["selected_lr"].update(sel)
        old["summary"].update(summary)
        out["selected_lr"], out["summary"] = old["selected_lr"], old["summary"]
    save_json(out, prev)
    if not smoke:
        with open(LR_FILE, "w", encoding="utf-8") as f:
            f.write("# GENERATED by the LR sweep (scripts/lr_sweep.py, RULE). Evidence: results/lr_sweep/"
                    "selected.json\n# Part II reuses the LSTM value.\n")
            yaml.safe_dump({"train": {"lr": out["selected_lr"]}}, f, sort_keys=False)
    cfg["train"]["lr"].update(out["selected_lr"])
    return out


def stage_train(cfg, smoke, device, log, variants, seeds):
    data = get_data(cfg, smoke)
    root = results_root(cfg, smoke)
    results = {}
    for seed in seeds:
        for v in variants:
            vcfg = variant_cfg(cfg, v)
            M, info = pretrained_for(v, cfg, smoke, log)
            d = run_dir(root, v, seed)
            extra = {"variant": v}
            if info is not None:
                extra.update({"embedding_matrix_sha256": info["matrix_sha256"], "embedding_source": info["source"]})
            log(f"train {v} seed {seed} -> {d}")
            m = train_run(vcfg, VARIANTS[v][1], seed, variant_lr(cfg, v), d, data, device, pretrained=M,
                          extra_manifest=extra)
            results[(v, seed)] = m
            if m.get("final") and "test" in m["final"]:
                log(f"  {v} seed {seed}: best epoch {m['best_epoch']} ({m['stopped_reason']}), "
                    f"val ppl {m['final']['val']['ppl']:.2f}, test ppl {m['final']['test']['ppl']:.2f}, "
                    f"time {m['total_time_s'] / 60:.1f} min")
    return results


def load_lm(cfg, smoke, variant, seed, device):
    """Model with the best checkpoint of a finished run; returns (model, info)."""
    root = results_root(cfg, smoke)
    d = run_dir(root, variant, seed)
    if not (d / "DONE").exists():
        raise FileNotFoundError(f"run {d} is not finished (no DONE marker)")
    data = get_data(cfg, smoke)
    vcfg = variant_cfg(cfg, variant)
    M, _ = pretrained_for(variant, cfg, smoke, Logger(None, echo=False))
    model = build_model(VARIANTS[variant][1], vcfg, len(data["vocab"]), data["vocab"].pad_id, seed, M)
    ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, {"checkpoint": rel_path(d / "best.pt"), "checkpoint_sha256": sha256_file(d / "best.pt"),
                   "best_epoch": ck["epoch"]}


def stage_generation(cfg, smoke, device, log, seed=0):
    root = results_root(cfg, smoke)
    data = get_data(cfg, smoke)
    vocab = data["vocab"]
    g = cfg["generation"]
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    L = cfg["data"]["context_length"]
    start = now_str()
    prompts = [vocab.encode(p.split()) for p in g["prompts"]]
    settings = {"greedy": {"mode": "greedy"},
                "sample_t0.8_k40": {"mode": "sample", "temperature": g["temperature"], "top_k": g["top_k"]}}

    def run_model(name, scorer):
        res = {}
        for sname, s in settings.items():
            outs = []
            for pi, p in enumerate(prompts):
                ids = generate(scorer, p, g["max_new_tokens"], vocab.pad_id, s["mode"], s.get("temperature", 1.0),
                               s.get("top_k", 0), seed=g["sample_seed"] + pi if s["mode"] == "sample" else None)
                outs.append(vocab.decode(ids))
            res[sname] = {"continuations": [{"prompt": g["prompts"][i], "text": " ".join(o)} for i, o in enumerate(outs)],
                          "metrics": text_metrics(outs)}
        log(f"generation {name}: " + "; ".join(f"{k} d1={v['metrics']['distinct_1']:.3f} "
                                                f"rep3={v['metrics']['repeated_trigram_rate']:.3f}" for k, v in res.items()))
        return res

    groups = {"part1": ["ngram"] + list(PART1_NEURAL), "part2": ["lstm"] + list(PART2_NEW)}
    cache = {}
    for part, names in groups.items():
        out_path = root / part / "generations.json"
        models = {}
        for name in names:
            if name not in cache:
                if name == "ngram":
                    ng = load_ngram(cfg, smoke)
                    cache[name] = (run_model(name, NgramScorer(ng)), {"lambdas": ng.lambdas})
                else:
                    model, info = load_lm(cfg, smoke, name, seed, device)
                    cache[name] = (run_model(name, NeuralScorer(model, device, amp, L)), info)
                    del model
            models[name] = {"source": cache[name][1], **cache[name][0]}
        save_json({"settings": settings, "max_new_tokens": g["max_new_tokens"], "sample_seed_base": g["sample_seed"],
                   "lm_seed": seed, "models": models,
                   "manifest": make_manifest(cfg, seed, None, start, device)}, out_path)
        log(f"saved {out_path}")


def part3_dir(root: Path, seed: int) -> Path:
    """Seed-0 Part III results live in results/part3/ (the LOCKED main comparison); additional LM seeds in
    results/part3/seed<k>/ (same protocol)."""
    return root / "part3" if seed == 0 else root / "part3" / f"seed{seed}"


def stage_part3(cfg, smoke, device, log, seed=0, optional=True):
    from . import part3 as p3

    root = results_root(cfg, smoke)
    out = part3_dir(root, seed)
    out.mkdir(parents=True, exist_ok=True)
    rlog = Logger(out / "log.txt")
    start = now_str()
    data = get_data(cfg, smoke)
    vocab = data["vocab"]
    vs = set(vocab.itos)
    pc = cfg["part3"]
    ag = p3.load_ag_news(cfg)
    if smoke:
        for s, n in (("train", 400), ("val", 200), ("test", 400)):
            ag[s] = {k: v[:n] for k, v in ag[s].items()}
    tok_before = {s: [p3.baseline_tokens(t) for t in ag[s]["text"]] for s in ag}
    tok_after = {s: [p3.convert_tokens(t, vs) for t in ag[s]["text"]] for s in ag}
    oov = {s: {"before": p3.oov_rate(tok_before[s], vs), "after": p3.oov_rate(tok_after[s], vs)} for s in ag}
    trunc = {s: float(np.mean([len(t) > pc["max_len"] for t in tok_after[s]])) for s in ag}
    rlog(f"Part III with LM seed {seed}. OOV rate before/after conversion: " + ", ".join(
        f"{s} {oov[s]['before']['oov_rate']:.4f}->{oov[s]['after']['oov_rate']:.4f}" for s in ag)
         + f"; truncated fraction {trunc}")
    seqs = {s: p3.encode_texts(tok_after[s], vocab, pc["max_len"]) for s in ag}
    y = {s: np.asarray(ag[s]["y"]) for s in ag}

    reps, preds, lm_info = {}, {}, {}

    def run_rep(name, feats):
        r = p3.fit_and_test(feats["train"], y["train"], feats["val"], y["val"], feats["test"], y["test"],
                            pc["C_grid"], pc["max_iter"])
        preds[name] = r.pop("pred")
        reps[name] = r
        rlog(f"{name}: C={r['selection']['best_C']} val_acc={max(g['val_acc'] for g in r['selection']['grid']):.4f} "
             f"test_acc={r['test_acc']:.4f} test_macroF1={r['test_macro_f1']:.4f}")

    def lm_feats(variant, poolings):
        model, info = load_lm(cfg, smoke, variant, seed, device)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        t0 = time.perf_counter()
        f = {s: p3.extract_features(model, seqs[s], vocab.pad_id, device, poolings) for s in seqs}
        info["feature_time_s"] = time.perf_counter() - t0
        info["params_unchanged"] = all(torch.equal(before[k], v) for k, v in model.state_dict().items())
        if not info["params_unchanged"]:
            raise RuntimeError(f"LM parameters of {variant} changed during feature extraction")
        lm_info[variant] = info
        return {p: {s: f[s][p] for s in f} for p in poolings}

    # ---- main comparison (LOCKED): Part I LSTM vs Transformer, seed 0, mean pooling
    feats = {v: lm_feats(v, ("mean", "last")) for v in ("lstm", "transformer")}
    for v in ("lstm", "transformer"):
        run_rep(f"{v}_mean", feats[v]["mean"])
    main = ["lstm_mean", "transformer_mean"]
    # ---- optional, cheap extras (after the main comparison)
    if optional:
        for v in ("lstm", "transformer"):
            run_rep(f"{v}_last", feats[v]["last"])
        for v in PART2_NEW:
            run_rep(f"{v}_mean", lm_feats(v, ("mean",))["mean"])
        G, ginfo = pretrained_for("glove_frozen", cfg, smoke, rlog)
        Gn = G.numpy()
        run_rep("glove_avg_baseline", {s: p3.glove_mean_features(seqs[s], Gn, vocab.pad_id) for s in seqs})

    # ---- errors
    texts = ag["test"]["text"]
    errors = {"misclassified": {}, "pairwise": {}}
    for name, pr in preds.items():
        wrong = np.where(pr != y["test"])[0][:pc["n_error_examples"]]
        errors["misclassified"][name] = [{"test_idx": int(i), "text": texts[i], "gold": p3.LABELS[y["test"][i]],
                                          "pred": p3.LABELS[pr[i]]} for i in wrong]
    a, b = preds["lstm_mean"], preds["transformer_mean"]
    for tag, mask in (("lstm_right_transformer_wrong", (a == y["test"]) & (b != y["test"])),
                      ("transformer_right_lstm_wrong", (b == y["test"]) & (a != y["test"]))):
        idx = np.where(mask)[0]
        errors["pairwise"][tag] = {"count": int(idx.size), "examples": [
            {"test_idx": int(i), "text": texts[i], "gold": p3.LABELS[y["test"][i]],
             "lstm_pred": p3.LABELS[a[i]], "transformer_pred": p3.LABELS[b[i]]} for i in idx[:pc["n_error_examples"]]]}
    errors["pairwise"]["both_wrong"] = int(((a != y["test"]) & (b != y["test"])).sum())
    save_json(errors, out / "errors.json")

    # ---- LM ppl vs downstream
    pair = []
    for name in reps:
        variant = name.rsplit("_", 1)[0] if name != "glove_avg_baseline" else None
        ppl = None
        if variant in VARIANTS:
            m = load_json(run_dir(root, variant, seed) / "metrics.json")
            ppl = m["final"]["test"]["ppl"]
        rec = {"representation": name, "lm_seed": seed, "lm_test_ppl": ppl, "test_acc": reps[name]["test_acc"],
               "test_macro_f1": reps[name]["test_macro_f1"]}
        if seed == 0:
            rec["lm_test_ppl_seed0"] = ppl  # key used by the original seed-0 results.json
        pair.append(rec)
    res = {"labels": p3.LABELS, "split_sizes": {s: len(ag[s]["text"]) for s in ag},
           "class_counts": {s: np.bincount(y[s], minlength=4).tolist() for s in ag},
           "oov": oov, "truncated_fraction": trunc, "main_comparison": main, "representations": reps,
           "ppl_vs_downstream": pair, "lm_checkpoints": lm_info, "lm_seed": seed,
           "test_predictions": {k: v.tolist() for k, v in preds.items()},
           "split_indices": {s: ag[s]["orig_idx"] for s in ("train", "val")},
           "conversion_examples": [{"raw": ag["test"]["text"][i], "converted": " ".join(tok_after["test"][i])}
                                   for i in range(5)],
           "manifest": make_manifest(cfg, seed, out / "log.txt", start, device)}
    save_json(res, out / "results.json")
    rlog(f"saved {out / 'results.json'}")
    return res


def stage_benchmark(cfg, smoke, device, log, seed=0):
    root = results_root(cfg, smoke)
    out = root / "part1" / "benchmark.json"
    blog = Logger(root / "part1" / "benchmark_log.txt")
    start = now_str()
    data = get_data(cfg, smoke)
    vocab = data["vocab"]
    te_x, te_y = data["chunks"]["test"]
    prompt = vocab.encode(cfg["generation"]["prompts"][0].split())
    L = cfg["data"]["context_length"]
    cpu = torch.device("cpu")
    res = {"protocol": bm.__doc__.strip(), "cpu_threads": torch.get_num_threads(), "models": {}}
    ng = load_ngram(cfg, smoke)
    res["models"]["ngram"] = {"cpu_throughput": bm.ngram_throughput(ng, te_x, te_y, cfg, vocab.pad_id),
                              "cpu_latency": bm.latency(NgramScorer(ng), prompt, cfg, vocab.pad_id)}
    blog(f"ngram: {res['models']['ngram']}")
    for v in PART1_NEURAL:
        model, info = load_lm(cfg, smoke, v, seed, cpu)
        r = {"checkpoint": info["checkpoint"], "params": param_split(model)}
        r["cpu_throughput"] = bm.neural_throughput(model, te_x, te_y, cpu, False, cfg, vocab.pad_id)
        r["cpu_latency"] = bm.latency(NeuralScorer(model, cpu, False, L), prompt, cfg, vocab.pad_id, cpu)
        if device.type == "cuda":
            model.to(device)
            amp = bool(cfg["train"]["amp"])
            r["gpu_throughput"] = bm.neural_throughput(model, te_x, te_y, device, amp, cfg, vocab.pad_id)
            r["gpu_latency"] = bm.latency(NeuralScorer(model, device, amp, L), prompt, cfg, vocab.pad_id, device)
            r["gpu_amp"] = amp
        res["models"][v] = r
        blog(f"{v}: " + ", ".join(f"{k}={r[k]['tokens_per_s_median']:.0f} tok/s" for k in r if k.endswith("throughput"))
             + ", " + ", ".join(f"{k}={r[k]['latency_s_median'] * 1000:.1f} ms" for k in r if k.endswith("latency")))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    res["manifest"] = make_manifest(cfg, seed, root / "part1" / "benchmark_log.txt", start, device)
    save_json(res, out)
    log(f"saved {out}")
    return res


def budget_check_after_seed0(cfg, smoke, seeds) -> tuple[bool, str]:
    """Budget re-estimate with real numbers: GPU hours of the sweep + seed-0 runs, and the remaining seeds projected
    from the seed-0 run times. Returns (within_budget, message); the message is also written to
    results/budget_check.json."""
    root = results_root(cfg, smoke)
    sweep = 0.0
    sel = root / "lr_sweep" / "selected.json"
    if sel.exists():
        sweep = sum(t for s in load_json(sel)["summary"].values() for v in s["tried"].values()
                    for t in v["epoch_time_s"]) / 3600
    seed0 = {v: load_json(run_dir(root, v, seeds[0]) / "metrics.json")["total_time_s"] / 3600 for v in VARIANTS}
    done_later = 0.0
    for s in seeds[1:]:
        for v in VARIANTS:
            mp = run_dir(root, v, s) / "metrics.json"
            if mp.exists():
                done_later += load_json(mp)["total_time_s"] / 3600
    projected = sum(seed0.values()) * len(seeds[1:])
    total = sweep + sum(seed0.values()) + max(projected, done_later)
    budget = cfg.get("budget", {}).get("total_gpu_hours", 8.0)
    ok = total <= budget
    msg = (f"budget check after seed {seeds[0]}: sweep {sweep:.2f} h + seed-0 runs {sum(seed0.values()):.2f} h "
           f"({', '.join(f'{k} {v:.2f}' for k, v in seed0.items())}) + projected seeds {seeds[1:]} {projected:.2f} h "
           f"= {total:.2f} h vs budget {budget:.1f} h -> {'WITHIN' if ok else 'OVER'}")
    save_json({"within_budget": ok, "message": msg, "sweep_h": sweep, "seed0_h": seed0, "projected_h": projected,
               "total_h": total, "budget_h": budget, "time": now_str()}, root / "budget_check.json")
    return ok, msg


def stage_speedcheck(cfg, smoke, device, log, steps=40, warmup=5, assumed_epochs=(12, 18, 25, 40)):
    """Times `steps` real training steps per architecture (micro-batch from config, AMP) without saving
    anything except results/profile/speedcheck.json, and projects s/epoch and the full-run total."""
    from .train import epoch_permutation, make_optimizer, make_scheduler, train_epoch

    data = get_data(cfg, smoke)
    tr_x, tr_y = (t.to(device) for t in data["chunks"]["train"])
    n_steps = (tr_x.size(0) + cfg["train"]["batch_size"] - 1) // cfg["train"]["batch_size"]
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    out = {"device": str(device), "time": now_str(), "per_arch": {}}
    for arch in ARCHS:
        model = build_model(arch, cfg, len(data["vocab"]), data["vocab"].pad_id, 0).to(device)
        opt = make_optimizer(model, cfg, 1e-3)
        sched = make_scheduler(opt, arch, cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        perm = epoch_permutation(tr_x.size(0), 0, 1).to(device)
        train_epoch(model, opt, sched, scaler, tr_x, tr_y, perm, cfg, device, data["vocab"].pad_id, amp, max_steps=warmup)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        r = train_epoch(model, opt, sched, scaler, tr_x, tr_y, perm, cfg, device, data["vocab"].pad_id, amp,
                        max_steps=steps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_step = (time.perf_counter() - t0) / r["steps"]
        out["per_arch"][arch] = {"s_per_step": per_step, "train_s_per_epoch": per_step * n_steps}
        del model, opt
    ep = {a: out["per_arch"][a]["train_s_per_epoch"] * 1.05 for a in ARCHS}  # +5% for validation
    per_seed_epoch = ep["rnn"] + ep["lstm"] + ep["transformer"] + 2 * ep["lstm"]
    seeds = len(cfg["train"]["seeds"])
    out["projected_full_runs_h"] = {f"{e}_epochs_per_run": seeds * per_seed_epoch * e / 3600 for e in assumed_epochs}
    log("speedcheck: " + ", ".join(f"{a} {v['train_s_per_epoch']:.0f} s/epoch" for a, v in out["per_arch"].items())
        + " | projected full runs (all 15 LM runs): "
        + ", ".join(f"{k.split('_')[0]} ep -> {v:.1f} h" for k, v in out["projected_full_runs_h"].items()))
    # Budget check: GPU hours already spent on the sweep are subtracted from the budget
    spent = 0.0
    sel = ROOT / "results" / "lr_sweep" / "selected.json"
    if sel.exists():
        spent = sum(t for s in load_json(sel)["summary"].values() for v in s["tried"].values()
                    for t in v["epoch_time_s"]) / 3600
    b = cfg.get("budget", {"total_gpu_hours": 8.0, "assumed_typical_epochs": 15})
    typical = seeds * per_seed_epoch * b["assumed_typical_epochs"] / 3600
    worst = seeds * per_seed_epoch * cfg["train"]["max_epochs"] / 3600
    left = b["total_gpu_hours"] - spent
    out.update({"sweep_hours_spent": spent, "typical_h": typical, "worst_h": worst, "budget_left_h": left,
                "within_budget": typical <= left})
    log(f"budget: {left:.2f} h left after sweep ({spent:.2f} h); typical ({b['assumed_typical_epochs']} epochs) "
        f"{typical:.1f} h, worst (max_epochs={cfg['train']['max_epochs']}) {worst:.1f} h -> "
        + ("WITHIN BUDGET: start the full runs" if typical <= left else
           "OVER BUDGET: escalate (Kaggle for seeds 1-2, or fewer seeds)"))
    if not smoke:
        save_json(out, ROOT / "results" / "profile" / "speedcheck.json")
    return out


def stage_profile(cfg, device, log, lr=1e-3):
    """One full-data epoch per neural model on the GPU (results/profile/, not used for any result)."""
    data = get_data(cfg, False)
    out = {}
    for arch in ARCHS:
        d = ROOT / "results" / "profile" / arch
        m = train_run(cfg, arch, 0, lr, d, data, device, max_epochs=1, eval_test=False,
                      extra_manifest={"purpose": "profiling"})
        h = m["history"][0]
        out[arch] = {"epoch_time_s": h["epoch_time_s"], "train_time_s": h["train_time_s"], "peak_mem_mb": h["peak_mem_mb"],
                     "micro_batch": m["micro_batch"], "steps": h["steps"], "train_loss": h["train_loss"],
                     "val_loss": h["val_loss"], "skipped_steps": h["skipped_steps"]}
        log(f"profile {arch}: {out[arch]}")
    save_json(out, ROOT / "results" / "profile" / "profile.json")
    return out
