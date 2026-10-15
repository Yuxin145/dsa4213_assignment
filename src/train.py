"""Training loop for neural LMs with AMP, gradient accumulation, early stopping and checkpoint/resume.

Each run directory contains config.yaml, log.txt, metrics.json (updated every epoch), last.pt, best.pt
and a DONE marker written last. Re-running skips DONE runs and resumes from last.pt otherwise.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import autocast_ctx, evaluate_lm, lm_loss_sum
from .models import build_model, param_split
from .utils import Logger, config_hash, load_json, make_manifest, now_str, save_json, save_yaml


def make_optimizer(model, cfg: dict, lr: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=cfg["train"]["weight_decay"])


def make_scheduler(opt, arch: str, cfg: dict):
    warm = cfg["train"]["warmup_steps"] if arch == "transformer" else 0
    if warm > 0:
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, (step + 1) / warm))
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)


def _raise_on_amp_skip(scaler, scale_before: float) -> None:
    """Escalate any GradScaler-detected non-finite gradient instead of silently skipping an update."""
    if scaler.is_enabled() and scaler.get_scale() < scale_before:
        raise FloatingPointError(
            "AMP GradScaler skipped an optimizer step after detecting non-finite scaled gradients. "
            "Stop this run and record the escalation in VERIFICATION.md; do not silently mix precision policies."
        )


def _rng_state() -> dict:
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def epoch_permutation(n: int, seed: int, epoch: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed * 100_003 + epoch)
    return torch.randperm(n, generator=g)


def train_epoch(model, opt, sched, scaler, inputs, targets, perm, cfg, device, pad_id, amp,
                max_steps: int | None = None, loss_trace: list | None = None) -> dict:
    """One pass over the training chunks in `perm` order. Loss per optimizer step = sum NLL over the
    effective batch / #non-pad targets in that batch (identical with or without accumulation)."""
    model.train()
    B = cfg["train"]["batch_size"]
    mb = cfg["train"]["micro_batch"]
    assert B % mb == 0, "batch_size must be a multiple of micro_batch"
    clip = cfg["train"]["grad_clip"]
    params = [p for p in model.parameters() if p.requires_grad]
    tot_nll, tot_cnt, steps, skipped, gnorms = 0.0, 0, 0, 0, []
    for s in range(0, perm.numel(), B):
        idx = perm[s:s + B]
        y_all = targets[idx]
        n_tok = int((y_all != pad_id).sum())
        if n_tok == 0:
            continue
        opt.zero_grad(set_to_none=True)
        step_nll = 0.0
        for ms in range(0, idx.numel(), mb):
            x = inputs[idx[ms:ms + mb]].to(device, non_blocking=True)
            y = targets[idx[ms:ms + mb]].to(device, non_blocking=True)
            with autocast_ctx(device, amp):
                logits = model(x)
            nll, _ = lm_loss_sum(logits, y, pad_id)
            scaler.scale(nll / n_tok).backward()
            step_nll += float(nll.detach())
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, clip)
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.is_enabled() and scaler.get_scale() < scale_before:
            skipped += 1
        _raise_on_amp_skip(scaler, scale_before)
        sched.step()
        gnorms.append(float(gn))
        tot_nll += step_nll
        tot_cnt += n_tok
        steps += 1
        if loss_trace is not None:
            loss_trace.append(step_nll / n_tok)
        if max_steps is not None and steps >= max_steps:
            break
    finite = [g for g in gnorms if np.isfinite(g)]
    return {"train_loss": tot_nll / max(tot_cnt, 1), "steps": steps, "tokens": tot_cnt,
            "skipped_steps": skipped, "grad_norm_mean": float(np.mean(finite)) if finite else float("nan")}


def _save_ckpt(path: Path, **obj) -> None:
    tmp = path.with_suffix(".pt.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train_run(cfg: dict, arch: str, seed: int, lr: float, run_dir: Path, data: dict, device: torch.device,
              pretrained: torch.Tensor | None = None, max_epochs: int | None = None, eval_test: bool = True,
              stop_after_epochs: int | None = None, max_steps_per_epoch: int | None = None,
              extra_manifest: dict | None = None, echo: bool = True) -> dict:
    """Train one (experiment, arch, seed) run. Returns metrics dict.
    stop_after_epochs simulates an interruption (used by the resume test)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    if (run_dir / "DONE").exists():
        return load_json(metrics_path)

    vocab = data["vocab"]
    pad_id = vocab.pad_id
    run_cfg = dict(cfg)
    run_cfg["run"] = {"arch": arch, "seed": seed, "lr": lr,
                      "max_epochs": max_epochs or cfg["train"]["max_epochs"], "vocab_sha256": vocab.sha256}
    save_yaml(run_cfg, run_dir / "config.yaml")
    log = Logger(run_dir / "log.txt", echo=echo)
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    max_ep = run_cfg["run"]["max_epochs"]
    patience = cfg["train"]["patience"]
    cap_s = cfg["train"]["wall_clock_cap_min"] * 60

    model = build_model(arch, cfg, len(vocab), pad_id, seed, pretrained).to(device)
    torch.manual_seed(seed + 1)  # training RNG independent of how much RNG model construction consumed
    opt = make_optimizer(model, cfg, lr)
    sched = make_scheduler(opt, arch, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    tr_x, tr_y = data["chunks"]["train"]
    va_x, va_y = data["chunks"]["validation"]
    if device.type == "cuda":
        tr_x, tr_y = tr_x.to(device), tr_y.to(device)

    state = {"epoch": 0, "global_step": 0, "tokens_seen": 0, "best_val": float("inf"), "best_epoch": 0,
             "bad_epochs": 0, "cum_time": 0.0, "time_to_best": None, "history": [], "start": now_str(),
             "stopped_reason": None, "first_step_losses": []}
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        _set_rng_state(ck["rng"])
        log(f"RESUME from {last_path} at epoch {state['epoch']} step {state['global_step']}")
    else:
        log(f"START arch={arch} seed={seed} lr={lr} device={device} amp={amp} "
            f"micro_batch={cfg['train']['micro_batch']} params={param_split(model)}")

    params = param_split(model)
    epochs_this_call = 0
    while state["stopped_reason"] is None:
        if state["epoch"] >= max_ep:
            state["stopped_reason"] = "max_epochs"
            break
        if state["cum_time"] >= cap_s:
            state["stopped_reason"] = "wall_clock_cap"
            log(f"wall-clock cap of {cap_s / 60:.0f} min reached")
            break
        ep = state["epoch"] + 1
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        perm = epoch_permutation(tr_x.size(0), seed, ep).to(tr_x.device)
        trace = []
        try:
            tr = train_epoch(model, opt, sched, scaler, tr_x, tr_y, perm, cfg, device, pad_id, amp,
                             max_steps=max_steps_per_epoch, loss_trace=trace)
        except FloatingPointError as exc:
            log(f"AMP ESCALATION at epoch {ep}: {exc}")
            raise
        state["first_step_losses"] = (state["first_step_losses"] + trace)[:50]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_train = time.perf_counter() - t0
        va = evaluate_lm(model, va_x, va_y, pad_id, device, amp, cfg["train"]["batch_size"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_epoch = time.perf_counter() - t0
        state["cum_time"] += t_epoch
        state["global_step"] += tr["steps"]
        state["tokens_seen"] += tr["tokens"]
        state["epoch"] = ep
        peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        rec = {"epoch": ep, "train_loss": tr["train_loss"], "val_loss": va["ce"], "val_ppl": va["ppl"],
               "epoch_time_s": t_epoch, "train_time_s": t_train, "cum_time_s": state["cum_time"],
               "lr": opt.param_groups[0]["lr"], "peak_mem_mb": peak, "steps": tr["steps"],
               "global_step": state["global_step"], "skipped_steps": tr["skipped_steps"],
               "grad_norm_mean": tr["grad_norm_mean"]}
        state["history"].append(rec)
        improved = va["ce"] < state["best_val"]
        if not np.isfinite(tr["train_loss"]) or not np.isfinite(va["ce"]):
            state["stopped_reason"] = "non_finite_loss"
            log(f"NON-FINITE LOSS at epoch {ep}: train={tr['train_loss']} val={va['ce']}")
        if improved:
            state["best_val"], state["best_epoch"], state["bad_epochs"] = va["ce"], ep, 0
            state["time_to_best"] = state["cum_time"]
            _save_ckpt(best_path, model=model.state_dict(), epoch=ep, val_ce=va["ce"])
        else:
            state["bad_epochs"] += 1
        log(f"epoch {ep:3d} | train {tr['train_loss']:.4f} | val {va['ce']:.4f} ppl {va['ppl']:.2f} | "
            f"{t_epoch:.1f}s (cum {state['cum_time'] / 60:.1f} min) | lr {rec['lr']:.2e} | "
            f"peak {peak if peak is None else round(peak)} MB | skipped {tr['skipped_steps']}"
            + (" *" if improved else ""))
        if state["stopped_reason"] is None and state["bad_epochs"] >= patience:
            state["stopped_reason"] = "early_stopping"
        _save_ckpt(last_path, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                   scaler=scaler.state_dict(), state=state, rng=_rng_state())
        save_json(_metrics(run_cfg, arch, seed, lr, state, params, None, log, device, extra_manifest),
                  metrics_path)
        epochs_this_call += 1
        if stop_after_epochs is not None and epochs_this_call >= stop_after_epochs and state["stopped_reason"] is None:
            log(f"simulated interruption after {epochs_this_call} epoch(s)")
            return load_json(metrics_path)

    log(f"STOP reason={state['stopped_reason']} best_epoch={state['best_epoch']} best_val={state['best_val']:.4f}")
    final = None
    if best_path.exists():
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        final = {"val": evaluate_lm(model, va_x, va_y, pad_id, device, amp, cfg["train"]["batch_size"])}
        if eval_test:
            te_x, te_y = data["chunks"]["test"]
            final["test"] = evaluate_lm(model, te_x, te_y, pad_id, device, amp, cfg["train"]["batch_size"])
            log(f"FINAL (best ckpt, epoch {state['best_epoch']}) val ppl {final['val']['ppl']:.2f} "
                f"test ppl {final['test']['ppl']:.2f}")
        else:
            log(f"FINAL (best ckpt, epoch {state['best_epoch']}) val ppl {final['val']['ppl']:.2f}")
    m = _metrics(run_cfg, arch, seed, lr, state, params, final, log, device, extra_manifest)
    save_json(m, metrics_path)
    (run_dir / "DONE").write_text(now_str(), encoding="utf-8")
    return m


def _metrics(run_cfg, arch, seed, lr, state, params, final, log, device, extra) -> dict:
    return {
        "arch": arch, "seed": seed, "lr": lr,
        "embedding": run_cfg["embedding"],
        "micro_batch": run_cfg["train"]["micro_batch"],
        "params": params,
        "history": state["history"],
        "epochs_run": state["epoch"],
        "best_epoch": state["best_epoch"],
        "best_val_ce": state["best_val"],
        "stopped_reason": state["stopped_reason"],
        "time_to_best_s": state["time_to_best"],
        "total_time_s": state["cum_time"],
        "steps": state["global_step"],
        "tokens_seen": state["tokens_seen"],
        "first_step_losses": state["first_step_losses"],
        "final": final,
        "manifest": make_manifest(run_cfg, seed, log.path, state["start"], device,
                                  dict(extra or {}, config_hash_run=config_hash(run_cfg))),
    }
