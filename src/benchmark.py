"""Inference cost protocol (§6.4).

Common workload: score every target token of the chunked test split (log-probability of each target),
batch of 32 chunks; the first `warmup` batches are excluded from timing; tokens/sec is the number of
non-pad targets in the timed batches divided by their summed time; median of `runs` runs.
Generation latency: greedy, 50 new tokens, one prompt, batch 1; one untimed warm-up, median of 5 runs.
"""
from __future__ import annotations

import statistics
import time

import numpy as np
import torch

from .evaluate import autocast_ctx
from .generate import generate


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def neural_score_batch(model, x, y, device, amp, pad_id):
    with torch.no_grad():
        with autocast_ctx(device, amp):
            logits = model(x)
        lp = torch.log_softmax(logits.float(), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        lp = lp[y != pad_id]
        _ = float(lp.sum())  # forces completion
    return lp


def throughput(score_fn, inputs, targets, batch_size: int, warmup: int, runs: int, pad_id: int,
               device=torch.device("cpu")) -> dict:
    n_batches = (inputs.size(0) + batch_size - 1) // batch_size
    warmup = min(warmup, n_batches - 1)  # only binds for tiny (smoke) splits; recorded below
    per_run = []
    for _ in range(runs):
        timed_tok, timed_s = 0, 0.0
        for bi, s in enumerate(range(0, inputs.size(0), batch_size)):
            x, y = inputs[s:s + batch_size], targets[s:s + batch_size]
            _sync(device)
            t0 = time.perf_counter()
            score_fn(x, y)
            _sync(device)
            dt = time.perf_counter() - t0
            if bi >= warmup:
                timed_tok += int((y != pad_id).sum())
                timed_s += dt
        per_run.append(timed_tok / timed_s)
    return {"tokens_per_s_median": statistics.median(per_run), "tokens_per_s_runs": per_run,
            "timed_tokens_per_run": timed_tok, "warmup_batches_used": warmup, "n_batches": n_batches}


def neural_throughput(model, inputs, targets, device, amp, cfg, pad_id) -> dict:
    model.eval()
    b = cfg["benchmark"]
    xin, yin = inputs.to(device), targets.to(device)
    fn = lambda x, y: neural_score_batch(model, x, y, device, amp, pad_id)
    return throughput(fn, xin, yin, b["batch_size"], b["warmup_batches"], b["throughput_runs"], pad_id, device)


def ngram_throughput(ngram, inputs, targets, cfg, pad_id) -> dict:
    b = cfg["benchmark"]

    def fn(x, y):
        P, A = ngram.chunk_components(x, y)
        return np.log(ngram.mix(P, A, ngram.lambdas)).sum()

    return throughput(fn, inputs, targets, b["batch_size"], b["warmup_batches"], b["throughput_runs"], pad_id)


def latency(scorer, prompt_ids, cfg, pad_id, device=torch.device("cpu")) -> dict:
    b = cfg["benchmark"]
    generate(scorer, prompt_ids, b["latency_new_tokens"], pad_id, "greedy")  # warm-up
    times = []
    for _ in range(b["latency_runs"]):
        _sync(device)
        t0 = time.perf_counter()
        generate(scorer, prompt_ids, b["latency_new_tokens"], pad_id, "greedy")
        _sync(device)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return {"latency_s_median": med, "latency_s_runs": times,
            "ms_per_token": 1000 * med / b["latency_new_tokens"]}
