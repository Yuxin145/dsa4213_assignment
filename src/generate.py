"""Autoregressive generation with matched decoding for all four models (§6.5).

At every step the model scores the next token given only the prompt and the tokens generated so far
(neural models: full forward pass on the last <= context_length tokens, last position). <pad> is never
generated. Settings: greedy, or temperature + top-k sampling with a fixed CPU generator seeded per prompt.
"""
from __future__ import annotations

import numpy as np
import torch

from .evaluate import autocast_ctx


class NeuralScorer:
    def __init__(self, model, device: torch.device, amp: bool, context_length: int):
        self.model, self.device, self.amp, self.L = model.eval(), device, amp, context_length

    @torch.no_grad()
    def next_logprobs(self, ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids[-self.L:]], dtype=torch.long, device=self.device)
        with autocast_ctx(self.device, self.amp):
            logits = self.model(x)[0, -1]
        return torch.log_softmax(logits.float(), dim=-1).cpu().double()


class NgramScorer:
    def __init__(self, ngram):
        self.ngram = ngram

    def next_logprobs(self, ids: list[int]) -> torch.Tensor:
        with np.errstate(divide="ignore"):
            return torch.from_numpy(np.log(self.ngram.distribution(ids)))


def choose_next(logp: torch.Tensor, pad_id: int, mode: str, temperature: float = 1.0, top_k: int = 0,
                gen: torch.Generator | None = None) -> int:
    logp = logp.clone()
    logp[pad_id] = -float("inf")
    if mode == "greedy":
        return int(torch.argmax(logp))
    if mode == "sample":
        logits = logp / temperature
        if top_k and top_k < logits.numel():
            kth = torch.topk(logits, top_k).values[-1]
            logits[logits < kth] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1, generator=gen))
    raise ValueError(mode)


def generate(scorer, prompt_ids: list[int], n_new: int, pad_id: int, mode: str, temperature: float = 1.0,
             top_k: int = 0, seed: int | None = None) -> list[int]:
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    ids = list(prompt_ids)
    out = []
    for _ in range(n_new):
        nxt = choose_next(scorer.next_logprobs(ids), pad_id, mode, temperature, top_k, gen)
        ids.append(nxt)
        out.append(nxt)
    return out


def text_metrics(continuations: list[list[str]], unk: str = "<unk>") -> dict:
    """distinct-1/2 pooled over continuations, repeated-trigram rate (trigram already seen earlier in the
    same continuation), <unk> rate among generated tokens."""
    toks = [t for c in continuations for t in c]
    uni = set(toks)
    bi_all, n_bi, rep, n_tri = set(), 0, 0, 0
    for c in continuations:
        for i in range(len(c) - 1):
            bi_all.add((c[i], c[i + 1]))
            n_bi += 1
        seen = set()
        for i in range(len(c) - 2):
            tri = tuple(c[i:i + 3])
            if tri in seen:
                rep += 1
            seen.add(tri)
            n_tri += 1
    return {
        "n_tokens": len(toks),
        "distinct_1": len(uni) / max(len(toks), 1),
        "distinct_2": len(bi_all) / max(n_bi, 1),
        "repeated_trigram_rate": rep / max(n_tri, 1),
        "unk_rate": sum(t == unk for t in toks) / max(len(toks), 1),
    }
