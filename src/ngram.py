"""Interpolated trigram LM implemented with dicts/Counters.

P(w|u,v) = l3*P_ML(w|u,v) + l2*P_ML(w|v) + l1*P_ML(w) + l0*(1/V), renormalising the lambdas over the
orders whose context is available (seen in training, and present in the chunk: the context is reset at
every chunk start, so the first target of a chunk only has a bigram context).
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict

import numpy as np
import torch


class InterpolatedTrigram:
    def __init__(self, vocab_size: int, pad_id: int = 0):
        self.V = vocab_size
        self.pad_id = pad_id
        self.uni = Counter()
        self.n_tokens = 0
        self.next2: dict[int, Counter] = defaultdict(Counter)            # v -> Counter(w)
        self.ctx2: Counter = Counter()                                     # v -> total
        self.next3: dict[tuple[int, int], Counter] = defaultdict(Counter)  # (u,v) -> Counter(w)
        self.ctx3: Counter = Counter()                                     # (u,v) -> total
        self.lambdas = None  # (l3, l2, l1, l0)
        self._p1 = None

    # ------------------------------------------------------------------ training
    def fit(self, stream) -> "InterpolatedTrigram":
        ids = [int(x) for x in stream]
        assert self.pad_id not in ids, "pad must not occur in the training stream"
        self.uni.update(ids)
        self.n_tokens = len(ids)
        for v, w in zip(ids[:-1], ids[1:]):
            self.next2[v][w] += 1
            self.ctx2[v] += 1
        for u, v, w in zip(ids[:-2], ids[1:-1], ids[2:]):
            self.next3[(u, v)][w] += 1
            self.ctx3[(u, v)] += 1
        self.next2 = dict(self.next2)
        self.next3 = dict(self.next3)
        p1 = np.zeros(self.V, dtype=np.float64)
        for w, c in self.uni.items():
            p1[w] = c / self.n_tokens
        self._p1 = p1
        return self

    def set_lambdas(self, lambdas) -> None:
        lam = tuple(float(x) for x in lambdas)
        if len(lam) != 4 or min(lam) < 0 or abs(sum(lam) - 1) > 1e-9:
            raise ValueError(f"lambdas must be 4 non-negative weights summing to 1, got {lam}")
        if lam[2] + lam[3] <= 0:
            raise ValueError("l1 + l0 must be > 0 so that every in-vocab word has non-zero probability")
        self.lambdas = lam

    # ------------------------------------------------------------------ components
    def components(self, u: int | None, v: int, w: int) -> tuple[float, float, float, float, bool, bool]:
        """(p3, p2, p1, p0, avail3, avail2) for target w given context (u, v); u=None if absent."""
        p1 = self.uni.get(w, 0) / self.n_tokens
        c2 = self.ctx2.get(v, 0)
        a2 = c2 > 0
        p2 = self.next2[v].get(w, 0) / c2 if a2 else 0.0
        a3 = False
        p3 = 0.0
        if u is not None:
            c3 = self.ctx3.get((u, v), 0)
            if c3 > 0:
                a3 = True
                p3 = self.next3[(u, v)].get(w, 0) / c3
        return p3, p2, p1, 1.0 / self.V, a3, a2

    def chunk_components(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Component probabilities for every non-pad target of the chunked split.
        Returns P (n,4) = [p3,p2,p1,p0] and A (n,4) availability (float). Context reset per chunk."""
        X = inputs.tolist()
        Y = targets.tolist()
        P, A = [], []
        pad = self.pad_id
        for x, y in zip(X, Y):
            for j, w in enumerate(y):
                if w == pad:
                    continue
                u = x[j - 1] if j >= 1 else None
                p3, p2, p1, p0, a3, a2 = self.components(u, x[j], w)
                P.append((p3, p2, p1, p0))
                A.append((float(a3), float(a2), 1.0, 1.0))
        return np.asarray(P, dtype=np.float64), np.asarray(A, dtype=np.float64)

    @staticmethod
    def mix(P: np.ndarray, A: np.ndarray, lambdas) -> np.ndarray:
        lam = np.asarray(lambdas, dtype=np.float64)
        num = (P * A) @ lam
        den = A @ lam
        with np.errstate(divide="ignore", invalid="ignore"):
            return num / den

    # ------------------------------------------------------------------ scoring
    def score_chunks(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        assert self.lambdas is not None
        P, A = self.chunk_components(inputs, targets)
        p = self.mix(P, A, self.lambdas)
        if not np.all(p > 0):
            raise RuntimeError(f"{int((p <= 0).sum())} targets received zero probability")
        nll = float(-np.log(p).sum())
        n = int(p.size)
        return {"nll_sum": nll, "count": n, "ce": nll / n, "ppl": math.exp(nll / n)}

    def distribution(self, context: list[int]) -> np.ndarray:
        """Full normalised distribution over V given the context (last two tokens used)."""
        assert self.lambdas is not None
        l3, l2, l1, l0 = self.lambdas
        v = context[-1]
        u = context[-2] if len(context) >= 2 else None
        dist = np.full(self.V, l0 / self.V, dtype=np.float64)
        dist += l1 * self._p1
        total = l0 + l1
        c2 = self.ctx2.get(v, 0)
        if c2 > 0:
            row = self.next2[v]
            idx = np.fromiter(row.keys(), dtype=np.int64, count=len(row))
            cnt = np.fromiter(row.values(), dtype=np.float64, count=len(row))
            dist[idx] += l2 * cnt / c2
            total += l2
        if u is not None and self.ctx3.get((u, v), 0) > 0:
            row = self.next3[(u, v)]
            c3 = self.ctx3[(u, v)]
            idx = np.fromiter(row.keys(), dtype=np.int64, count=len(row))
            cnt = np.fromiter(row.values(), dtype=np.float64, count=len(row))
            dist[idx] += l3 * cnt / c3
            total += l3
        return dist / total


def lambda_grid(step: float) -> list[tuple[float, float, float, float]]:
    n = int(round(1 / step))
    out = []
    for a, b, c in itertools.product(range(n + 1), repeat=3):
        d = n - a - b - c
        if d >= 0:
            out.append((a / n, b / n, c / n, d / n))
    return out


def grid_search_lambdas(P: np.ndarray, A: np.ndarray, step: float, batch: int = 64) -> dict:
    """RULE: lambda grid search on VALIDATION components. Returns best + full table."""
    grid = lambda_grid(step)
    G = np.asarray(grid, dtype=np.float64)  # (g,4)
    PA = P * A
    ces = np.empty(len(grid))
    n = P.shape[0]
    for s in range(0, len(grid), batch):
        g = G[s:s + batch].T                      # (4,b)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = (PA @ g) / (A @ g)                # (n,b)
            ll = np.log(p)
        ll[~np.isfinite(ll)] = -np.inf
        ces[s:s + batch] = -ll.sum(0) / n
    ces[~np.isfinite(ces)] = np.inf
    # a valid model needs l1 + l0 > 0 (non-zero probability for every in-vocab word)
    ces[(G[:, 2] + G[:, 3]) <= 0] = np.inf
    order = np.argsort(ces, kind="stable")
    best = order[0]
    return {
        "best_lambdas": grid[best],
        "best_val_ce": float(ces[best]),
        "n_grid": len(grid),
        "top10": [{"lambdas": grid[i], "val_ce": float(ces[i])} for i in order[:10]],
    }
