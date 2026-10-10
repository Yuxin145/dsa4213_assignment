"""Test 2 (ngram_normalized) + interpolation/renormalisation semantics and lambda search on validation."""
import math

import numpy as np
import pytest
import torch

from src.ngram import InterpolatedTrigram, grid_search_lambdas, lambda_grid

LAMBDAS = (0.4, 0.3, 0.2, 0.1)


@pytest.fixture(scope="module")
def full_ngram(tokenized, smoke_data):
    from src.data import lines_to_stream

    v = smoke_data["vocab"]
    ng = InterpolatedTrigram(len(v), v.pad_id).fit(lines_to_stream(tokenized["train"], v))
    ng.set_lambdas(LAMBDAS)
    return ng


def _contexts(ng, vocab):
    rng = np.random.default_rng(0)
    seen_tri = list(ng.ctx3.keys())[:6]
    ctxs = [list(k) for k in seen_tri]                                   # seen trigram contexts
    ctxs += [[int(x)] for x in rng.integers(3, len(vocab), 4)]            # bigram-only (chunk start)
    while len(ctxs) < 16:                                                 # random pairs, mostly unseen
        u, w = (int(x) for x in rng.integers(3, len(vocab), 2))
        ctxs.append([u, w])
    ctxs += [[vocab.pad_id], [vocab.pad_id, vocab.pad_id]]                 # unseen bigram AND trigram
    ctxs += [[vocab.stoi["the"], vocab.stoi["the"]], [vocab.eos_id, vocab.stoi["the"]]]
    assert len(ctxs) == 20
    assert any(len(c) == 2 and (c[0], c[1]) not in ng.ctx3 for c in ctxs), "need unseen trigram contexts"
    return ctxs


def test_distribution_normalised_20_contexts(full_ngram, smoke_data):
    ng, vocab = full_ngram, smoke_data["vocab"]
    for ctx in _contexts(ng, vocab):
        d = ng.distribution(ctx)
        assert d.shape == (len(vocab),)
        assert abs(d.sum() - 1.0) < 1e-6, (ctx, d.sum())
        assert (d[1:] > 0).all(), "zero probability for an in-vocab word"


def test_distribution_matches_components_and_renormalisation(full_ngram, smoke_data):
    ng, vocab = full_ngram, smoke_data["vocab"]
    l3, l2, l1, l0 = LAMBDAS
    the = vocab.stoi["the"]
    # unseen trigram context -> renormalise over (l2, l1, l0)
    u, v = vocab.pad_id, the
    assert (u, v) not in ng.ctx3 and ng.ctx2[v] > 0
    d = ng.distribution([u, v])
    for w in [the, vocab.stoi["city"], vocab.eos_id, vocab.unk_id]:
        p2 = ng.next2[v].get(w, 0) / ng.ctx2[v]
        p1 = ng.uni[w] / ng.n_tokens
        expect = (l2 * p2 + l1 * p1 + l0 / len(vocab)) / (l2 + l1 + l0)
        assert d[w] == pytest.approx(expect, rel=1e-12)
    # seen trigram context -> all four orders
    (u, v), _ = next(iter(ng.ctx3.items()))
    d = ng.distribution([u, v])
    for w in list(ng.next3[(u, v)].keys())[:3] + [vocab.unk_id]:
        p3, p2, p1, p0, a3, a2 = ng.components(u, v, w)
        assert a3 and a2
        assert d[w] == pytest.approx(l3 * p3 + l2 * p2 + l1 * p1 + l0 * p0, rel=1e-12)


def test_no_zero_probability_on_test(full_ngram, tokenized, smoke_data):
    from src.data import lines_to_stream, make_chunks

    v = smoke_data["vocab"]
    x, y = make_chunks(lines_to_stream(tokenized["test"], v), 128, v.pad_id)
    P, A = full_ngram.chunk_components(x, y)
    p = full_ngram.mix(P, A, LAMBDAS)
    assert p.size == int((y != 0).sum())
    assert (p > 0).all() and np.isfinite(np.log(p)).all()
    r = full_ngram.score_chunks(x, y)
    assert r["count"] == p.size and math.isfinite(r["ppl"])


def test_context_reset_at_chunk_start(full_ngram, smoke_data):
    """The first target of a chunk only gets a bigram context (no trigram from the previous chunk)."""
    ng = full_ngram
    (u, v), _ = next(iter(ng.ctx3.items()))
    w = next(iter(ng.next3[(u, v)]))
    x = torch.zeros(2, 128, dtype=torch.long)
    y = torch.zeros(2, 128, dtype=torch.long)
    x[0, 127], y[0, 127] = u, v          # chunk 0 ends with u -> v
    x[1, 0], y[1, 0] = v, w              # chunk 1 starts with v -> w
    P, A = ng.chunk_components(x, y)
    assert A[-1, 0] == 0.0, "trigram context leaked across a chunk boundary"
    assert A[-1, 1] == 1.0


def test_lambda_grid_search_toy():
    assert len(lambda_grid(0.05)) == 1771
    for g in lambda_grid(0.05):
        assert abs(sum(g) - 1) < 1e-9 and min(g) >= 0
    # toy problem where the bigram component is always right -> l2 should dominate
    n = 1000
    P = np.column_stack([np.full(n, 0.01), np.full(n, 0.9), np.full(n, 0.05), np.full(n, 1e-4)])
    A = np.ones((n, 4))
    r = grid_search_lambdas(P, A, 0.05)
    assert r["best_lambdas"][1] >= 0.9
    assert r["best_lambdas"][2] + r["best_lambdas"][3] > 0
