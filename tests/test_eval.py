"""Test 5 (ppl_definition) and test 4 (padding_excluded)."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.data import make_chunks
from src.evaluate import evaluate_lm, lm_loss_sum
from src.models import ARCHS, build_model
from src.ngram import InterpolatedTrigram


@pytest.fixture(scope="module")
def small_split(smoke_data):
    """A short real stream whose last chunk is partial (so padding is exercised)."""
    s = smoke_data["streams"]["validation"][: 128 * 3 + 50]
    x, y = make_chunks(s, 128, 0)
    assert (y == 0).any()
    return s, x, y


@pytest.mark.parametrize("arch", ARCHS)
def test_ppl_equals_naive_loop(cfg, smoke_data, small_split, cpu, arch):
    V = len(smoke_data["vocab"])
    model = build_model(arch, cfg, V, 0, seed=3).eval()
    s, x, y = small_split
    r = evaluate_lm(model, x, y, 0, cpu, amp=False, batch_size=2)
    # naive reference: one chunk at a time, one position at a time
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for c in range(x.size(0)):
            logp = torch.log_softmax(model(x[c:c + 1])[0].double(), -1)
            for j in range(128):
                t = int(y[c, j])
                if t == 0:
                    continue
                tot -= float(logp[j, t])
                cnt += 1
    assert cnt == s.numel() - 1 == r["count"]
    assert r["ce"] == pytest.approx(tot / cnt, rel=1e-5)
    assert r["ppl"] == pytest.approx(math.exp(tot / cnt), rel=1e-5)


def test_ngram_and_neural_score_same_targets(cfg, smoke_data, cpu):
    v = smoke_data["vocab"]
    ng = InterpolatedTrigram(len(v), 0).fit(smoke_data["streams"]["train"])
    ng.set_lambdas((0.3, 0.3, 0.3, 0.1))
    model = build_model("lstm", cfg, len(v), 0, seed=0)
    for split in ("validation", "test"):
        x, y = smoke_data["chunks"][split]
        a = ng.score_chunks(x, y)
        b = evaluate_lm(model, x, y, 0, cpu, amp=False)
        assert a["count"] == b["count"] == smoke_data["streams"][split].numel() - 1


def test_ngram_ppl_equals_naive_loop(smoke_data, small_split):
    v = smoke_data["vocab"]
    ng = InterpolatedTrigram(len(v), 0).fit(smoke_data["streams"]["train"])
    ng.set_lambdas((0.3, 0.3, 0.3, 0.1))
    s, x, y = small_split
    r = ng.score_chunks(x, y)
    tot, cnt = 0.0, 0
    for c in range(x.size(0)):
        for j in range(128):
            t = int(y[c, j])
            if t == 0:
                continue
            ctx = x[c, :j + 1].tolist()       # only this chunk's inputs (context reset)
            tot -= math.log(ng.distribution(ctx)[t])
            cnt += 1
    assert cnt == r["count"] == s.numel() - 1
    assert r["ce"] == pytest.approx(tot / cnt, rel=1e-9)


@pytest.mark.parametrize("arch", ARCHS)
def test_padding_excluded(cfg, smoke_data, cpu, arch):
    V = len(smoke_data["vocab"])
    model = build_model(arch, cfg, V, 0, seed=1).eval()
    g = torch.Generator().manual_seed(0)
    real = torch.randint(3, V, (1, 89), generator=g)
    x_unpad, y_unpad = real[:, :-1], real[:, 1:]                 # 88 targets
    x_pad = torch.zeros(1, 128, dtype=torch.long)
    y_pad = torch.zeros(1, 128, dtype=torch.long)
    x_pad[:, :88], y_pad[:, :88] = x_unpad, y_unpad
    with torch.no_grad():
        nll_pad, n_pad = lm_loss_sum(model(x_pad), y_pad, 0)
        nll_unpad, n_unpad = lm_loss_sum(model(x_unpad), y_unpad, 0)
    assert n_pad == n_unpad == 88
    assert float(nll_pad) == pytest.approx(float(nll_unpad), rel=1e-5)
    # same through the evaluator (padding changes neither the sum nor the count)
    a = evaluate_lm(model, x_pad, y_pad, 0, cpu, amp=False)
    b = evaluate_lm(model, x_unpad, y_unpad, 0, cpu, amp=False)
    assert a["count"] == b["count"] == 88 and a["ce"] == pytest.approx(b["ce"], rel=1e-5)
    # explicit check that pad targets are ignored by the loss
    with torch.no_grad():
        logits = model(x_pad).float()
    manual = F.cross_entropy(logits[0, :88], y_pad[0, :88], reduction="sum")
    assert float(nll_pad) == pytest.approx(float(manual), rel=1e-6)


def test_training_loss_ignores_padding(cfg, tiny_data, cpu):
    """train_epoch counts only non-pad targets, and its token-weighted loss for a padded batch equals the
    loss computed on the unpadded sequences (dropout disabled to compare exactly)."""
    import copy

    from src.train import make_optimizer, make_scheduler, train_epoch

    V = len(tiny_data["vocab"])
    c = copy.deepcopy(cfg)
    c["train"]["batch_size"] = c["train"]["micro_batch"] = 2
    c["model"]["dropout"] = 0.0
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(2, 128, dtype=torch.long)
    y = torch.zeros(2, 128, dtype=torch.long)
    x[0], y[0] = torch.randint(3, V, (128,), generator=g), torch.randint(3, V, (128,), generator=g)
    x[1, :40], y[1, :40] = torch.randint(3, V, (40,), generator=g), torch.randint(3, V, (40,), generator=g)
    model = build_model("lstm", c, V, 0, seed=0)
    ref = copy.deepcopy(model).eval()
    with torch.no_grad():
        l_a = lm_loss_sum(ref(x[:1]), y[:1], 0)[0]
        l_b = lm_loss_sum(ref(x[1:, :40]), y[1:, :40], 0)[0]
    opt = make_optimizer(model, c, 1e-3)
    out = train_epoch(model, opt, make_scheduler(opt, "lstm", c), torch.amp.GradScaler("cuda", enabled=False),
                      x, y, torch.arange(2), c, cpu, 0, False)
    assert out["tokens"] == 168
    assert out["train_loss"] == pytest.approx(float((l_a + l_b) / 168), rel=1e-5)
