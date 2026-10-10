"""Tests 11 (pooling_excludes_pad) and 12 (part3_no_leak) + tokenization conversion."""
import inspect

import numpy as np
import pytest
import torch

from src import part3 as p3
from src.models import build_model


@pytest.mark.parametrize("arch", ["lstm", "transformer", "rnn"])
@pytest.mark.parametrize("how", ["mean", "last"])
def test_pooling_excludes_pad(cfg, smoke_data, cpu, arch, how):
    V = len(smoke_data["vocab"])
    model = build_model(arch, cfg, V, 0, seed=0).eval()
    g = torch.Generator().manual_seed(0)
    short = torch.randint(3, V, (20,), generator=g).tolist()
    long = torch.randint(3, V, (77,), generator=g).tolist()
    batch = p3.extract_features(model, [short, long], 0, cpu, poolings=(how,))[how]
    alone = p3.extract_features(model, [short], 0, cpu, poolings=(how,))[how]
    assert np.allclose(batch[0], alone[0], atol=1e-5), "padding changed the pooled representation"
    # explicit reference for mean pooling of the unpadded sequence
    with torch.no_grad():
        h = model.encode(torch.tensor([short]))[0]
    ref = h.mean(0) if how == "mean" else h[-1]
    assert np.allclose(alone[0], ref.numpy(), atol=1e-5)
    # including pad positions would give a different answer (the test is sensitive)
    if how == "mean":
        with torch.no_grad():
            hp = model.encode(torch.tensor([short + [0] * 57]))[0]
        assert not np.allclose(hp.mean(0).numpy(), ref.numpy(), atol=1e-4)


def test_select_C_never_sees_test():
    assert set(inspect.signature(p3.select_C).parameters) == {"Xtr", "ytr", "Xva", "yva", "C_grid", "max_iter"}
    rng = np.random.default_rng(0)
    Xtr, Xva = rng.normal(size=(200, 8)), rng.normal(size=(80, 8))
    ytr, yva = (Xtr[:, 0] > 0).astype(int), (Xva[:, 0] > 0).astype(int)
    Xte1, yte1 = rng.normal(size=(50, 8)), rng.integers(0, 2, 50)
    Xte2, yte2 = rng.normal(size=(50, 8)) * 100, rng.integers(0, 2, 50)
    a = p3.fit_and_test(Xtr, ytr, Xva, yva, Xte1, yte1, [0.01, 0.1, 1, 10], 2000)
    b = p3.fit_and_test(Xtr, ytr, Xva, yva, Xte2, yte2, [0.01, 0.1, 1, 10], 2000)
    assert a["selection"] == b["selection"], "C selection depends on test data"


def test_lm_params_unchanged_after_feature_extraction(cfg, smoke_data, cpu):
    V = len(smoke_data["vocab"])
    for arch in ("lstm", "transformer"):
        model = build_model(arch, cfg, V, 0, seed=0)
        model.train()  # extraction must switch to eval itself
        before = {k: v.clone() for k, v in model.state_dict().items()}
        seqs = [torch.randint(3, V, (n,)).tolist() for n in (5, 30, 128)]
        f1 = p3.extract_features(model, seqs, 0, cpu)["mean"]
        f2 = p3.extract_features(model, seqs, 0, cpu)["mean"]
        assert np.array_equal(f1, f2), "features not deterministic (dropout active?)"
        for k, v in model.state_dict().items():
            assert torch.equal(before[k], v), k
        assert all(p.grad is None for p in model.parameters())


def test_conversion_to_wikitext_conventions(smoke_data):
    vs = set(smoke_data["vocab"].itos)
    raw = ("Wall St. Bears Claw Back (Reuters) Reuters - Short-sellers, Wall Street's dwindling\\band "
           "of ultra-cynics, are seeing green again. It's $1,234.5 for 3.5% &lt;b&gt;now&lt;/b&gt; #39;s don't")
    toks = p3.convert_tokens(raw, vs)
    s = " ".join(toks)
    assert "short @-@ sellers" in s and "ultra @-@ cynics" in s
    assert "1 @,@ 234 @.@ 5" in s and "3 @.@ 5 %" in s
    assert "street 's" in s and "don 't" in s and "it 's" in s
    assert "( reuters )" in s and "\\" not in s and "<b>" not in s and "&lt;" not in s
    assert "again ." in s and "$" in toks
    before = p3.oov_rate([p3.baseline_tokens(raw)], vs)["oov_rate"]
    after = p3.oov_rate([toks], vs)["oov_rate"]
    assert after < before
