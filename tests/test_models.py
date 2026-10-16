"""Tests 3 (causality), 6 (initial_loss), 7 (overfit_one_batch), 9 (no_weight_tying) + param split."""
import math

import pytest
import torch

from src.evaluate import autocast_ctx, evaluate_lm, lm_loss_sum
from src.models import ARCHS, build_model, param_split
from src.utils import get_device


@pytest.mark.parametrize("arch", ARCHS)
def test_causality(cfg, smoke_data, arch):
    V = len(smoke_data["vocab"])
    model = build_model(arch, cfg, V, 0, seed=0).eval()
    g = torch.Generator().manual_seed(1)
    x = torch.randint(3, V, (2, 128), generator=g)
    with torch.no_grad():
        base = model(x)
        for t in (0, 1, 17, 64, 126):
            x2 = x.clone()
            x2[:, t + 1:] = torch.randint(3, V, (2, 127 - t), generator=g)
            out = model(x2)
            assert torch.allclose(out[:, :t + 1], base[:, :t + 1], atol=1e-5), f"future leaked into t<={t}"
            assert not torch.allclose(out[:, t + 1:], base[:, t + 1:], atol=1e-5)
        # and the model does use the past: changing position 0 changes later logits
        x3 = x.clone()
        x3[:, 0] = (x3[:, 0] + 1) % (V - 3) + 3
        assert not torch.allclose(model(x3)[:, 5], base[:, 5], atol=1e-5)


@pytest.mark.parametrize("arch", ARCHS)
def test_causality_encode_shapes_and_eval_determinism(cfg, smoke_data, arch):
    V = len(smoke_data["vocab"])
    model = build_model(arch, cfg, V, 0, seed=0).eval()
    x = torch.randint(3, V, (2, 50))
    with torch.no_grad():
        h1, h2 = model.encode(x), model.encode(x)
    assert h1.shape == (2, 50, cfg["model"]["hidden"])
    assert torch.equal(h1, h2), "eval-mode encode must be deterministic (no dropout)"


@pytest.mark.parametrize("arch", ARCHS)
def test_initial_loss_is_log_V(cfg, smoke_data, arch):
    V = len(smoke_data["vocab"])
    device = get_device()
    model = build_model(arch, cfg, V, 0, seed=0).to(device)
    x, y = smoke_data["chunks"]["validation"]
    r = evaluate_lm(model, x, y, 0, device, amp=True)  # AMP when on CUDA, as in training
    assert math.isfinite(r["ce"])
    assert abs(r["ce"] - math.log(V)) < 1.0, (r["ce"], math.log(V))


@pytest.mark.parametrize("arch", ARCHS)
def test_overfit_one_batch(cfg, smoke_data, arch):
    """Real V, AMP on CUDA (as in training), AdamW(wd 0.01) + clip 1.0: eval loss on one small fixed batch
    (4 x 32 tokens) < 0.5 within 300 steps. Identical settings for all models: dropout disabled (a capacity /
    optimisation check, not a regularisation check) and lr 1e-2 -- the stacked 2-layer LSTM with small default
    init memorises much more slowly than the RNN/Transformer."""
    import copy

    V = len(smoke_data["vocab"])
    device = get_device()
    c = copy.deepcopy(cfg)
    c["model"]["dropout"] = 0.0
    model = build_model(arch, c, V, 0, seed=0).to(device)
    torch.manual_seed(0)
    x_all, y_all = smoke_data["chunks"]["train"]
    x, y = x_all[:4, :32].to(device), y_all[:4, :32].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    losses = []
    for step in range(300):
        model.train()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device, amp):
            logits = model(x)
        nll, n = lm_loss_sum(logits, y, 0)
        loss = nll / n
        assert torch.isfinite(loss), f"non-finite loss at step {step} (AMP={amp})"
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach()))
        if step % 25 == 24:
            model.eval()
            with torch.no_grad(), autocast_ctx(device, amp):
                ev = float(lm_loss_sum(model(x), y, 0)[0]) / n
            if ev < 0.5:
                break
    assert ev < 0.5, f"{arch} could not overfit one batch: eval loss {ev:.3f} (train {losses[-1]:.3f})"


@pytest.mark.parametrize("arch", ARCHS)
def test_no_weight_tying(cfg, smoke_data, arch):
    V = len(smoke_data["vocab"])
    model = build_model(arch, cfg, V, 0, seed=0)
    emb, out = model.embedding.weight, model.out.weight
    assert emb is not out
    assert emb.data_ptr() != out.data_ptr()
    assert emb.untyped_storage().data_ptr() != out.untyped_storage().data_ptr()
    before = emb.detach().clone()
    with torch.no_grad():
        out.add_(1.0)
    assert torch.equal(emb, before), "modifying the output layer changed the input embedding"
    names = [n for n, _ in model.named_parameters()]
    assert "embedding.weight" in names and "out.weight" in names


@pytest.mark.parametrize("arch", ARCHS)
def test_param_split_and_pad_row(cfg, smoke_data, arch):
    V = len(smoke_data["vocab"])
    model = build_model(arch, cfg, V, 0, seed=0)
    p = param_split(model)
    assert p["input_embedding"] == V * 100
    assert p["output_layer"] == 256 * V + V
    assert p["total"] == sum(x.numel() for x in model.parameters()) == p["trainable"]
    assert p["frozen"] == 0
    assert torch.all(model.embedding.weight[0] == 0)
    assert abs(float(model.embedding.weight[3:].std()) - 0.1) < 0.005
