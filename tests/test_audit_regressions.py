"""Regression tests for Major findings from the independent audit."""
from types import SimpleNamespace

import pytest
import torch


def test_amp_scaler_skip_escalates():
    from src.train import _raise_on_amp_skip, train_epoch

    class FakeScaler:
        def __init__(self, enabled: bool, scale: float):
            self.enabled = enabled
            self._scale = scale

        def is_enabled(self):
            return self.enabled

        def get_scale(self):
            return self._scale

    with pytest.raises(FloatingPointError, match="AMP GradScaler skipped"):
        _raise_on_amp_skip(FakeScaler(True, 512.0), scale_before=1024.0)

    _raise_on_amp_skip(FakeScaler(True, 1024.0), scale_before=1024.0)
    _raise_on_amp_skip(FakeScaler(False, 512.0), scale_before=1024.0)

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.out = torch.nn.Linear(4, 8)

        def forward(self, x):
            return self.out(self.embedding(x))

    class SkippingScaler(FakeScaler):
        def scale(self, loss):
            return loss

        def unscale_(self, opt):
            pass

        def step(self, opt):
            pass

        def update(self):
            self._scale /= 2

    class Scheduler:
        stepped = False

        def step(self):
            self.stepped = True

    model = TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = Scheduler()
    scaler = SkippingScaler(True, 1024.0)
    x = torch.tensor([[1, 2, 3]])
    y = torch.tensor([[2, 3, 4]])
    cfg = {"train": {"batch_size": 1, "micro_batch": 1, "grad_clip": 1.0}}
    with pytest.raises(FloatingPointError, match="AMP GradScaler skipped"):
        train_epoch(
            model, opt, sched, scaler, x, y, torch.tensor([0]), cfg,
            torch.device("cpu"), pad_id=0, amp=False,
        )
    assert not sched.stepped, "LR scheduler advanced after a skipped optimizer step"


def test_real_runs_reject_dirty_experiment_source(monkeypatch):
    from src import utils

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1:3] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if args[1:3] == ["status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout=" M run_all.py\n", stderr="")
        raise AssertionError(args)

    monkeypatch.setattr(utils.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="clean committed experiment source"):
        utils.require_clean_source()

    status_call = next(args for args in calls if args[1:3] == ["status", "--porcelain"])
    assert "run_all.py" in status_call, "entry point was omitted from dirty-source detection"
