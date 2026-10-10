"""Test 14 (checkpoint_roundtrip): save -> load -> identical logits; resume continues at the right
epoch/step and reproduces the uninterrupted run exactly (CPU)."""
import pytest
import torch

from src.models import build_model
from src.train import train_run

from .conftest import cfg_copy


@pytest.mark.parametrize("arch", ["lstm", "transformer"])
def test_save_load_identical_logits(cfg, tiny_data, cpu, tmp_path, arch):
    c = cfg_copy(cfg)
    m = train_run(c, arch, 0, 1e-3, tmp_path / "r", tiny_data, cpu, max_epochs=1, eval_test=False,
                  max_steps_per_epoch=2, echo=False)
    assert (tmp_path / "r" / "DONE").exists() and (tmp_path / "r" / "best.pt").exists()
    V = len(tiny_data["vocab"])
    a = build_model(arch, c, V, 0, seed=0)
    a.load_state_dict(torch.load(tmp_path / "r" / "best.pt", weights_only=False)["model"])
    b = build_model(arch, c, V, 0, seed=123)  # different init, then loaded
    b.load_state_dict(torch.load(tmp_path / "r" / "best.pt", weights_only=False)["model"])
    fresh = build_model(arch, c, V, 0, seed=0)
    x = tiny_data["chunks"]["validation"][0][:2]
    a.eval(), b.eval(), fresh.eval()
    with torch.no_grad():
        assert torch.equal(a(x), b(x))
        assert not torch.allclose(a(x), fresh(x)), "checkpoint did not contain trained weights"
    # best.pt's val loss is reproduced by evaluation of the loaded model
    from src.evaluate import evaluate_lm
    r = evaluate_lm(b, *tiny_data["chunks"]["validation"], 0, cpu, amp=False)
    assert r["ce"] == pytest.approx(m["best_val_ce"], rel=1e-6)


@pytest.mark.parametrize("arch", ["lstm", "transformer"])
def test_resume_matches_uninterrupted(cfg, tiny_data, cpu, tmp_path, arch):
    c = cfg_copy(cfg, patience=100)
    kw = dict(max_epochs=3, eval_test=False, max_steps_per_epoch=3, echo=False)
    full = train_run(c, arch, 0, 1e-3, tmp_path / "full", tiny_data, cpu, **kw)
    part = train_run(c, arch, 0, 1e-3, tmp_path / "part", tiny_data, cpu, stop_after_epochs=1, **kw)
    assert part["epochs_run"] == 1 and part["steps"] == 3
    assert not (tmp_path / "part" / "DONE").exists() and (tmp_path / "part" / "last.pt").exists()
    resumed = train_run(c, arch, 0, 1e-3, tmp_path / "part", tiny_data, cpu, **kw)
    assert "RESUME from" in (tmp_path / "part" / "log.txt").read_text(encoding="utf-8")
    assert resumed["epochs_run"] == 3 and resumed["steps"] == 9 == full["steps"]
    assert [h["epoch"] for h in resumed["history"]] == [1, 2, 3]
    for a, b in zip(full["history"], resumed["history"]):
        assert a["global_step"] == b["global_step"]
        assert a["train_loss"] == b["train_loss"], "resumed run diverged from the uninterrupted run"
        assert a["val_loss"] == b["val_loss"]
    # a DONE run is skipped on re-run (no new epochs)
    again = train_run(c, arch, 0, 1e-3, tmp_path / "part", tiny_data, cpu, **kw)
    assert again["history"] == resumed["history"]
