"""Test 13 (determinism): two CPU smoke runs with the same seed give identical first 20 losses."""
import pytest

from src.train import train_run

from .conftest import cfg_copy


@pytest.mark.parametrize("arch", ["lstm", "transformer"])
def test_determinism_cpu(cfg, tiny_data, cpu, tmp_path, arch):
    smoke_data = tiny_data
    c = cfg_copy(cfg, patience=100)
    n_steps_per_epoch = (smoke_data["chunks"]["train"][0].shape[0] + 31) // 32
    epochs = -(-20 // n_steps_per_epoch)
    runs = []
    for k in range(2):
        m = train_run(c, arch, 0, 1e-3, tmp_path / f"run{k}", smoke_data, cpu, max_epochs=epochs,
                      eval_test=False, echo=False)
        runs.append(m["first_step_losses"][:20])
    assert len(runs[0]) == 20
    assert runs[0] == runs[1], "same seed gave different losses"
    # a different seed must give a different trajectory (the seed is actually used)
    m = train_run(c, arch, 1, 1e-3, tmp_path / "run_seed1", smoke_data, cpu, max_epochs=1, eval_test=False,
                  echo=False)
    assert m["first_step_losses"][:5] != runs[0][:5]
