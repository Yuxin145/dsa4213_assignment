import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_lm_data, load_tokenized  # noqa: E402
from src.utils import load_config  # noqa: E402

torch.set_num_threads(max(1, torch.get_num_threads()))


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c["train"]["lr"] = {"rnn": 1e-3, "lstm": 1e-3, "transformer": 1e-3}  # tests do not depend on the sweep
    return c


@pytest.fixture(scope="session")
def tokenized(cfg):
    return load_tokenized(cfg)


@pytest.fixture(scope="session")
def smoke_data(cfg, tmp_path_factory):
    """Real vocab (train only) + first 1% of each split, built in a temp dir (never touches results/)."""
    d = tmp_path_factory.mktemp("smoke_results")
    return load_lm_data(cfg, d, smoke=True)


@pytest.fixture
def cpu():
    return torch.device("cpu")


def cfg_copy(cfg, **train_overrides):
    c = copy.deepcopy(cfg)
    c["train"].update(train_overrides)
    return c


@pytest.fixture(scope="session")
def tiny_data(smoke_data, cfg):
    """Same pipeline/code path as the smoke data but with the vocab truncated to the 2000 most frequent
    training types (others -> <unk>) so that CPU training-loop tests run in seconds."""
    from src.data import Vocab, make_chunks

    full = smoke_data["vocab"]
    K = 2000
    v = Vocab(full.itos[:K])
    remap = lambda s: torch.where(s < K, s, torch.full_like(s, v.unk_id))
    streams = {k: remap(s) for k, s in smoke_data["streams"].items()}
    L = cfg["data"]["context_length"]
    chunks = {k: make_chunks(s, L, v.pad_id) for k, s in streams.items()}
    return {"vocab": v, "streams": streams, "chunks": chunks, "lines": smoke_data["lines"]}
