"""Tests 8 (frozen_embeddings) and 10 (part2_configs_match)."""
import copy

import numpy as np
import pytest
import torch

from src.embeddings import build_matrix
from src.models import build_model
from src.pipeline import VARIANTS, variant_cfg, variant_lr
from src.train import make_optimizer, make_scheduler, train_epoch, train_run
from src.utils import Logger


@pytest.fixture(scope="module")
def glove_vectors(cfg):
    from src.embeddings import load_glove

    vecs, info = load_glove(cfg, Logger(None, echo=False))
    return vecs


@pytest.fixture(scope="module")
def glove_matrix(glove_vectors, smoke_data):
    v = smoke_data["vocab"]
    return build_matrix(v, glove_vectors, 100, 0, smoke_data["streams"]["train"])


def _flatten(d, prefix=""):
    out = {}
    for k, val in d.items():
        key = f"{prefix}{k}"
        if isinstance(val, dict):
            out.update(_flatten(val, key + "."))
        else:
            out[key] = val
    return out


@pytest.mark.parametrize("kind", ["own_frozen", "glove_frozen"])
def test_frozen_embeddings(cfg, smoke_data, glove_matrix, kind, cpu):
    vocab = smoke_data["vocab"]
    V = len(vocab)
    if kind == "glove_frozen":
        M, stats = glove_matrix
    else:  # any fixed matrix exercises the same freezing path as the own word2vec table
        M = torch.randn(V, 100, generator=torch.Generator().manual_seed(0))
        M[0] = 0
    vcfg = variant_cfg(cfg, kind)
    model = build_model("lstm", vcfg, V, 0, seed=0, pretrained=M)
    emb = model.embedding.weight
    assert emb.requires_grad is False
    assert torch.equal(emb, M.float()) and torch.all(emb[0] == 0)
    opt = make_optimizer(model, vcfg, 1e-3)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(emb) not in opt_ids, "embedding parameters are in the optimizer"
    assert opt_ids == {id(p) for n, p in model.named_parameters() if not n.startswith("embedding.")}
    before = emb.detach().clone()
    other_before = model.out.weight.detach().clone()
    x, y = smoke_data["chunks"]["train"]
    c = copy.deepcopy(vcfg)
    c["train"]["batch_size"] = c["train"]["micro_batch"] = 8
    train_epoch(model, opt, make_scheduler(opt, "lstm", c), torch.amp.GradScaler("cuda", enabled=False),
                x[:24], y[:24], torch.arange(24), c, cpu, 0, False)
    assert torch.equal(model.embedding.weight, before), "frozen embedding table changed during training"
    assert model.embedding.weight.grad is None
    assert not torch.equal(model.out.weight, other_before), "the rest of the model did not train"


def test_glove_rows_equal_loaded_vectors(glove_vectors, glove_matrix, smoke_data, cfg):
    vocab = smoke_data["vocab"]
    M, stats = glove_matrix
    model = build_model("lstm", variant_cfg(cfg, "glove_frozen"), len(vocab), 0, seed=0, pretrained=M)
    for w in ["the", "city", "music", "war", "species"]:
        i = vocab.stoi[w]
        assert np.array_equal(model.embedding.weight[i].numpy(), glove_vectors[w]), w
    # missing-word rules
    assert torch.all(M[vocab.pad_id] == 0)
    assert not stats["unk_covered"] and stats["unk_init"] == "mean_of_covered"
    covered = torch.tensor([vocab.itos[i] in glove_vectors for i in range(len(vocab))])
    covered[0] = False
    assert torch.allclose(M[vocab.unk_id], M[covered].mean(0), atol=1e-5)
    assert stats["types_covered"] == int(covered.sum())
    assert 0.5 < stats["type_coverage"] < 1.0 and 0.5 < stats["token_coverage"] <= 1.0
    # random rows for missing words are reproducible (seeded) and roughly sigma-scaled
    M2, _ = build_matrix(vocab, glove_vectors, 100, 0, smoke_data["streams"]["train"])
    assert torch.equal(M, M2)
    miss = ~covered
    miss[0] = miss[vocab.unk_id] = False
    assert abs(float(M[miss].std()) - stats["sigma_covered"]) / stats["sigma_covered"] < 0.05


@pytest.mark.parametrize("kind", ["own_frozen", "glove_frozen"])
def test_part2_configs_match(cfg, kind):
    base, vc = _flatten(cfg), _flatten(variant_cfg(cfg, kind))
    diff = {k for k in set(base) | set(vc) if base.get(k) != vc.get(k)}
    assert diff and all(k.startswith("embedding.") for k in diff), diff
    assert variant_lr(cfg, kind) == variant_lr(cfg, "lstm"), "Part II must reuse the Part I LSTM LR"
    assert VARIANTS[kind][1] == "lstm"


@pytest.mark.parametrize("kind", ["own_frozen", "glove_frozen"])
def test_part2_init_matches_fresh_and_not_part1_checkpoint(cfg, tiny_data, glove_matrix, kind, cpu, tmp_path):
    vocab = tiny_data["vocab"]
    V = len(vocab)
    M = torch.randn(V, 100, generator=torch.Generator().manual_seed(5))
    M[0] = 0
    for seed in (0, 1):
        p2 = build_model("lstm", variant_cfg(cfg, kind), V, 0, seed=seed, pretrained=M)
        fresh = build_model("lstm", cfg, V, 0, seed=seed)  # a fresh Part I LSTM with the same seed
        n2 = {n: p for n, p in p2.named_parameters() if not n.startswith("embedding.")}
        nf = {n: p for n, p in fresh.named_parameters() if not n.startswith("embedding.")}
        assert n2.keys() == nf.keys()
        for n in n2:
            assert torch.equal(n2[n], nf[n]), f"non-embedding init differs at {n} (seed {seed})"
    # Part I checkpoint (trained a few steps) differs in every non-embedding tensor from the Part II init
    train_run(cfg, "lstm", 0, 1e-3, tmp_path / "p1", tiny_data, cpu, max_epochs=1, eval_test=False,
              max_steps_per_epoch=3, echo=False)
    ck = torch.load(tmp_path / "p1" / "best.pt", weights_only=False)["model"]
    p2 = build_model("lstm", variant_cfg(cfg, kind), V, 0, seed=0, pretrained=M)
    for n, p in p2.named_parameters():
        if not n.startswith("embedding."):
            assert not torch.equal(p, ck[n]), f"Part II model shares {n} with the Part I checkpoint"


def test_part2_saved_run_configs_match_part1(cfg):
    """If smoke/full runs exist, their saved resolved configs differ from the Part I LSTM only in embedding."""
    from src.utils import ROOT, load_config

    checked = 0
    for rdir in (ROOT / "results_smoke", ROOT / "results"):
        p1 = rdir / "part1" / "lstm" / "seed0" / "config.yaml"
        for kind in ("own_frozen", "glove_frozen"):
            p2 = rdir / "part2" / kind / "seed0" / "config.yaml"
            if p1.exists() and p2.exists():
                a, b = _flatten(load_config(p1)), _flatten(load_config(p2))
                diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
                assert all(k.startswith("embedding.") for k in diff), (p2, diff)
                checked += 1
    if checked == 0:
        pytest.skip("no saved Part I/II runs yet (checked again after the smoke run)")
