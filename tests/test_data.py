"""Test 1 (vocab_train_only) + chunking behaviour of the common evaluation setup."""
from pathlib import Path

import torch

from src.data import EOS, PAD, UNK, Vocab, lines_to_stream, make_chunks, preprocess_lines
from src.utils import ROOT, load_json, load_config


def test_vocab_train_only_synthetic():
    train = preprocess_lines(["The cat sat", "", "  ", "a dog <unk> ran"])
    val = preprocess_lines(["the zebra sat"])
    v = Vocab.build(train)
    assert v.itos[:3] == [PAD, UNK, EOS] and v.pad_id == 0
    assert "zebra" not in v.stoi, "validation-only word leaked into the vocabulary"
    ids = v.encode(val[0])
    assert ids == [v.stoi["the"], v.unk_id, v.stoi["sat"], v.eos_id]
    # lowercase, empty lines dropped, <eos> appended, dataset <unk> == our single <unk>
    assert train == [["the", "cat", "sat", EOS], ["a", "dog", "<unk>", "ran", EOS]]
    assert v.encode(["<unk>"]) == [v.unk_id]


def test_vocab_train_only_real(tokenized):
    v = Vocab.build(tokenized["train"])
    train_types = {t for line in tokenized["train"] for t in line}
    assert set(v.itos) == train_types | {PAD, UNK, EOS}
    for split in ("validation", "test"):
        types = {t for line in tokenized[split] for t in line}
        only = types - train_types
        assert not (only & set(v.itos)), f"{split}-only types in vocab"
        # every such word maps to <unk>
        stream = lines_to_stream(tokenized[split], v)
        flat = [t for line in tokenized[split] for t in line]
        for i, t in enumerate(flat):
            if t in only:
                assert stream[i] == v.unk_id
        assert stream.numel() == len(flat)


def test_vocab_hash_shared_by_all_consumers(smoke_data, tmp_path):
    v = smoke_data["vocab"]
    v.save(tmp_path / "vocab.json")
    assert Vocab.load(tmp_path / "vocab.json").sha256 == v.sha256
    # tampering is detected
    d = load_json(tmp_path / "vocab.json")
    d["itos"][5], d["itos"][6] = d["itos"][6], d["itos"][5]
    import json
    (tmp_path / "bad.json").write_text(json.dumps(d), encoding="utf-8")
    try:
        Vocab.load(tmp_path / "bad.json")
        raise AssertionError("tampered vocab accepted")
    except ValueError:
        pass
    # every saved consumer artifact (if present) records the same hash as results*/vocab.json
    for rdir in (ROOT / "results", ROOT / "results_smoke"):
        if not (rdir / "vocab.json").exists():
            continue
        h = Vocab.load(rdir / "vocab.json").sha256
        assert h == v.sha256
        assert load_json(rdir / "data_stats.json")["vocab_sha256"] == h
        for cfgfile in rdir.rglob("config.yaml"):
            c = load_config(cfgfile)
            assert c["run"]["vocab_sha256"] == h, cfgfile
        for info in (rdir / "part2" / "embeddings").glob("*_info.json"):
            assert load_json(info)["vocab_sha256"] == h, info


def test_make_chunks_targets_each_token_once():
    stream = torch.arange(1, 301)  # 300 tokens, no pad
    x, y = make_chunks(stream, 128, pad_id=0)
    assert x.shape == (3, 128)
    # inputs t[i:i+L], targets t[i+1:i+L+1]
    assert torch.equal(x[0], stream[:128]) and torch.equal(y[0], stream[1:129])
    assert torch.equal(x[1], stream[128:256]) and torch.equal(y[1], stream[129:257])
    real = y[y != 0]
    assert torch.equal(real, stream[1:]), "every token after the first is a target exactly once"
    # padded inputs exactly where targets are padded
    assert torch.equal(x == 0, y == 0)
