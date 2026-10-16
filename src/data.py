"""WikiText-2 loading, preprocessing, vocabulary and the common chunked evaluation setup."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from .utils import load_json, save_json

PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"
SPLITS = ("train", "validation", "test")


# ----------------------------------------------------------------------------- raw text
def load_raw_lines(cfg: dict) -> dict[str, list[str]]:
    """Official splits, raw lines (no filtering)."""
    from datasets import load_dataset

    ds = load_dataset(cfg["data"]["hf_name"], cfg["data"]["hf_config"])
    return {s: list(ds[s]["text"]) for s in SPLITS}


def preprocess_lines(lines: list[str], lowercase: bool = True) -> list[list[str]]:
    """Lowercase, whitespace split, drop empty lines, append <eos> to each non-empty line.
    The dataset's own '<unk>' token is kept as the literal '<unk>' (our single UNK)."""
    out = []
    for line in lines:
        if lowercase:
            line = line.lower()
        toks = line.split()
        if not toks:
            continue
        out.append(toks + [EOS])
    return out


def load_tokenized(cfg: dict) -> dict[str, list[list[str]]]:
    raw = load_raw_lines(cfg)
    return {s: preprocess_lines(raw[s], cfg["data"]["lowercase"]) for s in SPLITS}


# ----------------------------------------------------------------------------- vocab
class Vocab:
    def __init__(self, itos: list[str]):
        assert itos[0] == PAD, "<pad> must be id 0"
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        assert len(self.stoi) == len(self.itos), "duplicate vocab entries"
        self.pad_id, self.unk_id, self.eos_id = self.stoi[PAD], self.stoi[UNK], self.stoi[EOS]

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def sha256(self) -> str:
        return vocab_hash(self.itos)

    def encode(self, toks: list[str]) -> list[int]:
        unk = self.unk_id
        return [self.stoi.get(t, unk) for t in toks]

    def decode(self, ids) -> list[str]:
        return [self.itos[int(i)] for i in ids]

    @classmethod
    def build(cls, train_lines: list[list[str]], specials=(PAD, UNK, EOS)) -> "Vocab":
        """All types of the TRAINING split only (no min_freq). Specials first, then frequency desc,
        ties broken alphabetically (deterministic)."""
        counts = Counter(t for line in train_lines for t in line)
        rest = sorted((w for w in counts if w not in specials), key=lambda w: (-counts[w], w))
        return cls(list(specials) + rest)

    def save(self, path: Path) -> None:
        save_json({"itos": self.itos, "sha256": self.sha256, "size": len(self)}, path)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        d = load_json(path)
        v = cls(d["itos"])
        if v.sha256 != d["sha256"]:
            raise ValueError(f"vocab hash mismatch in {path}: file says {d['sha256']}, got {v.sha256}")
        return v


def vocab_hash(itos: list[str]) -> str:
    return hashlib.sha256(json.dumps(itos, ensure_ascii=False).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- streams/chunks
def lines_to_stream(lines: list[list[str]], vocab: Vocab) -> torch.Tensor:
    ids = []
    for line in lines:
        ids.extend(vocab.encode(line))
    return torch.tensor(ids, dtype=torch.long)


def make_chunks(stream: torch.Tensor, L: int, pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-overlapping chunks: inputs t[i:i+L], targets t[i+1:i+L+1] for i = 0, L, 2L, ...
    The last partial chunk is padded with pad_id (targets too, excluded via ignore_index).
    Every token except the first is a target exactly once."""
    n_targets = stream.numel() - 1
    n_chunks = (n_targets + L - 1) // L
    inputs = torch.full((n_chunks, L), pad_id, dtype=torch.long)
    targets = torch.full((n_chunks, L), pad_id, dtype=torch.long)
    for c in range(n_chunks):
        i = c * L
        x = stream[i:min(i + L, n_targets)]
        y = stream[i + 1:min(i + L + 1, n_targets + 1)]
        inputs[c, :x.numel()] = x
        targets[c, :y.numel()] = y
    return inputs, targets


def load_lm_data(cfg: dict, results_dir: Path, smoke: bool = False) -> dict:
    """Returns {'vocab', 'streams', 'chunks'} for the three splits. Builds and saves the vocab
    (from train only) and data_stats.json if missing; otherwise loads the saved vocab."""
    tok = load_tokenized(cfg)
    vocab_path = results_dir / "vocab.json"
    if vocab_path.exists():
        vocab = Vocab.load(vocab_path)
        if vocab.itos != Vocab.build(tok["train"]).itos:
            raise ValueError(f"{vocab_path} does not match a vocab rebuilt from the training split")
    else:
        vocab = Vocab.build(tok["train"])
        vocab.save(vocab_path)
    streams = {s: lines_to_stream(tok[s], vocab) for s in SPLITS}
    stats_path = results_dir / "data_stats.json"
    if not stats_path.exists():
        stats = {"vocab_size": len(vocab), "vocab_sha256": vocab.sha256, "splits": {}}
        for s in SPLITS:
            st = streams[s]
            stats["splits"][s] = {
                "lines": len(tok[s]),
                "tokens": int(st.numel()),
                "unk_tokens": int((st == vocab.unk_id).sum()),
                "unk_rate": float((st == vocab.unk_id).float().mean()),
                "targets": int(st.numel() - 1),
            }
        save_json(stats, stats_path)
    if smoke:
        frac = cfg["data"]["smoke_fraction"]
        streams = {s: st[: max(2 * cfg["data"]["context_length"], int(st.numel() * frac))]
                   for s, st in streams.items()}
    L = cfg["data"]["context_length"]
    chunks = {s: make_chunks(streams[s], L, vocab.pad_id) for s in SPLITS}
    return {"vocab": vocab, "streams": streams, "chunks": chunks, "lines": tok}
