"""RNN / LSTM / Transformer language models with the shared interface (§6.2).

forward(tokens[B,T]) -> logits[B,T,V];  encode(tokens[B,T]) -> hidden[B,T,256] (last layer, no dropout
in eval mode). Construction order is core -> output layer -> embedding so that, for a given seed, the
non-embedding initialisation does not depend on the embedding variant (Part II controlled comparison).
"""
from __future__ import annotations

import torch
import torch.nn as nn

ARCHS = ("rnn", "lstm", "transformer")


def make_embedding(V: int, emb_dim: int, pad_id: int, init_std: float,
                   pretrained: torch.Tensor | None, freeze: bool) -> nn.Embedding:
    if pretrained is not None:
        assert pretrained.shape == (V, emb_dim), f"pretrained shape {tuple(pretrained.shape)} != {(V, emb_dim)}"
        if not freeze:
            raise ValueError("pretrained embeddings are only used frozen in this project")
        return nn.Embedding.from_pretrained(pretrained.float().clone(), freeze=True, padding_idx=pad_id)
    if freeze:
        raise ValueError("frozen embedding requires a pretrained matrix")
    emb = nn.Embedding(V, emb_dim, padding_idx=pad_id)
    with torch.no_grad():
        emb.weight.normal_(0.0, init_std)
        emb.weight[pad_id].zero_()
    return emb


class RecurrentLM(nn.Module):
    def __init__(self, cell: str, V: int, emb_dim: int, hidden: int, layers: int, dropout: float,
                 pad_id: int, init_std: float, pretrained=None, freeze=False):
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM}[cell]
        kw = {"nonlinearity": "tanh"} if cell == "rnn" else {}
        self.core = rnn_cls(emb_dim, hidden, num_layers=layers, dropout=dropout, batch_first=True, **kw)
        self.out = nn.Linear(hidden, V)
        self.embedding = make_embedding(V, emb_dim, pad_id, init_std, pretrained, freeze)
        self.drop = nn.Dropout(dropout)
        self.arch = cell

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        h, _ = self.core(self.drop(self.embedding(tokens)))
        return h

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.out(self.drop(self.encode(tokens)))


class TransformerLM(nn.Module):
    def __init__(self, V: int, emb_dim: int, hidden: int, layers: int, dropout: float, pad_id: int,
                 init_std: float, nhead: int, dim_feedforward: int, max_len: int,
                 pretrained=None, freeze=False):
        super().__init__()
        self.proj = nn.Linear(emb_dim, hidden)
        self.pos = nn.Embedding(max_len, hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.core = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, V)
        self.embedding = make_embedding(V, emb_dim, pad_id, init_std, pretrained, freeze)
        self.drop = nn.Dropout(dropout)
        self.max_len = max_len
        self.arch = "transformer"
        self.register_buffer("causal_mask",
                             torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1),
                             persistent=False)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} > max_len {self.max_len}")
        pos = torch.arange(T, device=tokens.device)
        x = self.drop(self.proj(self.embedding(tokens)) + self.pos(pos)[None])
        x = self.core(x, mask=self.causal_mask[:T, :T], is_causal=True)
        return self.ln_f(x)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.out(self.drop(self.encode(tokens)))


def build_model(arch: str, cfg: dict, V: int, pad_id: int, seed: int,
                pretrained: torch.Tensor | None = None) -> nn.Module:
    """Seeds the global RNG and builds the model. Callers re-seed afterwards for training RNG."""
    m = cfg["model"]
    freeze = bool(cfg["embedding"]["freeze"])
    torch.manual_seed(seed)
    if arch in ("rnn", "lstm"):
        return RecurrentLM(arch, V, m["emb_dim"], m["hidden"], m["layers"], m["dropout"], pad_id,
                           m["emb_init_std"], pretrained, freeze)
    if arch == "transformer":
        t = m["transformer"]
        return TransformerLM(V, m["emb_dim"], m["hidden"], m["layers"], m["dropout"], pad_id,
                             m["emb_init_std"], t["nhead"], t["dim_feedforward"],
                             cfg["data"]["context_length"], pretrained, freeze)
    raise ValueError(f"unknown arch {arch}")


def param_split(model: nn.Module) -> dict:
    """Parameter counts: input embedding / core / output layer / total; trainable vs frozen."""
    groups = {"input_embedding": 0, "core": 0, "output_layer": 0}
    trainable = frozen = 0
    for name, p in model.named_parameters():
        n = p.numel()
        if name.startswith("embedding."):
            groups["input_embedding"] += n
        elif name.startswith("out."):
            groups["output_layer"] += n
        else:
            groups["core"] += n  # recurrent/attention layers, input projection, positions, final LN
        if p.requires_grad:
            trainable += n
        else:
            frozen += n
    groups["total"] = sum(groups.values())
    groups["trainable"] = trainable
    groups["frozen"] = frozen
    return groups
