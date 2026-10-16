"""Part III: AG News topic classification from frozen LM representations."""
from __future__ import annotations

import html
import re

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Vocab

LABELS = ["World", "Sports", "Business", "Sci/Tech"]


# ----------------------------------------------------------------------------- data
def load_ag_news(cfg: dict) -> dict:
    """Stratified 10k sample of the official train split -> 8k train / 2k val (seed 0); full official test."""
    from datasets import load_dataset

    p = cfg["part3"]
    ds = load_dataset(p["hf_name"])
    tr_text, tr_y = list(ds["train"]["text"]), np.asarray(ds["train"]["label"])
    te_text, te_y = list(ds["test"]["text"]), np.asarray(ds["test"]["label"])
    idx = np.arange(len(tr_text))
    sample, _ = train_test_split(idx, train_size=p["n_sample"], stratify=tr_y, random_state=p["split_seed"])
    tr_idx, va_idx = train_test_split(sample, train_size=p["n_train"], test_size=p["n_val"],
                                      stratify=tr_y[sample], random_state=p["split_seed"])
    return {
        "train": {"text": [tr_text[i] for i in tr_idx], "y": tr_y[tr_idx], "orig_idx": tr_idx.tolist()},
        "val": {"text": [tr_text[i] for i in va_idx], "y": tr_y[va_idx], "orig_idx": va_idx.tolist()},
        "test": {"text": te_text, "y": te_y, "orig_idx": list(range(len(te_text)))},
    }


# ----------------------------------------------------------------------------- tokenization
_LEAD = set("\"'([{$#`")
_TRAIL = set(".,;:!?)]}\"'%`")
_CONTRACTION = re.compile(r"^(.+?)('s|'t|'re|'m|'ve|'ll|'d)$")


def baseline_tokens(text: str) -> list[str]:
    """'Before conversion' tokenization: lowercase + whitespace split only."""
    return text.lower().split()


def clean_text(text: str) -> str:
    t = text
    for _ in range(2):  # AG News contains doubly escaped entities, e.g. '&amp;lt;b&amp;gt;'
        t = html.unescape(t)
    t = re.sub(r"#(\d+);", lambda m: chr(int(m.group(1))), t)  # broken entities such as ' #39;s'
    t = re.sub(r"\b(quot|amp|lt|gt);", lambda m: {"quot": '"', "amp": "&", "lt": "<", "gt": ">"}[m.group(1)], t)
    t = t.replace("\\", " ")
    t = re.sub(r"<[^>]{0,200}>", " ", t)  # HTML tags
    return t


def convert_tokens(text: str, vocab_set: set[str]) -> list[str]:
    """Convert raw AG News text to WikiText-2 conventions (lowercase, split punctuation, @-@/@,@/@.@)."""
    t = clean_text(text).lower()
    t = re.sub(r"(?<=\d),(?=\d)", " @,@ ", t)
    t = re.sub(r"(?<=\d)\.(?=\d)", " @.@ ", t)
    t = re.sub(r"(?<=[a-z0-9])-(?=[a-z0-9])", " @-@ ", t)
    t = t.replace("", " — ").replace("", " – ")  # cp1252 dashes -> WikiText's dash tokens
    t = re.sub(r"(?<=[a-z0-9])&(?=[a-z0-9])", " & ", t)
    t = t.replace("--", " -- ").replace("/", " / ")
    out = []
    for raw in t.split():
        out.extend(_split_token(raw, vocab_set))
    return out


def _split_token(w: str, vocab_set: set[str]) -> list[str]:
    if w in vocab_set:
        return [w]
    pre, post = [], []
    while len(w) > 1 and w[0] in _LEAD and w not in vocab_set:
        pre.append(w[0])
        w = w[1:]
    while len(w) > 1 and w[-1] in _TRAIL and w not in vocab_set:
        post.append(w[-1])
        w = w[:-1]
    mid = [w]
    if w not in vocab_set:
        m = _CONTRACTION.match(w)
        if m:
            mid = [m.group(1), m.group(2)]
        elif w.endswith("s'") and len(w) > 2:
            mid = [w[:-1], "'"]
    return pre + mid + post[::-1]


def oov_rate(token_lists: list[list[str]], vocab_set: set[str]) -> dict:
    n = sum(len(t) for t in token_lists)
    oov = sum(1 for toks in token_lists for x in toks if x not in vocab_set)
    return {"tokens": n, "oov_tokens": oov, "oov_rate": oov / max(n, 1)}


def encode_texts(token_lists: list[list[str]], vocab: Vocab, max_len: int) -> list[list[int]]:
    out = []
    for toks in token_lists:
        ids = vocab.encode(toks[:max_len])
        if not ids:
            ids = [vocab.unk_id]
        out.append(ids)
    return out


# ----------------------------------------------------------------------------- features
def pad_batch(seqs: list[list[int]], pad_id: int) -> torch.Tensor:
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return x


def pool(hidden: torch.Tensor, mask: torch.Tensor, how: str) -> torch.Tensor:
    """hidden [B,T,H], mask [B,T] (True = real token). 'mean' over non-pad positions, or 'last' non-pad."""
    if how == "mean":
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)
    if how == "last":
        last = mask.long().sum(1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last]
    raise ValueError(how)


@torch.no_grad()
def extract_features(model, seqs: list[list[int]], pad_id: int, device: torch.device,
                     poolings=("mean",), batch_size: int = 64) -> dict[str, np.ndarray]:
    """Frozen LM (eval, no_grad, fp32) -> pooled last-layer encode() states."""
    model.eval()
    feats = {p: [] for p in poolings}
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))  # length-bucketing for speed
    for s in range(0, len(order), batch_size):
        ids = order[s:s + batch_size]
        x = pad_batch([seqs[i] for i in ids], pad_id).to(device)
        h = model.encode(x).float()
        mask = x != pad_id
        for p in poolings:
            feats[p].append((ids, pool(h, mask, p).cpu().numpy()))
    out = {}
    for p in poolings:
        arr = np.zeros((len(seqs), feats[p][0][1].shape[1]), dtype=np.float32)
        for ids, f in feats[p]:
            arr[ids] = f
        out[p] = arr
    return out


def glove_mean_features(seqs: list[list[int]], matrix: np.ndarray, pad_id: int) -> np.ndarray:
    out = np.zeros((len(seqs), matrix.shape[1]), dtype=np.float32)
    for i, s in enumerate(seqs):
        ids = [t for t in s if t != pad_id]
        out[i] = matrix[ids].mean(0)
    return out


# ----------------------------------------------------------------------------- classifier
def select_C(Xtr, ytr, Xva, yva, C_grid, max_iter: int) -> dict:
    """C chosen on VALIDATION accuracy (ties -> smaller C). Never receives test data."""
    scores = []
    for C in C_grid:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter))
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)
        scores.append({"C": C, "val_acc": float(accuracy_score(yva, pred)),
                       "val_macro_f1": float(f1_score(yva, pred, average="macro"))})
    best = max(scores, key=lambda r: (r["val_acc"], -r["C"]))
    return {"best_C": best["C"], "grid": scores}


def fit_and_test(Xtr, ytr, Xva, yva, Xte, yte, C_grid, max_iter: int) -> dict:
    sel = select_C(Xtr, ytr, Xva, yva, C_grid, max_iter)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=sel["best_C"], max_iter=max_iter))
    clf.fit(Xtr, ytr)  # fit on train only
    pred = clf.predict(Xte)  # test evaluated once, with the selected C
    return {
        "selection": sel,
        "test_acc": float(accuracy_score(yte, pred)),
        "test_macro_f1": float(f1_score(yte, pred, average="macro")),
        "confusion_matrix": confusion_matrix(yte, pred, labels=list(range(len(LABELS)))).tolist(),
        "pred": pred,
    }
