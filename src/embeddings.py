"""Part II embedding tables: own Word2Vec (skip-gram) and public GloVe 6B-100d, aligned to our vocab (§7).

Missing words: <pad> = 0; <unk> (if missing) = mean of covered vectors; other missing words ~ N(0, sigma^2)
with sigma = std of all covered vector entries, drawn from a fixed seed so the frozen table is identical
for every LM seed.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .data import PAD, UNK, Vocab
from .utils import Logger, save_json, sha256_file


def train_word2vec(train_lines: list[list[str]], cfg: dict, out_path: Path, log: Logger) -> dict:
    """Skip-gram on the TRAINING split only; sentences = preprocessed lines (incl. <eos>)."""
    from gensim.models import Word2Vec

    w = cfg["word2vec"]
    t0 = time.perf_counter()
    model = Word2Vec(sentences=train_lines, vector_size=w["vector_size"], window=w["window"],
                     min_count=w["min_count"], negative=w["negative"], epochs=w["epochs"], seed=w["seed"],
                     workers=w["workers"], sg=w["sg"])
    secs = time.perf_counter() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.wv.save(str(out_path))
    log(f"word2vec trained in {secs:.1f}s on {sum(len(l) for l in train_lines)} tokens; "
        f"{len(model.wv)} vectors -> {out_path}")
    return {"train_time_s": secs, "n_vectors": len(model.wv), "path": str(out_path),
            "sha256": sha256_file(out_path), "params": w,
            "n_sentences": len(train_lines), "n_tokens": sum(len(l) for l in train_lines)}


def load_word2vec(path: Path) -> dict[str, np.ndarray]:
    from gensim.models import KeyedVectors

    kv = KeyedVectors.load(str(path))
    return {w: kv[w] for w in kv.index_to_key}


def load_glove(cfg: dict, log: Logger) -> tuple[dict[str, np.ndarray], dict]:
    import gensim.downloader as api

    from gensim.models import KeyedVectors

    from .utils import ROOT

    name = cfg["glove"]["gensim_name"]
    path = api.load(name, return_path=True)
    cache = ROOT / ".cache" / f"{name}.kv"  # native-format copy of the same vectors (fast to reload)
    if cache.exists():
        kv = KeyedVectors.load(str(cache))
    else:
        kv = api.load(name)
        cache.parent.mkdir(parents=True, exist_ok=True)
        kv.save(str(cache))
    log(f"GloVe loaded via gensim.downloader '{name}' from {path}: {len(kv)} vectors, dim {kv.vector_size}")
    if kv.vector_size != cfg["model"]["emb_dim"]:
        raise ValueError(f"GloVe dim {kv.vector_size} != emb_dim {cfg['model']['emb_dim']}")
    return ({w: kv[w] for w in kv.index_to_key},
            {"source": f"gensim.downloader:{name}", "path": str(path), "n_vectors": len(kv)})


def build_matrix(vocab: Vocab, vectors: dict[str, np.ndarray], dim: int, seed: int,
                 train_stream: torch.Tensor) -> tuple[torch.Tensor, dict]:
    V = len(vocab)
    M = np.zeros((V, dim), dtype=np.float32)
    covered = np.zeros(V, dtype=bool)
    for i, w in enumerate(vocab.itos):
        if w == PAD:
            continue
        v = vectors.get(w)
        if v is not None:
            M[i] = v
            covered[i] = True
    if not covered.any():
        raise RuntimeError("no vocabulary word is covered by the embedding source")
    cov_vecs = M[covered]
    mean, sigma = cov_vecs.mean(0), float(cov_vecs.std())
    rng = np.random.default_rng(seed)
    missing = [i for i in range(V) if not covered[i] and i != vocab.pad_id]
    unk_missing = not covered[vocab.unk_id]
    n_random = 0
    for i in missing:
        if i == vocab.unk_id:
            M[i] = mean
        else:
            M[i] = rng.normal(0.0, sigma, size=dim).astype(np.float32)
            n_random += 1
    M[vocab.pad_id] = 0.0
    # coverage over non-pad types, and over training tokens
    counts = torch.bincount(train_stream, minlength=V).numpy()
    nonpad = np.ones(V, dtype=bool)
    nonpad[vocab.pad_id] = False
    stats = {
        "types_total": int(nonpad.sum()),
        "types_covered": int(covered.sum()),
        "type_coverage": float(covered.sum() / nonpad.sum()),
        "train_tokens_total": int(counts.sum()),
        "train_tokens_covered": int(counts[covered].sum()),
        "token_coverage": float(counts[covered].sum() / counts.sum()),
        "missing_types": len(missing),
        "missing_random_init": n_random,
        "unk_covered": bool(covered[vocab.unk_id]),
        "unk_init": "mean_of_covered" if unk_missing else "source_vector",
        "eos_covered": bool(covered[vocab.eos_id]),
        "sigma_covered": sigma,
        "missing_seed": seed,
        "top_missing_by_train_freq": [(vocab.itos[i], int(counts[i]))
                                      for i in sorted(missing, key=lambda i: -counts[i])[:30]],
    }
    return torch.from_numpy(M), stats


def get_pretrained(kind: str, cfg: dict, vocab: Vocab, train_lines, train_stream, out_dir: Path,
                   log: Logger) -> tuple[torch.Tensor, dict]:
    """Builds (and caches) the frozen table for 'own_frozen' or 'glove_frozen'."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mat_path = out_dir / f"{kind}_matrix.pt"
    info_path = out_dir / f"{kind}_info.json"
    if mat_path.exists() and info_path.exists():
        from .utils import load_json
        info = load_json(info_path)
        if info["vocab_sha256"] != vocab.sha256:
            raise ValueError(f"{mat_path} was built for a different vocab")
        return torch.load(mat_path), info
    dim = cfg["model"]["emb_dim"]
    if kind == "own_frozen":
        w2v_path = out_dir / "word2vec.kv"
        train_info = train_word2vec(train_lines, cfg, w2v_path, log)
        vectors = load_word2vec(w2v_path)
        source = {"source": "own word2vec skip-gram (gensim)", **train_info}
    elif kind == "glove_frozen":
        vectors, source = load_glove(cfg, log)
    else:
        raise ValueError(kind)
    M, stats = build_matrix(vocab, vectors, dim, cfg["glove"]["missing_seed"], train_stream)
    torch.save(M, mat_path)
    info = {"kind": kind, "vocab_sha256": vocab.sha256, "matrix_path": str(mat_path),
            "matrix_sha256": sha256_file(mat_path), "coverage": stats, **source}
    save_json(info, info_path)
    log(f"{kind}: type coverage {stats['type_coverage']:.4f}, token coverage {stats['token_coverage']:.4f}, "
        f"missing {stats['missing_types']} (random {stats['missing_random_init']}, unk_init {stats['unk_init']})")
    return M, info
