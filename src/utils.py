"""Shared helpers: config, seeding, device, run manifests, JSON/text IO, logging."""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "base.yaml"
EXPERIMENT_SOURCE_PATHS = ("src", "scripts", "configs", "run_all.py", "requirements.txt")


# ----------------------------------------------------------------------------- config
def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_hash(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def save_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def results_root(cfg: dict, smoke: bool) -> Path:
    return ROOT / (cfg["paths"]["results_smoke"] if smoke else cfg["paths"]["results"])


# ----------------------------------------------------------------------------- IO
def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Logger:
    """Appends timestamped lines to a log file and echoes them to stdout."""

    def __init__(self, path: Path | None, echo: bool = True):
        self.path = Path(path) if path else None
        self.echo = echo
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[{now_str()}] {msg}"
        if self.echo:
            print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


# ----------------------------------------------------------------------------- env
def now_str() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hardware_string(device: torch.device | None = None) -> str:
    cpu = platform.processor() or platform.machine()
    s = f"{platform.system()} {platform.release()}; CPU: {cpu}; torch {torch.__version__}"
    if device is not None and device.type == "cuda":
        s += f"; GPU: {torch.cuda.get_device_name(device)}"
    elif torch.cuda.is_available():
        s += f"; GPU available: {torch.cuda.get_device_name(0)} (not used)"
    return s


def require_clean_source() -> str:
    """Return HEAD, or refuse a real-result run whose experiment source is not exactly committed."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=10
    )
    commit = head.stdout.strip() if head.returncode == 0 else ""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *EXPERIMENT_SOURCE_PATHS],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=10,
    )
    dirty = status.stdout.strip() if status.returncode == 0 else "unable to inspect source status"
    if not commit or dirty:
        detail = dirty or "repository has no readable HEAD commit"
        raise RuntimeError(
            "Real-result runs require clean committed experiment source so manifests identify exact code. "
            f"Commit or revert the source changes first. Git status: {detail}"
        )
    return commit


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, encoding="utf-8", timeout=10)
        commit = out.stdout.strip() if out.returncode == 0 else "no-commit"
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *EXPERIMENT_SOURCE_PATHS],
                               cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                               timeout=10).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception as e:  # git missing is not fatal for the manifest, but is recorded
        return f"unknown ({e})"


def rel_path(p) -> str:
    p = Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def make_manifest(cfg: dict, seed: int | None, log_path: Path | None, start: str,
                  device: torch.device | None = None, extra: dict | None = None) -> dict:
    m = {
        "start": start,
        "end": now_str(),
        "git_commit": git_commit(),
        "config_hash": config_hash(cfg),
        "seed": seed,
        "hardware": hardware_string(device),
        "log_file": rel_path(log_path) if log_path else None,
    }
    if extra:
        m.update(extra)
    return m
