"""Shared utilities: seeding, logging, hashing, run bookkeeping."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("azsent")


def setup_logging(logfile: str | Path | None = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_uid(text: str, extra: str = "") -> str:
    """Stable 16-hex uid from normalized text (+extra discriminator)."""
    return hashlib.sha1((text + "\x1f" + extra).encode("utf-8")).hexdigest()[:16]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mark_done(run_dir: str | Path, payload: dict | None = None) -> None:
    write_json(Path(run_dir) / "DONE.json", {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), **(payload or {})})


def is_done(run_dir: str | Path) -> bool:
    return (Path(run_dir) / "DONE.json").exists()


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": sys.platform}
    for mod in ("torch", "transformers", "numpy", "pandas", "sklearn", "scipy"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0
