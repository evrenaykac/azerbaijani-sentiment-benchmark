"""Configuration loading with environment overrides."""
from __future__ import annotations

import os
import copy
from pathlib import Path

import yaml

_PKG_ROOT = Path(__file__).resolve().parents[2]   # the azsent/ folder that contains configs/


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


class Config(dict):
    """Dict with attribute access, nested."""

    def __getattr__(self, item):
        try:
            v = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        if isinstance(v, dict) and not isinstance(v, Config):
            v = Config(v)
            self[item] = v
        return v


def load_config(overrides: dict | None = None, config_path: str | None = None) -> Config:
    path = Path(config_path) if config_path else _PKG_ROOT / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        _deep_update(cfg, overrides)

    # resolve paths
    data_dir = cfg["paths"].get("data_dir") or os.environ.get("AZSENT_DATA")
    if not data_dir:
        # default: the data/ folder next to the azsent package folder
        data_dir = str(_PKG_ROOT.parent / "data")
    runs_dir = cfg["paths"].get("runs_dir") or os.environ.get("AZSENT_RUNS")
    if not runs_dir:
        runs_dir = r"C:\azsent_runs" if os.name == "nt" else str(Path.home() / "azsent_runs")
    cfg["paths"]["data_dir"] = str(Path(data_dir))
    cfg["paths"]["runs_dir"] = str(Path(runs_dir))
    # AZSENT_DATASET lets the smoke run (and any one-off) point the whole
    # pipeline at a different export without editing configs/default.yaml -
    # editing the YAML by hand and forgetting to change it back is exactly how
    # a "full" run silently trained on the smoke subset.
    ds_env = os.environ.get("AZSENT_DATASET")
    if ds_env:
        cfg["paths"]["dataset_name"] = ds_env
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    return Config(copy.deepcopy(cfg))


def load_systems() -> Config:
    with open(_PKG_ROOT / "configs" / "systems.yaml", "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))


def prep_dir(cfg: Config) -> Path:
    p = Path(cfg.paths.runs_dir) / "prepared"
    p.mkdir(parents=True, exist_ok=True)
    return p


def runs_root(cfg: Config) -> Path:
    p = Path(cfg.paths.runs_dir) / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p
