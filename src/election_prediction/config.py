"""Load YAML configs and resolve paths relative to the repo root."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/baseline.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(p)
    return cfg


def repo_path(relative: str | Path) -> Path:
    """Turn a config path like 'data/raw/x.csv' into an absolute path."""
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p


class ConfigError(ValueError):
    pass


def validate_config(cfg: dict) -> dict:
    """
    Catch settings mistakes before anything trains. Collects every problem:
      * predict year also listed in train years (the model would see the answers)
      * a feature listed twice, or in both the baseline and residual lists
      * a residual feature '<base>_chg_<N>y' whose horizon N isn't in feature_horizons
    """
    problems = []
    years = cfg.get("years", {})
    if years.get("predict") in (years.get("train") or []):
        problems.append(f"predict year {years['predict']} is also in train years")

    base = list(cfg.get("baseline", {}).get("features") or [])
    resid = list(cfg.get("residual_model", {}).get("features") or [])
    for name, feats in (("baseline", base), ("residual_model", resid)):
        dupes = sorted({f for f in feats if feats.count(f) > 1})
        if dupes:
            problems.append(f"{name} lists features twice: {dupes}")
    both = sorted(set(base) & set(resid))
    if both:
        problems.append(f"features in both baseline and residual_model: {both}")

    horizons = cfg.get("feature_horizons") or {}
    for f in resid:
        m = re.fullmatch(r"(.+)_chg_(\d+)y", f)
        if m and int(m.group(2)) not in (horizons.get(m.group(1)) or []):
            problems.append(f"{f}: horizon {m.group(2)} not listed under feature_horizons.{m.group(1)}")

    if problems:
        raise ConfigError("; ".join(problems))
    return cfg
