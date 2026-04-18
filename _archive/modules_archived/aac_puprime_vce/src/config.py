"""Shared configuration loader for aac_puprime_vce."""

from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load(name: str) -> dict[str, Any]:
    path = _CONFIG_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_instruments() -> dict[str, Any]:
    """Load instruments."""
    return _load("instruments")["instruments"]


def load_strategy() -> dict[str, Any]:
    """Load strategy."""
    return _load("strategy")["strategy"]


def load_risk() -> dict[str, Any]:
    """Load risk."""
    return _load("risk")["risk"]


def load_costs() -> dict[str, Any]:
    """Load costs."""
    return _load("costs")["costs"]
