#!/usr/bin/env python3
"""Export the built-in doctrine registry to config/doctrine_packs.yaml."""

from __future__ import annotations

from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aac.doctrine.doctrine_engine import DOCTRINE_PACKS
from aac.doctrine.pack_registry import serialize_doctrine_packs


def main() -> int:
    output_path = PROJECT_ROOT / "config" / "doctrine_packs.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = serialize_doctrine_packs(DOCTRINE_PACKS)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

    print(f"Exported doctrine packs to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())