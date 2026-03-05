"""AAC Configuration Package — System-wide configuration management."""

import os
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
CRYPTO_DIR = CONFIG_DIR / "crypto"
