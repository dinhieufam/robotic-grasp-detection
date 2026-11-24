"""Config loading. Relative paths in configs are relative to the repo root."""
from __future__ import annotations

import os

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)
