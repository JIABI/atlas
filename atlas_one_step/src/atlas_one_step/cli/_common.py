from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from ..utils.simple_config import compose_config


def load_cfg(argv: list[str] | None = None) -> SimpleNamespace:
    args = argv if argv is not None else sys.argv[1:]
    repo_root = Path(__file__).resolve().parents[3]
    return compose_config(repo_root, args)
