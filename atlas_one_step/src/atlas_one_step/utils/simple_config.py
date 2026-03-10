from __future__ import annotations

"""Lightweight config composition for key=value CLI overrides and simple YAML files."""

from dataclasses import dataclass
from pathlib import Path
import ast
from types import SimpleNamespace
from typing import Any


def _parse_scalar(raw: str) -> Any:
    txt = raw.strip()
    if txt.lower() in {"true", "false"}:
        return txt.lower() == "true"
    try:
        return ast.literal_eval(txt)
    except Exception:
        return txt.strip('"').strip("'")


def parse_yaml_simple(path: Path) -> dict[str, Any]:
    """Parse a constrained YAML subset used by this repository configs."""
    out: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line == "defaults:" or line.startswith("-"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = _parse_scalar(v)
    return out


def parse_defaults(path: Path) -> dict[str, str]:
    groups: dict[str, str] = {}
    lines = path.read_text().splitlines()
    for ln in lines:
        t = ln.strip()
        if t.startswith("- ") and ":" in t and "_self_" not in t:
            x = t[2:]
            g, n = [p.strip() for p in x.split(":", 1)]
            groups[g] = n
    return groups


def _deep_ns(d: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**{k: _deep_ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def compose_config(repo_root: Path, overrides: list[str]) -> SimpleNamespace:
    """Compose config from defaults + group yaml files + key=value overrides."""
    cfg_root = repo_root / "configs"
    base = parse_yaml_simple(cfg_root / "defaults.yaml")
    selected = parse_defaults(cfg_root / "defaults.yaml")
    for ov in overrides:
        if "=" in ov:
            k, v = ov.split("=", 1)
            if "." not in k and (cfg_root / k / f"{v}.yaml").exists():
                selected[k] = v
    cfg: dict[str, Any] = base.copy()
    for grp, name in selected.items():
        cfg[grp] = parse_yaml_simple(cfg_root / grp / f"{name}.yaml")
    for ov in overrides:
        if "=" not in ov:
            continue
        k, v = ov.split("=", 1)
        if "." not in k and (cfg_root / k / f"{v}.yaml").exists():
            continue
        if "." in k:
            p0, p1 = k.split(".", 1)
            cfg.setdefault(p0, {})
            cfg[p0][p1] = _parse_scalar(v)
        else:
            cfg[k] = _parse_scalar(v)
    return _deep_ns(cfg)
