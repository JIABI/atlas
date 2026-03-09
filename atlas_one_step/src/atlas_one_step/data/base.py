
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetSpec:
    name: str
    root: Path
    resolution: int
