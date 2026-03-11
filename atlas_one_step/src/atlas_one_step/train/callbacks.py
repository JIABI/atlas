from __future__ import annotations

"""Training callbacks for structured logging."""

from pathlib import Path
import json


class JsonLoggerCallback:
    """Persist per-epoch metrics to a JSONL file."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        record = {"epoch": epoch, **metrics}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
