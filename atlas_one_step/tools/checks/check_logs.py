from __future__ import annotations

from pathlib import Path
import json


def main() -> None:
    log_dir = Path("logs")
    entries = []
    for p in log_dir.glob("*.jsonl"):
        for line in p.read_text().splitlines():
            entries.append(json.loads(line))
    print({"log_files": len(list(log_dir.glob('*.jsonl'))), "entries": len(entries)})


if __name__ == "__main__":
    main()
