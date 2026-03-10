from __future__ import annotations

from pathlib import Path
import json


def main() -> None:
    p = Path("outputs/atlas/atlas.jsonl")
    if not p.exists():
        raise SystemExit("atlas.jsonl missing")
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    Path("outputs/atlas/atlas_compact.json").write_text(json.dumps(rows[:50], indent=2))
    print({"rows": len(rows), "saved": "outputs/atlas/atlas_compact.json"})


if __name__ == "__main__":
    main()
