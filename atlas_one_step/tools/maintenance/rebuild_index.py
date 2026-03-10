from __future__ import annotations

from pathlib import Path
import json


def main() -> None:
    out = {}
    for folder in ["outputs", "checkpoints", "paper_artifacts"]:
        p = Path(folder)
        out[folder] = [str(x) for x in sorted(p.rglob("*")) if x.is_file()]
    Path("outputs/index.json").parent.mkdir(parents=True, exist_ok=True)
    Path("outputs/index.json").write_text(json.dumps(out, indent=2))
    print("outputs/index.json")


if __name__ == "__main__":
    main()
