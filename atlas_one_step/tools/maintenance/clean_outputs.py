from __future__ import annotations

from pathlib import Path
import shutil


def main() -> None:
    for d in ["outputs", "logs", "paper_artifacts"]:
        p = Path(d)
        if p.exists():
            shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
    print("cleaned")


if __name__ == "__main__":
    main()
