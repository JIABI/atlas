from __future__ import annotations

from pathlib import Path


def main() -> None:
    ckpts = sorted(Path("checkpoints").glob("*.pt"))
    if not ckpts:
        raise SystemExit("No checkpoints found")
    print({"count": len(ckpts), "latest": str(ckpts[-1])})


if __name__ == "__main__":
    main()
