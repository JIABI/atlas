from __future__ import annotations

from pathlib import Path
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as wf:
        for src in args.input:
            for line in Path(src).read_text().splitlines():
                if line.strip():
                    wf.write(line + "\n")
    print(out)


if __name__ == "__main__":
    main()
