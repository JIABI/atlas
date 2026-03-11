from __future__ import annotations

from pathlib import Path
import json

from ..atlas.surrogate import fit_from_atlas


def main() -> None:
    report = fit_from_atlas(Path("outputs/atlas/atlas.parquet"), Path("outputs/atlas/surrogate"))
    Path("outputs/atlas/surrogate/report.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    main()
