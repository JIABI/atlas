from __future__ import annotations

from pathlib import Path
import json

from ..selection.select_target import select_from_atlas


def main() -> None:
    selection = select_from_atlas(Path("outputs/atlas/atlas.parquet"), strategy="pathology_min")
    Path("outputs/selection.json").write_text(json.dumps(selection, indent=2))
    print(selection)


if __name__ == "__main__":
    main()
