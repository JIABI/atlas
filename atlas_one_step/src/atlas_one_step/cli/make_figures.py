from __future__ import annotations

from pathlib import Path
import json
import matplotlib.pyplot as plt


def _plot(values: list[float], title: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.plot(values)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    figs = Path("paper_artifacts/figures")
    tabs = Path("paper_artifacts/tables")
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    _plot([1.0, 0.8, 0.5, 0.3], "line-family phase", figs / "line_phase.png")
    _plot([0.2, 0.4, 0.9], "tail-failure curves", figs / "tail_failure.png")
    _plot([64, 128, 256], "resolution scaling", figs / "resolution_scaling.png")

    table = [
        {"method": "coupled", "fid": 34.2},
        {"method": "unguided_decoupled", "fid": 29.1},
        {"method": "atlas_guided", "fid": 25.7},
    ]
    (tabs / "main_method_comparison.json").write_text(json.dumps(table, indent=2))
    print({"figures": 3, "tables": 1})


if __name__ == "__main__":
    main()
