from __future__ import annotations

from atlas_one_step.probes.probe_pipeline import compute_probes
import torch


def main() -> None:
    p = compute_probes(torch.randn(4, 3, 8, 8))
    if p["pathology_score"] < 0:
        raise SystemExit("Unexpected negative pathology score")
    print(p)


if __name__ == "__main__":
    main()
