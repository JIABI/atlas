from __future__ import annotations

from atlas_one_step.targets.line_families import line_x0_u, line_x0_r, line_x0_eps
import torch


def main() -> None:
    x0 = torch.randn(2, 3, 8, 8)
    xt = torch.randn(2, 3, 8, 8)
    eps = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    vals = [line_x0_u(0.3, x0, xt, eps, t), line_x0_r(0.3, x0, xt, eps, t), line_x0_eps(0.3, x0, xt, eps, t)]
    print({"ok": all(v.shape == x0.shape for v in vals)})


if __name__ == "__main__":
    main()
