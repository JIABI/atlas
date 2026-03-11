python - <<'PY'
import json, glob, os, pandas as pd

rows = []
for p in glob.glob("/home/ubuntu/PyCharmMiscProject/atlas_one_step/outputs/cifar10_line_sweep/sweeps/line_x0_u_*_seed*/summary.json"):
    with open(p, "r") as f:
        d = json.load(f)
    d["run_dir"] = os.path.dirname(p)
    rows.append(d)

df = pd.json_normalize(rows)
print("rows:", len(df))
print("cols:", df.columns.tolist())
out = "/home/ubuntu/PyCharmMiscProject/atlas_one_step/outputs/cifar10_line_sweep/line_x0_u_all_runs.csv"
df.to_csv(out, index=False)
print("saved:", out)
PY

python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/cifar10_line_sweep/line_x0_u_all_runs.csv")
alpha_cols = [c for c in df.columns if "alpha" in c.lower()]
print("alpha columns:", alpha_cols)
for c in alpha_cols:
    vals = sorted(df[c].dropna().unique().tolist())
    print(c, vals)
PY

python - <<'PY'
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

inp = "outputs/cifar10_line_sweep/line_x0_u_all_runs.csv"
outdir = Path("outputs/cifar10_line_sweep/analysis")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(inp)

# try to find columns
def pick(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

alpha_col = pick([
    "prediction_spec.params.alpha",
    "target_lambda.alpha",
    "loss_target.alpha",
    "alpha",
])

psnr_col = pick(["quality.psnr", "psnr"])
ffd_col = pick(["quality.feature_fd", "feature_fd"])
collapse_col = pick(["quality.collapse_rate", "collapse_rate"])
path_col = pick(["pathology.pathology_score", "pathology_score"])
rho_col = pick(["pathology.rho_nor", "rho_nor"])
cond_col = pick(["pathology.conditioning", "conditioning"])
conv_col = pick(["trainability.converged", "converged"])

print("alpha_col:", alpha_col)
print("psnr_col:", psnr_col)
print("path_col:", path_col)

if alpha_col is None:
    raise RuntimeError("No alpha column found.")

# numeric cast
for c in [alpha_col, psnr_col, ffd_col, collapse_col, path_col, rho_col, cond_col, conv_col]:
    if c is not None:
        df[c] = pd.to_numeric(df[c], errors="coerce")

agg_cols = {}
for c in [psnr_col, ffd_col, collapse_col, path_col, rho_col, cond_col, conv_col]:
    if c is not None:
        agg_cols[c] = ["mean", "std", "count"]

g = df.groupby(alpha_col).agg(agg_cols)
g.columns = [f"{a}__{b}" for a,b in g.columns.to_flat_index()]
g = g.reset_index().sort_values(alpha_col)
g.to_csv(outdir / "alpha_summary.csv", index=False)
print("saved:", outdir / "alpha_summary.csv")

# plotting helper
def lineplot(ymean, ystd, ylabel, fname):
    if ymean not in g.columns:
        return
    plt.figure(figsize=(6,4))
    plt.plot(g[alpha_col], g[ymean], marker="o")
    if ystd in g.columns:
        lo = g[ymean] - g[ystd].fillna(0)
        hi = g[ymean] + g[ystd].fillna(0)
        plt.fill_between(g[alpha_col], lo, hi, alpha=0.2)
    plt.xlabel("alpha")
    plt.ylabel(ylabel)
    plt.title(f"alpha vs {ylabel}")
    plt.tight_layout()
    plt.savefig(outdir / fname, dpi=200)
    plt.close()

if psnr_col:
    lineplot(f"{psnr_col}__mean", f"{psnr_col}__std", "PSNR", "alpha_vs_psnr.png")
if ffd_col:
    lineplot(f"{ffd_col}__mean", f"{ffd_col}__std", "feature_fd", "alpha_vs_feature_fd.png")
if collapse_col:
    lineplot(f"{collapse_col}__mean", f"{collapse_col}__std", "collapse_rate", "alpha_vs_collapse.png")
if path_col:
    lineplot(f"{path_col}__mean", f"{path_col}__std", "pathology_score", "alpha_vs_pathology.png")
if rho_col:
    lineplot(f"{rho_col}__mean", f"{rho_col}__std", "rho_nor", "alpha_vs_rho_nor.png")
if cond_col:
    lineplot(f"{cond_col}__mean", f"{cond_col}__std", "conditioning", "alpha_vs_conditioning.png")
if conv_col:
    lineplot(f"{conv_col}__mean", f"{conv_col}__std", "converged_rate", "alpha_vs_converged.png")

# pathology vs quality scatter
if path_col and psnr_col:
    plt.figure(figsize=(5,4))
    plt.scatter(df[path_col], df[psnr_col], alpha=0.8)
    plt.xlabel("pathology_score")
    plt.ylabel("PSNR")
    plt.title("pathology vs PSNR")
    plt.tight_layout()
    plt.savefig(outdir / "pathology_vs_psnr.png", dpi=200)
    plt.close()

if path_col and collapse_col:
    plt.figure(figsize=(5,4))
    plt.scatter(df[path_col], df[collapse_col], alpha=0.8)
    plt.xlabel("pathology_score")
    plt.ylabel("collapse_rate")
    plt.title("pathology vs collapse")
    plt.tight_layout()
    plt.savefig(outdir / "pathology_vs_collapse.png", dpi=200)
    plt.close()

print("done.")
PY