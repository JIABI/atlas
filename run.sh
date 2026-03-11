python -m atlas_one_step.cli run-sweep --config configs/cifar10_line_sweep.yaml

python analysis/summarize_tail_failures.py  \
   --input /home/ubuntu/PyCharmMiscProject/atlas_one_step/outputs/smoke/atlas/atlas.csv  \
   --outdir analysis_outputs/tail_summary

# compare multi runs
python analysis/compare_runs.py \
  --inputs outputs/run1/results.csv outputs/run2/results.csv outputs/run3/results.csv \
  --outdir analysis_outputs/compare_runs

# generate diagram
python analysis/make_phase_diagram.py \
  --input /home/ubuntu/PyCharmMiscProject/atlas_one_step/outputs/smoke/atlas/atlas.csv \
  --outdir analysis_outputs/phase_diagram



python -m atlas_one_step.cli run-sweep --config configs/cifar10_line_sweep.yaml

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