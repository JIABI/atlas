
from pathlib import Path
import torch
from ..probes.probe_pipeline import compute_probes
from .atlas_io import write_record

def run_line_sweep(out_dir='outputs/atlas/sweeps', seeds=(0,1), family='line_x0_u'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        torch.manual_seed(seed)
        x=torch.randn(8,3,32,32)
        probes=compute_probes(x)
        rec={"exp_id":"EXP-A1","dataset":"cifar10","split":"train","model":"unet_small","corruption":"diffusion_like","target_family":family,"target_lambda":{"alpha":0.5},"prediction_family":"identity","prediction_eta":{},"lambda_loss":{},"seed":seed,"resolution":32,"regularization":{"mu":0.1,"tau":0.1},"trainability":{"converged":True,"diverged":False,"nan":False,"time_to_threshold":1.0,"collapse_rate":0.0},"quality":{"fid":50.0,"sfid":40.0,"lpips":0.3},"tail":{"worst_k_score":0.2,"percentile_95":0.1,"percentile_99":0.15,"rare_failure_rate":0.0},"pathology":probes,"artifacts":{"checkpoint":"checkpoints/mock.pt","samples_dir":"outputs/samples","plots_dir":"outputs/plots","config_path":"outputs/config.yaml"}}
        write_record(Path(out_dir)/f'{family}_seed{seed}.json',rec)
