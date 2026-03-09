from atlas_one_step.probes.probe_pipeline import compute_probes
import torch
def test_probes():
 d=compute_probes(torch.randn(2,3,4,4)); assert "pathology_score" in d
