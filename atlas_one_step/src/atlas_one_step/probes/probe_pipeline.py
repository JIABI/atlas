
from .support_deviation import compute_support_deviation
from .normal_burden import compute_normal_burden
from .target_jacobian import jacobian_proxy
from .early_training_pathology import early_pathology
from .pathology_score import aggregate_pathology

def compute_probes(x):
    out={}
    out.update(compute_support_deviation(x))
    out.update(compute_normal_burden(x))
    out.update(jacobian_proxy(x))
    out.update(early_pathology())
    out['pathology_score']=aggregate_pathology(out)
    return out
