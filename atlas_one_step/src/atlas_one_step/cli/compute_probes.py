from pathlib import Path
import json, torch
from ..probes.probe_pipeline import compute_probes

def main():
 out=Path("outputs/atlas/probes.json"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(compute_probes(torch.randn(8,3,32,32)),indent=2))

if __name__=="__main__": main()
