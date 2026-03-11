from pathlib import Path
def latest_checkpoint(folder="checkpoints"):
    p=Path(folder); return sorted(p.glob("*.pt"))[-1] if p.exists() and list(p.glob("*.pt")) else None
