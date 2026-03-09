
import json
from pathlib import Path
import pandas as pd

def write_record(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2))

def read_records(folder: Path):
    return [json.loads(p.read_text()) for p in folder.glob('*.json')]

def to_parquet(records, out: Path):
    pd.DataFrame(records).to_parquet(out, index=False)
