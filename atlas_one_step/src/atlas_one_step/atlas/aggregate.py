from pathlib import Path
from .atlas_io import read_records,to_parquet
def aggregate(in_dir,out_path):
    recs=read_records(Path(in_dir)); to_parquet(recs,Path(out_path)); return len(recs)
