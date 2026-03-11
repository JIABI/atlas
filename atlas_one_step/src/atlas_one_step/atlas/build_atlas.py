from .aggregate import aggregate
def build_atlas(in_dir="outputs/atlas/sweeps", out="outputs/atlas/atlas.parquet"):
    return aggregate(in_dir,out)
