import pandas as pd
from pathlib import Path
from ..atlas.surrogate import fit_surrogate
def main():
 p=Path("outputs/atlas/atlas.parquet")
 if not p.exists(): print("atlas missing; run build_atlas first"); return
 _,s=fit_surrogate(pd.read_parquet(p)); Path("outputs/atlas/surrogate.txt").write_text(str(s))
if __name__=="__main__": main()
