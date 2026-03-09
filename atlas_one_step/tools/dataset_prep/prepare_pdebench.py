from pathlib import Path
import argparse
if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--root",required=True); a=p.parse_args(); Path(a.root).mkdir(parents=True,exist_ok=True); print("prepared",a.root)
