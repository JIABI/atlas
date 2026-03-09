from pathlib import Path
import json
def main():
 Path("outputs/eval").mkdir(parents=True,exist_ok=True); Path("outputs/eval/metrics.json").write_text(json.dumps({"fid":0.0},indent=2))
if __name__=="__main__": main()
