import json
from pathlib import Path
def log_json(path,data): Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(data,indent=2))
