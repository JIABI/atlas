import json
from pathlib import Path
def save_selection(path,sel): Path(path).write_text(json.dumps(sel,indent=2))
