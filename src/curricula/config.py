from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json

# writes curriculum metadata
def write_config_json(path: Path, config: Dict[str, Any]) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, sort_keys=True)