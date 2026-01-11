# curricula/path_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Union

PKG_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINING_ROOT = PKG_DIR.parent.parent / "data" / "processed" / "corpora" / "training"


def resolve_curriculum_path(
  curriculum_path: Union[str, Path, int],
  training_root: Path = DEFAULT_TRAINING_ROOT,
) -> Path:
  """
  Resolve a curriculum identifier into a filesystem path.

  Accepts:
    - int index: 7
    - digit string index: "7" or "007"
    - path-like string / Path: "data/.../training/007__curr=..." or "./..."

  Behavior:
    - If `curriculum_path` is an index, it resolves to the unique directory under
      `training_root` whose name starts with the zero-padded index (e.g. "007*").
    - Otherwise, returns it as a Path (no existence check beyond what callers do).

  Raises:
    - FileNotFoundError if no matching curriculum folder exists for an index
    - RuntimeError if multiple folders match (shouldn't happen if indices are unique)
  """
  # int index
  if isinstance(curriculum_path, int):
    idx = curriculum_path
    return _resolve_index(idx, training_root)

  s = str(curriculum_path).strip()

  # digit string index
  if s.isdigit():
    idx = int(s)
    return _resolve_index(idx, training_root)

  # normal path
  return Path(s)

def _parse_tranche_idx(name: str) -> int:
  # tranche_0001 -> 1
  try:
    return int(name.split("_", 1)[1])
  except Exception:
    return 10**9

def find_tranche_dirs(tranches_dir: Path) -> List[Path]:
  tranche_dirs = [p for p in Path(tranches_dir).iterdir() if p.is_dir() and p.name.startswith("tranche_")]
  tranche_dirs.sort(key=lambda p: _parse_tranche_idx(p.name))
  return tranche_dirs

def get_tranches_dir(curriculum_root_or_tranches: Path) -> Path:
  p = Path(curriculum_root_or_tranches)
  if (p / "tranches").is_dir():
    return p / "tranches"
  return p

def _resolve_index(idx: int, training_root: Path) -> Path:
  training_root = Path(training_root)
  pattern = f"{idx:03d}*"
  matches = sorted(training_root.glob(pattern))
  if len(matches) == 0:
    raise FileNotFoundError(
      f"No curriculum folder starting with index {idx:03d} found under {training_root}"
    )
  if len(matches) > 1:
    names = ", ".join(m.name for m in matches[:5])
    raise RuntimeError(
      f"Multiple curriculum folders matched index {idx:03d} under {training_root}: {names}"
    )
  return matches[0]
