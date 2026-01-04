from __future__ import annotations

from pathlib import Path
import re

# next index to name curriculum
def _next_index(out_base: Path) -> int:
  out_base = Path(out_base)
  out_base.mkdir(parents=True, exist_ok=True)

  max_idx = -1
  for p in out_base.iterdir():
    if not p.is_dir():
      continue
    m = re.match(r"^(\d+)", p.name)
    if m:
      max_idx = max(max_idx, int(m.group(1)))
  return max_idx + 1

def make_curriculum_dir(
  out_base: Path,
  curriculum: str,
  scoring_method: str,
  sort_order: str,
  tranche_type: str,
  tranche_size: int,
  aoa_agnostic: bool,
  multiword: bool,
  skip_stopwords: bool,
  inflect: bool,
) -> tuple[str, Path]:
  """
  Returns (idx, out_dir)
  
  Uses abbreviated names to stay under Windows MAX_PATH (260 chars).
  """
  idx = _next_index(out_base)
  
  # Abbreviate tranche_type for shorter paths
  tt_abbrev = "wb" if tranche_type == "word-based" else "sb"
  
  name = (
    f"{idx:03d}"
    f"_c={curriculum}"
    f"_m={scoring_method}"
    f"_o={sort_order}"
    f"_t={tt_abbrev}"
    f"_s={tranche_size}"
    f"_ag={int(aoa_agnostic)}"
    f"_mw={int(multiword)}"
    f"_ss={int(skip_stopwords)}"
    f"_in={int(inflect)}"
  )
  return str(idx), Path(out_base) / name

def make_run_dir(out_dir: Path) -> Path:
  return Path(out_dir) / "_run"