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
) -> Path:
  idx = _next_index(out_base)
  name = (
    f"{idx:03d}"
    f"__curr={curriculum}"
    f"__method={scoring_method}"
    f"__order={sort_order}"
    f"__tranche={tranche_type}"
    f"__size={tranche_size}"
    f"__aoaAgn={int(aoa_agnostic)}"
    f"__mw={int(multiword)}"
    f"__skipStop={int(skip_stopwords)}"
    f"__inflect={int(inflect)}"
  )
  return Path(out_base) / name

def make_run_dir(out_dir: Path) -> Path:
  return Path(out_dir) / "_run"