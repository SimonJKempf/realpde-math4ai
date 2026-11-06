"""Canonical filesystem paths for this project.

Every path is derived from this file's own location, so imports resolve
correctly from anywhere: a notebook in notebooks/, a script in scripts/,
the repo root, or another machine.

    from pde_competition.paths import EXAMPLE_H5, RAW
"""
import os
from pathlib import Path

# src/pde_competition/paths.py -> src/pde_competition -> src -> repo root
REPO = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    """Relocate one root through the environment. Unset -> the repo-relative
    default, so nothing changes for the ordinary checkout. The final training
    package (final/solution.py) runs the same scripts inside a container where
    the raw data is a read-only mount and every artefact must land in the
    mounted output directory, and these three are the roots it moves."""
    v = os.environ.get(var)
    return Path(v).expanduser().resolve() if v else default


DATA        = REPO / "data"
RAW         = _env_path("RPDE_RAW", DATA / "raw")
PROCESSED   = _env_path("RPDE_PROCESSED", DATA / "processed")
EXAMPLE     = DATA / "example"

CHECKPOINTS = REPO / "checkpoints"
CKPT_FLAT   = CHECKPOINTS / "all"          # flat symlinks the kit expects
OUTPUTS     = _env_path("RPDE_OUTPUTS", REPO / "outputs")   # gitignored: runs/, scores/
KIT         = REPO / "kit"

# Where summarize_runs.py writes its CSV (created on demand).
EXPERIMENTS = REPO / "experiments"
QUEUE       = EXPERIMENTS / "queue"
RESULTS     = EXPERIMENTS / "results"      # summary.csv

EXAMPLE_H5  = EXAMPLE / "3750_0.h5"
TRAIN_REAL  = RAW / "train_real"
TRAIN_SIM   = RAW / "train_sim"
