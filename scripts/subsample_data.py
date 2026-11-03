#!/usr/bin/env python
"""Subsample the raw trajectories to eval resolution once, into data/processed/.

Raw PIV is 64x128 float64 gzipped; scoring runs at 32x64. Reading windows
straight from the raw files costs ~25 ms each because every read has to gunzip
the full spatial plane, so training would be I/O bound. This writes the 32x64
float32 store that training reads instead: ~3.6 GB, small enough to hold in RAM.

Stores whole trajectories, not pre-cut windows, so stride / T_in / T_out stay
free parameters at training time. Takes about a minute -- to rebuild, just
rerun it.

    python scripts/subsample_data.py
"""
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

from pde_competition.dataset import EXCLUDED_REAL
from pde_competition.paths import PROCESSED, TRAIN_REAL, TRAIN_SIM

# Raw PIV is 64x128; Track 1 strides it by 2 (strided, NOT averaged) so scoring
# runs at 32x64. This lives here rather than in the package because nothing that
# reads data/processed needs it -- the store is already subsampled. It is the
# parameter that produced that store, so changing it invalidates everything
# under data/processed and this script must be rerun.
#
# The coordinate grid is not stored: it was verified identical (to 1e-9 m) across
# all 182 raw files, real and sim alike -- a uniform rectilinear grid in metres --
# so nothing downstream ever needed it. (scripts/build_grid.py established that
# and was retired 2026-09-09; `git show pre-restructure:scripts/build_grid.py`.)
SUB_S = 2

# THE FOUR PHASES. Raw PIV is 64x128 and Track 1 strides it by 2, taking offset (0, 0).
# The other three offsets -- (0,1), (1,0), (1,1) -- are equally valid 32x64 samples of the
# SAME flow, so the real data is 4x larger than we have ever used, and real rows are the
# scarce resource (17 of them against an unlimited sim).
# THE CATCH, and it decides how they must be used: the grader ALWAYS scores offset (0,0),
# and the airfoil sits at a different sub-pixel position in the other three. A model
# trained on all four learns a BLURRED body position, which is exactly the wake region
# where rel_l2 and tke are won. So the phases are for a pretraining/curriculum stage and
# the final fine-tune must be on (0,0) alone.
PHASES = {"p00": (0, 0), "p01": (0, 1), "p10": (1, 0), "p11": (1, 1)}


def subsample_h5(path: Path, sub_s: int = SUB_S, phase: tuple = (0, 0)) -> tuple[np.ndarray, dict]:
    """Return (data, meta) for one trajectory.

    data is (T, H//sub_s, W//sub_s, C) float32 -- [u, v] for real, [u, v, p]
    for sim. Real data has no pressure field; the zero p channel the models
    expect is cheaper to broadcast at batch time than to store 1.1 GB of zeros.

    meta is the manifest row for this file, minus re_nominal, which only the
    filename knows.
    """
    with h5py.File(path, "r") as f:
        # Read whole, then stride in numpy: h5py's strided selection is ~30%
        # slower than decompressing and letting numpy do the striding.
        oy, ox = phase
        fields = [np.asarray(f[k][:][:, oy::sub_s, ox::sub_s], dtype=np.float32)
                  for k in ("u", "v", "p") if k in f]
        meta = {"re_measured": int(f["re"][()]), "aoa": int(f["aoa"][()])}

    data = np.stack(fields, axis=-1)
    meta["n_frames"], meta["channels"] = data.shape[0], data.shape[-1]
    return data, meta


FIELDS = ["file", "re_nominal", "re_measured", "aoa", "n_frames", "channels"]


def build(root: Path, out_dir: Path, exclude: frozenset[str] = frozenset(), phase: tuple = (0, 0)):
    """Subsample every .h5 under root into out_dir, plus a manifest.csv."""
    all_paths = sorted(root.glob("*.h5"))
    # STEM, not name: EXCLUDED_REAL holds "7575_0" while p.name is "7575_0.h5", so matching
    # on .name is a silent no-op -- the build even prints "(0 excluded)". train_real predates
    # this and is correct at 81; the three phase stores built 2026-09-21 each shipped 82 and
    # the Mac caught it by diffing file lists, not from any error. Checked both ways so a
    # future EXCLUDED holding either form still filters.
    paths = [p for p in all_paths if p.stem not in exclude and p.name not in exclude]
    if not paths:
        sys.exit(f"no .h5 files under {root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows, total = [], 0
    for i, p in enumerate(paths, 1):
        data, meta = subsample_h5(p, phase=phase)
        out = out_dir / f"{p.stem}.npy"
        np.save(out, data)
        total += out.stat().st_size

        re_nominal, aoa = map(int, p.stem.split("_"))
        rows.append({"file": out.name, "re_nominal": re_nominal, **meta})
        print(f"  [{i:3d}/{len(paths)}] {out.name:16s} {str(data.shape):24s} "
              f"{out.stat().st_size / 1e6:7.1f} MB")

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"{out_dir}: {len(rows)} trajectories, {total / 1e9:.2f} GB "
          f"({len(all_paths) - len(paths)} excluded)")
    return total


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=sorted(PHASES), default="p00",
                    help="which of the four sub-grid offsets to take. p00 is what Track 1 scores and "
                         "what every store built before 2026-09-21 contains; p01/p10/p11 are the same "
                         "flow sampled on a half-pixel-shifted grid, i.e. 3x more REAL data.")
    ap.add_argument("--real-only", action="store_true",
                    help="skip the sim split. Right for the extra phases: sim is unlimited already, and "
                         "the phases exist to multiply the 17 scarce REAL rows.")
    a_ = ap.parse_args()
    off = PHASES[a_.phase]
    suffix = "" if a_.phase == "p00" else f"_{a_.phase}"
    print(f"phase {a_.phase} offset {off}")
    print(f"{TRAIN_REAL} -> {PROCESSED / ('train_real' + suffix)}")
    a = build(TRAIN_REAL, PROCESSED / ("train_real" + suffix), EXCLUDED_REAL, phase=off)
    b = 0
    if not a_.real_only:
        print(f"\n{TRAIN_SIM} -> {PROCESSED / ('train_sim' + suffix)}")
        b = build(TRAIN_SIM, PROCESSED / ("train_sim" + suffix), phase=off)
    print(f"\ntotal {(a + b) / 1e9:.2f} GB")
