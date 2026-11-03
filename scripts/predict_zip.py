#!/usr/bin/env python
"""Run one submission zip's predict() on a saved input array and save its outputs.

    python scripts/predict_zip.py <zip> <x.npy> <out.npz>

Used by score_local.py --bounds-from: two zips cannot both be imported as `submission`
in one process, so the second one runs here, in its own interpreter. CEN (2026-09-22).
"""
import sys, tempfile, zipfile, shutil
from pathlib import Path
import numpy as np


def main(zip_path: str, x_path: str, out_path: str) -> None:
    x = np.load(x_path)
    work = Path(tempfile.mkdtemp(prefix="predict_zip_"))
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(work)
        sys.path.insert(0, str(work))
        import submission  # noqa: E402
        out = submission.predict(x, {})
        arrays = {"prediction": np.asarray(out["prediction"], dtype=np.float32)}
        for k in ("lower", "upper"):
            if k in out:
                arrays[k] = np.asarray(out[k], dtype=np.float32)
        np.savez(out_path, **arrays)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
