#!/usr/bin/env python
"""Verify a submission zip the way the grader will run it.

Extracts the archive to a temp dir, blocks the packages the eval container does
NOT ship, imports submission.py and calls predict(). Catches missing imports,
wrong shapes and bad bounds before you spend a daily submission.

    python scripts/verify_submission.py submission/dist/flower_b16_bounds.zip

Run the check under a torch matching the eval image (2.2.2), NOT the training
env's torch. That distinction is not academic: one of our submissions passed this
script on torch 2.13 and then scored 0.0 on every subscore, because
`@torch.compile` inside flower_standalone drives the inductor backend on 2.2.2
-- writing a cache dir and shelling out to a C++ compiler, which the eval
sandbox denies -- while doing nothing at all on 2.13.

The interpreter is resolved in this order, and the one used is always printed
so a PASS is never ambiguous:

    --python PATH  ->  $RPDE_EVAL_PYTHON  ->  this interpreter

Point $RPDE_EVAL_PYTHON at an environment with torch 2.2.2 (for example a
Python 3.10 or 3.11 venv with ``pip install torch==2.2.2 numpy==1.26.4``).
Without it the check still runs, but says plainly that it cannot catch
version-specific failures. Note that torch 2.2.2's MPS backend cannot run
Flower at all ("Unsupported Border padding mode" in grid_sample), so on Apple
Silicon the eval environment is CPU-only.

WHAT THIS STILL CANNOT CATCH: the container is CUDA on an A800; this is CPU on
Apple Silicon. Device-specific failures, the 5-minute wall clock under real
load, and GPU memory limits remain out of reach.
"""
import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path

# The eval image is pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime: no scipy,
# h5py, einops, matplotlib or pandas.
MISSING_IN_EVAL_IMAGE = ("h5py", "scipy", "matplotlib", "pandas")

CHILD = textwrap.dedent('''
    import sys, os, time, numpy as np

    # Import torch BEFORE arming the Blocker. torch is guaranteed present in
    # the eval image, and it probes for optional packages during its own
    # import: torch._dynamo calls find_spec("pandas"), which returns None for a
    # genuinely absent module but would hit the Blocker below and raise.
    # Blocking torch's internal probes tests nothing real -- what matters is
    # whether the SUBMISSION reaches for a package the image lacks.
    #
    # torch._dynamo is imported explicitly because einops pulls it in
    # (_torch_specific.py: `from torch._dynamo import allow_in_graph`), and any
    # submission using einops -- Flower, Transolver -- would otherwise trip on
    # that probe rather than on a real missing dependency.
    import torch
    import torch._dynamo  # noqa: F401

    MISSING = set(%r)
    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in MISSING:
                raise ModuleNotFoundError(
                    "No module named %%r (blocked: absent from eval image)" %% name)
            return None
    sys.meta_path.insert(0, Blocker())
    sys.path.insert(0, os.getcwd())

    import submission
    N, T_IN, T_OUT, H, W, C = 2, 20, 20, 32, 64, 3
    x = np.random.randn(N, T_IN, H, W, C).astype(np.float32)
    x[..., 2] = 0.0

    t0 = time.time(); out = submission.predict(x, {}); dt = time.time() - t0

    if isinstance(out, dict):
        missing = {"prediction"} - set(out)
        assert not missing, "dict output missing key: %%s" %% missing
        pred = np.asarray(out["prediction"])
        if "lower" in out or "upper" in out:
            assert {"lower", "upper"} <= set(out), "bounds must be all-or-nothing"
            lo, hi = np.asarray(out["lower"]), np.asarray(out["upper"])
            assert lo.shape == hi.shape == pred.shape, "bounds shape mismatch"
            assert np.isfinite(lo).all() and np.isfinite(hi).all(), "non-finite bounds"
            assert (lo <= hi).all(), "lower > upper somewhere -> scores ZERO on every metric"
            note = ""
            try:
                import json as _j
                if "width_head" in _j.load(open("bounds.json")):
                    note = ("  [input-dependent width model: on this N(0,1) noise input the "
                            "mean says NOTHING about scored widths -- see score_local on a twin]")
            except Exception:  # noqa: BLE001 -- the note is advisory
                pass
            print("bounds : ok, width mean %%.4g%%s" %% ((hi - lo).mean(), note))
    else:
        pred = np.asarray(out)

    assert pred.shape == (N, T_OUT, H, W, C), "bad shape: %%s" %% (pred.shape,)
    assert np.isfinite(pred).all(), "non-finite prediction"
    print("shape  :", pred.shape, "| dtype:", pred.dtype)
    print("finite : True")
    print("time   : %%.2fs for %%d windows" %% (dt, N))
    # the parent prints the PASS line, qualified by torch version
''') % (MISSING_IN_EVAL_IMAGE,)


EVAL_TORCH = "2.2.2"


def resolve_python(explicit: str | None) -> Path:
    """Which interpreter runs the clean-room check. See the module docstring."""
    for cand in (explicit, os.environ.get("RPDE_EVAL_PYTHON")):
        if cand and Path(cand).exists():
            return Path(cand)
    return Path(sys.executable)


def torch_version_of(python: Path) -> str:
    r = subprocess.run([str(python), "-c", "import torch; print(torch.__version__)"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def main(zip_path: Path, python: str | None = None):
    if not zip_path.exists():
        raise SystemExit(f"no such zip: {zip_path}")
    mb = zip_path.stat().st_size / 1e6
    print(f"{zip_path.name}  {mb:.1f} MB", "OVER 256 MB CAP" if mb > 256 else "(under cap)")

    py = resolve_python(python)
    tv = torch_version_of(py)
    matches = tv.startswith(EVAL_TORCH)
    print(f"python : {py}")
    print(f"torch  : {tv} " + ("(matches eval image)" if matches else
          f"!= eval image {EVAL_TORCH} -- version-specific failures CANNOT be "
          f"caught; set $RPDE_EVAL_PYTHON to an interpreter with torch {EVAL_TORCH}"))

    tmp = Path(tempfile.mkdtemp(prefix="verify_sub_"))
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        roots = sorted(p.name for p in tmp.iterdir())
        print("root   :", roots)
        if "submission.py" not in roots:
            raise SystemExit("submission.py is not at the archive ROOT")

        r = subprocess.run([str(py), "-c", CHILD], cwd=tmp,
                           capture_output=True, text=True)
        print(r.stdout.strip())
        if r.stderr.strip():
            # ALWAYS, not only on failure. submission.py prints one line per process saying
            # which path it actually took (path=graph / path=fast / path=eager, member count,
            # head layout). On 2026-09-21 H1g verified a fast_predict zip under torch 2.2.2,
            # printed PASS, and proved nothing about the fast path, because that line was
            # discarded on success -- the exact "passes while proving nothing" trap the
            # parcel's own header warned about. A submission writing to stderr is interesting
            # by definition; print it.
            for line in r.stderr.strip().splitlines():
                print(f"  child stderr: {line}")
        if r.returncode != 0:
            raise SystemExit(f"FAILED (on torch {tv})")
        print(f"PASS on torch {tv}" if matches else
              f"PASS, but on torch {tv} -- NOT the eval image's {EVAL_TORCH}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", type=Path)
    ap.add_argument("--python", default=None, metavar="PATH",
                    help="interpreter to run the clean-room check under; "
                         "defaults to $RPDE_EVAL_PYTHON, then this interpreter")
    a = ap.parse_args()
    main(a.zip, a.python)
