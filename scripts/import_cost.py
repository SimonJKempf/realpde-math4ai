#!/usr/bin/env python
"""How long does a submission zip take to IMPORT, as a function of member count?

The grader runs a fresh process per predict() call and does not time the import --
but the container still has a 5-minute execution limit (2026-09-17-codabench-rules.md),
and everything we do at import is paid once per call against that budget: CUDA init,
loading k checkpoints, and the import-time CUDA-graph captures for sizes 1 and 16,
which scale with k.

Nobody has measured this. For a single model it is irrelevant; for an 8-member
ensemble it is the only route to a TIMEOUT, which scores 0 rather than merely
costing time points. This measures it per zip, in a FRESH process each time, the
way the grader would pay it.

    python scripts/import_cost.py submission/dist/a.zip submission/dist/b.zip
"""
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

CHILD = r'''
import sys, time, json, importlib.util
t0 = time.perf_counter()
spec = importlib.util.spec_from_file_location("sub", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t_import = time.perf_counter() - t0
import numpy as np
x = np.zeros((1, 20, 32, 64, 3), np.float32); x[..., 0] = 0.1; x[..., 1] = 0.01
t1 = time.perf_counter(); m.predict(x); t_first = time.perf_counter() - t1
t2 = time.perf_counter(); m.predict(x); t_second = time.perf_counter() - t2
print(json.dumps(dict(t_import=t_import, t_first=t_first, t_second=t_second)))
'''


def main():
    zips = [Path(z) for z in sys.argv[1:]]
    if not zips:
        raise SystemExit("usage: import_cost.py <zip> [<zip> ...]")
    print(f"{'zip':<44} {'members':>7} {'import s':>9} {'1st call s':>11} {'2nd call':>11} {'per-call s':>11}")
    for z in zips:
        with tempfile.TemporaryDirectory() as d:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(d)
            # The member list is NOT in bounds.json -- it lives in the members manifest the
            # module reads at import. Counting the shipped .pth files and subtracting the
            # width head is the reliable measure, and it is what the zip actually carries.
            pths = [q for q in Path(d).rglob("*.pth")]
            k = max(1, len(pths) - sum(1 for q in pths if "width_head" in q.name))
            sub = Path(d) / "submission.py"
            if not sub.exists():
                cands = [p for p in Path(d).glob("*.py") if "submission" in p.name]
                sub = cands[0] if cands else None
            child = Path(d) / "_child.py"
            child.write_text(CHILD)
            r = subprocess.run([sys.executable, str(child), str(sub)],
                               capture_output=True, text=True, cwd=d)
            if r.returncode != 0:
                print(f"{z.name:<44} {k:>7}  FAILED: {r.stderr.strip().splitlines()[-1][:60]}")
                continue
            out = json.loads(r.stdout.strip().splitlines()[-1])
            # what the container actually pays for one grader call in a fresh process
            per_call = out["t_import"] + out["t_first"]
            print(f"{z.name:<44} {k:>7} {out['t_import']:9.2f} {out['t_first']:11.2f} "
                  f"{out['t_second']*1e3:8.2f} ms {per_call:11.2f}")
    print("\nThe container limit is 300 s. 'per-call s' is import + first predict in a fresh")
    print("process -- what the grader pays per call. Divide 300 by it for the call budget.")


if __name__ == "__main__":
    main()
