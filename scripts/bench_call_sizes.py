#!/usr/bin/env python
"""Price a submission zip per window at LARGE call sizes and with two variants
(fresh-eyes time review, 2026-09-18): the grader's number of windows per
predict() call is unknown, and both v2 (n < 16) and v3 (n <= 16) run their
graphs only for small calls -- a big call takes the eager path, which
bench_graph_path.py never timed. Variants:
    base   : the zip as shipped
    nohead : the same zip with the "width_head" block removed from bounds.json
             (falls back to the constant/spatial widths) -> the head's own cost
    tf32   : the zip with TF32 re-enabled after import (matmul + cudnn)

    python scripts/bench_call_sizes.py submission/dist/<zip> [--sizes 1 16 64 256] [--variants base nohead tf32] [--profile]

Prints ms/window per variant and size (median of --reps calls after one warm
call), the max |base - variant| on prediction and widths at the largest size,
and with --profile the top CUDA kernels of one base call at the largest size.
"""
import argparse, importlib.util, json, os, shutil, sys, tempfile, time, zipfile
from pathlib import Path

import numpy as np, torch


def load_module(zip_path: Path, strip_head: bool):
    tmp = Path(tempfile.mkdtemp(prefix="bench_sizes_"))
    zipfile.ZipFile(zip_path).extractall(tmp)
    if strip_head:
        b = json.loads((tmp / "bounds.json").read_text())
        b.pop("width_head", None)
        (tmp / "bounds.json").write_text(json.dumps(b))
    sys.path.insert(0, str(tmp))
    spec = importlib.util.spec_from_file_location(f"submission_{tmp.name}", tmp / "submission.py")
    sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
    return sub, tmp


def outputs(o):
    if isinstance(o, dict):
        w = (o["upper"] - o["lower"]) if "upper" in o else None
        return np.asarray(o["prediction"], np.float32), (None if w is None else np.asarray(w, np.float32))
    return np.asarray(o, np.float32), None


def timeit(sub, x, reps):
    sub.predict(x, {}); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); sub.predict(x, {}); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / x.shape[0]


def main(a):
    rng = np.random.default_rng(0)
    xs = {n: rng.normal(0, 0.1, (n, 20, 32, 64, 3)).astype(np.float32) for n in a.sizes}
    for arr in xs.values():
        arr[..., 2] = 0
    print(f"zip {a.zip}  torch {torch.__version__}  cuda {torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    ref = {}
    tmps = []
    try:
        for var in a.variants:
            sub, tmp = load_module(a.zip, strip_head=(var == "nohead")); tmps.append(tmp)
            if var == "tf32":
                torch.backends.cudnn.allow_tf32 = True; torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(sub, "_graphs"): sub._graphs.clear()
                if hasattr(sub, "_GRAPH_OK"): sub._GRAPH_OK[0] = None
            else:
                torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
            row = []
            for n in a.sizes:
                row.append(f"n={n:>4}: {timeit(sub, xs[n], a.reps) * 1000:7.3f} ms/win")
            print(f"{var:<7} " + "   ".join(row), flush=True)
            big = xs[max(a.sizes)]
            p, w = outputs(sub.predict(big, {}))
            if var == "base":
                ref = dict(p=p, w=w)
            else:
                dp = float(np.abs(p - ref["p"]).max())
                dw = "n/a" if (w is None or ref["w"] is None) else f"{float(np.abs(w - ref['w']).max()):.2e}"
                print(f"        max |{var} - base| at n={big.shape[0]}: prediction {dp:.2e}, widths {dw}")
            if a.profile and var == "base" and torch.cuda.is_available():
                from torch.profiler import profile, ProfilerActivity
                sub.predict(big, {}); torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                    sub.predict(big, {}); torch.cuda.synchronize()
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
    finally:
        for t in tmps:
            shutil.rmtree(t, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", type=Path)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 16, 64, 256])
    ap.add_argument("--variants", nargs="+", default=["base", "nohead", "tf32"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--profile", action="store_true")
    main(ap.parse_args())
