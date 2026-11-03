#!/usr/bin/env python
"""Cold-start cost of a submission zip, the way a grader that times the FIRST
predict() would see it (audit finding 1, 2026-09-17).

    python scripts/bench_cold_start.py submission/dist/<zip> [--windows 136]

Runs in THIS process: timestamps after `import torch`, after CUDA init, after
importing submission.py (model load), after the first predict (1 window: CUDA
graph capture + warm-up), after the second predict, then a steady-state
average. Prints each as absolute ms and as ms/window for --windows windows, so
the one-time cost can be compared with the grader's ~5.4 ms/window. Run once
in a fresh process per zip; run under the t222 env as well as the training env.
"""
import argparse, importlib.util, os, sys, tempfile, time, zipfile

ap = argparse.ArgumentParser()
ap.add_argument("zip"); ap.add_argument("--windows", type=int, default=136)
a = ap.parse_args()
T = [("start", time.perf_counter())]
import numpy as np  # noqa: E402
import torch  # noqa: E402
T.append(("import torch", time.perf_counter()))
if torch.cuda.is_available():
    torch.cuda.init(); torch.zeros(1, device="cuda"); torch.cuda.synchronize()
T.append(("cuda init", time.perf_counter()))
tmp = tempfile.mkdtemp(prefix="coldstart-")
zipfile.ZipFile(a.zip).extractall(tmp)
sys.path.insert(0, tmp); os.chdir(tmp)
spec = importlib.util.spec_from_file_location("submission", os.path.join(tmp, "submission.py"))
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
T.append(("import submission", time.perf_counter()))
x1 = np.random.default_rng(0).normal(0, 0.1, (1, 20, 32, 64, 3)).astype(np.float32); x1[..., 2] = 0
sub.predict(x1, {}); torch.cuda.synchronize() if torch.cuda.is_available() else None
T.append(("first predict (1 window)", time.perf_counter()))
sub.predict(x1, {}); torch.cuda.synchronize() if torch.cuda.is_available() else None
T.append(("second predict", time.perf_counter()))
t0 = time.perf_counter()
for _ in range(50): sub.predict(x1, {})
torch.cuda.synchronize() if torch.cuda.is_available() else None
steady = (time.perf_counter() - t0) / 50
print(f"zip {a.zip}  cuda={torch.cuda.is_available()}  torch {torch.__version__}")
for (name, t), (_, tp) in zip(T[1:], T[:-1]):
    d = (t - tp) * 1000
    print(f"  {name:<28} {d:9.1f} ms   = {d / a.windows:7.3f} ms/window over {a.windows} windows")
one_time = (T[4][1] - T[0][1]) * 1000 - steady * 1000
print(f"  steady state                 {steady*1000:9.3f} ms/window")
print(f"  one-time total (to first predict, minus one steady call): {one_time:.0f} ms = {one_time / a.windows:.3f} ms/window over {a.windows}")
