"""Fresh-process A/B of the timed call, the way the grader runs it: each trial is a NEW python process that imports
the zip's submission module (untimed) and calls predict() once on n=5 windows; the module's own 'timing first_call'
stderr line is parsed. Arms alternate A,B,A,B,... so host drift hits both. Also checks that predict() allocates no
device memory (torch.cuda.memory_stats num_alloc_retries / allocated bytes unchanged).

    python scripts/diag/fresh_timing.py ZIP_A ZIP_B [ZIP_C ...] [--n N]

The wall-minus-replay distribution is BIMODAL on the cluster (~1.6 or ~5.2 ms per call), so per arm we also print the
fraction of "slow" trials (> 3 ms) and every trial value.
"""
import re, statistics, subprocess, sys, tempfile, zipfile

CHILD = r'''
import sys, time, numpy as np
sys.path.insert(0, sys.argv[1])
import submission, torch
x = np.ascontiguousarray(np.random.default_rng(0).standard_normal((5, 20, 32, 64, 3)).astype(np.float32) * 0.1)
x[..., 2] = 0.0
torch.cuda.synchronize()
a0 = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
t = time.perf_counter(); out = submission.predict(x); dt = (time.perf_counter() - t) * 1e3
a1 = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
print(f"CHILD wall_ms={dt:.3f} new_device_allocs={a1 - a0}")
'''

def extract(z):
    d = tempfile.mkdtemp(); zipfile.ZipFile(z).extractall(d); return d

args = sys.argv[1:]
n = 10
if '--n' in args:
    i = args.index('--n'); n = int(args[i + 1]); del args[i:i + 2]
elif len(args) == 3 and args[2].isdigit():
    n = int(args.pop())
zs = args
dirs = [extract(z) for z in zs]
res = {z: [] for z in zs}
for i in range(n):
    for z, d in zip(zs, dirs):
        p = subprocess.run([sys.executable, "-c", CHILD, d], capture_output=True, text=True)
        m = re.search(r"replay_dev_ms=([\d.]+) replay_and_d2h_wall_ms=([\d.]+) total_ms=([\d.]+)", p.stderr)
        c = re.search(r"CHILD wall_ms=([\d.]+) new_device_allocs=(-?\d+)", p.stdout)
        if not (m and c):
            print("TRIAL FAILED", z, p.stderr[-400:]); continue
        rd, rw, tot = map(float, m.groups()); wall, na = float(c.group(1)), int(c.group(2))
        res[z].append((rd, rw - rd, tot, wall, na))
for z in zs:
    r = res[z]
    if not r:
        continue
    cols = list(zip(*r))
    f = lambda v: f"{statistics.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]"
    print(f"{z.split('/')[-1]}  n={len(r)}  replay_dev {f(cols[0])}  wall-minus-replay {f(cols[1])}  total {f(cols[2])}  "
          f"outer_wall {f(cols[3])}  new_device_allocs {sorted(set(cols[4]))}")
    slow = sum(v > 3.0 for v in cols[1])
    print(f"    slow trials (wall-minus-replay > 3 ms): {slow}/{len(r)}   values: {' '.join(f'{v:.2f}' for v in cols[1])}")
