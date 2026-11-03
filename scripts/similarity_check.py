#!/usr/bin/env python
"""Reynolds-similarity sanity check (fresh-eyes proposition 7b, 2026-09-17).

If a row at Re, read with frame step 2 and amplitude x2, looks like the row at
2*Re, then the rows we own can manufacture the hidden extrapolation regime
(2 x 13950 = 27900 ~ the hidden 27975). Before training anything on that
idea, compare, for every real (Re, ~2Re) pair at the same angle:
    mean |u|, mean |v|, TKE level (time variance of u,v averaged over pixels),
    and the 20-frame persistence rel-L2 (scale-free dynamics check)
of the EMULATED row (low Re, step 2, x2) against the REAL high-Re row.
Ratios near 1 (within ~10%) say the similarity holds well enough to try
augmentation (b); a persistence ratio far from 1 says the time scale does not
double with Re and the emulation is wrong.

    python scripts/similarity_check.py [--frames 400]
CPU, a minute.
"""
import argparse
import numpy as np

from pde_competition.dataset import list_runs

T_IN, T_OUT = 20, 20


def stats(arr: np.ndarray, step: int, amp: float, frames: int):
    """arr (T,H,W,C) memmap -> dict of level and dynamics statistics on frames[::step] x amp."""
    a = np.asarray(arr[: frames * step: step, ..., :2], dtype=np.float32) * amp
    meas = a[..., 0] != 0
    u, v = a[..., 0], a[..., 1]
    mean_abs_u = float(np.abs(u[meas]).mean()); mean_abs_v = float(np.abs(v[meas]).mean())
    # TKE level: time variance per pixel of the measured field, averaged
    m = a.mean(0, keepdims=True)
    tke = float((0.5 * ((a - m) ** 2).sum(-1))[meas].mean())
    # persistence rel-L2 over a 20-frame horizon: ||x[t+k] - x[t]|| / ||x[t+k]|| averaged over starts
    errs = []
    for s in range(0, a.shape[0] - T_IN - T_OUT, 10):
        last = a[s + T_IN - 1]
        tgt = a[s + T_IN: s + T_IN + T_OUT]
        errs.append(float(np.linalg.norm(tgt - last[None]) / max(np.linalg.norm(tgt), 1e-8)))
    return dict(mean_abs_u=mean_abs_u, mean_abs_v=mean_abs_v, tke=tke, persistence=float(np.mean(errs)) if errs else float("nan"))


def main(a):
    runs = list_runs("train_real")
    by_key = {r.key: r for r in runs}
    res_rows = sorted({r.re_nominal for r in runs})
    pairs = []
    for lo in res_rows:
        cands = [hi for hi in res_rows if abs(2 * lo - hi) / hi < 0.02]
        for hi in cands:
            pairs.append((lo, hi))
    print(f"(Re, ~2Re) pairs among real rows: {pairs}")
    print(f"{'lo':>6} {'hi':>6} {'aoa':>4} | {'|u| emu/real':>12} {'|v| emu/real':>12} {'tke emu/real':>12} {'persist emu':>11} {'persist real':>12} {'ratio':>6}")
    agg = []
    for lo, hi in pairs:
        for aoa in sorted({k[1] for k in by_key if k[0] == lo}):
            if (hi, aoa) not in by_key:
                continue
            e = stats(by_key[(lo, aoa)].load(), step=2, amp=2.0, frames=a.frames)
            r = stats(by_key[(hi, aoa)].load(), step=1, amp=1.0, frames=a.frames)
            row = (e["mean_abs_u"] / r["mean_abs_u"], e["mean_abs_v"] / r["mean_abs_v"], e["tke"] / r["tke"],
                   e["persistence"], r["persistence"], e["persistence"] / r["persistence"])
            agg.append(row)
            print(f"{lo:>6} {hi:>6} {aoa:>4} | {row[0]:12.3f} {row[1]:12.3f} {row[2]:12.3f} {row[3]:11.4f} {row[4]:12.4f} {row[5]:6.3f}")
    if agg:
        m = np.array(agg).mean(0)
        print(f"\nmean over {len(agg)} pairs: |u| {m[0]:.3f}  |v| {m[1]:.3f}  tke {m[2]:.3f}  persistence ratio {m[5]:.3f}")
        print("read: ratios within 0.9..1.1 on |u|, tke and persistence -> similarity holds, augmentation (b) is worth training;"
              " persistence ratio far from 1 -> the time scale does not follow 2x Re, drop (b).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400, help="frames of the high-Re row used (the low-Re row uses 2x as many, stepped)")
    main(ap.parse_args())
