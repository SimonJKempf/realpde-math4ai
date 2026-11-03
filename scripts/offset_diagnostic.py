#!/usr/bin/env python
"""Should the uncertainty interval be centred on the prediction at all?

The scorer pays 1{lower <= target <= upper} * exp(-2*(upper-lower)/sigma): it
charges for the WIDTH and rewards covering the target. Nothing requires the
interval to be centred on the point prediction. For a fixed width, coverage is
maximised by centring on the conditional mean of the TARGET, which is the
prediction plus whatever conditional bias the model has.

W154 measured exactly such a bias, and that is what makes this worth testing
now. Over the predicted fluctuation f = p - mean_t(p), the model's KE level is
a = 0.725 of the target's on row 26700: the model UNDER-DISPERSES by ~27%. If
target ~ mean + f/a, then target - p ~ (1/a - 1) * f ~ 0.38 * f -- a bias that
is conditionally signed WITH the local fluctuation, and therefore invisible to
any static per-pixel map. W48 closed "asymmetric offsets" by measuring the mean
per-pixel residual (0.0008 u, 0.0003 v) and concluding the residuals are not
systematically signed -- but a bias that flips sign with the shedding phase
averages to ~0 in exactly that statistic, so W48's evidence does not bear on a
CONDITIONAL offset.

Why this is not W154's failed gain. Scaling the prediction (p -> mean + g*f)
moves the POINT prediction off the conditional mean, which costs rel_l2 and
tke -- W154 measured that cost and it exceeded the sps gain. Offsetting only
the INTERVAL leaves the prediction, and therefore rel_l2, tke and mvpe, exactly
untouched. It is free by construction if it works at all.

    python scripts/offset_diagnostic.py --head outputs/width_head/head_ext_tau05_w48.pth \
        --folds 'outputs/runs/ws_cfk_fold*/best_score.pth' --stride 40

Reads: a best beta at 0 confirms W48 on the conditional statistic too and
closes the centring axis for good. A best beta near 0.3-0.4 with a real gain
says the interval should be shifted, and the shift costs nothing anywhere else.
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_bounds import objective_map  # noqa: E402
from oof_disagreement import collect as collect_rows  # noqa: E402
from oof_width_model import window_speed  # noqa: E402
from train_width_head import WidthHead, features, extra_planes  # noqa: E402


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.head, map_location="cpu", weights_only=False)
    folds = sorted(p for g in a.folds for p in glob.glob(g))
    print(f"head {a.head} (planes {ck['planes']}, width {ck['width']}, val rows {ck['val_rows']})")
    print(f"{len(folds)} fold checkpoints; stride {a.stride}; device {device}\n")
    rows = collect_rows(folds, a.stride, device)

    from pde_competition.dataset import build_eval_arrays, list_runs
    speeds, extras = {}, {}
    for re in rows:
        runs = [r for r in list_runs("train_real") if r.re_nominal == re]
        x, _ = build_eval_arrays(runs, stride=a.stride)
        speeds[re] = window_speed(x)
        if ck["planes"] == "ext":
            extras[re] = extra_planes(x, rows[re][0])

    head = WidthHead(ck["width"], ck["n_feat"], ck.get("n_pfeat", 0), ck.get("n_bottle", 8)).to(device)
    head.load_state_dict(ck["state_dict"]); head.eval()

    betas = np.round(np.arange(a.beta_min, a.beta_max + 1e-9, a.beta_step), 3)
    val = set(ck["val_rows"])
    tot = {b: [0.0, 0] for b in betas}
    print(f"{'row':>7s} {'seen':>5s} {'ch':>3s} {'obj at b=0':>11s} {'best b':>7s} {'obj at best':>12s} {'gain':>8s}")
    per_row = {}
    for re in sorted(rows):
        m, z, y = rows[re]
        f = features(m, z, speeds[re], ck["norm"], extras.get(re) if ck["planes"] == "ext" else None)
        if ck.get("drop_z"):
            f[:, 2:4] = 0.0
        with torch.no_grad():
            h = torch.cat([head(f[i:i + 256].to(device)).cpu() for i in range(0, f.shape[0], 256)])
        h = h.numpy().reshape(m.shape[0], m.shape[1], 2, *m.shape[2:4]).transpose(0, 1, 3, 4, 2)
        fluct = m - m.mean(1, keepdims=True)          # (N,T,H,W,2) the predicted fluctuation
        for c, name in enumerate(["u", "v"]):
            n_sc = int(np.count_nonzero(y[..., c] != 0))
            objs = {}
            for b in betas:
                shifted = m[..., c] + b * fluct[..., c]
                objs[b] = objective_map(shifted, y[..., c], h[..., c])[0]
            o0 = objs[0.0]
            bb = max(objs, key=objs.get)
            for b in betas:
                tot[b][0] += objs[b] * n_sc; tot[b][1] += n_sc
            per_row.setdefault(re, {})[name] = bb
            print(f"{re:7d} {'yes' if re in val else 'no':>5s} {name:>3s} {o0:11.4f} {bb:7.2f} {objs[bb]:12.4f} {objs[bb]/o0-1:+7.2%}")

    print("\npooled over channels and rows, by beta:")
    base = tot[0.0][0] / tot[0.0][1]
    best = max(betas, key=lambda b: tot[b][0] / tot[b][1])
    for b in betas:
        v = tot[b][0] / tot[b][1]
        print(f"  beta {b:5.2f}   {v:.4f}  ({v / base - 1:+.2%}){'   <- best' if b == best else ''}")
    seen = [per_row[r][c] for r in per_row if r in val for c in ("u", "v")]
    unseen = [per_row[r][c] for r in per_row if r not in val for c in ("u", "v")]
    print(f"\nbest beta on the head's validation rows: mean {np.mean(seen):+.3f}")
    print(f"best beta on every other row           : mean {np.mean(unseen):+.3f}  n={len(unseen)}")
    print(f"pooled best beta {best:+.2f}, gain {tot[best][0]/tot[best][1]/base - 1:+.2%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--head", required=True)
    p.add_argument("--folds", nargs="+", required=True)
    p.add_argument("--stride", type=int, default=40)
    p.add_argument("--beta-min", type=float, default=-0.4)
    p.add_argument("--beta-max", type=float, default=0.6)
    p.add_argument("--beta-step", type=float, default=0.1)
    main(p.parse_args())
