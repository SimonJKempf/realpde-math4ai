"""Disagreement widths validated OUT OF FOLD on all 18 Reynolds rows.

Widths of the form h = a + b*z (z = per-element std over ensemble members),
validated honestly. This uses the cross-fit fold ensembles (three seeds per
fold, each fold holding out a third of the rows) so every row has a member mean,
a member disagreement and a target from models that never saw it. Then:
  * leave-one-row-out over the rows in --loo-rows (fit a,b on the other 17,
    score the left-out row) for lin_z and lin_z x shape;
  * a fit on all 18 rows, reported next to a one-row fit so one can see
    whether a single-row fit is representative.

train_width_head.py imports ``collect`` and ``fit_ab`` from here.

    python scripts/oof_disagreement.py --folds 'outputs/runs/ws_cfk_fold*/best_score.pth' \
        --loo-rows 3750 13950 26700 --stride 40
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch
from pde_competition.dataset import build_eval_arrays, list_runs
from pde_competition.model import FlowerS2S, predict_numpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_bounds import fit_channel, objective_map  # noqa: E402
from oof_width_model import GRID, window_speed  # noqa: E402

A_GRID = np.linspace(0.0, 0.02, 21)
B_GRID = np.linspace(0.0, 0.03, 7)


def predict_with_feats(model, x, batch_size, device):
    """predict_numpy + the model's PENULTIMATE feature map (fresh-eyes proposition 8).

    The hook sits on the ReLU inside Flower's final projection, so the captured tensor is the
    last representation before the 1x1 conv that writes the prediction: (B, lifting_dim, H, W),
    one map per WINDOW (the core emits all t_out frames from this single map). Returned as
    float16 -- 128 planes per window is the largest array in the head's data path and the head
    reduces it through a learned 1x1 bottleneck anyway.
    """
    grabbed = []
    h = model.core.project[1].register_forward_hook(lambda _m, _i, o: grabbed.append(o.detach()))
    try:
        cfg = model.cfg
        model.eval().to(device)
        x = np.asarray(x, dtype=np.float32)
        pred = np.zeros((x.shape[0], cfg.t_out, *cfg.resolution, 3), np.float32)
        feats = None
        for i in range(0, x.shape[0], batch_size):
            grabbed.clear()
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch_size, ..., :cfg.c_in]))
            with torch.no_grad():
                yb = model(xb.to(device)).float().cpu().numpy()
            pred[i:i + batch_size, ..., :cfg.c_out] = yb
            # under rollout the hook fires once per pass; the first pass is the one that saw the real window
            fb = grabbed[0].float().cpu().numpy().astype(np.float16)
            if feats is None:
                feats = np.zeros((x.shape[0], fb.shape[1], *fb.shape[2:]), np.float16)
            feats[i:i + batch_size] = fb
    finally:
        h.remove()
    return pred, feats


def collect(fold_ckpts, stride, device, pool_members=False, with_feats=False):
    """rows: re -> (mean (N,T,H,W,2), std (N,T,H,W,2), y (N,T,H,W,2)) from the fold that held re out.

    pool_members=True (fresh-eyes proposition 4, 2026-09-18): instead of the M-member MEAN per window,
    every member's own prediction is a sample -- (M*N, ...) with the disagreement std repeated and y
    tiled -- so a width head learns the residual family of SINGLE checkpoints (which is what ships),
    not the residual of a 3-mean. Pass best_score.pth AND last.pth per fold run to pool over
    checkpoints as well as seeds.

    with_feats=True (proposition 8) additionally returns feats: re -> (N, lifting_dim, H, W) float16,
    the model's penultimate feature map per window (the member mean, or per member when pooling), in
    the same leading order as rows[re][0]."""
    by_fold = {}
    feats = {}
    for ck in fold_ckpts:
        args = json.load(open(Path(ck).parent / "meta.json"))["launches"][-1]["args"]
        by_fold.setdefault(tuple(sorted(args["holdout_re"])), []).append(ck)
    rows = {}
    for holdout, cks in by_fold.items():
        models = [FlowerS2S.from_checkpoint(c) for c in cks]
        for re in holdout:
            runs = [r for r in list_runs("train_real") if r.re_nominal == re]
            if not runs:
                continue
            x, y = build_eval_arrays(runs, stride=stride)
            if with_feats:
                got = [predict_with_feats(m, x, 16, device) for m in models]
                P = np.stack([g[0][..., :2] for g in got]).astype(np.float32)
                Fm = np.stack([g[1] for g in got])                 # (M, N, F, H, W) float16
            else:
                P = np.stack([predict_numpy(m, x, batch_size=16, device=device)[..., :2] for m in models]).astype(np.float32)
            yy = y[..., :2].astype(np.float32)
            if pool_members:
                M = P.shape[0]
                rows[re] = (P.reshape(-1, *P.shape[2:]), np.tile(P.std(0), (M, 1, 1, 1, 1)), np.tile(yy, (M, 1, 1, 1, 1)))
                if with_feats:
                    feats[re] = Fm.reshape(-1, *Fm.shape[2:])      # same leading order as rows[re][0]
            else:
                rows[re] = (P.mean(0), P.std(0), yy)
                if with_feats:
                    feats[re] = Fm.mean(0).astype(np.float16)      # the mean map, matching the mean prediction
            print(f"  row {re}: {len(cks)} members, {x.shape[0]} windows{' (pooled per member)' if pool_members else ''}", flush=True)
    return (rows, feats) if with_feats else rows


def fit_ab(rows, c, keys):
    """Grid-search (a, b) of h = a + b * z/zref for one channel over the given rows, pooled."""
    P = np.concatenate([rows[k][0][..., c] for k in keys]); Y = np.concatenate([rows[k][2][..., c] for k in keys])
    Z = np.concatenate([rows[k][1][..., c] for k in keys])
    zref = float(Z[Y != 0].mean()); Zn = Z / max(zref, 1e-8)
    best = (-1.0, 0.0, 0.0)
    for b in B_GRID:
        for a in A_GRID:
            o, _ = objective_map(P, Y, a + b * Zn)
            if o > best[0]:
                best = (o, float(a), float(b))
    return best[1], best[2], zref


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = sorted(p for g in a.folds for p in glob.glob(g))
    print(f"{len(folds)} fold checkpoints; stride {a.stride}; device {device}")
    rows = collect(folds, a.stride, device)
    keys = sorted(rows); print(f"{len(keys)} rows with out-of-fold ensembles\n")
    for c, name in enumerate(["u", "v"]):
        Z = np.concatenate([rows[k][1][..., c] for k in keys]); R = np.concatenate([np.abs(rows[k][0][..., c] - rows[k][2][..., c]) for k in keys])
        S = np.concatenate([rows[k][2][..., c] for k in keys]) != 0
        print(f"channel {name}: corr(z, |r|) over all 18 rows = {np.corrcoef(Z[S], R[S])[0, 1]:.3f}")
    loo_rows = [r for r in (a.loo_rows or keys) if r in rows]
    print("\nLEAVE-ONE-ROW-OUT (fit a,b on the other rows; pooled over both channels):")
    tot = {"const": [0.0, 0], "lin_z": [0.0, 0]}
    for re in loo_rows:
        train = [k for k in keys if k != re]
        num = {"const": 0.0, "lin_z": 0.0}; den = 0
        for c in range(2):
            p, z, y = rows[re][0][..., c], rows[re][1][..., c], rows[re][2][..., c]
            n = int(np.count_nonzero(y != 0))
            h0 = fit_channel(np.concatenate([rows[k][0][..., c] for k in train]), np.concatenate([rows[k][2][..., c] for k in train]), GRID)[0]
            aa, bb, zref = fit_ab(rows, c, train)
            num["const"] += objective_map(p, y, h0)[0] * n
            num["lin_z"] += objective_map(p, y, aa + bb * z / zref)[0] * n
            den += n
        for k in num:
            tot[k][0] += num[k] / den; tot[k][1] += 1
        print(f"  row {re:>5}: const {num['const'] / den:.4f}  lin_z {num['lin_z'] / den:.4f}  ({num['lin_z'] / num['const'] - 1:+.2%})", flush=True)
    print(f"  mean over rows: const {tot['const'][0] / tot['const'][1]:.4f}  lin_z {tot['lin_z'][0] / tot['lin_z'][1]:.4f}  "
          f"({tot['lin_z'][0] / tot['const'][0] - 1:+.2%})")
    print("\nFIT ON ALL 18 ROWS (compare with W44's one-row fit: u a=0.0045 b=0.0050, v a=0.0035 b=0.0025):")
    for c, name in enumerate(["u", "v"]):
        aa, bb, zref = fit_ab(rows, c, keys)
        print(f"  {name}: a={aa:.4f} b={bb:.4f} zref={zref:.5f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", nargs="+", required=True)
    p.add_argument("--loo-rows", type=int, nargs="*", default=None)
    p.add_argument("--stride", type=int, default=40)
    main(p.parse_args())
