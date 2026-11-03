#!/usr/bin/env python
"""Where does the TKE error actually live, and what is the CEILING of fixing it?

``tke_score`` is the weakest accuracy subscore (77.65 on #11 against rel_l2's
94.75) and four queued proposals all assume the same thing -- that the model
under-predicts fluctuation energy and a rescale recovers it. Nobody has priced
the ceiling. This does, on a checkpoint, on CPU, in one job.

The decomposition, per window s, on the kinetic-energy field k (the scorer's
own ``kinetic_energy``: the time-mean is subtracted first, so k is FLUCTUATION
energy):

    rel_s          = ||k_p - k_t|| / ||k_t||                    what we score
    rel_min_s      = sqrt(1 - rho_s^2)                          after the best per-window scalar
    rho_s          = <k_p, k_t> / (||k_p|| ||k_t||)

``rel_s - rel_min_s`` is everything ANY rescaling family can remove; ``rel_min_s``
is decorrelation and no gain touches it. Three gains are then priced, from
unreachable to shippable:

    oracle/window   a*_s per window in KE space          the absolute ceiling
    oracle/row      one a* per Reynolds row              needs Re at test time
    |u|-predicted   a* regressed on mean|u| of the INPUT  SHIPPABLE -- the input is all it uses
    global          one constant for everything          what tke_gain_diagnostic swept

and finally the same thing in VELOCITY space, where a gain g on the fluctuation
scales KE by g^2 but also moves rel_l2 and mvpe -- which is the version we could
actually ship, and the reason the KE-space number is a ceiling and not a plan.

    python scripts/tke_decomposition.py --ckpt 'outputs/runs/<run>-*/best_score.pth'
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pde_competition.paths import KIT, OUTPUTS                      # noqa: E402
from pde_competition.dataset import build_eval_arrays, list_runs    # noqa: E402

sys.path.insert(0, str(KIT))
import scoring as ks                                                # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_bounds import _predict                                     # noqa: E402


def ke(a, c):
    return ks.kinetic_energy(a[..., :c]).reshape(a.shape[0], -1)


def fluct_gain(pred, g, c):
    """Scale the prediction's TEMPORAL FLUCTUATION by g, leaving the time-mean alone.

    This is the shippable knob: KE is built from the fluctuation, so KE scales as
    g^2, while the mean flow -- which rel_l2 and mvpe mostly measure -- is untouched.
    """
    out = pred.copy()
    m = out[..., :c].mean(axis=1, keepdims=True)
    out[..., :c] = m + g * (out[..., :c] - m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--re", nargs="+", type=int, default=[3750, 8850, 13950, 19050, 26700])
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(OUTPUTS / "tke_decomposition.json"))
    a = ap.parse_args()

    # arrays_for takes a PATH, not a pattern: torch.load does not glob. It also
    # returns one key per RUN, and every per-window statistic below needs one key
    # per WINDOW -- hence build_eval_arrays(return_keys=True) here rather than
    # arrays_for, which is what score_local.per_row_scores does for the same reason.
    hits = sorted(Path().glob(a.ckpt)) if any(ch in a.ckpt for ch in "*?[") else [Path(a.ckpt)]
    if len(hits) != 1:
        raise SystemExit(f"--ckpt must resolve to exactly one file, got {len(hits)}: {hits[:4]}")
    runs = [r for r in list_runs("train_real") if r.re_nominal in a.re]
    if not runs:
        raise SystemExit(f"no trajectories with Re in {a.re}")
    x, y, keys = build_eval_arrays(runs, stride=a.stride, return_keys=True)
    pred = _predict(hits[0], None, x, a.device)
    assert len(keys) == pred.shape[0], (len(keys), pred.shape)
    c = ks.measured_channels(y)
    kp, kt = ke(pred, c), ke(y, c)
    nt = np.linalg.norm(kt, axis=1).clip(min=1e-8)
    npd = np.linalg.norm(kp, axis=1).clip(min=1e-8)
    dot = (kp * kt).sum(axis=1)
    rho = dot / (npd * nt)
    rel = np.linalg.norm(kp - kt, axis=1) / nt
    rel_min = np.sqrt(np.clip(1.0 - rho ** 2, 0.0, None))
    astar = dot / (npd ** 2)                       # the per-window optimal KE gain
    rows = np.array([k[0] for k in keys])
    # mean |u| of the INPUT -- the only row information a shipped model has
    umag = np.abs(x[..., 0]).mean(axis=(1, 2, 3))

    def score_with(k_scaled):
        return ks.score_error(float(np.mean(np.linalg.norm(k_scaled - kt, axis=1) / nt)))

    print(f"windows {len(rel)}  rows {sorted(set(rows))}  measured channels {c}")
    print(f"\nCURRENT           tke_rel_l2 {rel.mean():.5f}   tke_score {ks.score_error(float(rel.mean())):.3f}")
    print(f"DECORRELATION     sqrt(1-rho^2) {rel_min.mean():.5f}  <- no rescale of any kind touches this")
    print(f"GAIN-REMOVABLE    {rel.mean() - rel_min.mean():.5f}  ({100*(rel.mean()-rel_min.mean())/rel.mean():.1f}% of the error)")
    print(f"mean optimal KE gain a* {astar.mean():.4f} (spread {astar.min():.3f}-{astar.max():.3f})"
          f"  -> velocity-space g = sqrt(a*) = {np.sqrt(astar.mean()):.4f}")

    print("\nWHAT EACH GAIN FAMILY BUYS, in KE space (a ceiling: velocity space cannot reach it)")
    print(f"  {'family':<22} {'tke_rel_l2':>11} {'tke_score':>10} {'d(score)':>9}")
    base = ks.score_error(float(rel.mean()))
    fam = {}
    fam["none (current)"] = kp
    fam["global a*"] = kp * (dot.sum() / (npd ** 2).sum())
    per_row = np.array([astar[rows == r].mean() for r in rows])
    fam["per Reynolds row"] = kp * per_row[:, None]
    A = np.polyfit(umag, astar, 1)
    fam["|u|-predicted (ship)"] = kp * np.polyval(A, umag)[:, None]
    fam["oracle per window"] = kp * astar[:, None]
    for nm, kk in fam.items():
        s = score_with(kk)
        print(f"  {nm:<22} {float(np.mean(np.linalg.norm(kk-kt,axis=1)/nt)):11.5f} {s:10.3f} {s-base:+9.3f}")

    print("\nVELOCITY SPACE -- the shippable version, and its cost on the other subscores")
    print(f"  {'g':>6} {'rel_l2':>8} {'tke':>8} {'mvpe':>8}")
    vel = {}
    for g in [1.0, np.sqrt(float(np.polyval(A, umag).mean())), float(np.sqrt(astar.mean()))]:
        p2 = fluct_gain(pred, g, c)
        r1 = ks.score_error(float(np.mean(ks.rel_l2_per_sample(p2, y, c))))
        t1 = ks.score_error(float(np.mean(ks.tke_rel_l2_per_sample(p2, y, c))))
        m1 = ks.score_error(ks.mvpe_rel_l2(p2, y))
        print(f"  {g:6.4f} {r1:8.3f} {t1:8.3f} {m1:8.3f}")
        vel[f"g={g:.4f}"] = dict(rel_l2=r1, tke=t1, mvpe=m1)

    print("\nWHERE THE IRREDUCIBLE PART LIVES")
    resid = kp * astar[:, None] - kt                 # after the oracle gain
    per_cell = (resid ** 2).mean(axis=0).reshape(y.shape[2], y.shape[3])
    tot = per_cell.sum()
    order = np.sort(per_cell.ravel())[::-1]
    for q in (0.05, 0.10, 0.25):
        n = max(1, int(q * per_cell.size))
        print(f"  top {q:4.0%} of cells carry {100*order[:n].sum()/tot:5.1f}% of the residual energy")
    cols = per_cell.mean(axis=0)
    print(f"  worst 5 columns (streamwise x): {np.argsort(cols)[::-1][:5].tolist()} of {len(cols)}")
    print(f"  wake band rows (y): {np.argsort(per_cell.mean(axis=1))[::-1][:5].tolist()} of {per_cell.shape[0]}")

    print(f"\n  {'row':>6} {'n':>4} {'rel':>8} {'1-rho^2':>9} {'a*':>7} {'mean|u|':>8}")
    out_rows = {}
    for r in sorted(set(rows)):
        m = rows == r
        out_rows[str(r)] = dict(n=int(m.sum()), rel=float(rel[m].mean()),
                                rel_min=float(rel_min[m].mean()), astar=float(astar[m].mean()),
                                umag=float(umag[m].mean()))
        print(f"  {r:>6} {m.sum():4d} {rel[m].mean():8.5f} {rel_min[m].mean():9.5f} "
              f"{astar[m].mean():7.4f} {umag[m].mean():8.4f}")

    Path(a.out).write_text(json.dumps(dict(
        ckpt=a.ckpt, windows=int(len(rel)), tke_rel_l2=float(rel.mean()),
        decorrelation=float(rel_min.mean()), gain_removable=float(rel.mean() - rel_min.mean()),
        astar_mean=float(astar.mean()), umag_fit=[float(A[0]), float(A[1])],
        families={k: float(np.mean(np.linalg.norm(v - kt, axis=1) / nt)) for k, v in fam.items()},
        velocity=vel, per_row=out_rows), indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
