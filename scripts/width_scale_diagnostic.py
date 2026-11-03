"""Does the trained width head under-widen on rows it never validated on?

Fresh-eyes proposition 3 asked for a global width scale k and priced it as a
board day: ship k > 1 and see what the hidden rows pay. That is expensive (one
upload) and, since 2026-09-18, unreadable in the time subscore -- but the
question itself is local. The head is re-levelled on --val-re (13950/26700) by
construction, so k = 1 is optimal THERE by definition. What nobody has measured
is the best k on the other fifteen rows, which stand in for the hidden ones.

This walks the out-of-fold ensembles (the same data path train_width_head fits
on), applies a trained head, and reports the best global k per row and channel
against fit_bounds.objective_map -- the exact scorer objective, not a surrogate.

    python scripts/width_scale_diagnostic.py --head outputs/width_head/head_ext_tau05_w48.pth \
        --folds 'outputs/runs/ws_cfk_fold*/best_score.pth' --stride 20

Reads: a best k near 1.0 on the unseen rows means the head generalises its
LEVEL and proposition 3 closes without a board day. A best k systematically
above 1 means the head under-widens away from its validation pair, and the
gain from the pooled best k is what a board day would have bought.
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


def head_widths(ck, rows, speeds, extras, re, device):
    """Half-widths (N, T, H, W, 2) from a trained head for one row."""
    m, z, _ = rows[re]
    ext = ck["planes"] == "ext"
    n_out = ck.get("n_out", 2)          # 2 = symmetric half-widths; 4 = H1's lo_u, lo_v, up_u, up_v (2026-09-22)
    head = WidthHead(ck["width"], ck["n_feat"], ck.get("n_pfeat", 0), ck.get("n_bottle", 8), n_out=n_out).to(device)
    head.load_state_dict(ck["state_dict"]); head.eval()
    f = features(m, z, speeds[re], ck["norm"], extras.get(re) if ext else None)
    if ck.get("drop_z"):
        f[:, 2:4] = 0.0
    with torch.no_grad():
        h = torch.cat([head(f[i:i + 256].to(device)).cpu() for i in range(0, f.shape[0], 256)])
    return h.numpy().reshape(m.shape[0], m.shape[1], n_out, *m.shape[2:4]).transpose(0, 1, 3, 4, 2)


def objective_asym(pred, target, lo, up, sigma=None):
    """The exact per-element SPS width reward for an ASYMMETRIC interval [pred - lo, pred + up]:
    mean over scored elements of 1{inside} * exp(-(lo + up) / sigma). Reduces to objective_map when
    lo == up. Written for the H1 head (n_out = 4); kit/scoring.py never compares the interval to pred,
    so this is the scorer's own form with the centre at pred."""
    from fit_bounds import SIGMA_GLOBAL
    sigma = SIGMA_GLOBAL if sigma is None else sigma
    scored = target != 0.0
    n = int(np.count_nonzero(scored))
    if n == 0:
        return float("nan"), float("nan")
    lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), pred.shape)
    up = np.broadcast_to(np.asarray(up, dtype=np.float64), pred.shape)
    inside = (target >= pred - lo) & (target <= pred + up)
    val = np.where(inside, np.exp(-(lo + up) / sigma), 0.0)
    return float(val[scored].mean()), float(inside[scored].mean())


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.head, map_location="cpu", weights_only=False)
    if ck.get("n_pfeat", 0):
        sys.exit("this head needs the model's penultimate planes; not supported here")
    folds = sorted(p for g in a.folds for p in glob.glob(g))
    print(f"head {a.head} (planes {ck['planes']}, width {ck['width']}, val rows {ck['val_rows']})")
    print(f"{len(folds)} fold checkpoints; stride {a.stride}; device {device}\n")
    rows = collect_rows(folds, a.stride, device, pool_members=a.pool_members)

    from pde_competition.dataset import build_eval_arrays, list_runs
    speeds, extras = {}, {}
    for re in rows:
        runs = [r for r in list_runs("train_real") if r.re_nominal == re]
        x, _ = build_eval_arrays(runs, stride=a.stride)
        rep = rows[re][0].shape[0] // x.shape[0]          # members pooled per window (1 unless --pool-members)
        if rep > 1:
            x = np.tile(x, (rep, 1, 1, 1, 1))
        speeds[re] = window_speed(x)
        if ck["planes"] == "ext":
            extras[re] = extra_planes(x, rows[re][0])

    ks = np.round(np.arange(a.k_min, a.k_max + 1e-9, a.k_step), 3)
    val = set(ck["val_rows"])
    per_row, tot = {}, {k: [0.0, 0] for k in ks}
    print(f"{'row':>7s} {'seen':>5s} {'ch':>3s} {'obj at k=1':>11s} {'best k':>7s} {'obj at best':>12s} {'gain':>8s}")
    for re in sorted(rows):
        m, _, y = rows[re]
        h = head_widths(ck, rows, speeds, extras, re, device)
        for c, name in enumerate(["u", "v"]):
            n_sc = int(np.count_nonzero(y[..., c] != 0))
            if h.shape[-1] == 4:      # asymmetric head: lo = planes 0:2, up = planes 2:4
                obj = lambda k, c=c: objective_asym(m[..., c], y[..., c], k * h[..., c], k * h[..., 2 + c])[0]
            else:
                obj = lambda k, c=c: objective_map(m[..., c], y[..., c], k * h[..., c])[0]
            objs = {k: obj(k) for k in ks}
            o1 = objs[1.0] if 1.0 in objs else obj(1.0)
            bk = max(objs, key=objs.get)
            for k in ks:
                tot[k][0] += objs[k] * n_sc; tot[k][1] += n_sc
            per_row.setdefault(re, {})[name] = (o1, bk, objs[bk])
            print(f"{re:7d} {'yes' if re in val else 'no':>5s} {name:>3s} {o1:11.4f} {bk:7.2f} {objs[bk]:12.4f} {objs[bk]/o1-1:+7.2%}")

    print("\npooled over channels and rows, by k:")
    base = tot[1.0][0] / tot[1.0][1]
    for k in ks:
        v = tot[k][0] / tot[k][1]
        print(f"  k {k:4.2f}   {v:.4f}  ({v / base - 1:+.2%}){'   <- best' if v == max(t[0]/t[1] for t in tot.values()) else ''}")

    seen = [per_row[r][c][1] for r in per_row if r in val for c in ("u", "v")]
    unseen = [per_row[r][c][1] for r in per_row if r not in val for c in ("u", "v")]
    print(f"\nbest k on the head's OWN validation rows : mean {np.mean(seen):.3f}  ({sorted(seen)})")
    print(f"best k on rows it never validated on     : mean {np.mean(unseen):.3f}  n={len(unseen)}")
    hi = sorted(r for r in per_row if r not in val)[-3:]
    print(f"best k on the three highest unseen rows {hi}: "
          f"{[per_row[r][c][1] for r in hi for c in ('u', 'v')]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--head", required=True)
    p.add_argument("--folds", nargs="+", required=True)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--k-min", type=float, default=0.80)
    p.add_argument("--k-max", type=float, default=1.40)
    p.add_argument("--k-step", type=float, default=0.05)
    p.add_argument("--pool-members", action="store_true",
                   help="score every fold member's OWN prediction as a sample instead of the member mean: "
                        "the residual family a SINGLE shipped checkpoint has (needs a coarse --stride; "
                        "the pooled arrays are ~6x the memory)")
    main(p.parse_args())
