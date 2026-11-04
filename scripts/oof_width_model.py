"""SPS width model fitted on OUT-OF-FOLD residuals from every Reynolds row.

Constant widths fitted on two held-out rows (13950, 26700) are the simplest
honest choice, because those are the only cells a model trained on the rest has
out-of-sample residuals on. Two observations motivate more: the Reynolds-LEVEL
of the width matters more than its spatial structure, and the optimal width
tracks the window's mean |u| at r = 0.98 across rows. The hidden test rows are
15225 (interpolation) and 27975 (extrapolation past every real row). A width
model fitted on 18 rows can be validated for extrapolation by leaving the top
row out; one fitted on 2 rows cannot.

Inputs: K fold checkpoints, each trained with --holdout-re on disjoint row sets
covering all 18. Every row is predicted by the fold that never saw it.

Width models, per channel c, for window w with mean input speed m_w and pixel p:
    const        h = h0                       one number
    affine       h = a + b*m_w                fitted on 18 rows
    shape        h = (a + b*m_w) * s[p]       times a Re-free per-pixel shape
    blend        h = (a + b*m_w) * (0.5*s[p] + 0.5)
Validation is leave-one-row-out over the 18 rows: fit on 17, score the
width reward on the 18th. The rows 3750 and 26700 are the extrapolation tests.

Shipping: fit on all 18 rows, then RE-LEVEL on the shipped checkpoint's own
honest rows (13950, 26700) with one scalar per channel -- the folds trained on
12 rows each and their residuals are larger than the shipped model's. The
result is written beside the shipped checkpoint as bounds.json with a
`half_width_affine_map` block that submission_flower.py applies per window.

    python scripts/oof_width_model.py --folds outputs/runs/ws_cf_fold*/best_score.pth \
        --ship outputs/runs/ws_ft_d128_str1_s1-*/best_score.pth --stride 20
"""
from __future__ import annotations
import argparse, glob, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_bounds as fb                                   # noqa: E402
from fit_bounds import SIGMA_GLOBAL, fit_channel, fit_map, objective_map   # noqa: E402

GRID = np.concatenate([np.linspace(1e-4, 0.02, 400), np.linspace(0.02, 0.20, 200)])
NAMES = ["u", "v"]


# ----------------------------------------------------------------- pure numpy
def window_speed(x):
    """Mean |u| over measured points of each input window: (N, T, H, W, C) -> (N,)."""
    u = x[..., 0]
    meas = u != 0
    return ((np.abs(u) * meas).sum(axis=(1, 2, 3)) / np.maximum(meas.sum(axis=(1, 2, 3)), 1)).astype(np.float32)


def fit_affine(rows, c):
    """Least-squares (a, b) of each row's optimal constant width on its mean speed."""
    hs, ms = [], []
    for pred, y, m in rows.values():
        if not np.any(y[..., c] != 0.0):
            continue
        hs.append(fit_channel(pred[..., c], y[..., c], GRID)[0]); ms.append(float(m.mean()))
    A = np.stack([np.ones(len(ms)), np.asarray(ms)], 1)
    (a, b), *_ = np.linalg.lstsq(A, np.asarray(hs), rcond=None)
    return float(a), float(b)


def level(a, b, m):
    return np.maximum(a + b * np.asarray(m, dtype=np.float64), 1e-4)


def fit_shape(rows, c, a, b):
    """Re-free per-pixel shape: pooled optimal map after bringing every row to one level.

    Each window's residuals are rescaled by g_ref / g_w, i.e. to the scale of a
    REFERENCE level, not to unity: the width reward exp(-2h/sigma) uses the
    scorer's fixed sigma, so it is not scale-invariant, and residuals rescaled
    to O(1) would make every pixel's optimum collapse onto its smallest
    residuals (found on synthetic rows 2026-09-15: the shape came out flat).
    The map fitted at the reference level is then normalised to mean 1 over
    scored pixels, so it carries shape only and the level stays in (a, b, k).
    """
    levels = [level(a, b, m) for _, _, m in rows.values()]
    g_ref = float(np.mean(np.concatenate(levels)))
    preds, ys = [], []
    for (pred, y, m), g in zip(rows.values(), levels):
        scale = (g_ref / g)[:, None, None, None].astype(np.float32)        # per window
        preds.append(y[..., c] + (pred[..., c] - y[..., c]) * scale); ys.append(y[..., c])
    P, Y = np.concatenate(preds), np.concatenate(ys)
    hmap = fit_map(P, Y)                                                   # (H, W), at the reference level
    scored = np.any(Y != 0.0, axis=(0, 1))
    ref = float(hmap[scored].mean()) if scored.any() else 1.0
    s = np.where(scored, hmap / max(ref, 1e-8), 1.0).astype(np.float32)
    return s


def widths(model, m, c):
    """Per-window per-pixel half-widths (N, 1, H, W) for one channel."""
    a, b = model["a"][c], model["b"][c]
    g = level(a, b, m)[:, None, None, None]
    s = np.asarray(model["shape"])[..., c]
    alpha = model["alpha"]
    return (model["k"][c] * g * (alpha * s + (1.0 - alpha))).astype(np.float32)


def fit_model(rows, alpha):
    model = {"a": [], "b": [], "k": [1.0, 1.0], "alpha": alpha, "shape": None}
    shapes = []
    for c in range(2):
        a, b = fit_affine(rows, c)
        model["a"].append(a); model["b"].append(b)
        shapes.append(fit_shape(rows, c, a, b))
    model["shape"] = np.stack(shapes, -1)                                   # (H, W, 2)
    return model


def fit_const(rows, c):
    P = np.concatenate([p[..., c] for p, _, _ in rows.values()])
    Y = np.concatenate([y[..., c] for _, y, _ in rows.values()])
    return fit_channel(P, Y, GRID)[0]


def score(pred, y, m, h_fn):
    """Pooled width reward over both channels for one row, h_fn(c) -> broadcastable widths."""
    num = den = 0.0
    for c in range(2):
        if not np.any(y[..., c] != 0.0):
            continue
        o, _ = objective_map(pred[..., c], y[..., c], h_fn(c))
        n = int(np.count_nonzero(y[..., c] != 0.0)); num += o * n; den += n
    return num / max(den, 1)


def leave_one_row_out(rows, alphas=(1.0, 0.5), only=None):
    """Return {re: {model_name: pooled val objective}} fitting on the other rows.

    ``only`` restricts which rows are left out (each is a full refit on the
    other 17 and costs ~3.5 CPU-minutes at stride 20). The default is all 18;
    W31 (2026-09-15) showed that occupies a 4090 for an hour doing numpy, so
    pass the rows that decide something -- the two ends and the two standard
    held-out rows -- when a card is waiting.
    """
    out = {}
    for re in sorted(rows):
        if only is not None and re not in only:
            continue
        train = {k: v for k, v in rows.items() if k != re}
        pred, y, m = rows[re]
        res = {}
        hc = [fit_const(train, c) for c in range(2)]
        res["const"] = score(pred, y, m, lambda c: hc[c])
        ab = [fit_affine(train, c) for c in range(2)]
        res["affine"] = score(pred, y, m, lambda c: level(*ab[c], m)[:, None, None, None])
        for alpha in alphas:
            mdl = fit_model(train, alpha)
            res[f"shape_a{alpha:g}"] = score(pred, y, m, lambda c: widths(mdl, m, c))
        out[re] = res
        print(f"  LOO row {re:>5}: " + "  ".join(f"{k} {v:.4f}" for k, v in res.items())
              + f"   (affine {res['affine'] / res['const'] - 1:+.2%}, "
              + ", ".join(f"{k} {res[k] / res['const'] - 1:+.2%}" for k in res if k.startswith('shape')) + ")",
              flush=True)
    return out


def relevel(model, ship_rows):
    """One scalar per channel maximising the pooled reward on the shipped model's honest rows."""
    ks = np.linspace(0.6, 1.6, 101)
    for c in range(2):
        best, bk = -1.0, 1.0
        for k in ks:
            mdl = dict(model, k=[k if i == c else 1.0 for i in range(2)])
            tot = num = 0.0
            for pred, y, m in ship_rows.values():
                if not np.any(y[..., c] != 0.0):
                    continue
                o, _ = objective_map(pred[..., c], y[..., c], widths(mdl, m, c))
                n = int(np.count_nonzero(y[..., c] != 0.0)); tot += o * n; num += n
            if tot / max(num, 1) > best:
                best, bk = tot / max(num, 1), float(k)
        model["k"][c] = bk
    return model


# ----------------------------------------------------------------- torch side
def collect(ckpts, stride, device, rows_filter=None):
    import torch  # noqa: F401
    from pde_competition.dataset import build_eval_arrays, list_runs
    from pde_competition.model import FlowerS2S, predict_numpy
    rows = {}
    for ck in ckpts:
        meta = json.load(open(Path(ck).parent / "meta.json"))["launches"][-1]["args"]
        holdout = meta.get("holdout_re") or [13950, 26700]
        model = FlowerS2S.from_checkpoint(ck)
        for re in holdout:
            if rows_filter is not None and re not in rows_filter:
                continue
            runs = [r for r in list_runs("train_real") if r.re_nominal == re]
            if not runs:
                continue
            x, y = build_eval_arrays(runs, stride=stride)
            pred = predict_numpy(model, x, batch_size=16, device=device)
            rows[re] = (pred[..., :2].astype(np.float32), y[..., :2].astype(np.float32), window_speed(x))
            print(f"  {Path(ck).parent.name}: row {re} -> {x.shape[0]} windows, mean speed {rows[re][2].mean():.4f}", flush=True)
    return rows


def main(a):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = sorted(p for g in a.folds for p in glob.glob(g))
    print(f"folds: {folds}\nstride {a.stride}  device {device}  sigma {SIGMA_GLOBAL}\n")
    rows = collect(folds, a.stride, device)
    print(f"\n{len(rows)} rows with out-of-fold predictions: {sorted(rows)}\n")

    if not a.no_loo:
        print("LEAVE-ONE-ROW-OUT (fit on 17, score the 18th; 3750 and 26700 are extrapolation):")
        loo = leave_one_row_out(rows, only=set(a.loo_rows) if a.loo_rows else None)
        names = list(next(iter(loo.values())).keys())
        print("\n  pooled over rows (unweighted mean of row objectives):")
        for k in names:
            v = np.mean([r[k] for r in loo.values()]); v0 = np.mean([r["const"] for r in loo.values()])
            print(f"    {k:<12}{v:.4f}   {v / v0 - 1:+.2%}")
        for re in (min(rows), max(rows)):
            if re not in loo:
                continue
            r = loo[re]; print(f"  extrapolation row {re}: " + ", ".join(f"{k} {r[k] / r['const'] - 1:+.2%}" for k in names if k != 'const'))

    if not a.ship:
        return
    print(f"\nFIT ON ALL {len(rows)} ROWS, alpha {a.alpha}, then re-level on the shipped model's rows {a.ship_re}")
    model = fit_model(rows, a.alpha)
    ship_rows = collect([a.ship], a.stride, device, rows_filter=set(a.ship_re))
    model = relevel(model, ship_rows)
    print(f"  a {model['a']}  b {model['b']}  k {model['k']}")
    # what the shipped model would score on its honest rows: const (as shipped) vs this
    for re, (pred, y, m) in ship_rows.items():
        hc = [fit_channel(pred[..., c], y[..., c], GRID)[0] for c in range(2)]
        sc = score(pred, y, m, lambda c: hc[c]); sm = score(pred, y, m, lambda c: widths(model, m, c))
        print(f"  shipped model, row {re}: const(own fit) {sc:.4f}  model {sm:.4f}  ({sm / sc - 1:+.2%})")
    ship = Path(a.ship)
    payload = {
        "half_width": [float(np.mean(widths(model, np.concatenate([r[2] for r in ship_rows.values()]), c))) for c in range(2)] + [0.0],
        "channels": ["u", "v", "p"],
        "half_width_affine_map": {
            "a": model["a"] + [0.0], "b": model["b"] + [0.0], "k": model["k"] + [0.0],
            "alpha": model["alpha"],
            "shape": np.concatenate([model["shape"], np.ones(model["shape"].shape[:2] + (1,), np.float32)], -1).round(5).tolist(),
            "speed": "mean |u| over measured (u != 0) points of the input window",
            "formula": "h[w,p,c] = k_c * max(a_c + b_c * m_w, 1e-4) * (alpha * shape[p,c] + 1 - alpha)",
        },
        "ckpt": ship.name, "ckpt_sha256": hashlib.sha256(ship.read_bytes()).hexdigest(),
        "fitted_on": {"folds": [str(f) for f in folds], "rows": sorted(rows), "stride": a.stride,
                      "relevel_rows": sorted(a.ship_re)},
        "sigma": SIGMA_GLOBAL, "fitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None,
        "tool": "scripts/oof_width_model.py",
    }
    out = fb.bounds_path_for(ship)
    out.write_text(json.dumps(payload) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", nargs="+", required=True, help="fold checkpoints or globs")
    p.add_argument("--ship", default=None, help="shipped checkpoint to re-level on and write bounds for")
    p.add_argument("--ship-re", type=int, nargs="+", default=[13950, 26700])
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--no-loo", action="store_true",
                   help="skip the leave-one-row-out table (already measured in W31); "
                        "just fit on all rows, re-level on --ship and write its bounds")
    p.add_argument("--loo-rows", type=int, nargs="*", default=None,
                   help="rows to leave out (default all); e.g. 3750 13950 26700 "
                        "keeps the fit to minutes instead of an hour")
    p.add_argument("--alpha", type=float, default=0.5, help="shape blend for the shipped model")
    main(p.parse_args())
