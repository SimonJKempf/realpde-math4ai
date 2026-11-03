#!/usr/bin/env python
"""Is the width head's OUTPUT mis-shaped, and can a monotone recalibration fix it?

sps is our worst subscore (43.06 on the board) and the richest non-time marginal
we have (+0.244 final per point), so it is where a post-hoc change is worth most.
The head's own exact validation already sweeps a single global scalar k and lands
at 0.98-1.00, so a global rescale is exhausted -- the head is well calibrated ON
AVERAGE. That says nothing about its SHAPE: it may still be systematically too
narrow where it predicts small widths and too wide where it predicts large ones,
and the scorer's reward exp(-2h/sigma) is strongly nonlinear in h, so shape is
paid for.

This fits a two-parameter MONOTONE recalibration

    g(h) = h0 * k * (h / h0) ** gamma        h0 = median predicted width

and evaluates it with the exact objective (fit_bounds.objective_map). gamma = 1
is the identity; gamma < 1 compresses the spread of widths toward h0, gamma > 1
stretches it. Two parameters, not eight free bins, BECAUSE the honest protocol
here is leave-one-Reynolds-row-out and there are only five rows: a flexible
recalibration would fit the row it is scored on. The free-per-bin fit is still
reported, but only as an ORACLE ceiling -- what any monotone reshaping could ever
buy -- never as a shippable number.

    python scripts/width_recalibration.py --head outputs/width_head/head_ext_tau05_w48.pth \
        --folds 'outputs/runs/ws_cfk_fold*/best_score.pth' --stride 40

Reads: if LOO gain is <= 0, the head's shape is already right and the whole
output-recalibration axis (sps P2, and P3's quantile head with it, since P3 is a
richer way of doing the same thing) is closed. A positive LOO gain is worth its
9 lines in submission_flower.py, which already applies a scalar to the head.
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_bounds import objective_map                          # noqa: E402
from oof_disagreement import collect as collect_rows          # noqa: E402
from oof_width_model import window_speed                      # noqa: E402
from train_width_head import WidthHead, features, extra_planes  # noqa: E402


def recal(h, h0, k, gamma):
    """Monotone in h for k, gamma > 0. Identity at k = gamma = 1."""
    return h0 * k * np.power(np.clip(h, 1e-12, None) / h0, gamma)


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

    # per (row, channel): the predicted widths, the target, and the scored count
    cells = {}
    for re in sorted(rows):
        m, z, y = rows[re]
        f = features(m, z, speeds[re], ck["norm"], extras.get(re) if ck["planes"] == "ext" else None)
        if ck.get("drop_z"):
            f[:, 2:4] = 0.0
        with torch.no_grad():
            h = torch.cat([head(f[i:i + 256].to(device)).cpu() for i in range(0, f.shape[0], 256)])
        h = h.numpy().reshape(m.shape[0], m.shape[1], 2, *m.shape[2:4]).transpose(0, 1, 3, 4, 2)
        for c, name in enumerate(["u", "v"]):
            n_sc = int(np.count_nonzero(y[..., c] != 0))
            if n_sc:
                cells[(re, name)] = (m[..., c], y[..., c], h[..., c], n_sc)

    ks = np.round(np.arange(a.k_min, a.k_max + 1e-9, a.k_step), 4)
    gs = np.round(np.arange(a.g_min, a.g_max + 1e-9, a.g_step), 4)
    h0 = {key: float(np.median(v[2])) for key, v in cells.items()}

    def obj(key, k, g):
        m, y, h, _ = cells[key]
        return objective_map(m, y, recal(h, h0[key], k, g))[0]

    # grid once per cell: cheap, and every fit below is a lookup
    grid = {key: np.array([[obj(key, k, g) for g in gs] for k in ks]) for key in cells}
    ident = {key: obj(key, 1.0, 1.0) for key in cells}

    def pooled(keys, table):
        num = sum(table[key] * cells[key][3] for key in keys)
        return num / sum(cells[key][3] for key in keys)

    print(f"{'row':>7} {'ch':>3} {'identity':>10} {'best k':>7} {'best g':>7} {'in-sample':>10} {'gain':>8}")
    for key in sorted(cells):
        i, j = np.unravel_index(np.argmax(grid[key]), grid[key].shape)
        print(f"{key[0]:>7} {key[1]:>3} {ident[key]:10.4f} {ks[i]:7.3f} {gs[j]:7.3f} "
              f"{grid[key][i, j]:10.4f} {grid[key][i, j] / ident[key] - 1:+7.2%}")

    print("\nLEAVE-ONE-REYNOLDS-ROW-OUT -- the only honest number here")
    print(f"{'held-out row':>13} {'k*':>7} {'g*':>7} {'identity':>10} {'recal':>10} {'gain':>8}")
    tot_i = tot_r = tot_n = 0.0
    all_rows = sorted({key[0] for key in cells})
    for re in all_rows:
        tr = [key for key in cells if key[0] != re]
        te = [key for key in cells if key[0] == re]
        stack = sum(grid[key] * cells[key][3] for key in tr) / sum(cells[key][3] for key in tr)
        i, j = np.unravel_index(np.argmax(stack), stack.shape)
        oi = pooled(te, ident)
        orr = pooled(te, {key: grid[key][i, j] for key in te})
        n = sum(cells[key][3] for key in te)
        tot_i += oi * n; tot_r += orr * n; tot_n += n
        print(f"{re:>13} {ks[i]:7.3f} {gs[j]:7.3f} {oi:10.4f} {orr:10.4f} {orr / oi - 1:+7.2%}")
    print(f"{'POOLED LOO':>13} {'':>7} {'':>7} {tot_i / tot_n:10.4f} {tot_r / tot_n:10.4f} "
          f"{tot_r / tot_i - 1:+7.2%}")

    # oracle ceiling of the two-parameter family: best (k, g) chosen ON the row it is scored on
    orc = sum(grid[key].max() * cells[key][3] for key in cells) / sum(cells[key][3] for key in cells)
    idp = pooled(list(cells), ident)
    print(f"\nORACLE, power family (k,g fitted on the row scored -- NOT shippable): {orc:.4f} ({orc / idp - 1:+.2%})")

    # EXACT ceiling of ANY recalibration h -> h'. The objective is
    #     mean over scored elements of 1{|r| <= h'} * exp(-2 h' / sigma)
    # which is a SUM OF PER-ELEMENT TERMS, each depending only on that element's own h'.
    # So a recalibration that is a function of the predicted h alone can be optimised
    # independently in each bin of h, and the per-bin optimum is exact rather than a search.
    # This bounds every monotone reshaping AND every non-monotone one -- a strict superset.
    from fit_bounds import SIGMA_GLOBAL
    num = den = 0.0
    for key, (m, y, h, n_sc) in cells.items():
        sc = y != 0
        r = np.abs(y - m)[sc].astype(np.float64)
        hh = h[sc].astype(np.float64)
        edges = np.quantile(hh, np.linspace(0, 1, a.bins + 1))
        edges[-1] += 1e-12
        idx = np.clip(np.searchsorted(edges, hh, side="right") - 1, 0, a.bins - 1)
        best = 0.0
        for b in range(a.bins):
            sel = idx == b
            if not sel.any():
                continue
            rb = np.sort(r[sel])
            # EXACT per-bin optimum, no grid. Choosing one width c for the bin gives
            #     count(r <= c) * exp(-2c/sigma),
            # whose count term only steps up AT a residual value while the exponential
            # strictly decreases, so the maximum is always attained at some c = r_(i).
            # Taking i * exp(-2 r_(i) / sigma) over the sorted residuals is therefore the
            # true optimum -- an earlier version searched a grid of h-quantiles instead,
            # which cannot represent widths outside the head's own output range and gave
            # a "ceiling" BELOW the power family's, i.e. not a bound at all.
            best += float(np.max(np.arange(1, rb.size + 1) * np.exp(-2.0 * rb / SIGMA_GLOBAL)))
        num += best / np.count_nonzero(sc) * n_sc
        den += n_sc
    free = num / den
    print(f"ORACLE, ANY h-only recalibration ({a.bins} bins, EXACT per-bin optimum): "
          f"{free:.4f} ({free / idp - 1:+.2%})")
    # THE ABSOLUTE CEILING OF ANY BOUNDS METHOD WHATSOEVER. If the width could depend on the
    # actual residual -- which no model can do -- the best choice per element is h = |r| exactly,
    # covering it with zero slack, for a reward of exp(-2|r|/sigma). This bounds the head, a
    # quantile head (sps P3), and anything else that predicts widths from anything at all.
    num = den = 0.0
    for key, (m, y, h, n_sc) in cells.items():
        sc = y != 0
        r = np.abs(y - m)[sc].astype(np.float64)
        num += float(np.mean(np.exp(-2.0 * r / SIGMA_GLOBAL))) * n_sc
        den += n_sc
    per_el = num / den
    print(f"ORACLE, PER-ELEMENT h = |r| (no model can reach this -- it uses the answer): "
          f"{per_el:.4f} ({per_el / idp - 1:+.2%})")
    print(f"  the shipped head sits at {idp:.4f}, i.e. {idp / per_el:.1%} of that absolute ceiling")
    print("  -> headroom for ANY better width model, sps P3 included, is at most "
          f"{per_el / idp - 1:+.1%} on this objective.")

    print("\nThat second ceiling is the honest one: it bounds every function of the predicted width,")
    print("monotone or not, fitted directly on the data it is scored on. It closes sps P2 (recalibrating")
    print("THIS head's output) if it is small. It does NOT bound sps P3 (a quantile head trained from")
    print("FEATURES), which is a different model and not a function of h alone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--head", required=True)
    p.add_argument("--folds", nargs="+", required=True)
    p.add_argument("--stride", type=int, default=40)
    p.add_argument("--k-min", type=float, default=0.85)
    p.add_argument("--k-max", type=float, default=1.20)
    p.add_argument("--k-step", type=float, default=0.025)
    p.add_argument("--g-min", type=float, default=0.70)
    p.add_argument("--g-max", type=float, default=1.40)
    p.add_argument("--g-step", type=float, default=0.05)
    p.add_argument("--bins", type=int, default=40, help="quantile bins of the predicted width for the exact free ceiling")
    main(p.parse_args())
