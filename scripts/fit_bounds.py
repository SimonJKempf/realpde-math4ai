#!/usr/bin/env python
"""Fit SPS interval half-widths for a trained checkpoint.

    python scripts/fit_bounds.py --ckpt outputs/runs/<run>/best.pth
    # -> writes outputs/runs/<run>/best.bounds.json, which build_submission.py
    #    ships in the zip and submission.py reads. Nothing to paste.

The scorer's default band is ``+-0.05*|pred|``, which covers barely a third of
measured elements -- and an element whose target falls outside its interval
scores exactly zero. Fitted constant widths are worth far more than they look:
the same FNO predictions went from sps 20.39 to 46.73 with no retraining.

Widths are model-specific, so this has to be re-run for every checkpoint. The
FNO ones survive only as a literal in submission/archive/submission_fno_bounds.py
with the method in a comment; this script exists so ours are reproducible.

What is being maximised
-----------------------
From kit/scoring.py's aggregate_sps, an element contributes

    (1 - pm) * exp(-(upper - lower)/sigma)     if lower <= target <= upper
    0                                          otherwise

and elements with ``target == 0`` are not scored at all -- those are outside the
PIV field of view or inside the airfoil body.

``pm`` is a per-window accuracy factor that does not depend on the width, so it
drops out of the fit. Each element belongs to exactly one channel, so the
objective separates and each channel's half-width ``h`` is an independent 1-D
maximisation of

    mean over scored elements of  1{|target - pred| <= h} * exp(-2h/sigma)

Which cells to fit on
---------------------
Fit and validate on DISJOINT cells the model has never trained on. Fitting on
training cells would tune the widths to optimistically small residuals and
produce intervals that are systematically too narrow at test time. The default
splits the held-out set: Re 13950 fits, Re 26700 validates.

``--gapped-holdout [N G]`` (direction GTH, default 100 40) fits instead on the
LAST N frames of EVERY trajectory -- the validation side of
``split_temporal_gapped``, ALL Re rows -- for a checkpoint trained with train.py
``--gapped-holdout``, which held out no condition. There is no disjoint row to
validate on, so fit and val are the same windows and the val columns are
in-sample. ``--fit-re``/``--val-re`` are refused with it. The bounds file then
carries ``"gapped_holdout": [N, G]``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from pde_competition.dataset import (build_eval_arrays, list_runs,
                                     parse_gapped_holdout, split_temporal_gapped)
from pde_competition.geometry import C
from pde_competition.model import FlowerS2S, predict_numpy, window_scale
from pde_competition.paths import KIT

# Read sigma from the scorer rather than retyping it: a wrong sigma would
# silently mis-fit every width, and nothing downstream would complain.
sys.path.insert(0, str(KIT))
from scoring import SIGMA_GLOBAL  # noqa: E402


def objective(pred, target, h, sigma=SIGMA_GLOBAL):
    """Mean SPS width-reward over scored elements, for one channel.

    ``target == 0`` marks unscored elements (outside the PIV field of view, or
    inside the airfoil body); they leave both numerator and denominator, exactly
    as aggregate_sps does.
    """
    scored = target != 0.0
    n = int(np.count_nonzero(scored))
    if n == 0:
        return float("nan"), float("nan")
    inside = (np.abs(target - pred) <= h) & scored
    coverage = np.count_nonzero(inside) / n
    return coverage * float(np.exp(-2.0 * h / sigma)), coverage


def default_band_objective(pred, target, sigma=SIGMA_GLOBAL):
    """The scorer's fallback: lower = p - 0.05|p|, upper = p + 0.05|p|.

    Width varies per element, so this cannot reuse ``objective`` -- the
    exponential has to be evaluated elementwise.
    """
    scored = target != 0.0
    n = int(np.count_nonzero(scored))
    half = 0.05 * np.abs(pred)
    inside = (np.abs(target - pred) <= half) & scored
    reward = np.where(inside, np.exp(-2.0 * half / sigma), 0.0)
    return float(reward.sum() / n), float(np.count_nonzero(inside) / n)


def fit_channel(pred, target, grid):
    best_h, best_obj, best_cov = 0.0, -1.0, 0.0
    for h in grid:
        obj, cov = objective(pred, target, h)
        if obj > best_obj:
            best_h, best_obj, best_cov = float(h), obj, cov
    return best_h, best_obj, best_cov


def objective_perwindow(pred, target, h, sigma=SIGMA_GLOBAL):
    """Like ``objective`` but with a per-window half-width.

    ``h`` has one entry per window, so the exponential penalty varies window to
    window and cannot be factored out of the mean the way a constant can.
    """
    scored = target != 0.0
    n = int(np.count_nonzero(scored))
    hb = h.reshape((-1,) + (1,) * (target.ndim - 1))
    inside = (np.abs(target - pred) <= hb) & scored
    reward = np.where(inside, np.exp(-2.0 * hb / sigma), 0.0)
    return float(reward.sum() / n), float(np.count_nonzero(inside) / n)


# --------------------------------------------------------------------------
# PER-ELEMENT WIDTHS. The objective is a mean over scored elements of
# 1{|r| <= h} * exp(-2h/sigma), and nothing couples one element's h to
# another's, so h may vary freely over (t, y, x) and each cell's optimum is a
# separate 1-D problem on that cell's own residuals. The scorer accepts
# per-element bounds (they must match the prediction's shape), and the
# submission already ships a per-element array -- filled with a constant.
# Free stream, wake and body boundary have very different residual scales, so
# a constant leaves reward on the table wherever the residual is reliably
# small. A pooled map h(y, x) has N*T samples per pixel (2720 at stride 50);
# a frame-resolved map h(t, y, x) has only N per cell and can overfit -- both
# are validated on the disjoint row, which is the honest check.
# --------------------------------------------------------------------------
def fit_map(pred, target, per_frame=False, sigma=SIGMA_GLOBAL):
    """Optimal half-width per cell for one channel. pred/target: (N, T, H, W).

    Returns (H, W) with frames pooled, or (T, H, W) with ``per_frame``. For a
    cell with sorted residuals r_(1) <= ... <= r_(n) the objective is piecewise
    constant in h and maximal at some r_(k): (k/n) * exp(-2 r_(k)/sigma). So the
    exact optimum is an argmax over the sorted residuals -- no grid needed.
    Unscored samples (target == 0) leave numerator and denominator.
    """
    r = np.abs(target - pred).astype(np.float64)
    scored = target != 0.0
    R = np.where(scored, r, np.inf)
    if not per_frame:
        R = R.reshape(-1, *R.shape[2:])                    # (N*T, H, W)
    Rs = np.sort(R, axis=0)
    n = np.count_nonzero(np.isfinite(Rs), axis=0)          # scored per cell
    k = np.arange(1, Rs.shape[0] + 1, dtype=np.float64)
    k = k.reshape((-1,) + (1,) * (Rs.ndim - 1))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        obj = (k / np.maximum(n, 1)) * np.exp(-2.0 * Rs / sigma)   # inf -> 0
    obj = np.where(np.isfinite(obj), obj, 0.0)
    idx = np.argmax(obj, axis=0)
    h = np.take_along_axis(Rs, idx[None], axis=0)[0]
    h = np.where((n > 0) & np.isfinite(h), h, 0.0)
    return h.astype(np.float32)


def objective_map(pred, target, hmap, sigma=SIGMA_GLOBAL):
    """Like ``objective`` with a per-element half-width broadcastable to pred."""
    scored = target != 0.0
    n = int(np.count_nonzero(scored))
    if n == 0:
        return float("nan"), float("nan")
    h = np.broadcast_to(np.asarray(hmap, dtype=np.float64), pred.shape)
    inside = (np.abs(target - pred) <= h) & scored
    reward = np.where(inside, np.exp(-2.0 * h / sigma), 0.0)
    return float(reward.sum() / n), float(np.count_nonzero(inside) / n)


def smooth3(h):
    """3x3 mean over the last two axes, edges replicated. Cheap regulariser."""
    p = np.pad(h, [(0, 0)] * (h.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    out = np.zeros_like(h, dtype=np.float64)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out += p[..., dy:dy + h.shape[-2], dx:dx + h.shape[-1]]
    return (out / 9.0).astype(np.float32)


SPATIAL_VARIANTS = ("const", "map", "map_smooth", "map_blend", "frame", "frame_smooth")


def spatial_variants(fit_pred, fit_y, grid):
    """Every candidate width for one channel, fitted on the fit rows."""
    h_const = fit_channel(fit_pred, fit_y, grid)[0]
    m = fit_map(fit_pred, fit_y)
    f = fit_map(fit_pred, fit_y, per_frame=True)
    return {
        "const": np.float32(h_const),
        "map": m,
        "map_smooth": smooth3(m),
        "map_blend": (0.5 * m + 0.5 * h_const).astype(np.float32),
        "frame": f,
        "frame_smooth": smooth3(f),
    }


def run_spatial(a, device, grid):
    fit_pred, fit_y, fit_cells = arrays_for(a.fit_re, a.ckpt, a.stride, device, a.fit_ensemble)
    val_pred, val_y, val_cells = arrays_for(a.val_re, a.ckpt, a.stride, device, a.fit_ensemble)
    if a.fit_ensemble:
        print(f"fitting on the MEAN of {len(a.fit_ensemble)} members, not on --ckpt alone")
    print(f"fit  cells {fit_cells}  -> {fit_pred.shape[0]} windows")
    print(f"val  cells {val_cells}  -> {val_pred.shape[0]} windows\n")
    names = ["u", "v", "p"]
    chosen = {}
    n_fit = {}; n_val = {}
    rows = {v: [0.0, 0.0] for v in SPATIAL_VARIANTS}   # pooled numerators
    for c in range(C):
        if not np.any(fit_y[..., c] != 0.0):
            continue
        fp, fy, vp, vy = fit_pred[..., c], fit_y[..., c], val_pred[..., c], val_y[..., c]
        n_fit[c] = int(np.count_nonzero(fy != 0.0)); n_val[c] = int(np.count_nonzero(vy != 0.0))
        cands = spatial_variants(fp, fy, grid)
        print(f"channel {names[c]}   (const half-width {float(cands['const']):.4f})")
        print(f"  {'variant':<12}{'fit obj':>9}{'fit cov':>9}{'val obj':>9}{'val cov':>9}{'val gain':>10}")
        base_val = objective_map(vp, vy, cands["const"])[0]
        best_name, best_val = "const", base_val
        for name in SPATIAL_VARIANTS:
            fo, fc = objective_map(fp, fy, cands[name])
            vo, vc = objective_map(vp, vy, cands[name])
            rows[name][0] += fo * n_fit[c]; rows[name][1] += vo * n_val[c]
            print(f"  {name:<12}{fo:9.4f}{fc:9.1%}{vo:9.4f}{vc:9.1%}{vo / base_val - 1:+10.2%}")
            if vo > best_val:
                best_name, best_val = name, vo
        pick = best_name if a.spatial_variant == "auto" else a.spatial_variant
        chosen[c] = (pick, cands[pick])
        print(f"  -> channel {names[c]} picks {pick!r}  (auto: best VAL objective for THIS "
              f"channel; --spatial-variant overrides)\n")
    nf, nv = sum(n_fit.values()), sum(n_val.values())
    print("pooled over scored elements of all measured channels (this ratio is the sps ratio,")
    print("up to the per-window accuracy factor, which the width does not touch):")
    print(f"  {'variant':<12}{'fit':>9}{'val':>9}{'val gain':>10}")
    for name in SPATIAL_VARIANTS:
        print(f"  {name:<12}{rows[name][0] / nf:9.4f}{rows[name][1] / nv:9.4f}"
              f"{rows[name][1] / rows['const'][1] - 1:+10.2%}")
    # The rows above use ONE variant for every channel. The picker chooses per
    # channel, so its combination is not in the table; price it the same way.
    mix_fit = sum(objective_map(fit_pred[..., c], fit_y[..., c], h)[0] * n_fit[c]
                  for c, (_, h) in chosen.items()) / nf
    mix_val = sum(objective_map(val_pred[..., c], val_y[..., c], h)[0] * n_val[c]
                  for c, (_, h) in chosen.items()) / nv
    picks = {names[c]: p for c, (p, _) in chosen.items()}
    print(f"  {'PICKED':<12}{mix_fit:9.4f}{mix_val:9.4f}{mix_val / (rows['const'][1] / nv) - 1:+10.2%}"
          f"   per channel {picks}")
    print("  (the pooled ratio and the per-channel objective are the SAME quantity; the pooled row")
    print("   is its element-weighted mean over channels, so PICKED >= every uniform row on val)")
    if a.no_write:
        print("\n(not written)")
        return
    # Assemble a channels-last map, (H, W, C) or (T, H, W, C), zero on p.
    per_frame = any(v.ndim == 3 for _, v in chosen.values())
    shape = (val_pred.shape[1],) * per_frame + tuple(val_pred.shape[2:4]) + (C,)
    hmap = np.zeros(shape, dtype=np.float32)
    widths = np.zeros(C, dtype=np.float32)
    for c, (pick, h) in chosen.items():
        widths[c] = float(np.mean(h))
        hmap[..., c] = np.broadcast_to(h, shape[:-1])
    kinds = {names[c]: pick for c, (pick, _) in chosen.items()}
    out = write_bounds(a, widths, fit_cells, val_cells, hmap=hmap, kinds=kinds)
    print(f"\nwrote {out}  (half_width_map {hmap.shape}, variants {kinds}; half_width holds the map means)")


def input_magnitude(x):
    """Per-window mean |u| over measured elements, from the INPUT window.

    This is what predict() can see; the fit's per-row optima were measured
    against targets. The two agree to r = 0.9995 (mean relative difference
    ~1%), which is what makes conditioning on the input legitimate.

    Delegates to ``model.window_scale``, which is the same quantity used as the
    per-window divisor by ``FlowerConfig.window_norm``. One definition, so the
    affine width fit and the model cannot end up disagreeing about what the
    velocity scale of a window is.
    """
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return window_scale(t).numpy().reshape(-1).astype(np.float64)


def _predict(ckpt_path, members, x, device):
    """The prediction the SUBMISSION will make: one model, or the mean over members.

    Widths are fitted to residuals, and an ensemble's residuals are smaller than any
    member's -- so fitting on a single model and shipping the widths with an ensemble
    over-widens every element and gives away sps that the ensemble has already earned.
    Every ensemble zip built before 2026-09-18 did exactly that (score_twin fits on
    --ckpt and builds with --ensemble).
    """
    cks = [ckpt_path] if not members else list(members)
    preds = [predict_numpy(FlowerS2S.from_checkpoint(c), x, batch_size=16, device=device) for c in cks]
    return preds[0] if len(preds) == 1 else np.mean(preds, axis=0)


def arrays_for(re_values, ckpt_path, stride, device, members=None):
    runs = [r for r in list_runs("train_real") if r.re_nominal in re_values]
    if not runs:
        raise SystemExit(f"no trajectories with Re in {re_values}")
    x, y = build_eval_arrays(runs, stride=stride)
    pred = _predict(ckpt_path, members, x, device)
    return pred, y, sorted(r.key for r in runs)


def arrays_for_gth(gapped_holdout, ckpt_path, stride, device, members=None):
    """GTH: predictions and targets on the TAIL windows of every real trajectory.

    The validation side of ``split_temporal_gapped`` with the same two numbers
    train.py used, so the residuals come from frames the checkpoint never saw --
    across all 18 Re rows, not 2. Returns ``(pred, y, cells)`` like ``arrays_for``.
    """
    n_hold, gap = gapped_holdout
    _train, val = split_temporal_gapped(list_runs("train_real"),
                                        holdout_frames=n_hold, gap_frames=gap)
    x, y = build_eval_arrays(val, stride=stride)
    pred = _predict(ckpt_path, members, x, device)
    return pred, y, sorted(r.key for r in val)


def resolve_rows(a):
    """Resolve ``--fit-re``/``--val-re`` against ``--gapped-holdout`` into one design.

    Sets ``a.gth`` to ``None`` or ``(N, G)`` and fills the row defaults (fit 13950,
    val 26700) only on the condition-holdout path. Under GTH the two row flags are
    REFUSED rather than reinterpreted: the fit uses the tail of ALL rows, there is
    no disjoint row, and a filter would only produce a third kind of width.
    """
    a.gth = parse_gapped_holdout(a.gapped_holdout)
    if a.gth is not None:
        if a.fit_re is not None or a.val_re is not None:
            raise SystemExit(
                "--fit-re/--val-re with --gapped-holdout is refused: a GTH checkpoint held out "
                "no condition, so the widths are fitted on the TAIL windows of ALL rows and "
                "there is no disjoint row to validate on. Pass one of the two designs.")
        if a.affine or a.spatial:
            raise SystemExit("--affine/--spatial with --gapped-holdout is not implemented: both "
                             "are per-Re-row fits with a disjoint val row, which GTH has none of.")
        a.fit_re = a.val_re = []          # filled with the actual rows once the arrays exist
    else:
        a.fit_re = list(a.fit_re) if a.fit_re else [13950]
        a.val_re = list(a.val_re) if a.val_re else [26700]
    return a


def per_row(re_values, ckpt_path, stride, device, grid):
    """Per Reynolds row: its input mean|u| and its own optimal width per channel."""
    model = FlowerS2S.from_checkpoint(ckpt_path)
    out = []
    for re in sorted(re_values):
        rr = [r for r in list_runs("train_real") if r.re_nominal == re]
        if not rr:
            continue
        x, y = build_eval_arrays(rr, stride=stride)
        pred = predict_numpy(model, x, batch_size=16, device=device)
        m = float(input_magnitude(x).mean())
        hs = [fit_channel(pred[..., c], y[..., c], grid)[0]
              if np.any(y[..., c] != 0.0) else 0.0 for c in range(C)]
        out.append((re, m, hs, pred, x, y))
    return out


def run_affine(a, device, grid):
    """Fit h_c = beta_c + alpha_c * mean|u_input|.

    The slope comes from every row -- the training rows span the |u| range and
    pin it well. The intercept is then shifted so the line passes through the
    HELD-OUT rows, because training rows have optimistically small residuals and
    bias the level, not the slope. One parameter calibrated on unseen data
    instead of two, which is all two held-out rows can support.
    """
    held = set(a.val_re)
    rows = per_row(sorted(set(a.fit_re) | held), a.ckpt, a.stride, device, grid)
    names = ["u", "v", "p"]

    print(f"{'Re':>6} {'held?':>6} {'|u| in':>8} " +
          "".join(f"{'h_'+names[c]:>8}" for c in range(C)))
    for re, m, hs, *_ in rows:
        print(f"{re:6d} {'HELD' if re in held else '':>6} {m:8.4f} " +
              "".join(f"{h:8.4f}" for h in hs))

    mags = np.array([r[1] for r in rows])
    slope, inter = np.zeros(C), np.zeros(C)
    print()
    for c in range(C):
        hc = np.array([r[2][c] for r in rows])
        if not np.any(hc > 0):
            continue
        al, be_all = np.polyfit(mags, hc, 1)
        # Re-level on held-out rows only.
        hm = [(m, h) for (re, m, hs, *_) in rows for h in [hs[c]] if re in held]
        be = float(np.mean([h - al * m for m, h in hm]))
        slope[c], inter[c] = al, be
        print(f"  {names[c]}: slope {al:.5f}  intercept {be_all:+.5f} (all rows) "
              f"-> {be:+.5f} (held-out re-levelled, shift {be - be_all:+.5f})")

    print(f"\nobjective on each row: constant {a.const} vs affine vs that row's own best")
    print(f"{'Re':>6} {'held?':>6} {'ch':>3} {'constant':>9} {'affine':>9} "
          f"{'row best':>9} {'affine/const':>13}")
    for re, m, hs, pred, x, y in rows:
        mag = input_magnitude(x)
        for c in range(C):
            if hs[c] <= 0:
                continue
            o_const, _ = objective(pred[..., c], y[..., c], a.const[c])
            h_aff = np.maximum(inter[c] + slope[c] * mag, 1e-6)
            o_aff, _ = objective_perwindow(pred[..., c], y[..., c], h_aff)
            o_best, _ = objective(pred[..., c], y[..., c], hs[c])
            print(f"{re:6d} {'HELD' if re in held else '':>6} {names[c]:>3} "
                  f"{o_const:9.4f} {o_aff:9.4f} {o_best:9.4f} "
                  f"{o_aff / o_const:12.3f}x")

    print("\npaste into submission/submission_flower.py:\n")
    print(f"_W_SLOPE     = np.array({[round(float(s), 5) for s in slope]}, dtype=np.float32)")
    print(f"_W_INTERCEPT = np.array({[round(float(b), 5) for b in inter]}, dtype=np.float32)")


def main(a):
    # RPDE_DEVICE overrides the pick (final/solution.py passes the device it was handed; on a Mac
    # the torch-2.2.2 smoke must not land on MPS, which cannot run Flower's border grid_sample).
    device = os.environ.get("RPDE_DEVICE") or ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"checkpoint: {a.ckpt}\ndevice: {device}  sigma: {SIGMA_GLOBAL}\n")

    grid_ = np.concatenate([np.linspace(1e-4, 0.02, 400),
                            np.linspace(0.02, 0.20, 200)])
    if a.affine:
        return run_affine(a, device, grid_)
    if a.spatial:
        return run_spatial(a, device, grid_)

    if getattr(a, "gth", None) is not None:
        n_hold, gap = a.gth
        print("=" * 96 + f"\nGTH TAIL FIT: residuals from the LAST {n_hold} frames of EVERY trajectory "
              f"(gap {gap}), ALL Re rows.\nfit == val (no disjoint row exists): the val columns "
              f"below are IN-SAMPLE for the width and only sanity-check the fit.\nNOT comparable "
              f"with a condition-holdout fit. bounds.json carries \"gapped_holdout\": [{n_hold}, {gap}].\n"
              + "=" * 96)
        fit_pred, fit_y, fit_cells = arrays_for_gth(a.gth, a.ckpt, a.stride, device, a.fit_ensemble)
        val_pred, val_y, val_cells = fit_pred, fit_y, fit_cells
        a.fit_re = a.val_re = sorted({c[0] for c in fit_cells})
    else:
        fit_pred, fit_y, fit_cells = arrays_for(a.fit_re, a.ckpt, a.stride, device, a.fit_ensemble)
        val_pred, val_y, val_cells = arrays_for(a.val_re, a.ckpt, a.stride, device, a.fit_ensemble)
    if a.fit_ensemble:
        print(f"fitting on the MEAN of {len(a.fit_ensemble)} members, not on --ckpt alone")
    print(f"fit  cells {fit_cells}  -> {fit_pred.shape[0]} windows")
    print(f"val  cells {val_cells}  -> {val_pred.shape[0]} windows\n")

    grid = np.concatenate([np.linspace(1e-4, 0.02, 400),
                           np.linspace(0.02, 0.20, 200)])
    names = ["u", "v", "p"]
    widths = np.zeros(C, dtype=np.float32)

    print(f"{'ch':>3} {'half-width':>11} {'fit obj':>9} {'fit cov':>8} "
          f"{'val obj':>9} {'val cov':>8}")
    for c in range(C):
        # p is unmeasured on real data: every target is 0, so no element is
        # scored and no width can earn anything. Zero width is legal
        # (lower == upper) and costs nothing.
        if not np.any(fit_y[..., c] != 0.0):
            print(f"{names[c]:>3} {0.0:11.4f} {'--':>9} {'--':>8} "
                  f"{'--':>9} {'--':>8}   (unmeasured, not scored)")
            continue
        h, obj, cov = fit_channel(fit_pred[..., c], fit_y[..., c], grid)
        vobj, vcov = objective(val_pred[..., c], val_y[..., c], h)
        widths[c] = h
        print(f"{names[c]:>3} {h:11.4f} {obj:9.4f} {cov:8.1%} {vobj:9.4f} {vcov:8.1%}")

    print("\nagainst the scorer's default +-5%|pred| band:")
    print(f"{'ch':>3} {'fit obj':>9} {'fit cov':>8} {'val obj':>9} {'val cov':>8}")
    for c in range(C):
        if not np.any(fit_y[..., c] != 0.0):
            continue
        o, cv = default_band_objective(fit_pred[..., c], fit_y[..., c])
        vo, vcv = default_band_objective(val_pred[..., c], val_y[..., c])
        print(f"{names[c]:>3} {o:9.4f} {cv:8.1%} {vo:9.4f} {vcv:8.1%}")

    if len(set(a.fit_re)) == 1 and getattr(a, "gth", None) is None:
        print("\nWARNING: fitted on a SINGLE Reynolds row. Submission 6 shipped "
              "widths fitted on Re 13950 alone (0.0079/0.0057) and lost 1.44 sps "
              "against the all-conditions fit (0.0090/0.0064): Re 26700 wants "
              "44% wider bands. Consider --fit-re 13950 26700 before submitting.")

    if a.no_write:
        print(f"\nhalf-widths {[round(float(w), 4) for w in widths]}  (not written)")
        return
    out = write_bounds(a, widths, fit_cells, val_cells)
    print(f"\nwrote {out}\n  build_submission.py picks it up from beside the "
          f"checkpoint; nothing to paste.")


def bounds_path_for(ckpt: Path) -> Path:
    """best.pth -> best.bounds.json, beside it.

    Keyed to the checkpoint's stem because a run directory holds several
    checkpoints (best / best_score / last) and each needs its own widths.
    """
    return ckpt.with_name(ckpt.stem + ".bounds.json")


def write_bounds(a, widths, fit_cells, val_cells, hmap=None, kinds=None) -> Path:
    """Record the fitted half-widths as an artifact beside the checkpoint.

    This file, not a literal in submission.py, is what ships in the zip. So the
    widths a submission carries are always traceable to the checkpoint, the
    cells and the commit that produced them -- the edit that cost submission 6
    (hand-pasting a single-row fit) cannot recur as an invisible one-line diff.
    """
    ckpt = Path(a.ckpt)
    # --write-for: the widths were FITTED on a.ckpt (a held-out twin) but SHIP
    # with another checkpoint -- the --train-all final model, which has seen
    # every cell and has no honest residuals of its own. The file goes beside
    # the target and carries the target's sha, which is what
    # build_submission.py checks; the fit's own provenance is kept alongside.
    target = Path(a.write_for) if a.write_for else ckpt
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    payload = {
        "half_width": [round(float(w), 6) for w in widths],
        "channels": ["u", "v", "p"],
        "ckpt": target.name,
        "ckpt_sha256": sha,
        "fitted_on_ckpt": str(ckpt) if a.write_for else None,
        "fitted_on_sha256": (hashlib.sha256(ckpt.read_bytes()).hexdigest()
                             if a.write_for else None),
        "fit_re": sorted(a.fit_re), "val_re": sorted(a.val_re),
        # GTH: None for every ordinary fit, [N, G] when the residuals came from the
        # gapped temporal TAIL of all rows (fit_re then lists every row and fit == val).
        "gapped_holdout": (list(a.gth) if getattr(a, "gth", None) is not None else None),
        "fit_cells": [list(c) for c in fit_cells],
        "val_cells": [list(c) for c in val_cells],
        "stride": a.stride,
        "sigma": SIGMA_GLOBAL,
        "fitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "tool": "scripts/fit_bounds.py",
    }
    if hmap is not None:
        # Channels-last, broadcastable to the prediction (N, T, H, W, C):
        # (H, W, C) pooled over frames or (T, H, W, C) frame-resolved.
        payload["half_width_map"] = np.asarray(hmap, dtype=np.float32).round(6).tolist()
        payload["half_width_map_shape"] = list(np.asarray(hmap).shape)
        payload["half_width_map_variant"] = kinds
    out = bounds_path_for(target)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--fit-re", type=int, nargs="+", default=None,
                   help="Reynolds rows to fit the widths on (default 13950). Refused with "
                        "--gapped-holdout.")
    p.add_argument("--val-re", type=int, nargs="+", default=None,
                   help="disjoint rows to validate them on (default 26700). Refused with "
                        "--gapped-holdout.")
    p.add_argument("--gapped-holdout", type=int, nargs="*", default=None, metavar="N_G",
                   help="direction GTH: fit on the LAST N frames of EVERY trajectory (gap G), the "
                        "validation side of split_temporal_gapped, ALL Re rows -- for a checkpoint "
                        "trained with train.py --gapped-holdout. Bare = 100 40 (the cc_gth_* pilot); "
                        "or give N G. fit == val (no disjoint row). Writes 'gapped_holdout': [N, G] "
                        "into bounds.json. Not with --fit-re/--val-re/--affine/--spatial.")
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--affine", action="store_true",
                   help="fit h = intercept + slope*mean|u_input| per channel, "
                        "instead of one constant width. Slope from all rows, "
                        "intercept re-levelled on --val-re.")
    p.add_argument("--spatial", action="store_true",
                   help="fit per-element half-widths (a pixel map, optionally "
                        "frame-resolved) instead of one constant per channel; "
                        "prints every variant against the constant on the val "
                        "rows and writes half_width_map into bounds.json")
    p.add_argument("--spatial-variant", default="auto", choices=("auto",) + SPATIAL_VARIANTS,
                   help="which spatial variant to write (auto: best VAL objective "
                        "per channel). Pick on a disjoint fit/val split first, "
                        "then refit that variant on all held-out rows to ship.")
    p.add_argument("--const", type=float, nargs="+", default=[0.0090, 0.0065, 0.0],
                   help="currently shipped constant widths, for comparison")
    p.add_argument("--fit-ensemble", type=Path, nargs="+", default=None,
                   help="fit the widths on the MEAN of these members instead of on --ckpt alone. An ensemble's "
                        "residuals are smaller than any member's, so widths fitted on one model and shipped with "
                        "an ensemble over-widen every element. Pass the same members build_submission --ensemble "
                        "will get.")
    p.add_argument("--write-for", type=Path, default=None,
                   help="write the bounds beside THIS checkpoint (with its sha) "
                        "instead of --ckpt: fit on a held-out twin, ship with a "
                        "--train-all model that has no honest residuals")
    p.add_argument("--no-write", action="store_true",
                   help="print the fit but do not write <ckpt>.bounds.json")
    main(resolve_rows(p.parse_args()))
