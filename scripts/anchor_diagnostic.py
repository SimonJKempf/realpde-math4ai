#!/usr/bin/env python
"""Input-statistics anchoring, as a post-processor on a zip's predictions
(fresh-eyes proposition 1, 2026-09-17): shift each predicted window's per-pixel
time-mean toward the INPUT window's time-mean (alpha), and/or rescale its
fluctuation toward the input's per-pixel std (beta). The input window is the
one thing at an unseen Re that carries the true regime, so a model that drifts
in mean level or turbulence level could be corrected from it at zero cost.

    python scripts/anchor_diagnostic.py submission/dist/<twin zip> [--holdout-re 13950 26700] [--stride 50]

Runs the zip's predict() once on the held-out windows (like score_local), then
evaluates the grid alpha x beta with kit/scoring.py's metrics per Re row: rel_l2,
tke, mvpe, sps (bounds move with the prediction, half-widths preserved) and
est.final at the reference time. Also prints the input-mean persistence
baseline. CPU is fine (a few minutes). Writes outputs/analyses/anchor_<zip>.json.
Ship criterion (pre-registered): on the extrapolation row, mvpe error -10% on
every twin with rel_l2 within +0.0003 and sps not down.
"""
import argparse, importlib.util, json, shutil, sys, tempfile, zipfile
from pathlib import Path

import numpy as np

from pde_competition.dataset import build_eval_arrays, list_runs, split_by_condition
from pde_competition.paths import KIT, OUTPUTS, REPO

sys.path.insert(0, str(KIT))
import scoring as ks  # noqa: E402
from pde_competition.composite import estimate  # noqa: E402

REF_TIME = 91.370559
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BETAS = (0.0, 0.5, 1.0)


def input_stats(x: np.ndarray):
    """Per-pixel time-mean and std of the measured input frames, (N,H,W,2) each, and a (N,H,W,1) has-data mask."""
    u = x[..., :2].astype(np.float32)
    meas = (x[..., 0] != 0).astype(np.float32)[..., None]          # (N,T,H,W,1)
    cnt = meas.sum(1)                                              # (N,H,W,1)
    ok = cnt > 0
    m = (u * meas).sum(1) / np.maximum(cnt, 1.0)
    sd = np.sqrt(np.maximum(((u - m[:, None]) ** 2 * meas).sum(1) / np.maximum(cnt, 1.0), 0.0))
    return m, sd, ok


def anchor(pred: np.ndarray, in_mean, in_std, ok, alpha: float, beta: float) -> np.ndarray:
    """pred (N,T,H,W,C) -> anchored copy; channels 0..1 only, only where the input has data."""
    out = pred.copy()
    p = pred[..., :2]
    pm = p.mean(1, keepdims=True)                                  # (N,1,H,W,2)
    fl = p - pm
    if beta:
        ps = np.sqrt(np.maximum((fl ** 2).mean(1, keepdims=True), 1e-12))
        ratio = np.clip(in_std[:, None] / ps, 0.5, 2.0)
        fl = fl * ((1.0 - beta) + beta * ratio)
    new_mean = pm + alpha * (in_mean[:, None] - pm)
    anchored = new_mean + fl
    mask = ok[:, None]                                             # (N,1,H,W,1)
    out[..., :2] = np.where(mask, anchored, p)
    return out


def metrics(pred, target, lower, upper, idx):
    p, t = pred[idx], target[idx]
    c = ks.measured_channels(t)
    rel = float(np.mean(ks.rel_l2_per_sample(p, t, c)))
    tke = float(np.mean(ks.tke_rel_l2_per_sample(p, t, c)))
    mvpe = ks.mvpe_rel_l2(p, t)
    sps, cov = ks.aggregate_sps(p, t, c, lower=None if lower is None else lower[idx],
                                upper=None if upper is None else upper[idx])
    r = dict(rel_l2_err=rel, tke_err=tke, mvpe_err=mvpe, rel_l2_score=ks.score_error(rel),
             tke_score=ks.score_error(tke), mvpe_score=ks.score_error(mvpe), sps_score=ks.score_sps(sps), coverage=float(cov))
    r["est_final"] = estimate(r["rel_l2_score"], r["tke_score"], r["mvpe_score"], REF_TIME, r["sps_score"])
    return r


def main(a):
    runs = list_runs()
    _tr, val = split_by_condition(runs, holdout_re=tuple(a.holdout_re))
    x, y, keys = build_eval_arrays(val, stride=a.stride, return_keys=True)
    print(f"held-out cells: {sorted(r.key for r in val)}; windows {x.shape[0]}")
    work = Path(tempfile.mkdtemp(prefix="anchor_"))
    try:
        zipfile.ZipFile(a.zip).extractall(work)
        sys.path.insert(0, str(work))
        spec = importlib.util.spec_from_file_location("submission", work / "submission.py")
        sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
        out = sub.predict(x, {})
    finally:
        shutil.rmtree(work, ignore_errors=True)
    pred = np.asarray(out["prediction"], np.float32)
    lower = np.asarray(out["lower"], np.float32) if "lower" in out else None
    upper = np.asarray(out["upper"], np.float32) if "upper" in out else None
    hw_lo = None if lower is None else pred - lower
    hw_up = None if upper is None else upper - pred
    in_mean, in_std, ok = input_stats(x)
    rows = sorted({k[0] for k in keys})
    groups = {str(re): np.array([i for i, k in enumerate(keys) if k[0] == re]) for re in rows}
    groups["all"] = np.arange(len(keys))

    results = {}
    # baseline: input-mean persistence (the regime carrier alone)
    persist = pred.copy(); persist[..., :2] = np.where(ok[:, None], np.broadcast_to(in_mean[:, None], pred[..., :2].shape), pred[..., :2])
    for g, idx in groups.items():
        results.setdefault("persistence", {})[g] = metrics(persist, y, None, None, idx)
    for al in ALPHAS:
        for be in BETAS:
            pa = anchor(pred, in_mean, in_std, ok, al, be)
            lo = None if hw_lo is None else pa - hw_lo
            up = None if hw_up is None else pa + hw_up
            key = f"a{al:.2f}_b{be:.2f}"
            for g, idx in groups.items():
                results.setdefault(key, {})[g] = metrics(pa, y, lo, up, idx)

    base = "a0.00_b0.00"
    for g in groups:
        print(f"\n=== row {g} ({len(groups[g])} windows)   [deltas vs the unanchored zip]")
        b = results[base][g]
        print(f"  {'variant':<14} {'rel_l2':>8} {'tke':>8} {'mvpe':>8} {'sps':>8} {'est.final':>9}   {'d rel':>7} {'d tke':>7} {'d mvpe':>7} {'d sps':>7} {'d est':>7}")
        pr = results["persistence"][g]
        print(f"  {'input-mean':<14} {pr['rel_l2_score']:8.3f} {pr['tke_score']:8.3f} {pr['mvpe_score']:8.3f} {'-':>8} {'-':>9}")
        for key in results:
            if key == "persistence":
                continue
            r = results[key][g]
            print(f"  {key:<14} {r['rel_l2_score']:8.3f} {r['tke_score']:8.3f} {r['mvpe_score']:8.3f} {r['sps_score']:8.3f} {r['est_final']:9.3f}   "
                  f"{r['rel_l2_score']-b['rel_l2_score']:+7.3f} {r['tke_score']-b['tke_score']:+7.3f} {r['mvpe_score']-b['mvpe_score']:+7.3f} "
                  f"{r['sps_score']-b['sps_score']:+7.3f} {r['est_final']-b['est_final']:+7.3f}")
    out_dir = OUTPUTS / "analyses"; out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"anchor_{Path(a.zip).stem}.json"
    path.write_text(json.dumps({"zip": str(a.zip), "holdout_re": a.holdout_re, "stride": a.stride, "results": results}, indent=1))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", type=Path)
    ap.add_argument("--holdout-re", type=int, nargs="+", default=[13950, 26700])
    ap.add_argument("--stride", type=int, default=50)
    main(ap.parse_args())
