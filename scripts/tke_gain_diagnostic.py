#!/usr/bin/env python
"""Post-hoc fluctuation LEVEL on a zip's predictions: how much of the tke error
is a level defect (fixable for free) rather than a shape defect (not fixable)?

    p' = mean_t(p) + g * (p - mean_t(p))

kit/scoring.py's kinetic_energy is a per-pixel TEMPORAL VARIANCE over the 20
predicted frames, so this multiplies the predicted KE map by exactly g**2, and
leaves the time-mean -- hence mvpe, which reads only the time-mean at 36 probe
points -- exactly unchanged. Both invariants are asserted below.

Writing a = ||ke_p|| / ||ke_t|| and rho = cos(ke_p, ke_t), the metric is
err**2 = a**2 - 2*a*rho + 1, whose minimum over a pure level change is
sqrt(1 - rho**2) at g**2 = rho / a. rho is a ceiling no post-hoc knob can move;
a is a closed-form one. An L2-trained predictor is a conditional mean, whose own
variance is Var(E[y|x]) while the target needs E[Var(y|x)] + Var(E[y|x]), so
a < 1 generically -- the question is by how much on the EXTRAPOLATION row, which
is what the board scores and what no training row can have calibrated.

This is NOT W144's beta (closed as a null). That rescaled per PIXEL toward the
input's per-pixel std, clipped to [0.5, 2]; a 20-frame per-pixel std carries
~30% sampling noise, and a noisy multiplicative gain ADDS uncorrelated variance
to the KE map. One scalar per window averages that noise over ~2048 pixels.

Variants:
  gain g      : global constant gains
  tau         : per-window g from the INPUT window's measured TKE,
                g = (tke_in / tke_pred) ** (tau/2) -- the input window is 20
                frames of the same trajectory at the hidden Re, so it measures
                the turbulence level the model has to extrapolate. (The width
                head is already fed the input's per-pixel std as planes 10-11;
                the point prediction is not.)
  ramp c      : g_k = 1 + c*(k-1)/(T-1), re-centred so the time-mean is exact
                (error_by_frame: TKE err 0.652 on frames 1-10 vs 0.850 on 11-20)
  mask        : zero the prediction where the input window has no measured data
                (tke/mvpe/rel_l2 are all UNMASKED in the scorer; but our training
                rel_l2 is unmasked too, so the model already sees those zeros --
                expect little)

    python scripts/tke_gain_diagnostic.py submission/dist/<twin zip> [--holdout-re 13950 26700] [--stride 50]

CPU is fine. Writes outputs/analyses/tkegain_<zip>.json.
"""
import argparse, importlib.util, json, shutil, sys, tempfile, zipfile
from pathlib import Path

import numpy as np

from pde_competition.dataset import build_eval_arrays, list_runs, split_by_condition
from pde_competition.paths import KIT, OUTPUTS

sys.path.insert(0, str(KIT))
import scoring as ks  # noqa: E402
from pde_competition.composite import estimate  # noqa: E402

REF_TIME = 91.370559
GAINS = (0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30, 1.50)
TAUS = (0.25, 0.50, 1.00)
RAMPS = (0.20, 0.40)


def split(p):
    """(N,T,H,W,C) -> time-mean (N,1,H,W,2) and fluctuation (N,T,H,W,2), channels 0..1."""
    q = p[..., :2]
    m = q.mean(1, keepdims=True)
    return m, q - m


def apply_gain(pred, g):
    """g: scalar, or (N,1,1,1,1), or (1,T,1,1,1) for a ramp. Time-mean preserved exactly."""
    out = pred.copy()
    m, fl = split(pred)
    fl = fl * g
    fl = fl - fl.mean(1, keepdims=True)      # re-centre: a ramped gain would otherwise move the mean
    out[..., :2] = m + fl
    return out


def window_tke(p):
    """Field-summed TKE per window, (N,) -- the scorer's kinetic_energy summed over pixels."""
    return ks.kinetic_energy(p[..., :2]).reshape(p.shape[0], -1).sum(1)


def a_rho(pred, target, idx):
    """Pooled level a = ||ke_p||/||ke_t|| and shape rho = cos(ke_p, ke_t) over a row."""
    kp = ks.kinetic_energy(pred[idx][..., :2]).reshape(len(idx), -1)
    kt = ks.kinetic_energy(target[idx][..., :2]).reshape(len(idx), -1)
    a = float(np.linalg.norm(kp) / max(np.linalg.norm(kt), 1e-12))
    rho = float((kp * kt).sum() / max(np.linalg.norm(kp) * np.linalg.norm(kt), 1e-12))
    return a, rho


def metrics(pred, target, lower, upper, idx):
    p, t = pred[idx], target[idx]
    c = ks.measured_channels(t)
    rel = float(np.mean(ks.rel_l2_per_sample(p, t, c)))
    tke = float(np.mean(ks.tke_rel_l2_per_sample(p, t, c)))
    mvpe = ks.mvpe_rel_l2(p, t)
    sps, cov = ks.aggregate_sps(p, t, c, lower=None if lower is None else lower[idx],
                                upper=None if upper is None else upper[idx])
    r = dict(rel_l2_err=rel, tke_err=tke, mvpe_err=mvpe, rel_l2_score=ks.score_error(rel),
             tke_score=ks.score_error(tke), mvpe_score=ks.score_error(mvpe),
             sps_score=ks.score_sps(sps), coverage=float(cov))
    r["est_final"] = estimate(r["rel_l2_score"], r["tke_score"], r["mvpe_score"], REF_TIME, r["sps_score"])
    return r


def main(a):
    runs = list_runs()
    _tr, val = split_by_condition(runs, holdout_re=tuple(a.holdout_re))
    x, y, keys = build_eval_arrays(val, stride=a.stride, return_keys=True)
    print(f"held-out cells: {sorted(r.key for r in val)}; windows {x.shape[0]}")
    work = Path(tempfile.mkdtemp(prefix="tkegain_"))
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

    rows = sorted({k[0] for k in keys})
    groups = {str(re): np.array([i for i, k in enumerate(keys) if k[0] == re]) for re in rows}
    groups["all"] = np.arange(len(keys))

    # --- invariants: the two properties the whole diagnostic rests on -------
    g_test = apply_gain(pred, 1.30)
    ke0 = ks.kinetic_energy(pred[:4, ..., :2]); ke1 = ks.kinetic_energy(g_test[:4, ..., :2])
    assert np.allclose(ke1, ke0 * 1.30 ** 2, rtol=1e-4, atol=1e-9), "gain must scale the KE map by g^2"
    assert np.allclose(ks.mvpe_rel_l2(g_test, y), ks.mvpe_rel_l2(pred, y), rtol=0, atol=1e-6), "gain must leave mvpe exact"
    print("invariants OK: KE scales as g^2, mvpe exactly unchanged\n")

    # --- P1: the floor. The input window as a predictor of the target window.
    print("FLOOR -- the input window scored as if it were the prediction")
    print("(two independent 20-frame samples of the same flow; a perfect predictor of the")
    print(" CONDITIONAL mean sits near this/sqrt(2), and higher, since adjacent windows correlate)")
    xp = np.zeros_like(pred); xp[..., :2] = x[..., :2]
    for g, idx in groups.items():
        c = ks.measured_channels(y[idx])
        ft = float(np.mean(ks.tke_rel_l2_per_sample(xp[idx], y[idx], c)))
        fm = ks.mvpe_rel_l2(xp[idx], y[idx])
        print(f"  row {g:>6}: tke_err {ft:.4f} (floor ~{ft/np.sqrt(2):.4f})   mvpe_err {fm:.4f} (floor ~{fm/np.sqrt(2):.4f})")

    results, variants = {}, {}
    for g in GAINS:
        variants[f"g{g:.2f}"] = apply_gain(pred, g)
    tke_in, tke_pr = window_tke(x), np.maximum(window_tke(pred), 1e-12)
    for tau in TAUS:
        gn = (tke_in / tke_pr) ** (tau / 2.0)
        variants[f"tau{tau:.2f}"] = apply_gain(pred, gn.reshape(-1, 1, 1, 1, 1).astype(np.float32))
    T = pred.shape[1]
    for c_ in RAMPS:
        ramp = (1.0 + c_ * np.arange(T) / max(T - 1, 1)).reshape(1, T, 1, 1, 1).astype(np.float32)
        variants[f"ramp{c_:.2f}"] = apply_gain(pred, ramp)
    meas = (x[..., 0] != 0).any(1)                                   # (N,H,W) -- measured anywhere in the input
    masked = pred.copy(); masked[..., :2] *= meas[:, None, ..., None]
    variants["mask0"] = masked

    for key, pv in variants.items():
        lo = None if hw_lo is None else pv - hw_lo
        up = None if hw_up is None else pv + hw_up
        for g, idx in groups.items():
            results.setdefault(key, {})[g] = metrics(pv, y, lo, up, idx)

    base = "g1.00"
    for g, idx in groups.items():
        av, rv = a_rho(pred, y, idx)
        print(f"\n=== row {g} ({len(idx)} windows)   a {av:.3f}  rho {rv:.3f}  "
              f"g* = sqrt(rho/a) {np.sqrt(max(rv, 1e-9) / max(av, 1e-9)):.3f}  "
              f"best possible tke_err sqrt(1-rho^2) {np.sqrt(max(1 - rv * rv, 0)):.3f}")
        b = results[base][g]
        print(f"  {'variant':<10} {'rel_l2':>8} {'tke':>8} {'mvpe':>8} {'sps':>8} {'est.final':>9}   "
              f"{'d rel':>7} {'d tke':>7} {'d mvpe':>7} {'d sps':>7} {'d est':>7}")
        for key in variants:
            r = results[key][g]
            print(f"  {key:<10} {r['rel_l2_score']:8.3f} {r['tke_score']:8.3f} {r['mvpe_score']:8.3f} {r['sps_score']:8.3f} {r['est_final']:9.3f}   "
                  f"{r['rel_l2_score']-b['rel_l2_score']:+7.3f} {r['tke_score']-b['tke_score']:+7.3f} {r['mvpe_score']-b['mvpe_score']:+7.3f} "
                  f"{r['sps_score']-b['sps_score']:+7.3f} {r['est_final']-b['est_final']:+7.3f}")

    out_dir = OUTPUTS / "analyses"; out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tkegain_{Path(a.zip).stem}.json"
    path.write_text(json.dumps({"zip": str(a.zip), "holdout_re": a.holdout_re, "stride": a.stride, "results": results}, indent=1))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", type=Path)
    ap.add_argument("--holdout-re", type=int, nargs="+", default=[13950, 26700])
    ap.add_argument("--stride", type=int, default=50)
    main(ap.parse_args())
