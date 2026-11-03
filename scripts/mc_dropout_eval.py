#!/usr/bin/env python
"""MC-dropout inference for the PNO_D probe: is test-time dropout a better point or a width signal?

    python scripts/mc_dropout_eval.py --bounds-zip Z --ckpt P [P ...] --samples 1 4 8 --stride 20 \\
        [--holdout-re 13950 26700] [--force-dropout 0.1] [--seed 0]

PNO_D (Buelte / Scholl / Kutyniok, arXiv 2502.12902) fine-tunes a deterministic operator with dropout
and the energy score (scripts/train.py --energy-score-samples) and then predicts with MC dropout. This
scores that on the standard held-out windows, the same way scripts/variance_inflation.py scores its
post-hoc transforms: windows from split_by_condition + build_eval_arrays, bounds PINNED to what the
given twin zip's predict() returns, per Reynolds row with the kit scorer and the exact composite.

For each M in --samples, each member k runs M stochastic forwards -- model.eval() first, then ONLY the
nn.Dropout modules back to train mode (set_mc_dropout), so no other layer ever sees train mode -- and
the prediction is the mean over the M*k samples (= mean over members of each member's sample mean).
The deterministic reference, labelled 'M=0 (dropout off)', is one plain eval forward per member and
is always computed; every M in --samples, 1 included, is stochastic.

Per configuration it prints, per held-out row and pooled:
  * rel_l2 / tke / mvpe / sps / est.final with the zip's bounds PINNED (absolute lower/upper, as
    variance_inflation does), and sps_rc / final_rc with the bounds RE-CENTRED on this prediction
    (lower' = p + (lower - p_zip), upper' = p + (upper - p_zip): the head's offsets, moved with the
    point). Pinned penalises any change of the point; re-centred is the fair read for fine-tuned members.
  * r(spr,|e|): Pearson over the row's MEASURED elements (target u != 0) between the per-element
    sample spread (population std over the M*k samples; for M=0 the std over the k deterministic
    members, i.e. plain ensemble disagreement) and |p - y|. r(head,|e|) is the same for the zip's
    half-width (upper - lower)/2 -- what the shipped head already achieves -- and r(spr,head) says
    whether the spread is a NEW signal or the one the head already has.
  * per-lead rel_l2 at leads 0 5 10 15 19.

The first block scores the zip's own predict() output (the twin's known number); the M=0 block also
prints its max |difference| to that prediction, which is ~0 when --ckpt are exactly the zip's members
(a check that this script's forward reproduces the shipped path) and is the size of the point change
otherwise. Timings are this script's eager fp32 loop, NOT the grader's path.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kit"))
sys.path.insert(0, str(ROOT / "scripts"))

# Imported BEFORE the zip's submission module: both ship a vendored einops, and the package's copy
# being first in sys.modules is what variance_inflation relies on too (same files, same code).
from pde_competition.model import FlowerConfig, FlowerS2S  # noqa: E402

LEADS = (0, 5, 10, 15, 19)


def set_mc_dropout(model: nn.Module) -> int:
    """MC-dropout mode: ``model.eval()``, then ONLY the ``nn.Dropout`` modules back to train mode.

    Every other module -- GroupNorm, the warps, anything added later that behaves differently in
    train mode -- stays in eval. (Flower has no BatchNorm today, so this equals model.train() for the
    current architecture; the helper is written so it stays right if that ever changes.) Returns how
    many of the switched modules are ACTIVE (p > 0): a checkpoint built with dropout_rate 0 has
    nn.Identity in their place and returns 0, i.e. its "samples" are all identical."""
    model.eval()
    active = 0
    for q in model.modules():
        if isinstance(q, nn.Dropout):
            q.train()
            active += int(q.p > 0)
    return active


def load_member(path, force_dropout: float | None = None) -> FlowerS2S:
    """FlowerS2S.from_checkpoint, or -- with ``force_dropout`` -- the same config rebuilt at that
    dropout rate before the strict load (nn.Dropout / nn.Identity carry no parameters). The latter is
    the free baseline: MC dropout injected into members that were never trained with it."""
    if force_dropout is None:
        return FlowerS2S.from_checkpoint(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"])
    cfg["resolution"] = tuple(cfg["resolution"])
    cfg["dropout_rate"] = float(force_dropout)
    model = FlowerS2S(FlowerConfig(**cfg))
    model.load_state_dict(ckpt["state_dict"])
    return model


@torch.no_grad()
def accumulate(model: FlowerS2S, x: np.ndarray, n_samples: int, device, batch_size: int,
               acc_sum: np.ndarray, acc_sq: np.ndarray, c: int) -> None:
    """Add ``n_samples`` forwards of ``model`` on every window of ``x`` into the running sums.

    The caller sets the mode (eval, or set_mc_dropout). Batching and channel handling follow
    model.predict_numpy: the input is sliced to cfg.c_in, the output fills channels [0, c_out) of the
    grader's C and anything beyond stays 0. --mask-channel models get the per-frame measured mask in
    channel 2 exactly as the shipped submission.py writes it. ``acc_sum`` (N, T, H, W, C) and
    ``acc_sq`` (N, T, H, W, c) are float64: the spread is sqrt(E[s^2] - E[s]^2) of O(0.1) fields with
    O(0.01) spread, which float32 would cancel badly."""
    cfg = model.cfg
    for i in range(0, x.shape[0], batch_size):
        xb = np.array(x[i:i + batch_size, ..., :cfg.c_in], dtype=np.float32, copy=True)
        if cfg.mask_channel and xb.shape[-1] > 2:
            xb[..., 2] = (xb[..., 0] != 0).astype(np.float32)
        xt = torch.from_numpy(xb).to(device)
        s = sq = None
        for _ in range(n_samples):
            yb = model(xt).double()
            s = yb if s is None else s + yb
            q = yb[..., :c] ** 2
            sq = q if sq is None else sq + q
        b = xb.shape[0]
        acc_sum[i:i + b, ..., :cfg.c_out] += s.cpu().numpy()
        acc_sq[i:i + b] += sq.cpu().numpy()


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def main() -> None:
    from pde_competition.dataset import build_eval_arrays, list_runs, split_by_condition
    from pde_competition.composite import estimate
    from variance_inflation import REF_TIME
    import scoring as ks

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bounds-zip", type=pathlib.Path, required=True,
                    help="twin zip whose predict() supplies the (pinned) lower/upper")
    ap.add_argument("--ckpt", type=pathlib.Path, nargs="+", required=True,
                    help="member checkpoints; the prediction is the mean over members")
    ap.add_argument("--samples", type=int, nargs="+", default=[1, 4, 8],
                    help="MC-dropout sample counts per member. The deterministic 'M=0 (dropout off)' row "
                         "is always computed; a 0 here is ignored.")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--holdout-re", type=int, nargs="+", default=[13950, 26700])
    ap.add_argument("--force-dropout", type=float, default=None, metavar="P",
                    help="rebuild every member at dropout rate P before loading (MC dropout injected into "
                         "members not trained with it, or at a different rate). Default: each "
                         "checkpoint's own rate.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch seed, reset before every configuration so each is reproducible alone")
    a = ap.parse_args()
    if any(m < 0 for m in a.samples):
        ap.error("--samples must be >= 0")
    if a.force_dropout is not None and not 0.0 <= a.force_dropout < 1.0:
        ap.error("--force-dropout must be in [0, 1)")
    samples = sorted({m for m in a.samples if m > 0})
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)

    _t, val = split_by_condition(list_runs(), holdout_re=tuple(a.holdout_re))
    x, y, keys = build_eval_arrays(val, stride=a.stride, return_keys=True)
    c = ks.measured_channels(y)
    rows = sorted({k[0] for k in keys}) + ["all"]
    row_idx = {re: np.array([i for i, k in enumerate(keys) if re == "all" or k[0] == re]) for re in rows}

    work = pathlib.Path(tempfile.mkdtemp(prefix="mcdrop_"))
    with zipfile.ZipFile(a.bounds_zip) as z:
        z.extractall(work / "sub")
    sys.path.insert(0, str(work / "sub"))
    import submission                                                                 # noqa: E402
    # Copies, as in variance_inflation: the fast path may hand out views of pinned buffers, and a
    # --mask-channel zip writes into its input's channel 2 in place.
    out = submission.predict(x.copy(), {})
    p_zip = np.array(out["prediction"], dtype=np.float32, copy=True)
    if out.get("lower") is None or out.get("upper") is None:
        raise SystemExit(f"{a.bounds_zip.name} returns no lower/upper; nothing to pin")
    lower = np.array(out["lower"], dtype=np.float32, copy=True)
    upper = np.array(out["upper"], dtype=np.float32, copy=True)
    del out
    off_lo, off_up = lower - p_zip, upper - p_zip
    half = ((upper - lower) / 2.0)[..., :c]
    scored = np.broadcast_to((y[..., :1] != 0), y[..., :c].shape)                  # measured elements

    print(f"windows {x.shape[0]}  holdout {a.holdout_re}  stride {a.stride}  measured channels {c}  "
          f"device {device}")
    print(f"bounds pinned from {a.bounds_zip}")
    print(f"members ({len(a.ckpt)}): " + ", ".join(str(p) for p in a.ckpt))
    if a.force_dropout is not None:
        print(f"--force-dropout {a.force_dropout}: every member rebuilt at that rate")

    def report(label, p, spread, secs=None, extra=""):
        print(f"\n{label}" + (f"   [{secs:.1f}s, {1e3 * secs / x.shape[0]:.2f} ms/window, eager fp32]"
                               if secs is not None else "") + extra)
        print(f"  {'row':>6} {'rel_l2':>8} {'tke':>8} {'mvpe':>8} {'sps':>8} {'est.final':>10} "
              f"{'sps_rc':>8} {'final_rc':>9} {'r(spr,|e|)':>11} {'r(head,|e|)':>12} {'r(spr,head)':>12} "
              f"{'mean spr':>9}")
        lo_rc, up_rc = p + off_lo, p + off_up
        for re in rows:
            idx = row_idx[re]
            pp, tt = p[idx], y[idx]
            rel = float(np.mean(ks.rel_l2_per_sample(pp, tt, c)))
            tke = float(np.mean(ks.tke_rel_l2_per_sample(pp, tt, c)))
            mv = ks.mvpe_rel_l2(pp, tt)
            sps, _ = ks.aggregate_sps(pp, tt, c, lower=lower[idx], upper=upper[idx])
            sps_rc, _ = ks.aggregate_sps(pp, tt, c, lower=lo_rc[idx], upper=up_rc[idx])
            s = (ks.score_error(rel), ks.score_error(tke), ks.score_error(mv))
            fin = estimate(*s, REF_TIME, ks.score_sps(sps))
            fin_rc = estimate(*s, REF_TIME, ks.score_sps(sps_rc))
            m = scored[idx]
            err = np.abs(pp[..., :c] - tt[..., :c])[m]
            hw = half[idx][m]
            if spread is None:
                r_se = r_sh = msp = float("nan")
            else:
                sp = spread[idx][m]
                r_se, r_sh, msp = pearson(sp, err), pearson(sp, hw), float(sp.mean())
            print(f"  {str(re):>6} {s[0]:8.3f} {s[1]:8.3f} {s[2]:8.3f} {ks.score_sps(sps):8.3f} {fin:10.3f} "
                  f"{ks.score_sps(sps_rc):8.3f} {fin_rc:9.3f} {r_se:11.4f} {pearson(hw, err):12.4f} "
                  f"{r_sh:12.4f} {msp:9.5f}")
        print(f"  per-lead rel_l2 (raw, lower is better)")
        print(f"  {'row':>6} " + " ".join(f"{'L' + str(j):>8}" for j in LEADS))
        for re in rows:
            idx = row_idx[re]
            cells = [float(np.mean(ks.rel_l2_per_sample(p[idx, j:j + 1], y[idx, j:j + 1], c))) for j in LEADS]
            print(f"  {str(re):>6} " + " ".join(f"{v:8.4f}" for v in cells))

    report(f"zip predict() of {a.bounds_zip.name} (reference: the twin's own point and bounds)", p_zip, None)

    def run_config(n_samples, stochastic):
        torch.manual_seed(a.seed)
        acc_sum = np.zeros(y.shape, np.float64)
        acc_sq = np.zeros(y.shape[:-1] + (c,), np.float64)
        t0 = time.time()
        for path in a.ckpt:
            model = load_member(path, a.force_dropout).to(device).float()
            if stochastic:
                if set_mc_dropout(model) == 0:
                    print(f"  WARNING: {path} has no active dropout (dropout_rate "
                          f"{model.cfg.dropout_rate}): its {n_samples} samples are identical. "
                          f"--force-dropout P injects it.")
            else:
                model.eval()
            accumulate(model, x, n_samples, device, a.batch_size, acc_sum, acc_sq, c)
            del model
        if device.type == "cuda":
            torch.cuda.synchronize()
        secs = time.time() - t0
        k = n_samples * len(a.ckpt)
        mean = acc_sum / k
        var = np.maximum(acc_sq / k - mean[..., :c] ** 2, 0.0)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32), secs

    p0, spread0, secs = run_config(1, stochastic=False)
    d = np.abs(p0[..., :c] - p_zip[..., :c])
    report(f"M=0 (dropout off): plain eval, mean of {len(a.ckpt)} members; spread = member disagreement",
           p0, spread0 if len(a.ckpt) > 1 else None, secs,
           extra=f"\n  vs zip predict(): max|dp| {float(d.max()):.3g}, mean|dp| {float(d.mean()):.3g}")
    for m in samples:
        p, spread, secs = run_config(m, stochastic=True)
        d = np.abs(p[..., :c] - p0[..., :c])
        report(f"M={m}: MC dropout, {m} samples x {len(a.ckpt)} members = {m * len(a.ckpt)} forwards per window",
               p, spread, secs,
               extra=f"\n  vs M=0: max|dp| {float(d.max()):.3g}, mean|dp| {float(d.mean()):.3g}")


if __name__ == "__main__":
    main()
