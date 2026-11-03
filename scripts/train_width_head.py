"""The trained SPS width head (W52).

A small CNN maps per-element features to a per-element HALF-WIDTH h >= 0,
symmetric about the point prediction (so lower <= pred <= upper holds by
construction), trained to maximise a smooth surrogate of the scorer's reward
    1{|r| <= h} * exp(-2h/sigma)       ->   sigmoid((h - |r|)/tau) * exp(-2h/sigma)
on OUT-OF-FOLD residuals (W51a's fold ensembles: every row predicted by three
recipe models that never saw it). Features per element (channel-stacked over
time): the member-mean prediction u,v; member disagreement z_u, z_v; the
window's mean input speed; normalised pixel coordinates; the frame index.

Validation is exact (fit_bounds.objective_map) on --val-re rows, which are
excluded from training, against the constant and the linear-in-disagreement
width (a,b fitted on the training rows). The head then re-levels one scalar per
channel on --val-re -- the same convention as every shipped width -- and saves
head.pth + head.json beside the first fold checkpoint's directory.

    python scripts/train_width_head.py --folds 'outputs/runs/ws_cfk_fold*/best_score.pth' \
        --val-re 13950 26700 --stride 20 --epochs 30

--outputs (2026-09-21, directions.md H1/H2) relaxes the SYMMETRY that every head
so far has assumed. Nothing in kit/scoring.py::aggregate_sps requires it: the
reward is 1{lower <= target <= upper} * exp(-(upper-lower)/sigma), which pays for
the WIDTH and for covering the target, not for the interval being centred.

    --outputs 1   (default)  h            -> lower = p - h,  upper = p + h
    --outputs 2   H1         (lo, up)     -> lower = p - lo, upper = p + up
    --outputs 3   H2         (d, lo, up)  -> lower = p+d-lo, upper = p+d+up

--outputs 1 is bit-identical to everything shipped. --outputs 3 also moves the
POINT PREDICTION, which feeds rel_l2, tke and mvpe (0.7 of the composite), so it
reports the accuracy of the corrected field next to the width reward and --delta-l2
lets the correction be trained against the residual as well as against the scorer.
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pde_competition.paths import OUTPUTS
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_bounds import SIGMA_GLOBAL, fit_channel, objective_map  # noqa: E402
from oof_width_model import GRID, window_speed  # noqa: E402
from oof_disagreement import collect as collect_rows, fit_ab  # noqa: E402

# feature planes per frame: u, v, z_u, z_v, speed, y, x, t  -> 8 channels ("base")
N_FEAT = 8
# "ext" (W81, 2026-09-17) appends 9 planes computed from the INPUT window and
# the prediction: input temporal mean u, v; input temporal std u, v; last input
# frame u, v; measured fraction; |frame-to-frame change| of the prediction u, v.
N_EXTRA = 9
# "ext2" (2026-09-18) appends 3 more, all about UNSTEADINESS rather than level:
# the per-pixel mean |frame-to-frame change| of the INPUT window (u, v), and its
# ratio to the prediction's own |change| -- the model's local damping factor.
# The existing ext planes give the input's time mean, time std, last frame and
# measured fraction; none of them separates a slow drift from fast unsteadiness,
# which is what sets the residual scale. The input increment is MEASURED at test
# time, so it carries the hidden Reynolds regime instead of extrapolating it.
N_EXTRA2 = 3
PLANES = {"base": N_FEAT, "ext": N_FEAT + N_EXTRA, "ext2": N_FEAT + N_EXTRA + N_EXTRA2}


# Output layouts. n_out=2 is the symmetric head everything has shipped; the two
# asymmetric layouts put the DELTA first (signed, so it cannot share softplus) and
# the offsets last, so `out[:, -4:]` is (lo_u, lo_v, up_u, up_v) in both of them.
N_OUT = {1: 2, 2: 4, 3: 6}


class WidthHead(nn.Module):
    """(B, n_feat, H, W) -> (B, n_out, H, W). ~20k parameters at width 24.

    n_out = 2 (default, everything shipped): half-widths h_u, h_v, symmetric about the
    prediction. n_out = 4 (H1): lo_u, lo_v, up_u, up_v, so lower = p - lo and upper = p + up.
    n_out = 6 (H2): d_u, d_v then the same four offsets, with the interval centred on p + d.
    The delta planes are signed and bounded by a tanh times ``delta_scale`` (a buffer, so it
    travels with the state_dict); they start at exactly zero, so an H2 head at initialisation
    predicts the uncorrected field.

    With n_pfeat > 0 (fresh-eyes proposition 8) the head also takes the model's penultimate
    feature map, (B, n_pfeat, H, W) per frame, and reduces it through a learned 1x1 bottleneck
    to n_bottle planes that are concatenated to the hand-built ones. n_pfeat = 0 -- everything
    shipped so far -- leaves the module bit-identical to the two-argument-free original.
    """
    def __init__(self, width=24, n_feat=N_FEAT, n_pfeat=0, n_bottle=8, n_out=2, delta_scale=(0.0097, 0.0016)):
        super().__init__()
        assert n_out in (2, 4, 6), f"n_out must be 2, 4 or 6, got {n_out}"
        self.n_pfeat, self.n_bottle, self.n_out = n_pfeat, (n_bottle if n_pfeat else 0), n_out
        if n_pfeat:
            self.bottle = nn.Conv2d(n_pfeat, n_bottle, 1)
            nn.init.zeros_(self.bottle.bias)
        self.net = nn.Sequential(
            nn.Conv2d(n_feat + self.n_bottle, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, dilation=1), nn.GELU(),
            nn.Conv2d(width, n_out, 1))
        # start near the constant width: softplus(bias) ~ 0.009. With n_out = 4 both
        # offsets start at 0.009, so the interval is the symmetric head's at init.
        nn.init.constant_(self.net[-1].bias, np.log(np.exp(0.009) - 1))
        nn.init.zeros_(self.net[-1].weight)
        if n_out == 6:
            with torch.no_grad():
                self.net[-1].bias[:2].zero_()                       # tanh(0) = 0: no correction at init
            self.register_buffer("delta_scale", torch.tensor(delta_scale, dtype=torch.float32).view(1, 2, 1, 1))

    def forward(self, f, pf=None):
        if self.n_pfeat:
            if pf is None:
                raise ValueError("this head was built with n_pfeat > 0 and needs the penultimate planes")
            f = torch.cat([f, F.gelu(self.bottle(pf))], dim=1)
        o = self.net(f)
        if self.n_out < 6:
            return F.softplus(o).clamp(max=0.5)
        return torch.cat([self.delta_scale * torch.tanh(o[:, :2]),
                          F.softplus(o[:, 2:]).clamp(max=0.5)], dim=1)


def extra_planes(x, mean, ext2=False):
    """Raw "ext" planes. x: (N, T_in, H, W, >=2) input window (zeros = unmeasured);
    mean: (N, T, H, W, 2) prediction. -> (N, T, 9, H, W) float32, un-normalised:
    [in_mean_u, in_mean_v, in_std_u, in_std_v, last_u, last_v, measured_frac, |du|, |dv|]."""
    N, T, H, W = mean.shape[:4]
    u = x[..., :2].astype(np.float32)
    meas = (x[..., 0] != 0).astype(np.float32)[..., None]            # (N, T_in, H, W, 1)
    cnt = np.maximum(meas.sum(1), 1.0)                                 # (N, H, W, 1)
    m = (u * meas).sum(1) / cnt                                        # (N, H, W, 2)
    sd = np.sqrt(np.maximum(((u - m[:, None]) ** 2 * meas).sum(1) / cnt, 0.0))
    last = u[:, -1]                                                    # (N, H, W, 2)
    frac = meas.mean(1)                                                # (N, H, W, 1)
    d = np.abs(np.diff(mean, axis=1, prepend=mean[:, :1]))             # (N, T, H, W, 2), 0 at t=0
    n_e = N_EXTRA + (N_EXTRA2 if ext2 else 0)
    e = np.empty((N, T, n_e, H, W), np.float32)
    e[:, :, 0:2] = m.transpose(0, 3, 1, 2)[:, None]
    e[:, :, 2:4] = sd.transpose(0, 3, 1, 2)[:, None]
    e[:, :, 4:6] = last.transpose(0, 3, 1, 2)[:, None]
    e[:, :, 6:7] = frac.transpose(0, 3, 1, 2)[:, None]
    e[:, :, 7:9] = d.transpose(0, 1, 4, 2, 3)
    if ext2:
        # mean |frame-to-frame change| of the MEASURED input, per pixel: (N, H, W, 2).
        # Only steps where both frames are measured count, so the mask edge does not
        # register as motion.
        both = (meas[:, 1:] * meas[:, :-1])                        # (N, T_in-1, H, W, 1)
        du = np.abs(np.diff(u, axis=1)) * both
        cnt2 = np.maximum(both.sum(1), 1.0)
        ind = du.sum(1) / cnt2                                     # (N, H, W, 2)
        e[:, :, 9:11] = ind.transpose(0, 3, 1, 2)[:, None]
        # damping factor: how much smaller the PREDICTION's motion is than the input's.
        # Dimensionless, so it needs no normalisation constant; clipped because the
        # prediction's increment is ~0 at t=0 and in the unmeasured region.
        e[:, :, 11] = np.clip(ind.mean(-1).transpose(0, 1, 2)[:, None]
                              / (d.mean(-1) + 1e-6), 0.0, 10.0)
    return e


def features(mean, std, speed, norm, extra=None):
    """mean/std: (N, T, H, W, 2) numpy; speed: (N,); extra: raw ext planes (N, T, 9, H, W)
    or None. -> torch (N*T, n_feat, H, W) normalised."""
    N, T, H, W = mean.shape[:4]
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    tt = np.linspace(-1, 1, T)
    n_feat = N_FEAT + (extra.shape[2] if extra is not None else 0)
    f = np.empty((N, T, n_feat, H, W), np.float32)
    f[:, :, 0] = (mean[..., 0] - norm["mu"][0]) / norm["sd"][0]
    f[:, :, 1] = (mean[..., 1] - norm["mu"][1]) / norm["sd"][1]
    f[:, :, 2] = std[..., 0] / norm["zref"][0]
    f[:, :, 3] = std[..., 1] / norm["zref"][1]
    f[:, :, 4] = ((speed - norm["speed_mu"]) / norm["speed_sd"])[:, None, None, None]
    f[:, :, 5] = yy; f[:, :, 6] = xx
    f[:, :, 7] = tt[None, :, None, None]
    if extra is not None:
        e = extra
        for c in range(2):
            f[:, :, 8 + c] = (e[:, :, c] - norm["mu"][c]) / norm["sd"][c]          # input mean
            f[:, :, 10 + c] = e[:, :, 2 + c] / norm["in_std_ref"][c]                  # input std
            f[:, :, 12 + c] = (e[:, :, 4 + c] - norm["mu"][c]) / norm["sd"][c]      # last frame
            f[:, :, 15 + c] = e[:, :, 7 + c] / norm["delta_ref"][c]                   # |d pred|
        f[:, :, 14] = e[:, :, 6]                                                        # measured fraction
        if e.shape[2] > N_EXTRA:
            for c in range(2):
                f[:, :, 17 + c] = e[:, :, 9 + c] / norm["in_d_ref"][c]                   # input |d| per pixel
            f[:, :, 19] = e[:, :, 11]                                                    # damping ratio (dimensionless)
    return torch.from_numpy(f.reshape(N * T, n_feat, H, W))


def surrogate_loss(h, r, scored, tau, sigma=SIGMA_GLOBAL, w=None, straight_through=False):
    """Negative smooth reward, mean over scored elements. h, r: (B, 2, H, W).

    ``w`` (B, 1, 1, 1) re-weights each FRAME by the scorer's own per-sample accuracy factor.
    This is not a regularisation choice, it is what ``kit/scoring.py::aggregate_sps`` does:

        aggregate = 0.5*branch(dm) + 0.3*branch(tke) + 0.2*branch(mvpe)

    and every branch multiplies the SAME per-element width reward by a per-sample factor
    ``(1 - metric/(0.5+metric))``. So the scorer pays less for covering an element in a
    window the model predicts badly, and the head -- which weights every window equally --
    is optimising the wrong average. A covered element in an accurate window is worth
    strictly more than one in an inaccurate window, and the head has never known that.

    ``straight_through`` makes the FORWARD pass use the exact indicator 1{r <= h} while the
    BACKWARD pass keeps the sigmoid's gradient. The smoothed forward is biased: at tau the
    head is rewarded for near-misses the scorer pays nothing for. W194 showed the surrogate
    beats a proper scoring rule precisely because it optimises the right functional, so
    removing its remaining bias is the natural next step rather than replacing it.
    """
    soft = torch.sigmoid((h - r) / tau)
    inside = soft + ((r <= h).to(soft.dtype) - soft).detach() if straight_through else soft
    reward = inside * torch.exp(-2.0 * h / sigma)
    if w is not None:
        reward = reward * w
    return -(reward * scored).sum() / scored.sum().clamp(min=1)


def surrogate_loss_asym(lo, up, r, scored, tau, sigma=SIGMA_GLOBAL, w=None, straight_through=False):
    """H1's loss: the same smooth surrogate with the interval free to be lopsided.

    ``r`` is the SIGNED residual target - pred (the symmetric loss takes |r|), ``lo``/``up``
    the non-negative offsets, so the scorer's element reward is

        1{-lo <= r <= up} * exp(-(lo + up)/sigma)

    and the two one-sided indicators are smoothed separately. At lo == up == h and r >= 0
    this is the symmetric surrogate times sigmoid((r+h)/tau), which is 1 to machine precision
    for any residual the head is not already covering on the wrong side.
    """
    soft = torch.sigmoid((r + lo) / tau) * torch.sigmoid((up - r) / tau)
    if straight_through:
        exact = ((r >= -lo) & (r <= up)).to(soft.dtype)
        inside = soft + (exact - soft).detach()
    else:
        inside = soft
    reward = inside * torch.exp(-(lo + up) / sigma)
    if w is not None:
        reward = reward * w
    return -(reward * scored).sum() / scored.sum().clamp(min=1)


def objective_map_asym(pred, target, lo, up, sigma=SIGMA_GLOBAL):
    """Exact SPS width reward for an ASYMMETRIC per-element interval, one channel.

    The generalisation of fit_bounds.objective_map: with lo == up == h it returns exactly
    what objective_map returns, to the bit. That needs care -- ``target - pred`` is left in
    the inputs' own dtype, as objective_map computes it, so that |r| <= h and -h <= r <= h
    test the same float; and lo + up is exactly 2h, so the exponent matches too. It is what
    makes an --outputs 2 number comparable to the incumbent's 0.6572.
    """
    scored = target != 0.0
    n = int(np.count_nonzero(scored))
    if n == 0:
        return float("nan"), float("nan")
    r = np.asarray(target) - np.asarray(pred)
    lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), pred.shape)
    up = np.broadcast_to(np.asarray(up, dtype=np.float64), pred.shape)
    inside = (r >= -lo) & (r <= up) & scored
    reward = np.where(inside, np.exp(-(lo + up) / sigma), 0.0)
    return float(reward.sum() / n), float(np.count_nonzero(inside) / n)


def accuracy_pair(pred, y):
    """(rel_l2, tke_rel_l2) of a 2-channel field against its target, mean over windows.

    Only used to report what an --outputs 3 correction does to the 0.7 of the composite
    weight that is NOT sps; the head's own objective never sees these.
    """
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parents[1] / "kit"))
    import scoring as ks
    p3 = np.concatenate([pred, np.zeros_like(pred[..., :1])], axis=-1)
    y3 = np.concatenate([y, np.zeros_like(y[..., :1])], axis=-1)
    return float(ks.rel_l2_per_sample(p3, y3, 2).mean()), float(ks.tke_rel_l2_per_sample(p3, y3, 2).mean())


def scorer_sample_weights(mean, y, c=2, normalize=0.5, wd=0.5, wt=0.3, wm=0.2):
    """The per-WINDOW factor aggregate_sps applies to the width reward, as (N,) float32.

    Mirrors kit/scoring.py exactly: each of the three branches scales the width reward by
    (1 - metric/(normalize + metric)) for its own per-sample metric, and the three are
    combined 0.5/0.3/0.2. Returned normalised to mean 1 so it re-weights without changing
    the loss scale (and so tau, tuned against the unweighted loss, still means what it did).
    """
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parents[1] / "kit"))
    import scoring as ks
    p3 = np.concatenate([mean, np.zeros_like(mean[..., :1])], axis=-1)
    y3 = np.concatenate([y, np.zeros_like(y[..., :1])], axis=-1)
    dm = ks.rel_l2_per_sample(p3, y3, c)
    tk = ks.tke_rel_l2_per_sample(p3, y3, c)
    mv = ks.mvpe_rel_l2_per_sample(p3, y3)
    f = (wd * (1.0 - dm / (normalize + dm))
         + wt * (1.0 - tk / (normalize + tk))
         + wm * (1.0 - mv / (normalize + mv)))
    f = np.asarray(f, np.float32)
    return f / max(float(f.mean()), 1e-8)


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    folds = sorted(p for g in a.folds for p in glob.glob(g))
    if a.pfeat:
        rows, pfeats = collect_rows(folds, a.stride, device, pool_members=a.pool_members, with_feats=True)
    else:
        rows, pfeats = collect_rows(folds, a.stride, device, pool_members=a.pool_members), {}   # re -> (mean, std, y), 2 channels
    # window speeds: recompute inputs per row (collect() does not keep x) -- cheap at this stride
    from pde_competition.dataset import build_eval_arrays, list_runs
    speeds = {}; extras = {}
    for re in rows:
        runs = [r for r in list_runs("train_real") if r.re_nominal == re]
        x, _ = build_eval_arrays(runs, stride=a.stride)
        rep = rows[re][0].shape[0] // x.shape[0]               # members pooled per window (1 unless --pool-members)
        if rep > 1:
            x = np.tile(x, (rep, 1, 1, 1, 1))
        speeds[re] = window_speed(x)
        if a.planes in ("ext", "ext2"):
            extras[re] = extra_planes(x, rows[re][0], ext2=a.planes == "ext2")
    ext = a.planes in ("ext", "ext2")
    train_keys = [k for k in sorted(rows) if k not in a.val_re and k >= a.min_train_re]
    if a.train_re is not None:
        want = set(a.train_re)
        missing = sorted(want - set(train_keys))
        if missing:
            raise SystemExit(f"--train-re names rows that are not available to train on: {missing}\n"
                             f"  available (not in --val-re, above --min-train-re): {train_keys}")
        train_keys = [k for k in train_keys if k in want]
        if not train_keys:
            raise SystemExit("--train-re selected no rows")
        print(f"--train-re: training on {len(train_keys)} of the available rows")
    val_keys = [k for k in sorted(rows) if k in a.val_re]
    if a.min_train_re > 0:
        dropped = [k for k in sorted(rows) if k not in a.val_re and k < a.min_train_re]
        print(f"--min-train-re {a.min_train_re}: dropped {len(dropped)} low rows {dropped}")
    print(f"train rows {train_keys}\nval rows   {val_keys}")

    # normalisation from the training rows
    Mtr = np.concatenate([rows[k][0] for k in train_keys]); Ztr = np.concatenate([rows[k][1] for k in train_keys])
    Ytr = np.concatenate([rows[k][2] for k in train_keys]); Str = np.concatenate([speeds[k] for k in train_keys])
    sc = Ytr != 0
    norm = {"mu": [float(Mtr[..., c][sc[..., c]].mean()) for c in range(2)],
            "sd": [float(Mtr[..., c][sc[..., c]].std() + 1e-8) for c in range(2)],
            "zref": [float(Ztr[..., c][sc[..., c]].mean() + 1e-8) for c in range(2)],
            "speed_mu": float(Str.mean()), "speed_sd": float(Str.std() + 1e-8)}
    if ext:
        Etr = np.concatenate([extras[k] for k in train_keys])
        norm["in_std_ref"] = [float(Etr[:, :, 2 + c].mean() + 1e-8) for c in range(2)]
        norm["delta_ref"] = [float(Etr[:, :, 7 + c].mean() + 1e-8) for c in range(2)]
        print(f"ext planes: in_std_ref {norm['in_std_ref']}, delta_ref {norm['delta_ref']}")
        if a.planes == "ext2":
            norm["in_d_ref"] = [float(Etr[:, :, 9 + c].mean() + 1e-8) for c in range(2)]
            print(f"ext2 planes: in_d_ref {norm['in_d_ref']}, damping ratio mean {float(Etr[:, :, 11].mean()):.3f}")

    # --outputs 1 trains on |residual| (the symmetric head cannot see a sign); 2 and 3 need
    # the SIGNED residual target - pred, because which side the target falls on is the whole
    # point of letting the interval be lopsided.
    n_out = N_OUT[a.outputs]
    signed = a.outputs > 1

    def tensors(keys):
        f = torch.cat([features(rows[k][0], rows[k][1], speeds[k], norm, extras.get(k) if ext else None) for k in keys])
        res = [(rows[k][2] - rows[k][0]) if signed else np.abs(rows[k][0] - rows[k][2]) for k in keys]
        r = torch.from_numpy(np.concatenate([e.reshape(-1, *rows[k][0].shape[2:4], 2) for e, k in zip(res, keys)])).permute(0, 3, 1, 2)
        s = torch.from_numpy(np.concatenate([(rows[k][2] != 0).reshape(-1, *rows[k][2].shape[2:4], 2) for k in keys]).astype(np.float32)).permute(0, 3, 1, 2)
        return f, r, s
    ftr, rtr, str_ = tensors(train_keys)
    if signed:
        # THE MECHANISM, measured before anything is trained: if the out-of-fold residual is
        # symmetric about zero on the scored elements, H1 has nothing to find and the run is
        # a formality. P(r > 0) far from 0.5, or a median far from the mean, says otherwise.
        # Every 8th frame -- rtr is ~1 GB and a skew over 30 M samples is not the uncertainty here.
        for c, nm in enumerate(["u", "v"]):
            rc = rtr[::8, c][str_[::8, c] > 0].numpy().astype(np.float64)
            print(f"signed OOF residual {nm}: mean {rc.mean():+.2e}  median {np.median(rc):+.2e}  "
                  f"P(r>0) {float((rc > 0).mean()):.4f}  |r| mean {np.abs(rc).mean():.2e}  "
                  f"skew {float(((rc - rc.mean()) ** 3).mean() / max(rc.std() ** 3, 1e-30)):+.3f}"
                  f"  (n={rc.size})")
            del rc
    mse0 = 1.0
    if a.outputs == 3:                     # --delta-l2's reference scale, chunked: rtr is ~1 GB
        num = den = 0.0
        for i in range(0, rtr.shape[0], 4096):
            num += float(((rtr[i:i + 4096] ** 2) * str_[i:i + 4096]).sum())
            den += float(str_[i:i + 4096].sum())
        mse0 = num / max(den, 1.0)
    wtr = None
    if a.scorer_weight:
        # one factor per WINDOW, expanded to the per-frame rows the head trains on
        per_row = {k: scorer_sample_weights(rows[k][0], rows[k][2]) for k in train_keys}
        T_f = rows[train_keys[0]][0].shape[1]
        wtr = torch.from_numpy(np.concatenate([np.repeat(per_row[k], T_f) for k in train_keys])).float()
        wtr = wtr.view(-1, 1, 1, 1)
        print(f"scorer sample weights: mean {float(wtr.mean()):.4f} "
              f"range {float(wtr.min()):.3f}-{float(wtr.max()):.3f} (the scorer's own accuracy factor)")
        assert wtr.shape[0] == ftr.shape[0], (wtr.shape, ftr.shape)
    if a.re_weight > 0:
        # H3: TILT the head toward the board-like rows instead of RESTRICTING to them.
        # --min-train-re dropped the low rows outright and lost (0.6535 against 0.6572): the
        # low rows still teach the head what a small residual looks like. This keeps every
        # window and weights row k by (k / max training row) ** P, normalised to mean 1, so
        # P = 0 is the shipped head and P -> large approaches --min-train-re's restriction.
        # NOTE the val rows (13950, 26700) are held out, so the top TRAINING rows carry the
        # tilt as proxies for the board's Re, which is what row 26700 stands in for.
        T_f = rows[train_keys[0]][0].shape[1]
        remax = float(max(train_keys))
        rw = torch.from_numpy(np.concatenate(
            [np.full(rows[k][0].shape[0] * T_f, (k / remax) ** a.re_weight, np.float32)
             for k in train_keys])).float().view(-1, 1, 1, 1)
        rw = rw / rw.mean()
        wtr = rw if wtr is None else (wtr * rw) / (wtr * rw).mean()
        print(f"--re-weight {a.re_weight:g}: row factors " +
              " ".join(f"{k}:{(k / remax) ** a.re_weight:.4f}" for k in train_keys) +
              f"  -> combined weight range {float(wtr.min()):.4f}-{float(wtr.max()):.4f}, "
              f"mean {float(wtr.mean()):.4f}")
        assert wtr.shape[0] == ftr.shape[0], (wtr.shape, ftr.shape)
    T_FR = rows[train_keys[0]][0].shape[1]                 # frames per window: flat index // T_FR = window index
    n_pfeat = 0
    if a.pfeat:
        assert all(rows[k][0].shape[1] == T_FR for k in rows), "rows disagree on frames per window"
        pf_tr = torch.from_numpy(np.concatenate([pfeats[k] for k in train_keys]))   # (Nwin, F, H, W) float16
        n_pfeat = pf_tr.shape[1]
        assert pf_tr.shape[0] * T_FR == ftr.shape[0], (pf_tr.shape, ftr.shape)
        print(f"penultimate planes: {n_pfeat} per window -> bottleneck {a.pfeat}; "
              f"{pf_tr.numel() * 2 / 2**30:.1f} GiB held (float16)")
    if a.drop_z:
        # ABLATION (independent review, 2026-09-16): a head WITHOUT the
        # disagreement planes ships with a single model at zero time cost. If
        # it keeps most of lin_z's gain from prediction/speed/coordinates alone,
        # the ensemble is not needed for the widths.
        ftr[:, 2:4] = 0.0
        print("ablation: disagreement planes zeroed (--drop-z)")
    print(f"training elements: {ftr.shape[0]} frames x {ftr.shape[2]}x{ftr.shape[3]}; scored fraction {str_.mean():.3f}")

    dscale = tuple(float(a.delta_frac * s) for s in norm["sd"])          # per-channel bound on the H2 correction
    if a.outputs == 3:
        print(f"--outputs 3: delta bounded by +-{dscale[0]:.4g} (u), +-{dscale[1]:.4g} (v) "
              f"= {a.delta_frac:g} x the training sd; --delta-l2 {a.delta_l2:g} (mse reference {mse0:.3e})")
    head = WidthHead(a.width, PLANES[a.planes], n_pfeat, a.pfeat, n_out=n_out, delta_scale=dscale).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((ftr.shape[0] + a.batch - 1) // a.batch))
    n = ftr.shape[0]
    for ep in range(1, a.epochs + 1):
        head.train(); perm = torch.randperm(n); ep_loss = 0.0; t0 = time.time()
        tau = a.tau * (0.5 ** (ep / a.epochs))            # anneal the smoothing toward the indicator
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            f, r, s = ftr[idx].to(device), rtr[idx].to(device), str_[idx].to(device)
            pf = pf_tr[idx // T_FR].float().to(device) if a.pfeat else None
            w = None if wtr is None else wtr[idx].to(device)
            out = head(f, pf)
            if a.outputs == 1:
                loss = surrogate_loss(out, r, s, tau, w=w, straight_through=a.straight_through)
            else:
                rr = r if a.outputs == 2 else r - out[:, :2]     # residual of the CORRECTED prediction
                loss = surrogate_loss_asym(out[:, -4:-2], out[:, -2:], rr, s, tau,
                                           w=w, straight_through=a.straight_through)
                if a.outputs == 3 and a.delta_l2 > 0:
                    loss = loss + a.delta_l2 * ((rr ** 2) * s).sum() / s.sum().clamp(min=1) / mse0
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            ep_loss += loss.item() * len(idx)
        print(f"epoch {ep:3d}  surrogate reward {-ep_loss / n:.4f}  tau {tau:.5f}  {time.time() - t0:.0f}s", flush=True)

    tot = {}          # validation accumulator: name -> [weighted sum, n]; empty when there are no val rows (W63 bug)
    corrected = {}    # --outputs 3 only: re -> the prediction with the head's delta applied
    acc = {}          # --outputs 3 only: re -> {"rel_l2": [before, after], "tke": [...]}
    if not val_keys:
        print("no validation rows: the head is trained on every row (shipping head); nothing to validate")
    else:
        # exact validation on the held-out rows, per channel, against const and lin_z fitted on the training rows
        head.eval()
        print("\nEXACT validation on the held-out rows (fit_bounds.objective_map):")
        k_lvl = [1.0, 1.0]; tot = {}
        for c, name in enumerate(["u", "v"]):
            h0 = fit_channel(Mtr[..., c], Ytr[..., c], GRID)[0]
            aa, bb, zref = fit_ab(rows, c, train_keys)
            for re in val_keys:
                m, z, y = rows[re]
                note = ""
                with torch.no_grad():
                    f = features(m, z, speeds[re], norm, extras.get(re) if ext else None)
                    if a.drop_z:
                        f[:, 2:4] = 0.0
                    pfv = torch.from_numpy(pfeats[re]) if a.pfeat else None
                    hh = torch.cat([head(f[i:i + 256].to(device),
                                         pfv[torch.arange(i, min(i + 256, f.shape[0])) // T_FR].float().to(device)
                                         if a.pfeat else None).cpu()
                                    for i in range(0, f.shape[0], 256)])
                hh = hh.numpy().reshape(m.shape[0], m.shape[1], n_out, *m.shape[2:4]).transpose(0, 1, 3, 4, 2)
                pc = m[..., c]                                  # the point prediction the interval sits on
                if a.outputs == 3:
                    pc = pc + hh[..., c]
                    corrected.setdefault(re, m[..., :2].copy())[..., c] = pc
                n_sc = int(np.count_nonzero(y[..., c] != 0))
                res = {"const": objective_map(m[..., c], y[..., c], h0)[0],
                       "lin_z": objective_map(m[..., c], y[..., c], aa + bb * z[..., c] / zref)[0]}
                if a.outputs == 1:
                    hh = hh[..., c]
                    res["head"] = objective_map(pc, y[..., c], hh)[0]
                    obj_k = lambda k: objective_map(pc, y[..., c], k * hh)[0]       # noqa: E731
                else:
                    lo, up = hh[..., n_out - 4 + c], hh[..., n_out - 2 + c]
                    res["head"] = objective_map_asym(pc, y[..., c], lo, up)[0]
                    obj_k = lambda k: objective_map_asym(pc, y[..., c], k * lo, k * up)[0]   # noqa: E731
                    sc_c = y[..., c] != 0
                    note = (f"   up/lo {float(np.mean(up[sc_c]) / max(np.mean(lo[sc_c]), 1e-12)):.3f}"
                            f"  width {float(np.mean(lo[sc_c] + up[sc_c])):.4f}")
                # re-level: one scalar on the head's output, chosen on this row (reported separately)
                ks = np.linspace(0.6, 1.6, 51); best = max(((obj_k(k), k) for k in ks))
                res["head_relevel"] = best[0]
                for kk, v in res.items():
                    tot.setdefault(kk, [0.0, 0]); tot[kk][0] += v * n_sc; tot[kk][1] += n_sc
                print(f"  {name} row {re}: " + "  ".join(f"{kk} {v:.4f} ({v / res['const'] - 1:+.2%})" for kk, v in res.items()) + f"   best k {best[1]:.2f}" + note)
        print("pooled over channels and held-out rows:")
        for kk, (num, nn_) in tot.items():
            print(f"  {kk:<13}{num / nn_:.4f}  ({num / nn_ / (tot['const'][0] / tot['const'][1]) - 1:+.2%})")
        if a.outputs == 3:
            # THE RISK H2 CARRIES, measured: the corrected field feeds rel_l2, tke and mvpe,
            # which are 0.7 of the composite weight. A width gain bought with a worse
            # prediction is not a gain.
            print("\naccuracy of the CORRECTED prediction (the head's delta applied), held-out rows:")
            for re in val_keys:
                m, _, y = rows[re]
                b_dm, b_tke = accuracy_pair(m[..., :2], y)
                c_dm, c_tke = accuracy_pair(corrected[re], y)
                print(f"  row {re}: rel_l2 {b_dm:.4f} -> {c_dm:.4f} ({c_dm / b_dm - 1:+.2%})   "
                      f"tke {b_tke:.4f} -> {c_tke:.4f} ({c_tke / b_tke - 1:+.2%})")
                acc[str(re)] = {"rel_l2": [b_dm, c_dm], "tke": [b_tke, c_tke]}

    out_dir = OUTPUTS / "width_head"; out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "width": a.width, "norm": norm, "n_feat": PLANES[a.planes],
                "planes": a.planes, "drop_z": bool(a.drop_z),
                "scorer_weight": bool(a.scorer_weight), "straight_through": bool(a.straight_through), "n_pfeat": n_pfeat, "n_bottle": int(a.pfeat),
                "n_out": n_out, "outputs": int(a.outputs), "delta_scale": list(dscale), "delta_l2": float(a.delta_l2),
                "train_rows": train_keys, "val_rows": val_keys, "folds": [str(f) for f in folds], "pool_members": bool(a.pool_members)}, out_dir / f"{a.name}.pth")
    (out_dir / f"{a.name}.json").write_text(json.dumps(
        {"norm": norm, "pooled": {k: v[0] / v[1] for k, v in tot.items()}, "n_out": n_out, "accuracy": acc}, indent=2))
    print(f"\nsaved {out_dir / a.name}.pth")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", nargs="+", required=True)
    p.add_argument("--val-re", type=int, nargs="*", default=[13950, 26700],
                   help="held-out rows for exact validation; pass with no values to train on all rows")
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--tau", type=float, default=0.002, help="initial smoothing of the coverage indicator, in field units; annealed by 2x over training")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--name", default="head_v1")
    p.add_argument("--planes", choices=sorted(PLANES), default="base",
                   help="input planes: base (8) or ext (+9 from the input window and the prediction's motion)")
    p.add_argument("--re-weight", type=float, default=0.0,
                   help="H3: weight each training row by (re / max_train_re) ** P, normalised to mean 1. "
                        "0 (default) is the shipped head, bit-identical. Composes with --scorer-weight.")
    p.add_argument("--scorer-weight", action="store_true",
                   help="weight each window's loss by the factor aggregate_sps actually applies to the width "
                        "reward: 0.5(1-dm~) + 0.3(1-tke~) + 0.2(1-mvpe~). The head has always weighted every "
                        "window equally, which is not what the scorer pays.")
    p.add_argument("--straight-through", action="store_true",
                   help="exact indicator 1{r<=h} in the forward pass, sigmoid gradient in the backward. "
                        "Removes the smoothed forward's bias toward near-misses the scorer pays nothing for.")
    p.add_argument("--drop-z", action="store_true", help="zero the disagreement planes: the single-model head")
    p.add_argument("--outputs", type=int, choices=(1, 2, 3), default=1,
                   help="1 (default, bit-identical to every shipped head): one symmetric half-width per "
                        "channel. 2 (H1): separate lower and upper offsets -- aggregate_sps pays for "
                        "upper-lower and for covering the target, and requires no symmetry, so a "
                        "conditionally skewed residual is covered by less total width. 3 (H2): also a "
                        "signed correction to the prediction, which makes the head a post-processor and "
                        "moves rel_l2/tke/mvpe as well as sps.")
    p.add_argument("--delta-frac", type=float, default=0.10, metavar="F",
                   help="--outputs 3 only: bound the prediction correction at F x the training sd of that "
                        "channel (tanh), so a diverging head cannot destroy the field it is correcting")
    p.add_argument("--delta-l2", type=float, default=0.0, metavar="LAM",
                   help="--outputs 3 only: add LAM x (mse of the corrected residual / mse of the raw one) "
                        "to the loss, so the correction is trained against accuracy and not only against "
                        "the width reward. 0 = the pure-sps arm.")
    p.add_argument("--train-re", type=int, nargs="*", default=None, metavar="RE",
                   help="train on EXACTLY these rows (still excluding --val-re). Default None = every row "
                        "that is not a validation row. Two uses: an ALL-DATA head (pass --val-re with no "
                        "rows, so nothing is held out and every row trains, the head analogue of shipping "
                        "all-data model members), and a LEARNING CURVE in the number of training rows, "
                        "which is the only honest way to justify the all-data head -- if the curve is still "
                        "rising at 16 rows, the 2 extra rows are worth having. Unlike --min-train-re this "
                        "selects rows without confounding count with Reynolds scale.")
    p.add_argument("--min-train-re", type=int, default=0, metavar="RE",
                   help="drop training rows below RE. The board's scored rows sit at the TOP of our range -- board #11's "
                        "subscores match the twin's row-26700 numbers, not the pool's -- while 52%% of the head's "
                        "training windows come from rows below 13950, whose residual scale is ~2.3x smaller. "
                        "Validation rows (--val-re) are unaffected.")
    p.add_argument("--pfeat", type=int, default=0, metavar="K",
                   help="fresh-eyes proposition 8: feed the model's penultimate feature map through a learned "
                        "1x1 bottleneck to K extra planes (0 = off, the shipping head). A head fitted with "
                        "--pfeat cannot be shipped until submission_flower.py grows the same hook.")
    p.add_argument("--pool-members", action="store_true",
                   help="train on every fold member's OWN residuals (pooled) instead of the member-mean's; "
                        "pass best_score.pth and last.pth per fold run to pool over checkpoints too (proposition 4)")
    main(p.parse_args())
