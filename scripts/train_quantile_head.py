#!/usr/bin/env python
"""sps P3: predict the DISTRIBUTION of |r|, then pick the width the scorer actually wants.

The shipped head is trained on a smooth surrogate of the scorer's reward,

    1{|r| <= h} * exp(-2h/sigma)   ->   sigmoid((h - |r|)/tau) * exp(-2h/sigma)

and outputs one width per element. That surrogate has two known weaknesses: its
gradient vanishes both for elements comfortably inside the interval and for
elements hopelessly outside it, and tau is a free parameter that had to be tuned
(W122/W128/W158). This takes the other route. A head predicts Q quantiles of |r|
by PINBALL loss -- a proper scoring rule with clean gradients everywhere and no
free smoothing parameter -- and at inference the width is chosen by maximising
the scorer's own reward over the predicted quantiles:

    h* = argmax over q of   tau_q * exp(-2 h_q / sigma)

because tau_q IS the estimated F(h_q). So the training objective is proper and
the selection rule is exact, where the incumbent approximates both at once.

WHY THIS IS NOT ONE OF THE FOUR NULL PROBES. Recalibrating the head's output
(W191, +0.05%), adding disagreement planes (W184z, null), head width (saturates
at w48) and the pooled-member head all left the TRAINING OBJECTIVE alone. This
is the only thing that changes it, and it is the last named proposition on the
sps axis. The prior is still poor -- four probes say these features are close to
exhausted, and W191's per-element oracle says the absolute ceiling is +23% with
most of it irreducible scatter -- but the axis is untested.

Quantile crossing is prevented by construction: the head emits the lowest
quantile and non-negative increments, so h_1 <= h_2 <= ... <= h_Q always.

    python scripts/train_quantile_head.py --head outputs/width_head/head_ext_tau05_w48.pth \
        --folds 'outputs/runs/ws_cfk_fold*/best_score.pth' --stride 20 --epochs 30
"""
from __future__ import annotations
import argparse, glob, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fit_bounds import objective_map, SIGMA_GLOBAL                # noqa: E402
from oof_disagreement import collect as collect_rows              # noqa: E402
from oof_width_model import window_speed                          # noqa: E402
from train_width_head import features, extra_planes               # noqa: E402
from pde_competition.paths import OUTPUTS                         # noqa: E402


class QuantileHead(nn.Module):
    """(B, n_feat, H, W) -> (B, 2, Q, H, W) non-crossing quantiles of |r| per channel."""
    def __init__(self, width, n_feat, n_q):
        super().__init__()
        self.n_q = n_q
        self.net = nn.Sequential(
            nn.Conv2d(n_feat, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, 2 * n_q, 1))
        # start every quantile near the constant width, as the incumbent head does
        nn.init.zeros_(self.net[-1].weight)
        b = torch.full((2 * n_q,), float(np.log(np.exp(0.009 / n_q) - 1)))
        nn.init.constant_(self.net[-1].bias, 0.0)
        with torch.no_grad():
            self.net[-1].bias.copy_(b)

    def forward(self, f):
        o = F.softplus(self.net(f))                       # (B, 2Q, H, W), all positive
        o = o.reshape(o.shape[0], 2, self.n_q, *o.shape[2:])
        return torch.cumsum(o, dim=2).clamp(max=0.5)      # non-crossing by construction


def pinball(pred, target, taus, scored):
    """pred (B,2,Q,H,W), target (B,2,H,W), taus (Q,). Mean over scored elements."""
    d = target.unsqueeze(2) - pred
    t = taus.view(1, 1, -1, 1, 1)
    l = torch.maximum(t * d, (t - 1.0) * d).mean(dim=2)   # average over quantiles
    return (l * scored).sum() / scored.sum().clamp(min=1.0)


def pick(h, taus, sigma=SIGMA_GLOBAL):
    """h (..., Q) -> the width maximising tau_q * exp(-2 h_q / sigma), and that reward."""
    val = taus * np.exp(-2.0 * h / sigma)
    i = np.argmax(val, axis=-1)
    return np.take_along_axis(h, i[..., None], axis=-1)[..., 0]


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.head, map_location="cpu", weights_only=False)
    norm, planes, ext = ck["norm"], ck["planes"], ck["planes"] != "base"
    folds = sorted(p for g in a.folds for p in glob.glob(g))
    print(f"incumbent head {a.head} (planes {planes}, width {ck['width']}, val rows {ck['val_rows']})")
    print(f"{len(folds)} folds; stride {a.stride}; device {device}; Q={a.n_q}\n")
    rows = collect_rows(folds, a.stride, device)

    from pde_competition.dataset import build_eval_arrays, list_runs
    speeds, extras = {}, {}
    for re in rows:
        runs = [r for r in list_runs("train_real") if r.re_nominal == re]
        x, _ = build_eval_arrays(runs, stride=a.stride)
        speeds[re] = window_speed(x)
        if ext:
            extras[re] = extra_planes(x, rows[re][0])

    val_keys = [k for k in rows if k in set(ck["val_rows"])]
    train_keys = [k for k in rows if k not in set(ck["val_rows"])]
    print(f"train rows {sorted(train_keys)}\nval rows   {sorted(val_keys)}\n")

    def tensors(keys):
        f = torch.cat([features(rows[k][0], rows[k][1], speeds[k], norm, extras.get(k) if ext else None) for k in keys])
        r = torch.from_numpy(np.concatenate([np.abs(rows[k][0] - rows[k][2]).reshape(-1, *rows[k][0].shape[2:4], 2) for k in keys])).permute(0, 3, 1, 2)
        s = torch.from_numpy(np.concatenate([(rows[k][2] != 0).reshape(-1, *rows[k][2].shape[2:4], 2) for k in keys]).astype(np.float32)).permute(0, 3, 1, 2)
        return f, r, s

    ftr, rtr, str_ = tensors(train_keys)
    if ck.get("drop_z"):
        ftr[:, 2:4] = 0.0                                  # the shipped head is drop-z; match it
        print("disagreement planes zeroed, matching the incumbent")
    taus_np = np.linspace(a.tau_lo, a.tau_hi, a.n_q)
    taus = torch.tensor(taus_np, dtype=torch.float32, device=device)
    print(f"quantile levels: {np.round(taus_np, 3).tolist()}")
    print(f"training elements: {ftr.shape[0]} frames x {ftr.shape[2]}x{ftr.shape[3]}; scored {str_.mean():.3f}\n")

    head = QuantileHead(a.width, ftr.shape[1], a.n_q).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    n = ftr.shape[0]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((n + a.batch - 1) // a.batch))
    for ep in range(1, a.epochs + 1):
        head.train(); perm = torch.randperm(n); tot = 0.0; t0 = time.time()
        for i in range(0, n, a.batch):
            j = perm[i:i + a.batch]
            f, r, s = ftr[j].to(device), rtr[j].to(device), str_[j].to(device)
            loss = pinball(head(f), r, taus, s)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss) * len(j)
        print(f"epoch {ep:3d}  pinball {tot / n:.6f}  {time.time() - t0:.0f}s", flush=True)

    print("\nEXACT validation on the held-out rows (fit_bounds.objective_map), against the incumbent:")
    print(f"  {'row':>7} {'ch':>3} {'incumbent':>10} {'quantile':>10} {'gain':>8}")
    head.eval(); tot = {"inc": [0.0, 0], "qh": [0.0, 0]}
    from train_width_head import WidthHead
    inc = WidthHead(ck["width"], ck["n_feat"], ck.get("n_pfeat", 0), ck.get("n_bottle", 8)).to(device)
    inc.load_state_dict(ck["state_dict"]); inc.eval()
    for re in sorted(val_keys):
        m, z, y = rows[re]
        f = features(m, z, speeds[re], norm, extras.get(re) if ext else None)
        if ck.get("drop_z"):
            f[:, 2:4] = 0.0
        with torch.no_grad():
            hq = torch.cat([head(f[i:i + 256].to(device)).cpu() for i in range(0, f.shape[0], 256)])
            hi = torch.cat([inc(f[i:i + 256].to(device)).cpu() for i in range(0, f.shape[0], 256)])
        # (N*T, 2, Q, H, W) -> per channel (N, T, H, W, Q)
        hq = hq.numpy().reshape(m.shape[0], m.shape[1], 2, a.n_q, *m.shape[2:4])
        hi = hi.numpy().reshape(m.shape[0], m.shape[1], 2, *m.shape[2:4]).transpose(0, 1, 3, 4, 2)
        for c, name in enumerate(["u", "v"]):
            n_sc = int(np.count_nonzero(y[..., c] != 0))
            if not n_sc:
                continue
            hqc = np.moveaxis(hq[:, :, c], 2, -1)          # (N,T,Q,H,W) -> (N,T,H,W,Q); axis 2 is Q, not T
            o_q = objective_map(m[..., c], y[..., c], pick(hqc, taus_np))[0]
            o_i = objective_map(m[..., c], y[..., c], hi[..., c])[0]
            tot["inc"][0] += o_i * n_sc; tot["inc"][1] += n_sc
            tot["qh"][0] += o_q * n_sc; tot["qh"][1] += n_sc
            print(f"  {re:>7} {name:>3} {o_i:10.4f} {o_q:10.4f} {o_q / o_i - 1:+7.2%}")
    pi, pq = tot["inc"][0] / tot["inc"][1], tot["qh"][0] / tot["qh"][1]
    print(f"\n  POOLED  incumbent {pi:.4f}   quantile head {pq:.4f}   {pq / pi - 1:+.2%}")
    out = OUTPUTS / "width_head" / f"{a.name}.pth"
    torch.save({"state_dict": head.state_dict(), "width": a.width, "norm": norm, "n_feat": ftr.shape[1],
                "planes": planes, "drop_z": bool(ck.get("drop_z")), "n_q": a.n_q, "taus": taus_np.tolist(),
                "pooled": {"incumbent": pi, "quantile": pq}}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--head", required=True, help="the incumbent head: supplies norm, planes, drop_z and the comparison")
    p.add_argument("--folds", nargs="+", required=True)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--n-q", type=int, default=9)
    p.add_argument("--tau-lo", type=float, default=0.50)
    p.add_argument("--tau-hi", type=float, default=0.97)
    p.add_argument("--name", default="qhead_ext_w48")
    main(p.parse_args())
