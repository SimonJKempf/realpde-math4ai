"""Do member-disagreement-conditioned widths beat the constant and the pixel map?

W39: the std over ensemble members correlates with |residual| at r ~ 0.6.
This fits, on the FIT row, half-widths of the form
    const        h = h0
    map_blend    h = map_blend(y, x)                (W24's winner)
    lin_z        h = a + b * z                      z = member std per element
    lin_z x map  h = (a + b * z) * (0.5 s + 0.5)    s = map shape, mean 1
and scores them on the disjoint VAL row. The point prediction is the member
MEAN (what a shipped ensemble would predict). Widths on p are 0.

    python scripts/disagreement_widths.py --ckpts 'outputs/runs/ws_ftk003_s*/best_score.pth'
"""
from __future__ import annotations
import argparse, glob, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
from pde_competition.dataset import build_eval_arrays, list_runs
from pde_competition.model import FlowerS2S, predict_numpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_bounds import SIGMA_GLOBAL, fit_channel, fit_map, objective_map, spatial_variants  # noqa: E402

GRID = np.concatenate([np.linspace(1e-4, 0.02, 400), np.linspace(0.02, 0.20, 200)])


def rows(ckpts, re_values, stride, device):
    runs = [r for r in list_runs("train_real") if r.re_nominal in re_values]
    x, y = build_eval_arrays(runs, stride=stride)
    P = np.stack([predict_numpy(FlowerS2S.from_checkpoint(c), x, batch_size=16, device=device)[..., :2]
                  for c in ckpts]).astype(np.float32)
    return P.mean(0), P.std(0), y[..., :2].astype(np.float32)


def fit_lin_z(pred, y, z, shape=None):
    """Grid-search (a, b) for h = (a + b z) * shape on one channel."""
    zs = z / max(float(z[y != 0].mean()), 1e-8)                 # z normalised to mean 1
    sh = 1.0 if shape is None else shape
    best = (-1.0, 0.0, 0.0)
    for b in np.linspace(0.0, 0.03, 13):
        for a in np.linspace(0.0, 0.02, 41):
            h = (a + b * zs) * sh
            o, _ = objective_map(pred, y, h)
            if o > best[0]:
                best = (o, a, b)
    return best[1], best[2]


def offset_map(pred, y):
    """Per-pixel median of (target - pred) over scored elements: the residual BIAS.

    An interval centred on pred + offset covers a biased residual at a smaller
    width than one centred on pred. The point prediction is untouched (rel_l2
    is scored on pred); only the interval moves. Simon's asymmetric-bounds idea
    in its cheapest form.
    """
    r = np.where(y != 0, y - pred, np.nan)
    with np.errstate(all="ignore"):
        o = np.nanmedian(r.reshape(-1, *r.shape[2:]), axis=0)
    return np.nan_to_num(o, nan=0.0).astype(np.float32)


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted(p for g in a.ckpts for p in glob.glob(g))
    print(f"{len(ckpts)} members: {[Path(c).parent.name for c in ckpts]}")
    fm, fz, fy = rows(ckpts, a.fit_re, a.stride, device)
    vm, vz, vy = rows(ckpts, a.val_re, a.stride, device)
    print(f"fit rows {a.fit_re}: {fm.shape[0]} windows; val rows {a.val_re}: {vm.shape[0]} windows\n")
    names = ["u", "v"]
    tot = {}
    for c in range(2):
        p, y, z = fm[..., c], fy[..., c], fz[..., c]
        vp, vyy, vzz = vm[..., c], vy[..., c], vz[..., c]
        scored = y != 0
        r = np.abs(p - y)
        print(f"channel {names[c]}: corr(z, |r|) fit {np.corrcoef(z[scored], r[scored])[0,1]:.3f}")
        h0 = fit_channel(p, y, GRID)[0]
        cands = spatial_variants(p, y, GRID)
        s = cands["map_blend"] / max(float(cands["map_blend"][np.any(y != 0, axis=(0, 1))].mean()), 1e-8)
        a1, b1 = fit_lin_z(p, y, z)
        a2, b2 = fit_lin_z(p, y, z, shape=(0.5 * s + 0.5))
        zref_f = float(z[scored].mean()); zref_v = zref_f            # normalise val z by the FIT mean
        variants = {
            "const": h0,
            "map_blend": cands["map_blend"],
            "lin_z": (a1 + b1 * vzz / zref_v),
            "lin_z_x_map": (a2 + b2 * vzz / zref_v) * (0.5 * s + 0.5),
        }
        # ASYMMETRIC: the same widths around pred + per-pixel residual bias
        # (fitted on the fit row), i.e. lower = pred + o - h, upper = pred + o + h.
        off = offset_map(p, y)
        # BRACKET CONSTRAINT (Simon, 2026-09-16): the interval must always contain
        # the point prediction, lower <= pred <= upper, so |offset| <= h. Clip the
        # offset to half the constant width so the shift can never leave pred
        # outside, nor move the interval by more than half its own half-width.
        off = np.clip(off, -0.5 * h0, 0.5 * h0)
        print(f"  offset map (clipped to |o| <= 0.5 h0 = {0.5 * h0:.5f}): mean |o| {np.abs(off).mean():.5f}, max |o| {np.abs(off).max():.5f}")
        shifted = {f"{k}+off": (h, off) for k, h in variants.items() if k in ("const", "lin_z_x_map")}
        # refit the level of the shifted constant on the fit row (bias removal may want a narrower h)
        h0s = fit_channel(p + off, y, GRID)[0]
        shifted["const+off+refit"] = (h0s, off)
        base = objective_map(vp, vyy, h0)[0]
        n = int(np.count_nonzero(vyy != 0))
        print(f"  {'variant':<18}{'val obj':>9}{'val cov':>9}{'gain':>8}")
        for k, (h, o_) in shifted.items():
            o, cov = objective_map(vp + o_, vyy, h)
            tot.setdefault(k, [0.0, 0]); tot[k][0] += o * n; tot[k][1] += n
            print(f"  {k:<18}{o:9.4f}{cov:9.1%}{o / base - 1:+8.2%}")
        for k, h in variants.items():
            o, cov = objective_map(vp, vyy, h)
            tot.setdefault(k, [0.0, 0]); tot[k][0] += o * n; tot[k][1] += n
            print(f"  {k:<18}{o:9.4f}{cov:9.1%}{o / base - 1:+8.2%}")
        print(f"  lin_z: a={a1:.5f} b={b1:.5f}   lin_z_x_map: a={a2:.5f} b={b2:.5f}\n")
    print("pooled over scored elements of both channels (= the sps ratio up to the accuracy factor):")
    base = tot["const"][0] / tot["const"][1]
    for k, (num, n) in tot.items():
        print(f"  {k:<18}{num / n:9.4f}{num / n / base - 1:+8.2%}")

    if a.write_for_members:
        # SHIP: refit the chosen variant on the FIT rows (pass both held-out rows
        # for a shipping fit) and write bounds.json beside member 0 with the
        # 'disagreement' block submission_flower.py applies per element.
        A, Bc, Z, S = [], [], [], []
        for c in range(2):
            p, y, z = fm[..., c], fy[..., c], fz[..., c]
            scored = y != 0
            if a.variant == "lin_z":
                aa, bb = fit_lin_z(p, y, z); shape_c = np.ones(p.shape[2:], np.float32)
            else:
                cands = spatial_variants(p, y, GRID)
                s = cands["map_blend"] / max(float(cands["map_blend"][np.any(y != 0, axis=(0, 1))].mean()), 1e-8)
                aa, bb = fit_lin_z(p, y, z, shape=(0.5 * s + 0.5)); shape_c = s
            A.append(aa); Bc.append(bb); Z.append(float(z[scored].mean())); S.append(shape_c)
        shape = np.stack(S + [np.ones_like(S[0])], -1)                    # (H, W, C)
        alpha = 0.0 if a.variant == "lin_z" else 0.5
        member0 = Path(ckpts[0])
        # mean width for the record (over fit windows)
        zs = fz / np.maximum(np.asarray(Z, np.float32), 1e-8)
        lvl = np.asarray(A, np.float32) + np.asarray(Bc, np.float32) * zs
        hw = [float(np.mean(lvl[..., c] * (alpha * shape[..., c] + 1 - alpha))) for c in range(2)] + [0.0]
        payload = {
            "half_width": hw, "channels": ["u", "v", "p"],
            "disagreement": {"a": A + [0.0], "b": Bc + [0.0], "zref": Z + [1.0], "alpha": alpha,
                             "shape": shape.round(5).tolist() if alpha > 0 else None,
                             "variant": a.variant, "members": [str(c) for c in ckpts],
                             "formula": "h = (a_c + b_c * z_c / zref_c) * (alpha * shape + 1 - alpha), z = std over members"},
            "ckpt": member0.name, "ckpt_sha256": hashlib.sha256(member0.read_bytes()).hexdigest(),
            "fit_re": sorted(a.fit_re), "val_re": sorted(a.val_re), "stride": a.stride,
            "sigma": SIGMA_GLOBAL, "fitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None,
            "tool": "scripts/disagreement_widths.py",
        }
        out = member0.with_name(member0.stem + ".bounds.json")
        out.write_text(json.dumps(payload) + "\n")
        print(f"\nwrote {out}  variant {a.variant}  a {A} b {Bc} zref {Z}  mean widths {hw[:2]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--fit-re", type=int, nargs="+", default=[13950])
    p.add_argument("--val-re", type=int, nargs="+", default=[26700])
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--write-for-members", action="store_true",
                   help="also refit --variant on the fit rows and write bounds.json "
                        "beside member 0 (for build_submission.py --ensemble)")
    p.add_argument("--variant", default="lin_z", choices=("lin_z", "lin_z_x_map"))
    main(p.parse_args())
