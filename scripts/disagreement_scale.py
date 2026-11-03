#!/usr/bin/env python
"""Does the ensemble we SHIP disagree on the same scale as the one the head was TRAINED on?

The width head's disagreement planes are z = per-element std over members, normalised
by a fixed ``zref`` baked into the head at training time. The head therefore assumes a
scale. Three different ensembles appear in this pipeline and nothing has ever checked
that they agree:

  * the FOLD ensemble (ws_cfk_fold*), which the head trains on -- 3 cross-fit members
  * the HELD-OUT ensemble (cc_allsim_ftk008_cov04_s*), which the TWIN scores with
  * the ALL-DATA ensemble (cc_allsim_all_tke008_cov04_s*/epoch020), which the ZIP SHIPS

All-data members saw every real row in training, so they may agree MORE than held-out
members do. If they do, the shipped zip feeds the head systematically smaller z than
the twin measured, the head predicts different widths than the twin scored, and the
twin's sps is not the zip's sps. That is the W178 failure shape -- a model scored on an
input it was not calibrated for -- and it is invisible in the twin by construction.

Prints mean, median and p90 of z per channel for each ensemble at matched member
counts, on the rows the twin scores.

    python scripts/disagreement_scale.py --re 13950 26700 --stride 20
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pde_competition.dataset import build_eval_arrays, list_runs   # noqa: E402
from pde_competition.model import FlowerS2S, predict_numpy         # noqa: E402


def member_std(ckpts, x, device):
    """Per-element std over members, exactly as submission_flower.py computes it."""
    preds = []
    for c in ckpts:
        model = FlowerS2S.from_checkpoint(c)
        preds.append(predict_numpy(model, x, batch_size=16, device=device))
    p = np.stack(preds, 0)
    return p.std(axis=0), p.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", nargs="+", type=int, default=[13950, 26700])
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, default=3, help="members per ensemble to compare at matched count")
    a = ap.parse_args()

    runs = [r for r in list_runs("train_real") if r.re_nominal in a.re]
    if not runs:
        raise SystemExit(f"no trajectories with Re in {a.re}")
    x, y = build_eval_arrays(runs, stride=a.stride)
    print(f"{x.shape[0]} windows, rows {sorted(a.re)}, {a.k} members per ensemble\n")

    fams = {
        "fold (head trained on)": sorted(Path().glob("outputs/runs/ws_cfk_fold*/best_score.pth")),
        "held-out (twin scores)": sorted(Path().glob("outputs/runs/cc_allsim_ftk008_cov04_s*/best_score.pth")),
        "all-data (zip ships)": sorted(Path().glob("outputs/runs/cc_allsim_all_tke008_cov04_s*/epoch020.pth")),
    }
    print(f"{'ensemble':<26} {'n':>2} {'mean z_u':>10} {'mean z_v':>10} {'p90 z_u':>10} {'p90 z_v':>10}")
    ref = None
    for nm, cks in fams.items():
        cks = cks[:a.k]
        if len(cks) < 2:
            print(f"{nm:<26} {len(cks):>2}  (need >= 2 members, skipped)")
            continue
        std, _ = member_std(cks, x, a.device)
        zu, zv = std[..., 0].ravel(), std[..., 1].ravel()
        row = (zu.mean(), zv.mean(), np.percentile(zu, 90), np.percentile(zv, 90))
        print(f"{nm:<26} {len(cks):>2} {row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:10.6f}")
        if "fold" in nm:
            ref = row
    if ref:
        print("\nRATIO to the fold ensemble the head was trained on (1.00 = the head sees what it learned):")
        for nm, cks in fams.items():
            cks = cks[:a.k]
            if len(cks) < 2 or "fold" in nm:
                continue
            std, _ = member_std(cks, x, a.device)
            zu, zv = std[..., 0].ravel(), std[..., 1].ravel()
            print(f"  {nm:<26} z_u x{zu.mean()/ref[0]:.3f}   z_v x{zv.mean()/ref[1]:.3f}")
        print("\nA ratio far from 1.0 means the head is fed out-of-distribution disagreement and its")
        print("widths are mis-calibrated by that factor -- the twin cannot see this, because the twin")
        print("and the zip use DIFFERENT ensembles.")


if __name__ == "__main__":
    main()
