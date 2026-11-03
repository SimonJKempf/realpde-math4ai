"""Does averaging N checkpoints pay on ACCURACY? W21b priced the time; this
measures the other half, with no change to the submission module.

For each checkpoint set: predict the held-out windows (same 136 windows at
stride 50 that evaluate.py scores), average the predictions, fit constant
half-widths on the averaged residuals (both held-out rows, as submission 8),
and run the kit scorer. Each member is scored alone the same way. est.final
holds time at composite.REFERENCE (local time is a per-host artefact).

    python scripts/eval_ensemble.py --name ens_ftk003 --ckpts outputs/runs/ws_ftk003_s*/best_score.pth
"""
from __future__ import annotations
import argparse, glob, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, torch
from pde_competition.dataset import build_eval_arrays, list_runs, split_by_condition
from pde_competition.model import FlowerS2S, predict_numpy
from pde_competition.paths import KIT, OUTPUTS
from pde_competition import composite
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_bounds import fit_channel  # noqa: E402

GRID = np.concatenate([np.linspace(1e-4, 0.02, 400), np.linspace(0.02, 0.20, 200)])


def score(name, pred, y, x_shape):
    widths = np.zeros(3, np.float32)
    for c in range(2):
        widths[c] = fit_channel(pred[..., c], y[..., c], GRID)[0]
    half = np.broadcast_to(widths, pred.shape).astype(np.float32)
    work = Path(tempfile.mkdtemp(prefix="ens_"))
    (work / "in" / "ref").mkdir(parents=True)
    np.savez(work / "in" / "predictions.npz", prediction=pred.astype(np.float32),
             lower=(pred - half).astype(np.float32), upper=(pred + half).astype(np.float32),
             mean_t_neural_s=np.float32(1e-3))
    np.savez(work / "in" / "ref" / "targets.npz", target=y)
    out = OUTPUTS / "scores" / name; out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(KIT / "scoring.py"), str(work / "in"), str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr)
    s = json.loads((out / "scores.json").read_text())
    est = composite.estimate(s["rel_l2_score"], s["tke_score"], s["mvpe_score"],
                             composite.REFERENCE["time"], s["sps_score"])
    print(f"  {name:<34} rel_l2 {s['rel_l2_score']:7.3f}  tke {s['tke_score']:7.3f}  mvpe {s['mvpe_score']:7.3f}"
          f"  sps {s['sps_score']:7.3f}  est.final@ref {est:7.3f}   widths {widths[:2].round(6).tolist()}", flush=True)
    return est


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted(p for g in a.ckpts for p in glob.glob(g))
    _train, val = split_by_condition(list_runs())
    x, y = build_eval_arrays(val, stride=a.stride)
    print(f"{a.name}: {len(ckpts)} members, {x.shape[0]} held-out windows, device {device}")
    preds = []
    for ck in ckpts:
        m = FlowerS2S.from_checkpoint(ck)
        preds.append(predict_numpy(m, x, batch_size=16, device=device).astype(np.float32))
        score(f"{a.name}_member{len(preds)}", preds[-1], y, x.shape)
    P = np.stack(preds)
    mean = P.mean(0)
    dis = P[..., :2].std(0)
    resid = np.abs(mean[..., :2] - y[..., :2])
    scored = y[..., :2] != 0
    corr = np.corrcoef(dis[scored], resid[scored])[0, 1]
    print(f"  member disagreement (std over members): mean {dis[scored].mean():.5f}; "
          f"corr(disagreement, |residual|) = {corr:.3f}  <- the width-head feature")
    score(f"{a.name}_mean{len(ckpts)}", mean, y, x.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--stride", type=int, default=50)
    main(p.parse_args())
