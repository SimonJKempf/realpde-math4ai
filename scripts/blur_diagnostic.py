#!/usr/bin/env python
"""Is the prediction BLURRED? Measure fluctuation energy, predicted vs true, against lead time.

    python scripts/blur_diagnostic.py <zip> [--stride 50]

WHY THIS EXISTS. tke_score sits at 78 while rel_l2 sits at 95, and the sps review put 15.9 sps points
behind tke alone -- more than rel_l2 and mvpe combined. The standard explanation is that an L2 objective
is minimised by the conditional MEAN, which at a 20-frame lead averages over an ambiguous shedding phase
and produces a field with too little fluctuation energy. If that is what is happening, an adversarial or
likelihood-based head (direction A7) has a real target; if the energy is right and only its PHASE is wrong,
A7 is still the answer but for a different reason; and if neither, A7 is a large build against nothing.

This distinguishes those three without training anything:
  * energy ratio  E_pred/E_true  per lead time. Blur predicts a ratio that FALLS with lead time.
  * correlation   corr(pred', true') per lead time on the fluctuation fields. Phase error predicts
    correlation falling while the ratio stays near 1.
Fluctuation = field minus its time mean over the 20-frame output window, per pixel per channel, so the
mean flow is removed and what is left is the unsteady part tke_score is computed from.
"""
import argparse, pathlib, sys, tempfile, zipfile
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from pde_competition.dataset import list_runs, split_by_condition, build_eval_arrays  # noqa: E402
# build_eval_arrays lives in dataset, not a separate module


def fluct(a):
    """a: (N, T, H, W, C) -> fluctuation about the per-pixel time mean of the window."""
    return a - a.mean(axis=1, keepdims=True)


def report(p, y, label: str) -> None:
    """Print BLUR's table for one prediction array. Unchanged numbers; ``label`` says whose."""
    c = 2                                    # real PIV: u, v only
    fp, ft = fluct(p[..., :c]), fluct(y[..., :c])
    print(f"\n== {label}")
    print("\n  IDENTITY: for the conditional mean (the L2 optimum) corr^2 = r EXACTLY, because")
    print("  y = yhat + eps with eps orthogonal to yhat gives corr = sqrt(Var(yhat)/Var(y)) = sqrt(r).")
    print("  So r/corr^2 > 1 means the prediction carries MORE energy than L2 wants, and beta = corr/sqrt(r)")
    print("  is the rescale that would minimise rel_l2 alone.")
    print(f"\n{'lead':>5} {'E_pred/E_true':>14} {'corr(pred,true)':>16} {'corr^2':>9} {'r/corr^2':>9} {'beta':>7}   interpretation")
    corrs = []
    for t in range(fp.shape[1]):
        ep = float((fp[:, t] ** 2).mean())
        et = float((ft[:, t] ** 2).mean())
        num = float((fp[:, t] * ft[:, t]).mean())
        den = float(np.sqrt((fp[:, t] ** 2).mean() * (ft[:, t] ** 2).mean()))
        r = ep / et if et else float("nan")
        cr = num / den if den else float("nan")
        corrs.append(cr)
        if t in (0, 4, 9, 14, 19):
            tag = "energy kept, phase lost" if r > 0.85 and cr < 0.7 else (
                  "BLURRED (energy lost)" if r < 0.85 else "both intact")
            beta = cr / (r ** 0.5) if r > 0 else float("nan")
            print(f"{t:5d} {r:14.3f} {cr:16.3f} {cr*cr:9.3f} {r/(cr*cr) if cr else float('nan'):9.2f} "
                  f"{beta:7.3f}   {tag}")
    ep = float((fp ** 2).mean()); et = float((ft ** 2).mean())
    print(f"\n  pooled energy ratio {ep / et:.3f}   (1.0 = the prediction carries the true unsteady energy)")
    # THE A7 GATE NUMBER. mean_t corr_t is exactly what --corr-weight optimises (model.fluct_corr
    # returns 1 minus it), and 1 - corr^2 is the fluctuation error left after the optimal rescale --
    # i.e. the part of the blur no post-hoc alpha can remove. corr(19) is the single number A7 is
    # pre-registered on, because that is where the deficit is largest.
    print(f"  GATE  mean_t corr {np.mean(corrs):.4f}   corr(19) {corrs[-1]:.4f}   "
          f"1 - corr(19)^2 {1 - corrs[-1] ** 2:.4f}")
    print("  READ IT: a ratio falling with lead time is blur, and an adversarial/likelihood head (A7) has a")
    print("  target proportional to the shortfall. A flat ratio near 1 with falling correlation is PHASE")
    print("  error, which A7 also addresses but by committing to a mode rather than by adding energy.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", type=pathlib.Path, nargs="?",
                    help="a built submission zip (the ensemble the board would run). Omit it and pass "
                         "--ckpt instead to diagnose raw training checkpoints.")
    ap.add_argument("--ckpt", type=pathlib.Path, nargs="+", default=None, metavar="PTH",
                    help="diagnose these checkpoints instead of a zip, EACH ONE SEPARATELY. This is the "
                         "A7 gate: a fine-tune's effect on corr is a property of the model, and building "
                         "a zip per arm would add bounds fitting and ensembling between the change and "
                         "the number. Same windows, same table, one block per checkpoint.")
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--holdout-re", type=int, nargs="+", default=[13950, 26700])
    ap.add_argument("--split", default="train_real",
                    help="train_real (the scored p00 data) or train_sim. THE SIM CEILING: sim has no PIV "
                         "measurement noise and a deterministic trajectory, so if corr at lead 19 is ~0.5 "
                         "there too, the phase ambiguity is ALEATORIC and no training objective can raise "
                         "it; if sim corr stays high, the real-data phase skill is recoverable and a "
                         "coherence-targeting loss has headroom.")
    a = ap.parse_args()
    if (a.zip_path is None) == (a.ckpt is None):
        ap.error("give exactly one of: a zip path, or --ckpt PTH [PTH ...]")

    _train, val = split_by_condition(list_runs(a.split), holdout_re=tuple(a.holdout_re))
    x, y = build_eval_arrays(val, stride=a.stride)[:2]
    print(f"windows {x.shape[0]}  holdout {a.holdout_re}  split {a.split}")

    if a.ckpt is not None:
        import torch                                                     # noqa: E402
        from pde_competition.model import FlowerS2S, predict_numpy       # noqa: E402
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for ck in a.ckpt:
            m = FlowerS2S.from_checkpoint(ck)
            p = predict_numpy(m, x, batch_size=16, device=device).astype(np.float32)
            report(p, y, str(ck))
        return

    work = pathlib.Path(tempfile.mkdtemp(prefix="blur_"))
    with zipfile.ZipFile(a.zip_path) as z:
        z.extractall(work / "sub")
    sys.path.insert(0, str(work / "sub"))
    import submission                                                    # noqa: E402
    p = np.asarray(submission.predict(x, {})["prediction"], dtype=np.float32)
    report(p, y, str(a.zip_path))


if __name__ == "__main__":
    main()
