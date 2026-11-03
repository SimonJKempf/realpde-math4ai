#!/usr/bin/env python
"""Score a submission zip locally against held-out real trajectories.

Builds a validation set from (Re, aoa) cells the model never trained on,
runs the zip's predict() exactly as the grader would, and hands the result
to kit/scoring.py for the five official subscores.

    python scripts/score_local.py submission/dist/baseline_fno.zip

``--gapped-holdout [N G]`` (direction GTH) scores instead on the LAST N frames of
EVERY trajectory (gap G, default 100 40) -- the validation side of
``split_temporal_gapped`` -- for a model trained with train.py
``--gapped-holdout``. All 18 Re rows then appear in the per-row table. Those
numbers are NOT comparable with condition-holdout twins, and the scores files
carry ``"gapped_holdout": [N, G]`` so nothing downstream can mistake them.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from pde_competition.dataset import (build_eval_arrays, list_runs,
                                     parse_gapped_holdout, split_by_condition,
                                     split_temporal_gapped)
from pde_competition.paths import KIT, OUTPUTS

DEFAULT_HOLDOUT_RE = (13950, 26700)


def gth_banner(gapped_holdout) -> str:
    """The loud line printed wherever a GTH number appears. One string, so the
    header above the table and the one above the run cannot drift apart."""
    n_hold, gap = gapped_holdout
    bar = "=" * 96
    return (f"{bar}\n"
            f"GTH TAIL SCORING: every number here is measured on the LAST {n_hold} frames of EVERY\n"
            f"trajectory (gap {gap} frames before them), i.e. on ALL Re rows, IN-SAMPLE in condition and\n"
            f"out-of-sample in time only. NOT comparable with any condition-holdout twin (different\n"
            f"windows, different persistence floor). Files carry \"gapped_holdout\": [{n_hold}, {gap}].\n"
            f"{bar}")


def resolve_holdout(holdout_re, gapped_values):
    """The two row-choosing flags, resolved into exactly one split.

    Returns ``(holdout_re, gapped_holdout)`` where exactly one is not None:
    ``((13950, 26700), None)`` by default, ``(tuple(holdout_re), None)`` for a
    condition holdout, ``(None, (N, G))`` for GTH. Passing ``--holdout-re`` with
    ``--gapped-holdout`` is REFUSED rather than reinterpreted as a row filter:
    the GTH table already shows every row and its 'all' row pools all 18, so a
    filter would only create a third kind of number with its own pooled row.
    """
    gth = parse_gapped_holdout(gapped_values)
    if gth is not None:
        if holdout_re is not None:
            raise SystemExit(
                f"--holdout-re {list(holdout_re)} with --gapped-holdout is refused: the gapped "
                f"split holds out no condition, every Re row is scored on its own tail and the "
                f"per-row table already shows each one. Pass one of the two flags.")
        return None, gth
    return tuple(holdout_re) if holdout_re else DEFAULT_HOLDOUT_RE, None


def row_groups(keys: list) -> list[tuple[str, np.ndarray]]:
    """``[(row_label, window indices), ...]``: one entry per Re row in ascending
    order, then ``("all", every index)``. Labels are strings, as the json keys are."""
    rows = sorted({k[0] for k in keys})
    out = [(str(re), np.array([i for i, k in enumerate(keys) if k[0] == re], dtype=int))
           for re in rows]
    out.append(("all", np.arange(len(keys), dtype=int)))
    return out


def gth_eval_windows(runs, gapped_holdout, stride: int, limit: int | None = None):
    """``(x, y, keys)`` from the GTH TAIL of every run -- the validation side of
    ``split_temporal_gapped`` -- exactly what a GTH twin is scored on."""
    n_hold, gap = gapped_holdout
    _train, val = split_temporal_gapped(runs, holdout_frames=n_hold, gap_frames=gap)
    return build_eval_arrays(val, stride=stride, limit_per_traj=limit, return_keys=True)


def per_row_scores(arrays: dict, y: np.ndarray, keys: list, out_dir: Path,
                   gapped_holdout: tuple[int, int] | None = None) -> None:
    """The five subscores (time = REFERENCE) and est.final per held-out Re row.

    The pooled local number is ~63% the interpolation row (85 of 136 windows
    are 13950), while the board scores an unseen mid row and an extrapolation
    row -- so a change can look flat pooled and still be right on 26700, or
    the reverse (fresh-eyes proposition 2, 2026-09-17). Written to
    scores_by_row.json beside scores.json; never fed to the grader.

    Under ``gapped_holdout`` the rows are ALL 18 (each scored on its own tail)
    and the file gains a top-level ``"gapped_holdout": [N, G]`` marker.
    """
    sys.path.insert(0, str(KIT))
    import scoring as ks                                   # noqa: E402
    from pde_competition.composite import estimate         # noqa: E402
    REF_TIME = 91.370559
    pred = arrays["prediction"]; lower = arrays.get("lower"); upper = arrays.get("upper")
    c = ks.measured_channels(y)
    by_row = {}
    for re, idx in row_groups(keys):
        p, t = pred[idx], y[idx]
        rel = float(np.mean(ks.rel_l2_per_sample(p, t, c)))
        tke = float(np.mean(ks.tke_rel_l2_per_sample(p, t, c)))
        mvpe = ks.mvpe_rel_l2(p, t)
        sps, cov = ks.aggregate_sps(p, t, c, lower=None if lower is None else lower[idx],
                                    upper=None if upper is None else upper[idx])
        sc = dict(windows=int(len(idx)), rel_l2_score=ks.score_error(rel), tke_score=ks.score_error(tke),
                  mvpe_score=ks.score_error(mvpe), sps_score=ks.score_sps(sps), coverage=float(cov))
        sc["est_final"] = estimate(sc["rel_l2_score"], sc["tke_score"], sc["mvpe_score"], REF_TIME, sc["sps_score"])
        by_row[re] = sc
    if gapped_holdout is not None:
        print("\n" + gth_banner(gapped_holdout))
    print("\nper row (time = reference 91.37):")
    print(f"  {'row':>6} {'n':>4} {'rel_l2':>8} {'tke':>8} {'mvpe':>8} {'sps':>8} {'cover':>7} {'est.final':>9}")
    for re, sc in by_row.items():
        print(f"  {re:>6} {sc['windows']:>4} {sc['rel_l2_score']:8.3f} {sc['tke_score']:8.3f} {sc['mvpe_score']:8.3f} "
              f"{sc['sps_score']:8.3f} {sc['coverage']:7.4f} {sc['est_final']:9.3f}")
    if gapped_holdout is not None:
        by_row["gapped_holdout"] = [int(gapped_holdout[0]), int(gapped_holdout[1])]
    (out_dir / "scores_by_row.json").write_text(json.dumps(by_row, indent=2))


def recentre_bounds(lower: np.ndarray, upper: np.ndarray, scale: float):
    """Scale an interval's width about its own centre: (lower, upper) -> centre -/+ scale * half.
    scale = 1 returns the inputs unchanged (bit-identical). CEN: prices a global re-levelling of
    bounds that come from a different field than the scored prediction."""
    if scale == 1.0:
        return lower, upper
    centre = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * np.float32(scale)
    return (centre - half).astype(np.float32), (centre + half).astype(np.float32)


def main(zip_path: Path, stride: int, limit: int | None, holdout_re: tuple[int, ...] | None,
         bounds_from: Path | None = None, bounds_scale: float = 1.0,
         gapped_holdout: tuple[int, int] | None = None):
    runs = list_runs()
    if gapped_holdout is not None:
        # GTH: the validation TAIL of every trajectory, all rows. Mirrors train.py's
        # --gapped-holdout split exactly (same function, same two numbers), so the
        # windows scored here are the ones the model never saw.
        print(gth_banner(gapped_holdout))
        x, y, keys = gth_eval_windows(runs, gapped_holdout, stride, limit)
        rows = sorted({k[0] for k in keys})
        print(f"GTH tail windows from {len(runs)} trajectories over {len(rows)} Re rows: {rows}")
    else:
        _train, val = split_by_condition(runs, holdout_re=holdout_re)
        print(f"held-out cells: {sorted(r.key for r in val)}")
        x, y, keys = build_eval_arrays(val, stride=stride, limit_per_traj=limit, return_keys=True)
    print(f"validation windows: x{x.shape}  y{y.shape}  ({x.nbytes/1e6:.0f} MB each)")

    work = Path(tempfile.mkdtemp(prefix="score_local_"))
    try:
        sub_dir = work / "sub"
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(sub_dir)
        sys.path.insert(0, str(sub_dir))
        import submission                                  # noqa: E402

        t0 = time.time()
        out = submission.predict(x, {})
        elapsed = time.time() - t0
        per_window = elapsed / x.shape[0]
        print(f"predict: {elapsed:.1f}s total, {per_window:.4f}s/window")

        arrays = {}
        if isinstance(out, dict):
            arrays["prediction"] = np.asarray(out["prediction"], dtype=np.float32)
            for k in ("lower", "upper"):
                if k in out:
                    arrays[k] = np.asarray(out[k], dtype=np.float32)
        else:
            arrays["prediction"] = np.asarray(out, dtype=np.float32)
        arrays["mean_t_neural_s"] = np.float32(per_window)

        tag = ""
        if bounds_from is not None:
            # CEN (2026-09-22): the sps scorer never compares the interval to the prediction
            # (kit/scoring.py: inside = (t >= lower) & (t <= upper)), so score THIS zip's prediction
            # with ANOTHER zip's lower/upper. The second zip runs in its own interpreter.
            np.save(work / "x.npy", x)
            r2 = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "predict_zip.py"),
                                 str(bounds_from), str(work / "x.npy"), str(work / "b.npz")],
                                capture_output=True, text=True)
            if r2.returncode != 0:
                print(r2.stderr.strip(), file=sys.stderr)
                raise SystemExit("--bounds-from predict failed")
            b = np.load(work / "b.npz")
            if "lower" not in b or "upper" not in b:
                raise SystemExit("--bounds-from zip returned no bounds")
            lo, up = recentre_bounds(b["lower"], b["upper"], bounds_scale)
            arrays["lower"], arrays["upper"] = lo, up
            tag = f"__bounds_from_{bounds_from.stem}" + (f"_k{bounds_scale:g}" if bounds_scale != 1.0 else "")
            print(f"bounds taken from {bounds_from.name} (scale {bounds_scale:g}); prediction from {zip_path.name}")

        in_dir = work / "in"
        (in_dir / "ref").mkdir(parents=True)
        np.savez(in_dir / "predictions.npz", **arrays)
        np.savez(in_dir / "ref" / "targets.npz", target=y)

        out_dir = OUTPUTS / "scores" / (zip_path.stem + tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [sys.executable, str(KIT / "scoring.py"), str(in_dir), str(out_dir)],
            capture_output=True, text=True)
        print(r.stdout.strip())
        if r.returncode != 0:
            print(r.stderr.strip(), file=sys.stderr)
            raise SystemExit("scoring failed")

        if gapped_holdout is not None:
            # Mark the kit's own file too: scores.json is what evaluate.py and the notes
            # quote, and a GTH number quoted beside a condition-holdout one is the misread
            # this flag exists to prevent.
            sp = out_dir / "scores.json"
            d = json.loads(sp.read_text())
            d["gapped_holdout"] = [int(gapped_holdout[0]), int(gapped_holdout[1])]
            sp.write_text(json.dumps(d, indent=2))
        scores = (out_dir / "scores.json").read_text()
        print(f"\n--- {out_dir/'scores.json'} ---\n{scores}")
        per_row_scores(arrays, y, keys, out_dir, gapped_holdout=gapped_holdout)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", type=Path)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None,
                    help="max windows per trajectory (for a quick check)")
    ap.add_argument("--bounds-from", type=Path, default=None,
                    help="CEN: take lower/upper from THIS zip (run separately) and score them with the main zip's prediction")
    ap.add_argument("--bounds-scale", type=float, default=1.0,
                    help="CEN: scale the borrowed interval's width about its centre (1.0 = as returned)")
    ap.add_argument("--holdout-re", type=int, nargs="+", default=None,
                    help="the Re rows to score on -- MUST be the rows the model held out "
                         "(a model trained on a row scores in-sample there; W136, 2026-09-18). "
                         f"Default {list(DEFAULT_HOLDOUT_RE)}. Refused with --gapped-holdout.")
    ap.add_argument("--gapped-holdout", type=int, nargs="*", default=None, metavar="N_G",
                    help="direction GTH: score on the LAST N frames of EVERY trajectory (gap G "
                         "frames before them), the validation side of split_temporal_gapped -- for "
                         "a model trained with train.py --gapped-holdout. Bare = 100 40 (the "
                         "cc_gth_* pilot); or give N G. All 18 Re rows appear in the per-row table. "
                         "NOT comparable with condition-holdout numbers; scores.json and "
                         "scores_by_row.json gain a 'gapped_holdout': [N, G] marker.")
    return ap


if __name__ == "__main__":
    a = build_parser().parse_args()
    holdout_re, gth = resolve_holdout(a.holdout_re, a.gapped_holdout)
    main(a.zip, a.stride, a.limit, holdout_re, a.bounds_from, a.bounds_scale,
         gapped_holdout=gth)
