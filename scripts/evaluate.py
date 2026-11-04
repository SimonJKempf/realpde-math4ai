#!/usr/bin/env python
"""Checkpoint -> the five local subscores, in one call.

    python scripts/evaluate.py outputs/runs/<run>/              # best_score.pth if present, else best.pth
    python scripts/evaluate.py outputs/runs/<run>/best.pth --name flower_v7
    python scripts/evaluate.py <run> --fit-re 13950 26700       # widths fitted on both held-out rows

Chains the four existing scripts, each in its own subprocess, and stops at the
first failure:

    fit_bounds.py         -> <ckpt stem>.bounds.json beside the checkpoint
    build_submission.py   -> submission/dist/<name>.zip  (ckpt + bounds + model)
    verify_submission.py  -> the zip imports and predicts under torch 2.2.2
    score_local.py        -> outputs/scores/<name>/scores.json  (tracked in git)

then prints one line: the five subscores and our composite estimate. Nothing
here is new logic -- the point is that "how good is this checkpoint" is one
command instead of four, so it gets asked after every run rather than before
every submission.

Costs a few minutes on the M2 (two inference passes over the held-out windows
plus the verifier). Local numbers run ~2-4 points optimistic on the accuracy
subscores and local time_score is meaningless (the grader's hardware differs).

Choice of fit rows is passed straight through to fit_bounds.py, whose default
fits on Re 13950 and validates on 26700. Submission 5 (the best composite so
far) used widths fitted on both rows; submission 6 lost 1.44 sps on a
single-row fit. That is a methodology choice this script does not make for you.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pde_competition import composite
from pde_competition.paths import OUTPUTS, REPO

SCRIPTS = REPO / "scripts"


def run(step: str, *args: str) -> None:
    cmd = [sys.executable, str(SCRIPTS / step), *args]
    print(f"\n=== {step} {' '.join(args)}", flush=True)
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        raise SystemExit(f"{step} failed (exit {r.returncode}); stopping here")


def resolve_ckpt(target: Path) -> Path:
    if target.is_file():
        return target
    if target.is_dir():
        for name in ("best_score.pth", "best.pth"):
            if (target / name).exists():
                return target / name
        raise SystemExit(f"{target} holds neither best_score.pth nor best.pth")
    raise SystemExit(f"no such checkpoint or run directory: {target}")


def main(a: argparse.Namespace) -> None:
    ckpt = resolve_ckpt(Path(a.target)).resolve()
    run_dir = ckpt.parent
    name = a.name or f"{run_dir.name}-{ckpt.stem}"
    zip_path = REPO / "submission" / "dist" / f"{name}.zip"
    bounds = ckpt.with_name(ckpt.stem + ".bounds.json")
    scores_path = OUTPUTS / "scores" / name / "scores.json"

    print(f"checkpoint : {ckpt.relative_to(REPO)}\nname       : {name}")

    fit_args = []
    if a.fit_re:
        fit_args += ["--fit-re", *map(str, a.fit_re)]
    if a.val_re:
        fit_args += ["--val-re", *map(str, a.val_re)]
    if a.spatial:
        fit_args += ["--spatial", "--spatial-variant", a.spatial_variant]
    if a.fit_stride:
        fit_args += ["--stride", str(a.fit_stride)]
    if a.skip_fit and bounds.exists():
        print(f"\n=== fit_bounds.py skipped, reusing {bounds.name}")
    else:
        run("fit_bounds.py", "--ckpt", str(ckpt), *fit_args)

    run("build_submission.py", "--ckpt", str(ckpt), "--name", name)
    if not a.skip_verify:
        run("verify_submission.py", str(zip_path))
    run("score_local.py", str(zip_path), "--stride", str(a.stride))

    s = json.loads(scores_path.read_text())
    b = json.loads(bounds.read_text())
    # Local time_score is wall-clock on whatever CPU/GPU ran score_local and
    # means nothing (CLAUDE.md); on 2026-09-15 a candidate scored on wst004
    # instead of wst002 picked up +5.8 time points = +0.58 est.final for an
    # identical model. Hold time at the leaderboard reference so est.final
    # ranks only what the local run can measure; print the local time
    # separately, labelled.
    est = composite.estimate(s["rel_l2_score"], s["tke_score"], s["mvpe_score"],
                             composite.REFERENCE["time"], s["sps_score"])
    print("\n=== summary")
    print(f"{'name':12s} {'rel_l2':>8s} {'tke':>8s} {'mvpe':>8s} {'time*':>8s} "
          f"{'sps':>8s} {'est.final':>9s}   half-width (fit_re)      "
          f"(*local, NOT in est.final; time held at {composite.REFERENCE['time']:.2f})")
    print(f"{name[:12]:12s} {s['rel_l2_score']:8.3f} {s['tke_score']:8.3f} "
          f"{s['mvpe_score']:8.3f} {s['time_score']:8.3f} {s['sps_score']:8.3f} "
          f"{est:9.3f}   {b['half_width'][:2]} ({b.get('fit_re')})")
    print(f"\nscores : {scores_path.relative_to(REPO)}  (tracked -- commit it)\n"
          f"zip    : {zip_path.relative_to(REPO)}\n"
          f"est.final uses composite.WEIGHTS {composite.WEIGHTS} -- a ranking "
          f"device, not a prediction; local time_score is CPU and meaningless.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", help="run directory or .pth checkpoint")
    p.add_argument("--name", default=None,
                   help="zip / scores name (default: <run_id>-<ckpt stem>)")
    p.add_argument("--fit-re", type=int, nargs="+", default=None,
                   help="passed to fit_bounds.py (its default: 13950)")
    p.add_argument("--fit-stride", type=int, default=None,
                   help="window stride for fit_bounds.py (its default 50; 10 gives "
                        "a per-pixel map 5x the samples)")
    p.add_argument("--spatial", action="store_true",
                   help="passed to fit_bounds.py: per-element half-width map")
    p.add_argument("--spatial-variant", default="auto",
                   help="passed to fit_bounds.py --spatial-variant")
    p.add_argument("--val-re", type=int, nargs="+", default=None,
                   help="passed to fit_bounds.py (its default: 26700)")
    p.add_argument("--stride", type=int, default=50,
                   help="score_local.py window stride (default 50 = 136 windows)")
    p.add_argument("--skip-fit", action="store_true",
                   help="reuse an existing <ckpt>.bounds.json instead of refitting")
    p.add_argument("--skip-verify", action="store_true",
                   help="skip the torch-2.2.2 clean-room check (do NOT skip "
                        "before a real submission)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
