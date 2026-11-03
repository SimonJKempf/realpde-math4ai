#!/usr/bin/env python
"""One command for the thing every scoring parcel repeated in three lines:
fit constant bounds FRESH for a checkpoint, build its zip (optionally as an
ensemble, optionally with a width head), and score it locally.

    python scripts/score_twin.py --ckpt <glob or path> --name <zip name> [--width-head H.pth]
                                 [--ensemble <glob or path> ...] [--stride 50]

Globs are resolved here (sorted, last match) so array manifests can carry
them unexpanded. The bounds file beside the checkpoint is removed before the
fit -- a persisted fit is only a saving if its source has not changed (W92b,
W105, 2026-09-17). Prints the scores.json path at the end.

``--gapped-holdout [N G]`` (direction GTH, bare = 100 40) makes a GTH twin of a
checkpoint trained with train.py ``--gapped-holdout``: the bounds are fitted and
the zip is scored on the LAST N frames of EVERY trajectory (gap G) -- ALL 18 Re
rows -- instead of on held-out Re rows. ``--fit-re``/``--holdout-re`` are refused
with it. The numbers are NOT comparable with condition-holdout twins.
"""
import argparse, glob, os, subprocess, sys
from pathlib import Path

from pde_competition.dataset import parse_gapped_holdout
from pde_competition.paths import REPO

DEFAULT_FIT_RE = ["13950", "26700"]


def resolve(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"no checkpoint matches {pattern}")
    return Path(hits[-1])


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--width-head", default=None)
    ap.add_argument("--width-scale", type=float, default=1.0,
                    help="passed to build_submission: multiplies the head's half-widths. Needed when the "
                         "shipped ensemble is larger than the 3-member fold ensemble the head trained on.")
    ap.add_argument("--width-head-use-z", action="store_true",
                    help="feed the head its member-DISAGREEMENT planes instead of zeroing them. Only legal "
                         "with --ensemble: the planes are the per-element std over members, so a single "
                         "model has none. Every head we ship today was trained --drop-z BECAUSE we ship one "
                         "model; an ensemble makes the planes free at inference (they are already computed) "
                         "and build_submission has carried the flag since the start. Without this, "
                         "score_twin could not score the head the ensemble actually unlocks.")
    ap.add_argument("--ensemble", nargs="*", default=None)
    ap.add_argument("--fit-on-ensemble", action="store_true",
                    help="fit the bounds on the ENSEMBLE mean rather than on --ckpt alone. Every ensemble zip built "
                         "before 2026-09-18 fitted on one member, which over-widens.")
    ap.add_argument("--fit-re", nargs="+", default=None,
                    help=f"rows the bounds are fitted on; default {DEFAULT_FIT_RE}. Refused with --gapped-holdout.")
    ap.add_argument("--fit-stride", default="10")
    ap.add_argument("--stride", default="50")
    ap.add_argument("--holdout-re", nargs="+", default=None,
                    help="rows to SCORE on; defaults to --fit-re. A twin trained with --holdout-re X must be "
                         "scored on X: W136 scored split-A/B twins on the standard rows they had trained on. "
                         "Refused with --gapped-holdout.")
    ap.add_argument("--gapped-holdout", type=int, nargs="*", default=None, metavar="N_G",
                    help="direction GTH: fit the bounds AND score on the LAST N frames of EVERY trajectory "
                         "(gap G), all 18 Re rows -- for members trained with train.py --gapped-holdout. "
                         "Bare = 100 40 (the cc_gth_* pilot); or give N G. Passed through to fit_bounds.py "
                         "and score_local.py. NOT comparable with condition-holdout twins.")
    a = ap.parse_args()
    gth = parse_gapped_holdout(a.gapped_holdout)
    if gth is not None:
        if a.fit_re is not None or a.holdout_re is not None:
            raise SystemExit(
                "--fit-re/--holdout-re with --gapped-holdout is refused: a GTH twin is fitted and "
                "scored on the tail windows of ALL rows, so there is no row to choose. Pass one design.")
        gth_flag = ["--gapped-holdout", str(gth[0]), str(gth[1])]     # explicit, never bare, downstream
        print("=" * 96 + f"\nGTH TWIN: bounds fitted and zip scored on the LAST {gth[0]} frames of EVERY "
              f"trajectory (gap {gth[1]}), ALL 18 Re rows.\nNOT comparable with any condition-holdout "
              f"twin. Name it so: {a.name!r}.\n" + "=" * 96, flush=True)
    else:
        gth_flag = []
        a.fit_re = a.fit_re or DEFAULT_FIT_RE
    holdout = a.holdout_re or a.fit_re
    py = sys.executable
    ckpt = resolve(a.ckpt)
    bounds = ckpt.with_name(ckpt.stem + ".bounds.json")
    if bounds.exists():
        bounds.unlink()
    fit = [py, str(REPO / "scripts/fit_bounds.py"), "--ckpt", str(ckpt), "--stride", a.fit_stride]
    fit += gth_flag if gth is not None else ["--fit-re", *a.fit_re]
    if a.ensemble and a.fit_on_ensemble:
        # Widths are fitted to residuals, and an ensemble's residuals are SMALLER than any
        # member's. Fitting on --ckpt alone and shipping those widths with --ensemble
        # over-widens every element and gives back sps the ensemble already earned.
        fit += ["--fit-ensemble", *[str(resolve(m)) for m in a.ensemble]]
    run(fit)
    build = [py, str(REPO / "scripts/build_submission.py"), "--ckpt", str(ckpt), "--name", a.name]
    if a.ensemble:
        build += ["--ensemble", *[str(resolve(m)) for m in a.ensemble]]
    if a.width_head:
        build += ["--width-head", a.width_head]
        if a.width_scale != 1.0:
            build += ["--width-scale", str(a.width_scale)]
        if a.width_head_use_z:
            if not a.ensemble:
                raise SystemExit("--width-head-use-z needs --ensemble: disagreement is a std over members")
            build += ["--width-head-use-z"]
    run(build)
    score = [py, str(REPO / "scripts/score_local.py"), str(REPO / "submission/dist" / f"{a.name}.zip"), "--stride", a.stride]
    score += gth_flag if gth is not None else ["--holdout-re", *holdout]
    run(score)
    print(f"scores: {REPO / 'outputs/scores' / a.name / 'scores.json'}")


if __name__ == "__main__":
    os.chdir(REPO)
    main()
