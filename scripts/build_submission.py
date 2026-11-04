#!/usr/bin/env python
"""Assemble a Track 1 submission zip.

Usage:
    python scripts/build_submission.py --ckpt outputs/runs/<run>/best_score.pth --name flower_v7
    python scripts/build_submission.py --model baseline           # organizers' FNO control

For --model flower (the default) the archive ROOT contains, side by side:
    submission.py, model.pth, bounds.json, flower_standalone.py, geometry.py, _vendor/
--ensemble adds model_1.pth.. and ensemble.json; --width-head adds width_head.pth; --centre
(CEN) adds centre_0.pth.. , an ensemble.json "centre" list and bounds.json "centre_bounds":
true, which centres the shipped INTERVAL on that second ensemble while the prediction stays
the first one's mean.

bounds.json is REQUIRED and is looked for beside the checkpoint as
<ckpt stem>.bounds.json -- run scripts/fit_bounds.py --ckpt <ckpt> first.
The widths ship as data, not as a literal in submission.py, so which widths a
zip carries is a recorded artifact of the checkpoint they were fitted to.

For --model baseline the root holds the kit's plumbing instead:
    submission.py, <checkpoint>.pth, load_baseline.py, rpde_baselines/, _vendor/
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

from pde_competition.paths import REPO, KIT, CHECKPOINTS


def _stage_weights(src: Path, dst: Path) -> None:
    """Copy a checkpoint into the staging dir WEIGHTS-ONLY. last.pth is the resume
    artefact and carries AdamW's moments, scheduler and RNG state (3x the file:
    7.4 MB vs 2.5 MB; W100, 2026-09-17) -- nothing in submission.py reads any
    of it. Keep config, state_dict and the small provenance fields; drop the
    rest. best.pth / best_score.pth / epochNNN.pth are already weights-only and
    are copied byte-for-byte (their sha then matches bounds.json's record)."""
    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    heavy = {"optimizer", "scheduler", "rng_torch", "rng_numpy", "best", "best_tke", "score_best", "wandb_id"}
    if not (heavy & set(ck)):
        shutil.copy2(src, dst)
        return
    slim = {k: v for k, v in ck.items() if k not in heavy}
    slim["stripped_from"] = str(src)
    torch.save(slim, dst)
    print(f"staged {src.name} weights-only ({src.stat().st_size / 1e6:.1f} MB -> {dst.stat().st_size / 1e6:.1f} MB)")

# The staging directory is created PER BUILD (tempfile.mkdtemp under
# submission/) and removed afterwards. It used to be one fixed path on shared
# NFS, and on 2026-09-16 two builds launched from two workstations in the same
# second collided in it -- one saw "File exists", the other lost its files
# mid-copy. `STAGING` is rebound at the start of build(); the payload helpers
# read the module global so they need no signature change.
STAGING_ROOT = REPO / "submission"
STAGING = STAGING_ROOT / "staging"            # rebound per build; never used as is
DIST = REPO / "submission" / "dist"
SRC = REPO / "src" / "pde_competition"
CAP_MB = 256


IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def _payload_baseline(ckpt: Path) -> None:
    """The organizers' FNO/CNO/Transolver plumbing."""
    shutil.copy2(ckpt, STAGING / ckpt.name)   # submission.py names it directly
    shutil.copy2(KIT / "load_baseline.py", STAGING / "load_baseline.py")
    for d in ("rpde_baselines", "_vendor"):
        shutil.copytree(KIT / d, STAGING / d, ignore=IGNORE)


def bounds_path_for(ckpt: Path) -> Path:
    """Where fit_bounds.py leaves the widths for this checkpoint."""
    return ckpt.with_name(ckpt.stem + ".bounds.json")


# Flags that change what a checkpoint's OUTPUT MEANS. submission_flower.py builds every
# member with the SHIPPED cfg (from --ckpt) and loads only its state_dict, so a member that
# disagrees on any of these is silently run under the wrong semantics. W202 arm C did exactly
# that -- three predict_delta members averaged with three ordinary ones, each individually
# healthy at val_rel_l2 0.0811, and the ensemble scored rel_l2 87.88 against 96.15 and sps
# 16.87 against 50.90, because the deltas were read as absolute fields. Nothing warned.
_SEMANTIC = ("predict_delta", "mask_channel", "lagrangian", "lagrangian_c", "rans",
             "temporal_lift", "c_in", "c_out", "lifting_dim", "n_levels", "t_in", "t_out",
             "window_norm", "window_scale_mode", "boundary_condition_types", "rollout_frames",
             "rollout_feed")


def _assert_members_agree(ckpt: Path, members: list[Path]) -> None:
    """Refuse an ensemble whose members do not share the shipped config's semantics."""
    import torch
    def cfg_of(p):
        c = torch.load(p, map_location="cpu", weights_only=False).get("config", {})
        return c if isinstance(c, dict) else vars(c)
    ref = cfg_of(ckpt)
    bad = []
    for m in members:
        mc = cfg_of(m)
        for k in _SEMANTIC:
            if k in ref and k in mc and mc[k] != ref[k]:
                bad.append(f"{Path(m).parent.name}: {k}={mc[k]!r} but --ckpt has {ref[k]!r}")
    if bad:
        raise SystemExit(
            "REFUSING to build: ensemble members disagree with --ckpt on flags that change what a\n"
            "checkpoint's output MEANS. Every member is run under the shipped config, so these would\n"
            "be silently misinterpreted (W202 arm C: rel_l2 96.15 -> 87.88, sps 50.90 -> 16.87).\n  "
            + "\n  ".join(bad))


def _payload_members(members: list[Path]) -> None:
    """Ensemble: member 0 is model.pth (so everything single-model still works),
    the others model_<i>.pth, and ensemble.json names them in order."""
    import hashlib
    names = ["model.pth"]
    for i, m in enumerate(members[1:], start=1):
        _stage_weights(m, STAGING / f"model_{i}.pth")
        names.append(f"model_{i}.pth")
    (STAGING / "ensemble.json").write_text(json.dumps({
        "members": names, "point": "mean",
        "sources": [str(m) for m in members],
        "sha256": [hashlib.sha256(m.read_bytes()).hexdigest() for m in members]}, indent=2) + "\n")


def _payload_centre(centre: list[Path]) -> None:
    """CEN: the SECOND ensemble, the one the shipped interval is centred on.

    prediction  = mean of the --ensemble members       (unchanged, scored by rel_l2/tke/mvpe)
    lower/upper = mean of THESE members  -+  the width head's offsets from their own mean
    kit/scoring.py:203 tests the truth against the interval and never the interval against the
    prediction, so the interval's centre is a free parameter; CEN-3 measured +0.083 sps for a
    k=6 coverage-trained centre. This is the in-zip form of score_local.py --bounds-from.

    EVERY centre member is staged under its own name, centre_<i>.pth, starting at 0 -- none is
    aliased to model.pth. That is deliberate: it is exactly the aliasing that made --ckpt and
    --ensemble[0] able to disagree (the guard in __main__, and the 145 corrupted twin zips it
    was added for). There is nothing here for that bug to reach.

    Must run LAST: _payload_width_head rewrites bounds.json and _payload_members writes
    ensemble.json, and this updates both. ensemble.json is created if absent -- a centre is
    legal beside a single-model prediction, and submission.py reads ["members"] unconditionally,
    so the fresh file must carry it.
    """
    import hashlib
    names = []
    for i, m in enumerate(centre):
        _stage_weights(m, STAGING / f"centre_{i}.pth")
        names.append(f"centre_{i}.pth")
    ep = STAGING / "ensemble.json"
    e = json.loads(ep.read_text()) if ep.exists() else {"members": ["model.pth"], "point": "mean"}
    e["centre"] = names
    e["centre_sources"] = [str(m) for m in centre]
    # The SOURCE bytes, like "sha256" above: _stage_weights may drop optimizer state on the way
    # in, so the staged file's hash need not match and the provenance that matters is the run's.
    e["centre_sha256"] = [hashlib.sha256(m.read_bytes()).hexdigest() for m in centre]
    ep.write_text(json.dumps(e, indent=2) + "\n")
    bp = STAGING / "bounds.json"
    b = json.loads(bp.read_text())
    b["centre_bounds"] = True
    bp.write_text(json.dumps(b))
    print(f"decoupled interval centre: {len(centre)} members -> centre_0..centre_{len(centre) - 1}.pth, "
          f'bounds.json "centre_bounds": true')


def _payload_flower(ckpt: Path, bounds: Path) -> None:
    """Our Flower model: the vendored net, the shape constants, einops, bounds.

    The checkpoint is renamed to a fixed ``model.pth`` and the widths to a fixed
    ``bounds.json`` so submission.py does not have to know which run produced
    them. einops comes from the KIT's vendored copy rather than a second one of
    our own -- that is the exact tree we train against, so what runs here and
    what runs locally cannot drift.
    """
    _stage_weights(ckpt, STAGING / "model.pth")
    shutil.copy2(bounds, STAGING / "bounds.json")
    shutil.copy2(SRC / "_vendor" / "flower_standalone.py",
                 STAGING / "flower_standalone.py")
    shutil.copy2(SRC / "geometry.py", STAGING / "geometry.py")
    shutil.copytree(KIT / "_vendor", STAGING / "_vendor", ignore=IGNORE)


def head_n_out(head: Path) -> int:
    """The head's output layout: 2 = symmetric half-width (everything shipped), 4 = H1's
    separate lower/upper offsets, 6 = H2's (delta, lower, upper). Heads trained before
    2026-09-21 carry no key and are symmetric."""
    import torch
    return int(torch.load(head, map_location="cpu", weights_only=False).get("n_out", 2))


def _payload_width_head(head: Path, use_disagreement: bool, scale: float = 1.0) -> None:
    """Trained width model (scripts/train_width_head.py): the weights ship as
    width_head.pth and the STAGED bounds.json gains a "width_head" block naming
    it, so submission.py switches to the head. The fitted widths in bounds.json
    stay (they still cover any channel the head does not)."""
    import hashlib
    n_out = head_n_out(head)
    if n_out not in (2, 4):
        # n_out = 6 is H2: a signed correction to the PREDICTION plus the two offsets.
        # predict() returns the model's own prediction, so shipping it would silently
        # drop the correction and keep the offsets that were fitted with it. H2 is also
        # closed (directions.md): moving the centre by d and the offsets by -+d leaves
        # lower and upper unchanged, so the correction cannot pay in the width reward.
        # Refuse here rather than in the grader -- half a head still scores, just wrongly.
        raise SystemExit(
            f"{head.name} has n_out={n_out}; submission.py consumes the symmetric (2) and the "
            "asymmetric lower/upper (4) layouts only; the n_out=6 layout (a point correction "
            "plus lower/upper) is not supported by the shipped module.")
    shutil.copy2(head, STAGING / "width_head.pth")
    bp = STAGING / "bounds.json"
    b = json.loads(bp.read_text())
    b["width_head"] = {"file": "width_head.pth", "use_disagreement": bool(use_disagreement),
                       "scale": float(scale),
                       "source": str(head), "sha256": hashlib.sha256(head.read_bytes()).hexdigest()}
    bp.write_text(json.dumps(b))
    print(f"width head: {head}  use_disagreement={use_disagreement}  scale={scale:g}  n_out={n_out}")


def build(submission_py: Path, ckpt: Path, name: str, model: str,
          bounds: Path | None = None, members: list[Path] | None = None,
          width_head: Path | None = None, width_head_use_z: bool = False,
          width_scale: float = 1.0, centre: list[Path] | None = None) -> Path:
    for p in (submission_py, ckpt):
        if not p.exists():
            raise SystemExit(f"missing: {p}")
    if centre:
        if model != "flower":
            raise SystemExit("--centre is a flower-payload option")
        missing = [str(p) for p in centre if not p.exists()]
        if missing:
            raise SystemExit("missing centre checkpoint(s): " + ", ".join(missing))
        if width_head is None:
            # Mirrors submission.py's import-time refusal, so a bad build fails HERE and not on
            # the leaderboard: with a centre and no head, predict()'s elif-chain falls through to
            # the disagreement / affine / constant width models, whose inputs the centre changes
            # silently (a k=1 centre has no member std at all).
            raise SystemExit(
                "--centre needs --width-head. A decoupled interval centre is supported with the "
                "trained width head only; the legacy width models would change silently under it.")

    if model == "flower":
        bounds = bounds or bounds_path_for(ckpt)
        if not bounds.exists():
            raise SystemExit(
                f"no bounds file for this checkpoint: {bounds}\n"
                f"  run: python scripts/fit_bounds.py --ckpt {ckpt}")
        b = json.loads(bounds.read_text())
        if b.get("ckpt_sha256"):
            import hashlib
            sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
            if sha != b["ckpt_sha256"]:
                raise SystemExit(
                    f"{bounds.name} was fitted to a different checkpoint "
                    f"({b['ckpt_sha256'][:12]}... vs {sha[:12]}...). Re-run "
                    f"fit_bounds.py, or pass --bounds explicitly to override.")
        print(f"bounds: half-width {b['half_width']}  fit_re {b.get('fit_re')}  "
              f"from {bounds.relative_to(REPO) if bounds.is_relative_to(REPO) else bounds}")
    elif model != "baseline":
        raise SystemExit(f"unknown --model {model!r}; choose baseline or flower")

    global STAGING
    import tempfile
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    STAGING = Path(tempfile.mkdtemp(prefix="staging-", dir=STAGING_ROOT))
    DIST.mkdir(parents=True, exist_ok=True)

    # 1. the entry point, always named submission.py at the root
    shutil.copy2(submission_py, STAGING / "submission.py")
    # 2. whatever that entry point needs beside it
    if model == "flower":
        _payload_flower(ckpt, bounds)
        if width_head is not None:
            if not width_head.exists():
                raise SystemExit(f"missing width head: {width_head}")
            _payload_width_head(width_head, width_head_use_z, width_scale)
        if members:
            _assert_members_agree(ckpt, members)
            _payload_members(members)
            print(f"ensemble: {len(members)} members; bounds must carry a 'disagreement' "
                  f"block for disagreement widths ({'present' if 'disagreement' in b else 'ABSENT -> constant/map widths on the mean'})")
        if centre:
            # LAST: it updates the bounds.json _payload_width_head just rewrote and the
            # ensemble.json _payload_members just wrote. Same semantic guard as the prediction
            # ensemble -- every centre member is run under the SHIPPED cfg from --ckpt, so one
            # that disagrees on predict_delta, window_norm, ... is read under the wrong meaning.
            _assert_members_agree(ckpt, centre)
            _payload_centre(centre)
    else:
        _payload_baseline(ckpt)

    out = DIST / f"{name}.zip"
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(STAGING.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(STAGING))   # relative -> lands at archive root
    finally:
        shutil.rmtree(STAGING, ignore_errors=True)

    mb = out.stat().st_size / 1e6
    print(f"{out}  ({mb:.1f} MB)")
    if mb > CAP_MB:
        raise SystemExit(f"OVER THE {CAP_MB} MB CAP by {mb - CAP_MB:.1f} MB")
    print(f"under the {CAP_MB} MB cap, {CAP_MB - mb:.1f} MB spare")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # flower is the default: every submission since #4 has been ours. The
    # baseline payload stays reachable for control submissions.
    ap.add_argument("--model", choices=("flower", "baseline"), default="flower",
                    help="which set of files to ship beside submission.py")
    ap.add_argument("--submission", type=Path, default=None,
                    help="entry point; defaults to the one matching --model")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint; required for flower, defaults to the FNO "
                         "baseline for --model baseline")
    ap.add_argument("--bounds", type=Path, default=None,
                    help="bounds.json from fit_bounds.py; defaults to "
                         "<ckpt stem>.bounds.json beside the checkpoint")
    ap.add_argument("--name", default=None, help="output zip basename")
    ap.add_argument("--ensemble", type=Path, nargs="+", default=None,
                    help="ship these member checkpoints as an ensemble (member 0 "
                         "doubles as --ckpt; its bounds.json must carry the "
                         "'disagreement' block from disagreement_widths.py "
                         "--write-for-members for disagreement widths)")
    ap.add_argument("--width-head", type=Path, default=None,
                    help="trained width head (.pth from train_width_head.py) to ship as the width model")
    ap.add_argument("--width-scale", type=float, default=1.0,
                    help="multiply every half-width the head predicts. The head trains on 3-member "
                         "fold-ensemble residuals and ships --drop-z, so it cannot see the shipped "
                         "member count; a larger ensemble needs a smaller scale. 1.0 is the identity "
                         "and reproduces every zip built before 2026-09-20.")
    ap.add_argument("--width-head-use-z", action="store_true",
                    help="feed member disagreement to the head (only for a head trained WITH the planes and an ensemble zip)")
    ap.add_argument("--centre", type=Path, nargs="+", default=None,
                    help="CEN: ship these checkpoints as a SECOND ensemble and centre the shipped "
                         "interval on THEIR mean, while the prediction stays the --ensemble mean. "
                         "Sets bounds.json 'centre_bounds': true. Needs --width-head. They are "
                         "staged as centre_0.pth.. and are never --ckpt, so this list is "
                         "independent of --ensemble and may overlap it or not.")
    a = ap.parse_args()
    if a.ensemble:
        a.ckpt = a.ckpt or a.ensemble[0]
        # model.pth is written from --ckpt (_payload_flower) while _payload_members stages members[1:]
        # and RECORDS members[0] in ensemble.json. If --ckpt is not members[0] the zip therefore ships
        # --ckpt in place of members[0]: members[0] is never written, --ckpt appears twice when it is
        # elsewhere in the list, and ensemble.json certifies a set the zip does not contain.
        # Found 2026-09-23 by a code audit AFTER it had corrupted 145 of 611 built ensemble zips --
        # every one of them a twin whose --ckpt was rotated to dodge the bounds.json packing race.
        # No submitted zip was affected. Refuse rather than silently substitute.
        if Path(a.ckpt).resolve() != Path(a.ensemble[0]).resolve():
            raise SystemExit(
                f"--ckpt must be ensemble member 0, or model.pth silently replaces it.\n"
                f"  --ckpt        {a.ckpt}\n"
                f"  --ensemble[0] {a.ensemble[0]}\n"
                f"Reorder --ensemble so the intended reference is first, or drop --ckpt.")

    if a.model == "flower":
        submission = a.submission or REPO / "submission" / "submission_flower.py"
        if a.ckpt is None:
            raise SystemExit("--ckpt is required "
                             "(e.g. outputs/runs/<run>/best_score.pth)")
        name = a.name or "flower"
    else:
        submission = a.submission or KIT / "submission_example_fno.py"
        a.ckpt = a.ckpt or CHECKPOINTS / "sim_real_ft" / "sim_real_fno_fp16.pth"
        name = a.name or "baseline_fno"

    build(submission, a.ckpt, name, a.model, a.bounds, members=a.ensemble, width_scale=a.width_scale,
          width_head=a.width_head, width_head_use_z=a.width_head_use_z, centre=a.centre)
