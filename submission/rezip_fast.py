#!/usr/bin/env python
"""Rebuild an EXISTING submission zip with a different submission.py and/or the T7
fast_predict switch, without touching any checkpoint.

build_submission.py is the real build route and needs the member .pth files. This is
for the two things that do not need them: verifying that a module edit still loads
under the eval image's torch, and A/B-ing the fast switch on weights that are already
packaged.

    python submission/rezip_fast.py submission/dist/sub22_k03.zip \\
        submission/dist/sub22_k03_fast.zip --src submission/submission_flower.py --on
    python scripts/verify_submission.py submission/dist/sub22_k03_fast.zip
"""
import argparse
import json
import pathlib
import shutil
import tempfile
import zipfile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_zip", type=pathlib.Path)
    ap.add_argument("dst_zip", type=pathlib.Path)
    ap.add_argument("--src", type=pathlib.Path, default=None,
                    help="replace submission.py with this file")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--on", action="store_true", help='bounds.json fast_predict: true')
    g.add_argument("--off", action="store_true", help='bounds.json fast_predict: false')
    t = ap.add_mutually_exclusive_group()
    t.add_argument("--timing", action="store_true", help='bounds.json phase_timing: true (one diagnostic stderr line on the first call)')
    t.add_argument("--no-timing", action="store_true", help='bounds.json phase_timing: false')
    # its own group: fp16 and the timing line are independent switches and a zip may want both
    h = ap.add_mutually_exclusive_group()
    h.add_argument("--fp16-head", action="store_true", help='bounds.json head_fp16: true (width head in half precision; prediction unaffected)')
    h.add_argument("--no-fp16-head", action="store_true", help='bounds.json head_fp16: false')
    ap.add_argument("--fluct-scale", type=float, default=None,
                    help='bounds.json fluct_scale: re-inflate the prediction fluctuation about its own time '
                         'mean by this factor (VINF). 1.0 is a bit-identical no-op; ~1.05 was the measured '
                         'optimum. Changes the prediction only, after the bounds are formed.')
    ap.add_argument("--mean-anchor", type=float, default=None,
                    help='bounds.json mean_anchor: shift the prediction\'s per-pixel time mean toward the input '
                         'window\'s masked mean by this fraction, p\' = p + A*(m_in - mean_t(p)), channels u/v '
                         'only (ANCH). 0.0 is a bit-identical no-op. Prediction only, after the bounds and after '
                         'fluct_scale, on both paths; band 0..1.')
    # RBLEND. Two flags because the key has three fields: S W, then the member subset of the second pass.
    r = ap.add_mutually_exclusive_group()
    r.add_argument("--rollout-blend", type=float, nargs=2, default=None, metavar=("S", "W"),
                   help='bounds.json rollout_blend: a second forward of the --rollout-members subset on the '
                        'window ending S frames into the forecast, blended in as p[:, S:] = (1-W) p0[:, S:] + '
                        'W p2[:, :20-S] (scripts/variance_inflation.py --rollout-blend S W). S integer in 1..19, '
                        'W in (0, 1]. Prediction only, bounds stay p0\'s, applied BEFORE fluct_scale and '
                        'mean_anchor on both paths. Needs --rollout-members.')
    r.add_argument("--no-rollout-blend", action="store_true", help="remove bounds.json rollout_blend (off)")
    ap.add_argument("--rollout-members", type=int, nargs="+", default=None, metavar="I",
                    help="member indices (0 = model.pth, ensemble.json order) of the rollout blend's second "
                         "pass, e.g. 0 1; distinct, non-empty. Only with --rollout-blend.")
    ap.add_argument("--pressure-half-width", type=float, default=None,
                    help='bounds.json half_width[2] (the pressure channel). PINS: at c=2 the grader slices '
                         'the bounds to the measured channel count, so this is discarded and costs exactly '
                         'nothing; if a pressure leak ever flipped c to 3, a zero-width band would cover '
                         'nothing and sps would fall by about a third. Free insurance.')
    f = ap.add_mutually_exclusive_group()
    f.add_argument("--tf32", action="store_true", help='bounds.json tf32: true (TF32 matmul/conv; CHANGES the prediction)')
    f.add_argument("--no-tf32", action="store_true", help='bounds.json tf32: false')
    # CEN. --no-centre-bounds is the interesting direction: build ONCE with --centre, then rezip
    # the control off the same archive, so the A/B differs in one JSON key and in nothing else --
    # not the prediction members, not the head, not the zip's byte layout of anything shared.
    # --centre-bounds cannot ADD the centre .pth files (rezip never touches checkpoints), so it
    # refuses on a zip whose ensemble.json names no "centre" list rather than shipping a key
    # submission.py would then raise on.
    c = ap.add_mutually_exclusive_group()
    c.add_argument("--centre-bounds", action="store_true",
                   help='bounds.json centre_bounds: true (centre the interval on the zip\'s "centre" members)')
    c.add_argument("--no-centre-bounds", action="store_true", help="bounds.json centre_bounds: false")
    a = ap.parse_args()
    if (a.rollout_blend is None) != (a.rollout_members is None):
        ap.error("--rollout-blend S W and --rollout-members I [I ...] go together")

    tmp = pathlib.Path(tempfile.mkdtemp())
    with zipfile.ZipFile(a.src_zip) as z:
        z.extractall(tmp)
    if a.src is not None:
        shutil.copy(a.src, tmp / "submission.py")
    if (a.on or a.off or a.timing or a.no_timing or a.fp16_head or a.no_fp16_head or a.tf32 or a.no_tf32
            or a.centre_bounds or a.no_centre_bounds
            or a.pressure_half_width is not None or a.fluct_scale is not None
            or a.mean_anchor is not None
            or a.rollout_blend is not None or a.no_rollout_blend):
        b = json.loads((tmp / "bounds.json").read_text())
        if a.on or a.off:
            b["fast_predict"] = bool(a.on)
        if a.centre_bounds or a.no_centre_bounds:
            if a.centre_bounds:
                ep = tmp / "ensemble.json"
                names = json.loads(ep.read_text()).get("centre") if ep.exists() else None
                if not names:
                    raise SystemExit(
                        f"{a.src_zip.name} names no 'centre' members in ensemble.json, and this script "
                        "cannot add checkpoints. Rebuild with build_submission.py --centre.")
                missing = [n for n in names if not (tmp / n).exists()]
                if missing:
                    raise SystemExit(f"centre members missing from the archive: {', '.join(missing)}")
            b["centre_bounds"] = bool(a.centre_bounds)
        if a.timing or a.no_timing:
            b["phase_timing"] = bool(a.timing)
        if a.fp16_head or a.no_fp16_head:
            b["head_fp16"] = bool(a.fp16_head)
        if a.tf32 or a.no_tf32:
            b["tf32"] = bool(a.tf32)
        if a.fluct_scale is not None:
            b["fluct_scale"] = float(a.fluct_scale)
        if a.mean_anchor is not None:
            b["mean_anchor"] = float(a.mean_anchor)
        if a.no_rollout_blend:
            b.pop("rollout_blend", None)
        if a.rollout_blend is not None:
            S, W = a.rollout_blend
            ep = tmp / "ensemble.json"
            k = len(json.loads(ep.read_text())["members"]) if ep.exists() else 1
            k = max(k, 1)
            m = list(a.rollout_members)
            # the same refusals submission.py raises at import, here where they cost nothing
            if int(S) != S or not 1 <= int(S) <= 19:
                raise SystemExit(f"--rollout-blend S must be an integer in 1..19, got {S:g}")
            if not 0.0 < W <= 1.0:
                raise SystemExit(f"--rollout-blend W must be in (0, 1], got {W:g}")
            if len(set(m)) != len(m) or not all(0 <= i < k for i in m):
                raise SystemExit(f"--rollout-members must be distinct indices in [0, {k}), got {m}")
            if '_BOUNDS.get("rollout_blend")' not in (tmp / "submission.py").read_text():
                # an older module would ignore the key and ship the unblended prediction under its name
                raise SystemExit("the archive's submission.py does not read rollout_blend; pass "
                                 "--src submission/submission_flower.py")
            b["rollout_blend"] = {"shift": int(S), "weight": float(W), "members": m}
        if a.pressure_half_width is not None:
            hw = list(b["half_width"])
            if len(hw) < 3:
                raise SystemExit(f"half_width has {len(hw)} entries; expected 3 (u, v, p)")
            hw[2] = float(a.pressure_half_width)
            b["half_width"] = hw
        (tmp / "bounds.json").write_text(json.dumps(b, indent=2) + "\n")
    with zipfile.ZipFile(a.dst_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(tmp))
    bj = json.loads((tmp / "bounds.json").read_text())
    print(f"{a.src_zip} -> {a.dst_zip}  submission.py={'replaced' if a.src else 'kept'}  "
          f"fast_predict={bj.get('fast_predict', False)} phase_timing={bj.get('phase_timing', False)} "
          f"head_fp16={bj.get('head_fp16', False)} tf32={bj.get('tf32', False)} "
          f"half_width={bj.get('half_width')} fluct_scale={bj.get('fluct_scale', 1.0)} "
          f"mean_anchor={bj.get('mean_anchor', 0.0)} rollout_blend={bj.get('rollout_blend')} "
          f"centre_bounds={bj.get('centre_bounds', False)}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
