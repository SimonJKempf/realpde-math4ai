#!/usr/bin/env python
"""Gate a built submission on HOW its uncertainty bands were fitted.

Widths are never typed into a source file: `fit_bounds.py` writes `<ckpt>.bounds.json`
and `build_submission.py` ships it. That leaves one silent failure mode, which cost
submission 6 about 1.44 sps and reappeared in SUB28/SUB29 on 2026-09-23: `fit_bounds`
defaults to fitting on Re 13950 ALONE and validating on 26700, while every good
submission since has passed `--fit-re 13950 26700`. Re 26700 wants ~44% wider bands,
so a one-row fit ships bands that are too narrow everywhere and loses coverage --
and it does it without an error, a traceback, or a shape change.

Reads the bounds.json actually inside the zip, so it gates the artefact, not the
intent recorded in a queue file.
"""
import argparse, json, sys, zipfile

ap = argparse.ArgumentParser()
ap.add_argument("zips", nargs="+")
ap.add_argument("--require-re", type=int, nargs="*", default=[13950, 26700],
                help="Reynolds rows the fit must cover (default: both)")
ap.add_argument("--ref-half-width", type=float, nargs=2, metavar=("U", "V"),
                help="reference u,v half-widths to compare against, e.g. board #15's")
ap.add_argument("--tol", type=float, default=0.05,
                help="fractional tolerance against --ref-half-width (default 0.05)")
a = ap.parse_args()

bad = []
for z in a.zips:
    try:
        with zipfile.ZipFile(z) as f:
            names = [n for n in f.namelist() if n.endswith("bounds.json")]
            if not names:
                print(f"FAIL  {z}: no bounds.json in the zip")
                bad.append(z)
                continue
            b = json.loads(f.read(names[0]))
    except Exception as e:                                   # noqa: BLE001
        print(f"FAIL  {z}: unreadable ({e})")
        bad.append(z)
        continue

    hw = b.get("half_width") or []
    fr = b.get("fit_re")
    why = []
    if a.require_re and sorted(fr or []) != sorted(a.require_re):
        why.append(f"fit_re={fr}, required {a.require_re}")
    if a.ref_half_width and len(hw) >= 2:
        for i, (got, ref, ch) in enumerate(zip(hw, a.ref_half_width, "uv")):
            d = got / ref - 1 if ref else 0.0
            if abs(d) > a.tol:
                why.append(f"{ch} half-width {got:.6f} is {d:+.1%} vs reference {ref:.6f}")
    tag = "FAIL" if why else "PASS"
    print(f"{tag}  {z.split('/')[-1]:42s} half_width={[round(x, 6) for x in hw[:2]]} "
          f"fit_re={fr} stride={b.get('stride')}")
    for w in why:
        print(f"        {w}")
    if why:
        bad.append(z)

if bad:
    sys.exit(f"BOUNDS GATE FAILED for {len(bad)} zip(s): " + ", ".join(bad))
print(f"BOUNDS GATE PASSED for {len(a.zips)} zip(s)")
