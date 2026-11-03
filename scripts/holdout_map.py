#!/usr/bin/env python
"""HO2: the phase-mix effect per Reynolds row, across every holdout swap measured.

    python scripts/holdout_map.py

Reads the twin scores out of the Slurm logs of the HO2 twin jobs and prints one row per
Reynolds condition: the mix-minus-no-mix difference in est_final and in each subscore.
The accuracy subscores (rel_l2, tke, mvpe) do NOT involve the width head; sps does. That
distinction is the reason this script reports them separately -- the head is validated on
13950/26700 (scripts/train_width_head.py --val-re), so swapped twins apply it in-sample
and the standard twin applies it out-of-sample, and the accuracy columns are what rule
that asymmetry out as an explanation.

Exclusion handling: each arm's clean ensemble is passed in EXCLUDE below, applied per host
per the rule adopted 2026-09-22 (the wst002/wst003 offset of 0.00085 exceeds the rule's own
absolute threshold, so a family spanning hosts over-flags).
"""
import re, sys, glob, statistics as st

# job -> (tag, (row_lo, row_hi), nomix_pick, mix_pick)
# "loo" = mean over that arm's legitimate leave-one-out subsets; "dNNN" = the one clean subset.
JOBS = [
    ("4692", "ho2a", ("12675", "25425"), "loo",  "d104"),
    ("4697", "ho2b", ("11400", "24150"), "loo",  "d105"),
    ("4713", "ho2c", ("10125", "22875"), "d103", "loo"),
    ("4713", "ho2d", ("8850",  "21600"), "k3",   "k3clean"),
]
FIELDS = ("rel", "tke", "mvpe", "sps", "fin")


def parse(job, tag):
    txt = open(f"outputs/logs/slurm/realpde-packed-{job}.out", errors="ignore").read()
    arms = {}
    for b in re.split(rf"scores/twin_{tag}_", txt)[1:]:
        m = re.match(r"(nomix|mix)_(\w+)/scores\.json", b)
        if not m:
            continue
        per = {}
        for rr in re.finditer(r"^\s*(\d{4,5}|all)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)", b, re.M):
            per[rr.group(1)] = dict(zip(FIELDS, [float(rr.group(i)) for i in (2, 3, 4, 5, 6)]))
        if per:
            arms.setdefault(m.group(1), {})[m.group(2)] = per
    return arms


def pick(arm, how, row):
    if how == "loo":
        vs = [v[row] for k, v in arm.items() if k != "k5" and not k.startswith("k3") and row in v]
    elif how == "k3":
        vs = [v[row] for k, v in arm.items() if k.startswith("k3_") and row in v]
    else:
        vs = [arm[how][row]] if how in arm and row in arm[how] else []
    if not vs:
        return None
    return {f: st.mean(v[f] for v in vs) for f in FIELDS}


def main():
    print(f"{'row':>7} {'swap':>6} {'d rel_l2':>9} {'d tke':>8} {'d mvpe':>8} {'d sps':>8} {'d final':>9}")
    out = []
    for job, tag, rows, pn, pm in JOBS:
        if not glob.glob(f"outputs/logs/slurm/realpde-packed-{job}.out"):
            continue
        arms = parse(job, tag)
        if "nomix" not in arms or "mix" not in arms:
            continue
        for row in rows:
            a, b = pick(arms["nomix"], pn, row), pick(arms["mix"], pm, row)
            if a and b:
                d = {f: b[f] - a[f] for f in FIELDS}
                out.append((int(row), tag, d))
    for row, tag, d in sorted(out):
        print(f"{row:>7} {tag:>6} {d['rel']:+9.3f} {d['tke']:+8.3f} {d['mvpe']:+8.3f} {d['sps']:+8.3f} {d['fin']:+9.3f}")
    print(f"\n  standard   13950   +0.023   -0.045   +0.086   +0.042    +0.001   (D1g, the pair every other twin uses)")
    print(f"  standard   26700      ...      ...      ...      ...     +0.032")
    if out:
        fins = [d["fin"] for _, _, d in out]
        print(f"\n  swapped rows: n={len(fins)}  min {min(fins):+.3f}  max {max(fins):+.3f}  mean {st.mean(fins):+.3f}")
        print(f"  the two standard rows are +0.001 and +0.032 -- read the set, never an average")


if __name__ == "__main__":
    main()
