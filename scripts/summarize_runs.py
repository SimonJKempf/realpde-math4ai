#!/usr/bin/env python
"""Roll every training run up into one machine-readable table.

    python scripts/summarize_runs.py                  # print + write the CSV
    python scripts/summarize_runs.py --top 15         # just look at the best
    python scripts/summarize_runs.py --sort score     # by composite estimate
    python scripts/summarize_runs.py --filter mix     # run ids containing "mix"

Reads only `outputs/runs/*/log.jsonl` and `*/meta.json` -- no torch, no data,
no network -- and writes `experiments/results/summary.csv`. Because it is
derived, the CSV can be regenerated at any time and is never hand-edited: the
run directories are the source of truth, and `submissions.tsv` beside it remains
the separate record of what was actually submitted to Codabench.

One row per run directory. Best epochs are recomputed from the log rather than
read from a checkpoint, and `best_val_score` is recomputed from the logged
errors with the current `pde_competition.composite` -- so a changed formula is
reflected for every run, historical ones included, and the log's own
`val_score` (frozen at training time) is only what selected `best_score.pth`.

Columns that older runs cannot fill are left empty rather than guessed:
`git_commit`, `split` and `seed` only entered the log format in September 2026,
and for runs before that the split is recoverable from `last.pth`'s `guard`
dict. Nothing here loads a checkpoint to find out.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from pde_competition import composite
from pde_competition.paths import REPO, OUTPUTS, RESULTS

DEFAULT_CSV = RESULTS / "summary.csv"

COLUMNS = ("run_id", "name", "started", "epochs", "best_val_rel_l2",
           "best_epoch", "val_tke_at_best", "best_val_tke", "best_val_score",
           "val_re13950_at_best", "val_re26700_at_best", "train_all",
           "in_sample_val_rel_l2",
           "best_score_epoch", "floor_rel_l2", "floor_ratio", "split", "seed",
           "channels", "lifting_dim", "n_levels", "batch_size", "lr", "stride",
           "window_norm", "boundary", "predict_delta", "mix_sim", "init_from",
           "minutes", "pack", "n_params", "git_commit", "git_dirty", "wandb_id",
           "wandb_group")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A run killed mid-write leaves a truncated final line. That is
                # not a reason to lose the epochs before it.
                pass
    return rows


def summarize(run_dir: Path) -> dict | None:
    """One row for one run directory, or None if it has no usable epochs."""
    log = run_dir / "log.jsonl"
    if not log.exists():
        return None
    epochs = read_jsonl(log)
    if not epochs:
        return None

    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
    launches = meta.get("launches") or []
    first, last_launch = (launches[0], launches[-1]) if launches else ({}, {})
    args = last_launch.get("args", {})

    def best_by(key, lower_is_better=True):
        have = [e for e in epochs if isinstance(e.get(key), (int, float))
                and not math.isnan(e[key])]
        if not have:
            return None
        return min(have, key=lambda e: e[key]) if lower_is_better else \
            max(have, key=lambda e: e[key])

    best = best_by("val_rel_l2")
    best_tke = best_by("val_tke")
    # The composite is recomputed here from the two logged errors with the
    # CURRENT composite.WEIGHTS / P, not read from the log's val_score, which
    # was computed with whatever the weights were at training time. train.py
    # logs the errors every epoch for exactly this reason. Consequences: runs
    # that predate val_score get a value too, and when the formula changes (it
    # was solved exactly on 2026-09-10) every row moves together.
    for e in epochs:
        if isinstance(e.get("val_rel_l2"), (int, float)) and isinstance(e.get("val_tke"), (int, float)):
            e["_score_now"] = composite.from_errors(e["val_rel_l2"], e["val_tke"])
    best_score = best_by("_score_now", lower_is_better=False)
    tail = epochs[-1]
    cfg = tail.get("config", {})
    floor = (tail.get("persistence_floor") or {}).get("rel_l2")
    bc = cfg.get("boundary_condition_types") or []

    row = {
        "run_id": run_dir.name,
        # The timestamp suffix is the run's identity, so the bare name is what
        # groups repeats of one configuration together.
        "name": run_dir.name.rsplit("-", 2)[0],
        "started": first.get("started", ""),
        "epochs": len(epochs),
        "best_val_rel_l2": best and best["val_rel_l2"],
        "best_epoch": best and best["epoch"],
        "val_tke_at_best": best and best.get("val_tke"),
        # Per held-out row at the best (pooled) epoch; empty for runs that
        # predate per-row logging (2026-09-15).
        "val_re13950_at_best": best and ((best.get("val_by_re") or {}).get("13950") or {}).get("rel_l2"),
        "val_re26700_at_best": best and ((best.get("val_by_re") or {}).get("26700") or {}).get("rel_l2"),
        "train_all": tail.get("train_all", ""),
        "best_val_tke": best_tke and best_tke.get("val_tke"),
        "best_val_score": best_score and best_score.get("_score_now"),
        "best_score_epoch": best_score and best_score["epoch"],
        "floor_rel_l2": floor,
        "floor_ratio": (best["val_rel_l2"] / floor
                        if best and floor else None),
        # Present in the log only for runs from 2026-09-09 on; before that the
        # CLI arg is the only record, and meta.json did not exist either.
        "split": tail.get("split", args.get("split", "")),
        "seed": tail.get("seed", args.get("seed", "")),
        "channels": cfg.get("c_in"),
        "lifting_dim": cfg.get("lifting_dim"),
        "n_levels": cfg.get("n_levels"),
        "batch_size": args.get("batch_size", ""),
        "lr": args.get("lr", ""),
        "stride": args.get("stride", ""),
        "window_norm": cfg.get("window_norm"),
        "boundary": (bc[0] if bc else ""),
        "predict_delta": cfg.get("predict_delta"),
        "mix_sim": tail.get("mix_sim"),
        "init_from": tail.get("init_from") or "",
        "minutes": round(sum(e.get("seconds", 0) for e in epochs) / 60, 1),
        # Runs that shared a GPU. >1 means `minutes` measures the pack, not the
        # model: filter on pack==1 before reading any timing across runs.
        "pack": last_launch.get("pack", 1),
        "n_params": last_launch.get("n_params", ""),
        "git_commit": (tail.get("git_commit")
                       or (first.get("git") or {}).get("commit") or "")[:12],
        "git_dirty": tail.get("git_dirty",
                              (first.get("git") or {}).get("dirty", "")),
        "wandb_id": last_launch.get("wandb_id") or "",
        "wandb_group": args.get("wandb_group") or "",
    }
    if tail.get("train_all"):
        # --train-all runs saw every cell: their val is IN-SAMPLE and 30x the
        # noise floor in the flattering direction (0.0708 next to 0.0785 for
        # the same recipe, W26). Keep the number, but never in the columns
        # anything groups or sorts on. Same class of fix as the pack column.
        row["in_sample_val_rel_l2"] = row["best_val_rel_l2"]
        for k in ("best_val_rel_l2", "best_val_score", "floor_ratio",
                  "val_re13950_at_best", "val_re26700_at_best"):
            row[k] = None
    return {k: ("" if row.get(k) is None else row[k]) for k in COLUMNS}


def main(a: argparse.Namespace) -> None:
    runs = sorted(p for p in (OUTPUTS / "runs").glob("*") if p.is_dir())
    rows = [r for r in (summarize(p) for p in runs) if r]
    empty = len(runs) - len(rows)
    if empty:
        # Kept, not deleted: a directory with no epochs is the record of a run
        # that died before its first validation pass.
        print(f"note: {empty} run directories have no usable log.jsonl "
              f"(died before epoch 1); they are not in the table")
    if a.filter:
        rows = [r for r in rows if a.filter in r["run_id"]]
    if not rows:
        raise SystemExit(f"no runs with a log.jsonl under {OUTPUTS / 'runs'}")

    keys = {"rel_l2": ("best_val_rel_l2", False),
            "score": ("best_val_score", True),
            "tke": ("best_val_tke", False),
            "date": ("run_id", True)}
    key, reverse = keys[a.sort]
    rows.sort(key=lambda r: (r[key] == "", r[key]), reverse=reverse)

    if not a.no_write:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            # Written in run_id order regardless of --sort: a stable order keeps
            # the diff of this file readable as runs are added.
            w.writerows(sorted(rows, key=lambda r: r["run_id"]))
        print(f"{len(rows)} runs -> {a.out.relative_to(REPO)}\n")

    shown = rows[:a.top] if a.top else rows
    head = (f"{'run_id':40s} {'ep':>3s} {'val_rel_l2':>10s} {'@':>3s} "
            f"{'val_tke':>8s} {'score':>7s} {'x floor':>7s} {'min':>6s} "
            f"{'commit':>12s}")
    print(head)
    print("-" * len(head))
    def num(v, width, prec=None):
        """Right-aligned number, or a padded dash -- runs predating a metric
        must not shift the columns of the ones that have it."""
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return "-".rjust(width)
        return f"{v:{width}d}" if prec is None else f"{v:{width}.{prec}f}"

    for r in shown:
        print(f"{r['run_id']:40s} {num(r['epochs'], 3)} "
              f"{num(r['best_val_rel_l2'], 10, 5)} "
              f"{num(r['best_epoch'], 3)} "
              f"{num(r['val_tke_at_best'], 8, 4)} "
              f"{num(r['best_val_score'], 7, 3)} "
              f"{num(r['floor_ratio'], 7, 3)} "
              f"{num(r['minutes'], 6, 1)} "
              f"{r['git_commit'] or '-':>12s}")
    if a.top and len(rows) > a.top:
        print(f"... {len(rows) - a.top} more")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_CSV,
                   help=f"CSV to write (default: {DEFAULT_CSV.relative_to(REPO)})")
    p.add_argument("--no-write", action="store_true",
                   help="print the table only, leave the CSV alone")
    p.add_argument("--sort", default="rel_l2",
                   choices=("rel_l2", "score", "tke", "date"),
                   help="ordering of the printed table (default: rel_l2, best "
                        "first). The CSV is always written in run_id order.")
    p.add_argument("--top", type=int, default=0, metavar="N",
                   help="print only the first N rows (default: all)")
    p.add_argument("--filter", default=None, metavar="SUBSTR",
                   help="keep only runs whose id contains SUBSTR")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
