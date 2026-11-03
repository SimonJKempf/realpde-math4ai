"""Refuse unless every checkpoint's config is the rollout recipe we mean to ship.

    python scripts/diag/assert_rollout_members.py --frames 10 --feed 2 --tlift 5 CKPT [CKPT ...]
"""
import argparse
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, required=True)
ap.add_argument("--feed", type=int, required=True)
ap.add_argument("--tlift", type=int, required=True)
ap.add_argument("ckpts", nargs="+")
a = ap.parse_args()
bad = 0
for p in a.ckpts:
    c = torch.load(p, map_location="cpu", weights_only=False)["config"]
    c = c if isinstance(c, dict) else vars(c)
    got = (c.get("rollout_frames"), c.get("rollout_feed"), c.get("temporal_lift"))
    ok = got == (a.frames, a.feed, a.tlift)
    bad += not ok
    print(("member ok " if ok else "MEMBER WRONG ") + p, "rollout_frames/feed/tlift =", got)
sys.exit(1 if bad else 0)
