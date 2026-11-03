"""Refuse unless every checkpoint's config is identical (the shipped module builds every member from member 0's config).

    python scripts/diag/assert_members_same_config.py CKPT [CKPT ...]
"""
import sys

import torch

cfgs = []
for p in sys.argv[1:]:
    c = torch.load(p, map_location="cpu", weights_only=False)["config"]
    cfgs.append(c if isinstance(c, dict) else vars(c))
    print(p, {k: cfgs[-1].get(k) for k in ("rollout_frames", "rollout_feed", "temporal_lift", "predict_delta", "c_in", "lifting_dim")})
bad = [(i, k) for i, c in enumerate(cfgs[1:], 1) for k in set(c) | set(cfgs[0]) if c.get(k) != cfgs[0].get(k)]
print("member configs identical" if not bad else f"MEMBER CONFIGS DIFFER: {bad}")
sys.exit(1 if bad else 0)
