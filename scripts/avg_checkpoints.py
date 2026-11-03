"""Average the weights of several checkpoints of ONE run (e.g. epoch030 + epoch040) into a new checkpoint.

    python scripts/avg_checkpoints.py OUT.pth IN1.pth IN2.pth [...]

Every input must share config and state_dict keys/shapes; the first input's dict (config, stats, ...) is kept and
only "state_dict" (or "model") is replaced by the element-wise mean. Integer buffers are taken from the first input.
"""
import sys

import torch

out, ins = sys.argv[1], sys.argv[2:]
cks = [torch.load(p, map_location="cpu", weights_only=False) for p in ins]
key = "state_dict" if "state_dict" in cks[0] else "model"
sds = [c[key] for c in cks]
for c in cks[1:]:
    if c.get("config") != cks[0].get("config"):
        raise SystemExit("configs differ -- refusing to average across runs")
avg = {}
for k, v in sds[0].items():
    if torch.is_floating_point(v):
        avg[k] = sum(sd[k].float() for sd in sds).div(len(sds)).to(v.dtype)
    else:
        avg[k] = v
res = dict(cks[0])
res[key] = avg
torch.save(res, out)
print(f"averaged {len(ins)} checkpoints ({key}, {len(avg)} tensors) -> {out}")
