"""Average the weights of several FlowerS2S checkpoints into one (weight
averaging / "model soup").

    python scripts/average_checkpoints.py OUT.pth IN1.pth IN2.pth ...

Every input must carry the same ``config``. Floating-point tensors are
averaged uniformly; every other entry (integer buffers such as counters)
must be identical across inputs and is copied from the first. The output
is a weights-only checkpoint in the same format ``FlowerS2S.save`` writes
-- ``config`` + ``state_dict`` -- so ``evaluate.py``, ``fit_bounds.py`` and
``build_submission.py`` accept it unchanged. Provenance travels with it as
``averaged_from`` (input paths) and ``averaged_epochs``.

Two uses, both W58 (2026-09-16):
  tail averaging -- the ``epochNNN.pth`` snapshots of ONE run written by
      ``train.py --save-epochs`` (SWA-style, no training change);
  seed soup -- the ``best_score.pth`` of several fine-tunes from the SAME
      pretrain checkpoint. Members from different inits are not in one
      basin and must not be souped; the script cannot detect that, the
      parcel must.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def average_state_dicts(states: list[dict]) -> dict:
    """Uniform mean of float tensors; non-float entries must agree and are copied."""
    if not states:
        raise ValueError("no state_dicts to average")
    keys = list(states[0].keys())
    for i, s in enumerate(states[1:], 1):
        if list(s.keys()) != keys:
            raise ValueError(f"state_dict {i} has different keys from state_dict 0")
    out = {}
    for k in keys:
        ts = [s[k] for s in states]
        if ts[0].is_floating_point():
            out[k] = torch.stack([t.to(torch.float64) for t in ts]).mean(0).to(ts[0].dtype)
        else:
            for i, t in enumerate(ts[1:], 1):
                if not torch.equal(t, ts[0]):
                    raise ValueError(f"non-float entry {k!r} differs between inputs 0 and {i}")
            out[k] = ts[0].clone()
    return out


def average_checkpoints(inputs: list[Path], out: Path) -> dict:
    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in inputs]
    cfg0 = ckpts[0]["config"]
    for p, c in zip(inputs[1:], ckpts[1:]):
        if c["config"] != cfg0:
            raise ValueError(f"config of {p} differs from {inputs[0]}")
    state = average_state_dicts([c["state_dict"] for c in ckpts])
    payload = {"config": cfg0, "state_dict": state,
               "averaged_from": [str(p) for p in inputs],
               "averaged_epochs": [c.get("epoch") for c in ckpts],
               "n_averaged": len(ckpts)}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("inputs", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if len(a.inputs) < 2:
        ap.error("need at least two checkpoints to average")
    for p in a.inputs:
        if not p.is_file():
            ap.error(f"missing checkpoint: {p}")
    payload = average_checkpoints(a.inputs, a.out)
    print(f"averaged {payload['n_averaged']} checkpoints (epochs {payload['averaged_epochs']}) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
