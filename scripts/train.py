#!/usr/bin/env python
"""Train a Flower model on the real PIV trajectories.

    python scripts/train.py --smoke                 # 2 min, checks the wiring
    python scripts/train.py --name flower_v1 --epochs 40

Reads the 32x64 store under data/processed/ (build it once with
scripts/subsample_data.py), holds out whole (Re, aoa) cells -- the same ones
scripts/score_local.py scores on, so training-time validation and local scoring
never disagree about what the model has seen.

Every run writes one self-contained directory, named for the run and stamped
with the time it started, so two runs can never overwrite or interleave:

    outputs/runs/<name>-<YYYYmmdd-HHMMSS>/
        best.pth        lowest val rel-L2, weights + config + stats
        best_score.pth  highest composite estimate -- see pde_competition.composite
        last.pth        most recent epoch, for resuming
        log.jsonl       one line per epoch
        meta.json       provenance: commit, argv, host, torch, W&B run id

Two selected checkpoints because rel-L2 is not the objective: val TKE usually
bottoms out at a different epoch than val rel-L2, and TKE is worth more per
unit than rel-L2 at the operating point of this competition. best_score.pth is
the one to submit. log.jsonl records val_score every epoch, so a revised
weighting can be re-applied to past runs offline.

Add --wandb to mirror the metrics to Weights & Biases. It is strictly optional:
the JSONL is written either way and stays the authoritative record.
scripts/summarize_runs.py rolls every run directory up into one CSV table.

Validation loss here is mean relative L2, the quantity behind rel_l2_score --
but it is NOT the leaderboard number. Run scripts/score_local.py on a built zip
for that; it is the only thing that computes all five subscores.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from pde_competition import composite
from pde_competition.dataset import (WindowDataset, channel_stats, list_runs,
                                    measured_channels, split_by_condition,
                                    split_temporal_gapped)
from pde_competition.model import (rel_l2_per_channel, FlowerConfig, FlowerS2S, horizon_weights,
                                   mvpe_rel_l2, persistence, rel_l2,
                                   rel_l2_per_frame, rel_l2_per_sample, tke_rel_l2, vrmse, div_rel_l2, amse_time,
                                   fluct_corr,
                                   rel_l1, coverage_surrogate, exp_reward,
                                   ensemble_ambiguity)
from pde_competition.paths import OUTPUTS, REPO


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# The scorer's own width constant, read from the kit rather than re-typed: kit/scoring.py
# defines SIGMA_GLOBAL = 0.0563870259 and fit_bounds.py imports it the same way.
SIGMA_SPS = 0.0563870259


# ---------------------------------------------------------------------------
# PNO_D probe (2026-09-25, Simon-approved): FINE-TUNE WITH THE ENERGY SCORE ON
# MC-DROPOUT SAMPLES (Buelte / Scholl / Kutyniok, arXiv 2502.12902). Reached only
# under --energy-score-samples M >= 2; at the default 0 not one line of it runs
# and a run is bit-identical to before.
# ---------------------------------------------------------------------------

def _safe_norm(v: torch.Tensor, dim: int) -> torch.Tensor:
    """L2 norm over ``dim`` whose VALUE is exactly 0 and whose GRADIENT is 0 where the
    vector is exactly zero. ``torch.linalg.norm`` has an undefined gradient there, and
    the energy score hits that point by construction (a sample equal to the target, or
    two identical samples); the double-``where`` keeps the clamp out of the value."""
    sq = (v * v).sum(dim=dim)
    return torch.where(sq > 0, sq.clamp_min(1e-30).sqrt(), torch.zeros_like(sq))


def energy_score(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of the per-window RELATIVE (fair) energy score.

    ``samples`` is (M, B, ...), M >= 2 predictions of the SAME windows, already sliced
    to the measured channels; ``target`` is (B, ...). Per window b, with norms over
    every remaining axis (T, H, W, C_measured) and y = target_b:

        ES_b = (1/M) sum_i ||s_i - y|| / ||y||  -  1/(2 M (M-1)) sum_{i != j} ||s_i - s_j|| / ||y||

    The first term is the samples' mean rel-L2 against the truth; the second rewards
    SPREAD between samples, and is what makes this a proper scoring rule for the
    predictive distribution rather than a point loss. Dividing by ||y|| makes it
    scale-free across the Reynolds rows exactly as ``rel_l2`` is (the real rows span
    5.89x in amplitude). The M(M-1) normalisation is the FAIR (unbiased) estimator.

    Properties the tests pin: 0 when every sample equals y; >= 0 always (triangle
    inequality: sum_{i<j} ||s_i - s_j|| <= (M-1) sum_i ||s_i - y||). NOT strictly
    positive for finite M: two samples placed symmetrically about y (y +- d) score
    exactly 0 -- the fair estimator is proper in expectation, not per draw.
    Unmeasured target elements (exact zeros) are LEFT IN, as ``rel_l2`` leaves them.
    """
    m, n = samples.shape[0], samples.shape[1]
    if m < 2:
        raise ValueError(f"energy_score needs at least 2 samples, got {m}")
    if samples.shape[1:] != target.shape:
        raise ValueError(f"samples are {tuple(samples.shape)}, target is {tuple(target.shape)}")
    s = samples.reshape(m, n, -1)
    t = target.reshape(n, -1)
    ynorm = torch.linalg.norm(t, dim=1).clamp(min=1e-8)                         # (B,)
    acc = _safe_norm(s - t.unsqueeze(0), dim=2).mean(dim=0)                     # (B,)
    # sum over i != j is twice the sum over i < j, so the 1/(2M(M-1)) becomes 1/(M(M-1)).
    pair = sum(_safe_norm(s[i] - s[j], dim=1) for i in range(m) for j in range(i + 1, m))
    return ((acc - pair / (m * (m - 1))) / ynorm).mean()


def check_energy_score_model(model, m: int) -> int:
    """The number of ACTIVE nn.Dropout modules (p > 0) in ``model``; SystemExit when the
    energy score is asked for (m >= 2) and there are none -- the M samples would then be
    identical, the spread term exactly 0, and ES would reduce to plain rel-L2: an
    unwitting control. m == 1 is refused too (no spread term exists for one sample)."""
    if m == 1 or m < 0:
        raise SystemExit(f"--energy-score-samples {m}: needs M >= 2 (0 = off)")
    n = sum(1 for q in model.modules() if isinstance(q, torch.nn.Dropout) and q.p > 0)
    if m >= 2 and n == 0:
        raise SystemExit(
            f"--energy-score-samples {m} on a model with no active dropout (dropout_rate "
            f"{getattr(getattr(model, 'cfg', None), 'dropout_rate', '?')}): the {m} samples would be "
            f"identical and the energy score would be plain rel-L2. Pass --dropout P > 0 (it "
            f"overrides the checkpoint's value under --init-from).")
    return n


def run_epoch(model, loader, device, measured, optimizer, scheduler=None, input_noise=0.0, loss_kind="rel_l2", amp_jitter=None,
              div_w=0.0, speed_power=0.0, l1_w=0.0, cov_w=0.0, cov_half=None, exp_w=0.0, chan_w=None,
              max_steps=None, horizon_w=None, tke_w=0.0, mvpe_w=0.0, amse_w=0.0, corr_w=0.0,
              teachers=None, distill_w=0.0, truth_w=1.0, distill_tke=False, distill_stats=None,
              rollout_w=0.0, rollout_detach=False, rollout_stats=None,
              es_m=0, es_w=1.0, es_replace=False, es_stats=None) -> float:
    """One training pass. Returns mean relative L2.

    ``es_m`` (PNO_D probe, 2026-09-25) is 0 for every ordinary run, and then not
    one line of the energy-score code runs -- the path is bit-identical to before.
    At M >= 2 the batch is pushed through the model M times with independent
    dropout masks (as ONE forward of the batch repeated M times: nothing in
    FlowerS2S mixes windows -- GroupNorm, window_scale and the Lagrangian shear are
    all per window -- so this is the same distribution as M separate passes, at
    one kernel launch sequence). ``pred`` below is then the SAMPLE MEAN, so every
    point term (rel_l2, tke, coverage, ...) and the RETURNED train_rel_l2 are the
    mean's, and the loss gains ``es_w * energy_score(samples, tgt)`` -- or, with
    ``es_replace``, the primary point term (rel_l2, or whatever --loss /
    --horizon-weight / --chan-weight / --speed-weight-power selected) is REPLACED
    by it. ``es_stats``, if a dict, receives the epoch mean under ``"energy"``.

    ``rollout_w`` (direction ROLL, 2026-09-24) is 0.0 for every ordinary run, and
    then the loader yields ``(x, y)`` and not one line of the rollout code runs --
    the path is bit-identical to before. When positive, the loader yields
    ``(x, y, y2)`` (WindowDataset extra_out = t_out) and the batch's loss gains

        rollout_w * ( rel_l2(pred2, y2) + tke_w * tke_rel_l2(pred2, y2) )

    where ``pred2 = model(x2)`` is a SECOND pass on the model's own first-pass
    prediction composed as the next input window (``rollout_input``), and ``y2``
    is the TRUE frames [t_in + t_out, t_in + 2 t_out). Gradients flow through the
    first pass into ``x2`` unless ``rollout_detach``. Training only: the shipped
    module never rolls out. ``rollout_stats``, if a dict, receives the epoch mean of
    the term under ``"rollout"``; the RETURN stays the first pass's plain rel-L2.

    ``teachers`` (direction DIST, 2026-09-24) is None for every ordinary run, and
    then not one line of the distillation code runs -- the path is bit-identical
    to before. Given a list of frozen models, the batch's loss becomes

        truth_w * L_truth  +  distill_w * rel_l2(pred, t)          [+ distill_w * tke_w * tke_rel_l2(pred, t)]

    where ``L_truth`` is the WHOLE objective composed below (rel_l2 + tke + cov
    + ...) and ``t`` is the teachers' mean prediction on the SAME input the
    student sees, augmentation included. The bracketed term is ``distill_tke``:
    the teacher mean is smoother than any member, so matching it in rel-L2 alone
    would pull the student's fluctuation energy DOWN, and TKE is the subscore the
    ensemble is shipped for. ``distill_stats``, if a dict, receives the epoch means
    of the two components (``truth``, ``distill``) so log.jsonl can carry them
    without changing the return value, which stays the plain rel-L2.

    ``tke_w`` adds ``tke_w * tke_rel_l2(pred, tgt)`` to the loss -- the
    composite-mirroring objective. At submission 8's operating point one unit
    of TKE error is worth 0.136 of one unit of rel-L2 error on final_score
    (marginals from the solved power mean: 0.102*28.3 vs 0.473*44.8), so 0.14
    weights the two terms exactly as the leaderboard does. The RETURNED number
    stays the plain rel-L2.

    ``measured`` is how many channels to score. Everything past it is dropped
    from both sides, exactly as ``kit/scoring.py`` drops the unmeasured ``p``
    before computing any metric -- otherwise the loss spends gradient on a
    channel the leaderboard never reads.

    ``horizon_w`` is None for the historical objective -- one relative L2 over
    the flattened window -- or a (T,) tensor of per-frame weights, in which case
    the loss is the weighted mean of per-frame relative L2. The RETURNED number
    is always the unweighted global rel-L2, so train_rel_l2 stays comparable
    across arms even when the objective does not.
    """
    model.train()
    if es_m:
        check_energy_score_model(model, es_m)
    total, n = 0.0, 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        if rollout_w:
            # ROLL: a third tensor, the truth for the second pass. Only ever
            # yielded by a dataset built with extra_out, which main() does iff
            # rollout_w > 0 -- a 3-tuple against rollout_w == 0 fails loudly here.
            x, y, y2 = batch
            y2 = y2.to(device, non_blocking=True)
        else:
            x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if input_noise:
            # PIV-like measurement noise on the INPUT window only (W65, 2026-09-16):
            # Gaussian with sd = input_noise x the batch's per-channel std, kept
            # off unmeasured (zero) points so real masks stay exact. Targets are
            # untouched -- the model learns to denoise, not to predict noise.
            sd = x.flatten(0, -2).std(0).clamp(min=1e-8) * input_noise      # (C,)
            x = x + torch.randn_like(x) * sd * (x != 0)
        if amp_jitter:
            # Reynolds-similarity augmentation (a) (fresh-eyes proposition 7, 2026-09-18): one
            # amplitude factor s ~ U[lo, hi] per window on INPUT AND TARGET -- the same flow at a
            # different velocity scale. Unmeasured zeros stay zero.
            lo, hi = amp_jitter
            sc = torch.empty(x.shape[0], *([1] * (x.dim() - 1)), device=x.device).uniform_(lo, hi)
            x = x * sc; y = y * sc
            if rollout_w:
                y2 = y2 * sc          # the same flow at the same scale, 20 frames later
        # The FULL prediction is kept (all c_out channels): the rollout composes the
        # next input window from it. Slicing a view costs no arithmetic, so ``pred``
        # is what it always was.
        if es_m:
            # ES: M samples as one batched forward; copy k is rows [k*B, (k+1)*B).
            samples = model(x.repeat(es_m, *([1] * (x.dim() - 1))))
            samples = samples.reshape(es_m, x.shape[0], *samples.shape[1:])     # (M, B, T, H, W, C)
            out = samples.mean(0)
        else:
            out = model(x)
        pred = out[..., :measured]
        tgt = y[..., :measured]
        if loss_kind == "vrmse":
            loss = vrmse(pred, tgt)
            reported = rel_l2(pred, tgt).detach()
        elif horizon_w is None and speed_power:
            # Per-window weights from the INPUT window's mean |u| (the Reynolds
            # proxy the grader cannot hide; W99, 2026-09-17): w ~ speed^p,
            # normalised to mean 1, so faster (higher-Re) windows count more.
            per = rel_l2_per_sample(pred, tgt)
            u = x[..., 0]; meas = (u != 0).to(u.dtype)
            spd = (u.abs() * meas).sum(dim=(1, 2, 3)) / meas.sum(dim=(1, 2, 3)).clamp(min=1.0)
            w = (spd / spd.mean().clamp(min=1e-8)) ** speed_power
            w = w / w.mean()
            loss = (per * w).mean()
            reported = per.mean().detach()
        elif horizon_w is None and chan_w is not None:
            # PER-CHANNEL relative L2. rel_l2 flattens u and v into one norm, so the
            # ratio is dominated by u (sd 0.106 against v's 0.0157, ~7x). The score
            # normalises over the measured channels together, so v is scored but barely
            # trained. chan_w = (1, 1) trains them equally in RELATIVE terms.
            loss = rel_l2_per_channel(pred, tgt, chan_w)
            reported = rel_l2(pred, tgt).detach()
        elif horizon_w is None:
            loss = rel_l2(pred, tgt)
            reported = loss.detach()
        else:
            per_frame = rel_l2_per_frame(pred, tgt)          # (B, T)
            loss = (per_frame * horizon_w).mean()
            reported = rel_l2(pred, tgt).detach()
        if es_m:
            es = energy_score(samples[..., :measured], tgt)
            loss = es_w * es if es_replace else loss + es_w * es
            if es_stats is not None:
                es_stats["energy"] = es_stats.get("energy", 0.0) + es.item() * x.shape[0]
        if tke_w:
            loss = loss + tke_w * tke_rel_l2(pred, tgt)
        if amse_w:
            loss = loss + amse_w * amse_time(pred, tgt)
        if corr_w:
            # A7: the per-lead fluctuation CORRELATION, scale-invariant, so it spends its
            # whole gradient on phase and none on the amplitude VINF already sets post-hoc.
            loss = loss + corr_w * fluct_corr(pred, tgt)
        if mvpe_w:
            loss = loss + mvpe_w * mvpe_rel_l2(pred, tgt)
        if div_w:
            loss = loss + div_w * div_rel_l2(pred, tgt)
        if l1_w:
            loss = loss + l1_w * rel_l1(pred, tgt)
        if cov_w:
            loss = loss + cov_w * coverage_surrogate(pred, tgt, cov_half)
        if exp_w:
            loss = loss + exp_w * exp_reward(pred, tgt, SIGMA_SPS)
        if rollout_w:
            roll = rollout_term(model, out, x, y2, measured, tke_w, rollout_detach)
            loss = loss + rollout_w * roll
            if rollout_stats is not None:
                rollout_stats["rollout"] = rollout_stats.get("rollout", 0.0) + roll.item() * x.shape[0]
        if teachers is not None:
            # DIST: the teachers see ``x`` AFTER input_noise / amp_jitter, so the
            # student is asked to match what the ensemble would have said about the
            # same augmented window, not about a clean one it never saw. Sliced to
            # the measured channels exactly as ``tgt`` is.
            t = _teacher_mean(teachers, x)[..., :measured]
            truth = loss
            dist = rel_l2(pred, t)
            if distill_tke and tke_w:
                dist = dist + tke_w * tke_rel_l2(pred, t)
            loss = truth_w * truth + distill_w * dist
            if distill_stats is not None:
                distill_stats["truth"] = distill_stats.get("truth", 0.0) + truth.item() * x.shape[0]
                distill_stats["distill"] = distill_stats.get("distill", 0.0) + dist.item() * x.shape[0]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # These fields are smooth but the warping layers can spike early;
        # clipping costs nothing and removes a class of NaN run.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += reported.item() * x.shape[0]
        n += x.shape[0]
    if distill_stats is not None and teachers is not None:
        for k in ("truth", "distill"):
            distill_stats[k] = distill_stats.get(k, 0.0) / max(n, 1)
    if rollout_stats is not None and rollout_w:
        rollout_stats["rollout"] = rollout_stats.get("rollout", 0.0) / max(n, 1)
    if es_stats is not None and es_m:
        es_stats["energy"] = es_stats.get("energy", 0.0) / max(n, 1)
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# ROLL (2026-09-24): TRAINING-TIME ROLLOUT CONSISTENCY. The model is trained to
# predict [20, 40) from [0, 20); here it is ALSO asked to predict [40, 60) from
# its own [20, 40), against the truth. Reached only under --train-rollout-weight
# > 0; at 0 the loader yields 2-tuples and none of this runs. Inference is
# untouched: the shipped module makes one pass, as always. NOT the same thing as
# --rollout-frames, which is an inference-time K-frames-per-pass mode inside
# FlowerS2S.forward and is refused together with this.
# ---------------------------------------------------------------------------

def rollout_input(pred: torch.Tensor, x: torch.Tensor, measured: int,
                  detach: bool = False) -> torch.Tensor:
    """Compose the second pass's input window from the first pass's prediction.

    ``pred`` is ``(B, t_out, H, W, c_out)`` physical, ``x`` the first pass's
    ``(B, t_in, H, W, c_in)`` input (after augmentation). Requires t_out == t_in,
    which FlowerS2S.forward enforces on the result. Channel by channel:

    * channels ``[0, measured)`` -- u, v (and p on sim) -- come from ``pred``;
    * channels ``[measured, c_in)`` are CARRIED OVER from x's LAST input frame,
      broadcast over the t_out frames. On real data that plane is either the
      identically-zero p the grader hands over, or under --mask-channel the
      measured mask (u != 0), which is near-static within a run (the body moves
      with angle of attack, not with time). The model's own channel-2 output is
      never used there: the loss never trains it, so it is noise.
    * every channel from ``pred`` is multiplied by the last input frame's
      measured mask (u != 0), so the PIV holes the first pass saw as zeros stay
      zeros in the second pass's input. Holes are a property of the sensor, not
      of time (the same convention the Lagrangian path and mask_channel use), and
      y2 carries them as zeros too. Without this the second pass would see an
      input distribution -- no holes -- that neither training nor the grader
      ever produces.

    Under --delta nothing extra is needed: predict_delta adds the residual to the
    last input frame INSIDE _one_pass, in normalised space, and returns physical
    units -- so ``pred`` is already the full field and x2 is fed like any input.

    ``detach`` cuts the graph at ``pred``: the rollout term then trains the
    second pass only and sends no gradient into the first pass.
    """
    if detach:
        pred = pred.detach()
    if pred.shape[-1] < measured or x.shape[-1] < measured:
        raise ValueError(f"rollout_input: measured={measured} exceeds pred {pred.shape[-1]} "
                         f"or x {x.shape[-1]} channels")
    t2 = pred.shape[1]
    last = x[:, -1:]                                          # (B, 1, H, W, c_in)
    meas = (last[..., :1] != 0).to(pred.dtype)                # (B, 1, H, W, 1)
    x2 = pred[..., :measured] * meas
    c_in = x.shape[-1]
    if c_in > measured:
        carry = last[..., measured:c_in].expand(-1, t2, -1, -1, -1)
        x2 = torch.cat((x2, carry), dim=-1)
    return x2


def rollout_term(model, out: torch.Tensor, x: torch.Tensor, y2: torch.Tensor,
                 measured: int, tke_w: float, detach: bool) -> torch.Tensor:
    """rel_l2(pred2, y2) + tke_w * tke_rel_l2(pred2, y2) on the measured channels,
    where pred2 is the model's second pass on ``rollout_input(out, x)``. Mirrors the
    first pass's slicing exactly; the primary-loss variants (--loss vrmse, --horizon-
    weight, --chan-weight, --speed-weight-power) are deliberately NOT mirrored -- the
    rollout term is always the plain shipping objective."""
    x2 = rollout_input(out, x, measured, detach)
    pred2 = model(x2)[..., :measured]
    tgt2 = y2[..., :measured]
    roll = rel_l2(pred2, tgt2)
    if tke_w:
        roll = roll + tke_w * tke_rel_l2(pred2, tgt2)
    return roll


@torch.no_grad()
def _eval_pass(predict, loader, device, measured, max_steps=None) -> dict:
    """Accumulate validation errors over ``loader``, one forward pass.

    ``predict`` maps x -> prediction, so the model and the persistence baseline
    are scored by exactly the same code and cannot drift apart.

    tke_rel_l2 does its own [u, v] slicing, so ``measured`` applies to rel-L2
    only -- TKE is a velocity quantity and never involves p.
    """
    total = {"rel_l2": 0.0, "tke": 0.0}
    n = 0
    for step, (x, y) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = predict(x)
        b = x.shape[0]
        total["rel_l2"] += rel_l2(pred[..., :measured], y[..., :measured]).item() * b
        total["tke"] += tke_rel_l2(pred, y).item() * b
        n += b
    return {k: v / max(n, 1) for k, v in total.items()}


def evaluate(model, loader, device, measured, max_steps=None) -> dict:
    """Validation errors for the model: {"rel_l2": ..., "tke": ...}."""
    model.eval()
    return _eval_pass(model, loader, device, measured, max_steps)


# ---------------------------------------------------------------------------
# DIST (2026-09-24): ENSEMBLE -> SINGLE-MODEL DISTILLATION AS A FINE-TUNE.
# Every shipped member costs ~0.24 ms of timed inference (~0.013 leaderboard
# points), so a k=1 or k=2 student that keeps most of the k=5 mean's accuracy
# buys time_score for nothing. Reached only under --teacher-ensemble; without it
# ``teachers`` is None everywhere and a run is bit-identical to before.
# ---------------------------------------------------------------------------

# Config fields on which a teacher MUST agree with the student. The first block is
# what the data path feeds and what the loss reads -- a teacher with a different
# c_in / mask_channel / t_in would be handed the wrong input and answer garbage
# without a warning, and a different c_out / t_out / rans would return a field the
# loss cannot be sliced against. rollout_frames is NOT refused (2026-09-25, rollout
# students): FlowerS2S.forward returns (B, t_out, H, W, c_out) whatever the rollout
# setting, so a one-pass teacher's field lines up frame-for-frame with a rollout
# student's, and the teachers never ship (only students reach build_submission). predict_delta, lifting_dim and
# n_levels are refused too although FlowerS2S.forward already returns physical
# units for any of them: the student is meant to REPLACE the teachers in the
# shipped ensemble path, and build_submission._assert_members_agree refuses
# exactly these keys, so a mismatch here would only be discovered at build time.
# (A bigger-teacher/smaller-student distillation is a legitimate future use; it
# needs only lifting_dim / n_levels dropped from this tuple.) Everything else --
# window_norm, lagrangian, boundary padding, dropout -- lives inside the
# teacher's own forward and travels with its checkpoint, so it is harmless.
_TEACHER_MUST_MATCH = ("c_in", "c_out", "t_in", "t_out", "mask_channel", "rans",
                       "predict_delta", "lifting_dim", "n_levels")


def load_teachers(paths, student_cfg, device) -> list:
    """Load frozen fp32 teachers: eval mode, no grad, on ``device``.

    Each teacher keeps the normalisation statistics its checkpoint carries --
    they are part of what that model IS, and re-fitting them on the student's
    split would change every teacher prediction. from_checkpoint loads them with
    the state_dict, so nothing here touches set_stats.
    """
    teachers = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise SystemExit(f"--teacher-ensemble: no such checkpoint {p}")
        t = FlowerS2S.from_checkpoint(p)
        bad = [f"{k}: teacher {getattr(t.cfg, k)!r} vs student {getattr(student_cfg, k)!r}"
               for k in _TEACHER_MUST_MATCH if getattr(t.cfg, k) != getattr(student_cfg, k)]
        if bad:
            raise SystemExit(f"--teacher-ensemble: {p} disagrees with the student's config on "
                             f"fields the data path or the shipped ensemble depends on:\n  "
                             + "\n  ".join(bad))
        t = t.to(device).float().eval()
        for q in t.parameters():
            q.requires_grad_(False)
        teachers.append(t)
    return teachers


@torch.no_grad()
def _teacher_mean(teachers, x):
    """The ensemble's prediction: members averaged in OUTPUT (physical) space, the
    same mean build_submission ships and evaluate_ens scores. Full channels;
    the caller slices to the measured ones."""
    return torch.stack([t(x) for t in teachers]).mean(0)


# ---------------------------------------------------------------------------
# E3 / L8: CO-TRAINED MEMBERS WITH AN EXPLICIT DIVERSITY OBJECTIVE.
# Everything below is reached only when --co-members is given; with the flag at
# its default of 0 not one line of it runs and a run is bit-identical to before.
# ---------------------------------------------------------------------------

def jitter_member(mdl, seed: int, scale: float) -> float:
    """Break the symmetry between co-trained members. Returns the relative size applied.

    Under --init-from every member starts from the SAME pretrained weights, and
    co-training means every member sees the SAME batch in the same order -- so
    without this they stay bit-identical forever. That is not a near-miss: the
    diversity gradient of ``ensemble_ambiguity`` is exactly zero when all members
    agree, so identical members are a FIXED POINT of the co-trained objective,
    not an unstable one. A diversity term with nothing to act on is a control
    that looks like a treatment.

    Independent seed members differ only by their shuffle order, and they end up
    a long way apart: measured on cc_allsim_ftk008_cov04_s1/s2/s3, pairwise
    relative weight distance 0.29-0.31, against 0.46 for the whole fine-tune's
    move away from the pretrain. This is the co-trained equivalent of that
    shuffle: each weight tensor is perturbed by ``scale`` times its OWN std, so
    the perturbation means the same thing in every layer. Buffers -- the
    normalisation mean and std -- are deliberately untouched: they are fitted
    statistics of the split, not parameters, and perturbing them would make the
    members disagree about what the data IS rather than about what it does.
    """
    g = torch.Generator().manual_seed(int(seed))
    moved_sq = 0.0
    total_sq = 0.0
    with torch.no_grad():
        for q in mdl.parameters():
            if q.numel() < 2:
                continue
            sd = q.detach().float().std()
            if not torch.isfinite(sd) or float(sd) == 0.0:
                continue
            d = (torch.randn(q.shape, generator=g) * (scale * float(sd))).to(q.device, q.dtype)
            moved_sq += float((d.float() ** 2).sum())
            total_sq += float((q.detach().float() ** 2).sum())
            q.add_(d)
    return (moved_sq / total_sq) ** 0.5 if total_sq > 0 else 0.0


def run_epoch_co(members, loader, device, measured, optimizer, scheduler=None,
                 max_steps=None, tke_w=0.0, cov_w=0.0, cov_half=None,
                 co_lambda=0.0, co_kind="var", bootstrap=False, co_energy_w=0.0) -> dict:
    """One co-training pass over M members that SHARE every batch.

    Each member pays exactly the objective a solo run pays -- rel_l2
    + tke_w * tke + cov_w * coverage -- and the member losses are SUMMED, not
    averaged, so at ``co_lambda = 0`` every member receives the gradient a solo
    run would give it (AdamW's per-parameter scaling would mostly hide an
    averaging factor, but "mostly" is not a control). Gradient clipping is also
    applied per member, at the same 1.0, for the same reason. The coupling is

        total = sum_m L_m  -  co_lambda * M * ensemble_ambiguity(preds, tgt)

    The factor M makes ``co_lambda`` independent of the member count: with it,
    the diversity gradient on one member has ``co_lambda`` times the magnitude
    of that member's own accuracy gradient (see ``ensemble_ambiguity``), so
    co_lambda = 0.3 means "push the members apart with 30% of the force that
    pushes them toward the data", whatever M is.

    ``bootstrap`` resamples the batch WITH REPLACEMENT, independently per member,
    for that member's OWN loss only -- the diversity term always sees every
    member on the whole batch. It is off by default. It exists because the
    jitter is the only thing decorrelating the members otherwise, and bagging is
    the textbook way to give each member an independent gradient-noise stream;
    the sample COUNT is unchanged (B draws from B), so the gradient scale and
    the step count are the recipe's, not a smaller batch's.

    ``co_energy_w`` (BET A, 2026-09-25; --co-energy-weight, default 0 = off and
    not one extra op runs): adds ``co_energy_w * energy_score(preds, tgt)`` -- the
    RELATIVE FAIR energy score with the M members as the M samples, on the measured
    channels, over the WHOLE batch (like the diversity term, never bootstrapped).
    The per-member point losses stay as they are; ES is added on top. Unlike the
    ambiguity reward, ES is a proper scoring rule: its spread term is bounded by
    its accuracy term (triangle inequality), so it rewards spread only where the
    truth is uncertain. It is NOT multiplied by M: per member, ES's accuracy and
    spread gradients are each ~``co_energy_w / M`` of that member's own rel-L2 push.
    parse_args refuses it together with --co-lambda (one ensemble term at a time).
    When on, the returned dict also has ``energy`` (the epoch mean of ES).

    Returns the epoch means of: the members' own rel-L2 (``rel_l2``, so
    train_rel_l2 keeps meaning what it means everywhere else -- a member's
    error), the rel-L2 of the members' MEAN prediction (``ens_rel_l2``), and the
    diversity term itself (``ambiguity``), which is the only way to tell a
    penalty that moved the members from one that never did.
    """
    for mdl in members:
        mdl.train()
    m = len(members)
    tot = {"rel_l2": 0.0, "ens_rel_l2": 0.0, "ambiguity": 0.0}
    if co_energy_w:
        tot["energy"] = 0.0
    n = 0
    for step, (x, y) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        tgt = y[..., :measured]
        preds = torch.stack([mdl(x)[..., :measured] for mdl in members])       # (M, B, ...)
        loss = 0.0
        reported = 0.0
        for k in range(m):
            p_own, t_own = preds[k], tgt
            if bootstrap:
                idx = torch.randint(0, x.shape[0], (x.shape[0],), device=preds.device)
                p_own, t_own = preds[k][idx], tgt[idx]
            lm = rel_l2(p_own, t_own)
            reported = reported + lm.detach()
            if tke_w:
                lm = lm + tke_w * tke_rel_l2(p_own, t_own)
            if cov_w:
                lm = lm + cov_w * coverage_surrogate(p_own, t_own, cov_half)
            loss = loss + lm
        amb = ensemble_ambiguity(preds, tgt, weight_by_error=(co_kind == "err"))
        if co_lambda:
            loss = loss - co_lambda * m * amb
        if co_energy_w:
            ces = energy_score(preds, tgt)
            loss = loss + co_energy_w * ces
        with torch.no_grad():
            ens = rel_l2(preds.mean(0), tgt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for mdl in members:
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        b = x.shape[0]
        tot["rel_l2"] += float(reported) / m * b
        tot["ens_rel_l2"] += ens.item() * b
        tot["ambiguity"] += amb.item() * b
        if co_energy_w:
            tot["energy"] += ces.item() * b
        n += b
    return {k: v / max(n, 1) for k, v in tot.items()}


@torch.no_grad()
def evaluate_ens(members, loader, device, measured, max_steps=None) -> dict:
    """Validation for a co-trained set, in ONE pass over the loader.

    ``rel_l2`` and ``tke`` are the ENSEMBLE MEAN's -- the ensemble is what ships,
    so it is what selects the checkpoints and what the composite estimate is
    computed from. ``member_rel_l2`` / ``member_tke`` are the mean over members
    of each member's own error, which is the number comparable to an ordinary
    run's val_rel_l2. ``ambiguity`` is the validation-set disagreement.

    NOTE for anyone reading summary.csv: a co-trained run's ``val_rel_l2`` is an
    ENSEMBLE number and is NOT comparable to a single-model run's. The member
    mean is, and it is in the log line as ``val_member_rel_l2``.
    """
    for mdl in members:
        mdl.eval()
    tot = {"rel_l2": 0.0, "tke": 0.0, "member_rel_l2": 0.0, "member_tke": 0.0,
           "ambiguity": 0.0}
    n = 0
    for step, (x, y) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = torch.stack([mdl(x) for mdl in members])
        mean = preds.mean(0)
        b = x.shape[0]
        tot["rel_l2"] += rel_l2(mean[..., :measured], y[..., :measured]).item() * b
        tot["tke"] += tke_rel_l2(mean, y).item() * b
        tot["member_rel_l2"] += sum(rel_l2(q[..., :measured], y[..., :measured]).item()
                                    for q in preds) / len(members) * b
        tot["member_tke"] += sum(tke_rel_l2(q, y).item() for q in preds) / len(members) * b
        tot["ambiguity"] += ensemble_ambiguity(preds[..., :measured],
                                               y[..., :measured]).item() * b
        n += b
    return {k: v / max(n, 1) for k, v in tot.items()}


def persistence_floor(loader, device, measured, t_out, max_steps=None) -> dict:
    """The same errors for repeating the last input frame.

    The number every val score should be read against. TKE's floor is near 1.0
    by construction: a repeated frame has zero fluctuation about its own time
    mean, so persistence predicts no turbulent energy at all.
    """
    return _eval_pass(lambda x: persistence(x, t_out), loader, device,
                      measured, max_steps)


def git_state() -> dict:
    """The commit this run is training at, or Nones if git cannot say.

    Never raises: a missing git, a missing repo or a detached worktree must not
    be able to abort a training run. ``dirty`` true means the tree carried
    uncommitted changes, so the commit alone does NOT reproduce the run -- which
    is exactly the thing worth knowing six weeks later.
    """
    def run(*args):
        try:
            out = subprocess.run(("git", "-C", str(REPO)) + args,
                                 capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    # --no-optional-locks: plain `git status` refreshes the index under .git/index.lock; on this
    # NFS repo it can outlive the 10 s timeout, and the killed process leaves the lock behind
    # (2026-09-25/26: stale 0-byte locks at every batch start blocked sync.sh and commits).
    status = run("--no-optional-locks", "status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def write_meta(run_dir, run_id, a, cfg, git, device, n_params, wb) -> None:
    """Record how this run was launched, in run_dir/meta.json.

    The JSONL says what happened per epoch; this says what produced it -- the
    commit, the exact argv, the interpreter and torch version, the W&B run it
    mirrors to. Enough to relaunch the run, and enough for
    scripts/summarize_runs.py to build the cross-run table without loading a
    checkpoint.

    A --resume appends a launch rather than replacing the file: one run
    directory can span several invocations, and the earlier ones are part of its
    history. Nothing here is ever overwritten.
    """
    path = run_dir / "meta.json"
    meta = {"run_id": run_id, "launches": []}
    if path.exists():
        try:
            meta = json.loads(path.read_text())
            meta.setdefault("launches", [])
        except (OSError, json.JSONDecodeError):
            meta = {"run_id": run_id, "launches": []}

    meta["launches"].append({
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "argv": sys.argv,
        "args": {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(a).items()},
        "config": asdict(cfg),
        "n_params": n_params,
        "device": str(device),
        "host": platform.node(),
        # How many runs shared this GPU, set by queue.sh --jobs. 1 or absent
        # means sole occupancy. Anything above 1 makes epoch_seconds and
        # `minutes` a property of the pack, not of the model -- never read a
        # timing across different pack sizes.
        "pack": int(os.environ.get("RPDE_PACK", "1")),
        # Set only under Slurm: the link from a run directory to its
        # outputs/logs/slurm/*-<job id>.out and to `sacct`.
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "wandb_id": wb.run.id if wb is not None else None,
        "wandb_url": wb.run.url if wb is not None else None,
    })
    path.write_text(json.dumps(meta, indent=2) + "\n")


def init_wandb(a, run_id, cfg, floor, measured, n_params, git=None,
               resume_id=None):
    """Open a Weights & Biases run, or return None if --wandb was not given.

    Imported lazily and only on demand: wandb is an optional extra, and nothing
    in the training path may depend on it being installed.

    The whole point of W&B over the JSONL is the searchable table of run
    configs, so everything that distinguishes this run goes into ``config`` --
    the architecture, the CLI arguments, the commit it was trained at, and the
    persistence floor the metrics have to be read against.

    Identity and grouping, since the runs table is what experiments are compared
    in: ``name`` is the run_id, which carries a UTC timestamp and so is unique
    even across two runs sharing a --name; ``group`` collects the arms of one
    experiment (pass the same --wandb-group to each); ``tags`` are for filtering
    across groups. The split is tagged automatically because sim and real
    val numbers are not comparable and must never be averaged together.
    """
    if not a.wandb:
        return None
    try:
        import wandb
    except ImportError:
        raise SystemExit(
            "--wandb needs the wandb package, which is not installed.\n"
            '  pip install "pde-competition[train]"   (or: pip install wandb)\n'
            "Then `wandb login`, or run with --wandb-mode offline to record "
            "locally and sync later.")

    # resume="must" rather than "allow": if the stored id is missing or wrong
    # we want a loud failure, not a silent second run split across two charts.
    extra = {"id": resume_id, "resume": "must"} if resume_id else {}
    git = git_state() if git is None else git
    tags = sorted({a.split, *(a.wandb_tags or []),
                   *(["smoke"] if a.smoke else []),
                   *(["dirty-tree"] if git["dirty"] else [])})
    wandb.init(project=a.wandb_project, name=run_id, mode=a.wandb_mode,
               group=a.wandb_group, tags=tags,
               config=asdict(cfg) | vars(a) | {"run_id": run_id,
                                               "persistence_floor": floor,
                                               "measured_channels": measured,
                                               "n_params": n_params,
                                               "git_commit": git["commit"],
                                               "git_branch": git["branch"],
                                               "git_dirty": git["dirty"]},
               **extra)
    return wandb



_BC = {"periodic": "PERIODIC", "border": "BORDER", "zeros": "ZEROS", "reflection": "REFLECTION"}


def _boundary_types(a):
    """The warping padding, per axis if asked for.

    The vendored sampler has ALWAYS taken padding_modes=(padding_h, padding_w) with each
    axis independently one of {zeros, border, reflection, periodic}, and it allows them to
    differ provided one is periodic. Every run this repo has ever done used a single value
    on BOTH axes, because --boundary is a single choice -- so the mixed configurations have
    never been tested, and the two axes are not alike: W = 64 is streamwise, where the flow
    convects OUT of the domain, and H = 32 is cross-stream free stream.

    That matters because instrumenting the sampler found 8.58% of warp coordinates land
    outside the grid (cross-stream 5.95%, streamwise 2.87%), and `periodic` answers those
    by wrapping -- so the far wake at x = 63 feeds the inflow at x = 0. W182 localised the
    tke residual to x = 40-44 of 64, downstream of the body.

    --boundary-hw takes exactly two values, H then W, and is refused unless one of them is
    periodic, which is the sampler's own constraint (flower_standalone.py:168).
    """
    if not getattr(a, "boundary_hw", None):
        return {"periodic": ("PERIODIC",), "border": ("BORDER",)}.get(a.boundary, ("ZEROS",))
    h, w = a.boundary_hw
    if h != w and "periodic" not in (h, w):
        raise SystemExit(f"--boundary-hw {h} {w}: when the axes differ one must be periodic "
                         f"(custom_grid_sample2d rejects mixed non-periodic padding)")
    print(f"  per-axis boundary: H(cross-stream)={h}  W(streamwise)={w}")
    return (_BC[h], _BC[w])


# ARCHITECTURE UNDER --init-from. The checkpoint defines the model, so every architecture
# flag on the command line used to be dropped without a word -- and a dropped flag makes the
# arm an unwitting control. It happened to --delta (fixed by the override below), and then,
# found 2026-09-24, to --rollout-frames (09-17 W87 K=10/K=5 arms, 09-23 AR1 cc_arft10, today's
# RINF: every checkpoint config says rollout_frames=0) and to --temporal-lift (09-21 W203 /
# A1: the adopted config says temporal_lift=0 and the non-strict load printed "tlift.weight
# left at its identity init" about a parameter the model did not have).
#
# "Given" means: differs from the parser's default. So the parcels' habit of passing
# --lifting-dim 128 --n-levels 2 --channels 3 --boundary periodic beside --init-from of the
# d128 pretrain stays accepted (they MATCH the checkpoint), while a flag set to a value the
# checkpoint does not have is refused. Blind spot, by construction: a flag typed at its
# default value (--rollout-frames 0 against a K=10 checkpoint) is indistinguishable from an
# absent one and is adopted from the checkpoint, as before.
#
# (CLI dest, FlowerConfig field). Each changes a parameter SHAPE (channels -> dim_in/dim_out,
# lifting_dim / n_levels -> every layer, num_heads -> the warp's flow conv, temporal_lift ->
# tlift.weight, rans / rollout_frames -> dim_out) or, for groups, the GroupNorm partition the
# affine weights were trained under.
_INIT_REFUSED = (("channels", "c_in"), ("channels", "c_out"), ("lifting_dim", "lifting_dim"),
                 ("n_levels", "n_levels"), ("num_heads", "num_heads"), ("groups", "groups"),
                 ("temporal_lift", "temporal_lift"), ("rans", "rans"),
                 ("rollout_frames", "rollout_frames"))
# Shape-free, NOT adopted (today's behaviour), but no longer silent: a differing value warns.
# --dropout left this list on 2026-09-25 (PNO_D probe): it is now an OVERRIDE, like --delta.
_INIT_WARNED = (("window_norm", "window_norm"),
                ("window_scale", "window_scale_mode"))


def _flag_given(a, parser, dest):
    """A flag counts as GIVEN when its value differs from the parser default (a flag typed at its default
    value is indistinguishable from an absent one -- the known blind spot of adopt_init_config)."""
    return getattr(a, dest) != parser.get_default(dest)


def adopt_init_config(saved: dict, a: argparse.Namespace,
                      parser: argparse.ArgumentParser) -> tuple[dict, bool]:
    """The FlowerConfig kwargs for a --init-from run, and whether to slice the output head.

    ``saved`` is the checkpoint's ``config`` dict. Starts from it; applies the shape-free
    overrides (--mask-channel, --lagrangian(-c), --delta, --dropout, and --temporal-lift K against a
    temporal_lift-0 checkpoint -- BET B, a parameter addition); REFUSES any shape-changing flag
    given with a value the checkpoint does not have (SystemExit naming flag, inherited and
    requested values); warns for the shape-free ones it ignores. The one sanctioned shape
    change is --rollout-frames K with --init-slice-output: rollout_frames becomes K and the
    caller must load through slice_output_state_dict. Pure: no torch, no file access.
    """
    saved = dict(saved)
    saved["resolution"] = tuple(saved["resolution"])
    saved["boundary_condition_types"] = tuple(saved.get("boundary_condition_types", ("PERIODIC",)))
    src = a.init_from

    refused = []
    slice_output = False
    add_tlift = False
    for dest, field in _INIT_REFUSED:
        if not _flag_given(a, parser, dest):
            continue
        want, have = getattr(a, dest), saved.get(field, FlowerConfig.__dataclass_fields__[field].default)
        if want == have:
            continue
        if dest == "rollout_frames" and a.init_slice_output:
            slice_output = True
            continue
        if dest == "temporal_lift" and not have and want > 0 and want % 2 == 1:
            # BET B (2026-09-25): ADDING a temporal lift to a checkpoint without one is the one
            # sanctioned parameter ADDITION. The new tlift.weight has no counterpart in the
            # checkpoint and stays at its identity init, so step 0 equals the checkpoint;
            # load_init_state_dict allows exactly that missing key and nothing else. A K that
            # differs from a NON-ZERO checkpoint value (a reshape, not an addition) stays refused.
            add_tlift = True
            continue
        flag = "--" + dest.replace("_", "-")
        refused.append(f"  {flag} {want}: requested {field}={want!r}, the checkpoint has {field}={have!r}")
    if refused:
        hint = ("\n  (--rollout-frames only: add --init-slice-output to start the rollout model from "
                "the checkpoint's first K output frames)" if any("--rollout-frames" in r for r in refused)
                else "")
        raise SystemExit(
            f"--init-from {src}: the checkpoint defines the architecture, and these flags ask for a "
            f"different one. Under --init-from they would be silently dropped (the arm an unwitting "
            f"control), so they are refused:\n" + "\n".join(refused) + hint
            + "\n  Drop the flag, or start from a checkpoint that has it (or train from scratch).")

    if add_tlift:
        if a.init_slice_output:
            raise SystemExit("--temporal-lift (added under --init-from) with --init-slice-output: "
                             "not supported together; pass one")
        saved["temporal_lift"] = a.temporal_lift
        print(f"  (--temporal-lift {a.temporal_lift} OVERRIDES the inherited temporal_lift=0: it ADDS "
              f"tlift.weight ({a.temporal_lift} params), which stays at its identity init, so step 0 "
              f"equals the checkpoint; the load allows exactly that missing key)")

    if a.init_slice_output:
        k, t_out, c_out = a.rollout_frames, saved["t_out"], saved["c_out"]
        have_k = saved.get("rollout_frames", 0)
        if not slice_output:
            raise SystemExit(f"--init-slice-output needs --rollout-frames K different from the "
                             f"checkpoint's rollout_frames={have_k}; got --rollout-frames {k}. "
                             f"Without a rollout change there is nothing to slice.")
        have_fpp = have_k or t_out
        if k <= 0 or t_out % k or k >= have_fpp:
            raise SystemExit(f"--init-slice-output --rollout-frames {k}: K must divide t_out={t_out} "
                             f"and be smaller than the checkpoint's {have_fpp} frames per pass")
        if saved.get("rans"):
            raise SystemExit("--init-slice-output with a rans checkpoint: its head ends in c_out "
                             "mean-field channels that a frame slice would cut off; not supported")
        saved["rollout_frames"] = k
        print(f"  (--rollout-frames {k} --init-slice-output: rollout_frames {have_k} -> {k}; the output "
              f"projection will be sliced to its first {k * c_out} of {have_fpp * c_out} rows)")

    for dest, field in _INIT_WARNED:
        have = saved.get(field, FlowerConfig.__dataclass_fields__[field].default)
        if _flag_given(a, parser, dest) and getattr(a, dest) != have:
            print(f"  WARNING: --{dest.replace('_', '-')} {getattr(a, dest)!r} IGNORED: the checkpoint "
                  f"has {field}={have!r} and --init-from keeps it")
    if _flag_given(a, parser, "boundary") or getattr(a, "boundary_hw", None):
        bc = _boundary_types(a)
        if tuple(bc) != saved["boundary_condition_types"]:
            print(f"  WARNING: boundary {bc} IGNORED: the checkpoint has "
                  f"boundary_condition_types={saved['boundary_condition_types']} and --init-from keeps it")

    if a.mask_channel:
        saved["mask_channel"] = True
        print("  (--mask-channel OVERRIDES the inherited config: it changes no parameter shape)")
    if a.lagrangian_c is not None:
        saved["lagrangian_c"] = a.lagrangian_c
        print(f"  (--lagrangian-c {a.lagrangian_c} OVERRIDES the inherited config)")
    if a.lagrangian != "off":
        # Same argument as --delta: the Lagrangian shear is a pre-processing step inside
        # _one_pass and changes no parameter shape, so the state_dict loads strict either way.
        saved["lagrangian"] = a.lagrangian
        print(f"  (--lagrangian {a.lagrangian} OVERRIDES the inherited config: it changes no parameter shape)")
    if a.delta and not saved.get("predict_delta"):
        # predict_delta changes no parameter shape -- it adds the last input frame to the
        # output inside _one_pass, so the state_dict loads strict either way. Without this
        # override --delta is silently dropped on any fine-tune and the arm is an unwitting
        # control.
        saved["predict_delta"] = True
        print("  (--delta OVERRIDES the inherited config: it changes no parameter shape)")
    if _flag_given(a, parser, "rollout_feed"):
        # ROLLOUT FEED (2026-09-26): shape-free -- which channels the rollout loop feeds back.
        have = saved.get("rollout_feed", 0)
        if a.rollout_feed != have:
            saved["rollout_feed"] = a.rollout_feed
            print(f"  (--rollout-feed {a.rollout_feed} OVERRIDES the inherited rollout_feed={have}: it "
                  f"changes no parameter shape)")
    if _flag_given(a, parser, "dropout"):
        # PNO_D (2026-09-25): an MC-dropout fine-tune of a dropout-free checkpoint needs this.
        # nn.Dropout and the nn.Identity it replaces have no parameters or buffers, so the
        # state_dict loads strict either way. Same blind spot as every flag here: --dropout 0
        # against a checkpoint trained WITH dropout equals the parser default, reads as
        # absent, and the checkpoint's rate is kept.
        have = saved.get("dropout_rate", FlowerConfig.__dataclass_fields__["dropout_rate"].default)
        if a.dropout != have:
            saved["dropout_rate"] = a.dropout
            print(f"  (--dropout {a.dropout} OVERRIDES the inherited dropout_rate={have}: it changes "
                  f"no parameter shape)")
    return saved, slice_output


def load_init_state_dict(model, state_dict: dict, added_tlift: bool) -> None:
    """Load --init-from weights into ``model``. STRICT, except in the one sanctioned case of
    BET B: the adopted config ADDED a temporal lift the checkpoint does not have. Then the
    load is non-strict and must report exactly the tlift module's keys as missing and nothing
    unexpected -- anything else is a SystemExit, not a warning -- and tlift is left at the
    identity init FlowerS2S.__init__ gave it, so the first forward equals the checkpoint's."""
    if not added_tlift:
        model.load_state_dict(state_dict)
        return
    want = sorted(k for k in model.state_dict() if k.startswith("tlift."))
    if not want:
        raise SystemExit("load_init_state_dict: added_tlift but the model has no tlift module")
    res = model.load_state_dict(state_dict, strict=False)
    if res.unexpected_keys or sorted(res.missing_keys) != want:
        raise SystemExit(f"--temporal-lift under --init-from: the non-strict load may miss ONLY "
                         f"{want}; it missed {sorted(res.missing_keys)} and found unexpected "
                         f"{sorted(res.unexpected_keys)}")
    k = model.cfg.temporal_lift
    w = model.tlift.weight.detach()
    ident = torch.zeros_like(w)
    ident[0, 0, k // 2] = 1.0
    if not torch.equal(w, ident):
        raise SystemExit("--temporal-lift under --init-from: tlift.weight is not the identity kernel")
    print(f"  loaded non-strict for the ADDED temporal lift: missing exactly {want}, nothing "
          f"unexpected; tlift (K={k}) left at its identity init, so step 0 equals the checkpoint")


def slice_output_state_dict(state_dict: dict, model_state: dict, k_frames: int, c_out: int,
                            t_out: int) -> dict:
    """``state_dict`` with the output projection cut to the first ``k_frames`` frames.

    FlowerS2S._one_pass reads the core's output as ``reshape(b, frames, c_out, h, w)``, so
    channel ``t * c_out + c`` is frame t, channel c: frame-major, and rows [0:k*c_out] of the
    last conv ARE the first k frames. The parameters to slice are identified by shape, not by
    name: exactly those whose checkpoint leading dim is t_out*c_out and whose model leading dim
    is k*c_out with every other dim equal, and they must all belong to one module (in FlowerS2S:
    core.project.2.weight and .bias). Any other key or shape difference is refused, so
    everything else loads strictly.
    """
    full, keep = t_out * c_out, k_frames * c_out
    if set(state_dict) != set(model_state):
        raise SystemExit(f"--init-slice-output: checkpoint and model disagree on keys: missing "
                         f"{sorted(set(model_state) - set(state_dict))}, unexpected "
                         f"{sorted(set(state_dict) - set(model_state))}")
    out, sliced, bad = dict(state_dict), [], []
    for name, v in state_dict.items():
        src, dst = tuple(v.shape), tuple(model_state[name].shape)
        if src == dst:
            continue
        if src and dst and src[0] == full and dst[0] == keep and src[1:] == dst[1:]:
            out[name] = v[:keep].clone()
            sliced.append((name, src, dst))
        else:
            bad.append(f"{name} {src} -> {dst}")
    if bad:
        raise SystemExit("--init-slice-output: shape differences other than the output frames: "
                         + "; ".join(bad))
    if not sliced or len({n.rsplit(".", 1)[0] for n, _, _ in sliced}) != 1:
        raise SystemExit(f"--init-slice-output: cannot identify ONE output layer to slice "
                         f"(candidates {[n for n, _, _ in sliced]})")
    for name, src, dst in sliced:
        print(f"  sliced {name}: {src} -> {dst} (rows 0:{keep} = frames 0:{k_frames} x {c_out} channels)")
    return out


def check_rollout_detach(a: argparse.Namespace, cfg: FlowerConfig) -> None:
    """--rollout-detach-passes needs a rollout in the FINAL config, not on the CLI: under
    --init-from rollout_frames comes from the checkpoint or from --rollout-frames K
    --init-slice-output (adopt_init_config), under --resume from the saved config. Without a
    rollout there are no fed-back frames to detach and the arm would be a silent control."""
    if getattr(a, "rollout_detach_passes", False) and not cfg.rollout_frames:
        raise SystemExit("--rollout-detach-passes detaches the frames fed back between rollout passes, "
                         f"but the model config has rollout_frames={cfg.rollout_frames}: there is no "
                         "rollout. Add --rollout-frames K (with --init-from: plus --init-slice-output).")


def main(a: argparse.Namespace) -> None:
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device(a.device)
    runs = list_runs(a.split)

    # ONE split function for this run, used at EVERY call site below -- p00 here, the
    # three PIV phase stores in the --mix-phases block, and sim in --mix-sim. The
    # 2026-09-23 leak was exactly a second call site that split differently from this
    # one (the phase stores took split_by_condition's DEFAULT holdout instead of the
    # run's), so the fix is structural: there is no second way to split.
    if a.gapped_holdout:
        # GTH: hold out the last --gth-holdout-frames frames of EVERY trajectory with
        # --gth-gap frames belonging to neither side. Every condition trains and every
        # condition validates, so the instrument spans all 18 Reynolds rows instead of 2
        # -- at the price of being in-sample in condition and out-of-sample only in time.
        # val_rel_l2 from this mode is NOT comparable to any condition-holdout run.
        if a.train_all:
            raise SystemExit("--train-all with --gapped-holdout would train on the held-out "
                             "TAIL as well, leaving no holdout at all. Under the gapped split "
                             "every condition already trains; drop --train-all.")
        if a.mix_sim:
            raise SystemExit("--mix-sim with --gapped-holdout is refused: sim trajectories are "
                             "1000 frames of a simulation with no frame-level correspondence to "
                             "the 868-frame PIV recording, so no temporal cut of sim can be shown "
                             "to exclude the validation interval. The condition guard that makes "
                             "--mix-sim honest does not exist in this mode.")
        if tuple(a.holdout_re) != (13950, 26700):
            raise SystemExit(f"--holdout-re {list(a.holdout_re)} with --gapped-holdout: the gapped "
                             f"split holds out no condition, so --holdout-re would be silently "
                             f"ignored. Composing the two splits is not implemented; pass one.")

        def _split(rs):
            return split_temporal_gapped(rs, holdout_frames=a.gth_holdout_frames,
                                         gap_frames=a.gth_gap)
        print(f"GAPPED TEMPORAL HOLDOUT (GTH): last {a.gth_holdout_frames} frames of every "
              f"trajectory are validation, the {a.gth_gap} frames before them belong to "
              f"NEITHER split, the rest trains. --holdout-re is not used; every condition "
              f"is in both splits. val_rel_l2 here is NOT comparable to a condition-holdout run.")
    else:
        def _split(rs):
            return split_by_condition(rs, holdout_re=tuple(a.holdout_re))

    train_runs, val_runs = _split(runs)
    if a.gapped_holdout:
        _t0, _v0 = train_runs[0], val_runs[0]
        print(f"  first trajectory: train frames [0, {_t0.n_frames}), gap "
              f"[{_t0.n_frames}, {_v0.frame_start}), val frames "
              f"[{_v0.frame_start}, {_v0.frame_start + _v0.n_frames}) "
              f"of {_v0.frame_start + _v0.n_frames}")
    if not val_runs:
        raise SystemExit("empty validation split -- check --split and holdout Re")
    if a.train_all:
        # THE FINAL-CHECKPOINT MODE. Every cell trains, including the two Re
        # rows normally held out -- the rows closest to the hidden set's
        # (15225 and 27975). Validation still runs on those rows but is
        # IN-SAMPLE from here on: it is a sanity check, not a selection
        # criterion. Choose the epoch from the held-out twin run and take it
        # from --save-epochs; fit the SPS widths on the twin as well
        # (fit_bounds.py --write-for), never on cells this model has seen.
        train_runs = list(runs)
        print("TRAIN-ALL: every cell trains; val below is IN-SAMPLE and must "
              "not select the checkpoint or fit the bounds.")
    if a.drop_train_re:
        # S1c: THIN THE TRAINING SET WITHOUT TOUCHING THE VALIDATION SET. --holdout-re moves
        # a row into val, which changes what the number means; this drops it from training
        # only, so models trained on different amounts of data are scored on exactly the
        # same held-out rows. That is the whole point: the question is how the val-optimal
        # EPOCH moves with the amount of training data, and it can only be asked if the val
        # set is identical across arms.
        drop = set(a.drop_train_re)
        overlap = drop & set(a.holdout_re)
        if overlap:
            raise SystemExit(f"--drop-train-re {sorted(overlap)} is already held out; it trains on neither path")
        before = len(train_runs)
        train_runs = [r for r in train_runs if r.re_nominal not in drop]
        if not train_runs:
            raise SystemExit("--drop-train-re dropped every training run")
        kept = sorted({r.re_nominal for r in train_runs})
        print(f"--drop-train-re {sorted(drop)}: {before} -> {len(train_runs)} train runs, "
              f"{len(kept)} rows {kept}")
    print(f"{len(runs)} runs -> {len(train_runs)} train / {len(val_runs)} val")
    print(f"held-out cells: {sorted(r.key for r in val_runs)}")

    # --resume continues THIS run: same directory, same schedule, restored
    # optimizer. --init-from starts a NEW run from someone else's weights.
    # Opposite meanings, so they are mutually exclusive.
    resume = None
    if a.resume:
        resume_path = Path(a.resume)
        if resume_path.is_dir():
            resume_path = resume_path / "last.pth"
        if not resume_path.exists():
            raise SystemExit(f"no checkpoint to resume from: {resume_path}")
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "guard" not in resume:
            raise SystemExit(
                f"{resume_path} predates --resume support (no training state). "
                f"Only its weights are usable, via --init-from.")
        # OneCycleLR bakes total_steps = epochs * steps_per_epoch at
        # construction, so any of these changing makes the restored scheduler
        # position meaningless. Refuse rather than silently retrain on a
        # different schedule.
        diffs = [f"    {k}: {v!r} -> {getattr(a, k)!r}"
                 for k, v in resume["guard"].items()
                 if k != "steps_per_epoch" and getattr(a, k) != v]
        if diffs:
            raise SystemExit("cannot resume: the run has changed.\n"
                             + "\n".join(diffs)
                             + "\nRe-run without --resume, or restore these values.")

    # The checkpoint defines the model, the CLI defines the run: with
    # --init-from the architecture comes from the saved config, so
    # load_state_dict cannot shape-mismatch, while split / lr / epochs / stride
    # stay under CLI control.
    if resume:
        saved = dict(resume["config"])
        saved["resolution"] = tuple(saved["resolution"])
        cfg = FlowerConfig(**saved)
        print(f"resuming {resume_path} from epoch {resume['epoch']}")
    elif a.init_from:
        ckpt = torch.load(a.init_from, map_location="cpu", weights_only=False)
        saved, slice_output = adopt_init_config(ckpt["config"], a, build_parser())
        cfg = FlowerConfig(**saved)
        print(f"adopted architecture from {a.init_from}: {cfg}")
        print("  (shape-changing flags that differ are refused above; shape-free ones are overridden or warned)")
    else:
        cfg = FlowerConfig(
            c_in=a.channels, c_out=a.channels,
            lifting_dim=a.lifting_dim, n_levels=a.n_levels,
            num_heads=a.num_heads, groups=a.groups,
            dropout_rate=a.dropout, predict_delta=a.delta, lagrangian=a.lagrangian, mask_channel=a.mask_channel,
            temporal_lift=a.temporal_lift, rans=a.rans,
            **({} if a.lagrangian_c is None else {'lagrangian_c': a.lagrangian_c}),
            window_norm=a.window_norm,
            window_scale_mode=a.window_scale,
            rollout_frames=a.rollout_frames, rollout_feed=a.rollout_feed,
            boundary_condition_types=_boundary_types(a),
        )

    check_rollout_detach(a, cfg)
    if a.energy_score_samples and cfg.dropout_rate <= 0:
        # Refused before any data is loaded; check_energy_score_model repeats it on the
        # built model (and run_epoch on every epoch) as the second layer.
        raise SystemExit(f"--energy-score-samples {a.energy_score_samples} needs dropout: the config "
                         f"has dropout_rate={cfg.dropout_rate}. Pass --dropout P > 0 (under "
                         f"--init-from it overrides the checkpoint's value).")

    # Optional second training domain. Two-stage pretraining failed by
    # catastrophic forgetting -- the LR probe got monotonically better the more
    # thoroughly warm-up erased the pretrained weights -- so here the model
    # never stops seeing sim, and there is nothing to forget.
    #
    # VALIDATION STAYS REAL-ONLY. The leaderboard scores real, and a mixed val
    # set would make val_rel_l2 incomparable with every run already in
    # outputs/runs/ and with score_local.py. Sim contributes training windows
    # and nothing else.
    mix_runs = []
    if a.mix_phases > 0:
        # THE FOUR PIV PHASES. Raw PIV is 64x128; Track 1 strides by 2 taking offset (0,0),
        # which is the only thing train_real has ever held and the only thing the grader
        # scores. The other three offsets are the SAME flow sampled on a half-pixel-shifted
        # grid, so they are 3x more REAL data -- and real rows are the scarce resource, 17
        # of them against an unlimited sim.
        # THE TRAP, and it is why this is a mix and not a plain --split. The airfoil sits at
        # a different sub-pixel position on each offset grid, so a model trained only on all
        # four learns a BLURRED body position, and the wake anchored to that body is exactly
        # where rel_l2 and tke are won. The curriculum is: this stage for representation,
        # then a FINAL fine-tune on p00 alone to re-anchor the geometry.
        # THE LEAK THIS GUARDS, which is the part that would have invalidated everything:
        # the phase stores contain the SAME trajectories, including the held-out rows. Adding
        # them raw would put row 26700 into training at a half-pixel shift -- essentially the
        # validation flow itself. split_by_condition is applied to every phase and only the
        # TRAINING cells are kept, exactly as --mix-sim does for sim.
        if a.split != "train_real":
            raise SystemExit(f"--mix-phases adds the shifted-grid copies of train_real; "
                             f"--split is {a.split!r}. Drop one of them.")
        for ph in ("train_real_p01", "train_real_p10", "train_real_p11"):
            # THE SPLIT MUST BE THIS RUN'S, at every call site. Passing nothing here silently
            # held out split_by_condition's DEFAULT (13950, 26700) from the phase stores while
            # p00 held out --holdout-re's rows, so any run with a non-default holdout trained on
            # half-pixel-shifted copies of its OWN validation flow. Found 2026-09-23 after it had
            # inflated every HO2 swap; the shipped models use the default and were never affected.
            # _split is the run's one split function, so this cannot drift from p00 again.
            # Under GTH the same argument applies FRAME-WISE: the phase stores are the same
            # trajectories on a shifted grid, frame i for frame i, so the held-out TAIL must be
            # dropped from them too or the model trains on a half-pixel-shifted copy of its own
            # validation frames. The n_frames check below is what makes "frame i for frame i" a
            # fact rather than an assumption.
            _ph_all = list_runs(ph)
            if a.gapped_holdout:
                _ref = {r.key: r.n_frames for r in runs}
                _bad = {r.key: (r.n_frames, _ref.get(r.key)) for r in _ph_all
                        if _ref.get(r.key) != r.n_frames}
                if _bad:
                    raise SystemExit(
                        f"--gapped-holdout --mix-phases: {ph} disagrees with {a.split} on "
                        f"trajectory length for {sorted(_bad)[:5]}: (phase, p00) = "
                        f"{[_bad[k] for k in sorted(_bad)[:5]]}. A temporal holdout is a FRAME "
                        f"INDEX, so a store of a different length holds out a different "
                        f"interval and the tail leaks in shifted. Rebuild the phase store.")
            ph_runs, _ph_held = _split(_ph_all)
            if not ph_runs:
                raise SystemExit(f"--mix-phases: {ph} has no training cells. Build it with "
                                 f"scripts/subsample_data.py --phase {ph[-3:]} --real-only")
            mix_runs += ph_runs
        print(f"phase mixing: {len(mix_runs)} extra trajectories from p01/p10/p11 "
              f"at {a.mix_phases:.0%} of each epoch (held-out cells removed from all three)")
    if a.mix_sim > 0:
        if a.split != "train_real":
            raise SystemExit(f"--mix-sim adds sim to a REAL run; --split is "
                             f"{a.split!r}. Drop one of them.")
        # _split holds out the same Re rows here as for real, and both rows exist
        # in both splits -- so the held-out conditions are withheld from the sim
        # half too. Without this the val cells would leak in through the sim data.
        # (Unreachable under --gapped-holdout: that mode refuses --mix-sim above,
        # because sim has no frame-level correspondence to the PIV recording.)
        mix_runs, _mix_held = _split(list_runs("train_sim"))
        if not cfg.window_norm:
            print("WARNING: --mix-sim without window_norm. real/sim u std "
                  "0.1137 vs 0.4436 (3.9x) and v 0.0159 vs 0.2148 (13.5x), so "
                  "one global normalizer fits neither domain.")

    # Score only what the leaderboard scores. Real data measures [u, v] and the
    # scorer drops p; sim measures all three. Mixed, the common width is real's.
    store_ch = 2 if mix_runs else measured_channels(runs)
    measured = min(store_ch, cfg.c_out)

    # coverage half: one constant per channel, or a per-element MAP from a --spatial bounds.json
    cov_half = None
    if a.coverage_weight:
        if a.coverage_half_map:
            _b = json.load(open(a.coverage_half_map))
            if 'half_width_map' not in _b:
                raise SystemExit(f"{a.coverage_half_map} has no 'half_width_map' -- fit it with fit_bounds.py --spatial")
            # the map is written with all C channels; the loss sees only the MEASURED ones,
            # exactly as the constant path is sliced by a.coverage_half[:measured]
            _m = np.asarray(_b['half_width_map'], np.float32)[..., :measured]
            cov_half = torch.from_numpy(np.ascontiguousarray(_m)).to(device)
            print(f"coverage half: per-element map {tuple(_m.shape)} (sliced to {measured} measured channels) "
                  f"from {a.coverage_half_map} "
                  f"(mean {_m.mean():.5f}, min {_m.min():.5f}, max {_m.max():.5f})", flush=True)
        else:
            cov_half = torch.tensor(a.coverage_half[:measured], device=device)

    print(f"optimizing {measured} of {cfg.c_out} output channels "
          f"(store has {store_ch})")

    # Measured on wst004 during a fine-tune: GPU 20-21% busy, 1.0 of 24 GB used,
    # train.py at 1323% CPU across 62 threads. The GPU is not the constraint --
    # the batches are. Workers feed it in parallel; pinned host memory makes the
    # copy asynchronous. The store is RAM-resident and Linux forks copy-on-write,
    # so workers share it rather than duplicating it.
    pin = (device.type == "cuda")
    if a.threads:
        # That 1323% is largely torch's intra-op pool spinning on tiny 32x64
        # kernels, where the threads cost more than they save. Capping it leaves
        # cores for the workers.
        torch.set_num_threads(a.threads)

    mask_bank = None
    if a.mask_from_real:
        from pde_competition.dataset import real_mask_bank
        mask_bank = real_mask_bank()
        frac = float(mask_bank.mean())
        print(f"--mask-from-real: {len(mask_bank)} real PIV masks, measured fraction "
              f"{frac:.3f} (so ~{1 - frac:.1%} of each window is holed), applied to the TRAINING windows only",
              flush=True)
    # ROLL: 0 for every ordinary run. With --train-rollout-weight the TRAINING
    # windows -- p00 AND the mix stores, which share one loader -- are cut
    # t_out frames longer and yield (x, y, y2). Validation windows are unchanged:
    # val_rel_l2 keeps meaning what it means everywhere else. Checked against cfg,
    # not only the CLI, because under --init-from rollout_frames / rans come from
    # the checkpoint; and the second pass feeds t_out frames back in as t_in.
    extra_out = 0
    if a.train_rollout_weight > 0:
        if cfg.rollout_frames or cfg.rans:
            raise SystemExit(f"--train-rollout-weight with a model whose config has "
                             f"rollout_frames={cfg.rollout_frames} / rans={cfg.rans}: not "
                             f"supported (inherited from --init-from?)")
        if cfg.t_in != cfg.t_out:
            raise SystemExit(f"--train-rollout-weight needs t_in == t_out to feed the prediction "
                             f"back as the next window, got {cfg.t_in}/{cfg.t_out}")
        extra_out = cfg.t_out
        print(f"TRAIN-ROLLOUT: loss += {a.train_rollout_weight} * (rel_l2 + {a.tke_weight} * tke) on a "
              f"second pass fed the first pass's own prediction"
              f"{' (DETACHED: no gradient into the first pass)' if a.train_rollout_detach else ''}; "
              f"training windows are {cfg.t_in + 2 * cfg.t_out} frames, so slightly fewer of them.")
    train_ds = WindowDataset(train_runs, stride=a.stride, channels=cfg.c_in,
                             mask_bank=mask_bank, mask_seed=a.seed, mask_channel=cfg.mask_channel,
                             extra_out=extra_out)
    val_ds = WindowDataset(val_runs, stride=a.val_stride, channels=cfg.c_in, mask_channel=cfg.mask_channel)
    sim_ds = (WindowDataset(mix_runs, stride=a.stride, channels=cfg.c_in, extra_out=extra_out)
              if mix_runs else None)
    print(f"windows: {len(train_ds)} train / {len(val_ds)} val "
          f"(stride {a.stride}/{a.val_stride})")
    if sim_ds is not None:
        print(f"mixing in {len(sim_ds)} sim windows from {len(mix_runs)} cells "
              f"at {a.mix_sim:.0%} of each epoch")

    # Fit the normalizer on training runs only, then hand it to the model so it
    # lands in the checkpoint. window_norm comes from cfg, never from the CLI:
    # under --init-from the architecture is the checkpoint's, and statistics
    # fitted in the wrong space would be silently wrong rather than loud.
    mean, std = channel_stats(train_runs + mix_runs, channels=cfg.c_in,
                              window_norm=cfg.window_norm,
                              window_scale_mode=cfg.window_scale_mode)
    print(f"channel mean {np.round(mean, 6).tolist()}  std {np.round(std, 6).tolist()}")

    model = FlowerS2S(cfg)
    # Truncated backprop through the rollout: a model ATTRIBUTE, not a config field, so the
    # checkpoint config (and the shipped module built from it) never sees it. Set before any
    # copy.deepcopy of the model below so co-members / copies inherit it. No-grad eval is
    # bit-identical either way.
    model.rollout_detach = bool(a.rollout_detach_passes)
    if model.rollout_detach:
        print(f"ROLLOUT-DETACH: the {cfg.rollout_frames} frames fed back after each of the "
              f"{cfg.t_out // cfg.rollout_frames} passes are detached during training (truncated "
              f"backprop: each pass trains on its own loss, no gradient into earlier passes)")
    if resume:
        # Stats are buffers inside the state_dict and are already correct for
        # this split -- do NOT re-fit. (--init-from does re-fit, deliberately:
        # it changes split.)
        model.load_state_dict(resume["state_dict"])
    elif a.init_from:
        # Weights from the checkpoint, statistics from THIS split. A sim
        # checkpoint carries sim's normalization; fine-tuning real data in that
        # space does not crash, it just trains badly -- so the fresh fit has to
        # land after the load, not before.
        # STRICT. adopt_init_config refuses every flag that would make this model's parameters
        # differ from the checkpoint's. The load used to be non-strict under --temporal-lift and
        # print "tlift.weight left at its identity init" -- but cfg came from the checkpoint
        # (temporal_lift=0), so there was no tlift at all and W203 / A1's arms were controls.
        # The one sanctioned difference is the sliced output head of --init-slice-output.
        # BET B (2026-09-25): the second sanctioned difference is a temporal lift ADDED to a
        # temporal_lift-0 checkpoint (adopt_init_config); load_init_state_dict then allows exactly
        # the missing tlift key and leaves it at identity. Every other load stays strict.
        sd = ckpt["state_dict"]
        if slice_output:
            sd = slice_output_state_dict(sd, model.state_dict(), cfg.rollout_frames, cfg.c_out, cfg.t_out)
        added_tlift = bool(cfg.temporal_lift) and not ckpt["config"].get("temporal_lift", 0)
        load_init_state_dict(model, sd, added_tlift)
        print(f"loaded weights from {a.init_from} "
              f"(epoch {ckpt.get('epoch', '?')}, "
              f"val_rel_l2 {ckpt.get('val_rel_l2', float('nan')):.4f})")
    if not resume:
        model.set_stats(mean, std)

    if a.freeze_encoder:
        # Transfer without letting the fine-tune un-learn what was pretrained.
        # The LR probe showed the opposite of what you'd hope: results improved
        # monotonically with max_lr (1e-4 -> 3e-4 -> 1e-3), i.e. the more the
        # pretrained weights were erased during OneCycle warm-up the better the
        # outcome. Holding the encoder fixed removes that failure mode -- only
        # the decoder and projection adapt.
        if not a.init_from:
            raise SystemExit("--freeze-encoder is meaningless without --init-from: "
                             "it would freeze randomly initialised weights")
        frozen = 0
        for mod in (model.core.lift, model.core.encoder_blocks, model.core.bottleneck):
            for p in mod.parameters():
                p.requires_grad_(False)
                frozen += p.numel()
        total = model.n_params()
        print(f"froze lift + encoder + bottleneck: {frozen:,} of {total:,} params "
              f"({frozen / total:.0%}); training decoder + projection only")

    model.to(device)

    print(f"Flower on {device}: {model.n_params():,} params "
          f"({model.n_params() * 4 / 1e6:.1f} MB fp32), dim_in {cfg.dim_in} -> "
          f"dim_out {cfg.dim_out}")
    if a.energy_score_samples:
        n_drop = check_energy_score_model(model, a.energy_score_samples)
        print(f"ENERGY SCORE (PNO_D): {a.energy_score_samples} MC-dropout samples per batch "
              f"({n_drop} nn.Dropout modules at p={cfg.dropout_rate}); loss = "
              + (f"{a.energy_score_weight} * ES" if a.energy_score_replace
                 else f"<primary point term> + {a.energy_score_weight} * ES")
              + " + the other point terms on the SAMPLE MEAN. Each step costs "
              f"~{a.energy_score_samples}x a plain one (forward and backward).")

    # DIST. None for every ordinary run. Loaded after model.to(device) so the
    # teachers share the student's device, and never added to ``trainable`` below
    # (that list is built from ``model`` alone), so AdamW cannot touch them.
    teachers = None
    if a.teacher_ensemble:
        teachers = load_teachers(a.teacher_ensemble, cfg, device)
        print(f"DISTILLING from {len(teachers)} frozen teachers "
              f"({teachers[0].n_params():,} params each): loss = {a.distill_truth_weight} * truth "
              f"+ {a.distill_weight} * rel_l2(pred, teacher mean)"
              + (f" [+ {a.distill_weight} * {a.tke_weight} * tke_rel_l2(pred, teacher mean)]"
                 if a.distill_tke else ""))
        for p_ in a.teacher_ensemble:
            print(f"  teacher {p_}")

    # E3 / L8. ``members`` is None for every ordinary run and nothing below
    # changes; with --co-members it is the list of M models trained together.
    # Member 0 IS ``model``, so the normalisation statistics, the --init-from
    # weights and --freeze-encoder are inherited by every member by
    # construction rather than by a second copy of that code.
    members = None
    if a.co_members:
        members = [model] + [copy.deepcopy(model) for _ in range(a.co_members - 1)]
        # Every member is jittered, member 0 included: no member may be the
        # "real" one. Own generator per member, seeded from --seed, so the
        # global RNG stream -- and therefore the batch order -- is identical to
        # the solo run with the same seed.
        moved = [jitter_member(mdl, a.seed * 1000 + k, a.co_jitter)
                 for k, mdl in enumerate(members)]
        print(f"CO-TRAINING {a.co_members} members in one process, "
              f"lambda {a.co_lambda} on the {a.co_diversity!r} diversity term"
              f"{', bootstrapped batches' if a.co_bootstrap else ''}"
              + (f"; + {a.co_energy_weight} * relative fair ENERGY SCORE over the "
                 f"{a.co_members} members (BET A)" if a.co_energy_weight else "") + ".")
        print(f"  init jitter {a.co_jitter}: members moved "
              f"{', '.join(f'{v:.4f}' for v in moved)} relative from the shared init "
              f"(independent seed members end 0.29-0.31 apart).")
        print("  val_rel_l2 for this run is the ENSEMBLE MEAN's and is NOT comparable "
              "to a single-model run; val_member_rel_l2 in the log line is.")

    if sim_ds is None:
        train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                              num_workers=a.workers, drop_last=True,
                              pin_memory=pin, persistent_workers=a.workers > 0)
    else:
        # num_samples = len(train_ds) keeps the epoch the same size as a
        # real-only run, so "30 epochs" means the same compute as every run
        # already recorded. Letting the epoch grow with the added data would
        # confound the comparison with a budget change.
        _frac = a.mix_phases if a.mix_phases > 0 else a.mix_sim
        w = ([(1 - _frac) / len(train_ds)] * len(train_ds)
             + [_frac / len(sim_ds)] * len(sim_ds))
        # D1p (2026-09-22, Simon-approved): with num_samples = len(train_ds) the real p00 windows
        # get only (1 - frac) of each epoch -- a quarter at frac 0.75 -- which is one reason the mix
        # needed 40 epochs. --mix-grow-epoch scales the epoch by 1 / (1 - frac) so the EXPECTED
        # number of p00 draws per epoch equals a real-only epoch and the phases come on top. It is a
        # budget change (OneCycle's total_steps grows with it), stated as such in the parcel.
        n_samples = len(train_ds)
        if a.mix_grow_epoch:
            if not 0.0 < _frac < 1.0:
                raise SystemExit("--mix-grow-epoch needs a mix fraction strictly between 0 and 1")
            n_samples = int(round(len(train_ds) / (1.0 - _frac)))
            print(f"--mix-grow-epoch: epoch grows {len(train_ds)} -> {n_samples} samples "
                  f"(p00 exposure kept at frac {_frac:g})")
        sampler = WeightedRandomSampler(w, num_samples=n_samples,
                                        replacement=True)
        train_dl = DataLoader(ConcatDataset([train_ds, sim_ds]),
                              batch_size=a.batch_size, sampler=sampler,
                              num_workers=a.workers, drop_last=True,
                              pin_memory=pin, persistent_workers=a.workers > 0)
    # One loader per held-out Re row, so val is reported per row as well as
    # pooled. The pooled number is unchanged and stays the selection criterion;
    # the rows are the proxies for the hidden set's interpolation (13950 ->
    # 15225) and extrapolation (26700 -> 27975) regimes, and most of the
    # hidden error is expected at the top row.
    val_rows = sorted({r.re_nominal for r in val_runs})
    val_dl_by_re = {re: DataLoader(WindowDataset([r for r in val_runs if r.re_nominal == re],
                                                 stride=a.val_stride, channels=cfg.c_in),
                                   batch_size=a.batch_size, shuffle=False)
                    for re in val_rows} if len(val_rows) > 1 else {}
    val_dl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, pin_memory=pin,
                        persistent_workers=a.workers > 0)

    # Frozen parameters must not reach the optimizer: AdamW would still apply
    # weight decay to them, so they would drift despite requires_grad=False.
    trainable = [p for p in model.parameters() if p.requires_grad]
    if members is not None:
        trainable = [q for mdl in members for q in mdl.parameters() if q.requires_grad]
    max_lr = a.lr
    if a.llrd < 1.0:
        # Layer-wise LR decay (fresh-eyes proposition 9): lift + encoder + bottleneck at lr * llrd,
        # decoder + projection at lr -- the soft version of --freeze-encoder.
        enc_ids = {id(q) for mod in (model.core.lift, model.core.encoder_blocks, model.core.bottleneck)
                   for q in mod.parameters()}
        enc = [q for q in trainable if id(q) in enc_ids]; rest = [q for q in trainable if id(q) not in enc_ids]
        optimizer = torch.optim.AdamW([{"params": enc, "lr": a.lr * a.llrd}, {"params": rest, "lr": a.lr}],
                                      lr=a.lr, weight_decay=a.weight_decay)
        max_lr = [a.lr * a.llrd, a.lr]
        print(f"layer-wise lr decay {a.llrd}: {sum(q.numel() for q in enc):,} encoder params at lr {a.lr * a.llrd:.2e}, "
              f"{sum(q.numel() for q in rest):,} at {a.lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(trainable, lr=a.lr,
                                      weight_decay=a.weight_decay)
    steps = a.max_steps or len(train_dl)
    total_steps = max(a.epochs * steps, 1)
    if a.schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=a.pct_start)
    else:
        # The Flowers paper's recipe: linear warm-up then cosine decay to zero.
        # Differs from OneCycle in two ways that could matter -- OneCycle also
        # anneals MOMENTUM in antiphase with the LR, and it decays to max_lr/1e4
        # rather than to 0. Which is better here is an open question and the
        # reason this flag exists; warm-up length stays --pct-start either way,
        # so the two are compared at the same warm-up.
        warm = max(int(a.pct_start * total_steps), 1)
        def lr_lambda(step):
            if step < warm:
                return (step + 1) / warm
            import math as _m
            prog = (step - warm) / max(total_steps - warm, 1)
            return 0.5 * (1.0 + _m.cos(_m.pi * min(prog, 1.0)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        torch.set_rng_state(resume["rng_torch"])
        np.random.set_state(resume["rng_numpy"])

    # One directory per run, stamped with the start time. Two runs sharing a
    # --name used to append to the same log and overwrite the same checkpoint,
    # which made the epoch axis meaningless and lost the earlier weights.
    if resume:
        run_dir = resume_path.parent          # append to the same run
        run_id = run_dir.name
    else:
        # The host suffix is not decoration. outputs/runs/ lives on an NFS home
        # shared by the cluster login node and wst001-004, and the UTC stamp has
        # one-second resolution -- so two machines starting the same parcel line
        # together generate the SAME run_id and then interleave into one
        # log.jsonl and overwrite each other's best.pth. That happened on
        # 2026-09-14 (ws_ctl_d64_s1-20260914-061154: 45 log lines, 23 distinct
        # epochs, two W&B ids). A short host tag makes the collision impossible
        # rather than unlikely.
        # FULL short hostname, not a prefix. [:8] mapped cc-gpu01-l01 (login)
        # and every cc-gpu01-n0x (compute) onto the same "cc-gpu01", which threw
        # away exactly the distinction the tag exists to make -- and would let a
        # login-node run and a compute-node run started in the same second
        # collide on the shared NFS, which is the failure the tag was added for.
        host = platform.node().split(".")[0]
        run_id = (a.run_id
                  or f"{a.name}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{host}")
        run_dir = OUTPUTS / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    print(f"run: {run_dir}")

    # Read once, here, so the commit reported by meta.json, W&B and every log
    # line is the same string even if the tree is edited mid-run.
    git = git_state()
    print(f"git: {(git['commit'] or 'unknown')[:12]}"
          f"{' DIRTY' if git['dirty'] else ''} on {git['branch'] or '?'}")

    # The number every val score is read against. Computed once, before any
    # training, and carried in every log line so a log stays interpretable on
    # its own months later.
    floor = persistence_floor(val_dl, device, measured, cfg.t_out,
                              max_steps=a.max_val_steps)
    print(f"persistence floor (val): rel_l2 {floor['rel_l2']:.4f}  "
          f"tke {floor['tke']:.4f}  -- anything above these is worse than "
          f"repeating the last frame")

    # Optional mirror to Weights & Biases. Initialised here, after cfg and the
    # floor exist, so both land in the run's searchable config -- that table of
    # configs and summaries is what W&B gives you over a pile of JSONL files.
    wb = init_wandb(a, run_id, cfg, floor, measured, model.n_params(), git=git,
                    resume_id=resume["wandb_id"] if resume else None)

    # After init_wandb so the W&B run id is on record next to the commit: that
    # pairing is the only link from a local run directory to its dashboard run.
    write_meta(run_dir, run_id, a, cfg, git, device, model.n_params(), wb)

    best = resume["best"] if resume else math.inf
    best_tke = resume["best_tke"] if resume else math.nan
    # Highest composite estimate so far -- selects best_score.pth, independently
    # of best.pth. Checkpoints predating this field carry no score_best, and
    # seeding -inf would let the first resumed epoch overwrite best_score.pth
    # with worse weights. The run's own log has every epoch's metrics, so
    # recompute it there instead. Recomputing (rather than reading a stored
    # value) also means a changed composite.WEIGHTS applies to the whole run.
    score_best = -math.inf
    if resume:
        score_best = resume.get("score_best", -math.inf)
        if not math.isfinite(score_best) and log_path.exists():
            score_best = max((composite.from_errors(e["val_rel_l2"], e["val_tke"])
                              for e in map(json.loads, log_path.read_text().splitlines())
                              if e.get("val_tke") is not None), default=-math.inf)
            print(f"recovered score_best {score_best:.4f} from {log_path.name}")
    start_epoch = resume["epoch"] + 1 if resume else 1
    if start_epoch > a.epochs:
        raise SystemExit(f"nothing to do: run already reached epoch {a.epochs}")

    # Per-frame loss weights, or None for the historical objective. Built once:
    # t_out frames, mean 1, on the training device. VALIDATION is deliberately
    # left on the unweighted global rel-L2 -- every val_rel_l2 in summary.csv
    # has to keep meaning the same thing, or arms with different objectives
    # could not be compared at all.
    if a.tke_weight or a.mvpe_weight:
        print(f"composite objective: rel_l2 + {a.tke_weight} * tke_rel_l2 + {a.mvpe_weight} * mvpe_rel_l2")
    horizon_w = None
    if a.horizon_weight != "none":
        horizon_w = horizon_weights(cfg.t_out, a.horizon_weight, device)
        print(f"horizon weighting {a.horizon_weight!r}: per-frame rel-L2, "
              f"weights {np.round(horizon_w.cpu().numpy(), 3).tolist()}")

    try:
        for epoch in range(start_epoch, a.epochs + 1):
            t0 = time.time()
            co = None
            dist = {} if teachers is not None else None
            roll = {} if a.train_rollout_weight > 0 else None
            esd = {} if a.energy_score_samples else None
            if members is None:
                tr = run_epoch(model, train_dl, device, measured, optimizer,
                               scheduler, max_steps=a.max_steps,
                               horizon_w=horizon_w, tke_w=a.tke_weight, mvpe_w=a.mvpe_weight, amse_w=a.amse_weight,
                               corr_w=a.corr_weight,
                               input_noise=a.input_noise, loss_kind=a.loss, amp_jitter=a.amp_jitter,
                               div_w=a.div_weight, speed_power=a.speed_weight_power, chan_w=a.chan_weight,
                               l1_w=a.l1_weight, cov_w=a.coverage_weight, exp_w=a.exp_weight,
                               cov_half=cov_half,
                               teachers=teachers, distill_w=a.distill_weight,
                               truth_w=a.distill_truth_weight, distill_tke=a.distill_tke,
                               distill_stats=dist,
                               rollout_w=a.train_rollout_weight, rollout_detach=a.train_rollout_detach,
                               rollout_stats=roll,
                               es_m=a.energy_score_samples, es_w=a.energy_score_weight,
                               es_replace=a.energy_score_replace, es_stats=esd)
                va = evaluate(model, val_dl, device, measured,
                              max_steps=a.max_val_steps)
                va_by_re = {str(re): evaluate(model, dl, device, measured,
                                              max_steps=a.max_val_steps)
                            for re, dl in val_dl_by_re.items()}
            else:
                co = run_epoch_co(members, train_dl, device, measured, optimizer,
                                  scheduler, max_steps=a.max_steps,
                                  tke_w=a.tke_weight, cov_w=a.coverage_weight,
                                  cov_half=cov_half, co_lambda=a.co_lambda,
                                  co_kind=a.co_diversity, bootstrap=a.co_bootstrap,
                                  co_energy_w=a.co_energy_weight)
                # train_rel_l2 stays a MEMBER's error, as in every other run.
                tr = co["rel_l2"]
                va = evaluate_ens(members, val_dl, device, measured,
                                  max_steps=a.max_val_steps)
                va_by_re = {str(re): evaluate_ens(members, dl, device, measured,
                                                  max_steps=a.max_val_steps)
                            for re, dl in val_dl_by_re.items()}
            dt = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]

            # Our estimate of the leaderboard composite. Higher is better, so
            # this one is maximized. See pde_competition.composite -- the two
            # errors go through the official score_error and the other three
            # subscores are held at fixed references, which cannot change an
            # argmax within a run.
            sc = composite.from_errors(va["rel_l2"], va["tke"])

            def save_ckpt(stem, **extra):
                """One checkpoint for a solo run; one PER MEMBER for a co-trained set.

                A co-trained run has no single "the model" -- what it produces is M
                member checkpoints meant to be scored together (score_twin.py
                --ensemble m0 m1 m2). Each is an ordinary FlowerS2S checkpoint with
                the same config, so build_submission's member-agreement check and
                everything downstream keep working unchanged.
                """
                if members is None:
                    return model.save(run_dir / f"{stem}.pth", **extra)
                for k, mdl in enumerate(members):
                    mdl.save(run_dir / f"{stem}_m{k}.pth", member=k,
                             co_members=len(members), **extra)

            flags = []
            # best.pth stays exactly what it has always been: lowest val rel-L2.
            # Every recorded analysis and every run-to-run comparison assumes
            # that, so it must not shift under them.
            if va["rel_l2"] < best:
                best = va["rel_l2"]
                best_tke = va["tke"]
                save_ckpt("best", epoch=epoch,
                          val_rel_l2=va["rel_l2"], val_tke=va["tke"],
                          run_id=run_id,
                          args=vars(a) | {"device": str(device)})
                flags.append("*best")
            # best_score.pth is the SECOND selection, on the composite estimate.
            # Weights-only like best.pth. The weights that chose it travel with
            # it, so a checkpoint can always say which estimate picked it.
            if sc > score_best:
                score_best = sc
                save_ckpt("best_score", epoch=epoch,
                          val_rel_l2=va["rel_l2"], val_tke=va["tke"],
                          val_score=sc, composite_weights=dict(composite.WEIGHTS),
                          run_id=run_id,
                          args=vars(a) | {"device": str(device)})
                flags.append("*score")
            if epoch in a.save_epochs:
                save_ckpt(f"epoch{epoch:03d}", epoch=epoch,
                          val_rel_l2=va["rel_l2"], val_tke=va["tke"],
                          run_id=run_id,
                          args=vars(a) | {"device": str(device)})
                flags.append(f"*ep{epoch}")
            flag = ("  " + " ".join(flags)) if flags else ""
            # last.pth is the RESUME artifact and carries training state;
            # best.pth is the SUBMISSION artifact and stays weights-only --
            # AdamW's moments are ~2x the model and buy nothing in a zip.
            if members is None:
                model.save(run_dir / "last.pth", epoch=epoch,
                           val_rel_l2=va["rel_l2"], val_tke=va["tke"],
                           run_id=run_id,
                           optimizer=optimizer.state_dict(),
                           scheduler=scheduler.state_dict(),
                           best=best, best_tke=best_tke, score_best=score_best,
                           rng_torch=torch.get_rng_state(),
                           rng_numpy=np.random.get_state(),
                           wandb_id=wb.run.id if wb is not None else None,
                           guard={"split": a.split, "epochs": a.epochs,
                                  "batch_size": a.batch_size, "stride": a.stride,
                                  "steps_per_epoch": steps})
            else:
                # --resume is refused for a co-trained run (parse_args), so
                # last_m*.pth is weights only: one shared optimizer state copied
                # M times is bytes nothing can consume.
                save_ckpt("last", epoch=epoch,
                          val_rel_l2=va["rel_l2"], val_tke=va["tke"],
                          run_id=run_id,
                          args=vars(a) | {"device": str(device)})

            print(f"epoch {epoch:3d}/{a.epochs}  train {tr:.4f}  "
                  f"val {va['rel_l2']:.4f} ({va['rel_l2'] / floor['rel_l2']:.2f}x)  "
                  f"tke {va['tke']:.4f} ({va['tke'] / floor['tke']:.2f}x)  "
                  f"score {sc:.3f}  lr {lr_now:.2e}  {dt:.1f}s{flag}"
                  + (f"  [ens; member {va['member_rel_l2']:.4f}"
                     f"  amb {va['ambiguity']:.4f}"
                     + (f"  co_energy {co['energy']:.4f}" if "energy" in co else "")
                     + "]" if co is not None else "")
                  + (f"  [truth {dist['truth']:.4f}  distill {dist['distill']:.4f}]"
                     if dist else "")
                  + (f"  [rollout {roll['rollout']:.4f}]" if roll else "")
                  + (f"  [energy {esd['energy']:.4f}]" if esd else "")
                  + ("".join(f"  re{re} {v['rel_l2']:.4f}" for re, v in va_by_re.items())))
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "run_id": run_id, "epoch": epoch,
                    # Per line, not once per file: a JSONL that is appended to
                    # by a --resume can span two commits, and then only the
                    # line knows which code produced it.
                    "git_commit": git["commit"], "git_dirty": git["dirty"],
                    "split": a.split, "seed": a.seed, "train_rel_l2": tr,
                    "val_rel_l2": va["rel_l2"], "val_tke": va["tke"],
                    # Per held-out Re row; the pooled numbers above are unchanged.
                    "val_by_re": va_by_re, "train_all": bool(a.train_all),
                    "holdout_re": list(a.holdout_re),
                    # GTH: None for every ordinary run, [holdout_frames, gap] when the
                    # gapped temporal split is on. val_rel_l2 on a line where this is not
                    # None was measured on a DIFFERENT validation set (the tail of all 18
                    # rows, not 2 whole rows) and must not be compared with one where it is.
                    "gapped_holdout": ([a.gth_holdout_frames, a.gth_gap]
                                       if a.gapped_holdout else None),
                    "drop_train_re": list(a.drop_train_re),
                    # val_score is logged EVERY epoch, so a later change to
                    # composite.WEIGHTS can be re-applied offline to every past
                    # run. Only checkpoint selection is frozen at training time.
                    "val_score": sc,
                    "composite_weights": dict(composite.WEIGHTS),
                    "persistence_floor": floor,
                    "measured_channels": measured, "mix_sim": a.mix_sim,
                    "init_from": str(a.init_from) if a.init_from else None,
                    # DIST provenance, on every line so a distilled run can be told
                    # from its own control without meta.json: None / 0.0 / 1.0 for
                    # every ordinary run. train_truth / train_distill are the epoch
                    # means of the two loss components and exist only when distilling.
                    "teacher_ensemble": ([str(p_) for p_ in a.teacher_ensemble]
                                         if a.teacher_ensemble else None),
                    "distill_weight": a.distill_weight,
                    "distill_truth_weight": a.distill_truth_weight,
                    **({} if not dist else {"train_truth": dist["truth"],
                                            "train_distill": dist["distill"],
                                            "distill_tke": bool(a.distill_tke)}),
                    # ROLL provenance on every line: 0.0 / False for every ordinary
                    # run. train_rollout is the epoch mean of the rollout term
                    # (rel_l2 + tke_w * tke on the second pass) and exists only when on.
                    "train_rollout_weight": a.train_rollout_weight,
                    "train_rollout_detach": bool(a.train_rollout_detach),
                    # Truncated backprop between --rollout-frames passes (model attribute,
                    # absent from "config"): False for every ordinary run.
                    "rollout_detach_passes": bool(a.rollout_detach_passes),
                    **({} if not roll else {"train_rollout": roll["rollout"]}),
                    # PNO_D provenance on every line: 0 / 1.0 / False for every ordinary run.
                    # train_energy is the epoch mean of the relative energy score and exists
                    # only when it is on; train_rel_l2 is then the SAMPLE MEAN's.
                    "energy_score_samples": a.energy_score_samples,
                    "energy_score_weight": a.energy_score_weight,
                    "energy_score_replace": bool(a.energy_score_replace),
                    **({} if not esd else {"train_energy": esd["energy"]}),
                    "lr": lr_now, "seconds": dt, "config": asdict(cfg),
                    # E3/L8 only, absent from every ordinary run. val_rel_l2 and
                    # val_tke above are the ENSEMBLE MEAN's for a co-trained run;
                    # val_member_rel_l2 is the number comparable to other runs.
                    **({} if co is None else {
                        "co_members": len(members), "co_lambda": a.co_lambda,
                        "co_diversity": a.co_diversity, "co_jitter": a.co_jitter,
                        "co_bootstrap": bool(a.co_bootstrap),
                        # BET A: 0.0 on every co-trained run without it; train_co_energy
                        # (epoch mean of the members' relative energy score) only when on.
                        "co_energy_weight": a.co_energy_weight,
                        **({"train_co_energy": co["energy"]} if "energy" in co else {}),
                        "val_member_rel_l2": va["member_rel_l2"],
                        "val_member_tke": va["member_tke"],
                        "val_ambiguity": va["ambiguity"],
                        "train_member_rel_l2": co["rel_l2"],
                        "train_ens_rel_l2": co["ens_rel_l2"],
                        "train_ambiguity": co["ambiguity"]})}) + "\n")

            if wb is not None:
                # floor_ratio is the curve worth watching: crossing 1.0 is the
                # moment the model starts beating "repeat the last frame".
                wb.log({"train/rel_l2": tr,
                        "val/rel_l2": va["rel_l2"],
                        "val/floor_ratio": va["rel_l2"] / floor["rel_l2"],
                        "val/tke": va["tke"],
                        "val/tke_floor_ratio": va["tke"] / floor["tke"],
                        "val/score": sc,
                        **{f"val/rel_l2_re{re}": v["rel_l2"] for re, v in va_by_re.items()},
                        **{f"val/tke_re{re}": v["tke"] for re, v in va_by_re.items()},
                        **({} if co is None else {
                            "val/member_rel_l2": va["member_rel_l2"],
                            "val/member_tke": va["member_tke"],
                            "val/ambiguity": va["ambiguity"],
                            "train/ambiguity": co["ambiguity"],
                            "train/ens_rel_l2": co["ens_rel_l2"],
                            **({"train/co_energy": co["energy"]} if "energy" in co else {})}),
                        **({} if not dist else {"train/truth": dist["truth"],
                                                "train/distill": dist["distill"]}),
                        **({} if not roll else {"train/rollout": roll["rollout"]}),
                        **({} if not esd else {"train/energy": esd["energy"]}),
                        "lr": lr_now, "epoch_seconds": dt}, step=epoch)

        verdict = ("beats persistence" if best < floor["rel_l2"] else
                   "WORSE THAN PERSISTENCE -- the model has learned nothing useful")
        print(f"\nbest val rel-L2 {best:.4f}  vs floor {floor['rel_l2']:.4f}  "
              f"({best / floor['rel_l2']:.2f}x)  -- {verdict}")
        print(f"     its val TKE {best_tke:.4f}  vs floor {floor['tke']:.4f}  "
              f"({best_tke / floor['tke']:.2f}x)")
        print(f"best composite estimate {score_best:.4f}  -> best_score.pth "
              f"(weights {composite.WEIGHTS})")
        if members is not None:
            # M members live in one process, so the memory bill is the thing that
            # decides how many such runs may be packed on one GPU. Printed for
            # co-trained runs only, where it is a planning number rather than noise.
            print(f"co-trained {len(members)} members -> "
                  + ", ".join(f"best_score_m{k}.pth" for k in range(len(members))))
            if device.type == "cuda":
                print(f"peak CUDA memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
        print(f"  -> {run_dir}")

        if wb is not None:
            # summary is the one headline number per run -- what the runs table
            # sorts on, so it should be the number that decides anything.
            wb.run.summary["best_val_rel_l2"] = best
            wb.run.summary["best_val_tke"] = best_tke
            wb.run.summary["best_floor_ratio"] = best / floor["rel_l2"]
            wb.run.summary["best_val_score"] = score_best
    finally:
        # Close the W&B run even on Ctrl-C or a crash, or it is left marked
        # "running" forever in the dashboard.
        if wb is not None:
            wb.finish()


def build_parser() -> argparse.ArgumentParser:
    """The CLI, without parsing. Separate from parse_args so adopt_init_config can ask the
    parser for its DEFAULTS -- that is how it tells a flag given on the command line from one
    left alone -- and so tests can build a Namespace without touching sys.argv."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="flower",
                   help="run name; the run directory is <name>-<timestamp>")
    p.add_argument("--run-id", default=None,
                   help="override the generated run id (and so the output "
                        "directory name). Reusing one overwrites that run.")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                   help="mirror metrics to Weights & Biases (default: on; "
                        "--no-wandb to disable). The JSONL log is written either "
                        "way and stays the authoritative record. Needs `pip "
                        "install wandb` and `wandb login`, or --wandb-mode "
                        "offline.")
    p.add_argument("--wandb-project", default="realpde-competition")
    p.add_argument("--wandb-group", default=None, metavar="NAME",
                   help="W&B group: give every arm of one experiment the same "
                        "value and the dashboard can compare them as a set "
                        "(e.g. --wandb-group mix-sim-sweep). Does not affect "
                        "the run's own name or output directory.")
    p.add_argument("--wandb-tags", nargs="*", default=None, metavar="TAG",
                   help="extra W&B tags for filtering across groups. The split "
                        "is tagged automatically, as is a dirty working tree.")
    p.add_argument("--wandb-mode", default="online",
                   choices=("online", "offline", "disabled"),
                   help="offline records to ./wandb/ and sends nothing; sync "
                        "later with `wandb sync wandb/offline-run-*`")
    p.add_argument("--split", default="train_real",
                   help="processed subdir: train_real or train_sim")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--rollout-frames", type=int, default=0, metavar="K",
                   help="emit K frames per forward pass and roll out t_out/K "
                        "times, sliding the input window over the model's own "
                        "predictions. 0 (default) emits the whole horizon in one "
                        "pass, which is what every run so far has done. K must "
                        "divide t_out. This is the mode the Flowers paper "
                        "actually evaluates -- their headline claim is low "
                        "ROLLOUT error -- and it costs t_out/K passes at "
                        "inference, which time_score charges for.")
    p.add_argument("--schedule", default="onecycle", choices=("onecycle", "warmcos"),
                   help="LR schedule. 'onecycle' is ours (OneCycleLR, which also "
                        "anneals momentum in antiphase and decays to max_lr/1e4). "
                        "'warmcos' is the Flowers paper's: linear warm-up then "
                        "cosine decay to zero. Warm-up length is --pct-start for "
                        "both, so they are compared at the same warm-up.")
    p.add_argument("--pct-start", type=float, default=0.1, metavar="FRAC",
                   help="fraction of OneCycle spent warming up to --lr "
                        "(default 0.1). Raising it is the standard remedy for "
                        "a run that picks a bad basin early: d128 at lr 1e-3 "
                        "peaked at epoch 2 and never recovered, and window-norm "
                        "seeds are bimodal.")
    p.add_argument("--drop-train-re", type=int, nargs="*", default=[],
                   help="S1c: drop these Re rows from TRAINING only, leaving --holdout-re's validation "
                        "set untouched, so arms trained on different amounts of data are comparable.")
    p.add_argument("--holdout-re", type=int, nargs="+", default=[13950, 26700],
                   help="Reynolds rows held out of training (val). Default is "
                        "D3's split. Other choices are for CROSS-FITTING: a "
                        "model that never saw rows R predicts them honestly, so "
                        "K models with disjoint holdouts give out-of-fold "
                        "residuals on every row -- the training set for a width "
                        "model. Runs with a non-default holdout are NOT "
                        "comparable to anything else on val.")
    p.add_argument("--gapped-holdout", action="store_true",
                   help="direction GTH: validate on the LAST --gth-holdout-frames frames of "
                        "EVERY trajectory instead of on whole Reynolds rows, with --gth-gap "
                        "frames in between belonging to neither split. The instrument then "
                        "spans all 18 Re rows rather than 2, which is the axis HO2 said our "
                        "twin cannot see; the price is that it is out-of-sample in TIME only, "
                        "in-sample in condition. --holdout-re is unused and a non-default value "
                        "is refused; --train-all and --mix-sim are refused. The phase stores are "
                        "cut at the same frame indices (they are the same trajectories on a "
                        "shifted grid). OFF by default and then not one line of it runs. "
                        "val_rel_l2 from this mode is NOT comparable to any other run.")
    p.add_argument("--gth-holdout-frames", type=int, default=100, metavar="N",
                   help="--gapped-holdout: how many frames at the END of each trajectory are "
                        "validation. 100 gives 2 windows per run at stride 50 (162 windows over "
                        "all 18 rows) and leaves 728 of 868 frames to train on.")
    p.add_argument("--gth-gap", type=int, default=40, metavar="G",
                   help="--gapped-holdout: how many frames between the two splits belong to "
                        "NEITHER. The u-fluctuation autocorrelation of this flow is 0.87 at lag "
                        "1 and ~0.2 at lag 10, so without a gap the first val window is nearly "
                        "the same flow state as the last train window. At lag 40 it is in "
                        "[-0.07, +0.12] on rows 3750/10125/13950/26700 and oscillates rather "
                        "than decaying beyond, so a larger gap buys little.")
    p.add_argument("--train-all", action="store_true",
                   help="train on EVERY real cell, held-out rows included -- the "
                        "final-checkpoint mode for a submission. Val is then "
                        "in-sample: pick the epoch from the held-out twin run "
                        "(--save-epochs) and fit the bounds on the twin "
                        "(fit_bounds.py --write-for).")
    p.add_argument("--save-epochs", type=int, nargs="*", default=[],
                   help="also save weights-only epochNNN.pth at these epochs")
    p.add_argument("--amse-weight", type=float, default=0.0,
                   help="coherence-decoupled spectral loss along time (Subich et al., ICML 2025). The only "
                        "candidate objective whose mechanism targets corr(pred,true) rather than amplitude, "
                        "and corr is the only lever that moves the rel_l2-vs-tke frontier. See amse_time.")
    p.add_argument("--corr-weight", type=float, default=0.0,
                   help="direction A7: add this multiple of (1 - per-lead fluctuation correlation) to "
                        "the loss. BLUR measured corr falling 0.881 -> 0.506 over the 20 leads while "
                        "r/corr^2 = 1.34-1.78 says the prediction is ALREADY over-amplified relative to "
                        "the L2 optimum, so corr is the only quantity that moves the rel_l2-vs-tke "
                        "frontier rather than walking it. The term is scale-invariant, so it cannot "
                        "trade energy for phase and leaves the amplitude to rel_l2 and to the post-hoc "
                        "fluct_scale. 0.0 is OFF and bit-identical to before. See fluct_corr.")
    p.add_argument("--mvpe-weight", type=float, default=0.0,
                   help="add this multiple of the probe mean-velocity-profile "
                        "relative L2 (mvpe_score's metric) to the loss. 0.19 "
                        "mirrors the solved composite at submission 8's operating "
                        "point (d(final)/d(err_mvpe) is 0.19 of rel-L2's).")
    p.add_argument("--l1-weight", type=float, default=0.0,
                   help="add this multiple of the relative L1 (median-seeking) to the loss.")
    p.add_argument("--coverage-weight", type=float, default=0.0,
                   help="add this multiple of (1 - smooth coverage) at --coverage-half per channel.")
    p.add_argument("--mask-from-real", action="store_true",
                   help="multiply every TRAINING window by a real PIV data hole, drawn per window from the real "
                        "trajectories' masks. For the SIM pretrain: sim is a full field, real has a ~9%% hole whose "
                        "shape moves with angle of attack, so today the pretrain never sees one and the fine-tune "
                        "has to learn 'infer around a hole and emit zeros in it' from scratch -- which W146 showed "
                        "it does by overwriting the pretrain wholesale. Changes the INPUT DISTRIBUTION only: no loss "
                        "term and no target semantics move.")
    p.add_argument("--mask-channel", action="store_true",
                   help="write the per-frame measured mask (u != 0) into input channel 2, which is identically "
                        "zero on real PIV and therefore a dead input. Distinct from W154, which zeroed the "
                        "OUTPUT outside the mask and measured +0.000; nothing has ever given the mask as an INPUT.")
    p.add_argument("--lagrangian-c", type=float, default=None, metavar="PX",
                   help="pixels per frame per unit mean|u| for --lagrangian (default: the measured 4.95). "
                        "0 makes the shear an exact no-op, which the tests pin as bit-identical to 'off'.")
    p.add_argument("--lagrangian", choices=("off", "input"), default="off",
                   help="de-shear the input window into the frame co-moving with the flow before the lift. "
                        "Structures convect at ~4.95 px/frame per unit mean|u|, so at Re 26700 they cross 32 px "
                        "of a 64 px domain over the 20 input frames while the correlation length is 3-6 px -- and "
                        "Flower's first op is a 1x1 conv over all 60 channels, which cannot re-align them.")
    p.add_argument("--exp-weight", type=float, default=0.0,
                   help="add this multiple of -mean(exp(-2|r|/sigma)) to the loss: the SPS reward integrated over "
                        "EVERY half-width, of which --coverage-weight's term is a single fixed-h slice. Gradient "
                        "redescends on hopeless elements, matching the scorer's economics, and it has no free "
                        "half-width parameter (sigma is the scorer's own constant).")
    p.add_argument("--coverage-half-map", default=None, metavar="BOUNDS_JSON",
                   help="aim the coverage surrogate at a per-element half-width MAP (the 'half_width_map' block "
                        "written by fit_bounds.py --spatial) instead of one constant per channel. The shipped "
                        "interval is wide in the wake and at late frames and narrow in the free stream; a constant "
                        "spends gradient where residuals would have scored anyway and ignores residuals that will "
                        "score exactly zero. W128 closed the constant's LEVEL, not its SHAPE.")
    p.add_argument("--coverage-half", type=float, nargs="+", default=[0.0093, 0.0063, 0.0],
                   help="per-channel half-widths for --coverage-weight (default: submission 9's constant fit).")
    p.add_argument("--div-weight", type=float, default=0.0,
                   help="add this multiple of the relative L2 between the prediction's and the "
                        "target's in-plane divergence (a physics term valid on real PIV slices).")
    p.add_argument("--speed-weight-power", type=float, default=0.0,
                   help="weight each window's rel-L2 by (mean |u| of its input / batch mean)^p, "
                        "normalised to mean 1: p > 0 up-weights high-Re windows. 0 = off.")
    p.add_argument("--loss", choices=("rel_l2", "vrmse"), default="rel_l2",
                   help="primary loss: rel_l2 (the metric; default) or vrmse (per-channel "
                        "variance-normalised RMSE). Reported train_rel_l2 stays rel-L2.")
    p.add_argument("--amp-jitter", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Reynolds-similarity augmentation: scale input AND target of each window by "
                        "s ~ U[LO, HI] (proposition 7a; e.g. 0.85 1.2)")
    p.add_argument("--llrd", type=float, default=1.0,
                   help="layer-wise LR decay: lift/encoder/bottleneck train at lr * LLRD (proposition 9); 1 = off")
    p.add_argument("--input-noise", type=float, default=0.0,
                   help="training-time Gaussian noise on the input window, sd = this "
                        "fraction of each channel's batch std, only on measured points "
                        "(default 0: none). Sim2real augmentation for the pretrain.")
    p.add_argument("--tke-weight", type=float, default=0.0,
                   help="add this multiple of the TKE relative L2 to the loss "
                        "(default 0: the historical rel-L2-only objective). "
                        "0.14 mirrors the solved composite at submission 8's "
                        "operating point -- d(final)/d(err_tke) is 0.136 of "
                        "d(final)/d(err_rel_l2). Logged train_rel_l2 stays the "
                        "plain rel-L2 so arms remain comparable.")
    p.add_argument("--horizon-weight", default="none",
                   choices=("none", "uniform", "linear", "sqrt"),
                   help="per-frame loss weighting. 'none' is the historical "
                        "objective (one relative L2 over the flattened window). "
                        "The other three score each frame against its OWN norm "
                        "and weight the result; 'uniform' is the control that "
                        "isolates that change of metric from the weighting, "
                        "because a mean of ratios is not the ratio of sums. "
                        "analyses/error_by_frame.md: late frames carry ~2/3 of "
                        "the error and the objective is horizon-blind.")
    p.add_argument("--stride", type=int, default=10,
                   help="frames between training windows; lower = more, more "
                        "correlated, windows per epoch")
    p.add_argument("--val-stride", type=int, default=50)
    # Default read off FlowerConfig rather than restated, so the CLI cannot
    # silently disagree with the library about what a model is.
    p.add_argument("--channels", type=int, default=FlowerConfig.c_in, choices=(2, 3),
                   help="3 = [u, v, p] as the grader sends them, and what sim "
                        "carries; 2 = [u, v] only, which is all real data "
                        "measures and all scoring reads")
    p.add_argument("--lifting-dim", type=int, default=64)
    p.add_argument("--n-levels", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--groups", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="dropout rate inside every Flower residual unit (after each FlowerBlock's "
                        "GELU) and before the output projection; active in train mode only. Under "
                        "--init-from it OVERRIDES the checkpoint's rate (no parameter shape changes). "
                        "Blind spot: --dropout 0 is the default, so against a checkpoint trained with "
                        "dropout it reads as absent and the checkpoint's rate is kept.")
    p.add_argument("--window-scale", default="u", choices=("u", "speed"),
                   help="which scalar --window-norm divides by. 'u' is mean|u| "
                        "(the original). 'speed' is sqrt(mean(u^2+v^2)): mean|u| "
                        "is the streamwise component, not the flow speed, so it "
                        "understates the free stream by ~cos(aoa) -- and aoa "
                        "spans 0-20 degrees here, so the understatement varies "
                        "with the condition and leaves an angle-dependent bias "
                        "in exactly the space window-norm exists to clean up. "
                        "Ignored without --window-norm; inherited from the "
                        "checkpoint under --init-from.")
    p.add_argument("--window-norm", action="store_true",
                   help="divide each window by its own free-stream |u| before "
                        "the global normalizer. Real PIV is dimensional and "
                        "sim is not, so mean|u| spans 5.89x across the real "
                        "Reynolds rows and is flat at ~0.87 for sim; this "
                        "collapses that to 1.01x and puts both in one space. "
                        "Needs --channels 2 (p scales as U^2). Part of the "
                        "architecture, so a --init-from fine-tune inherits it "
                        "from the checkpoint -- do not pass it there.")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="hold lift + encoder + bottleneck fixed and train only "
                        "the decoder and projection. Requires --init-from; "
                        "stops a fine-tune from un-learning the pretrained "
                        "features during OneCycle warm-up.")
    p.add_argument("--mix-sim", type=float, default=0.0, metavar="FRAC",
                   help="fraction of each training epoch drawn from train_sim, "
                        "on top of a real run. Validation stays real-only and "
                        "the epoch keeps its real-only size, so the number "
                        "remains comparable. Implies --channels 2 (real has no "
                        "p) and wants --window-norm (the domains differ 3.9x in "
                        "u and 13.5x in v otherwise).")
    p.add_argument("--mix-grow-epoch", action="store_true",
                   help="D1p: with --mix-phases/--mix-sim, grow the epoch by 1/(1-FRAC) so the real p00 windows "
                        "keep a full epoch's exposure and the mixed copies come on top (a budget change; "
                        "OneCycle's total_steps grows with it).")
    p.add_argument("--mix-phases", type=float, default=0.0, metavar="FRAC",
                   help="add the three shifted-grid PIV phases (p01/p10/p11) as extra TRAINING windows at "
                        "this fraction of each epoch. They are the same flow on a half-pixel-shifted grid, "
                        "so they are 3x more REAL data -- but the airfoil sits at a different sub-pixel "
                        "position in each, so a model trained on them alone learns a blurred body. Use as a "
                        "curriculum stage and fine-tune on p00 afterwards. Held-out cells are removed from "
                        "every phase, or row 26700 would enter training at a half-pixel shift.")
    p.add_argument("--chan-weight", type=float, nargs="+", default=None, metavar="W",
                   help="per-channel weights for a PER-CHANNEL relative L2 instead of the flattened one. "
                        "rel_l2 puts u and v in one norm, so the ratio is dominated by u (sd 0.106 against "
                        "v's 0.0157). The score normalises over the measured channels together, so v is "
                        "scored but barely trained. '1 1' trains them equally in RELATIVE terms.")
    p.add_argument("--temporal-lift", type=int, default=0, metavar="K",
                   help="depthwise conv of width K over the TIME axis before the lift, identity-initialised. "
                        "The core's first op is a 1x1 conv over all t_in*c_in channels, so at a pixel the 20 "
                        "frames are mixed globally -- but fluctuations convect ~4.95 px/frame per unit mean|u|, "
                        "so at Re 26700 they are ~10 unrelated eddies against a 3 px correlation length. K must "
                        "be odd. 0 = off and bit-identical to every run so far.")
    p.add_argument("--rans", action="store_true",
                   help="Reynolds decomposition: the core emits c_out extra channels holding a TIME-CONSTANT "
                        "mean field and the rest are the fluctuation about it. W182 measured 95.5%% of the tke "
                        "error as decorrelation of the fluctuation while the mean is easy. CHANGES dim_out, so "
                        "this cannot --init-from an existing checkpoint and must train from scratch (which W177 "
                        "priced at only -0.005 on row 26700).")
    p.add_argument("--boundary-hw", nargs=2, default=None,
                   choices=("periodic", "zeros", "border", "reflection"),
                   metavar=("H", "W"),
                   help="per-axis warping padding: H (cross-stream, 32) then W (streamwise, 64). "
                        "The sampler has always supported this and nothing has ever used it -- every "
                        "run so far applied ONE value to both axes. One of the two must be periodic "
                        "when they differ. Overrides --boundary.")
    p.add_argument("--boundary", choices=("periodic", "zeros", "border"), default="zeros",
                   help="warping padding at the domain edge. zeros (our default) "
                        "treats outside-the-domain as 0; periodic is the UPSTREAM "
                        "default and WRAPS both axes, which an airfoil in a "
                        "channel is not. Only the literal PERIODIC is special "
                        "upstream -- everything else maps to zeros, so these are "
                        "the only two reachable.")
    p.add_argument("--delta", action="store_true",
                   help="predict the change from the last input frame")
    p.add_argument("--co-members", type=int, default=0, metavar="M",
                   help="E3/L8: CO-TRAIN M members in one process on the same batches, with a "
                        "diversity term coupling them (--co-lambda). 0, the default, is the "
                        "ordinary single-model run and changes nothing. The run directory then "
                        "holds best_m0.pth ... best_score_m0.pth ..., one per member, to be "
                        "scored together with score_twin.py --ensemble.")
    p.add_argument("--co-lambda", type=float, default=0.0, metavar="LAM",
                   help="weight on the diversity reward. The total loss is sum_m L_m - LAM * M * "
                        "ensemble_ambiguity, so LAM is the diversity push as a FRACTION of each "
                        "member's own accuracy push, independent of M. 0 is the co-trained "
                        "CONTROL: members share batches but are not coupled at all.")
    p.add_argument("--co-energy-weight", type=float, default=0.0, metavar="W",
                   help="BET A: add W * the RELATIVE FAIR ENERGY SCORE over the --co-members "
                        "predictions (members as samples, measured channels; energy_score) to the "
                        "summed per-member point losses. A proper scoring rule, so members spread "
                        "where the truth is uncertain. Not scaled by M. 0 (default) is off and "
                        "bit-identical. Refused with --co-lambda != 0 and --energy-score-samples.")
    p.add_argument("--co-diversity", choices=("var", "err"), default="var",
                   help="which disagreement to reward. 'var' is the plain ambiguity (identical, up "
                        "to a factor 2, to mean pairwise MSE between members). 'err' weights each "
                        "element by the ensemble mean's own squared error there -- disagreement "
                        "where the mean is already right is pure cost.")
    p.add_argument("--co-jitter", type=float, default=0.05, metavar="FRAC",
                   help="relative weight perturbation that separates the members at init, as a "
                        "fraction of each tensor's own std. Required, not optional: identical "
                        "members are an exact fixed point of the co-trained objective. Seed "
                        "members end up 0.29-0.31 apart in the same units.")
    p.add_argument("--co-bootstrap", action="store_true",
                   help="also resample each member's OWN batch with replacement (B draws from B), "
                        "independently per member. Bagging: it gives every member an independent "
                        "gradient-noise stream, which is what actually decorrelates independent "
                        "seed runs. The diversity term always sees the whole batch.")
    p.add_argument("--teacher-ensemble", type=str, nargs="+", default=None, metavar="PTH",
                   help="DIST: distil from the MEAN of these frozen checkpoints. Loaded in eval mode "
                        "with no grad on the training device; each keeps its own normalisation "
                        "statistics. Refused if a teacher disagrees with the student on the fields "
                        "the data path or the shipped ensemble depends on (channels, delta, "
                        "lifting_dim, ...). Needs --distill-weight > 0. Plain strings, not Paths, "
                        "so the list serialises into meta.json as-is.")
    p.add_argument("--distill-weight", type=float, default=0.0, metavar="W",
                   help="DIST: weight on rel_l2(pred, teacher mean). 0 (default) is OFF and the "
                        "training path is bit-identical to a run without --teacher-ensemble.")
    p.add_argument("--distill-truth-weight", type=float, default=1.0, metavar="V",
                   help="DIST: weight on the WHOLE existing objective against the truth (rel_l2 + "
                        "--tke-weight + --coverage-weight + ...). 0 trains on the teachers alone; "
                        "1 (default) keeps the truth term exactly as it was and adds the teacher on top.")
    p.add_argument("--distill-tke", action="store_true",
                   help="DIST: also add W * --tke-weight * tke_rel_l2(pred, teacher mean). The ensemble "
                        "mean is smoother than any member, so matching it in rel-L2 alone pulls the "
                        "student's fluctuation energy down; this asks it to match the teacher's TKE "
                        "as well. Needs --tke-weight > 0 or it would be a silent no-op.")
    p.add_argument("--train-rollout-weight", type=float, default=0.0, metavar="W",
                   help="ROLL: TRAINING-ONLY rollout consistency. After predicting frames [20,40) "
                        "from [0,20), feed that prediction back as the next input window, predict "
                        "[40,60), and add W * (rel_l2 + --tke-weight * tke) against the TRUE frames "
                        "[40,60). Training windows become 60 frames (slightly fewer per epoch); "
                        "validation and the shipped module are untouched. 0 (default) is OFF and "
                        "bit-identical to before. Two forwards per batch, so an epoch is ~2x slower. "
                        "The rollout term is always the plain rel_l2 + tke objective, whatever "
                        "--loss / --horizon-weight / --chan-weight the first pass uses. Refused with "
                        "--rollout-frames, --rans, --teacher-ensemble and --co-members.")
    p.add_argument("--train-rollout-detach", action="store_true",
                   help="ROLL: detach the first-pass prediction before feeding it back, so the rollout "
                        "term trains the second pass only and sends no gradient into the first. Default "
                        "off: gradients flow through both passes.")
    p.add_argument("--rollout-feed", type=int, default=0, metavar="M",
                   help="rollout members: feed back only the first M output channels (masked by the "
                        "input's PIV holes) and carry the rest over from the last input frame. 2 on "
                        "real data (u, v measured; p is never trained). 0 = legacy (feed all).")
    p.add_argument("--rollout-detach-passes", action="store_true",
                   help="TRAINING-ONLY truncated backprop for a --rollout-frames K model: detach the "
                        "K frames fed back into the input window between passes, so each pass trains "
                        "on its own loss (inputs still include the earlier passes' outputs) but no "
                        "gradient flows back through earlier passes. The standard stabiliser for a "
                        "rollout that diverges at high lr. Not saved in the checkpoint config; "
                        "inference/eval is bit-identical. Refused unless the final config has "
                        "rollout_frames > 0 (under --init-from: --rollout-frames K --init-slice-output).")
    p.add_argument("--energy-score-samples", type=int, default=0, metavar="M",
                   help="PNO_D probe (arXiv 2502.12902): at M >= 2, push every batch through the model M "
                        "times with independent dropout masks and add --energy-score-weight * the RELATIVE "
                        "energy score of the M samples (per window: mean_i ||s_i - y||/||y|| - "
                        "1/(2M(M-1)) sum_{i!=j} ||s_i - s_j||/||y||, norms over T,H,W and the measured "
                        "channels, then mean over the batch). The point terms (rel_l2, tke, coverage, ...) "
                        "and train_rel_l2 are computed on the SAMPLE MEAN; train_energy is logged. COST: "
                        "each step is ~Mx a plain one (forward and backward; one batched forward of M*B "
                        "windows, so also ~Mx the activation memory). Needs dropout > 0 (--dropout P, "
                        "which overrides the checkpoint under --init-from). Refused with "
                        "--teacher-ensemble, --train-rollout-weight > 0 and --co-members. 0 (default) is "
                        "OFF and bit-identical to before.")
    p.add_argument("--energy-score-weight", type=float, default=1.0, metavar="W",
                   help="PNO_D: weight on the energy score (default 1.0). Must be > 0.")
    p.add_argument("--energy-score-replace", action="store_true",
                   help="PNO_D: the energy score REPLACES the primary point term (rel_l2, or whatever "
                        "--loss/--horizon-weight/--chan-weight/--speed-weight-power selects) instead of "
                        "being added to it. --tke-weight / --coverage-weight etc. still apply, on the "
                        "sample mean.")
    p.add_argument("--init-from", type=Path, default=None, metavar="CKPT",
                   help="start from a trained checkpoint instead of random "
                        "weights -- this is how sim pretraining feeds real "
                        "fine-tuning. The checkpoint supplies the architecture; "
                        "an architecture flag given with a value that DIFFERS from "
                        "the checkpoint's (--channels, --lifting-dim, --n-levels, "
                        "--num-heads, --groups, --temporal-lift, --rans, "
                        "--rollout-frames) is REFUSED rather than silently dropped. "
                        "--delta, --mask-channel, --lagrangian(-c), --dropout override "
                        "it (no shape change); --temporal-lift K overrides a checkpoint "
                        "WITHOUT a temporal lift (tlift added at identity init, step 0 = "
                        "the checkpoint; a different non-zero K is refused); "
                        "--window-norm, --window-scale, "
                        "--boundary(-hw) are ignored with a warning. Normalization "
                        "statistics are re-fitted on this split.")
    p.add_argument("--init-slice-output", action="store_true",
                   help="with --init-from and --rollout-frames K different from the "
                        "checkpoint's: build the checkpoint's architecture with "
                        "rollout_frames=K and load its weights with the OUTPUT "
                        "projection sliced to the first K frames (channels are "
                        "frame-major, so rows [0:K*c_out]). Every other parameter "
                        "loads strictly. At step 0 the first pass reproduces the "
                        "checkpoint's first K frames exactly.")
    p.add_argument("--resume", type=Path, default=None, metavar="RUN_DIR",
                   help="continue an interrupted run: pass its outputs/runs/ "
                        "directory. Restores optimizer, LR schedule, epoch, "
                        "best-so-far and RNG, and appends to the same log and "
                        "W&B run. Refuses if --epochs/--batch-size/--stride/"
                        "--split differ from the original. Not the same as "
                        "--init-from, which starts a NEW run from weights only.")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader worker processes (default 4). Was 0 until "
                        "2026-09-14, when the GPU was measured 20%% busy while "
                        "train.py burned 1323%% CPU: the feed was the "
                        "bottleneck, not the maths. Batch CONTENT and ORDER are "
                        "unchanged -- the sampler runs in the main process and "
                        "the dataset is deterministic -- so this is plumbing, "
                        "not a result-affecting knob.")
    p.add_argument("--threads", type=int, default=4, metavar="N",
                   help="cap torch's intra-op thread pool (0 = leave alone). "
                        "Kernels here are 32x64 at batch 16; the pool spins "
                        "more than it computes.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap training batches per epoch")
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: 2 epochs, a handful of batches, small model")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_parser()
    a = p.parse_args(argv)
    if a.init_slice_output and not a.init_from:
        p.error("--init-slice-output slices a checkpoint's output projection; it needs --init-from")
    if a.resume and a.init_from:
        p.error("--resume and --init-from are mutually exclusive: --resume "
                "continues this run, --init-from starts a new one from "
                "someone else's weights")
    if a.co_members:
        # The co-trained path reimplements only the objective that ships
        # (rel_l2 + tke + coverage). Anything else would be silently dropped,
        # which is how an arm becomes an unwitting control -- refuse instead.
        if a.co_members < 2:
            p.error("--co-members must be at least 2 (or 0 for an ordinary run)")
        unsupported = [n for n, v in (("--loss vrmse", a.loss != "rel_l2"),
                                      ("--horizon-weight", a.horizon_weight != "none"),
                                      ("--speed-weight-power", a.speed_weight_power),
                                      ("--chan-weight", a.chan_weight),
                                      ("--mvpe-weight", a.mvpe_weight),
                                      ("--div-weight", a.div_weight),
                                      ("--l1-weight", a.l1_weight),
                                      ("--exp-weight", a.exp_weight),
                                      ("--input-noise", a.input_noise),
                                      ("--amp-jitter", a.amp_jitter),
                                      ("--rollout-frames", a.rollout_frames),
                                      ("--llrd", a.llrd < 1.0),
                                      ("--resume", a.resume)) if v]
        if unsupported:
            p.error("--co-members supports the shipping objective only "
                    "(rel_l2 + --tke-weight + --coverage-weight); these are not "
                    "implemented on that path: " + ", ".join(unsupported))
        # BET A: one ensemble term at a time, and never on top of the dropout-sample ES.
        if a.co_energy_weight < 0:
            p.error(f"--co-energy-weight must be >= 0, got {a.co_energy_weight}")
        if a.co_energy_weight and a.co_lambda:
            p.error("--co-energy-weight with --co-lambda != 0: one ensemble term at a time")
        if a.co_energy_weight and a.energy_score_samples:
            p.error("--co-energy-weight with --energy-score-samples: one energy score at a time")
    elif a.co_lambda or a.co_bootstrap or a.co_energy_weight:
        p.error("--co-lambda / --co-bootstrap / --co-energy-weight do nothing without --co-members")
    # DIST: every combination that would silently be a control is refused, in the
    # spirit of the --co-lambda check above -- an arm that looks distilled but is not
    # is worse than no arm.
    if a.teacher_ensemble:
        if a.distill_weight <= 0:
            p.error("--teacher-ensemble without --distill-weight > 0 would load the teachers "
                    "and never use them: an unwitting control. Pass --distill-weight W.")
        if a.co_members:
            p.error("--teacher-ensemble is not implemented on the --co-members path")
        if a.distill_tke and not a.tke_weight:
            p.error("--distill-tke adds W * --tke-weight * tke(pred, teacher); with --tke-weight 0 "
                    "it is a no-op. Pass --tke-weight or drop --distill-tke.")
    elif a.distill_weight or a.distill_truth_weight != 1.0 or a.distill_tke:
        p.error("--distill-weight / --distill-truth-weight / --distill-tke do nothing without "
                "--teacher-ensemble")
    # ROLL. Refused where the second pass would be undefined or the loss composition
    # ambiguous: --rollout-frames already rolls out INSIDE forward (a different
    # mechanism, never combined); --rans changes dim_out and has never been in the
    # recipe; --teacher-ensemble wraps the whole objective in truth_w and the teachers
    # would have to be rolled out too; --co-members has its own epoch loop that reads
    # 2-tuples. A detach flag without the weight is a no-op and is refused like
    # --co-lambda without --co-members.
    if a.train_rollout_weight < 0:
        p.error("--train-rollout-weight must be >= 0")
    if a.train_rollout_weight > 0:
        clash = [n for n, v in (("--rollout-frames", a.rollout_frames),
                                ("--rans", a.rans),
                                ("--teacher-ensemble", a.teacher_ensemble),
                                ("--co-members", a.co_members)) if v]
        if clash:
            p.error("--train-rollout-weight is not implemented together with: " + ", ".join(clash))
    elif a.train_rollout_detach:
        p.error("--train-rollout-detach does nothing without --train-rollout-weight > 0")
    if not 0.0 <= a.dropout < 1.0:
        p.error(f"--dropout must be in [0, 1), got {a.dropout}")
    # PNO_D. Kept to the plain single-model path: the teachers, the rollout second pass
    # and the co-trained loop each compose the loss their own way, and an ES arm that
    # silently skipped one of them would be an unwitting control.
    if a.energy_score_samples < 0 or a.energy_score_samples == 1:
        p.error(f"--energy-score-samples must be 0 (off) or >= 2, got {a.energy_score_samples}")
    if a.energy_score_samples:
        clash = [n for n, v in (("--teacher-ensemble", a.teacher_ensemble),
                                ("--train-rollout-weight", a.train_rollout_weight > 0),
                                ("--co-members", a.co_members)) if v]
        if clash:
            p.error("--energy-score-samples is not implemented together with: " + ", ".join(clash))
        if a.energy_score_weight <= 0:
            p.error(f"--energy-score-weight must be > 0, got {a.energy_score_weight}")
    elif a.energy_score_replace or a.energy_score_weight != 1.0:
        p.error("--energy-score-weight / --energy-score-replace do nothing without "
                "--energy-score-samples M >= 2")
    if a.smoke:
        a.name = a.name if a.name != "flower" else "flower_smoke"
        a.epochs, a.max_steps, a.max_val_steps = 2, 6, 3
        if not a.init_from:
            # Under --init-from the checkpoint supplies the architecture, and these would
            # read as explicitly-given, differing flags and be refused by adopt_init_config.
            a.lifting_dim, a.num_heads, a.groups = 32, 8, 8
        a.stride, a.val_stride, a.batch_size = 200, 200, 4
    return a


if __name__ == "__main__":
    main(parse_args())
