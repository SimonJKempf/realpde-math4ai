"""The data layer: the 32x64 store under ``data/processed/``.

``scripts/subsample_data.py`` writes that store once -- whole trajectories,
float32, already strided to the scored resolution -- and everything downstream
reads it from here. Windows are cut at read time, so ``stride`` / ``t_in`` /
``t_out`` stay free parameters.

Arrays leave this module in the competition's own layout, ``(T, H, W, C)`` with
channels ``[u, v, p]``, in physical units. Two things deliberately do NOT happen
here:

* **No normalization.** It belongs to the model, so the statistics live in the
  checkpoint and a submission cannot run with a different set.
* **No folding of time into channels.** That is one architecture's convention;
  baking it in here would tie the loader to a single model.

The raw HDF5 release is not read here at all. ``subsample_data.py`` is the only
thing that touches ``data/raw/``, as a one-time preprocessing step.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import C, T_IN, T_OUT
from .paths import PROCESSED

# Files in the release that must never be trained or validated on.
# ``train_real/7575_0.h5`` holds the field data of ``6300_0`` instead of
# measurements of case 7575 (organizers' announcement, 2026-08-12). Its own
# ``re`` scalar still reads 7585, so nothing inside the file gives the
# duplication away -- only the u/v payload does. Case 7575 at 0 degrees cannot
# be re-measured, so the file stays in the release and is excluded in code.
# The other angles of attack at Re 7575 are fine, as is 6300_0 itself.
#
# REAL ONLY. The corruption is in the PIV measurement; ``train_sim/7575_0`` is
# a perfectly good simulation of that case and must not be dropped with it.
EXCLUDED_REAL = frozenset({"7575_0"})

# ``train_sim/13950_15.h5`` carries a broken pressure field: p in
# [+60.4, +241.4], mean +119.8, std 36.8, where all 99 other sim trajectories
# have mean p in [-1.015, +0.009] and std in [0.100, 0.290]. Its u and v are
# entirely normal, and its own ``re``/``aoa`` scalars are correct, so -- as with
# 7575_0 -- nothing but the payload gives it away.
#
# NOT a gauge-pressure convention: a constant shift preserves the standard
# deviation, and this inflates it 220x. After removing the mean the range is
# still [-59.4, +121.6] against a neighbour's [-1.9, +0.8], and the per-frame
# mean drifts 204 -> 84 across the trajectory. Nothing recovers it.
#
# Present in the raw HDF5, so it is the release, not our preprocessing.
#
# The damage is real: normalized against training statistics fitted on the good
# 99 (p std 0.358), +241 arrives as ~670 sigma. SelfWarp's flow head turns that
# into an enormous warping displacement and corrupts u and v through the shared
# trunk -- predicted u reached 12.8 against a target range of [-0.60, +1.51].
# Keeping this one cell in sim validation cost rel_l2 0.111 -> 0.180 and tke
# 0.46 -> 31.9. It sits in the held-out Re rows, so it never entered training.
EXCLUDED_SIM = frozenset({"13950_15"})

#: Per-split exclusions, so nothing is dropped from a split it does not apply to.
EXCLUDED = {"train_real": EXCLUDED_REAL, "train_sim": EXCLUDED_SIM}


@dataclass(frozen=True)
class Trajectory:
    """One trajectory in the processed store, as described by its manifest row.

    ``re_nominal`` comes from the filename and is the join key between real and
    sim; ``re_measured`` is what the file recorded and is the physical truth
    (they differ by ~0.2% for real data).
    """
    path: Path
    re_nominal: int
    re_measured: int
    aoa: int
    n_frames: int
    channels: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.re_nominal, self.aoa)

    def load(self) -> np.ndarray:
        """(T, H, W, C) float32, memory mapped -- never read whole."""
        return np.load(self.path, mmap_mode="r")


def list_runs(split: str = "train_real",
              exclude: frozenset[str] | None = None) -> list[Trajectory]:
    """Read ``data/processed/<split>/manifest.csv``.

    The manifest rather than a glob: it is written in the same pass that writes
    the arrays, so a half-finished ``subsample_data.py`` fails loudly here
    instead of silently training on a partial dataset.

    ``exclude`` defaults to whatever ``EXCLUDED`` holds for this split, and is
    re-applied even though ``subsample_data.py`` already applied it: the store
    on disk may have been built by an older run of that script, and the whole
    point of the exclusion is that a bad trajectory is invisible from its own
    contents -- so it is checked where the data is consumed, not only where it
    is produced.
    """
    if exclude is None:
        # The four PIV phase splits are train_real on a SHIFTED SUB-GRID, so they inherit
        # train_real's exclusions. Looking up "train_real_p01" in EXCLUDED misses and
        # returns an empty set, which would silently train a curriculum stage on 7575_0 --
        # the one row this project drops -- three times over. Match the PREFIX too.
        exclude = EXCLUDED.get(split, frozenset())
        if not exclude:
            for _k, _v in EXCLUDED.items():
                if split.startswith(_k + "_"):
                    exclude = _v
                    break
    root = PROCESSED / split
    manifest = root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found -- run `python scripts/subsample_data.py` first")

    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    runs = []
    for r in rows:
        if Path(r["file"]).stem in exclude:
            continue
        runs.append(Trajectory(path=root / r["file"],
                        re_nominal=int(r["re_nominal"]),
                        re_measured=int(r["re_measured"]),
                        aoa=int(r["aoa"]),
                        n_frames=int(r["n_frames"]),
                        channels=int(r["channels"])))
    if not runs:
        raise ValueError(f"{manifest} listed no usable trajectories")
    return runs


def measured_channels(runs: list[Trajectory]) -> int:
    """How many channels these runs actually measure: 2 for real, 3 for sim.

    The counterpart of ``kit/scoring.py``'s ``measured_channels``, which decides
    the same thing from the target array and drops the unmeasured ``p`` before
    computing any metric. Here it comes from the manifest instead, so a training
    loss can be masked to exactly the channels the leaderboard will look at.

    Mixed lists raise rather than returning a ``min``: concatenating sim and real
    runs is a deliberate decision about what to optimize, not something to
    resolve silently.
    """
    widths = {r.channels for r in runs}
    if len(widths) != 1:
        raise ValueError(f"runs disagree on channel count: {sorted(widths)} -- "
                         f"mask the loss explicitly rather than guessing")
    return widths.pop()


class HasRe(Protocol):
    """Anything carrying a nominal Reynolds number."""
    re_nominal: int


#: ``Trajectory`` is the only case type now, but the split is generic over anything
#: carrying a nominal Reynolds number.
_Case = TypeVar("_Case", bound=HasRe)


def split_by_condition(runs: list[_Case],
                       holdout_re: tuple[int, ...] = (13950, 26700),
                       ) -> tuple[list[_Case], list[_Case]]:
    """Split into (train, val) by whole (Re, aoa) cells.

    NEVER split by window: consecutive windows from one trajectory overlap by
    ~95%, so a random window split leaks the validation set into training and
    produces a validation score that means nothing.

    The default holds out one mid-range and one high Reynolds row, mirroring
    the two Re rows (15225, 27975) that have sim but no real data -- i.e. the
    interpolation and extrapolation regimes the private test set targets.
    """
    val = [r for r in runs if r.re_nominal in holdout_re]
    train = [r for r in runs if r.re_nominal not in holdout_re]
    return train, val


@dataclass(frozen=True)
class TrajectorySlice(Trajectory):
    """A contiguous frame range of one trajectory, usable anywhere a ``Trajectory`` is.

    ``n_frames`` is the length of the SLICE and ``load()`` returns exactly those
    frames, so every consumer -- ``WindowDataset``'s ``window_starts``,
    ``channel_stats``' normalizer fit, ``build_eval_arrays`` -- sees the slice and
    nothing else, with no changes of their own.

    That is the whole reason the restriction lives in ``load()`` rather than in
    the window index: ``channel_stats`` reads ``r.load()`` directly, so a split
    that only restricted window STARTS would still fit the normalizer on the
    held-out frames -- a leak no test on ``WindowDataset`` could see.
    """
    frame_start: int = 0

    def load(self) -> np.ndarray:
        """(n_frames, H, W, C) float32, memory mapped -- the slice, never the whole run."""
        return np.load(self.path, mmap_mode="r")[
            self.frame_start:self.frame_start + self.n_frames]


def split_temporal_gapped(runs: list[Trajectory], *, holdout_frames: int,
                          gap_frames: int, t_in: int = T_IN, t_out: int = T_OUT,
                          ) -> tuple[list[TrajectorySlice], list[TrajectorySlice]]:
    """Split into (train, val) IN TIME, holding out the tail of EVERY trajectory.

    Direction GTH. ``split_by_condition`` holds out whole Reynolds rows, so every
    number this project measures comes from 2 of 18 rows. This holds out the last
    ``holdout_frames`` frames of every trajectory instead, so the validation set
    spans ALL conditions -- at the price of being in-sample in condition and
    out-of-sample only in time. The two instruments have different blind spots;
    neither replaces the other, and their numbers are NOT comparable (different
    validation set, different persistence floor).

    Per trajectory of ``T`` frames, with ``H = holdout_frames``, ``G = gap_frames``::

        train  frames [0, T-H-G)          <- this split
        gap    frames [T-H-G, T-H)        <- NEITHER split: exactly G frames
        val    frames [T-H, T)            <- this split

    **The gap semantics, exactly.** ``G`` frames belong to neither split. So if
    ``i`` is the last frame of any training window and ``j`` the first frame of
    any validation window, ``j - i - 1 >= G`` (the count of frames strictly
    between them), with equality for the tightest pair; equivalently ``j - i >=
    G + 1``. No training window can overlap, touch, or come within ``G`` frames
    of a validation window, at any stride, because the two frame ranges are
    disjoint and separated by the gap and each window lies wholly inside its own
    range.

    Why a gap at all: consecutive frames of this flow are ~0.87 correlated and
    the correlation is still ~0.2 at lag 10, so a validation window that starts
    one frame after a training window ends is nearly the same flow state. The
    gap buys the short-range decorrelation; it cannot buy independence, because
    stationary vortex shedding keeps a phase-locked |rho| ~ 0.1-0.2 at every lag
    (measured on 3750/10125/13950/26700: rho(40) in [-0.07, +0.12], oscillatory
    beyond). Choose ``G`` past the decay of the short-range ramp and do not
    expect more from it.

    ``holdout_frames`` and ``gap_frames`` are KEYWORD-ONLY AND HAVE NO DEFAULTS,
    deliberately. The 2026-09-23 leak was a defaulted split argument at one call
    site of several; a split that cannot be called without stating its holdout
    cannot repeat it.

    Raises rather than dropping: a trajectory too short for both a training and a
    validation window would otherwise vanish from the split silently.
    """
    span = t_in + t_out
    if holdout_frames < span:
        raise ValueError(f"holdout_frames={holdout_frames} is shorter than one "
                         f"{span}-frame window: the holdout would be empty")
    if gap_frames < 0:
        raise ValueError(f"gap_frames={gap_frames} is negative")

    short = [f"{r.path.stem} ({r.n_frames} frames)" for r in runs
             if r.n_frames - holdout_frames - gap_frames < span]
    if short:
        raise ValueError(
            f"holdout {holdout_frames} + gap {gap_frames} leaves less than one "
            f"{span}-frame window to train on in: {', '.join(short)}. "
            f"Shorten the holdout or the gap rather than dropping the run.")

    train, val = [], []
    for r in runs:
        n_train = r.n_frames - holdout_frames - gap_frames
        train.append(TrajectorySlice(path=r.path, re_nominal=r.re_nominal,
                                     re_measured=r.re_measured, aoa=r.aoa,
                                     n_frames=n_train, channels=r.channels,
                                     frame_start=0))
        val.append(TrajectorySlice(path=r.path, re_nominal=r.re_nominal,
                                   re_measured=r.re_measured, aoa=r.aoa,
                                   n_frames=holdout_frames, channels=r.channels,
                                   frame_start=r.n_frames - holdout_frames))
    return train, val


#: The scoring scripts' ``--gapped-holdout`` given bare: (holdout_frames, gap_frames).
#: The same two numbers as train.py's ``--gth-holdout-frames`` / ``--gth-gap`` defaults
#: and as the cc_gth_* pilot (queue 2026-09-24-cc-GTH-pilot), so a bare flag scores a
#: pilot model on exactly the tail it held out.
GTH_DEFAULT = (100, 40)


def parse_gapped_holdout(values) -> tuple[int, int] | None:
    """``--gapped-holdout [N G]`` (argparse ``nargs="*"``) -> ``None`` or ``(N, G)``.

    Shared by score_local.py, fit_bounds.py and score_twin.py so the three cannot
    disagree on what a bare flag means:

    * ``None`` (flag absent)      -> ``None``: the condition-holdout path, untouched.
    * ``[]``   (flag given bare)  -> ``GTH_DEFAULT``.
    * ``[N, G]``                  -> ``(N, G)``.

    One value or more than two raise ``SystemExit``: a lone ``N`` would silently
    pair with a default gap, and the gap is the whole point of the split.
    """
    if values is None:
        return None
    vals = [int(v) for v in values]
    if not vals:
        return GTH_DEFAULT
    if len(vals) != 2:
        raise SystemExit(f"--gapped-holdout takes no values (-> {list(GTH_DEFAULT)}) or exactly "
                         f"two, N G (holdout frames, gap frames); got {vals}")
    return vals[0], vals[1]


def window_starts(n_frames: int, t_in: int = T_IN, t_out: int = T_OUT,
                  stride: int = 10) -> range:
    """First-frame indices of every complete window in a trajectory of n frames.

    The ``+ 1`` is load-bearing: without it the last valid window of every
    trajectory is silently dropped. Written once and shared, because this is
    the arithmetic that quietly disagrees between two copies.
    """
    return range(0, max(n_frames - (t_in + t_out) + 1, 0), stride)


def _fit_channels(w: np.ndarray, channels: int = 3) -> np.ndarray:
    """Pad the channel axis to 3 channels.

    Real PIV has no pressure field, so the store holds 2 channels while sim
    holds 3. Padding a zero ``p`` plane here is what lets a sim-pretrained
    3-channel model fine-tune on real data through this same loader -- and it
    matches what the grader hands ``predict()``, which is always 3 channels
    with ``p`` zeroed.
    """
    have = w.shape[-1]
    if have < channels:
        pad = np.zeros((*w.shape[:-1], channels - have), dtype=w.dtype)
        return np.concatenate([w, pad], axis=-1)
    if have > channels:
        return w[..., :channels]
    return w


class WindowDataset(Dataset):
    """Fixed-stride ``(x, y)`` windows cut from a list of runs.

    Yields ``x`` of shape ``(t_in, H, W, channels)`` and ``y`` of
    ``(t_out, H, W, channels)`` as float32 tensors in physical units.

    ``mask_bank`` (2026-09-18) is an optional ``(M, H, W)`` bool array of REAL
    PIV data holes. When given, every window is multiplied by one of them,
    drawn per window -- so a SIM pretrain sees the same kind of hole the real
    fine-tune will, instead of learning "infer around a hole and emit zeros in
    it" from scratch in 30 fine-tune epochs. It changes the pretrain's INPUT
    DISTRIBUTION only: no loss term and no target semantics move, so it cannot
    rotate the rel_l2/tke frontier the way a loss change does.

    ``extra_out`` (direction ROLL, 2026-09-24) lengthens every window by that many
    frames and yields ``(x, y, y2)`` with ``y2`` of ``(extra_out, H, W, channels)``:
    the true frames AFTER ``y``, for a training-time rollout loss. 0 (default)
    is the ordinary ``(x, y)`` dataset, unchanged.

    ``stride`` is not only a speed knob. At stride 1 consecutive windows share
    39 of 40 frames, so an epoch gets 40x longer while carrying almost no extra
    information. Keep it well above 1 and let ``split_by_condition`` -- never a
    random window split -- decide what the model is allowed to see.
    """

    def __init__(self, runs: list[Trajectory], t_in: int = T_IN, t_out: int = T_OUT,
                 stride: int = 20, channels: int = 3, mask_bank: np.ndarray | None = None,
                 mask_seed: int = 0, mask_channel: bool = False, extra_out: int = 0):
        if not runs:
            raise ValueError("no runs given")
        if channels > C:
            raise ValueError(f"channels={channels} exceeds the {C} the problem has")
        if extra_out < 0:
            raise ValueError(f"extra_out={extra_out} is negative")

        self.runs = runs
        self.t_in, self.t_out, self.channels = t_in, t_out, channels
        self.stride = stride
        self.mask_bank = mask_bank
        self.mask_seed = mask_seed
        self.mask_channel = mask_channel
        # Direction ROLL (2026-09-24): ``extra_out > 0`` cuts windows of
        # t_in + t_out + extra_out frames and yields a THIRD tensor, y2, holding the
        # extra frames [t_in + t_out, t_in + t_out + extra_out) -- the truth the
        # training-time rollout (train.py --train-rollout-weight) scores its second
        # pass against. Windows are only cut where the longer span fits, so the
        # count drops (868 frames at stride 10: 83 -> 81 for extra_out = 20). At 0,
        # the default, span and index are what they always were and __getitem__
        # returns the same 2-tuple, bit for bit.
        self.extra_out = extra_out

        # Flat (run index, first frame) list, built once up front so __len__ is
        # exact and shuffling is a permutation of real windows. Per-run,
        # because trajectory lengths vary (868 frames for most, but also 607,
        # 492 and 282).
        self.index: list[tuple[int, int]] = [
            (i, s)
            for i, r in enumerate(runs)
            for s in window_starts(r.n_frames, t_in, t_out + extra_out, stride)
        ]
        if not self.index:
            raise ValueError(
                f"no windows: no run is at least {t_in + t_out + extra_out} frames long")

        # Opened on first access and cached per process. NOT in __init__: with
        # num_workers > 0 the dataset is forked into each worker, and each
        # worker should own its own mappings rather than inherit one.
        self._mm: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def key_of(self, k: int) -> tuple[int, int]:
        """The (Re, aoa) cell window ``k`` came from -- for per-cell analysis."""
        return self.runs[self.index[k][0]].key

    def _array(self, i: int) -> np.ndarray:
        if i not in self._mm:
            self._mm[i] = self.runs[i].load()
        return self._mm[i]

    def __getitem__(self, k: int):
        i, s = self.index[k]
        span = self.t_in + self.t_out + self.extra_out

        # np.array, not np.asarray: the store is float32 already, so asarray
        # would hand back a read-only view onto the mmap. torch.from_numpy on
        # that warns and returns a tensor aliasing the mapping. Copy the 40
        # frames out -- 320 KB -- and be done with it.
        w = np.array(self._array(i)[s:s + span], dtype=np.float32)
        w = _fit_channels(w, self.channels)
        if self.mask_channel and w.shape[-1] > 2:
            # per-frame, not "measured in every frame": 11.6% of pixels are u-zero in SOME
            # frames only, and the per-frame form tells the model exactly which pixels carry
            # a measurement at each step. INPUT FRAMES ONLY -- the slice w[:t_in] matters: the
            # window is one array that is later split into x and y, so writing w[..., 2] would put
            # the mask in the TARGET as well. Harmless today (the loss slices to the measured
            # channels) but wrong the moment measured becomes 3, and it contradicted this comment.
            # Caught by tests/test_dataset.py::MaskChannel, which asserts y is untouched.
            w[:self.t_in, ..., 2] = (w[:self.t_in, ..., 0] != 0).astype(w.dtype)
        if self.mask_bank is not None:
            # One real hole per window, drawn deterministically from (seed, k) so
            # an epoch is reproducible and every worker agrees. Applied to input
            # AND target: the hole is a property of the measurement, not of time.
            m = self.mask_bank[(hash((self.mask_seed, k)) & 0x7fffffff) % len(self.mask_bank)]
            w = w * m[None, :, :, None]

        if self.extra_out:
            # (x, y, y2): y2 is cut from the SAME array by the same slicing, so it
            # carries the same channel padding, the same mask_bank hole and -- like
            # y -- no mask_channel plane (that is written into w[:t_in] only above).
            # Physical units like everything else here; the model normalises.
            t1 = self.t_in + self.t_out
            return (torch.from_numpy(w[:self.t_in]), torch.from_numpy(w[self.t_in:t1]),
                    torch.from_numpy(w[t1:]))
        return torch.from_numpy(w[:self.t_in]), torch.from_numpy(w[self.t_in:])


def real_mask_bank(runs: list[Trajectory] | None = None) -> np.ndarray:
    """(M, H, W) bool: the measured-point masks of the real trajectories.

    ``u == 0`` marks a point outside the PIV field of view or inside the airfoil
    body -- the same convention ``window_scale`` and the scorer use. The mask is
    near-static within a run (the body moves with angle of attack, not with
    time), so one mask per trajectory taken from its first frame is enough, and
    taking it from frame 0 costs one frame of IO per run.
    """
    if runs is None:
        runs = list_runs("train_real")
    out = []
    for r in runs:
        a = r.load()
        out.append(np.asarray(a[0, ..., 0]) != 0.0)
    return np.stack(out)


def build_eval_arrays(runs: list[Trajectory], stride: int = 50,
                      limit_per_traj: int | None = None,
                      t_in: int = T_IN, t_out: int = T_OUT,
                      channels: int = C, return_keys: bool = False):
    """Stack windows from several runs into ``(N, T, H, W, C)`` arrays.

    This is what ``scripts/score_local.py`` feeds to a submission's
    ``predict()`` and to ``kit/scoring.py`` as the reference targets, so it
    defaults to the grader's 3 channels -- real data is zero-padded in ``p``,
    exactly as the grader hands it over -- rather than the 2 training reads.

    ``limit_per_traj`` caps windows per run, not overall: a global cap would
    take every window from the first trajectory and none from the last.
    """
    ds = WindowDataset(runs, t_in=t_in, t_out=t_out, stride=stride,
                       channels=channels)

    taken: dict[int, int] = {}
    keep = []
    for k, (i, _s) in enumerate(ds.index):
        if limit_per_traj is not None and taken.get(i, 0) >= limit_per_traj:
            continue
        taken[i] = taken.get(i, 0) + 1
        keep.append(k)

    xs, ys = zip(*(ds[k] for k in keep))
    x = np.stack([a.numpy() for a in xs]); y = np.stack([a.numpy() for a in ys])
    if return_keys:
        # (Re, aoa) per window, in array order -- for per-row scoring.
        return x, y, [ds.key_of(k) for k in keep]
    return x, y


def channel_stats(runs: list[Trajectory], channels: int = 3,
                  max_frames: int = 200,
                  window_norm: bool = False,
                  window_scale_mode: str = "u") -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (mean, std) over ``runs``, for a model's normalizer.

    Fit on the TRAINING runs only. Statistics taken over the held-out cells
    leak the validation set, quietly, in a way no score will reveal.

    ``window_norm`` must match the model's ``FlowerConfig.window_norm``: that
    model divides by the window's velocity scale *before* applying these
    statistics, so they have to describe the divided fields or they describe
    the wrong space entirely. Here the divisor is each trajectory's own
    ``mean|u|`` rather than each window's -- this is stationary vortex
    shedding, so the two agree far past the precision a normalizer needs.
    Expect roughly ``mean [0.996, ~0]  std [~0.39, ~0.09]`` on real data.

    Reads at most ``max_frames`` evenly spaced frames per run: this is
    stationary vortex shedding, so a couple hundred frames pin the mean and std
    far past the precision that matters. Accumulated in float64 -- the naive
    sum-of-squares variance is badly conditioned here, because u has a mean of
    ~0.15 and a std of ~0.1.

    A channel that is identically zero (``p`` in real data) gets std 1, not 0,
    matching what the benchmark's own GaussianNormalizer does.
    """
    n = 0
    total = np.zeros(channels, np.float64)
    total_sq = np.zeros(channels, np.float64)

    for r in runs:
        a = r.load()
        step = max(1, a.shape[0] // max_frames)
        w = _fit_channels(np.asarray(a[::step], dtype=np.float64), channels)
        if window_norm:
            # Imported in the function body, not at module scope: this file is
            # the data layer and stays clear of the model layer (and of pulling
            # the vendored Flower into every subsample_data.py run). But the
            # floor is the model's, and a second copy of it here would be free
            # to drift out of agreement with the divisor the model applies.
            from .model import U_FLOOR
            u = w[..., 0]
            measured = u != 0
            # The divisor MUST be the same quantity the model divides by, or
            # these statistics describe a different space than the one the
            # network sees. window_scale_mode is therefore threaded from the
            # model config rather than defaulted here.
            if window_scale_mode == "u":
                scale = np.abs(u[measured]).mean()
            elif window_scale_mode == "speed":
                v = w[..., 1]
                scale = np.sqrt((u[measured] ** 2 + v[measured] ** 2).mean())
            else:
                raise ValueError(f"unknown window scale mode "
                                 f"{window_scale_mode!r}; use 'u' or 'speed'")
            w = w / max(scale, U_FLOOR)
        w = w.reshape(-1, channels)
        n += w.shape[0]
        total += w.sum(axis=0)
        total_sq += (w ** 2).sum(axis=0)

    if n == 0:
        raise ValueError("no frames read")

    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 0.0))
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)
