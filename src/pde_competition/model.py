"""Our Track 1 model: Flower wrapped for this competition's tensor layout.

The Flower source is vendored at ``_vendor/flower_standalone.py`` -- see that
directory's README for provenance. Import it through the package (as below) and
never as a bare top-level module: ``_vendor/__init__.py`` is what puts the
vendored einops on sys.path.

Three mismatches sit between upstream Flower and this competition, and this
module is where each is resolved once:

1. **Layout.** Flower is a 2-D image-to-image operator: ``(B, C, H, W)`` in,
   ``(B, C, H, W)`` out, channels first. The competition speaks
   ``(N, T, H, W, C)``, channels last, with a time axis. We fold time into
   channels -- ``dim_in = t_in * c_in``, ``dim_out = t_out * c_out`` -- so one
   forward pass maps the whole 20-frame input to the whole 20-frame output.
   No autoregressive rollout, so no error accumulation and one kernel launch
   per window (which is also what ``time_score`` rewards).

2. **Channels.** The default is ``c_in = c_out = 3``, matching the ``[u, v, p]``
   arrays the grader hands ``predict()`` and the pressure field sim data
   actually carries. Real PIV does not measure ``p``: it is identically 0, and
   ``scoring.measured_channels`` drops it before computing any metric. So for
   real-only training ``c_in = c_out = 2`` is a sound economy -- it spends no
   capacity learning to reproduce an all-zero target that is never scored.

3. **Normalization.** Fields are O(0.1) with a mean far from zero; feeding them
   raw trains badly. The statistics live in registered buffers, so they are
   saved and loaded as part of the ``state_dict`` and a submission physically
   cannot run with a different set -- the failure mode the kit's FNO example
   calls "the single most common reason a correctly-loaded baseline still
   scores badly". ``FlowerConfig.window_norm`` adds a second, per-window stage
   in front of it: real PIV is dimensional and sim is not, so a global
   normalizer alone leaves a 5.89x amplitude spread across the Reynolds rows.

Flower also bakes ``spatial_resolution`` into its coordinate grids and warping
layers at construction, so an instance is tied to one input size. Ours is
32x64, the scored resolution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._vendor.flower_standalone import Flower
from .geometry import C, H_S, T_IN, T_OUT, W_S


#: Floor on the per-window velocity scale, so a degenerate window cannot divide
#: by ~0. Real ``mean|u|`` bottoms out at 0.049 (Re 3750), two orders above this.
U_FLOOR = 1e-3


def window_scale(x, mode: str = "u"):
    """Per-window free-stream velocity scale, over measured points.

    ``mode="u"`` is ``mean|u|``, the original and the default, so every existing
    checkpoint keeps its meaning. ``mode="speed"`` is ``sqrt(mean(u^2 + v^2))``.

    Why "speed" exists: ``mean|u|`` is the streamwise component, not the flow
    speed, so it understates the free stream by roughly ``cos(aoa)`` -- and the
    dataset spans aoa 0, 5, 10, 15, 20 degrees, so the understatement VARIES
    with the condition (about 6% at 20 degrees). That leaves an angle-dependent
    bias in the normalized space, which is the one thing window-norm is supposed
    to remove. Both are one scalar per window and both preserve the u:v ratio,
    so "speed" changes the level and not the direction information.

    Takes ``(B, T, H, W, C)`` and returns the scale with the trailing axes kept,
    ``(B, 1, 1, 1, 1)``, ready to divide ``x``. Torch only -- numpy callers
    (``scripts/fit_bounds.py``) wrap their array rather than keeping a second
    implementation, since the whole point is that there is one definition of
    what the velocity scale is.

    Computed from the INPUT window, because that is all ``predict()`` ever has.
    That is legitimate here: the input and target windows agree on this quantity
    to r = 0.9995, measured over the held-out cells in ``scripts/fit_bounds.py``.

    ``u == 0`` marks points outside the PIV field of view or inside the airfoil
    body, and that mask moves with angle of attack -- so it has to be excluded
    rather than averaged in, or the scale would carry the mask geometry as well
    as the flow speed.
    """
    u = x[..., 0]
    measured = u != 0
    n = measured.sum(dim=(1, 2, 3))
    if mode == "u":
        total = (u.abs() * measured).sum(dim=(1, 2, 3))
        scale = total / n.clamp(min=1)
    elif mode == "speed":
        # Same mask: u == 0 is what marks a point as unmeasured, and v can be
        # legitimately 0 where the flow is measured.
        sq = (x[..., 0] ** 2 + x[..., 1] ** 2) * measured
        scale = torch.sqrt(sq.sum(dim=(1, 2, 3)) / n.clamp(min=1))
    else:
        raise ValueError(f"unknown window scale mode {mode!r}; use 'u' or 'speed'")
    return scale.clamp(min=U_FLOOR).reshape(-1, 1, 1, 1, 1)


#: Convection speed of the fluctuation, in pixels per frame per unit mean|u|.
#: Measured 2026-09-18 by cross-correlating the u-fluctuation at lag 10 over 40 real
#: trajectories (all aoa, Re 3750-26700): 4.95 +- 0.3, with only a weak aoa trend.
#: At Re 26700 that is 1.68 px/frame, so structures cross 32 px of a 64 px domain over
#: the 20 input frames while the streamwise correlation length is 3-6 px -- the frames
#: at a fixed pixel are ~10 statistically independent eddies. Flower's first operation
#: is a 1x1 conv over all t_in*c_in channels, so nothing can re-align them afterwards.
PX_PER_FRAME_PER_U = 4.95


def shear_window(x: torch.Tensor, shift_px: torch.Tensor) -> torch.Tensor:
    """Translate each frame of ``(B, T, H, W, C)`` downstream by ``shift_px`` (B, T) pixels.

    Positive shift moves content toward +x. Uses one ``grid_sample`` over the folded
    ``(B*T, C, H, W)`` view with ``padding_mode='border'`` so the inflow/outflow edges
    replicate rather than wrap -- the domain is periodic in neither direction.
    """
    b, t, h, w, c = x.shape
    q = x.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
    yy, xx = torch.meshgrid(torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
                            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
                            indexing="ij")
    # sample from (x - shift): moving the sample point upstream moves content downstream
    dx = (2.0 * shift_px.reshape(b * t, 1, 1) / max(w - 1, 1))
    grid = torch.stack((xx.unsqueeze(0) - dx, yy.unsqueeze(0).expand(b * t, h, w)), dim=-1)
    out = F.grid_sample(q, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.reshape(b, t, c, h, w).permute(0, 1, 3, 4, 2)


@dataclass(frozen=True)
class FlowerConfig:
    """Everything needed to rebuild the network. Saved beside the weights.

    ``lifting_dim`` / ``n_levels`` are the size knobs: channels double each
    level, so cost scales roughly as ``lifting_dim^2``. Upstream's default
    (160 / 4, 17.3 M params) is aimed at 128x128 datasets far larger than 81
    real trajectories; 64 / 4 is 2.8 M params and a saner starting point here.
    """
    t_in: int = T_IN
    t_out: int = T_OUT
    c_in: int = 3                     # [u, v, p]; p=0 for real
    c_out: int = 3
    resolution: tuple[int, int] = (H_S, W_S)
    lifting_dim: int = 64
    n_levels: int = 4
    num_heads: int = 16
    groups: int = 16
    dropout_rate: float = 0.0
    predict_delta: bool = False       # learn the change from the last input frame
    temporal_lift: int = 0            # depthwise conv over TIME before the lift; 0 = off
    rans: bool = False                # split the output into a time-constant mean + a fluctuation

    # Upstream's default is ["PERIODIC"], which derive_padding_strategies turns
    # into ['periodic', 'periodic'] -- BOTH axes wrap. An airfoil in a channel
    # is periodic in neither: streamwise has inflow/outflow, cross-stream has
    # walls. Wrapping means SelfWarp can sample the outlet when its learned
    # displacement walks off the inlet.
    #
    # This is effectively a BOOLEAN PER AXIS, not a physics vocabulary.
    # derive_padding_strategies is
    #     ['periodic' if bc.upper() == 'PERIODIC' else 'zeros' for bc in bc_list]
    # so only the literal "PERIODIC" is recognised and EVERY other string --
    # "ZEROS", "DIRICHLET", a typo -- lands on 'zeros'. Upstream's own docstring
    # says as much: "Only PERIODIC is treated differently". 'border' and
    # 'reflection' exist inside custom_grid_sample2d but are unreachable from
    # here, so we say "ZEROS" for what we actually get rather than inventing a
    # boundary-condition name the code does not honour.
    #
    # Order is per spatial dim, i.e. [H, W] for our (32, 64). Left at the
    # upstream default so every existing checkpoint reloads unchanged.
    boundary_condition_types: tuple[str, ...] = ("PERIODIC",)

    # Divide each window by its own free-stream speed before the global
    # normalizer. Real PIV is dimensional and sim is non-dimensionalized, so
    # real ``mean|u|`` climbs 0.049 -> 0.29 across the Reynolds rows (5.89x)
    # while sim sits at ~0.87 for every row. Under one global (mean, std) that
    # puts real Re 3750 at -0.92 sigma and Re 26700 at +1.22 sigma -- the
    # normalized input spends its whole dynamic range encoding Reynolds -- and
    # puts every sim row at ~0. Dividing by U collapses the real spread to
    # 1.01x and lands both datasets in the same space.
    #
    # Off by default so every existing checkpoint reloads unchanged, and
    # inherited through --init-from like the rest of the architecture.
    window_norm: bool = False

    # Which scalar window_norm divides by. "u" is mean|u| (the original, and the
    # default so every existing checkpoint reloads unchanged); "speed" is
    # sqrt(mean(u^2 + v^2)), which does not carry mean|u|'s angle-of-attack
    # bias. Only consulted when window_norm is on.
    window_scale_mode: str = "u"

    #: Lagrangian input alignment (2026-09-18). "off" ships today's behaviour. "input"
    #: de-shears the input window into the frame co-moving with the flow before the lift,
    #: so the 20 frames at a pixel are one parcel's history instead of ~10 unrelated eddies.
    #: The shift law is measured, not learned, so it is exactly right at Reynolds numbers
    #: the model has never seen -- which is what the hidden rows require.
    #: Write the per-frame measured mask (u != 0) into input channel 2 (2026-09-19).
    #: On real PIV that channel is identically zero -- a dead input -- while "unmeasured"
    #: is signalled only by u == 0, which model.py:93-94 itself flags as ambiguous (v is
    #: legitimately 0 in places). The mask also encodes the body outline, which MOVES with
    #: angle of attack, so it is per-row geometry the network currently has to infer.
    #: Costs nothing: the plane exists, the loss never sees it (run_epoch slices to the
    #: measured channels), and the grader already hands us the array it is built from.
    mask_channel: bool = False
    lagrangian: str = "off"
    lagrangian_c: float = PX_PER_FRAME_PER_U

    # How many frames the network emits per forward pass.
    #
    # 0 (default) is what this project has always done: one pass emits the whole
    # t_out horizon, no rollout, no error accumulation, one kernel launch --
    # which is also what time_score rewards.
    #
    # k > 0 emits k frames per pass and applies the network t_out/k times,
    # sliding the input window and feeding its own predictions back. That is the
    # mode the Flowers paper actually evaluates (they do 4->1 next-step and then
    # roll out), and their headline claim is that rollout error stays low where
    # baselines blow up -- Rayleigh-Benard 1:20 rollout 2.17 against 12.5-39.0.
    # We have been using the architecture in a mode its authors never tested.
    #
    # Costs t_out/k forward passes at inference, which time_score charges for.
    rollout_frames: int = 0
    # ROLLOUT FEED (2026-09-26, reviewer bug). 0 = legacy: the next window gets ALL c_out channels
    # of the step just predicted. On real data that feeds back the pressure output, which the loss
    # never trains (measured = 2), into input columns that only ever saw p = 0, and fills the PIV
    # holes the input had as exact zeros. m > 0: feed back only the first m channels, multiplied
    # by the ORIGINAL input's last-frame measured mask (u != 0; holes are a property of the sensor),
    # and carry channels [m, c_in) over from that last input frame -- the convention
    # scripts/train.py:rollout_input has always used for the training-only rollout. Changes no
    # parameter shape; the first pass is identical either way.
    rollout_feed: int = 0

    def __post_init__(self) -> None:
        if self.rollout_feed and not (self.rollout_frames and 0 < self.rollout_feed <= self.c_out):
            raise ValueError(f"rollout_feed={self.rollout_feed} needs rollout_frames > 0 and "
                             f"0 < rollout_feed <= c_out={self.c_out} (rollout_frames={self.rollout_frames})")
        if self.rollout_frames:
            if self.rollout_frames < 0 or self.t_out % self.rollout_frames:
                raise ValueError(
                    f"rollout_frames must divide t_out={self.t_out}, "
                    f"got {self.rollout_frames}")
            if self.c_in != self.c_out:
                # A rollout step feeds its own output back in as input, so the
                # two widths have to match or the window cannot be slid.
                raise ValueError(
                    f"rollout needs c_in == c_out, got {self.c_in}/{self.c_out}")
        if self.window_scale_mode not in ("u", "speed"):
            raise ValueError(f"window_scale_mode must be 'u' or 'speed', "
                             f"got {self.window_scale_mode!r}")
        # p scales as U^2, not U, so it cannot ride the same factor. Real data
        # has no p to get wrong, which is exactly why this has to be checked
        # here: at c_in=3 on real data the mistake is invisible in every
        # metric, because scoring drops the channel.
        if self.window_norm and (self.c_in > 2 or self.c_out > 2):
            raise ValueError(
                f"window_norm needs c_in = c_out = 2, got {self.c_in}/{self.c_out}: "
                f"pressure scales as U^2 and cannot share the velocity factor")
        # Fail here with a sentence, rather than 200 frames down inside a
        # rearrange or a GroupNorm.
        divisor = 2 ** (self.n_levels - 1)
        for name, d in zip("HW", self.resolution):
            if d % divisor:
                raise ValueError(
                    f"resolution {name}={d} must be divisible by 2^(n_levels-1)"
                    f"={divisor}; lower n_levels or change the resolution")
        for level in range(self.n_levels):
            ch = self.lifting_dim * 2 ** level
            if ch % self.num_heads:
                raise ValueError(f"level {level} has {ch} channels, not divisible "
                                 f"by num_heads={self.num_heads}")
            if ch % self.groups:
                raise ValueError(f"level {level} has {ch} channels, not divisible "
                                 f"by groups={self.groups}")

        # Mirror custom_grid_sample2d's own constraint so an illegal pair fails
        # at construction rather than partway through a forward pass.
        pad = ["periodic" if b.upper() == "PERIODIC" else "zeros"
               for b in self.boundary_condition_types]
        if len(pad) == 1:
            pad *= 2
        if len(pad) != 2:
            raise ValueError(f"boundary_condition_types needs 1 or 2 entries for "
                             f"2-D, got {self.boundary_condition_types}")
        if pad[0] != pad[1] and "periodic" not in pad:
            raise ValueError(
                f"{self.boundary_condition_types} -> {pad}: when the axes differ "
                f"one must be PERIODIC (custom_grid_sample2d rejects mixed "
                f"non-periodic padding)")

    @property
    def dim_in(self) -> int:
        return self.t_in * self.c_in

    @property
    def frames_per_pass(self) -> int:
        """Frames emitted by one forward pass of the core network."""
        return self.rollout_frames or self.t_out

    @property
    def dim_out(self) -> int:
        # +c_out for the RANS mean field, which is time-constant and so costs one frame's
        # worth of channels however long the horizon is.
        return self.frames_per_pass * self.c_out + (self.c_out if self.rans else 0)


class FlowerS2S(nn.Module):
    """Flower as a sequence-to-sequence map on ``(N, T, H, W, C)`` fields.

    Input and output are in **physical units**. Normalization, the layout fold
    and the optional delta parameterization all happen inside, so callers --
    the training loop, local scoring, the submission -- only ever handle raw
    fields and cannot disagree about the convention.
    """

    # TRUNCATED BACKPROP THROUGH THE ROLLOUT (2026-09-25). A plain attribute, deliberately NOT a
    # FlowerConfig field: it changes only how gradients flow during training, never a forward
    # value, so it must not reach the checkpoint config the shipped module rebuilds from.
    # train.py sets it on the live model under --rollout-detach-passes; everywhere else
    # (checkpoint loads, submission, eval) it is this class default, False. Read only inside
    # forward's rollout loop, and only while grad is enabled.
    rollout_detach: bool = False

    def __init__(self, cfg: FlowerConfig = FlowerConfig(),
                 mean: np.ndarray | None = None, std: np.ndarray | None = None):
        super().__init__()
        self.cfg = cfg
        self.core = Flower(
            dim_in=cfg.dim_in,
            dim_out=cfg.dim_out,
            n_spatial_dims=2,
            spatial_resolution=list(cfg.resolution),
            lifting_dim=cfg.lifting_dim,
            n_levels=cfg.n_levels,
            num_heads=cfg.num_heads,
            groups=cfg.groups,
            dropout_rate=cfg.dropout_rate,
            boundary_condition_types=list(cfg.boundary_condition_types),
        )
        # Buffers, not attributes: they belong to the state_dict and move with
        # .to(device). Identity until set_stats() is called.
        if cfg.temporal_lift:
            if cfg.temporal_lift % 2 != 1:
                raise ValueError(f"temporal_lift must be odd so the window stays centred, got {cfg.temporal_lift}")
            self.tlift = nn.Conv1d(1, 1, cfg.temporal_lift, padding=cfg.temporal_lift // 2, bias=False)
            # IDENTITY INIT: a delta kernel, so an untrained model reproduces today's exactly
            # and every change from here is something the optimiser chose.
            with torch.no_grad():
                self.tlift.weight.zero_()
                self.tlift.weight[0, 0, cfg.temporal_lift // 2] = 1.0
        self.register_buffer("mean", torch.zeros(cfg.c_in))
        self.register_buffer("std", torch.ones(cfg.c_in))
        if mean is not None and std is not None:
            self.set_stats(mean, std)

    def set_stats(self, mean, std) -> None:
        mean = torch.as_tensor(np.asarray(mean, np.float32))
        std = torch.as_tensor(np.asarray(std, np.float32))
        if mean.shape != (self.cfg.c_in,) or std.shape != (self.cfg.c_in,):
            raise ValueError(f"stats must be ({self.cfg.c_in},), got "
                             f"{tuple(mean.shape)} and {tuple(std.shape)}")
        if (std <= 0).any():
            raise ValueError(f"non-positive std {std.tolist()}")
        self.mean.copy_(mean.to(self.mean.device))
        self.std.copy_(std.to(self.std.device))

    @property
    def stats_are_set(self) -> bool:
        return not (torch.all(self.mean == 0) and torch.all(self.std == 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, t_in, H, W, c_in) physical -> (B, t_out, H, W, c_out) physical."""
        cfg = self.cfg
        b, t, h, w, c = x.shape
        if (t, h, w, c) != (cfg.t_in, *cfg.resolution, cfg.c_in):
            raise ValueError(
                f"expected (B, {cfg.t_in}, {cfg.resolution[0]}, {cfg.resolution[1]}, "
                f"{cfg.c_in}), got {tuple(x.shape)}")

        # Divide by the window's own velocity scale FIRST, then centre. The
        # other order -- (x - mean)/std/U -- does not remove the Reynolds
        # dependence, because a global mean subtracted before the division
        # leaves a U-dependent offset behind.
        #
        # Detached: U is a property of the input, and letting gradients flow
        # through the scaling buys nothing and destabilises the warping layers.
        if not cfg.rollout_frames:
            return self._one_pass(x)

        # ROLLOUT. Emit frames_per_pass at a time and slide the input window,
        # feeding predictions back. The window is kept in PHYSICAL units and
        # renormalized each step, deliberately: with window_norm the velocity
        # scale is a property of the current window, and after a few steps that
        # window is mostly the model's own output, so reusing the first step's
        # scale would silently drift.
        #
        # rollout_detach (training only): the frames fed back into the next window are
        # detached, so each pass is trained on its own loss with inputs that contain the
        # earlier passes' outputs, but no gradient flows back through earlier passes -- the
        # feedback loop that diverges at lr 3e-3 is cut. The returned step keeps its graph,
        # and every forward VALUE is identical either way (detach changes no number).
        k = cfg.rollout_frames
        detach = self.rollout_detach and torch.is_grad_enabled()
        m = cfg.rollout_feed
        if m:
            last = x[:, -1:]                                          # (B, 1, H, W, c_in)
            meas = (last[..., :1] != 0).to(x.dtype)                   # holes of the ORIGINAL input
            carry = last[..., m:].expand(-1, k, -1, -1, -1) if x.shape[-1] > m else None
        window, preds = x, []
        for _ in range(cfg.t_out // k):
            step = self._one_pass(window)                # (B, k, H, W, c_out)
            preds.append(step)
            fb = step.detach() if detach else step
            if m:
                fb = fb[..., :m] * meas
                if carry is not None:
                    fb = torch.cat((fb, carry), dim=-1)
            window = torch.cat((window[:, k:], fb), dim=1)
        return torch.cat(preds, dim=1)

    def _one_pass(self, x: torch.Tensor) -> torch.Tensor:
        """One forward pass: (B, t_in, H, W, c_in) -> (B, frames_per_pass, ...)."""
        cfg = self.cfg
        b, _, h, w, _ = x.shape

        u_scale = (window_scale(x, cfg.window_scale_mode).detach()
                   if cfg.window_norm else 1.0)

        if cfg.lagrangian == "input" and cfg.lagrangian_c:
            # de-shear to the LAST input frame's position: frame i moves downstream by
            # (t_in-1-i) * c * mean|u| px, so every frame lands where the last one is.
            speed = window_scale(x, "u").detach().reshape(-1)          # (B,) mean|u|
            k = torch.arange(x.shape[1] - 1, -1, -1, device=x.device, dtype=x.dtype)
            shift = cfg.lagrangian_c * speed[:, None] * k[None, :]     # (B, t_in)
            meas = (x[..., :1] != 0).to(x.dtype)                       # holes are a property of the sensor
            x = shear_window(x, shift) * meas

        z = (x / u_scale - self.mean) / self.std              # broadcast over C
        if cfg.temporal_lift:
            # THE ADVECTION PROBLEM, attacked without moving any data. The core's first
            # operation is a 1x1 conv over all t_in*c_in channels, so at a fixed pixel the
            # 20 input frames are mixed globally -- but fluctuations convect at ~4.95
            # px/frame per unit mean|u|, so at Re 26700 those 20 frames are ~10 UNRELATED
            # eddies and the correlation length is 3 px. W178/W180 tried to fix this by
            # shearing the window into a Lagrangian frame and that closed (a perfectly
            # padded version still cost ~1.9%, and the shear fabricated 25% of the input).
            # This instead gives the network LOCAL temporal structure to form features from
            # before the global mix: one depthwise conv over t, per (channel, pixel).
            # Identity-initialised, so an untrained model is bit-identical to today's.
            # (b, t, h, w, c) -> flatten every (b, h, w, c) into a batch of length-t signals.
            # The flattened order is (b, h, w, c) and the reshape back MUST match it -- an
            # earlier version reshaped to (b, c, h, w, t) and silently transposed the field,
            # which the identity-init test caught at 100% mismatch.
            cc = z.shape[-1]
            zz = z.permute(0, 2, 3, 4, 1).reshape(-1, 1, cfg.t_in)     # (B*H*W*C, 1, t_in)
            zz = self.tlift(zz).reshape(b, h, w, cc, cfg.t_in)
            z = zz.permute(0, 4, 1, 2, 3)                              # -> (b, t_in, h, w, c)
        z = z.permute(0, 1, 4, 2, 3).reshape(b, cfg.dim_in, h, w)

        out = self.core(z)                                   # (B, dim_out, H, W)
        if cfg.rans:
            # REYNOLDS DECOMPOSITION: u(t) = u_bar + u'(t). The core emits c_out extra
            # channels holding a TIME-CONSTANT mean field, and the remaining t_out*c_out
            # are the fluctuation about it. W182 measured that 95.5% of the tke error is
            # decorrelation of the FLUCTUATION while the mean flow is easy, so giving the
            # two separate capacity -- and a separate, explicit sum -- is the structural
            # version of that finding. It also means the fluctuation head is never asked
            # to carry the mean, which is what an L2 loss otherwise spends it on.
            nf = cfg.frames_per_pass * cfg.c_out
            mean_f = out[:, nf:].reshape(b, 1, cfg.c_out, h, w)
            out = out[:, :nf].reshape(b, cfg.frames_per_pass, cfg.c_out, h, w) + mean_f
            out = out.permute(0, 1, 3, 4, 2)
        else:
            out = out.reshape(b, cfg.frames_per_pass, cfg.c_out, h, w).permute(0, 1, 3, 4, 2)

        if cfg.predict_delta:
            # Residual on the last observed frame, in normalized space, so the
            # network starts from persistence rather than from zero.
            last = z.reshape(b, cfg.t_in, cfg.c_in, h, w)[:, -1]     # (B, C, H, W)
            last = last.permute(0, 2, 3, 1)[:, None, ..., :cfg.c_out]
            out = out + last

        return (out * self.std[:cfg.c_out] + self.mean[:cfg.c_out]) * u_scale

    # -- checkpoints ------------------------------------------------------
    # Always state_dict + config, never torch.save(model). Pickling the module
    # records the class path pde_competition.model.FlowerS2S, which does not
    # exist inside the flat submission zip and fails to load there.

    def save(self, path: str | Path, **extra) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.cfg),
                    "state_dict": self.state_dict(), **extra}, path)
        return path

    @classmethod
    def from_checkpoint(cls, path: str | Path, map_location="cpu") -> "FlowerS2S":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt["config"]
        cfg["resolution"] = tuple(cfg["resolution"])
        model = cls(FlowerConfig(**cfg))
        model.load_state_dict(ckpt["state_dict"])
        return model

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def rel_l2_per_channel(pred: torch.Tensor, target: torch.Tensor, w) -> torch.Tensor:
    """Weighted mean over channels of each channel's OWN relative L2.

    ``rel_l2`` flattens all channels into one norm, so the ratio is dominated by
    whichever channel carries the most energy -- and u and v do not: on this data
    sd(u) = 0.106 against sd(v) = 0.0157, roughly 7x. The SCORE, however, normalises
    over all measured channels together, so v's error is scored but barely trained.
    This computes each channel's relative L2 separately and weights them, so w = (1, 1)
    trains u and v equally hard in RELATIVE terms while the flattened default trains
    them in ABSOLUTE terms.
    Not a reweighting of the metric -- the metric is unchanged -- a reweighting of what
    the optimiser spends capacity on.
    """
    n, c = pred.shape[0], pred.shape[-1]
    out = 0.0
    tot = float(sum(w[:c]))
    for k in range(c):
        p, t = pred[..., k].reshape(n, -1), target[..., k].reshape(n, -1)
        out = out + w[k] * (torch.linalg.norm(p - t, dim=1)
                            / torch.linalg.norm(t, dim=1).clamp(min=1e-8)).mean()
    return out / max(tot, 1e-8)


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-sample relative L2 -- the same quantity ``rel_l2_score`` scores.

    Training directly on the metric beats training on MSE here: MSE is
    dominated by the high-|u| free stream, while the score normalizes each
    window by its own norm and so weights the low-speed wake, which is the part
    that is actually hard.
    """
    n = pred.shape[0]
    p, t = pred.reshape(n, -1), target.reshape(n, -1)
    return (torch.linalg.norm(p - t, dim=1)
            / torch.linalg.norm(t, dim=1).clamp(min=1e-8)).mean()


def rel_l2_per_frame(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 computed FRAME BY FRAME: (B,T,...) -> (B, T).

    ``rel_l2`` flattens the whole window into one norm, so a late frame and an
    early frame contribute in proportion to their absolute error. This
    normalizes each frame by its own norm instead, which is what lets a
    per-frame weight mean anything -- with one global norm, weighting frames
    would just re-weight the numerator and leave the denominator shared.

    Note the two are NOT equal at uniform weights: a mean of ratios is not the
    ratio of sums. That is why ``--horizon-weight uniform`` exists as its own
    arm; it isolates the change of metric from the change of weighting.
    """
    b, t = pred.shape[0], pred.shape[1]
    p = pred.reshape(b, t, -1)
    tg = target.reshape(b, t, -1)
    return (torch.linalg.norm(p - tg, dim=2)
            / torch.linalg.norm(tg, dim=2).clamp(min=1e-8))


def horizon_weights(t_out: int, mode: str, device=None) -> torch.Tensor:
    """Per-frame loss weights, normalized to mean 1 so the loss keeps its scale.

    Mean 1 rather than sum 1 matters: it leaves the weighted loss on the same
    numeric scale as the unweighted one, so the LR, the gradient clip at 1.0 and
    every logged train_rel_l2 stay comparable across arms.

    ``error_by_frame.md`` measured late frames carrying roughly two thirds of
    the error, and nothing in the objective knows about the horizon.
    """
    k = torch.arange(1, t_out + 1, dtype=torch.float32, device=device)
    if mode == "uniform":
        w = torch.ones_like(k)
    elif mode == "linear":
        w = k
    elif mode == "sqrt":
        w = torch.sqrt(k)
    else:
        raise ValueError(f"unknown horizon weighting {mode!r}; "
                         f"choose from uniform, linear, sqrt")
    return w / w.mean()


def kinetic_energy(x: torch.Tensor) -> torch.Tensor:
    """Turbulent kinetic energy per cell: (B, T, H, W, C) -> (B, H, W).

    **Fluctuation** energy, not energy: the time-mean is subtracted first, so
    this measures unsteadiness. That is exactly why it catches what rel-L2
    misses -- a smooth, blurred prediction, the classic failure of an
    L2-trained operator, has little fluctuation and scores badly here while
    looking fine on rel-L2.

    Always ``[u, v]`` whatever ``C`` is, matching ``kit/scoring.py`` which
    indexes channels 0 and 1 explicitly. TKE is a velocity quantity; pressure
    has no part in it.

    ``.mean(dim=1)`` is the population mean, matching ``np.mean`` -- there is
    no ddof discrepancy against the scorer.
    """
    if x.shape[-1] < 2:
        raise ValueError(f"TKE needs u and v; got {x.shape[-1]} channel(s)")
    uv = x[..., :2]
    fluct = uv - uv.mean(dim=1, keepdim=True)
    return 0.5 * (fluct ** 2).mean(dim=1).sum(dim=-1)


# The grader's mean-velocity-profile probes, from kit/scoring.py's
# mvpe_rel_l2_per_sample at the real-data geometry (H=32, W=64, sub_s_real=2):
# nine y positions around the airfoil centre-line and four x stations in the
# wake. Fixed for the competition; asserted against the kit by the tests.
MVPE_PROBE_Y = (8, 10, 12, 14, 16, 18, 20, 22, 24)
MVPE_PROBE_X = (13, 21, 29, 37)


def mvpe_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-sample relative L2 of the time-mean velocity at the grader's
    probes -- what ``mvpe_score`` scores. Torch port of the kit's
    ``mvpe_rel_l2_per_sample`` (tested equal), differentiable so it can be a
    loss term. Uses channels u, v only; (B, T, H, W, C) in.
    """
    n = pred.shape[0]
    ys = list(MVPE_PROBE_Y)
    errs = []
    for x in MVPE_PROBE_X:
        pp = pred[:, :, ys, x, :2].mean(dim=1).reshape(n, -1)
        tt = target[:, :, ys, x, :2].mean(dim=1).reshape(n, -1)
        errs.append(torch.linalg.norm(pp - tt, dim=1) / torch.linalg.norm(tt, dim=1).clamp(min=1e-8))
    return torch.stack(errs, 0).mean(0).mean()


def vrmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Variance-normalised RMSE, the other common PDE-benchmark loss (The Well /
    PDEBench "VRMSE"): per sample and channel, sqrt(mean((p - t)^2) / var(t)),
    averaged over channels and samples. Unlike rel-L2 it normalises each channel
    by its own variance, so v (small) counts as much as u (large) and the
    free-stream mean of u is removed before normalising. Pretrain arm W88."""
    b, c = pred.shape[0], pred.shape[-1]
    p = pred.reshape(b, -1, c); t = target.reshape(b, -1, c)
    mse = ((p - t) ** 2).mean(dim=1)                                   # (B, C)
    var = t.var(dim=1, unbiased=False).clamp(min=1e-8)                 # (B, C)
    return torch.sqrt(mse / var).mean()


def rel_l2_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 per sample, (B,) -- ``rel_l2`` is its mean."""
    n = pred.shape[0]
    p, t = pred.reshape(n, -1), target.reshape(n, -1)
    return torch.linalg.norm(p - t, dim=1) / torch.linalg.norm(t, dim=1).clamp(min=1e-8)


def divergence(field: torch.Tensor) -> torch.Tensor:
    """In-plane divergence du/dx + dv/dy of (B, T, H, W, >=2) fields by central
    differences on the interior grid, grid spacing 1 -> (B, T, H-2, W-2). Axis 2 is
    y (H), axis 3 is x (W); channel 0 is u, channel 1 is v."""
    u, v = field[..., 0], field[..., 1]
    du_dx = 0.5 * (u[:, :, 1:-1, 2:] - u[:, :, 1:-1, :-2])
    dv_dy = 0.5 * (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1])
    return du_dx + dv_dy


def div_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 between the prediction's and the TARGET's in-plane divergence,
    per sample, averaged (W98, 2026-09-17). Real PIV is a 2-D slice of a 3-D flow, so
    the target's divergence is not zero; matching it is valid on both sim and real,
    whereas a divergence-free penalty would fight the real data."""
    dp, dt = divergence(pred), divergence(target)
    n = pred.shape[0]
    dp, dt = dp.reshape(n, -1), dt.reshape(n, -1)
    return (torch.linalg.norm(dp - dt, dim=1) / torch.linalg.norm(dt, dim=1).clamp(min=1e-8)).mean()


def rel_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-sample relative L1: sum|p - t| / sum|t|. Its minimiser is the
    conditional MEDIAN, which is what a fixed-width interval wants covered
    (proposition 5, 2026-09-17): coverage 1{|r| <= h} counts every residual
    under h the same, so pulling the bulk of residuals toward zero matters
    more than the mean-square tail rel-L2 spends its gradient on."""
    n = pred.shape[0]
    p, t = pred.reshape(n, -1), target.reshape(n, -1)
    return ((p - t).abs().sum(dim=1) / t.abs().sum(dim=1).clamp(min=1e-8)).mean()


def coverage_surrogate(pred: torch.Tensor, target: torch.Tensor, half: torch.Tensor,
                       tau: float = 0.002) -> torch.Tensor:
    """1 - smooth coverage at fixed per-channel half-widths: mean over scored
    elements of 1 - sigmoid((h_c - |r|) / tau). Differentiable stand-in for the
    SPS indicator, so the point predictor itself pushes residuals under the
    interval (proposition 5, 2026-09-17). `half` is either one entry per channel
    of pred (1-D, broadcast over every element) or already broadcastable to
    pred's shape -- a per-element MAP, so the surrogate can aim at the width
    profile that actually ships rather than at a constant. W128 swept the
    constant's LEVEL (x0.7 / x1.4) and found it null; the SHAPE has never been
    tested. Unmeasured target elements (exact zeros) are not scored."""
    r = (pred - target).abs()
    scored = (target != 0).to(pred.dtype)
    h = half.to(pred.dtype)
    h = h.view(*([1] * (pred.dim() - 1)), -1) if h.dim() == 1 else h
    inside = torch.sigmoid((h - r) / tau)
    return ((1.0 - inside) * scored).sum() / scored.sum().clamp(min=1.0)


def exp_reward(pred: torch.Tensor, target: torch.Tensor, sigma: float) -> torch.Tensor:
    """The SPS reward integrated over every half-width: mean over scored elements of
    ``exp(-2|r|/sigma)``, returned as a LOSS (negated by the caller's weight).

    The identity is exact:  integral over h of (2/sigma) e^{-2h/sigma} 1{|r| <= h} = e^{-2|r|/sigma}.

    ``coverage_surrogate`` is one slice of that integral, at a single fixed half-width,
    and its gradient vanishes both for elements comfortably inside the interval and for
    elements hopelessly outside it. But the scorer DOES pay for tightening an
    already-covered residual, because a tighter residual lets the width head choose a
    smaller h and collect a larger exp(-2h/sigma). This term has gradient
    (2/sigma) sign(r) exp(-2|r|/sigma): an L1-family loss whose influence REDESCENDS to
    zero on hopeless elements, which is the scorer's own economics. It has no free
    parameter -- sigma is the scorer's constant -- where the coverage term's half-width
    needed three sweeps (W122 weight, W128 level, W158 shape) to pin down.

    Unmeasured target elements (exact zeros) are not scored, as everywhere else.
    """
    r = (pred - target).abs()
    scored = (target != 0).to(pred.dtype)
    return -((torch.exp(-2.0 * r / sigma) * scored).sum() / scored.sum().clamp(min=1.0))


def amse_time(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Coherence-decoupled spectral loss along TIME, per pixel (Subich et al., ICML 2025, "AMSE").

    THE PATHOLOGY IT FIXES. Per temporal frequency bin k, MSE decomposes exactly as

        MSE_k = (sqrt(P_x) - sqrt(P_y))^2 + 2*sqrt(P_x*P_y)*(1 - Coh_k)

    so at fixed coherence the amplitude minimising it is sqrt(P_x) = rho_k*sqrt(P_y) -- the blur -- and
    worse, the coherence term's gradient is weighted by sqrt(P_x), so once a scale has been blurred the
    network stops learning its phase there. AMSE replaces the geometric mean with the max:

        AMSE_k = (sqrt(P_x) - sqrt(P_y))^2 + 2*max(P_x, P_y)*(1 - Coh_k)

    The amplitude term pins P_x = P_y; the coherence term keeps a FULL-STRENGTH phase gradient weighted by
    the truth's own power. Parameter-free.

    WHY THIS IS THE ONE NEW OBJECTIVE WORTH TRYING HERE (2026-09-23). BLUR measured our fluctuation energy
    ratio r against the phase correlation, and for the conditional mean corr^2 = r EXACTLY. We measure
    r/corr^2 = 1.34-1.78 at every lead, so the model is already OVER-amplified relative to the L2 optimum:
    every "restore the energy" objective merely walks the rel_l2-vs-tke frontier that the fluct_scale sweep
    already mapped. **Only corr moves that frontier**, and the coherence term is the only mechanism here
    that targets corr directly rather than amplitude.

    Fluctuations only (the per-pixel time mean is removed, so this is orthogonal to the mean-flow error and
    to mvpe). The DC bin is dropped. Coherence is pooled over the batch: a single pixel's coherence from 20
    samples is noise. Normalised by the truth's mean power so it is scale-free and comparable to rel_l2.
    """
    p = pred[..., :2] - pred[..., :2].mean(dim=1, keepdim=True)
    t = target[..., :2] - target[..., :2].mean(dim=1, keepdim=True)
    Pf = torch.fft.rfft(p, dim=1)[:, 1:]
    Tf = torch.fft.rfft(t, dim=1)[:, 1:]
    Pxx = Pf.real ** 2 + Pf.imag ** 2
    Pyy = Tf.real ** 2 + Tf.imag ** 2
    amp = (Pxx.clamp_min(eps).sqrt() - Pyy.clamp_min(eps).sqrt()) ** 2
    cross = (Pf * Tf.conj()).sum(dim=0)
    coh = cross.abs() / (Pxx.sum(0).clamp_min(eps).sqrt() * Pyy.sum(0).clamp_min(eps).sqrt())
    incoh = (1.0 - coh).clamp_min(0.0).unsqueeze(0)
    return (amp + 2.0 * torch.maximum(Pxx, Pyy) * incoh).mean() / Pyy.mean().clamp_min(eps)


def fluct_corr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - the per-lead CORRELATION of the fluctuation fields, pooled (direction A7).

    Returns ``mean_t (1 - corr_t)``, where ``corr_t`` is the correlation between the
    predicted and the true fluctuation at lead ``t``, pooled over the batch, both grid
    axes and the measured channels. Fluctuation means the field minus its own per-pixel
    mean over the 20-frame output window -- exactly how ``scripts/blur_diagnostic.py``
    defines it, so THIS LOSS IS THE DIAGNOSTIC'S OWN NUMBER. That is the same relationship
    ``tke_rel_l2`` has to ``kit/scoring.py``, and it is deliberate: the acceptance test for
    this term is the quantity it optimises, read on held-out data.

    WHY CORRELATION AND NOT ENERGY (2026-09-23).
    BLUR (Slurm 4838) measured the shipped prediction against truth: fluctuation energy
    ratio r = 1.072 at lead 0 falling to 0.457 at lead 19, correlation 0.881 -> 0.506.
    For the L2 optimum -- the conditional mean -- ``y = yhat + e`` with e orthogonal to
    yhat, so ``corr^2 = r`` EXACTLY. We measure ``r / corr^2 = 1.34 to 1.78`` at every
    lead, so the prediction already carries MORE fluctuation energy than L2 wants, and the
    rescale minimising rel_l2 alone is ``beta = corr/sqrt(r) = 0.75 to 0.87`` -- a SHRINK.
    Every objective whose mechanism is "put the energy back" therefore only walks the
    rel_l2-vs-tke frontier, which VINF already mapped post-hoc for free (constant
    ``fluct_scale`` 1.07, +0.072/+0.064 board, no retraining) and whose far end VINFb/VINFc
    found. ``corr`` is the only quantity that MOVES that frontier, and it moves both halves
    the same way: with ``m = E|p'-t'|^2 / E|t'|^2 = r + 1 - 2*corr*sqrt(r)`` we have
    ``dm/dcorr = -2*sqrt(r) < 0`` at fixed r, and at the optimal rescale ``m = 1 - corr^2``
    -- raising corr lowers the fluctuation error AND raises the energy affordable at that
    error.

    WHY THIS SHAPE RATHER THAN ``amse_time``'s.
    ``amse_time`` is the same idea from the spectral side, but half of it -- the
    ``(sqrt(P_x) - sqrt(P_y))^2`` amplitude term -- is exactly a "put the energy back"
    term, so an AMSE arm confounds the two mechanisms this project has just separated.
    ``fluct_corr`` is SCALE-INVARIANT: multiplying the predicted fluctuation by any
    positive constant leaves it unchanged, so it has exactly zero gradient along the one
    direction VINF already optimises and cannot buy corr by trading energy. Amplitude stays
    where rel_l2 and the post-hoc alpha put it; this term spends its whole gradient on
    phase. It is also parameter-free, costs nothing at inference (the model's output form
    is unchanged), and needs no second network.

    Range [0, 2]. Zero when the predicted fluctuation is any POSITIVE MULTIPLE of the true
    one at every lead, and so in particular when ``pred == target``. Pooling is per lead and
    then averaged over leads, so lead 19 -- where ``1 - corr`` is nearly twice lead 0's --
    attracts the most gradient with no hand-set horizon weight.

    Fluctuations of [u, v] only: the per-pixel time mean is removed, so this is orthogonal
    to the mean flow and to mvpe, and the unmeasured third channel is dropped exactly as
    ``amse_time`` and ``blur_diagnostic`` drop it. Unmeasured PIV elements (exact zeros) are
    LEFT IN, again to match the diagnostic: their target fluctuation is identically zero, so
    they contribute nothing to the numerator and only dilute the denominator.
    """
    p = pred[..., :2] - pred[..., :2].mean(dim=1, keepdim=True)
    t = target[..., :2] - target[..., :2].mean(dim=1, keepdim=True)
    dims = (0,) + tuple(range(2, p.dim()))          # pool everything except lead time
    num = (p * t).sum(dim=dims)                                              # (T,)
    den = (p.pow(2).sum(dim=dims).clamp_min(eps).sqrt()
           * t.pow(2).sum(dim=dims).clamp_min(eps).sqrt())                   # (T,)
    return (1.0 - num / den).mean()


def tke_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-sample relative L2 of the TKE field -- what ``tke_score`` scores.

    A torch port of ``kit/scoring.py``'s ``tke_rel_l2_per_sample`` (verified
    equal to it elementwise). Kept in torch rather than imported so the
    validation pass stays on-device, so nothing in this package depends on the
    disposable ``kit/`` tree, and so this stays differentiable -- TKE is the
    weakest accuracy subscore, and making it a loss term should not require a
    reimplementation.
    """
    n = pred.shape[0]
    p = kinetic_energy(pred).reshape(n, -1)
    t = kinetic_energy(target).reshape(n, -1)
    return (torch.linalg.norm(p - t, dim=1)
            / torch.linalg.norm(t, dim=1).clamp(min=1e-8)).mean()


# How far above the mean a single element's error-weight may rise in the `err`
# variant. NOT a tuning knob -- a stability bound, and 2026-09-21's arm D is why
# it exists. Without it the variant diverges in ONE EPOCH: w is renormalised to
# mean 1, which fixes the TOTAL reward per window but not its CONCENTRATION, and
# putting all of it on a single element multiplies the reward for disagreeing
# there by sqrt(N) -- about 286 at N = 20x32x64x2. The loop closes because
# disagreeing makes the ensemble mean wrong exactly where the members disagree.
# Measured: train ambiguity 621 at epoch 1 and 8.5e4 by epoch 4, against 0.037
# for the unweighted term at the same lambda.
# WHAT THE CAP BUYS, exactly:  A_err <= sqrt(CAP) * A_var, because w <= CAP
# everywhere. So `err` at lambda behaves at worst like `var` at sqrt(CAP)*lambda,
# and `var`'s own boundary was bracketed in the same sweep: stable at 0.3,
# divergent at 1.0. At CAP = 2 that is at most 1.41x, which puts lambda 0.3 at an
# effective 0.42, inside the measured stable region with room to spare. CAP = 10
# would allow 3.2x and put lambda 0.3 at an effective 0.95, back at the boundary
# that diverged -- the first draft of this constant did exactly that, and the
# test below is what caught it.
ERR_WEIGHT_CAP = 2.0


def ensemble_ambiguity(preds: torch.Tensor, target: torch.Tensor,
                       weight_by_error: bool = False) -> torch.Tensor:
    """Member DISAGREEMENT of a CO-TRAINED ensemble, in rel-L2 units (E3 / L8).

    ``preds`` is (M, B, ...): the M members' predictions of the SAME batch,
    already sliced to the measured channels. ``target`` is (B, ...). Returns

        A = mean_b  sqrt( (1/M) sum_m ||p_mb - pbar_b||^2 )  /  ||y_b||

    WHY THIS SHAPE, term by term.

    * Two of the three obvious candidates are the same term. Mean pairwise MSE
      between members and the variance of the members differ by exactly a
      factor of two:  (1/M^2) sum_ij ||p_i - p_j||^2 = 2 * (1/M) sum_m
      ||p_m - pbar||^2, because the cross terms sum to zero about the mean. So
      there is ONE unweighted candidate here, not two, and lambda absorbs the 2.
    * The square root and the ||y_b|| denominator put it in the units of
      ``rel_l2``, which is what this project's objective is made of -- a mean of
      RATIOS OF NORMS, never of squares. It also makes the gradient behave:
      d A / d p_m has norm 1 / (M ||y_b||) whatever the disagreement level,
      i.e. a constant push of 1/M of the accuracy gradient's own magnitude. So
      lambda reads directly as "the diversity push as a fraction of the accuracy
      push" and nothing blows up while the members are still nearly identical,
      which is exactly when a squared term would have no gradient at all.
    * Krogh & Vedelsby's ambiguity decomposition is the reason to expect
      anything: for squared error, ensemble error = mean member error - ambiguity,
      EXACTLY. Members that disagree more at the same individual accuracy give a
      better mean. Independent seeds leave that ambiguity to chance; this prices it.

    ``weight_by_error=True`` is the variant that follows the mechanism rather
    than the identity: disagreement where the ensemble mean is already RIGHT is
    pure cost, so weight each element by the ensemble mean's own squared error
    there -- detached, and renormalised to mean 1 over the window. The
    renormalisation fixes the total reward available per window however wrong the
    model is, so no gradient pays the model for being uniformly worse -- but it
    does NOT fix the CONCENTRATION, and concentration is what pays: putting all
    the weight on one element multiplies the reward for disagreeing there by
    sqrt(N). Uncapped, that loop diverged in one epoch on 2026-09-21 (arm D of
    the E3 sweep: train ambiguity 621 at epoch 1 against 0.037 for the
    unweighted term at the same lambda). ``ERR_WEIGHT_CAP`` bounds it, applied
    after the renormalisation and deliberately WITHOUT renormalising again, which
    bounds the weighted term by sqrt(ERR_WEIGHT_CAP) times the unweighted one --
    so `err` at lambda is at worst `var` at sqrt(CAP)*lambda, and `var`'s
    stability boundary is measured (0.3 stable, 1.0 divergent). Unmeasured target elements (exact zeros) get weight 0, as in
    ``coverage_surrogate``; the unweighted variant leaves them in, exactly as
    ``rel_l2`` does, so each variant keeps the convention of the loss it is
    commensurate with.

    NOT a metric and never validated on: this is a training-time coupling only.
    The number it returns is logged as a diagnostic (how far apart the members
    actually are), which is the only way to tell a diversity term that worked
    from one that never moved.
    """
    m, n = preds.shape[0], preds.shape[1]
    if m < 2:
        raise ValueError(f"ensemble_ambiguity needs at least 2 members, got {m}")
    if preds.shape[1:] != target.shape:
        raise ValueError(f"members are {tuple(preds.shape)}, target is {tuple(target.shape)}")
    dev = (preds - preds.mean(0)).reshape(m, n, -1)                    # (M, B, N)
    t = target.reshape(n, -1)                                          # (B, N)
    if weight_by_error:
        e = ((preds.mean(0).reshape(n, -1) - t) ** 2).detach() * (t != 0).to(preds.dtype)
        w = e / e.mean(dim=1, keepdim=True).clamp(min=1e-12)           # mean 1 per sample
        w = w.clamp(max=ERR_WEIGHT_CAP)                                # see ERR_WEIGHT_CAP
        amb = (w.unsqueeze(0) * dev ** 2).sum(dim=2).mean(dim=0)       # (B,)
    else:
        amb = (dev ** 2).sum(dim=2).mean(dim=0)                        # (B,)
    # clamp before the sqrt: its gradient is zero where the clamp bites, so
    # exactly-identical members give no gradient rather than an infinite one.
    return (amb.clamp(min=1e-12).sqrt()
            / torch.linalg.norm(t, dim=1).clamp(min=1e-8)).mean()

def persistence(x: torch.Tensor, t_out: int = T_OUT) -> torch.Tensor:
    """Repeat the last input frame -- the predictor any model has to beat.

    The trivial forecast for a stationary flow, and the reference every training
    number should be read against: a val rel-L2 means nothing until you know
    what doing nothing scores.

    Also exactly what ``FlowerConfig.predict_delta`` learns a residual on, so
    the baseline and that parameterization cannot drift apart.
    """
    return x[:, -1:].expand(-1, t_out, -1, -1, -1)


@torch.no_grad()
def predict_numpy(model: FlowerS2S, x: np.ndarray, batch_size: int = 16,
                  device: str | torch.device = "cpu") -> np.ndarray:
    """Track 1 ``predict()`` semantics: (N, t_in, H, W, C) -> (N, t_out, H, W, C).

    Always returns the full ``C`` channels the grader expects, whatever the
    model predicts. The input is sliced down to ``cfg.c_in``; any channel beyond
    ``cfg.c_out`` is left at the zero it was initialised to. At ``c_out = 2``
    that means the unmeasured ``p`` plane comes back as zeros; at ``c_out = 3``
    it is whatever the model produced, which is harmless either way because
    ``scoring.measured_channels`` drops ``p`` for real targets before every
    metric.
    """
    cfg = model.cfg
    model.eval().to(device)
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros((x.shape[0], cfg.t_out, *cfg.resolution, C), np.float32)
    for i in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch_size, ..., :cfg.c_in]))
        yb = model(xb.to(device)).float().cpu().numpy()
        out[i:i + batch_size, ..., :cfg.c_out] = yb
    return out
