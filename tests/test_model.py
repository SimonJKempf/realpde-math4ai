"""Shape, dtype and checkpoint round-trip of the Flower wrapper. Seconds, CPU, no data.

    python -m unittest discover tests        # or: pytest tests/

These exist so a small code change can be checked without a 20-minute
training run. They do not measure quality -- train.py --smoke does the wiring
end to end, and evaluate.py measures.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pde_competition.geometry import H_S, T_IN, T_OUT, W_S
from pde_competition.model import (FlowerConfig, FlowerS2S, horizon_weights,
                                   persistence, rel_l2, rel_l2_per_frame,
                                   tke_rel_l2, window_scale)

# The --smoke configuration: small enough for a CPU test, same code path.
SMALL = dict(lifting_dim=32, num_heads=8, groups=8)


def _x(b=2, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, T_IN, H_S, W_S, c, generator=g)
    if c == 3:
        x[..., 2] = 0.0                      # real data: p is identically zero
    return x


class ForwardShape(unittest.TestCase):
    def test_three_channels(self):
        m = FlowerS2S(FlowerConfig(**SMALL)).eval()
        with torch.no_grad():
            y = m(_x(c=3))
        self.assertEqual(tuple(y.shape), (2, T_OUT, H_S, W_S, 3))
        self.assertEqual(y.dtype, torch.float32)
        self.assertTrue(torch.isfinite(y).all())

    def test_two_channels(self):
        m = FlowerS2S(FlowerConfig(c_in=2, c_out=2, **SMALL)).eval()
        with torch.no_grad():
            y = m(_x(c=2))
        self.assertEqual(tuple(y.shape), (2, T_OUT, H_S, W_S, 2))

    def test_window_norm_and_delta_paths(self):
        for kw in (dict(window_norm=True, c_in=2, c_out=2), dict(predict_delta=True)):
            m = FlowerS2S(FlowerConfig(**SMALL, **kw)).eval()
            with torch.no_grad():
                y = m(_x(c=m.cfg.c_in))
            self.assertTrue(torch.isfinite(y).all(), kw)

    def test_wrong_shape_is_rejected_loudly(self):
        m = FlowerS2S(FlowerConfig(**SMALL))
        with self.assertRaises(ValueError):
            m(torch.zeros(1, T_IN, H_S, W_S, 2))       # c_in is 3
        with self.assertRaises(ValueError):
            m(torch.zeros(1, T_IN - 1, H_S, W_S, 3))   # wrong window length


class CheckpointRoundTrip(unittest.TestCase):
    def test_save_then_from_checkpoint_is_identical(self):
        cfg = FlowerConfig(c_in=2, c_out=2, window_norm=True,
                           boundary_condition_types=("ZEROS",), **SMALL)
        m = FlowerS2S(cfg, mean=np.array([0.15, 0.0]), std=np.array([0.11, 0.016]))
        m.eval()
        x = _x(c=2)
        with tempfile.TemporaryDirectory() as d:
            p = m.save(Path(d) / "best.pth", epoch=3, val_rel_l2=0.08)
            m2 = FlowerS2S.from_checkpoint(p).eval()
            ck = torch.load(p, map_location="cpu", weights_only=False)
        # Config survives, including the two fields that silently change the
        # forward pass if dropped (boundary types, window_norm).
        self.assertEqual(m2.cfg, cfg)
        self.assertEqual(tuple(ck["config"]["boundary_condition_types"]), ("ZEROS",))
        self.assertTrue(ck["config"]["window_norm"])
        self.assertEqual(ck["epoch"], 3)
        # Normalization stats are buffers and travel with the weights.
        self.assertTrue(m2.stats_are_set)
        with torch.no_grad():
            torch.testing.assert_close(m(x), m2(x))


class Metrics(unittest.TestCase):
    def test_rel_l2_is_zero_on_identity_and_positive_otherwise(self):
        y = _x()
        self.assertAlmostEqual(float(rel_l2(y, y)), 0.0, places=6)
        self.assertGreater(float(rel_l2(y + 0.1, y)), 0.0)

    def test_persistence_repeats_last_frame(self):
        x = _x()
        p = persistence(x)
        self.assertEqual(tuple(p.shape), (2, T_OUT, H_S, W_S, 3))
        torch.testing.assert_close(p[:, 0], x[:, -1])
        torch.testing.assert_close(p[:, -1], x[:, -1])

    def test_mvpe_matches_the_kit_and_is_differentiable(self):
        import sys
        from pde_competition.paths import KIT
        sys.path.insert(0, str(KIT))
        from scoring import mvpe_rel_l2_per_sample
        from pde_competition.model import mvpe_rel_l2
        g = torch.Generator().manual_seed(7)
        pred = torch.randn(3, T_OUT, H_S, W_S, 3, generator=g)
        tgt = torch.randn(3, T_OUT, H_S, W_S, 3, generator=g)
        ours = mvpe_rel_l2(pred, tgt).item()
        kit = float(mvpe_rel_l2_per_sample(pred.numpy(), tgt.numpy()).mean())
        self.assertAlmostEqual(ours, kit, places=5)
        p2 = pred.clone().requires_grad_(True)
        mvpe_rel_l2(p2, tgt).backward()
        self.assertTrue(torch.isfinite(p2.grad).all())

    def test_tke_term_is_differentiable_and_adds_to_the_loss(self):
        # --tke-weight: loss = rel_l2 + w * tke_rel_l2 must backpropagate and
        # be strictly larger than rel_l2 alone when the prediction is off.
        x = _x(); tgt = _x(seed=1)
        pred = (x[:, :T_OUT] if x.shape[1] >= T_OUT else x).clone().requires_grad_(True)
        tgt = tgt[:, :pred.shape[1]]
        base = rel_l2(pred, tgt)
        loss = base + 0.14 * tke_rel_l2(pred, tgt)
        self.assertGreater(loss.item(), base.item())
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.abs().sum().item(), 0.0)

    def test_tke_of_persistence_is_the_floor(self):
        # A repeated frame has no fluctuation about its own time mean, so its
        # TKE error against any target is ~1.0 -- the floor train.py reports.
        x, y = _x(seed=1), _x(seed=2)
        self.assertAlmostEqual(float(tke_rel_l2(persistence(x), y)), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()


class HorizonWeighting(unittest.TestCase):
    """The per-frame objective added 2026-09-11."""

    def test_weights_have_mean_one_so_the_loss_keeps_its_scale(self):
        # Mean 1, not sum 1: the LR, the 1.0 gradient clip and every logged
        # train_rel_l2 are calibrated against the unweighted loss's scale.
        for mode in ("uniform", "linear", "sqrt"):
            w = horizon_weights(T_OUT, mode)
            self.assertEqual(w.shape, (T_OUT,))
            self.assertAlmostEqual(float(w.mean()), 1.0, places=5, msg=mode)

    def test_linear_and_sqrt_increase_with_lead_time(self):
        for mode in ("linear", "sqrt"):
            w = horizon_weights(T_OUT, mode)
            self.assertTrue(bool((w[1:] > w[:-1]).all()), mode)
        self.assertTrue(bool((horizon_weights(T_OUT, "uniform") == 1).all()))

    def test_unknown_mode_is_rejected_loudly(self):
        with self.assertRaises(ValueError):
            horizon_weights(T_OUT, "quadratic")

    def test_per_frame_is_zero_on_identity_and_shaped_B_by_T(self):
        t = torch.randn(3, T_OUT, H_S, W_S, 2)
        pf = rel_l2_per_frame(t, t)
        self.assertEqual(tuple(pf.shape), (3, T_OUT))
        self.assertTrue(torch.allclose(pf, torch.zeros_like(pf), atol=1e-6))

    def test_per_frame_uniform_is_NOT_the_global_rel_l2(self):
        # A mean of ratios is not the ratio of sums. This is exactly why
        # --horizon-weight uniform exists as its own arm: without it, a win
        # from 'linear' could not be separated from the change of metric.
        torch.manual_seed(0)
        target = torch.randn(4, T_OUT, H_S, W_S, 2)
        # make the frames differ in norm, or the two agree by construction
        target *= torch.linspace(0.2, 3.0, T_OUT).reshape(1, T_OUT, 1, 1, 1)
        pred = target + 0.1 * torch.randn_like(target)
        uniform = (rel_l2_per_frame(pred, target)
                   * horizon_weights(T_OUT, "uniform")).mean()
        self.assertNotAlmostEqual(float(uniform), float(rel_l2(pred, target)),
                                  places=3)

    def test_late_weighting_penalises_late_error_more(self):
        torch.manual_seed(0)
        target = torch.randn(2, T_OUT, H_S, W_S, 2)
        early, late = target.clone(), target.clone()
        early[:, 0] += 1.0
        late[:, -1] += 1.0
        w = horizon_weights(T_OUT, "linear")
        loss = lambda p: float((rel_l2_per_frame(p, target) * w).mean())
        self.assertGreater(loss(late), loss(early))


class WindowScaleMode(unittest.TestCase):
    """The sqrt(mean(u^2+v^2)) divisor added 2026-09-11."""

    def _window(self, u_val, v_val, masked=True):
        x = torch.zeros(1, T_IN, H_S, W_S, 2)
        x[..., 0] = u_val
        x[..., 1] = v_val
        if masked:                      # u == 0 is what marks an unmeasured point
            x[:, :, :4, :, :] = 0.0
        return x

    def test_u_mode_is_mean_abs_u_and_ignores_v(self):
        a = window_scale(self._window(0.2, 0.0), "u")
        b = window_scale(self._window(0.2, 5.0), "u")
        self.assertAlmostEqual(float(a), 0.2, places=5)
        self.assertAlmostEqual(float(a), float(b), places=6)

    def test_speed_mode_includes_v_and_exceeds_u_mode(self):
        x = self._window(0.3, 0.4)
        self.assertAlmostEqual(float(window_scale(x, "speed")), 0.5, places=5)
        self.assertGreater(float(window_scale(x, "speed")),
                           float(window_scale(x, "u")))

    def test_modes_agree_when_v_is_zero(self):
        x = self._window(0.25, 0.0)
        self.assertAlmostEqual(float(window_scale(x, "u")),
                               float(window_scale(x, "speed")), places=5)

    def test_unmeasured_points_are_excluded_by_both_modes(self):
        # Masking a quarter of the field must not change either scale: the
        # excluded cells leave numerator and denominator together.
        for mode in ("u", "speed"):
            full = window_scale(self._window(0.3, 0.4, masked=False), mode)
            part = window_scale(self._window(0.3, 0.4, masked=True), mode)
            self.assertAlmostEqual(float(full), float(part), places=5, msg=mode)

    def test_unknown_mode_is_rejected_loudly(self):
        with self.assertRaises(ValueError):
            window_scale(self._window(0.2, 0.1), "magnitude")
        with self.assertRaises(ValueError):
            FlowerConfig(c_in=2, c_out=2, window_norm=True,
                         window_scale_mode="magnitude")

    def test_default_is_u_so_existing_checkpoints_are_unchanged(self):
        self.assertEqual(FlowerConfig().window_scale_mode, "u")

    def test_speed_mode_runs_end_to_end_and_round_trips(self):
        cfg = FlowerConfig(c_in=2, c_out=2, window_norm=True,
                           window_scale_mode="speed", **SMALL)
        model = FlowerS2S(cfg).eval()
        with torch.no_grad():
            y = model(_x(c=2))
        self.assertEqual(y.shape[-1], 2)
        with tempfile.TemporaryDirectory() as d:
            path = model.save(Path(d) / "m.pth")
            back = FlowerS2S.from_checkpoint(path)
        # The mode silently changes the forward pass, so it has to survive the
        # round trip the same way window_norm and the boundary types do.
        self.assertEqual(back.cfg.window_scale_mode, "speed")
        self.assertEqual(back.cfg, cfg)


class BorderPadding(unittest.TestCase):
    """'border' became reachable 2026-09-14; upstream mapped it onto zeros."""

    def test_border_reaches_grid_sample_and_is_not_zeros(self):
        from pde_competition._vendor.flower_standalone import derive_padding_strategies
        self.assertEqual(derive_padding_strategies(["BORDER"], 2), ["border", "border"])
        self.assertEqual(derive_padding_strategies(["PERIODIC"], 2), ["periodic", "periodic"])
        self.assertEqual(derive_padding_strategies(["ZEROS"], 2), ["zeros", "zeros"])
        # an unrecognised string still falls back, as it always did
        self.assertEqual(derive_padding_strategies(["DIRICHLET"], 2), ["zeros", "zeros"])

    def test_border_model_runs_and_round_trips(self):
        cfg = FlowerConfig(c_in=2, c_out=2, boundary_condition_types=("BORDER",), **SMALL)
        m = FlowerS2S(cfg).eval()
        with torch.no_grad():
            self.assertEqual(m(_x(c=2)).shape[-1], 2)
        with tempfile.TemporaryDirectory() as d:
            back = FlowerS2S.from_checkpoint(m.save(Path(d) / "m.pth"))
        self.assertEqual(tuple(back.cfg.boundary_condition_types), ("BORDER",))


class SingleLevel(unittest.TestCase):
    """n_levels=1 -- a flat warp operator, unsupported upstream until patched."""

    def test_one_level_builds_and_runs(self):
        # Upstream raised UnboundLocalError: out_ch, because the decoder loop
        # that assigns it is empty at one level.
        cfg = FlowerConfig(c_in=2, c_out=2, n_levels=1, lifting_dim=32,
                           resolution=(H_S, W_S))
        m = FlowerS2S(cfg).eval()
        with torch.no_grad():
            y = m(torch.randn(1, T_IN, H_S, W_S, 2))
        self.assertEqual(tuple(y.shape), (1, T_OUT, H_S, W_S, 2))

    def test_deeper_models_are_unaffected_by_the_patch(self):
        for nl in (2, 3, 4):
            cfg = FlowerConfig(c_in=2, c_out=2, n_levels=nl, **SMALL)
            m = FlowerS2S(cfg).eval()
            with torch.no_grad():
                self.assertEqual(m(_x(c=2)).shape[-1], 2)


class WarmCosineSchedule(unittest.TestCase):
    """--schedule warmcos, the Flowers paper's recipe, added 2026-09-14."""

    def _lr_curve(self, total_steps, warm_frac):
        import math
        warm = max(int(warm_frac * total_steps), 1)
        def f(step):
            if step < warm:
                return (step + 1) / warm
            prog = (step - warm) / max(total_steps - warm, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
        return [f(i) for i in range(total_steps)]

    def test_warms_up_then_decays_to_zero(self):
        c = self._lr_curve(100, 0.1)
        self.assertAlmostEqual(c[9], 1.0, places=6)      # peak at end of warm-up
        self.assertTrue(all(c[i] <= c[i + 1] for i in range(9)))    # rising
        self.assertTrue(all(c[i] >= c[i + 1] for i in range(10, 99)))  # then falling
        self.assertLess(c[-1], 1e-3)                     # cosine reaches ~0
        self.assertGreater(min(c), -1e-12)               # never negative


class Rollout(unittest.TestCase):
    """--rollout-frames, added 2026-09-14."""

    def test_default_is_off_and_unchanged(self):
        cfg = FlowerConfig(c_in=2, c_out=2, **SMALL)
        self.assertEqual(cfg.rollout_frames, 0)
        self.assertEqual(cfg.frames_per_pass, T_OUT)
        self.assertEqual(cfg.dim_out, T_OUT * 2)

    def test_rollout_shrinks_the_core_output_and_keeps_the_horizon(self):
        cfg = FlowerConfig(c_in=2, c_out=2, rollout_frames=5, **SMALL)
        self.assertEqual(cfg.frames_per_pass, 5)
        self.assertEqual(cfg.dim_out, 5 * 2)          # core emits 5 frames
        m = FlowerS2S(cfg).eval()
        with torch.no_grad():
            y = m(_x(c=2))
        self.assertEqual(y.shape[1], T_OUT)           # but the model still emits 20

    def test_rollout_equal_to_horizon_is_one_pass(self):
        # k == t_out means a single step, so it must match the non-rollout path
        # given identical weights.
        torch.manual_seed(0)
        a = FlowerS2S(FlowerConfig(c_in=2, c_out=2, **SMALL)).eval()
        b = FlowerS2S(FlowerConfig(c_in=2, c_out=2, rollout_frames=T_OUT, **SMALL)).eval()
        b.load_state_dict(a.state_dict())
        x = _x(c=2)
        with torch.no_grad():
            torch.testing.assert_close(a(x), b(x))

    def test_k_must_divide_the_horizon(self):
        with self.assertRaises(ValueError):
            FlowerConfig(c_in=2, c_out=2, rollout_frames=7, **SMALL)

    def test_rollout_needs_matching_channel_widths(self):
        # a step feeds its own output back in, so c_in must equal c_out
        with self.assertRaises(ValueError):
            FlowerConfig(c_in=3, c_out=2, rollout_frames=5, **SMALL)

    def test_config_survives_a_checkpoint_round_trip(self):
        cfg = FlowerConfig(c_in=2, c_out=2, rollout_frames=5, **SMALL)
        m = FlowerS2S(cfg).eval()
        with tempfile.TemporaryDirectory() as d:
            back = FlowerS2S.from_checkpoint(m.save(Path(d) / "m.pth"))
        self.assertEqual(back.cfg.rollout_frames, 5)
        self.assertEqual(back.cfg, cfg)


class VRMSELoss(unittest.TestCase):
    def test_vrmse_is_zero_at_target_and_one_for_the_channel_mean(self):
        from pde_competition.model import vrmse
        g = torch.Generator().manual_seed(0)
        t = torch.randn(2, 4, 8, 8, 3, generator=g) * torch.tensor([1.0, 0.1, 0.01])
        self.assertAlmostEqual(float(vrmse(t, t)), 0.0, places=6)
        # predicting each channel's own mean gives RMSE = std -> VRMSE = 1 per channel
        m = t.reshape(2, -1, 3).mean(dim=1, keepdim=True).expand(-1, 4 * 8 * 8, -1).reshape_as(t)
        self.assertAlmostEqual(float(vrmse(m, t)), 1.0, places=5)
        self.assertTrue(vrmse(t * 1.1, t).requires_grad is False)


class PhysicsAndWeights(unittest.TestCase):
    def test_divergence_is_zero_for_a_solenoidal_field_and_loss_is_zero_at_target(self):
        from pde_competition.model import divergence, div_rel_l2
        B, T, H, W = 2, 3, 8, 12
        y = torch.linspace(0, 1, H).view(1, 1, H, 1, 1).expand(B, T, H, W, 1)
        x = torch.linspace(0, 1, W).view(1, 1, 1, W, 1).expand(B, T, H, W, 1)
        f = torch.cat([y, -x, torch.zeros_like(x)], dim=-1)        # u = y, v = -x -> div = 0
        self.assertLess(float(divergence(f).abs().max()), 1e-6)
        g = torch.randn(B, T, H, W, 3)
        self.assertAlmostEqual(float(div_rel_l2(g, g)), 0.0, places=6)
        self.assertGreater(float(div_rel_l2(g * 1.3, g)), 0.0)

    def test_per_sample_rel_l2_means_to_rel_l2(self):
        from pde_competition.model import rel_l2, rel_l2_per_sample
        p, t = torch.randn(4, 3, 8, 12, 3), torch.randn(4, 3, 8, 12, 3)
        self.assertAlmostEqual(float(rel_l2_per_sample(p, t).mean()), float(rel_l2(p, t)), places=6)


class LagrangianShear(unittest.TestCase):
    """shear_window and cfg.lagrangian: correctness, and that 'off' is bit-identical."""

    def test_zero_shift_is_identity_and_integer_shift_translates(self):
        from pde_competition.model import shear_window
        g = torch.Generator().manual_seed(0)
        x = torch.randn(2, 4, 8, 16, 3, generator=g)
        z = torch.zeros(2, 4)
        torch.testing.assert_close(shear_window(x, z), x)                 # no shift -> identity
        one = torch.ones(2, 4)
        out = shear_window(x, one)
        # a +1 px shift moves content downstream: out[..., j, :] == x[..., j-1, :] away from the edge
        torch.testing.assert_close(out[:, :, :, 2:-1, :], x[:, :, :, 1:-2, :], atol=1e-5, rtol=1e-5)

    def test_lagrangian_off_is_bit_identical_and_input_changes_the_field(self):
        from pde_competition.model import FlowerS2S, FlowerConfig
        base = dict(t_in=4, t_out=4, c_in=3, c_out=3, resolution=(8, 16), lifting_dim=16,
                    n_levels=1, num_heads=4, groups=4)
        torch.manual_seed(0)
        m_off = FlowerS2S(FlowerConfig(**base))
        torch.manual_seed(0)
        m_lag = FlowerS2S(FlowerConfig(**base, lagrangian="input"))
        g = torch.Generator().manual_seed(1)
        x = torch.randn(2, 4, 8, 16, 3, generator=g).abs() * 0.2          # mean|u| > 0 so the shift is nonzero
        with torch.no_grad():
            a, b = m_off(x), m_lag(x)
        self.assertGreater(float((a - b).abs().max()), 1e-6)              # the flag does something
        torch.manual_seed(0)
        m_c0 = FlowerS2S(FlowerConfig(**base, lagrangian="input", lagrangian_c=0.0))
        with torch.no_grad():
            torch.testing.assert_close(m_c0(x), a)                        # c=0 recovers 'off' exactly

    def test_unmeasured_holes_stay_holes_after_the_shear(self):
        from pde_competition.model import FlowerS2S, FlowerConfig
        cfg = FlowerConfig(t_in=4, t_out=4, c_in=3, c_out=3, resolution=(8, 16), lifting_dim=16,
                           n_levels=1, num_heads=4, groups=4, lagrangian="input")
        m = FlowerS2S(cfg)
        g = torch.Generator().manual_seed(2)
        x = torch.randn(1, 4, 8, 16, 3, generator=g).abs() * 0.2
        x[:, :, :, 3:6, :] = 0.0                                          # a hole that must not smear
        # reach into the shear the way _one_pass does and check the mask survives
        from pde_competition.model import shear_window, window_scale
        speed = window_scale(x, "u").detach().reshape(-1)
        k = torch.arange(3, -1, -1, dtype=x.dtype)
        shift = cfg.lagrangian_c * speed[:, None] * k[None, :]
        meas = (x[..., :1] != 0).to(x.dtype)
        out = shear_window(x, shift) * meas
        self.assertTrue(bool((out[:, :, :, 3:6, :] == 0).all()))


class ExpReward(unittest.TestCase):
    """exp_reward: the coverage term integrated over every half-width."""

    def test_matches_the_analytic_integral_of_the_coverage_indicator(self):
        """MC-integrate (2/s)e^{-2h/s}*1{|r|<=h} dh over h and compare to exp(-2|r|/s)."""
        import numpy as np
        from pde_competition.model import exp_reward
        s = 0.0563870259
        g = torch.Generator().manual_seed(0)
        t = torch.randn(4, 3, 8, 8, 2, generator=g) * 0.05
        p = t + torch.randn(4, 3, 8, 8, 2, generator=g) * 0.01
        got = -float(exp_reward(p, t, s))
        r = (p - t).abs().numpy().ravel()
        # Deterministic quadrature, not Monte Carlo: the first version sampled 400 draws
        # and had a ~0.02 standard error against a places=2 assertion, so it failed on a
        # correct implementation. Integrate (2/s)e^{-2h/s} * mean(1{|r|<=h}) dh directly.
        h = np.linspace(0.0, 40.0 * s, 20001)
        w = (2.0 / s) * np.exp(-2.0 * h / s)
        frac = (r[None, :] <= h[:, None]).mean(axis=1)
        quad = float(np.trapezoid(w * frac, h)) if hasattr(np, "trapezoid") else float(np.trapz(w * frac, h))
        self.assertAlmostEqual(got, quad, places=4)

    def test_scored_mask_bounds_and_gradient(self):
        from pde_competition.model import exp_reward
        s = 0.0563870259
        g = torch.Generator().manual_seed(1)
        t = torch.randn(2, 3, 8, 8, 3, generator=g) * 0.05
        t[..., 2] = 0.0                                    # unmeasured channel, never scored
        self.assertAlmostEqual(float(exp_reward(t, t, s)), -1.0, places=6)   # zero residual -> reward 1
        far = t + 1.0
        self.assertGreater(float(exp_reward(far, t, s)), -0.01)             # hopeless -> reward ~0
        q = (t * 1.1).requires_grad_(True)
        exp_reward(q, t, s).backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(float(q.grad.abs().sum()), 0.0)


class CoverageObjective(unittest.TestCase):
    def test_rel_l1_and_coverage_surrogate_behave(self):
        from pde_competition.model import rel_l1, coverage_surrogate
        g = torch.Generator().manual_seed(0)
        t = torch.randn(2, 3, 8, 8, 3, generator=g) * 0.1
        t[..., 2] = 0.0          # real PIV: the pressure channel is exact zeros = unmeasured, never scored
        self.assertAlmostEqual(float(rel_l1(t, t)), 0.0, places=6)
        self.assertGreater(float(rel_l1(t * 1.2, t)), 0.0)
        half = torch.tensor([0.05, 0.05, 0.0])
        # h = 0 on the unmeasured channel is fine only because that channel is not scored;
        # on a MEASURED channel h = 0 puts r = 0 on the sigmoid's tie (0.5) -- W143, 2026-09-18.
        self.assertLess(float(coverage_surrogate(t, t, half)), 0.01)                 # all inside -> ~0
        t_meas = t.clone(); t_meas[..., 2] = 0.1
        self.assertAlmostEqual(float(coverage_surrogate(t_meas, t_meas, half)), 0.5 / 3, places=5)
        self.assertGreater(float(coverage_surrogate(t + 1.0, t, half)), 0.99)        # all outside -> ~1
        p = (t * 1.1).requires_grad_(True)
        coverage_surrogate(p, t, half).backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_coverage_surrogate_accepts_a_per_element_map(self):
        """A (H,W,C) or (T,H,W,C) map must broadcast, and a CONSTANT map must equal the 1-D path."""
        from pde_competition.model import coverage_surrogate
        g = torch.Generator().manual_seed(1)
        t = torch.randn(2, 3, 8, 8, 3, generator=g) * 0.1
        t[..., 2] = 0.0
        p = t * 1.15
        half = torch.tensor([0.05, 0.05, 0.0])
        ref = float(coverage_surrogate(p, t, half))
        for shape in ((8, 8, 3), (3, 8, 8, 3)):
            flat = torch.zeros(shape); flat[..., 0] = 0.05; flat[..., 1] = 0.05
            self.assertAlmostEqual(float(coverage_surrogate(p, t, flat)), ref, places=6)
        # a genuinely varying map must differ from the constant, and stay differentiable
        m = torch.zeros(8, 8, 3)
        m[..., 0] = torch.linspace(0.01, 0.2, 8)[:, None].expand(8, 8)
        m[..., 1] = 0.05
        self.assertNotAlmostEqual(float(coverage_surrogate(p, t, m)), ref, places=4)
        q = p.clone().requires_grad_(True)
        coverage_surrogate(q, t, m).backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(float(q.grad.abs().sum()), 0.0)

    def test_coverage_map_channel_count_must_match_the_prediction(self):
        """The map on disk carries all C channels; the loss sees only the measured ones.

        W158's first launch died here: a (20, 32, 64, 3) map against 2-channel tensors raises
        rather than broadcasting, so the slice in train.py is load-bearing, not cosmetic."""
        from pde_competition.model import coverage_surrogate
        g = torch.Generator().manual_seed(2)
        t = torch.randn(2, 20, 8, 8, 2, generator=g) * 0.1        # measured channels only
        p = t * 1.15
        m3 = torch.full((20, 8, 8, 3), 0.05)                      # as fit_bounds --spatial writes it
        with self.assertRaises(RuntimeError):
            coverage_surrogate(p, t, m3)
        self.assertLess(float(coverage_surrogate(p, t, m3[..., :2])), 1.0)   # sliced: fine


class TemporalLiftAndRans(unittest.TestCase):
    """Two architecture changes aimed at W182's decorrelation finding.

    temporal_lift gives the network LOCAL temporal structure before the core's 1x1 conv
    mixes all t_in*c_in channels globally -- the advection problem, attacked without
    moving data the way W178's Lagrangian shear did (and which closed).
    rans splits the output into a time-constant mean and a fluctuation, because 95.5% of
    the tke error is decorrelation of the fluctuation while the mean flow is easy."""

    def test_identity_init_makes_temporal_lift_a_no_op(self):
        """The whole point of the delta-kernel init: an untrained model must reproduce
        today's exactly, so any later change is something the optimiser chose."""
        torch.manual_seed(0)
        base = FlowerS2S(FlowerConfig(lifting_dim=32, n_levels=1, c_in=3, c_out=3)).eval()
        torch.manual_seed(0)
        tl = FlowerS2S(FlowerConfig(lifting_dim=32, n_levels=1, c_in=3, c_out=3,
                                    temporal_lift=5)).eval()
        x = torch.randn(2, base.cfg.t_in, 32, 64, 3)
        with torch.no_grad():
            torch.testing.assert_close(base(x), tl(x), rtol=1e-5, atol=1e-6)

    def test_temporal_lift_must_be_odd(self):
        with self.assertRaises(ValueError):
            FlowerS2S(FlowerConfig(lifting_dim=16, n_levels=1, temporal_lift=4))

    def test_temporal_lift_actually_mixes_time_once_trained(self):
        cfg = FlowerConfig(lifting_dim=16, n_levels=1, c_in=3, c_out=3, temporal_lift=3)
        m = FlowerS2S(cfg).eval()
        with torch.no_grad():                      # perturb away from the delta kernel
            m.tlift.weight.copy_(torch.tensor([[[0.3, 0.4, 0.3]]]))
        x = torch.randn(1, cfg.t_in, 32, 64, 3)
        y_mixed = m(x)
        with torch.no_grad():
            m.tlift.weight.zero_(); m.tlift.weight[0, 0, 1] = 1.0
        self.assertFalse(torch.allclose(y_mixed, m(x), atol=1e-6), "the temporal conv did nothing")

    def test_rans_widens_dim_out_by_exactly_one_frame(self):
        a = FlowerConfig(lifting_dim=16, n_levels=1, c_in=3, c_out=3)
        b = FlowerConfig(lifting_dim=16, n_levels=1, c_in=3, c_out=3, rans=True)
        self.assertEqual(b.dim_out, a.dim_out + a.c_out)

    def test_rans_output_shape_and_that_the_mean_is_shared(self):
        cfg = FlowerConfig(lifting_dim=16, n_levels=1, c_in=3, c_out=3, rans=True)
        m = FlowerS2S(cfg).eval()
        x = torch.randn(2, cfg.t_in, 32, 64, 3)
        with torch.no_grad():
            y = m(x)
        self.assertEqual(y.shape, (2, cfg.t_out, 32, 64, 3))
        self.assertTrue(torch.isfinite(y).all())

    def test_rans_mean_channel_shifts_every_frame_equally(self):
        """The mean field is time-constant by construction: adding a constant to it must
        move all t_out frames by the same amount, or the split is not a decomposition."""
        cfg = FlowerConfig(lifting_dim=16, n_levels=1, c_in=3, c_out=3, rans=True)
        m = FlowerS2S(cfg).eval()
        x = torch.randn(1, cfg.t_in, 32, 64, 3)
        with torch.no_grad():
            y0 = m(x)
            last = [mod for mod in m.core.modules() if isinstance(mod, torch.nn.Conv2d)][-1]
            last.bias[cfg.frames_per_pass * cfg.c_out:] += 1.0
            y1 = m(x)
        d = (y1 - y0)
        self.assertTrue(torch.allclose(d, d[:, :1].expand_as(d), atol=1e-5),
                        "the RANS mean did not shift every frame equally")


class PerChannelRelL2(unittest.TestCase):
    """rel_l2 flattens u and v into ONE norm, so the ratio is dominated by whichever
    channel carries the energy -- and on this data sd(u)=0.106 against sd(v)=0.0157,
    about 7x. The score normalises over the measured channels together, so v's error is
    scored but barely trained. rel_l2_per_channel scores each channel's own ratio."""

    def test_equal_weights_upweight_the_small_channel(self):
        from pde_competition.model import rel_l2, rel_l2_per_channel
        torch.manual_seed(0)
        t = torch.stack([torch.randn(2, 20, 32, 64) * 0.106,
                         torch.randn(2, 20, 32, 64) * 0.0157], dim=-1)
        p = t.clone()
        p[..., 1] += 0.0157 * 0.5                       # a 50% relative error in v only
        flat = float(rel_l2(p, t))
        perch = float(rel_l2_per_channel(p, t, (1.0, 1.0)))
        self.assertGreater(perch, flat,
                           "an error confined to the small channel must cost MORE per-channel than flattened")

    def test_it_reduces_to_rel_l2_when_the_channels_match(self):
        from pde_competition.model import rel_l2, rel_l2_per_channel
        torch.manual_seed(0)
        t = torch.randn(3, 20, 8, 8, 2)
        p = t * 1.1                                      # identical RELATIVE error in both
        self.assertAlmostEqual(float(rel_l2_per_channel(p, t, (1.0, 1.0))), float(rel_l2(p, t)), places=4)

    def test_weights_are_normalised_so_the_scale_is_unchanged(self):
        from pde_competition.model import rel_l2_per_channel
        torch.manual_seed(0)
        t = torch.randn(2, 20, 8, 8, 2); p = t * 1.2
        a = float(rel_l2_per_channel(p, t, (1.0, 1.0)))
        b = float(rel_l2_per_channel(p, t, (2.0, 2.0)))
        self.assertAlmostEqual(a, b, places=5, msg="doubling both weights must not change the loss scale")


# ---------------------------------------------------------------------------
# E3 / L8: the diversity objective for CO-TRAINED ensemble members.
# ---------------------------------------------------------------------------

def _members(spread, b=2, t=4, h=8, w=8, c=2, m=3, seed=0):
    """M member predictions that agree up to ``spread``, plus a target."""
    g = torch.Generator().manual_seed(seed)
    tgt = torch.randn(b, t, h, w, c, generator=g)
    base = tgt + 0.1 * torch.randn(b, t, h, w, c, generator=g)
    dev = torch.randn(m, b, t, h, w, c, generator=g)
    dev = dev - dev.mean(0)                       # exactly zero-mean over members
    return base.unsqueeze(0) + spread * dev, tgt


class EnsembleAmbiguity(unittest.TestCase):
    """The diversity term co-trained members are coupled by (E3 / L8).

    It is the ONLY unweighted candidate: "negative mean pairwise MSE" and
    "negative variance of the members" are the same quantity up to a factor 2.
    """

    def test_identical_members_have_no_disagreement(self):
        from pde_competition.model import ensemble_ambiguity
        p, t = _members(0.0)
        self.assertAlmostEqual(float(ensemble_ambiguity(p, t)), 0.0, places=6)
        self.assertAlmostEqual(float(ensemble_ambiguity(p, t, True)), 0.0, places=6)

    def test_it_is_the_pairwise_identity_not_a_second_candidate(self):
        """(1/M^2) sum_ij ||p_i - p_j||^2 == 2 * (1/M) sum_m ||p_m - pbar||^2, exactly.
        So rewarding pairwise disagreement and rewarding member variance are ONE term."""
        from pde_competition.model import ensemble_ambiguity
        p, t = _members(0.3)
        m, n = p.shape[0], p.shape[1]
        f = p.reshape(m, n, -1)
        pair = sum(((f[i] - f[j]) ** 2).sum(dim=1) for i in range(m) for j in range(m)) / m ** 2
        want = ((0.5 * pair).sqrt() / torch.linalg.norm(t.reshape(n, -1), dim=1)).mean()
        self.assertAlmostEqual(float(ensemble_ambiguity(p, t)), float(want), places=5)

    def test_it_is_in_rel_l2_units_homogeneous_degree_one(self):
        """Doubling the spread doubles it, and rescaling the whole problem leaves it
        alone -- both required for lambda to be read against rel_l2."""
        from pde_competition.model import ensemble_ambiguity
        p1, t = _members(0.1)
        p2, _ = _members(0.2)
        self.assertAlmostEqual(float(ensemble_ambiguity(p2, t)),
                               2 * float(ensemble_ambiguity(p1, t)), places=5)
        self.assertAlmostEqual(float(ensemble_ambiguity(7.5 * p1, 7.5 * t)),
                               float(ensemble_ambiguity(p1, t)), places=5)

    def test_the_gradient_magnitude_does_not_depend_on_how_far_apart_they_are(self):
        """The docstring's claim, and the reason for the square root: the push is a
        constant 1/M of the accuracy push whatever the disagreement, so it neither
        vanishes while the members are nearly identical nor blows up."""
        from pde_competition.model import ensemble_ambiguity
        norms = []
        for spread in (0.001, 0.05, 1.0):
            p, t = _members(spread)
            p = p.clone().requires_grad_(True)
            ensemble_ambiguity(p, t).backward()
            norms.append(float(p.grad.norm()))
        self.assertLess(max(norms) / min(norms), 1.05,
                        f"gradient magnitude moved with the spread: {norms}")

    def test_error_weighting_moves_the_reward_to_where_the_mean_is_wrong(self):
        """The variant that matches the mechanism. Same member spread in both cases;
        only WHERE the ensemble mean is wrong changes."""
        from pde_competition.model import ensemble_ambiguity
        t = torch.ones(1, 2, 4, 4, 1)
        dev = torch.zeros(3, 1, 2, 4, 4, 1)
        dev[0, :, 0] = 1.0; dev[1, :, 0] = -1.0          # members disagree on frame 0 only
        hit = t.clone(); hit[:, 0] += 0.5                 # mean is wrong on frame 0
        miss = t.clone(); miss[:, 1] += 0.5               # mean is wrong on frame 1
        a_hit = float(ensemble_ambiguity(hit.unsqueeze(0) + 0.1 * dev, t, True))
        a_miss = float(ensemble_ambiguity(miss.unsqueeze(0) + 0.1 * dev, t, True))
        self.assertGreater(a_hit, 3 * a_miss,
                           "error weighting did not concentrate on the wrong region")

    def test_error_weighting_is_renormalised_so_being_worse_buys_nothing(self):
        """w is detached AND scaled to mean 1, so the total reward per window is the
        same however large the errors are -- only its allocation moves. Without the
        renormalisation the term would pay the model for being wrong."""
        from pde_competition.model import ensemble_ambiguity
        p, t = _members(0.2)
        far = t + 5.0 * (p.mean(0) - t)                   # same error PATTERN, 5x bigger
        p_far = far.unsqueeze(0) + (p - p.mean(0))        # identical member deviations
        self.assertAlmostEqual(float(ensemble_ambiguity(p_far, t, True)),
                               float(ensemble_ambiguity(p, t, True)), places=5)

    def test_error_weighting_cannot_be_amplified_by_concentrating_the_error(self):
        """The bug that diverged arm D of the E3 sweep on 2026-09-21, in one test.
        Renormalising w to mean 1 fixes the TOTAL reward but not its CONCENTRATION:
        all the weight on one element multiplies the reward for disagreeing there by
        sqrt(N). ERR_WEIGHT_CAP bounds the weighted term by sqrt(CAP) times the
        unweighted one, and that is the bound asserted here -- because the first draft
        of the cap claimed a stronger one (<= the unweighted term) which is false, and
        this test is what caught it."""
        import math
        from pde_competition.model import ERR_WEIGHT_CAP, ensemble_ambiguity
        t = torch.ones(1, 4, 16, 16, 1)
        n = t[0].numel()
        dev = torch.zeros(3, 1, 4, 16, 16, 1)
        dev[0, 0, 0, 0, 0, 0] = 1.0; dev[1, 0, 0, 0, 0, 0] = -1.0   # disagree on ONE element
        point = t.clone(); point[0, 0, 0, 0, 0] += 0.1              # wrong on that ONE element
        spread = t.clone() + 0.1                                    # wrong everywhere, evenly
        plain = float(ensemble_ambiguity(t.unsqueeze(0) + dev, t))
        conc = float(ensemble_ambiguity(point.unsqueeze(0) + dev, t, True))
        even = float(ensemble_ambiguity(spread.unsqueeze(0) + dev, t, True))
        self.assertLessEqual(conc, math.sqrt(ERR_WEIGHT_CAP) * plain + 1e-6,
                             f"the cap did not hold: {conc} > sqrt({ERR_WEIGHT_CAP}) * {plain}")
        self.assertLess(conc, 0.2 * (n ** 0.5) * plain,
                        "still within reach of the uncapped sqrt(N) amplification")
        self.assertAlmostEqual(even, plain, places=6,
                               msg="a uniformly wrong mean must reduce err to var")

    def test_unmeasured_elements_earn_no_error_weighted_reward(self):
        """Exact zeros in the target are PIV holes. coverage_surrogate drops them and
        so does this; the unweighted variant keeps rel_l2's own convention and does not."""
        from pde_competition.model import ensemble_ambiguity
        t = torch.randn(2, 3, 4, 4, 1)
        t[:, :, :2] = 0.0                                  # half the grid unmeasured
        dev = torch.zeros(3, 2, 3, 4, 4, 1)
        dev[0, :, :, :2] = 1.0; dev[1, :, :, :2] = -1.0    # disagreement only in the holes
        p = t.unsqueeze(0) + 0.2 * dev
        self.assertLess(float(ensemble_ambiguity(p, t, True)), 1e-5)
        self.assertGreater(float(ensemble_ambiguity(p, t, False)), 0.0)

    def test_it_refuses_a_single_member_and_a_shape_mismatch(self):
        from pde_competition.model import ensemble_ambiguity
        p, t = _members(0.1)
        with self.assertRaises(ValueError):
            ensemble_ambiguity(p[:1], t)
        with self.assertRaises(ValueError):
            ensemble_ambiguity(p, t[:1])


def _train_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "trn_e3", str(Path(__file__).resolve().parents[1] / "scripts" / "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny():
    from pde_competition.model import FlowerConfig, FlowerS2S
    torch.manual_seed(0)
    return FlowerS2S(FlowerConfig(lifting_dim=16, n_levels=1, num_heads=8, groups=8,
                                  c_in=2, c_out=2))


class CoTrainingSymmetryBreak(unittest.TestCase):
    """--init-from gives every co-trained member the same weights and co-training gives
    them the same batches, so without a perturbation they stay bit-identical forever:
    the diversity gradient is exactly zero when all members agree, which makes identical
    members a FIXED POINT rather than an unstable equilibrium."""

    def test_the_jitter_is_the_size_it_says_and_is_seed_reproducible(self):
        m = _train_module()
        a, b, c = _tiny(), _tiny(), _tiny()
        moved = m.jitter_member(a, 7, 0.05)
        self.assertAlmostEqual(moved, 0.05, delta=0.015)
        m.jitter_member(b, 7, 0.05)
        m.jitter_member(c, 8, 0.05)
        ka = dict(a.named_parameters())
        self.assertTrue(all(torch.allclose(v, dict(b.named_parameters())[k])
                            for k, v in ka.items()), "same seed gave a different jitter")
        self.assertFalse(all(torch.allclose(v, dict(c.named_parameters())[k])
                             for k, v in ka.items()), "a different seed gave the same jitter")

    def test_it_does_not_touch_the_normalisation_buffers(self):
        """mean/std are fitted statistics of the split, not parameters. Perturbing them
        would make the members disagree about what the data IS."""
        m = _train_module()
        mdl = _tiny()
        mdl.set_stats(np.array([0.5, 0.25], np.float32), np.array([2.0, 3.0], np.float32))
        m.jitter_member(mdl, 1, 0.1)
        self.assertTrue(torch.allclose(mdl.mean, torch.tensor([0.5, 0.25])))
        self.assertTrue(torch.allclose(mdl.std, torch.tensor([2.0, 3.0])))


class CoTrainingEpoch(unittest.TestCase):
    """The co-trained epoch. The control has to be a real control: at lambda = 0 every
    member must receive exactly the gradient a solo run would have given it."""

    def _batch(self, b=2):
        g = torch.Generator().manual_seed(11)
        x = torch.randn(b, T_IN, H_S, W_S, 2, generator=g)
        y = torch.randn(b, T_OUT, H_S, W_S, 2, generator=g)
        return [(x, y)]

    def _members(self, n=2):
        mm = [_tiny() for _ in range(n)]
        trn = _train_module()
        for k, mdl in enumerate(mm):
            trn.jitter_member(mdl, 100 + k, 0.05)
        return mm, trn

    def test_lambda_zero_reproduces_the_solo_gradient_exactly(self):
        from pde_competition.model import coverage_surrogate, rel_l2, tke_rel_l2
        mm, trn = self._members()
        batch = self._batch()
        half = torch.tensor([0.0093, 0.0063])
        # lr 0: the step runs but moves nothing, so .grad is the epoch's gradient.
        opt = torch.optim.AdamW([q for mdl in mm for q in mdl.parameters()], lr=0.0)
        trn.run_epoch_co(mm, batch, torch.device("cpu"), 2, opt, None,
                         tke_w=0.08, cov_w=0.4, cov_half=half, co_lambda=0.0)
        got = [q.grad.clone() for q in mm[0].parameters()]
        # the same member, alone, with the solo objective
        mm[0].zero_grad(set_to_none=True)
        x, y = batch[0]
        pred, tgt = mm[0](x)[..., :2], y[..., :2]
        (rel_l2(pred, tgt) + 0.08 * tke_rel_l2(pred, tgt)
         + 0.4 * coverage_surrogate(pred, tgt, half)).backward()
        torch.nn.utils.clip_grad_norm_(mm[0].parameters(), 1.0)
        for a, b in zip(got, [q.grad for q in mm[0].parameters()]):
            self.assertTrue(torch.allclose(a, b, atol=1e-6),
                            "lambda=0 is not the solo gradient, so it is not a control")

    def test_a_positive_lambda_changes_the_gradient(self):
        mm, trn = self._members()
        batch = self._batch()
        half = torch.tensor([0.0093, 0.0063])
        grads = {}
        for lam in (0.0, 0.5):
            for mdl in mm:
                mdl.zero_grad(set_to_none=True)
            opt = torch.optim.AdamW([q for mdl in mm for q in mdl.parameters()], lr=0.0)
            trn.run_epoch_co(mm, batch, torch.device("cpu"), 2, opt, None,
                             tke_w=0.08, cov_w=0.4, cov_half=half, co_lambda=lam)
            grads[lam] = torch.cat([q.grad.flatten() for q in mm[0].parameters()])
        self.assertFalse(torch.allclose(grads[0.0], grads[0.5], atol=1e-7),
                         "the diversity term had no effect on the gradient")

    def test_it_reports_member_error_ambiguity_and_the_ensemble(self):
        mm, trn = self._members()
        half = torch.tensor([0.0093, 0.0063])
        opt = torch.optim.AdamW([q for mdl in mm for q in mdl.parameters()], lr=0.0)
        out = trn.run_epoch_co(mm, self._batch(), torch.device("cpu"), 2, opt, None,
                               tke_w=0.08, cov_w=0.4, cov_half=half, co_lambda=0.3)
        self.assertEqual(set(out), {"rel_l2", "ens_rel_l2", "ambiguity"})
        self.assertGreater(out["ambiguity"], 0.0, "jittered members did not disagree at all")
        # Jensen: ||mean(p) - y|| <= mean ||p_m - y||, so the ensemble can never be worse.
        self.assertLessEqual(out["ens_rel_l2"], out["rel_l2"] + 1e-6)

    def test_bootstrap_resamples_the_own_loss_but_not_the_diversity_term(self):
        mm, trn = self._members()
        half = torch.tensor([0.0093, 0.0063])
        opt = torch.optim.AdamW([q for mdl in mm for q in mdl.parameters()], lr=0.0)
        torch.manual_seed(5)
        a = trn.run_epoch_co(mm, self._batch(b=4), torch.device("cpu"), 2, opt, None,
                             tke_w=0.08, cov_w=0.4, cov_half=half, co_lambda=0.3)
        torch.manual_seed(5)
        b = trn.run_epoch_co(mm, self._batch(b=4), torch.device("cpu"), 2, opt, None,
                             tke_w=0.08, cov_w=0.4, cov_half=half, co_lambda=0.3,
                             bootstrap=True)
        self.assertAlmostEqual(a["ambiguity"], b["ambiguity"], places=6,
                               msg="the diversity term must see the whole batch")
        self.assertNotAlmostEqual(a["rel_l2"], b["rel_l2"], places=6,
                                  msg="bootstrap did not change the member's own loss")

    def test_evaluate_ens_scores_the_mean_and_the_members_separately(self):
        mm, trn = self._members(3)
        out = trn.evaluate_ens(mm, self._batch(), torch.device("cpu"), 2)
        self.assertEqual(set(out), {"rel_l2", "tke", "member_rel_l2", "member_tke",
                                    "ambiguity"})
        self.assertLessEqual(out["rel_l2"], out["member_rel_l2"] + 1e-6,
                             "the ensemble mean cannot have a larger relative L2 than "
                             "the mean of its members (Jensen)")
        self.assertGreater(out["ambiguity"], 0.0)


# ---------------------------------------------------------------------------
# DIRECTION A7 (2026-09-23): the phase objectives. `fluct_corr` is the new one;
# `amse_time` shipped for SPEC on the same day and never had a test, because its
# only run (Slurm 4936) died on an import race 84 s after the job started.
# ---------------------------------------------------------------------------

def _fluct_pair(b=3, t=T_OUT, h=6, w=8, c=2, noise=0.7, seed=0):
    """A target and a partially-correlated prediction, both (B, T, H, W, C)."""
    g = torch.Generator().manual_seed(seed)
    tgt = torch.randn(b, t, h, w, c, generator=g)
    pred = tgt + noise * torch.randn(b, t, h, w, c, generator=g)
    return pred, tgt


def _corr_by_hand(pred, tgt):
    """blur_diagnostic.py's own per-lead number, in numpy, written independently."""
    p = pred.detach().numpy()[..., :2]
    t = tgt.detach().numpy()[..., :2]
    p = p - p.mean(axis=1, keepdims=True)
    t = t - t.mean(axis=1, keepdims=True)
    out = []
    for k in range(p.shape[1]):
        num = float((p[:, k] * t[:, k]).mean())
        den = float(np.sqrt((p[:, k] ** 2).mean() * (t[:, k] ** 2).mean()))
        out.append(num / den)
    return float(np.mean([1.0 - c for c in out]))


class FluctCorr(unittest.TestCase):
    """--corr-weight: 1 - the per-lead fluctuation correlation (A7)."""

    def test_zero_at_identity_and_positive_otherwise(self):
        from pde_competition.model import fluct_corr
        pred, tgt = _fluct_pair()
        self.assertAlmostEqual(float(fluct_corr(tgt, tgt)), 0.0, places=5)
        v = float(fluct_corr(pred, tgt))
        self.assertGreater(v, 0.0)
        self.assertLess(v, 2.0)                  # corr in [-1, 1] bounds the loss to [0, 2]
        self.assertEqual(fluct_corr(pred, tgt).shape, torch.Size([]))

    def test_it_is_the_blur_diagnostic_number(self):
        """The claim in the docstring: this loss IS scripts/blur_diagnostic.py's
        per-lead corr, averaged over leads. Checked against an independent numpy
        transcription of that script's formula."""
        from pde_competition.model import fluct_corr
        pred, tgt = _fluct_pair(seed=3)
        self.assertAlmostEqual(float(fluct_corr(pred, tgt)), _corr_by_hand(pred, tgt),
                               places=5)

    def test_scale_invariance_is_exact(self):
        """THE LOAD-BEARING PROPERTY. r/corr^2 = 1.34-1.78 says the prediction is already
        over-amplified relative to the L2 optimum, so a phase term that could be paid off
        in energy would just walk VINF's frontier. Rescaling the prediction must not move
        this term by anything."""
        from pde_competition.model import fluct_corr
        pred, tgt = _fluct_pair(seed=5)
        base = float(fluct_corr(pred, tgt))
        for s in (0.25, 1.07, 4.0):
            self.assertAlmostEqual(float(fluct_corr(s * pred, tgt)), base, places=5,
                                   msg=f"scaling the prediction by {s} moved the phase term")
        # and the gradient carries no component along the scaling direction
        p = pred.clone().requires_grad_(True)
        fluct_corr(p, tgt).backward()
        pf = p.detach() - p.detach().mean(dim=1, keepdim=True)
        self.assertAlmostEqual(float((p.grad * pf).sum()), 0.0,
                               delta=1e-4 * float(p.grad.norm() * pf.norm()))

    def test_the_time_mean_is_removed_from_both_sides(self):
        """A per-pixel, time-constant offset on the prediction is the mean flow, which
        mvpe owns and this term must not see."""
        from pde_competition.model import fluct_corr
        pred, tgt = _fluct_pair(seed=7)
        g = torch.Generator().manual_seed(8)
        off = torch.randn(pred.shape[0], 1, *pred.shape[2:], generator=g)
        self.assertAlmostEqual(float(fluct_corr(pred + off, tgt)),
                               float(fluct_corr(pred, tgt)), places=5)

    def test_a_better_correlated_prediction_scores_lower(self):
        from pde_competition.model import fluct_corr
        good, tgt = _fluct_pair(noise=0.2, seed=9)
        bad, _ = _fluct_pair(noise=2.0, seed=9)
        self.assertLess(float(fluct_corr(good, tgt)), float(fluct_corr(bad, tgt)))

    def test_it_is_differentiable_and_adds_to_the_loss(self):
        from pde_competition.model import fluct_corr
        pred, tgt = _fluct_pair(seed=11)
        p = pred.clone().requires_grad_(True)
        base = rel_l2(p, tgt)
        loss = base + 0.2 * fluct_corr(p, tgt)
        self.assertGreater(loss.item(), base.item())
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(p.grad.abs().sum().item(), 0.0)

    def test_the_unmeasured_third_channel_is_dropped(self):
        """Real PIV has no pressure; blur_diagnostic and amse_time both slice to [u, v]
        and this must agree, or a 3-channel run would optimise a channel nobody scores."""
        from pde_competition.model import fluct_corr
        pred, tgt = _fluct_pair(c=3, seed=13)
        moved = pred.clone()
        moved[..., 2] += 5.0
        self.assertAlmostEqual(float(fluct_corr(moved, tgt)), float(fluct_corr(pred, tgt)),
                               places=6)

    def test_a_degenerate_prediction_does_not_produce_nan(self):
        """A constant-in-time prediction has no fluctuation at all: eps must catch the
        zero denominator rather than returning nan into the optimiser."""
        from pde_competition.model import fluct_corr
        _pred, tgt = _fluct_pair(seed=17)
        flat = torch.zeros_like(tgt)
        v = fluct_corr(flat, tgt)
        self.assertTrue(torch.isfinite(v))
        self.assertAlmostEqual(float(v), 1.0, places=5)     # corr = 0


class AmseTime(unittest.TestCase):
    """--amse-weight (SPEC). Never had a test: 4936 died before it ran."""

    def test_zero_at_identity_and_positive_otherwise(self):
        from pde_competition.model import amse_time
        pred, tgt = _fluct_pair(seed=1)
        self.assertAlmostEqual(float(amse_time(tgt, tgt)), 0.0, places=5)
        self.assertGreater(float(amse_time(pred, tgt)), 0.0)

    def test_it_is_differentiable_and_finite(self):
        from pde_competition.model import amse_time
        pred, tgt = _fluct_pair(seed=2)
        p = pred.clone().requires_grad_(True)
        v = amse_time(p, tgt)
        self.assertTrue(torch.isfinite(v))
        v.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(p.grad.abs().sum().item(), 0.0)

    def test_it_drops_the_third_channel_like_fluct_corr(self):
        from pde_competition.model import amse_time
        pred, tgt = _fluct_pair(c=3, seed=4)
        moved = pred.clone()
        moved[..., 2] += 5.0
        self.assertAlmostEqual(float(amse_time(moved, tgt)), float(amse_time(pred, tgt)),
                               places=5)

    def test_at_batch_one_the_coherence_half_vanishes(self):
        """Coherence is pooled over the batch, and a one-sample estimate is identically 1
        by Cauchy-Schwarz. So at B = 1 amse_time is its AMPLITUDE term alone -- which is
        exactly the 'put the energy back' mechanism VINF already prices. Not a bug, but it
        means a B = 1 arm is not testing what the flag claims to test."""
        from pde_competition.model import amse_time
        pred, tgt = _fluct_pair(b=1, seed=6)
        p = pred[..., :2] - pred[..., :2].mean(dim=1, keepdim=True)
        t = tgt[..., :2] - tgt[..., :2].mean(dim=1, keepdim=True)
        Pf = torch.fft.rfft(p, dim=1)[:, 1:]
        Tf = torch.fft.rfft(t, dim=1)[:, 1:]
        amp = ((Pf.real ** 2 + Pf.imag ** 2).clamp_min(1e-8).sqrt()
               - (Tf.real ** 2 + Tf.imag ** 2).clamp_min(1e-8).sqrt()) ** 2
        want = amp.mean() / (Tf.real ** 2 + Tf.imag ** 2).mean().clamp_min(1e-8)
        self.assertAlmostEqual(float(amse_time(pred, tgt)), float(want), places=5)


class CorrWeightDefaultsOff(unittest.TestCase):
    """--corr-weight 0.0 must leave the default training path bit-identical."""

    def _batch(self, b=2):
        g = torch.Generator().manual_seed(11)
        x = torch.randn(b, T_IN, H_S, W_S, 2, generator=g)
        y = torch.randn(b, T_OUT, H_S, W_S, 2, generator=g)
        return [(x, y)]

    def test_the_argparse_default_is_zero(self):
        from unittest import mock
        trn = _train_module()
        with mock.patch("sys.argv", ["train.py"]):        # parse_args() reads sys.argv itself
            a = trn.parse_args()
        self.assertEqual(a.corr_weight, 0.0)
        self.assertEqual(a.amse_weight, 0.0)

    def test_weight_zero_gives_the_same_weights_as_not_passing_it(self):
        trn = _train_module()
        outs = []
        for kw in ({}, {"corr_w": 0.0}):
            torch.manual_seed(3)
            mdl = _tiny()
            opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
            trn.run_epoch(mdl, self._batch(), torch.device("cpu"), 2, opt, None,
                          tke_w=0.08, **kw)
            outs.append([q.detach().clone() for q in mdl.parameters()])
        for a, b in zip(*outs):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_a_nonzero_weight_actually_changes_the_step(self):
        trn = _train_module()
        outs = []
        for w in (0.0, 0.5):
            torch.manual_seed(3)
            mdl = _tiny()
            opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
            trn.run_epoch(mdl, self._batch(), torch.device("cpu"), 2, opt, None,
                          tke_w=0.08, corr_w=w)
            outs.append([q.detach().clone() for q in mdl.parameters()])
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(*outs)),
                         "--corr-weight 0.5 left every parameter where 0.0 did")
