"""INFERENCE-TIME ROLLOUT in scripts/variance_inflation.py (--rollout-blend, 2026-09-24). Numpy only, no data.

    x2 = concat(x[:, S:], fb)          fb = p0[:, :S], PIV holes of x's last frame zeroed, channel 2 carried
    p'[:, S:] = (1-W) p0[:, S:] + W p2[:, :T-S]        frames [0, S) untouched, bounds pinned by the caller
    --rollout-ramp: W is per frame, w(j) = W19 (j-S)/(19-S) for j in [S, 20)  (ramp_weights)

Tiny arrays; every value is distinct so a misplaced frame or channel cannot pass by coincidence.
"""
import sys
import unittest

import numpy as np

from pde_competition.paths import REPO

sys.path.insert(0, str(REPO / "scripts"))
import variance_inflation as vi  # noqa: E402

N, T, H, W, C = 2, 20, 3, 4, 3
HOLE = (1, 2, 3)            # (n, h, w): u == 0 exactly in x's LAST input frame -> a PIV hole


def _arrays(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, 1.5, size=(N, T, H, W, C)).astype(np.float32)   # u never 0 except the hole
    x[..., 2] = 0.0                                                        # real PIV ships no pressure
    n, h, w = HOLE
    x[n, -1, h, w, :] = 0.0
    p0 = rng.normal(size=(N, T, H, W, C)).astype(np.float32)
    p2 = rng.normal(size=(N, T, H, W, C)).astype(np.float32)
    return x, p0, p2


class RolloutInput(unittest.TestCase):
    def setUp(self):
        self.x, self.p0, _ = _arrays()

    def test_places_the_input_tail_then_the_prediction_head(self):
        for s in (1, 5, 19):
            x2 = vi.rollout_input(self.x, self.p0, s)
            self.assertEqual(x2.shape, (N, T, H, W, C))
            self.assertEqual(x2.dtype, np.float32)
            self.assertEqual(x2[:, :T - s].tobytes(), self.x[:, s:].tobytes(), msg=f"S={s}")
            fb = x2[:, T - s:]
            ok = np.ones((N, H, W), bool); ok[HOLE] = False
            for k in range(s):
                self.assertTrue(np.array_equal(fb[:, k][ok][:, :2], self.p0[:, k][ok][:, :2]), msg=f"S={s} k={k}")

    def test_channel_two_of_the_fed_back_frames_is_zero_like_the_input(self):
        p0 = self.p0.copy()
        p0[..., 2] = 7.0                                  # the model's own channel 2 must never leak in
        x2 = vi.rollout_input(self.x, p0, 5)
        self.assertTrue(np.all(x2[:, -5:, ..., 2] == 0))
        self.assertFalse(np.any(np.signbit(x2[:, -5:, ..., 2])))

    def test_channel_two_is_carried_from_the_last_input_frame(self):
        x = self.x.copy()
        x[..., 2] = np.arange(T, dtype=np.float32)[None, :, None, None]   # distinct per frame
        x2 = vi.rollout_input(x, self.p0, 4)
        self.assertTrue(np.all(x2[:, -4:, ..., 2] == x[:, -1:, ..., 2]))

    def test_holes_of_the_last_input_frame_are_zeroed_in_every_fed_back_frame(self):
        n, h, w = HOLE
        self.assertTrue(np.all(self.p0[n, :5, h, w, :2] != 0))     # the prediction there is not zero
        x2 = vi.rollout_input(self.x, self.p0, 5)
        self.assertTrue(np.all(x2[n, -5:, h, w, :] == 0))
        self.assertFalse(np.any(np.signbit(x2[n, -5:, h, w, :])))   # +0.0, not -0.0
        # the other window has no hole there and keeps the prediction
        self.assertTrue(np.array_equal(x2[1 - n, -5:, h, w, :2], self.p0[1 - n, :5, h, w, :2]))

    def test_inputs_are_not_modified_and_the_result_is_fresh(self):
        x0, p00 = self.x.copy(), self.p0.copy()
        x2 = vi.rollout_input(self.x, self.p0, 3)
        self.assertEqual(self.x.tobytes(), x0.tobytes())
        self.assertEqual(self.p0.tobytes(), p00.tobytes())
        self.assertFalse(np.shares_memory(x2, self.x))
        self.assertFalse(np.shares_memory(x2, self.p0))

    def test_shift_outside_1_to_19_raises(self):
        for s in (0, 20, -1, 25, 2.5):
            with self.assertRaises(ValueError, msg=f"S={s}"):
                vi.rollout_input(self.x, self.p0, s)

    def test_shape_mismatches_raise(self):
        with self.assertRaises(ValueError):
            vi.rollout_input(self.x[:1], self.p0, 5)                  # N differs
        with self.assertRaises(ValueError):
            vi.rollout_input(self.x[:, :, :2], self.p0, 5)            # H differs
        with self.assertRaises(ValueError):
            vi.rollout_input(self.x[0], self.p0, 5)                   # not 5-D
        with self.assertRaises(ValueError):
            vi.rollout_input(self.x, self.p0[..., :1], 5)             # fewer channels than measured


class RolloutBlend(unittest.TestCase):
    def setUp(self):
        _, self.p0, self.p2 = _arrays(1)

    def test_w_zero_is_the_identity_bitwise(self):
        p0 = self.p0.copy()
        p0[0, 10, 0, 0, 0] = -0.0
        out = vi.rollout_blend(p0, self.p2, 5, 0.0)
        self.assertEqual(out.tobytes(), p0.tobytes())
        self.assertIsNot(out, p0)
        self.assertFalse(np.shares_memory(out, p0))

    def test_w_one_replaces_exactly_frames_s_to_20(self):
        p2 = self.p2.copy()
        p2[0, 0, 0, 0, 0] = -0.0                                      # lands at frame S: must stay -0.0
        for s in (1, 5, 19):
            out = vi.rollout_blend(self.p0, p2, s, 1.0)
            self.assertEqual(out[:, s:].tobytes(), p2[:, :T - s].tobytes(), msg=f"S={s}")
            self.assertEqual(out[:, :s].tobytes(), self.p0[:, :s].tobytes(), msg=f"S={s}")

    def test_intermediate_w_blends_the_overlap_and_leaves_the_head_alone(self):
        s, w = 5, 0.25
        out = vi.rollout_blend(self.p0, self.p2, s, w)
        self.assertEqual(out[:, :s].tobytes(), self.p0[:, :s].tobytes())
        want = 0.75 * self.p0[:, s:].astype(np.float64) + 0.25 * self.p2[:, :T - s].astype(np.float64)
        self.assertTrue(np.allclose(out[:, s:], want, atol=1e-5, rtol=0))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, self.p0.shape)

    def test_inputs_are_not_modified(self):
        a, b = self.p0.copy(), self.p2.copy()
        vi.rollout_blend(self.p0, self.p2, 7, 0.5)
        self.assertEqual(self.p0.tobytes(), a.tobytes())
        self.assertEqual(self.p2.tobytes(), b.tobytes())

    def test_shift_outside_1_to_19_raises(self):
        for s in (0, 20, -3, 1.5):
            with self.assertRaises(ValueError, msg=f"S={s}"):
                vi.rollout_blend(self.p0, self.p2, s, 0.5)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            vi.rollout_blend(self.p0, self.p2[:, :19], 5, 0.5)
        with self.assertRaises(ValueError):
            vi.rollout_blend(self.p0[0], self.p2[0], 5, 0.5)


class RampWeights(unittest.TestCase):
    """--rollout-ramp: w(j) = W19 (j - S) / (19 - S) over the overlap frames j in [S, 20)."""

    def test_length_endpoints_and_values(self):
        for s, w19 in ((1, 1.0), (5, 0.5), (10, 1.0), (18, 0.8)):
            w = vi.ramp_weights(s, w19)
            self.assertEqual(w.shape, (T - s,), msg=f"S={s}")
            self.assertEqual(w[0], 0.0, msg=f"S={s}")
            self.assertEqual(w[-1], w19, msg=f"S={s}")
            want = np.array([w19 * (j - s) / (19 - s) for j in range(s, 20)])
            self.assertTrue(np.allclose(w, want, atol=1e-15, rtol=0), msg=f"S={s}")
            self.assertTrue(np.all(np.diff(w) > 0), msg=f"S={s}")
        self.assertTrue(np.allclose(vi.ramp_weights(10, 1.0), np.arange(10) / 9.0, atol=1e-15, rtol=0))

    def test_other_t(self):
        w = vi.ramp_weights(2, 0.6, t=6)                 # leads 2..5 -> 0, 0.2, 0.4, 0.6
        self.assertTrue(np.allclose(w, [0.0, 0.2, 0.4, 0.6], atol=1e-15, rtol=0))

    def test_invalid_shift_refused(self):
        for s in (19, 0, 20, -1, 2.5):
            with self.assertRaises(ValueError, msg=f"S={s}"):
                vi.ramp_weights(s, 1.0)


class RolloutBlendPerFrame(unittest.TestCase):
    def setUp(self):
        _, self.p0, self.p2 = _arrays(2)

    def test_array_w_equals_a_manual_per_frame_blend(self):
        for s, w19 in ((5, 1.0), (10, 0.5), (1, 0.7)):
            w = vi.ramp_weights(s, w19)
            out = vi.rollout_blend(self.p0, self.p2, s, w)
            self.assertEqual(out.dtype, np.float32)
            self.assertEqual(out.shape, self.p0.shape)
            self.assertEqual(out[:, :s].tobytes(), self.p0[:, :s].tobytes(), msg=f"S={s}")
            for k in range(T - s):
                wk = float(w[k])
                want = (np.float32(1.0 - wk) * self.p0[:, s + k] + np.float32(wk) * self.p2[:, k]).astype(np.float32)
                if wk == 1.0:
                    want = self.p2[:, k]
                self.assertEqual(out[:, s + k].tobytes(), want.tobytes(), msg=f"S={s} k={k}")
                ref = (1 - wk) * self.p0[:, s + k].astype(np.float64) + wk * self.p2[:, k].astype(np.float64)
                self.assertTrue(np.allclose(out[:, s + k], ref, atol=1e-5, rtol=0), msg=f"S={s} k={k}")

    def test_zero_and_one_frames_are_bitwise(self):
        p0, p2 = self.p0.copy(), self.p2.copy()
        s = 10
        p0[0, s, 0, 0, 0] = -0.0                          # lead S has w = 0: must stay p0's -0.0
        p2[1, T - s - 1, 0, 0, 0] = -0.0                  # lead 19 has w = W19 = 1: must be p2's -0.0
        p2[0, 0, 1, 1, 1] = np.inf                        # an inf in p2 at a w = 0 frame must not leak
        out = vi.rollout_blend(p0, p2, s, vi.ramp_weights(s, 1.0))
        self.assertEqual(out[:, s].tobytes(), p0[:, s].tobytes())
        self.assertEqual(out[:, -1].tobytes(), p2[:, T - s - 1].tobytes())

    def test_constant_array_equals_the_scalar_bitwise(self):
        for s, w in ((5, 0.25), (10, 0.5), (3, 0.0), (7, 1.0)):
            a = vi.rollout_blend(self.p0, self.p2, s, np.full(T - s, w))
            b = vi.rollout_blend(self.p0, self.p2, s, w)
            self.assertEqual(a.tobytes(), b.tobytes(), msg=f"S={s} w={w}")

    def test_scalar_path_unchanged(self):
        """The scalar path is the pre-ramp formula, reproduced here verbatim."""
        for s, w in ((5, 0.25), (10, 0.5), (19, 0.75)):
            want = self.p0.copy()
            want[:, s:] = np.float32(1.0 - w) * self.p0[:, s:] + np.float32(w) * self.p2[:, :T - s]
            for ww in (w, np.float32(w), np.float64(w), np.array(w)):
                out = vi.rollout_blend(self.p0, self.p2, s, ww)
                self.assertEqual(out.tobytes(), want.tobytes(), msg=f"S={s} w={ww!r}")

    def test_wrong_length_array_raises(self):
        for w in (np.zeros(14), np.zeros(16), np.zeros((15, 1)), np.zeros((1, 15))):
            with self.assertRaises(ValueError, msg=f"shape {w.shape}"):
                vi.rollout_blend(self.p0, self.p2, 5, w)

    def test_inputs_are_not_modified(self):
        a, b = self.p0.copy(), self.p2.copy()
        vi.rollout_blend(self.p0, self.p2, 8, vi.ramp_weights(8, 0.9))
        self.assertEqual(self.p0.tobytes(), a.tobytes())
        self.assertEqual(self.p2.tobytes(), b.tobytes())


class RolloutEndToEnd(unittest.TestCase):
    def test_a_perfect_shift_invariant_model_reproduces_itself(self):
        """With a 'model' that is exact on a travelling sequence, the second pass forecasts the same frames
        as the first, so any W leaves p0 unchanged -- the frame bookkeeping (p2 index j-S <-> truth j) is
        right end to end."""
        seq = np.arange(60, dtype=np.float32)[None, :, None, None, None] + np.float32(1.0)
        seq = np.broadcast_to(seq, (N, 60, H, W, C)).copy()
        seq[..., 2] = 0.0
        x, y = seq[:, :20], seq[:, 20:40]

        def model(inp):                                   # forecast: the next 20 frames of the sequence
            start = int(inp[0, -1, 0, 0, 0])              # value k + 1 at absolute frame k
            return seq[:, start:start + 20].copy()

        p0 = model(x)
        self.assertEqual(p0.tobytes(), y.tobytes())
        for s in (1, 5, 19):
            p2 = model(vi.rollout_input(x, p0, s))
            self.assertEqual(p2[:, :T - s].tobytes(), y[:, s:].tobytes(), msg=f"S={s}")
            self.assertEqual(vi.rollout_blend(p0, p2, s, 0.5).tobytes(), y.tobytes(), msg=f"S={s}")
            if s <= T - 2:                                # (1-w)y + wy is y only to float32 rounding
                self.assertTrue(np.allclose(vi.rollout_blend(p0, p2, s, vi.ramp_weights(s, 1.0)), y,
                                            atol=0, rtol=1e-6), msg=f"ramp S={s}")


if __name__ == "__main__":
    unittest.main()
