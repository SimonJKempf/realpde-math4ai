"""The post-hoc MEAN ANCHOR in scripts/variance_inflation.py (diagnostic, 2026-09-24). Numpy only, no data.

    p' = p + alpha * (m_in - mean_t(p))     on channels [u, v]; channel 2 and the bounds untouched

m_in is the input window's per-pixel mean over MEASURED frames (exact 0.0 = unmeasured PIV hole, the same
convention as submission_flower._head_extra and anchor_diagnostic.input_stats); a pixel with no measured
frame is not shifted at all.

The fixture is built on a dyadic grid so that every sum, mean and shift is EXACT in float32: base levels are
multiples of 1/8 and every per-pixel fluctuation is a zero-sum symmetric ramp (t - 9.5)/8. That is what makes
the bitwise assertions (identity at alpha 0, unchanged fluctuation at alpha 1) meaningful rather than lucky.
A separate random-data test checks the same properties to 1e-6.
"""
import sys
import unittest

import numpy as np

from pde_competition.paths import REPO

sys.path.insert(0, str(REPO / "scripts"))
import variance_inflation as vi  # noqa: E402

N, T, H, W, C = 2, 20, 4, 6, 3
HOLE_HALF = (0, 1, 2)      # (n, h, w): input frames 0..9 are 0.0 (unmeasured), 10..19 measured
HOLE_ALL = (1, 3, 5)       # (n, h, w): no measured input frame at all


def _fixture():
    """pred (N,T,H,W,C), x (N,T,H,W,C), base level of pred (N,H,W,C) and of x (N,H,W,C), all float32,
    all exactly representable, with mean_t(pred) == base and masked mean_t(x) == xbase exactly."""
    idx = np.arange(N * H * W * C, dtype=np.float32).reshape(N, H, W, C)
    base = (idx % 13 - 6) / 8                                   # in [-0.75, 0.75], multiples of 1/8
    ramp = ((np.arange(T, dtype=np.float32) - 9.5) / 8)         # symmetric, zero-sum, multiples of 1/16
    pred = base[:, None] + ramp[None, :, None, None, None]
    xbase = np.empty((N, H, W, C), np.float32)
    xbase[..., 0] = (idx[..., 0] % 7 - 3) / 4 + 2.0             # u in [1.25, 2.75]: never 0 with the ramp
    xbase[..., 1] = (idx[..., 1] % 5 - 2) / 4                   # v around 0 (mask comes from u only)
    xbase[..., 2] = 0.0                                         # real PIV ships no pressure
    x = xbase[:, None] + ramp[None, :, None, None, None]
    x[..., 2] = 0.0
    n, h, w = HOLE_HALF
    x[n, :10, h, w, :] = 0.0
    ramp10 = (np.arange(10, dtype=np.float32) - 4.5) / 8        # zero-sum over the 10 measured frames
    x[n, 10:, h, w, :2] = xbase[n, h, w, :2][None] + ramp10[:, None]
    n, h, w = HOLE_ALL
    x[n, :, h, w, :] = 0.0
    return pred.astype(np.float32), x.astype(np.float32), base.astype(np.float32), xbase


def _masked_input_mean(x):
    """Independent float64 derivation of the measured-frames mean, (N,H,W,2), and the has-data mask (N,H,W)."""
    meas = (x[..., 0] != 0)
    cnt = meas.sum(1)
    m = (x[..., :2].astype(np.float64) * meas[..., None]).sum(1) / np.maximum(cnt, 1)[..., None]
    return m, cnt > 0


def _fluct(p):
    return p - p.mean(axis=1, keepdims=True)


class MeanAnchor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred, cls.x, cls.base, cls.xbase = _fixture()

    def test_alpha_zero_is_the_identity_bitwise(self):
        pred = self.pred.copy()
        pred[0, 0, 0, 0, 0] = -0.0                              # -0.0 + 0.0 would round to +0.0: the trap
        out, shift = vi.mean_anchor(pred, self.x, 0.0)
        self.assertEqual(out.tobytes(), pred.tobytes())
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(shift.shape, (N, 1, H, W, 2))
        self.assertTrue(np.all(shift == 0))
        self.assertIsNot(out, pred)                             # a copy, the caller's array is left alone

    def test_alpha_one_matches_the_masked_input_mean_and_keeps_the_fluctuation_bitwise(self):
        out, shift = vi.mean_anchor(self.pred, self.x, 1.0)
        m_in, ok = _masked_input_mean(self.x)
        m_out = out[..., :2].astype(np.float64).mean(1)
        self.assertTrue(np.allclose(m_out[ok], m_in[ok], atol=1e-6, rtol=0))
        self.assertTrue(np.array_equal(m_out[ok], self.xbase[..., :2][ok]))   # exact on the dyadic fixture
        self.assertEqual(_fluct(out)[..., :2].tobytes(), _fluct(self.pred)[..., :2].tobytes())
        # the shift is exactly (m_in - mean_t(p)) where there is data
        want = (self.xbase[..., :2] - self.base[..., :2])[:, None]
        self.assertTrue(np.array_equal(shift[ok[:, None]], want[ok[:, None]]))

    def test_intermediate_alpha_moves_a_fraction_of_the_gap(self):
        out, shift = vi.mean_anchor(self.pred, self.x, 0.25)
        _m_in, ok = _masked_input_mean(self.x)
        want = np.float32(0.25) * (self.xbase[..., :2] - self.base[..., :2])[:, None]
        self.assertTrue(np.array_equal(shift[ok[:, None]], want[ok[:, None]]))
        self.assertTrue(np.array_equal(out[..., :2], self.pred[..., :2] + shift))
        self.assertEqual(_fluct(out)[..., :2].tobytes(), _fluct(self.pred)[..., :2].tobytes())

    def test_channel_two_is_untouched(self):
        for alpha in (0.0, 0.25, 1.0, 2.0):
            out, _ = vi.mean_anchor(self.pred, self.x, alpha)
            self.assertEqual(out[..., 2].tobytes(), self.pred[..., 2].tobytes(), msg=f"alpha {alpha}")

    def test_input_zeros_are_excluded_from_the_input_mean(self):
        n, h, w = HOLE_HALF
        out, shift = vi.mean_anchor(self.pred, self.x, 1.0)
        measured = self.x[n, :, h, w, 0] != 0
        self.assertEqual(int(measured.sum()), 10)               # the fixture really has a half hole here
        over_measured = self.x[n, measured, h, w, :2].astype(np.float64).mean(0)
        over_all = self.x[n, :, h, w, :2].astype(np.float64).mean(0)   # what a naive mean would give
        self.assertFalse(np.allclose(over_measured, over_all))
        got = out[n, :, h, w, :2].astype(np.float64).mean(0)
        self.assertTrue(np.allclose(got, over_measured, atol=1e-6, rtol=0))
        self.assertTrue(np.array_equal(got, self.xbase[n, h, w, :2]))
        self.assertTrue(np.array_equal(shift[n, 0, h, w], self.xbase[n, h, w, :2] - self.base[n, h, w, :2]))

    def test_a_pixel_with_no_measured_input_gets_no_shift(self):
        n, h, w = HOLE_ALL
        self.assertTrue(np.all(self.x[n, :, h, w] == 0))
        for alpha in (0.5, 1.0):
            out, shift = vi.mean_anchor(self.pred, self.x, alpha)
            self.assertTrue(np.all(shift[n, 0, h, w] == 0), msg=f"alpha {alpha}")
            self.assertEqual(out[n, :, h, w].tobytes(), self.pred[n, :, h, w].tobytes(), msg=f"alpha {alpha}")
            # and every other pixel of that window did move (the no-shift is the convention, not a bug)
            other = np.ones((H, W), bool); other[h, w] = False
            self.assertTrue(np.all(shift[n, 0, other][:, 0] != 0))    # u: xbase >= 1.25 vs base <= 0.75

    def test_persist_source_anchors_to_the_last_input_frame(self):
        out, shift = vi.mean_anchor(self.pred, self.x, 1.0, source="persist")
        last = self.x[:, -1, ..., :2]
        ok = self.x[:, -1, ..., 0] != 0
        m_out = out[..., :2].astype(np.float64).mean(1)
        self.assertTrue(np.allclose(m_out[ok], last[ok], atol=1e-6, rtol=0))
        n, h, w = HOLE_ALL
        self.assertTrue(np.all(shift[n, 0, h, w] == 0))
        self.assertEqual(out[..., 2].tobytes(), self.pred[..., 2].tobytes())
        with self.assertRaises(ValueError):
            vi.mean_anchor(self.pred, self.x, 1.0, source="median")

    def test_random_data_alpha_one_to_1e6(self):
        """The dyadic fixture proves exactness; this proves the arithmetic on ordinary values."""
        rng = np.random.default_rng(0)
        pred = rng.normal(size=(N, T, H, W, C)).astype(np.float32) * np.float32(0.5)
        x = np.empty((N, T, H, W, C), np.float32)
        x[..., 0] = rng.uniform(0.2, 1.0, size=(N, T, H, W))
        x[..., 1] = rng.uniform(-0.5, 0.5, size=(N, T, H, W))
        x[..., 2] = 0.0
        x[0, :7, 2, 3, :] = 0.0                                  # a partial hole
        out, shift = vi.mean_anchor(pred, x, 1.0)
        m_in, ok = _masked_input_mean(x)
        self.assertTrue(ok.all())
        m_out = out[..., :2].astype(np.float64).mean(1)
        self.assertTrue(np.allclose(m_out, m_in, atol=1e-6, rtol=0))
        self.assertTrue(np.allclose(_fluct(out.astype(np.float64))[..., :2],
                                    _fluct(pred.astype(np.float64))[..., :2], atol=1e-6, rtol=0))
        self.assertEqual(out[..., 2].tobytes(), pred[..., 2].tobytes())
        out0, _ = vi.mean_anchor(pred, x, 0.0)
        self.assertEqual(out0.tobytes(), pred.tobytes())


if __name__ == "__main__":
    unittest.main()
