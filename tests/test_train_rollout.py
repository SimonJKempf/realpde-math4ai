"""Direction ROLL (2026-09-24): the training-time rollout consistency loss.

    python -m unittest tests.test_train_rollout

Seconds, CPU, no data. Two layers: ``WindowDataset(extra_out=...)`` on a tiny
synthetic store whose values ENCODE the frame index (the test_gth pattern), and
``scripts/train.py``'s ``run_epoch`` / ``rollout_input`` / ``rollout_term`` /
``parse_args`` on a tiny model (the test_distill pattern). The control has to be a
real control -- W = 0 and extra_out = 0 must be bit-identical to a run that never
heard of a rollout -- and the detach flag has to do exactly what it says.
"""
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from pde_competition.dataset import Trajectory, WindowDataset, window_starts
from pde_competition.geometry import H_S, T_IN, T_OUT, W_S
from pde_competition.model import FlowerConfig, FlowerS2S, rel_l2, tke_rel_l2

REPO = pathlib.Path(__file__).resolve().parents[1]
SPAN = T_IN + T_OUT
H_, W_ = 4, 6


def _train_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("trn_roll", str(REPO / "scripts" / "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny(c=2, lifting_dim=16, delta=False, mask_channel=False):
    torch.manual_seed(0)
    return FlowerS2S(FlowerConfig(lifting_dim=lifting_dim, n_levels=1, num_heads=8, groups=8,
                                  c_in=c, c_out=c, predict_delta=delta, mask_channel=mask_channel))


def _batch(b=2, c=2, seed=11, with_y2=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, T_IN, H_S, W_S, c, generator=g)
    y = torch.randn(b, T_OUT, H_S, W_S, c, generator=g)
    y2 = torch.randn(b, T_OUT, H_S, W_S, c, generator=g)
    if c == 3:
        x[..., 2] = 0.0; y[..., 2] = 0.0; y2[..., 2] = 0.0     # real data: p is identically zero
    x[:, :, :3, :, :] = 0.0                                    # a PIV hole across the top rows
    return [(x, y, y2) if with_y2 else (x, y)]


def _params(mdl):
    return [q.detach().clone() for q in mdl.parameters()]


def _frame_indexed(n_frames, channels=2):
    a = np.empty((n_frames, H_, W_, channels), np.float32)
    a[:] = np.arange(n_frames, dtype=np.float32)[:, None, None, None]
    return a


class _Store(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def make_run(self, re_nominal, aoa, n_frames, array=None):
        if array is None:
            array = _frame_indexed(n_frames)
        path = self.root / f"{re_nominal}_{aoa}.npy"
        np.save(path, array)
        return Trajectory(path=path, re_nominal=re_nominal, re_measured=re_nominal + 7,
                          aoa=aoa, n_frames=n_frames, channels=array.shape[-1])


class DatasetExtraOutZeroIsToday(_Store):
    """(a) extra_out=0 -- the default -- is the dataset as it was: same index, same items."""

    def test_default_and_explicit_zero_are_identical_and_match_the_arithmetic(self):
        runs = [self.make_run(3750, 0, 100), self.make_run(10125, 5, 61)]
        base = WindowDataset(runs, stride=10, channels=2)
        zero = WindowDataset(runs, stride=10, channels=2, extra_out=0)
        self.assertEqual(base.index, zero.index)
        self.assertEqual(base.index, [(i, s) for i, r in enumerate(runs)
                                      for s in window_starts(r.n_frames, T_IN, T_OUT, 10)])
        self.assertEqual(len(base), 7 + 3)       # (100-40)//10+1, (61-40)//10+1
        for k in range(len(base)):
            a, b = base[k], zero[k]
            self.assertEqual(len(a), 2); self.assertEqual(len(b), 2)
            for u, v in zip(a, b):
                self.assertTrue(torch.equal(u, v))
            i, s = base.index[k]
            np.testing.assert_array_equal(a[0][:, 0, 0, 0].numpy(), np.arange(s, s + T_IN))
            np.testing.assert_array_equal(a[1][:, 0, 0, 0].numpy(), np.arange(s + T_IN, s + SPAN))

    def test_negative_extra_out_is_refused(self):
        with self.assertRaises(ValueError):
            WindowDataset([self.make_run(3750, 0, 100)], stride=10, extra_out=-1)


class DatasetExtraOutYieldsTheNextFrames(_Store):
    """(b) extra_out=T_OUT: fewer windows, and y2 is exactly frames [s+40, s+60)."""

    def test_count_drops_and_y2_is_the_true_continuation(self):
        runs = [self.make_run(3750, 0, 100), self.make_run(10125, 5, 61), self.make_run(21600, 10, 59)]
        ds = WindowDataset(runs, stride=10, channels=2, extra_out=T_OUT)
        # 100 frames: starts 0..40 -> 5 (was 7); 61 frames: start 0 only -> 1 (was 3);
        # 59 frames: no 60-frame window at all (was 2).
        self.assertEqual(ds.index, [(0, 0), (0, 10), (0, 20), (0, 30), (0, 40), (1, 0)])
        self.assertEqual(len(ds), 6)
        for k in range(len(ds)):
            item = ds[k]
            self.assertEqual(len(item), 3)
            x, y, y2 = item
            i, s = ds.index[k]
            self.assertEqual(tuple(x.shape), (T_IN, H_, W_, 2))
            self.assertEqual(tuple(y.shape), (T_OUT, H_, W_, 2))
            self.assertEqual(tuple(y2.shape), (T_OUT, H_, W_, 2))
            np.testing.assert_array_equal(x[:, 0, 0, 0].numpy(), np.arange(s, s + T_IN))
            np.testing.assert_array_equal(y[:, 0, 0, 0].numpy(), np.arange(s + T_IN, s + SPAN))
            np.testing.assert_array_equal(y2[:, 0, 0, 0].numpy(), np.arange(s + SPAN, s + SPAN + T_OUT))
            self.assertTrue(bool((y2 == y2[:, :1, :1, :1]).all()), "y2 is not frame-constant: wrong frames")
        # Every window is complete: the last y2 frame exists in the array.
        for i, s in ds.index:
            self.assertLessEqual(s + SPAN + T_OUT, runs[i].n_frames)

    def test_x_and_y_are_bit_identical_to_the_plain_dataset_at_the_same_start(self):
        runs = [self.make_run(3750, 0, 100)]
        plain = WindowDataset(runs, stride=10, channels=2)
        roll = WindowDataset(runs, stride=10, channels=2, extra_out=T_OUT)
        by_start = {s: k for k, (_i, s) in enumerate(plain.index)}
        for k, (_i, s) in enumerate(roll.index):
            x, y, _y2 = roll[k]
            xp, yp = plain[by_start[s]]
            self.assertTrue(torch.equal(x, xp)); self.assertTrue(torch.equal(y, yp))

    def test_channel_padding_mask_channel_and_mask_bank_apply_to_y2_like_y(self):
        a = _frame_indexed(100) + 1.0                # no zeros, so the mask is all-ones
        a[:, 0, :, 0] = 0.0                          # except an unmeasured top row in u
        runs = [self.make_run(26700, 0, 100, a)]
        ds = WindowDataset(runs, stride=10, channels=3, mask_channel=True, extra_out=T_OUT)
        x, y, y2 = ds[0]
        self.assertEqual(y2.shape[-1], 3)
        self.assertTrue(bool((y2[..., 2] == 0).all()), "y2's padded p must stay zero, like y's")
        self.assertTrue(bool((y[..., 2] == 0).all()))
        self.assertTrue(bool((x[:, 1:, :, 2] == 1).all()) and bool((x[:, 0, :, 2] == 0).all()),
                        "the mask plane goes into the INPUT frames only")
        bank = np.ones((1, H_, W_), bool); bank[0, :, :2] = False
        dsb = WindowDataset(runs, stride=10, channels=3, mask_bank=bank, extra_out=T_OUT)
        xb, yb, y2b = dsb[0]
        for t in (xb, yb, y2b):
            self.assertTrue(bool((t[:, :, :2, :] == 0).all()), "the mask_bank hole must hit y2 too")
        self.assertTrue(bool((y2b[:, 1:, 2:, 0] != 0).all()))


class RolloutInputConstruction(unittest.TestCase):
    """How x2 is built: measured channels from the prediction (holes re-zeroed), the
    rest carried over from x's last input frame."""

    def test_three_channel_real_layout(self):
        trn = _train_module()
        (x, _y, _y2), = _batch(c=3)
        x[:, -1, 5, 7, 0] = 0.0                       # a dead pixel in the LAST frame only
        x[:, 0, 9, 9, 0] = 0.0                        # and one in the FIRST frame only
        pred = torch.randn(2, T_OUT, H_S, W_S, 3)
        x2 = trn.rollout_input(pred, x, measured=2)
        self.assertEqual(tuple(x2.shape), tuple(x.shape))
        # channel 2 is x's last-frame plane (zeros here), broadcast over the 20 frames
        self.assertTrue(torch.equal(x2[..., 2], x[:, -1:, ..., 2].expand(-1, T_OUT, -1, -1)))
        # u, v are the prediction outside the last frame's holes ...
        meas = (x[:, -1, ..., 0] != 0)
        self.assertTrue(torch.equal(x2[..., :2][:, :, meas[0]], pred[..., :2][:, :, meas[0]]))
        # ... and exactly zero inside them (top rows + the last-frame dead pixel)
        self.assertTrue(bool((x2[:, :, :3, :, :2] == 0).all()))
        self.assertTrue(bool((x2[:, :, 5, 7, :2] == 0).all()))
        # the first-frame-only hole is NOT propagated: the last frame decides
        self.assertTrue(bool((x2[:, :, 9, 9, :2] == pred[:, :, 9, 9, :2]).all()))

    def test_mask_channel_plane_is_carried_from_the_last_frame(self):
        trn = _train_module()
        (x, _y, _y2), = _batch(c=3)
        x[..., 2] = (x[..., 0] != 0).float()         # what WindowDataset(mask_channel=True) writes
        pred = torch.randn(2, T_OUT, H_S, W_S, 3)
        x2 = trn.rollout_input(pred, x, measured=2)
        self.assertTrue(torch.equal(x2[..., 2], x[:, -1:, ..., 2].expand(-1, T_OUT, -1, -1)))
        # consistent with the u-plane it describes: x2's u is zero exactly where the plane is 0
        self.assertTrue(bool(((x2[..., 0] == 0) | (x2[..., 2] == 1)).all()))

    def test_two_channel_layout_is_the_masked_prediction(self):
        trn = _train_module()
        (x, _y, _y2), = _batch(c=2)
        pred = torch.randn(2, T_OUT, H_S, W_S, 2)
        x2 = trn.rollout_input(pred, x, measured=2)
        meas = (x[:, -1:, ..., :1] != 0).float()
        self.assertTrue(torch.equal(x2, pred * meas))

    def test_detach_cuts_the_graph(self):
        trn = _train_module()
        (x, _y, _y2), = _batch(c=2)
        pred = torch.randn(2, T_OUT, H_S, W_S, 2, requires_grad=True)
        self.assertTrue(trn.rollout_input(pred, x, 2).requires_grad)
        self.assertFalse(trn.rollout_input(pred, x, 2, detach=True).requires_grad)

    def test_delta_model_feeds_back_without_any_extra_step(self):
        """predict_delta adds the residual inside _one_pass and returns physical units,
        so the second pass takes x2 like any input: it runs and matches a manual pass."""
        trn = _train_module()
        mdl = _tiny(c=3, delta=True).eval()
        (x, _y, y2), = _batch(c=3)
        with torch.no_grad():
            out = mdl(x)
            x2 = trn.rollout_input(out, x, 2)
            manual = rel_l2(mdl(x2)[..., :2], y2[..., :2]) + 0.08 * tke_rel_l2(mdl(x2)[..., :2], y2[..., :2])
            got = trn.rollout_term(mdl, out, x, y2, 2, 0.08, False)
        torch.testing.assert_close(got, manual)
        self.assertTrue(torch.isfinite(got))


class WeightZeroIsBitIdentical(unittest.TestCase):
    """(c) The default path must not move by a single bit."""

    def test_no_flags_and_explicit_defaults_give_the_same_weights(self):
        trn = _train_module()
        outs = []
        for kw in ({}, {"rollout_w": 0.0, "rollout_detach": False, "rollout_stats": None}):
            torch.manual_seed(3)
            mdl = _tiny()
            opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
            tr = trn.run_epoch(mdl, _batch(with_y2=False), torch.device("cpu"), 2, opt, None,
                               tke_w=0.08, **kw)
            outs.append((tr, _params(mdl)))
        self.assertEqual(outs[0][0], outs[1][0])
        for a, b in zip(outs[0][1], outs[1][1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_a_three_tuple_loader_against_weight_zero_fails_loudly(self):
        trn = _train_module()
        mdl = _tiny()
        opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
        with self.assertRaises(ValueError):
            trn.run_epoch(mdl, _batch(with_y2=True), torch.device("cpu"), 2, opt, None)


class WeightChangesTheStep(unittest.TestCase):
    """(d) W > 0 moves the parameters differently, the logged term is finite and positive,
    and the RETURN stays the first pass's plain rel-L2."""

    def test_step_differs_and_stats_are_finite_positive(self):
        trn = _train_module()
        outs = []
        for w in (0.0, 1.0):
            torch.manual_seed(3)
            mdl = _tiny()
            opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
            stats = {} if w else None
            trn.run_epoch(mdl, _batch(with_y2=bool(w)), torch.device("cpu"), 2, opt, None,
                          tke_w=0.08, rollout_w=w, rollout_stats=stats)
            outs.append(_params(mdl))
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(*outs)),
                         "--train-rollout-weight 1.0 left every parameter where 0.0 did")
        self.assertIn("rollout", stats)
        self.assertTrue(np.isfinite(stats["rollout"]))
        self.assertGreater(stats["rollout"], 0.0)

    def test_returned_number_is_the_first_pass_rel_l2(self):
        trn = _train_module()
        torch.manual_seed(3)
        mdl = _tiny()
        (x, y, _y2), = _batch()
        with torch.no_grad():
            expected = rel_l2(mdl(x), y).item()
        opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
        tr = trn.run_epoch(mdl, _batch(), torch.device("cpu"), 2, opt, None, tke_w=0.08, rollout_w=0.5)
        self.assertAlmostEqual(tr, expected, places=6)

    def test_three_channel_delta_model_trains_with_rollout(self):
        """The shipped layout: c_in = c_out = 3, measured 2, --delta."""
        trn = _train_module()
        mdl = _tiny(c=3, delta=True)
        opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
        stats = {}
        trn.run_epoch(mdl, _batch(c=3), torch.device("cpu"), 2, opt, None, tke_w=0.08,
                      rollout_w=0.5, rollout_stats=stats)
        self.assertTrue(np.isfinite(stats["rollout"]) and stats["rollout"] > 0)


class DetachFlag(unittest.TestCase):
    """(e) With detach the rollout term sends NO gradient into the first pass; without
    it, it does -- and the two produce different parameter gradients."""

    def _first_pass_grad(self, detach):
        trn = _train_module()
        mdl = _tiny()
        (x, _y, y2), = _batch()
        out = mdl(x)
        out.retain_grad()
        roll = trn.rollout_term(mdl, out, x, y2, 2, 0.08, detach)
        roll.backward()
        return out.grad, [torch.zeros_like(q) if q.grad is None else q.grad.clone()
                          for q in mdl.parameters()]

    def test_detached_term_has_zero_gradient_at_the_first_pass_output(self):
        g_det, p_det = self._first_pass_grad(detach=True)
        g_full, p_full = self._first_pass_grad(detach=False)
        self.assertIsNone(g_det, "detach still let the rollout term reach the first pass")
        self.assertIsNotNone(g_full)
        self.assertGreater(float(g_full.abs().sum()), 0.0)
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(p_det, p_full)),
                         "detach changed nothing about the parameter gradients")
        self.assertTrue(all(torch.isfinite(g).all() for g in p_det + p_full))

    def test_detach_changes_the_training_step(self):
        trn = _train_module()
        outs = []
        for flag in (False, True):
            torch.manual_seed(3)
            mdl = _tiny()
            opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
            trn.run_epoch(mdl, _batch(), torch.device("cpu"), 2, opt, None, tke_w=0.08,
                          rollout_w=1.0, rollout_detach=flag)
            outs.append(_params(mdl))
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(*outs)))


class ArgparseDefaultsAndGuards(unittest.TestCase):
    """(f) Off by default; the combinations that are undefined or silent are refused."""

    def _parse(self, argv):
        trn = _train_module()
        with mock.patch("sys.argv", ["train.py", *argv]):
            return trn.parse_args()

    def test_defaults_are_off(self):
        a = self._parse([])
        self.assertEqual(a.train_rollout_weight, 0.0)
        self.assertFalse(a.train_rollout_detach)
        self.assertEqual(a.rollout_frames, 0)      # the inference-time flag is untouched

    def test_a_full_line_parses(self):
        a = self._parse(["--train-rollout-weight", "0.5", "--train-rollout-detach", "--delta",
                         "--mix-phases", "0.75", "--tke-weight", "0.08", "--coverage-weight", "0.4"])
        self.assertEqual(a.train_rollout_weight, 0.5)
        self.assertTrue(a.train_rollout_detach)
        self.assertEqual(a.mix_phases, 0.75)

    def test_refusals(self):
        for argv in (["--train-rollout-weight", "0.5", "--rollout-frames", "4"],
                     ["--train-rollout-weight", "0.5", "--rans"],
                     ["--train-rollout-weight", "0.5", "--teacher-ensemble", "a.pth", "--distill-weight", "1.0"],
                     ["--train-rollout-weight", "0.5", "--co-members", "2"],
                     ["--train-rollout-weight", "-0.5"],
                     ["--train-rollout-detach"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                self._parse(argv)

    def test_gapped_holdout_and_mix_phases_are_allowed(self):
        a = self._parse(["--train-rollout-weight", "1.0", "--gapped-holdout", "--mix-phases", "0.75"])
        self.assertEqual(a.train_rollout_weight, 1.0)


class MainWiring(unittest.TestCase):
    """The behaviour needs the processed store, so the wiring is asserted as source
    strings (the test_gth / PhaseMixingLeak pattern): both training stores get
    extra_out, validation does not, and the term is logged."""

    def test_both_training_datasets_get_extra_out_and_validation_does_not(self):
        src = (REPO / "scripts" / "train.py").read_text()
        i = src.index("extra_out = 0\n")
        block = src[i:i + 3000]
        self.assertIn("extra_out = cfg.t_out", block)
        self.assertIn("train_ds = WindowDataset(train_runs", block)
        self.assertIn("extra_out=extra_out)", block)
        self.assertIn("sim_ds = (WindowDataset(mix_runs, stride=a.stride, channels=cfg.c_in, extra_out=extra_out)", block)
        self.assertIn("val_ds = WindowDataset(val_runs, stride=a.val_stride, channels=cfg.c_in, mask_channel=cfg.mask_channel)", block)
        self.assertNotIn("val_runs, stride=a.val_stride, channels=cfg.c_in, extra_out", src)

    def test_log_and_wandb_carry_the_term_and_the_flags(self):
        src = (REPO / "scripts" / "train.py").read_text()
        for key in ('"train_rollout_weight": a.train_rollout_weight',
                    '"train_rollout_detach": bool(a.train_rollout_detach)',
                    '{"train_rollout": roll["rollout"]}',
                    '{"train/rollout": roll["rollout"]}',
                    "rollout_w=a.train_rollout_weight, rollout_detach=a.train_rollout_detach"):
            self.assertIn(key, src, key)

    def test_the_inference_rollout_is_untouched(self):
        """--rollout-frames is FlowerS2S.forward's K-frames-per-pass mode and stays as it was."""
        src = (REPO / "scripts" / "train.py").read_text()
        self.assertIn('rollout_frames=a.rollout_frames,', src)
        self.assertIn('p.add_argument("--rollout-frames", type=int, default=0, metavar="K",', src)


if __name__ == "__main__":
    unittest.main()
