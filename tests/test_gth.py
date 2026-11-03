"""Direction GTH: the gapped temporal holdout (``split_temporal_gapped``, ``TrajectorySlice``).

``split_by_condition`` holds out whole Reynolds rows; GTH holds out the TAIL of every
trajectory with a gap of frames that belong to neither split. The leak this design
prevents is a subtle one: ``channel_stats`` reads ``r.load()`` directly, so a split
that only restricted window STARTS would still fit the normalizer on held-out frames.
``TrajectorySlice`` therefore restricts ``load()`` itself, and these tests exercise
every consumer -- ``window_starts``, ``WindowDataset``, ``channel_stats``,
``build_eval_arrays`` -- against a tiny synthetic store whose values ENCODE the frame
index, so "which frames did this consumer see" is a direct read of the values.

The train.py wiring is asserted as source strings, in the style of
``tests/test_dataset.py::PhaseMixingLeak``: the behaviour needs the processed store,
which a fresh clone does not have.
"""
import pathlib
import tempfile
import unittest

import numpy as np

from pde_competition.dataset import (Trajectory, TrajectorySlice, WindowDataset,
                                     build_eval_arrays, channel_stats,
                                     split_temporal_gapped, window_starts)
from pde_competition.geometry import T_IN, T_OUT

SPAN = T_IN + T_OUT          # 40 frames per window
H_, W_ = 4, 6                # deliberately tiny; no consumer checks the grid size
TRAIN_PY = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train.py"


def _frame_indexed(n_frames: int, channels: int = 2) -> np.ndarray:
    """(n_frames, H_, W_, channels) float32 with frame i filled with the value i."""
    a = np.empty((n_frames, H_, W_, channels), np.float32)
    a[:] = np.arange(n_frames, dtype=np.float32)[:, None, None, None]
    return a


class _Store(unittest.TestCase):
    """A temp dir of small .npy trajectories, one per (Re, aoa), with frame index encoded."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def make_run(self, re_nominal: int, aoa: int, n_frames: int,
                 array: np.ndarray | None = None) -> Trajectory:
        if array is None:
            array = _frame_indexed(n_frames)
        assert array.shape[0] == n_frames
        path = self.root / f"{re_nominal}_{aoa}.npy"
        np.save(path, array)
        return Trajectory(path=path, re_nominal=re_nominal, re_measured=re_nominal + 7,
                          aoa=aoa, n_frames=n_frames, channels=array.shape[-1])


class SplitArithmetic(_Store):
    """Per trajectory of T frames: train [0, T-H-G), gap [T-H-G, T-H), val [T-H, T)."""

    T, H, G = 130, 50, 10     # train [0,70), gap [70,80), val [80,130)

    def test_slice_bounds_are_exactly_the_docstring(self):
        """If this fails the split is not the one the docstring describes: either frames
        are shared between the sides, or the gap is not G frames wide, and every GTH
        number would have been measured on a different holdout than the one claimed."""
        run = self.make_run(13950, 0, self.T)
        (tr,), (va,) = split_temporal_gapped([run], holdout_frames=self.H, gap_frames=self.G)
        self.assertEqual((tr.frame_start, tr.n_frames), (0, self.T - self.H - self.G))
        self.assertEqual((va.frame_start, va.n_frames), (self.T - self.H, self.H))
        # exactly G frames between the end of train and the start of val
        self.assertEqual(va.frame_start - (tr.frame_start + tr.n_frames), self.G)
        # identity fields ride along unchanged
        for f in ("path", "re_nominal", "re_measured", "aoa", "channels", "key"):
            self.assertEqual(getattr(tr, f), getattr(run, f), f)
            self.assertEqual(getattr(va, f), getattr(run, f), f)

    def test_load_returns_exactly_the_slice_frames(self):
        """``load()`` is where the restriction lives. If it returned the whole array,
        n_frames would say one thing and every consumer would read another."""
        run = self.make_run(13950, 0, self.T)
        (tr,), (va,) = split_temporal_gapped([run], holdout_frames=self.H, gap_frames=self.G)
        got_tr = np.asarray(tr.load())[:, 0, 0, 0]
        got_va = np.asarray(va.load())[:, 0, 0, 0]
        np.testing.assert_array_equal(got_tr, np.arange(0, self.T - self.H - self.G))
        np.testing.assert_array_equal(got_va, np.arange(self.T - self.H, self.T))
        self.assertEqual(tr.load().shape[0], tr.n_frames)
        self.assertEqual(va.load().shape[0], va.n_frames)

    def test_gap_frames_are_in_neither_slice(self):
        """The G frames [T-H-G, T-H) must be unreachable from both sides. A frame in
        both, or a gap frame in either, is the leak GTH exists to rule out."""
        run = self.make_run(13950, 0, self.T)
        (tr,), (va,) = split_temporal_gapped([run], holdout_frames=self.H, gap_frames=self.G)
        seen_tr = set(np.asarray(tr.load())[:, 0, 0, 0].astype(int).tolist())
        seen_va = set(np.asarray(va.load())[:, 0, 0, 0].astype(int).tolist())
        gap = set(range(self.T - self.H - self.G, self.T - self.H))
        self.assertEqual(len(gap), self.G)
        self.assertFalse(seen_tr & seen_va, "a frame is in both splits")
        self.assertFalse(seen_tr & gap, "a gap frame is in TRAIN")
        self.assertFalse(seen_va & gap, "a gap frame is in VAL")
        self.assertEqual(seen_tr | seen_va | gap, set(range(self.T)), "a frame went missing")

    def test_zero_gap_makes_the_slices_adjacent(self):
        """gap_frames=0 is legal and means adjacent, not overlapping: the last train
        frame is T-H-1 and the first val frame T-H."""
        run = self.make_run(13950, 0, self.T)
        (tr,), (va,) = split_temporal_gapped([run], holdout_frames=self.H, gap_frames=0)
        self.assertEqual(tr.n_frames, self.T - self.H)
        self.assertEqual(va.frame_start, tr.frame_start + tr.n_frames)


class GapSemantics(_Store):
    """The docstring's exact claim: for any training window ending at frame i and any
    validation window starting at frame j, j - i >= G + 1, with equality attained."""

    T, H, G = 130, 50, 10

    def _abs_windows(self, sl: TrajectorySlice, stride: int):
        """(first, last) ABSOLUTE frame indices of every window in the slice."""
        return [(sl.frame_start + s, sl.frame_start + s + SPAN - 1)
                for s in window_starts(sl.n_frames, stride=stride)]

    def test_min_separation_is_g_plus_one_at_stride_one(self):
        """Stride 1 enumerates EVERY window on each side, so this is the tightest pair
        that exists. If the minimum were <= G the gap would be narrower than claimed
        and the first val window nearly the same flow state as the last train window."""
        run = self.make_run(3750, 5, self.T)
        (tr,), (va,) = split_temporal_gapped([run], holdout_frames=self.H, gap_frames=self.G)
        tr_w = self._abs_windows(tr, stride=1)
        va_w = self._abs_windows(va, stride=1)
        self.assertTrue(tr_w and va_w, "both sides must have windows for this to mean anything")
        seps = [j - i for (_, i) in tr_w for (j, _) in va_w]
        self.assertEqual(min(seps), self.G + 1)             # equality is attained
        self.assertTrue(all(s >= self.G + 1 for s in seps))  # and never undercut
        # the docstring's other form: G frames strictly between the tightest pair
        i_last = max(i for _, i in tr_w)
        j_first = min(j for j, _ in va_w)
        self.assertEqual(j_first - i_last - 1, self.G)

    def test_no_window_crosses_its_slice_boundary(self):
        """Each window lies wholly inside its own range, at any stride -- the premise
        of the separation argument. A window starting inside the slice but running
        past its end would reach into the gap or the other split."""
        run = self.make_run(3750, 5, self.T)
        (tr,), (va,) = split_temporal_gapped([run], holdout_frames=self.H, gap_frames=self.G)
        for stride in (1, 7, 50):
            for first, last in self._abs_windows(tr, stride):
                self.assertGreaterEqual(first, 0)
                self.assertLess(last, self.T - self.H - self.G, f"train window at stride {stride}")
            for first, last in self._abs_windows(va, stride):
                self.assertGreaterEqual(first, self.T - self.H, f"val window at stride {stride}")
                self.assertLess(last, self.T)


class NormalizerFitSeesOnlyTrainingFrames(_Store):
    """``channel_stats`` reads ``r.load()`` directly. A split that restricted only window
    starts would fit the normalizer on held-out frames -- a leak invisible to any test
    on WindowDataset. This is the reason the restriction lives in ``load()``."""

    T, H, G = 130, 50, 10

    def test_channel_stats_on_train_slices_ignores_gap_and_holdout(self):
        """Train frames are 1.0, gap frames 500.0, held-out frames 1000.0. The training
        normalizer must come out at mean exactly 1.0 with zero variance (std -> 1.0 by
        the identically-constant rule). Any admixture of gap or tail frames would drag
        the mean far above 1 and give a large std -- unmistakable."""
        a = np.empty((self.T, H_, W_, 2), np.float32)
        a[:self.T - self.H - self.G] = 1.0
        a[self.T - self.H - self.G:self.T - self.H] = 500.0
        a[self.T - self.H:] = 1000.0
        run = self.make_run(10125, 10, self.T, array=a)
        train, val = split_temporal_gapped([run], holdout_frames=self.H, gap_frames=self.G)

        mean, std = channel_stats(train, channels=2)
        np.testing.assert_allclose(mean, [1.0, 1.0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(std, [1.0, 1.0], rtol=0, atol=1e-6)   # zero var -> 1.0

        # Sanity on the instrument itself: the whole run, and the val side, are NOT 1.0.
        mean_all, _ = channel_stats([run], channels=2)
        self.assertGreater(float(mean_all[0]), 100.0)
        mean_val, _ = channel_stats(val, channels=2)
        np.testing.assert_allclose(mean_val, [1000.0, 1000.0], rtol=0, atol=1e-3)

    def test_channel_stats_subsampling_stays_inside_the_slice(self):
        """channel_stats strides frames with ``a[::step]`` for ``max_frames``. With a small
        max_frames on the slice the stride is computed from the SLICE length and every
        frame read is still a training frame."""
        a = np.empty((self.T, H_, W_, 2), np.float32)
        a[:self.T - self.H - self.G] = 1.0
        a[self.T - self.H - self.G:] = 1000.0
        run = self.make_run(10125, 10, self.T, array=a)
        train, _ = split_temporal_gapped([run], holdout_frames=self.H, gap_frames=self.G)
        mean, _ = channel_stats(train, channels=2, max_frames=7)
        np.testing.assert_allclose(mean, [1.0, 1.0], rtol=0, atol=1e-6)


class EvalArraysLieInsideTheHoldout(_Store):
    """``build_eval_arrays`` on the val slices is what score_local.py would feed a
    submission; every window must lie inside [T-H, T)."""

    T, H, G = 130, 50, 10

    def test_every_eval_window_is_a_tail_window(self):
        """A window with any frame < T-H would score the model on frames it trained on
        (or on gap frames), and the GTH holdout would be no holdout."""
        runs = [self.make_run(13950, 0, self.T), self.make_run(26700, 5, self.T)]
        _train, val = split_temporal_gapped(runs, holdout_frames=self.H, gap_frames=self.G)
        x, y, keys = build_eval_arrays(val, stride=5, return_keys=True)
        self.assertEqual(x.shape[1:], (T_IN, H_, W_, 3))     # grader's 3 channels, p padded
        self.assertEqual(y.shape[1:], (T_OUT, H_, W_, 3))
        self.assertEqual(len(keys), x.shape[0])
        frames = np.concatenate([x[..., 0, 0, 0], y[..., 0, 0, 0]], axis=1)   # (N, 40) frame ids
        self.assertTrue((frames >= self.T - self.H).all(), frames.min())
        self.assertTrue((frames < self.T).all(), frames.max())
        # windows are contiguous 40-frame runs, x followed by y
        np.testing.assert_array_equal(np.diff(frames, axis=1), 1.0)
        # the padded p plane is zero, as the grader hands it over
        self.assertTrue((x[..., 2] == 0).all() and (y[..., 2] == 0).all())
        # both conditions present
        self.assertEqual(set(keys), {(13950, 0), (26700, 5)})

    def test_eval_window_count_follows_the_holdout_length(self):
        """H=50 at stride 5 gives (50-40)/5+1 = 3 windows per trajectory; at stride 50,
        exactly one. If the count came from the whole trajectory the holdout would
        silently be the whole run."""
        run = self.make_run(13950, 0, self.T)
        _train, val = split_temporal_gapped([run], holdout_frames=self.H, gap_frames=self.G)
        x5, _ = build_eval_arrays(val, stride=5)
        x50, _ = build_eval_arrays(val, stride=50)
        self.assertEqual(x5.shape[0], len(window_starts(self.H, stride=5)))
        self.assertEqual(x5.shape[0], 3)
        self.assertEqual(x50.shape[0], 1)


class Raises(_Store):
    """The split raises rather than silently producing an empty or shrunken holdout."""

    def test_holdout_shorter_than_one_window_raises(self):
        """holdout_frames < T_IN+T_OUT would leave a validation slice with zero windows,
        and the run would report a validation score over nothing."""
        run = self.make_run(13950, 0, 130)
        with self.assertRaises(ValueError) as cm:
            split_temporal_gapped([run], holdout_frames=SPAN - 1, gap_frames=0)
        self.assertIn("holdout", str(cm.exception).lower())
        # exactly one window's worth is the smallest legal holdout
        _t, (va,) = split_temporal_gapped([run], holdout_frames=SPAN, gap_frames=0)
        self.assertEqual(len(window_starts(va.n_frames, stride=1)), 1)

    def test_negative_gap_raises(self):
        """A negative gap would make train and val OVERLAP by |G| frames -- worse than no gap."""
        run = self.make_run(13950, 0, 130)
        with self.assertRaises(ValueError) as cm:
            split_temporal_gapped([run], holdout_frames=SPAN, gap_frames=-1)
        self.assertIn("negative", str(cm.exception))

    def test_too_short_trajectory_raises_and_names_the_run(self):
        """A trajectory that cannot hold one training window after the holdout and gap
        must raise and NAME the run. Dropping it silently would remove a condition
        from training with nothing in any log to say so. (The store has 607-, 492- and
        282-frame trajectories beside the 868-frame majority, so this is reachable.)"""
        ok = self.make_run(13950, 0, 130)                     # 130-50-10 = 70 >= 40
        short = self.make_run(3750, 15, 95)                   # 95-50-10 = 35 < 40
        with self.assertRaises(ValueError) as cm:
            split_temporal_gapped([ok, short], holdout_frames=50, gap_frames=10)
        msg = str(cm.exception)
        self.assertIn(short.path.stem, msg)                   # "3750_15"
        self.assertIn("95", msg)                              # its length
        self.assertNotIn(ok.path.stem, msg)                   # not the innocent one
        # the boundary case -- exactly one training window -- is accepted
        exact = self.make_run(3750, 20, 50 + 10 + SPAN)
        (tr,), _v = split_temporal_gapped([exact], holdout_frames=50, gap_frames=10)
        self.assertEqual(tr.n_frames, SPAN)
        self.assertEqual(len(window_starts(tr.n_frames, stride=1)), 1)

    def test_too_short_does_not_silently_drop(self):
        """Complement of the above: no combination of inputs returns fewer slices than
        runs. Either every run is split or the call raises."""
        runs = [self.make_run(13950, 0, 130), self.make_run(3750, 15, 95)]
        try:
            train, val = split_temporal_gapped(runs, holdout_frames=50, gap_frames=10)
        except ValueError:
            return
        self.fail(f"returned {len(train)}/{len(val)} slices for {len(runs)} runs instead of raising")

    def test_holdout_and_gap_are_keyword_only(self):
        """The 2026-09-23 leak was a defaulted split argument at one of several call
        sites. Positional or defaulted holdout arguments must be impossible here."""
        run = self.make_run(13950, 0, 130)
        with self.assertRaises(TypeError):
            split_temporal_gapped([run], 50, 10)                  # positional
        with self.assertRaises(TypeError):
            split_temporal_gapped([run], holdout_frames=50)       # gap defaulted
        with self.assertRaises(TypeError):
            split_temporal_gapped([run], gap_frames=10)           # holdout defaulted
        with self.assertRaises(TypeError):
            split_temporal_gapped([run])                          # both defaulted


class EveryConditionOnBothSides(_Store):
    """The point of GTH: the validation set spans ALL conditions, not 2 of 18 rows."""

    def test_every_key_is_in_train_and_in_val(self):
        """If a condition were missing from one side the instrument would have a
        condition blind spot again -- the thing HO2 said the twin could not see."""
        keys = [(re, aoa) for re in (3750, 10125, 13950, 26700) for aoa in (0, 5, 10)]
        runs = [self.make_run(re, aoa, 130) for re, aoa in keys]
        train, val = split_temporal_gapped(runs, holdout_frames=50, gap_frames=10)
        self.assertEqual(sorted(r.key for r in train), sorted(keys))
        self.assertEqual(sorted(r.key for r in val), sorted(keys))
        self.assertEqual(len(train), len(runs))
        self.assertEqual(len(val), len(runs))

    def test_order_is_preserved_and_slices_pair_up(self):
        """train[i] and val[i] are the same trajectory: train.py's first-trajectory print
        line reads train_runs[0] and val_runs[0] as a pair, and per-row analysis zips them."""
        keys = [(26700, 20), (3750, 0), (13950, 10)]
        runs = [self.make_run(re, aoa, 130) for re, aoa in keys]
        train, val = split_temporal_gapped(runs, holdout_frames=50, gap_frames=10)
        self.assertEqual([r.key for r in train], keys)
        self.assertEqual([r.key for r in val], keys)
        for t, v, r in zip(train, val, runs):
            self.assertEqual(t.path, r.path)
            self.assertEqual(v.path, r.path)

    def test_varying_trajectory_lengths_hold_out_the_same_number_of_tail_frames(self):
        """The holdout is a count from the END, so a 607-frame run and an 868-frame run
        both contribute H validation frames; only the training length differs."""
        long = self.make_run(13950, 0, 200)
        short = self.make_run(13950, 5, 130)
        train, val = split_temporal_gapped([long, short], holdout_frames=50, gap_frames=10)
        self.assertEqual([v.n_frames for v in val], [50, 50])
        self.assertEqual([v.frame_start for v in val], [150, 80])
        self.assertEqual([t.n_frames for t in train], [140, 70])


class SliceIsATrajectory(_Store):
    """``TrajectorySlice`` must be usable wherever a ``Trajectory`` is, unchanged."""

    T, H, G = 130, 50, 10

    def test_isinstance_and_key(self):
        """Consumers type on Trajectory (WindowDataset, channel_stats, measured_channels);
        a slice that were not one would need changes in every consumer -- the design
        says none are needed."""
        run = self.make_run(13950, 0, self.T)
        (tr,), (va,) = split_temporal_gapped([run], holdout_frames=self.H, gap_frames=self.G)
        self.assertIsInstance(tr, Trajectory)
        self.assertIsInstance(va, Trajectory)
        self.assertIsInstance(tr, TrajectorySlice)
        self.assertEqual(tr.key, run.key)
        self.assertEqual(va.key, run.key)
        # a plain Trajectory behaves as a slice with frame_start 0 over the whole run
        self.assertEqual(TrajectorySlice(**vars(run)).frame_start, 0)
        np.testing.assert_array_equal(TrajectorySlice(**vars(run)).load(), run.load())

    def test_window_dataset_windows_stay_inside_the_slice(self):
        """WindowDataset indexes by ``r.n_frames`` and reads ``r.load()``. On the train
        slice the first window is frames 0..39 and the last window ends at T-H-G-1; on
        the val slice the first window starts at T-H and the last ends at T-1."""
        run = self.make_run(13950, 0, self.T)
        (tr,), (va,) = split_temporal_gapped([run], holdout_frames=self.H, gap_frames=self.G)
        stride = 10
        ds_tr = WindowDataset([tr], stride=stride, channels=2)
        ds_va = WindowDataset([va], stride=stride, channels=2)
        self.assertEqual(len(ds_tr), len(window_starts(tr.n_frames, stride=stride)))
        self.assertEqual(len(ds_va), len(window_starts(va.n_frames, stride=stride)))

        def frames_of(ds, k):
            x, y = ds[k]
            return np.concatenate([x[:, 0, 0, 0].numpy(), y[:, 0, 0, 0].numpy()])

        first_tr = frames_of(ds_tr, 0)
        last_tr = frames_of(ds_tr, len(ds_tr) - 1)
        np.testing.assert_array_equal(first_tr, np.arange(0, SPAN))
        self.assertLess(last_tr[-1], self.T - self.H - self.G)
        last_start = list(window_starts(tr.n_frames, stride=stride))[-1]
        np.testing.assert_array_equal(last_tr, np.arange(last_start, last_start + SPAN))

        first_va = frames_of(ds_va, 0)
        last_va = frames_of(ds_va, len(ds_va) - 1)
        np.testing.assert_array_equal(first_va, np.arange(self.T - self.H, self.T - self.H + SPAN))
        self.assertLess(last_va[-1], self.T)
        self.assertGreaterEqual(first_va[0], self.T - self.H)
        # every window on both sides, not only the ends
        for k in range(len(ds_tr)):
            f = frames_of(ds_tr, k)
            self.assertTrue((f < self.T - self.H - self.G).all(), f)
        for k in range(len(ds_va)):
            f = frames_of(ds_va, k)
            self.assertTrue((f >= self.T - self.H).all() and (f < self.T).all(), f)

    def test_slice_dataset_is_the_matching_subset_of_the_full_dataset(self):
        """Window k of the val slice equals window (frame_start + k*stride) of the full
        run when the strides line up: the slice adds no interpolation or reindexing."""
        run = self.make_run(13950, 0, self.T)
        _t, (va,) = split_temporal_gapped([run], holdout_frames=self.H, gap_frames=self.G)
        full = WindowDataset([run], stride=1, channels=2)
        part = WindowDataset([va], stride=5, channels=2)
        for k in range(len(part)):
            xp, yp = part[k]
            xf, yf = full[va.frame_start + 5 * k]
            np.testing.assert_array_equal(xp.numpy(), xf.numpy())
            np.testing.assert_array_equal(yp.numpy(), yf.numpy())


class TrainPyWiring(unittest.TestCase):
    """train.py's GTH wiring, as source strings (the behaviour needs the processed store)."""

    @classmethod
    def setUpClass(cls):
        cls.src = TRAIN_PY.read_text()
        cls.i_gth = cls.src.index("if a.gapped_holdout:")
        cls.i_split_def = cls.src.index("train_runs, val_runs = _split(runs)")
        cls.i_phases = cls.src.index("if a.mix_phases > 0:")
        cls.i_sim = cls.src.index("if a.mix_sim > 0:")

    def test_flag_exists_and_is_store_true(self):
        """A flag with a default other than False would turn GTH on for ordinary runs."""
        self.assertIn('p.add_argument("--gapped-holdout", action="store_true"', self.src)
        self.assertIn('p.add_argument("--gth-holdout-frames", type=int, default=100', self.src)
        self.assertIn('p.add_argument("--gth-gap", type=int, default=40', self.src)

    def test_phase_stores_go_through_the_runs_split_function(self):
        """The 2026-09-23 leak was the phase stores taking split_by_condition's DEFAULT
        holdout. Under GTH the same second call site would train on half-pixel-shifted
        copies of the held-out TAIL. The stores must go through ``_split`` and never
        call split_by_condition themselves."""
        block = self.src[self.i_phases:self.i_sim]
        self.assertIn("ph_runs, _ph_held = _split(_ph_all)", block)
        self.assertIn("_ph_all = list_runs(ph)", block)
        self.assertNotIn("split_by_condition(", block,
                         "the phase block must not split on its own")
        self.assertNotIn("split_temporal_gapped(", block,
                         "nor call the gapped split directly -- only via _split")
        for ph in ("train_real_p01", "train_real_p10", "train_real_p11"):
            self.assertIn(ph, block)

    def test_split_by_condition_has_exactly_one_call_site(self):
        """The structural form of the leak fix: ONE split function per run. A second
        ``split_by_condition(`` call anywhere in train.py is a second way to split."""
        calls = self.src.count("split_by_condition(")
        self.assertEqual(calls, 1, f"{calls} call sites of split_by_condition( in train.py")
        self.assertEqual(self.src.count("split_temporal_gapped("), 1,
                         "split_temporal_gapped( must be called only inside _split")

    def test_incompatible_flags_are_refused_under_gapped_holdout(self):
        """--train-all would train on the held-out tail; --mix-sim has no frame-level
        correspondence to the PIV recording; a non-default --holdout-re would be
        silently ignored. Each must raise SystemExit inside the gapped branch, before
        _split is defined."""
        branch = self.src[self.i_gth:self.i_split_def]
        for guard, msg in (("if a.train_all:",
                            "--train-all with --gapped-holdout would train on the held-out"),
                           ("if a.mix_sim:",
                            "--mix-sim with --gapped-holdout is refused"),
                           ("if tuple(a.holdout_re) != (13950, 26700):",
                            "--holdout-re would be silently")):
            self.assertIn(guard, branch, guard)
            j = branch.index(guard)
            after = branch[j:j + 700]
            self.assertIn("raise SystemExit(", after, f"{guard} does not raise")
            self.assertIn(msg, after, f"{guard} raises the wrong message")
            self.assertLess(after.index("raise SystemExit("), after.index(msg))
        # all three guards precede the gapped _split definition
        self.assertLess(branch.rindex("raise SystemExit("), branch.index("def _split(rs):"))

    def test_log_record_carries_gapped_holdout(self):
        """Without the key a GTH val_rel_l2 line is indistinguishable from a condition-
        holdout line in log.jsonl, and summarize_runs.py would merge the two instruments."""
        self.assertIn('"gapped_holdout": ([a.gth_holdout_frames, a.gth_gap]', self.src)
        j = self.src.index('"gapped_holdout": (')
        self.assertIn("if a.gapped_holdout else None", self.src[j:j + 200])
        # sits in the same record as the condition holdout
        self.assertIn('"holdout_re": list(a.holdout_re)', self.src[j - 600:j])

    def test_phase_store_length_guard(self):
        """A temporal holdout is a FRAME INDEX. A phase store of a different length would
        hold out a different interval and the tail would leak in shifted. The guard
        must compare n_frames per key against p00 and raise SystemExit naming
        'trajectory length' under --gapped-holdout --mix-phases."""
        block = self.src[self.i_phases:self.i_sim]
        j = block.index("if a.gapped_holdout:")
        guard = block[j:j + 1200]
        self.assertIn("_ref = {r.key: r.n_frames for r in runs}", guard)
        self.assertIn("_ref.get(r.key) != r.n_frames", guard)
        self.assertIn("raise SystemExit(", guard)
        self.assertIn("--gapped-holdout --mix-phases", guard)
        self.assertIn("trajectory length", guard)
        # the guard runs BEFORE the store is split
        self.assertLess(j, block.index("ph_runs, _ph_held = _split(_ph_all)"))

    def test_gapped_split_threads_both_flags(self):
        """_split under GTH must pass BOTH --gth-holdout-frames and --gth-gap through;
        the function has no defaults, so omitting one is a TypeError, but a typo that
        passed the wrong attribute would not be."""
        branch = self.src[self.i_gth:self.i_split_def]
        self.assertIn("split_temporal_gapped(rs, holdout_frames=a.gth_holdout_frames,", branch)
        self.assertIn("gap_frames=a.gth_gap)", branch)

    def test_the_first_trajectory_print_reads_slice_fields(self):
        """The launch log states the frame ranges from the slices themselves, so a
        misconfigured run is visible in its first lines."""
        after = self.src[self.i_split_def:self.i_split_def + 800]
        self.assertIn("_t0, _v0 = train_runs[0], val_runs[0]", after)
        self.assertIn("_v0.frame_start", after)
        self.assertIn("_t0.n_frames", after)


class DefaultOff(unittest.TestCase):
    """With no --gapped-holdout, not one line of GTH runs and the split is unchanged."""

    def test_else_branch_is_split_by_condition_with_this_runs_holdout(self):
        """Ordinary runs must keep holding out whole Re rows by --holdout-re. If the
        else branch changed, every non-GTH number after this commit would be on a
        different validation set from every number before it."""
        src = TRAIN_PY.read_text()
        i = src.index("if a.gapped_holdout:")
        j = src.index("train_runs, val_runs = _split(runs)")
        branch = src[i:j]
        self.assertIn("else:", branch)
        else_part = branch[branch.rindex("else:"):]
        self.assertIn("def _split(rs):", else_part)
        self.assertIn("return split_by_condition(rs, holdout_re=tuple(a.holdout_re))", else_part)
        self.assertNotIn("split_temporal_gapped", else_part)
        self.assertNotIn("gth", else_part)

    def test_log_field_is_none_when_off(self):
        """The record must say None, not [100, 40], for an ordinary run -- the defaults
        of the two --gth flags exist even when the mode is off."""
        src = TRAIN_PY.read_text()
        j = src.index('"gapped_holdout": (')
        self.assertIn("if a.gapped_holdout else None", src[j:j + 200])

    def test_a_plain_trajectory_loads_the_whole_run(self):
        """The base class is untouched: Trajectory.load() still returns every frame."""
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "13950_0.npy"
            np.save(path, _frame_indexed(60))
            run = Trajectory(path=path, re_nominal=13950, re_measured=13957, aoa=0,
                             n_frames=60, channels=2)
            self.assertEqual(run.load().shape[0], 60)
            self.assertFalse(hasattr(run, "frame_start"))


if __name__ == "__main__":
    unittest.main()
