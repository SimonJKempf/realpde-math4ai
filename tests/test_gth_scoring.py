"""Direction GTH, the SCORING side: a GTH-trained model is scored on its own held-out TAIL
windows, per Reynolds row (all 18), by score_local.py / fit_bounds.py / score_twin.py
``--gapped-holdout [N G]``.

No model inference and no processed store: the pure helpers run on the synthetic
frame-indexed store of ``tests/test_gth.py`` (values encode the frame index, so "which
frames were scored" is a direct read), and the flag wiring of the three scripts is
asserted as source strings, as ``test_gth.TrainPyWiring`` does for train.py.
"""
import importlib.util
import pathlib
import sys
import unittest

import numpy as np

from pde_competition.dataset import GTH_DEFAULT, parse_gapped_holdout, split_temporal_gapped
from pde_competition.geometry import T_IN, T_OUT
from pde_competition.paths import REPO

try:                                     # `python -m unittest discover tests` imports it as test_gth
    from test_gth import _Store, H_, W_
except ImportError:                      # `python -m unittest tests.test_gth_scoring`
    from tests.test_gth import _Store, H_, W_

SCRIPTS = REPO / "scripts"
SCORE_LOCAL = SCRIPTS / "score_local.py"
FIT_BOUNDS = SCRIPTS / "fit_bounds.py"
SCORE_TWIN = SCRIPTS / "score_twin.py"


def _load_score_local():
    """score_local.py as a module (it imports the data layer and paths only, no torch model)."""
    spec = importlib.util.spec_from_file_location("score_local_gth", SCORE_LOCAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ParseGappedHoldout(unittest.TestCase):
    """The one shared reading of ``--gapped-holdout [N G]``."""

    def test_absent_is_off(self):
        self.assertIsNone(parse_gapped_holdout(None))

    def test_bare_is_the_pilot_default(self):
        self.assertEqual(parse_gapped_holdout([]), GTH_DEFAULT)
        self.assertEqual(GTH_DEFAULT, (100, 40))

    def test_two_values_are_taken_as_given(self):
        self.assertEqual(parse_gapped_holdout([80, 20]), (80, 20))
        self.assertEqual(parse_gapped_holdout(["80", "20"]), (80, 20))

    def test_one_or_three_values_are_refused(self):
        for bad in ([100], [1, 2, 3]):
            with self.assertRaises(SystemExit) as cm:
                parse_gapped_holdout(bad)
            self.assertIn("exactly two", str(cm.exception))


class ScoreLocalHelpers(_Store):
    """The pure helpers score_local.py scores a GTH twin with, on the synthetic store."""

    T, H, G = 130, 50, 10     # train [0,70), gap [70,80), val [80,130)

    @classmethod
    def setUpClass(cls):
        cls.sl = _load_score_local()

    def test_resolve_holdout_default_is_the_standard_pair(self):
        self.assertEqual(self.sl.resolve_holdout(None, None), ((13950, 26700), None))

    def test_resolve_holdout_condition_rows_pass_through(self):
        self.assertEqual(self.sl.resolve_holdout([3750, 26700], None), ((3750, 26700), None))

    def test_resolve_holdout_gth_bare_and_explicit(self):
        self.assertEqual(self.sl.resolve_holdout(None, []), (None, (100, 40)))
        self.assertEqual(self.sl.resolve_holdout(None, [80, 20]), (None, (80, 20)))

    def test_holdout_re_with_gapped_holdout_is_refused(self):
        """Refused, not reinterpreted as a row filter: the GTH table shows every row already."""
        with self.assertRaises(SystemExit) as cm:
            self.sl.resolve_holdout([13950], [])
        msg = str(cm.exception)
        self.assertIn("--holdout-re", msg)
        self.assertIn("--gapped-holdout", msg)
        self.assertIn("refused", msg)

    def test_row_groups_orders_rows_and_pools_all_last(self):
        keys = [(26700, 0), (13950, 5), (13950, 0), (3750, 10)]
        groups = self.sl.row_groups(keys)
        self.assertEqual([g[0] for g in groups], ["3750", "13950", "26700", "all"])
        got = {g[0]: g[1].tolist() for g in groups}
        self.assertEqual(got["3750"], [3])
        self.assertEqual(got["13950"], [1, 2])
        self.assertEqual(got["26700"], [0])
        self.assertEqual(got["all"], [0, 1, 2, 3])
        for _, idx in groups:
            self.assertEqual(idx.dtype.kind, "i")

    def test_gth_eval_windows_are_tail_windows_grouped_by_row(self):
        """Every scored window lies in [T-H, T) of its own trajectory, every row appears in
        row_groups, and the 'all' group is the union -- on three runs over two Re rows."""
        runs = [self.make_run(13950, 0, self.T), self.make_run(26700, 5, self.T),
                self.make_run(13950, 10, self.T)]
        x, y, keys = self.sl.gth_eval_windows(runs, (self.H, self.G), stride=5)
        self.assertEqual(x.shape[1:], (T_IN, H_, W_, 3))
        self.assertEqual(y.shape[1:], (T_OUT, H_, W_, 3))
        self.assertEqual(len(keys), x.shape[0])
        frames = np.concatenate([x[..., 0, 0, 0], y[..., 0, 0, 0]], axis=1)   # (N, 40) frame ids
        self.assertTrue((frames >= self.T - self.H).all(), frames.min())
        self.assertTrue((frames < self.T).all(), frames.max())
        np.testing.assert_array_equal(np.diff(frames, axis=1), 1.0)
        # 3 windows per run at stride 5 over a 50-frame tail
        per_run = len(range(0, self.H - (T_IN + T_OUT) + 1, 5))
        self.assertEqual(per_run, 3)
        groups = dict(self.sl.row_groups(keys))
        self.assertEqual(sorted(groups), ["13950", "26700", "all"])
        self.assertEqual(len(groups["13950"]), 2 * per_run)
        self.assertEqual(len(groups["26700"]), per_run)
        self.assertEqual(len(groups["all"]), 3 * per_run)
        self.assertEqual(sorted(groups["13950"].tolist() + groups["26700"].tolist()),
                         groups["all"].tolist())
        # and the keys really are the (Re, aoa) of the slice each window came from
        self.assertEqual(set(keys), {(13950, 0), (26700, 5), (13950, 10)})

    def test_gth_eval_windows_match_split_temporal_gapped_val(self):
        """The helper is nothing but build_eval_arrays on split_temporal_gapped's VAL side --
        the same function train.py validated on, so scoring and training agree on the tail."""
        runs = [self.make_run(13950, 0, self.T)]
        x, y, keys = self.sl.gth_eval_windows(runs, (self.H, self.G), stride=50)
        _tr, val = split_temporal_gapped(runs, holdout_frames=self.H, gap_frames=self.G)
        self.assertEqual(x.shape[0], 1)
        self.assertEqual(float(x[0, 0, 0, 0, 0]), float(val[0].frame_start))

    def test_gth_eval_windows_limit_is_per_trajectory(self):
        runs = [self.make_run(13950, 0, self.T), self.make_run(26700, 5, self.T)]
        x, _y, keys = self.sl.gth_eval_windows(runs, (self.H, self.G), stride=5, limit=1)
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(set(keys), {(13950, 0), (26700, 5)})

    def test_banner_says_not_comparable(self):
        b = self.sl.gth_banner((100, 40))
        self.assertIn("NOT comparable", b)
        self.assertIn("LAST 100 frames", b)
        self.assertIn("gap 40", b)


class ScoreLocalParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sl = _load_score_local()

    def test_flag_exists_bare_and_with_values(self):
        ap = self.sl.build_parser()
        self.assertEqual(ap.parse_args(["x.zip"]).gapped_holdout, None)
        self.assertEqual(ap.parse_args(["x.zip", "--gapped-holdout"]).gapped_holdout, [])
        self.assertEqual(ap.parse_args(["x.zip", "--gapped-holdout", "80", "20"]).gapped_holdout, [80, 20])

    def test_holdout_re_default_is_none_so_the_exclusion_can_see_it(self):
        """If --holdout-re defaulted to the pair, a bare --gapped-holdout could not tell
        'not given' from 'given the default' and the refusal would fire on every GTH call."""
        ap = self.sl.build_parser()
        self.assertIsNone(ap.parse_args(["x.zip"]).holdout_re)
        self.assertEqual(ap.parse_args(["x.zip", "--holdout-re", "3750"]).holdout_re, [3750])


class SourceWiring(unittest.TestCase):
    """What the behaviour-level tests cannot reach without the store or a checkpoint."""

    @classmethod
    def setUpClass(cls):
        cls.sl = SCORE_LOCAL.read_text()
        cls.fb = FIT_BOUNDS.read_text()
        cls.st = SCORE_TWIN.read_text()

    def test_all_three_scripts_take_the_flag_as_zero_or_two_ints(self):
        for name, src in (("score_local", self.sl), ("fit_bounds", self.fb), ("score_twin", self.st)):
            self.assertIn('"--gapped-holdout", type=int, nargs="*", default=None', src, name)
            self.assertIn("parse_gapped_holdout", src, name)

    def test_score_local_uses_the_gapped_val_side_and_marks_both_files(self):
        self.assertIn("split_temporal_gapped(runs, holdout_frames=n_hold, gap_frames=gap)", self.sl)
        self.assertIn('by_row["gapped_holdout"] = [int(gapped_holdout[0]), int(gapped_holdout[1])]', self.sl)
        self.assertIn('d["gapped_holdout"] = [int(gapped_holdout[0]), int(gapped_holdout[1])]', self.sl)
        self.assertIn("per_row_scores(arrays, y, keys, out_dir, gapped_holdout=gapped_holdout)", self.sl)
        # the standard path is untouched
        self.assertIn("_train, val = split_by_condition(runs, holdout_re=holdout_re)", self.sl)

    def test_fit_bounds_refuses_row_flags_and_fits_on_the_tail_of_all_rows(self):
        i = self.fb.index("def resolve_rows(a):")
        block = self.fb[i:self.fb.index("def per_row(", i)]
        self.assertIn("if a.fit_re is not None or a.val_re is not None:", block)
        self.assertIn("raise SystemExit(", block)
        self.assertIn("--fit-re/--val-re with --gapped-holdout is refused", block)
        self.assertIn("--affine/--spatial with --gapped-holdout is not implemented", block)
        # defaults moved off the parser so the exclusion can see 'not given'
        self.assertIn('"--fit-re", type=int, nargs="+", default=None', self.fb)
        self.assertIn('"--val-re", type=int, nargs="+", default=None', self.fb)
        self.assertIn("a.fit_re = list(a.fit_re) if a.fit_re else [13950]", block)
        self.assertIn("a.val_re = list(a.val_re) if a.val_re else [26700]", block)
        self.assertIn("main(resolve_rows(p.parse_args()))", self.fb)
        # the GTH arrays: split_temporal_gapped's VAL side over every real trajectory
        j = self.fb.index("def arrays_for_gth(")
        gth = self.fb[j:self.fb.index("def resolve_rows(", j)]
        self.assertIn('split_temporal_gapped(list_runs("train_real"),', gth)
        self.assertIn("holdout_frames=n_hold, gap_frames=gap)", gth)
        self.assertIn("build_eval_arrays(val, stride=stride)", gth)
        # fit == val, rows recorded as all rows seen, marker in the payload
        self.assertIn("val_pred, val_y, val_cells = fit_pred, fit_y, fit_cells", self.fb)
        self.assertIn("a.fit_re = a.val_re = sorted({c[0] for c in fit_cells})", self.fb)
        self.assertIn('"gapped_holdout": (list(a.gth) if getattr(a, "gth", None) is not None else None)', self.fb)

    def test_score_twin_threads_explicit_n_g_to_both_children_and_refuses_row_flags(self):
        self.assertIn("--fit-re/--holdout-re with --gapped-holdout is refused", self.st)
        self.assertIn("raise SystemExit(", self.st[self.st.index("gth = parse_gapped_holdout"):])
        self.assertIn('gth_flag = ["--gapped-holdout", str(gth[0]), str(gth[1])]', self.st)
        self.assertIn('fit += gth_flag if gth is not None else ["--fit-re", *a.fit_re]', self.st)
        self.assertIn('score += gth_flag if gth is not None else ["--holdout-re", *holdout]', self.st)
        self.assertIn('"--fit-re", nargs="+", default=None', self.st)
        self.assertIn("a.fit_re = a.fit_re or DEFAULT_FIT_RE", self.st)
        self.assertIn('DEFAULT_FIT_RE = ["13950", "26700"]', self.st)

    def test_every_script_prints_the_not_comparable_header(self):
        self.assertIn("NOT comparable", self.sl)
        self.assertIn("NOT comparable", self.fb)
        self.assertIn("NOT comparable", self.st)


if __name__ == "__main__":
    unittest.main()
