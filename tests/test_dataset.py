"""The split and windowing arithmetic -- the two places a leak would hide. No data needed."""
import pathlib
import unittest
from dataclasses import dataclass

from pde_competition.dataset import (EXCLUDED, split_by_condition,
                                     window_starts)
from pde_competition.geometry import T_IN, T_OUT


@dataclass
class Case:
    re_nominal: int
    aoa: int


class SplitByCondition(unittest.TestCase):
    def test_holds_out_whole_reynolds_rows(self):
        cases = [Case(re, aoa) for re in (3750, 13950, 21600, 26700) for aoa in (0, 5, 10)]
        train, val = split_by_condition(cases)
        self.assertEqual({c.re_nominal for c in val}, {13950, 26700})
        self.assertEqual({c.re_nominal for c in train}, {3750, 21600})
        # Nothing lost, nothing in both.
        self.assertEqual(len(train) + len(val), len(cases))
        self.assertFalse(set(map(id, train)) & set(map(id, val)))

    def test_every_aoa_of_a_held_out_row_is_held_out(self):
        # A cell-level split must not let one angle of a held-out Re leak.
        cases = [Case(13950, aoa) for aoa in range(0, 25, 5)]
        train, val = split_by_condition(cases)
        self.assertEqual(train, [])
        self.assertEqual(len(val), 5)


class WindowStarts(unittest.TestCase):
    def test_last_complete_window_is_included(self):
        n = T_IN + T_OUT              # exactly one window fits
        self.assertEqual(list(window_starts(n, stride=10)), [0])
        n = T_IN + T_OUT + 10         # two windows at stride 10
        self.assertEqual(list(window_starts(n, stride=10)), [0, 10])

    def test_too_short_yields_nothing(self):
        self.assertEqual(list(window_starts(T_IN + T_OUT - 1)), [])

    def test_every_window_is_complete(self):
        n = 1000
        for s in window_starts(n, stride=7):
            self.assertLessEqual(s + T_IN + T_OUT, n)


class Exclusions(unittest.TestCase):
    def test_known_bad_trajectories_stay_excluded(self):
        # 7575_0: announced by the organizers 2026-08-12.
        # 13950_15: broken pressure field, experiments/notes/analyses/data-13950_15-broken-pressure.md
        self.assertIn("7575_0", EXCLUDED["train_real"])
        self.assertIn("13950_15", EXCLUDED["train_sim"])


if __name__ == "__main__":
    unittest.main()


class RealMaskBank(unittest.TestCase):
    """--mask-from-real: the bank is boolean and a masked window is zero exactly where the mask is."""

    def test_masked_window_is_zero_where_the_mask_is(self):
        import numpy as np, torch
        from pde_competition.dataset import WindowDataset
        from pde_competition.geometry import H_S, W_S

        class _Run:
            key = (13950, 0); re_nominal = 13950; n_frames = 60
            def load(self):
                return np.full((60, H_S, W_S, 3), 0.5, np.float32)

        bank = np.ones((2, H_S, W_S), bool)
        bank[0, :4, :] = False                      # a hole across the top rows
        ds = WindowDataset([_Run()], stride=20, channels=3, mask_bank=bank, mask_seed=0)
        seen_holed = False
        for k in range(len(ds)):
            x, y = ds[k]
            for w in (x, y):
                holed = (w[:, :4, :, 0] == 0).all()
                if holed:
                    seen_holed = True
                    self.assertTrue(bool((w[:, 4:, :, 0] == 0.5).all()))   # untouched elsewhere
                else:
                    self.assertTrue(bool((w[..., 0] == 0.5).all()))        # the all-ones mask
        self.assertTrue(seen_holed, "no window drew the holed mask")
        # without a bank nothing is masked -- the default path must be untouched
        plain = WindowDataset([_Run()], stride=20, channels=3)
        x0, _ = plain[0]
        self.assertTrue(bool((x0 == 0.5).all()))


class MaskChannel(unittest.TestCase):
    """--mask-channel: the dataset and the shipped module must build the SAME plane."""

    def _run(self):
        import numpy as np
        from pde_competition.geometry import H_S, W_S
        class _Run:
            key = (26700, 0); re_nominal = 26700; n_frames = 60
            def load(self):
                a = np.full((60, H_S, W_S, 3), 0.25, np.float32)
                a[:, :3, :, 0] = 0.0          # an unmeasured band in u
                a[:, 10, 5, 0] = 0.0          # and one isolated dead pixel
                a[..., 2] = 0.0               # p is identically zero on real PIV
                return a
        return _Run()

    def test_dataset_writes_the_mask_into_the_dead_channel(self):
        import numpy as np, torch
        from pde_competition.dataset import WindowDataset
        off = WindowDataset([self._run()], stride=20, channels=3)
        on = WindowDataset([self._run()], stride=20, channels=3, mask_channel=True)
        x_off, y_off = off[0]
        x_on, y_on = on[0]
        self.assertTrue(bool((x_off[..., 2] == 0).all()))          # dead input today
        self.assertTrue(bool((x_on[..., :3, :, 2] == 0).all()))    # unmeasured band -> 0
        self.assertTrue(bool((x_on[:, 10, 5, 2] == 0).all()))      # dead pixel -> 0
        self.assertTrue(bool((x_on[:, 20:, :, 2] == 1).all()))     # measured -> 1
        torch.testing.assert_close(x_on[..., :2], x_off[..., :2])  # u, v untouched
        torch.testing.assert_close(y_on, y_off)                    # the TARGET is not touched

    def test_the_shipped_module_builds_an_identical_plane(self):
        """A train/score mismatch here would score the model on an input it never saw."""
        import numpy as np, torch
        from pde_competition.dataset import WindowDataset
        on = WindowDataset([self._run()], stride=20, channels=3, mask_channel=True)
        x_ds, _ = on[0]
        # the submission's three lines, verbatim in behaviour
        raw = self._run().load()[:x_ds.shape[0]].copy()
        xin = np.ascontiguousarray(raw[..., :3])
        xin[..., 2] = (xin[..., 0] != 0).astype(xin.dtype)
        np.testing.assert_array_equal(xin[..., 2], x_ds[..., 2].numpy())


class PhaseSplitExclusions(unittest.TestCase):
    """The four PIV phase splits are train_real on a shifted sub-grid. EXCLUDED is keyed
    {"train_real","train_sim"}, so a lookup for "train_real_p01" MISSES and returns an
    empty set -- a curriculum stage over all four phases would then train on 7575_0, the
    one row this project drops, three times over. The Mac caught the sibling defect in
    subsample_data.py by diffing file lists; nothing errored."""

    def test_phase_splits_inherit_train_real_exclusions(self):
        from pde_competition.dataset import EXCLUDED, EXCLUDED_REAL
        for split in ("train_real_p01", "train_real_p10", "train_real_p11"):
            exclude = EXCLUDED.get(split, frozenset())
            if not exclude:
                for k, v in EXCLUDED.items():
                    if split.startswith(k + "_"):
                        exclude = v
                        break
            self.assertEqual(exclude, EXCLUDED_REAL, f"{split} did not inherit the exclusion")

    def test_train_sim_is_not_captured_by_a_real_prefix(self):
        from pde_competition.dataset import EXCLUDED, EXCLUDED_SIM
        split = "train_sim_p01"
        exclude = EXCLUDED.get(split, frozenset())
        if not exclude:
            for k, v in EXCLUDED.items():
                if split.startswith(k + "_"):
                    exclude = v
                    break
        self.assertEqual(exclude, EXCLUDED_SIM)

    def test_the_subsampler_filters_on_the_stem(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "subsample_data.py").read_text()
        self.assertIn("p.stem not in exclude", src,
                      "EXCLUDED_REAL holds stems; matching on p.name is a silent no-op")


class PhaseMixingLeak(unittest.TestCase):
    """--mix-phases adds the three shifted-grid PIV copies as extra training windows. The
    phase stores contain the SAME trajectories as train_real, held-out rows included, so
    adding them raw would put row 26700 into TRAINING at a half-pixel shift -- essentially
    the validation flow itself, and the twin would then be meaningless. These assert the
    guard exists and that the held-out cells really are dropped."""

    def test_split_by_condition_is_applied_to_every_phase(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train.py").read_text()
        i = src.index("if a.mix_phases > 0:")
        # 4000, not 2000: the fix for the 2026-09-23 leak added a long comment above the call, which
        # pushed the line itself outside a 2000-char window and made this test fail on correct code.
        block = src[i:i + 4000]
        # The ORIGINAL form of this assertion was `split_by_condition(list_runs(ph))` -- which is
        # the bug. It checks that a split happens, not that it uses THIS RUN's holdout, and
        # split_by_condition defaults to (13950, 26700). Every run with --holdout-re set to anything
        # else therefore trained on half-pixel-shifted copies of its own validation rows, and this
        # test passed throughout. Assert the holdout is threaded through, not merely that a call exists.
        # Since 2026-09-23 (GTH) train.py has ONE split function per run, `_split`, defined once
        # from a.holdout_re (or the gapped split) and used at every call site. The phase stores
        # must go through it -- that is the structural form of the leak fix.
        self.assertIn("ph_runs, _ph_held = _split(_ph_all)", block,
                      "the phase split must use THIS RUN's split function, or the held-out rows "
                      "leak in through the shifted-grid copies")
        self.assertIn("return split_by_condition(rs, holdout_re=tuple(a.holdout_re))", src,
                      "_split must thread the run's --holdout-re into split_by_condition")
        self.assertNotIn("split_by_condition(list_runs(ph))", block,
                         "the bare form uses split_by_condition's DEFAULT holdout, not the run's")
        self.assertNotIn("split_by_condition(list_runs(ph)", block,
                         "the phase stores must not call split_by_condition directly at all")
        for ph in ("train_real_p01", "train_real_p10", "train_real_p11"):
            self.assertIn(ph, block)

    def test_mix_sim_also_uses_this_runs_holdout(self):
        """--mix-sim had the identical defect: the sim split also defaulted to (13950, 26700)."""
        src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train.py").read_text()
        self.assertIn('_split(list_runs("train_sim"))', src,
                      "the sim split must go through the run's one split function")
        self.assertNotIn('split_by_condition(list_runs("train_sim")', src,
                         "the sim stores must not call split_by_condition directly at all")

    def test_mix_phases_refuses_a_non_real_split(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train.py").read_text()
        i = src.index("if a.mix_phases > 0:")
        self.assertIn("--mix-phases adds the shifted-grid copies", src[i:i + 2000])

    def test_the_held_out_cells_are_the_same_in_a_phase_as_in_train_real(self):
        """The guard only works if split_by_condition picks the SAME cells for a phase
        store as for train_real -- it keys on (Re, aoa), which the shifted grid does not
        change, so this should hold by construction. Verified rather than assumed."""
        from pde_competition.dataset import list_runs, split_by_condition, PROCESSED
        if not (PROCESSED / "train_real_p01").exists():
            self.skipTest("phase store not built on this machine")
        _tr, held_a = split_by_condition(list_runs("train_real"))
        _tp, held_b = split_by_condition(list_runs("train_real_p01"))
        self.assertEqual(sorted(r.key for r in held_a), sorted(r.key for r in held_b))
