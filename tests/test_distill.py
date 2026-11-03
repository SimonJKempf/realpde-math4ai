"""DIST (2026-09-24): ensemble -> single-model distillation in scripts/train.py.

    python -m unittest tests.test_distill

Seconds, CPU, no data. The control has to be a real control (W = 0 must be
bit-identical to a run that never heard of teachers), the teachers must stay
frozen, and a student trained on two teachers must actually move toward their
mean -- otherwise a distilled arm is an unwitting control.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from pde_competition.geometry import H_S, T_IN, T_OUT, W_S
from pde_competition.model import FlowerConfig, FlowerS2S, rel_l2

# Same pattern as tests/test_model.py: the training script is loaded from its
# file (it is a script, not a package module), under a module name of its own so
# it cannot collide with test_model's copy in the import cache.


def _train_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "trn_dist", str(Path(__file__).resolve().parents[1] / "scripts" / "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny(lifting_dim=16):
    torch.manual_seed(0)
    return FlowerS2S(FlowerConfig(lifting_dim=lifting_dim, n_levels=1, num_heads=8, groups=8,
                                  c_in=2, c_out=2))


def _batch(b=2, seed=11):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, T_IN, H_S, W_S, 2, generator=g)
    y = torch.randn(b, T_OUT, H_S, W_S, 2, generator=g)
    return [(x, y)]


def _teachers(trn, n=2, scale=0.3):
    """n distinct frozen teachers: _tiny() is seeded, so every copy starts identical
    and jitter_member (a different seed each) is what makes them disagree."""
    out = []
    for k in range(n):
        t = _tiny()
        trn.jitter_member(t, 100 + k, scale)
        t.eval()
        for q in t.parameters():
            q.requires_grad_(False)
        out.append(t)
    return out


def _params(mdl):
    return [q.detach().clone() for q in mdl.parameters()]


class WeightZeroIsBitIdentical(unittest.TestCase):
    """(a) The default path must not move by a single bit."""

    def test_no_teachers_and_explicit_defaults_give_the_same_weights(self):
        trn = _train_module()
        outs = []
        for kw in ({}, {"teachers": None, "distill_w": 0.0, "truth_w": 1.0,
                        "distill_tke": False, "distill_stats": None}):
            torch.manual_seed(3)
            mdl = _tiny()
            opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
            trn.run_epoch(mdl, _batch(), torch.device("cpu"), 2, opt, None, tke_w=0.08, **kw)
            outs.append(_params(mdl))
        for a, b in zip(*outs):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_the_returned_number_is_still_the_plain_rel_l2(self):
        """With teachers the RETURN stays the truth rel-L2 -- the number every
        train_rel_l2 in summary.csv means -- and the components land in the dict."""
        trn = _train_module()
        torch.manual_seed(3)
        mdl = _tiny()
        teachers = _teachers(trn)
        opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
        (x, y), = _batch()
        with torch.no_grad():
            expected = rel_l2(mdl(x), y).item()
        stats = {}
        tr = trn.run_epoch(mdl, _batch(), torch.device("cpu"), 2, opt, None,
                           teachers=teachers, distill_w=1.0, distill_stats=stats)
        self.assertAlmostEqual(tr, expected, places=6)
        self.assertIn("truth", stats)
        self.assertIn("distill", stats)
        self.assertGreater(stats["distill"], 0.0)


class StudentMovesTowardTheTeacherMean(unittest.TestCase):
    """(b) With the truth switched off, a few steps must bring the student's
    prediction closer to the teacher mean in rel-L2 -- and W > 0 must change the
    step at all, or the flag is decoration."""

    def _distance(self, mdl, teachers, trn):
        (x, _), = _batch()
        mdl.eval()
        with torch.no_grad():
            d = rel_l2(mdl(x), trn._teacher_mean(teachers, x)).item()
        mdl.train()
        return d

    def test_distance_to_the_teacher_mean_decreases(self):
        trn = _train_module()
        torch.manual_seed(3)
        mdl = _tiny()
        teachers = _teachers(trn)
        before = self._distance(mdl, teachers, trn)
        self.assertGreater(before, 0.0, "student and teachers start identical; the jitter did nothing")
        opt = torch.optim.AdamW(mdl.parameters(), lr=1e-2)
        stats_first, stats_last = None, None
        for step in range(20):
            stats = {}
            trn.run_epoch(mdl, _batch(), torch.device("cpu"), 2, opt, None,
                          teachers=teachers, distill_w=1.0, truth_w=0.0, distill_stats=stats)
            stats_first = stats_first or stats
            stats_last = stats
        after = self._distance(mdl, teachers, trn)
        self.assertLess(after, before,
                        f"20 distillation steps did not bring the student closer: {before:.4f} -> {after:.4f}")
        self.assertLess(stats_last["distill"], stats_first["distill"])

    def test_a_nonzero_weight_actually_changes_the_step(self):
        trn = _train_module()
        outs = []
        teachers = _teachers(trn)
        for w in (0.0, 1.0):
            torch.manual_seed(3)
            mdl = _tiny()
            opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
            trn.run_epoch(mdl, _batch(), torch.device("cpu"), 2, opt, None, tke_w=0.08,
                          teachers=teachers if w else None, distill_w=w)
            outs.append(_params(mdl))
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(*outs)),
                         "--distill-weight 1.0 left every parameter where 0.0 did")

    def test_distill_tke_changes_the_step_too(self):
        trn = _train_module()
        outs = []
        teachers = _teachers(trn)
        for flag in (False, True):
            torch.manual_seed(3)
            mdl = _tiny()
            opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
            trn.run_epoch(mdl, _batch(), torch.device("cpu"), 2, opt, None, tke_w=0.08,
                          teachers=teachers, distill_w=1.0, distill_tke=flag)
            outs.append(_params(mdl))
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(*outs)),
                         "--distill-tke left every parameter where the plain distillation did")


class TeachersStayFrozen(unittest.TestCase):
    """(c) Loaded through the same helper main() uses: eval mode, requires_grad
    False, and bitwise unchanged by a training pass."""

    def test_loaded_teachers_are_frozen_and_unchanged_by_training(self):
        trn = _train_module()
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for k in range(2):
                t = _tiny()
                trn.jitter_member(t, 200 + k, 0.3)
                paths.append(t.save(Path(d) / f"teacher{k}.pth"))
            student = _tiny()
            teachers = trn.load_teachers(paths, student.cfg, torch.device("cpu"))
            self.assertEqual(len(teachers), 2)
            for t in teachers:
                self.assertFalse(t.training, "teacher not in eval mode")
                self.assertFalse(any(q.requires_grad for q in t.parameters()))
                self.assertTrue(all(q.dtype == torch.float32 for q in t.parameters()))
            before = [{k: v.clone() for k, v in t.state_dict().items()} for t in teachers]
            opt = torch.optim.AdamW(student.parameters(), lr=1e-2)
            for _ in range(3):
                trn.run_epoch(student, _batch(), torch.device("cpu"), 2, opt, None, tke_w=0.08,
                              teachers=teachers, distill_w=1.0, distill_tke=True)
            for t, sd in zip(teachers, before):
                for k, v in t.state_dict().items():
                    self.assertTrue(torch.equal(v, sd[k]), f"teacher parameter {k} moved")
                self.assertFalse(t.training, "run_epoch flipped a teacher into train mode")
                self.assertFalse(any(q.requires_grad for q in t.parameters()))

    def test_teacher_mean_is_the_output_space_average(self):
        trn = _train_module()
        teachers = _teachers(trn, n=3)
        (x, _), = _batch()
        with torch.no_grad():
            expected = sum(t(x) for t in teachers) / 3
        got = trn._teacher_mean(teachers, x)
        self.assertFalse(got.requires_grad)
        torch.testing.assert_close(got, expected)


class ArgparseDefaultsAndGuards(unittest.TestCase):
    """(d) Off by default; every combination that would be a silent control is refused."""

    def _parse(self, argv):
        trn = _train_module()
        with mock.patch("sys.argv", ["train.py", *argv]):     # parse_args() reads sys.argv itself
            return trn.parse_args()

    def test_defaults_are_off(self):
        a = self._parse([])
        self.assertIsNone(a.teacher_ensemble)
        self.assertEqual(a.distill_weight, 0.0)
        self.assertEqual(a.distill_truth_weight, 1.0)
        self.assertFalse(a.distill_tke)

    def test_a_full_distillation_line_parses(self):
        a = self._parse(["--teacher-ensemble", "a.pth", "b.pth", "--distill-weight", "1.0",
                         "--distill-truth-weight", "0.5", "--distill-tke", "--tke-weight", "0.08"])
        self.assertEqual(a.teacher_ensemble, ["a.pth", "b.pth"])
        self.assertEqual(a.distill_weight, 1.0)
        self.assertEqual(a.distill_truth_weight, 0.5)
        self.assertTrue(a.distill_tke)

    def test_silent_controls_are_refused(self):
        for argv in (["--teacher-ensemble", "a.pth"],                                  # W = 0
                     ["--distill-weight", "1.0"],                                      # no teachers
                     ["--distill-truth-weight", "0.5"],
                     ["--distill-tke"],
                     ["--teacher-ensemble", "a.pth", "--distill-weight", "1.0", "--distill-tke"],   # tke_w 0
                     ["--teacher-ensemble", "a.pth", "--distill-weight", "1.0", "--co-members", "2"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                self._parse(argv)


class ConfigMismatchIsRefused(unittest.TestCase):
    """(e) A teacher of another lifting_dim is refused before any batch is seen."""

    def test_lifting_dim_mismatch_raises_system_exit(self):
        trn = _train_module()
        with tempfile.TemporaryDirectory() as d:
            path = _tiny(lifting_dim=32).save(Path(d) / "teacher.pth")
            student = _tiny(lifting_dim=16)
            with self.assertRaises(SystemExit) as cm:
                trn.load_teachers([path], student.cfg, torch.device("cpu"))
            self.assertIn("lifting_dim", str(cm.exception))

    def test_a_rollout_student_accepts_one_pass_teachers(self):
        """Rollout students (2026-09-25): a K=10 student distils from one-pass teachers --
        both forwards return t_out frames, so the distillation target lines up."""
        trn = _train_module()
        with tempfile.TemporaryDirectory() as d:
            path = _tiny().save(Path(d) / "teacher.pth")
            torch.manual_seed(0)
            student = FlowerS2S(FlowerConfig(lifting_dim=16, n_levels=1, num_heads=8, groups=8,
                                             c_in=2, c_out=2, rollout_frames=10))
            teachers = trn.load_teachers([path], student.cfg, torch.device("cpu"))
            (x, _), = _batch()
            with torch.no_grad():
                self.assertEqual(tuple(student(x).shape), tuple(trn._teacher_mean(teachers, x).shape))
            opt = torch.optim.AdamW(student.parameters(), lr=1e-2)
            trn.run_epoch(student, _batch(), torch.device("cpu"), 2, opt, None, tke_w=0.08,
                          teachers=teachers, distill_w=1.0, distill_tke=True)

    def test_missing_checkpoint_raises_system_exit(self):
        trn = _train_module()
        with self.assertRaises(SystemExit):
            trn.load_teachers(["/nonexistent/teacher.pth"], _tiny().cfg, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
