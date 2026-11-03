"""PNO_D probe (2026-09-25): energy-score fine-tuning in scripts/train.py, the --dropout override
under --init-from, and the MC-dropout switch of scripts/mc_dropout_eval.py.

    python -m unittest tests.test_energy_score

Seconds, CPU, no data. The control has to be a real control (M = 0 bit-identical to a run that
never heard of the energy score), an ES arm on a dropout-free model has to be refused (its samples
would be identical and ES would be plain rel-L2), and the eval switch must put ONLY the dropout
layers in train mode.
"""
import contextlib
import io
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pde_competition.geometry import H_S, T_IN, T_OUT, W_S
from pde_competition.model import FlowerConfig, FlowerS2S

TINY = dict(lifting_dim=16, n_levels=1, num_heads=8, groups=8, c_in=2, c_out=2)


def _load(name, rel):
    # Same pattern as tests/test_model.py / test_distill.py: scripts are loaded from their file,
    # under a module name of their own so they cannot collide with another test's copy.
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parents[1] / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _train_module():
    return _load("trn_es", "scripts/train.py")


def _eval_module():
    return _load("mcd_es", "scripts/mc_dropout_eval.py")


def _tiny(dropout=0.2):
    torch.manual_seed(0)
    return FlowerS2S(FlowerConfig(**TINY, dropout_rate=dropout))


def _batch(b=2, seed=11):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, T_IN, H_S, W_S, 2, generator=g)
    y = torch.randn(b, T_OUT, H_S, W_S, 2, generator=g)
    return [(x, y)]


def _params(mdl):
    return [q.detach().clone() for q in mdl.parameters()]


def _quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kw)


def _np_energy(s, y):
    """The formula as written in the brief, loop by loop: per window
    mean_i ||s_i - y||/||y|| - 1/(2M(M-1)) sum_{i != j} ||s_i - s_j||/||y||, then the batch mean."""
    m, b = s.shape[:2]
    out = []
    for k in range(b):
        yn = np.linalg.norm(y[k].ravel())
        acc = sum(np.linalg.norm((s[i, k] - y[k]).ravel()) for i in range(m)) / m
        spr = sum(np.linalg.norm((s[i, k] - s[j, k]).ravel())
                  for i in range(m) for j in range(m) if i != j) / (2 * m * (m - 1))
        out.append((acc - spr) / yn)
    return float(np.mean(out))


class EnergyScoreFormula(unittest.TestCase):
    """(1) Identity, sign, and agreement with a hand-written numpy formula."""

    @classmethod
    def setUpClass(cls):
        cls.trn = _train_module()

    def test_all_samples_equal_to_the_target_score_exactly_zero_with_a_finite_gradient(self):
        g = torch.Generator().manual_seed(0)
        y = torch.randn(3, 4, 5, 6, 2, generator=g, dtype=torch.float64)
        s = y.unsqueeze(0).repeat(4, 1, 1, 1, 1, 1).requires_grad_(True)
        es = self.trn.energy_score(s, y)
        self.assertEqual(float(es), 0.0)
        es.backward()
        self.assertTrue(torch.isfinite(s.grad).all(), "zero-distance norms produced a NaN gradient")

    def test_random_samples_are_positive_and_match_numpy(self):
        g = torch.Generator().manual_seed(1)
        for m in (2, 3, 5):
            with self.subTest(m=m):
                y = torch.randn(3, 4, 5, 6, 2, generator=g, dtype=torch.float64)
                s = y + 0.3 * torch.randn(m, *y.shape, generator=g, dtype=torch.float64)
                es = float(self.trn.energy_score(s, y))
                self.assertGreater(es, 0.0)
                self.assertAlmostEqual(es, _np_energy(s.numpy(), y.numpy()), places=12)

    def test_it_is_relative_so_scaling_everything_leaves_it_unchanged(self):
        g = torch.Generator().manual_seed(2)
        y = torch.randn(2, 3, 4, 5, 2, generator=g, dtype=torch.float64)
        s = y + 0.2 * torch.randn(4, *y.shape, generator=g, dtype=torch.float64)
        self.assertAlmostEqual(float(self.trn.energy_score(7.5 * s, 7.5 * y)),
                               float(self.trn.energy_score(s, y)), places=12)

    def test_fair_estimator_is_zero_for_two_samples_symmetric_about_the_target(self):
        # Documented, not a bug: the fair (M(M-1)) estimator is proper in EXPECTATION only.
        # s = y +- d gives acc = ||d||, spread term = ||2d|| / 2 = ||d||.
        g = torch.Generator().manual_seed(3)
        y = torch.randn(2, 3, 4, 5, 2, generator=g, dtype=torch.float64)
        d = torch.randn(y.shape, generator=g, dtype=torch.float64)
        self.assertAlmostEqual(float(self.trn.energy_score(torch.stack([y + d, y - d]), y)), 0.0, places=12)

    def test_bad_inputs_are_refused(self):
        y = torch.zeros(2, 3, 4, 5, 2)
        with self.assertRaises(ValueError):
            self.trn.energy_score(y.unsqueeze(0), y)                    # M = 1
        with self.assertRaises(ValueError):
            self.trn.energy_score(torch.zeros(3, 2, 3, 4, 5, 1), y)     # shape mismatch


class OffIsBitIdentical(unittest.TestCase):
    """(2) M = 0 must not move a single bit -- on a model WITH dropout, so the RNG stream matters."""

    def test_no_es_kwargs_and_explicit_defaults_give_the_same_weights(self):
        trn = _train_module()
        outs = []
        for kw in ({}, {"es_m": 0, "es_w": 1.0, "es_replace": False, "es_stats": None}):
            torch.manual_seed(3)
            mdl = _tiny(0.2)
            opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
            trn.run_epoch(mdl, _batch(), torch.device("cpu"), 2, opt, None, tke_w=0.08, **kw)
            outs.append(_params(mdl))
        for a, b in zip(*outs):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


class Refusals(unittest.TestCase):
    """(3) An ES arm that would silently be a control, or a combination that is not implemented,
    is refused."""

    @classmethod
    def setUpClass(cls):
        cls.trn = _train_module()

    def test_es_on_a_model_without_dropout_is_refused(self):
        mdl = _tiny(0.0)
        opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
        with self.assertRaises(SystemExit) as cm:
            self.trn.run_epoch(mdl, _batch(), torch.device("cpu"), 2, opt, None, es_m=2)
        self.assertIn("--dropout", str(cm.exception))

    def test_cli_refusals(self):
        bad = [["--energy-score-samples", "1"],
               ["--energy-score-samples", "-2"],
               ["--energy-score-samples", "4", "--train-rollout-weight", "0.5"],
               ["--energy-score-samples", "4", "--co-members", "2"],
               ["--energy-score-samples", "4", "--teacher-ensemble", "t.pth", "--distill-weight", "1"],
               ["--energy-score-samples", "4", "--energy-score-weight", "0"],
               ["--energy-score-replace"],
               ["--energy-score-weight", "2"],
               ["--dropout", "1.0"]]
        for argv in bad:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                _quiet(self.trn.parse_args, argv)

    def test_the_probe_command_line_parses(self):
        a = _quiet(self.trn.parse_args, ["--energy-score-samples", "4", "--dropout", "0.1",
                                         "--energy-score-weight", "0.5", "--energy-score-replace"])
        self.assertEqual((a.energy_score_samples, a.dropout, a.energy_score_weight, a.energy_score_replace),
                         (4, 0.1, 0.5, True))
        d = _quiet(self.trn.parse_args, [])
        self.assertEqual((d.energy_score_samples, d.energy_score_weight, d.energy_score_replace), (0, 1.0, False))


class OnChangesTheStep(unittest.TestCase):
    """(4) M >= 2 must change the step (or the flag is decoration) and log a finite train_energy."""

    def _step(self, trn, **kw):
        torch.manual_seed(3)
        mdl = _tiny(0.2)
        opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
        tr = trn.run_epoch(mdl, _batch(), torch.device("cpu"), 2, opt, None, tke_w=0.08, **kw)
        return tr, _params(mdl)

    def test_es_changes_the_step_and_logs_a_finite_energy(self):
        trn = _train_module()
        _, base = self._step(trn)
        stats = {}
        tr, on = self._step(trn, es_m=4, es_stats=stats)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(base, on)), "ES did not change the step")
        self.assertIn("energy", stats)
        self.assertTrue(np.isfinite(stats["energy"]) and stats["energy"] > 0.0, stats)
        self.assertTrue(np.isfinite(tr))

    def test_replace_differs_from_add(self):
        trn = _train_module()
        _, add = self._step(trn, es_m=2)
        _, rep = self._step(trn, es_m=2, es_replace=True)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(add, rep)))


class McDropoutSwitch(unittest.TestCase):
    """(5) 'dropout on, rest eval' sets EXACTLY the nn.Dropout modules to train mode."""

    @classmethod
    def setUpClass(cls):
        cls.mcd = _eval_module()

    def test_only_dropout_modules_are_in_train_mode(self):
        mdl = _tiny(0.2)
        mdl.train()                                   # start from the wrong state on purpose
        n = self.mcd.set_mc_dropout(mdl)
        drops = [q for q in mdl.modules() if isinstance(q, nn.Dropout)]
        self.assertEqual(n, len(drops))
        self.assertGreater(n, 0)
        for name, q in mdl.named_modules():
            self.assertEqual(q.training, isinstance(q, nn.Dropout), name or "<root>")

    def test_samples_differ_in_mc_mode_and_not_in_eval(self):
        mdl = _tiny(0.2)
        (x, _), = _batch()
        with torch.no_grad():
            self.mcd.set_mc_dropout(mdl)
            a, b = mdl(x), mdl(x)
            self.assertFalse(torch.equal(a, b))
            mdl.eval()
            self.assertTrue(torch.equal(mdl(x), mdl(x)))

    def test_a_dropout_free_model_reports_no_active_dropout(self):
        self.assertEqual(self.mcd.set_mc_dropout(_tiny(0.0)), 0)

    def test_force_dropout_rebuilds_the_rate_and_loads_strict(self):
        src = _tiny(0.0)
        with tempfile.TemporaryDirectory() as d:
            path = src.save(Path(d) / "m.pth")
            m = self.mcd.load_member(path, force_dropout=0.3)
            self.assertEqual(m.cfg.dropout_rate, 0.3)
            self.assertGreater(self.mcd.set_mc_dropout(m), 0)
            for k, v in src.state_dict().items():
                self.assertTrue(torch.equal(v, m.state_dict()[k]), k)
            self.assertEqual(self.mcd.load_member(path).cfg.dropout_rate, 0.0)


class DropoutOverrideUnderInitFrom(unittest.TestCase):
    """(6) --dropout OVERRIDES the inherited dropout_rate under --init-from (pure adopt_init_config)."""

    @classmethod
    def setUpClass(cls):
        cls.trn = _train_module()

    def _adopt(self, flags, saved_dropout):
        saved = asdict(FlowerConfig(**TINY, dropout_rate=saved_dropout))
        a = _quiet(self.trn.parse_args, flags + ["--init-from", "unused.pth"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cfg_d, slice_output = self.trn.adopt_init_config(saved, a, self.trn.build_parser())
        self.assertFalse(slice_output)
        return cfg_d, out.getvalue()

    def test_dropout_overrides_and_the_dropout_free_weights_load_strict(self):
        cfg_d, msg = self._adopt(["--dropout", "0.1"], 0.0)
        self.assertEqual(cfg_d["dropout_rate"], 0.1)
        self.assertIn("OVERRIDES", msg)
        self.assertNotIn("WARNING", msg)
        model = FlowerS2S(FlowerConfig(**cfg_d))
        model.load_state_dict(_tiny(0.0).state_dict())          # strict: dropout carries no parameters
        self.assertGreater(sum(isinstance(q, nn.Dropout) for q in model.modules()), 0)

    def test_without_the_flag_the_checkpoint_rate_is_kept(self):
        self.assertEqual(self._adopt([], 0.2)[0]["dropout_rate"], 0.2)
        self.assertEqual(self._adopt([], 0.0)[0]["dropout_rate"], 0.0)

    def test_known_blind_spot_dropout_zero_equals_the_default_and_keeps_the_checkpoint_rate(self):
        # --dropout 0 is the parser default, so it reads as "not given" (the documented blind spot
        # of adopt_init_config): a dropout-0 control must start from a dropout-0 checkpoint.
        self.assertEqual(self._adopt(["--dropout", "0"], 0.2)[0]["dropout_rate"], 0.2)


if __name__ == "__main__":
    unittest.main()
