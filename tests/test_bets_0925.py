"""BET A (--co-energy-weight) and BET B (--temporal-lift as an override under --init-from),
2026-09-25 ("run every bet"). Seconds, CPU, no data.

A: the co-trained members' relative fair energy score, added to the summed per-member point
   losses. At weight 0 the step must be BITWISE the step without the flag (the control is a
   real control); > 0 must change it and report a finite train_co_energy.
B: a temporal lift ADDED to a temporal_lift-0 checkpoint: the adopted config gains it, the
   load misses exactly tlift.weight and nothing else, tlift stays at identity, and the first
   forward equals the checkpoint's. A K that differs from a non-zero checkpoint K is refused.
"""
import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path

import torch

from pde_competition.geometry import H_S, T_IN, T_OUT, W_S
from pde_competition.model import FlowerConfig, FlowerS2S

from tests.test_model import _tiny, _train_module

TINY = dict(lifting_dim=16, n_levels=1, num_heads=8, groups=8, c_in=2, c_out=2)


def _quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kw)


def _batch(b=2, seed=11):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, T_IN, H_S, W_S, 2, generator=g)
    y = torch.randn(b, T_OUT, H_S, W_S, 2, generator=g)
    return [(x, y)]


class CoEnergyScore(unittest.TestCase):
    """BET A."""

    @classmethod
    def setUpClass(cls):
        cls.trn = _train_module()

    def _members(self, n=3):
        mm = [_tiny() for _ in range(n)]
        for k, mdl in enumerate(mm):
            self.trn.jitter_member(mdl, 100 + k, 0.05)
        return mm

    def _step(self, lr=1e-3, **kw):
        mm = self._members()
        opt = torch.optim.AdamW([q for mdl in mm for q in mdl.parameters()], lr=lr)
        out = self.trn.run_epoch_co(mm, _batch(), torch.device("cpu"), 2, opt, None,
                                    tke_w=0.09, cov_w=0.4,
                                    cov_half=torch.tensor([0.0093, 0.0063]), **kw)
        return mm, out

    def test_weight_zero_is_bitwise_the_step_without_the_flag(self):
        ref, out_ref = self._step()
        got, out_got = self._step(co_energy_w=0.0)
        self.assertEqual(out_ref, out_got)
        self.assertNotIn("energy", out_got)
        for a, b in zip(ref, got):
            for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
                self.assertEqual(ka, kb)
                self.assertTrue(torch.equal(va, vb), f"{ka} differs at co_energy_w=0")

    def test_positive_weight_changes_the_step_and_reports_a_finite_energy(self):
        ref, _ = self._step()
        got, out = self._step(co_energy_w=0.5)
        self.assertIn("energy", out)
        self.assertTrue(math.isfinite(out["energy"]))
        self.assertGreaterEqual(out["energy"], -1e-6, "the fair ES is >= 0 by the triangle inequality")
        moved = any(not torch.equal(va, vb) for a, b in zip(ref, got)
                    for va, vb in zip(a.state_dict().values(), b.state_dict().values()))
        self.assertTrue(moved, "co_energy_w=0.5 left every parameter where weight 0 put it")

    def test_the_reported_energy_is_energy_score_over_the_members(self):
        # lr 0: the step moves nothing, so the members after it are the members it scored.
        mm, out = self._step(lr=0.0, co_energy_w=0.5)
        (x, y), = _batch()
        with torch.no_grad():
            for mdl in mm:
                mdl.train()
            preds = torch.stack([mdl(x)[..., :2] for mdl in mm])
            want = self.trn.energy_score(preds, y[..., :2]).item()
        self.assertAlmostEqual(out["energy"], want, places=5)

    def test_parse_args(self):
        p = lambda argv: _quiet(self.trn.parse_args, argv)
        self.assertEqual(p([]).co_energy_weight, 0.0)
        self.assertEqual(p(["--co-members", "5", "--co-energy-weight", "0.5"]).co_energy_weight, 0.5)
        for bad in (["--co-energy-weight", "0.5"],                                   # no --co-members
                    ["--co-members", "5", "--co-energy-weight", "0.5", "--co-lambda", "0.1"],
                    ["--co-members", "5", "--co-energy-weight", "0.5",
                     "--energy-score-samples", "4", "--dropout", "0.1"],
                    ["--co-members", "5", "--co-energy-weight", "-0.1"]):
            with self.assertRaises(SystemExit, msg=f"accepted {bad}"):
                p(bad)


class TemporalLiftOverride(unittest.TestCase):
    """BET B."""

    @classmethod
    def setUpClass(cls):
        cls.trn = _train_module()
        cls.tmp = tempfile.TemporaryDirectory()
        torch.manual_seed(0)
        cls.src = FlowerS2S(FlowerConfig(**TINY))
        cls.src.set_stats([0.1, -0.02], [0.3, 0.05])
        cls.ckpt_path = cls.src.save(Path(cls.tmp.name) / "best.pth", epoch=3, val_rel_l2=0.5)
        cls.ckpt = torch.load(cls.ckpt_path, map_location="cpu", weights_only=False)
        torch.manual_seed(1)
        cls.src3 = FlowerS2S(FlowerConfig(**(TINY | {"temporal_lift": 3})))
        cls.ckpt3_path = cls.src3.save(Path(cls.tmp.name) / "tl3.pth", epoch=3, val_rel_l2=0.5)
        cls.ckpt3 = torch.load(cls.ckpt3_path, map_location="cpu", weights_only=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _adopt(self, flags, ckpt_path, saved):
        a = _quiet(self.trn.parse_args, flags + ["--init-from", str(ckpt_path)])
        return _quiet(self.trn.adopt_init_config, saved, a, self.trn.build_parser())

    def _lifted(self, k=5):
        cfg_d, slice_output = self._adopt(["--temporal-lift", str(k)], self.ckpt_path, self.ckpt["config"])
        self.assertFalse(slice_output)
        model = FlowerS2S(FlowerConfig(**cfg_d))
        _quiet(self.trn.load_init_state_dict, model, self.ckpt["state_dict"], True)
        return cfg_d, model

    def test_adopt_accepts_an_added_lift_and_says_so(self):
        a = _quiet(self.trn.parse_args, ["--temporal-lift", "5", "--init-from", str(self.ckpt_path)])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cfg_d, _ = self.trn.adopt_init_config(self.ckpt["config"], a, self.trn.build_parser())
        self.assertEqual(cfg_d["temporal_lift"], 5)
        self.assertIn("--temporal-lift 5 OVERRIDES", out.getvalue())
        # everything else is the checkpoint's
        self.assertEqual(FlowerConfig(**(cfg_d | {"temporal_lift": 0})), FlowerConfig(**TINY))

    def test_load_leaves_tlift_at_identity_and_everything_else_bitwise(self):
        _, model = self._lifted(5)
        want = torch.zeros(1, 1, 5)
        want[0, 0, 2] = 1.0
        self.assertTrue(torch.equal(model.tlift.weight.detach(), want))
        sd = model.state_dict()
        self.assertEqual(set(sd) - set(self.ckpt["state_dict"]), {"tlift.weight"})
        for k, v in self.ckpt["state_dict"].items():
            self.assertTrue(torch.equal(sd[k], v), f"{k} not loaded bitwise")

    def test_step_zero_equals_the_checkpoint(self):
        _, model = self._lifted(5)
        self.src.eval()
        model.eval()
        g = torch.Generator().manual_seed(3)
        x = torch.randn(2, T_IN, H_S, W_S, 2, generator=g)
        with torch.no_grad():
            ref, got = self.src(x), model(x)
        self.assertLess((ref - got).abs().max().item(), 1e-6)

    def test_the_non_strict_load_allows_nothing_but_tlift(self):
        cfg_d, _ = self._adopt(["--temporal-lift", "5"], self.ckpt_path, self.ckpt["config"])
        model = FlowerS2S(FlowerConfig(**cfg_d))
        sd = dict(self.ckpt["state_dict"])
        sd.pop(next(k for k in sd if k.startswith("core.")))              # another key missing
        with self.assertRaises(SystemExit):
            _quiet(self.trn.load_init_state_dict, model, sd, True)
        sd = dict(self.ckpt["state_dict"]) | {"bogus.weight": torch.zeros(1)}  # an unexpected key
        with self.assertRaises(SystemExit):
            _quiet(self.trn.load_init_state_dict, model, sd, True)
        # without the flag the load stays strict: the missing tlift key is an error
        with self.assertRaises(RuntimeError):
            _quiet(self.trn.load_init_state_dict, model, self.ckpt["state_dict"], False)

    def test_a_k_that_differs_from_a_non_zero_checkpoint_k_is_refused(self):
        with self.assertRaises(SystemExit) as cm:
            self._adopt(["--temporal-lift", "5"], self.ckpt3_path, self.ckpt3["config"])
        self.assertIn("temporal_lift=3", str(cm.exception))
        # the matching K is accepted and changes nothing
        cfg_d, _ = self._adopt(["--temporal-lift", "3"], self.ckpt3_path, self.ckpt3["config"])
        self.assertEqual(cfg_d["temporal_lift"], 3)

    def test_an_even_k_is_still_refused(self):
        with self.assertRaises(SystemExit):
            self._adopt(["--temporal-lift", "4"], self.ckpt_path, self.ckpt["config"])


if __name__ == "__main__":
    unittest.main()
