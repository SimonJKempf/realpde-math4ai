"""Truncated backprop through a --rollout-frames rollout (2026-09-25): FlowerS2S.rollout_detach,
set by scripts/train.py under --rollout-detach-passes.

    python -m unittest tests.test_rollout_detach

Seconds, CPU, no data. The attribute must change GRADIENTS only -- never a forward value, never
the checkpoint config the shipped module rebuilds from -- and the flag must be refused when the
final config has no rollout (it would be a silent control).
"""
import contextlib
import io
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from pde_competition.geometry import H_S, T_IN, T_OUT, W_S
from pde_competition.model import FlowerConfig, FlowerS2S

# tests/test_model.py's _tiny() configuration, extended with rollout_frames. predict_delta makes
# each pass depend on its input window through the residual, so the fed-back frames provably
# carry gradient when they are not detached.
TINY = dict(lifting_dim=16, n_levels=1, num_heads=8, groups=8, c_in=2, c_out=2,
            dropout_rate=0.0, predict_delta=True)


def _tiny(k, seed=0):
    torch.manual_seed(seed)
    m = FlowerS2S(FlowerConfig(**TINY, rollout_frames=k))
    m.set_stats([0.1, -0.02], [0.3, 0.05])
    return m


def _x(b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, T_IN, H_S, W_S, 2, generator=g)


def _train_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "trn_rd", str(Path(__file__).resolve().parents[1] / "scripts" / "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kw)


def _record_passes(m):
    """Wrap m._one_pass so every pass's output is kept (the tensors stay in the graph)."""
    passes, orig = [], m._one_pass

    def wrapped(w):
        out = orig(w)
        passes.append(out)
        return out
    m._one_pass = wrapped
    return passes


class ForwardUnchanged(unittest.TestCase):
    """(1) The attribute changes no forward value: eval, no-grad, and train mode with grad."""

    def test_outputs_identical_with_and_without(self):
        x = _x()
        for k in (10, 5):
            m = _tiny(k)
            self.assertFalse(m.rollout_detach, "class default must be False")
            for train in (False, True):
                m.train(train)
                for grad in (False, True):
                    with torch.set_grad_enabled(grad):
                        m.rollout_detach = False
                        a = m(x)
                        m.rollout_detach = True
                        b = m(x)
                    self.assertTrue(torch.equal(a, b), f"K={k} train={train} grad={grad}")
            m.rollout_detach = False


class GradientTruncated(unittest.TestCase):
    """(2) With the attribute set, the last pass's loss sends no gradient into earlier passes."""

    def _grads(self, k, detach):
        m = _tiny(k).train()
        m.rollout_detach = detach
        passes = _record_passes(m)
        out = m(_x())
        self.assertEqual(len(passes), T_OUT // k)
        loss = out[:, -k:].pow(2).mean()                     # the LAST pass only
        params = [p for p in m.parameters() if p.requires_grad]
        g = torch.autograd.grad(loss, params + [passes[0]], allow_unused=True)
        g_first = g[-1] if g[-1] is not None else torch.zeros_like(passes[0])
        return [gi if gi is not None else torch.zeros_like(p) for gi, p in zip(g[:-1], params)], g_first

    def test_last_pass_gradient_is_truncated(self):
        for k in (10, 5):
            gp_full, gf_full = self._grads(k, False)
            gp_det, gf_det = self._grads(k, True)
            # Without detach the last pass's loss reaches the first pass through the fed-back window.
            self.assertGreater(gf_full.abs().max().item(), 0.0, f"K={k}: no gradient even without detach")
            # With it, exactly zero.
            self.assertEqual(gf_det.abs().max().item(), 0.0, f"K={k}: gradient leaked into pass 1")
            # And the parameter gradient differs (it lost the through-earlier-passes paths).
            self.assertFalse(all(torch.allclose(a, b) for a, b in zip(gp_full, gp_det)),
                             f"K={k}: parameter gradient unchanged by detach")

    def test_detached_gradient_equals_the_last_pass_alone(self):
        """The truncated gradient is exactly the last pass trained on its (constant) input window."""
        k = 5
        m = _tiny(k).train()
        m.rollout_detach = True
        x = _x()
        loss = m(x)[:, -k:].pow(2).mean()
        params = list(m.parameters())
        g_det = torch.autograd.grad(loss, params, allow_unused=True)
        # Rebuild the last pass's input window without any graph, then run that pass alone.
        with torch.no_grad():
            m.rollout_detach = False
            full = m(x)
        window = torch.cat((x[:, 3 * k:], full[:, :3 * k]), dim=1)
        loss_last = m._one_pass(window).pow(2).mean()
        g_last = torch.autograd.grad(loss_last, params, allow_unused=True)
        for a, b in zip(g_det, g_last):
            if a is None or b is None:
                self.assertIs(a, b)
            else:
                self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-5))


class NotInConfig(unittest.TestCase):
    """(3) The attribute never reaches the checkpoint: a loaded model has the default."""

    def test_save_load_round_trip(self):
        m = _tiny(5)
        m.rollout_detach = True
        with tempfile.TemporaryDirectory() as d:
            p = m.save(Path(d) / "best.pth", epoch=1)
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m2 = FlowerS2S.from_checkpoint(p)
        self.assertNotIn("rollout_detach", ck["config"])
        self.assertNotIn("rollout_detach", ck["state_dict"])
        self.assertEqual(ck["config"], asdict(FlowerConfig(**TINY, rollout_frames=5)))
        self.assertFalse(m2.rollout_detach)
        self.assertEqual(m2.cfg, m.cfg)
        # and a copy of the live model (co-members are deepcopies) keeps it
        import copy
        self.assertTrue(copy.deepcopy(m).rollout_detach)


class TrainFlag(unittest.TestCase):
    """(4) train.py refuses --rollout-detach-passes when the FINAL config has no rollout."""

    @classmethod
    def setUpClass(cls):
        cls.trn = _train_module()

    def test_refused_without_rollout(self):
        a = _quiet(self.trn.parse_args, ["--rollout-detach-passes"])
        self.assertTrue(a.rollout_detach_passes)
        with self.assertRaises(SystemExit):
            self.trn.check_rollout_detach(a, FlowerConfig(**TINY))

    def test_accepted_with_rollout_and_default_off(self):
        a = _quiet(self.trn.parse_args, ["--rollout-frames", "5", "--rollout-detach-passes"])
        self.trn.check_rollout_detach(a, FlowerConfig(**TINY, rollout_frames=5))
        a0 = _quiet(self.trn.parse_args, [])
        self.assertFalse(a0.rollout_detach_passes)
        self.trn.check_rollout_detach(a0, FlowerConfig(**TINY))      # off: nothing to check

    def test_init_from_rollout_comes_from_slice_output(self):
        """Under --init-from the rollout arrives via --init-slice-output (adopt_init_config)."""
        saved = asdict(FlowerConfig(**TINY))                         # a rollout_frames=0 pretrain
        a = _quiet(self.trn.parse_args, ["--init-from", "unused.pth", "--rollout-frames", "5",
                                         "--init-slice-output", "--rollout-detach-passes"])
        cfg_d, slice_output = _quiet(self.trn.adopt_init_config, saved, a, self.trn.build_parser())
        self.assertTrue(slice_output)
        self.trn.check_rollout_detach(a, FlowerConfig(**cfg_d))      # accepted
        # the same pretrain without a rollout request: refused
        a = _quiet(self.trn.parse_args, ["--init-from", "unused.pth", "--rollout-detach-passes"])
        cfg_d, _ = _quiet(self.trn.adopt_init_config, saved, a, self.trn.build_parser())
        with self.assertRaises(SystemExit):
            self.trn.check_rollout_detach(a, FlowerConfig(**cfg_d))


if __name__ == "__main__":
    unittest.main()
