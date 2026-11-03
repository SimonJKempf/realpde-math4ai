"""--init-from and the architecture flags. Seconds, CPU, no data.

Under --init-from train.py adopts the checkpoint's architecture. Until 2026-09-24 every other
architecture flag was dropped without a word, so --rollout-frames (W87, AR1, RINF) and
--temporal-lift (W203, A1) arms silently trained the baseline recipe. These tests pin the
guard (a differing shape-changing flag is refused), the escape hatch for the rollout case
(--init-slice-output: slice the output head to the first K frames), and the parcels' habit of
passing default-matching flags beside --init-from (still accepted).
"""
import contextlib
import io
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from pde_competition.model import FlowerConfig, FlowerS2S


def _train_module():
    # same pattern as test_model.py / test_train_rollout.py: load scripts/train.py by path
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "trn_initarch", str(Path(__file__).resolve().parents[1] / "scripts" / "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

TINY = dict(lifting_dim=16, n_levels=1, num_heads=8, groups=8, c_in=2, c_out=2)


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args)


class InitFromArchitecture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.trn = _train_module()
        cls.tmp = tempfile.TemporaryDirectory()
        torch.manual_seed(0)
        cls.src = FlowerS2S(FlowerConfig(**TINY))
        cls.src.set_stats([0.1, -0.02], [0.3, 0.05])
        # exactly as train.py writes best.pth: FlowerS2S.save -> {"config", "state_dict", **extra}
        cls.ckpt_path = cls.src.save(Path(cls.tmp.name) / "best.pth", epoch=3, val_rel_l2=0.5)
        cls.ckpt = torch.load(cls.ckpt_path, map_location="cpu", weights_only=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _adopt(self, flags, saved=None):
        a = _quiet(self.trn.parse_args, flags + ["--init-from", str(self.ckpt_path)])
        return _quiet(self.trn.adopt_init_config, saved or self.ckpt["config"], a,
                      self.trn.build_parser())

    # (a)
    def test_rollout_frames_without_slice_is_refused(self):
        with self.assertRaises(SystemExit) as cm:
            self._adopt(["--rollout-frames", "10"])
        msg = str(cm.exception)
        self.assertIn("--rollout-frames 10", msg)
        self.assertIn("rollout_frames=10", msg)
        self.assertIn("rollout_frames=0", msg)
        self.assertIn("--init-slice-output", msg)

    # (b)
    def _sliced_model(self, k):
        cfg_d, slice_output = self._adopt(["--rollout-frames", str(k), "--init-slice-output"])
        self.assertTrue(slice_output)
        cfg = FlowerConfig(**cfg_d)
        model = FlowerS2S(cfg)
        sd = _quiet(self.trn.slice_output_state_dict, self.ckpt["state_dict"], model.state_dict(),
                    cfg.rollout_frames, cfg.c_out, cfg.t_out)
        model.load_state_dict(sd)                       # strict
        return cfg, model

    def test_slice_output_builds_a_rollout_model_with_the_first_k_frames(self):
        for k in (10, 5):
            with self.subTest(k=k):
                cfg, model = self._sliced_model(k)
                self.assertEqual(cfg.rollout_frames, k)
                keep = k * cfg.c_out
                src = self.ckpt["state_dict"]
                got = model.state_dict()
                self.assertEqual(set(got), set(src))
                sliced = sorted(n for n in src if src[n].shape != got[n].shape)
                self.assertEqual(sliced, ["core.project.2.bias", "core.project.2.weight"])
                for n in sliced:
                    self.assertEqual(src[n].shape[0], cfg.t_out * cfg.c_out)
                    self.assertTrue(torch.equal(got[n], src[n][:keep]), n)
                for n in set(src) - set(sliced):
                    self.assertTrue(torch.equal(got[n], src[n]), n)

    def test_sliced_first_pass_reproduces_the_checkpoints_first_k_frames(self):
        # Frame-major channels: rows 0:k*c_out of the head ARE frames 0:k, so the rollout
        # model's first pass equals the checkpoint's one-pass output on those frames.
        torch.manual_seed(1)
        x = torch.randn(2, 20, 32, 64, 2) * 0.1
        self.src.eval()
        with torch.no_grad():
            ref = self.src(x)
            for k in (10, 5):
                cfg, model = self._sliced_model(k)
                model.eval()
                first = model._one_pass(x)
                self.assertEqual(tuple(first.shape), (2, k, 32, 64, 2))
                torch.testing.assert_close(first, ref[:, :k], rtol=1e-5, atol=1e-6)
                self.assertEqual(tuple(model(x).shape), tuple(ref.shape))

    def test_slice_output_without_a_rollout_change_is_refused(self):
        with self.assertRaises(SystemExit) as cm:
            self._adopt(["--init-slice-output"])
        self.assertIn("--init-slice-output", str(cm.exception))

    def test_slice_output_needs_init_from(self):
        with self.assertRaises(SystemExit):
            _quiet(self.trn.parse_args, ["--rollout-frames", "10", "--init-slice-output"])

    def test_slice_output_k_must_divide_t_out(self):
        with self.assertRaises(SystemExit) as cm:
            self._adopt(["--rollout-frames", "3", "--init-slice-output"])
        self.assertIn("divide", str(cm.exception))

    def test_slice_refuses_any_other_shape_difference(self):
        other = FlowerS2S(FlowerConfig(**(TINY | {"lifting_dim": 32, "rollout_frames": 10})))
        with self.assertRaises(SystemExit):
            _quiet(self.trn.slice_output_state_dict, self.ckpt["state_dict"], other.state_dict(), 10, 2, 20)

    # (c)
    def test_todays_parcel_flags_against_the_real_d128_pretrain_are_accepted(self):
        # The d128 pretrain's config as its adoption line prints it (lifting 128, 2 levels,
        # 16 heads, 16 groups, c 3/3, PERIODIC, rollout 0, rans False, temporal_lift 0). Built
        # from FlowerConfig as a dict only: adopt_init_config is pure, no model is needed.
        real = asdict(FlowerConfig(lifting_dim=128, n_levels=2, num_heads=16, groups=16,
                                   c_in=3, c_out=3, boundary_condition_types=("PERIODIC",)))
        recipe = ["--delta", "--mix-phases", "0.75", "--lifting-dim", "128", "--n-levels", "2",
                  "--channels", "3", "--batch-size", "16", "--stride", "5", "--epochs", "40",
                  "--lr", "3e-3", "--tke-weight", "0.08", "--coverage-weight", "0.4",
                  "--boundary", "periodic"]
        cfg_d, slice_output = self._adopt(recipe, saved=real)
        self.assertFalse(slice_output)
        cfg = FlowerConfig(**cfg_d)
        self.assertTrue(cfg.predict_delta)
        self.assertEqual((cfg.lifting_dim, cfg.n_levels, cfg.c_in, cfg.rollout_frames), (128, 2, 3, 0))
        for k in (10, 5):
            cfg_d, slice_output = self._adopt(recipe + ["--rollout-frames", str(k), "--init-slice-output"],
                                              saved=real)
            self.assertTrue(slice_output)
            self.assertEqual(FlowerConfig(**cfg_d).rollout_frames, k)

    def test_the_same_flags_against_a_checkpoint_they_do_not_match_are_refused(self):
        with self.assertRaises(SystemExit) as cm:
            self._adopt(["--lifting-dim", "128", "--n-levels", "2", "--channels", "3"])
        msg = str(cm.exception)
        # --channels 3 is the parser DEFAULT, so it counts as not given (the documented blind spot) and is
        # adopted from the checkpoint rather than refused -- only the non-default flags are named.
        for part in ("--lifting-dim 128", "lifting_dim=16", "--n-levels 2", "n_levels=1"):
            self.assertIn(part, msg)

    # (d)
    def test_rans_against_a_rans_free_checkpoint_is_refused(self):
        with self.assertRaises(SystemExit) as cm:
            self._adopt(["--rans"])
        self.assertIn("--rans True", str(cm.exception))
        self.assertIn("rans=False", str(cm.exception))

    def test_temporal_lift_against_a_checkpoint_without_it_now_overrides(self):
        # Refused until 2026-09-25; BET B made an ADDED lift a sanctioned override (tlift at
        # identity, step 0 = checkpoint). tests/test_bets_0925.py pins the load and the refusals.
        cfg_d, slice_output = self._adopt(["--temporal-lift", "5"])
        self.assertFalse(slice_output)
        self.assertEqual(cfg_d["temporal_lift"], 5)

    def test_shape_free_flags_still_warn_and_are_not_adopted(self):
        # --dropout used to be the example here; since 2026-09-25 (PNO_D probe) it OVERRIDES the
        # inherited rate like --delta (tests/test_energy_score.py pins that), so the still-warned
        # --window-scale takes its place.
        a = _quiet(self.trn.parse_args, ["--window-scale", "speed", "--init-from", str(self.ckpt_path)])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cfg_d, _ = self.trn.adopt_init_config(self.ckpt["config"], a, self.trn.build_parser())
        self.assertEqual(cfg_d["window_scale_mode"], "u")
        self.assertIn("WARNING: --window-scale", out.getvalue())

    def test_dropout_now_overrides_instead_of_warning(self):
        a = _quiet(self.trn.parse_args, ["--dropout", "0.1", "--init-from", str(self.ckpt_path)])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cfg_d, _ = self.trn.adopt_init_config(self.ckpt["config"], a, self.trn.build_parser())
        self.assertEqual(cfg_d["dropout_rate"], 0.1)
        self.assertNotIn("WARNING: --dropout", out.getvalue())
        self.assertIn("--dropout 0.1 OVERRIDES", out.getvalue())

    def test_no_architecture_flags_adopts_the_checkpoint_unchanged(self):
        cfg_d, slice_output = self._adopt([])
        self.assertFalse(slice_output)
        self.assertEqual(FlowerConfig(**cfg_d), FlowerConfig(**TINY))


if __name__ == "__main__":
    unittest.main()
