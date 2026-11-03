"""ROLLOUT FEED (2026-09-26, reviewer bug): FlowerConfig.rollout_feed.

    python -m unittest tests.test_rollout_feed

Seconds, CPU, no data. With rollout_feed=m the window fed to pass 2+ must hold the prediction's first
m channels times the ORIGINAL input's last-frame mask, and channels [m, c_in) carried over from that
frame -- never the model's own (untrained) pressure output. The first pass, the default (0 = legacy)
and the parameter shapes must be untouched; the flag must survive --init-from as an override.
"""
import unittest

import torch

from pde_competition.geometry import H_S, T_IN, T_OUT, W_S
from pde_competition.model import FlowerConfig, FlowerS2S

TINY = dict(lifting_dim=16, n_levels=1, num_heads=8, groups=8, c_in=3, c_out=3,
            dropout_rate=0.0, predict_delta=True)


def _tiny(k, feed, seed=0):
    torch.manual_seed(seed)
    m = FlowerS2S(FlowerConfig(**TINY, rollout_frames=k, rollout_feed=feed))
    with torch.no_grad():
        for q in m.parameters():
            q.normal_(0.0, 0.05)
    m.set_stats([0.1, -0.02, 0.0], [0.3, 0.05, 1.0])
    return m.eval()


def _x(b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = 0.1 * torch.randn(b, T_IN, H_S, W_S, 3, generator=g)
    x[..., 0] += 0.5
    x[..., 2] = 0.0                              # real data: no pressure
    x[:, :, :3, :4, :] = 0.0                     # a PIV hole (u == 0), static in time
    return x


class FeedComposesTheWindowLikeRolloutInput(unittest.TestCase):
    def _windows(self, m, x):
        seen = []
        orig = m._one_pass
        def spy(w):
            seen.append(w.clone())
            return orig(w)
        m._one_pass = spy
        with torch.no_grad():
            out = m(x)
        return seen, out

    def test_pass_two_sees_masked_uv_and_the_carried_p(self):
        k = 10
        m, x = _tiny(k, 2), _x()
        seen, out = self._windows(m, x)
        self.assertEqual(len(seen), T_OUT // k)
        w2 = seen[1]
        torch.testing.assert_close(w2[:, :T_IN - k], x[:, k:])            # the slid real frames
        fed = w2[:, T_IN - k:]
        self.assertTrue(torch.equal(fed[..., 2], torch.zeros_like(fed[..., 2])), "p must be the carried zero")
        self.assertTrue(torch.equal(fed[:, :, :3, :4, :2], torch.zeros_like(fed[:, :, :3, :4, :2])),
                        "holes must stay zero in the fed-back frames")
        torch.testing.assert_close(fed[:, :, 3:, :, :2], out[:, :k, 3:, :, :2])

    def test_legacy_feeds_the_raw_prediction(self):
        k = 10
        m, x = _tiny(k, 0), _x()
        seen, out = self._windows(m, x)
        torch.testing.assert_close(seen[1][:, T_IN - k:], out[:, :k])
        self.assertGreater(seen[1][:, T_IN - k:, ..., 2].abs().max().item(), 0.0)

    def test_first_pass_and_shapes_are_unchanged(self):
        for k in (10, 5):
            a, b = _tiny(k, 0), _tiny(k, 2)
            self.assertEqual({n: t.shape for n, t in a.state_dict().items()},
                             {n: t.shape for n, t in b.state_dict().items()})
            x = _x()
            with torch.no_grad():
                torch.testing.assert_close(a(x)[:, :k], b(x)[:, :k])

    def test_bad_values_are_refused(self):
        with self.assertRaises(ValueError):
            FlowerConfig(**TINY, rollout_frames=0, rollout_feed=2)          # no rollout
        with self.assertRaises(ValueError):
            FlowerConfig(**TINY, rollout_frames=10, rollout_feed=4)         # > c_out

    def test_gradients_flow_through_the_fed_channels(self):
        m = _tiny(10, 2)
        m.train()
        x = _x()
        m(x).pow(2).mean().backward()
        self.assertTrue(all(q.grad is not None for q in m.parameters() if q.requires_grad))


class InitFromOverride(unittest.TestCase):
    def test_the_flag_overrides_an_inherited_zero(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "trn_feed", str(Path(__file__).resolve().parents[1] / "scripts" / "train.py"))
        trn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trn)
        parser = trn.build_parser()
        from dataclasses import asdict
        saved = asdict(FlowerConfig(**TINY))
        a = parser.parse_args(["--rollout-frames", "10", "--init-slice-output", "--rollout-feed", "2"])
        out, _ = trn.adopt_init_config(dict(saved), a, parser)
        self.assertEqual(out["rollout_feed"], 2)
        self.assertEqual(out["rollout_frames"], 10)
        FlowerConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in out.items()})


if __name__ == "__main__":
    unittest.main()
