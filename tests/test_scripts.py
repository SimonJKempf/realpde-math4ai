"""The bookkeeping scripts, on synthetic run directories. No torch inference, no data."""
import json
import pathlib
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path

from pde_competition.paths import REPO

sys.path.insert(0, str(REPO / "scripts"))
import summarize_runs  # noqa: E402
import build_submission  # noqa: E402


def _write_run(root: Path, run_id: str, epochs, meta=True) -> Path:
    """A run directory as train.py writes it: log.jsonl (+ meta.json)."""
    d = root / run_id
    d.mkdir()
    with open(d / "log.jsonl", "w") as f:
        for i, (rl2, tke) in enumerate(epochs, start=1):
            f.write(json.dumps({
                "run_id": run_id, "epoch": i, "git_commit": "abc123def456789",
                "git_dirty": False, "split": "train_real", "seed": 0,
                "train_rel_l2": rl2 + 0.01, "val_rel_l2": rl2, "val_tke": tke,
                "val_score": 100 - 100 * rl2, "seconds": 30.0, "mix_sim": 0.0,
                "init_from": None,
                "persistence_floor": {"rel_l2": 0.137, "tke": 1.0},
                "config": {"c_in": 3, "lifting_dim": 64, "n_levels": 4,
                           "window_norm": False, "predict_delta": False,
                           "boundary_condition_types": ["ZEROS"]}}) + "\n")
    if meta:
        (d / "meta.json").write_text(json.dumps({
            "run_id": run_id,
            "launches": [{"started": "2026-09-09T10:00:00+00:00",
                          "git": {"commit": "abc123def456789", "dirty": False},
                          "args": {"batch_size": 16, "lr": 1e-3, "stride": 5,
                                   "wandb_group": "stride-seeds"},
                          "n_params": 2800000, "wandb_id": "xyz789"}]}))
    return d


class Summarize(unittest.TestCase):
    def test_best_epoch_is_recomputed_from_the_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _write_run(Path(tmp), "stride5_s1-20260909-100000",
                           [(0.10, 0.9), (0.081, 0.70), (0.085, 0.65)])
            row = summarize_runs.summarize(d)
        self.assertEqual(row["epochs"], 3)
        self.assertEqual(row["best_epoch"], 2)                 # lowest val_rel_l2
        self.assertAlmostEqual(row["best_val_rel_l2"], 0.081)
        self.assertAlmostEqual(row["val_tke_at_best"], 0.70)
        # best_score is recomputed with the CURRENT composite from the logged
        # errors, not read from the log's val_score -- so the expectation is
        # computed the same way rather than hard-coded to one set of weights.
        from pde_competition import composite
        scores = {2: composite.from_errors(0.081, 0.70), 3: composite.from_errors(0.085, 0.65)}
        want = max(scores, key=scores.get)
        self.assertEqual(row["best_score_epoch"], want)
        self.assertAlmostEqual(row["best_val_score"], scores[want])
        self.assertAlmostEqual(row["best_val_tke"], 0.65)      # separate optimum
        self.assertEqual(row["name"], "stride5_s1")
        self.assertEqual(row["git_commit"], "abc123def456")
        self.assertEqual(row["wandb_group"], "stride-seeds")
        self.assertEqual(row["stride"], 5)
        self.assertEqual(row["boundary"], "ZEROS")
        self.assertEqual(row["minutes"], 1.5)

    def test_run_without_meta_still_summarizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _write_run(Path(tmp), "old-20260901-000000", [(0.09, 0.8)], meta=False)
            row = summarize_runs.summarize(d)
        self.assertEqual(row["best_epoch"], 1)
        self.assertEqual(row["wandb_id"], "")
        self.assertEqual(row["batch_size"], "")

    def test_empty_run_directory_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "died-20260909-071715"
            d.mkdir()
            self.assertIsNone(summarize_runs.summarize(d))
            (d / "log.jsonl").write_text("")
            self.assertIsNone(summarize_runs.summarize(d))

    def test_truncated_last_line_is_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _write_run(Path(tmp), "cut-20260909-000000", [(0.09, 0.8)])
            with open(d / "log.jsonl", "a") as f:
                f.write('{"run_id": "cut", "epoch": 2, "val_rel')   # killed mid-write
            row = summarize_runs.summarize(d)
        self.assertEqual(row["epochs"], 1)

    def test_every_column_is_present_in_every_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _write_run(Path(tmp), "x-20260909-000000", [(0.09, 0.8)])
            row = summarize_runs.summarize(d)
        self.assertEqual(tuple(row), summarize_runs.COLUMNS)


class BoundsPlumbing(unittest.TestCase):
    def test_fit_bounds_and_build_submission_agree_on_the_bounds_path(self):
        # Two scripts, one convention: <ckpt stem>.bounds.json beside the ckpt.
        # Import fit_bounds lazily -- it pulls in torch and the kit scorer.
        import fit_bounds
        ckpt = Path("/x/outputs/runs/r-1/best_score.pth")
        self.assertEqual(fit_bounds.bounds_path_for(ckpt),
                         build_submission.bounds_path_for(ckpt))
        self.assertEqual(build_submission.bounds_path_for(ckpt).name,
                         "best_score.bounds.json")

    def test_flower_build_refuses_without_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "best.pth"
            ckpt.write_bytes(b"not a real checkpoint")
            sub = Path(tmp) / "submission.py"
            sub.write_text("")
            with self.assertRaises(SystemExit) as cm:
                build_submission.build(sub, ckpt, "t", "flower")
        self.assertIn("fit_bounds.py", str(cm.exception))


if __name__ == "__main__":
    unittest.main()


class SpatialBounds(unittest.TestCase):
    """fit_bounds.fit_map / objective_map on synthetic residuals. No model, no data."""

    def _fb(self):
        import fit_bounds  # lazy: pulls in torch and the kit scorer
        return fit_bounds

    def test_map_optimum_tracks_each_pixels_own_residual_scale(self):
        fb = self._fb()
        import numpy as np
        rng = np.random.default_rng(0)
        N, T, H, W = 40, 4, 3, 5
        target = np.ones((N, T, H, W), np.float32)
        scale = np.full((H, W), 0.002, np.float32)
        scale[:, -1] = 0.02                      # one column is 10x noisier
        pred = target + rng.normal(0, 1, (N, T, H, W)).astype(np.float32) * scale
        h = fb.fit_map(pred, target)
        self.assertEqual(h.shape, (H, W))
        self.assertTrue(np.all(h[:, :-1] < h[:, -1:]))
        # Per-element widths must beat the best single constant on the fit set.
        grid = np.linspace(1e-4, 0.1, 500)
        h_const = fb.fit_channel(pred, target, grid)[0]
        self.assertGreater(fb.objective_map(pred, target, h)[0],
                           fb.objective(pred, target, h_const)[0])

    def test_constant_map_reproduces_the_constant_objective(self):
        fb = self._fb()
        import numpy as np
        rng = np.random.default_rng(1)
        target = rng.normal(0, 1, (10, 2, 4, 4)).astype(np.float32)
        pred = target + rng.normal(0, 0.01, target.shape).astype(np.float32)
        o1, c1 = fb.objective(pred, target, 0.01)
        o2, c2 = fb.objective_map(pred, target, np.full((4, 4), 0.01, np.float32))
        self.assertAlmostEqual(o1, o2, places=7)
        self.assertAlmostEqual(c1, c2, places=7)

    def test_unscored_cells_get_zero_width_and_frame_map_has_time_axis(self):
        fb = self._fb()
        import numpy as np
        target = np.ones((6, 3, 2, 2), np.float32)
        target[..., 0, 0] = 0.0                  # inside the body: never scored
        pred = target + 0.001
        h = fb.fit_map(pred, target)
        self.assertEqual(h[0, 0], 0.0)
        self.assertEqual(fb.fit_map(pred, target, per_frame=True).shape, (3, 2, 2))
        self.assertEqual(fb.smooth3(h).shape, h.shape)


class PerRowValColumns(unittest.TestCase):
    def test_columns_fill_from_val_by_re_and_stay_empty_without_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = _write_run(root, "r-20260915-000000-h", [(0.09, 0.7), (0.08, 0.6)])
            # Rewrite the second epoch with per-row numbers, as train.py logs them now.
            lines = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines()]
            lines[1]["val_by_re"] = {"13950": {"rel_l2": 0.07, "tke": 0.5},
                                     "26700": {"rel_l2": 0.09, "tke": 0.7}}
            lines[1]["train_all"] = False
            (d / "log.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")
            row = summarize_runs.summarize(d)
            self.assertAlmostEqual(row["val_re13950_at_best"], 0.07)
            self.assertAlmostEqual(row["val_re26700_at_best"], 0.09)
            d2 = _write_run(root, "s-20260915-000000-h", [(0.09, 0.7)])
            row2 = summarize_runs.summarize(d2)
            self.assertIn(row2["val_re26700_at_best"], (None, ""))


class GraphPlan(unittest.TestCase):
    def test_chunking_covers_every_window_once_and_pads_only_lone_remainders(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "subflower", str(REPO / "submission" / "submission_flower.py"))
        # Importing the module needs model.pth/bounds.json beside it; test the
        # pure helper by extracting it from the source instead.
        src = (REPO / "submission" / "submission_flower.py").read_text()
        # _bucket reads the module-level ladder, so the isolated namespace needs it too.
        # Parsed from the source rather than hard-coded, so this test tracks the shipped
        # default instead of silently diverging from it.
        lad = re.search(r'RPDE_GRAPH_LADDER", "([0-9,]+)"', src).group(1)
        ns = {"_GRAPH_LADDER": tuple(int(s) for s in lad.split(","))}
        for name in ("def _bucket(", "def _plan("):
            if name in src:
                start = src.index(name); end = src.index("\n\n\n", start)
                exec(src[start:end], ns)
        plan = ns["_plan"]
        self.assertEqual(plan(1, 16), [(0, 1, 1)])
        self.assertEqual(plan(16, 16), [(0, 16, 16)])
        self.assertEqual(plan(17, 16), [(0, 16, 16), (16, 1, 1)])
        # LADDER {1,2,3,4,5,6,8,12,16} since 2026-09-21: sizes up to 6 are exact, because
        # the grader calls with n=5 and powers of two padded that to an 8-window graph.
        self.assertEqual(plan(2, 16), [(0, 2, 5)])
        self.assertEqual(plan(5, 16), [(0, 5, 5)])       # the grader's size: EXACT, worth 0.95 ms/window
        self.assertEqual(plan(7, 16), [(0, 7, 8)])
        self.assertEqual(plan(12, 16), [(0, 12, 16)])
        self.assertEqual(plan(20, 16), [(0, 16, 16), (16, 4, 5)])   # ladder {1,5,16}: a 4-remainder pads to 5
        covered = sorted(i for s, n, _ in plan(37, 16) for i in range(s, s + n))
        self.assertEqual(covered, list(range(37)))


class OOFWidthModel(unittest.TestCase):
    """oof_width_model's numpy fitting on synthetic rows whose residual scale grows with speed."""

    def _rows(self):
        import numpy as np
        rng = np.random.default_rng(3)
        rows = {}
        for i, re in enumerate([3750, 8850, 13950, 19050, 26700]):
            speed = 0.05 + 0.01 * i
            N, T, H, W = 30, 4, 6, 8
            y = np.ones((N, T, H, W, 2), np.float32); y[..., 0, 0, :] = 0.0      # a body pixel, unscored
            scale = 0.002 + 0.06 * speed                                           # width grows with speed
            r = rng.normal(0, 1, y.shape).astype(np.float32) * scale
            r[..., 3:, :, :] *= 3.0                                                # bottom half 3x noisier
            rows[re] = (y + r, y, np.full(N, speed, np.float32))
        return rows

    def test_affine_beats_constant_when_left_out_row_extrapolates(self):
        import oof_width_model as om
        rows = self._rows()
        loo = om.leave_one_row_out(rows, alphas=(0.5,))
        top = loo[26700]
        self.assertGreater(top["affine"], top["const"])
        self.assertGreater(top["shape_a0.5"], top["affine"])
        a, b = om.fit_affine(rows, 0)
        self.assertGreater(b, 0.0)

    def test_widths_are_finite_positive_and_shaped_per_window(self):
        import oof_width_model as om
        import numpy as np
        rows = self._rows()
        mdl = om.fit_model(rows, 0.5)
        h = om.widths(mdl, rows[13950][2], 0)
        self.assertEqual(h.shape, (30, 1, 6, 8))
        self.assertTrue(np.all(np.isfinite(h)) and np.all(h > 0))


class TrainAllIsQuarantined(unittest.TestCase):
    def test_in_sample_val_leaves_the_sortable_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _write_run(Path(tmp), "all-20260915-000000-h", [(0.071, 0.6)])
            lines = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines()]
            lines[-1]["train_all"] = True
            (d / "log.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")
            row = summarize_runs.summarize(d)
            self.assertEqual(row["best_val_rel_l2"], "")
            self.assertEqual(row["best_val_score"], "")
            self.assertAlmostEqual(row["in_sample_val_rel_l2"], 0.071)
            self.assertTrue(row["train_all"])


class DisagreementHalf(unittest.TestCase):
    def test_formula_is_linear_in_normalised_disagreement_and_clipped(self):
        import numpy as np
        src = (REPO / "submission" / "submission_flower.py").read_text()
        start = src.index("def _disagreement_half("); end = src.index("\n\n\n", start)
        ns = {"np": np}; exec(src[start:end], ns)
        f = ns["_disagreement_half"]
        z = np.zeros((2, 3, 4, 5, 3), np.float32); z[..., 0] = 0.002; z[..., 1] = 0.004
        a = np.array([0.004, 0.003, 0.0], np.float32); b = np.array([0.005, 0.0025, 0.0], np.float32)
        zref = np.array([0.002, 0.002, 1.0], np.float32)
        h = f(z, a, b, zref, 0.0, np.ones((1, 1, 3), np.float32))
        self.assertEqual(h.shape, z.shape)
        self.assertAlmostEqual(float(h[0, 0, 0, 0, 0]), 0.004 + 0.005 * 1.0, places=6)
        self.assertAlmostEqual(float(h[0, 0, 0, 0, 1]), 0.003 + 0.0025 * 2.0, places=6)
        self.assertEqual(float(h[0, 0, 0, 0, 2]), 0.0)
        big = f(z * 1e3, a, b, zref, 0.0, np.ones((1, 1, 3), np.float32))
        self.assertTrue(np.all(big <= 0.5))


class WidthHeadUnit(unittest.TestCase):
    """train_width_head: forward shape, bounded output, finite surrogate loss with gradients."""

    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("twh", str(REPO / "scripts" / "train_width_head.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m

    def test_forward_is_nonnegative_bounded_and_trainable(self):
        import torch
        twh = self._mod()
        head = twh.WidthHead(8)
        f = torch.randn(3, twh.N_FEAT, 32, 64)
        h = head(f)
        self.assertEqual(tuple(h.shape), (3, 2, 32, 64))
        self.assertTrue(torch.all(h >= 0) and torch.all(h <= 0.5))
        self.assertAlmostEqual(float(h.mean()), 0.009, places=3)      # initialised near the constant width
        r = torch.rand(3, 2, 32, 64) * 0.02
        s = torch.ones(3, 2, 32, 64)
        loss = twh.surrogate_loss(h, r, s, 0.002)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None))

    def test_penultimate_branch_is_opt_in_and_trainable(self):
        """Proposition 8: --pfeat adds a learned 1x1 bottleneck; n_pfeat=0 stays the shipping module."""
        import torch
        twh = self._mod()
        plain = twh.WidthHead(8, twh.N_FEAT)
        self.assertFalse(any(k.startswith("bottle") for k in plain.state_dict()))   # shipping head unchanged
        head = twh.WidthHead(8, twh.N_FEAT, n_pfeat=16, n_bottle=4)
        f = torch.randn(3, twh.N_FEAT, 32, 64)
        pf = torch.randn(3, 16, 32, 64)
        with self.assertRaises(ValueError):
            head(f)                                                    # the planes are not optional once built in
        h = head(f, pf)
        self.assertEqual(tuple(h.shape), (3, 2, 32, 64))
        self.assertTrue(torch.all(h >= 0) and torch.all(h <= 0.5))
        self.assertAlmostEqual(float(h.mean()), 0.009, places=3)       # still starts at the constant width
        # the output layer starts at exactly zero weight, which zeroes every upstream gradient by
        # construction; perturb it first, or "the bottleneck gets no gradient" is true of any wiring
        with torch.no_grad():
            head.net[-1].weight.add_(torch.randn_like(head.net[-1].weight) * 0.05)
        loss = twh.surrogate_loss(head(f, pf), torch.rand(3, 2, 32, 64) * 0.02, torch.ones(3, 2, 32, 64), 0.002)
        loss.backward()
        self.assertTrue(torch.isfinite(head.bottle.weight.grad).all())
        self.assertTrue(float(head.bottle.weight.grad.abs().sum()) > 0)  # the branch is in the graph

    def test_frame_index_maps_to_its_window(self):
        """The --pfeat gather is pf[idx // T]; it is only correct if features() is row-major (window, frame)."""
        import numpy as np
        twh = self._mod()
        N, T, H, W = 3, 4, 8, 12
        m = np.zeros((N, T, H, W, 2), np.float32); z = np.zeros_like(m)
        speed = np.array([0.10, 0.20, 0.30], np.float32)
        norm = {"mu": [0.0, 0.0], "sd": [1.0, 1.0], "zref": [1.0, 1.0], "speed_mu": 0.0, "speed_sd": 1.0}
        f = twh.features(m, z, speed, norm)
        self.assertEqual(f.shape[0], N * T)
        tt = np.linspace(-1, 1, T)
        for k in range(N * T):
            self.assertAlmostEqual(float(f[k, 4, 0, 0]), float(speed[k // T]), places=5)   # window plane
            self.assertAlmostEqual(float(f[k, 7, 0, 0]), float(tt[k % T]), places=5)       # frame plane

    def test_ext2_planes_are_additive_and_leave_ext_untouched(self):
        """ext2 appends 3 unsteadiness planes; the first 17 must be bit-identical to ext."""
        import numpy as np
        twh = self._mod()
        rng = np.random.default_rng(7)
        N, T, H, W, Tin = 2, 4, 8, 12, 5
        mean = rng.normal(0.16, 0.1, (N, T, H, W, 2)).astype(np.float32)
        x = rng.normal(0.16, 0.1, (N, Tin, H, W, 3)).astype(np.float32)
        x[:, :, :2, :, :] = 0.0                       # an unmeasured band
        e1 = twh.extra_planes(x, mean)
        e2 = twh.extra_planes(x, mean, ext2=True)
        self.assertEqual(e1.shape, (N, T, twh.N_EXTRA, H, W))
        self.assertEqual(e2.shape, (N, T, twh.N_EXTRA + twh.N_EXTRA2, H, W))
        np.testing.assert_array_equal(e2[:, :, :twh.N_EXTRA], e1)   # ext is a prefix of ext2
        self.assertTrue(np.all(e2[:, :, 9:11] >= 0.0))              # |increment| is non-negative
        self.assertTrue(np.all(e2[:, :, 9:11, :2, :] == 0.0))       # unmeasured band -> no motion
        self.assertTrue(np.all((e2[:, :, 11] >= 0.0) & (e2[:, :, 11] <= 10.0)))   # ratio clipped
        self.assertEqual(twh.PLANES["ext2"], 20)

    def test_features_have_eight_planes_per_frame(self):
        import numpy as np
        twh = self._mod()
        m = np.random.rand(2, 20, 32, 64, 2).astype(np.float32)
        z = (np.random.rand(2, 20, 32, 64, 2) * 0.002).astype(np.float32)
        norm = {"mu": [0.5, 0.5], "sd": [0.3, 0.3], "zref": [0.001, 0.001], "speed_mu": 0.11, "speed_sd": 0.01}
        F = twh.features(m, z, np.array([0.1, 0.12], np.float32), norm)
        self.assertEqual(tuple(F.shape), (40, twh.N_FEAT, 32, 64))


class AverageCheckpoints(unittest.TestCase):
    """average_checkpoints: the mean of two saved models is elementwise the mean, keeps the config, and loads back."""

    def test_average_is_the_mean_and_reloads(self):
        import importlib.util
        import torch
        from pde_competition.model import FlowerConfig, FlowerS2S
        spec = importlib.util.spec_from_file_location("avg", str(REPO / "scripts" / "average_checkpoints.py"))
        avg = importlib.util.module_from_spec(spec); spec.loader.exec_module(avg)
        small = dict(lifting_dim=32, num_heads=8, groups=8)
        with tempfile.TemporaryDirectory() as d:
            paths = []
            models = []
            for i in range(2):
                torch.manual_seed(i)
                m = FlowerS2S(FlowerConfig(**small))
                m.save(Path(d) / f"m{i}.pth", epoch=10 + i)
                paths.append(Path(d) / f"m{i}.pth"); models.append(m)
            out = Path(d) / "avg.pth"
            payload = avg.average_checkpoints(paths, out)
            self.assertEqual(payload["averaged_epochs"], [10, 11])
            got = FlowerS2S.from_checkpoint(out)
            s0, s1, sg = models[0].state_dict(), models[1].state_dict(), got.state_dict()
            for k in s0:
                if s0[k].is_floating_point():
                    torch.testing.assert_close(sg[k], (s0[k] + s1[k]) / 2)
                else:
                    self.assertTrue(torch.equal(sg[k], s0[k]))
            self.assertEqual(got.cfg.lifting_dim, 32)
            # a different config must be refused
            other = FlowerS2S(FlowerConfig(lifting_dim=16, num_heads=8, groups=8))
            other.save(Path(d) / "other.pth")
            with self.assertRaises(ValueError):
                avg.average_checkpoints([paths[0], Path(d) / "other.pth"], Path(d) / "bad.pth")


class WidthHeadInSubmission(unittest.TestCase):
    """submission_flower's _WidthHead / _head_features are exact twins of train_width_head's."""

    def _pieces(self):
        import importlib.util
        import numpy as np
        import torch
        spec = importlib.util.spec_from_file_location("twh", str(REPO / "scripts" / "train_width_head.py"))
        twh = importlib.util.module_from_spec(spec); spec.loader.exec_module(twh)
        src = (REPO / "submission" / "submission_flower.py").read_text()
        ns = {"np": np, "torch": torch}
        for name in ("class _WidthHead(", "def _head_extra(", "def _head_features(",
                     "def _head_extra_torch(", "def _head_features_torch("):
            start = src.index(name); end = src.index("\n\n\n", start)
            exec(src[start:end], ns)
        self._extra = ns["_head_extra"]; self._extra_t = ns["_head_extra_torch"]; self._feats_t = ns["_head_features_torch"]
        return twh, ns["_WidthHead"], ns["_head_features"]

    def test_features_and_forward_match_the_training_script(self):
        import numpy as np
        import torch
        twh, Head, feats = self._pieces()
        rng = np.random.default_rng(0)
        N, T, H, W = 2, 3, 8, 12
        mean = rng.normal(0.16, 0.1, (N, T, H, W, 2)).astype(np.float32)
        std = np.abs(rng.normal(0.001, 0.0005, (N, T, H, W, 2))).astype(np.float32)
        speed = rng.uniform(0.1, 0.3, N).astype(np.float32)
        norm = {"mu": [0.16, 0.0], "sd": [0.1, 0.015], "zref": [0.0013, 0.0007], "speed_mu": 0.17, "speed_sd": 0.07}
        torch.testing.assert_close(feats(mean, std, speed, norm), twh.features(mean, std, speed, norm))
        # std=None must equal the training script's --drop-z ablation (planes 2, 3 zeroed)
        ref = twh.features(mean, std, speed, norm); ref[:, 2:4] = 0.0
        torch.testing.assert_close(feats(mean, None, speed, norm), ref)
        torch.manual_seed(0)
        trained = twh.WidthHead(24)
        with torch.no_grad():
            for p in trained.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        twin = Head(24); twin.load_state_dict(trained.state_dict())
        f = feats(mean, std, speed, norm)
        with torch.no_grad():
            torch.testing.assert_close(twin(f), trained(f))
        h = twin(f)
        self.assertEqual(tuple(h.shape), (N * T, 2, H, W))
        self.assertTrue(bool((h >= 0).all()) and bool((h <= 0.5).all()))

    def test_ext_planes_match_and_feed_a_17_plane_head(self):
        import numpy as np
        import torch
        twh, Head, feats = self._pieces()
        rng = np.random.default_rng(1)
        N, T, H, W, Tin = 2, 3, 8, 12, 5
        mean = rng.normal(0.16, 0.1, (N, T, H, W, 2)).astype(np.float32)
        x = rng.normal(0.16, 0.1, (N, Tin, H, W, 3)).astype(np.float32)
        x[:, :, :2, :, :] = 0.0                      # an unmeasured band, as real PIV has
        speed = rng.uniform(0.1, 0.3, N).astype(np.float32)
        norm = {"mu": [0.16, 0.0], "sd": [0.1, 0.015], "zref": [0.0013, 0.0007], "speed_mu": 0.17, "speed_sd": 0.07,
                "in_std_ref": [0.05, 0.01], "delta_ref": [0.02, 0.005]}
        e_train = twh.extra_planes(x, mean); e_sub = self._extra(x, mean)
        self.assertEqual(e_train.shape, (N, T, 9, H, W))
        np.testing.assert_allclose(e_sub, e_train, rtol=0, atol=1e-6)
        self.assertTrue(np.all(e_train[:, :, 6] <= 1.0) and np.all(e_train[:, :, 6] >= 0.0))
        self.assertTrue(np.all(e_train[:, 0, 7:9] == 0.0))    # no motion before the first predicted frame
        self.assertTrue(np.all(e_train[:, :, 0:2, :2, :] == 0.0))  # unmeasured band -> zero input mean
        f_train = twh.features(mean, None if False else np.zeros_like(mean), speed, norm, e_train)
        f_sub = feats(mean, None, speed, norm, e_sub)
        f_train[:, 2:4] = 0.0
        torch.testing.assert_close(f_sub, f_train)
        self.assertEqual(f_sub.shape[1], 17)
        head = Head(24, 17); trained = twh.WidthHead(24, 17); head.load_state_dict(trained.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(head(f_sub), trained(f_train))

    def test_device_side_features_equal_the_numpy_ones(self):
        """The fused CUDA-graph path builds the head's planes in torch; they must equal the numpy build."""
        import numpy as np
        import torch
        twh, Head, feats = self._pieces()
        rng = np.random.default_rng(2)
        N, T, H, W, Tin = 2, 3, 8, 12, 5
        mean = rng.normal(0.16, 0.1, (N, T, H, W, 2)).astype(np.float32)
        std = np.abs(rng.normal(0.001, 0.0005, (N, T, H, W, 2))).astype(np.float32)
        x = rng.normal(0.16, 0.1, (N, Tin, H, W, 3)).astype(np.float32); x[:, :, :2] = 0.0
        speed = rng.uniform(0.1, 0.3, N).astype(np.float32)
        norm = {"mu": [0.16, 0.0], "sd": [0.1, 0.015], "zref": [0.0013, 0.0007], "speed_mu": 0.17, "speed_sd": 0.07,
                "in_std_ref": [0.05, 0.01], "delta_ref": [0.02, 0.005]}
        t = lambda a: torch.from_numpy(a)
        # base planes, with and without disagreement
        torch.testing.assert_close(self._feats_t(t(mean), t(std), t(speed), norm), feats(mean, std, speed, norm))
        torch.testing.assert_close(self._feats_t(t(mean), None, t(speed), norm), feats(mean, None, speed, norm))
        # ext planes
        e_np = self._extra(x, mean); e_t = self._extra_t(t(x), t(mean))
        torch.testing.assert_close(e_t, t(e_np), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(self._feats_t(t(mean), None, t(speed), norm, e_t), feats(mean, None, speed, norm, e_np), rtol=1e-5, atol=1e-6)
        # population std over members, as the fused graph computes it, equals numpy's default
        y = t(rng.normal(0, 1, (3, N, T, H, W, 2)).astype(np.float32))
        torch.testing.assert_close(y.std(0, unbiased=False), t(y.numpy().std(0)), rtol=1e-5, atol=1e-6)


class QuantileHead(unittest.TestCase):
    """sps P3's head: quantiles must not cross, and the width picked must be the one
    the scorer would pick. Both are load-bearing -- crossing quantiles make F(h)
    non-monotone and the argmax meaningless."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "tqh", str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train_quantile_head.py"))
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_quantiles_never_cross(self):
        import torch
        torch.manual_seed(0)
        h = self.m.QuantileHead(8, 5, 7)
        # perturb away from the zero-init so the test sees a real forward, not the constant
        with torch.no_grad():
            h.net[-1].weight.normal_(0, 0.5)
        out = h(torch.randn(3, 5, 8, 8))
        self.assertEqual(out.shape, (3, 2, 7, 8, 8))
        d = out[:, :, 1:] - out[:, :, :-1]
        self.assertGreaterEqual(float(d.min()), 0.0, "quantiles crossed")

    def test_pick_returns_the_scorer_s_argmax(self):
        import numpy as np
        sigma = self.m.SIGMA_GLOBAL
        taus = np.array([0.5, 0.8, 0.95])
        h = np.array([[[0.002, 0.010, 0.400]]])          # (1,1,Q)
        got = self.m.pick(h, taus)
        want = h[0, 0][int(np.argmax(taus * np.exp(-2.0 * h[0, 0] / sigma)))]
        self.assertAlmostEqual(float(got[0, 0]), float(want), places=12)
        # a hopeless third quantile must never be chosen: exp(-2*0.4/sigma) underflows the gain
        self.assertNotAlmostEqual(float(got[0, 0]), 0.400, places=6)

    def test_pinball_is_minimised_at_the_true_quantile(self):
        import torch
        torch.manual_seed(0)
        target = torch.rand(1, 2, 1, 64) * 0.02            # (B,2,H,W)
        scored = torch.ones_like(target)
        taus = torch.tensor([0.5])
        q = torch.quantile(target.reshape(-1), 0.5)
        at_truth = self.m.pinball(q.expand(1, 2, 1, 1, 64), target, taus, scored)
        for off in (-0.004, +0.004):
            self.assertGreater(float(self.m.pinball((q + off).expand(1, 2, 1, 1, 64), target, taus, scored)),
                               float(at_truth), f"pinball not minimised at the median (offset {off})")


class PerAxisBoundary(unittest.TestCase):
    """--boundary-hw exposes the sampler's per-axis padding, which the CLI never has.
    The constraint it enforces is the sampler's own (flower_standalone.py:168): when the
    axes differ, one must be periodic. Getting that wrong would fail deep inside a forward
    pass instead of at argument parsing."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "trn", str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train.py"))
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def _a(self, hw=None, boundary="periodic"):
        import types
        return types.SimpleNamespace(boundary_hw=hw, boundary=boundary)

    def test_single_value_is_unchanged(self):
        self.assertEqual(self.m._boundary_types(self._a(None, "periodic")), ("PERIODIC",))
        self.assertEqual(self.m._boundary_types(self._a(None, "border")), ("BORDER",))
        self.assertEqual(self.m._boundary_types(self._a(None, "zeros")), ("ZEROS",))

    def test_mixed_with_one_periodic_is_accepted(self):
        self.assertEqual(self.m._boundary_types(self._a(["periodic", "border"])), ("PERIODIC", "BORDER"))
        self.assertEqual(self.m._boundary_types(self._a(["border", "periodic"])), ("BORDER", "PERIODIC"))

    def test_equal_non_periodic_axes_are_accepted(self):
        self.assertEqual(self.m._boundary_types(self._a(["border", "border"])), ("BORDER", "BORDER"))

    def test_mixed_without_periodic_is_refused(self):
        with self.assertRaises(SystemExit):
            self.m._boundary_types(self._a(["border", "zeros"]))

    def test_the_model_accepts_what_the_flag_produces(self):
        from pde_competition.model import FlowerConfig, FlowerS2S
        cfg = FlowerConfig(lifting_dim=32, n_levels=1, c_in=3, c_out=3,
                           boundary_condition_types=("PERIODIC", "BORDER"))
        FlowerS2S(cfg)          # construction runs the same validation the sampler does


class WidthScale(unittest.TestCase):
    """The head's half-widths get one multiplier for the shipped ensemble size. The default
    must be bit-exact against every zip built before it existed, and the two inference paths
    (numpy and the captured-graph torch twin) must apply it identically -- they are scored by
    the same grader and a divergence would be invisible until the board."""

    def test_default_is_identity_in_the_module_source(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / "submission" / "submission_flower.py").read_text()
        self.assertIn("_HEAD_SCALE = 1.0", src)
        # both paths must multiply, or the graph and eager results diverge. The torch path's
        # forward acquired an fp16 branch on 2026-09-22 (FP16H), so the invariant is checked on
        # the assignment rather than on one literal spelling of the call: whichever branch runs,
        # its result is what gets scaled.
        i = src.index("def _head_half_torch(")
        line = [l for l in src[i:].splitlines() if l.strip().startswith("hh = ")][0]
        self.assertTrue(line.split("#")[0].rstrip().endswith("* _HEAD_SCALE"),
                        f"torch path must scale the head output, got: {line}")
        self.assertIn("head(f)", line)          # the float32 branch
        self.assertIn("head(f.half()).float()", line)   # the fp16 branch, back to float32 first
        self.assertIn("hh.numpy() * _HEAD_SCALE", src)

    def test_build_writes_the_scale_into_bounds(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_submission.py").read_text()
        self.assertIn('"scale": float(scale)', src)
        self.assertIn("--width-scale", src)

    def test_the_sane_band_is_enforced(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / "submission" / "submission_flower.py").read_text()
        self.assertIn("0.2 <= _HEAD_SCALE <= 3.0", src)

    def test_scaling_a_width_moves_the_objective_the_way_the_scorer_says(self):
        """A larger width must buy coverage and cost exp(-2h/sigma); the product is what
        fit_bounds.objective_map computes, and the scale must move it through that product."""
        import numpy as np
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
        from fit_bounds import objective_map, SIGMA_GLOBAL
        rng = np.random.default_rng(0)
        pred = np.zeros((4, 3, 8, 8), np.float32)
        targ = (rng.normal(0, 0.008, pred.shape)).astype(np.float32)
        targ[targ == 0] = 1e-6
        h = np.full_like(pred, 0.009)
        base = objective_map(pred, targ, h)[0]
        wide = objective_map(pred, targ, h * 2.0)[0]
        narrow = objective_map(pred, targ, h * 0.5)[0]
        self.assertNotAlmostEqual(base, wide, places=6)
        self.assertNotAlmostEqual(base, narrow, places=6)
        # doubling the width can only raise coverage, never lower it
        self.assertGreaterEqual(objective_map(pred, targ, h * 2.0)[1], objective_map(pred, targ, h)[1])


class HeadLossVariants(unittest.TestCase):
    """Two new head objectives. The scorer weights each WINDOW by its accuracy factor and the
    head never has; and the smoothed forward is biased toward near-misses the scorer pays
    nothing for. Both must leave the default path bit-identical."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "twh", str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train_width_head.py"))
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_default_path_is_unchanged(self):
        import torch
        torch.manual_seed(0)
        h = torch.rand(4, 2, 6, 6) * 0.01
        r = torch.rand(4, 2, 6, 6) * 0.02
        s = torch.ones_like(h)
        a = self.m.surrogate_loss(h, r, s, 0.0005)
        b = self.m.surrogate_loss(h, r, s, 0.0005, w=None, straight_through=False)
        self.assertAlmostEqual(float(a), float(b), places=12)

    def test_straight_through_forward_is_the_exact_indicator(self):
        import torch
        h = torch.full((1, 2, 2, 2), 0.010, requires_grad=True)
        r = torch.full((1, 2, 2, 2), 0.011)          # every element OUTSIDE -> exact reward 0
        s = torch.ones_like(r)
        st = self.m.surrogate_loss(h, r, s, 0.0005, straight_through=True)
        self.assertAlmostEqual(float(st), 0.0, places=9)
        st.backward()                                 # and it must still carry gradient
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_scorer_weights_are_normalised_and_reward_accuracy(self):
        import numpy as np
        rng = np.random.default_rng(0)
        y = rng.normal(0, 0.1, (6, 4, 8, 16, 2)).astype(np.float32)
        good = y + rng.normal(0, 0.001, y.shape).astype(np.float32)   # accurate windows
        bad = y + rng.normal(0, 0.05, y.shape).astype(np.float32)     # inaccurate windows
        wg = self.m.scorer_sample_weights(good, y)
        wb = self.m.scorer_sample_weights(bad, y)
        self.assertAlmostEqual(float(wg.mean()), 1.0, places=5)
        self.assertAlmostEqual(float(wb.mean()), 1.0, places=5)
        # the scorer pays MORE for coverage in accurate windows: raw factors must order that way
        mixed = np.concatenate([good[:3], bad[3:]], axis=0)
        w = self.m.scorer_sample_weights(mixed, y)
        self.assertGreater(float(w[:3].mean()), float(w[3:].mean()))

    def test_weighting_changes_the_loss_but_not_its_sign(self):
        import torch
        torch.manual_seed(0)
        h = torch.rand(4, 2, 6, 6) * 0.01
        r = torch.rand(4, 2, 6, 6) * 0.02
        s = torch.ones_like(h)
        w = torch.tensor([0.5, 1.0, 1.5, 1.0]).view(-1, 1, 1, 1)
        plain = float(self.m.surrogate_loss(h, r, s, 0.0005))
        wt = float(self.m.surrogate_loss(h, r, s, 0.0005, w=w))
        self.assertNotAlmostEqual(plain, wt, places=9)
        self.assertLess(wt, 0.0)


class NoUndefinedLocals(unittest.TestCase):
    """Catch the class of bug that killed W197b: a NameError on a local that only fires
    deep inside a training loop, minutes into a GPU job. Our unit tests exercised
    surrogate_loss() directly and never the loop that indexes into its weights, so
    `wtr[j]` where the loop defines `idx` sailed through 102 green tests and cost a
    28-minute job. A whole-function name check is cheap and would have caught it."""

    SCRIPTS = ["train_width_head.py", "train_quantile_head.py", "train.py",
               "width_recalibration.py", "tke_decomposition.py", "import_cost.py",
               "disagreement_scale.py", "score_twin.py", "build_submission.py"]

    def test_every_name_used_in_a_function_is_bound_somewhere(self):
        import ast, builtins
        root = pathlib.Path(__file__).resolve().parents[1] / "scripts"
        bad = []
        for name in self.SCRIPTS:
            p = root / name
            if not p.exists():
                continue
            tree = ast.parse(p.read_text())
            # module-level dunders are bound in every function scope by the import system
            module_names = set(dir(builtins)) | {
                "__file__", "__name__", "__doc__", "__package__", "__spec__",
                "__loader__", "__builtins__", "__debug__"}
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for al in node.names:
                        module_names.add((al.asname or al.name).split(".")[0])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    module_names.add(node.name)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        for n in ast.walk(t):
                            if isinstance(n, ast.Name):
                                module_names.add(n.id)
            for fn in [n for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                bound = set(module_names)
                for n in ast.walk(fn):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                        bound.add(n.id)
                    elif isinstance(n, ast.arg):
                        bound.add(n.arg)
                    elif isinstance(n, (ast.Import, ast.ImportFrom)):
                        for al in n.names:
                            bound.add((al.asname or al.name).split(".")[0])
                    elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        bound.add(n.name)
                    elif isinstance(n, ast.ExceptHandler) and n.name:
                        bound.add(n.name)
                    elif isinstance(n, (ast.comprehension,)):
                        for nn in ast.walk(n.target):
                            if isinstance(nn, ast.Name):
                                bound.add(nn.id)
                for n in ast.walk(fn):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound:
                        bad.append(f"{name}:{n.lineno} {fn.name}() uses undefined '{n.id}'")
        self.assertEqual(bad, [], "undefined locals:\n  " + "\n  ".join(bad))


class GraphLadder(unittest.TestCase):
    """The grader calls predict with n=5 (submission 13's Codabench log). Under the old
    powers-of-two buckets that padded to an 8-window graph -- 60% more model and head work
    than the call needed, on every one of its 17 calls. These assert the ladder covers 5
    exactly, never pads DOWN (which would drop windows), and still caps at the batch."""

    def setUp(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / "submission" / "submission_flower.py").read_text()
        self.src = src
        self.LAD = (1, 5, 8, 16)         # exact at the grader's n=5, and 8 keeps the mid-size net

    def bucket(self, size, batch=16):
        if size >= batch:
            return batch
        for b in self.LAD:
            if b >= size and b <= batch:
                return b
        return batch

    def test_the_graders_call_size_is_exact(self):
        self.assertEqual(self.bucket(5), 5, "n=5 is the grader's call size and must not pad")

    def test_the_ladder_is_no_larger_than_the_powers_of_two_it_replaced(self):
        """E1 OOM-killed at k=12 on a 2080Ti with 8.58 GiB in graph pools. Every captured
        shape holds its own buffers, so shape COUNT is a memory budget."""
        self.assertLessEqual(len(self.LAD), 5)

    def test_a_mid_size_call_does_not_fall_all_the_way_to_16(self):
        """With {1,5,16} an 8-window call paid a 16-window graph: 2.83 ms/window at k=3
        against 1.75 at n=5. The grader uses n=5 so it never bit, but removing the
        mid-size net was an unforced regression."""
        self.assertEqual(self.bucket(8), 8)
        self.assertLessEqual(self.bucket(7), 8)

    def test_the_dense_ladder_is_still_reachable(self):
        """The dense ladder measured null (W200) and cost graph memory (E1 OOM at k=12 on a
        2080Ti), so the DEFAULT is powers of two again -- but the knob must survive so the
        question can be re-asked without re-deriving it."""
        self.assertIn("RPDE_GRAPH_LADDER", self.src)
        dense = (1, 2, 3, 4, 5, 6, 8, 12, 16)
        self.assertEqual(min(b for b in dense if b >= 5), 5)

    def test_a_bucket_never_drops_windows(self):
        for n in range(1, 17):
            self.assertGreaterEqual(self.bucket(n), n, f"bucket({n}) padded DOWN, which loses windows")

    def test_buckets_are_capped_at_the_batch(self):
        for n in range(1, 40):
            self.assertLessEqual(self.bucket(n), 16)

    def test_the_ladder_is_the_shipped_default_and_is_sorted(self):
        self.assertIn('"1,5,8,16"', self.src)
        self.assertEqual(list(self.LAD), sorted(self.LAD))

    def test_import_warmup_captures_every_ladder_shape(self):
        # a shape captured lazily would capture INSIDE the timed region, ~50-100 ms
        self.assertIn("for b in sorted({b for b in _GRAPH_LADDER if b <= _BATCH} | {_BATCH})", self.src)


class MemberConfigGuard(unittest.TestCase):
    """submission_flower.py builds EVERY ensemble member with the shipped cfg and loads only
    its state_dict, so a member that disagrees on a semantic flag is run under the wrong
    semantics with no warning. W202 arm C averaged three predict_delta members with three
    ordinary ones -- every member individually healthy at val_rel_l2 0.0811 -- and the
    ensemble scored rel_l2 87.88 against 96.15 and sps 16.87 against 50.90."""

    def test_the_guard_lists_the_flags_that_change_meaning(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_submission.py").read_text()
        for flag in ("predict_delta", "mask_channel", "rans", "temporal_lift", "c_in", "c_out",
                     "lifting_dim", "n_levels", "rollout_frames", "boundary_condition_types"):
            self.assertIn(f'"{flag}"', src, f"{flag} changes output semantics but is not guarded")

    def test_the_guard_runs_before_the_members_are_written(self):
        src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_submission.py").read_text()
        i_guard = src.index("_assert_members_agree(ckpt, members)")
        i_write = src.index("_payload_members(members)", i_guard - 400)
        self.assertLess(i_guard, i_write, "the guard must refuse BEFORE anything is staged")

    def test_a_disagreeing_member_is_refused(self):
        import torch, tempfile, importlib.util
        spec = importlib.util.spec_from_file_location(
            "bs", str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_submission.py"))
        bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)
        with tempfile.TemporaryDirectory() as d:
            a = pathlib.Path(d) / "a.pth"; b = pathlib.Path(d) / "b.pth"
            torch.save({"config": {"predict_delta": False, "c_in": 3}}, a)
            torch.save({"config": {"predict_delta": True, "c_in": 3}}, b)
            bs._assert_members_agree(a, [a])                      # agreeing: fine
            with self.assertRaises(SystemExit):
                bs._assert_members_agree(a, [a, b])               # disagreeing: refused


# 2026-09-23: `prediction` became `base` and `mean` became `ybase`/`base` at these sites when the
# decoupled-centre path landed. `base` is the CENTRE field when `bounds.json centre_bounds` is set and is
# the prediction itself otherwise -- so with the key absent these lines are byte-for-byte what they were.
# These assertions pin source text, which is why a correct change breaks them; the behavioural guarantee
# they stand in for is the off-path identity, which the exec-based tests in this file cover.
class AsymmetricWidthHead(unittest.TestCase):
    """directions.md H1/H2. aggregate_sps pays 1{lower <= target <= upper} * exp(-(upper-lower)/sigma):
    it buys WIDTH and COVERAGE and says nothing about the interval being centred. --outputs 2 lets
    the head emit the two offsets separately, --outputs 3 adds a signed correction to the prediction.
    --outputs 1 must remain the head that ships, bit for bit, and an asymmetric head must never
    reach a zip through the symmetric plumbing."""

    def _twh(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("twh_asym", str(REPO / "scripts" / "train_width_head.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m

    def _head_region(self, **glb):
        """submission_flower's head functions exec'd with stub module globals, as the
        existing WidthHeadInSubmission tests do -- importing the module would load a model."""
        import numpy as np
        import torch
        src = (REPO / "submission" / "submission_flower.py").read_text()
        ns = {"np": np, "torch": torch}
        ns.update(glb)
        for name in ("class _WidthHead(", "def _head_extra(", "def _head_features(",
                     "def _head_raw(", "def _head_half(", "def _head_offsets("):
            start = src.index(name); end = src.index("\n\n\n", start)
            exec(src[start:end], ns)
        return ns

    def test_the_default_layout_is_the_shipped_head(self):
        import torch
        twh = self._twh()
        torch.manual_seed(0); a = twh.WidthHead(8, twh.N_FEAT)
        torch.manual_seed(0); b = twh.WidthHead(8, twh.N_FEAT, n_out=2)
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))
        for k in a.state_dict():
            torch.testing.assert_close(a.state_dict()[k], b.state_dict()[k], rtol=0, atol=0)
        self.assertFalse(any("delta_scale" in k for k in a.state_dict()))   # no buffer on the shipped layout
        h = a(torch.randn(3, twh.N_FEAT, 32, 64))
        self.assertEqual(tuple(h.shape), (3, 2, 32, 64))
        self.assertAlmostEqual(float(h.mean()), 0.009, places=3)
        self.assertEqual(twh.N_OUT[1], 2)
        # the shipped submission path is untouched: one half-width, mirrored about the
        # prediction. Since H1 was wired the mirror is `up = half` -- the SAME array, so a
        # symmetric head still costs one allocation and produces the identical interval.
        src = (REPO / "submission" / "submission_flower.py").read_text()
        self.assertIn('_HEAD_CKPT.get("n_out", 2)', src)
        self.assertIn("lower = base - half", src)
        self.assertIn("upper = base + up", src)
        self.assertIn("up = half", src)
        self.assertIn("_HEAD_NOUT = 2", src)
        # and a head trained before any of this loads into the twin unchanged
        ck = REPO / "outputs" / "width_head" / "head_ext_tau05_w48.pth"
        if ck.exists():
            d = torch.load(ck, map_location="cpu", weights_only=False)
            self.assertEqual(int(d.get("n_out", 2)), 2)
            ns = self._head_region()
            ns["_WidthHead"](int(d.get("width", 24)), int(d.get("n_feat", 8))).load_state_dict(d["state_dict"])

    def test_the_asymmetric_objective_reduces_exactly_to_the_symmetric_one(self):
        """The control arm's 0.6572 is an objective_map number; an --outputs 2 number is only
        comparable to it if the two agree bit for bit at lo == up."""
        import numpy as np
        twh = self._twh()
        sys.path.insert(0, str(REPO / "scripts"))
        from fit_bounds import objective_map
        rng = np.random.default_rng(11)
        pred = rng.normal(0.16, 0.1, (5, 4, 8, 12)).astype(np.float32)
        targ = (pred + rng.normal(0, 0.01, pred.shape)).astype(np.float32)
        targ[rng.random(targ.shape) < 0.1] = 0.0                       # unscored elements
        for h in (0.009, np.abs(rng.normal(0.009, 0.003, pred.shape))):
            o_sym, c_sym = objective_map(pred, targ, h)
            o_asym, c_asym = twh.objective_map_asym(pred, targ, h, h)
            self.assertEqual(o_sym, o_asym)
            self.assertEqual(c_sym, c_asym)

    def test_a_lopsided_interval_beats_a_centred_one_on_skewed_residuals(self):
        """H1's premise, on synthetic data: when the residual is skewed the same total width
        covers more if it is not centred, and when it is symmetric there is nothing to win."""
        import numpy as np
        twh = self._twh()
        rng = np.random.default_rng(3)
        pred = np.zeros((120, 1, 6, 6), np.float32)
        g = np.linspace(0.0, 0.04, 41)

        def best(targ):
            sym = max(twh.objective_map_asym(pred, targ, h, h)[0] for h in g)
            asym = max(twh.objective_map_asym(pred, targ, lo, up)[0] for lo in g for up in g)
            return sym, asym

        skew = (rng.gamma(2.0, 0.004, pred.shape) - 0.002).astype(np.float32)
        skew[skew == 0] = 1e-7
        s_sym, s_asym = best(skew)
        self.assertGreater(s_asym, s_sym * 1.02)                       # the mechanism is real
        sym_r = rng.normal(0, 0.006, pred.shape).astype(np.float32)
        sym_r[sym_r == 0] = 1e-7
        b_sym, b_asym = best(sym_r)
        self.assertGreaterEqual(b_asym, b_sym)                         # symmetric is a special case
        self.assertLess(b_asym, b_sym * 1.02)                          # and nothing is left on the table

    def test_the_asymmetric_surrogate_trains_and_its_exact_forward_is_the_scorer(self):
        import numpy as np
        import torch
        twh = self._twh()
        torch.manual_seed(1)
        head = twh.WidthHead(8, twh.N_FEAT, n_out=4)
        f = torch.randn(3, twh.N_FEAT, 16, 16)
        o = head(f)
        self.assertEqual(tuple(o.shape), (3, 4, 16, 16))
        self.assertTrue(bool((o >= 0).all()) and bool((o <= 0.5).all()))
        self.assertAlmostEqual(float(o.mean().detach()), 0.009, places=4)   # starts as the symmetric head
        loss = twh.surrogate_loss_asym(o[:, :2], o[:, 2:], torch.randn(3, 2, 16, 16) * 0.01,
                                       torch.ones(3, 2, 16, 16), 0.0005)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None))
        # straight-through: the forward VALUE is the scorer's exact reward, not the smoothed one
        lo = torch.full((2, 2, 4, 4), 0.006); up = torch.full((2, 2, 4, 4), 0.012)
        r = torch.randn(2, 2, 4, 4) * 0.01
        got = float(twh.surrogate_loss_asym(lo, up, r, torch.ones_like(r), 1e-6, straight_through=True))
        want = -float((((r >= -lo) & (r <= up)).double()
                       * float(np.exp(-0.018 / twh.SIGMA_GLOBAL))).mean())
        self.assertAlmostEqual(got, want, places=6)
        # a one-sided miss is not covered: r above `up` scores zero however wide `lo` is
        r1 = torch.full((1, 2, 2, 2), 0.02)
        self.assertAlmostEqual(
            float(twh.surrogate_loss_asym(torch.full_like(r1, 0.4), torch.full_like(r1, 0.012),
                                          r1, torch.ones_like(r1), 1e-6, straight_through=True)), 0.0, places=9)

    def test_the_prediction_correction_starts_at_zero_and_is_bounded(self):
        import torch
        twh = self._twh()
        ds = (0.0097, 0.0016)
        head = twh.WidthHead(8, twh.N_FEAT, n_out=6, delta_scale=ds)
        f = torch.randn(3, twh.N_FEAT, 16, 16)
        o = head(f)
        self.assertEqual(tuple(o.shape), (3, 6, 16, 16))
        self.assertTrue(bool((o[:, :2] == 0).all()))                   # H2 at init predicts the raw field
        self.assertAlmostEqual(float(o[:, 2:].mean()), 0.009, places=4)
        self.assertIn("delta_scale", head.state_dict())                # travels with the weights
        with torch.no_grad():
            for p in head.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        o = head(f)
        self.assertTrue(float(o[:, :2].abs().max()) > 0.0)             # the correction is live
        for c, s in enumerate(ds):
            self.assertLessEqual(float(o[:, c].abs().max()), s + 1e-7)  # and cannot run away
        ns = self._head_region()
        twin = ns["_WidthHead"](8, twh.N_FEAT, 6)
        twin.load_state_dict(head.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(twin(f), head(f))

    def test_head_offsets_reproduces_head_half_and_the_symmetric_path_refuses_an_asym_head(self):
        import numpy as np
        import torch
        twh = self._twh()
        rng = np.random.default_rng(5)
        N, T, H, W, Tin, C = 2, 3, 8, 12, 5, 3
        pred = rng.normal(0.16, 0.1, (N, T, H, W, C)).astype(np.float32)
        x = rng.normal(0.16, 0.1, (N, Tin, H, W, C)).astype(np.float32); x[:, :, :2] = 0.0
        norm = {"mu": [0.16, 0.0], "sd": [0.1, 0.015], "zref": [0.0013, 0.0007],
                "speed_mu": 0.17, "speed_sd": 0.07}
        halfw = np.array([0.009, 0.004, 0.02], np.float32)
        ns = self._head_region(_HEAD_USE_Z=False, _HEAD_PLANES="base", _HEAD_NORM=norm,
                               _HEAD_SCALE=1.0, _HEAD_NOUT=2, _HALF_WIDTH=halfw)

        def wire(n_out):
            torch.manual_seed(2)
            trained = twh.WidthHead(16, twh.N_FEAT, n_out=n_out)
            with torch.no_grad():
                for p in trained.parameters():
                    p.add_(torch.randn_like(p) * 0.05)
            twin = ns["_WidthHead"](16, twh.N_FEAT, n_out)
            twin.load_state_dict(trained.state_dict())
            ns["_HEAD_NOUT"] = n_out
            ns["_get_head"] = lambda device, _t=twin.eval(): _t

        wire(2)
        half = ns["_head_half"](pred, None, x, "cpu")
        d, lo, up = ns["_head_offsets"](pred, None, x, "cpu")
        np.testing.assert_array_equal(d, np.zeros_like(pred))
        np.testing.assert_array_equal(lo, half)
        np.testing.assert_array_equal(up, half)                        # symmetric -> the shipped interval

        wire(4)
        with self.assertRaises(NotImplementedError):
            ns["_head_half"](pred, None, x, "cpu")                     # cannot ship half an interval
        d, lo, up = ns["_head_offsets"](pred, None, x, "cpu")
        self.assertEqual(lo.shape, pred.shape)
        np.testing.assert_array_equal(d, np.zeros_like(pred))          # n_out=4 moves no prediction
        self.assertTrue(np.any(lo[..., :2] != up[..., :2]))            # the interval really is lopsided
        self.assertTrue(np.all(lo >= 0) and np.all(up >= 0))
        np.testing.assert_array_equal(lo[..., 2], np.full(pred.shape[:-1], halfw[2], np.float32))
        np.testing.assert_array_equal(up[..., 2], np.full(pred.shape[:-1], halfw[2], np.float32))

    def test_the_build_ships_lower_upper_and_refuses_the_prediction_correction(self):
        """n_out 2 and 4 both have a consumer in predict(); 6 (H2) does not and is closed."""
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "bounds.json").write_text(json.dumps({"half_width": [0.009, 0.004, 0.02]}))
            sym, asym, delta, old_style = d / "sym.pth", d / "asym.pth", d / "delta.pth", d / "old.pth"
            torch.save({"state_dict": {}, "n_out": 2}, sym)
            torch.save({"state_dict": {}, "n_out": 4}, asym)
            torch.save({"state_dict": {}, "n_out": 6}, delta)
            torch.save({"state_dict": {}}, old_style)
            self.assertEqual(build_submission.head_n_out(sym), 2)
            self.assertEqual(build_submission.head_n_out(asym), 4)
            self.assertEqual(build_submission.head_n_out(delta), 6)
            self.assertEqual(build_submission.head_n_out(old_style), 2)     # pre-H1 heads are symmetric
            keep = build_submission.STAGING
            build_submission.STAGING = d
            try:
                with self.assertRaises(SystemExit) as cm:
                    build_submission._payload_width_head(delta, False)
                self.assertIn("n_out=6", str(cm.exception))
                build_submission._payload_width_head(asym, False)            # H1 ships
                build_submission._payload_width_head(sym, False)             # the default still ships
            finally:
                build_submission.STAGING = keep
            b = json.loads((d / "bounds.json").read_text())
            self.assertEqual(b["width_head"]["file"], "width_head.pth")
            self.assertEqual(b["width_head"]["scale"], 1.0)

    def test_predict_ships_two_offsets_when_the_head_is_asymmetric(self):
        """The wiring H1 was blocked on. A 4-plane head output must reach lower and upper as
        two DIFFERENT offsets; a 2-plane one must still produce the mirrored interval, element
        for element, so every zip built before 2026-09-21 is reproduced exactly."""
        import numpy as np
        src = (REPO / "submission" / "submission_flower.py").read_text()
        i = src.index("def predict(")
        body = src[i:src.index("\n\n\n", i)]
        rng = np.random.default_rng(3)
        N, T, H, W = 2, 3, 4, 5
        pred = rng.normal(0.16, 0.1, (N, T, H, W, 3)).astype(np.float32)
        halfw = np.array([0.009, 0.004, 0.02], np.float32)
        x = np.zeros((N, 20, H, W, 3), np.float32)

        def run(**glb):
            ns = {"np": np, "_FAST": False, "_fast_out": {"off": True}, "_HEAD": True,
                  "C": 3, "_HALF_WIDTH": halfw, "_HEAD_NOUT": 2,
                  "_state": {"graph_half": None, "device": "cpu"},
                  # VINF wrapped predict()'s returns in _inflate_fluct, which is defined above the
                  # slice this test execs. Bind the REAL function, compiled from the same file, so the
                  # test still exercises shipped code rather than a stub that could mask a change.
                  "_inflate_fluct": _real_inflate(np),
                  # ANCH (2026-09-24) nested _anchor_mean around it on the same return line; the real
                  # function again, at its default A = 0.0, where it returns its argument itself.
                  "_anchor_mean": _real_anchor(np),
                  "_predict_fields": lambda a: (pred, None)}
            ns.update(glb)
            exec(compile(body, "submission_flower.py", "exec"), ns)
            return ns["predict"](x)

        lo2 = rng.uniform(0.001, 0.02, (N, T, H, W, 2)).astype(np.float32)
        up2 = (lo2 * 2.0 + 0.001).astype(np.float32)
        exp_lo = np.empty_like(pred); exp_lo[..., :2] = lo2; exp_lo[..., 2] = halfw[2]
        exp_up = np.empty_like(pred); exp_up[..., :2] = up2; exp_up[..., 2] = halfw[2]

        # symmetric, from the fused path: unchanged, and the two sides are the same array
        o = run(_state={"graph_half": lo2, "device": "cpu"})
        np.testing.assert_array_equal(o["prediction"], pred)
        np.testing.assert_array_equal(o["lower"], pred - exp_lo)
        np.testing.assert_array_equal(o["upper"], pred + exp_lo)

        # asymmetric, from the fused path: planes 0:2 low, 2:4 high
        gh = np.concatenate([lo2, up2], axis=-1)
        o = run(_state={"graph_half": gh, "device": "cpu"}, _HEAD_NOUT=4)
        np.testing.assert_array_equal(o["lower"], pred - exp_lo)
        np.testing.assert_array_equal(o["upper"], pred + exp_up)
        self.assertTrue(np.all((o["upper"] - pred)[..., :2] > (pred - o["lower"])[..., :2]))

        # asymmetric on the numpy reference path (no graph): _head_offsets is consulted
        seen = {}

        def offsets(prediction, dis, arr, device):
            seen["called"] = True
            return np.zeros_like(pred), exp_lo, exp_up
        o = run(_HEAD_NOUT=4, _head_offsets=offsets)
        self.assertTrue(seen.get("called"))
        np.testing.assert_array_equal(o["lower"], pred - exp_lo)
        np.testing.assert_array_equal(o["upper"], pred + exp_up)


class DropTrainRe(unittest.TestCase):
    """directions.md S1c. --drop-train-re thins TRAINING without touching validation, so arms
    trained on different amounts of data are scored on exactly the same held-out rows. S1b's
    flaw was the opposite: it scored all-data snapshots on rows those models had trained on,
    which favours later epochs because the in-sample advantage grows with epoch."""

    @classmethod
    def setUpClass(cls):
        cls.src = (REPO / "scripts" / "train.py").read_text()

    def test_the_default_is_empty_so_every_existing_run_is_unchanged(self):
        self.assertIn('p.add_argument("--drop-train-re", type=int, nargs="*", default=[],', self.src)
        self.assertIn("if a.drop_train_re:", self.src)

    def test_a_row_that_is_already_held_out_is_refused_not_silently_dropped_twice(self):
        i = self.src.index("if a.drop_train_re:")
        body = self.src[i:self.src.index("print(f\"{len(runs)} runs", i)]
        self.assertIn("overlap = drop & set(a.holdout_re)", body)
        self.assertIn("raise SystemExit", body)
        self.assertIn("val_runs", self.src[:i])          # val is computed BEFORE the drop

    def test_it_filters_training_only_and_the_val_set_is_untouched(self):
        """Exec the block itself against a stub run list, so the test tracks the source."""
        import textwrap
        i = self.src.index("    if a.drop_train_re:")
        j = self.src.index('    print(f"{len(runs)} runs', i)
        block = textwrap.dedent(self.src[i:j])
        rows = [3750, 5025, 6300, 8850, 13950, 26700]
        runs = [types.SimpleNamespace(re_nominal=r, key=f"{r}_0") for r in rows]
        val_runs = [r for r in runs if r.re_nominal in (13950, 26700)]
        train_runs = [r for r in runs if r.re_nominal not in (13950, 26700)]
        ns = {"a": types.SimpleNamespace(drop_train_re=[5025], holdout_re=[13950, 26700]),
              "train_runs": list(train_runs), "val_runs": val_runs,
              "SystemExit": SystemExit}
        exec(compile(block, "train.py", "exec"), ns)
        self.assertEqual([r.re_nominal for r in ns["train_runs"]], [3750, 6300, 8850])
        self.assertEqual([r.re_nominal for r in ns["val_runs"]], [13950, 26700])   # untouched
        # a held-out row cannot also be "dropped": it trains on neither path
        ns2 = dict(ns, a=types.SimpleNamespace(drop_train_re=[13950], holdout_re=[13950, 26700]),
                   train_runs=list(train_runs))
        with self.assertRaises(SystemExit):
            exec(compile(block, "train.py", "exec"), ns2)


class QueueFileMarkers(unittest.TestCase):
    """scripts/ws/launch.sh greps the parcel markers ANCHORED (`grep -cE '^# TIMING'`),
    so a marker written mid-line is INERT and the protection it stands for is not armed.
    On 2026-09-21 H1h carried `# DIRECTION ID: H1 / T7.   # TIMING -- unpacked, clean host
    only.` on one line and launched without its co-schedule refusal; the Mac caught it by
    reading launch.sh rather than by trusting the file. Nothing in the repo would have."""

    MARKERS = ("# TIMING", "# CPU-ONLY")

    def test_every_marker_is_at_the_start_of_its_own_line(self):
        bad = []
        for f in sorted((REPO / "experiments" / "queue").glob("*.sh")):
            text = f.read_text()
            for m in self.MARKERS:
                if m in text and not any(l.startswith(m) for l in text.splitlines()):
                    n = next(i for i, l in enumerate(text.splitlines(), 1) if m in l)
                    bad.append(f"{f.name}:{n} has {m!r} but never at column 0")
        self.assertEqual(bad, [], "launch.sh greps these anchored, so these are inert:\n"
                                  + "\n".join(bad))


class ReWeightTheBoardLikeRows(unittest.TestCase):
    """directions.md H3. --min-train-re RESTRICTED the head to the high rows and lost
    (0.6535 against 0.6572). --re-weight keeps every window and tilts instead: row k gets
    (k / max training row) ** P, normalised to mean 1. P = 0 must be the shipped head."""

    @classmethod
    def setUpClass(cls):
        cls.src = (REPO / "scripts" / "train_width_head.py").read_text()

    def test_the_default_is_off_and_the_shipped_path_never_enters_the_branch(self):
        self.assertIn('p.add_argument("--re-weight", type=float, default=0.0,', self.src)
        self.assertIn("if a.re_weight > 0:", self.src)          # P = 0 skips the block entirely

    def _weights(self, power, windows, frames=2, scorer=None):
        """Exec the --re-weight block itself, so the test cannot drift from the source."""
        import textwrap
        import numpy as np
        import torch
        i = self.src.index("    if a.re_weight > 0:")
        j = self.src.index("assert wtr.shape[0] == ftr.shape[0]", i)
        block = textwrap.dedent(self.src[i:j]).replace("if a.re_weight > 0:", "if True:")
        rows = {k: (np.zeros((w, frames, 1, 1, 2), np.float32),) for k, w in windows.items()}
        ns = {"np": np, "torch": torch, "rows": rows, "train_keys": sorted(windows),
              "a": types.SimpleNamespace(re_weight=power), "wtr": scorer,
              "ftr": np.zeros((sum(w * frames for w in windows.values()),), np.float32)}
        exec(compile(block, "train_width_head.py", "exec"), ns)
        return ns["wtr"]

    def test_the_weights_are_mean_one_monotone_in_re_and_per_frame(self):
        import torch
        windows = {3750: 3, 13050: 2, 25425: 4}
        for power in (1.0, 2.0, 4.0):
            w = self._weights(power, windows)
            self.assertEqual(w.shape, (2 * (3 + 2 + 4), 1, 1, 1))   # one entry per FRAME
            self.assertAlmostEqual(float(w.mean()), 1.0, places=5)  # tilts without rescaling the loss
            flat = w.view(-1)
            lo, mid, hi = flat[0], flat[6], flat[-1]                # first row, second row, last row
            self.assertLess(float(lo), float(mid))
            self.assertLess(float(mid), float(hi))
            self.assertAlmostEqual(float(hi) / float(lo),
                                   (25425 / 3750) ** power, places=3)
            # every frame of one window carries that window's factor
            self.assertTrue(bool(torch.all(flat[:6] == flat[0])))
        # a bigger power concentrates harder, which is what makes P=4 the turn-over probe
        self.assertGreater(float(self._weights(4.0, windows).max()),
                           float(self._weights(1.0, windows).max()))

    def test_it_composes_with_scorer_weight_and_stays_mean_one(self):
        import torch
        windows = {3750: 2, 25425: 2}
        scorer = torch.linspace(0.5, 1.5, 8).view(-1, 1, 1, 1)
        w = self._weights(2.0, windows, scorer=scorer)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)
        self.assertFalse(bool(torch.allclose(w, scorer)))           # the row tilt really applied


# ---------------------------------------------------------------------------
# T6 (alternatives to vmap) and E4 (per-member width heads), 2026-09-21.
# Both are experiments behind default-off env vars. The first test in each
# class is the one that matters: THE DEFAULT PATH IS UNCHANGED.
# ---------------------------------------------------------------------------
class T6EnsembleModes(unittest.TestCase):
    """RPDE_ENSEMBLE_MODE swaps how the M members are combined. Unset, the module
    must take exactly the vmap path that shipped in every submission to date."""

    SRC = None

    def setUp(self):
        if T6EnsembleModes.SRC is None:
            T6EnsembleModes.SRC = (REPO / "submission" / "submission_flower.py").read_text()
        self.src = T6EnsembleModes.SRC

    def test_default_is_vmap_and_the_alternatives_are_skipped_entirely(self):
        self.assertIn('os.environ.get("RPDE_ENSEMBLE_MODE", "vmap")', self.src)
        # the alternative branch must be guarded on mode != "vmap", so an unset
        # env var cannot even reach _build_alt_stacked
        i = self.src.index("def _build_stacked(")
        body = self.src[i:self.src.index("\n\n\n", i)]
        self.assertIn('if mode != "vmap":', body)
        self.assertLess(body.index('if mode != "vmap":'), body.index("_build_alt_stacked"))

    def test_an_unknown_mode_is_refused_not_silently_ignored(self):
        i = self.src.index("def _build_alt_stacked(")
        body = self.src[i:self.src.index("\n\n\n", i)]
        self.assertIn("raise ValueError", body)

    def test_the_ceiling_probe_is_labelled_as_not_shippable(self):
        i = self.src.index("def _build_alt_stacked(")
        body = self.src[i:self.src.index("\n\n\n", i)]
        self.assertIn("Never ship it.", body)

    def _ns(self):
        """The grouped-ensemble machinery, exec'd out of the module source (importing
        submission_flower needs model.pth/bounds.json beside it)."""
        import torch  # noqa: F401
        from pde_competition._vendor.flower_standalone import Flower  # noqa: F401
        ns = {"torch": torch, "Flower": Flower, "sys": sys}
        for name in ("class _FlowerS2S(", "class _GroupedStacked(", "def _getattr_path(",
                     "def _setattr_path(", "def _build_alt_stacked("):
            s = self.src.index(name)
            ns_src = self.src[s:self.src.index("\n\n\n", s)]
            exec(compile(ns_src, "submission_flower.py", "exec"), ns)
        return ns

    def _members(self, ns, cfg, m=3):
        import torch
        torch.manual_seed(0)
        out = []
        for _ in range(m):
            mm = ns["_FlowerS2S"](cfg)
            with torch.no_grad():
                for p in mm.parameters():
                    p.normal_(0.0, 0.05)
                mm.mean.copy_(torch.randn(cfg["c_in"]) * 0.01)
                mm.std.copy_(1.0 + torch.rand(cfg["c_in"]) * 0.1)
            out.append(mm.eval())
        return out

    def _cfg(self, **kw):
        cfg = dict(t_in=2, c_in=2, t_out=2, c_out=2, resolution=(8, 16), lifting_dim=8,
                   n_levels=2, num_heads=2, groups=2,
                   boundary_condition_types=["PERIODIC", "PERIODIC"])
        cfg.update(kw)
        return cfg

    def _check_grouped(self, cfg):
        """T6(b): a leading member axis in every weight, realised as grouped convs,
        must reproduce the members run one at a time."""
        import torch
        ns = self._ns()
        members = self._members(ns, cfg)
        x = torch.randn(2, cfg["t_in"], *cfg["resolution"], cfg["c_in"])
        with torch.no_grad():
            ref = torch.stack([m(x) for m in members])
            got = ns["_build_alt_stacked"](members, cfg, "cpu", "grouped")(x)
        self.assertEqual(tuple(got.shape), tuple(ref.shape))
        d = (got - ref).abs().max().item()
        self.assertLess(d, 1e-5, f"grouped ensemble differs from the loop by {d:.2e}")

    def test_grouped_convs_reproduce_the_members_exactly(self):
        self._check_grouped(self._cfg())

    def test_grouped_convs_reproduce_a_delta_model_too(self):
        # predict_delta and window_norm both touch the member axis by hand
        self._check_grouped(self._cfg(predict_delta=True, window_norm=True))

    def test_the_loop_mode_is_the_plain_stack(self):
        import torch
        ns = self._ns()
        cfg = self._cfg()
        members = self._members(ns, cfg)
        x = torch.randn(2, cfg["t_in"], *cfg["resolution"], cfg["c_in"])
        with torch.no_grad():
            ref = torch.stack([m(x) for m in members])
            got = ns["_build_alt_stacked"](members, cfg, "cpu", "loop")(x)
        self.assertTrue(torch.equal(got, ref))


class E4PerMemberWidths(unittest.TestCase):
    """RPDE_HEAD_PER_MEMBER runs the width head on every member's prediction and
    combines the k half-widths. Unset, the head runs once on the mean, as shipped."""

    SRC = None

    def setUp(self):
        if E4PerMemberWidths.SRC is None:
            E4PerMemberWidths.SRC = (REPO / "submission" / "submission_flower.py").read_text()
        self.src = E4PerMemberWidths.SRC

    def test_default_is_the_single_head_call_on_the_mean(self):
        self.assertIn('os.environ.get("RPDE_HEAD_PER_MEMBER", "")', self.src)
        i = self.src.index("def _fused(")
        body = self.src[i:self.src.index("\n\n\n", i)]
        self.assertIn("if _HEAD and _HEAD_PER_MEMBER and ybase.shape[0] > 1:", body)
        self.assertIn("half = _head_half_torch(base, std, static_x, device) if _HEAD else None", body)

    def test_an_unparsable_combiner_fails_at_import_not_on_the_board(self):
        self.assertIn("expected '', mean, median or q<p>", self.src)

    def _per_member_half(self, combiner):
        import torch
        src = self.src
        s = src.index("def _per_member_half(")
        seen = {}

        def stub(mean_t, std_t, x_t, device):
            seen["x"] = tuple(x_t.shape)
            seen["std"] = None if std_t is None else tuple(std_t.shape)
            return mean_t[..., :2].abs()
        ns = {"torch": torch, "_HEAD_PER_MEMBER": combiner, "_head_half_torch": stub}
        exec(compile(src[s:src.index("\n\n\n", s)], "submission_flower.py", "exec"), ns)
        return ns["_per_member_half"], seen

    def test_the_member_axis_folds_into_the_batch_and_reduces_correctly(self):
        import torch
        M, N, T, H, W, Cc = 4, 3, 2, 5, 6, 2
        torch.manual_seed(1)
        y = torch.randn(M, N, T, H, W, Cc)
        x = torch.randn(N, 7, H, W, 3)
        for combiner, expect in (("mean", lambda h: h.mean(0)),
                                 ("median", lambda h: h.median(0).values),
                                 ("q0.75", lambda h: h.quantile(0.75, dim=0))):
            fn, seen = self._per_member_half(combiner)
            got = fn(y, None, x, "cpu")
            self.assertEqual(tuple(got.shape), (N, T, H, W, 2))
            self.assertTrue(torch.allclose(got, expect(y[..., :2].abs()), atol=1e-6),
                            f"{combiner} combined the members wrongly")
            # one head call over M*N windows, not M calls
            self.assertEqual(seen["x"][0], M * N)

    def test_identical_members_leave_the_widths_unchanged(self):
        # the sanity property: if every member predicts the same thing, running the
        # head per member and combining must equal running it once on the mean
        import torch
        y = torch.randn(1, 3, 2, 4, 5, 2).expand(6, -1, -1, -1, -1, -1).contiguous()
        x = torch.randn(3, 7, 4, 5, 3)
        for combiner in ("mean", "median", "q0.75"):
            fn, _ = self._per_member_half(combiner)
            got = fn(y, None, x, "cpu")
            self.assertTrue(torch.allclose(got, y[0][..., :2].abs(), atol=1e-6))


class T7FastPredictPath(unittest.TestCase):
    """The fast predict path (T7): a build-time switch, a decline rule, and an
    aliasing guard. Everything here is source-level or pure logic -- the path
    itself needs a CUDA graph, which the timing parcel exercises
    (experiments/queue/2026-09-21-cc-T7-fresh-process-cost-of-predict.sh)."""

    @classmethod
    def setUpClass(cls):
        cls.src = (REPO / "submission" / "submission_flower.py").read_text()

    def _body(self, name):
        i = self.src.index(name)
        return self.src[i:self.src.index("\n\n\n", i)]

    def _flag(self, bounds, env):
        """Exec the four lines that resolve _FAST, with a stub bounds.json and env."""
        i = self.src.index("_FAST = bool(_BOUNDS")
        snippet = self.src[i:self.src.index("\n_fast_out", i)]
        ns = {"_BOUNDS": bounds, "os": type("o", (), {"environ": dict(env)})()}
        exec(compile(snippet, "submission_flower.py", "exec"), ns)
        return ns["_FAST"]

    def test_the_conservative_path_is_the_default(self):
        """Every bounds.json written so far has no fast_predict key, so every zip
        already built keeps the shipped path with no edit of any kind."""
        self.assertFalse(self._flag({}, {}))
        self.assertFalse(self._flag({"fast_predict": False}, {}))

    def test_bounds_json_alone_turns_it_on(self):
        """The grader gives us no environment, so the switch must be DATA in the zip."""
        self.assertTrue(self._flag({"fast_predict": True}, {}))

    def test_the_env_override_wins_both_ways_for_benching(self):
        on = {"RPDE_FAST_PREDICT": "1"}
        off = {"RPDE_FAST_PREDICT": "0"}
        self.assertTrue(self._flag({}, on))
        self.assertFalse(self._flag({"fast_predict": True}, off))

    def test_the_switch_is_read_from_bounds_json_not_typed_into_the_source(self):
        self.assertIn('_BOUNDS.get("fast_predict"', self.src)

    def test_a_call_the_fast_path_cannot_serve_is_declined_not_guessed(self):
        """_fast_ready must refuse anything outside one captured chunk with the
        symmetric head, so predict() falls through to the shipped path."""
        ns = {"os": type("o", (), {"environ": {}})()}
        for name in ("def _bucket(", "def _plan(", "def _fast_ready("):
            s = self.src.index(name)
            exec(compile(self.src[s:self.src.index("\n\n\n", s)], "sf.py", "exec"), ns)
        ns.update(_FAST=True, _fast_out={"off": False}, _HEAD=True, _HEAD_NOUT=2,
                  C=3, _GRAPH_MAX=16, _BATCH=16, _GRAPH_LADDER=(1, 5, 8, 16),
                  _graphs_enabled=lambda d: True)
        cfg = {"c_out": 3}
        self.assertTrue(ns["_fast_ready"](5, "cuda", cfg))
        self.assertFalse(ns["_fast_ready"](20, "cuda", cfg), "two chunks must decline")
        self.assertFalse(ns["_fast_ready"](0, "cuda", cfg))
        self.assertFalse(ns["_fast_ready"](5, "cuda", {"c_out": 2}), "c_out < C must decline")
        ns["_HEAD_NOUT"] = 4
        self.assertTrue(ns["_fast_ready"](5, "cuda", cfg), "H1's lower/upper head is served")
        ns["_HEAD_NOUT"] = 6
        self.assertFalse(ns["_fast_ready"](5, "cuda", cfg), "H2's prediction correction must decline")
        ns["_HEAD_NOUT"] = 2
        ns["_HEAD"] = None
        self.assertFalse(ns["_fast_ready"](5, "cuda", cfg), "no head must decline")
        ns["_HEAD"] = True
        ns["_fast_out"] = {"off": True}
        self.assertFalse(ns["_fast_ready"](5, "cuda", cfg), "a disabled path must stay disabled")
        ns["_fast_out"] = {"off": False}
        ns["_FAST"] = False
        self.assertFalse(ns["_fast_ready"](5, "cuda", cfg))

    def test_the_pinned_buffers_are_handed_out_at_most_once(self):
        """The grader promises one predict() per process. If it calls twice, the
        second call must NOT write into buffers the first call handed to the caller."""
        body = self._body("def _predict_fast(")
        self.assertIn('if _fast_out["handed_out"]:', body)
        self.assertIn('_fast_out["handed_out"] = True', body)
        # the copy-out branch must come from the DEVICE tensors, never the pinned ones
        i = body.index('if _fast_out["handed_out"]:')
        j = body.index("else:", i)
        self.assertNotIn("pinned[", body[i:j], "the second call reused the pinned buffers")
        self.assertIn(".cpu().numpy()", body[i:j])

    def test_the_host_buffers_are_page_touched_at_import(self):
        """The whole point: 13.6 MB of first-touch page faults move out of the timed
        region. An allocation without a write leaves the faults where they were."""
        body = self._body("def _capture_fast(")
        self.assertIn("pin_memory=True", body)
        self.assertIn("p.zero_()", body)
        self.assertIn("g.replay()", body)          # pre-upload, so the first launch is cheap

    def test_the_dead_member_std_is_not_computed_when_nothing_reads_it(self):
        body = self._body("def _fused_fast(")
        self.assertIn("if (y.shape[0] > 1 and _HEAD_USE_Z) else None", body)

    def test_the_fast_path_checks_the_two_things_that_score_zero(self):
        """A non-finite bound or lower > upper zeroes every subscore, and the host
        never sees the arrays, so the check has to be inside the graph."""
        body = self._body("def _fused_fast(")
        self.assertIn("torch.isfinite(lower).all()", body)
        self.assertIn("torch.isfinite(upper).all()", body)
        self.assertIn("(lower <= upper).all()", body)

    def test_a_failure_falls_back_to_the_shipped_path_rather_than_raising(self):
        i = self.src.index("def predict(")
        body = self.src[i:self.src.index("\n\n\n", i)]
        self.assertIn('if _FAST and not _fast_out["off"]:', body)
        self.assertIn('_fast_out["off"] = True', body)
        self.assertIn("prediction, disagreement = _predict_fields(input_array)", body)
        self.assertIn("lower = base - half", body)     # the shipped path, untouched

    def test_the_fast_graphs_replace_the_shipped_ones_rather_than_joining_them(self):
        """Two ladders of captured graphs would double the graph pool; E1 OOM-killed
        at k=12 with 8.58 GiB in graph pools."""
        body = self._body("def _warm_up(")
        self.assertIn("if fast:", body)
        self.assertIn("elif b not in _graphs:", body)


def _real_inflate(np, scale=1.0):
    """Compile submission_flower's own _inflate_fluct so exec-based tests use shipped code."""
    src = (REPO / "submission" / "submission_flower.py").read_text()
    i = src.index("def _inflate_fluct(")
    body = src[i:src.index("\n\n\ndef ", i)]
    ns = {"np": np, "_FLUCT_SCALE": scale}
    exec(compile(body, "submission_flower.py", "exec"), ns)
    return ns["_inflate_fluct"]


def _real_anchor(np, alpha=0.0):
    """Compile submission_flower's own _anchor_mean (ANCH) so exec-based tests use shipped code."""
    src = (REPO / "submission" / "submission_flower.py").read_text()
    i = src.index("def _anchor_mean(")
    body = src[i:src.index("\n\n\ndef ", i)]
    ns = {"np": np, "_MEAN_ANCHOR": alpha}
    exec(compile(body, "submission_flower.py", "exec"), ns)
    return ns["_anchor_mean"]


class MemberZeroIsTheCkpt(unittest.TestCase):
    """2026-09-23: build_submission wrote model.pth from --ckpt but staged members[1:] and recorded
    members[0] in ensemble.json. With --ckpt != members[0] the zip shipped --ckpt in place of
    members[0] -- dropping one member, duplicating another, and certifying a set it did not contain.
    145 of 611 built ensemble zips were affected (no submitted zip). The guard must REFUSE."""

    def test_the_guard_exists_and_compares_resolved_paths(self):
        src = (REPO / "scripts" / "build_submission.py").read_text()
        i = src.index("if a.ensemble:")
        block = src[i:i + 1800]
        self.assertIn("Path(a.ckpt).resolve() != Path(a.ensemble[0]).resolve()", block)
        self.assertIn("raise SystemExit", block)

    def test_member_zero_is_never_staged_so_the_guard_is_load_bearing(self):
        """_payload_members starts at members[1:] on purpose; that is WHY member 0 must be --ckpt."""
        src = (REPO / "scripts" / "build_submission.py").read_text()
        i = src.index("def _payload_members(")
        body = src[i:src.index("\n\n\ndef ", i)]
        self.assertIn("members[1:]", body)
        self.assertIn('names = ["model.pth"]', body)

    def test_every_submitted_zip_ships_exactly_what_it_records(self):
        """The defect is observable in the artefacts, so assert on them rather than on source text.

        THIS TEST WAS FIRST WRITTEN TO CHECK ONLY THAT MEMBERS WERE DISTINCT, AND THAT WAS NOT ENOUGH.
        The bug has two modes. When --ckpt is INSIDE --ensemble it duplicates a member, which a
        distinctness check catches. When --ckpt is OUTSIDE --ensemble it SUBSTITUTES one: a model the
        record does not name ships, members[0] is dropped, and every member stays distinct -- invisible
        to the first version of this test. 17 of the 162 corrupted zips are that mode.
        So compare each shipped member's CRC against the CRC of the source ensemble.json names. A
        source that no longer exists on disk is skipped rather than failed; a `last.pth` source is
        tolerated because _stage_weights re-saves those (build_submission.py:28-44)."""
        import zipfile, zlib, json as _json
        dist = REPO / "submission" / "dist"
        if not dist.is_dir():
            self.skipTest("no dist directory")

        def crc_of(path):
            h = 0
            with open(path, "rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        return h
                    h = zlib.crc32(b, h)

        offenders = []
        for z in sorted(dist.glob("sub*.zip")):      # submitted-style only: those must always be clean
            with zipfile.ZipFile(z) as zf:
                if "ensemble.json" not in zf.namelist():
                    continue
                ej = _json.loads(zf.read("ensemble.json"))
                shipped = {i.filename: i.CRC for i in zf.infolist()
                           if i.filename == "model.pth" or i.filename.startswith("model_")}
                for name, src in zip(ej["members"], ej["sources"]):
                    if not pathlib.Path(src).exists() or src.endswith("last.pth"):
                        continue
                    if crc_of(src) != shipped.get(name):
                        offenders.append(f"{z.name}:{name}")
        self.assertEqual(offenders, [], f"zips shipping something other than what they record: {offenders}")


class FluctScale(unittest.TestCase):
    """VINF 2026-09-23: re-inflate the prediction's fluctuation about its own time mean.

    BLUR measured the shipped prediction at 65% of the true unsteady energy (46% at the longest lead),
    and a sweep found alpha ~1.05 worth +0.06 composite on BOTH held-out reference rows. The switch is
    DATA in bounds.json, default 1.0, which must be a bit-identical no-op.
    """

    @classmethod
    def setUpClass(cls):
        cls.src = (REPO / "submission" / "submission_flower.py").read_text()
        cls.rezip = (REPO / "submission" / "rezip_fast.py").read_text()

    def test_default_is_a_no_op(self):
        self.assertIn('_BOUNDS.get("fluct_scale", 1.0)', self.src)
        i = self.src.index("def _inflate_fluct(")
        body = self.src[i:self.src.index("\n\n\ndef ", i)]
        self.assertIn("if _FLUCT_SCALE == 1.0:", body)
        self.assertIn("return pred", body)

    def test_it_preserves_the_time_mean_and_is_a_no_op_at_one(self):
        """The whole point: the mean flow is untouched, so mvpe (a time-averaged probe quantity) is
        invariant. Run the SHIPPED function, not a re-derivation of its arithmetic.
        At alpha = 1.0 the guarantee is BIT-identity, which float32 (m + 1.0*(p-m)) does NOT give --
        that is exactly why the function early-returns instead of computing. Test the guarantee."""
        import numpy as _np
        rng = _np.random.default_rng(0)
        p = rng.normal(size=(3, 20, 8, 16, 2)).astype(_np.float32)
        _np.testing.assert_array_equal(_real_inflate(_np, 1.0)(p), p)     # bit-identical no-op
        for alpha in (0.9, 1.05, 1.3):
            q = _real_inflate(_np, alpha)(p)
            _np.testing.assert_allclose(q.mean(axis=1), p.mean(axis=1), rtol=0, atol=2e-6)
            self.assertGreater(float(((q - q.mean(1, keepdims=True)) ** 2).mean()) /
                               float(((p - p.mean(1, keepdims=True)) ** 2).mean()),
                               0.5 if alpha < 1 else 1.0, "fluctuation energy must scale with alpha")

    def test_it_is_applied_after_the_bounds_at_every_return(self):
        """Applied to the prediction ONLY and after lower/upper exist. kit/scoring.py tests the truth
        against the interval and never compares the interval to the prediction, so the bounds stay
        valid -- but only if they were built from the un-inflated field, which is what was measured."""
        lines = self.src.splitlines()
        rets = [(i, l) for i, l in enumerate(lines) if l.strip().startswith('return {"prediction"')]
        self.assertTrue(rets, "no predict() return found")
        # Since SUB27b (2026-09-23) the FAST path applies the inflation DEVICE-SIDE inside the
        # captured graph (_fused_fast: `mean = _inflate_fluct_t(mean)` after lower/upper), so its
        # return in _predict_fast is deliberately NOT wrapped -- wrapping it again would give
        # alpha^2, which is exactly what the SUB27 gate caught. Every other return wraps.
        self.assertIn("mean = _inflate_fluct_t(mean)", self.src)
        fused_fast = self.src[self.src.index("def _fused_fast("):]
        fused_fast = fused_fast[:fused_fast.index("\ndef ")]
        self.assertLess(fused_fast.index("lower"), fused_fast.index("_inflate_fluct_t(mean)"),
                        "the device-side inflation must come AFTER the bounds are formed")
        for i, r in rets:
            self.assertIn('"lower": lower', r)
            if "_inflate_fluct(" in r:
                continue
            owner = next(l for l in reversed(lines[:i]) if l.startswith("def "))
            self.assertTrue(owner.startswith("def _predict_fast("),
                            f"unwrapped return outside the fast path: {owner.strip()}: {r.strip()}")

    def test_the_band_is_enforced_and_rezip_exposes_it(self):
        self.assertIn('assert 0.5 <= _FLUCT_SCALE <= 2.0', self.src)
        self.assertIn('"--fluct-scale"', self.rezip)
        self.assertIn('b["fluct_scale"] = float(a.fluct_scale)', self.rezip)


class Fp16WidthHead(unittest.TestCase):
    """2026-09-22 (FP16H): the width head may run in half precision on the captured graph path.
    The head produces interval offsets only, never the prediction, so rel_l2/tke/mvpe are
    bit-identical by construction and only the half-widths can move. The switch is DATA in
    bounds.json, default off, so a zip built without it is the shipped path unchanged."""

    @classmethod
    def setUpClass(cls):
        cls.src = (REPO / "submission" / "submission_flower.py").read_text()
        cls.rezip = (REPO / "submission" / "rezip_fast.py").read_text()

    def test_head_fp16_is_read_from_bounds_json_and_defaults_off(self):
        self.assertIn('_BOUNDS.get("head_fp16", False)', self.src)
        line = [l for l in self.src.splitlines() if l.startswith("_HEAD_FP16 = ")]
        self.assertEqual(len(line), 1, "exactly one _HEAD_FP16 definition")
        ns = {"_BOUNDS": {}}
        exec(compile(line[0], "submission_flower.py", "exec"), ns)
        self.assertFalse(ns["_HEAD_FP16"], "default must be off")
        ns = {"_BOUNDS": {"head_fp16": True}}
        exec(compile(line[0], "submission_flower.py", "exec"), ns)
        self.assertTrue(ns["_HEAD_FP16"])

    def test_half_is_confined_to_the_heads_forward_and_only_on_cuda(self):
        """.half() must be gated on a CUDA device (CPU half convolutions are unimplemented or
        far slower, and verify_submission runs the zip on CPU under torch 2.2.2), and the head's
        output must return to float32 before any consumer sees it."""
        i = self.src.index("def _get_head(")
        body = self.src[i:self.src.index("\n\n\ndef ", i)]
        self.assertIn("_HEAD_FP16", body)
        self.assertIn('str(device).startswith("cuda")', body)
        self.assertIn(".half()", body)
        j = self.src.index("def _head_half_torch(")
        fwd = self.src[j:self.src.index("\n\n\ndef ", j)]
        self.assertIn("head(f.half()).float()", fwd)
        self.assertIn("f.is_cuda", fwd)

    def test_the_eager_path_follows_the_heads_dtype(self):
        """_get_head caches ONE module and both paths share it. The eager path must therefore cast
        its features to whatever dtype the head is, or head_fp16 raises the first time a call too
        large for the captured graphs falls through to it."""
        i = self.src.index("def _head_raw(")
        body = self.src[i:self.src.index("\n\n\ndef ", i)]
        self.assertIn("next(head.parameters()).dtype", body)
        self.assertIn("dtype=_hdt", body)
        self.assertIn(".float()", body)

    def test_rezip_exposes_the_switch_both_ways(self):
        self.assertIn('"--fp16-head"', self.rezip)
        self.assertIn('"--no-fp16-head"', self.rezip)
        # fp16 and the timing line are INDEPENDENT switches: a zip may want both at once. Putting
        # them in one mutually exclusive group made `--timing --fp16-head` a parse error, which is
        # exactly how Slurm 4545 died.
        import argparse as _ap
        self.assertNotEqual(
            self.rezip.index('"--fp16-head"') and self.rezip[:self.rezip.index('"--fp16-head"')].rfind("add_mutually_exclusive_group"),
            self.rezip[:self.rezip.index('"--timing"')].rfind("add_mutually_exclusive_group"),
            "--fp16-head must not share a mutually exclusive group with --timing")
        self.assertIn('b["head_fp16"] = bool(a.fp16_head)', self.rezip)
        self.assertIn("head_fp16=", self.rezip)


class PhaseTimingAndBoundsFrom(unittest.TestCase):
    """2026-09-22, from the three reviews: a diagnostic timing line behind a bounds.json switch,
    and score_local's ability to score one zip's prediction with another zip's bounds (CEN)."""

    @classmethod
    def setUpClass(cls):
        cls.src = (REPO / "submission" / "submission_flower.py").read_text()

    def test_phase_timing_is_read_from_bounds_json_and_defaults_off(self):
        self.assertIn('_BOUNDS.get("phase_timing", False)', self.src)
        i = self.src.index("_TIMING = bool(_BOUNDS")
        snippet = self.src[i:self.src.index("\n_fast_out", i)]
        ns = {"_BOUNDS": {}}
        exec(compile(snippet, "submission_flower.py", "exec"), ns)
        self.assertFalse(ns["_TIMING"])
        ns = {"_BOUNDS": {"phase_timing": True}}
        exec(compile(snippet, "submission_flower.py", "exec"), ns)
        self.assertTrue(ns["_TIMING"])

    def test_the_timing_line_is_one_shot_and_read_after_the_existing_sync(self):
        """The CUDA events must be read only after the synchronize() the fast path already does,
        and the line must print once per process (the _TIMED latch)."""
        i = self.src.index("def _predict_fast(")
        body = self.src[i:self.src.index("\n\n\ndef ", i)]
        self.assertIn("_TIMING and not _TIMED[0]", body)
        self.assertIn("_TIMED[0] = True", body)
        self.assertLess(body.index("torch.cuda.current_stream().synchronize()"), body.index("elapsed_time"))
        self.assertIn('"[submission] timing first_call', body)
        # the module records both ends of the untimed import region
        self.assertIn("_T_IMPORT_START = time.perf_counter()", self.src)
        self.assertTrue(self.src.rstrip().endswith("# phase_timing: end of the untimed import region"))

    def test_rezip_fast_can_toggle_phase_timing(self):
        rz = (REPO / "submission" / "rezip_fast.py").read_text()
        self.assertIn('"--timing"', rz)
        self.assertIn('b["phase_timing"] = bool(a.timing)', rz)

    def test_recentre_bounds_is_identity_at_scale_one_and_scales_the_width(self):
        import importlib.util
        import numpy as np
        spec = importlib.util.spec_from_file_location("score_local", REPO / "scripts" / "score_local.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        lo = np.array([0.0, 1.0], dtype=np.float32); up = np.array([2.0, 5.0], dtype=np.float32)
        l1, u1 = mod.recentre_bounds(lo, up, 1.0)
        self.assertIs(l1, lo); self.assertIs(u1, up)
        l2, u2 = mod.recentre_bounds(lo, up, 1.5)
        np.testing.assert_allclose(0.5 * (l2 + u2), 0.5 * (lo + up))          # centre unchanged
        np.testing.assert_allclose(u2 - l2, 1.5 * (up - lo))                  # width scaled
        self.assertTrue((REPO / "scripts" / "predict_zip.py").exists())


class MixGrowEpoch(unittest.TestCase):
    """D1p: --mix-grow-epoch scales the sampler's epoch by 1/(1-frac) so p00 exposure is kept."""

    def test_flag_exists_and_the_sampler_uses_the_grown_count(self):
        src = (REPO / "scripts" / "train.py").read_text()
        self.assertIn('"--mix-grow-epoch"', src)
        i = src.index("n_samples = len(train_ds)")
        body = src[i:src.index("WeightedRandomSampler(w, num_samples=n_samples", i)]
        self.assertIn("round(len(train_ds) / (1.0 - _frac))", body)
        self.assertIn("if a.mix_grow_epoch:", body)

    def test_the_arithmetic_keeps_p00_exposure(self):
        n_real, frac = 11617, 0.75
        n = int(round(n_real / (1.0 - frac)))
        self.assertEqual(n, 46468)
        # expected p00 draws per epoch = (1 - frac) * n == n_real (to rounding)
        self.assertAlmostEqual((1.0 - frac) * n, n_real, delta=1.0)


class MeanAnchorKey(unittest.TestCase):
    """ANCH 2026-09-24: a second post-hoc transform behind a bounds.json key, the MEAN ANCHOR

        p' = p + A * (m_in - mean_t(p))     on [u, v]; channel 2 and the bounds untouched; A = 0.0 = identity

    under the rules that made fluct_scale safe: the default is a bit-identical no-op, it touches the
    PREDICTION only, AFTER the bounds, exactly once on every path -- device-side in _fused_fast for the
    fast route, numpy in predict()'s return for the graph/eager route, and NOT in _predict_fast's return
    (the SUB27 double application). Order on both paths: bounds -> fluct_scale -> mean_anchor.
    """

    @classmethod
    def setUpClass(cls):
        cls.src = (REPO / "submission" / "submission_flower.py").read_text()
        cls.rezip = (REPO / "submission" / "rezip_fast.py").read_text()

    def test_key_is_read_next_to_fluct_scale_defaults_to_zero_and_is_banded(self):
        self.assertIn('_BOUNDS.get("mean_anchor", 0.0)', self.src)
        self.assertLess(self.src.index('_BOUNDS.get("fluct_scale", 1.0)'),
                        self.src.index('_BOUNDS.get("mean_anchor", 0.0)'))
        lines = [l for l in self.src.splitlines()
                 if l.startswith("_MEAN_ANCHOR = ") or l.startswith("assert 0.0 <= _MEAN_ANCHOR <= 1.0")]
        self.assertEqual(len(lines), 2, "exactly one definition and one band assert")
        block = compile("\n".join(lines), "submission_flower.py", "exec")
        ns = {"_BOUNDS": {}}; exec(block, ns)
        self.assertEqual(ns["_MEAN_ANCHOR"], 0.0); self.assertIsInstance(ns["_MEAN_ANCHOR"], float)
        ns = {"_BOUNDS": {"mean_anchor": 0.1}}; exec(block, ns)
        self.assertEqual(ns["_MEAN_ANCHOR"], 0.1)
        for bad in (-0.1, 1.5):
            with self.assertRaises(AssertionError, msg=f"mean_anchor {bad} must be refused at import"):
                exec(block, {"_BOUNDS": {"mean_anchor": bad}})

    def test_both_twins_exist_early_return_at_zero_and_touch_u_v_only(self):
        for name, clamp in (("def _anchor_mean_t(", "cnt.clamp(min=1.0)"),
                            ("def _anchor_mean(", "np.maximum(cnt, 1.0)")):
            i = self.src.index(name)
            body = self.src[i:self.src.index("\n\n\n", i)]
            self.assertIn("if _MEAN_ANCHOR == 0.0:", body, name)
            self.assertIn("[..., :2]", body, name)                      # channels u, v only
            self.assertIn("!= 0", body, name)                           # exact-zero hole convention
            self.assertIn(clamp, body, name)                            # a pixel with no data: no division blow-up
            self.assertIn("cnt > 0", body, name)                        # ... and no shift

    def test_fast_path_order_is_bounds_then_fluct_then_anchor_inside_the_capture(self):
        fused_fast = self.src[self.src.index("def _fused_fast("):]
        fused_fast = fused_fast[:fused_fast.index("\ndef ")]
        i_lower = fused_fast.index("lower = (base - half)")
        i_fluct = fused_fast.index("mean = _inflate_fluct_t(mean)")
        i_anchor = fused_fast.index("mean = _anchor_mean_t(mean, static_x)")
        i_ret = fused_fast.index("return mean, lower, upper, ok")
        self.assertLess(i_lower, i_fluct, "bounds before fluct_scale")
        self.assertLess(i_fluct, i_anchor, "fluct_scale before mean_anchor")
        self.assertLess(i_anchor, i_ret)
        self.assertNotIn(".item()", fused_fast[i_anchor - 1200:i_anchor])   # nothing host-syncing was added
        # _fused (the graph/eager route) stays untouched: that route's twin runs in predict()
        fused = self.src[self.src.index("def _fused("):]
        fused = fused[:fused.index("\ndef ")]
        self.assertNotIn("_anchor_mean", fused)
        self.assertNotIn("_inflate_fluct", fused)

    def test_eager_path_nests_anchor_around_fluct_and_predict_fast_stays_bare(self):
        i = self.src.index("def predict(")
        body = self.src[i:self.src.index("\n\n\n", i)]
        rets = [l for l in body.splitlines() if l.strip().startswith('return {"prediction"')]
        self.assertEqual(len(rets), 1)
        self.assertIn('_anchor_mean(_inflate_fluct(prediction), input_array)', rets[0])
        self.assertIn('"lower": lower, "upper": upper', rets[0])
        self.assertLess(body.index("upper = base + up"), body.index("_anchor_mean("), "bounds first")
        i = self.src.index("def _predict_fast(")
        pf = self.src[i:self.src.index("\n\n\n", i)]
        self.assertNotIn("_anchor_mean(", pf, "the fast path applies it device-side; wrapping again is SUB27")
        self.assertIn('return {"prediction": mean, "lower": lower, "upper": upper}', pf)

    def test_numpy_twin_matches_variance_inflation_to_1e6_and_is_the_identity_at_zero(self):
        """The shipped numpy _anchor_mean against scripts/variance_inflation.mean_anchor (the reference the
        A sweep was measured with), on random data with a partial hole and an all-hole pixel."""
        import numpy as np
        import variance_inflation as vi          # scripts/ is on sys.path; numpy-only at import
        rng = np.random.default_rng(0)
        N, T, H, W, C = 2, 20, 4, 6, 3
        pred = (rng.normal(size=(N, T, H, W, C)) * 0.5).astype(np.float32)
        x = np.empty((N, T, H, W, C), np.float32)
        x[..., 0] = rng.uniform(0.2, 1.0, size=(N, T, H, W))
        x[..., 1] = rng.uniform(-0.5, 0.5, size=(N, T, H, W))
        x[..., 2] = 0.0
        x[0, :7, 2, 3, :] = 0.0                  # a partial hole: 13 measured frames
        x[1, :, 3, 5, :] = 0.0                   # no measured frame at all: no shift
        for alpha in (0.1, 0.25, 1.0):
            got = _real_anchor(np, alpha)(pred, x)
            want, _shift = vi.mean_anchor(pred, x, alpha)
            self.assertEqual(got.dtype, np.float32)
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-6, err_msg=f"alpha {alpha}")
            self.assertEqual(got[..., 2].tobytes(), pred[..., 2].tobytes())
            self.assertEqual(got[1, :, 3, 5].tobytes(), pred[1, :, 3, 5].tobytes())
            np.testing.assert_allclose((got - got.mean(1, keepdims=True))[..., :2],
                                       (pred - pred.mean(1, keepdims=True))[..., :2], rtol=0, atol=1e-6,
                                       err_msg="only the level moves; the fluctuation is carried over")
            self.assertGreater(float(np.abs(got[0, :, 0, 0, :2] - pred[0, :, 0, 0, :2]).max()), 0.0)
            # the input is read, not written: predict() hands the grader's own array in
            self.assertTrue(np.all(x[0, :7, 2, 3] == 0))
        pred[0, 0, 0, 0, 0] = -0.0                # -0.0 + 0.0 = +0.0: the identity must not compute
        out0 = _real_anchor(np, 0.0)(pred, x)
        self.assertIs(out0, pred, "A = 0.0 returns the argument itself: no copy, no arithmetic")
        # a float64 input window (the grader's dtype is not promised) gives the same float32 answer
        got64 = _real_anchor(np, 0.25)(pred, x.astype(np.float64))
        np.testing.assert_array_equal(got64, _real_anchor(np, 0.25)(pred, x))

    def test_rezip_exposes_it_and_the_board_log_shows_it(self):
        self.assertIn('"--mean-anchor"', self.rezip)
        self.assertIn('b["mean_anchor"] = float(a.mean_anchor)', self.rezip)
        self.assertIn("or a.mean_anchor is not None", self.rezip)      # the key is written even when nothing else changes
        self.assertIn("mean_anchor={bj.get('mean_anchor', 0.0)}", self.rezip)
        # the one-shot "[submission] call n=" line carries it on BOTH paths, next to scale=
        self.assertEqual(self.src.count("scale={_HEAD_SCALE:g} anchor={_MEAN_ANCHOR:g} "), 2)


class RolloutBlendKey(unittest.TestCase):
    """RBLEND 2026-09-24: an INFERENCE-TIME ROLLOUT BLEND behind bounds.json's rollout_blend key,

        x2 = concat(x[:, S:], fb)    fb = p0[:, :S], holes of x's last frame -> +0.0, channel 2 carried
        p2 = mean over the member SUBSET of the forward on x2
        prediction[:, S:] = (1-W) p0[:, S:] + W p2[:, :20-S]          frames [0, S) untouched

    exactly scripts/variance_inflation.py's rollout_input + rollout_blend. Contract: absent = off with no
    code path run (the fast graph captures what it captured before), the bounds stay p0's, and the order on
    BOTH paths is bounds (p0) -> blend -> fluct_scale -> mean_anchor -- the measured arm (alpha after the
    blend). Fast path: device-side in _fused_fast; graph/eager path: device-side in _fused, p0 handed on as
    the interval centre, and predict()'s return applies the numpy fluct/anchor twins to the blend.
    """

    HELPERS = ("_rollout_input", "_rollout_input_t", "_rollout_blend", "_rollout_blend_t", "_rollout_pass_t")

    @classmethod
    def setUpClass(cls):
        cls.src = (REPO / "submission" / "submission_flower.py").read_text()
        cls.rezip = (REPO / "submission" / "rezip_fast.py").read_text()

    def _body(self, name):
        i = self.src.index(name)
        return self.src[i:self.src.index("\n\n\n", i)]

    def _fn(self, name, **glb):
        """Compile one top-level function of the module into a namespace (the module itself needs a zip)."""
        import numpy as np
        ns = {"np": np}
        ns.update(glb)
        exec(compile(self._body(f"def {name}("), "submission_flower.py", "exec"), ns)
        return ns[name]

    def _import_block(self, bounds, members):
        """Exec the import-time lines that read the key, with a stub bounds.json and ensemble."""
        i = self.src.index('_RBLEND = _BOUNDS.get("rollout_blend")')
        block = self.src[i:self.src.index("\n\n", i)]
        ns = {"_BOUNDS": bounds, "_MEMBERS": members,
              "_check_rollout_blend": self._fn("_check_rollout_blend")}
        exec(compile(block, "submission_flower.py", "exec"), ns)
        return ns

    def test_key_absent_is_off_and_every_rollout_line_is_guarded_by_it(self):
        ns = self._import_block({}, ["model.pth", "model_1.pth"])
        self.assertIsNone(ns["_RBLEND"])
        self.assertIsNone(ns["_RB_SHIFT"]); self.assertIsNone(ns["_RB_WEIGHT"]); self.assertIsNone(ns["_RB_MEMBERS"])
        self.assertEqual(ns["_RB_TAG"], "off")
        ns = self._import_block({"rollout_blend": {"shift": 10, "weight": 0.5, "members": [0, 1]}},
                                ["m%d" % i for i in range(5)])
        self.assertEqual((ns["_RB_SHIFT"], ns["_RB_WEIGHT"], ns["_RB_MEMBERS"]), (10, 0.5, [0, 1]))
        self.assertEqual(ns["_RB_TAG"], "10/0.5/0+1")
        # Every use of the rollout machinery -- a helper call, the subset stack, the parsed fields -- sits
        # under `if _RBLEND is not None:` or inside a rollout helper (which only guarded code calls). With the
        # key absent nothing runs: no subset stack is built and _fused/_fused_fast capture today's kernels.
        import ast
        tree = ast.parse(self.src)
        parents = {}
        for node in ast.walk(tree):
            for ch in ast.iter_child_nodes(node):
                parents[ch] = node
        inside_ok = set(self.HELPERS) | {"_check_rollout_blend"}

        def guarded(node):
            n = node
            while n in parents:
                child, n = n, parents[n]
                if isinstance(n, ast.If) and ast.unparse(n.test) == "_RBLEND is not None" and child in n.body:
                    return True
                if isinstance(n, ast.FunctionDef) and n.name in inside_ok:
                    return True
            return False
        uses = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in inside_ok:
                uses.append((node, node.func.id))
            elif isinstance(node, ast.Constant) and node.value == "stacked_rb":
                uses.append((node, "stacked_rb"))
            elif (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                  and node.id in ("_RB_SHIFT", "_RB_WEIGHT", "_RB_MEMBERS")):
                uses.append((node, node.id))
        self.assertGreater(len(uses), 8)
        bad = [(n.lineno, label) for n, label in uses if not guarded(n)]
        self.assertEqual(bad, [], f"rollout code reachable with the key absent: {bad}")
        # the three sites that matter, named: the subset stack, the fast capture, the graph/eager capture
        self.assertIn('_state["stacked_rb"] = stacked_rb', self._body("def _get_model("))
        for fn in ("def _fused_fast(", "def _fused("):
            self.assertIn("    if _RBLEND is not None:\n", self._body(fn), fn)

    def test_the_key_is_validated_at_import(self):
        chk = self._fn("_check_rollout_blend")
        self.assertEqual(chk({"shift": 10, "weight": 0.5, "members": [0, 1]}, 5), (10, 0.5, [0, 1]))
        self.assertEqual(chk({"shift": 1, "weight": 1.0, "members": [4]}, 5), (1, 1.0, [4]))
        self.assertEqual(chk({"shift": 19.0, "weight": 1, "members": [1, 0]}, 2), (19, 1.0, [1, 0]))
        self.assertEqual(chk({"shift": 5, "weight": 0.25, "members": [0]}, 1), (5, 0.25, [0]))
        good = {"shift": 10, "weight": 0.5, "members": [0, 1]}
        bad = [dict(good, shift=0), dict(good, shift=20), dict(good, shift=10.5), dict(good, shift=True),
               dict(good, shift="10"), dict(good, shift=None),
               dict(good, weight=0.0), dict(good, weight=-0.1), dict(good, weight=1.01),
               dict(good, weight=float("nan")), dict(good, weight=float("inf")), dict(good, weight=True),
               dict(good, weight="0.5"),
               dict(good, members=[]), dict(good, members=[0, 0]), dict(good, members=[5]),
               dict(good, members=[-1]), dict(good, members=[True]), dict(good, members=[0.0]),
               dict(good, members="0"), dict(good, members=0),
               {"shift": 10, "weight": 0.5}, dict(good, alpha=1.0), [10, 0.5, [0]], None]
        for b in bad:
            with self.assertRaises(ValueError, msg=f"must be refused at import: {b!r}"):
                chk(b, 5)
        with self.assertRaises(ValueError):
            chk(dict(good, members=[1]), 1)             # a single-model zip has member 0 only
        # the import block wires the member count from ensemble.json (1 for a single-model zip)
        with self.assertRaises(ValueError):
            self._import_block({"rollout_blend": dict(good, members=[0, 1])}, None)
        self._import_block({"rollout_blend": dict(good, members=[0])}, None)

    def test_fast_path_order_is_bounds_blend_fluct_anchor_inside_the_capture(self):
        ff = self._body("def _fused_fast(")
        i_lower = ff.index("lower = (base - half)")
        i_upper = ff.index("upper = (base + up)")
        i_guard = ff.index("    if _RBLEND is not None:\n        mean = _rollout_pass_t(mean, static_x)")
        i_blend = ff.index("mean = _rollout_pass_t(mean, static_x)")
        i_fluct = ff.index("mean = _inflate_fluct_t(mean)")
        i_anchor = ff.index("mean = _anchor_mean_t(mean, static_x)")
        i_ret = ff.index("return mean, lower, upper, ok")
        self.assertLess(max(i_lower, i_upper), i_guard, "bounds from p0 first")
        self.assertLess(i_guard, i_blend)
        self.assertLess(i_blend, i_fluct, "blend before fluct_scale")
        self.assertLess(i_fluct, i_anchor, "fluct_scale before mean_anchor")
        self.assertLess(i_anchor, i_ret)
        self.assertEqual(ff.count("_rollout_pass_t("), 1, "exactly once per path")
        # the prediction is no longer implied finite by the bounds: checked, and only when the key is on
        self.assertIn("if _CENTRE or _RBLEND is not None:", ff)
        # capture-safe: nothing in the second pass syncs with the host or allocates per call on the host
        for name in ("def _rollout_pass_t(", "def _rollout_input_t(", "def _rollout_blend_t("):
            body = self._body(name)
            for sync in (".item()", ".cpu()", ".numpy()", "synchronize", "nonzero", "tolist", "_build_stacked"):
                self.assertNotIn(sync, body, f"{name} {sync}")
        self.assertIn('_state["stacked_rb"](x2)', self._body("def _rollout_pass_t("))
        # the subset stack is built once, in _get_model, never in a per-call function
        for fn in ("def _fused_fast(", "def _fused(", "def _predict_fast(", "def _rollout_pass_t("):
            self.assertNotIn("_build_stacked", self._body(fn), fn)
        gm = self._body("def _get_model(")
        self.assertIn("_build_stacked([members[i] for i in _RB_MEMBERS], cfg, device)", gm)
        self.assertLess(gm.index('_state["stacked_rb"] = stacked_rb'),
                        gm.index("_state.update(model=model"), "built before the model is committed")

    def test_eager_path_blends_after_the_head_keeps_p0_as_centre_and_before_fluct(self):
        fu = self._body("def _fused(")
        i_head = fu.index("half = _head_half_torch(base, std, static_x, device) if _HEAD else None")
        i_guard = fu.index("    if _RBLEND is not None:\n")
        i_centre = fu.index("centre = mean ")
        i_blend = fu.index("mean = _rollout_pass_t(mean, static_x)")
        i_ret = fu.index("return mean, std, half, centre, ok")
        self.assertLess(i_head, i_guard, "the head reads p0")
        self.assertLess(i_guard, i_centre)
        self.assertLess(i_centre, i_blend, "p0 is kept as the interval's centre before the blend replaces mean")
        self.assertLess(i_blend, i_ret)
        self.assertEqual(fu.count("_rollout_pass_t("), 1)
        self.assertIn("if centre is None:", fu[i_guard:i_blend], "a CEN centre is kept, not overwritten")
        self.assertNotIn("_inflate_fluct", fu)
        self.assertNotIn("_anchor_mean", fu)
        # the numpy reference path (RPDE_NUMPY_HEAD=1): same order, p0 saved as the centre first
        pf = self._body("def _predict_fields(")
        self.assertLess(pf.index("cen = out.copy()"), pf.index('out[..., :cfg["c_out"]] = _rollout_blend('))
        self.assertLess(pf.index('out[..., :cfg["c_out"]] = _rollout_blend('), pf.index('_state["centre_mean"] = cen\n    return out, dis'))
        # predict() itself is unchanged: bounds from centre_mean (= p0), then fluct, then anchor on the blend
        body = self._body("def predict(")
        rets = [l for l in body.splitlines() if l.strip().startswith('return {"prediction"')]
        self.assertEqual(len(rets), 1)
        self.assertIn('_anchor_mean(_inflate_fluct(prediction), input_array)', rets[0])
        self.assertIn('base = _state.get("centre_mean")', body)

    def test_predict_bounds_are_p0s_and_the_blend_gets_fluct_then_anchor(self):
        """Run the SHIPPED predict() body on the graph/eager route's outputs as _predict_fields hands them
        over under the key: prediction = the blend, centre_mean = p0, graph_half = the head's offsets."""
        import numpy as np
        rng = np.random.default_rng(7)
        N, T, H, W = 2, 20, 3, 4
        p0 = rng.normal(0.2, 0.1, (N, T, H, W, 3)).astype(np.float32)
        p2 = rng.normal(0.2, 0.1, (N, T, H, W, 3)).astype(np.float32)
        blend = self._fn("_rollout_blend")(p0, p2, 10, 0.5)
        x = rng.uniform(0.2, 1.0, (N, 20, H, W, 3)).astype(np.float32); x[..., 2] = 0
        halfw = np.array([0.009, 0.004, 0.02], np.float32)
        gh = rng.uniform(0.001, 0.02, (N, T, H, W, 2)).astype(np.float32)
        ns = {"np": np, "_FAST": False, "_fast_out": {"off": True}, "_HEAD": True, "C": 3,
              "_HALF_WIDTH": halfw, "_HEAD_NOUT": 2,
              "_state": {"graph_half": gh, "device": "cpu", "centre_mean": p0},
              "_inflate_fluct": _real_inflate(np, 1.04), "_anchor_mean": _real_anchor(np, 0.1),
              "_predict_fields": lambda a: (blend, None)}
        exec(compile(self._body("def predict("), "submission_flower.py", "exec"), ns)
        o = ns["predict"](x)
        half = np.empty_like(p0); half[..., :2] = gh; half[..., 2] = halfw[2]
        np.testing.assert_array_equal(o["lower"], p0 - half)
        np.testing.assert_array_equal(o["upper"], p0 + half)
        want = _real_anchor(np, 0.1)(_real_inflate(np, 1.04)(blend), x)
        np.testing.assert_array_equal(o["prediction"], want)
        # and the reference arm: blend -> alpha as variance_inflation --rollout-alpha writes it -> anchor
        import variance_inflation as vi
        p = vi.rollout_blend(p0, p2, 10, 0.5)
        m_b = p.mean(axis=1, keepdims=True)
        p = m_b + np.float32(1.04) * (p - m_b)
        np.testing.assert_array_equal(_real_inflate(np, 1.04)(blend), p)

    def test_predict_fast_return_stays_bare(self):
        pf = self._body("def _predict_fast(")
        self.assertIn('return {"prediction": mean, "lower": lower, "upper": upper}', pf)
        code = "\n".join(l.split("#")[0] for l in pf.splitlines())         # comments may name them
        self.assertNotIn("_rollout", code, "the graph already blended; blending again here is the SUB27 bug")
        self.assertNotIn("_inflate_fluct(", code)
        self.assertNotIn("_anchor_mean(", code)

    def test_numpy_helpers_equal_variance_inflation_on_data_with_holes(self):
        import numpy as np
        import variance_inflation as vi          # scripts/ is on sys.path; numpy-only at import
        rin, rbl = self._fn("_rollout_input"), self._fn("_rollout_blend")
        rng = np.random.default_rng(1)
        N, T, H, W, C = 3, 20, 4, 6, 3
        x = rng.uniform(0.2, 1.5, size=(N, T, H, W, C)).astype(np.float32)
        x[..., 1] = rng.uniform(-0.5, 0.5, size=(N, T, H, W))
        x[..., 2] = 0.0
        x[0, -1, 1, 2, :] = 0.0                  # holes in the LAST input frame
        x[2, -1, 3, :, :] = 0.0
        x[1, :5, 0, 0, :] = 0.0                  # a hole in an earlier frame only: not a hole for fb
        p0 = rng.normal(size=(N, T, H, W, C)).astype(np.float32)
        p0[0, 0, 0, 0, 0] = -0.0
        p2 = rng.normal(size=(N, T, H, W, C)).astype(np.float32)
        p2[1, 0, 1, 1, 1] = -0.0
        x_keep = x.copy(); p0_keep = p0.copy()
        for S in (1, 5, 10, 19):
            got = rin(x, p0, S)
            want = vi.rollout_input(x, p0, S)
            self.assertEqual(got.dtype, np.float32)
            self.assertEqual(got.shape, want.shape)
            self.assertEqual(got.tobytes(), want.tobytes(), f"rollout_input S={S}")
            self.assertTrue(np.all(got[0, -S:, 1, 2, :2] == 0) and not np.any(np.signbit(got[0, -S:, 1, 2, :2])))
            for Wt in (0.25, 0.5, 0.75, 1.0):
                g = rbl(p0, p2, S, Wt)
                w = vi.rollout_blend(p0, p2, S, Wt)
                self.assertEqual(g.tobytes(), w.tobytes(), f"rollout_blend S={S} W={Wt}")
                self.assertEqual(g[:, :S].tobytes(), p0[:, :S].tobytes(), "frames [0, S) untouched")
        self.assertEqual(x.tobytes(), x_keep.tobytes(), "the input is read, never written")
        self.assertEqual(p0.tobytes(), p0_keep.tobytes())
        # mask_channel: the module feeds its own mask-processed input; predict() in the reference run
        # rewrites x2's channel 2 as (u != 0), so the fed-back frames' channel 2 is (fb_u != 0)
        xm = x.copy(); xm[..., 2] = (xm[..., 0] != 0).astype(np.float32)
        ref = vi.rollout_input(x, p0, 7); ref[..., 2] = (ref[..., 0] != 0).astype(np.float32)
        self.assertEqual(rin(xm, p0, 7, True).tobytes(), ref.tobytes())
        # a 2-channel input (c_in = 2) has nothing to carry
        self.assertEqual(rin(x[..., :2], p0, 7).tobytes(), vi.rollout_input(x, p0, 7)[..., :2].tobytes())

    def test_torch_twins_match_the_numpy_twins(self):
        """CPU torch, no model: the device helpers the fast and graph paths run, against the numpy ones.
        Imports torch -- run it with the suite (Slurm / workstation), not on the login node."""
        import numpy as np
        import torch
        rin, rbl = self._fn("_rollout_input"), self._fn("_rollout_blend")
        rin_t = self._fn("_rollout_input_t", torch=torch)
        rbl_t = self._fn("_rollout_blend_t", torch=torch)
        rng = np.random.default_rng(2)
        N, T, H, W = 2, 20, 4, 6
        x = rng.uniform(0.2, 1.5, size=(N, T, H, W, 3)).astype(np.float32); x[..., 2] = 0.0
        x[1, -1, 2, 3, :] = 0.0
        p0 = rng.normal(size=(N, T, H, W, 3)).astype(np.float32); p0[0, 0, 0, 0, 1] = -0.0
        p2 = rng.normal(size=(N, T, H, W, 3)).astype(np.float32)
        for S in (1, 10, 19):
            for mask in (False, True):
                xx = x.copy()
                if mask:
                    xx[..., 2] = (xx[..., 0] != 0).astype(np.float32)
                got = rin_t(torch.from_numpy(xx), torch.from_numpy(p0), S, mask).numpy()
                self.assertEqual(got.tobytes(), rin(xx, p0, S, mask).tobytes(), f"S={S} mask={mask}")
            for Wt in (0.3, 0.5, 1.0):
                got = rbl_t(torch.from_numpy(p0), torch.from_numpy(p2), S, Wt).numpy()
                want = rbl(p0, p2, S, Wt)
                self.assertEqual(got[:, :S].tobytes(), want[:, :S].tobytes(), f"S={S}: frames [0, S) are p0's")
                if Wt == 1.0:
                    self.assertEqual(got.tobytes(), want.tobytes(), f"S={S}: W=1 copies p2 verbatim")
                else:   # the same three float32 ops; bitwise expected, float32 rounding is the contract
                    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7, err_msg=f"S={S} W={Wt}")

    def test_rezip_exposes_it_and_the_board_log_shows_it(self):
        self.assertIn('"--rollout-blend"', self.rezip)
        self.assertIn('"--rollout-members"', self.rezip)
        self.assertIn('"--no-rollout-blend"', self.rezip)
        self.assertIn('b["rollout_blend"] = {"shift": int(S), "weight": float(W), "members": m}', self.rezip)
        self.assertIn("or a.rollout_blend is not None", self.rezip)
        self.assertIn("rollout_blend={bj.get('rollout_blend')}", self.rezip)
        self.assertEqual(self.src.count("anchor={_MEAN_ANCHOR:g} rblend={_RB_TAG} "), 2)
        self.assertEqual(self.src.count("scale={_HEAD_SCALE:g} anchor={_MEAN_ANCHOR:g} "), 2)

    def test_rezip_writes_the_key_and_refuses_what_import_would_refuse(self):
        """rezip_fast.py end to end on a stub archive (no checkpoint is opened; pure python)."""
        import subprocess, zipfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            src_zip = d / "in.zip"
            with zipfile.ZipFile(src_zip, "w") as z:
                z.writestr("bounds.json", json.dumps({"half_width": [0.01, 0.005, 0.02], "fluct_scale": 1.07}))
                z.writestr("ensemble.json", json.dumps({"members": ["model.pth"] + [f"model_{i}.pth" for i in range(1, 5)]}))
                z.writestr("submission.py", "# stale module\n")
            new_src = REPO / "submission" / "submission_flower.py"

            def run(*args):
                return subprocess.run([sys.executable, str(REPO / "submission" / "rezip_fast.py"),
                                       str(src_zip), str(d / "out.zip"), *args],
                                      capture_output=True, text=True)
            r = run("--src", str(new_src), "--rollout-blend", "10", "0.5", "--rollout-members", "0", "1",
                    "--fluct-scale", "1.04")
            self.assertEqual(r.returncode, 0, r.stderr)
            with zipfile.ZipFile(d / "out.zip") as z:
                b = json.loads(z.read("bounds.json"))
                self.assertEqual(z.read("submission.py").decode(), new_src.read_text())
            self.assertEqual(b["rollout_blend"], {"shift": 10, "weight": 0.5, "members": [0, 1]})
            self.assertIsInstance(b["rollout_blend"]["shift"], int)
            self.assertEqual(b["fluct_scale"], 1.04)
            self.assertEqual(self._fn("_check_rollout_blend")(b["rollout_blend"], 5), (10, 0.5, [0, 1]))
            for bad in (["--rollout-blend", "10", "0.5"],                               # no members
                        ["--rollout-members", "0"],                                     # no S W
                        ["--rollout-blend", "20", "0.5", "--rollout-members", "0"],
                        ["--rollout-blend", "10.5", "0.5", "--rollout-members", "0"],
                        ["--rollout-blend", "10", "0", "--rollout-members", "0"],
                        ["--rollout-blend", "10", "0.5", "--rollout-members", "0", "0"],
                        ["--rollout-blend", "10", "0.5", "--rollout-members", "5"]):
                r = run("--src", str(new_src), *bad)
                self.assertNotEqual(r.returncode, 0, f"accepted {bad}")
            r = run("--rollout-blend", "10", "0.5", "--rollout-members", "0", "1")    # stale submission.py kept
            self.assertNotEqual(r.returncode, 0, "a module that ignores the key must not get it")
            self.assertIn("does not read rollout_blend", r.stdout + r.stderr)


class RolloutFramesMembers(unittest.TestCase):
    """2026-09-24, job 5321: a `--rollout-frames K` member (core emits K frames per pass, rolled t_out/K
    times over its own predictions) could not be loaded by the shipped module -- _FlowerS2S built the core
    with dim_out = t_out*c_out and did one pass (size mismatch on core.project.2, 30 vs 60 rows). The
    module now runs src FlowerS2S.forward's rollout loop; with rollout_frames 0/absent the one-pass path
    must be exactly what shipped in every submission to date."""

    SRC = None
    # sha256 of _FlowerS2S.forward's one-pass body, from `b, _, h, w, _ = x.shape` through its final
    # `return`, as it stood at 80607e6 (before rollout support). A correct change to the one-pass path
    # breaks this on purpose: re-derive it only together with a leaderboard-identity check.
    ONE_PASS_SHA256 = "bd6a4ddfa19854dd47b150cd3894f90bcef3b4785354e9ab8e340786ce31ad14"

    def setUp(self):
        if RolloutFramesMembers.SRC is None:
            RolloutFramesMembers.SRC = (REPO / "submission" / "submission_flower.py").read_text()
        self.src = RolloutFramesMembers.SRC

    def _class_src(self, name):
        s = self.src.index(name)
        return self.src[s:self.src.index("\n\n\n", s)]

    # -- source checks (torch-free) --------------------------------------------------------
    def test_the_one_pass_body_is_byte_for_byte_what_shipped(self):
        import hashlib
        cls = self._class_src("class _FlowerS2S(")
        f = cls.index("    def forward(self, x: torch.Tensor) -> torch.Tensor:\n")
        s = cls.index("        b, _, h, w, _ = x.shape\n", f)
        e = cls.index("        return out if isinstance(u_scale, float) else out * u_scale\n", s)
        e = cls.index("\n", e) + 1
        self.assertEqual(hashlib.sha256(cls[s:e].encode()).hexdigest(), self.ONE_PASS_SHA256)
        self.assertIn('out = out.reshape(b, cfg["t_out"], cfg["c_out"], h, w)', cls[s:e])

    def test_forward_branches_to_the_rollout_only_on_a_truthy_key(self):
        import ast
        tree = ast.parse(self._class_src("class _FlowerS2S("))
        fwd = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "forward")
        self.assertEqual(ast.unparse(fwd.body[0]), "cfg = self.cfg")
        guard = fwd.body[1]
        self.assertIsInstance(guard, ast.If)
        self.assertEqual(ast.unparse(guard.test), "cfg.get('rollout_frames')")
        self.assertEqual([ast.unparse(s) for s in guard.body], ["return self._rollout(x)"])
        self.assertEqual(guard.orelse, [])
        self.assertEqual(ast.unparse(fwd.body[2]), "b, _, h, w, _ = x.shape")
        # the guard is the only mention of the rollout in forward
        self.assertEqual(sum("rollout" in ast.unparse(s) for s in fwd.body), 1)

    def test_the_core_is_t_out_frames_wide_without_the_key(self):
        cls = self._class_src("class _FlowerS2S(")
        self.assertIn('dim_out=(cfg.get("rollout_frames") or cfg["t_out"]) * cfg["c_out"],', cls)
        self.assertNotIn('dim_out=cfg["t_out"] * cfg["c_out"],', cls)

    def test_the_rollout_loop_is_static(self):
        cls = self._class_src("class _FlowerS2S(")
        body = cls[cls.index("    def _rollout(self"):cls.index("    def _rollout_pass(self")]
        self.assertIn('for _ in range(cfg["t_out"] // k):', body)
        self.assertIn("window = torch.cat((window[:, k:], fb), dim=1)", body)
        self.assertIn("fb = step\n", body)
        for bad in (".item()", ".cpu()", ".numpy()", "if step", "while "):
            self.assertNotIn(bad, body, f"{bad!r} would break capture or vmap")

    def test_grouped_mode_refuses_rather_than_running_one_pass(self):
        cls = self._class_src("class _GroupedStacked(")
        init = cls[cls.index("    def __init__("):cls.index("import copy")]
        self.assertIn('if cfg.get("rollout_frames"):', init)
        self.assertIn("raise NotImplementedError", init)

    # -- behaviour against src FlowerS2S (CPU torch) ----------------------------------------
    def _ns(self, mode="vmap"):
        import torch
        from pde_competition._vendor.flower_standalone import Flower
        ns = {"torch": torch, "Flower": Flower, "sys": sys, "_ENS_MODE": mode}
        for name in ("class _FlowerS2S(", "class _GroupedStacked(", "def _getattr_path(",
                     "def _setattr_path(", "def _build_stacked(", "def _build_alt_stacked("):
            exec(compile(self._class_src(name), "submission_flower.py", "exec"), ns)
        return ns

    def _pair(self, seed=0, drop_key=False, **kw):
        """(src FlowerS2S, shipped _FlowerS2S with the same weights, the shipped cfg dict)."""
        import torch
        from dataclasses import asdict
        from pde_competition.model import FlowerConfig, FlowerS2S
        base = dict(t_in=20, t_out=20, c_in=3, c_out=3, resolution=(8, 16), lifting_dim=8,
                    n_levels=2, num_heads=2, groups=2)
        base.update(kw)
        fc = FlowerConfig(**base)
        torch.manual_seed(seed)
        ref = FlowerS2S(fc)
        with torch.no_grad():
            for p in ref.parameters():
                p.normal_(0.0, 0.03)
        ref.set_stats(0.3 + 0.05 * torch.rand(fc.c_in).numpy(), 0.8 + 0.4 * torch.rand(fc.c_in).numpy())
        ref.eval()
        cfg = dict(asdict(fc))                      # what _get_model builds from ckpt["config"]
        cfg["resolution"] = tuple(cfg["resolution"])
        if drop_key:
            del cfg["rollout_frames"]               # checkpoints predating the field
        got = self._ns()["_FlowerS2S"](cfg)
        got.load_state_dict(ref.state_dict())       # strict: the core widths must agree
        return ref, got.eval(), cfg

    def _x(self, c, b=2, seed=1):
        import torch
        g = torch.Generator().manual_seed(seed)
        x = 0.1 * torch.randn(b, 20, 8, 16, c, generator=g)
        x[..., 0] += 0.4 + 0.2 * torch.arange(b).view(b, 1, 1, 1)   # a per-window free stream
        x[:, :, :2, :3, :] = 0.0                                     # an unmeasured patch (u == 0)
        return x

    def _assert_close(self, got, ref, what):
        self.assertEqual(tuple(got.shape), tuple(ref.shape), what)
        d = (got - ref).abs().max().item()
        self.assertLess(d, 1e-5, f"{what}: shipped module differs from src FlowerS2S by {d:.2e}")

    def test_rollout_matches_src(self):
        import torch
        for k in (10, 5):
            for delta in (False, True):
                with self.subTest(k=k, delta=delta):
                    ref, got, _ = self._pair(rollout_frames=k, predict_delta=delta)
                    x = self._x(3)
                    with torch.no_grad():
                        want, have = ref(x), got(x)
                    self.assertEqual(tuple(have.shape), (2, 20, 8, 16, 3))
                    self._assert_close(have, want, f"K={k} delta={delta}")

    def test_rollout_with_window_norm_renormalises_each_pass_like_src(self):
        import torch
        for k in (10, 5):
            with self.subTest(k=k):
                ref, got, _ = self._pair(rollout_frames=k, predict_delta=True, c_in=2, c_out=2,
                                         window_norm=True)
                x = self._x(2)
                with torch.no_grad():
                    self._assert_close(got(x), ref(x), f"K={k} window_norm")

    def test_the_one_pass_path_still_matches_src(self):
        import torch
        for drop_key in (False, True):
            with self.subTest(drop_key=drop_key):
                ref, got, _ = self._pair(predict_delta=True, drop_key=drop_key)
                x = self._x(3)
                with torch.no_grad():
                    self._assert_close(got(x), ref(x), f"one pass, key dropped={drop_key}")

    def test_the_vmapped_ensemble_runs_the_loop(self):
        import torch
        for k in (10, 5):
            with self.subTest(k=k):
                pairs = [self._pair(seed=s, rollout_frames=k, predict_delta=True) for s in (0, 1, 2)]
                ns, cfg = self._ns("vmap"), pairs[0][2]
                members = []
                for ref, _, _ in pairs:
                    m = ns["_FlowerS2S"](cfg); m.load_state_dict(ref.state_dict()); members.append(m.eval())
                stacked, how = ns["_build_stacked"](members, cfg, "cpu")
                self.assertEqual(how, "vmap", "torch.vmap could not trace the rollout loop")
                x = self._x(3)
                with torch.no_grad():
                    want = torch.stack([ref(x) for ref, _, _ in pairs])
                    self._assert_close(stacked(x), want, f"vmap K={k}")

    def test_rollout_feed_matches_src(self):
        """2026-09-26: rollout_feed=2 (feed back u, v masked by the input's holes; carry p) -- the
        shipped module must match src FlowerS2S.forward, alone and vmapped, and differ from feed-all."""
        import torch
        for k in (10, 5):
            with self.subTest(k=k):
                ref, got, cfg = self._pair(rollout_frames=k, predict_delta=True, rollout_feed=2)
                x = self._x(3)
                x[..., 2] = 0.0                                     # real data: p is identically zero
                with torch.no_grad():
                    want = ref(x)
                    self._assert_close(got(x), want, f"feed=2 K={k}")
                    legacy, _, _ = self._pair(rollout_frames=k, predict_delta=True)
                    self.assertGreater((legacy(x) - want).abs().max().item(), 1e-6,
                                       "rollout_feed changed nothing: the test cannot see the feed")
                pairs = [self._pair(seed=s, rollout_frames=k, predict_delta=True, rollout_feed=2)
                         for s in (0, 1)]
                ns = self._ns("vmap")
                members = []
                for r, _, _ in pairs:
                    m = ns["_FlowerS2S"](cfg); m.load_state_dict(r.state_dict()); members.append(m.eval())
                stacked, how = ns["_build_stacked"](members, cfg, "cpu")
                self.assertEqual(how, "vmap")
                with torch.no_grad():
                    self._assert_close(stacked(x), torch.stack([r(x) for r, _, _ in pairs]), f"vmap feed=2 K={k}")

    def test_grouped_mode_falls_back_to_vmap_not_one_pass(self):
        import torch
        pairs = [self._pair(seed=s, rollout_frames=10, predict_delta=True) for s in (0, 1)]
        ns, cfg = self._ns("grouped"), pairs[0][2]
        members = []
        for ref, _, _ in pairs:
            m = ns["_FlowerS2S"](cfg); m.load_state_dict(ref.state_dict()); members.append(m.eval())
        with self.assertRaises(NotImplementedError):
            ns["_GroupedStacked"](members, cfg, "cpu")
        stacked, how = ns["_build_stacked"](members, cfg, "cpu")
        self.assertEqual(how, "vmap")
        x = self._x(3)
        with torch.no_grad():
            self._assert_close(stacked(x), torch.stack([ref(x) for ref, _, _ in pairs]), "grouped->vmap")

    def test_an_illegal_k_is_refused_at_construction(self):
        ns = self._ns()
        cfg = dict(t_in=20, t_out=20, c_in=3, c_out=3, resolution=(8, 16), lifting_dim=8,
                   n_levels=2, num_heads=2, groups=2, rollout_frames=3)
        with self.assertRaises(ValueError):
            ns["_FlowerS2S"](cfg)
        with self.assertRaises(ValueError):
            ns["_FlowerS2S"](dict(cfg, rollout_frames=10, c_out=2))


class TemporalLiftMembers(unittest.TestCase):
    """2026-09-25: a `--temporal-lift K` member (src FlowerS2S applies a depthwise Conv1d(1, 1, K) over TIME
    to the normalised window, per (pixel, channel), before the layout fold) could not be loaded by the
    shipped module -- _FlowerS2S had no tlift, so the strict load_state_dict failed on `tlift.weight`. The
    module now builds tlift under the key and applies it in _rollout_pass (a one-pass tlift member is
    routed there with a private rollout_frames = t_out, i.e. src's frames_per_pass). Without the key the
    one-pass forward is exactly what shipped: same pinned sha256 as RolloutFramesMembers."""

    SRC = None
    # Borrowed, not inherited: subclassing would run RolloutFramesMembers' tests a second time.
    _class_src = RolloutFramesMembers._class_src
    _ns = RolloutFramesMembers._ns
    _x = RolloutFramesMembers._x
    _assert_close = RolloutFramesMembers._assert_close

    def setUp(self):
        if TemporalLiftMembers.SRC is None:
            TemporalLiftMembers.SRC = (REPO / "submission" / "submission_flower.py").read_text()
        self.src = TemporalLiftMembers.SRC

    def _fn(self, cls_name, fn_name):
        import ast
        tree = ast.parse(self._class_src(cls_name))
        return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn_name)

    # -- source checks (torch-free) --------------------------------------------------------
    def test_the_one_pass_forward_is_untouched(self):
        import ast
        import hashlib
        cls = self._class_src("class _FlowerS2S(")
        f = cls.index("    def forward(self, x: torch.Tensor) -> torch.Tensor:\n")
        s = cls.index("        b, _, h, w, _ = x.shape\n", f)
        e = cls.index("        return out if isinstance(u_scale, float) else out * u_scale\n", s)
        e = cls.index("\n", e) + 1
        self.assertEqual(hashlib.sha256(cls[s:e].encode()).hexdigest(),
                         RolloutFramesMembers.ONE_PASS_SHA256)
        fwd = self._fn("class _FlowerS2S(", "forward")
        for stmt in fwd.body:
            code = ast.unparse(stmt)
            self.assertNotIn("tlift", code)
            self.assertNotIn("temporal", code)

    def test_tlift_is_built_only_under_its_key_and_cfg_is_copied_not_mutated(self):
        import ast
        init = self._fn("class _FlowerS2S(", "__init__")
        lines = [ast.unparse(s) for s in init.body]
        self.assertIn("tl = cfg.get('temporal_lift') or 0", lines)
        guard = next(s for s in init.body if isinstance(s, ast.If) and ast.unparse(s.test) == "tl")
        inner = [ast.unparse(s) for s in guard.body]
        self.assertIn("self.tlift = torch.nn.Conv1d(1, 1, tl, padding=tl // 2, bias=False)", inner)
        # the only self.tlift / self.cfg-reassignment in __init__ live under the guard
        outside = [ast.unparse(s) for s in init.body if s is not guard]
        self.assertFalse(any("tlift" in c for c in outside))
        self.assertEqual(sum(c.startswith("self.cfg =") for c in outside), 1)    # the original `self.cfg = cfg`
        # the guard body also holds the odd-kernel ValueError check (an earlier If); the ROUTING If is the one on `not k`
        route = next((s for s in guard.body if isinstance(s, ast.If) and ast.unparse(s.test) == "not k"), None)
        self.assertIsNotNone(route, "no `if not k:` routing branch under the temporal_lift guard")
        self.assertEqual([ast.unparse(s) for s in route.body],
                         ["self.cfg = dict(cfg, rollout_frames=cfg['t_out'])"])
        # nothing in __init__ writes INTO cfg (it is _state["cfg"], shared with the stack builders)
        for node in ast.walk(init):
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                self.assertNotEqual(ast.unparse(node.value), "cfg", ast.unparse(node))

    def test_the_lift_sits_between_centring_and_the_fold_in_the_pass(self):
        import ast
        body = self._fn("class _FlowerS2S(", "_rollout_pass").body
        code = [ast.unparse(s) for s in body]
        norm = next(i for i, c in enumerate(code) if c.startswith("z = ((x if isinstance(u_scale, float)"))
        fold = next(i for i, c in enumerate(code) if c.startswith("z = z.permute(0, 1, 4, 2, 3)"))
        core = code.index("out = self.core(z)")
        lift = [i for i, s in enumerate(body)
                if isinstance(s, ast.If) and ast.unparse(s.test) == "cfg.get('temporal_lift')"]
        self.assertEqual(lift, [norm + 1], "tlift must follow centring directly")
        self.assertEqual(fold, norm + 2)
        self.assertLess(fold, core)
        block = code[lift[0]]
        self.assertIn("z.permute(0, 2, 3, 4, 1).reshape(-1, 1, cfg['t_in'])", block)
        self.assertIn("self.tlift(zz).reshape(b, h, w, cc, cfg['t_in'])", block)
        self.assertIn("z = zz.permute(0, 4, 1, 2, 3)", block)
        for bad in (".item()", ".cpu()", ".numpy()", "while "):
            self.assertNotIn(bad, block, f"{bad!r} would break capture or vmap")
        # the one and only application of tlift in the module
        self.assertEqual(self.src.count("self.tlift("), 1)

    def test_rollout_short_circuits_a_single_pass(self):
        import ast
        body = self._fn("class _FlowerS2S(", "_rollout").body
        short = next(s for s in body if isinstance(s, ast.If))
        self.assertEqual(ast.unparse(short.test), "k == cfg['t_out']")
        self.assertEqual([ast.unparse(s) for s in short.body], ["return self._rollout_pass(x)"])

    def test_grouped_mode_refuses_temporal_lift(self):
        cls = self._class_src("class _GroupedStacked(")
        init = cls[cls.index("    def __init__("):cls.index("import copy")]
        self.assertIn('if cfg.get("temporal_lift"):', init)
        self.assertIn('raise NotImplementedError("grouped ensemble mode does not implement temporal_lift")', init)

    # -- behaviour against src FlowerS2S (CPU torch) ----------------------------------------
    def _pair(self, tl, seed=0, **kw):
        """(src FlowerS2S with a clearly NON-identity tlift, shipped _FlowerS2S with the same weights, cfg)."""
        import torch
        from dataclasses import asdict
        from pde_competition.model import FlowerConfig, FlowerS2S
        base = dict(t_in=20, t_out=20, c_in=3, c_out=3, resolution=(8, 16), lifting_dim=8,
                    n_levels=2, num_heads=2, groups=2, temporal_lift=tl)
        base.update(kw)
        fc = FlowerConfig(**base)
        torch.manual_seed(seed)
        ref = FlowerS2S(fc)
        with torch.no_grad():
            for p in ref.parameters():
                p.normal_(0.0, 0.03)
            if tl:
                ref.tlift.weight.normal_(0.0, 0.25)
                ref.tlift.weight[0, 0, tl // 2] += 1.0
        ref.set_stats(0.3 + 0.05 * torch.rand(fc.c_in).numpy(), 0.8 + 0.4 * torch.rand(fc.c_in).numpy())
        ref.eval()
        cfg = dict(asdict(fc))
        cfg["resolution"] = tuple(cfg["resolution"])
        snapshot = dict(cfg)
        got = self._ns()["_FlowerS2S"](cfg)
        self.assertEqual(cfg, snapshot, "_FlowerS2S mutated the caller's cfg dict")
        got.load_state_dict(ref.state_dict())       # strict: tlift.weight must be expected
        return ref, got.eval(), cfg

    def test_one_pass_matches_src(self):
        import torch
        for tl in (5, 9):
            for delta in (False, True):
                with self.subTest(tl=tl, delta=delta):
                    ref, got, cfg = self._pair(tl, predict_delta=delta)
                    self.assertEqual(cfg["rollout_frames"], 0)
                    self.assertEqual(got.cfg["rollout_frames"], 20)
                    x = self._x(3)
                    with torch.no_grad():
                        have = got(x)
                        self.assertEqual(tuple(have.shape), (2, 20, 8, 16, 3))
                        self._assert_close(have, ref(x), f"tl={tl} delta={delta}")

    def test_the_lift_is_not_a_no_op(self):
        import copy
        import torch
        ref, got, _ = self._pair(5, predict_delta=True)
        ident = copy.deepcopy(ref)
        with torch.no_grad():
            ident.tlift.weight.zero_()
            ident.tlift.weight[0, 0, 2] = 1.0       # the delta kernel == no lift at all
            x = self._x(3)
            d = (got(x) - ident(x)).abs().max().item()
        self.assertGreater(d, 1e-3, "the test kernel is too close to identity to prove anything")

    def test_window_norm_matches_src(self):
        import torch
        ref, got, _ = self._pair(5, predict_delta=True, c_in=2, c_out=2, window_norm=True)
        x = self._x(2)
        with torch.no_grad():
            self._assert_close(got(x), ref(x), "tl=5 window_norm")

    def test_with_rollout_the_lift_runs_every_pass_like_src(self):
        import torch
        for k in (10, 5):
            with self.subTest(k=k):
                ref, got, cfg = self._pair(5, rollout_frames=k, predict_delta=True)
                self.assertIs(got.cfg, cfg)          # a real rollout member keeps the shared cfg
                x = self._x(3)
                with torch.no_grad():
                    self._assert_close(got(x), ref(x), f"tl=5 K={k}")

    def test_without_the_key_nothing_changes(self):
        ns = self._ns()
        cfg = dict(t_in=20, t_out=20, c_in=3, c_out=3, resolution=(8, 16), lifting_dim=8,
                   n_levels=2, num_heads=2, groups=2)
        for c in (cfg, dict(cfg, temporal_lift=0)):
            m = ns["_FlowerS2S"](c)
            self.assertIs(m.cfg, c)
            self.assertFalse(hasattr(m, "tlift"))
            self.assertFalse(any("tlift" in k for k in m.state_dict()))

    def test_the_vmapped_stack_matches_src(self):
        import torch
        for tl in (5, 9):
            for delta in (False, True):
                with self.subTest(tl=tl, delta=delta):
                    pairs = [self._pair(tl, seed=s, predict_delta=delta) for s in (0, 1, 2)]
                    ns, cfg = self._ns("vmap"), pairs[0][2]
                    members = []
                    for ref, _, _ in pairs:
                        m = ns["_FlowerS2S"](cfg); m.load_state_dict(ref.state_dict()); members.append(m.eval())
                    stacked, how = ns["_build_stacked"](members, cfg, "cpu")
                    self.assertEqual(how, "vmap", "torch.vmap could not trace the temporal lift")
                    x = self._x(3)
                    with torch.no_grad():
                        want = torch.stack([ref(x) for ref, _, _ in pairs])
                        self._assert_close(stacked(x), want, f"vmap tl={tl} delta={delta}")

    def test_grouped_mode_falls_back_to_vmap(self):
        import torch
        pairs = [self._pair(5, seed=s, predict_delta=True) for s in (0, 1)]
        ns, cfg = self._ns("grouped"), pairs[0][2]
        members = []
        for ref, _, _ in pairs:
            m = ns["_FlowerS2S"](cfg); m.load_state_dict(ref.state_dict()); members.append(m.eval())
        with self.assertRaises(NotImplementedError):
            ns["_GroupedStacked"](members, cfg, "cpu")
        stacked, how = ns["_build_stacked"](members, cfg, "cpu")
        self.assertEqual(how, "vmap")
        x = self._x(3)
        with torch.no_grad():
            self._assert_close(stacked(x), torch.stack([ref(x) for ref, _, _ in pairs]), "grouped->vmap")

    def test_an_illegal_kernel_is_refused(self):
        ns = self._ns()
        cfg = dict(t_in=20, t_out=20, c_in=3, c_out=3, resolution=(8, 16), lifting_dim=8,
                   n_levels=2, num_heads=2, groups=2)
        for tl in (4, -3):
            with self.subTest(tl=tl), self.assertRaises(ValueError):
                ns["_FlowerS2S"](dict(cfg, temporal_lift=tl))


class GcFreezeAtImportEnd(unittest.TestCase):
    """GCF (2026-09-26): gc is collected, frozen and disabled after the import-time warm-up and before the
    untimed-import marker, inside a try that cannot break the import."""

    def test_gc_freeze_sits_between_warm_up_and_import_end(self):
        src = (REPO / "submission" / "submission_flower.py").read_text()
        w = src.index("\n_warm_up()\n")
        g = src.index("_gc.freeze()", w)
        e = src.index("_T_IMPORT_END = time.perf_counter()", w)
        self.assertLess(w, g)
        self.assertLess(g, e)
        block = src[w:e]
        for needed in ("try:", "import gc as _gc", "_gc.collect()", "_gc.disable()", "except Exception"):
            self.assertIn(needed, block)

