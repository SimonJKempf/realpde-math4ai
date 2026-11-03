"""Invariants of the composite estimate that selects best_score.pth."""
import unittest

from pde_competition import composite


class Weights(unittest.TestCase):
    def test_weights_cover_exactly_the_five_subscores(self):
        self.assertEqual(set(composite.WEIGHTS), set(composite.SUBSCORES))

    def test_equal_subscores_map_to_themselves(self):
        # Any real aggregation rule satisfies f(c,c,c,c,c) = c -- see
        # experiments/notes/analyses/composite-score.md. Ours must too.
        for c in (0.0, 50.0, 79.117, 100.0):
            self.assertAlmostEqual(composite.estimate(c, c, c, c, c), c, places=9)


class ReproducesTheLeaderboard(unittest.TestCase):
    """The formula was solved, not fitted -- so it must reproduce every row.

    These are the real leaderboard subscores and final_scores from
    experiments/results/submissions.tsv. Tolerance is 1e-5, well inside the
    5e-7 actually achieved but loose enough not to trip on float noise. If a
    future submission breaks this, the rule changed (or the row is mistyped)
    -- do NOT relax the tolerance to make it pass.
    """

    ROWS = [
        # rel_l2,    tke,       mvpe,      time,      sps,       final
        (94.168150, 74.025866, 92.836278, 92.512037, 14.079725, 72.635304),
        (94.168150, 74.025866, 92.836278, 91.347185, 32.031602, 77.791654),
        (94.503637, 73.923468, 92.398429, 91.370559, 35.596935, 78.828987),
        (94.536536, 74.636151, 92.912553, 91.430936, 36.161764, 79.116637),
        (94.154009, 76.187976, 92.015548, 91.428910, 34.724113, 78.637900),
        (94.536536, 74.636151, 92.912553, 91.397614, 36.382652, 79.169921),
    ]

    def test_every_submitted_row(self):
        for *subs, final in self.ROWS:
            self.assertAlmostEqual(composite.estimate(*subs), final, delta=1e-5)

    def test_weights_are_the_organizers_round_numbers(self):
        self.assertEqual(composite.WEIGHTS,
                         {"rel_l2": 0.5, "tke": 0.1, "mvpe": 0.1,
                          "time": 0.1, "sps": 0.2})
        self.assertAlmostEqual(sum(composite.WEIGHTS.values()), 1.0, places=12)
        self.assertEqual(composite.P, 0.6854)

    def test_concave_so_the_weakest_subscore_has_extra_leverage(self):
        # P < 1 pulls the mean toward the minimum. Spending a point on the
        # weakest subscore must beat spending it on an equally-weighted
        # stronger one -- this is why sps outranks its 0.2 weight.
        base = composite.estimate(90.0, 90.0, 90.0, 90.0, 40.0)
        lift_weak = composite.estimate(90.0, 90.0, 90.0, 90.0, 41.0) - base
        lift_strong = composite.estimate(90.0, 90.0, 90.0, 91.0, 40.0) - base
        self.assertGreater(lift_weak, 2 * lift_strong)


class FromErrors(unittest.TestCase):
    def test_lower_error_is_higher_score(self):
        base = composite.from_errors(0.08, 0.70)
        self.assertGreater(composite.from_errors(0.07, 0.70), base)
        self.assertGreater(composite.from_errors(0.08, 0.60), base)
        self.assertLess(composite.from_errors(0.09, 0.70), base)

    def test_fixed_subscores_can_be_overridden_but_not_invented(self):
        a = composite.from_errors(0.08, 0.70)
        b = composite.from_errors(0.08, 0.70, sps=composite.REFERENCE["sps"] + 10)
        self.assertGreater(b, a)
        with self.assertRaises(ValueError):
            composite.from_errors(0.08, 0.70, final=90.0)    # not a subscore at all


if __name__ == "__main__":
    unittest.main()
