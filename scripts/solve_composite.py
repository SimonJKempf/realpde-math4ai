#!/usr/bin/env python
"""What our submissions do and do not tell us about the composite score.

    python scripts/solve_composite.py

The leaderboard combines five subscores into ``final_score`` by a rule the
organizers do not publish -- ``kit/scoring.py`` says so and stops at the five.
Every submission is one equation toward recovering it. This reads
``experiments/results/submissions.tsv`` and reports what the evidence now
supports.

It deliberately does NOT print a confident answer from too little data. Five
weights plus an intercept needs six independent rows; below that it says what
is pinned, what is not, and which submission would be most informative next.

CAUTION -- read before believing any fit this prints. Row COUNT is not the
binding constraint; row SPREAD is. Across every submission so far, sps varies by
22 points while rel_l2 varies by 0.38, tke 2.26, mvpe 0.90 and time 1.17. All
the information lies along one axis, so any weight this reports for the other
four subscores is fitted to noise, and wildly different formulas fit every row
to <0.05 while disagreeing everywhere we have no data. Reaching six rows does
not fix that -- six rows that all differ only in sps are still one equation.

Two sanity checks before trusting a candidate rule:
  * it must satisfy f(c,c,c,c,c) = c, which rules out any free intercept;
  * judge it by leave-one-out, never by fit residual -- a 6-parameter model fits
    5 rows exactly and still predicts worse than a 1-parameter one.

Nothing here writes to pde_competition.composite. Updating WEIGHTS there stays
a manual, reviewed step -- this only prints what it would say.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

from pde_competition.composite import SUBSCORES, WEIGHTS
from pde_competition.paths import RESULTS

TSV = RESULTS / "submissions.tsv"


def load(path: Path = TSV) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, labels) for the informative rows.

    Rows that scored 0 on everything are dropped: a failed submission is a real
    event worth recording, but ``0 = w . 0`` constrains nothing and would drag
    any least-squares fit toward the origin.
    """
    rows, labels = [], []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if f[0] == "id":
            head = f
            continue
        rec = dict(zip(head, f))
        vals = [float(rec[k]) for k in SUBSCORES]
        if not any(vals):
            print(f"  skipping #{rec['id']} ({rec['name']}): all-zero, no information")
            continue
        rows.append(vals + [float(rec["final"])])
        labels.append(f"#{rec['id']} {rec['name']}")
    a = np.array(rows, float)
    return a[:, :5], a[:, 5], labels


def pinned_by_pairs(X: np.ndarray, y: np.ndarray, labels: list[str]) -> None:
    """Weight intervals from row PAIRS that differ in only one or two subscores.

    This is the strongest evidence available and needs no global assumption:
    if two submissions differ in one place, the difference in final_score is
    attributable there whatever the functional form, as long as it is locally
    smooth. It is what pins w_sps.
    """
    print("\n== Pinned by row pairs (no global model assumed) ==")
    found = False
    for i, j in itertools.combinations(range(len(X)), 2):
        d, dy = X[j] - X[i], y[j] - y[i]
        moved = [k for k in range(5) if abs(d[k]) > 1e-9]
        if not 1 <= len(moved) <= 2:
            continue
        found = True
        terms = " + ".join(f"{d[k]:+.4f}*w_{SUBSCORES[k]}" for k in moved)
        print(f"\n  {labels[i]}  ->  {labels[j]}")
        print(f"    {terms} = {dy:+.6f}")
        if len(moved) == 1:
            k = moved[0]
            print(f"    => w_{SUBSCORES[k]} = {dy / d[k]:.4f}  (exact)")
        else:
            a, b = moved
            print(f"    => w_{SUBSCORES[b]} as a function of w_{SUBSCORES[a]}:")
            for wa in (0.0, 0.1, 0.2, 0.3):
                print(f"         w_{SUBSCORES[a]}={wa:.2f} -> "
                      f"w_{SUBSCORES[b]}={(dy - d[a] * wa) / d[b]:.4f}")
    if not found:
        print("  none -- no two submissions differ in fewer than three subscores.")
        print("  Re-submitting one model with ONLY the bounds changed creates such a pair.")


def linear_family(X: np.ndarray, y: np.ndarray) -> None:
    """Least squares under sum(w)=1, plus the nonnegativity test.

    The nonnegativity test is the one that matters. With few rows the system is
    underdetermined, so an exact fit is unsurprising and proves nothing. But if
    NO member of the exact-fit family has all weights >= 0, then no convex
    weighting reproduces the leaderboard, and the true rule must have a
    nonlinearity, an intercept, or a term we are not modelling.
    """
    print("\n== Linear family under sum(w) = 1 ==")
    A = np.vstack([X, np.ones(5)])
    b = np.append(y, 1.0)
    w, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    resid = np.abs(X @ w - y).max()
    print(f"  rank {rank} of 5, exact-fit family is {5 - rank}-dimensional")
    print(f"  min-norm solution: "
          + "  ".join(f"{k} {v:+.4f}" for k, v in zip(SUBSCORES, w)))
    print(f"  max |residual| {resid:.2e}"
          + ("  (fits exactly -- expected while underdetermined)"
             if resid < 1e-6 else "  (does NOT fit: the rule is not linear)"))

    null = np.linalg.svd(A)[2][rank:]
    if len(null) == 0:
        print(f"  unique solution; nonnegative: {(w >= 0).all()}")
        return
    if len(null) > 1:
        print(f"  {len(null)}-dimensional family -- too underdetermined to test "
              f"nonnegativity cheaply; more rows needed.")
        return
    # 1-D family w(t) = w0 + t*n: intersect the per-weight nonnegativity
    # intervals. Empty means no convex weighting exists.
    n = null[0]
    lo, hi = -np.inf, np.inf
    for i in range(5):
        if abs(n[i]) < 1e-12:
            if w[i] < 0:
                lo, hi = 1.0, -1.0
        elif n[i] > 0:
            lo = max(lo, -w[i] / n[i])
        else:
            hi = min(hi, -w[i] / n[i])
    if lo > hi:
        print(f"  NONNEGATIVE MEMBER: none (interval empty, [{lo:.4f}, {hi:.4f}])")
        print("  => no convex weighting fits these rows EXACTLY. Weak evidence:")
        print("     demanding an exact fit from few rows is a strong requirement,")
        print("     and an approximate nonnegative fit may well exist. Do not")
        print("     conclude nonlinearity from this alone.")
    else:
        print(f"  nonnegative members exist for t in [{lo:.4f}, {hi:.4f}]:")
        for t in (lo, hi):
            print("    " + "  ".join(f"{k} {v:.4f}"
                                     for k, v in zip(SUBSCORES, w + t * n)))


def next_submission(X: np.ndarray) -> None:
    """Which subscore is least constrained, so a slot buys evidence."""
    print("\n== What to spend the next submission on ==")
    spread = X.max(0) - X.min(0)
    order = np.argsort(spread)
    print("  observed range per subscore (a subscore that never moves is")
    print("  perfectly confounded with the intercept):")
    for k in order:
        print(f"    {SUBSCORES[k]:7s} {spread[k]:7.3f}"
              + ("   <- least informative so far" if k == order[0] else ""))
    print(f"\n  Most valuable next row: one that moves {SUBSCORES[order[0]]} "
          f"while holding the others still.")


def simplex_grid(k: int, n: int) -> np.ndarray:
    """All nonnegative integer k-vectors summing to n, one per row."""
    if k == 1:
        return np.array([[n]])
    out = []
    for i in range(n + 1):
        sub = simplex_grid(k - 1, n - i)
        out.append(np.c_[np.full(len(sub), i), sub])
    return np.vstack(out)


def power_mean_family(X: np.ndarray, y: np.ndarray, labels: list[str],
                      step: int = 50) -> None:
    """Search concave means  M_p(S;w) = (sum w_i S_i^p)^(1/p),  w >= 0, sum w = 1.

    Unlike the linear fits above this obeys ``f(c,c,c,c,c) = c``, which any real
    aggregation rule must. It is the only family found so far that fits every
    row; see experiments/notes/analyses/composite-score.md.

    Reports the FEASIBLE SET rather than a point estimate. With four free
    weights plus an exponent against six rows the best-fit point is nearly
    saturated and means little; what the rows genuinely constrain is the range
    each parameter can take while still fitting, and that separates the pinned
    parameters from the unidentified ones.
    """
    print("\n== Concave power-mean family (obeys f(c,...,c) = c) ==")
    W = simplex_grid(len(SUBSCORES), step) / step
    ps = np.round(np.arange(0.30, 1.3001, 0.02), 3)
    S = X / 100.0

    # pred[i, j, r] = prediction for exponent ps[i], weights W[j], row r
    pred = np.empty((len(ps), len(W), len(X)))
    for i, p in enumerate(ps):
        pred[i] = (100.0 * (((S ** p) @ W.T) ** (1.0 / p))).T
    resid = np.abs(pred - y).max(2)

    i, j = np.unravel_index(resid.argmin(), resid.shape)
    p_hat, w_hat = ps[i], W[j]
    print(f"  grid {len(W):,} weight vectors x {len(ps)} exponents "
          f"(weights on a 1/{step} grid)")
    print(f"  best fit: p = {p_hat:.2f}   max|residual| = {resid[i, j]:.4f}")
    print("    " + "  ".join(f"{k} {v:.2f}" for k, v in zip(SUBSCORES, w_hat)))

    for tol in (0.05, 0.10, 0.25):
        m = resid <= tol
        n = int(m.sum())
        print(f"\n  feasible set at max|residual| <= {tol}: {n:,} points")
        if n == 0:
            continue
        pi, wi = np.nonzero(m)
        print(f"    {'p':10s} {ps[pi].min():+.2f} .. {ps[pi].max():+.2f}")
        for c, k in enumerate(SUBSCORES):
            lo, hi = W[wi, c].min(), W[wi, c].max()
            flag = "  <== pinned" if hi - lo <= 0.06 else ""
            print(f"    w_{k:8s} {lo:.2f} .. {hi:.2f}{flag}")

    # Leave-one-out: the only honest check that this predicts rather than fits.
    print("\n  leave-one-out (refit p and w on the other rows):")
    errs = []
    for k in range(len(X)):
        keep = np.ones(len(X), bool)
        keep[k] = False
        r = np.abs(pred[:, :, keep] - y[keep]).max(2)
        a, b = np.unravel_index(r.argmin(), r.shape)
        e = pred[a, b, k] - y[k]
        errs.append(e)
        print(f"    {labels[k]:22s} p {ps[a]:.2f}  w_sps {W[b][-1]:.2f}  "
              f"predicted {pred[a, b, k]:8.4f}  actual {y[k]:8.4f}  {e:+.4f}")
    print(f"    LOO rms {np.sqrt(np.mean(np.square(errs))):.4f}   "
          f"(equal-weight mean: {np.sqrt(np.mean(np.square(X.mean(1) - y))):.4f})")

    # Marginal rates at the most recent row -- what a subscore point is worth.
    x = X[-1]
    m = float((((x / 100.0) ** p_hat) @ w_hat) ** (1.0 / p_hat) * 100.0)
    print(f"\n  d(final)/d(subscore) at {labels[-1]} (best-fit point):")
    for c, k in enumerate(SUBSCORES):
        d = w_hat[c] * (x[c] / 100.0) ** (p_hat - 1) * (m / 100.0) ** (1 - p_hat)
        print(f"    {k:8s} S = {x[c]:7.3f}   {d:+.4f}")
    print("  Trust only the subscores whose weight came out pinned above.")


def verify_solved(X: np.ndarray, y: np.ndarray, labels: list[str]) -> bool:
    """Check every row against the SOLVED formula in pde_competition.composite.

    The rule was identified on 2026-09-10 (weights 0.5/0.1/0.1/0.1/0.2, power
    mean exponent 0.6854). This is now a regression test, not a fit: a new row
    that does not land on the formula means the organizers changed the rule,
    and everything downstream needs revisiting. Say so loudly.
    """
    from pde_competition.composite import P, WEIGHTS, estimate

    print("\n== Verifying the solved formula ==")
    print("  final = ( " + " + ".join(f"{WEIGHTS[k]:.1f}*{k}**P" for k in SUBSCORES)
          + f" ) ** (1/P),   P = {P}")
    worst = 0.0
    print(f"\n  {'row':22s} {'actual':>12} {'formula':>12} {'residual':>11}")
    for lab, x, t in zip(labels, X, y):
        got = estimate(*x)
        worst = max(worst, abs(got - t))
        print(f"  {lab:22s} {t:12.6f} {got:12.6f} {got - t:+11.2e}")
    ok = worst < 1e-4
    print(f"\n  max |residual| = {worst:.2e}   "
          f"(subscores are reported to 1e-6, so ~5e-7 is exact)")
    if ok:
        print("  OK -- every row lands on the formula.")
    else:
        print("  *** MISMATCH. A row does not fit the solved rule. Either the")
        print("  *** row is mistyped in submissions.tsv or the organizers")
        print("  *** changed the composite. Do not paper over this.")
    return ok


def rounded_weight_search(X: np.ndarray, y: np.ndarray) -> None:
    """How the weights were found: assume they are multiples of 0.1.

    1001 vectors instead of a continuum. That prior is what turned an
    underdetermined fit into an exact identification -- keep the search here so
    the claim stays reproducible and so a future rule change can be re-solved
    the same way.
    """
    print("\n== Round-weight search (weights as multiples of 0.1) ==")
    W = simplex_grid(len(SUBSCORES), 10) / 10.0
    S = X / 100.0
    ps = np.round(np.arange(0.05, 2.0001, 0.002), 4)
    pred = np.empty((len(ps), len(W), len(X)))
    for i, p in enumerate(ps):
        pred[i] = (100.0 * (((S ** p) @ W.T) ** (1.0 / p))).T
    resid = np.abs(pred - y).max(2)
    best_p, best_r = resid.argmin(0), resid.min(0)
    order = np.argsort(best_r)

    print(f"  {len(W)} weight vectors x {len(ps)} exponents")
    print(f"  {'':4} " + " ".join(f"{k:>7}" for k in SUBSCORES) + f"  {'p':>7} {'max|r|':>10}")
    for rank, j in enumerate(order[:5]):
        print(f"  {rank+1:>4} " + " ".join(f"{v:7.1f}" for v in W[j])
              + f"  {ps[best_p[j]]:7.3f} {best_r[j]:10.2e}")
    if len(order) > 1 and best_r[order[0]] > 0:
        print(f"\n  runner-up is {best_r[order[1]] / best_r[order[0]]:,.0f}x worse "
              f"-- the winner is unique, not the top of a plateau.")


def main() -> None:
    print(f"reading {TSV}")
    X, y, labels = load()
    print(f"{len(labels)} informative rows: " + ", ".join(labels))

    ok = verify_solved(X, y, labels)
    if ok:
        # The rule is known; the exploratory fits below only add noise.
        print("\n(The exploratory fits are skipped while the formula verifies. "
              "Run with --explore to see them.)")
        if "--explore" not in sys.argv:
            return
    rounded_weight_search(X, y)
    pinned_by_pairs(X, y, labels)
    linear_family(X, y)
    if len(labels) >= 5:
        power_mean_family(X, y, labels)

    need = 6
    print(f"\n== Fitting a full model ==")
    if len(labels) < need:
        print(f"  {len(labels)} rows, need {need} (5 weights + intercept). "
              f"Not fitting.")
        print(f"  Current placeholder in pde_competition.composite.WEIGHTS:")
        print("    " + "  ".join(f"{k} {v:.4f}" for k, v in WEIGHTS.items()))
    else:
        # Model comparison by leave-one-out: linear, linear+intercept, and a
        # min-penalised mean, which is the shape suggested by the deviation
        # from the equal-weight mean tracking the WEAKEST subscore.
        def loo(fit, predict) -> float:
            e = []
            for i in range(len(X)):
                m = np.ones(len(X), bool)
                m[i] = False
                e.append(predict(fit(X[m], y[m]), X[i]) - y[i])
            return float(np.sqrt(np.mean(np.square(e))))

        def f_lin(Xt, yt):
            return np.linalg.lstsq(np.vstack([Xt, np.ones(5)]),
                                   np.append(yt, 1.0), rcond=None)[0]

        def f_int(Xt, yt):
            return np.linalg.lstsq(np.c_[Xt, np.ones(len(Xt))], yt, rcond=None)[0]

        def f_min(Xt, yt):
            F = np.c_[Xt.mean(1), Xt.mean(1) - Xt.min(1), np.ones(len(Xt))]
            return np.linalg.lstsq(F, yt, rcond=None)[0]

        models = {
            "linear (sum w = 1)": (f_lin, lambda w, x: x @ w),
            "linear + intercept": (f_int, lambda w, x: x @ w[:5] + w[5]),
            "mean - lam*(mean-min)": (
                f_min, lambda w, x: w[0] * x.mean() + w[1] * (x.mean() - x.min()) + w[2]),
        }
        print("  leave-one-out RMSE, lower is better:")
        for name, (fit, pred) in sorted(
                models.items(), key=lambda kv: loo(*kv[1])):
            print(f"    {name:24s} {loo(fit, pred):8.4f}")
        print("\n  Best-fit weights it WOULD write into composite.WEIGHTS:")
        w = f_lin(X, y)
        print("    " + "  ".join(f"{k} {v:+.4f}" for k, v in zip(SUBSCORES, w)))
        print("  (review before copying -- negative weights mean the linear form is wrong)")

    next_submission(X)


if __name__ == "__main__":
    main()
