"""
pareto_frontier.py
==================

Nemhauser-Ullmann (Pareto Frontier) solver for the 0/1 knapsack problem.

This implements the algorithm in `Pareto_Frontier.pdf` (Section 1):

    alpha = max_x  [ I(w^T x <= L) + rho^T x ]
          = max_x  sum_i rho_i x_i   s.t.   sum_i w_i x_i <= L,   x_i in {0,1}

with  w_i, rho_i >= 0.

Why a *frontier* and not just a single knapsack value?
------------------------------------------------------
Inside Capacity-Constrained Affinity Propagation (see `cap.py`) the availability
message alpha~_{ik} requires the knapsack *optimal value as a function of the
capacity budget* P_k^{-A}(L), queried at several capacities
(Q - w_k, Q - w_k - w_i, ...).  The Nemhauser-Ullmann frontier gives the entire
value-vs-capacity staircase in one pass, so we build it once per exemplar and
then answer every capacity query by an O(log |F|) lookup.

The output of this algorithm is provably optimal (Pareto optimality => no
dominated partial solution is ever discarded).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


class ParetoFrontier:
    """A monotone staircase of Pareto-optimal (weight, value) pairs.

    Invariants of the stored arrays (`self.w`, `self.v`):
      * sorted by weight strictly ascending,
      * value strictly ascending (a heavier point that is not strictly more
        valuable is dominated and removed),
      * always contains the empty-selection point (0.0, 0.0).
    """

    __slots__ = ("w", "v")

    def __init__(self, w: np.ndarray | None = None, v: np.ndarray | None = None):
        if w is None:
            self.w = np.array([0.0])
            self.v = np.array([0.0])
        else:
            self.w = np.asarray(w, dtype=float)
            self.v = np.asarray(v, dtype=float)

    # ------------------------------------------------------------------ build
    @staticmethod
    def build(item_w, item_r, cap: float = np.inf) -> "ParetoFrontier":
        """Build the frontier for items with weights `item_w` and prizes `item_r`.

        Items with prize <= 0 are never beneficial (they only add weight), so
        they are skipped -- this is exactly the `r = max(0, rho~)` rule used by
        CAP.  Points whose weight exceeds `cap` are pruned early to keep the
        frontier small (in CAP, `cap = Q - w_k`).
        """
        item_w = np.asarray(item_w, dtype=float)
        item_r = np.asarray(item_r, dtype=float)

        w = np.array([0.0])
        v = np.array([0.0])

        for wi, ri in zip(item_w, item_r):
            if ri <= _EPS:                       # prize 0 item -> never selected
                continue
            # Candidate set = keep item i out  OR  add item i to every point.
            cw = np.concatenate([w, w + wi])
            cv = np.concatenate([v, v + ri])

            # Capacity prune.
            keep = cw <= cap + _EPS
            cw, cv = cw[keep], cv[keep]

            # Sort by weight ascending, value descending (so equal weights keep
            # the largest value first).
            order = np.lexsort((-cv, cw))
            cw, cv = cw[order], cv[order]

            # Pareto prune: keep a point iff its value strictly exceeds the best
            # value seen at any smaller-or-equal weight.
            run_max = np.maximum.accumulate(cv)
            prev_best = np.concatenate(([-np.inf], run_max[:-1]))
            mask = cv > prev_best + _EPS
            w, v = cw[mask], cv[mask]

        return ParetoFrontier(w, v)

    # ------------------------------------------------------------------ query
    def value_at(self, L):
        """Maximum achievable prize using total weight <= L.

        Accepts a scalar or an array of budgets.  A budget L < 0 is infeasible
        (returns -inf): in CAP this means an item cannot even fit alongside the
        exemplar, which correctly drives the corresponding availability to -inf.
        """
        L = np.asarray(L, dtype=float)
        scalar = L.ndim == 0
        Lf = np.atleast_1d(L)

        out = np.empty_like(Lf)
        infeasible = Lf < -_EPS
        # searchsorted gives the rightmost frontier point with weight <= L.
        idx = np.searchsorted(self.w, Lf, side="right") - 1
        idx = np.clip(idx, 0, len(self.w) - 1)
        out[:] = self.v[idx]
        out[infeasible] = -np.inf
        return float(out[0]) if scalar else out

    def __len__(self):
        return len(self.w)

    def __repr__(self):
        return f"ParetoFrontier(points={len(self)}, max_value={self.v[-1]:.4g})"


# ---------------------------------------------------------------------------
# Head-knapsack helper used by CAP: one frontier per candidate exemplar k,
# with cheap "leave-one-out" queries P_k^{-{i,k}}(L).
# ---------------------------------------------------------------------------
class HeadKnapsack:
    """Knapsack over all nodes j != k for a fixed exemplar k.

    Provides P_k^{-A}(L) for A = {k} (exclude=None) and A = {i, k}
    (exclude=i).  Leave-one-out frontiers are only rebuilt for items that carry
    a strictly positive prize (others are absent from the frontier anyway, so
    excluding them changes nothing) and are cached.
    """

    def __init__(self, weights: np.ndarray, prizes: np.ndarray, head: int, cap: float):
        self.weights = np.asarray(weights, dtype=float)
        self.prizes = np.asarray(prizes, dtype=float)
        self.head = head
        self.cap = cap
        n = len(weights)

        # Items eligible to enter the knapsack: j != head and prize > 0.
        self._pos = np.array(
            [j for j in range(n) if j != head and self.prizes[j] > _EPS], dtype=int
        )
        self.base = ParetoFrontier.build(
            self.weights[self._pos], self.prizes[self._pos], cap
        )
        self._loo_cache: dict[int, ParetoFrontier] = {}

    def value(self, L, exclude: int | None = None):
        """P_k^{-A}(L).  exclude=i removes item i (=> A = {i, k})."""
        if exclude is None or exclude == self.head or self.prizes[exclude] <= _EPS:
            return self.base.value_at(L)
        if exclude not in self._loo_cache:
            sel = self._pos[self._pos != exclude]
            self._loo_cache[exclude] = ParetoFrontier.build(
                self.weights[sel], self.prizes[sel], self.cap
            )
        return self._loo_cache[exclude].value_at(L)


# ---------------------------------------------------------------------------
# Reference solver (dynamic programming) -- used only for unit tests.
# ---------------------------------------------------------------------------
def knapsack_dp(weights, prizes, cap, scale: int = 1):
    """Exact 0/1 knapsack by integer DP, for validating the frontier solver.

    Weights are multiplied by `scale` and rounded so that non-integer weights
    can still be checked on a grid.
    """
    weights = np.asarray(weights, dtype=float)
    prizes = np.asarray(prizes, dtype=float)
    W = int(round(cap * scale))
    if W < 0:
        return -np.inf
    dp = np.full(W + 1, 0.0)
    for wi, ri in zip(weights, prizes):
        if ri <= 0:
            continue
        wi_int = int(round(wi * scale))
        if wi_int > W:
            continue
        for c in range(W, wi_int - 1, -1):
            cand = dp[c - wi_int] + ri
            if cand > dp[c]:
                dp[c] = cand
    return float(dp[W])
