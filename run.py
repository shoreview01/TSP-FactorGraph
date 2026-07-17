"""
run.py
======

Driver for the Coupled CAP-Trellis Message Passing framework (paper Sec. III,
Algorithm 1), with two comparison baselines: brute force and nearest-neighbour.

    python run.py --capacity 12 --weights uniform
    python run.py --capacity 12 --weights uniform --pref-scale 1.5
    python run.py --capacity 12 --weights uniform --compare   # vs brute-force + NN

The proposed method runs Algorithm 1 on the full node set: CAP messages
((37)-(42), with the Pareto-frontier knapsack inside phi~) exchange soft
information with per-cluster trellises through the bridging messages delta~
((40), (44), (45)) until the assignment {b_ij} reaches a fixed point.  Each
cluster's exemplar is its depot (v_{1,k} = e_k), and the final tours are exact
(Held-Karp min-sum trellis, no pruning, no beam).  The reported total is the
(P1.2) objective: the sum of regional tour costs.

Knobs:

  --tau        : candidate threshold  V^cand_k = {i : rho~_i > tau}  (Alg. 1)
  --rounds     : maximum coupling rounds R
  --pref-scale : AP cluster *strength*.  Multiplies the AP preference (the
                 median/min/float self-similarity).  Similarities are negative,
                 so a LARGER scale pushes the preference more negative -> fewer,
                 larger clusters; a SMALLER scale -> more, smaller clusters.

The per-cluster capacity report (load vs Q, utilisation, slack) is always
printed.  Algorithm 1 decodes {b_ij} purely from the messages (no repair), so
capacity feasibility is reported rather than forced.
"""

import argparse
import time

import numpy as np
import pandas as pd

from coupled import solve_coupled
from cap import similarity_from_distance
from benchmarks import brute_force_tsp, nearest_neighbor

BRUTE_LIMIT = 10          # max nodes for which to run brute force


def load_matrix(path: str) -> np.ndarray:
    return pd.read_csv(path, index_col=0).values.astype(float)


def make_weights(kind: str, m: int, seed: int = 0) -> np.ndarray:
    if kind == "uniform":
        return np.ones(m)
    return np.random.default_rng(seed).integers(1, 10, size=m).astype(float)


def resolve_preference(D, bins, similarity, preference, scale):
    """Resolve the AP preference to a float and apply the cluster-strength scale.

    Mirrors cap.py's median/min resolution on the same similarity matrix, then
    multiplies by ``scale`` so the float can be handed straight to the solver.
    Returns (effective_preference, base_preference).
    """
    sub = D[np.ix_(bins, bins)]
    S = similarity_from_distance(sub, kind=similarity)
    off = S[~np.eye(len(S), dtype=bool)]
    if preference == "median":
        base = float(np.median(off))
    elif preference == "min":
        base = float(np.min(off))
    else:
        base = float(preference)
    return base * scale, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/seongbuk.csv")
    ap.add_argument("--capacity", type=float, default=12.0)
    ap.add_argument("--weights", choices=["uniform", "random"], default="uniform")
    ap.add_argument("--similarity", choices=["neg", "neg_sq"], default="neg")
    ap.add_argument("--preference", default="median",
                    help="AP self-similarity: 'median', 'min', or a float")
    ap.add_argument("--pref-scale", type=float, default=1.0,
                    help="AP cluster strength: multiplies the preference "
                         "(>1 -> fewer/larger clusters, <1 -> more/smaller)")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="candidate threshold: V^cand_k = {i : rho~_i > tau}")
    ap.add_argument("--rounds", type=int, default=10,
                    help="maximum coupling rounds R of Algorithm 1")
    ap.add_argument("--max-cand", type=int, default=16,
                    help="cap on |V^cand_k| (guards the 2^m trellis table)")
    ap.add_argument("--bridge-damping", type=float, default=0.5,
                    help="damping gamma on the bridging messages delta~ "
                         "(0 = undamped Algorithm 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="compare each cluster route against brute-force + nearest-neighbour")
    args = ap.parse_args()

    D = load_matrix(args.csv)
    n = len(D)
    weights = make_weights(args.weights, n, args.seed)

    eff_pref, base_pref = resolve_preference(
        D, list(range(n)), args.similarity, args.preference, args.pref_scale)

    print(f"\nmatrix: {n} nodes  weights={args.weights}  Q={args.capacity}")
    print(f"AP preference: {eff_pref:,.1f}  "
          f"(base {base_pref:,.1f} x strength {args.pref_scale:g})   "
          f"tau={args.tau:g}  R={args.rounds}")

    t0 = time.perf_counter()
    sol = solve_coupled(D, weights, args.capacity,
                        similarity=args.similarity, preference=eff_pref,
                        tau=args.tau, max_rounds=args.rounds,
                        max_cand=args.max_cand,
                        bridge_damping=args.bridge_damping,
                        verbose=args.verbose)
    dt = time.perf_counter() - t0

    print(f"Algorithm 1: {sol.n_clusters} clusters  "
          f"rounds={sol.n_rounds}  fixed_point={sol.converged}  "
          f"cap_converged={sol.cap_converged}")
    print(f"Coupled CAP-trellis total route: {sol.total_length:,.1f}   ({dt:.1f}s)")

    # ---- per-cluster capacity satisfaction -------------------------------
    print(f"\n{'cluster':>7}{'depot':>7}{'size':>5}{'load':>9}{'cap Q':>8}"
          f"{'util%':>9}{'slack':>9}  status")
    print("-" * 63)
    utils = []
    for i, c in enumerate(sol.clusters):
        size = len(c.nodes) - 1                    # closed tour [e, ..., e]
        util = 100.0 * c.load / args.capacity if args.capacity else 0.0
        slack = args.capacity - c.load
        ok = c.load <= args.capacity + 1e-6
        utils.append(util)
        status = "OK" if ok else "OVER!"
        print(f"{i:>7}{c.exemplar:>7}{size:>5}{c.load:>9.1f}{args.capacity:>8.1f}"
              f"{util:>8.1f}%{slack:>9.1f}  {status}")
    print("-" * 63)
    feasible = sol.is_feasible(args.capacity)
    print(f"all clusters within capacity: {feasible}   "
          f"util min {min(utils):.0f}% / mean {np.mean(utils):.0f}% / max {max(utils):.0f}%")

    # ---- comparison baselines --------------------------------------------
    if args.compare:
        print(f"\n{'cluster':>7}{'size':>5}{'coupled':>13}{'brute-force':>13}{'NN':>11}")
        print("-" * 49)
        tr_total = nn_total = 0.0
        checked = mismatches = 0
        for i, c in enumerate(sol.clusters):
            nodes = c.nodes[:-1]                   # drop the closing depot
            Dsub = D[np.ix_(nodes, nodes)]
            tr = c.length
            nn = nearest_neighbor(Dsub, 0)[1]
            tr_total += tr
            nn_total += nn
            if len(nodes) <= BRUTE_LIMIT:          # brute force only when cheap
                bf = brute_force_tsp(Dsub, 0)[1]
                checked += 1
                if abs(bf - tr) > 1e-6:
                    mismatches += 1
                bf_s = f"{bf:,.0f}"
            else:
                bf_s = "-"
            print(f"{i:>7}{len(nodes) - 1:>5}{tr:>13,.0f}{bf_s:>13}{nn:>11,.0f}")
        print("-" * 49)
        print(f"{'TOTAL':>7}{'':>5}{tr_total:>13,.0f}{'':>13}{nn_total:>11,.0f}")

        gap_nn = 100.0 * (nn_total - tr_total) / tr_total if tr_total else 0.0
        verdict = "OPTIMAL" if mismatches == 0 else "NOT optimal"
        print(f"\nbrute-force optimality check on {checked} clusters "
              f"(<= {BRUTE_LIMIT} nodes): {mismatches} mismatch(es) -> "
              f"per-cluster routing is {verdict}")
        print(f"nearest-neighbour total: {nn_total:,.0f}  "
              f"(coupled saves {gap_nn:.1f}% vs NN)")


if __name__ == "__main__":
    main()
