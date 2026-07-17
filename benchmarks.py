"""
benchmarks.py
=============

Comparison baselines for the coupled CAP-trellis pipeline (paper Sec. IV-A):

  * brute_force_tsp   : O(n!) exhaustive optimum -- an independent exact check.
  * nearest_neighbor  : O(n^2) greedy heuristic   -- a quality lower bound.
  * genetic_algorithm : population-based metaheuristic -- tournament selection,
                        order crossover (OX), swap mutation; population 100,
                        500 generations, exactly as configured in the paper.

The proposed per-cluster router is the exact trellis (``trellis_tsp.trellis_tsp``).
Brute force confirms that route is optimal on the small clusters the capacity
budget produces; NN and GA show how much the exact route saves over heuristics.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np


def route_cost(D: np.ndarray, route: list[int]) -> float:
    """Total length of a closed route ``[start, ..., start]`` (final hop included)."""
    D = np.asarray(D, dtype=float)
    return float(sum(D[route[i], route[i + 1]] for i in range(len(route) - 1)))


def brute_force_tsp(D: np.ndarray, start: int = 0,
                    end: int | None = None) -> tuple[list[int], float]:
    """Exhaustive O(n!) optimum.  Exact; use on small instances only.

    ``end=None``(기본): 폐투어 ``[start, ..., start]``.
    ``end`` 지정: start 에서 출발해 모든 노드를 거쳐 end 에서 끝나는 open path
    ``[start, ..., end]`` (온라인 재계획: 현재 위치 -> 남은 노드 -> depot).
    """
    D = np.asarray(D, dtype=float)
    n = len(D)
    term = start if end is None else end
    if n <= 1:
        return [start], 0.0
    others = [c for c in range(n) if c != start and (end is None or c != end)]
    if not others:
        return [start, term], float(D[start, term])
    best_r, best_c = None, np.inf
    for perm in permutations(others):
        r = [start, *perm, term]
        c = sum(D[r[i], r[i + 1]] for i in range(len(r) - 1))
        if c < best_c:
            best_c, best_r = c, r
    return best_r, float(best_c)


def genetic_algorithm(D: np.ndarray, start: int = 0, pop_size: int = 100,
                      generations: int = 500, tournament: int = 3,
                      crossover_rate: float = 0.9, mutation_rate: float = 0.2,
                      seed: int = 0, end: int | None = None) -> tuple[list[int], float]:
    """Genetic algorithm for the closed tour from ``start`` (paper Sec. IV-A).

    Population-based metaheuristic with tournament selection, order crossover
    (OX) and swap mutation, population 100 and 500 generations by default --
    matching the GA benchmark configured in the paper.  Stochastic: pass
    ``seed`` for reproducibility.  ``end=None``: closed tour
    ``[start, ..., start]``; ``end`` 지정 시 open path ``[start, ..., end]``.
    """
    D = np.asarray(D, dtype=float)
    n = len(D)
    term = start if end is None else end
    if n <= 1:
        return [start], 0.0
    others = np.array([c for c in range(n)
                       if c != start and (end is None or c != end)])
    m = len(others)
    if m <= 2:                                # <=2 free nodes: exact is instant
        return brute_force_tsp(D, start, end=end)

    rng = np.random.default_rng(seed)

    def tour_cost(perm: np.ndarray) -> float:
        seq = others[perm]
        return float(D[start, seq[0]] + D[seq[:-1], seq[1:]].sum() + D[seq[-1], term])

    pop = [rng.permutation(m) for _ in range(pop_size)]
    costs = np.array([tour_cost(p) for p in pop])

    def ox(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Order crossover: copy a slice of p1, fill the rest in p2's order."""
        a, b = sorted(rng.integers(0, m, size=2))
        child = -np.ones(m, dtype=int)
        child[a:b + 1] = p1[a:b + 1]
        used = set(child[a:b + 1].tolist())
        fill = [g for g in p2 if g not in used]
        k = 0
        for i in range(m):
            if child[i] < 0:
                child[i] = fill[k]
                k += 1
        return child

    for _ in range(generations):
        elite = pop[int(np.argmin(costs))].copy()          # elitism
        new_pop = [elite]
        while len(new_pop) < pop_size:
            # tournament selection of two parents
            cand = rng.integers(0, pop_size, size=tournament)
            p1 = pop[cand[np.argmin(costs[cand])]]
            cand = rng.integers(0, pop_size, size=tournament)
            p2 = pop[cand[np.argmin(costs[cand])]]
            child = ox(p1, p2) if rng.random() < crossover_rate else p1.copy()
            if rng.random() < mutation_rate:               # swap mutation
                i, j = rng.integers(0, m, size=2)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        pop = new_pop
        costs = np.array([tour_cost(p) for p in pop])

    best = pop[int(np.argmin(costs))]
    route = [start, *others[best].tolist(), term]
    return route, float(np.min(costs))


def nearest_neighbor(D: np.ndarray, start: int = 0,
                     end: int | None = None) -> tuple[list[int], float]:
    """Greedy nearest-neighbour heuristic.  O(n^2).

    ``end=None``: 폐투어 ``[start, ..., start]``; ``end`` 지정 시 남은 노드를
    greedy 로 다 돌고 마지막에 end 로 향하는 open path ``[start, ..., end]``.
    """
    D = np.asarray(D, dtype=float)
    n = len(D)
    term = start if end is None else end
    if n <= 1:
        return [start], 0.0
    visited = {start} if end is None else {start, end}
    route = [start]
    cost = 0.0
    cur = start
    for _ in range(n - len(visited)):
        best_d, best_c = np.inf, -1
        for c in range(n):
            if c not in visited and D[cur, c] < best_d:
                best_d, best_c = D[cur, c], c
        visited.add(best_c)
        route.append(best_c)
        cost += best_d
        cur = best_c
    cost += D[cur, term]
    route.append(term)
    return route, float(cost)
