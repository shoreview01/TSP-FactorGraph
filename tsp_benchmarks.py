"""
TSP 벤치마크 솔버 모음
======================
Held-Karp (Exact DP), Nearest Neighbor, 2-opt, OR-Tools, Genetic Algorithm

사용법:
  from tsp_benchmarks import (
      held_karp, nearest_neighbor, two_opt, ortools_solve, genetic_algorithm,
      run_all_benchmarks
  )

  D = np.random.rand(11, 11); np.fill_diagonal(D, 0)
  results = run_all_benchmarks(D, start=0, seed=42)
  for name, (route, cost, elapsed) in results.items():
      print(f"{name:20s}  cost={cost:.4f}  time={elapsed:.4f}s")

CLI:
  python tsp_benchmarks.py --n 12 --trials 10
  python tsp_benchmarks.py --tsplib ALL_tsp/gr17.tsp.gz
"""

import time
import itertools
import argparse
import numpy as np
from typing import List, Tuple, Dict, Optional


# =================================================================
#  유틸: 경로 비용 계산
# =================================================================
def route_cost(D: np.ndarray, route: List[int]) -> float:
    """경로의 총 비용 계산 (마지막 → 처음 포함)."""
    c = 0.0
    for i in range(len(route) - 1):
        c += D[route[i], route[i + 1]]
    return c


# =================================================================
#  1. Held-Karp (Exact DP) — O(2^N · N²)
# =================================================================
def held_karp(D: np.ndarray, start: int = 0) -> Tuple[List[int], float]:
    """
    Held-Karp 알고리즘. Exact optimal 보장.
    N ≤ 20 권장 (메모리: O(2^N · N)).

    Returns: (route, cost)
    """
    N = D.shape[0]
    if N <= 1:
        return [start, start], 0.0

    # dp[S][i] = start에서 집합 S의 도시들을 거쳐 i에 도달하는 최소 비용
    # S는 bitmask (start 제외)
    cities = [c for c in range(N) if c != start]
    n = len(cities)
    city_to_idx = {c: i for i, c in enumerate(cities)}

    M = 1 << n
    INF = float('inf')

    dp = np.full((M, n), INF, dtype=np.float64)
    parent = np.full((M, n), -1, dtype=np.int32)

    # 초기: start → city
    for i, c in enumerate(cities):
        dp[1 << i, i] = D[start, c]

    # DP 전이
    for mask in range(1, M):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            if dp[mask, last] == INF:
                continue

            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                new_cost = dp[mask, last] + D[cities[last], cities[nxt]]
                if new_cost < dp[new_mask, nxt]:
                    dp[new_mask, nxt] = new_cost
                    parent[new_mask, nxt] = last

    # 종료: 마지막 도시 → start
    full = M - 1
    best_cost = INF
    best_last = -1
    for i in range(n):
        total = dp[full, i] + D[cities[i], start]
        if total < best_cost:
            best_cost = total
            best_last = i

    # 역추적
    route_inner = []
    mask = full
    cur = best_last
    for _ in range(n):
        route_inner.append(cities[cur])
        prev = parent[mask, cur]
        mask ^= (1 << cur)
        cur = prev

    route_inner.reverse()
    route = [start] + route_inner + [start]

    return route, best_cost


# =================================================================
#  2. Nearest Neighbor — O(N²)
# =================================================================
def nearest_neighbor(D: np.ndarray, start: int = 0) -> Tuple[List[int], float]:
    """
    Greedy nearest neighbor 휴리스틱.

    Returns: (route, cost)
    """
    N = D.shape[0]
    visited = {start}
    route = [start]
    cost = 0.0
    cur = start

    for _ in range(N - 1):
        best_d, best_c = float('inf'), -1
        for c in range(N):
            if c not in visited and D[cur, c] < best_d:
                best_d = D[cur, c]
                best_c = c
        visited.add(best_c)
        route.append(best_c)
        cost += best_d
        cur = best_c

    cost += D[cur, start]
    route.append(start)
    return route, cost


# =================================================================
#  3. 2-opt Improvement — O(N² × iterations)
# =================================================================
def two_opt(D: np.ndarray, start: int = 0,
            initial_route: Optional[List[int]] = None,
            max_iters: int = 1000) -> Tuple[List[int], float]:
    """
    2-opt local search. NN 초기해에서 시작하여 교차 제거.

    Returns: (route, cost)
    """
    if initial_route is None:
        route, _ = nearest_neighbor(D, start)
    else:
        route = list(initial_route)

    N = len(route) - 1  # route[-1] == route[0]

    def calc_cost(r):
        return sum(D[r[i], r[i + 1]] for i in range(len(r) - 1))

    best_cost = calc_cost(route)
    improved = True
    itr = 0

    while improved and itr < max_iters:
        improved = False
        itr += 1
        for i in range(1, N - 1):
            for j in range(i + 1, N):
                # 2-opt swap: reverse route[i:j+1]
                # 비용 차이 = 새 엣지 - 기존 엣지
                delta = (D[route[i - 1], route[j]] +
                         D[route[i], route[j + 1]]) - \
                        (D[route[i - 1], route[i]] +
                         D[route[j], route[j + 1]])

                if delta < -1e-12:
                    route[i:j + 1] = route[i:j + 1][::-1]
                    best_cost += delta
                    improved = True

    return route, best_cost


# =================================================================
#  4. OR-Tools (Google Operations Research)
# =================================================================
def ortools_solve(D: np.ndarray, start: int = 0,
                  time_limit_sec: float = 1.0) -> Tuple[List[int], float]:
    """
    Google OR-Tools TSP 솔버.
    pip install ortools 필요.

    Returns: (route, cost)
    """
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    except ImportError:
        print("  ⚠ OR-Tools 미설치: pip install ortools")
        return [], float('inf')

    N = D.shape[0]

    # OR-Tools는 정수 거리 선호 → 스케일링
    scale = 1_000_000
    D_int = (D * scale).astype(np.int64)

    manager = pywrapcp.RoutingIndexManager(N, 1, start)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(D_int[from_node, to_node])

    transit_id = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_id)

    # 탐색 파라미터
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = int(time_limit_sec)

    solution = routing.SolveWithParameters(params)

    if solution:
        route = []
        idx = routing.Start(0)
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = solution.Value(routing.NextVar(idx))
        route.append(start)  # 돌아오기
        cost = route_cost(D, route)
        return route, cost
    else:
        return [], float('inf')


# =================================================================
#  5. Genetic Algorithm — 메타휴리스틱
# =================================================================
def genetic_algorithm(D: np.ndarray, start: int = 0,
                      pop_size: int = 200,
                      generations: int = 500,
                      elite_ratio: float = 0.1,
                      mutation_rate: float = 0.15,
                      tournament_k: int = 5,
                      seed: int = 42) -> Tuple[List[int], float]:
    """
    Genetic Algorithm with:
      - Ordered Crossover (OX)
      - Inversion Mutation
      - Tournament Selection
      - Elitism

    Returns: (route, cost)
    """
    rng = np.random.default_rng(seed)
    N = D.shape[0]
    cities = [c for c in range(N) if c != start]
    n = len(cities)

    if n == 0:
        return [start, start], 0.0

    # ── 적합도 (총 비용, 낮을수록 좋음) ──
    def fitness(perm):
        c = D[start, cities[perm[0]]]
        for i in range(n - 1):
            c += D[cities[perm[i]], cities[perm[i + 1]]]
        c += D[cities[perm[-1]], start]
        return c

    # ── 초기 population ──
    population = np.array([rng.permutation(n) for _ in range(pop_size)])
    costs = np.array([fitness(p) for p in population])

    n_elite = max(1, int(pop_size * elite_ratio))

    # ── Tournament Selection ──
    def tournament_select():
        candidates = rng.choice(pop_size, size=tournament_k, replace=False)
        winner = candidates[np.argmin(costs[candidates])]
        return population[winner].copy()

    # ── Ordered Crossover (OX) ──
    def ox_crossover(p1, p2):
        c1, c2 = sorted(rng.choice(n, size=2, replace=False))
        child = np.full(n, -1, dtype=np.int64)
        # 부모1의 구간 복사
        child[c1:c2 + 1] = p1[c1:c2 + 1]
        # 나머지를 부모2 순서대로 채움
        used = set(child[c1:c2 + 1])
        fill = [g for g in p2 if g not in used]
        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[idx]
                idx += 1
        return child

    # ── Inversion Mutation ──
    def mutate(perm):
        if rng.random() < mutation_rate:
            i, j = sorted(rng.choice(n, size=2, replace=False))
            perm[i:j + 1] = perm[i:j + 1][::-1]
        return perm

    # ── 진화 루프 ──
    best_cost = costs.min()
    best_perm = population[costs.argmin()].copy()

    for gen in range(generations):
        # Elitism: 상위 n_elite 보존
        elite_idx = np.argsort(costs)[:n_elite]
        new_pop = [population[i].copy() for i in elite_idx]

        # 나머지 자식 생성
        while len(new_pop) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            child = ox_crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)

        population = np.array(new_pop[:pop_size])
        costs = np.array([fitness(p) for p in population])

        gen_best = costs.min()
        if gen_best < best_cost:
            best_cost = gen_best
            best_perm = population[costs.argmin()].copy()

    # 경로 복원
    route = [start] + [cities[i] for i in best_perm] + [start]
    return route, best_cost


# =================================================================
#  Constrained 버전: NN, 2-opt, GA
# =================================================================
def nearest_neighbor_constrained(
    D: np.ndarray, start: int = 0,
    constraints: Optional[np.ndarray] = None
) -> Tuple[List[int], float]:
    """
    제약 조건 인식 Nearest Neighbor.
    constraints: (N-1, N-1) 행렬 (depot 제외).
      0 = 허용, -inf = 금지, 음수 = 페널티.
    """
    N = D.shape[0]
    n = N - 1  # depot 제외 도시 수
    visited = {start}
    route = [start]
    cost = 0.0
    cur = start

    # 도시 인덱스 매핑: 원래 도시 → constraint 인덱스
    other_cities = [c for c in range(N) if c != start]
    city_to_cidx = {c: i for i, c in enumerate(other_cities)}

    for step in range(N - 1):
        best_d, best_c = float('inf'), -1
        for c in range(N):
            if c in visited:
                continue
            d = D[cur, c]

            # 제약 확인
            if constraints is not None and c != start:
                cidx = city_to_cidx[c]
                bias = constraints[cidx, step]
                if np.isneginf(bias):
                    continue  # 금지
                d -= bias  # 페널티를 거리에 추가 (음수 페널티 → 거리 증가)

            if d < best_d:
                best_d = d
                best_c = c

        if best_c == -1:
            # 모든 도시 금지된 경우 → 가장 가까운 미방문 도시 강제 선택
            for c in range(N):
                if c not in visited and D[cur, c] < best_d:
                    best_d = D[cur, c]
                    best_c = c

        visited.add(best_c)
        route.append(best_c)
        cost += D[cur, best_c]
        cur = best_c

    cost += D[cur, start]
    route.append(start)
    return route, cost


def genetic_algorithm_constrained(
    D: np.ndarray, start: int = 0,
    constraints: Optional[np.ndarray] = None,
    pop_size: int = 200,
    generations: int = 500,
    penalty_weight: float = 1000.0,
    seed: int = 42
) -> Tuple[List[int], float]:
    """
    제약 인식 GA. 금지 위반 시 penalty_weight 만큼 비용 추가.
    """
    rng = np.random.default_rng(seed)
    N = D.shape[0]
    cities = [c for c in range(N) if c != start]
    n = len(cities)

    if n == 0:
        return [start, start], 0.0

    def fitness(perm):
        c = D[start, cities[perm[0]]]
        for i in range(n - 1):
            c += D[cities[perm[i]], cities[perm[i + 1]]]
        c += D[cities[perm[-1]], start]

        # 제약 위반 페널티
        if constraints is not None:
            for step, idx in enumerate(perm):
                bias = constraints[idx, step]
                if np.isneginf(bias):
                    c += penalty_weight
                elif bias < 0:
                    c -= bias  # 음수 페널티 → 양수 추가
        return c

    population = np.array([rng.permutation(n) for _ in range(pop_size)])
    costs = np.array([fitness(p) for p in population])

    n_elite = max(1, int(pop_size * 0.1))
    tournament_k = 5

    def tournament_select():
        candidates = rng.choice(pop_size, size=tournament_k, replace=False)
        return population[candidates[np.argmin(costs[candidates])]].copy()

    def ox_crossover(p1, p2):
        c1, c2 = sorted(rng.choice(n, size=2, replace=False))
        child = np.full(n, -1, dtype=np.int64)
        child[c1:c2 + 1] = p1[c1:c2 + 1]
        used = set(child[c1:c2 + 1])
        fill = [g for g in p2 if g not in used]
        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[idx]; idx += 1
        return child

    def mutate(perm):
        if rng.random() < 0.15:
            i, j = sorted(rng.choice(n, size=2, replace=False))
            perm[i:j + 1] = perm[i:j + 1][::-1]
        return perm

    best_cost = costs.min()
    best_perm = population[costs.argmin()].copy()

    for gen in range(generations):
        elite_idx = np.argsort(costs)[:n_elite]
        new_pop = [population[i].copy() for i in elite_idx]
        while len(new_pop) < pop_size:
            child = ox_crossover(tournament_select(), tournament_select())
            new_pop.append(mutate(child))
        population = np.array(new_pop[:pop_size])
        costs = np.array([fitness(p) for p in population])
        if costs.min() < best_cost:
            best_cost = costs.min()
            best_perm = population[costs.argmin()].copy()

    route = [start] + [cities[i] for i in best_perm] + [start]
    actual_cost = route_cost(D, route)
    return route, actual_cost


# =================================================================
#  통합 벤치마크 실행
# =================================================================
def run_all_benchmarks(
    D: np.ndarray,
    start: int = 0,
    seed: int = 42,
    include_ortools: bool = True,
    ortools_time_limit: float = 1.0,
    ga_generations: int = 500,
    ga_pop_size: int = 200,
    constraints: Optional[np.ndarray] = None,
) -> Dict[str, Tuple[List[int], float, float]]:
    """
    모든 벤치마크를 실행하고 결과 반환.

    Returns: {name: (route, cost, elapsed_seconds)}
    """
    results = {}
    N = D.shape[0]

    # 1. Held-Karp (N ≤ 22)
    if N <= 22:
        t0 = time.time()
        r, c = held_karp(D, start)
        results['Held-Karp (Exact)'] = (r, c, time.time() - t0)

    # 2. Nearest Neighbor
    t0 = time.time()
    if constraints is not None:
        r, c = nearest_neighbor_constrained(D, start, constraints)
    else:
        r, c = nearest_neighbor(D, start)
    results['Nearest Neighbor'] = (r, c, time.time() - t0)

    # 3. NN + 2-opt
    t0 = time.time()
    if constraints is not None:
        r_nn, _ = nearest_neighbor_constrained(D, start, constraints)
        r, c = two_opt(D, start, initial_route=r_nn)
    else:
        r, c = two_opt(D, start)
    results['NN + 2-opt'] = (r, c, time.time() - t0)

    # 4. Genetic Algorithm
    t0 = time.time()
    if constraints is not None:
        r, c = genetic_algorithm_constrained(
            D, start, constraints,
            pop_size=ga_pop_size, generations=ga_generations, seed=seed
        )
    else:
        r, c = genetic_algorithm(
            D, start, pop_size=ga_pop_size,
            generations=ga_generations, seed=seed
        )
    results['Genetic Algorithm'] = (r, c, time.time() - t0)

    # 5. OR-Tools
    if include_ortools:
        t0 = time.time()
        r, c = ortools_solve(D, start, time_limit_sec=ortools_time_limit)
        if r:
            results['OR-Tools'] = (r, c, time.time() - t0)

    return results


def print_results(results: Dict, optimal_cost: float = None):
    """벤치마크 결과 테이블 출력."""
    print(f"\n{'Method':<22s} {'Cost':>12s} {'Time (s)':>10s}", end='')
    if optimal_cost:
        print(f" {'Gap':>10s}", end='')
    print()
    print("-" * (56 if optimal_cost else 44))

    for name, (route, cost, elapsed) in sorted(results.items(), key=lambda x: x[1][1]):
        print(f"{name:<22s} {cost:>12.4f} {elapsed:>10.4f}", end='')
        if optimal_cost and optimal_cost > 0:
            gap = (cost - optimal_cost) / optimal_cost * 100
            print(f" {gap:>+9.2f}%", end='')
        print()


# =================================================================
#  CLI
# =================================================================
def main():
    parser = argparse.ArgumentParser(description='TSP 벤치마크 비교')
    parser.add_argument('--n', type=int, default=12, help='도시 수 (기본 12)')
    parser.add_argument('--trials', type=int, default=1, help='랜덤 trial 수')
    parser.add_argument('--seed', type=int, default=42, help='시드')
    parser.add_argument('--no-ortools', action='store_true', help='OR-Tools 제외')
    parser.add_argument('--ga-gen', type=int, default=500, help='GA 세대 수')
    parser.add_argument('--ga-pop', type=int, default=200, help='GA 인구 수')

    # TSPLIB 파일
    parser.add_argument('--tsplib', type=str, default=None,
                        help='TSPLIB 파일 (.tsp 또는 .tsp.gz)')

    # 제약 조건
    parser.add_argument('--constrained', action='store_true',
                        help='랜덤 time window 제약 추가')
    parser.add_argument('--constraint-ratio', type=float, default=0.3)

    # Beam search
    parser.add_argument('--beam', type=int, default=None,
                        help='Beam width (None=exact DP, 정수=beam search). N>20이면 필수.')
    parser.add_argument('--bp-iters', type=int, default=50, help='FG-BP 최대 반복 (기본 50)')
    parser.add_argument('--bp-patience', type=int, default=10, help='FG-BP early stop patience (기본 10)')
    parser.add_argument('--bp-damping', type=float, default=0.3, help='FG-BP damping (기본 0.3)')

    args = parser.parse_args()

    if args.tsplib:
        # TSPLIB 파일 사용
        from run_tsplib import parse_tsplib, get_known_optimal
        from tsp_factor_graph_gpu import TSPFactorGraphSolverGPU
        tsp = parse_tsplib(args.tsplib)
        D = tsp['D']
        N = tsp['dimension'] - 1
        known_opt = get_known_optimal(tsp['name'])

        print(f"\n{'='*60}")
        print(f"  Benchmark: {tsp['name']} (N={tsp['dimension']})")
        print(f"  BP: iters={args.bp_iters}, patience={args.bp_patience}, damping={args.bp_damping}")
        print(f"{'='*60}")

        results = run_all_benchmarks(
            D, start=0, seed=args.seed,
            include_ortools=not args.no_ortools,
            ga_generations=args.ga_gen, ga_pop_size=args.ga_pop,
        )

        # FG-BP (Exact)
        if N <= 20:
            t0 = time.time()
            solver = TSPFactorGraphSolverGPU(
                D, start_city=0, damping=args.bp_damping,
                iters=args.bp_iters,
                verbose=False, patience=args.bp_patience, seed=args.seed,
            )
            r, c = solver.run()
            results['FG-BP (Exact)'] = (r, c, time.time() - t0)
            solver.cleanup()

        # FG-BP (Beam)
        if args.beam is not None or N > 20:
            bw = args.beam if args.beam is not None else min(1000, 2**N)
            t0 = time.time()
            solver = TSPFactorGraphSolverGPU(
                D, start_city=0, damping=args.bp_damping,
                iters=args.bp_iters,
                verbose=False, patience=args.bp_patience, seed=args.seed,
                beam_width=bw,
            )
            r, c = solver.run()
            results[f'FG-BP (Beam={bw})'] = (r, c, time.time() - t0)
            solver.cleanup()

        print_results(results, optimal_cost=known_opt)

    else:
        # 랜덤 행렬
        from tsp_factor_graph_gpu import TSPFactorGraphSolverGPU

        all_results = {}  # {method: [costs]}

        for trial in range(args.trials):
            seed = args.seed + trial
            rng = np.random.default_rng(seed)
            D = rng.random((args.n, args.n))
            np.fill_diagonal(D, 0)
            D = (D + D.T) / 2  # symmetric

            N = args.n - 1

            # 제약 조건
            constraints = None
            if args.constrained:
                constraints = TSPFactorGraphSolverGPU.make_constraints(N)
                for c in rng.choice(N, size=max(1, int(N * args.constraint_ratio)),
                                     replace=False):
                    width = rng.integers(max(1, N * 3 // 10), max(2, N * 7 // 10) + 1)
                    start_t = rng.integers(0, max(1, N - width + 1))
                    TSPFactorGraphSolverGPU.time_window(
                        constraints, city=c, earliest=start_t, latest=start_t + width - 1
                    )

            results = run_all_benchmarks(
                D, start=0, seed=seed,
                include_ortools=not args.no_ortools,
                ga_generations=args.ga_gen, ga_pop_size=args.ga_pop,
                constraints=constraints,
            )

            # Factor-Graph BP 솔버 (Exact)
            beam_width = args.beam

            if N <= 20:
                # Exact DP 가능
                t0 = time.time()
                solver = TSPFactorGraphSolverGPU(
                    D, start_city=0, damping=args.bp_damping,
                    iters=args.bp_iters,
                    verbose=False, patience=args.bp_patience, seed=seed,
                    constraints=constraints,
                )
                route_bp, cost_bp = solver.run()
                elapsed_bp = time.time() - t0
                results['FG-BP (Exact)'] = (route_bp, cost_bp, elapsed_bp)

                if args.constrained and solver.has_constraints:
                    t0 = time.time()
                    r_exact, c_exact = solver.solve_exact_constrained()
                    results['Exact Constrained'] = (r_exact, c_exact, time.time() - t0)

                solver.cleanup()

            # Beam search (항상 --beam 지정 시, 또는 N>20일 때 자동)
            if beam_width is not None or N > 20:
                bw = beam_width if beam_width is not None else min(1000, 2**N)
                t0 = time.time()
                solver_beam = TSPFactorGraphSolverGPU(
                    D, start_city=0, damping=args.bp_damping,
                    iters=args.bp_iters,
                    verbose=False, patience=args.bp_patience, seed=seed,
                    constraints=constraints,
                    beam_width=bw,
                )
                route_bm, cost_bm = solver_beam.run()
                elapsed_bm = time.time() - t0
                results[f'FG-BP (Beam={bw})'] = (route_bm, cost_bm, elapsed_bm)
                solver_beam.cleanup()

            # optimal cost
            opt = results.get('Held-Karp (Exact)', (None, None, None))[1]
            if opt is None:
                opt = results.get('Exact Constrained', (None, None, None))[1]

            if args.trials == 1:
                print(f"\n{'='*60}")
                print(f"  Random D ({args.n}×{args.n}), seed={seed}")
                if args.constrained:
                    print(f"  Constrained: {args.constraint_ratio*100:.0f}% cities")
                if beam_width:
                    print(f"  Beam width: {beam_width}")
                print(f"  BP: iters={args.bp_iters}, patience={args.bp_patience}, damping={args.bp_damping}")
                print(f"{'='*60}")
                print_results(results, optimal_cost=opt)
            else:
                for name, (r, c, t) in results.items():
                    if name not in all_results:
                        all_results[name] = {'costs': [], 'times': []}
                    all_results[name]['costs'].append(c)
                    all_results[name]['times'].append(t)

        # 여러 trial 평균
        if args.trials > 1:
            print(f"\n{'='*70}")
            print(f"  {args.trials} trials, N={args.n}, seed={args.seed}~{args.seed+args.trials-1}")
            if args.constrained:
                print(f"  Constrained: {args.constraint_ratio*100:.0f}% cities")
            print(f"  BP: iters={args.bp_iters}, patience={args.bp_patience}, damping={args.bp_damping}")
            print(f"{'='*70}")

            # optimal 기준
            opt_key = 'Held-Karp (Exact)' if 'Held-Karp (Exact)' in all_results else None
            if opt_key is None and 'Exact Constrained' in all_results:
                opt_key = 'Exact Constrained'
            opt_costs = np.array(all_results[opt_key]['costs']) if opt_key else None

            print(f"\n{'Method':<22s} {'Mean Cost':>10s} {'Time (mean ± std)':>20s}", end='')
            if opt_costs is not None:
                print(f" {'Gap (mean ± std)':>20s}", end='')
            print()
            print("-" * (64 if opt_costs is not None else 42))

            for name in sorted(all_results.keys(),
                              key=lambda k: np.mean(all_results[k]['costs'])):
                costs_arr = np.array(all_results[name]['costs'])
                times_arr = np.array(all_results[name]['times'])
                mean_c = costs_arr.mean()
                print(f"{name:<22s} {mean_c:>10.4f}   {times_arr.mean():>7.4f}s ± {times_arr.std():>6.4f}s", end='')
                if opt_costs is not None:
                    gaps = (costs_arr - opt_costs) / opt_costs * 100
                    print(f"   {gaps.mean():>+6.2f}% ± {gaps.std():>5.2f}%", end='')
                print()


if __name__ == '__main__':
    main()