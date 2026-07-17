"""
exp3_routing.py
===============

실험 3 — 라우팅 기법별 누적 이동거리 비교 (논문 Sec. IV).

파이프라인은 계층 구조 그대로 고정하고, 라우팅 단계의 solver 만 바꾼다:

  1) 클러스터링   : coupled CAP-trellis (Algorithm 1) -- 모든 방법이 공유
  2) 클러스터 내부: depot(=exemplar)에서 출발하는 폐투어  <- solver 교체
  3) depot 간 TSP : 클러스터 depot 들을 잇는 상위 폐투어  <- solver 교체

Solver: Brute Force(BF) / Trellis(제안, exact) / Nearest Neighbor(NN) /
Genetic Algorithm(GA; population 100, 500 generations, tournament + OX +
swap mutation).

시뮬레이션: 매 time step 마다 모든 클러스터의 차량(+depot 투어 차량)이
병렬로 정확히 한 노드씩 이동한다.  x 축 = time step, y 축 = 전 차량 누적
이동거리.  IEEE Transactions 단일 컬럼 규격 figure 로 저장한다.

Monte Carlo (weights=random 일 때): --seeds 회(기본 100) 가중치 실현을
바꿔가며 [가중치 -> CAP 클러스터링 -> 방법별 라우팅] 전체를 반복하고,
평균 곡선(선) + min-max 범위(같은 색 반투명 구름)를 그린다.  uniform
가중치는 결정적이라 1회만 실행하며 구름 없이 기존과 동일하게 그려진다.

BF 주의: n 노드 클러스터의 BF 는 (n-1)! 이라 BF_LIMIT 초과 클러스터는
exact trellis 의 최적해로 대신한다(둘 다 exact 최적이므로 값이 같다 --
run.py --compare 의 0-mismatch 검증과 동일한 근거).  콘솔에 표시된다.

실행:
  C:/Users/guild/.conda/envs/tsp/python.exe exp3_routing.py --capacity 12 --weights uniform
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from ieee_style import ieee_rc, save_fig, COL_W
import matplotlib.pyplot as plt

from coupled import solve_coupled
from trellis_tsp import trellis_tsp
from benchmarks import brute_force_tsp, nearest_neighbor, genetic_algorithm, route_cost
from run import load_matrix, make_weights, resolve_preference

BF_LIMIT = 11        # BF 를 실제로 돌릴 최대 노드 수 ((n-1)! 폭발 가드)

STYLE = {   # method -> (label, color, linestyle, marker)
    "bf":      ("Brute force (optimal)", "#000000", "--", "o"),
    "trellis": ("Proposed (trellis)",    "#0072bd", "-",  "s"),
    "nn":      ("Nearest neighbor",      "#d7191c", "-.", "^"),
    "ga":      ("Genetic algorithm",     "#1a9641", ":",  "D"),
}
ORDER = ["nn", "ga", "bf", "trellis"]      # 그리기 순서 (제안 기법을 맨 위에)


def route_with(method: str, D: np.ndarray, nodes: list[int], depot: int,
               seed: int = 0) -> tuple[list[int], float]:
    """global `nodes` 를 `depot` 에서 출발/복귀하는 폐투어로 라우팅."""
    if len(nodes) == 1:
        return [depot, depot], 0.0
    Dsub = D[np.ix_(nodes, nodes)]
    s = nodes.index(depot)
    if method == "trellis":
        local, cost = trellis_tsp(Dsub, start=s)
    elif method == "nn":
        local, cost = nearest_neighbor(Dsub, start=s)
    elif method == "ga":
        local, cost = genetic_algorithm(Dsub, start=s, seed=seed)
    elif method == "bf":
        if len(nodes) <= BF_LIMIT:
            local, cost = brute_force_tsp(Dsub, start=s)
        else:
            # (n-1)! 이 불가한 크기 -> exact trellis 최적해로 대체 (동일 최적값)
            print(f"    [bf] {len(nodes)} nodes > BF_LIMIT={BF_LIMIT} -> "
                  f"exact trellis optimum substituted (identical value)")
            local, cost = trellis_tsp(Dsub, start=s)
    else:
        raise ValueError(method)
    return [nodes[i] for i in local], float(cost)


def cumulative_curve(D: np.ndarray, routes: list[list[int]], T: int) -> np.ndarray:
    """매 step 모든 차량이 병렬로 한 노드씩 이동할 때의 누적거리 곡선.

    routes: 폐투어 리스트 (클러스터별 차량 + depot 투어 차량).
    반환 y[0..T]: step t 까지 전 차량이 이동한 거리 합 (일찍 끝난 차량은 정지).
    """
    y = np.zeros(T + 1)
    for r in routes:
        hops = np.array([D[r[i], r[i + 1]] for i in range(len(r) - 1)], dtype=float)
        c = np.concatenate([[0.0], np.cumsum(hops)])
        if len(c) < T + 1:                     # 끝난 차량은 마지막 값 유지
            c = np.concatenate([c, np.full(T + 1 - len(c), c[-1])])
        y += c[:T + 1]
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/seongbuk.csv")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--capacity", type=float, default=12.0)
    ap.add_argument("--weights", choices=["uniform", "random"], default="uniform")
    ap.add_argument("--similarity", choices=["neg", "neg_sq"], default="neg")
    ap.add_argument("--preference", default="median")
    ap.add_argument("--pref-scale", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=100,
                    help="Monte Carlo 반복 횟수 -- weights=random 일 때만 사용 "
                         "(seed 마다 가중치 실현 -> 클러스터링 -> 라우팅을 "
                         "반복해 평균+min-max 구름을 그림). uniform 은 "
                         "결정적이라 1회로 고정.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ieee_rc()
    os.makedirs(args.outdir, exist_ok=True)

    D = load_matrix(args.csv)
    n = len(D)
    eff_pref, _ = resolve_preference(D, list(range(n)), args.similarity,
                                     args.preference, args.pref_scale)

    from trellis_tsp import MEM_LIMIT_NODES
    METHODS = ["bf", "trellis", "nn", "ga"]

    seeds = args.seeds if args.weights == "random" else 1
    if args.weights != "random" and args.seeds != 1:
        print("[note] uniform 가중치는 결정적이므로 1회만 실행합니다 "
              "(--seeds 는 weights=random 에서만 의미가 있음).")
    print(f"exp2: {n} nodes  Q={args.capacity:g}  weights={args.weights}  "
          f"MC runs={seeds}")

    # ---- Monte Carlo: 가중치 실현 -> 클러스터링 -> 방법별 라우팅 -------------
    all_routes = {m: [] for m in METHODS}      # method -> seed 별 routes 목록
    all_totals = {m: [] for m in METHODS}
    for s in range(seeds):
        t0 = time.perf_counter()
        weights = make_weights(args.weights, n, args.seed + s)
        sol = solve_coupled(D, weights, args.capacity,
                            similarity=args.similarity, preference=eff_pref)
        clusters = [(c.exemplar,
                     [i for i in range(n) if sol.labels[i] == c.exemplar])
                    for c in sol.clusters]
        depots = [e for e, _ in clusters]
        if len(depots) > MEM_LIMIT_NODES:
            raise SystemExit(
                f"K={len(depots)} depots > exact-trellis limit "
                f"{MEM_LIMIT_NODES}: depot 간 상위 TSP 를 exact 로 풀 수 "
                f"없습니다. 용량 Q 를 키워 클러스터 수를 줄이세요 (예: random "
                f"가중치는 --capacity 40, uniform 은 --capacity 12).")
        for method in METHODS:
            base = args.seed + 7919 * s
            routes = [route_with(method, D, members, e, seed=base + ci)[0]
                      for ci, (e, members) in enumerate(clusters)]
            routes.append(route_with(method, D, list(depots), depots[0],
                                     seed=base + 1000)[0])
            all_routes[method].append(routes)
            all_totals[method].append(sum(route_cost(D, r) for r in routes))
        print(f"  [run {s + 1}/{seeds}] K={len(clusters)}  "
              f"proposed={all_totals['trellis'][-1]:,.0f} m  "
              f"({time.perf_counter() - t0:.1f}s)")

    # sanity: BF 와 trellis 는 둘 다 exact -> run 마다 총거리 일치해야 함
    gap = max(abs(b - t) for b, t in zip(all_totals["bf"], all_totals["trellis"]))
    print(f"  [check] max_run |BF - trellis| = {gap:.6f} m (both exact)")

    # ---- 시뮬레이션: time step 별 누적거리 (run 별 -> 평균/min/max) ----------
    T = max(len(r) - 1
            for m in METHODS for routes in all_routes[m] for r in routes)
    results = {}
    for m in METHODS:
        arr = np.vstack([cumulative_curve(D, routes, T)
                         for routes in all_routes[m]])
        results[m] = {"mean": arr.mean(axis=0), "lo": arr.min(axis=0),
                      "hi": arr.max(axis=0),
                      "total_mean": float(np.mean(all_totals[m])),
                      "total_std": float(np.std(all_totals[m]))}

    # ---- IEEE figure: 평균 곡선(선) + min-max 범위(구름) --------------------
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.72))
    steps = np.arange(T + 1)

    def plot_curves(target, methods, ms, clip=True, cloud=True):
        for method in methods:
            label, color, ls, marker = STYLE[method]
            r = results[method]
            if cloud and seeds > 1:            # min-max 범위 구름
                target.fill_between(steps, r["lo"] / 1000.0, r["hi"] / 1000.0,
                                    color=color, alpha=0.10, linewidth=0,
                                    zorder=1)
            target.plot(steps, r["mean"] / 1000.0, linestyle=ls,
                        color=color, marker=marker, markersize=ms,
                        markerfacecolor="none" if method == "bf" else color,
                        markeredgewidth=0.8, linewidth=1.1, label=label,
                        clip_on=clip, zorder=3)

    plot_curves(ax, ORDER, ms=3.2, clip=False)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative travel distance (km)")
    ax.set_xlim(0, T)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="major", linestyle=":", linewidth=0.4)
    ax.legend(loc="upper left", handlelength=2.2, labelspacing=0.35,
              borderpad=0.5)

    # ---- 확대 inset: 마지막 구간의 GA / BF / proposed (평균선만) ------------
    zoom_methods = ["ga", "bf", "trellis"]
    z0, z1 = max(0, T - 2), T
    zy = np.concatenate([results[m]["mean"][z0:z1 + 1]
                         for m in zoom_methods]) / 1000.0
    pad = (zy.max() - zy.min()) * 0.10
    axins = ax.inset_axes([0.585, 0.075, 0.395, 0.36])
    plot_curves(axins, zoom_methods, ms=2.6, clip=True, cloud=False)
    axins.set_xlim(z0 - 0.15, z1 + 0.15)
    axins.set_ylim(zy.min() - pad, zy.max() + pad)
    axins.set_xticks(list(range(z0, z1 + 1)))
    axins.tick_params(labelsize=6, length=2, pad=1.5)
    axins.grid(True, linestyle=":", linewidth=0.4)
    for sp in axins.spines.values():
        sp.set_linewidth(0.6)
    ax.indicate_inset_zoom(axins, edgecolor="0.35", linewidth=0.7, alpha=0.9)

    fig.tight_layout(pad=0.1)
    save_fig(fig, os.path.join(args.outdir, "exp3_cumdist"))
    plt.close(fig)

    # ---- 요약 ---------------------------------------------------------------
    base = results["trellis"]["total_mean"]
    print(f"\nMonte Carlo over {seeds} weight realisation(s)")
    print(f"{'method':<26s}{'total (m)':>16s}{'vs proposed':>13s}")
    print("-" * 55)
    for method in METHODS:
        r = results[method]
        print(f"{STYLE[method][0]:<26s}"
              f"{r['total_mean']:>10,.0f} +/-{r['total_std']:>6,.0f}"
              f"{100 * (r['total_mean'] - base) / base:>+12.1f}%")
    print("\ndone -> figs/exp3_cumdist.pdf")


if __name__ == "__main__":
    main()
