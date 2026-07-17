"""
exp4_online.py
==============

실험 4 — 시변 disruption 환경에서의 online adaptiveness (논문 Sec. IV,
Algorithm 2 의 online re-solve 정책).

환경 변화 모델:
  * 매 time step 전체 노드의 5% 가 무작위로 새 사고 지점이 되고,
  * 각 사고 지점에서 도로거리 500 m 이내 노드로 향하는 엣지(양방향)의
    통행 비용이 3배로 증가한다:  D_t[i,j] = 3 * D_0[i,j].
  * 각 사고는 duration 스텝(기본 3) 동안 지속된 뒤 해소된다.  disruption
    시퀀스는 seed 로 고정되어 모든 방법/정책이 동일한 환경을 겪는다.
  * (duration=1 로 두면 매 스텝 완전히 재추첨되는 무기억 환경이 되는데, 이
    경우 사고가 다음 스텝에 사라져 어떤 계획자도 첫 hop 이상을 적응시킬 수
    없으므로 online ~= static 이 된다.  적응성이 의미를 가지려면 환경에
    시간적 지속성이 있어야 한다.)

정책 비교 (Algorithm 2):
  * static : 출발 전 D_0 로 투어를 한 번 계획하고 그대로 주행.
             단, 지불 비용은 그 시점의 D_t (disruption 을 그대로 맞는다).
  * online : 매 step, 각 차량이 현재 D_t 위에서 "현재 위치 -> 남은 노드 전부
             -> depot 복귀" open-path TSP 를 재계획하고 첫 노드로 이동
             (Algorithm 2 lines 4-7; T_c 예산/fallback 은 여기선 미사용).

Solver 4종 (BF / Trellis(제안) / NN / GA) x 정책 2종 = 8 곡선.
클러스터링은 coupled CAP-trellis 로 고정하고, 매 step 모든 클러스터 차량
(+depot 투어 차량)이 병렬로 한 노드씩 이동한다 (exp2 와 동일한 시간 축).

Monte Carlo: 서로 다른 disruption 시퀀스 --seeds 개(기본 100)를 반복해,
평균 곡선(선) + min-max 범위(같은 색의 반투명 구름)를 그린다.  static 계획은
D_0 에만 의존하므로 method 당 한 번만 세우고, 지불 비용/재계획만 seed 마다
반복한다.  (GA online 은 재계획마다 500세대를 다시 돌므로 100회 반복에
수 시간이 걸린다 -- 진행 상황이 25 seed 마다 출력된다.)

BF 주의: 남은 노드가 BF_PATH_LIMIT 개를 넘는 재계획은 (n-1)! 폭발 때문에
exact Held-Karp path 의 최적해로 대신한다 (둘 다 exact 라 값이 같다).

실행:
  C:/Users/guild/.conda/envs/tsp/python.exe exp4_online.py --capacity 12 --weights uniform
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from ieee_style import ieee_rc, save_fig, COL_W
import matplotlib.pyplot as plt

from coupled import solve_coupled
from benchmarks import brute_force_tsp, nearest_neighbor, genetic_algorithm
from exp3_routing import route_with          # static 계획 (D0 위 폐투어)
from run import load_matrix, make_weights, resolve_preference

BF_PATH_LIMIT = 9      # 남은 노드가 이보다 많으면 BF 재계획 -> exact HK path 대체

STYLE = {   # method -> (label, color)
    "bf":      ("BF",       "#000000"),
    "trellis": ("Proposed", "#0072bd"),
    "nn":      ("NN",       "#d7191c"),
    "ga":      ("GA",       "#1a9641"),
}
# 그리기/legend 순서: NN, GA, BF, Proposed (각각 static -> online)
METHODS = ["nn", "ga", "bf", "trellis"]


# ---------------------------------------------------------------------------
# 시변 disruption
# ---------------------------------------------------------------------------
def disruption_sequence(D0: np.ndarray, T: int, seed: int, frac: float = 0.05,
                        radius: float = 500.0, alpha: float = 3.0,
                        duration: int = 3):
    """t = 1..T 의 비용행렬 목록.

    매 step 노드 frac(기본 5%) 가 새 사고 지점이 되고, 각 사고는 duration
    스텝 동안 지속된다.  사고 지점 c 에서 도로거리 <= radius 인 노드로의
    엣지(양방향) 비용이 alpha 배가 된다 (사고가 겹쳐도 배수는 alpha 로 고정).
    """
    rng = np.random.default_rng(seed)
    n = len(D0)
    k = max(1, int(round(frac * n)))
    onset_centers = [rng.choice(n, size=k, replace=False) for _ in range(T)]
    seq = []
    for t in range(T):
        M = np.zeros((n, n), dtype=bool)
        for s in range(max(0, t - duration + 1), t + 1):   # 아직 지속 중인 사고
            for c in onset_centers[s]:
                near = D0[c] <= radius
                near[c] = False
                M[c, near] = True
                M[near, c] = True
        Dt = D0.copy()
        Dt[M] = alpha * D0[M]
        seq.append(Dt)
    return seq


# ---------------------------------------------------------------------------
# open-path TSP solvers: 현재 위치 u -> 남은 노드 전부 -> depot
# ---------------------------------------------------------------------------
def held_karp_path(Dsub: np.ndarray, start: int, end: int):
    """start 에서 출발해 모든 노드를 정확히 한 번 방문하고 end 에서 끝나는
    최단 open path (exact DP).  반환 (route, cost)."""
    n = len(Dsub)
    mids = [i for i in range(n) if i not in (start, end)]
    m = len(mids)
    if m == 0:
        return [start, end], float(Dsub[start, end])
    size = 1 << m
    g = np.full((size, m), np.inf)
    parent = np.full((size, m), -1, dtype=int)
    for c in range(m):
        g[1 << c, c] = Dsub[start, mids[c]]
    for mask in range(size):
        for j in range(m):
            if not (mask >> j) & 1 or not np.isfinite(g[mask, j]):
                continue
            base = g[mask, j]
            for k in range(m):
                if (mask >> k) & 1:
                    continue
                nm, cand = mask | (1 << k), base + Dsub[mids[j], mids[k]]
                if cand < g[nm, k]:
                    g[nm, k] = cand
                    parent[nm, k] = j
    full = size - 1
    close = g[full] + Dsub[[mids[j] for j in range(m)], end]
    j = int(np.argmin(close))
    cost = float(close[j])
    order = []
    mask = full
    while j >= 0:
        order.append(mids[j])
        pj = parent[mask, j]
        mask ^= (1 << j)
        j = pj
    order.reverse()
    return [start, *order, end], cost


def replan(method: str, Dt: np.ndarray, pos: int, remaining: set[int],
           depot: int, seed: int) -> int:
    """현재 비용행렬 Dt 로 남은 투어를 재계획하고 다음 방문 노드를 반환."""
    R = sorted(remaining)
    if not R:
        return depot                                   # 마지막 hop: depot 복귀
    if len(R) == 1:
        return R[0]
    nodes = [pos, *R, depot]                           # local: 0=pos, last=depot
    Dsub = Dt[np.ix_(nodes, nodes)]
    e = len(nodes) - 1
    if method == "trellis":
        route, _ = held_karp_path(Dsub, 0, e)
    elif method == "nn":
        route, _ = nearest_neighbor(Dsub, 0, end=e)
    elif method == "ga":
        route, _ = genetic_algorithm(Dsub, 0, end=e, seed=seed)
    elif method == "bf":
        if len(R) <= BF_PATH_LIMIT:
            route, _ = brute_force_tsp(Dsub, 0, end=e)
        else:                                          # (n-1)! 폭발 -> exact DP 대체
            route, _ = held_karp_path(Dsub, 0, e)
    else:
        raise ValueError(method)
    return nodes[route[1]]


# ---------------------------------------------------------------------------
# 시뮬레이션: 매 step 모든 차량이 병렬로 한 노드씩 이동
# ---------------------------------------------------------------------------
def plan_static(method: str, D0, clusters, depots, seed) -> list[list[int]]:
    """D0 위에서 차량별 폐투어를 한 번 계획한다 (disruption seed 와 무관)."""
    routes = [route_with(method, D0, members, e, seed=seed + ci)[0]
              for ci, (e, members) in enumerate(clusters)]
    routes.append(route_with(method, D0, list(depots), depots[0],
                             seed=seed + 1000)[0])
    return routes


def simulate(method: str, policy: str, Dts, clusters, depots, seed,
             static_routes: list[list[int]] | None = None):
    """반환 (누적비용 곡선 y[0..T], 총비용, 재계획 총 소요시간).

    static 정책은 미리 계산된 static_routes(D0 계획)를 그대로 주행한다.
    """
    vehicles = []
    for ci, (e, members) in enumerate(clusters):
        vehicles.append({"pos": e, "depot": e, "remaining": set(members) - {e},
                         "done": False, "seed": seed + ci})
    vehicles.append({"pos": depots[0], "depot": depots[0],
                     "remaining": set(depots) - {depots[0]},
                     "done": False, "seed": seed + 1000})

    if policy == "static":                 # D0 위에서 계획된 고정 경로 주행
        for v, r in zip(vehicles, static_routes):
            v["route"] = r
            v["step"] = 0

    y = [0.0]
    solve_time = 0.0
    t = 0
    while any(not v["done"] for v in vehicles):
        Dt = Dts[t]                        # step t+1 에서 적용되는 비용행렬
        t += 1
        inc = 0.0
        for v in vehicles:
            if v["done"]:
                continue
            if policy == "static":
                v["step"] += 1
                nxt = v["route"][v["step"]]
            else:
                t0 = time.perf_counter()
                nxt = replan(method, Dt, v["pos"], v["remaining"], v["depot"],
                             seed=v["seed"] + 17 * t)
                solve_time += time.perf_counter() - t0
            inc += float(Dt[v["pos"], nxt])
            v["pos"] = nxt
            v["remaining"].discard(nxt)
            if not v["remaining"] and v["pos"] == v["depot"]:
                v["done"] = True
        y.append(y[-1] + inc)
    return np.asarray(y), y[-1], solve_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/seongbuk.csv")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--capacity", type=float, default=12.0)
    ap.add_argument("--weights", choices=["uniform", "random"], default="uniform")
    ap.add_argument("--similarity", choices=["neg", "neg_sq"], default="neg")
    ap.add_argument("--preference", default="median")
    ap.add_argument("--pref-scale", type=float, default=1.0)
    ap.add_argument("--frac", type=float, default=0.05,
                    help="매 step 사고 지점이 되는 노드 비율")
    ap.add_argument("--radius", type=float, default=500.0,
                    help="사고 지점의 영향 반경 (도로거리, m)")
    ap.add_argument("--alpha", type=float, default=3.0,
                    help="disruption 구간의 비용 배수")
    ap.add_argument("--duration", type=int, default=3,
                    help="사고 지속 스텝 수 (1 = 매 스텝 완전 재추첨)")
    ap.add_argument("--seeds", type=int, default=100,
                    help="Monte Carlo 반복 횟수 (disruption 시퀀스 개수; "
                         "GA online 재계획 시간이 비례 증가)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ieee_rc()
    os.makedirs(args.outdir, exist_ok=True)

    D0 = load_matrix(args.csv)
    n = len(D0)
    weights = make_weights(args.weights, n, args.seed)
    eff_pref, _ = resolve_preference(D0, list(range(n)), args.similarity,
                                     args.preference, args.pref_scale)

    print(f"exp3: {n} nodes  Q={args.capacity:g}  weights={args.weights}  "
          f"disruption: {args.frac:.0%} nodes/step, r<={args.radius:g}m, "
          f"x{args.alpha:g}, lasts {args.duration} steps")

    # ---- 클러스터링 (모든 방법/정책 공유) -----------------------------------
    sol = solve_coupled(D0, weights, args.capacity, similarity=args.similarity,
                        preference=eff_pref)
    clusters = [(c.exemplar, [i for i in range(n) if sol.labels[i] == c.exemplar])
                for c in sol.clusters]
    depots = [e for e, _ in clusters]
    print(f"clustering: K={len(clusters)}  sizes={[len(m) for _, m in clusters]}")
    from trellis_tsp import MEM_LIMIT_NODES
    if len(depots) > MEM_LIMIT_NODES:
        raise SystemExit(
            f"K={len(depots)} depots > exact-trellis limit "
            f"{MEM_LIMIT_NODES}: depot 간 상위 TSP 를 exact 로 풀 수 없습니다. "
            f"용량 Q 를 키워 클러스터 수를 줄이세요 (예: random 가중치는 "
            f"--capacity 40, uniform 은 --capacity 12).")

    # ---- 공유 disruption 시퀀스 (seed 별로 하나, 모든 방법/정책이 공유) -----
    T_max = max(len(m) for _, m in clusters) + 4
    Dts_list = [
        disruption_sequence(D0, T_max + 4, seed=args.seed + 7 + 100 * s,
                            frac=args.frac, radius=args.radius,
                            alpha=args.alpha, duration=args.duration)
        for s in range(args.seeds)
    ]

    # ---- static 계획 (D0 만 사용 -> method 당 1회) ---------------------------
    static_routes = {m: plan_static(m, D0, clusters, depots, seed=args.seed)
                     for m in METHODS}

    # ---- Monte Carlo 시뮬레이션 ---------------------------------------------
    results = {}
    for method in METHODS:
        for policy in ["static", "online"]:
            curves, totals, st_sum = [], [], 0.0
            for si, Dts in enumerate(Dts_list):
                y, total, st = simulate(method, policy, Dts, clusters, depots,
                                        seed=args.seed,
                                        static_routes=static_routes[method])
                curves.append(y)
                totals.append(total)
                st_sum += st
                if policy == "online" and (si + 1) % 25 == 0:
                    print(f"    [{STYLE[method][0]} online] "
                          f"{si + 1}/{len(Dts_list)} seeds "
                          f"(replan {st_sum:.0f}s so far)")
            arr = np.vstack(curves)
            results[(method, policy)] = {
                "mean": arr.mean(axis=0),
                "lo": arr.min(axis=0),
                "hi": arr.max(axis=0),
                "total_mean": float(np.mean(totals)),
                "total_std": float(np.std(totals)),
            }
            extra = f"   (replan {st_sum:.1f}s)" if policy == "online" else ""
            print(f"  {STYLE[method][0]:<9s} {policy:<7s} "
                  f"total={np.mean(totals):>10,.0f} m "
                  f"+/-{np.std(totals):>6,.0f}{extra}")

    # ---- IEEE figure: 평균 곡선(선) + min-max 범위(구름), 8 곡선 -------------
    T = len(results[(METHODS[0], "static")]["mean"]) - 1
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.75))
    POLICY_STYLE = [("static", "--", "o", "none"), ("online", "-", "s", None)]

    def plot_curves(target, methods, ms, clip, with_label=True, cloud=True):
        for method in methods:
            label, color = STYLE[method]
            for policy, ls, mk, mfc in POLICY_STYLE:
                r = results[(method, policy)]
                steps = np.arange(len(r["mean"]))
                if cloud:                      # min-max 범위 구름
                    target.fill_between(steps, r["lo"] / 1000.0,
                                        r["hi"] / 1000.0, color=color,
                                        alpha=0.10, linewidth=0, zorder=1)
                target.plot(steps, r["mean"] / 1000.0, linestyle=ls,
                            color=color, marker=mk, markersize=ms,
                            markerfacecolor=mfc if mfc else color,
                            markeredgewidth=0.7, linewidth=1.0,
                            label=f"{label} ({policy})" if with_label else None,
                            clip_on=clip, zorder=3)

    plot_curves(ax, METHODS, ms=2.6, clip=False)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative travel cost (km)")
    ax.set_xlim(0, T)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="major", linestyle=":", linewidth=0.4)
    ax.legend(loc="upper left", fontsize=6, handlelength=2.4,
              labelspacing=0.3, borderpad=0.4, ncol=1)

    # ---- 확대 inset: 마지막 구간의 GA/BF/Proposed 평균선 (구름 제외) ---------
    zoom_methods = ["ga", "bf", "trellis"]
    z0, z1 = max(0, T - 2), T
    zy = np.concatenate([results[(m, p)]["mean"][z0:z1 + 1]
                         for m in zoom_methods
                         for p in ("static", "online")]) / 1000.0
    pad = (zy.max() - zy.min()) * 0.10
    axins = ax.inset_axes([0.615, 0.075, 0.365, 0.335])
    plot_curves(axins, zoom_methods, ms=2.2, clip=True, with_label=False,
                cloud=False)
    axins.set_xlim(z0 - 0.15, z1 + 0.15)
    axins.set_ylim(zy.min() - pad, zy.max() + pad)
    axins.set_xticks(list(range(z0, z1 + 1)))
    axins.tick_params(labelsize=6, length=2, pad=1.5)
    axins.grid(True, linestyle=":", linewidth=0.4)
    for sp in axins.spines.values():
        sp.set_linewidth(0.6)
    ax.indicate_inset_zoom(axins, edgecolor="0.35", linewidth=0.7, alpha=0.9)

    fig.tight_layout(pad=0.1)
    save_fig(fig, os.path.join(args.outdir, "exp4_online"))
    plt.close(fig)

    # ---- 요약 ---------------------------------------------------------------
    print(f"\nMonte Carlo over {args.seeds} disruption seeds")
    print(f"{'method':<11s}{'static (m)':>14s}{'online (m)':>14s}{'saving':>9s}")
    print("-" * 48)
    for method in METHODS:
        s = results[(method, "static")]
        o = results[(method, "online")]
        print(f"{STYLE[method][0]:<11s}"
              f"{s['total_mean']:>9,.0f} +/-{s['total_std']:>5,.0f}"
              f"{o['total_mean']:>9,.0f} +/-{o['total_std']:>5,.0f}"
              f"{100 * (s['total_mean'] - o['total_mean']) / s['total_mean']:>8.1f}%")
    print("\ndone -> figs/exp4_online.pdf")


if __name__ == "__main__":
    main()
