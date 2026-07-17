"""
exp2_dynamic.py
===============

실험 2 — 시변 수요(쓰레기양) 하의 용량 관리 비교 (clustering 기법 축).

노드별 weight w_i(t) 가 time step 마다 변한다:
  * 주기적 계절 변동  : w_i(t) = base_i * (1 + amp * sin(2*pi*t / period))
  * 폭증(surge) 구간   : step in [surge_start, surge_end] 에서 무작위 surge_frac
                        비율의 노드가 x(surge_lo..surge_hi) 배로 폭발적으로 증가.
  * 각 노드는 Q 이하로 clip (개별 노드는 항상 한 차량에 실려야 하므로).
수요 시퀀스는 seed 로 고정되어 모든 기법이 동일한 환경을 겪는다.

매 스텝, 세 기법으로 현재 수요를 **재클러스터링**(online) 하고 지표를 추적한다:
  * total          = sum_i w_i(t)                      (모든 기법 공통)
  * feasible-cover = sum_{k: load_k <= Q} load_k        (안전 수거량)
  * infeasible     = sum_{k: load_k >  Q} load_k        (관리 실패, 적체량)
    (feasible-cover + infeasible = total)

기법 (비교군은 공정성을 위해 **demand-weighted** 변형 사용):
  * proposed : coupled CAP-trellis (용량 인지, K 를 스스로 결정)
  * ap       : weighted AP -- s_w(i,k) = w_i * s(i,k) (가중 facility-location
               목적), preference 는 가중 유사도의 median 으로 스텝마다 재계산.
               weight 를 유사도에 반영하지만 용량 제약은 없음.
  * kmedoids : weighted K-medoids -- medoid 갱신이 sum_i w_i d(i,m) 최소화.
               K = 그 스텝의 proposed K 부여 (같은 차량 대수인데 용량 무지).

그림: 1x3 small multiples (기법별 패널), total 선 + feasible(초록)/infeasible(빨강)
누적 영역 + surge 구간 음영.  IEEE Transactions 2단 규격.

출력: figs/exp2_dynamic.pdf/.png + figs/exp2_dynamic.csv

실행 (conda env `tsp`):
  C:/Users/guild/.conda/envs/tsp/python.exe exp2_dynamic.py --capacity 40 --steps 12
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from ieee_style import ieee_rc, save_fig, COL_W2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from coupled import solve_coupled
from exp1_clustering import ap_labels, kmedoids_labels
from run import load_matrix, resolve_preference

GREEN = "#1a9641"      # feasible-covered
RED = "#d7191c"        # infeasible
BLACK = "#222222"      # total
SURGE = "#f0d000"      # surge-window shading

METHOD_TITLE = {"proposed": "Proposed", "ap": "Weighted AP",
                "kmedoids": "Weighted K-medoids"}
ORDER = ["proposed", "ap", "kmedoids"]


def weight_trajectory(n, T, Q, seed, base_lo=1, base_hi=6, amp=0.25, period=6.0,
                      surge=(5, 7), surge_frac=0.4, surge_lo=2.0, surge_hi=4.0):
    """(T, n) 시변 weight 행렬.  각 원소는 (0, Q] 로 clip."""
    rng = np.random.default_rng(seed)
    base = rng.integers(base_lo, base_hi + 1, size=n).astype(float)
    W = np.zeros((T, n))
    for t in range(T):
        season = 1.0 + amp * np.sin(2.0 * np.pi * t / period)
        w = base * season
        if surge[0] <= t <= surge[1]:                 # 폭증 구간
            k = max(1, int(round(surge_frac * n)))
            hit = rng.choice(n, size=k, replace=False)
            w[hit] = w[hit] * rng.uniform(surge_lo, surge_hi, size=k)
        W[t] = np.clip(w, 1e-6, Q)
    return W


def cluster_loads(method, D, w, Q, similarity, eff_pref, seed, K_hint=None):
    """한 스텝의 배정으로부터 클러스터 load 리스트와 클러스터 수를 반환."""
    n = len(D)
    if method == "proposed":
        sol = solve_coupled(D, w, Q, similarity=similarity, preference=eff_pref)
        loads = [c.load for c in sol.clusters]
    else:
        if method == "ap":
            # weighted AP: s_w(i,k) = w_i * s(i,k), preference = 내부 median
            labels = ap_labels(D, similarity, None, node_weights=w)
        elif method == "kmedoids":
            K = K_hint if K_hint else max(1, int(np.ceil(w.sum() / Q)))
            labels = kmedoids_labels(D, K, seed=seed, node_weights=w)
        else:
            raise ValueError(method)
        loads = [float(w[labels == e].sum()) for e in sorted(set(labels.tolist()))]
    return loads


def decompose(loads, Q):
    """(total, feasible-covered, infeasible)."""
    feas = sum(l for l in loads if l <= Q + 1e-9)
    infeas = sum(l for l in loads if l > Q + 1e-9)
    return feas + infeas, feas, infeas


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--csv", default="data/seongbuk.csv")
    ap_.add_argument("--outdir", default="figs")
    ap_.add_argument("--capacity", type=float, default=40.0)
    ap_.add_argument("--steps", type=int, default=12)
    ap_.add_argument("--similarity", choices=["neg", "neg_sq"], default="neg")
    ap_.add_argument("--preference", default="median")
    ap_.add_argument("--pref-scale", type=float, default=1.0)
    ap_.add_argument("--surge-start", type=int, default=5)
    ap_.add_argument("--surge-end", type=int, default=7)
    ap_.add_argument("--surge-frac", type=float, default=0.4)
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--from-csv", default=None,
                     help="이 CSV 로부터 계산 없이 그림만 다시 그린다(그림 반복 수정용)")
    args = ap_.parse_args()

    ieee_rc()
    os.makedirs(args.outdir, exist_ok=True)

    rec = {m: {"total": [], "feas": [], "infeas": [], "K": []} for m in ORDER}

    if args.from_csv:                        # ---- 재플롯 모드 --------------
        df = pd.read_csv(args.from_csv)
        args.steps = int(df["step"].max()) + 1
        for m in ORDER:
            sub = df[df["method"] == m].sort_values("step")
            rec[m]["total"] = sub["total"].tolist()
            rec[m]["feas"] = sub["feasible_covered"].tolist()
            rec[m]["infeas"] = sub["infeasible"].tolist()
            rec[m]["K"] = sub["K"].tolist()
        print(f"exp4: replot from {args.from_csv}  steps={args.steps}")
    else:                                    # ---- 계산 모드 ----------------
        D = load_matrix(args.csv)
        n = len(D)
        Q = args.capacity
        eff_pref, _ = resolve_preference(D, list(range(n)), args.similarity,
                                         args.preference, args.pref_scale)
        W = weight_trajectory(n, args.steps, Q, seed=args.seed + 11,
                              surge=(args.surge_start, args.surge_end),
                              surge_frac=args.surge_frac)
        print(f"exp4: {n} nodes  Q={Q:g}  steps={args.steps}  "
              f"surge=[{args.surge_start},{args.surge_end}]  "
              f"total weight {W.sum(1).min():.0f}..{W.sum(1).max():.0f}")

        for t in range(args.steps):
            w = W[t]
            t0 = time.perf_counter()
            # proposed 먼저 -> 그 K 를 kmedoids 에 넘겨 공정 비교
            prop_loads = cluster_loads("proposed", D, w, Q, args.similarity,
                                       eff_pref, args.seed)
            K_prop = len(prop_loads)
            for m in ORDER:
                if m == "proposed":
                    loads = prop_loads
                else:
                    loads = cluster_loads(m, D, w, Q, args.similarity, eff_pref,
                                          args.seed, K_hint=K_prop)
                tot, feas, infeas = decompose(loads, Q)
                rec[m]["total"].append(tot)
                rec[m]["feas"].append(feas)
                rec[m]["infeas"].append(infeas)
                rec[m]["K"].append(len(loads))
            print(f"  t={t:2d}  W={w.sum():6.0f}  K_prop={K_prop:2d}  "
                  + "  ".join(f"{METHOD_TITLE[m]} infeas={rec[m]['infeas'][-1]:5.0f}"
                              for m in ORDER)
                  + f"   ({time.perf_counter() - t0:.1f}s)")

        rows = []
        for m in ORDER:
            for t in range(args.steps):
                rows.append({"method": m, "step": t, "total": rec[m]["total"][t],
                             "feasible_covered": rec[m]["feas"][t],
                             "infeasible": rec[m]["infeas"][t], "K": rec[m]["K"][t]})
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "exp2_dynamic.csv"),
                                  index=False)

    # ---- figure: 1x3 small multiples ---------------------------------------
    steps = np.arange(args.steps)
    ymax = max(max(rec[m]["total"]) for m in ORDER) * 1.08
    fig, axes = plt.subplots(1, 3, figsize=(COL_W2, COL_W2 * 0.34), sharey=True)
    for ax, m in zip(axes, ORDER):
        tot = np.array(rec[m]["total"])
        feas = np.array(rec[m]["feas"])
        ax.axvspan(args.surge_start - 0.5, args.surge_end + 0.5,
                   color=SURGE, alpha=0.18, linewidth=0, zorder=0)
        ax.fill_between(steps, 0, feas, color=GREEN, alpha=0.85, linewidth=0,
                        zorder=1)
        ax.fill_between(steps, feas, tot, color=RED, alpha=0.85, linewidth=0,
                        zorder=1)
        ax.plot(steps, tot, color=BLACK, linewidth=1.1, zorder=3)
        cum_inf = np.sum(rec[m]["infeas"])
        ax.set_title(f"{METHOD_TITLE[m]}\n"
                     rf"($\Sigma$ infeasible $= {cum_inf:,.0f}$)", fontsize=7)
        ax.set_xlabel("Time step")
        ax.set_xlim(0, args.steps - 1)
        ax.set_ylim(0, ymax)
        ax.grid(True, which="major", linestyle=":", linewidth=0.4, zorder=0)
    axes[0].set_ylabel("Weight (garbage)")

    handles = [
        Line2D([], [], color=BLACK, lw=1.1, label="Total demand"),
        Patch(facecolor=GREEN, alpha=0.85, label="Feasible-covered"),
        Patch(facecolor=RED, alpha=0.85, label="Infeasible (uncovered)"),
        Patch(facecolor=SURGE, alpha=0.18, label="Surge window"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7,
               frameon=True, bbox_to_anchor=(0.5, 1.02), handlelength=1.6,
               columnspacing=1.2, borderpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.3)
    save_fig(fig, os.path.join(args.outdir, "exp2_dynamic"))
    plt.close(fig)

    # ---- 요약 ---------------------------------------------------------------
    print(f"\n{'method':<11s}{'cum infeasible':>16s}{'peak infeasible':>17s}"
          f"{'mean K':>9s}")
    print("-" * 53)
    for m in ORDER:
        print(f"{METHOD_TITLE[m]:<11s}{np.sum(rec[m]['infeas']):>16,.0f}"
              f"{np.max(rec[m]['infeas']):>17,.0f}{np.mean(rec[m]['K']):>9.1f}")
    print("\ndone -> figs/exp2_dynamic.pdf")


if __name__ == "__main__":
    main()
