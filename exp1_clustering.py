"""
exp1_clustering.py
==================

실험 1 — 클러스터링 기법별 용량 제약 Q 의 feasibility 비교 (논문 Sec. IV).

비교군 (공정성을 위해 **demand-weighted** 변형; exp4 와 동일):
  * proposed : coupled CAP-trellis (Algorithm 1) -- 용량 제약이 메시지에 내장
  * ap       : weighted AP -- s_w(i,k) = w_i * s(i,k) (가중 facility-location
               목적), preference 는 가중 유사도의 median.  weight 를 유사도에
               반영하지만 용량 제약은 없음.  K 는 스스로 결정.
  * kmedoids : weighted K-medoids -- medoid 갱신이 sum_i w_i d(i,m) 최소화.
               K 는 proposed 가 찾은 클러스터 수와 동일하게 맞춰 공정 비교.

출력 (figs/):
  exp1_proposed.pdf/.png, exp1_ap.pdf/.png, exp1_kmedoids.pdf/.png

각 지도는
  * OpenStreetMap 성북구 도로망(회색, 라벨/타일 없이 도로만) 위에
  * 데이터 좌표의 서비스 노드(점)와 클러스터 depot(별)을 표시하고,
  * 클러스터 투어 엣지를 **실제 도로를 따라**(osmnx 최단경로 geometry)
    - load <= Q 인 feasible 클러스터는 초록 실선,
    - load  > Q 인 infeasible 클러스터는 빨강 실선으로 그린다.
  * legend: green = feasible, red = infeasible.

실행 (conda env `tsp` 의 python -- osmnx 필요):
  C:/Users/guild/.conda/envs/tsp/python.exe exp1_clustering.py --capacity 12 --weights uniform
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ieee_style import ieee_rc, save_fig, COL_W
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from coupled import solve_coupled
from trellis_tsp import trellis_tsp, MEM_LIMIT_NODES
from benchmarks import nearest_neighbor
from run import load_matrix, make_weights

GRAPHML = os.path.join("data", "seongbuk_drive.graphml")

GREEN = "#1a9641"     # feasible
RED = "#d7191c"       # infeasible
GREEN_LT = "#b7e2b0"  # load bar (feasible)
RED_LT = "#f6b0ac"    # load bar (infeasible)
ROAD = "#c8c8c8"      # background road network
NODE = "#333333"


# ---------------------------------------------------------------------------
# Baseline clusterings (capacity-blind)
# ---------------------------------------------------------------------------
def _consistent_labels(E: np.ndarray) -> np.ndarray:
    """E = A+R 로부터 argmax 배정 + exemplar 자기 일관성 강제."""
    labels = np.argmax(E, axis=1)
    for k in sorted(set(labels.tolist())):
        labels[k] = k
    return labels


def ap_labels(D: np.ndarray, similarity: str, preference: float | None,
              damping: float = 0.7, max_iter: int = 300,
              conv_iter: int = 20,
              label_history: list | None = None,
              node_weights: np.ndarray | None = None) -> np.ndarray:
    """표준 affinity propagation [28] -> labels[i] = exemplar node id.

    proposed 와 동일한 유사도 행렬/preference 를 쓰되 용량 제약이 없는
    원형 AP 메시지(responsibility/availability)로 클러스터링한다.
    exemplar 가 곧 depot 이 된다.  label_history 가 주어지면 매 메시지
    반복의 (자기 일관성 강제된) labels 를 append 한다.

    node_weights 가 주어지면 **demand-weighted AP**: 할당 비용을
    w_i * d(i,k) 로 두는 가중 facility-location 목적에 해당하는
    s_w(i,k) = w_i * s(i,k) (행 스케일링) 을 쓴다.  무거운 노드가 가까운
    exemplar 를 강하게 요구해 수요 밀집 지역에 exemplar 가 더 열린다
    (용량 제약은 여전히 없음).  preference=None 이면 (가중) 유사도의
    비대각 median 을 내부에서 쓴다.
    """
    from cap import similarity_from_distance
    n = len(D)
    S = similarity_from_distance(D, kind=similarity).astype(float).copy()
    if node_weights is not None:
        S = np.asarray(node_weights, dtype=float)[:, None] * S
    if preference is None:
        preference = float(np.median(S[~np.eye(n, dtype=bool)]))
    np.fill_diagonal(S, preference)

    A = np.zeros((n, n))
    R = np.zeros((n, n))
    last = None
    stable = 0
    for _ in range(max_iter):
        # responsibility: r(i,k) = s(i,k) - max_{k'!=k} [a(i,k') + s(i,k')]
        AS = A + S
        idx1 = np.argmax(AS, axis=1)
        max1 = AS[np.arange(n), idx1]
        AS2 = AS.copy()
        AS2[np.arange(n), idx1] = -np.inf
        max2 = np.max(AS2, axis=1)
        R_new = S - max1[:, None]
        R_new[np.arange(n), idx1] = S[np.arange(n), idx1] - max2
        R = damping * R + (1.0 - damping) * R_new

        # availability: a(i,k) = min(0, r(k,k) + sum_{i' not in {i,k}} max(0, r(i',k)))
        #               a(k,k) = sum_{i'!=k} max(0, r(i',k))
        Rp = np.maximum(R, 0.0)
        np.fill_diagonal(Rp, 0.0)
        colsum = Rp.sum(axis=0)
        A_new = np.minimum(0.0, np.diag(R)[None, :] + colsum[None, :] - Rp)
        A_new[np.arange(n), np.arange(n)] = colsum
        A = damping * A + (1.0 - damping) * A_new

        lab = np.argmax(A + R, axis=1)
        if label_history is not None:           # 반복별 (일관성 강제된) labels
            label_history.append(_consistent_labels(A + R))
        stable = stable + 1 if (last is not None and np.array_equal(lab, last)) else 0
        last = lab
        if stable >= conv_iter:
            break

    return _consistent_labels(A + R)


def kmedoids_labels(D: np.ndarray, K: int, seed: int = 0,
                    iters: int = 300,
                    label_history: list | None = None,
                    node_weights: np.ndarray | None = None) -> np.ndarray:
    """PAM 스타일 K-medoids (도로거리행렬 D 사용) -> labels[i] = medoid node id.

    label_history 가 주어지면 매 PAM 반복(assign+update) 후 labels 를 append 한다.
    node_weights 가 주어지면 **demand-weighted PAM**: medoid 갱신을
    sum_i w_i * d(i, m) 최소화로 수행한다 (할당은 여전히 최근접 medoid --
    w_i * d(i, m) 의 argmin_m 은 w_i 와 무관하므로 동일).  medoid 가 수요가
    무거운 쪽으로 끌리지만 용량 제약은 여전히 없다.
    """
    n = len(D)
    rng = np.random.default_rng(seed)
    w = (np.ones(n) if node_weights is None
         else np.asarray(node_weights, dtype=float))
    med = list(rng.choice(n, size=K, replace=False))

    def to_labels(medoids):
        lab = np.argmin(D[:, medoids], axis=1)
        labels = np.array([medoids[k] for k in lab], dtype=int)
        for m_ in medoids:
            labels[m_] = m_
        return labels

    for _ in range(iters):
        lab = np.argmin(D[:, med], axis=1)
        new_med = []
        for k in range(K):
            idx = np.where(lab == k)[0]
            if len(idx) == 0:
                new_med.append(med[k])
                continue
            costs = (w[idx, None] * D[np.ix_(idx, idx)]).sum(axis=0)
            new_med.append(int(idx[np.argmin(costs)]))
        if label_history is not None:           # 반복별 labels (갱신된 medoid 기준)
            label_history.append(to_labels(new_med))
        if set(new_med) == set(med):
            break
        med = new_med
    return to_labels(med)


# ---------------------------------------------------------------------------
# 클러스터 요약 + 시각화용 투어
# ---------------------------------------------------------------------------
def clusters_from_labels(D, weights, labels, Q):
    """labels -> [{depot, members, load, feasible, tour}].

    tour 는 시각화용 폐투어: 노드 수가 exact trellis 한계 이내면 trellis(최적),
    넘으면(용량 무시 클러스터링이 만든 거대 infeasible 클러스터) NN 으로 그린다.
    """
    out = []
    for e in sorted(set(labels.tolist())):
        members = [i for i in range(len(D)) if labels[i] == e]
        load = float(weights[labels == e].sum())
        if len(members) == 1:
            tour = [int(e), int(e)]
        else:
            Dsub = D[np.ix_(members, members)]
            s = members.index(e)
            if len(members) <= MEM_LIMIT_NODES:
                local, _ = trellis_tsp(Dsub, start=s)
            else:
                local, _ = nearest_neighbor(Dsub, start=s)
            tour = [members[i] for i in local]
        out.append({"depot": int(e), "members": members, "load": load,
                    "feasible": load <= Q + 1e-9, "tour": tour})
    return out


# ---------------------------------------------------------------------------
# OSM 도로망 + 도로 따라가는 엣지
# ---------------------------------------------------------------------------
def load_road_graph():
    import osmnx as ox
    if not os.path.exists(GRAPHML):
        G = ox.graph_from_place("Seongbuk-gu, Seoul, South Korea",
                                network_type="drive")
        ox.save_graphml(G, GRAPHML)
    else:
        G = ox.load_graphml(GRAPHML)
    import osmnx.convert
    return osmnx.convert.to_undirected(G)


def map_osm_ids(Gu, mapping_csv: str, n: int) -> list[int]:
    """idx -> OSM node id.  그래프에 없는 id 는 좌표 최근접 노드로 대체."""
    df = pd.read_csv(mapping_csv).sort_values("idx")
    ids = df["node_id"].tolist()
    gx = np.array([Gu.nodes[v]["x"] for v in Gu.nodes])
    gy = np.array([Gu.nodes[v]["y"] for v in Gu.nodes])
    gids = list(Gu.nodes)
    out = []
    for _, row in df.iterrows():
        nid = int(row["node_id"])
        if nid in Gu.nodes:
            out.append(nid)
        else:
            j = int(np.argmin((gx - row["lon"]) ** 2 + (gy - row["lat"]) ** 2))
            out.append(gids[j])
            print(f"  [map] node_id {nid} not in graph -> nearest {gids[j]}")
    if len(out) != n:
        raise ValueError(f"mapping rows {len(out)} != n {n}")
    return out


def road_path_xy(Gu, a: int, b: int):
    """OSM 노드 a->b 최단경로의 도로 geometry (xs, ys).  경로 없으면 직선."""
    import networkx as nx
    try:
        path = nx.shortest_path(Gu, a, b, weight="length")
    except nx.NetworkXNoPath:
        return ([Gu.nodes[a]["x"], Gu.nodes[b]["x"]],
                [Gu.nodes[a]["y"], Gu.nodes[b]["y"]])
    xs, ys = [], []
    for u, v in zip(path[:-1], path[1:]):
        data = min(Gu.get_edge_data(u, v).values(),
                   key=lambda d: d.get("length", np.inf))
        if "geometry" in data:
            gx, gy = data["geometry"].xy
            gx, gy = list(gx), list(gy)
        else:
            gx = [Gu.nodes[u]["x"], Gu.nodes[v]["x"]]
            gy = [Gu.nodes[u]["y"], Gu.nodes[v]["y"]]
        # geometry 방향을 u -> v 로 정렬
        du = abs(gx[0] - Gu.nodes[u]["x"]) + abs(gy[0] - Gu.nodes[u]["y"])
        dv = abs(gx[-1] - Gu.nodes[u]["x"]) + abs(gy[-1] - Gu.nodes[u]["y"])
        if du > dv:
            gx, gy = gx[::-1], gy[::-1]
        xs += gx
        ys += gy
    return xs, ys


# ---------------------------------------------------------------------------
# 지도 figure
# ---------------------------------------------------------------------------
def draw_load_bars(ax, coords, clusters, Q):
    """각 클러스터 depot 옆에 load 미니 막대를 그린다.

    막대 높이 = load (Q 가 기준 높이), 연초록/연빨강 = feasible/infeasible,
    점선 = 용량 Q 기준선, 막대 위 숫자 = load 값.
    """
    lon_span = coords[:, 0].max() - coords[:, 0].min()
    lat_span = coords[:, 1].max() - coords[:, 1].min()
    bw = lon_span * 0.014          # 막대 폭
    hQ = lat_span * 0.085          # Q 에 해당하는 기준 높이
    dx = lon_span * 0.014          # depot 별표로부터의 가로 오프셋
    dy = lat_span * 0.020

    for cl in clusters:
        d = cl["depot"]
        x0 = coords[d, 0] + dx
        y0 = coords[d, 1] + dy
        h = hQ * cl["load"] / Q
        face, edge = (GREEN_LT, GREEN) if cl["feasible"] else (RED_LT, RED)
        ax.add_patch(Rectangle((x0, y0), bw, h, facecolor=face, edgecolor=edge,
                               linewidth=0.5, zorder=6))
        # 용량 Q 기준선 (막대보다 조금 넓게)
        ax.plot([x0 - 0.45 * bw, x0 + 1.45 * bw], [y0 + hQ, y0 + hQ],
                color="0.2", lw=0.55, ls=(0, (2, 1.2)), zorder=7)
        ax.text(x0 + 0.5 * bw, y0 + max(h, hQ) + lat_span * 0.010,
                f"{cl['load']:g}", ha="center", va="bottom", fontsize=7,
                zorder=8)


def draw_map(Gu, coords, osm_ids, clusters, Q, out_base):
    import osmnx as ox

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.92))
    # 배경: 도로망만 (타일/라벨 없음 -> OSM 표기 최소, 도로는 전부)
    ox.plot_graph(Gu, ax=ax, show=False, close=False, bgcolor="white",
                  node_size=0, edge_color=ROAD, edge_linewidth=0.35)

    # 클러스터 투어 엣지 -- 실제 도로를 따라
    for cl in clusters:
        color = GREEN if cl["feasible"] else RED
        tour = cl["tour"]
        for u, v in zip(tour[:-1], tour[1:]):
            if u == v:
                continue
            xs, ys = road_path_xy(Gu, osm_ids[u], osm_ids[v])
            ax.plot(xs, ys, "-", color=color, lw=1.1, alpha=0.95,
                    zorder=3, solid_capstyle="round")

    # 서비스 노드 + depot
    mem = [i for cl in clusters for i in cl["members"] if i != cl["depot"]]
    ax.scatter(coords[mem, 0], coords[mem, 1], s=7, c=NODE, zorder=4,
               linewidths=0)
    depots = [cl["depot"] for cl in clusters]
    ax.scatter(coords[depots, 0], coords[depots, 1], s=55, marker="*",
               c="black", edgecolors="white", linewidths=0.5, zorder=5)

    # 클러스터별 load 미니 막대 (연초록/연빨강 + Q 기준선 + 숫자)
    draw_load_bars(ax, coords, clusters, Q)

    # 화면을 노드 영역으로 크롭 (도로망은 잘린 범위 안에서 전부 보임)
    pad_x = (coords[:, 0].max() - coords[:, 0].min()) * 0.06
    pad_y = (coords[:, 1].max() - coords[:, 1].min()) * 0.06
    ax.set_xlim(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
    ax.set_ylim(coords[:, 1].min() - pad_y,
                coords[:, 1].max() + pad_y * 2.6)   # 위쪽 여유: 막대/숫자 클리핑 방지
    ax.set_aspect(1.0 / np.cos(np.radians(coords[:, 1].mean())))
    ax.set_axis_off()

    handles = [
        Line2D([], [], color=GREEN, lw=1.4, label="Feasible"),
        Line2D([], [], color=RED, lw=1.4, label="Infeasible"),
        Line2D([], [], color="none", marker="o", markerfacecolor=NODE,
               markeredgecolor="none", markersize=3.5, label="Service node"),
        Line2D([], [], color="none", marker="*", markerfacecolor="black",
               markeredgecolor="white", markersize=9, label="Depot"),
        Line2D([], [], color="0.2", lw=0.7, ls=(0, (2, 1.2)),
               label="Capacity $Q$"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, borderpad=0.4,
              handlelength=1.5, labelspacing=0.35)
    fig.tight_layout(pad=0.05)
    save_fig(fig, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
def summarize(name, clusters, Q):
    n_feas = sum(cl["feasible"] for cl in clusters)
    loads = [cl["load"] for cl in clusters]
    print(f"{name:>10s}: K={len(clusters):2d}  feasible {n_feas}/{len(clusters)}"
          f"   load max={max(loads):g} (Q={Q:g})"
          f"   overflow={sum(max(0.0, l - Q) for l in loads):g}")
    return n_feas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/seongbuk.csv")
    ap.add_argument("--coords", default="data/seongbuk_node_mapping.csv")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--capacity", type=float, default=12.0)
    ap.add_argument("--weights", choices=["uniform", "random"], default="uniform")
    ap.add_argument("--similarity", choices=["neg", "neg_sq"], default="neg")
    ap.add_argument("--preference", default="median")
    ap.add_argument("--pref-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ieee_rc()
    os.makedirs(args.outdir, exist_ok=True)

    D = load_matrix(args.csv)
    n = len(D)
    weights = make_weights(args.weights, n, args.seed)
    Q = args.capacity
    coords = pd.read_csv(args.coords).sort_values("idx")[["lon", "lat"]].to_numpy(float)

    # preference 해석 (run.py 와 동일)
    from run import resolve_preference
    eff_pref, _ = resolve_preference(D, list(range(n)), args.similarity,
                                     args.preference, args.pref_scale)

    print(f"exp1: {n} nodes  Q={Q:g}  weights={args.weights}")

    # ---- proposed: coupled CAP-trellis (Algorithm 1) ----------------------
    sol = solve_coupled(D, weights, Q, similarity=args.similarity,
                        preference=eff_pref)
    labels_cap = sol.labels
    K = sol.n_clusters
    print(f"proposed clustering done: K={K} (rounds={sol.n_rounds}, "
          f"fixed_point={sol.converged})")

    # ---- baselines (demand-weighted; 용량 제약 없음) ------------------------
    # weighted AP: 가중 유사도의 median preference 로 K 를 스스로 결정
    # weighted K-medoids: proposed 와 같은 K, 가중 PAM
    labels_ap = ap_labels(D, args.similarity, None, node_weights=weights)
    labels_kmed = kmedoids_labels(D, K, seed=args.seed, node_weights=weights)

    methods = [
        ("proposed", labels_cap),
        ("ap", labels_ap),
        ("kmedoids", labels_kmed),
    ]

    Gu = load_road_graph()
    osm_ids = map_osm_ids(Gu, args.coords, n)

    print(f"\n{'method':>10s}  feasibility under Q={Q:g}")
    print("-" * 60)
    for name, labels in methods:
        clusters = clusters_from_labels(D, weights, labels, Q)
        summarize(name, clusters, Q)
        draw_map(Gu, coords, osm_ids, clusters, Q,
                 os.path.join(args.outdir, f"exp1_{name}"))
    print("\ndone -> figs/exp1_*.pdf")


if __name__ == "__main__":
    main()
