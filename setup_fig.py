"""
setup_fig.py
============

Simulation setup figure (논문 Sec. IV 도입부) — digital-twin 구성.

  (a) 위성 사진 (실제 환경, 고려대 캠퍼스 빨간 윤곽 포함) 을 위에,
  (b) 성북구 digital twin (도로망 + 경계 + 84 service node + 고려대 윤곽) 을
      아래에 배치.

디지털 트윈 시각 문법:
  * 위성 사진은 둥근 모서리 + 테두리 + 그림자의 '실사 카드'
  * 사진 아래 모서리 -> 지도 위 모서리로 점선 투영선 (physical -> digital 층)
  * 중앙 아래 화살표 + "Digital twin" 라벨
  * 고려대 부지를 두 패널 모두 같은 빨간 윤곽으로 표시 (두 층의 앵커)

지도 스타일: 성북구 도로(파랑, 강조) + 주변 도로(연회색, 문맥) + 경계(남색)
+ service node(진홍 점).  zone 표시 없음.

실행 (conda env `tsp`):
  C:/Users/guild/.conda/envs/tsp/python.exe setup_fig.py
출력: figs/fig_setup.pdf/.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ieee_style import ieee_rc, save_fig, COL_W
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import ConnectionPatch

NAVY = "#17457c"       # 성북구 경계
ROAD_IN = "#4577b4"    # 성북구 내부 도로 (강조)
ROAD_OUT = "#d8d8d8"   # 주변 문맥 도로
NODE = "#c81e3c"       # service node
KU_RED = "#a50f15"     # 고려대 윤곽 (위성 사진의 빨강과 맞춤)


def draw_map(ax):
    import osmnx as ox
    import geopandas as gpd

    Gc = ox.load_graphml(os.path.join("data", "seongbuk_context.graphml"))
    Gs = ox.load_graphml(os.path.join("data", "seongbuk_drive.graphml"))
    bnd = gpd.read_file(os.path.join("data", "seongbuk_boundary.geojson"))
    ku = gpd.read_file(os.path.join("data", "ku_campus.geojson"))
    coords = pd.read_csv(os.path.join("data", "seongbuk_node_mapping.csv")) \
               .sort_values("idx")[["lon", "lat"]].to_numpy(float)

    ox.plot_graph(Gc, ax=ax, show=False, close=False, bgcolor="white",
                  node_size=0, edge_color=ROAD_OUT, edge_linewidth=0.28)
    ox.plot_graph(Gs, ax=ax, show=False, close=False, bgcolor="white",
                  node_size=0, edge_color=ROAD_IN, edge_linewidth=0.36)
    bnd.boundary.plot(ax=ax, color=NAVY, linewidth=1.6, zorder=4)
    ku.boundary.plot(ax=ax, color=KU_RED, linewidth=1.3, zorder=5)
    ax.scatter(coords[:, 0], coords[:, 1], s=13, c=NODE, edgecolors="white",
               linewidths=0.35, zorder=6)

    # KU 라벨 (윤곽 옆)
    kx, ky = ku.geometry.iloc[0].centroid.x, ku.geometry.iloc[0].centroid.y
    ax.annotate("Korea Univ.", xy=(kx, ky), xytext=(kx + 0.006, ky - 0.0048),
                fontsize=6, color=KU_RED, fontstyle="italic",
                arrowprops=dict(arrowstyle="-", color=KU_RED, lw=0.6,
                                shrinkA=0, shrinkB=2), zorder=7)

    minx, miny, maxx, maxy = bnd.total_bounds
    pad_x, pad_y = (maxx - minx) * 0.05, (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect(1.0 / np.cos(np.radians((miny + maxy) / 2)))
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figs")
    args = ap.parse_args()

    ieee_rc()
    os.makedirs(args.outdir, exist_ok=True)

    sat = mpimg.imread(os.path.join("data", "sate_map.png"))

    fig = plt.figure(figsize=(COL_W, COL_W * 1.46))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.95, 1.15],
                          hspace=0.22, left=0.02, right=0.98,
                          top=0.965, bottom=0.055)
    ax_sat = fig.add_subplot(gs[0])
    ax_map = fig.add_subplot(gs[1])

    # ---- (a) 위성 사진: 둥근 모서리 카드 + 그림자 ---------------------------
    ax_sat.imshow(sat, aspect="auto")
    ax_sat.set_axis_off()
    frame = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                           boxstyle="round,pad=0,rounding_size=0.035",
                           transform=ax_sat.transAxes,
                           facecolor="none", edgecolor="0.25", linewidth=0.9,
                           zorder=5)
    ax_sat.add_patch(frame)
    for im in ax_sat.get_images():
        im.set_clip_path(frame)
    shadow = FancyBboxPatch((0.012, -0.030), 1.0, 1.0,
                            boxstyle="round,pad=0,rounding_size=0.035",
                            transform=ax_sat.transAxes,
                            facecolor="0.72", edgecolor="none", zorder=-1)
    ax_sat.add_patch(shadow)
    ax_sat.set_title("(a) Physical environment (Seongbuk-gu, Seoul)",
                     fontsize=7.5, pad=3)

    # ---- (b) digital twin 지도 (캡션은 지도 아래) ---------------------------
    draw_map(ax_map)
    pos_map = ax_map.get_position()
    fig.text((pos_map.x0 + pos_map.x1) / 2, pos_map.y0 - 0.018,
             "(b) Digital twin: road network and service nodes",
             fontsize=7.5, ha="center", va="top")

    # ---- 투영 점선 (physical -> digital 층) ---------------------------------
    for x in (0.0, 1.0):
        cp = ConnectionPatch(xyA=(x, 0.0), coordsA=ax_sat.transAxes,
                             xyB=(x, 1.0), coordsB=ax_map.transAxes,
                             linestyle=(0, (2.5, 2.5)), color="0.55",
                             linewidth=0.7, zorder=1)
        fig.add_artist(cp)

    # ---- 중앙 화살표 + 라벨 (패널 사이 간격에 정확히) ----------------------
    arrow = ConnectionPatch(xyA=(0.5, 0.0), coordsA=ax_sat.transAxes,
                            xyB=(0.5, 1.0), coordsB=ax_map.transAxes,
                            arrowstyle="-|>", mutation_scale=13,
                            linewidth=1.5, color="0.15", zorder=6,
                            shrinkA=3, shrinkB=3)
    fig.add_artist(arrow)
    pos_sat = ax_sat.get_position()
    y_mid = (pos_sat.y0 + pos_map.y1) / 2
    fig.text(0.535, y_mid, "Digital twin", fontsize=7, fontstyle="italic",
             ha="left", va="center", color="0.15")

    save_fig(fig, os.path.join(args.outdir, "fig_setup"))
    plt.close(fig)
    print("done -> figs/fig_setup.pdf")


if __name__ == "__main__":
    main()
