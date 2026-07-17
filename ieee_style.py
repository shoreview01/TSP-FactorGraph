"""
ieee_style.py
=============

IEEE Transactions 규격 matplotlib 스타일.  실험 스크립트(exp1/exp2)가 공유한다.

  * 단일 컬럼 폭 3.5 in (double column 7.16 in), serif(Times) 폰트, 8 pt
  * 벡터 PDF(Type 42 폰트 임베딩) + 600 dpi PNG 동시 저장
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

COL_W = 3.5          # IEEE single-column width (inch)
COL_W2 = 7.16        # IEEE double-column width (inch)


def ieee_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.linewidth": 0.6,
        "legend.fontsize": 7,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "0.2",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,      # embed TrueType (IEEE PDF eXpress friendly)
        "ps.fonttype": 42,
    })


def save_fig(fig, out_base: str):
    """out_base(확장자 없는 경로)에 .pdf 와 .png 를 함께 저장."""
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png")
    print(f"  saved {out_base}.pdf / .png")
