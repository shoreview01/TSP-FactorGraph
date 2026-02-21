"""
BM 메시지 시각화 — iteration별 히트맵 + cost 추이
===================================================
사용법:
  python visualize_bm.py              # 기본 N=10
  python visualize_bm.py --n 15       # N=15
  python visualize_bm.py --save       # PNG 저장
  python visualize_bm.py --gif        # 애니메이션 GIF 저장
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import argparse
import os

# --- solver import ---
from tsp_factor_graph_gpu import TSPFactorGraphSolverGPU


def make_heatmap(ax, data, title, vmin, vmax, ytick_labels=None,
                 constraint_mask=None, cmap='RdBu_r'):
    """min-max 기준 diverging colormap 히트맵."""
    # 0 중심 diverging norm (양/음 구분)
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = None

    im = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm,
                   vmin=vmin if norm is None else None,
                   vmax=vmax if norm is None else None,
                   interpolation='nearest')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Time slot t', fontsize=8)
    ax.set_ylabel('City', fontsize=8)
    ax.tick_params(labelsize=7)

    # y축을 원래 도시 번호로 표시
    if ytick_labels is not None:
        ax.set_yticks(range(len(ytick_labels)))
        ax.set_yticklabels(ytick_labels)

    # 금지 셀에 × 표시
    if constraint_mask is not None:
        rows, cols = np.where(constraint_mask)
        ax.scatter(cols, rows, marker='x', color='black',
                   s=40, linewidths=1.5, alpha=0.7, zorder=5)

    return im


def plot_iteration(history, it, fig=None, global_ranges=None):
    """단일 iteration의 6개 BM 메시지 + cost 시각화."""
    if fig is None:
        fig = plt.figure(figsize=(18, 10))
    fig.clf()

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35)

    # 내부 인덱스 → 원래 도시 번호 매핑
    inv_perm = history.get('inv_perm', None)
    N = history.get('N', None)
    if inv_perm is not None and N is not None:
        ytick_labels = [str(inv_perm[i]) for i in range(N)]
    else:
        ytick_labels = None

    # 제약 마스크
    constraint_mask = history.get('constraint_mask', None)

    panels = [
        ('gamma', 'γ̃ (BM → Trellis)'),
        ('omega', 'ω̃ (BM → Trellis)'),
        ('rho',   'ρ̃ = η̃ + φ̃ (Assignment)'),
        ('delta', 'δ̃ (Trellis → BM)'),
        ('phi',   'φ̃ (Column msg)'),
        ('eta',   'η̃ (Row msg)'),
    ]

    for idx, (key, title) in enumerate(panels):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        data = history[key][it]

        if global_ranges:
            vmin, vmax = global_ranges[key]
        else:
            # 제약 셀 제외하고 min/max 계산 (constrained 값이 스케일 지배 방지)
            if constraint_mask is not None and constraint_mask.any():
                unconstrained = data[~constraint_mask]
                if unconstrained.size > 0:
                    vmin, vmax = unconstrained.min(), unconstrained.max()
                else:
                    vmin, vmax = data.min(), data.max()
            else:
                vmin, vmax = data.min(), data.max()

        # 값이 전부 0이면 범위 살짝 확장
        if abs(vmax - vmin) < 1e-15:
            vmin, vmax = -0.1, 0.1

        im = make_heatmap(ax, data,
                         f'{title}\niter {it+1}  [{vmin:.3f}, {vmax:.3f}]',
                         vmin, vmax, ytick_labels=ytick_labels,
                         constraint_mask=constraint_mask)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.2f')

    # ── Cost 추이 (우측 패널) ──
    ax_cost = fig.add_subplot(gs[:, 3])
    costs = history['cost'][:it+1]
    iters = range(1, len(costs) + 1)

    ax_cost.plot(iters, costs, 'o-', color='#2196F3', markersize=4, linewidth=1.5)
    ax_cost.axhline(y=min(costs), color='#4CAF50', linestyle='--',
                    linewidth=1, alpha=0.7, label=f'best={min(costs):.4f}')

    # 현재 iteration 강조
    ax_cost.plot(it+1, costs[it], 'o', color='#F44336', markersize=10, zorder=5)

    ax_cost.set_xlabel('Iteration', fontsize=10)
    ax_cost.set_ylabel('Route Cost', fontsize=10)
    ax_cost.set_title('Cost per Iteration', fontsize=10, fontweight='bold')
    ax_cost.legend(fontsize=8)
    ax_cost.grid(True, alpha=0.3)

    # 경로 텍스트
    route = history['route'][it]
    route_str = '→'.join(map(str, route))
    fig.suptitle(f'BM Message Heatmaps — Iteration {it+1}/{len(history["cost"])}    '
                 f'cost={costs[it]:.4f}    route: {route_str}',
                 fontsize=12, fontweight='bold', y=0.98)

    return fig


def compute_global_ranges(history):
    """전체 iteration에 걸친 global min/max (일관된 색상 스케일용)."""
    ranges = {}
    constraint_mask = history.get('constraint_mask', None)
    for key in ['gamma', 'omega', 'rho', 'delta', 'phi', 'eta']:
        all_data = np.stack(history[key])
        if constraint_mask is not None and constraint_mask.any():
            # broadcast mask to all iterations
            mask_3d = np.broadcast_to(constraint_mask, all_data.shape)
            unconstrained = all_data[~mask_3d]
            if unconstrained.size > 0:
                ranges[key] = (unconstrained.min(), unconstrained.max())
            else:
                ranges[key] = (all_data.min(), all_data.max())
        else:
            ranges[key] = (all_data.min(), all_data.max())
    return ranges


def interactive_viewer(history, global_scale=False):
    """키보드 ←→ 로 iteration 탐색하는 인터랙티브 뷰어."""
    global_ranges = compute_global_ranges(history) if global_scale else None
    total = len(history['cost'])

    fig = plt.figure(figsize=(18, 10))
    state = {'idx': 0}

    def update():
        plot_iteration(history, state['idx'], fig, global_ranges)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            state['idx'] = min(state['idx'] + 1, total - 1)
        elif event.key == 'left':
            state['idx'] = max(state['idx'] - 1, 0)
        elif event.key == 'home':
            state['idx'] = 0
        elif event.key == 'end':
            state['idx'] = total - 1
        elif event.key == 'q':
            plt.close(fig)
            return
        update()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update()
    plt.show()


def save_all_frames(history, out_dir='bm_frames', global_scale=False):
    """전체 iteration을 PNG로 저장."""
    os.makedirs(out_dir, exist_ok=True)
    global_ranges = compute_global_ranges(history) if global_scale else None
    total = len(history['cost'])
    fig = plt.figure(figsize=(18, 10))

    for it in range(total):
        plot_iteration(history, it, fig, global_ranges)
        path = os.path.join(out_dir, f'iter_{it+1:03d}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  saved {path}')

    plt.close(fig)
    print(f'Done: {total} frames → {out_dir}/')


def save_gif(history, out_path='bm_messages.gif', fps=3, global_scale=False):
    """전체 iteration을 GIF 애니메이션으로 저장."""
    try:
        from PIL import Image
    except ImportError:
        print("GIF 저장에는 Pillow가 필요합니다: pip install Pillow")
        return

    import io
    global_ranges = compute_global_ranges(history) if global_scale else None
    total = len(history['cost'])
    fig = plt.figure(figsize=(18, 10))
    frames = []

    for it in range(total):
        plot_iteration(history, it, fig, global_ranges)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)

    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0
    )
    print(f'GIF saved: {out_path} ({total} frames, {fps} fps)')


def main():
    parser = argparse.ArgumentParser(description='BM 메시지 히트맵 시각화')
    parser.add_argument('--n', type=int, default=10, help='도시 수 (depot 포함)')
    parser.add_argument('--iters', type=int, default=50, help='최대 반복')
    parser.add_argument('--damping', type=float, default=0.3, help='damping factor')
    parser.add_argument('--patience', type=int, default=20, help='early stop patience')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default=None, help='cuda/mps/cpu')
    parser.add_argument('--save', action='store_true', help='PNG 프레임 저장')
    parser.add_argument('--gif', action='store_true', help='GIF 애니메이션 저장')
    parser.add_argument('--global-scale', action='store_true', default=False,
                        help='전체 iteration 기준 색상 스케일 (기본 off, per-iteration)')
    parser.add_argument('--constrained', action='store_true', default=False,
                        help='time window 제약 조건 추가 (일부 도시에 무작위 제약)')
    parser.add_argument('--constraint-ratio', type=float, default=0.3,
                        help='제약할 도시 비율 (기본 30%%)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    D = np.random.rand(args.n, args.n)
    np.fill_diagonal(D, 0)

    N = args.n - 1  # depot 제외

    # 제약 조건 생성
    constraints = None
    if args.constrained:
        constraints = TSPFactorGraphSolverGPU.make_constraints(N)
        rng = np.random.default_rng(args.seed)
        n_constrained = max(1, int(N * args.constraint_ratio))
        cities = rng.choice(N, size=n_constrained, replace=False)

        for c in cities:
            # 각 도시에 랜덤 time window (폭 = N의 30~70%)
            width = rng.integers(max(1, N * 3 // 10), max(2, N * 7 // 10) + 1)
            start = rng.integers(0, max(1, N - width + 1))
            TSPFactorGraphSolverGPU.time_window(
                constraints, city=c, earliest=start, latest=start + width - 1
            )
            print(f"  City {c}: time window [{start}, {start+width-1}]")

        print(f"\n{n_constrained} cities constrained out of {N}")

    print(f"N={args.n}, iters={args.iters}, damping={args.damping}, "
          f"constrained={args.constrained}")

    solver = TSPFactorGraphSolverGPU(
        D, start_city=0,
        damping=args.damping,
        iters=args.iters,
        verbose=True,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        constraints=constraints,
    )

    route, cost = solver.run(record_history=True)
    history = solver.history

    # 제약 정보: 내부 인덱스 순서로 변환된 마스크 사용
    if constraints is not None and solver.has_constraints:
        # constraint_bias는 이미 내부 인덱스 순서
        cb = solver.constraint_bias.cpu().numpy()
        history['constraint_mask'] = (cb <= solver.HARD / 2)
    else:
        history['constraint_mask'] = None

    print(f"\nFinal: cost={cost:.6f}, route={route}")
    print(f"Total iterations recorded: {len(history['cost'])}")

    if args.save:
        save_all_frames(history, global_scale=args.global_scale)
    elif args.gif:
        save_gif(history, global_scale=args.global_scale)
    else:
        interactive_viewer(history, global_scale=args.global_scale)


if __name__ == '__main__':
    main()