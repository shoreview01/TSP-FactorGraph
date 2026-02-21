"""
TSPLIB 파일 파서 + Factor-Graph 솔버 실행기
=============================================
사용법:
  python run_tsplib.py ALL_tsp/gr17.tsp.gz
  python run_tsplib.py ALL_tsp/berlin52.tsp.gz --iters 50 --damping 0.5
  python run_tsplib.py ALL_tsp/eil51.tsp.gz --constrained --constraint-ratio 0.3
"""

import gzip
import math
import os
import sys
import time
import argparse
import numpy as np

from tsp_factor_graph_gpu import TSPFactorGraphSolverGPU, get_device


# =================================================================
#                      TSPLIB 파서
# =================================================================

def parse_tsplib(filepath: str) -> dict:
    """
    .tsp 또는 .tsp.gz 파일을 파싱.
    반환: {
        'name': str,
        'dimension': int,
        'edge_weight_type': str,
        'D': np.ndarray (N x N distance matrix),
        'coords': np.ndarray or None (N x 2),
    }
    """
    # 파일 열기 (gz 자동 감지)
    if filepath.endswith('.gz'):
        f = gzip.open(filepath, 'rt', encoding='utf-8', errors='replace')
    else:
        f = open(filepath, 'r', encoding='utf-8', errors='replace')

    with f:
        lines = f.readlines()

    # 헤더 파싱
    meta = {}
    data_section = None
    data_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('EOF'):
            if data_section:
                continue
            break

        # 섹션 시작 감지
        if line == 'NODE_COORD_SECTION':
            data_section = 'NODE_COORD'
            continue
        elif line == 'EDGE_WEIGHT_SECTION':
            data_section = 'EDGE_WEIGHT'
            continue
        elif line == 'DISPLAY_DATA_SECTION':
            data_section = 'DISPLAY_DATA'
            continue
        elif line == 'EOF':
            break

        if data_section:
            data_lines.append(line)
        else:
            # 메타데이터 키:값
            if ':' in line:
                key, val = line.split(':', 1)
                meta[key.strip().upper()] = val.strip()

    name = meta.get('NAME', os.path.basename(filepath))
    dimension = int(meta.get('DIMENSION', 0))
    edge_weight_type = meta.get('EDGE_WEIGHT_TYPE', 'EUC_2D').upper()
    edge_weight_format = meta.get('EDGE_WEIGHT_FORMAT', '').upper()

    print(f"  Name: {name}")
    print(f"  Dimension: {dimension}")
    print(f"  Edge weight type: {edge_weight_type}")
    if edge_weight_format:
        print(f"  Edge weight format: {edge_weight_format}")

    coords = None
    D = None

    if edge_weight_type == 'EXPLICIT':
        D = _parse_explicit_weights(data_lines, dimension, edge_weight_format)
    else:
        # 좌표 기반
        coords = _parse_coords(data_lines, dimension)
        D = _compute_distance_matrix(coords, edge_weight_type)

    assert D.shape == (dimension, dimension), \
        f"Distance matrix shape {D.shape} != ({dimension}, {dimension})"

    return {
        'name': name,
        'dimension': dimension,
        'edge_weight_type': edge_weight_type,
        'D': D,
        'coords': coords,
    }


def _parse_coords(data_lines: list, n: int) -> np.ndarray:
    """NODE_COORD_SECTION 파싱 → (N, 2) 좌표."""
    coords = np.zeros((n, 2), dtype=np.float64)
    count = 0
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 3:
            idx = int(parts[0]) - 1  # 1-indexed → 0-indexed
            x, y = float(parts[1]), float(parts[2])
            if 0 <= idx < n:
                coords[idx] = [x, y]
                count += 1
    assert count == n, f"Expected {n} coords, got {count}"
    return coords


def _compute_distance_matrix(coords: np.ndarray, edge_weight_type: str) -> np.ndarray:
    """좌표에서 거리 행렬 계산."""
    n = len(coords)
    D = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            d = _distance(coords[i], coords[j], edge_weight_type)
            D[i, j] = d
            D[j, i] = d
    return D


def _distance(c1, c2, edge_weight_type: str) -> float:
    """TSPLIB 거리 함수."""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]

    if edge_weight_type == 'EUC_2D':
        return round(math.sqrt(dx * dx + dy * dy))

    elif edge_weight_type == 'CEIL_2D':
        return math.ceil(math.sqrt(dx * dx + dy * dy))

    elif edge_weight_type == 'ATT':
        r = math.sqrt((dx * dx + dy * dy) / 10.0)
        t = round(r)
        return t + 1 if t < r else t

    elif edge_weight_type == 'GEO':
        # 지리적 좌표 → 라디안 변환
        def to_rad(c):
            deg = int(c)
            minute = c - deg
            return math.pi * (deg + 5.0 * minute / 3.0) / 180.0

        lat1, lon1 = to_rad(c1[0]), to_rad(c1[1])
        lat2, lon2 = to_rad(c2[0]), to_rad(c2[1])
        RRR = 6378.388
        q1 = math.cos(lon1 - lon2)
        q2 = math.cos(lat1 - lat2)
        q3 = math.cos(lat1 + lat2)
        return int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)

    elif edge_weight_type == 'MAN_2D':
        return round(abs(dx) + abs(dy))

    elif edge_weight_type == 'MAX_2D':
        return round(max(abs(dx), abs(dy)))

    else:
        # fallback: Euclidean
        return round(math.sqrt(dx * dx + dy * dy))


def _parse_explicit_weights(data_lines: list, n: int, fmt: str) -> np.ndarray:
    """EDGE_WEIGHT_SECTION 파싱 (다양한 포맷 지원)."""
    # 모든 숫자 추출
    values = []
    for line in data_lines:
        if line.strip() == 'EOF':
            break
        for token in line.split():
            try:
                values.append(float(token))
            except ValueError:
                break  # 섹션 끝

    D = np.zeros((n, n), dtype=np.float64)
    idx = 0

    if fmt == 'FULL_MATRIX':
        for i in range(n):
            for j in range(n):
                D[i, j] = values[idx]; idx += 1

    elif fmt == 'UPPER_ROW':
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = values[idx]; D[j, i] = values[idx]; idx += 1

    elif fmt == 'LOWER_ROW':
        for i in range(1, n):
            for j in range(i):
                D[i, j] = values[idx]; D[j, i] = values[idx]; idx += 1

    elif fmt == 'UPPER_DIAG_ROW':
        for i in range(n):
            for j in range(i, n):
                D[i, j] = values[idx]; D[j, i] = values[idx]; idx += 1

    elif fmt == 'LOWER_DIAG_ROW':
        for i in range(n):
            for j in range(i + 1):
                D[i, j] = values[idx]; D[j, i] = values[idx]; idx += 1

    else:
        # guess: try FULL_MATRIX if enough values
        if len(values) == n * n:
            for i in range(n):
                for j in range(n):
                    D[i, j] = values[idx]; idx += 1
        elif len(values) == n * (n - 1) // 2:
            # UPPER_ROW
            for i in range(n):
                for j in range(i + 1, n):
                    D[i, j] = values[idx]; D[j, i] = values[idx]; idx += 1
        elif len(values) == n * (n + 1) // 2:
            # LOWER_DIAG_ROW
            for i in range(n):
                for j in range(i + 1):
                    D[i, j] = values[idx]; D[j, i] = values[idx]; idx += 1
        else:
            raise ValueError(
                f"Unknown EDGE_WEIGHT_FORMAT '{fmt}', "
                f"got {len(values)} values for n={n}"
            )

    return D


# =================================================================
#                  알려진 최적해 (TSPLIB)
# =================================================================
KNOWN_OPTIMAL = {
    'gr17': 2085, 'gr21': 2707, 'gr24': 1272, 'gr48': 5046,
    'gr96': 55209, 'gr120': 6942, 'gr137': 69853, 'gr202': 40160,
    'gr229': 134602, 'gr431': 171414, 'gr666': 294358,
    'bayg29': 1610, 'bays29': 2020, 'fri26': 937,
    'dantzig42': 699, 'swiss42': 1273, 'att48': 10628,
    'hk48': 11461, 'eil51': 426, 'berlin52': 7542,
    'brazil58': 25395, 'st70': 675, 'eil76': 538,
    'pr76': 108159, 'kroA100': 21282, 'kroB100': 22141,
    'kroC100': 20749, 'kroD100': 21294, 'kroE100': 22068,
    'rd100': 7910, 'eil101': 629, 'lin105': 14379,
    'pr107': 44303, 'pr124': 59030, 'bier127': 118282,
    'ch130': 6110, 'pr136': 96772, 'pr144': 58537,
    'ch150': 6528, 'kroA150': 26524, 'kroB150': 26130,
    'pr152': 73682, 'u159': 42080, 'rat195': 2323,
    'd198': 15780, 'kroA200': 29368, 'kroB200': 29437,
    'ts225': 126643, 'tsp225': 3916, 'pr226': 80369,
    'a280': 2579, 'pr299': 48191, 'lin318': 42029,
    'linhp318': 41345, 'rd400': 15281, 'fl417': 11861,
    'pr439': 107217, 'pcb442': 50778, 'd493': 35002,
    'att532': 27686, 'ali535': 202310, 'si535': 48450,
    'pa561': 2763, 'u574': 36905, 'rat575': 6773,
    'p654': 34643, 'd657': 48912, 'u724': 41910,
    'rat783': 8806, 'pr1002': 259045, 'u1060': 224094,
    'vm1084': 239297, 'pcb1173': 56892, 'd1291': 50801,
    'rl1304': 252948, 'rl1323': 270199, 'nrw1379': 56638,
    'fl1400': 20127, 'u1432': 152970, 'fl1577': 22249,
    'vm1748': 336556, 'u1817': 57201, 'rl1889': 316536,
    'd2103': 80450, 'u2152': 64253, 'u2319': 234256,
    'pr2392': 378032, 'pcb3038': 137694, 'fl3795': 28772,
    'fnl4461': 182566, 'rl5915': 565530, 'rl5934': 556045,
    'burma14': 3323, 'ulysses16': 6859, 'ulysses22': 7013,
    'si175': 21407, 'si1032': 92650, 'brg180': 1950,
    'p43': 5620, 'ftv33': 1286, 'ftv35': 1473,
    'ftv38': 1530, 'ftv44': 1613, 'ftv47': 1776,
    'ftv55': 1608, 'ftv64': 1839, 'ftv70': 1950,
    'ft53': 6905, 'ft70': 38673, 'ry48p': 14422,
    'kro124p': 36230,
}


def get_known_optimal(name: str):
    """이름으로 최적해 검색. 없으면 None."""
    # 정확히 일치
    if name in KNOWN_OPTIMAL:
        return KNOWN_OPTIMAL[name]
    # 소문자 비교
    name_lower = name.lower()
    for k, v in KNOWN_OPTIMAL.items():
        if k.lower() == name_lower:
            return v
    return None


# =================================================================
#                         메인
# =================================================================
def main():
    parser = argparse.ArgumentParser(
        description='TSPLIB 파일을 Factor-Graph 솔버로 풀기'
    )
    parser.add_argument('file', type=str, help='.tsp 또는 .tsp.gz 파일 경로')
    parser.add_argument('--start', type=int, default=0,
                        help='시작 도시 (0-indexed, 기본 0)')
    parser.add_argument('--iters', type=int, default=100, help='최대 반복')
    parser.add_argument('--damping', type=float, default=0.3, help='damping factor')
    parser.add_argument('--patience', type=int, default=20, help='early stop patience')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default=None, help='cuda/mps/cpu')
    parser.add_argument('--verbose', action='store_true', help='매 iteration 출력')

    # 제약 조건
    parser.add_argument('--constrained', action='store_true',
                        help='랜덤 time window 제약 추가')
    parser.add_argument('--constraint-ratio', type=float, default=0.3,
                        help='제약할 도시 비율 (기본 30%%)')

    # 시각화
    parser.add_argument('--viz', action='store_true',
                        help='BM 메시지 히트맵 시각화')
    parser.add_argument('--gif', action='store_true',
                        help='GIF 애니메이션 저장')

    args = parser.parse_args()

    # ── 파일 파싱 ──
    print(f"\n{'='*60}")
    print(f"  Loading: {args.file}")
    print(f"{'='*60}")

    if not os.path.exists(args.file):
        print(f"Error: 파일 없음 → {args.file}")
        sys.exit(1)

    tsp = parse_tsplib(args.file)
    D = tsp['D']
    N_total = tsp['dimension']
    N = N_total - 1  # depot 제외

    known_opt = get_known_optimal(tsp['name'])
    if known_opt:
        print(f"  Known optimal: {known_opt}")

    # N 범위 확인
    if N > 20:
        print(f"\n  ⚠ N={N_total} (도시 {N}개 + depot)")
        print(f"    현재 exact 솔버는 N≤20까지 지원합니다.")
        print(f"    N={N}: M=2^{N} = {2**N:,} 상태 → "
              f"메모리 ~{4*(N+1)*(2**N)*N*4/(1024**3):.1f} GB (float32)")
        sys.exit(1)

    print(f"  Cities: {N_total} (N={N} + depot)")
    print(f"  D range: [{D.min():.1f}, {D.max():.1f}]")

    # ── 제약 조건 ──
    constraints = None
    if args.constrained:
        constraints = TSPFactorGraphSolverGPU.make_constraints(N)
        rng = np.random.default_rng(args.seed)
        n_constrained = max(1, int(N * args.constraint_ratio))
        cities = rng.choice(N, size=n_constrained, replace=False)

        print(f"\n  Constraints ({n_constrained}/{N} cities):")
        for c in cities:
            width = rng.integers(max(1, N * 3 // 10), max(2, N * 7 // 10) + 1)
            start = rng.integers(0, max(1, N - width + 1))
            TSPFactorGraphSolverGPU.time_window(
                constraints, city=c, earliest=start, latest=start + width - 1
            )
            print(f"    City {c+1}: time [{start}, {start+width-1}]")

    # ── 솔버 실행 ──
    print(f"\n{'='*60}")
    print(f"  Solving: {tsp['name']}")
    print(f"{'='*60}")

    record = args.viz or args.gif

    solver = TSPFactorGraphSolverGPU(
        D, start_city=args.start,
        damping=args.damping,
        iters=args.iters,
        verbose=args.verbose,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        constraints=constraints,
    )

    t0 = time.time()
    route, cost = solver.run(record_history=record)
    elapsed = time.time() - t0

    # ── 결과 출력 ──
    print(f"\n{'='*60}")
    print(f"  Result: {tsp['name']}")
    print(f"{'='*60}")
    print(f"  Cost:    {cost:.2f}")
    if known_opt:
        gap = (cost - known_opt) / known_opt * 100
        print(f"  Optimal: {known_opt}")
        print(f"  Gap:     {gap:+.2f}%")
    print(f"  Route:   {'→'.join(str(c) for c in route)}")
    print(f"  Time:    {elapsed:.2f}s")
    print(f"  Device:  {solver.device}")
    print(f"  Iters:   {len(solver.history['cost']) if record else '?'}")

    # ── 시각화 ──
    if record:
        history = solver.history
        if constraints is not None and solver.has_constraints:
            cb = solver.constraint_bias.cpu().numpy()
            history['constraint_mask'] = (cb <= solver.HARD / 2)
        else:
            history['constraint_mask'] = None

        if args.gif:
            from visualize_bm import save_gif
            out = f"bm_{tsp['name']}.gif"
            save_gif(history, out_path=out, global_scale=False)
            print(f"\n  GIF saved: {out}")
        elif args.viz:
            from visualize_bm import interactive_viewer
            interactive_viewer(history, global_scale=False)

    solver.cleanup()


if __name__ == '__main__':
    main()
