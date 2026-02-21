# TSP Factor-Graph Message-Passing Solver (GPU)

논문의 Factor-Graph BP 알고리즘을 PyTorch GPU 가속으로 구현한 솔버 + 벤치마크 + BM 메시지 시각화 도구.

## 파일 구성

| 파일 | 설명 |
|------|------|
| `tsp_factor_graph_gpu.py` | GPU 가속 솔버 (Exact DP + Beam Search) |
| `tsp_benchmarks.py` | 벤치마크 비교 (Held-Karp, NN, 2-opt, GA, OR-Tools) |
| `run_tsplib.py` | TSPLIB 파일 파서 + 솔버 실행 CLI |
| `visualize_bm.py` | BM 메시지 히트맵 시각화 (인터랙티브 / PNG / GIF) |

## 설치

```bash
pip install torch numpy matplotlib
pip install Pillow      # GIF 저장 시 필요
pip install ortools     # OR-Tools 벤치마크 (선택)
```

## 솔버 모드

| 모드 | 파라미터 | 복잡도 | N 범위 |
|------|---------|--------|--------|
| **Exact DP** | `beam_width=None` (기본) | O(2^N · N · T) | N ≤ 20 |
| **Beam Search** | `beam_width=B` | O(B · N² · T) | N ≤ 63 |

Exact DP는 2^N 상태를 전수 탐색하여 최적해 보장. Beam search는 각 step에서 top-B 상태만 유지하여 근사해를 빠르게 탐색.

## 벤치마크 비교

```bash
# 단일 인스턴스 비교 (N=12)
python tsp_benchmarks.py --n 12

# 10 trials 평균 ± std
python tsp_benchmarks.py --n 12 --trials 10

# 제약 조건 포함
python tsp_benchmarks.py --n 12 --constrained --trials 10

# Beam search 포함 (N>20)
python tsp_benchmarks.py --n 25 --beam 500 --trials 5

# TSPLIB 파일로 벤치마크
python tsp_benchmarks.py --tsplib ALL_tsp/gr17.tsp.gz

# OR-Tools 제외
python tsp_benchmarks.py --n 15 --no-ortools
```

출력 예시 (10 trials):

```
======================================================================
  10 trials, N=12, seed=42~51
======================================================================

Method                  Mean Cost    Time (mean ± std)     Gap (mean ± std)
--------------------------------------------------------------------------
Held-Karp (Exact)          2.8050    0.0047s ± 0.0002s    +0.00% ±  0.00%
FG-BP (Exact)              2.8050    0.1234s ± 0.0100s    +0.00% ±  0.00%
FG-BP (Beam=500)           2.8312    0.0456s ± 0.0030s    +0.93% ±  0.45%
Genetic Algorithm          2.8050    1.1059s ± 0.0207s    +0.00% ±  0.00%
NN + 2-opt                 2.8134    0.0001s ± 0.0000s    +0.26% ±  0.52%
Nearest Neighbor           3.0972    0.0000s ± 0.0000s    +9.98% ±  7.38%
```

### 벤치마크 솔버

| 솔버 | 복잡도 | 역할 |
|------|--------|------|
| **Held-Karp** | O(2^N · N²) exact | Ground truth |
| **Nearest Neighbor** | O(N²) greedy | 하한 baseline |
| **NN + 2-opt** | O(N² · k) local search | 실용적 휴리스틱 |
| **Genetic Algorithm** | Population-based meta | OX + Inversion + Elitism |
| **OR-Tools** | 산업 표준 | 상한 baseline |

## TSPLIB 파일 풀기

```bash
# 기본 실행
python run_tsplib.py ALL_tsp/gr17.tsp.gz

# Beam search로 큰 인스턴스
python run_tsplib.py ALL_tsp/att48.tsp.gz --beam 1000

# 파라미터 조정
python run_tsplib.py ALL_tsp/gr21.tsp.gz --iters 200 --damping 0.5

# 매 iteration 출력
python run_tsplib.py ALL_tsp/fri26.tsp.gz --verbose

# 제약 조건 추가
python run_tsplib.py ALL_tsp/bayg29.tsp.gz --constrained

# BM 메시지 시각화
python run_tsplib.py ALL_tsp/gr17.tsp.gz --viz

# GIF 저장
python run_tsplib.py ALL_tsp/gr17.tsp.gz --gif
```

지원 TSPLIB 포맷: EUC_2D, CEIL_2D, ATT, GEO, MAN_2D, MAX_2D, EXPLICIT (FULL_MATRIX, UPPER_ROW, LOWER_ROW, UPPER_DIAG_ROW, LOWER_DIAG_ROW). `.tsp`와 `.tsp.gz` 모두 지원.

알려진 최적해가 있으면 자동으로 gap을 계산해 줍니다. 제약 조건이 있으면 exact constrained baseline과 자동 비교.

## 솔버 단독 실행

```bash
# 기본 (무제약 vs 제약 vs Beam 비교)
python tsp_factor_graph_gpu.py
```

## 시각화

### 기본 (인터랙티브 뷰어)

```bash
# N=10 도시, 50 iterations
python visualize_bm.py --n 10 --iters 50

# 키보드 조작:
#   ← →   : iteration 이동
#   Home   : 첫 iteration
#   End    : 마지막 iteration
#   q      : 종료
```

### 제약 조건

```bash
# 30% 도시에 랜덤 time window 제약 (기본)
python visualize_bm.py --n 10 --constrained

# 50% 도시에 제약
python visualize_bm.py --n 10 --constrained --constraint-ratio 0.5
```

### Beam Search 시각화

```bash
# Beam 모드로 시각화
python visualize_bm.py --n 25 --beam 500 --iters 30

# Beam + 제약
python visualize_bm.py --n 20 --beam 200 --constrained
```

### 저장

```bash
# PNG 프레임 (bm_frames/ 폴더에 저장)
python visualize_bm.py --n 10 --iters 30 --save

# GIF 애니메이션
python visualize_bm.py --n 10 --iters 30 --gif
```

### GPU 지정

```bash
python visualize_bm.py --n 15 --device cuda
python visualize_bm.py --n 15 --device mps
python visualize_bm.py --n 15 --device cpu
```

## Python에서 직접 사용

### Exact DP (N ≤ 20)

```python
import numpy as np
from tsp_factor_graph_gpu import TSPFactorGraphSolverGPU

D = np.random.rand(11, 11)
np.fill_diagonal(D, 0)

solver = TSPFactorGraphSolverGPU(D, start_city=0, damping=0.3, iters=100)
route, cost = solver.run()
print(route, cost)
solver.cleanup()
```

### Beam Search (N > 20)

```python
D = np.random.rand(50, 50)
np.fill_diagonal(D, 0)

solver = TSPFactorGraphSolverGPU(
    D, start_city=0, damping=0.3, iters=50,
    beam_width=1000,  # top-1000 상태만 유지
)
route, cost = solver.run()
print(f"Cost: {cost:.4f}")
solver.cleanup()
```

### 제약 조건

```python
from tsp_factor_graph_gpu import TSPFactorGraphSolverGPU as Solver

N_CITIES = 11
N = N_CITIES - 1  # depot 제외
D = np.random.rand(N_CITIES, N_CITIES)
np.fill_diagonal(D, 0)

cons = Solver.make_constraints(N)
Solver.time_window(cons, city=2, earliest=0, latest=3)
Solver.forbid(cons, city=5, time=0)
Solver.penalize(cons, city=3, time=7, penalty=-10.0)
Solver.precedence(cons, city_before=1, city_after=4)

solver = Solver(D, start_city=0, constraints=cons, verbose=True)
route, cost = solver.run()

# Exact constrained baseline과 비교 (exact 모드에서만)
exact_route, exact_cost = solver.solve_exact_constrained()
print(f"BM: {cost:.4f}  Exact: {exact_cost:.4f}")
solver.cleanup()
```

### 벤치마크 API

```python
from tsp_benchmarks import run_all_benchmarks, print_results
import numpy as np

D = np.random.rand(12, 12)
np.fill_diagonal(D, 0)
D = (D + D.T) / 2

results = run_all_benchmarks(D, start=0, seed=42)
opt = results['Held-Karp (Exact)'][1]
print_results(results, optimal_cost=opt)
```

### 히스토리 기록

```python
solver = Solver(D, start_city=0, iters=50, verbose=False)
route, cost = solver.run(record_history=True)

h = solver.history
print(f"Iterations: {len(h['cost'])}")
print(f"Cost 추이: {h['cost']}")

for i in range(1, len(h['cost'])):
    ratio = np.abs(h['gamma'][i]).max() / max(np.abs(h['gamma'][i-1]).max(), 1e-15)
    print(f"  iter {i+1}: γ̃ growth = {ratio:.2f}x")

solver.cleanup()
```

## 제약 행렬 형식

`constraints`는 `(N, N)` numpy 배열 (N = 도시 수, depot 제외).

| 값 | 의미 |
|----|------|
| `0.0` | 허용 (기본) |
| `-np.inf` | 금지 (hard constraint) |
| 음수 (예: `-10.0`) | 페널티 (soft constraint) |

- 행: 도시 인덱스 (depot 제외, 0-indexed)
- 열: 시간 슬롯 (0 ~ N-1)

## 메모리 참고

### Exact DP

| N | M=2^N | 피크 메모리 (float64) |
|---|-------|----------------------|
| 10 | 1,024 | ~5 MB |
| 15 | 32,768 | ~300 MB |
| 20 | 1,048,576 | ~2 GB |

- N ≤ 15: GPU 8GB 이하 OK
- N ≤ 20: GPU 24GB 필요
- N > 20: Beam search 사용

### Beam Search

| N | beam_width | 피크 메모리 |
|---|-----------|------------|
| 30 | 1,000 | ~50 MB |
| 50 | 1,000 | ~100 MB |
| 50 | 5,000 | ~500 MB |

Beam search는 N에 선형, B에 선형으로 메모리 증가. N=63까지 bitmask 지원.