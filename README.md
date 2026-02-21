# TSP Factor-Graph Message-Passing Solver (GPU)

논문의 Factor-Graph BP 알고리즘을 PyTorch GPU 가속으로 구현한 솔버 + BM 메시지 시각화 도구.

## 파일 구성

| 파일 | 설명 |
|------|------|
| `tsp_factor_graph_gpu.py` | GPU 가속 솔버 (CUDA / MPS / CPU 자동 선택) |
| `visualize_bm.py` | BM 메시지 히트맵 시각화 (인터랙티브 / PNG / GIF) |
| `run_tsplib.py` | TSPLIB 파일 파서 + 솔버 실행 CLI |

## 설치

```bash
pip install torch numpy matplotlib
pip install Pillow   # GIF 저장 시 필요
```

## TSPLIB 파일 풀기

```bash
# 기본 실행
python run_tsplib.py ALL_tsp/gr17.tsp.gz

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

# GPU 지정
python run_tsplib.py ALL_tsp/gr17.tsp.gz --device mps
```

지원 TSPLIB 포맷: EUC_2D, CEIL_2D, ATT, GEO, MAN_2D, MAX_2D, EXPLICIT (FULL_MATRIX, UPPER_ROW, LOWER_ROW, UPPER_DIAG_ROW, LOWER_DIAG_ROW). `.tsp`와 `.tsp.gz` 모두 지원.

알려진 최적해가 있으면 자동으로 gap을 계산해 줍니다.

## 솔버 단독 실행

```bash
# 기본 실행 (N=11, 무제약 vs 제약 비교)
python tsp_factor_graph_gpu.py

# GPU 강제 지정
CUDA_VISIBLE_DEVICES=0 python tsp_factor_graph_gpu.py
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

### 도시 수 / 파라미터 변경

```bash
# N=15, damping 0.5, patience 30
python visualize_bm.py --n 15 --iters 100 --damping 0.5 --patience 30

# 시드 변경
python visualize_bm.py --n 10 --seed 123
```

### 제약 조건 추가

```bash
# 30% 도시에 랜덤 time window 제약 (기본)
python visualize_bm.py --n 10 --constrained

# 50% 도시에 제약
python visualize_bm.py --n 10 --constrained --constraint-ratio 0.5

# 제약 + 많은 iteration
python visualize_bm.py --n 12 --constrained --iters 100
```

### 색상 스케일

```bash
# per-iteration 스케일 (기본, 각 iteration마다 min/max 맞춤)
python visualize_bm.py --n 10

# global 스케일 (전체 iteration 걸쳐 고정, 절대 크기 비교용)
python visualize_bm.py --n 10 --global-scale
```

### 저장

```bash
# PNG 프레임 (bm_frames/ 폴더에 저장)
python visualize_bm.py --n 10 --iters 30 --save

# GIF 애니메이션
python visualize_bm.py --n 10 --iters 30 --gif

# 제약 + GIF
python visualize_bm.py --n 10 --constrained --iters 50 --gif
```

### GPU 지정

```bash
# CUDA
python visualize_bm.py --n 15 --device cuda

# MPS (Apple Silicon)
python visualize_bm.py --n 15 --device mps

# CPU 강제
python visualize_bm.py --n 15 --device cpu
```

## Python에서 직접 사용

### 무제약

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

### 제약 조건

```python
import numpy as np
from tsp_factor_graph_gpu import TSPFactorGraphSolverGPU as Solver

N_CITIES = 11
N = N_CITIES - 1  # depot 제외
D = np.random.rand(N_CITIES, N_CITIES)
np.fill_diagonal(D, 0)

# 제약 행렬 생성 (depot 제외 도시만, N×N)
cons = Solver.make_constraints(N)

# city 2: time 0~3에만 방문 가능
Solver.time_window(cons, city=2, earliest=0, latest=3)

# city 5: time 0 금지 (hard)
Solver.forbid(cons, city=5, time=0)

# city 3: time 7에 soft 페널티
Solver.penalize(cons, city=3, time=7, penalty=-10.0)

# city 1은 반드시 city 4보다 먼저
Solver.precedence(cons, city_before=1, city_after=4)

solver = Solver(D, start_city=0, constraints=cons, verbose=True)
route, cost = solver.run()
solver.cleanup()
```

### 히스토리 기록 + 커스텀 분석

```python
solver = Solver(D, start_city=0, iters=50, verbose=False)
route, cost = solver.run(record_history=True)

h = solver.history
print(f"Iterations: {len(h['cost'])}")
print(f"Cost 추이: {h['cost']}")
print(f"γ̃ 최종 range: [{h['gamma'][-1].min():.1f}, {h['gamma'][-1].max():.1f}]")

# 발산 비율 확인
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

| N | M=2^N | 피크 메모리 (float64) |
|---|-------|----------------------|
| 10 | 1,024 | ~5 MB |
| 15 | 32,768 | ~300 MB |
| 20 | 1,048,576 | ~2 GB |

- N ≤ 15: GPU 8GB 이하 OK
- N ≤ 20: GPU 24GB 필요
- N > 20: 메모리 초과 가능
- 80GB VRAM에서 N = 23까지 가능
