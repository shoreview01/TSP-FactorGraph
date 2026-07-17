# Coupled CAP–Trellis Message Passing — 성북구 쓰레기 수거 실험

논문 **Sec. III (Coupled CAP-Trellis Message Passing, Algorithm 1)** 의 구현과
**Sec. IV** 의 실험 4종. 용량 제약 쓰레기 수거 문제에서, 용량 인지 클러스터링
(CAP)과 exact trellis 라우팅(Held-Karp min-sum)이 bridging 메시지 δ̃ 로 양방향
결합되어 {b_ij} 고정점까지 서로를 되먹인다. edge pruning·beam·heuristic
fallback 은 쓰지 않는다.

```
성북구 도로거리행렬 (84노드, data/seongbuk.csv)
   → [라운드 r]  CAP (용량 Q, δ̃ 주입)   ─ Pareto-frontier knapsack 을 φ̃ 메시지에 내장
              → 활성 exemplar {j : b_jj = 1},  후보 V^cand_k = {i : ρ̃_i > τ}
              → 클러스터별 trellis leave-one-out  → δ̃  (unmasked 1회 + masked pass)
              → {b_ij} 불변이면 종료 (Algorithm 1)
   → 클러스터마다 exact trellis TSP     ─ exemplar = 지역 depot, 차량 1대 최적 경로
```

## 실행 환경

모든 스크립트는 conda env **`tsp`** 의 python 으로 실행한다
(osmnx·networkx·torch·matplotlib·geopandas·pypdfium2):

```powershell
& "C:\Users\guild\.conda\envs\tsp\python.exe" <script.py> [옵션]
```

## 파일 구성

### 핵심 알고리즘 (논문 Sec. III)

| 파일 | 역할 |
|---|---|
| `coupled.py` | **Algorithm 1**: CAP↔trellis 결합 루프 — ρ̃(식 40)·ŝ(식 44)·δ̃(식 43+45), δ̃ damping |
| `cap.py` | CAP max-sum 메시지 패싱 (식 37–42, bridging δ̃ 주입·warm start 지원) + 디코딩 |
| `pareto_frontier.py` | φ̃ 메시지 내부 0/1 knapsack (Nemhauser–Ullmann frontier, Appendix A) |
| `trellis_tsp.py` | exact trellis TSP — Held-Karp min-sum, torch/GPU, pruning·fallback 없음 |
| `benchmarks.py` | 비교 라우팅: brute force / nearest-neighbor / GA (pop 100, 500세대, OX+swap) — 폐투어·open-path 겸용 |
| `run.py` | Algorithm 1 단독 드라이버 (`--compare` 로 BF/NN 대조) + 공용 유틸(load_matrix 등) |
| `ieee_style.py` | IEEE Transactions figure 규격 (모든 실험 공유) |

### 실험 (논문 Sec. IV) — 출력은 모두 `figs/`

**Setup figure** — digital twin 구성도

```powershell
python setup_fig.py        # -> figs/fig_setup.pdf
```
위성사진(physical, 고려대 캠퍼스 빨간 윤곽) 위 + 성북구 digital twin(도로망·
경계·84 service node·고려대 윤곽) 아래, 모서리 투영 점선과 중앙 "Digital twin"
화살표로 연결한 2단 구성.

---

**실험 1** (`exp1_clustering.py`) — **클러스터링 기법별 용량 feasibility 지도**

```powershell
python exp1_clustering.py --capacity 40 --weights random
# -> figs/exp1_proposed.pdf, exp1_ap.pdf, exp1_kmedoids.pdf
```
proposed(CAP) vs **weighted AP** vs **weighted K-medoids** (같은 K).  비교군은
demand-weighted 변형 — AP 는 s_w(i,k)=w_i·s(i,k) (가중 facility-location),
K-medoids 는 가중 PAM — 으로 weight 를 유사도에 반영하되 용량 제약만 없다.
투어 엣지는 osmnx 최단경로로 실제 도로를 따라 그려지고 feasible=초록 /
infeasible=빨강, depot 옆 미니 막대 = load (점선 = Q).

* 결과 (Q=40, random w∈1–9): proposed **13/13 feasible**,
  weighted AP 9/13 (max 57), weighted K-medoids 7/13 (max 70)
  → weight 인지만으로는 부족하고 용량의 메시지 내장이 결정적.

---

**실험 2** (`exp2_dynamic.py`) — **시변 수요(폭증) 하의 적응력 (클러스터링 축)**

```powershell
python exp2_dynamic.py --capacity 40 --steps 12
# -> figs/exp2_dynamic.pdf (+ exp2_dynamic.csv)
# 그림만 재조정: python exp2_dynamic.py --from-csv figs/exp2_dynamic.csv
```
노드별 weight 가 주기적으로 변하고 steps 5–7 에서 무작위 40% 노드가 2–4배
폭증.  매 스텝 세 기법으로 재클러스터링해 total / feasible-covered(초록) /
infeasible(빨강) weight 를 추적 (1×3 패널, 폭증 구간 음영).

* 결과: 누적 infeasible — proposed **0**, weighted AP 1,127 (peak 447),
  weighted K-medoids 1,684 (peak 427).  weighted 비교군은 평시는 감당하지만
  폭증 구간에서 무너짐; proposed 는 차량을 11→18대로 열며 전량 수거.

---

**실험 3** (`exp3_routing.py`) — **라우팅 기법 성능 비교 (uniform / random)**

```powershell
python exp3_routing.py --capacity 12 --weights uniform     # 단일 실행
python exp3_routing.py --capacity 40 --weights random      # Monte Carlo 100회
# -> figs/exp3_cumdist.pdf
```
CAP 클러스터링 고정, 클러스터 내부 + depot 간 TSP 의 solver 만
BF / **Trellis(제안)** / NN / GA 로 교체.  매 time step 전 클러스터 차량이
병렬로 한 노드씩 이동할 때의 누적거리 (마지막 구간 확대 inset).
weights=random 이면 가중치 실현을 seed 마다 재추첨하는 Monte Carlo
(평균 곡선 + min-max 구름).

* 결과 (uniform Q=12): BF = Trellis = 63,004 m (둘 다 exact, 0 mismatch),
  GA +0.2%, NN +15.2%.
* 결과 (random Q=40, 100 runs): BF = Proposed 75,421 ± 5,585 m (전 run 일치),
  GA +0.7%, NN +10.9%.

---

**실험 4** (`exp4_online.py`) — **랜덤 disruption 하의 online 라우팅 (Algorithm 2)**

```powershell
python exp4_online.py --capacity 12 --weights uniform      # Monte Carlo 100회 (~3h)
# -> figs/exp4_online.pdf
```
매 step 전체 노드의 5% 가 무작위 사고 지점이 되고(지속 3 step), 도로거리
500 m 내 엣지 비용이 3배.  Static(출발 전 1회 계획) vs Online(매 step
"현재 위치 → 남은 노드 → depot" open-path TSP 재계획) × solver 4종 = 8곡선,
Monte Carlo 평균 + min-max 구름 + 마지막 구간 inset(평균선).

* 결과 (100 seeds, online 절감): **Proposed 1.8%** (재계획 총 0.13 s/run),
  BF 1.2% (448 s/run), NN 3.6%, GA 0.1% — GA 는 재계획마다 무작위
  재초기화(cold start)라 이득이 사실상 없음.
* `--duration 1`(무기억 환경)이면 online ≈ static — 적응성엔 환경의 시간적
  지속성이 필요.

### 데이터 (`data/`)

| 파일 | 내용 |
|---|---|
| `seongbuk.csv` | 성북구 도로거리행렬 (84×84, symmetric, m) |
| `seongbuk_node_mapping.csv` | 노드 idx ↔ OSM node id·위경도 |
| `seongbuk_drive.graphml` | 성북구 OSM 도로망 캐시 (실험 1 지도·경로용) |
| `seongbuk_context.graphml` | 성북구 +1.2 km 버퍼 도로망 (setup figure 문맥용) |
| `seongbuk_boundary.geojson` / `ku_campus.geojson` | 성북구 경계 / 고려대 캠퍼스 폴리곤 |
| `sate_map.png` | 위성사진 (setup figure 상단 패널) |

### 문서

* `unified_note.pdf` — 알고리즘 유도 노트 (Sec. III 수식 대응).

## 주요 검증

* δ̃ (식 43+45) 계산: 무작위 인스턴스 20개에서 부분집합×순열 전수조사와
  정확히 일치 (오차 < 1e-9).
* trellis 라우팅: brute force 와 모든 클러스터·모든 MC run 에서 0 mismatch
  (둘 다 exact).
* Algorithm 1: δ̃ damping γ=0.5 로 {b_ij} 고정점 수렴 (uniform Q=12 기준
  6 라운드); γ=0 이면 undamped 원형.
