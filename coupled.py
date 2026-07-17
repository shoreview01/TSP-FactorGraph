"""
coupled.py
==========

논문 Sec. III "Coupled CAP-Trellis Message Passing" (**Algorithm 1**) 의 정확한
구현.  모듈-수식 대응:

  CAP module        (37)-(42) : cap.cap_affinity_propagation(..., bridge=Delta)
      delta~ 는 활성 exemplar 열에서 s(i, e_k) 와 같은 자리로 omega~/gamma~ 에
      들어간다 ((41),(42)).  비활성 열은 delta~ = 0 이라 표준형 (37)-(39) 그대로.
  rho~ 메시지       (40)      : rho~^(k)_i = eta~_{i,e_k} + phi~_{i,e_k}.
      cap 이 반환하는 R 은 gamma~ = eta~ + (s + delta~) 이므로
      rho~ = R - S_eff + A  (S_eff = S + Delta, 대각은 preference) 로 복원한다.
  trellis 유틸리티  (14),(44) : s_t(u,v) = max_{i,j} d(i,j) - d(u,v)  이고
      새 노드 v 진입 전이에만 rho~_v 를 더한 hat-s_t(u,v) = s_t(u,v) + rho~_v.
      depot self-loop (a_{t-1}=a_t=e_k, 값 0) 은 부분집합 표현에서 암묵적으로
      처리된다: 방문집합 m 이 전체 후보보다 작은 궤적이 곧 self-loop 패딩이다.
  soft output       (43),(45) : delta~_i = max_{x_N: i in m_N} zeta_i
                                         - max_{x_N: i not in m_N} zeta_i.
      (43) 의 zeta_i 는 자기 자신의 rho~_i 를 합에서 제외하므로, hat-s 로 얻은
      전방 메시지 psi-hat 에서 첫째 항은 rho~_i 를 빼서 읽는다:
        max_{i in m} zeta_i  = [max_{i in m} psi-hat]  - rho~_i
        max_{i notin m} zeta_i = masked forward pass (i 로의 모든 전이 -inf,
                                  Algorithm 1 line 9) 의 최대값.
  스케줄            Alg. 1    : 메시지 1회 초기화(line 1) -> 라운드마다
      [CAP 갱신(delta~ 고정) -> 활성 exemplar {j: b_jj=1} -> V^cand_k =
      {i : rho~_i > tau} -> 클러스터별 leave-one-out delta~ (lines 5-12)] ->
      {b_ij} 불변이면 종료 (lines 13-15).

논문이 명시하지 않아 해석을 고정한 두 지점 (해당 코드에 주석 표시):

  * 투어 폐쇄: (14) 의 G 는 depot 복귀 전이를 정의하지 않지만 (9) 가
    a_0 = a_{N_k} = e_k 를 요구하므로, 종단 상태 평가에 복귀 유사도
    s(a_N, e_k) 를 더해 (P1.2) 의 폐투어 비용과 일치시킨다 (복귀는 "새 노드
    추가"가 아니므로 rho~ 편향은 없다).
  * depot 자신(i = e_k)의 bridging 인자 K^(k)_{e_k}: [m_N]_{e_k} = 1 이 항상
    성립해 제약이 자동 만족되므로 delta~_{e_k} 는 정의상 상수이고 계산하지
    않는다(0 으로 둔다).

최종 라우팅: 수렴한 {b_ij} 의 파티션 위에서 (P1.2) 를 클러스터별 exact
trellis(Held-Karp min-sum)로 푼다.  멤버 집합이 고정되면 rho~ 편향과 상수
max d 오프셋은 모든 방문 순서에 동일하므로 max sum(hat-s) 와 min sum(d) 의
argmax 는 같다 -- 거리 기반 trellis_tsp 가 곧 유사도 기반 최적 디코드다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cap import cap_affinity_propagation, similarity_from_distance
from trellis_tsp import trellis_tsp

_NEG = -np.inf

# (mask 배열, popcount) 캐시 -- 후보 수 m 별로 한 번만 만든다.
_MASK_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _masks_pc(m: int):
    if m not in _MASK_CACHE:
        masks = np.arange(1 << m, dtype=np.int64)
        pc = np.zeros(1 << m, dtype=np.int64)
        tmp = masks.copy()
        for _ in range(m):
            pc += tmp & 1
            tmp >>= 1
        _MASK_CACHE[m] = (masks, pc)
    return _MASK_CACHE[m]


# ---------------------------------------------------------------------------
# Trellis module: forward recursion (27) over utilities hat-s  (max-sum)
# ---------------------------------------------------------------------------
def _forward_table(S_hat: np.ndarray) -> np.ndarray:
    """전방 메시지 psi (식 (27)) 를 하이퍼큐브 trellis 전체에서 계산한다.

    로컬 인덱스 0 = depot e_k, 1..m = 후보 노드.  반환 g[mask, j] = depot 에서
    출발해 후보 부분집합 `mask` 를 정확히 방문하고 로컬 후보 j 에서 끝나는
    궤적의 최대 누적 유틸리티 (max-sum).  상태 x_t = (m_t, a_t) 의 (14) 전이
    조건 -- [m_{t-1}]_{a_{t-1}} = 1, m 에 새 노드 하나 추가(dH = 1) -- 이
    그대로 반영된다: 목적 노드 k 는 src mask 에 없어야 하고, 출발 노드는 src
    mask 안에 있어야 한다.
    """
    m = S_hat.shape[0] - 1
    size = 1 << m
    g = np.full((size, m), _NEG)
    if m == 0:
        return g
    ar = np.arange(m)
    g[1 << ar, ar] = S_hat[0, 1:]                    # depot -> 첫 노드 (bias 포함)

    masks, pc = _masks_pc(m)
    for t in range(1, m):                            # |m_t| = t -> t+1 층 확장
        layer = masks[pc == t]
        for k in range(m):
            bit = 1 << k
            src = layer[(layer & bit) == 0]          # k 를 아직 안 가진 집합들
            if src.size == 0:
                continue
            g_src = g[src]                                       # (M, m)
            cand = g_src + S_hat[1:, k + 1][None, :]             # 전이 j -> k
            in_mask = ((src[:, None] >> ar) & 1).astype(bool)    # 출발점은 방문집합 안
            cand[~in_mask] = _NEG
            best = cand.max(axis=1)
            tgt = src + bit
            g[tgt, k] = np.maximum(g[tgt, k], best)
    return g


# ---------------------------------------------------------------------------
# Bridging module: leave-one-out soft outputs delta~  (Sec. III-C3/C4, III-D)
# ---------------------------------------------------------------------------
def leave_one_out_soft(D: np.ndarray, rho: np.ndarray, cand: list[int],
                       exemplar: int, d_max: float) -> np.ndarray:
    """클러스터 k 의 모든 후보 i 에 대한 delta~^(k)_i (식 (45)).

    Algorithm 1 lines 7-11 그대로: unmasked 전방 패스 1회로 모든 i 의 첫째
    항을 동시에 읽고 (Sec. III-D), i 마다 hat-s_t(., i) = -inf 로 마스킹한
    패스(line 9)로 둘째 항을 얻는다.

    Parameters
    ----------
    D        : 전체 거리행렬.
    rho      : 열 e_k 의 rho~ 벡터 (전체 노드 길이) -- 식 (40).
    cand     : V^cand_k (exemplar 제외).
    exemplar : e_k.
    d_max    : max_{i,j in V} d(i,j) -- 유사도 변환 s_t = d_max - d 의 상수.
    """
    m = len(cand)
    nodes = [exemplar] + list(cand)
    S_route = d_max - D[np.ix_(nodes, nodes)]        # s_t(u,v) = max d - d(u,v)
    rho_c = np.asarray([rho[i] for i in cand], dtype=float)

    S_hat = S_route.copy()                           # (44): 새 노드 진입에만 rho~ 추가
    S_hat[:, 1:] = S_hat[:, 1:] + rho_c[None, :]

    g = _forward_table(S_hat)
    # 폐투어: 종단 상태에 복귀 유사도 s(a_N, e_k) 를 더한다 ((9); 복귀는 새
    # 노드 추가가 아니므로 rho~ 편향 없음).
    close = S_route[1:, 0]
    masks, _ = _masks_pc(m)
    term_best = (g + close[None, :]).max(axis=1)     # 방문집합별 최적 폐투어 유틸리티

    delta = np.empty(m)
    for c in range(m):
        bit = 1 << c
        # 첫째 항: max_{x_N: i in m_N} zeta_i.  psi-hat 은 rho~_i 를 포함하므로
        # (43) 의 "i' != i" 합에 맞춰 rho~_i 를 빼서 읽는다.
        inc = float(term_best[(masks & bit) != 0].max()) - rho_c[c]
        # 둘째 항: i 로의 모든 전이를 -inf 로 막은 masked pass (Alg. 1 line 9).
        # (trellis 상태가 방문집합을 명시하므로 unmasked 표의 i-미포함 mask 최대와
        #  동일한 값이지만, 논문 절차 그대로 재실행한다.)
        S_msk = S_hat.copy()
        S_msk[:, c + 1] = _NEG
        g_msk = _forward_table(S_msk)
        exc = float((g_msk + close[None, :]).max())
        exc = max(0.0, exc)                          # 빈 투어(depot self-loop 만) = 0
        delta[c] = inc - exc                         # (45)
    return delta


# ---------------------------------------------------------------------------
# Algorithm 1: Coupled CAP-Trellis Message Passing
# ---------------------------------------------------------------------------
@dataclass
class CoupledCluster:
    exemplar: int            # e_k = 지역 depot (v_{1,k} = e_k)
    nodes: list[int]         # 폐투어 [e_k, ..., e_k]
    length: float            # (P1.2) 의 지역 투어 비용
    load: float              # sum_i w_i b_{i,e_k} (exemplar 포함)


@dataclass
class CoupledSolution:
    clusters: list[CoupledCluster]
    labels: np.ndarray       # labels[i] = exemplar node id  ({b_ij} 의 디코드)
    n_rounds: int
    converged: bool          # {b_ij} 고정점 도달 (Alg. 1 lines 13-15)
    cap_converged: bool      # 마지막 라운드 CAP 내부 수렴 여부

    @property
    def exemplars(self) -> list[int]:
        return [c.exemplar for c in self.clusters]

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def total_length(self) -> float:
        # (P1.2): 지역 투어 비용의 합.  (지역 간 상위 투어는 논문 모델에 없다.)
        return sum(c.length for c in self.clusters)

    def is_feasible(self, Q: float) -> bool:
        return all(c.load <= Q + 1e-6 for c in self.clusters)


def solve_coupled(D: np.ndarray, weights: np.ndarray, Q: float,
                  similarity: str = "neg", preference="median",
                  tau: float = 0.0, max_rounds: int = 10, max_cand: int = 16,
                  damping: float = 0.7, bridge_damping: float = 0.5,
                  cap_max_iter: int = 300,
                  verbose: bool = False,
                  label_history: list | None = None) -> CoupledSolution:
    """논문 Algorithm 1 (Coupled CAP-Trellis Message Passing).

    Parameters
    ----------
    tau        : 후보 문턱값 -- V^cand_k = {i : rho~^(k)_i > tau} (Alg. 1 line 6).
    max_rounds : 최대 라운드 수 R.
    max_cand   : 클러스터당 후보 수 상한 (2^m trellis 표의 메모리/시간 가드).
                 초과 시 rho~ 상위 max_cand 개만 남기고, 잘렸음을 출력한다.
    damping    : CAP 메시지 damping (AP 계열 [28] 의 표준 안정화; Algorithm 1 의
                 메시지 갱신 자체에는 영향 없는 수렴 장치다).
    bridge_damping : bridging 메시지 damping gamma --
                 delta~ <- gamma * delta~_old + (1 - gamma) * delta~_new.
                 loopy max-sum 의 표준 진동 억제 장치로, CAP 쪽 메시지 damping
                 과 같은 역할을 라운드 단위 delta~ 갱신에 적용한 것이다.
                 gamma = 0 이면 undamped Algorithm 1 그대로다.
    나머지는 cap.cap_affinity_propagation 과 동일.
    """
    n = len(D)
    D = np.asarray(D, dtype=float)
    weights = np.asarray(weights, dtype=float)
    S = similarity_from_distance(D, kind=similarity)

    # preference(= s(j,j))를 float 로 고정해 라운드 간 동일하게 쓴다.
    off = S[~np.eye(n, dtype=bool)]
    if preference == "median":
        pref = float(np.median(off))
    elif preference == "min":
        pref = float(np.min(off))
    else:
        pref = float(preference)
    S_pref = S.copy()
    np.fill_diagonal(S_pref, pref)

    d_max = float(D.max())                    # s_t(u,v) = max_{i,j} d(i,j) - d(u,v)

    Delta = np.zeros((n, n))                  # bridging messages delta~ (line 1)
    A0 = R0 = None                            # CAP 메시지도 line 1 에서 1회 초기화
    prev_labels = None
    labels = None
    converged = False
    cap = None
    r = 0

    for r in range(1, max_rounds + 1):        # line 2
        # line 3: delta~ 고정한 채 CAP 메시지 갱신 ((37)-(42); delta~ 는 bridge 로 주입)
        cap = cap_affinity_propagation(S, weights, Q, preference=pref,
                                       damping=damping, max_iter=cap_max_iter,
                                       decode="raw", bridge=Delta, A0=A0, R0=R0)
        A0, R0 = cap.A, cap.R                 # 메시지는 라운드를 넘어 이어진다
        labels = cap.labels
        if label_history is not None:         # 라운드별 {b_ij} 디코드 기록
            label_history.append(np.asarray(labels, dtype=int).copy())

        # line 4: 활성 exemplar {j : b_jj = 1}
        exemplars = [int(k) for k in cap.exemplars]

        # lines 5-12: 활성 클러스터별 leave-one-out delta~
        S_eff = S_pref + Delta                # gamma~ = eta~ + (s + delta~) 의 s + delta~
        Delta_new = np.zeros((n, n))
        for e in exemplars:
            # (40): rho~ = eta~ + phi~,  eta~ = gamma~ - (s + delta~) = R - S_eff
            rho = cap.R[:, e] - S_eff[:, e] + cap.A[:, e]
            cand = [i for i in range(n) if i != e and rho[i] > tau]   # line 6
            if len(cand) > max_cand:          # 2^m 표 가드 -- 자르면 반드시 알린다
                order = np.argsort(-rho[cand])
                dropped = len(cand) - max_cand
                cand = [cand[int(o)] for o in order[:max_cand]]
                print(f"  [coupled r{r}] exemplar {e}: candidate set truncated "
                      f"to top {max_cand} by rho~ ({dropped} dropped)")
            if cand:                          # lines 7-11
                Delta_new[cand, e] = leave_one_out_soft(D, rho, cand, e, d_max)

        if verbose:
            changed = (n if prev_labels is None
                       else int(np.sum(labels != prev_labels)))
            print(f"  [coupled r{r}] {len(exemplars)} exemplars  "
                  f"labels changed={changed}  cap_iters={cap.n_iter}")

        # lines 13-15: {b_ij} 가 이전 라운드와 같으면 종료
        if prev_labels is not None and np.array_equal(labels, prev_labels):
            converged = True
            break
        prev_labels = labels
        # delta~ 갱신 (damping 으로 라운드 간 진동 억제; gamma=0 -> undamped)
        Delta = bridge_damping * Delta + (1.0 - bridge_damping) * Delta_new

    # line 17: {b_ij} 와 클러스터별 궤적 {a^(k)_t} 반환.
    # 파티션이 고정되면 (P1.2) 는 클러스터별 exact trellis 로 정확히 풀린다.
    clusters: list[CoupledCluster] = []
    for e in sorted(set(labels.tolist())):
        members = [i for i in range(n) if labels[i] == e]
        load = float(weights[labels == e].sum())
        if len(members) == 1:
            clusters.append(CoupledCluster(exemplar=int(e), nodes=[int(e), int(e)],
                                           length=0.0, load=load))
            continue
        Dsub = D[np.ix_(members, members)]
        local, length = trellis_tsp(Dsub, start=members.index(e))
        clusters.append(CoupledCluster(
            exemplar=int(e),
            nodes=[members[i] for i in local],
            length=float(length),
            load=load,
        ))

    return CoupledSolution(clusters=clusters, labels=np.asarray(labels, dtype=int),
                           n_rounds=r, converged=converged,
                           cap_converged=bool(cap.converged))
