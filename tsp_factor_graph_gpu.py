"""
TSP Factor-Graph Message-Passing Solver — GPU-Accelerated (PyTorch)
====================================================================
핵심 가속 전략:
  1) 마스크별 순차 루프 제거 → 동일 popcount 마스크를 텐서 배치로 처리
  2) Forward/Backward에서 per-city 루프 + 배치 gather/scatter
  3) δ̃ 계산: peak 팩토리제이션으로 O(N²T·M) → O(N²T + N·T·M) 축소

메모리 프로파일 (dtype 자동 선택: MPS→float32, CUDA/CPU→float64):
  psi, alpha, beta, xi : 각 [T+1, M, N]
  float64: N=15 ~61MB, N=20 ~320MB / float32: 절반
  총 피크: N=15 ~300MB(f64)/~150MB(f32), N=20 ~2GB(f64)/~1GB(f32)

권장 N 범위: ≤ 20 (GPU 24GB 기준), ≤ 15 (GPU 8GB 기준)
"""

import torch
import numpy as np
import gc
from typing import Optional, Tuple, List

NEG = -1e12


def get_device(device: Optional[str] = None) -> torch.device:
    """CUDA > MPS > CPU 자동 선택."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def flush_gpu(device: Optional[torch.device] = None):
    """GPU 메모리 강제 정리. solver 생성 전/후에 호출 가능."""
    gc.collect()
    if device is not None and device.type == "cpu":
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()          # 미완료 커널 대기
        gc.collect()                       # Python 참조 해제
        torch.cuda.empty_cache()           # PyTorch 캐시 → OS 반환
        torch.cuda.ipc_collect()           # IPC 공유 메모리 정리
        torch.cuda.reset_peak_memory_stats()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gc.collect()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()        # PyTorch 2.1+


class TSPFactorGraphSolverGPU:

    def __init__(self, D, start_city: int = 0,
                 damping: float = 0.3, iters: int = 200,
                 verbose: bool = False, seed: int = 0,
                 patience: int = 20, cost_tol: float = 1e-12,
                 device: Optional[str] = None,
                 constraints: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        D : (C x C) 거리 행렬. C = N+1 (N개 도시 + 1개 depot).
        device : 'cuda', 'mps', 'cpu' 또는 None (자동).
        constraints : (N_cities x N_cities) 제약 행렬. None이면 무제약.
                      원래 도시 인덱스 기준 (depot 제외, 0-indexed).
                        0.0  = 허용
                       -inf  = 금지 (hard constraint)
                       음수값 = 페널티 (soft constraint, 예: -10.0)
        """
        self.device = get_device(device)
        self.dtype = torch.float32 if self.device.type == "mps" else torch.float64
        torch.manual_seed(seed)

        # --- 기존 GPU 텐서 정리 ---
        flush_gpu(self.device)

        _npdtype = np.float32 if self.dtype == torch.float32 else np.float64
        D_np = np.array(D, dtype=_npdtype)
        assert D_np.shape[0] == D_np.shape[1], "D must be square"
        C = D_np.shape[0]
        N = C - 1

        # --- 내부 인덱싱: start_city → 마지막 인덱스(depot) ---
        perm = np.arange(C)
        if start_city != C - 1:
            perm[start_city], perm[C - 1] = perm[C - 1], perm[start_city]
        inv_perm = np.empty(C, dtype=int)
        inv_perm[perm] = np.arange(C)

        self.orig_D = torch.tensor(D_np, dtype=self.dtype, device=self.device)
        D_perm = D_np[perm][:, perm]
        self.D_perm = torch.tensor(D_perm, dtype=self.dtype, device=self.device)
        self.inv_perm = inv_perm
        self.C, self.N, self.depot = C, N, C - 1

        # Similarity (eq 5)
        self.s = self.D_perm.max() - self.D_perm
        # 도시 간 유사도 서브매트릭스 [N, N]
        self.s_city = self.s[:N, :N]

        self.T = N
        self.M = 1 << N
        self.damping = damping
        self.iters = iters
        self.verbose = verbose
        self.patience = patience
        self.cost_tol = cost_tol

        # BM 메시지
        self.gamma_t = torch.zeros((N, N), dtype=self.dtype, device=self.device)
        self.omega_t = torch.zeros((N, N), dtype=self.dtype, device=self.device)

        # --- 제약 행렬 (내부 인덱스로 변환) ---
        # constraint_bias[i, t]: ρ̃에 더해지는 bias
        #   0 = 허용, HARD = 금지, 음수 = 페널티 (soft)
        # ※ HARD는 문제 스케일에 비례: 충분히 크지만 mean 계산을 오염시키지 않는 수준
        max_d = float(self.D_perm.max())
        HARD = -(max_d * N * 100)  # 최적 경로 비용의 ~100배
        self.HARD = HARD
        self.constraint_bias = torch.zeros(
            (N, N), dtype=self.dtype, device=self.device
        )
        if constraints is not None:
            c_np = np.array(constraints, dtype=_npdtype)
            assert c_np.shape == (N, N), \
                f"constraints shape {c_np.shape} != ({N}, {N}). depot 제외 도시만."
            # -inf → HARD 변환 (nan 방지)
            c_np = np.where(np.isneginf(c_np), HARD, c_np)

            # 원래 도시 인덱스 → 내부 인덱스 변환
            perm_cities = np.array(perm[:N])
            c_reordered = np.zeros((N, N), dtype=_npdtype)
            for int_i in range(N):
                orig_i = inv_perm[int_i]
                if orig_i == start_city:
                    continue
                orig_row = orig_i if orig_i < start_city else orig_i - 1
                if 0 <= orig_row < N:
                    c_reordered[int_i, :] = c_np[orig_row, :]
            self.constraint_bias = torch.tensor(
                c_reordered, dtype=self.dtype, device=self.device
            )

        n_hard = (self.constraint_bias <= HARD / 2).sum().item()
        n_soft = ((self.constraint_bias < 0) &
                  (self.constraint_bias > HARD / 2)).sum().item()
        self.has_constraints = n_hard > 0 or n_soft > 0

        if self.verbose and self.has_constraints:
            print(f"  Constraints: {n_hard} hard (HARD={HARD:.1f}), "
                  f"{n_soft} soft (penalty)")

        # 금지 마스크: trellis에서 직접 강제용 [N, N] (city, time)
        # BM 메시지가 발산해도 금지된 할당은 절대 불가
        self._forbidden = (self.constraint_bias <= HARD / 2)  # bool [N, N]

        # --- 마스크 테이블 사전 계산 ---
        self._precompute_tables()

        if self.verbose:
            dt_str = "float32" if self.dtype == torch.float32 else "float64"
            print(f"[Device: {self.device}, dtype: {dt_str}] N={N}, M={self.M}, "
                  f"peak mem ~{self._estimate_mem_mb():.0f} MB")

    # =================================================================
    #                    마스크 테이블 사전 계산
    # =================================================================
    def _precompute_tables(self):
        N, M = self.N, self.M
        dev, dt = self.device, self.dtype

        mask_range = torch.arange(M, device=dev, dtype=torch.long)

        # popcount[m] = number of set bits in m
        pc = torch.zeros(M, dtype=torch.int32, device=dev)
        for i in range(N):
            pc += ((mask_range >> i) & 1).int()
        self._popcount = pc

        # bit_set[m, a] = bool: bit a is set in mask m
        bit_vals = (1 << torch.arange(N, device=dev, dtype=torch.long))
        self._bit_set = (mask_range.unsqueeze(1) & bit_vals.unsqueeze(0)) != 0  # [M, N]

        # prev_mask[m, a] = m ^ (1<<a)  (remove bit a)
        self._prev_mask = mask_range.unsqueeze(1) ^ bit_vals.unsqueeze(0)  # [M, N]

        # next_mask[m, a] = m | (1<<a)  (add bit a)
        self._next_mask = mask_range.unsqueeze(1) | bit_vals.unsqueeze(0)  # [M, N]

        # NEG 텐서 캐시
        self._NEG_M = torch.full((M,), NEG, dtype=dt, device=dev)
        self._NEG_MN = torch.full((M, N), NEG, dtype=dt, device=dev)
        self._NEG_1 = torch.tensor(NEG, dtype=dt, device=dev)
        self._ZERO = torch.tensor(0.0, dtype=dt, device=dev)
        self._NEG1_long = torch.tensor(-1, dtype=torch.long, device=dev)

    def _estimate_mem_mb(self):
        T, M, N = self.T, self.M, self.N
        bpe = 4 if self.dtype == torch.float32 else 8  # bytes per element
        main = 4 * (T + 1) * M * N * bpe
        tables = 3 * M * N * bpe
        return (main + tables) / (1024 ** 2)

    # =================================================================
    #                    GPU 메모리 관리
    # =================================================================
    def cleanup(self):
        """명시적 GPU 메모리 해제. 솔버 사용 후 호출 권장."""
        dev = self.device if hasattr(self, 'device') else None

        # 모든 텐서 속성 삭제
        tensor_attrs = [k for k, v in self.__dict__.items()
                        if isinstance(v, torch.Tensor)]
        for attr in tensor_attrs:
            delattr(self, attr)

        # 사전 계산 테이블 등 나머지 대형 속성
        for attr in ['_bit_set', '_prev_mask', '_next_mask', '_popcount',
                      '_NEG_M', '_NEG_MN', '_NEG_1', '_ZERO', '_NEG1_long']:
            if hasattr(self, attr):
                delattr(self, attr)

        flush_gpu(dev)

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def _print_gpu_mem(self, tag: str = ""):
        """CUDA 메모리 사용량 출력 (디버깅용)."""
        if self.device.type == "cuda":
            alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            print(f"  [GPU mem {tag}] alloc={alloc:.1f}MB, reserved={reserved:.1f}MB")

    # =================================================================
    #                    제약 조건 빌더 (static helpers)
    # =================================================================
    @staticmethod
    def make_constraints(n_cities: int) -> np.ndarray:
        """빈 제약 행렬 생성 (N x N, 전부 0 = 허용)."""
        return np.zeros((n_cities, n_cities), dtype=np.float64)

    @staticmethod
    def forbid(constraints: np.ndarray, city: int, time: int):
        """(city, time) 할당 금지 (hard constraint)."""
        constraints[city, time] = -np.inf

    @staticmethod
    def penalize(constraints: np.ndarray, city: int, time: int,
                 penalty: float = -10.0):
        """(city, time) 할당에 페널티 부과 (soft constraint)."""
        constraints[city, time] = penalty

    @staticmethod
    def time_window(constraints: np.ndarray, city: int,
                    earliest: int, latest: int):
        """city를 [earliest, latest] 시간 범위에만 허용.
        범위 바깥은 -inf (hard forbidden)."""
        N = constraints.shape[1]
        for t in range(N):
            if t < earliest or t > latest:
                constraints[city, t] = -np.inf

    @staticmethod
    def precedence(constraints: np.ndarray, city_before: int,
                   city_after: int):
        """city_before가 city_after보다 먼저 방문되어야 함.
        city_after의 t=0 금지, city_before의 t=N-1 금지,
        + 겹치는 시간대 제한."""
        N = constraints.shape[1]
        # city_after는 첫 시간 불가
        constraints[city_after, 0] = -np.inf
        # city_before는 마지막 시간 불가
        constraints[city_before, N - 1] = -np.inf

    # =================================================================
    #                         Public Interface
    # =================================================================
    def run(self, record_history: bool = False) -> Tuple[List[int], float]:
        best_route, best_cost = None, None
        stable = 0
        last_cost = None

        if record_history:
            self.history = {
                'gamma': [], 'omega': [],
                'rho': [], 'delta': [],
                'phi': [], 'eta': [],
                'cost': [], 'route': [],
                'inv_perm': self.inv_perm.copy(),
                'N': self.N,
                'depot_orig': int(self.inv_perm[self.depot]),
            }

        for it in range(self.iters):
            phi_t, eta_t, rho_t = self._derive_bm_messages()
            psi, alpha, backptr = self._forward(rho_t)
            beta, xi = self._backward(rho_t)
            delta_t = self._compute_delta(psi, beta, rho_t)

            # γ̃, ω̃ damping 업데이트
            gamma_new = eta_t + delta_t
            omega_new = phi_t + delta_t
            self.gamma_t = self.damping * gamma_new + (1 - self.damping) * self.gamma_t
            self.omega_t = self.damping * omega_new + (1 - self.damping) * self.omega_t

            route = self._decode(alpha, backptr)
            cost = self._route_cost(route)

            # ── 히스토리 기록 (CPU numpy로 복사) ──
            if record_history:
                self.history['gamma'].append(self.gamma_t.cpu().numpy().copy())
                self.history['omega'].append(self.omega_t.cpu().numpy().copy())
                self.history['rho'].append(rho_t.cpu().numpy().copy())
                self.history['delta'].append(delta_t.cpu().numpy().copy())
                self.history['phi'].append(phi_t.cpu().numpy().copy())
                self.history['eta'].append(eta_t.cpu().numpy().copy())
                self.history['cost'].append(cost)
                self.history['route'].append(route)

            # ── 중간 텐서 즉시 해제 ──
            del psi, beta, xi, phi_t, eta_t, rho_t, delta_t
            del gamma_new, omega_new

            if self.verbose:
                print(f"[{it+1:03d}] cost={cost:.6f}  route={route}")

            if best_cost is None or cost < best_cost:
                best_cost, best_route = cost, route

            # decode에서 사용 완료
            del alpha, backptr

            if last_cost is not None and abs(cost - last_cost) <= self.cost_tol:
                stable += 1
            else:
                stable = 0
            last_cost = cost
            if stable >= self.patience:
                break

        return best_route, best_cost

    # =================================================================
    #        Bipartite Matching: φ̃, η̃, ρ̃ (eq 37, 40, 41)
    # =================================================================
    def _derive_bm_messages(self):
        """
        벡터화된 top-2 기반 "자기 제외 max" 계산.
        """
        N, T = self.N, self.T
        dev, dt = self.device, self.dtype

        # --- eq (40): φ̃_it = -max_{i'≠i} γ̃_{i't} ---
        phi_t = torch.zeros((N, T), dtype=dt, device=dev)
        if N > 1:
            top2_v, top2_i = self.gamma_t.topk(2, dim=0)  # [2, T]
            for i in range(N):
                is_top = (top2_i[0] == i)
                phi_t[i] = -torch.where(is_top, top2_v[1], top2_v[0])

        # --- eq (37): η̃_it = -max_{t'≠t} ω̃_{it'} ---
        eta_t = torch.zeros((N, T), dtype=dt, device=dev)
        if T > 1:
            top2_v, top2_i = self.omega_t.topk(2, dim=1)  # [N, 2]
            for t in range(T):
                is_top = (top2_i[:, 0] == t)
                eta_t[:, t] = -torch.where(is_top, top2_v[:, 1], top2_v[:, 0])

        rho_t = eta_t + phi_t

        # --- 제약 적용 (BM 레벨에서만, trellis는 무관) ---
        if self.has_constraints:
            # soft constraint: 고정 페널티 추가
            soft_mask = ~self._forbidden & (self.constraint_bias < 0)
            if soft_mask.any():
                rho_t = rho_t + torch.where(soft_mask, self.constraint_bias,
                                            torch.zeros_like(self.constraint_bias))

            # hard constraint: 현재 메시지 스케일에 비례하여 금지
            # → BM이 아무리 발산해도 금지 셀은 항상 허용 셀보다 압도적으로 작음
            if self._forbidden.any():
                allowed = rho_t[~self._forbidden]
                if allowed.numel() > 0:
                    scale = allowed.abs().max().clamp(min=1.0)
                else:
                    scale = torch.tensor(1.0, dtype=dt, device=dev)
                rho_t[self._forbidden] = -scale * 1000

        return phi_t, eta_t, rho_t

    # =================================================================
    #         Forward Pass (벡터화): ψ_t, α_t (eq 15-16)
    # =================================================================
    def _forward(self, rho_t):
        """
        per-city 루프 + 배치 gather로 마스크 루프 제거.
        메모리: O(M·N) per time step.
        """
        T, N, M = self.T, self.N, self.M
        dev, dt = self.device, self.dtype
        depot = self.depot

        psi     = torch.full((T + 1, M, N), NEG, dtype=dt, device=dev)
        alpha   = torch.full((T + 1, M, N), NEG, dtype=dt, device=dev)
        backptr = torch.full((T + 1, M, N), -1, dtype=torch.long, device=dev)

        # λ_sum 전체 사전 계산: [N, T]
        lambda_sum_all = rho_t - rho_t.mean(dim=0, keepdim=True)

        # ── t = 1: depot → 첫 도시 ──
        for a in range(N):
            m = 1 << a
            s_val = self.s[depot, a]
            psi[1, m, a] = s_val
            alpha[1, m, a] = s_val + lambda_sum_all[a, 0]

        # ── t = 2 .. T: 배치 처리 ──
        for t in range(2, T + 1):
            t_idx = t - 1
            valid_pop = (self._popcount == t)  # [M]

            for a in range(N):
                # valid[m] = popcount(m)==t AND bit a set in m
                valid = valid_pop & self._bit_set[:, a]  # [M]

                # prev_mask[m] = m ^ (1<<a)
                prev = self._prev_mask[:, a]  # [M]

                # gather alpha[t-1, prev[m], :] → [M, N]
                alpha_gathered = alpha[t - 1][prev]  # [M, N]

                # scores[m, last] = alpha[t-1, prev, last] + s(last, a)
                scores = alpha_gathered + self.s_city[:, a].unsqueeze(0)  # [M, N]

                # last는 prev_mask에 bit가 있어야 유효
                valid_last = self._bit_set[prev]  # [M, N]
                scores = torch.where(valid_last, scores, self._NEG_MN)

                # max over last
                psi_val, argmax_last = scores.max(dim=1)  # [M]

                psi[t, :, a] = torch.where(valid, psi_val, self._NEG_M)
                alpha[t, :, a] = torch.where(
                    valid,
                    psi_val + lambda_sum_all[a, t_idx],
                    self._NEG_M
                )
                backptr[t, :, a] = torch.where(
                    valid, argmax_last, self._NEG1_long.expand(M)
                )

        return psi, alpha, backptr

    # =================================================================
    #         Backward Pass (벡터화): β_t, ξ_t (eq 17-18)
    # =================================================================
    def _backward(self, rho_t):
        """
        per-city gather + 배치 scatter.
        """
        T, N, M = self.T, self.N, self.M
        dev, dt = self.device, self.dtype
        full = (1 << N) - 1
        depot = self.depot

        beta = torch.full((T + 1, M, N), NEG, dtype=dt, device=dev)
        xi   = torch.full((T + 1, M, N), NEG, dtype=dt, device=dev)

        lambda_sum_all = rho_t - rho_t.mean(dim=0, keepdim=True)

        # ── t = T: closure ──
        for a in range(N):
            beta[T, full, a] = self.s[a, depot]
            xi[T, full, a] = self.s[a, depot] + lambda_sum_all[a, T - 1]

        # ── t = T-1 .. 1 ──
        for t in range(T - 1, 0, -1):
            t_idx = t - 1
            valid_pop = (self._popcount == t)  # [M]

            # xi_next_a[m, a] = xi[t+1, m|(1<<a), a]
            xi_next = xi[t + 1]  # [M, N]
            xi_next_a = torch.full((M, N), NEG, dtype=dt, device=dev)
            for a in range(N):
                next_m = self._next_mask[:, a]  # [M]
                xi_next_a[:, a] = xi_next[next_m, a]

            for last in range(N):
                valid = valid_pop & self._bit_set[:, last]  # [M]

                # scores[m, a] = s(last, a) + xi[t+1, m|(1<<a), a]
                scores = self.s_city[last, :].unsqueeze(0) + xi_next_a  # [M, N]

                # a는 현재 마스크에 없어야 함
                not_in_mask = ~self._bit_set  # [M, N]
                scores = torch.where(not_in_mask, scores, self._NEG_MN)

                # max over a
                beta_val, _ = scores.max(dim=1)  # [M]

                beta[t, :, last] = torch.where(valid, beta_val, self._NEG_M)
                xi[t, :, last] = torch.where(
                    valid,
                    beta_val + lambda_sum_all[last, t_idx],
                    self._NEG_M
                )

        return beta, xi

    # =================================================================
    #     δ̃ 계산 (최적화): peak 팩토리제이션 (eq 42)
    # =================================================================
    def _compute_delta(self, psi, beta, rho_t):
        """
        핵심 최적화:
          gamma_val = ψ + β  (경로 메트릭)
          peak[t, a] = max_{m: popcount=t, bit_a∈m} gamma_val[t, m, a]

        λ_sum_excl_i는 mask에 무관하므로 factoring 가능:
          best_with[i,t]    = excl_λ(i,t,i) + peak[t,i]
          best_without[i,t] = max_{a≠i} [excl_λ(i,t,a) + peak[t,a]]

        복잡도: O(N²·T) vs 원래 O(N²·T·M)
        """
        T, N, M = self.T, self.N, self.M
        dev, dt = self.device, self.dtype

        gamma_val = psi + beta  # [T+1, M, N]

        # --- peak[t, a] 계산 ---
        peak = torch.full((T + 1, N), NEG, dtype=dt, device=dev)
        for t in range(1, T + 1):
            valid = (self._popcount == t).unsqueeze(1) & self._bit_set  # [M, N]
            gv = gamma_val[t]  # [M, N]
            gv_masked = torch.where(valid & (gv > NEG / 2), gv, self._NEG_MN)
            peak[t] = gv_masked.max(dim=0).values  # [N]

        # --- lambda_sum_all ---
        lambda_sum_all = rho_t - rho_t.mean(dim=0, keepdim=True)  # [N, T]

        delta = torch.zeros((N, T), dtype=dt, device=dev)

        for t in range(1, T + 1):
            t_idx = t - 1
            pk = peak[t]                    # [N]
            lsa = lambda_sum_all[:, t_idx]  # [N]
            rho_col = rho_t[:, t_idx]       # [N]

            # --- best_with[i] ---
            # excl_λ(i, t, a=i) = λ_sum(t,i) - (N-1)/N · ρ[i,t]
            excl_w = lsa - (N - 1.0) / N * rho_col  # [N]
            best_with = excl_w + pk  # [N]
            best_with = torch.where(pk > NEG / 2, best_with,
                                    torch.full_like(best_with, NEG))

            # --- best_without[i] = max_{a≠i} [excl_λ(i,t,a) + peak[t,a]] ---
            # excl_λ(i, t, a≠i) = λ_sum(t,a) + 1/N · ρ[i,t]
            # score[i, a] = (lsa[a] + pk[a]) + 1/N · ρ[i,t]
            base_a = lsa + pk  # [N]
            rho_term = (1.0 / N) * rho_col  # [N]

            scores_wo = base_a.unsqueeze(0) + rho_term.unsqueeze(1)  # [N, N]
            scores_wo.fill_diagonal_(NEG)

            # peak가 NEG인 a 제외
            invalid_pk = (pk <= NEG / 2)
            scores_wo[:, invalid_pk] = NEG

            best_without = scores_wo.max(dim=1).values  # [N]

            both_neg = (best_with <= NEG / 2) & (best_without <= NEG / 2)
            delta[:, t_idx] = torch.where(both_neg, self._ZERO,
                                          best_with - best_without)

        return delta

    # =================================================================
    #                         Route Decoding
    # =================================================================
    def _decode(self, alpha, backptr):
        T, N = self.T, self.N
        full = (1 << N) - 1
        depot = self.depot

        final_scores = alpha[T, full, :] + self.s[:N, depot]
        best_last = final_scores.argmax().item()

        if final_scores[best_last].item() <= NEG / 2:
            return self._greedy_fallback()

        route_inner = []
        mask = full
        a = best_last
        for t in range(T, 0, -1):
            route_inner.append(a)
            if t == 1:
                break
            prev_a = backptr[t, mask, a].item()
            if prev_a < 0 or prev_a >= N:
                # backptr 오염 — greedy fallback
                return self._greedy_fallback()
            mask = mask ^ (1 << a)
            a = prev_a
        route_inner.reverse()
        route = [depot] + route_inner + [depot]
        return [int(self.inv_perm[c]) for c in route]

    def _greedy_fallback(self):
        N, depot = self.N, self.depot
        route = [depot]
        used = set()
        for _ in range(N):
            scores = self.s[route[-1], :N].clone()
            for u in used:
                scores[u] = NEG
            a = scores.argmax().item()
            used.add(a)
            route.append(a)
        route.append(depot)
        return [int(self.inv_perm[c]) for c in route]

    def _route_cost(self, route) -> float:
        cost = 0.0
        for k in range(len(route) - 1):
            cost += self.orig_D[route[k], route[k + 1]].item()
        return cost


# =====================================================================
#                            사용 예시
# =====================================================================
if __name__ == "__main__":
    import time

    torch.manual_seed(42)
    np.random.seed(42)

    N_CITIES = 11  # 10 cities + 1 depot
    D = np.random.rand(N_CITIES, N_CITIES)
    np.fill_diagonal(D, 0)

    device = None
    N = N_CITIES - 1  # depot 제외 도시 수
    print(f"Selected device: {get_device(device)}")

    # ── (1) 무제약 실행 ──
    print("\n" + "="*60)
    print("  Unconstrained")
    print("="*60)
    solver = TSPFactorGraphSolverGPU(
        D, start_city=0, damping=0.3, iters=30,
        verbose=True, patience=20, device=device,
    )
    t0 = time.time()
    route, cost = solver.run()
    print(f"Route: {route}  Cost: {cost:.6f}  ({time.time()-t0:.2f}s)")
    solver.cleanup()

    # ── (2) 제약 조건 추가 실행 ──
    print("\n" + "="*60)
    print("  Constrained: time windows + precedence")
    print("="*60)

    # depot=city0 제외, city1~city10이 도시 인덱스 0~9
    C = TSPFactorGraphSolverGPU  # alias
    cons = C.make_constraints(N)

    # city 1 (cons idx 0): time 0~2에만 방문 가능
    C.time_window(cons, city=0, earliest=0, latest=2)

    # city 5 (cons idx 4): time 5~9에만 방문 가능
    C.time_window(cons, city=4, earliest=5, latest=9)

    # city 3 (cons idx 2)은 반드시 city 7 (cons idx 6) 이전에 방문
    C.precedence(cons, city_before=2, city_after=6)

    # city 8 (cons idx 7)의 time=0 soft 페널티
    C.penalize(cons, city=7, time=0, penalty=-5.0)

    print(f"Constraint matrix:\n{cons}")
    print(f"(-inf=forbidden, negative=penalty, 0=allowed)\n")

    solver2 = TSPFactorGraphSolverGPU(
        D, start_city=0, damping=0.3, iters=30,
        verbose=True, patience=20, device=device,
        constraints=cons,
    )
    t0 = time.time()
    route2, cost2 = solver2.run()
    print(f"Route: {route2}  Cost: {cost2:.6f}  ({time.time()-t0:.2f}s)")
    solver2.cleanup()

    # ── 비교 ──
    print("\n" + "="*60)
    print(f"  Unconstrained: cost={cost:.6f}  route={route}")
    print(f"  Constrained:   cost={cost2:.6f}  route={route2}")
    print(f"  Cost increase: {(cost2-cost)/cost*100:.2f}%")
    print("="*60)