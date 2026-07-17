"""
cap.py
======

Capacity-Constrained Affinity Propagation (CAP).

Implements the max-sum message passing in `tsp_alg_new_with_pf.pdf`.  The model
clusters nodes (e.g. trash bins) under a per-cluster capacity budget Q on the
node weights (e.g. amount of trash):

    max_{x}  sum_{i,k} s(i,k) x_{ik}
    s.t.     sum_k x_{ik} = 1            (each node has one exemplar)
             x_{ik} <= x_{kk}            (assign only to open exemplars)
             sum_i w_i x_{ik} <= Q x_{kk} (capacity per cluster)

Message updates (using the net messages rho~ = rho(1)-rho(0), etc.):

  Responsibility (eq. 17,18,20):
        rho~_{ik} = s(i,k) - max_{k' != k} ( alpha~_{ik'} + s(i,k') )

  Availability (eq. 19):
        alpha~_{kk} = P_k^{-{k}}(Q - w_k)
        alpha~_{ik} = rho~_{kk} + P_k^{-{i,k}}(Q - w_i - w_k)
                      - max( 0, rho~_{kk} + P_k^{-{i,k}}(Q - w_k) ),   i != k

where P_k^{-A}(L) is the knapsack value over nodes j not in A with prizes
r_j = max(0, rho~_{jk}) and capacity L -- solved by the Pareto frontier
(`pareto_frontier.py`).

The raw max-sum decode is rounded to a hard assignment and then made strictly
capacity-feasible by a greedy repair (`decode_capacity_feasible`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pareto_frontier import HeadKnapsack

_NEG = -1e18  # finite stand-in for -inf to keep argmax/damping well behaved


@dataclass
class CAPResult:
    labels: np.ndarray                 # labels[i] = exemplar index of node i
    exemplars: np.ndarray              # sorted unique exemplar indices
    loads: np.ndarray                  # total weight per exemplar (indexed by exemplar id)
    n_iter: int
    converged: bool
    A: np.ndarray = field(repr=False)  # availability alpha~ (n x n)
    R: np.ndarray = field(repr=False)  # responsibility rho~ (n x n)

    @property
    def n_clusters(self) -> int:
        return len(self.exemplars)

    def clusters(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {int(k): [] for k in self.exemplars}
        for i, k in enumerate(self.labels):
            out[int(k)].append(i)
        return out


# ---------------------------------------------------------------------------
# Similarity construction
# ---------------------------------------------------------------------------
def similarity_from_distance(dist: np.ndarray, kind: str = "neg_sq") -> np.ndarray:
    """Build an AP similarity matrix from a distance matrix.

    `neg_sq` : s(i,k) = -dist^2   (compact, Gaussian-like clusters; AP default)
    `neg`    : s(i,k) = -dist
    The diagonal (preference) is set later by the solver.
    """
    if kind == "neg_sq":
        S = -(dist.astype(float) ** 2)
    elif kind == "neg":
        S = -dist.astype(float)
    else:
        raise ValueError(f"unknown similarity kind: {kind}")
    return S


# ---------------------------------------------------------------------------
# Core message passing
# ---------------------------------------------------------------------------
def cap_affinity_propagation(
    S: np.ndarray,
    w: np.ndarray,
    Q: float,
    preference: float | str | None = "median",
    damping: float = 0.7,
    max_iter: int = 300,
    conv_iter: int = 20,
    verbose: bool = False,
    decode: str = "greedy",
    bridge: np.ndarray | None = None,
    A0: np.ndarray | None = None,
    R0: np.ndarray | None = None,
) -> CAPResult:
    """Run capacity-constrained affinity propagation.

    Parameters
    ----------
    S : (n, n) similarity matrix (off-diagonal). Diagonal is overwritten by the
        preference.
    w : (n,) node weights (trash amounts). Must satisfy w_i <= Q.
    Q : capacity per cluster.
    preference : self-similarity s(k,k). Controls the number of clusters.
        "median"/"min" use the corresponding statistic of the off-diagonal
        similarities; a float sets it directly. Lower => fewer clusters.
    damping : message damping in [0.5, 1).
    max_iter / conv_iter : stop after `max_iter`, or once the hard assignment is
        unchanged for `conv_iter` consecutive iterations.
    bridge : optional (n, n) matrix of bridging messages delta~ from the routing
        layer (paper eqs. (41)-(42)): delta~^(k)_i sits at [i, e_k] on active
        exemplar columns and is zero elsewhere.  It enters omega~ and gamma~ in
        exactly the same position as s(i, e_k), so it is added to S once here
        and every message update below sees s + delta~.
    A0, R0 : optional warm starts for the availability/responsibility messages.
        Algorithm 1 of the paper initialises the messages ONCE (line 1) and then
        keeps updating the same message set across rounds while delta~ is held
        fixed per round -- pass the previous round's A/R to realise that.
    """
    n = len(w)
    S = S.astype(float).copy()
    w = np.asarray(w, dtype=float)
    if np.any(w > Q + 1e-9):
        bad = np.where(w > Q + 1e-9)[0]
        raise ValueError(f"nodes {bad.tolist()} have weight > capacity Q={Q}")

    off = S[~np.eye(n, dtype=bool)]
    if isinstance(preference, str):
        if preference == "median":
            pref = float(np.median(off))
        elif preference == "min":
            pref = float(np.min(off))
        else:
            raise ValueError("preference must be 'median', 'min', a float, or a length-n array")
    elif np.ndim(preference) >= 1:
        pref = np.asarray(preference, dtype=float)   # per-node self-preference
        if pref.shape[0] != n:
            raise ValueError(f"preference vector length {pref.shape[0]} != n={n}")
    elif isinstance(preference, (int, float)):
        pref = float(preference)
    else:
        raise ValueError("preference must be 'median', 'min', a float, or a length-n array")
    np.fill_diagonal(S, pref)

    if bridge is not None:
        # Bridging term delta~ enters omega~ (41) and gamma~ (42) alongside
        # s(i, e_k).  Adding it to S once makes every update below use
        # s + delta~ in exactly those two messages, while eta~/phi~ keep their
        # standard forms (37), (24).
        S = S + np.asarray(bridge, dtype=float)

    A = np.zeros((n, n)) if A0 is None else np.array(A0, dtype=float, copy=True)
    R = np.zeros((n, n)) if R0 is None else np.array(R0, dtype=float, copy=True)

    last_labels = None
    stable = 0
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        # ---- Responsibility: R[i,k] = S[i,k] - max_{k'!=k}(A[i,k'] + S[i,k'])
        AS = A + S
        idx1 = np.argmax(AS, axis=1)
        max1 = AS[np.arange(n), idx1]
        AS2 = AS.copy()
        AS2[np.arange(n), idx1] = _NEG
        max2 = np.max(AS2, axis=1)

        R_new = S - max1[:, None]
        R_new[np.arange(n), idx1] = S[np.arange(n), idx1] - max2
        R = damping * R + (1.0 - damping) * R_new

        # ---- Availability via Pareto-frontier knapsack (per exemplar k)
        A_new = np.full((n, n), _NEG)
        for k in range(n):
            rho_kk = R[k, k]
            cap_k = Q - w[k]
            prizes = np.maximum(0.0, R[:, k])      # r_j = max(0, rho~_{jk})

            hk = HeadKnapsack(weights=w, prizes=prizes, head=k, cap=cap_k)

            # Self-availability  alpha~_{kk} = P_k^{-{k}}(Q - w_k)
            A_new[k, k] = hk.base.value_at(cap_k)

            # P_k^{-{i,k}}(Q - w_k): for nodes with prize<=0 this equals the base
            # frontier (vectorised); positive-prize nodes need a leave-one-out.
            base_full = hk.base.value_at(cap_k)            # scalar
            base_minus = hk.base.value_at(cap_k - w)       # vector over i

            pos = np.where(prizes > 0)[0]
            P_w = np.full(n, base_full)                    # P^{-{i,k}}(Q - w_k)
            P_iw = base_minus.copy()                       # P^{-{i,k}}(Q - w_i - w_k)
            for i in pos:
                if i == k:
                    continue
                P_w[i] = hk.value(cap_k, exclude=i)
                P_iw[i] = hk.value(cap_k - w[i], exclude=i)

            term = rho_kk + P_iw
            sub = np.maximum(0.0, rho_kk + P_w)
            col = term - sub
            col[~np.isfinite(col)] = _NEG
            col = np.maximum(col, _NEG)
            col[k] = A_new[k, k]
            A_new[:, k] = col

        A = damping * A + (1.0 - damping) * A_new

        # ---- Convergence on the hard decode
        labels = np.argmax(A + R, axis=1)
        if last_labels is not None and np.array_equal(labels, last_labels):
            stable += 1
        else:
            stable = 0
        last_labels = labels
        if verbose and (it % 10 == 0 or it == 1):
            E = A + R
            ex = np.where(np.diag(E) > 0)[0]
            print(f"  [CAP] iter {it:3d}  exemplars≈{len(ex)}  stable={stable}")
        if stable >= conv_iter:
            converged = True
            break

    # ---- Decode
    if decode == "argmax":
        labels, exemplars, loads = decode_argmax_feasible(A, R, S, w, Q)
    elif decode == "raw":
        labels, exemplars, loads = decode_argmax_raw(A, R, w)
    else:
        labels, exemplars, loads = decode_capacity_feasible(A, R, S, w, Q)
    return CAPResult(
        labels=labels,
        exemplars=exemplars,
        loads=loads,
        n_iter=it,
        converged=converged,
        A=A,
        R=R,
    )


# ---------------------------------------------------------------------------
# Pure max-sum decode (no repair) -- used by Algorithm 1 (coupled.py)
# ---------------------------------------------------------------------------
def decode_argmax_raw(A, R, w):
    """순수 max-sum 디코드: b_i = argmax_k [a~(i,k) + r~(i,k)], 수선 없음.

    논문 Algorithm 1 이 반환하는 {b_ij} 그대로다.  유일한 후처리는 제약 (5)
    b_ij <= b_jj 의 강제: 누군가 exemplar 로 지목한 노드 k 는 자기 자신을
    exemplar 로 갖는다 (k 는 이미 label 집합에 있으므로 exemplar 집합은 변하지
    않고, 한 번의 패스로 충분하다).  용량 (6) 은 메시지(phi~ 의 knapsack)가
    소프트하게 강제한 결과를 그대로 두므로, 위반 여부는 호출자가 보고한다.
    """
    n = len(w)
    E = A + R
    labels = np.argmax(E, axis=1)
    for k in sorted(set(labels.tolist())):
        labels[k] = k
    exemplars = np.array(sorted(set(labels.tolist())), dtype=int)
    loads = np.zeros(n)
    for i in range(n):
        loads[labels[i]] += w[i]
    return labels, exemplars, loads


# ---------------------------------------------------------------------------
# Decoding with a hard capacity guarantee
# ---------------------------------------------------------------------------
def decode_capacity_feasible(A, R, S, w, Q):
    """Turn the soft max-sum solution into a strictly capacity-feasible labelling.

    1. Seed exemplars from the AP self-decision  E[k,k] > 0  (fallback: the most
       favourable diagonal). If total exemplar capacity is short, open extra
       exemplars greedily (the nodes least covered by the current set).
    2. Assign nodes most-committed-first (largest gap between best and 2nd-best
       exemplar) to the best exemplar that still has room; if none fits, the node
       opens its own cluster. This always terminates feasibly.
    """
    n = len(w)
    E = A + R

    exemplars = list(np.where(np.diag(E) > 0)[0])
    if not exemplars:
        exemplars = [int(np.argmax(np.diag(E)))]

    # Ensure enough capacity exists (lower bound = ceil(sum w / Q) clusters).
    total_w = float(np.sum(w))
    need = int(np.ceil(total_w / Q - 1e-9))
    while len(exemplars) < need:
        # add the node whose best similarity to current exemplars is worst
        cover = np.max(S[:, exemplars], axis=1)
        cand = int(np.argmin(np.where(np.isin(np.arange(n), exemplars), np.inf, cover)))
        exemplars.append(cand)
    exemplars = sorted(set(exemplars))

    # Commitment order: nodes with the clearest preference go first.
    Esub = E[:, exemplars]
    order_score = np.empty(n)
    for i in range(n):
        row = np.sort(Esub[i])[::-1]
        order_score[i] = (row[0] - row[1]) if len(row) > 1 else row[0]
    node_order = np.argsort(-order_score)

    labels = -np.ones(n, dtype=int)
    load = {k: 0.0 for k in exemplars}

    # Exemplars belong to their own cluster first.
    for k in exemplars:
        labels[k] = k
        load[k] += w[k]

    for i in node_order:
        if labels[i] != -1:
            continue
        # exemplars ranked by preference of i, restricted to those with room
        ranked = sorted(exemplars, key=lambda k: -E[i, k])
        placed = False
        for k in ranked:
            if load[k] + w[i] <= Q + 1e-9:
                labels[i] = k
                load[k] += w[i]
                placed = True
                break
        if not placed:
            # open a new singleton cluster for i
            labels[i] = i
            load[i] = w[i]
            exemplars = sorted(set(exemplars) | {i})

    exemplars = np.array(sorted(set(labels.tolist())), dtype=int)
    loads = np.zeros(n)
    for i in range(n):
        loads[labels[i]] += w[i]
    return labels, exemplars, loads


# ---------------------------------------------------------------------------
# Standard-AP decode (argmax) with *minimal* capacity repair
# ---------------------------------------------------------------------------
def decode_argmax_feasible(A, R, S, w, Q):
    """표준 AP 처럼 argmax 로 배정하고, **위반한 클러스터만 최소로** 수선한다.

    표준 binary AP 의 decode 는 `c_i = argmax_k (a(i,k)+r(i,k))` 한 방이다. 실측상
    이 배정은 용량을 거의 위반하지 않는다(uniform 0건, random 42개 중 1건 수준).
    그래서 전체를 헤집는 greedy(`decode_capacity_feasible`) 대신, argmax 를 그대로
    두고 **초과된 클러스터에서만** 가장 덜 붙어있는 노드를 최소한으로 빼낸다.

    빼낸 노드는 (자리 남는 가장 선호되는 head) 로, 없으면 자기 singleton 으로 간다.
    argmax 로 이미 feasible 한 클러스터는 손대지 않으므로 표준 AP 해에 가장 가깝다.
    """
    n = len(w)
    E = A + R
    labels = np.argmax(E, axis=1)

    # self-consistency: head 로 지목된 노드는 반드시 자기 클러스터에 속하게 한다
    # (argmax fixed point 가 완벽히 일관되지 않을 때 head 가 다른 곳에 배정되는 걸 방지).
    for k in set(labels.tolist()):
        labels[k] = k

    load = {}
    for k in set(labels.tolist()):
        load[k] = float(w[labels == k].sum())

    def overloaded():
        return [k for k, v in load.items() if v > Q + 1e-9]

    guard = 0
    while overloaded() and guard < 4 * n:
        guard += 1
        k = max(overloaded(), key=lambda k: load[k])          # 가장 초과된 클러스터
        members = [i for i in range(n) if labels[i] == k and i != k]
        if not members:                                       # head 혼자 초과면 손쓸 수 없음
            load[k] = Q                                        # (w_k<=Q 전제상 발생 안 함)
            continue
        # 이 head 에 가장 덜 붙은 노드(=E[i,k] 최소)를 후보로
        i = min(members, key=lambda i: E[i, k])
        # 자리 남는 다른 head 중 i 가 가장 선호(E 최대)하는 곳
        alts = [kk for kk in load if kk != k and load[kk] + w[i] <= Q + 1e-9]
        if alts:
            dst = max(alts, key=lambda kk: E[i, kk])
        else:
            dst = i                                            # 새 singleton
            load[dst] = 0.0
        labels[i] = dst
        load[k] -= w[i]
        load[dst] += w[i]

    exemplars = np.array(sorted(set(labels.tolist())), dtype=int)
    loads = np.zeros(n)
    for i in range(n):
        loads[labels[i]] += w[i]
    return labels, exemplars, loads
