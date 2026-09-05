"""Exact scalar max-sum updates for Eqs. (20), (21), and (34)-(36)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


EPS = 1e-12
HARD_NEGATIVE = -1e12
HARD_POSITIVE = 1e12


@dataclass
class MessageState:
    eta: np.ndarray
    phi: np.ndarray
    delta: np.ndarray
    omega: np.ndarray
    gamma: np.ndarray
    rho: np.ndarray
    belief: np.ndarray


def _frontier(weights: np.ndarray, prizes: np.ndarray,
              capacity: float) -> tuple[np.ndarray, np.ndarray]:
    frontier_w = np.array([0.0])
    frontier_v = np.array([0.0])
    for weight, prize in zip(weights, prizes):
        if prize <= EPS or weight > capacity + EPS:
            continue
        candidate_w = np.r_[frontier_w, frontier_w + weight]
        candidate_v = np.r_[frontier_v, frontier_v + prize]
        keep = candidate_w <= capacity + EPS
        candidate_w = candidate_w[keep]
        candidate_v = candidate_v[keep]
        order = np.lexsort((-candidate_v, candidate_w))
        candidate_w = candidate_w[order]
        candidate_v = candidate_v[order]
        running = np.maximum.accumulate(candidate_v)
        nondominated = candidate_v > np.r_[-np.inf, running[:-1]] + EPS
        frontier_w = candidate_w[nondominated]
        frontier_v = candidate_v[nondominated]
    return frontier_w, frontier_v


def _value_at(frontier: tuple[np.ndarray, np.ndarray], capacity: float) -> float:
    if capacity < -EPS:
        return -np.inf
    weights, values = frontier
    index = np.searchsorted(weights, capacity, side="right") - 1
    return float(values[max(0, index)])


def _nonempty_value_at(weights: np.ndarray, prizes: np.ndarray,
                       capacity: float) -> float:
    """Best knapsack value when at least one item must be selected.

    Positive-prize items use the ordinary optional frontier.  If none can
    improve on the empty set, the best feasible singleton supplies the exact
    non-empty value (including zero- and negative-prize cases).
    """

    feasible = weights <= capacity + EPS
    if capacity < -EPS or not np.any(feasible):
        return -np.inf
    local_weights = weights[feasible]
    local_prizes = prizes[feasible]
    positive = local_prizes > EPS
    optional = (_value_at(
        _frontier(local_weights[positive], local_prizes[positive], capacity),
        capacity) if np.any(positive) else 0.0)
    if optional > EPS:
        return optional
    return float(np.max(local_prizes))


def exclusivity_messages(omega: np.ndarray,
                         eligible: np.ndarray) -> np.ndarray:
    """Eq. (20), including the forced-one case for a singleton domain."""

    n, vehicles = omega.shape
    eta = np.full((n, vehicles), HARD_NEGATIVE, dtype=float)
    for i in range(n):
        choices = np.flatnonzero(eligible[i])
        if not len(choices):
            raise ValueError(f"bin {i} has no eligible vehicle")
        for vehicle in choices:
            others = choices[choices != vehicle]
            eta[i, vehicle] = (-float(np.max(omega[i, others]))
                               if len(others) else HARD_POSITIVE)
    return eta


def canonical_exclusivity_messages(omega: np.ndarray,
                                   eligible: np.ndarray) -> np.ndarray:
    """Exact max-sum messages for the assignment symmetry quotient.

    Identical vehicles make every unlabeled partition appear ``K!`` times in
    the binary assignment graph.  We retain exactly one representative by
    requiring the vehicle labels, read in bin-index order, to form a
    restricted-growth string: label zero appears first and label ``k`` may
    appear only after label ``k-1`` has appeared.  Every nonempty unlabeled
    partition has exactly one such labeling.

    The dynamic program computes the label max-marginals of this canonical
    one-hot factor.  It changes no physical solution when vehicles are truly
    identical; the caller is responsible for checking that precondition.
    """

    omega = np.asarray(omega, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if omega.shape != eligible.shape:
        raise ValueError("omega and eligible must have the same shape")
    n, vehicles = omega.shape
    if n == 0 or vehicles == 0:
        raise ValueError("canonical assignment factor cannot be empty")

    # State ``s`` stores one plus the largest vehicle label introduced so far;
    # s=0 is the empty prefix and s in 1..K means labels 0..s-1 exist.
    forward = np.full((n + 1, vehicles + 1), -np.inf)
    forward[0, 0] = 0.0
    for i in range(n):
        for introduced in range(vehicles + 1):
            prefix = forward[i, introduced]
            if not np.isfinite(prefix):
                continue
            maximum_label = min(introduced, vehicles - 1)
            for label in range(maximum_label + 1):
                if not eligible[i, label]:
                    continue
                next_introduced = max(introduced, label + 1)
                forward[i + 1, next_introduced] = max(
                    forward[i + 1, next_introduced],
                    prefix + float(omega[i, label]),
                )

    backward = np.full((n + 1, vehicles + 1), -np.inf)
    backward[n, :] = 0.0
    for i in range(n - 1, -1, -1):
        for introduced in range(vehicles + 1):
            maximum_label = min(introduced, vehicles - 1)
            best = -np.inf
            for label in range(maximum_label + 1):
                if not eligible[i, label]:
                    continue
                next_introduced = max(introduced, label + 1)
                suffix = backward[i + 1, next_introduced]
                if np.isfinite(suffix):
                    best = max(best, float(omega[i, label]) + suffix)
            backward[i, introduced] = best

    label_marginal = np.full((n, vehicles), -np.inf)
    for i in range(n):
        for introduced in range(vehicles + 1):
            prefix = forward[i, introduced]
            if not np.isfinite(prefix):
                continue
            maximum_label = min(introduced, vehicles - 1)
            for label in range(maximum_label + 1):
                if not eligible[i, label]:
                    continue
                next_introduced = max(introduced, label + 1)
                suffix = backward[i + 1, next_introduced]
                if np.isfinite(suffix):
                    label_marginal[i, label] = max(
                        label_marginal[i, label],
                        prefix + float(omega[i, label]) + suffix,
                    )

    eta = np.full((n, vehicles), HARD_NEGATIVE, dtype=float)
    for i in range(n):
        for vehicle in range(vehicles):
            if not eligible[i, vehicle]:
                continue
            include_full = label_marginal[i, vehicle]
            exclude_full = np.max(np.delete(label_marginal[i], vehicle)) \
                if vehicles > 1 else -np.inf
            if not np.isfinite(include_full):
                eta[i, vehicle] = HARD_NEGATIVE
            elif not np.isfinite(exclude_full):
                eta[i, vehicle] = HARD_POSITIVE
            else:
                # Remove the incoming message on the target edge.  The zero
                # state contributes no normalized scalar message.
                eta[i, vehicle] = (
                    include_full - float(omega[i, vehicle]) - exclude_full)
    return eta


def capacity_messages(weights: np.ndarray, gamma: np.ndarray,
                      remaining_capacity: np.ndarray,
                      eligible: np.ndarray,
                      require_nonempty: np.ndarray | None = None) -> np.ndarray:
    """Exact messages for joint non-empty and capacity factor ``V_k``.

    With ``b_ik=1`` the non-empty condition is already met.  With ``b_ik=0``
    at least one of the remaining eligible bins must be selected, in addition
    to satisfying the payload capacity.
    """

    n, vehicles = gamma.shape
    required = (np.ones(vehicles, dtype=bool)
                if require_nonempty is None
                else np.asarray(require_nonempty, dtype=bool))
    if required.shape != (vehicles,):
        raise ValueError("require_nonempty must have shape (vehicles,)")
    phi = np.full((n, vehicles), HARD_NEGATIVE, dtype=float)
    for vehicle in range(vehicles):
        choices = np.flatnonzero(eligible[:, vehicle])
        capacity = float(remaining_capacity[vehicle])
        prizes = np.maximum(0.0, gamma[choices, vehicle])
        positive_choices = choices[prizes > EPS]
        base = _frontier(
            weights[positive_choices], gamma[positive_choices, vehicle], capacity)
        for i in choices:
            other_choices = choices[choices != i]
            if i in positive_choices:
                without_positive = positive_choices[positive_choices != i]
                frontier = _frontier(
                    weights[without_positive],
                    gamma[without_positive, vehicle], capacity)
            else:
                frontier = base
            value_one = (
                _value_at(frontier, capacity - weights[i])
                if weights[i] <= capacity + EPS else -np.inf)
            value_zero = (
                _nonempty_value_at(
                    weights[other_choices], gamma[other_choices, vehicle],
                    capacity)
                if required[vehicle]
                else _value_at(frontier, capacity)
            )
            if np.isneginf(value_one) and np.isneginf(value_zero):
                raise ValueError(
                    f"vehicle {vehicle} cannot receive any eligible bin "
                    "within its remaining capacity")
            if np.isneginf(value_one):
                phi[i, vehicle] = HARD_NEGATIVE
            elif np.isneginf(value_zero):
                phi[i, vehicle] = HARD_POSITIVE
            else:
                phi[i, vehicle] = value_one - value_zero
    return phi


def initialize_messages(n: int, vehicles: int,
                        eligible: np.ndarray) -> MessageState:
    zeros = np.zeros((n, vehicles), dtype=float)
    delta = np.where(eligible, 0.0, HARD_NEGATIVE)
    return MessageState(
        eta=zeros.copy(), phi=zeros.copy(), delta=delta,
        omega=delta.copy(), gamma=delta.copy(), rho=zeros.copy(),
        belief=delta.copy())


def update_assignment_messages(
    weights: np.ndarray,
    remaining_capacity: np.ndarray,
    state: MessageState,
    eligible: np.ndarray,
    damping: float,
    *,
    canonical_vehicle_symmetry: bool = False,
    require_nonempty: np.ndarray | None = None,
) -> tuple[MessageState, float]:
    """Perform Algorithm 1 lines 4-6 and damp Eqs. (34)-(35).

    The returned value is the undamped fixed-point residual.  Consequently,
    the convergence tolerance has the same meaning for every damping value.
    """

    if not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")
    omega = state.phi + state.delta
    gamma = state.eta + state.delta
    raw_eta = (
        canonical_exclusivity_messages(omega, eligible)
        if canonical_vehicle_symmetry
        else exclusivity_messages(omega, eligible))
    raw_phi = capacity_messages(
        weights, gamma, remaining_capacity, eligible, require_nonempty)
    damped_eta = (1.0 - damping) * state.eta + damping * raw_eta
    damped_phi = (1.0 - damping) * state.phi + damping * raw_phi
    hard_eta = np.abs(raw_eta) >= 0.5 * HARD_POSITIVE
    eta = np.where(eligible, np.where(hard_eta, raw_eta, damped_eta),
                   HARD_NEGATIVE)
    # Infinite hard-factor messages are assigned directly, matching the
    # paper's instruction for impossible capacity choices and avoiding dozens
    # of meaningless rounds spent damping a finite stand-in for infinity.
    hard_phi = np.abs(raw_phi) >= 0.5 * HARD_POSITIVE
    phi = np.where(eligible, np.where(hard_phi, raw_phi, damped_phi),
                   HARD_NEGATIVE)
    rho = eta + phi
    omega = phi + state.delta
    gamma = eta + state.delta
    belief = rho + state.delta
    residual = max(
        float(np.max(np.abs(raw_eta[eligible] - state.eta[eligible]))),
        float(np.max(np.abs(raw_phi[eligible] - state.phi[eligible]))),
    )
    return MessageState(
        eta=eta, phi=phi, delta=state.delta.copy(), omega=omega,
        gamma=gamma, rho=rho, belief=belief), residual


def update_assignment_messages_trw(
    weights: np.ndarray,
    remaining_capacity: np.ndarray,
    state: MessageState,
    eligible: np.ndarray,
    tree_weight: float,
    relaxation: float = 1.0,
    *,
    canonical_vehicle_symmetry: bool = False,
    require_nonempty: np.ndarray | None = None,
) -> tuple[MessageState, float]:
    """Tree-reweighted I/V updates with reverse-message subtraction.

    For each factor-to-variable message, the proposal is

        w * raw_factor_message - (1-w) * reverse_variable_message.

    The reverse messages are ``phi + delta`` for I and ``eta + delta`` for V.
    This is the three-factor analogue of the correction in the HetNet update;
    it prevents information that arrived on an edge from being echoed back as
    fresh evidence.  ``relaxation`` is optional numerical under-relaxation of
    the already tree-reweighted proposal, not a replacement for the reverse
    subtraction.
    """

    if not 0 < tree_weight <= 1:
        raise ValueError("tree_weight must be in (0, 1]")
    if not 0 < relaxation <= 1:
        raise ValueError("relaxation must be in (0, 1]")
    reverse_to_i = state.phi + state.delta
    reverse_to_v = state.eta + state.delta
    raw_eta = (
        canonical_exclusivity_messages(reverse_to_i, eligible)
        if canonical_vehicle_symmetry
        else exclusivity_messages(reverse_to_i, eligible))
    raw_phi = capacity_messages(
        weights, reverse_to_v, remaining_capacity, eligible,
        require_nonempty)
    complement = 1.0 - tree_weight
    proposal_eta = tree_weight * raw_eta - complement * reverse_to_i
    proposal_phi = tree_weight * raw_phi - complement * reverse_to_v
    proposal_eta = np.clip(proposal_eta, HARD_NEGATIVE, HARD_POSITIVE)
    proposal_phi = np.clip(proposal_phi, HARD_NEGATIVE, HARD_POSITIVE)
    eta = ((1.0 - relaxation) * state.eta
           + relaxation * proposal_eta)
    phi = ((1.0 - relaxation) * state.phi
           + relaxation * proposal_phi)
    hard_eta = np.abs(raw_eta) >= 0.5 * HARD_POSITIVE
    hard_phi = np.abs(raw_phi) >= 0.5 * HARD_POSITIVE
    eta = np.where(eligible, np.where(hard_eta, raw_eta, eta), HARD_NEGATIVE)
    phi = np.where(eligible, np.where(hard_phi, raw_phi, phi), HARD_NEGATIVE)
    eta = np.clip(eta, HARD_NEGATIVE, HARD_POSITIVE)
    phi = np.clip(phi, HARD_NEGATIVE, HARD_POSITIVE)
    delta = state.delta.copy()
    rho = np.clip(eta + phi, HARD_NEGATIVE, HARD_POSITIVE)
    omega = np.clip(phi + delta, HARD_NEGATIVE, HARD_POSITIVE)
    gamma = np.clip(eta + delta, HARD_NEGATIVE, HARD_POSITIVE)
    belief = np.clip(rho + delta, HARD_NEGATIVE, HARD_POSITIVE)
    target_eta = np.where(hard_eta, raw_eta, proposal_eta)
    target_phi = np.where(hard_phi, raw_phi, proposal_phi)
    residual = max(
        float(np.max(np.abs(
            target_eta[eligible] - state.eta[eligible]))),
        float(np.max(np.abs(
            target_phi[eligible] - state.phi[eligible]))),
    )
    return MessageState(
        eta=eta, phi=phi, delta=delta, omega=omega, gamma=gamma,
        rho=rho, belief=belief), residual


def with_route_messages(state: MessageState, raw_delta: np.ndarray,
                        eligible: np.ndarray,
                        damping: float) -> tuple[MessageState, float]:
    """Apply Eq. (36) and return its undamped fixed-point residual."""

    residual = float(np.max(np.abs(
        raw_delta[eligible] - state.delta[eligible])))
    damped = (1.0 - damping) * state.delta + damping * raw_delta
    hard = np.abs(raw_delta) >= 0.5 * HARD_POSITIVE
    delta = np.where(eligible, np.where(hard, raw_delta, damped),
                     HARD_NEGATIVE)
    omega = state.phi + delta
    gamma = state.eta + delta
    rho = state.eta + state.phi
    belief = rho + delta
    return MessageState(
        eta=state.eta, phi=state.phi, delta=delta, omega=omega,
        gamma=gamma, rho=rho, belief=belief), residual


def with_route_messages_trw(
    state: MessageState,
    raw_delta: np.ndarray,
    eligible: np.ndarray,
    tree_weight: float,
    relaxation: float = 1.0,
) -> tuple[MessageState, float]:
    """Tree-reweighted R update using ``rho`` as the reverse edge message."""

    if not 0 < tree_weight <= 1:
        raise ValueError("tree_weight must be in (0, 1]")
    if not 0 < relaxation <= 1:
        raise ValueError("relaxation must be in (0, 1]")
    proposal = (tree_weight * np.asarray(raw_delta, dtype=float)
                - (1.0 - tree_weight) * state.rho)
    proposal = np.clip(proposal, HARD_NEGATIVE, HARD_POSITIVE)
    delta = (1.0 - relaxation) * state.delta + relaxation * proposal
    hard = np.abs(raw_delta) >= 0.5 * HARD_POSITIVE
    delta = np.where(eligible, np.where(hard, raw_delta, delta), HARD_NEGATIVE)
    delta = np.clip(delta, HARD_NEGATIVE, HARD_POSITIVE)
    target = np.where(hard, raw_delta, proposal)
    residual = float(np.max(np.abs(
        target[eligible] - state.delta[eligible])))
    rho = np.clip(state.eta + state.phi, HARD_NEGATIVE, HARD_POSITIVE)
    omega = np.clip(state.phi + delta, HARD_NEGATIVE, HARD_POSITIVE)
    gamma = np.clip(state.eta + delta, HARD_NEGATIVE, HARD_POSITIVE)
    belief = np.clip(rho + delta, HARD_NEGATIVE, HARD_POSITIVE)
    return MessageState(
        eta=state.eta, phi=state.phi, delta=delta, omega=omega,
        gamma=gamma, rho=rho, belief=belief), residual


def decode_assignment(
    belief: np.ndarray,
    weights: np.ndarray,
    remaining_capacity: np.ndarray,
    eligible: np.ndarray,
    *,
    maximum_bins_per_vehicle: int | None = None,
    reference_labels: np.ndarray | None = None,
    maximum_reassignments: int | None = None,
    canonical_vehicle_symmetry: bool = False,
    require_nonempty: np.ndarray | None = None,
) -> np.ndarray:
    """Decode a jointly feasible B with state-dependent nonempty columns."""

    n, vehicles = belief.shape
    required = (np.ones(vehicles, dtype=bool)
                if require_nonempty is None
                else np.asarray(require_nonempty, dtype=bool))
    if required.shape != (vehicles,):
        raise ValueError("require_nonempty must have shape (vehicles,)")
    if n < int(required.sum()):
        raise RuntimeError(
            f"{int(required.sum())} nonempty vehicles are impossible with "
            f"only {n} bins")
    c = -np.clip(belief, HARD_NEGATIVE, HARD_POSITIVE).ravel()
    # Deterministic tie break is far below the solver's numerical tolerance.
    c += 1e-10 * np.tile(np.arange(vehicles, dtype=float), n)
    exact = np.zeros((n, n * vehicles))
    for i in range(n):
        exact[i, i * vehicles:(i + 1) * vehicles] = 1.0
    capacity = np.zeros((vehicles, n * vehicles))
    cardinality = np.zeros_like(capacity)
    for vehicle in range(vehicles):
        capacity[vehicle, vehicle::vehicles] = weights
        cardinality[vehicle, vehicle::vehicles] = 1.0
    matrices = [exact, capacity]
    # I_i: every row sums to one.  Joint V_k: every column is non-empty and
    # its assigned payload remains within q_k.
    lower = [np.ones(n), required.astype(float)]
    upper = [np.ones(n), remaining_capacity]
    if canonical_vehicle_symmetry:
        # Restricted-growth representation of an unlabeled partition:
        # b[i,k] <= sum_{j<i} b[j,k-1].  Every partition into nonempty
        # identical-vehicle clusters has exactly one feasible labeling.
        canonical = np.zeros((n * max(vehicles - 1, 0), n * vehicles))
        row = 0
        for i in range(n):
            for vehicle in range(1, vehicles):
                canonical[row, i * vehicles + vehicle] = 1.0
                canonical[row, np.arange(i) * vehicles + vehicle - 1] = -1.0
                row += 1
        if row:
            matrices.append(canonical[:row])
            lower.append(np.full(row, -np.inf))
            upper.append(np.zeros(row))
    if maximum_bins_per_vehicle is not None:
        matrices.append(cardinality)
        lower.append(np.full(vehicles, -np.inf))
        upper.append(np.full(vehicles, maximum_bins_per_vehicle))
    if maximum_reassignments is not None:
        if reference_labels is None:
            raise ValueError(
                "reference_labels is required with maximum_reassignments")
        reference_labels = np.asarray(reference_labels, dtype=int)
        if reference_labels.shape != (n,):
            raise ValueError(
                f"reference_labels must have shape {(n,)}, "
                f"got {reference_labels.shape}")
        if not 0 <= maximum_reassignments <= n:
            raise ValueError("maximum_reassignments must be between 0 and N")
        retained = np.zeros((1, n * vehicles))
        retained[0, np.arange(n) * vehicles + reference_labels] = 1.0
        matrices.append(retained)
        lower.append(np.asarray([n - maximum_reassignments], dtype=float))
        upper.append(np.asarray([np.inf]))
    constraint = LinearConstraint(
        csc_matrix(np.vstack(matrices)), np.concatenate(lower),
        np.concatenate(upper))
    bounds = Bounds(np.zeros(n * vehicles), eligible.astype(float).ravel())
    result = milp(
        c=c, integrality=np.ones(n * vehicles), bounds=bounds,
        constraints=constraint,
        options={"disp": False, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"feasible assignment decode failed: {result.message}")
    matrix = result.x.reshape(n, vehicles)
    labels = np.argmax(matrix, axis=1)
    if not np.all(eligible[np.arange(n), labels]):
        raise AssertionError("decoder selected an ineligible assignment")
    return labels.astype(int)
