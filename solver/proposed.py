"""Exact small-N SOVA and assignment-pruned alternating trellises."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time

import numpy as np

from .trellis import state_aware_trellis

from .certified_route_factor import certified_route_max_marginals
from .messages import (HARD_NEGATIVE, MessageState, decode_assignment,
                       initialize_messages, update_assignment_messages,
                       update_assignment_messages_trw, with_route_messages,
                       with_route_messages_trw)
from .model import (PaperInstance, PlanEvaluation, evaluate_routes,
                    evaluate_vehicle_route, transition_costs)
from .route_factor import FullRouteTable, build_full_route_table


@dataclass(frozen=True)
class ProposedConfig:
    max_rounds: int = 32
    assignment_rounds: int = 24
    damping: float = 0.5
    tolerance: float = 1e-3
    improvement_tolerance: float = 1e-9
    exact_global_limit: int = 14
    maximum_exact_trellis_bins: int = 84
    hypercube_radii: tuple[int, ...] = (1, 2, 4, 8, 12, 16, 24, 32)
    stagnation_patience: int = 8
    fixed_factor_epoch_rounds: int = 1
    tree_reweight: float | None = None
    trw_relaxation: float = 1.0
    canonicalize_vehicle_symmetry: bool = True
    symmetry_dual_path: bool = True
    vehicle_gauss_seidel: bool = False
    certified_route_messages: bool = False


@dataclass
class ProposedResult:
    method: str
    routes: list[list[int]]
    evaluation: PlanEvaluation
    runtime_s: float
    labels: np.ndarray
    scopes: tuple[tuple[int, ...], ...]
    exact_global_factor_graph: bool
    full_assignment_graph: bool
    route_message_mode: str
    rounds: int
    converged: bool
    messages: MessageState
    diagnostics: list[dict] = field(default_factory=list)


def _remaining_capacity(instance: PaperInstance) -> np.ndarray:
    return np.asarray([
        instance.capacity_kg - state.payload_kg
        for state in instance.vehicle_states
    ], dtype=float)


def _identical_vehicle_states(instance: PaperInstance) -> bool:
    return all(state == instance.vehicle_states[0]
               for state in instance.vehicle_states[1:])


def _global_nearest_order(instance: PaperInstance) -> list[int]:
    """Deterministic spatial walk used only to seed feasible scopes."""

    remaining = set(range(instance.n_bins))
    order: list[int] = []
    current = instance.vehicle_states[0].node
    while remaining:
        transitions = transition_costs(instance.network,
            current, [instance.bin_nodes[index] for index in remaining],
            instance.start_time_s, 0.0)
        selected = min(
            remaining,
            key=lambda index: (
                transitions[instance.bin_nodes[index]][0],
                -instance.urgency[index], index,
            ),
        )
        order.append(selected)
        remaining.remove(selected)
        current = instance.bin_nodes[selected]
    return order


def _initial_assignment(instance: PaperInstance, order: list[int],
                        maximum_bins: int | None, *,
                        canonical_vehicle_symmetry: bool = False) -> np.ndarray:
    n, vehicles = instance.n_bins, instance.vehicles
    rank = np.empty(n, dtype=float)
    rank[np.asarray(order, dtype=int)] = np.arange(n, dtype=float)
    centers = (np.arange(vehicles, dtype=float) + 0.5) * n / vehicles - 0.5
    cost = ((rank[:, None] - centers[None, :]) / max(n, 1)) ** 2
    # A tiny urgency term gives deterministic preference without changing the
    # spatial partition scale.
    belief = -cost + 1e-9 * instance.urgency[:, None]
    return decode_assignment(
        belief, instance.demand_kg, _remaining_capacity(instance),
        np.ones((n, vehicles), dtype=bool),
        maximum_bins_per_vehicle=maximum_bins,
        canonical_vehicle_symmetry=canonical_vehicle_symmetry,
        require_nonempty=instance.nonempty_required)


def _full_scopes(instance: PaperInstance) -> tuple[tuple[int, ...], ...]:
    """Return the manuscript's complete N-by-K assignment connectivity."""

    scope = tuple(range(instance.n_bins))
    return tuple(scope for _ in range(instance.vehicles))


def _routes_from_labels(labels: np.ndarray,
                        tables: tuple[FullRouteTable, ...]
                        ) -> list[list[int]] | None:
    routes: list[list[int]] = []
    for vehicle, table in enumerate(tables):
        members = set(np.flatnonzero(labels == vehicle).tolist())
        try:
            route = table.route_for_members(members)
        except KeyError:
            return None
        if route is None:
            return None
        routes.append(route)
    return routes


@dataclass(frozen=True)
class _MemberRoute:
    route: tuple[int, ...]
    energy_kwh: float
    feasible: bool
    exact: bool


def _greedy_member_route(instance: PaperInstance, vehicle: int,
                         members: tuple[int, ...]) -> _MemberRoute:
    """Unrestricted fallback when an assigned trellis is exceptionally large."""

    state = instance.vehicle_states[vehicle]
    remaining = set(members)
    route: list[int] = []
    node = state.node
    payload = float(state.payload_kg)
    elapsed = 0.0
    while remaining:
        transitions = transition_costs(
            instance.network, node,
            [instance.bin_nodes[index] for index in remaining],
            instance.start_time_s + elapsed, payload)
        selected = min(
            remaining,
            key=lambda index: (
                transitions[instance.bin_nodes[index]][0],
                transitions[instance.bin_nodes[index]][1],
                index,
            ),
        )
        edge_energy, edge_time = transitions[instance.bin_nodes[selected]]
        del edge_energy
        route.append(selected)
        remaining.remove(selected)
        node = instance.bin_nodes[selected]
        payload += float(instance.demand_kg[selected])
        elapsed += float(edge_time) + float(
            instance.network.params.service_time_s)
    evaluation = evaluate_vehicle_route(instance, vehicle, route)
    return _MemberRoute(
        tuple(route), float(evaluation.energy_kwh), evaluation.feasible, False)


def _member_route_oracle(instance: PaperInstance, config: ProposedConfig):
    """Cache exact decoded-set trellises without pruning assignment edges."""

    cache: dict[tuple[int, tuple[int, ...]], _MemberRoute] = {}
    table_cache: dict[tuple[int, tuple[int, ...]], FullRouteTable | None] = {}
    insertion_cache: dict[tuple[int, tuple[int, ...], int], _MemberRoute] = {}
    statistics = {"exact": 0, "fallback": 0, "insertion": 0}

    def route(vehicle: int, members) -> _MemberRoute:
        member_tuple = tuple(sorted(int(index) for index in members))
        key = (int(vehicle), member_tuple)
        if key in cache:
            return cache[key]
        state = instance.vehicle_states[vehicle]
        assigned_load = float(instance.demand_kg[list(member_tuple)].sum()) \
            if member_tuple else 0.0
        if assigned_load > instance.capacity_kg - state.payload_kg + 1e-9:
            result = _MemberRoute((), float("inf"), False, True)
        elif not member_tuple:
            evaluation = evaluate_vehicle_route(instance, vehicle, [])
            result = _MemberRoute(
                (), float(evaluation.energy_kwh), evaluation.feasible, True)
        elif len(member_tuple) <= min(
                config.maximum_exact_trellis_bins, 14):
            try:
                table = build_full_route_table(
                    instance, vehicle, member_tuple,
                    maximum_scope_size=max(14, len(member_tuple)),
                )
                full_mask = (1 << len(member_tuple)) - 1
                global_route = table.route_by_mask[full_mask]
                energy = float(table.energy_by_mask[full_mask])
                result = _MemberRoute(
                    tuple(global_route or ()), energy,
                    bool(global_route is not None and np.isfinite(energy)), True)
                table_cache[key] = table
                statistics["exact"] += 1
            except (MemoryError, ValueError):
                result = _greedy_member_route(instance, vehicle, member_tuple)
                table_cache[key] = None
                statistics["fallback"] += 1
        elif len(member_tuple) <= config.maximum_exact_trellis_bins:
            try:
                trellis = state_aware_trellis(
                    instance.network,
                    instance.depot_node,
                    [instance.bin_nodes[index] for index in member_tuple],
                    [float(instance.demand_kg[index]) for index in member_tuple],
                    instance.start_time_s,
                    state.payload_kg,
                    state.battery_kwh,
                    instance.reserve_kwh,
                    start_node=state.node,
                )
                global_route = tuple(member_tuple[local] for local in trellis.route)
                result = _MemberRoute(
                    global_route, float(trellis.energy_kwh),
                    bool(trellis.feasible), True)
                table_cache[key] = None
                statistics["exact"] += 1
            except (MemoryError, ValueError):
                result = _greedy_member_route(instance, vehicle, member_tuple)
                table_cache[key] = None
                statistics["fallback"] += 1
        else:
            result = _greedy_member_route(instance, vehicle, member_tuple)
            table_cache[key] = None
            statistics["fallback"] += 1
        cache[key] = result
        return result

    def insertion(vehicle: int, base_route, index: int) -> _MemberRoute:
        route_tuple = tuple(int(member) for member in base_route)
        key = (int(vehicle), route_tuple, int(index))
        if key in insertion_cache:
            return insertion_cache[key]
        best_route: tuple[int, ...] = ()
        best_energy = float("inf")
        feasible = False
        for position in range(len(route_tuple) + 1):
            candidate = (*route_tuple[:position], int(index),
                         *route_tuple[position:])
            evaluation = evaluate_vehicle_route(instance, vehicle, candidate)
            if (evaluation.feasible
                    and evaluation.energy_kwh < best_energy - 1e-12):
                best_route = tuple(candidate)
                best_energy = float(evaluation.energy_kwh)
                feasible = True
        result = _MemberRoute(best_route, best_energy, feasible, False)
        insertion_cache[key] = result
        statistics["insertion"] += 1
        return result

    return route, insertion, cache, table_cache, statistics


def _dynamic_routes_from_labels(labels: np.ndarray, route_oracle,
                                vehicles: int) -> list[list[int]] | None:
    routes: list[list[int]] = []
    for vehicle in range(vehicles):
        result = route_oracle(vehicle, np.flatnonzero(labels == vehicle))
        if not result.feasible:
            return None
        routes.append(list(result.route))
    return routes


def _fixed_set_trellis_oracle(instance: PaperInstance,
                              config: ProposedConfig):
    """Exact route oracle for one hard-decoded vehicle assignment.

    Unlike a full SOVA route factor, this oracle represents only the bins in
    the decoded set.  Its subset trellis therefore has dimension ``n_k`` and
    is rebuilt only when the outer assignment phase changes that set.
    """

    cache: dict[tuple[int, tuple[int, ...]], _MemberRoute] = {}
    statistics = {"exact": 0, "oversize": 0}

    def route(vehicle: int, members) -> _MemberRoute:
        member_tuple = tuple(sorted(int(index) for index in members))
        key = (int(vehicle), member_tuple)
        if key in cache:
            return cache[key]
        state = instance.vehicle_states[vehicle]
        assigned_load = (float(instance.demand_kg[list(member_tuple)].sum())
                         if member_tuple else 0.0)
        remaining_capacity = instance.capacity_kg - state.payload_kg
        if assigned_load > remaining_capacity + 1e-9:
            result = _MemberRoute((), float("inf"), False, True)
        elif len(member_tuple) > config.maximum_exact_trellis_bins:
            statistics["oversize"] += 1
            result = _MemberRoute((), float("inf"), False, False)
        elif not member_tuple:
            evaluation = evaluate_vehicle_route(instance, vehicle, [])
            result = _MemberRoute(
                (), float(evaluation.energy_kwh), evaluation.feasible, True)
            statistics["exact"] += 1
        else:
            try:
                trellis = state_aware_trellis(
                    instance.network,
                    instance.depot_node,
                    [instance.bin_nodes[index] for index in member_tuple],
                    [float(instance.demand_kg[index])
                     for index in member_tuple],
                    instance.start_time_s,
                    state.payload_kg,
                    state.battery_kwh,
                    instance.reserve_kwh,
                    start_node=state.node,
                )
                global_route = tuple(
                    member_tuple[local] for local in trellis.route)
                result = _MemberRoute(
                    global_route, float(trellis.energy_kwh),
                    bool(trellis.feasible), True)
            except (MemoryError, ValueError):
                result = _MemberRoute((), float("inf"), False, False)
            statistics["exact"] += 1
        cache[key] = result
        return result

    return route, cache, statistics


def _uncapped_exact_route_oracle(instance: PaperInstance):
    """Exact decoded-set trellis oracle without a scope-size cutoff."""

    cache: dict[tuple[int, tuple[int, ...]], _MemberRoute] = {}
    statistics = {"exact": 0, "oversize": 0}

    def route(vehicle: int, members) -> _MemberRoute:
        member_tuple = tuple(sorted(int(index) for index in members))
        key = (int(vehicle), member_tuple)
        cached = cache.get(key)
        if cached is not None:
            return cached
        state = instance.vehicle_states[vehicle]
        assigned_load = (float(instance.demand_kg[list(member_tuple)].sum())
                         if member_tuple else 0.0)
        remaining_capacity = instance.capacity_kg - state.payload_kg
        if assigned_load > remaining_capacity + 1e-9:
            result = _MemberRoute((), float("inf"), False, True)
        elif not member_tuple:
            evaluation = evaluate_vehicle_route(instance, vehicle, [])
            result = _MemberRoute(
                (), float(evaluation.energy_kwh), evaluation.feasible, True)
        else:
            try:
                trellis = state_aware_trellis(
                    instance.network,
                    instance.depot_node,
                    [instance.bin_nodes[index] for index in member_tuple],
                    [float(instance.demand_kg[index])
                     for index in member_tuple],
                    instance.start_time_s,
                    state.payload_kg,
                    state.battery_kwh,
                    instance.reserve_kwh,
                    start_node=state.node,
                )
            except (MemoryError, ValueError) as error:
                raise RuntimeError(
                    "certified route-factor search could not evaluate exact "
                    f"subset of size {len(member_tuple)}") from error
            global_route = tuple(
                member_tuple[local] for local in trellis.route)
            result = _MemberRoute(
                global_route, float(trellis.energy_kwh),
                bool(trellis.feasible), True)
        cache[key] = result
        statistics["exact"] += 1
        return result

    return route, cache, statistics


def _certified_hypercube_messages(
    instance: PaperInstance,
    labels: np.ndarray,
    rho: np.ndarray,
    route_oracle,
) -> tuple[np.ndarray, list[list[int]], dict]:
    """Return exact full-cube route messages with proof-based pruning only."""

    n, vehicles = instance.n_bins, instance.vehicles
    preferences = np.empty((n, vehicles), dtype=float)
    routes: list[list[int]] = []
    aggregate: dict[str, int | float | bool] = {
        "certified": True,
        "search_nodes": 0,
        "bound_prunes": 0,
        "capacity_prunes": 0,
        "route_evaluations": 0,
        "unique_masks_evaluated": 0,
    }
    for vehicle in range(vehicles):
        assigned = tuple(int(index) for index in np.flatnonzero(
            labels == vehicle))
        marginal = certified_route_max_marginals(
            instance, vehicle, rho[:, vehicle], route_oracle,
            seed_sets=(assigned,))
        preferences[:, vehicle] = marginal.delta
        route = route_oracle(vehicle, assigned)
        if not route.feasible:
            raise RuntimeError(
                "certified route factor received an infeasible incumbent")
        routes.append(list(route.route))
        for key in ("search_nodes", "bound_prunes", "capacity_prunes",
                    "route_evaluations", "unique_masks_evaluated"):
            aggregate[key] = int(aggregate[key]) + int(
                marginal.statistics[key])
    aggregate["sova_updated_edges"] = n * vehicles
    aggregate["reference_terminal_masks"] = vehicles
    aggregate["full_route_tables_built"] = 0
    aggregate["maximum_scope_size"] = n
    return preferences, routes, aggregate


def _two_sided_exchange_preference(
    base_energy: float,
    removal_energy: dict[int, float],
    exchange_energy: dict[int, float],
    rho: np.ndarray,
) -> tuple[float, float]:
    """Return one-minus-zero max-marginal on the same exchange face.

    ``exchange_energy[r]`` represents the b_i=1 vertex obtained by replacing
    r with the target i.  Its paired b_i=0 vertex is ``removal_energy[r]``.
    The unchanged reference assignment is also retained on the zero side.
    Incoming rho_r therefore appears in both conditioned maxima and cancels
    when the same removal dominates them, instead of being echoed as an
    unopposed positive-feedback term.
    """

    best_zero = 0.0
    best_one = -np.inf
    for removed, exchanged_energy in exchange_energy.items():
        if removed not in removal_energy:
            continue
        reverse = float(rho[removed])
        best_zero = max(
            best_zero,
            base_energy - float(removal_energy[removed]) - reverse)
        best_one = max(
            best_one,
            base_energy - float(exchanged_energy) - reverse)
    if not np.isfinite(best_one):
        return HARD_NEGATIVE, best_zero
    return float(best_one - best_zero), float(best_zero)


def _pruned_hypercube_messages(
    instance: PaperInstance,
    labels: np.ndarray,
    rho: np.ndarray,
    route_oracle,
    *,
    update_vehicles: tuple[int, ...] | None = None,
    previous_preferences: np.ndarray | None = None,
) -> tuple[np.ndarray, list[list[int]], dict]:
    """Evaluate exact local route marginals around an assignment vertex.

    The hard assignment fixes one terminal visited-set mask per vehicle.  This
    cuts the manuscript's assignment cube before routing: exact trellises are
    built for the current mask and every feasible incident add, remove, and
    exchange mask.  If capacity requires an ejection, the b_ik=1 exchange
    maximum and the b_ik=0 removal maximum are evaluated on the same local
    face, including the corresponding incoming ``rho`` terms from Eq. (32).
    Every ``b_ik`` edge remains represented; only complete terminal masks
    farther from the decoded assignment are pruned.
    """

    n, vehicles = instance.n_bins, instance.vehicles
    if rho.shape != (n, vehicles):
        raise ValueError(f"rho must have shape {(n, vehicles)}, got {rho.shape}")
    if previous_preferences is None:
        preferences = np.full((n, vehicles), HARD_NEGATIVE, dtype=float)
    else:
        preferences = np.asarray(previous_preferences, dtype=float).copy()
        if preferences.shape != (n, vehicles):
            raise ValueError(
                "previous_preferences must match the assignment graph shape")
    active_vehicles = tuple(
        range(vehicles) if update_vehicles is None else update_vehicles)
    if (len(set(active_vehicles)) != len(active_vehicles)
            or any(vehicle < 0 or vehicle >= vehicles
                   for vehicle in active_vehicles)):
        raise ValueError("update_vehicles contains an invalid vehicle index")
    sets = [set(np.flatnonzero(labels == vehicle).tolist())
            for vehicle in range(vehicles)]
    base = [route_oracle(vehicle, members)
            for vehicle, members in enumerate(sets)]
    if not all(result.feasible for result in base):
        raise RuntimeError("hard assignment has no feasible fixed-mask trellis")

    remaining_capacity = _remaining_capacity(instance)
    loads = [float(instance.demand_kg[list(members)].sum()) if members else 0.0
             for members in sets]
    removals: list[dict[int, _MemberRoute]] = []
    exact_removal_branches = 0
    exact_addition_branches = 0
    exact_exchange_branches = 0
    exchange_zero_branches = 0
    exchange_denominator_corrections = 0
    maximum_exchange_denominator_correction = 0.0
    singleton_fallback_branches = 0

    for vehicle, members in enumerate(sets):
        local: dict[int, _MemberRoute] = {}
        if vehicle not in active_vehicles:
            removals.append(local)
            continue
        for index in members:
            without = route_oracle(vehicle, members - {index})
            local[index] = without
            exact_removal_branches += 1
            if without.feasible:
                # With all other b_jk fixed, this is the conditioned
                # R_k -> b_ik difference Q(S_k) - Q(S_k \ {i}).
                preferences[index, vehicle] = (
                    without.energy_kwh - base[vehicle].energy_kwh)
        removals.append(local)

    all_bins = set(range(n))
    for vehicle in active_vehicles:
        members = sets[vehicle]
        for index in all_bins - members:
            demand = float(instance.demand_kg[index])
            if loads[vehicle] + demand <= remaining_capacity[vehicle] + 1e-9:
                # This must be a new fixed-mask trellis, not insertion into
                # the incumbent order: Eq. (32) conditions on membership and
                # maximizes over every feasible route order for that mask.
                with_index = route_oracle(vehicle, members | {index})
                exact_addition_branches += 1
                if with_index.feasible:
                    preferences[index, vehicle] = (
                        base[vehicle].energy_kwh - with_index.energy_kwh)
                else:
                    empty = route_oracle(vehicle, ())
                    singleton = route_oracle(vehicle, (index,))
                    singleton_fallback_branches += 1
                    if empty.feasible and singleton.feasible:
                        preferences[index, vehicle] = (
                            empty.energy_kwh - singleton.energy_kwh)
                continue

            # The capacity factor can make an insertion feasible by turning
            # off another b_jk.  For every feasible exchange, evaluate both
            # sides of the same local face: S-r+i for b_ik=1 and S-r for
            # b_ik=0.  Comparing exchange vertices only against S would leave
            # -rho_r unopposed and create positive message feedback.
            exchange_energy: dict[int, float] = {}
            removal_energy: dict[int, float] = {}
            for removed in members:
                if (loads[vehicle] - instance.demand_kg[removed] + demand
                        > remaining_capacity[vehicle] + 1e-9):
                    continue
                without = removals[vehicle][removed]
                if not without.feasible:
                    continue
                exchanged = route_oracle(
                    vehicle, (members - {removed}) | {index})
                exact_exchange_branches += 1
                if exchanged.feasible:
                    exchange_energy[removed] = exchanged.energy_kwh
                    removal_energy[removed] = without.energy_kwh
            preference, zero_correction = _two_sided_exchange_preference(
                base[vehicle].energy_kwh, removal_energy, exchange_energy,
                rho[:, vehicle])
            exchange_zero_branches += len(exchange_energy)
            if preference > 0.5 * HARD_NEGATIVE:
                preferences[index, vehicle] = preference
                if zero_correction > 1e-12:
                    exchange_denominator_corrections += 1
                    maximum_exchange_denominator_correction = max(
                        maximum_exchange_denominator_correction,
                        zero_correction)
            else:
                # Eq. (32) also admits the pair of terminal configurations
                # empty versus singleton when all other b_jk are zero.  It is
                # farther from the reference vertex but supplies a finite
                # route-factor branch without inventing a candidate vehicle
                # restriction or damping a synthetic -infinity value.
                empty = route_oracle(vehicle, ())
                singleton = route_oracle(vehicle, (index,))
                singleton_fallback_branches += 1
                if empty.feasible and singleton.feasible:
                    preferences[index, vehicle] = (
                        empty.energy_kwh - singleton.energy_kwh)

    finite = preferences > 0.5 * HARD_NEGATIVE
    statistics = {
        "full_route_tables_built": 0,
        "reference_terminal_masks": len(active_vehicles),
        "exact_removal_branches": exact_removal_branches,
        "exact_addition_branches": exact_addition_branches,
        "exact_exchange_branches": exact_exchange_branches,
        "exchange_zero_branches": exchange_zero_branches,
        "exchange_denominator_corrections": (
            exchange_denominator_corrections),
        "maximum_exchange_denominator_correction": (
            maximum_exchange_denominator_correction),
        "singleton_fallback_branches": singleton_fallback_branches,
        "sova_updated_edges": int(np.count_nonzero(finite)),
        "maximum_scope_size": max((len(members) for members in sets), default=0),
    }
    return preferences, [list(result.route) for result in base], statistics


def _route_preferences(instance: PaperInstance, labels: np.ndarray,
                       route_oracle) -> tuple[np.ndarray, list[list[int]]]:
    """Exact local route sensitivities around one decoded assignment.

    The returned value for ``(i, k)`` is the energy preference for assigning
    bin ``i`` to vehicle ``k``.  For the current vehicle it is the exact
    remove marginal; for another vehicle it is the exact add marginal after
    full route reoptimization.  If the target is currently full, every
    feasible one-bin ejection is considered so that capacity-feasible swaps
    remain visible to the following assignment phase.
    """

    n, vehicles = instance.n_bins, instance.vehicles
    preferences = np.full((n, vehicles), HARD_NEGATIVE, dtype=float)
    sets = [set(np.flatnonzero(labels == vehicle).tolist())
            for vehicle in range(vehicles)]
    base_routes = [route_oracle(vehicle, members)
                   for vehicle, members in enumerate(sets)]
    if not all(route.feasible for route in base_routes):
        raise RuntimeError("decoded assignment has no exact feasible trellis")
    remaining_capacity = _remaining_capacity(instance)
    loads = [float(instance.demand_kg[list(members)].sum()) if members else 0.0
             for members in sets]

    for vehicle in range(vehicles):
        members = sets[vehicle]
        base = base_routes[vehicle]
        # Compute all exact removal sensitivities once.  They are reused to
        # select only the two most promising capacity-releasing ejections for
        # an otherwise infeasible addition, avoiding an O(n_k) exact-trellis
        # sweep for every full-vehicle edge.
        for index in members:
            without = route_oracle(vehicle, members - {index})
            if without.feasible:
                preferences[index, vehicle] = (
                    without.energy_kwh - base.energy_kwh)

        for index in set(range(n)) - members:
            demand = float(instance.demand_kg[index])
            if loads[vehicle] + demand <= remaining_capacity[vehicle] + 1e-9:
                with_index = route_oracle(vehicle, members | {index})
                if with_index.feasible:
                    preferences[index, vehicle] = (
                        base.energy_kwh - with_index.energy_kwh)
                continue

            # A full target vehicle can still receive i when the assignment
            # phase simultaneously ejects another bin.  The difference below
            # is conditioned on that ejection and gives an exact swap-local
            # preference instead of a hard, irreversible exclusion.
            feasible_ejections = [
                removed for removed in members
                if (loads[vehicle] - instance.demand_kg[removed] + demand
                    <= remaining_capacity[vehicle] + 1e-9)
            ]
            feasible_ejections.sort(
                key=lambda removed: preferences[removed, vehicle])
            best = HARD_NEGATIVE
            for removed in feasible_ejections[:2]:
                without = route_oracle(vehicle, members - {removed})
                exchanged = route_oracle(
                    vehicle, (members - {removed}) | {index})
                if without.feasible and exchanged.feasible:
                    best = max(
                        best, without.energy_kwh - exchanged.energy_kwh)
            if best <= 0.5 * HARD_NEGATIVE:
                # Keep the full assignment edge soft even when one ejection is
                # insufficient.  The following capacity factor still enforces
                # feasibility and a multi-bin trust-region proposal can free
                # the required load before exact trellis verification.
                empty = route_oracle(vehicle, ())
                singleton = route_oracle(vehicle, {index})
                if empty.feasible and singleton.feasible:
                    best = empty.energy_kwh - singleton.energy_kwh
            preferences[index, vehicle] = best

    return preferences, [list(route.route) for route in base_routes]


def _run_assignment_phase(instance: PaperInstance, route_preferences: np.ndarray,
                          eligible: np.ndarray, config: ProposedConfig
                          ) -> tuple[MessageState, int, bool, float]:
    """Converge assignment/capacity messages with routing costs frozen."""

    state = initialize_messages(instance.n_bins, instance.vehicles, eligible)
    state, _ = with_route_messages(
        state, route_preferences, eligible, damping=1.0)
    final_change = float("inf")
    for round_index in range(1, config.assignment_rounds + 1):
        state, final_change = update_assignment_messages(
            instance.demand_kg, _remaining_capacity(instance), state,
            eligible, config.damping,
            canonical_vehicle_symmetry=(
                config.canonicalize_vehicle_symmetry),
            require_nonempty=instance.nonempty_required)
        if final_change < config.tolerance:
            return state, round_index, True, final_change
    return state, config.assignment_rounds, False, final_change


def _solve_exact_global(instance: PaperInstance, config: ProposedConfig,
                        started: float) -> ProposedResult:
    """Exact Eq. (32)-(33) route factors for tractable small instances."""

    scopes = _full_scopes(instance)
    eligible = np.ones((instance.n_bins, instance.vehicles), dtype=bool)
    order = _global_nearest_order(instance)
    accepted_labels = _initial_assignment(
        instance, order, None,
        canonical_vehicle_symmetry=config.canonicalize_vehicle_symmetry)
    tables = tuple(
        build_full_route_table(
            instance, vehicle, scope,
            maximum_scope_size=config.maximum_exact_trellis_bins)
        for vehicle, scope in enumerate(scopes)
    )
    accepted_routes = _routes_from_labels(accepted_labels, tables)
    if accepted_routes is None:
        raise RuntimeError("initial assignment has no feasible complete routes")
    accepted_evaluation = evaluate_routes(instance, accepted_routes)
    if not accepted_evaluation.feasible:
        raise RuntimeError("initial assignment is not physically feasible")

    state = initialize_messages(instance.n_bins, instance.vehicles, eligible)
    diagnostics: list[dict] = []
    converged = False
    rounds_completed = 0
    remaining_capacity = _remaining_capacity(instance)
    for round_index in range(1, config.max_rounds + 1):
        rounds_completed = round_index
        previous_labels = accepted_labels.copy()
        state, assignment_change = update_assignment_messages(
            instance.demand_kg, remaining_capacity, state, eligible,
            config.damping,
            canonical_vehicle_symmetry=(
                config.canonicalize_vehicle_symmetry),
            require_nonempty=instance.nonempty_required)
        raw_delta = np.empty_like(state.delta)
        for vehicle, table in enumerate(tables):
            raw_delta[:, vehicle] = table.max_marginals(
                state.rho[:, vehicle]).delta
        state, route_change = with_route_messages(
            state, raw_delta, eligible, config.damping)
        candidate_labels = decode_assignment(
            state.belief, instance.demand_kg, remaining_capacity, eligible,
            canonical_vehicle_symmetry=(
                config.canonicalize_vehicle_symmetry),
            require_nonempty=instance.nonempty_required)
        candidate_routes = _routes_from_labels(candidate_labels, tables)
        candidate_evaluation = (
            evaluate_routes(instance, candidate_routes)
            if candidate_routes is not None else None)
        accepted = bool(
            candidate_evaluation is not None
            and candidate_evaluation.feasible
            and candidate_evaluation.energy_kwh
            < accepted_evaluation.energy_kwh - config.improvement_tolerance)
        if accepted:
            accepted_labels = candidate_labels
            accepted_routes = candidate_routes
            accepted_evaluation = candidate_evaluation
        message_change = max(assignment_change, route_change)
        assignment_unchanged = np.array_equal(accepted_labels, previous_labels)
        diagnostics.append({
            "round": round_index,
            "candidate_energy_kwh": (
                None if candidate_evaluation is None
                else candidate_evaluation.energy_kwh),
            "accepted_energy_kwh": accepted_evaluation.energy_kwh,
            "candidate_feasible": bool(
                candidate_evaluation is not None
                and candidate_evaluation.feasible),
            "accepted": accepted,
            "assignment_unchanged": bool(assignment_unchanged),
            "message_change": message_change,
            "message_residual": message_change,
            "assignment_edges": int(eligible.sum()),
        })
        if assignment_unchanged and message_change < config.tolerance:
            converged = True
            break
    return ProposedResult(
        method="proposed", routes=accepted_routes,
        evaluation=accepted_evaluation,
        runtime_s=time.perf_counter() - started,
        labels=accepted_labels, scopes=scopes,
        exact_global_factor_graph=True,
        full_assignment_graph=True,
        route_message_mode="exact_full_sova",
        rounds=rounds_completed, converged=converged,
        messages=state, diagnostics=diagnostics)


def _converge_assignment_phase(
    instance: PaperInstance,
    state: MessageState,
    eligible: np.ndarray,
    config: ProposedConfig,
) -> tuple[MessageState, int, bool, float]:
    """Converge only the I/V block while holding SOVA messages fixed."""

    final_change = float("inf")
    for round_index in range(1, config.assignment_rounds + 1):
        state, final_change = update_assignment_messages(
            instance.demand_kg, _remaining_capacity(instance), state,
            eligible, config.damping,
            canonical_vehicle_symmetry=(
                config.canonicalize_vehicle_symmetry),
            require_nonempty=instance.nonempty_required)
        if final_change < config.tolerance:
            return state, round_index, True, final_change
    return state, config.assignment_rounds, False, final_change


def _clustered_sova_messages(
    instance: PaperInstance,
    labels: np.ndarray,
    rho: np.ndarray,
    previous_delta: np.ndarray,
    config: ProposedConfig,
    table_cache: dict[tuple[int, tuple[int, ...]], FullRouteTable],
    *,
    update_vehicles: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, list[list[int]] | None, dict]:
    """Return exact local Eq. (32)-(33) messages after one hard assignment.

    Vehicle ``k`` builds one full subset trellis on its assigned set ``S_k``.
    SOVA updates exactly the route messages attached to that local hypercube;
    messages on edges cut by the current hard assignment retain their previous
    values until a later assignment activates them.  No insertion score,
    ejection heuristic, augmented external scope, or trust region is used.
    """

    n, vehicles = instance.n_bins, instance.vehicles
    active_vehicles = tuple(
        range(vehicles) if update_vehicles is None else update_vehicles)
    if (len(set(active_vehicles)) != len(active_vehicles)
            or any(vehicle < 0 or vehicle >= vehicles
                   for vehicle in active_vehicles)):
        raise ValueError("update_vehicles contains an invalid vehicle index")
    raw_delta = np.asarray(previous_delta, dtype=float).copy()
    if raw_delta.shape != (n, vehicles):
        raise ValueError(
            f"previous_delta must have shape {(n, vehicles)}, "
            f"got {raw_delta.shape}")
    routes: list[list[int]] = []
    statistics = {
        "bundle_table_builds": 0,
        "bundle_table_hits": 0,
        "bundle_trellis_masks_built": 0,
        "bundle_maximum_scope_size": 0,
        "bundle_sova_updated_edges": 0,
    }

    def table(vehicle: int, scope: tuple[int, ...]) -> FullRouteTable:
        key = (vehicle, scope)
        cached = table_cache.get(key)
        if cached is not None:
            statistics["bundle_table_hits"] += 1
            return cached
        built = build_full_route_table(
            instance, vehicle, scope,
            maximum_scope_size=config.maximum_exact_trellis_bins)
        table_cache[key] = built
        statistics["bundle_table_builds"] += 1
        statistics["bundle_trellis_masks_built"] += built.masks
        statistics["bundle_maximum_scope_size"] = max(
            statistics["bundle_maximum_scope_size"], built.size)
        return built

    complete_routes = True
    for vehicle in range(vehicles):
        assigned = tuple(sorted(
            int(index) for index in np.flatnonzero(labels == vehicle)))
        base = table(vehicle, assigned)
        route = base.route_for_members(assigned)
        if route is None:
            complete_routes = False
            routes.append([])
        else:
            routes.append(route)

        if assigned and vehicle in active_vehicles:
            assigned_indices = np.asarray(assigned, dtype=int)
            marginals = base.max_marginals(rho[assigned_indices, vehicle])
            raw_delta[assigned_indices, vehicle] = marginals.delta
            statistics["bundle_sova_updated_edges"] += len(assigned)

    return raw_delta, (routes if complete_routes else None), statistics


def _cavity_sova_messages(
    instance: PaperInstance,
    labels: np.ndarray,
    rho: np.ndarray,
    config: ProposedConfig,
    base_table_cache: dict[tuple[int, tuple[int, ...]], FullRouteTable],
) -> tuple[np.ndarray, list[list[int]] | None, dict]:
    """Compute fresh extrinsic route messages on every ``(i, k)`` edge.

    ``S_k`` selects one local face of the full assignment hypercube.  The
    assigned edges share an exact subset trellis on ``S_k``.  For every cut
    edge ``i not in S_k``, an exact augmented cavity on ``S_k union {i}``
    reopens that dimension and returns only its target max-marginal.  No old
    off-support route message is retained and no add/remove candidate is
    accepted or rejected here.
    """

    n, vehicles = instance.n_bins, instance.vehicles
    if rho.shape != (n, vehicles):
        raise ValueError(f"rho must have shape {(n, vehicles)}, got {rho.shape}")
    raw_delta = np.full((n, vehicles), HARD_NEGATIVE, dtype=float)
    routes: list[list[int]] = []
    all_bins = set(range(n))
    statistics = {
        "base_table_builds": 0,
        "base_table_hits": 0,
        "cavity_table_builds": 0,
        "base_masks_built": 0,
        "cavity_masks_built": 0,
        "maximum_base_scope_size": 0,
        "maximum_cavity_scope_size": 0,
        "sova_updated_edges": 0,
    }

    def build(vehicle: int, scope: tuple[int, ...]) -> FullRouteTable:
        if len(scope) > config.maximum_exact_trellis_bins:
            raise ValueError(
                "augmented cavity requires an exact local trellis of "
                f"{len(scope)} bins, above the implementation safety limit "
                f"{config.maximum_exact_trellis_bins}; no cavity was pruned")
        return build_full_route_table(
            instance, vehicle, scope,
            maximum_scope_size=config.maximum_exact_trellis_bins)

    complete_routes = True
    for vehicle in range(vehicles):
        assigned = tuple(sorted(
            int(index) for index in np.flatnonzero(labels == vehicle)))
        key = (vehicle, assigned)
        base = base_table_cache.get(key)
        if base is None:
            base = build(vehicle, assigned)
            base_table_cache[key] = base
            statistics["base_table_builds"] += 1
            statistics["base_masks_built"] += base.masks
        else:
            statistics["base_table_hits"] += 1
        statistics["maximum_base_scope_size"] = max(
            statistics["maximum_base_scope_size"], len(assigned))

        full_route = base.route_for_members(assigned)
        if full_route is None:
            complete_routes = False
            routes.append([])
        else:
            routes.append(full_route)

        if assigned:
            assigned_indices = np.asarray(assigned, dtype=int)
            marginal = base.max_marginals(rho[assigned_indices, vehicle])
            raw_delta[assigned_indices, vehicle] = marginal.delta
            statistics["sova_updated_edges"] += len(assigned)

        # Temporary augmented tables are intentionally not retained.  Their
        # physical DP is exact, but keeping N*K route-by-mask tables across
        # changing faces would make memory, rather than the factor graph,
        # determine the reachable search space.
        for index in sorted(all_bins - set(assigned)):
            cavity_scope = tuple(sorted((*assigned, int(index))))
            cavity = build(vehicle, cavity_scope)
            local = cavity_scope.index(index)
            cavity_rho = rho[np.asarray(cavity_scope, dtype=int), vehicle]
            raw_delta[index, vehicle] = cavity.max_marginal_for(
                cavity_rho, local)
            statistics["cavity_table_builds"] += 1
            statistics["cavity_masks_built"] += cavity.masks
            statistics["maximum_cavity_scope_size"] = max(
                statistics["maximum_cavity_scope_size"], len(cavity_scope))
            statistics["sova_updated_edges"] += 1

    if statistics["sova_updated_edges"] != n * vehicles:
        raise AssertionError("cavity SOVA did not refresh every assignment edge")
    return raw_delta, (routes if complete_routes else None), statistics


def _canonicalize_identical_vehicle_state(
    instance: PaperInstance,
    labels: np.ndarray,
    state: MessageState,
) -> tuple[np.ndarray, MessageState, bool]:
    """Quotient out pure vehicle-label permutations when states are equal."""

    if not _identical_vehicle_states(instance):
        return labels, state, False
    vehicles = instance.vehicles
    clusters = [tuple(np.flatnonzero(labels == vehicle).tolist())
                for vehicle in range(vehicles)]
    order = np.asarray(sorted(
        range(vehicles), key=lambda vehicle: clusters[vehicle]), dtype=int)
    if np.array_equal(order, np.arange(vehicles)):
        return labels, state, False
    old_to_new = np.empty(vehicles, dtype=int)
    old_to_new[order] = np.arange(vehicles)
    canonical_labels = old_to_new[np.asarray(labels, dtype=int)]

    def columns(values: np.ndarray) -> np.ndarray:
        return np.asarray(values[:, order], dtype=float).copy()

    canonical_state = MessageState(
        eta=columns(state.eta), phi=columns(state.phi),
        delta=columns(state.delta), omega=columns(state.omega),
        gamma=columns(state.gamma), rho=columns(state.rho),
        belief=columns(state.belief))
    return canonical_labels, canonical_state, True


def _vehicle_gauss_seidel_sweep(
    instance: PaperInstance,
    labels: np.ndarray,
    state: MessageState,
    eligible: np.ndarray,
    config: ProposedConfig,
    route_oracle,
    table_cache: dict[tuple[int, tuple[int, ...]], FullRouteTable],
) -> tuple[MessageState, float, float, list[list[int]], dict]:
    """Update one route factor at a time with immediate I/V feedback.

    This is a scheduling change only.  Every ``R_k`` uses the same corrected
    two-sided local max-marginal as the parallel solver, but vehicle ``k+1``
    sees the assignment messages produced after vehicle ``k``.  A final
    read-only full-factor evaluation reports the true residual of the state
    left by the sequential sweep.
    """

    remaining_capacity = _remaining_capacity(instance)
    applied_route_residual = 0.0
    applied_assignment_residual = 0.0

    for vehicle in range(instance.vehicles):
        raw_delta, _, _ = _pruned_hypercube_messages(
            instance, labels, state.rho, route_oracle,
            update_vehicles=(vehicle,),
            previous_preferences=state.delta)
        raw_delta, _, _ = _clustered_sova_messages(
            instance, labels, state.rho, raw_delta, config, table_cache,
            update_vehicles=(vehicle,))
        state, route_residual = with_route_messages(
            state, raw_delta, eligible, config.damping)
        state, assignment_residual = update_assignment_messages(
            instance.demand_kg, remaining_capacity, state, eligible,
            config.damping,
            canonical_vehicle_symmetry=(
                config.canonicalize_vehicle_symmetry),
            require_nonempty=instance.nonempty_required)
        applied_route_residual = max(
            applied_route_residual, route_residual)
        applied_assignment_residual = max(
            applied_assignment_residual, assignment_residual)

    # Re-evaluate every factor at the final sequential state.  The proposals
    # are deliberately discarded: this is a fixed-point residual check, not
    # an extra Jacobi update.
    check_delta, routes, statistics = _pruned_hypercube_messages(
        instance, labels, state.rho, route_oracle)
    check_delta, bundle_routes, bundle_statistics = _clustered_sova_messages(
        instance, labels, state.rho, check_delta, config, table_cache)
    if bundle_routes is None:
        raise RuntimeError("Gauss-Seidel sweep has no feasible bundle trellis")
    routes = bundle_routes
    statistics.update(bundle_statistics)
    statistics["full_route_tables_built"] = (
        bundle_statistics["bundle_table_builds"])
    _, route_residual = with_route_messages(
        state, check_delta, eligible, damping=1.0)
    _, assignment_residual = update_assignment_messages(
        instance.demand_kg, remaining_capacity, state, eligible,
        damping=1.0,
        canonical_vehicle_symmetry=(
            config.canonicalize_vehicle_symmetry),
        require_nonempty=instance.nonempty_required)
    statistics.update({
        "gauss_seidel_vehicle_updates": instance.vehicles,
        "gauss_seidel_applied_route_residual": applied_route_residual,
        "gauss_seidel_applied_assignment_residual": (
            applied_assignment_residual),
    })
    return (state, assignment_residual, route_residual, routes,
            statistics)


def solve_extrinsic_cavity(
    instance: PaperInstance,
    config: ProposedConfig = ProposedConfig(),
) -> ProposedResult:
    """Run soft assignment--trellis exchange without incumbent gating.

    A hard decode is used only to choose the next local trellis face.  Every
    decoded face advances the message trajectory, including temporarily worse
    or route-infeasible plans.  The best feasible physical plan is recorded on
    a separate output path and never controls subsequent messages.
    """

    if not 0 < config.damping <= 1:
        raise ValueError("damping must be in (0, 1]")
    if config.max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    if (config.canonicalize_vehicle_symmetry
            and not _identical_vehicle_states(instance)):
        raise ValueError(
            "vehicle symmetry can be quotiented only for identical states")
    started = time.perf_counter()
    n, vehicles = instance.n_bins, instance.vehicles
    scopes = _full_scopes(instance)
    eligible = np.ones((n, vehicles), dtype=bool)
    remaining_capacity = _remaining_capacity(instance)
    current_labels = _initial_assignment(
        instance, _global_nearest_order(instance), None,
        canonical_vehicle_symmetry=config.canonicalize_vehicle_symmetry)
    state = initialize_messages(n, vehicles, eligible)
    table_cache: dict[tuple[int, tuple[int, ...]], FullRouteTable] = {}

    raw_delta, current_routes, initial_statistics = _cavity_sova_messages(
        instance, current_labels, state.rho, config, table_cache)
    if current_routes is None:
        raise RuntimeError("initial cavity assignment has no feasible route")
    current_evaluation = evaluate_routes(instance, current_routes)
    if not current_evaluation.feasible:
        raise RuntimeError("initial cavity assignment is physically infeasible")
    best_labels = current_labels.copy()
    best_routes = [list(route) for route in current_routes]
    best_evaluation = current_evaluation
    if config.tree_reweight is None:
        state, initial_route_change = with_route_messages(
            state, raw_delta, eligible, damping=1.0)
    else:
        state, initial_route_change = with_route_messages_trw(
            state, raw_delta, eligible, config.tree_reweight,
            config.trw_relaxation)

    diagnostics: list[dict] = [{
        "round": 0,
        "outer_phase": "initial_face_then_all_edge_augmented_cavity_sova",
        "trajectory_energy_kwh": current_evaluation.energy_kwh,
        "best_energy_kwh": best_evaluation.energy_kwh,
        "trajectory_route_feasible": True,
        "best_updated": True,
        "assignment_unchanged": True,
        "assignment_message_change": 0.0,
        "route_message_change": initial_route_change,
        "message_change": initial_route_change,
        "assignment_message_residual": 0.0,
        "route_message_residual": initial_route_change,
        "message_residual": initial_route_change,
        "assignment_edges": int(eligible.sum()),
        "tree_reweight": config.tree_reweight,
        "assigned_scope_sizes": [
            int(np.count_nonzero(current_labels == vehicle))
            for vehicle in range(vehicles)],
        **initial_statistics,
    }]
    stable_rounds = 0
    converged = False
    rounds_completed = 0

    for cooperative_round in range(1, config.max_rounds + 1):
        rounds_completed = cooperative_round
        # One soft I/V sweep per exchange keeps the two modules coupled.  An
        # inner hard convergence would over-commit assignment before route
        # feedback arrives and recreates the local-search failure mode.
        if config.tree_reweight is None:
            state, assignment_change = update_assignment_messages(
                instance.demand_kg, remaining_capacity, state, eligible,
                config.damping,
                canonical_vehicle_symmetry=(
                    config.canonicalize_vehicle_symmetry),
                require_nonempty=instance.nonempty_required)
        else:
            state, assignment_change = update_assignment_messages_trw(
                instance.demand_kg, remaining_capacity, state, eligible,
                config.tree_reweight, config.trw_relaxation,
                canonical_vehicle_symmetry=(
                    config.canonicalize_vehicle_symmetry),
                require_nonempty=instance.nonempty_required)
        previous_labels = current_labels.copy()
        current_labels = decode_assignment(
            state.belief, instance.demand_kg, remaining_capacity, eligible,
            canonical_vehicle_symmetry=(
                config.canonicalize_vehicle_symmetry),
            require_nonempty=instance.nonempty_required)
        current_labels, state, labels_canonicalized = (
            _canonicalize_identical_vehicle_state(
                instance, current_labels, state))

        raw_delta, current_routes, trellis_statistics = _cavity_sova_messages(
            instance, current_labels, state.rho, config, table_cache)
        if config.tree_reweight is None:
            state, route_change = with_route_messages(
                state, raw_delta, eligible, config.damping)
        else:
            state, route_change = with_route_messages_trw(
                state, raw_delta, eligible, config.tree_reweight,
                config.trw_relaxation)

        current_evaluation = (
            evaluate_routes(instance, current_routes)
            if current_routes is not None else None)
        trajectory_feasible = bool(
            current_evaluation is not None and current_evaluation.feasible)
        best_updated = bool(
            trajectory_feasible
            and current_evaluation.energy_kwh
            < best_evaluation.energy_kwh - config.improvement_tolerance)
        if best_updated:
            best_labels = current_labels.copy()
            best_routes = [list(route) for route in current_routes]
            best_evaluation = current_evaluation

        assignment_unchanged = np.array_equal(
            current_labels, previous_labels)
        message_change = max(assignment_change, route_change)
        if assignment_unchanged and message_change < config.tolerance:
            stable_rounds += 1
        else:
            stable_rounds = 0
        diagnostics.append({
            "round": cooperative_round,
            "outer_phase": "one_I_V_sweep_then_all_edge_augmented_cavity_sova",
            "trajectory_energy_kwh": (
                None if current_evaluation is None
                else current_evaluation.energy_kwh),
            "best_energy_kwh": best_evaluation.energy_kwh,
            "trajectory_route_feasible": trajectory_feasible,
            "best_updated": best_updated,
            "assignment_unchanged": bool(assignment_unchanged),
            "vehicle_labels_canonicalized": labels_canonicalized,
            "assignment_message_change": assignment_change,
            "route_message_change": route_change,
            "message_change": message_change,
            "assignment_message_residual": assignment_change,
            "route_message_residual": route_change,
            "message_residual": message_change,
            "stable_rounds": stable_rounds,
            "assignment_edges": int(eligible.sum()),
            "assigned_scope_sizes": [
                int(np.count_nonzero(current_labels == vehicle))
                for vehicle in range(vehicles)],
            **trellis_statistics,
        })
        if stable_rounds >= 2:
            diagnostics[-1]["convergence_reason"] = "message_tolerance"
            converged = True
            break

    return ProposedResult(
        method=("proposed_cavity_trw" if config.tree_reweight is not None
                else "proposed_cavity"), routes=best_routes,
        evaluation=best_evaluation,
        runtime_s=time.perf_counter() - started,
        labels=best_labels, scopes=scopes,
        exact_global_factor_graph=False,
        full_assignment_graph=True,
        route_message_mode=(
            "all_edge_augmented_cavity_sova_trw"
            if config.tree_reweight is not None
            else "all_edge_augmented_cavity_sova"),
        rounds=rounds_completed, converged=converged,
        messages=state, diagnostics=diagnostics)


def _run_fixed_factor_epoch(
    instance: PaperInstance,
    labels: np.ndarray,
    state: MessageState,
    eligible: np.ndarray,
    config: ProposedConfig,
    route_oracle,
    table_cache: dict[tuple[int, tuple[int, ...]], FullRouteTable],
) -> tuple[MessageState, int, bool, float, float, list[list[int]], dict]:
    """Iterate I/V and R while holding one pruned hypercube fixed.

    The reference assignment ``labels`` is immutable inside the epoch.  Thus
    every route update is generated by the same assigned-set trellises and
    the same incident add/remove/exchange masks.  A new assignment is decoded
    only after this fixed operator has reached the residual tolerance or the
    epoch budget is exhausted.
    """

    assignment_residual = float("inf")
    route_residual = float("inf")
    routes: list[list[int]] | None = None
    statistics: dict = {}
    residual_trace: list[dict] = []
    rounds_completed = 0
    remaining_capacity = _remaining_capacity(instance)

    for epoch_round in range(1, config.fixed_factor_epoch_rounds + 1):
        rounds_completed = epoch_round
        state, assignment_residual = update_assignment_messages(
            instance.demand_kg, remaining_capacity, state, eligible,
            config.damping,
            canonical_vehicle_symmetry=(
                config.canonicalize_vehicle_symmetry),
            require_nonempty=instance.nonempty_required)
        raw_delta, routes, statistics = _pruned_hypercube_messages(
            instance, labels, state.rho, route_oracle)
        raw_delta, bundle_routes, bundle_statistics = (
            _clustered_sova_messages(
                instance, labels, state.rho, raw_delta, config, table_cache))
        if bundle_routes is None:
            raise RuntimeError(
                "fixed-factor epoch has no feasible bundle trellis")
        routes = bundle_routes
        statistics.update(bundle_statistics)
        statistics["full_route_tables_built"] = (
            bundle_statistics["bundle_table_builds"])
        state, route_residual = with_route_messages(
            state, raw_delta, eligible, config.damping)
        residual_trace.append({
            "epoch_round": epoch_round,
            "assignment_message_residual": assignment_residual,
            "route_message_residual": route_residual,
            "message_residual": max(assignment_residual, route_residual),
        })
        if max(assignment_residual, route_residual) < config.tolerance:
            break

    if routes is None:
        raise AssertionError("fixed-factor epoch performed no route update")
    converged = bool(
        max(assignment_residual, route_residual) < config.tolerance)
    statistics["fixed_factor_epoch_residual_trace"] = residual_trace
    return (state, rounds_completed, converged, assignment_residual,
            route_residual, routes, statistics)


def _solve_dynamic_full_assignment(instance: PaperInstance,
                                   config: ProposedConfig,
                                   started: float) -> ProposedResult:
    """Alternate full P2 decoding with assignment-pruned bundle SOVA."""

    n, vehicles = instance.n_bins, instance.vehicles
    if config.stagnation_patience < 1:
        raise ValueError("stagnation_patience must be positive")
    scopes = _full_scopes(instance)
    eligible = np.ones((n, vehicles), dtype=bool)
    remaining_capacity = _remaining_capacity(instance)
    accepted_labels = _initial_assignment(
        instance, _global_nearest_order(instance), None,
        canonical_vehicle_symmetry=config.canonicalize_vehicle_symmetry)
    state = initialize_messages(n, vehicles, eligible)
    initial_labels_canonicalized = False
    if config.canonicalize_vehicle_symmetry:
        (accepted_labels, state,
         initial_labels_canonicalized) = (
            _canonicalize_identical_vehicle_state(
                instance, accepted_labels, state))

    if config.certified_route_messages:
        route_oracle, route_cache, route_statistics = (
            _uncapped_exact_route_oracle(instance))
    else:
        route_oracle, route_cache, route_statistics = (
            _fixed_set_trellis_oracle(instance, config))
    bundle_table_cache: dict[
        tuple[int, tuple[int, ...]], FullRouteTable] = {}
    # B cuts the 84-dimensional vehicle cube to a small assigned-set cube.
    # Exact local add/exchange masks keep every external assignment edge alive,
    # while a compiled FullRouteTable supplies bundle max-marginals inside the
    # assigned cube.
    if config.certified_route_messages:
        initial_delta, accepted_routes, initial_statistics = (
            _certified_hypercube_messages(
                instance, accepted_labels, state.rho, route_oracle))
    else:
        initial_delta, accepted_routes, initial_statistics = (
            _pruned_hypercube_messages(
                instance, accepted_labels, state.rho, route_oracle))
        initial_delta, bundle_routes, initial_bundle_statistics = (
            _clustered_sova_messages(
                instance, accepted_labels, state.rho, initial_delta, config,
                bundle_table_cache))
        if bundle_routes is None:
            raise RuntimeError(
                "initial assignment has no feasible bundle trellis")
        accepted_routes = bundle_routes
        initial_statistics.update(initial_bundle_statistics)
        initial_statistics["full_route_tables_built"] = (
            initial_bundle_statistics["bundle_table_builds"])
    accepted_evaluation = evaluate_routes(instance, accepted_routes)
    if not accepted_evaluation.feasible:
        raise RuntimeError("initial hard assignment is not physically feasible")
    initial_energy = accepted_evaluation.energy_kwh
    initial_scope_sizes = [
        int(np.count_nonzero(accepted_labels == vehicle))
        for vehicle in range(vehicles)]
    initial_route_cache_size = len(route_cache)
    initial_exact_route_calls = route_statistics["exact"]
    initial_oversize_route_calls = route_statistics["oversize"]
    state, _ = with_route_messages(
        state, initial_delta, eligible, damping=1.0)

    diagnostics: list[dict] = []
    converged = False
    rounds_completed = 0
    stagnant_rounds = 0

    for cooperative_round in range(1, config.max_rounds + 1):
        rounds_completed = cooperative_round
        fixed_epoch = config.fixed_factor_epoch_rounds > 1
        epoch_routes: list[list[int]] | None = None
        epoch_statistics: dict = {}
        epoch_residual_trace: list[dict] = []
        epoch_converged = False
        epoch_route_change = float("inf")
        if fixed_epoch:
            (state, assignment_rounds, epoch_converged, assignment_change,
             epoch_route_change, epoch_routes, epoch_statistics) = (
                _run_fixed_factor_epoch(
                    instance, accepted_labels, state, eligible, config,
                    route_oracle, bundle_table_cache))
            epoch_residual_trace = list(epoch_statistics.get(
                "fixed_factor_epoch_residual_trace", []))
            assignment_converged = assignment_change < config.tolerance
        else:
            state, assignment_rounds, assignment_converged, assignment_change = (
                _converge_assignment_phase(instance, state, eligible, config))
        previous_labels = accepted_labels.copy()
        previous_energy = accepted_evaluation.energy_kwh

        # Decode local safeguards and the manuscript's unrestricted P2
        # candidate.  Every decoded vertex satisfies the complete I and V
        # factors through the exact MILP decoder.
        proposals: list[dict] = []
        seen: set[tuple[int, ...]] = set()
        best_labels = accepted_labels
        best_routes = accepted_routes
        best_evaluation = accepted_evaluation
        selected_radius: int | None = None
        unrestricted_candidate_reassignments = 0
        unrestricted_labels = accepted_labels.copy()
        # The manuscript decodes P2 over the complete feasible assignment
        # domain.  Retain local shells as monotone candidate safeguards, but
        # always include radius N, whose retained-assignment constraint is
        # vacuous and therefore performs the unrestricted P2 decode.
        decode_radii = tuple(dict.fromkeys(
            [*(min(int(value), n) for value in config.hypercube_radii), n]))
        for radius in decode_radii:
            if radius == n:
                candidate_labels = decode_assignment(
                    state.belief, instance.demand_kg, remaining_capacity,
                    eligible,
                    canonical_vehicle_symmetry=(
                        config.canonicalize_vehicle_symmetry),
                    require_nonempty=instance.nonempty_required)
            else:
                candidate_labels = decode_assignment(
                    state.belief, instance.demand_kg, remaining_capacity,
                    eligible, reference_labels=accepted_labels,
                    maximum_reassignments=radius,
                    canonical_vehicle_symmetry=(
                        config.canonicalize_vehicle_symmetry),
                    require_nonempty=instance.nonempty_required)
            actual_reassignments = int(np.count_nonzero(
                candidate_labels != accepted_labels))
            if radius == n:
                unrestricted_candidate_reassignments = actual_reassignments
                unrestricted_labels = candidate_labels.copy()
            signature = tuple(int(value) for value in candidate_labels)
            if signature in seen:
                continue
            seen.add(signature)
            candidate_routes = _dynamic_routes_from_labels(
                candidate_labels, route_oracle, vehicles)
            candidate_evaluation = (
                evaluate_routes(instance, candidate_routes)
                if candidate_routes is not None else None)
            candidate_feasible = bool(
                candidate_evaluation is not None
                and candidate_evaluation.feasible)
            proposals.append({
                "hypercube_radius": radius,
                "actual_reassignments": actual_reassignments,
                "energy_kwh": (
                    None if candidate_evaluation is None
                    else candidate_evaluation.energy_kwh),
                "feasible": candidate_feasible,
                "scope_sizes": [
                    int(np.count_nonzero(candidate_labels == vehicle))
                    for vehicle in range(vehicles)],
            })
            if (candidate_feasible
                    and candidate_evaluation.energy_kwh
                    < best_evaluation.energy_kwh
                    - config.improvement_tolerance):
                best_labels = candidate_labels.copy()
                best_routes = candidate_routes
                best_evaluation = candidate_evaluation
                selected_radius = radius

        improved = selected_radius is not None
        if improved:
            accepted_labels = best_labels
            accepted_routes = best_routes
            accepted_evaluation = best_evaluation
            stagnant_rounds = 0
        else:
            stagnant_rounds += 1

        labels_canonicalized = False
        if config.canonicalize_vehicle_symmetry:
            (accepted_labels, state, labels_canonicalized) = (
                _canonicalize_identical_vehicle_state(
                    instance, accepted_labels, state))

        if fixed_epoch and not improved:
            if epoch_routes is None:
                raise AssertionError("fixed-factor epoch routes are missing")
            reconstructed_routes = epoch_routes
            trellis_statistics = epoch_statistics
            route_change = epoch_route_change
        elif config.vehicle_gauss_seidel:
            (state, assignment_change, route_change, reconstructed_routes,
             trellis_statistics) = _vehicle_gauss_seidel_sweep(
                instance, accepted_labels, state, eligible, config,
                route_oracle, bundle_table_cache)
            assignment_converged = assignment_change < config.tolerance
        else:
            if config.certified_route_messages:
                (raw_delta, reconstructed_routes,
                 trellis_statistics) = _certified_hypercube_messages(
                    instance, accepted_labels, state.rho, route_oracle)
                state, route_change = with_route_messages(
                    state, raw_delta, eligible, config.damping)
            else:
                raw_delta, reconstructed_routes, trellis_statistics = (
                    _pruned_hypercube_messages(
                        instance, accepted_labels, state.rho, route_oracle))
                raw_delta, bundle_routes, bundle_statistics = (
                    _clustered_sova_messages(
                        instance, accepted_labels, state.rho, raw_delta, config,
                        bundle_table_cache))
                if bundle_routes is None:
                    raise RuntimeError(
                        "accepted assignment has no feasible bundle trellis")
                reconstructed_routes = bundle_routes
                trellis_statistics.update(bundle_statistics)
                trellis_statistics["full_route_tables_built"] = (
                    bundle_statistics["bundle_table_builds"])
                state, route_change = with_route_messages(
                    state, raw_delta, eligible, config.damping)

        # SOVA belongs to the trellis phase of the *accepted* hard assignment.
        # The unrestricted P2 vertex is still decoded and its exact routes are
        # evaluated above, but a rejected vertex must not inject route messages
        # into the next I/V phase.  Besides preserving the strict alternating
        # B -> trellis -> B semantics, this avoids constructing an exponential
        # full-mask table for a vehicle set that was never selected.
        candidate_scope_max = max(
            (int(np.count_nonzero(unrestricted_labels == vehicle))
             for vehicle in range(vehicles)), default=0)
        candidate_bundle_statistics: dict[str, int | bool] = {
            "rejected_candidate_bundle_sova_evaluated": False,
            "candidate_bundle_maximum_scope_size": candidate_scope_max,
        }
        reconstructed_evaluation = evaluate_routes(
            instance, reconstructed_routes)
        if (not reconstructed_evaluation.feasible
                or abs(reconstructed_evaluation.energy_kwh
                       - accepted_evaluation.energy_kwh) > 1e-8):
            raise AssertionError(
                "fixed-mask trellis reconstruction disagrees with incumbent")
        accepted_routes = reconstructed_routes
        accepted_evaluation = reconstructed_evaluation

        assignment_unchanged = np.array_equal(
            accepted_labels, previous_labels)
        message_change = max(assignment_change, route_change)
        proposal_energies = [
            proposal["energy_kwh"] for proposal in proposals
            if proposal["energy_kwh"] is not None]
        diagnostics.append({
            "round": cooperative_round,
            "outer_phase": (
                "fixed_factor_epoch_hypercube_then_decode"
                if fixed_epoch
                else "I_V_hypercube_shell_then_fixed_mask_trellis"),
            "candidate_energy_kwh": (
                min(proposal_energies) if proposal_energies else None),
            "accepted_energy_kwh": accepted_evaluation.energy_kwh,
            "previous_energy_kwh": previous_energy,
            "candidate_feasible": any(
                proposal["feasible"] for proposal in proposals),
            "accepted": improved,
            "assignment_unchanged": bool(assignment_unchanged),
            "assignment_inner_rounds": assignment_rounds,
            "assignment_converged": assignment_converged,
            "fixed_factor_epoch": fixed_epoch,
            "fixed_factor_epoch_converged": epoch_converged,
            "fixed_factor_epoch_rounds": assignment_rounds,
            "vehicle_labels_canonicalized": labels_canonicalized,
            "vehicle_gauss_seidel": config.vehicle_gauss_seidel,
            "assignment_message_change": assignment_change,
            "route_message_change": route_change,
            "message_change": message_change,
            "assignment_message_residual": assignment_change,
            "route_message_residual": route_change,
            "message_residual": message_change,
            "stagnant_rounds": stagnant_rounds,
            "assignment_edges": int(eligible.sum()),
            "selected_hypercube_radius": selected_radius,
            "unrestricted_decode_evaluated": True,
            "unrestricted_candidate_reassignments": (
                unrestricted_candidate_reassignments),
            **candidate_bundle_statistics,
            "proposals": proposals,
            "assigned_scope_sizes": [
                int(np.count_nonzero(accepted_labels == vehicle))
                for vehicle in range(vehicles)],
            "route_cache_size": len(route_cache),
            "exact_route_calls": route_statistics["exact"],
            "oversize_route_calls": route_statistics["oversize"],
            **trellis_statistics,
            "fixed_factor_epoch_residual_trace": epoch_residual_trace,
        })
        message_converged = bool(
            stagnant_rounds >= 2
            and ((epoch_converged and not improved) if fixed_epoch else (
                assignment_converged
                and route_change < config.tolerance)))
        incumbent_stagnant = (
            stagnant_rounds >= config.stagnation_patience
            and assignment_unchanged)
        diagnostics[-1]["incumbent_stagnation_observed"] = (
            bool(incumbent_stagnant))
        # A stationary incumbent is not a message-passing fixed point.  In
        # particular, route feedback can remain large and later move the hard
        # assignment after several rejected decodes.  Keep the best feasible
        # incumbent, but continue until the I/V and trellis messages themselves
        # converge or the configured round budget is exhausted.
        if message_converged:
            diagnostics[-1]["convergence_reason"] = "message_tolerance"
            converged = True
            break

    diagnostics.insert(0, {
        "round": 0,
        "outer_phase": "initial_B_then_pruned_hypercube_trellis",
        "candidate_energy_kwh": initial_energy,
        "accepted_energy_kwh": initial_energy,
        "candidate_feasible": True,
        "accepted": True,
        "assignment_unchanged": True,
        "vehicle_labels_canonicalized": initial_labels_canonicalized,
        "vehicle_gauss_seidel": config.vehicle_gauss_seidel,
        "assignment_edges": int(eligible.sum()),
        "assigned_scope_sizes": initial_scope_sizes,
        "route_cache_size": initial_route_cache_size,
        "exact_route_calls": initial_exact_route_calls,
        "oversize_route_calls": initial_oversize_route_calls,
        **initial_statistics,
    })
    return ProposedResult(
        method="proposed", routes=accepted_routes,
        evaluation=accepted_evaluation,
        runtime_s=time.perf_counter() - started,
        labels=accepted_labels, scopes=scopes,
        exact_global_factor_graph=False,
        full_assignment_graph=True,
        route_message_mode=(
            "bound_certified_full_hypercube_sova"
            if config.certified_route_messages
            else (
                "fixed_factor_epoch_bundle_sova"
                if config.fixed_factor_epoch_rounds > 1
                else (
                    "assignment_pruned_bundle_sova_gauss_seidel"
                    if config.vehicle_gauss_seidel
                    else "assignment_pruned_bundle_sova"))),
        rounds=rounds_completed, converged=converged,
        messages=state, diagnostics=diagnostics)


def solve_proposed(instance: PaperInstance,
                   config: ProposedConfig = ProposedConfig()) -> ProposedResult:
    """Run Algorithm 1 without fixed bin-to-vehicle scope pruning."""

    if not 0 < config.damping <= 1:
        raise ValueError("damping must be in (0, 1]")
    if config.maximum_exact_trellis_bins < 1:
        raise ValueError("maximum_exact_trellis_bins must be positive")
    if config.assignment_rounds < 1:
        raise ValueError("assignment_rounds must be positive")
    if (config.canonicalize_vehicle_symmetry
            and not _identical_vehicle_states(instance)):
        raise ValueError(
            "vehicle symmetry can be quotiented only for identical states")
    if config.fixed_factor_epoch_rounds < 1:
        raise ValueError("fixed_factor_epoch_rounds must be positive")
    if (config.vehicle_gauss_seidel
            and config.fixed_factor_epoch_rounds > 1):
        raise ValueError(
            "vehicle_gauss_seidel cannot be combined with a fixed-factor "
            "epoch")
    if (config.certified_route_messages
            and (config.vehicle_gauss_seidel
                 or config.fixed_factor_epoch_rounds > 1)):
        raise ValueError(
            "certified route messages require the direct parallel R update")
    if (not config.hypercube_radii
            or any(int(radius) < 1 for radius in config.hypercube_radii)):
        raise ValueError("hypercube_radii must contain positive integers")
    started = time.perf_counter()
    if instance.n_bins <= config.exact_global_limit:
        return _solve_exact_global(instance, config, started)
    if (config.canonicalize_vehicle_symmetry
            and config.symmetry_dual_path
            and not config.certified_route_messages):
        labeled_config = replace(
            config, canonicalize_vehicle_symmetry=False,
            symmetry_dual_path=False)
        labeled = _solve_dynamic_full_assignment(
            instance, labeled_config, started)
        quotient_config = replace(config, symmetry_dual_path=False)
        quotient = _solve_dynamic_full_assignment(
            instance, quotient_config, started)
        labeled_diagnostics = [
            {**entry, "dual_path_branch": "labeled"}
            for entry in labeled.diagnostics]
        quotient_diagnostics = [
            {**entry, "dual_path_branch": "canonical"}
            for entry in quotient.diagnostics]
        if (quotient.evaluation.energy_kwh
                <= labeled.evaluation.energy_kwh):
            selected, rejected = quotient, labeled
            selected_diagnostics = quotient_diagnostics
            rejected_diagnostics = labeled_diagnostics
            selected_branch = "canonical"
        else:
            selected, rejected = labeled, quotient
            selected_diagnostics = labeled_diagnostics
            rejected_diagnostics = quotient_diagnostics
            selected_branch = "labeled"
        selected.diagnostics = [
            {**entry, "dual_path_selected": False}
            for entry in rejected_diagnostics] + [
            {**entry, "dual_path_selected": True}
            for entry in selected_diagnostics]
        selected.rounds += rejected.rounds
        selected.runtime_s = time.perf_counter() - started
        selected.route_message_mode = (
            "dual_labeled_canonical_bundle_sova")
        selected.diagnostics[-1]["selected_dual_path_branch"] = (
            selected_branch)
        return selected
    return _solve_dynamic_full_assignment(instance, config, started)
