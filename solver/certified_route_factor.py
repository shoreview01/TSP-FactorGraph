"""Bound-certified max-marginals for the full vehicle assignment cube."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from simul.energy import service_energy_kwh

from .messages import HARD_NEGATIVE
from .model import PaperInstance


@dataclass(frozen=True)
class CertifiedRouteMarginals:
    delta: np.ndarray
    include_score: np.ndarray
    exclude_score: np.ndarray
    statistics: dict[str, int | float | bool]


def _travel_energy_floor(instance: PaperInstance, vehicle: int
                         ) -> tuple[float, float, float,
                                    np.ndarray, np.ndarray]:
    """Return safe start, inter-bin, and depot-return arc lower bounds.

    Operational route costs use a finite hourly slot and quantized payload
    state.  Taking the minimum over every slot and payload state therefore
    lower-bounds every transition that can occur in a route.  Networks used by
    unit tests need not expose the tensor; zero remains a valid lower bound.
    """

    network = instance.network
    if not hasattr(network, "relevant_cost_tensor"):
        zeros = np.zeros(instance.n_bins, dtype=float)
        return 0.0, 0.0, 0.0, zeros, zeros.copy()
    nodes, node_index, energy, _ = network.relevant_cost_tensor(
        instance.start_time_s, slot_count=24)
    del nodes
    lower = np.min(np.asarray(energy, dtype=float), axis=(0, 1))
    bin_positions = np.asarray(
        [node_index[node] for node in instance.bin_nodes], dtype=int)
    start = int(node_index[instance.vehicle_states[vehicle].node])
    depot = int(node_index[instance.depot_node])
    start_floor = float(np.min(lower[start, bin_positions]))
    return_floor = float(np.min(lower[bin_positions, depot]))
    if len(bin_positions) <= 1:
        inter_floor = 0.0
    else:
        between = lower[np.ix_(bin_positions, bin_positions)].copy()
        np.fill_diagonal(between, np.inf)
        inter_floor = float(np.min(between))
    inbound = np.empty(instance.n_bins, dtype=float)
    outbound = np.empty(instance.n_bins, dtype=float)
    for local, position in enumerate(bin_positions):
        other = np.delete(bin_positions, local)
        sources = np.r_[start, other]
        targets = np.r_[depot, other]
        inbound[local] = float(np.min(lower[sources, position]))
        outbound[local] = float(np.min(lower[position, targets]))
    return (
        max(0.0, start_floor), max(0.0, inter_floor),
        max(0.0, return_floor), np.maximum(inbound, 0.0),
        np.maximum(outbound, 0.0),
    )


def certified_route_max_marginals(
    instance: PaperInstance,
    vehicle: int,
    rho: np.ndarray,
    route_oracle: Callable[[int, Iterable[int]], object],
    *,
    seed_sets: Iterable[Iterable[int]] = (),
) -> CertifiedRouteMarginals:
    """Compute exact ``R_k -> b_ik`` messages with certified pruning.

    Search nodes are removed only when a capacity bound or a physical energy
    lower bound proves that the node cannot improve the relevant conditioned
    maximum.  The routine has no node, mask, radius, or wall-clock cap.  Its
    worst-case complexity remains exponential, as required by the exact
    high-order route factor.
    """

    rho = np.asarray(rho, dtype=float)
    n = instance.n_bins
    if rho.shape != (n,):
        raise ValueError(f"rho must have shape {(n,)}, got {rho.shape}")
    if not 0 <= int(vehicle) < instance.vehicles:
        raise IndexError("vehicle is outside the fleet")

    capacity = float(
        instance.capacity_kg - instance.vehicle_states[vehicle].payload_kg)
    weights = np.asarray(instance.demand_kg, dtype=float)
    service_floor = float(service_energy_kwh(instance.network.params))
    (start_floor, inter_floor, return_floor,
     inbound_floor, outbound_floor) = _travel_energy_floor(instance, vehicle)
    adjusted = rho - service_floor

    # High reward per capacity unit first makes the fractional upper bound
    # tight early.  Deterministic indices remove run-to-run ordering noise.
    ratio = np.divide(
        adjusted, weights, out=np.full(n, -np.inf), where=weights > 0)
    order = tuple(sorted(
        range(n), key=lambda i: (-max(float(ratio[i]), 0.0),
                                 -float(adjusted[i]), int(i))))

    route_cache: dict[tuple[int, ...], tuple[bool, float]] = {}
    route_evaluations = 0

    def evaluate(members: Iterable[int]) -> tuple[bool, float]:
        nonlocal route_evaluations
        key = tuple(sorted(int(index) for index in members))
        cached = route_cache.get(key)
        if cached is not None:
            return cached
        result = route_oracle(vehicle, key)
        feasible = bool(getattr(result, "feasible"))
        energy = float(getattr(result, "energy_kwh"))
        cached = (feasible, energy)
        route_cache[key] = cached
        route_evaluations += 1
        return cached

    include = np.full(n, -np.inf)
    exclude = np.full(n, -np.inf)

    def record(members: Iterable[int]) -> float:
        key = tuple(sorted(int(index) for index in members))
        load = float(weights[list(key)].sum()) if key else 0.0
        if load > capacity + 1e-9:
            return -np.inf
        feasible, energy = evaluate(key)
        if not feasible or not np.isfinite(energy):
            return -np.inf
        score = -energy + (float(rho[list(key)].sum()) if key else 0.0)
        selected = np.zeros(n, dtype=bool)
        if key:
            selected[np.asarray(key, dtype=int)] = True
        include[selected] = np.maximum(include[selected], score)
        exclude[~selected] = np.maximum(exclude[~selected], score)
        return float(score)

    # Finite conditioned incumbents are required before any proof-based prune.
    record(())
    for index in range(n):
        record((index,))
    for members in seed_sets:
        record(members)

    search_nodes = 0
    bound_prunes = 0
    capacity_prunes = 0

    def fractional_profit(indices: tuple[int, ...], remaining: float,
                          require_one: bool) -> float:
        positive = []
        best_single = -np.inf
        for index in indices:
            if weights[index] > remaining + 1e-9:
                continue
            value = float(adjusted[index])
            best_single = max(best_single, value)
            if value > 0.0:
                positive.append(index)
        if not positive:
            return best_single if require_one else 0.0
        value = 0.0
        capacity_left = remaining
        for index in sorted(
                positive, key=lambda i: (-float(ratio[i]), int(i))):
            if capacity_left <= 1e-12:
                break
            fraction = min(1.0, capacity_left / float(weights[index]))
            value += fraction * float(adjusted[index])
            capacity_left -= fraction * float(weights[index])
        return value

    def greedy_completion(included: tuple[int, ...], candidates: tuple[int, ...],
                          load: float) -> tuple[int, ...]:
        chosen = list(included)
        remaining = capacity - load
        for index in candidates:
            if adjusted[index] <= 0.0:
                continue
            if weights[index] <= remaining + 1e-9:
                chosen.append(index)
                remaining -= float(weights[index])
        return tuple(sorted(chosen))

    def mandatory_travel_floor(selected: tuple[int, ...]) -> float:
        if not selected:
            return start_floor + return_floor
        count_floor = (
            start_floor + return_floor
            + max(len(selected) - 1, 0) * inter_floor)
        indices = np.asarray(selected, dtype=int)
        incoming_floor = float(inbound_floor[indices].sum())
        outgoing_floor = float(outbound_floor[indices].sum())
        # Every mandatory visit has a distinct incoming and outgoing route arc.
        # Either degree sum is therefore a valid relaxation of the full tour.
        return max(count_floor, incoming_floor, outgoing_floor)

    def conditioned_search(target: int, fixed_one: bool) -> None:
        nonlocal search_nodes, bound_prunes, capacity_prunes
        included = (target,) if fixed_one else ()
        initial_load = float(weights[target]) if fixed_one else 0.0
        if initial_load > capacity + 1e-9:
            return
        candidates = tuple(index for index in order if index != target)
        record(greedy_completion(included, candidates, initial_load))

        def visit(position: int, selected: tuple[int, ...], load: float,
                  adjusted_sum: float) -> None:
            nonlocal search_nodes, bound_prunes, capacity_prunes
            search_nodes += 1
            remaining_indices = candidates[position:]
            remaining_capacity = capacity - load
            optional = fractional_profit(
                remaining_indices, remaining_capacity,
                require_one=(not selected and fixed_one is False))
            if selected:
                upper = (adjusted_sum + max(optional, 0.0)
                         - mandatory_travel_floor(selected))
            else:
                nonempty_upper = (
                    optional - mandatory_travel_floor(())
                    if np.isfinite(optional) else -np.inf)
                empty_feasible, empty_energy = evaluate(())
                empty_upper = -empty_energy if empty_feasible else -np.inf
                upper = max(empty_upper, nonempty_upper)
            incumbent = (include[target] if fixed_one else exclude[target])
            if upper <= incumbent + 1e-12:
                bound_prunes += 1
                return
            if position == len(candidates):
                record(selected)
                return

            index = candidates[position]
            next_position = position + 1
            # Exclusion is always feasible.
            visit(next_position, selected, load, adjusted_sum)
            if load + weights[index] <= capacity + 1e-9:
                expanded = tuple(sorted((*selected, index)))
                if position < 4:
                    record(greedy_completion(
                        expanded, candidates[next_position:],
                        load + float(weights[index])))
                visit(
                    next_position, expanded,
                    load + float(weights[index]),
                    adjusted_sum + float(adjusted[index]))
            else:
                capacity_prunes += 1

        visit(
            0, included, initial_load,
            float(adjusted[target]) if fixed_one else 0.0)

    for target in range(n):
        conditioned_search(target, False)
        conditioned_search(target, True)

    delta = include - rho - exclude
    delta = np.where(np.isfinite(delta), delta, HARD_NEGATIVE)
    return CertifiedRouteMarginals(
        delta=delta,
        include_score=include - rho,
        exclude_score=exclude,
        statistics={
            "certified": True,
            "search_nodes": search_nodes,
            "bound_prunes": bound_prunes,
            "capacity_prunes": capacity_prunes,
            "route_evaluations": route_evaluations,
            "unique_masks_evaluated": len(route_cache),
            "service_energy_floor_kwh": service_floor,
            "start_energy_floor_kwh": start_floor,
            "interbin_energy_floor_kwh": inter_floor,
            "return_energy_floor_kwh": return_floor,
        },
    )
