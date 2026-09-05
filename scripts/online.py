"""Reproducible wall-clock online comparison under localized disruptions.

Every method observes the same exogenous traffic function of absolute
simulation time.  Traffic is updated on a fine clock while adaptive solvers run
at a coarser fixed replanning interval.  Vehicle progress never controls the
onset or recovery of the disruption.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import heapq
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from simul.energy import service_energy_kwh

from .plotting.online import create_online_figures
from solver.baselines import solve_aco, solve_ga, solve_nn, solve_pso
from solver.model import (
    PaperInstance,
    PlanEvaluation,
    VehicleState,
    evaluate_routes,
    load_table_i_instance,
    prepare_fast_cost_oracle,
)
from solver.proposed import ProposedConfig, solve_proposed


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("proposed", "nn", "aco", "pso", "ga")
POLICIES = ("adaptive", "static_live", "static_frozen")


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class TrafficEpoch:
    epoch: int
    elapsed_s: float
    state: str
    change: str
    center_bins: tuple[int, ...]
    peak_multiplier: float
    speed_multiplier: float
    recovery_progress: float
    region_edges: tuple[str, ...]
    affected_edges: tuple[str, ...]
    multipliers: dict[str, float]


@dataclass
class RuntimeVehicleState:
    """Continuous execution state retained across wall-clock traffic steps."""

    node: str
    payload_kg: float
    battery_kwh: float
    active_edge: str | None = None
    edge_remaining_fraction: float = 0.0
    service_bin: int | None = None
    service_remaining_s: float = 0.0
    pending_path: tuple[str, ...] = ()


def _radius_edges(network, center_node: str, radius_m: float
                  ) -> dict[str, float]:
    """Return directed edges whose midpoint is within a road-distance radius."""

    if radius_m <= 0.0:
        raise ValueError("disruption radius must be positive")
    neighbors: dict[str, list[tuple[str, float]]] = {}
    for edge_id, values in network.edge_data.items():
        source = str(network.by_id.at[edge_id, "FROM_NODE"])
        target = str(values[0])
        length = float(values[1])
        neighbors.setdefault(source, []).append((target, length))
        neighbors.setdefault(target, []).append((source, length))

    center = str(center_node)
    distance = {center: 0.0}
    queue = [(0.0, center)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance != distance.get(node) or current_distance > radius_m:
            continue
        for neighbor, length in neighbors.get(node, ()):
            candidate = current_distance + length
            if candidate <= radius_m and candidate < distance.get(neighbor, np.inf):
                distance[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))

    edge_distance: dict[str, float] = {}
    for edge_id, values in network.edge_data.items():
        source = str(network.by_id.at[edge_id, "FROM_NODE"])
        target = str(values[0])
        length = float(values[1])
        midpoint_distance = min(
            distance.get(source, np.inf), distance.get(target, np.inf)
        ) + 0.5 * length
        if midpoint_distance <= radius_m:
            edge_distance[str(edge_id)] = float(midpoint_distance)
    return edge_distance


def _traffic_phase(elapsed_s: float, minimum_speed_factor: float,
                   disruption_delay_s: float, degradation_s: float,
                   disruption_hold_s: float, recovery_s: float,
                   ) -> tuple[float, str, str, float]:
    """Return the continuous speed factor and phase at absolute elapsed time."""

    elapsed = max(0.0, float(elapsed_s))
    degradation_start = float(disruption_delay_s)
    minimum_start = degradation_start + float(degradation_s)
    recovery_start = minimum_start + float(disruption_hold_s)
    recovery_end = recovery_start + float(recovery_s)
    amplitude = 1.0 - float(minimum_speed_factor)
    if elapsed < degradation_start:
        return 1.0, "normal", "waiting", 0.0
    if elapsed < minimum_start:
        progress = (elapsed - degradation_start) / float(degradation_s)
        speed = 1.0 - amplitude * progress
        return float(speed), "degrading", "degrading", 0.0
    if elapsed < recovery_start:
        return float(minimum_speed_factor), "disrupted", "holding", 0.0
    if elapsed < recovery_end:
        progress = (elapsed - recovery_start) / float(recovery_s)
        speed = minimum_speed_factor + amplitude * progress
        return float(speed), "recovering", "recovering", float(progress)
    return 1.0, "normal", "recovered", 1.0


def generate_traffic_trace(instance: PaperInstance, steps: int, seed: int,
                           disruption_fraction: float = 0.05,
                           disruption_radius_m: float = 500.0,
                           minimum_speed_factor: float = 0.35,
                           traffic_step_s: float = 60.0,
                           disruption_delay_s: float = 600.0,
                           degradation_s: float = 600.0,
                           disruption_hold_s: float = 600.0,
                           recovery_s: float = 1500.0,
                           ) -> list[TrafficEpoch]:
    """Sample a delayed, gradual disruption on an exogenous wall clock."""

    if steps < 1:
        raise ValueError("traffic steps must be positive")
    if not 0.0 < disruption_fraction <= 1.0:
        raise ValueError("disruption fraction must be in (0, 1]")
    if disruption_radius_m <= 0.0:
        raise ValueError("disruption radius must be positive")
    if not 0.0 < minimum_speed_factor < 1.0:
        raise ValueError("minimum speed factor must be in (0, 1)")
    if traffic_step_s <= 0.0:
        raise ValueError("traffic step must be positive")
    if disruption_delay_s < 0.0:
        raise ValueError("disruption delay must be nonnegative")
    if degradation_s <= 0.0:
        raise ValueError("degradation duration must be positive")
    if disruption_hold_s < 0.0:
        raise ValueError("disruption hold must be nonnegative")
    if recovery_s <= 0.0:
        raise ValueError("recovery duration must be positive")

    rng = np.random.default_rng(seed)
    center_count = max(1, int(round(instance.n_bins * disruption_fraction)))
    centers = tuple(sorted(int(value) for value in rng.choice(
        instance.n_bins, size=min(center_count, instance.n_bins),
        replace=False,
    )))
    affected_distance: dict[str, float] = {}
    for center in centers:
        for edge_id, distance_m in _radius_edges(
                instance.network, instance.bin_nodes[center],
                disruption_radius_m).items():
            affected_distance[edge_id] = min(
                affected_distance.get(edge_id, np.inf), distance_m)
    affected = tuple(sorted(affected_distance))

    trace = []
    previous_state = None
    for epoch in range(steps):
        elapsed_s = float(epoch) * float(traffic_step_s)
        (speed_multiplier, state, change,
         recovery_progress) = _traffic_phase(
             elapsed_s, minimum_speed_factor, disruption_delay_s,
             degradation_s, disruption_hold_s, recovery_s)
        if epoch == 0:
            change = "initial"
        elif state == previous_state:
            change = state
        peak = float(1.0 / speed_multiplier)
        multipliers = (
            {edge_id: peak for edge_id in affected}
            if speed_multiplier < 1.0 - 1e-12 else {})
        trace.append(TrafficEpoch(
            epoch=epoch,
            elapsed_s=elapsed_s,
            state=state,
            change=change,
            center_bins=centers,
            peak_multiplier=peak,
            speed_multiplier=speed_multiplier,
            recovery_progress=recovery_progress,
            region_edges=affected,
            affected_edges=(affected if multipliers else ()),
            multipliers=multipliers,
        ))
        previous_state = state
    return trace


def _remaining_instance(base: PaperInstance, states: list[VehicleState],
                        remaining: set[int], sim_time_s: float,
                        require_nonempty: tuple[bool, ...] | None = None,
                        ) -> tuple[PaperInstance, tuple[int, ...]]:
    global_indices = tuple(sorted(int(index) for index in remaining))
    local = PaperInstance(
        network=base.network,
        depot_node=base.depot_node,
        bin_nodes=tuple(base.bin_nodes[index] for index in global_indices),
        demand_kg=np.asarray(
            [base.demand_kg[index] for index in global_indices], float),
        urgency=np.asarray(
            [base.urgency[index] for index in global_indices], float),
        vehicle_states=tuple(states),
        start_time_s=float(sim_time_s),
        capacity_kg=base.capacity_kg,
        maximum_battery_kwh=base.maximum_battery_kwh,
        reserve_kwh=base.reserve_kwh,
        seed=base.seed,
        require_nonempty=require_nonempty,
    )
    return local, global_indices


def _solve(method: str, instance: PaperInstance, seed: int,
           args: argparse.Namespace):
    if method == "proposed":
        identical_states = all(
            state == instance.vehicle_states[0]
            for state in instance.vehicle_states[1:]
        )
        return solve_proposed(
            instance,
            ProposedConfig(
                max_rounds=args.max_rounds,
                assignment_rounds=args.assignment_rounds,
                damping=args.damping,
                tolerance=args.tolerance,
                improvement_tolerance=args.improvement_tolerance,
                exact_global_limit=args.exact_global_limit,
                maximum_exact_trellis_bins=args.maximum_exact_trellis_bins,
                hypercube_radii=tuple(args.hypercube_radii),
                stagnation_patience=args.stagnation_patience,
                canonicalize_vehicle_symmetry=(
                    args.canonicalize_vehicle_symmetry and identical_states),
                symmetry_dual_path=args.symmetry_dual_path,
                vehicle_gauss_seidel=args.vehicle_gauss_seidel,
                certified_route_messages=args.certified_route_messages,
            ),
        )
    if method == "nn":
        return solve_nn(instance)
    if method == "aco":
        return solve_aco(
            instance, seed, ants=args.aco_ants,
            iterations=args.aco_iterations)
    if method == "pso":
        return solve_pso(
            instance, seed, particles=args.pso_particles,
            iterations=args.pso_iterations)
    if method == "ga":
        return solve_ga(
            instance, seed, population=args.ga_population,
            generations=args.ga_generations)
    raise ValueError(f"unknown online method {method!r}")


def _global_routes(local_routes: list[list[int]],
                   global_indices: tuple[int, ...]) -> list[list[int]]:
    return [
        [global_indices[int(local_index)] for local_index in route]
        for route in local_routes
    ]


def _route_changes(previous: list[list[int]] | None,
                   current: list[list[int]]) -> tuple[int, int]:
    if previous is None:
        return 0, 0
    previous_owner = {
        int(index): vehicle
        for vehicle, route in enumerate(previous)
        for index in route
    }
    current_owner = {
        int(index): vehicle
        for vehicle, route in enumerate(current)
        for index in route
    }
    common = set(previous_owner) & set(current_owner)
    reassigned = sum(
        previous_owner[index] != current_owner[index] for index in common)
    next_changed = 0
    for vehicle in range(len(current)):
        old_next = previous[vehicle][0] if previous[vehicle] else None
        new_next = current[vehicle][0] if current[vehicle] else None
        next_changed += old_next != new_next
    return int(next_changed), int(reassigned)


def _evaluate_global_plan(base: PaperInstance, states: list[VehicleState],
                          remaining: set[int], routes: list[list[int]],
                          sim_time_s: float):
    local, global_indices = _remaining_instance(
        base, states, remaining, sim_time_s)
    lookup = {global_index: local_index
              for local_index, global_index in enumerate(global_indices)}
    local_routes = [
        [lookup[index] for index in route if index in lookup]
        for route in routes
    ]
    return evaluate_routes(local, local_routes)


def _candidate_improves_incumbent(incumbent_evaluation,
                                  candidate_evaluation,
                                  tolerance_kwh: float) -> bool:
    """Accept only a feasible candidate with lower residual route energy."""

    if tolerance_kwh < 0:
        raise ValueError("replan acceptance tolerance must be nonnegative")
    return bool(
        candidate_evaluation.feasible
        and candidate_evaluation.energy_kwh
        < incumbent_evaluation.energy_kwh - tolerance_kwh)


def _fixed_path_cost(network, path: tuple[str, ...] | list[str],
                     sim_time_s: float, payload_kg: float
                     ) -> tuple[float, float]:
    """Evaluate a fixed edge sequence under the current traffic snapshot."""

    energy = 0.0
    elapsed = 0.0
    for edge_id in path:
        edge_energy, edge_time = network.traversal(
            str(edge_id), sim_time_s, payload_kg)
        energy += float(edge_energy)
        elapsed += float(edge_time)
    return float(energy), float(elapsed)


def _freeze_route_paths(base: PaperInstance, routes: list[list[int]]
                        ) -> tuple[list[dict[int, tuple[str, ...]]],
                                   list[tuple[str, ...]]]:
    """Freeze every epoch-zero leg as a directed edge sequence."""

    stop_paths: list[dict[int, tuple[str, ...]]] = [
        {} for _ in range(base.vehicles)]
    return_paths: list[tuple[str, ...]] = []
    service_time = float(base.network.params.service_time_s)
    for vehicle, route in enumerate(routes):
        state = base.vehicle_states[vehicle]
        node = state.node
        payload = float(state.payload_kg)
        elapsed = 0.0
        for raw_index in route:
            index = int(raw_index)
            path, _, travel_time = base.network.shortest_path(
                node, base.bin_nodes[index],
                base.start_time_s + elapsed, payload)
            stop_paths[vehicle][index] = tuple(map(str, path))
            elapsed += float(travel_time) + service_time
            payload += float(base.demand_kg[index])
            node = base.bin_nodes[index]
        return_path, _, _ = base.network.shortest_path(
            node, base.depot_node, base.start_time_s + elapsed, payload)
        return_paths.append(tuple(map(str, return_path)))
    return stop_paths, return_paths


def _evaluate_frozen_global_plan(
        base: PaperInstance, states: list[VehicleState], remaining: set[int],
        routes: list[list[int]], sim_time_s: float,
        stop_paths: list[dict[int, tuple[str, ...]]],
        return_paths: list[tuple[str, ...]]) -> PlanEvaluation:
    """Evaluate a remaining plan without changing epoch-zero edge paths."""

    service_energy = service_energy_kwh(base.network.params)
    service_time = float(base.network.params.service_time_s)
    flattened = [
        int(index) for route in routes for index in route if index in remaining]
    exact_coverage = (
        len(flattened) == len(remaining)
        and sorted(flattened) == sorted(remaining))
    vehicle_energy: list[float] = []
    vehicle_time: list[float] = []
    capacity_feasible = True
    battery_feasible = True
    return_feasible = True
    for vehicle, route in enumerate(routes):
        state = states[vehicle]
        payload = float(state.payload_kg)
        energy = 0.0
        elapsed = 0.0
        for raw_index in route:
            index = int(raw_index)
            if index not in remaining:
                continue
            path_energy, path_time = _fixed_path_cost(
                base.network, stop_paths[vehicle][index],
                sim_time_s + elapsed, payload)
            energy += path_energy + service_energy
            elapsed += path_time + service_time
            payload += float(base.demand_kg[index])
            capacity_feasible &= payload <= base.capacity_kg + 1e-9
            battery_feasible &= (
                energy + base.reserve_kwh <= state.battery_kwh + 1e-9)
        try:
            path_energy, path_time = _fixed_path_cost(
                base.network, return_paths[vehicle],
                sim_time_s + elapsed, payload)
            energy += path_energy
            elapsed += path_time
        except (KeyError, ValueError):
            return_feasible = False
            energy = np.inf
            elapsed = np.inf
        battery_feasible &= (
            energy + base.reserve_kwh <= state.battery_kwh + 1e-9)
        vehicle_energy.append(float(energy))
        vehicle_time.append(float(elapsed))
    return PlanEvaluation(
        energy_kwh=float(sum(vehicle_energy)),
        makespan_s=float(max(vehicle_time, default=0.0)),
        served_bins=len(set(flattened)),
        served_demand_kg=(
            float(base.demand_kg[sorted(remaining)].sum())
            if remaining else 0.0),
        exact_coverage=bool(exact_coverage),
        capacity_feasible=bool(capacity_feasible),
        battery_feasible=bool(battery_feasible),
        return_feasible=bool(return_feasible),
        vehicle_energy_kwh=tuple(vehicle_energy),
        vehicle_time_s=tuple(vehicle_time),
    )


def _project_runtime_states(
        base: PaperInstance, runtime_states: list[RuntimeVehicleState],
        sim_time_s: float) -> tuple[list[VehicleState], dict[int, int]]:
    """Project unavoidable in-edge/service work to the next decision node."""

    service_energy = service_energy_kwh(base.network.params)
    service_time = float(base.network.params.service_time_s)
    projected: list[VehicleState] = []
    locked_service: dict[int, int] = {}
    for vehicle, state in enumerate(runtime_states):
        node = state.node
        payload = float(state.payload_kg)
        battery = float(state.battery_kwh)
        if state.active_edge is not None:
            edge_energy, _ = base.network.traversal(
                state.active_edge, sim_time_s, payload)
            battery -= float(edge_energy) * state.edge_remaining_fraction
            node = str(base.network.edge_data[state.active_edge][0])
        elif state.service_bin is not None:
            index = int(state.service_bin)
            fraction = state.service_remaining_s / service_time
            battery -= service_energy * fraction
            payload += float(base.demand_kg[index])
            node = base.bin_nodes[index]
            locked_service[vehicle] = index
        projected.append(VehicleState(node, payload, battery))
    return projected, locked_service


def _trim_route(route: list[int], remaining: set[int]) -> None:
    while route and route[0] not in remaining:
        route.pop(0)


def _advance_vehicle(
        base: PaperInstance, vehicle: int,
        state: RuntimeVehicleState, route: list[int], remaining: set[int],
        sim_time_s: float, duration_s: float, affected_edges: set[str],
        frozen_stop_paths: list[dict[int, tuple[str, ...]]] | None,
        ) -> dict:
    """Advance one vehicle continuously for a fixed wall-clock duration."""

    network = base.network
    service_energy = service_energy_kwh(network.params)
    service_time = float(network.params.service_time_s)
    energy = 0.0
    elapsed = 0.0
    served: list[int] = []
    traversed_edges = 0
    disrupted_edges = 0
    tolerance = 1e-9

    while elapsed < duration_s - tolerance:
        budget = duration_s - elapsed
        if state.service_bin is not None:
            used = min(budget, state.service_remaining_s)
            consumed = service_energy * used / service_time
            state.battery_kwh -= consumed
            state.service_remaining_s -= used
            energy += consumed
            elapsed += used
            if state.service_remaining_s <= tolerance:
                index = int(state.service_bin)
                state.service_bin = None
                state.service_remaining_s = 0.0
                state.payload_kg += float(base.demand_kg[index])
                if state.payload_kg > base.capacity_kg + tolerance:
                    raise RuntimeError(
                        "online execution exceeded vehicle capacity")
                if state.battery_kwh < base.reserve_kwh - tolerance:
                    raise RuntimeError(
                        "online execution violated battery reserve")
                remaining.remove(index)
                served.append(index)
                if route and route[0] == index:
                    route.pop(0)
                elif index in route:
                    route.remove(index)
                state.pending_path = ()
            continue

        if state.active_edge is not None:
            edge_id = state.active_edge
            edge_energy, edge_time = network.traversal(
                edge_id, sim_time_s + elapsed, state.payload_kg)
            remaining_time = float(edge_time) * state.edge_remaining_fraction
            used = min(budget, remaining_time)
            fraction_used = used / float(edge_time)
            consumed = float(edge_energy) * fraction_used
            state.battery_kwh -= consumed
            state.edge_remaining_fraction -= fraction_used
            energy += consumed
            elapsed += used
            if state.battery_kwh < base.reserve_kwh - tolerance:
                raise RuntimeError(
                    "online execution violated battery reserve")
            if state.edge_remaining_fraction <= tolerance:
                state.node = str(network.edge_data[edge_id][0])
                state.active_edge = None
                state.edge_remaining_fraction = 0.0
            continue

        _trim_route(route, remaining)
        if not route:
            break
        index = int(route[0])
        target = base.bin_nodes[index]
        if state.node == target:
            state.service_bin = index
            state.service_remaining_s = service_time
            continue

        if not state.pending_path:
            if frozen_stop_paths is None:
                path, _, _ = network.shortest_path(
                    state.node, target, sim_time_s + elapsed,
                    state.payload_kg)
                state.pending_path = tuple(map(str, path))
            else:
                state.pending_path = tuple(
                    frozen_stop_paths[vehicle][index])
        if not state.pending_path:
            state.node = target
            continue
        edge_id = str(state.pending_path[0])
        source = str(network.by_id.at[edge_id, "FROM_NODE"])
        if source != state.node:
            raise RuntimeError(
                f"fixed path discontinuity for vehicle {vehicle}: "
                f"{state.node!r} -> {edge_id!r}")
        state.pending_path = state.pending_path[1:]
        state.active_edge = edge_id
        state.edge_remaining_fraction = 1.0
        traversed_edges += 1
        disrupted_edges += edge_id in affected_edges

    return {
        "energy_kwh": float(energy),
        "elapsed_s": float(elapsed),
        "served": served,
        "traversed_edges": int(traversed_edges),
        "disrupted_edges": int(disrupted_edges),
    }


def _execute_time_step(
        base: PaperInstance, runtime_states: list[RuntimeVehicleState],
        remaining: set[int], routes: list[list[int]], sim_time_s: float,
        duration_s: float, affected_edges: set[str],
        frozen_stop_paths: list[dict[int, tuple[str, ...]]] | None = None,
        ) -> dict:
    epoch_energy = 0.0
    served_by_vehicle: list[list[int]] = []
    traversed_edges = 0
    disrupted_edges = 0
    vehicle_elapsed: list[float] = []
    for vehicle, state in enumerate(runtime_states):
        result = _advance_vehicle(
            base, vehicle, state, routes[vehicle], remaining,
            sim_time_s, duration_s, affected_edges,
            frozen_stop_paths)
        epoch_energy += result["energy_kwh"]
        vehicle_elapsed.append(result["elapsed_s"])
        served_by_vehicle.append(result["served"])
        traversed_edges += result["traversed_edges"]
        disrupted_edges += result["disrupted_edges"]
    return {
        "epoch_energy_kwh": float(epoch_energy),
        "epoch_operating_s": float(max(vehicle_elapsed, default=0.0)),
        "served_by_vehicle": served_by_vehicle,
        "served_bins_epoch": int(sum(map(len, served_by_vehicle))),
        "traversed_edges": int(traversed_edges),
        "disrupted_edges_traversed": int(disrupted_edges),
        "returned_to_depot": False,
    }


def _complete_returns(
        base: PaperInstance, runtime_states: list[RuntimeVehicleState],
        sim_time_s: float, trace: list[TrafficEpoch], traffic_step_s: float,
        frozen_return_paths: list[tuple[str, ...]] | None = None) -> dict:
    """Return every vehicle after service completion on the exogenous clock."""

    total_energy = 0.0
    vehicle_elapsed: list[float] = []
    traversed_edges = 0
    disrupted_edges = 0
    for vehicle, state in enumerate(runtime_states):
        elapsed = 0.0
        if frozen_return_paths is None:
            pending: tuple[str, ...] = ()
        else:
            pending = tuple(frozen_return_paths[vehicle])
        while state.node != base.depot_node:
            trace_index = min(
                int((sim_time_s - base.start_time_s + elapsed)
                    // traffic_step_s), len(trace) - 1)
            traffic = trace[trace_index]
            base.network.set_travel_time_multipliers(traffic.multipliers)
            if not pending:
                path, _, _ = base.network.shortest_path(
                    state.node, base.depot_node, sim_time_s + elapsed,
                    state.payload_kg)
                pending = tuple(map(str, path))
            if not pending:
                break
            edge_id = pending[0]
            pending = pending[1:]
            edge_energy, edge_time = base.network.traversal(
                edge_id, sim_time_s + elapsed, state.payload_kg)
            state.battery_kwh -= float(edge_energy)
            total_energy += float(edge_energy)
            elapsed += float(edge_time)
            traversed_edges += 1
            disrupted_edges += edge_id in set(traffic.affected_edges)
            state.node = str(base.network.edge_data[edge_id][0])
        if state.battery_kwh < base.reserve_kwh - 1e-9:
            raise RuntimeError("return-to-depot violated battery reserve")
        vehicle_elapsed.append(float(elapsed))
    return {
        "energy_kwh": float(total_energy),
        "makespan_s": float(max(vehicle_elapsed, default=0.0)),
        "traversed_edges": int(traversed_edges),
        "disrupted_edges": int(disrupted_edges),
    }


def _simulate_policy(base: PaperInstance, method: str, policy: str,
                     trace: list[TrafficEpoch], initial_routes: list[list[int]],
                     initial_solver_runtime_s: float,
                     initial_oracle_runtime_s: float,
                     frozen_stop_paths: list[dict[int, tuple[str, ...]]],
                     frozen_return_paths: list[tuple[str, ...]],
                     args: argparse.Namespace) -> list[dict]:
    if policy not in POLICIES:
        raise ValueError(f"unknown online policy {policy!r}")
    use_frozen_paths = policy == "static_frozen"
    runtime_states = [RuntimeVehicleState(
        state.node, state.payload_kg, state.battery_kwh)
        for state in base.vehicle_states]
    remaining = set(range(base.n_bins))
    routes = [list(route) for route in initial_routes]
    ever_served = np.zeros(base.vehicles, dtype=bool)
    cumulative_energy = 0.0
    cumulative_operating_s = 0.0
    cumulative_planning_s = 0.0
    rows: list[dict] = []

    replan_stride = int(round(
        args.replanning_interval_s / args.traffic_step_s))
    for traffic in trace:
        if not remaining:
            break
        base.network.set_travel_time_multipliers(traffic.multipliers)
        sim_time_s = base.start_time_s + traffic.elapsed_s
        if not use_frozen_paths:
            for runtime_state in runtime_states:
                runtime_state.pending_path = ()
        planner_states, locked_service = _project_runtime_states(
            base, runtime_states, sim_time_s)
        planning_remaining = remaining - set(locked_service.values())
        projected_ever_served = ever_served.copy()
        for vehicle in locked_service:
            projected_ever_served[vehicle] = True
        remaining_before = len(remaining)
        solver_runtime = 0.0
        oracle_runtime = 0.0
        converged = None
        rounds = None
        is_replan_epoch = traffic.epoch % replan_stride == 0
        replanned = traffic.epoch == 0
        replan_attempted = False
        replan_accepted = False
        acceptance_reason = "initial_plan" if traffic.epoch == 0 else "not_attempted"
        incumbent_plan_energy = float("nan")
        candidate_plan_energy = float("nan")
        candidate_next_choice_changes = 0
        candidate_reassigned_bins = 0
        incumbent = [
            [index for index in route if index in planning_remaining]
            for route in routes
        ]

        if traffic.epoch == 0:
            solver_runtime = initial_solver_runtime_s
            oracle_runtime = initial_oracle_runtime_s
        elif (policy == "adaptive" and is_replan_epoch
              and planning_remaining):
            replanned = True
            replan_attempted = True
            incumbent_evaluation = _evaluate_global_plan(
                base, planner_states, planning_remaining, incumbent,
                sim_time_s)
            if not incumbent_evaluation.feasible:
                raise RuntimeError(
                    f"{method} incumbent is infeasible at epoch "
                    f"{traffic.epoch}")
            incumbent_plan_energy = float(incumbent_evaluation.energy_kwh)
            local, global_indices = _remaining_instance(
                base, planner_states, planning_remaining, sim_time_s,
                require_nonempty=tuple(
                    (~projected_ever_served).tolist()))
            oracle_runtime = prepare_fast_cost_oracle(
                local, adjacent_slots=args.oracle_slots)
            replan_index = int(round(
                traffic.elapsed_s / args.replanning_interval_s))
            result = _solve(
                method, local,
                args.solver_seed + 1009 * replan_index,
                args,
            )
            if not result.evaluation.feasible:
                raise RuntimeError(
                    f"{method} returned an infeasible online plan at epoch "
                    f"{traffic.epoch}")
            candidate_routes = _global_routes(result.routes, global_indices)
            solver_runtime = float(result.runtime_s)
            converged = getattr(result, "converged", None)
            rounds = getattr(result, "rounds", None)
            candidate_evaluation = _evaluate_global_plan(
                base, planner_states, planning_remaining, candidate_routes,
                sim_time_s)
            candidate_plan_energy = float(candidate_evaluation.energy_kwh)
            (candidate_next_choice_changes,
             candidate_reassigned_bins) = _route_changes(
                 incumbent, candidate_routes)
            replan_accepted = _candidate_improves_incumbent(
                incumbent_evaluation, candidate_evaluation,
                args.replan_acceptance_tolerance_kwh)
            if replan_accepted:
                routes = [
                    ([locked_service[vehicle]] if vehicle in locked_service
                     else []) + list(candidate_routes[vehicle])
                    for vehicle in range(base.vehicles)
                ]
                acceptance_reason = "lower_residual_energy"
            else:
                routes = [
                    ([locked_service[vehicle]] if vehicle in locked_service
                     else []) + list(incumbent[vehicle])
                    for vehicle in range(base.vehicles)
                ]
                acceptance_reason = "incumbent_not_worse"

        if replanned and traffic.epoch > 0:
            next_choice_changes, reassigned_bins = _route_changes(
                incumbent, [
                    [index for index in route
                     if index in planning_remaining]
                    for route in routes
                ])
        else:
            next_choice_changes, reassigned_bins = 0, 0
        planning_routes = [
            [index for index in route if index in planning_remaining]
            for route in routes
        ]
        plan_evaluation = _evaluate_global_plan(
            base, planner_states, planning_remaining, planning_routes,
            sim_time_s)
        active_planning_vehicles = int(
            sum(bool(route) for route in planning_routes))

        execution = _execute_time_step(
            base, runtime_states, remaining, routes, sim_time_s,
            args.traffic_step_s, set(traffic.affected_edges),
            frozen_stop_paths if use_frozen_paths else None)
        for vehicle, served in enumerate(execution["served_by_vehicle"]):
            ever_served[vehicle] |= bool(served)
        planning_runtime = oracle_runtime + solver_runtime
        cumulative_energy += execution["epoch_energy_kwh"]
        cumulative_operating_s = (
            traffic.elapsed_s + execution["epoch_operating_s"])
        if not remaining:
            returns = _complete_returns(
                base, runtime_states,
                sim_time_s + args.traffic_step_s,
                trace, args.traffic_step_s,
                frozen_return_paths if use_frozen_paths else None)
            execution["epoch_energy_kwh"] += returns["energy_kwh"]
            execution["epoch_operating_s"] += returns["makespan_s"]
            execution["traversed_edges"] += returns["traversed_edges"]
            execution["disrupted_edges_traversed"] += (
                returns["disrupted_edges"])
            execution["returned_to_depot"] = True
            cumulative_energy += returns["energy_kwh"]
            cumulative_operating_s += returns["makespan_s"]
        cumulative_planning_s += planning_runtime
        hour = int(sim_time_s // 3600.0) % 24
        base.network.set_travel_time_multipliers(traffic.multipliers)
        network_speed = base.network.hourly_network_speed_kmh(hour)
        row = {
            "method": method,
            "policy": policy,
            "epoch": traffic.epoch,
            "elapsed_s": traffic.elapsed_s,
            "elapsed_min": traffic.elapsed_s / 60.0,
            "replan_epoch": int(traffic.epoch // replan_stride),
            "traffic_state": traffic.state,
            "traffic_change": traffic.change,
            "center_bins": " ".join(map(str, traffic.center_bins)),
            "affected_edges": len(traffic.affected_edges),
            "peak_tt_multiplier": traffic.peak_multiplier,
            "speed_multiplier": traffic.speed_multiplier,
            "recovery_progress": traffic.recovery_progress,
            "network_speed_kmh": network_speed,
            "sim_hour": sim_time_s / 3600.0 % 24.0,
            "remaining_bins_before": remaining_before,
            "served_bins_epoch": execution["served_bins_epoch"],
            "remaining_bins_after": len(remaining),
            "epoch_energy_kwh": execution["epoch_energy_kwh"],
            "epoch_energy_per_served_bin_kwh": (
                execution["epoch_energy_kwh"]
                / max(execution["served_bins_epoch"], 1)),
            "cumulative_energy_kwh": cumulative_energy,
            "epoch_operating_s": execution["epoch_operating_s"],
            "cumulative_operating_s": cumulative_operating_s,
            "oracle_runtime_s": oracle_runtime,
            "solver_runtime_s": solver_runtime,
            "planning_runtime_s": planning_runtime,
            "cumulative_planning_runtime_s": cumulative_planning_s,
            "compute_wait_aux_energy_kwh": 0.0,
            "cumulative_compute_wait_aux_energy_kwh": 0.0,
            "latency_adjusted_energy_kwh": cumulative_energy,
            "latency_adjusted_time_s": cumulative_operating_s,
            "planned_remaining_energy_kwh": plan_evaluation.energy_kwh,
            "planned_remaining_makespan_s": plan_evaluation.makespan_s,
            "incumbent_plan_energy_kwh": incumbent_plan_energy,
            "candidate_plan_energy_kwh": candidate_plan_energy,
            "candidate_energy_improvement_kwh": (
                incumbent_plan_energy - candidate_plan_energy),
            "replan_attempted": replan_attempted,
            "replan_accepted": replan_accepted,
            "acceptance_reason": acceptance_reason,
            "candidate_next_choice_changes": candidate_next_choice_changes,
            "candidate_reassigned_bins": candidate_reassigned_bins,
            "next_choice_changes": next_choice_changes,
            "reassigned_bins": reassigned_bins,
            "traversed_edges": execution["traversed_edges"],
            "disrupted_edges_traversed": (
                execution["disrupted_edges_traversed"]),
            "disrupted_edge_share": (
                execution["disrupted_edges_traversed"]
                / max(execution["traversed_edges"], 1)),
            "min_battery_kwh": min(
                state.battery_kwh for state in runtime_states),
            "max_payload_kg": max(
                state.payload_kg for state in runtime_states),
            "returned_to_depot": execution["returned_to_depot"],
            "converged": converged,
            "rounds": rounds,
            "replanned": replanned,
            "active_planning_vehicles": active_planning_vehicles,
            "vehicles_still_requiring_first_service": int(
                np.count_nonzero(~ever_served)),
        }
        rows.append(row)
        if not remaining:
            break

    if remaining:
        raise RuntimeError(
            f"traffic trace ended with {len(remaining)} unserved bins for "
            f"{method}/{policy}; increase --max-simulation-minutes")
    return rows


def _scenario_records(base: PaperInstance,
                      trace: list[TrafficEpoch]) -> list[dict]:
    rows = []
    previous_speed = None
    for traffic in trace:
        base.network.set_travel_time_multipliers(traffic.multipliers)
        sim_time_s = base.start_time_s + traffic.elapsed_s
        hour = int(sim_time_s // 3600.0) % 24
        speed = base.network.hourly_network_speed_kmh(hour)
        region_distance = sum(
            base.network.edge_data[edge_id][1]
            for edge_id in traffic.region_edges)
        region_elapsed = sum(
            base.network.edge_data[edge_id][3][hour]
            for edge_id in traffic.region_edges)
        region_speed = (
            3.6 * region_distance / region_elapsed
            if region_elapsed > 0.0 else float("nan"))
        affected_length = sum(
            base.network.edge_data[edge_id][1]
            for edge_id in traffic.affected_edges
        ) / 1000.0
        speed_change = (
            0.0 if previous_speed is None
            else 100.0 * (speed / previous_speed - 1.0))
        rows.append({
            "epoch": traffic.epoch,
            "elapsed_s": traffic.elapsed_s,
            "elapsed_min": traffic.elapsed_s / 60.0,
            "traffic_state": traffic.state,
            "traffic_change": traffic.change,
            "center_bins": " ".join(map(str, traffic.center_bins)),
            "peak_tt_multiplier": traffic.peak_multiplier,
            "speed_multiplier": traffic.speed_multiplier,
            "recovery_progress": traffic.recovery_progress,
            "disruption_region_edges": len(traffic.region_edges),
            "disruption_region_length_km": region_distance / 1000.0,
            "disruption_region_speed_kmh": region_speed,
            "affected_edges": len(traffic.affected_edges),
            "affected_length_km": affected_length,
            "network_speed_kmh_at_start_hour": speed,
            "speed_change_from_previous_pct": speed_change,
        })
        previous_speed = speed
    base.network.set_travel_time_multipliers({})
    return rows


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    final = (frame.sort_values("epoch")
             .groupby(["method", "policy"], as_index=False)
             .tail(1).copy())
    summary = final[[
        "method", "policy", "epoch", "cumulative_energy_kwh",
        "cumulative_operating_s", "cumulative_planning_runtime_s",
        "latency_adjusted_energy_kwh", "latency_adjusted_time_s",
        "min_battery_kwh", "returned_to_depot",
    ]].rename(columns={"epoch": "final_epoch"})
    first = (frame.sort_values("epoch")
             .groupby(["method", "policy"], as_index=False)
             .head(1)[["method", "policy", "planning_runtime_s",
                       "compute_wait_aux_energy_kwh"]]
             .rename(columns={
                 "planning_runtime_s": "initial_planning_runtime_s",
                 "compute_wait_aux_energy_kwh": (
                     "initial_compute_wait_aux_energy_kwh"),
             }))
    totals = (frame.groupby(["method", "policy"], as_index=False)
              .agg(total_reassigned_bins=("reassigned_bins", "sum"),
                   total_next_choice_changes=("next_choice_changes", "sum"),
                   replan_attempts=("replan_attempted", "sum"),
                   accepted_replans=("replan_accepted", "sum"),
                   disrupted_edges_traversed=(
                       "disrupted_edges_traversed", "sum"),
                   traversed_edges=("traversed_edges", "sum")))
    summary = summary.merge(first, on=["method", "policy"])
    summary = summary.merge(totals, on=["method", "policy"])
    summary["replanning_runtime_s"] = (
        summary["cumulative_planning_runtime_s"]
        - summary["initial_planning_runtime_s"])
    summary["post_departure_latency_adjusted_time_s"] = (
        summary["cumulative_operating_s"])
    summary["post_departure_latency_adjusted_energy_kwh"] = (
        summary["latency_adjusted_energy_kwh"]
        - summary["initial_compute_wait_aux_energy_kwh"])
    summary["disrupted_edge_share"] = (
        summary["disrupted_edges_traversed"]
        / summary["traversed_edges"].clip(lower=1))
    static_live_energy = (
        summary[summary["policy"] == "static_live"]
        .set_index("method")["cumulative_energy_kwh"])
    static_frozen_energy = (
        summary[summary["policy"] == "static_frozen"]
        .set_index("method")["cumulative_energy_kwh"])
    for reference, values in (
            ("static_live", static_live_energy),
            ("static_frozen", static_frozen_energy)):
        summary[f"adaptive_vs_{reference}_gain_pct"] = [
            (100.0 * (values[method] - energy) / values[method])
            if policy == "adaptive" else 0.0
            for method, policy, energy in zip(
                summary["method"], summary["policy"],
                summary["cumulative_energy_kwh"])
        ]
    summary["adaptive_energy_gain_pct"] = (
        summary["adaptive_vs_static_live_gain_pct"])
    return summary.sort_values(["method", "policy"])


def refresh_output_directory(output_dir: Path) -> pd.DataFrame:
    """Rebuild derived online metrics and figures without rerunning solvers."""

    output_dir = output_dir.resolve()
    epoch_output = output_dir / "online_epochs.csv"
    frame = pd.read_csv(epoch_output)
    frame["epoch_energy_per_served_bin_kwh"] = (
        frame["epoch_energy_kwh"]
        / frame["served_bins_epoch"].clip(lower=1))
    frame["disrupted_edge_share"] = (
        frame["disrupted_edges_traversed"]
        / frame["traversed_edges"].clip(lower=1))
    frame.to_csv(epoch_output, index=False)
    summary = _summary(frame)
    summary.to_csv(output_dir / "online_summary.csv", index=False)
    create_online_figures(
        epoch_output, output_dir / "traffic_trace.csv",
        output_dir / "figures")
    return summary


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = load_table_i_instance(
        args.n_bins, args.vehicles, args.start_hour, args.demand_seed)
    if args.max_simulation_minutes <= 0.0:
        raise ValueError("maximum simulation duration must be positive")
    if args.traffic_step_s <= 0.0:
        raise ValueError("traffic step must be positive")
    if args.replanning_interval_s <= 0.0:
        raise ValueError("replanning interval must be positive")
    ratio = args.replanning_interval_s / args.traffic_step_s
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            "replanning interval must be an integer multiple of the "
            "traffic step")
    steps = int(np.ceil(
        args.max_simulation_minutes * 60.0 / args.traffic_step_s)) + 1
    trace = generate_traffic_trace(
        base, steps, args.traffic_seed,
        args.disruption_fraction, args.disruption_radius_m,
        args.minimum_speed_factor, args.traffic_step_s,
        args.disruption_delay_minutes * 60.0,
        args.degradation_minutes * 60.0,
        args.disruption_hold_minutes * 60.0,
        args.recovery_minutes * 60.0)
    scenario = pd.DataFrame(_scenario_records(base, trace))
    scenario.to_csv(output_dir / "traffic_trace.csv", index=False)
    exact_trace = {
        "traffic_seed": args.traffic_seed,
        "traffic_model": "wall_clock_delayed_gradual_disruption",
        "maximum_simulation_s": args.max_simulation_minutes * 60.0,
        "disruption_fraction": args.disruption_fraction,
        "disruption_radius_m": args.disruption_radius_m,
        "minimum_speed_factor": args.minimum_speed_factor,
        "traffic_step_s": args.traffic_step_s,
        "replanning_interval_s": args.replanning_interval_s,
        "disruption_delay_s": args.disruption_delay_minutes * 60.0,
        "degradation_s": args.degradation_minutes * 60.0,
        "disruption_hold_s": args.disruption_hold_minutes * 60.0,
        "recovery_s": args.recovery_minutes * 60.0,
        "epochs": [
            {
                **asdict(epoch),
                "multipliers": sorted(epoch.multipliers.items()),
            }
            for epoch in trace
        ],
    }
    (output_dir / "traffic_trace.json").write_text(
        json.dumps(exact_trace, indent=2, ensure_ascii=False),
        encoding="utf-8")

    all_rows: list[dict] = []
    epoch_output = output_dir / "online_epochs.csv"
    initial_cache = {}
    if args.initial_plan_cache is not None:
        initial_cache = json.loads(
            args.initial_plan_cache.resolve().read_text(encoding="utf-8"))
    saved_initial_plans: dict[str, dict] = {}
    for method in args.methods:
        base.network.set_travel_time_multipliers(trace[0].multipliers)
        if method in initial_cache:
            cached = initial_cache[method]
            initial_routes = [list(route) for route in cached["routes"]]
            initial_oracle = float(cached["oracle_runtime_s"])
            initial_solver = float(cached["solver_runtime_s"])
            initial_evaluation = evaluate_routes(base, initial_routes)
            if not initial_evaluation.feasible:
                raise RuntimeError(
                    f"cached {method} initial online plan is infeasible")
            cache_source = cached.get(
                "source", _portable_path(args.initial_plan_cache))
        else:
            initial_oracle = prepare_fast_cost_oracle(
                base, adjacent_slots=args.oracle_slots)
            initial_result = _solve(method, base, args.solver_seed, args)
            if not initial_result.evaluation.feasible:
                raise RuntimeError(f"{method} initial online plan is infeasible")
            initial_routes = [list(route) for route in initial_result.routes]
            initial_solver = float(initial_result.runtime_s)
            initial_evaluation = initial_result.evaluation
            cache_source = "computed_in_this_run"
        frozen_stop_paths, frozen_return_paths = _freeze_route_paths(
            base, initial_routes)
        saved_initial_plans[method] = {
            "routes": initial_routes,
            "frozen_stop_paths": [
                {str(index): list(path) for index, path in vehicle.items()}
                for vehicle in frozen_stop_paths
            ],
            "frozen_return_paths": [
                list(path) for path in frozen_return_paths],
            "oracle_runtime_s": initial_oracle,
            "solver_runtime_s": initial_solver,
            "energy_kwh": initial_evaluation.energy_kwh,
            "makespan_s": initial_evaluation.makespan_s,
            "source": cache_source,
        }
        (output_dir / "initial_plans.json").write_text(
            json.dumps(saved_initial_plans, indent=2, ensure_ascii=False),
            encoding="utf-8")
        for policy in POLICIES:
            rows = _simulate_policy(
                base, method, policy, trace, initial_routes,
                initial_solver, initial_oracle,
                frozen_stop_paths, frozen_return_paths, args)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(epoch_output, index=False)
            final = rows[-1]
            print(
                f"DONE {method}/{policy}: "
                f"E={final['cumulative_energy_kwh']:.3f} kWh, "
                f"T={final['cumulative_operating_s']:.1f} s, "
                f"planning={final['cumulative_planning_runtime_s']:.1f} s",
                flush=True,
            )

    frame = pd.DataFrame(all_rows)
    summary = _summary(frame)
    summary.to_csv(output_dir / "online_summary.csv", index=False)
    manifest = {
        "n_bins": args.n_bins,
        "vehicles": args.vehicles,
        "start_hour": args.start_hour,
        "demand_seed": args.demand_seed,
        "traffic_seed": args.traffic_seed,
        "solver_seed": args.solver_seed,
        "traffic_steps_available": steps,
        "max_simulation_minutes": args.max_simulation_minutes,
        "traffic_step_s": args.traffic_step_s,
        "replanning_interval_s": args.replanning_interval_s,
        "traffic_model": "wall_clock_delayed_gradual_disruption",
        "disruption_fraction": args.disruption_fraction,
        "disruption_radius_m": args.disruption_radius_m,
        "minimum_speed_factor": args.minimum_speed_factor,
        "disruption_delay_s": args.disruption_delay_minutes * 60.0,
        "degradation_s": args.degradation_minutes * 60.0,
        "disruption_hold_s": args.disruption_hold_minutes * 60.0,
        "recovery_s": args.recovery_minutes * 60.0,
        "methods": list(args.methods),
        "policies": list(POLICIES),
        "residual_nonempty_rule": (
            "V_k remains nonempty only until physical vehicle k has served "
            "its first bin; already-used vehicles may be empty later"),
        "initial_plan_cache": (
            None if args.initial_plan_cache is None
            else _portable_path(args.initial_plan_cache)),
        "computation_accounting": {
            "planning_runtime_s": "oracle preparation plus solver wall time",
            "latency_adjusted_time_s": (
                "equal to operating makespan; planning is asynchronous"),
            "compute_wait_aux_energy_kwh": (
                "zero; vehicles do not stop while the controller computes"),
        },
        "execution_clock": (
            "vehicles advance continuously on fixed traffic steps; active "
            "edges and services persist across step boundaries"),
        "replanning_projection": (
            "unavoidable active-edge or active-service work is projected to "
            "its downstream decision state for route optimization, while "
            "physical execution itself is never teleported"),
        "incumbent_gate": True,
        "replan_acceptance_tolerance_kwh": (
            args.replan_acceptance_tolerance_kwh),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    if not getattr(args, "skip_figures", False):
        create_online_figures(
            epoch_output, output_dir / "traffic_trace.csv",
            output_dir / "figures")
    base.network.set_travel_time_multipliers({})
    return frame, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bins", type=int, default=84)
    parser.add_argument("--vehicles", type=int, default=10)
    parser.add_argument("--start-hour", type=int, default=18)
    parser.add_argument("--max-simulation-minutes", type=float, default=120.0)
    parser.add_argument("--traffic-step-s", type=float, default=60.0)
    parser.add_argument("--replanning-interval-s", type=float, default=300.0)
    parser.add_argument("--disruption-fraction", type=float, default=0.05)
    parser.add_argument("--disruption-radius-m", type=float, default=500.0)
    parser.add_argument("--minimum-speed-factor", type=float, default=0.35)
    parser.add_argument("--disruption-delay-minutes", type=float, default=10.0)
    parser.add_argument("--degradation-minutes", type=float, default=10.0)
    parser.add_argument("--disruption-hold-minutes", type=float, default=10.0)
    parser.add_argument("--recovery-minutes", type=float, default=25.0)
    parser.add_argument("--demand-seed", type=int, default=7)
    parser.add_argument("--traffic-seed", type=int, default=2026)
    parser.add_argument("--solver-seed", type=int, default=7)
    parser.add_argument(
        "--initial-plan-cache", type=Path,
        help=("optional measured initial routes/timings used to resume a "
              "terminal-policy rerun without repeating epoch zero"))
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--oracle-slots", type=int, default=2)
    parser.add_argument("--ga-population", type=int, default=100)
    parser.add_argument("--ga-generations", type=int, default=500)
    parser.add_argument("--pso-particles", type=int, default=60)
    parser.add_argument("--pso-iterations", type=int, default=200)
    parser.add_argument("--aco-ants", type=int, default=60)
    parser.add_argument("--aco-iterations", type=int, default=150)
    parser.add_argument("--max-rounds", type=int, default=32)
    parser.add_argument("--assignment-rounds", type=int, default=24)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--improvement-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--replan-acceptance-tolerance-kwh", type=float, default=1e-6,
        help=("minimum exact residual-energy reduction required to replace "
              "the incumbent route"))
    parser.add_argument("--exact-global-limit", type=int, default=14)
    parser.add_argument("--maximum-exact-trellis-bins", type=int, default=84)
    parser.add_argument(
        "--hypercube-radii", type=int, nargs="+",
        default=[1, 2, 4, 8, 12, 16, 24, 32])
    parser.add_argument("--stagnation-patience", type=int, default=8)
    parser.add_argument(
        "--canonicalize-vehicle-symmetry",
        dest="canonicalize_vehicle_symmetry", action="store_true",
        default=True)
    parser.add_argument(
        "--no-canonicalize-vehicle-symmetry",
        dest="canonicalize_vehicle_symmetry", action="store_false")
    parser.add_argument(
        "--symmetry-dual-path", dest="symmetry_dual_path",
        action="store_true", default=True)
    parser.add_argument(
        "--no-symmetry-dual-path", dest="symmetry_dual_path",
        action="store_false")
    parser.add_argument("--vehicle-gauss-seidel", action="store_true")
    parser.add_argument("--certified-route-messages", action="store_true")
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="skip per-seed figures when a batch-level plot will be produced")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PACKAGE_ROOT / "results" / "online" / "seed2026")
    return parser


def main(arguments: list[str] | None = None) -> None:
    run(build_parser().parse_args(arguments))


if __name__ == "__main__":
    main()
