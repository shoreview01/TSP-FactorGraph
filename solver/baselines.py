"""NN, GA, PSO, ACO, and exact small-N MILP on one common evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
import time

import numpy as np
from numba import njit
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from simul.energy import service_energy_kwh

from .model import (PaperInstance, PlanEvaluation, evaluate_routes,
                    evaluate_vehicle_route, transition_cost, transition_costs)


@dataclass
class SolverResult:
    method: str
    routes: list[list[int]]
    evaluation: PlanEvaluation
    runtime_s: float
    metadata: dict = field(default_factory=dict)


@njit(cache=False)
def _segment_tables_tensor(
    order: np.ndarray,
    demand_kg: np.ndarray,
    vehicle_node_index: np.ndarray,
    depot_index: int,
    initial_payload_kg: np.ndarray,
    battery_kwh: np.ndarray,
    energy_tensor: np.ndarray,
    time_tensor: np.ndarray,
    capacity_kg: float,
    reserve_kwh: float,
    payload_resolution_kg: float,
    service_energy_kwh_: float,
    service_time_s: float,
    start_offset_s: float,
    bin_node_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba decoder for exact time/load-dependent contiguous tour segments."""

    n = len(order)
    vehicles = len(vehicle_node_index)
    cost = np.full((vehicles, n + 1, n + 1), np.inf)
    elapsed_table = np.full((vehicles, n + 1, n + 1), np.inf)
    slot_count = energy_tensor.shape[0]
    payload_count = energy_tensor.shape[1]
    for vehicle in range(vehicles):
        initial_payload = initial_payload_kg[vehicle]
        payload_index = int(np.rint(initial_payload / payload_resolution_kg))
        payload_index = min(max(payload_index, 0), payload_count - 1)
        empty_energy = energy_tensor[
            0, payload_index, vehicle_node_index[vehicle], depot_index]
        empty_time = time_tensor[
            0, payload_index, vehicle_node_index[vehicle], depot_index]
        if np.isfinite(empty_energy) and empty_energy + reserve_kwh <= battery_kwh[vehicle] + 1e-9:
            for position in range(n + 1):
                cost[vehicle, position, position] = empty_energy
                elapsed_table[vehicle, position, position] = empty_time

        for begin in range(n):
            node = vehicle_node_index[vehicle]
            payload = initial_payload
            energy = 0.0
            elapsed = 0.0
            for end in range(begin, n):
                index = order[end]
                demand = demand_kg[index]
                if payload + demand > capacity_kg + 1e-9:
                    break
                slot = int((start_offset_s + elapsed) // 3600.0)
                if slot >= slot_count:
                    break
                payload_index = int(np.rint(payload / payload_resolution_kg))
                payload_index = min(max(payload_index, 0), payload_count - 1)
                target = bin_node_index[index]
                edge_energy = energy_tensor[
                    slot, payload_index, node, target]
                edge_time = time_tensor[
                    slot, payload_index, node, target]
                if not np.isfinite(edge_energy) or not np.isfinite(edge_time):
                    break
                energy += edge_energy + service_energy_kwh_
                elapsed += edge_time + service_time_s
                payload += demand
                node = target

                return_slot = int((start_offset_s + elapsed) // 3600.0)
                if return_slot >= slot_count:
                    continue
                payload_index = int(np.rint(payload / payload_resolution_kg))
                payload_index = min(max(payload_index, 0), payload_count - 1)
                return_energy = energy_tensor[
                    return_slot, payload_index, node, depot_index]
                return_time = time_tensor[
                    return_slot, payload_index, node, depot_index]
                total_energy = energy + return_energy
                if (np.isfinite(total_energy) and np.isfinite(return_time)
                        and total_energy + reserve_kwh
                        <= battery_kwh[vehicle] + 1e-9):
                    cost[vehicle, begin, end + 1] = total_energy
                    elapsed_table[vehicle, begin, end + 1] = (
                        elapsed + return_time)
    return cost, elapsed_table


@njit(cache=False)
def _split_boundaries(segment_cost: np.ndarray,
                      require_nonempty: np.ndarray) -> np.ndarray:
    """Optimal contiguous split with state-dependent nonempty vehicles."""

    vehicles = segment_cost.shape[0]
    n = segment_cost.shape[1] - 1
    dp = np.full((vehicles + 1, n + 1), np.inf)
    previous = np.full((vehicles + 1, n + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    for used in range(1, vehicles + 1):
        vehicle = used - 1
        for end in range(n + 1):
            for begin in range(end + 1):
                if require_nonempty[vehicle] and begin == end:
                    continue
                if not np.isfinite(dp[used - 1, begin]):
                    continue
                candidate = (dp[used - 1, begin]
                             + segment_cost[vehicle, begin, end])
                if candidate < dp[used, end] - 1e-12:
                    dp[used, end] = candidate
                    previous[used, end] = begin
    boundaries = np.full(vehicles + 1, -1, dtype=np.int64)
    if not np.isfinite(dp[vehicles, n]):
        return boundaries
    boundaries[vehicles] = n
    end = n
    for used in range(vehicles, 0, -1):
        begin = int(previous[used, end])
        if begin < 0:
            return np.full(vehicles + 1, -1, dtype=np.int64)
        boundaries[used - 1] = begin
        end = begin
    return boundaries


def solve_nn(instance: PaperInstance) -> SolverResult:
    """State-aware constrained nearest-neighbor fleet construction."""

    started = time.perf_counter()
    remaining = set(range(instance.n_bins))
    routes = [[] for _ in range(instance.vehicles)]
    nodes = [state.node for state in instance.vehicle_states]
    payload = np.asarray([state.payload_kg for state in instance.vehicle_states], float)
    energy = np.zeros(instance.vehicles)
    elapsed = np.zeros(instance.vehicles)
    service_energy = service_energy_kwh(instance.network.params)
    service_time = float(instance.network.params.service_time_s)
    empty_vehicles = set(np.flatnonzero(
        instance.nonempty_required).tolist())

    while remaining:
        best = None
        candidate_vehicles = (sorted(empty_vehicles) if empty_vehicles
                              else range(instance.vehicles))
        for vehicle in candidate_vehicles:
            state = instance.vehicle_states[vehicle]
            feasible = [
                index for index in remaining
                if payload[vehicle] + instance.demand_kg[index]
                <= instance.capacity_kg + 1e-9
            ]
            if not feasible:
                continue
            transitions = transition_costs(instance.network,
                nodes[vehicle], [instance.bin_nodes[index] for index in feasible],
                instance.start_time_s + elapsed[vehicle], payload[vehicle])
            for index in feasible:
                edge_energy, edge_time = transitions[instance.bin_nodes[index]]
                new_payload = payload[vehicle] + instance.demand_kg[index]
                try:
                    return_energy, _ = transition_cost(instance.network,
                        instance.bin_nodes[index], instance.depot_node,
                        instance.start_time_s + elapsed[vehicle]
                        + edge_time + service_time, new_payload)
                except ValueError:
                    continue
                incremental_energy = float(edge_energy) + service_energy
                if (energy[vehicle] + incremental_energy + return_energy
                        + instance.reserve_kwh > state.battery_kwh + 1e-9):
                    continue
                key = (incremental_energy, -instance.urgency[index],
                       vehicle, index, float(edge_time) + service_time)
                if best is None or key < best:
                    best = key
        if best is None:
            raise RuntimeError("NN found no constraint-feasible exact cover")
        incremental_energy, _, vehicle, index, incremental_time = best
        routes[vehicle].append(index)
        nodes[vehicle] = instance.bin_nodes[index]
        payload[vehicle] += instance.demand_kg[index]
        energy[vehicle] += incremental_energy
        elapsed[vehicle] += incremental_time
        remaining.remove(index)
        empty_vehicles.discard(vehicle)

    evaluation = evaluate_routes(instance, routes)
    if not evaluation.feasible:
        raise AssertionError("NN returned an infeasible plan")
    return SolverResult("nn", routes, evaluation,
                        time.perf_counter() - started)


def _segment_tables(instance: PaperInstance, order: list[int]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Physical energy for each vehicle and contiguous giant-tour segment."""

    tensor_getter = getattr(instance.network, "relevant_cost_tensor", None)
    if tensor_getter is not None:
        nodes, node_index, energy_tensor, time_tensor = tensor_getter(
            instance.start_time_s,
            int(getattr(instance.network, "active_relevant_slot_count", 1)))
        del nodes
        required_nodes = {
            instance.depot_node, *instance.bin_nodes,
            *(state.node for state in instance.vehicle_states),
        }
        if required_nodes <= set(node_index):
            return _segment_tables_tensor(
                np.asarray(order, dtype=np.int64),
                np.asarray(instance.demand_kg, dtype=np.float64),
                np.asarray([node_index[state.node]
                            for state in instance.vehicle_states], dtype=np.int64),
                int(node_index[instance.depot_node]),
                np.asarray([state.payload_kg
                            for state in instance.vehicle_states], dtype=np.float64),
                np.asarray([state.battery_kwh
                            for state in instance.vehicle_states], dtype=np.float64),
                np.asarray(energy_tensor, dtype=np.float64),
                np.asarray(time_tensor, dtype=np.float64),
                float(instance.capacity_kg),
                float(instance.reserve_kwh),
                float(instance.network.params.payload_state_kg),
                float(service_energy_kwh(instance.network.params)),
                float(instance.network.params.service_time_s),
                float(instance.start_time_s % 3600.0),
                np.asarray([node_index[node] for node in instance.bin_nodes],
                           dtype=np.int64),
            )

    n, vehicles = len(order), instance.vehicles
    cost = np.full((vehicles, n + 1, n + 1), np.inf)
    elapsed_table = np.full_like(cost, np.inf)
    service_energy = service_energy_kwh(instance.network.params)
    service_time = float(instance.network.params.service_time_s)
    for vehicle, state in enumerate(instance.vehicle_states):
        empty = evaluate_vehicle_route(instance, vehicle, [])
        if empty.feasible:
            for position in range(n + 1):
                cost[vehicle, position, position] = empty.energy_kwh
                elapsed_table[vehicle, position, position] = empty.elapsed_s
        for begin in range(n):
            node = state.node
            payload = float(state.payload_kg)
            energy = 0.0
            elapsed = 0.0
            for end in range(begin, n):
                index = order[end]
                demand = float(instance.demand_kg[index])
                if payload + demand > instance.capacity_kg + 1e-9:
                    break
                try:
                    edge_energy, edge_time = transition_cost(instance.network,
                        node, instance.bin_nodes[index],
                        instance.start_time_s + elapsed, payload)
                except ValueError:
                    break
                energy += float(edge_energy) + service_energy
                elapsed += float(edge_time) + service_time
                payload += demand
                node = instance.bin_nodes[index]
                try:
                    return_energy, return_time = transition_cost(instance.network,
                        node, instance.depot_node,
                        instance.start_time_s + elapsed, payload)
                except ValueError:
                    continue
                total_energy = energy + float(return_energy)
                total_elapsed = elapsed + float(return_time)
                if total_energy + instance.reserve_kwh <= state.battery_kwh + 1e-9:
                    cost[vehicle, begin, end + 1] = total_energy
                    elapsed_table[vehicle, begin, end + 1] = total_elapsed
    return cost, elapsed_table


def split_giant_tour(instance: PaperInstance,
                     permutation: np.ndarray | list[int]
                     ) -> list[list[int]] | None:
    """Optimal physical-energy split of a permutation across exactly K EVs."""

    order = [int(index) for index in permutation]
    if sorted(order) != list(range(instance.n_bins)):
        raise ValueError("permutation must contain every bin exactly once")
    n, vehicles = len(order), instance.vehicles
    segment_cost, _ = _segment_tables(instance, order)
    boundaries = _split_boundaries(
        segment_cost, np.asarray(instance.nonempty_required, dtype=np.bool_))
    if boundaries[0] < 0:
        return None
    return [order[int(boundaries[vehicle]):int(boundaries[vehicle + 1])]
            for vehicle in range(vehicles)]


def _score_order(instance: PaperInstance, permutation: np.ndarray,
                 cache: dict[tuple[int, ...], tuple[float, list[list[int]] | None]]
                 ) -> tuple[float, list[list[int]] | None]:
    key = tuple(int(index) for index in permutation)
    if key in cache:
        return cache[key]
    routes = split_giant_tour(instance, permutation)
    if routes is None:
        cache[key] = (np.inf, None)
        return cache[key]
    evaluation = evaluate_routes(instance, routes)
    cache[key] = (evaluation.energy_kwh if evaluation.feasible else np.inf,
                  routes if evaluation.feasible else None)
    return cache[key]


def solve_ga(instance: PaperInstance, seed: int = 0, *,
             population: int = 100, generations: int = 500,
             mutation_rate: float = 0.20,
             crossover_rate: float = 0.90) -> SolverResult:
    """Manuscript GA: tournament, order crossover, and swap mutation."""

    started = time.perf_counter()
    if population < 2:
        raise ValueError("GA population must be at least two")
    rng = np.random.default_rng(seed)
    n = instance.n_bins
    if n < 2:
        routes = split_giant_tour(instance, np.arange(n))
        if routes is None:
            raise RuntimeError("GA could not decode the singleton instance")
        return SolverResult("ga", routes, evaluate_routes(instance, routes),
                            time.perf_counter() - started)
    cache: dict[tuple[int, ...], tuple[float, list[list[int]] | None]] = {}
    nn = solve_nn(instance)
    seed_tour = np.asarray([index for route in nn.routes for index in route], int)

    def score(chromosome: np.ndarray) -> float:
        return _score_order(instance, chromosome, cache)[0]

    def tournament(population_: list[np.ndarray], costs: np.ndarray) -> np.ndarray:
        choices = rng.integers(0, len(population_), size=3)
        return population_[int(choices[np.argmin(costs[choices])])]

    def crossover(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        left, right = sorted(rng.choice(n, 2, replace=False))
        child = np.full(n, -1, int)
        child[left:right + 1] = first[left:right + 1]
        used = set(child[left:right + 1].tolist())
        fill = [int(gene) for gene in np.r_[second[right + 1:], second[:right + 1]]
                if int(gene) not in used]
        positions = list(range(right + 1, n)) + list(range(left))
        child[positions] = fill
        return child

    def mutate(chromosome: np.ndarray) -> np.ndarray:
        child = chromosome.copy()
        if rng.random() < mutation_rate:
            first, second = rng.choice(n, 2, replace=False)
            child[first], child[second] = child[second], child[first]
        return child

    population_list = [seed_tour.copy()]
    population_list.extend(
        rng.permutation(n) for _ in range(population - 1))
    for _ in range(generations):
        costs = np.asarray([score(chromosome) for chromosome in population_list])
        elite = population_list[int(np.argmin(costs))].copy()
        next_population = [elite]
        while len(next_population) < population:
            first = tournament(population_list, costs)
            second = tournament(population_list, costs)
            child = (crossover(first, second)
                     if rng.random() < crossover_rate else first.copy())
            next_population.append(mutate(child))
        population_list = next_population
    best = min(population_list, key=score)
    _, routes = _score_order(instance, best, cache)
    if routes is None:
        raise RuntimeError("GA found no feasible exact cover")
    evaluation = evaluate_routes(instance, routes)
    return SolverResult(
        "ga", routes, evaluation, time.perf_counter() - started,
        {"population": population, "generations": generations,
         "selection": "tournament", "crossover": "order",
         "mutation": "swap"})


def solve_pso(instance: PaperInstance, seed: int = 0, *,
              particles: int = 60, iterations: int = 200,
              inertia: float = 0.72, cognitive: float = 1.49,
              social: float = 1.49) -> SolverResult:
    """Random-key PSO with the common physical split decoder."""

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = instance.n_bins
    if particles < 2:
        raise ValueError("PSO needs at least two particles")
    nn = solve_nn(instance)
    seed_tour = np.asarray([index for route in nn.routes for index in route], int)

    def keys(permutation: np.ndarray) -> np.ndarray:
        result = np.empty(n, float)
        result[permutation] = np.linspace(0.0, 1.0, n, endpoint=False)
        return result

    def normalize(position: np.ndarray) -> np.ndarray:
        return keys(np.argsort(position, kind="stable"))

    positions = rng.random((particles, n))
    positions[0] = keys(seed_tour)
    velocities = rng.normal(0.0, 0.1, size=(particles, n))
    cache: dict[tuple[int, ...], tuple[float, list[list[int]] | None]] = {}

    def evaluate(position: np.ndarray):
        return _score_order(instance, np.argsort(position, kind="stable"), cache)

    personal_positions = positions.copy()
    personal_costs = np.asarray([evaluate(position)[0] for position in positions])
    best_index = int(np.argmin(personal_costs))
    global_position = personal_positions[best_index].copy()
    global_cost = float(personal_costs[best_index])
    for _ in range(iterations):
        r1 = rng.random((particles, n))
        r2 = rng.random((particles, n))
        velocities = (inertia * velocities
                      + cognitive * r1 * (personal_positions - positions)
                      + social * r2 * (global_position - positions))
        velocities = np.clip(velocities, -0.25, 0.25)
        positions = np.asarray([
            normalize(np.clip(position + velocity, 0.0, 1.0))
            for position, velocity in zip(positions, velocities)
        ])
        costs = np.asarray([evaluate(position)[0] for position in positions])
        improved = costs < personal_costs
        personal_positions[improved] = positions[improved]
        personal_costs[improved] = costs[improved]
        candidate = int(np.argmin(personal_costs))
        if personal_costs[candidate] < global_cost:
            global_cost = float(personal_costs[candidate])
            global_position = personal_positions[candidate].copy()
    _, routes = evaluate(global_position)
    if routes is None:
        raise RuntimeError("PSO found no feasible exact cover")
    return SolverResult(
        "pso", routes, evaluate_routes(instance, routes),
        time.perf_counter() - started,
        {"particles": particles, "iterations": iterations})


def _surrogate_matrix(instance: PaperInstance) -> np.ndarray:
    nodes = [instance.depot_node, *instance.bin_nodes]
    matrix = np.zeros((len(nodes), len(nodes)))
    payload = 0.5 * instance.capacity_kg
    for source_index, source in enumerate(nodes):
        targets = [target for target_index, target in enumerate(nodes)
                   if target_index != source_index]
        transitions = transition_costs(instance.network,
            source, targets, instance.start_time_s, payload)
        for target_index, target in enumerate(nodes):
            if target_index != source_index:
                matrix[source_index, target_index] = transitions[target][0]
    return matrix


def solve_aco(instance: PaperInstance, seed: int = 0, *, ants: int = 60,
              iterations: int = 150, alpha: float = 1.0,
              beta: float = 2.0, evaporation: float = 0.35) -> SolverResult:
    """Permutation ACO; selection and deposits use physical fleet energy."""

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = instance.n_bins
    matrix = _surrogate_matrix(instance)
    eta_depot = 1.0 / np.maximum(matrix[0, 1:], 1e-12)
    eta_pair = 1.0 / np.maximum(matrix[1:, 1:], 1e-12)
    np.fill_diagonal(eta_pair, 0.0)
    pheromone = np.ones((n + 1, n))
    cache: dict[tuple[int, ...], tuple[float, list[list[int]] | None]] = {}
    best_cost = np.inf
    best_routes = None

    for _ in range(iterations):
        colony = []
        for _ant in range(ants):
            remaining = set(range(n))
            order: list[int] = []
            last = n
            while remaining:
                candidates = np.asarray(sorted(remaining), int)
                heuristic = (eta_depot[candidates] if last == n
                             else eta_pair[last, candidates])
                weight = pheromone[last, candidates] ** alpha * heuristic ** beta
                if not np.isfinite(weight).all() or float(weight.sum()) <= 0:
                    probability = np.full(len(candidates), 1.0 / len(candidates))
                else:
                    probability = weight / weight.sum()
                selected = int(rng.choice(candidates, p=probability))
                order.append(selected)
                remaining.remove(selected)
                last = selected
            permutation = np.asarray(order, int)
            cost, routes = _score_order(instance, permutation, cache)
            colony.append((cost, permutation, routes))
            if cost < best_cost:
                best_cost, best_routes = cost, routes
        pheromone *= 1.0 - evaporation
        elite_count = max(1, ants // 10)
        for cost, order, _ in sorted(colony, key=lambda row: row[0])[:elite_count]:
            if not np.isfinite(cost):
                continue
            deposit = 1.0 / max(float(cost), 1e-12)
            last = n
            for selected in order:
                pheromone[last, selected] += deposit
                last = int(selected)
    if best_routes is None:
        raise RuntimeError("ACO found no feasible exact cover")
    return SolverResult(
        "aco", best_routes, evaluate_routes(instance, best_routes),
        time.perf_counter() - started,
        {"ants": ants, "iterations": iterations})


def solve_milp(instance: PaperInstance, *, max_bins: int = 9) -> SolverResult:
    """Exact vehicle-indexed ordered-route set-partitioning MILP."""

    started = time.perf_counter()
    n = instance.n_bins
    if n > max_bins:
        raise ValueError(f"exact MILP is limited to N<={max_bins}, got {n}")
    columns: list[tuple[int, tuple[int, ...], float]] = []
    for vehicle in range(instance.vehicles):
        for size in range(1, n + 1):
            for route in permutations(range(n), size):
                evaluation = evaluate_vehicle_route(instance, vehicle, route)
                if evaluation.feasible:
                    columns.append((vehicle, route, evaluation.energy_kwh))
    if not columns:
        raise RuntimeError("MILP has no feasible route columns")

    matrix = np.zeros((n + instance.vehicles, len(columns)))
    for column, (vehicle, route, _) in enumerate(columns):
        matrix[list(route), column] = 1.0
        matrix[n + vehicle, column] = 1.0
    constraint = LinearConstraint(
        csc_matrix(matrix),
        np.r_[np.ones(n), np.ones(instance.vehicles)],
        np.r_[np.ones(n), np.ones(instance.vehicles)],
    )
    result = milp(
        c=np.asarray([cost for _, _, cost in columns]),
        integrality=np.ones(len(columns)), bounds=Bounds(0, 1),
        constraints=constraint,
        options={"disp": False, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: {result.message}")
    routes = [[] for _ in range(instance.vehicles)]
    for column, selected in enumerate(result.x):
        if selected > 0.5:
            vehicle, route, _ = columns[column]
            routes[vehicle] = list(route)
    evaluation = evaluate_routes(instance, routes)
    if not evaluation.feasible:
        raise AssertionError("MILP decoder returned an infeasible plan")
    return SolverResult(
        "milp", routes, evaluation, time.perf_counter() - started,
        {"columns": len(columns), "solver_message": str(result.message)})


def solve_baseline(instance: PaperInstance, method: str, seed: int = 0,
                   **kwargs) -> SolverResult:
    dispatch = {
        "nn": solve_nn,
        "ga": solve_ga,
        "pso": solve_pso,
        "aco": solve_aco,
        "milp": solve_milp,
    }
    if method not in dispatch:
        raise ValueError(f"unknown baseline: {method}")
    if method == "nn":
        return solve_nn(instance)
    return dispatch[method](instance, seed=seed, **kwargs) if method != "milp" \
        else solve_milp(instance, **kwargs)
