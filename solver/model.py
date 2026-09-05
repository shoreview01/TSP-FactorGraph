"""Shared Section II instance model and exact fleet-plan evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from simul.energy import EVParameters, OperationalEnergyNetwork, service_energy_kwh


ROOT = Path(__file__).resolve().parents[1]


def transition_cost(network, source: str, target: str,
                    sim_time_s: float, payload_kg: float) -> tuple[float, float]:
    """Energy/time adapter supporting operational and lightweight test networks."""

    if hasattr(network, "cost"):
        return network.cost(source, target, sim_time_s, payload_kg)
    _, energy, elapsed = network.shortest_path(
        source, target, sim_time_s, payload_kg)
    return float(energy), float(elapsed)


def transition_costs(network, source: str, targets,
                     sim_time_s: float, payload_kg: float
                     ) -> dict[str, tuple[float, float]]:
    """One-to-many energy/time adapter for operational and test networks."""

    target_nodes = [str(target) for target in targets]
    if hasattr(network, "costs"):
        return network.costs(source, target_nodes, sim_time_s, payload_kg)
    paths = network.shortest_paths(
        source, target_nodes, sim_time_s, payload_kg)
    return {target: (float(paths[target][1]), float(paths[target][2]))
            for target in target_nodes}


@dataclass(frozen=True)
class VehicleState:
    """Measured vehicle state y_k(t_tau) from Eq. (2)."""

    node: str
    payload_kg: float
    battery_kwh: float


@dataclass(frozen=True)
class PaperInstance:
    network: OperationalEnergyNetwork
    depot_node: str
    bin_nodes: tuple[str, ...]
    demand_kg: np.ndarray
    urgency: np.ndarray
    vehicle_states: tuple[VehicleState, ...]
    start_time_s: float
    capacity_kg: float
    maximum_battery_kwh: float
    reserve_kwh: float
    seed: int = 7
    require_nonempty: tuple[bool, ...] | None = None

    @property
    def vehicles(self) -> int:
        return len(self.vehicle_states)

    @property
    def n_bins(self) -> int:
        return len(self.bin_nodes)

    @property
    def nonempty_required(self) -> np.ndarray:
        """Vehicles that still must receive a bin in the current horizon."""

        if self.require_nonempty is None:
            return np.ones(self.vehicles, dtype=bool)
        required = np.asarray(self.require_nonempty, dtype=bool)
        if required.shape != (self.vehicles,):
            raise ValueError(
                "require_nonempty must contain one flag per vehicle")
        return required


@dataclass(frozen=True)
class VehicleRouteEvaluation:
    energy_kwh: float
    elapsed_s: float
    final_payload_kg: float
    capacity_feasible: bool
    battery_feasible: bool
    return_feasible: bool

    @property
    def feasible(self) -> bool:
        return (self.capacity_feasible and self.battery_feasible
                and self.return_feasible)


@dataclass(frozen=True)
class PlanEvaluation:
    energy_kwh: float
    makespan_s: float
    served_bins: int
    served_demand_kg: float
    exact_coverage: bool
    capacity_feasible: bool
    battery_feasible: bool
    return_feasible: bool
    vehicle_energy_kwh: tuple[float, ...]
    vehicle_time_s: tuple[float, ...]

    @property
    def feasible(self) -> bool:
        return (self.exact_coverage and self.capacity_feasible
                and self.battery_feasible and self.return_feasible)


@dataclass(frozen=True)
class PlanEnergyBreakdown:
    """Auditable manuscript energy components of a complete fleet plan."""

    rolling_wheel_kwh: float
    grade_wheel_kwh: float
    aerodynamic_wheel_kwh: float
    propulsion_kwh: float
    recuperated_kwh: float
    recuperation_curtailed_kwh: float
    travel_auxiliary_kwh: float
    stop_go_kwh: float
    service_auxiliary_kwh: float
    lifting_compaction_kwh: float
    total_kwh: float
    congestion_penalty: float
    distance_km: float
    travel_time_s: float


def load_table_i_instance(
    n_bins: int = 84,
    vehicles: int = 10,
    start_hour: int = 8,
    seed: int = 7,
    *,
    bins_csv: Path | None = None,
    capacity_kg: float = 2000.0,
    maximum_battery_kwh: float = 180.0,
    reserve_kwh: float = 15.0,
    target_total_demand_kg: float | None = None,
) -> PaperInstance:
    """Load a Table-I instance from the authoritative operational inputs."""

    params = EVParameters(
        curb_mass_kg=7000.0,
        payload_capacity_kg=capacity_kg,
        rolling_resistance=0.010,
        drag_area_m2=5.5,
        drag_coefficient=0.60,
        drivetrain_efficiency=0.90,
        regen_efficiency=0.60,
        auxiliary_power_kw=3.0,
        stop_go_tolerance_fraction=0.25,
        stop_go_energy_kwh_per_kg=4.93e-6,
        service_auxiliary_power_kw=2.0,
        service_time_s=120.0,
        lifting_compaction_energy_kwh=0.12,
        payload_state_kg=100.0,
    )
    network = OperationalEnergyNetwork(
        ROOT / "simul/seongbuk_buffer_elevation.net.xml",
        ROOT / "simul/sumo_hourly_cost_operational_v3.csv",
        params,
    )
    path = bins_csv or ROOT / "simul/seongbuk_bins_84.csv"
    bins = pd.read_csv(path, dtype={"SUMO_EDGE_ID": str}, encoding="utf-8-sig")
    bins = bins[
        bins["ROUND_TRIP_FEASIBLE"].astype(str).str.lower() == "true"
    ].head(n_bins)
    if len(bins) != n_bins:
        raise ValueError(f"requested {n_bins} bins but only {len(bins)} are available")

    depot = pd.read_csv(
        ROOT / "simul/operational_depot_v3.csv",
        dtype={"SUMO_EDGE_ID": str}, encoding="utf-8-sig")
    depot_edge = str(depot.iloc[0]["SUMO_EDGE_ID"])
    depot_node = str(network.by_id.loc[depot_edge, "TO_NODE"])
    bin_nodes = tuple(
        str(network.by_id.loc[str(edge), "TO_NODE"])
        for edge in bins["SUMO_EDGE_ID"]
    )
    network.register_relevant_targets((depot_node, *bin_nodes))

    rng = np.random.default_rng(seed)
    demand = rng.integers(50, 251, size=n_bins).astype(float)
    if target_total_demand_kg is not None:
        target = float(target_total_demand_kg)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("target total demand must be positive and finite")
        if abs(target - round(target)) > 1e-9:
            raise ValueError("target total demand must use whole kilograms")
        minimum = 50.0
        minimum_total = minimum * n_bins
        if target < minimum_total - 1e-9:
            raise ValueError(
                "target total demand is below the 50 kg per-bin minimum")
        excess = demand - minimum
        excess_total = float(excess.sum())
        target_excess = target - minimum_total
        if excess_total <= 0.0 and target_excess > 0.0:
            raise ValueError("cannot rescale a zero-excess demand realization")
        scaled = (np.full(n_bins, minimum)
                  if target_excess == 0.0 else
                  minimum + excess * (target_excess / excess_total))
        rounded = np.floor(scaled).astype(float)
        residual = int(round(target - float(rounded.sum())))
        if residual < 0 or residual > n_bins:
            raise AssertionError("largest-remainder demand scaling failed")
        if residual:
            fractions = scaled - rounded
            order = np.argsort(-fractions, kind="stable")
            rounded[order[:residual]] += 1.0
        demand = rounded
        if abs(float(demand.sum()) - target) > 1e-9:
            raise AssertionError("scaled demand does not match requested total")
    urgency = rng.uniform(0.5, 1.0, size=n_bins) * demand
    if float(demand.sum()) > vehicles * capacity_kg + 1e-9:
        raise ValueError("fleet capacity is below total demand")
    states = tuple(
        VehicleState(depot_node, 0.0, maximum_battery_kwh)
        for _ in range(vehicles)
    )
    return PaperInstance(
        network=network,
        depot_node=depot_node,
        bin_nodes=bin_nodes,
        demand_kg=demand,
        urgency=urgency,
        vehicle_states=states,
        start_time_s=float(start_hour * 3600),
        capacity_kg=capacity_kg,
        maximum_battery_kwh=maximum_battery_kwh,
        reserve_kwh=reserve_kwh,
        seed=seed,
    )


def evaluate_vehicle_route(
    instance: PaperInstance,
    vehicle: int,
    route: list[int] | tuple[int, ...],
) -> VehicleRouteEvaluation:
    """Evaluate E_k(t_tau, r_k) with Eqs. (10)-(13)."""

    state = instance.vehicle_states[vehicle]
    node = state.node
    payload = float(state.payload_kg)
    energy = 0.0
    elapsed = 0.0
    capacity_ok = payload <= instance.capacity_kg + 1e-9
    battery_ok = state.battery_kwh <= instance.maximum_battery_kwh + 1e-9
    return_ok = True
    service_energy = service_energy_kwh(instance.network.params)
    service_time = float(instance.network.params.service_time_s)

    for raw_index in route:
        index = int(raw_index)
        if not 0 <= index < instance.n_bins:
            return VehicleRouteEvaluation(
                np.inf, np.inf, payload, False, False, False)
        try:
            de, dt = transition_cost(instance.network,
                node, instance.bin_nodes[index],
                instance.start_time_s + elapsed, payload)
        except ValueError:
            return VehicleRouteEvaluation(
                np.inf, np.inf, payload, False, False, False)
        energy += float(de) + service_energy
        elapsed += float(dt) + service_time
        payload += float(instance.demand_kg[index])
        capacity_ok &= payload <= instance.capacity_kg + 1e-9
        battery_ok &= energy + instance.reserve_kwh <= state.battery_kwh + 1e-9
        node = instance.bin_nodes[index]

    try:
        de, dt = transition_cost(instance.network,
            node, instance.depot_node,
            instance.start_time_s + elapsed, payload)
        energy += float(de)
        elapsed += float(dt)
    except ValueError:
        return_ok = False
        energy = np.inf
        elapsed = np.inf
    battery_ok &= energy + instance.reserve_kwh <= state.battery_kwh + 1e-9
    return VehicleRouteEvaluation(
        float(energy), float(elapsed), payload,
        bool(capacity_ok), bool(battery_ok), bool(return_ok))


def evaluate_routes(instance: PaperInstance,
                    routes: list[list[int]]) -> PlanEvaluation:
    """Common evaluator used by every solver, including exact coverage."""

    if len(routes) != instance.vehicles:
        raise ValueError("one route is required for every vehicle")
    flattened = [int(index) for route in routes for index in route]
    in_range = all(0 <= index < instance.n_bins for index in flattened)
    exact_coverage = (in_range and len(flattened) == instance.n_bins
                      and sorted(flattened) == list(range(instance.n_bins)))

    vehicle_evaluations = [
        evaluate_vehicle_route(instance, vehicle, route)
        for vehicle, route in enumerate(routes)
    ]
    served = sorted(set(index for index in flattened
                        if 0 <= index < instance.n_bins))
    energies = tuple(result.energy_kwh for result in vehicle_evaluations)
    times = tuple(result.elapsed_s for result in vehicle_evaluations)
    return PlanEvaluation(
        energy_kwh=float(sum(energies)),
        makespan_s=float(max(times, default=0.0)),
        served_bins=len(served),
        served_demand_kg=(float(instance.demand_kg[served].sum())
                          if served else 0.0),
        exact_coverage=bool(exact_coverage),
        capacity_feasible=all(result.capacity_feasible
                              for result in vehicle_evaluations),
        battery_feasible=all(result.battery_feasible
                             for result in vehicle_evaluations),
        return_feasible=all(result.return_feasible
                            for result in vehicle_evaluations),
        vehicle_energy_kwh=energies,
        vehicle_time_s=times,
    )


def evaluate_routes_energy_breakdown(
    instance: PaperInstance,
    routes: list[list[int]],
) -> PlanEnergyBreakdown:
    """Reconstruct every chosen path and decompose its battery consumption."""

    network = instance.network
    if not (hasattr(network, "traversal_breakdown")
            and hasattr(network, "edge_data")):
        raise TypeError("energy breakdown requires OperationalEnergyNetwork")
    totals = {
        "rolling_wheel_kwh": 0.0,
        "grade_wheel_kwh": 0.0,
        "aerodynamic_wheel_kwh": 0.0,
        "propulsion_kwh": 0.0,
        "recuperated_kwh": 0.0,
        "recuperation_curtailed_kwh": 0.0,
        "travel_auxiliary_kwh": 0.0,
        "stop_go_kwh": 0.0,
        "congestion_penalty": 0.0,
        "distance_km": 0.0,
        "travel_time_s": 0.0,
        "edge_net_kwh": 0.0,
    }
    service_count = 0
    for vehicle, route in enumerate(routes):
        state = instance.vehicle_states[vehicle]
        node = state.node
        payload = float(state.payload_kg)
        elapsed = 0.0
        targets = [instance.bin_nodes[int(index)] for index in route]
        targets.append(instance.depot_node)
        for position, target in enumerate(targets):
            departure = instance.start_time_s + elapsed
            if hasattr(network, "cost_breakdown"):
                part = network.cost_breakdown(
                    node, target, departure, payload)
                for name in (
                    "rolling_wheel_kwh", "grade_wheel_kwh",
                    "aerodynamic_wheel_kwh", "propulsion_kwh",
                    "recuperated_kwh", "recuperation_curtailed_kwh",
                    "travel_auxiliary_kwh", "stop_go_kwh",
                    "congestion_penalty",
                    "distance_km", "travel_time_s", "total_kwh",
                ):
                    target_name = "edge_net_kwh" if name == "total_kwh" else name
                    totals[target_name] += part[name]
                elapsed += part["travel_time_s"]
                node = target
                if position < len(route):
                    index = int(route[position])
                    payload += float(instance.demand_kg[index])
                    elapsed += float(network.params.service_time_s)
                    service_count += 1
                continue
            path, transition_energy, transition_time = network.shortest_path(
                node, target, departure, payload)
            reconstructed = 0.0
            cost_payload = network.quantized_payload(payload)
            for edge_id in path:
                part = network.traversal_breakdown(
                    edge_id, departure, cost_payload)
                totals["rolling_wheel_kwh"] += part.rolling_wheel_kwh
                totals["grade_wheel_kwh"] += part.grade_wheel_kwh
                totals["aerodynamic_wheel_kwh"] += part.aerodynamic_wheel_kwh
                totals["propulsion_kwh"] += part.propulsion_kwh
                totals["recuperated_kwh"] += part.recuperated_kwh
                totals["recuperation_curtailed_kwh"] += (
                    part.recuperation_curtailed_kwh)
                totals["travel_auxiliary_kwh"] += part.auxiliary_kwh
                totals["stop_go_kwh"] += part.stop_go_kwh
                totals["congestion_penalty"] += part.congestion_penalty
                totals["distance_km"] += network.edge_data[str(edge_id)][1] / 1000.0
                reconstructed += part.total_kwh
            if abs(reconstructed - float(transition_energy)) > 1e-7:
                raise AssertionError(
                    "path energy breakdown disagrees with route oracle: "
                    f"{reconstructed} != {transition_energy}")
            totals["edge_net_kwh"] += reconstructed
            totals["travel_time_s"] += float(transition_time)
            elapsed += float(transition_time)
            node = target
            if position < len(route):
                index = int(route[position])
                payload += float(instance.demand_kg[index])
                elapsed += float(network.params.service_time_s)
                service_count += 1

    service_auxiliary = (
        service_count * network.params.service_auxiliary_power_kw
        * network.params.service_time_s / 3600.0)
    lifting = service_count * network.params.lifting_compaction_energy_kwh
    total = totals["edge_net_kwh"] + service_auxiliary + lifting
    evaluated = evaluate_routes(instance, routes)
    if abs(total - evaluated.energy_kwh) > 1e-7:
        raise AssertionError(
            "fleet energy breakdown disagrees with common evaluator: "
            f"{total} != {evaluated.energy_kwh}")
    return PlanEnergyBreakdown(
        rolling_wheel_kwh=float(totals["rolling_wheel_kwh"]),
        grade_wheel_kwh=float(totals["grade_wheel_kwh"]),
        aerodynamic_wheel_kwh=float(totals["aerodynamic_wheel_kwh"]),
        propulsion_kwh=float(totals["propulsion_kwh"]),
        recuperated_kwh=float(totals["recuperated_kwh"]),
        recuperation_curtailed_kwh=float(
            totals["recuperation_curtailed_kwh"]),
        travel_auxiliary_kwh=float(totals["travel_auxiliary_kwh"]),
        stop_go_kwh=float(totals["stop_go_kwh"]),
        service_auxiliary_kwh=float(service_auxiliary),
        lifting_compaction_kwh=float(lifting),
        total_kwh=float(total),
        congestion_penalty=float(totals["congestion_penalty"]),
        distance_km=float(totals["distance_km"]),
        travel_time_s=float(totals["travel_time_s"]),
    )


def prepare_state_oracle(instance: PaperInstance) -> float:
    """Materialize shared Table-I payload transitions outside solver timing."""

    start = time.perf_counter()
    nodes = (instance.depot_node, *instance.bin_nodes,
             *(state.node for state in instance.vehicle_states))
    unique_nodes = tuple(dict.fromkeys(nodes))
    instance.network.register_relevant_targets(unique_nodes)
    instance.network.dense_transition_matrices(
        unique_nodes, unique_nodes, instance.start_time_s)
    return time.perf_counter() - start


def prepare_fast_cost_oracle(instance: PaperInstance,
                             adjacent_slots: int = 1) -> float:
    """Precompute compact exact pair costs for the current and next slots."""

    start = time.perf_counter()
    instance.network.active_relevant_slot_count = int(adjacent_slots)
    for offset in range(adjacent_slots):
        instance.network.precompute_relevant_costs(
            instance.start_time_s + offset * 3600.0)
    return time.perf_counter() - start
