"""Traffic-, grade-, and load-aware EV energy costs on a SUMO network."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=False)
def _target_elapsed_from_predecessors(
    predecessor: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    edge_indptr: np.ndarray,
    edge_indices: np.ndarray,
    edge_time: np.ndarray,
) -> np.ndarray:
    """Accumulate path time on Dijkstra trees for selected targets."""

    source_count = len(source_indices)
    node_count = predecessor.shape[1]
    target_count = len(target_indices)
    output = np.full((source_count, target_count), np.inf)
    memo = np.empty(node_count, dtype=np.float64)
    chain = np.empty(node_count, dtype=np.int64)
    for source_row in range(source_count):
        memo[:] = np.nan
        source = int(source_indices[source_row])
        memo[source] = 0.0
        for target_column in range(target_count):
            target = int(target_indices[target_column])
            current = target
            chain_length = 0
            while np.isnan(memo[current]):
                parent = int(predecessor[source_row, current])
                if parent < 0:
                    chain_length = -1
                    break
                chain[chain_length] = current
                chain_length += 1
                current = parent
            if chain_length < 0:
                continue
            total = memo[current]
            while chain_length > 0:
                chain_length -= 1
                child = int(chain[chain_length])
                parent = int(predecessor[source_row, child])
                left = int(edge_indptr[parent])
                right = int(edge_indptr[parent + 1])
                found_time = np.inf
                while left < right:
                    middle = (left + right) // 2
                    value = int(edge_indices[middle])
                    if value < child:
                        left = middle + 1
                    else:
                        right = middle
                if left < int(edge_indptr[parent + 1]) and int(edge_indices[left]) == child:
                    found_time = edge_time[left]
                total += found_time
                memo[child] = total
            output[source_row, target_column] = memo[target]
    return output


def _shape_points(text: str | None) -> list[tuple[float, ...]]:
    if not text:
        return []
    return [tuple(map(float, point.split(","))) for point in text.split()]


def read_sumo_grades(net_file: str | Path, *, require_elevation: bool = True,
                     required_edge_ids: set[str] | None = None) -> pd.DataFrame:
    """Read directed SUMO edges and calculate rise/grade from geometry z values.

    An elevation-enriched SUMO net has either junction ``z`` attributes or 3-D
    edge/lane shape points. Internal edges are excluded.
    """
    junctions: dict[str, tuple[float, float, float | None]] = {}
    edges: list[dict] = []
    for _, elem in ET.iterparse(net_file, events=("end",)):
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "junction":
            z = elem.get("z")
            junctions[elem.get("id", "")] = (
                float(elem.get("x", 0)), float(elem.get("y", 0)),
                None if z is None else float(z),
            )
        elif tag == "edge" and elem.get("function") != "internal":
            eid = elem.get("id", "")
            if eid and not eid.startswith(":"):
                shape = _shape_points(elem.get("shape"))
                if not shape:
                    lane = elem.find("lane")
                    shape = _shape_points(None if lane is None else lane.get("shape"))
                edges.append({"edge_id": eid, "from": elem.get("from"),
                              "to": elem.get("to"), "shape": shape})
        elem.clear()

    rows = []
    missing = []
    for edge in edges:
        points = edge["shape"]
        z0 = points[0][2] if points and len(points[0]) >= 3 else None
        z1 = points[-1][2] if points and len(points[-1]) >= 3 else None
        u = junctions.get(edge["from"])
        v = junctions.get(edge["to"])
        z0 = z0 if z0 is not None else (None if u is None else u[2])
        z1 = z1 if z1 is not None else (None if v is None else v[2])
        if z0 is None or z1 is None:
            if required_edge_ids is None or edge["edge_id"] in required_edge_ids:
                missing.append(edge["edge_id"])
            continue
        if len(points) >= 2:
            length = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                         for a, b in zip(points, points[1:]))
        elif u is not None and v is not None:
            length = math.hypot(v[0] - u[0], v[1] - u[1])
        else:
            length = 0.0
        rise = z1 - z0
        rows.append({"SUMO_EDGE_ID": edge["edge_id"], "FROM_NODE": edge["from"],
                     "TO_NODE": edge["to"], "LENGTH_M_GEOM": length,
                     "ELEVATION_FROM_M": z0, "ELEVATION_TO_M": z1,
                     "RISE_M": rise, "GRADE": rise / max(length, 1e-9)})

    if require_elevation and missing:
        raise ValueError(
            f"SUMO net has no usable z elevation for {len(missing):,} edges "
            f"(example: {missing[0]}). Enrich the network from a DEM first; "
            "flat-road substitution is disabled."
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class EVParameters:
    curb_mass_kg: float = 7000.0
    payload_capacity_kg: float = 2000.0
    rolling_resistance: float = 0.010
    drag_area_m2: float = 5.5
    drag_coefficient: float = 0.60
    air_density_kg_m3: float = 1.225
    drivetrain_efficiency: float = 0.90
    regen_efficiency: float = 0.60
    auxiliary_power_kw: float = 3.0
    stop_go_tolerance_fraction: float = 0.25
    stop_go_energy_kwh_per_kg: float = 4.93e-6
    service_auxiliary_power_kw: float = 2.0
    service_time_s: float = 120.0
    lifting_compaction_energy_kwh: float = 0.12
    gravity_m_s2: float = 9.80665
    # Native DEM resolution is 90 m. Very short SUMO edges can otherwise span
    # interpolated cells and inherit implausible endpoint grades.
    max_abs_road_grade: float = 0.20
    payload_state_kg: float = 100.0
    planning_objective: str = "energy"  # energy, time, or distance
    load_aware: bool = True


@dataclass(frozen=True)
class EdgeEnergyBreakdown:
    """Manuscript Eq. (6)-(8) battery terms for one directed edge."""

    rolling_wheel_kwh: float
    grade_wheel_kwh: float
    aerodynamic_wheel_kwh: float
    propulsion_kwh: float
    recuperated_kwh: float
    recuperation_curtailed_kwh: float
    auxiliary_kwh: float
    stop_go_kwh: float
    total_kwh: float
    congestion_penalty: float
    mean_speed_kmh: float
    freeflow_speed_kmh: float


def service_energy_kwh(p: EVParameters = EVParameters()) -> float:
    """Battery energy used while one bin is lifted and its waste is compacted."""
    if p.service_time_s < 0 or p.service_auxiliary_power_kw < 0:
        raise ValueError("service time and auxiliary power must be nonnegative")
    if p.lifting_compaction_energy_kwh < 0:
        raise ValueError("lifting/compaction energy must be nonnegative")
    return (p.lifting_compaction_energy_kwh
            + p.service_auxiliary_power_kw * p.service_time_s / 3600.0)


def correct_road_grades(frame: pd.DataFrame, max_abs_grade: float = 0.20) -> pd.DataFrame:
    """Winsorize only extreme short-edge DEM artifacts, retaining raw values."""
    if not 0 < max_abs_grade < 1:
        raise ValueError("max_abs_grade must be a fraction between 0 and 1")
    out = frame.copy()
    out["GRADE_RAW"] = out["GRADE"].astype(float)
    out["GRADE"] = out["GRADE_RAW"].clip(-max_abs_grade, max_abs_grade)
    out["GRADE_CORRECTED"] = ~np.isclose(out["GRADE"], out["GRADE_RAW"])
    out["GRADE_CORRECTION_METHOD"] = np.where(
        out["GRADE_CORRECTED"], f"WINSORIZED_{max_abs_grade:.3f}", "NONE"
    )
    return out


def edge_energy_breakdown(
    length_m: float,
    travel_time_s: float,
    grade: float,
    payload_kg: float,
    p: EVParameters = EVParameters(),
    freeflow_speed_m_s: float | None = None,
) -> EdgeEnergyBreakdown:
    """Return the edge-energy decomposition in manuscript Eqs. (6)-(8)."""

    if length_m < 0 or travel_time_s <= 0 or payload_kg < 0:
        raise ValueError("length, travel time, and payload must be physically valid")
    if not (0 < p.drivetrain_efficiency <= 1
            and 0 <= p.regen_efficiency <= 1):
        raise ValueError("propulsion and recuperation efficiencies are invalid")
    if p.stop_go_tolerance_fraction <= 0:
        raise ValueError("stop-go traversal-time tolerance must be positive")
    if p.stop_go_energy_kwh_per_kg < 0:
        raise ValueError("stop-go energy coefficient must be nonnegative")
    mass = p.curb_mass_kg + payload_kg
    speed = length_m / travel_time_s if length_m else 0.0
    free_speed = (speed if freeflow_speed_m_s is None
                  else max(float(freeflow_speed_m_s), speed, 1e-9))
    theta = math.atan(grade)
    rolling = (mass * p.gravity_m_s2 * p.rolling_resistance
               * math.cos(theta) * length_m / 3.6e6)
    grade_energy = (mass * p.gravity_m_s2 * math.sin(theta)
                    * length_m / 3.6e6)
    aerodynamic = (0.5 * p.air_density_kg_m3 * p.drag_coefficient
                   * p.drag_area_m2
                   * speed * speed * length_m / 3.6e6)
    road_wheel = rolling + grade_energy + aerodynamic

    propulsion = max(0.0, road_wheel) / p.drivetrain_efficiency
    recuperated = max(0.0, -road_wheel) * p.regen_efficiency
    auxiliary = p.auxiliary_power_kw * travel_time_s / 3600.0
    reference_time = (length_m / free_speed
                      if free_speed > 0 else travel_time_s)
    delta_time = p.stop_go_tolerance_fraction * reference_time
    deviation = ((travel_time_s - reference_time) / delta_time
                 if delta_time > 0 else 0.0)
    congestion_penalty = max(deviation - 1.0, 0.0)
    stop_go = p.stop_go_energy_kwh_per_kg * mass * congestion_penalty
    raw_average = propulsion + auxiliary - recuperated
    curtailed = max(0.0, -raw_average)
    total = raw_average + curtailed + stop_go
    return EdgeEnergyBreakdown(
        rolling_wheel_kwh=float(rolling),
        grade_wheel_kwh=float(grade_energy),
        aerodynamic_wheel_kwh=float(aerodynamic),
        propulsion_kwh=float(propulsion),
        recuperated_kwh=float(recuperated),
        recuperation_curtailed_kwh=float(curtailed),
        auxiliary_kwh=float(auxiliary),
        stop_go_kwh=float(stop_go),
        total_kwh=float(total),
        congestion_penalty=float(congestion_penalty),
        mean_speed_kmh=float(speed * 3.6),
        freeflow_speed_kmh=float(free_speed * 3.6),
    )


def edge_energy_kwh(length_m: float, travel_time_s: float, grade: float,
                    payload_kg: float, p: EVParameters = EVParameters(),
                    freeflow_speed_m_s: float | None = None) -> float:
    """Net nonnegative battery energy ``C_uv(t, load)`` for one edge."""

    return edge_energy_breakdown(
        length_m, travel_time_s, grade, payload_kg, p,
        freeflow_speed_m_s).total_kwh


class OperationalEnergyNetwork:
    """Directed operational graph backed by the final v3 hourly traffic layer."""

    def __init__(self, net_file: str | Path, traffic_csv: str | Path,
                 params: EVParameters = EVParameters()):
        traffic = pd.read_csv(traffic_csv, dtype={"SUMO_EDGE_ID": str}, encoding="utf-8-sig")
        required = set(traffic["SUMO_EDGE_ID"].astype(str))
        grades = read_sumo_grades(net_file, required_edge_ids=required)
        grades = correct_road_grades(grades, params.max_abs_road_grade)
        self.edges = grades.merge(traffic, on="SUMO_EDGE_ID", how="inner", validate="one_to_one")
        absent = required - set(self.edges["SUMO_EDGE_ID"])
        if absent:
            raise ValueError(f"{len(absent):,} operational traffic edges are absent from the elevated net")
        self.params = params
        self.by_id = self.edges.set_index("SUMO_EDGE_ID", drop=False)
        self.adj: dict[str, list[str]] = {}
        self.edge_data: dict[
            str, tuple[str, float, float, tuple[float, ...], float]
        ] = {}
        self._path_cache: dict[tuple[str, str, int, float],
                               tuple[tuple[str, ...], float, float]] = {}
        self._relevant_targets: set[str] = set()
        self._dense_transition_cache: dict[int, tuple[tuple[str, ...],
                                                       np.ndarray,
                                                       np.ndarray]] = {}
        self._relevant_cost_cache: dict[
            tuple[int, float],
            tuple[tuple[str, ...], dict[str, int], np.ndarray, np.ndarray]
        ] = {}
        self._relevant_breakdown_cache: dict[
            tuple[int, float],
            tuple[tuple[str, ...], dict[str, int], dict[str, np.ndarray]]
        ] = {}
        self._relevant_tensor_cache: dict[
            tuple[int, int],
            tuple[tuple[str, ...], dict[str, int], np.ndarray, np.ndarray]
        ] = {}
        self._baseline_hourly_tt: dict[str, tuple[float, ...]] = {}
        self._travel_time_multipliers: dict[str, float] = {}
        for row in self.edges.itertuples():
            self.adj.setdefault(row.FROM_NODE, []).append(row.SUMO_EDGE_ID)
            # itertuples renames non-identifier Korean column names, therefore
            # retrieve the hourly array once from the indexed frame instead.
            hourly_tt = tuple(float(self.by_id.at[row.SUMO_EDGE_ID, f"TT_~{h:02d}시"])
                              for h in range(1, 25))
            self.edge_data[row.SUMO_EDGE_ID] = (
                str(row.TO_NODE), float(row.SUMO_LENGTH), float(row.GRADE),
                hourly_tt, float(row.FREEFLOW_SPEED_KMH) / 3.6,
            )
            self._baseline_hourly_tt[str(row.SUMO_EDGE_ID)] = hourly_tt

    @staticmethod
    def slot(sim_time_s: float) -> int:
        return int(sim_time_s // 3600) % 24 + 1

    def quantized_payload(self, payload_kg: float) -> float:
        """Return the Table-I payload state used by every path-cost oracle."""

        resolution = self.params.payload_state_kg
        effective = payload_kg if self.params.load_aware else 0.0
        return float(np.clip(
            round(float(effective) / resolution) * resolution,
            0.0, self.params.payload_capacity_kg))

    def traversal(self, edge_id: str, sim_time_s: float, payload_kg: float) -> tuple[float, float]:
        slot = self.slot(sim_time_s)
        _, length, grade, hourly_tt, free_speed = self.edge_data[str(edge_id)]
        tt = hourly_tt[slot - 1]
        return edge_energy_kwh(
            length, tt, grade, payload_kg, self.params, free_speed), tt

    def traversal_breakdown(self, edge_id: str, sim_time_s: float,
                            payload_kg: float) -> EdgeEnergyBreakdown:
        """Return auditable energy terms for one network traversal."""

        slot = self.slot(sim_time_s)
        _, length, grade, hourly_tt, free_speed = self.edge_data[str(edge_id)]
        return edge_energy_breakdown(
            length, hourly_tt[slot - 1], grade, payload_kg, self.params,
            free_speed)

    def hourly_network_speed_kmh(self, hour: int) -> float:
        """Length-weighted harmonic link speed for an hour (0--23)."""

        if not 0 <= int(hour) <= 23:
            raise ValueError("hour must be between 0 and 23")
        slot = int(hour)
        distance = sum(values[1] for values in self.edge_data.values())
        elapsed = sum(values[3][slot] for values in self.edge_data.values())
        return float(3.6 * distance / elapsed)

    def shortest_paths(self, source: str, targets: list[str] | set[str],
                       sim_time_s: float, payload_kg: float
                       ) -> dict[str, tuple[list[str], float, float]]:
        """Exact one-to-many minimum-energy paths from a single Dijkstra pass."""
        source = str(source)
        unique_targets = {str(t) for t in targets}
        if not unique_targets:
            return {}
        slot = self.slot(sim_time_s)
        payload = self.quantized_payload(payload_kg)
        out: dict[str, tuple[list[str], float, float]] = {}
        missing = set()
        for target in unique_targets:
            if target == source:
                out[target] = ([], 0.0, 0.0)
                continue
            cached = self._path_cache.get((source, target, slot, payload))
            if cached is None:
                missing.add(target)
            else:
                path, energy, dt = cached
                out[target] = (list(path), energy, dt)
        if not missing:
            return out

        # Experiments repeatedly query different subsets of the same bin/depot
        # nodes from an identical (source, hour, payload-state).  Settle all
        # registered relevant targets once, otherwise each trellis marginal
        # launches another nearly identical Dijkstra pass.
        search_targets = missing | (self._relevant_targets - {source})

        queue = [(0.0, 0.0, source)]
        best = {source: 0.0}
        elapsed = {source: 0.0}
        parent: dict[str, tuple[str, str]] = {}
        settled_targets = set()
        while queue and settled_targets != search_targets:
            energy, dt, node = heapq.heappop(queue)
            if energy != best.get(node):
                continue
            if node in search_targets:
                settled_targets.add(node)
            for eid in self.adj.get(node, []):
                nxt, length, grade, hourly_tt, free_speed = self.edge_data[eid]
                edge_dt = hourly_tt[slot - 1]
                physical_energy = edge_energy_kwh(
                    length, edge_dt, grade, payload, self.params, free_speed)
                objective = self.params.planning_objective
                edge_cost = (physical_energy if objective == "energy" else
                             edge_dt if objective == "time" else length)
                cand = energy + edge_cost
                if cand < best.get(nxt, np.inf):
                    best[nxt], elapsed[nxt] = cand, dt + edge_dt
                    parent[nxt] = (node, eid)
                    heapq.heappush(queue, (cand, dt + edge_dt, nxt))
        unreachable = search_targets - settled_targets
        if unreachable:
            raise ValueError(f"no operational path from {source!r} to {next(iter(unreachable))!r}")
        for target in search_targets:
            path = []
            cur = target
            while cur != source:
                cur, eid = parent[cur]
                path.append(eid)
            path.reverse()
            cached = (tuple(path), best[target], elapsed[target])
            self._path_cache[(source, target, slot, payload)] = cached
            if target in unique_targets:
                out[target] = (path, cached[1], cached[2])
        return out

    def shortest_path(self, source: str, target: str, sim_time_s: float,
                      payload_kg: float) -> tuple[list[str], float, float]:
        """Minimum-energy path for C_uv at the transition departure time.

        The hourly IoT snapshot is frozen during one point-to-point transition,
        making edge weights static and Dijkstra exact. The returned travel time
        advances the trellis state before its next transition.
        """
        return self.shortest_paths(source, [target], sim_time_s, payload_kg)[str(target)]

    def costs(self, source: str, targets, sim_time_s: float,
              payload_kg: float) -> dict[str, tuple[float, float]]:
        """Return exact energy/time values, using a compact pair-cost oracle."""

        source = str(source)
        target_nodes = [str(target) for target in targets]
        slot = self.slot(sim_time_s)
        payload = self.quantized_payload(payload_kg)
        cached = self._relevant_cost_cache.get((slot, payload))
        if cached is not None:
            _, node_index, energy, elapsed = cached
            if source in node_index and all(target in node_index
                                            for target in target_nodes):
                source_index = node_index[source]
                return {
                    target: (
                        float(energy[source_index, node_index[target]]),
                        float(elapsed[source_index, node_index[target]]),
                    )
                    for target in target_nodes
                }
        paths = self.shortest_paths(source, target_nodes, sim_time_s, payload)
        return {target: (float(paths[target][1]), float(paths[target][2]))
                for target in target_nodes}

    def cost(self, source: str, target: str, sim_time_s: float,
             payload_kg: float) -> tuple[float, float]:
        """Return minimum-energy path energy and elapsed time."""

        return self.costs(source, [target], sim_time_s, payload_kg)[str(target)]

    def precompute_relevant_costs(self, sim_time_s: float) -> None:
        """Build all relevant endpoint costs with multi-source SciPy Dijkstra.

        This is equivalent to the energy-weighted Dijkstra in
        :meth:`shortest_paths`, including Table-I payload rounding and travel
        time accumulated along the chosen minimum-energy path.
        """

        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        if self.params.planning_objective != "energy":
            raise ValueError(
                "the compact relevant-cost oracle requires the energy objective")
        relevant = tuple(sorted(self._relevant_targets))
        if not relevant:
            return
        slot = self.slot(sim_time_s)
        resolution = float(self.params.payload_state_kg)
        payloads = np.arange(
            0.0, self.params.payload_capacity_kg + 0.5 * resolution,
            resolution, dtype=float)
        if all((slot, float(payload)) in self._relevant_cost_cache
               and (slot, float(payload)) in self._relevant_breakdown_cache
               for payload in payloads):
            return

        edge_ids = list(self.edge_data)
        from_nodes = [str(self.by_id.at[eid, "FROM_NODE"])
                      for eid in edge_ids]
        to_nodes = [self.edge_data[eid][0] for eid in edge_ids]
        nodes = tuple(sorted(set(from_nodes) | set(to_nodes) | set(relevant)))
        node_index = {node: index for index, node in enumerate(nodes)}
        relevant_indices = np.asarray([node_index[node] for node in relevant], int)
        u = np.asarray([node_index[node] for node in from_nodes], int)
        v = np.asarray([node_index[node] for node in to_nodes], int)
        length = np.asarray([self.edge_data[eid][1] for eid in edge_ids], float)
        grade = np.asarray([self.edge_data[eid][2] for eid in edge_ids], float)
        edge_time = np.asarray(
            [self.edge_data[eid][3][slot - 1] for eid in edge_ids], float)
        free_speed = np.asarray(
            [self.edge_data[eid][4] for eid in edge_ids], float)

        order = np.lexsort((v, u))
        sorted_u, sorted_v = u[order], v[order]
        starts = np.r_[0, 1 + np.flatnonzero(
            (sorted_u[1:] != sorted_u[:-1])
            | (sorted_v[1:] != sorted_v[:-1]))]
        group_u, group_v = sorted_u[starts], sorted_v[starts]
        group_sizes = np.diff(np.r_[starts, len(order)])

        theta = np.arctan(grade)
        speed = np.divide(length, edge_time, out=np.zeros_like(length),
                          where=edge_time > 0)
        reference_time = np.divide(
            length, free_speed, out=edge_time.copy(), where=free_speed > 0)
        delta_time = self.params.stop_go_tolerance_fraction * reference_time
        deviation = np.divide(
            edge_time - reference_time, delta_time,
            out=np.zeros_like(edge_time), where=delta_time > 0)
        congestion_penalty = np.maximum(deviation - 1.0, 0.0)
        aerodynamic_wheel = (0.5 * self.params.air_density_kg_m3
                             * self.params.drag_coefficient
                             * self.params.drag_area_m2 * speed * speed
                             * length / 3.6e6)
        rolling_factor = (self.params.gravity_m_s2
                          * self.params.rolling_resistance
                          * np.cos(theta) * length / 3.6e6)
        grade_factor = (self.params.gravity_m_s2 * np.sin(theta)
                        * length / 3.6e6)
        relevant_local_index = {
            node: index for index, node in enumerate(relevant)
        }
        for payload in payloads:
            mass = self.params.curb_mass_kg + float(payload)
            road_wheel = (mass * (rolling_factor + grade_factor)
                          + aerodynamic_wheel)
            propulsion = (np.maximum(0.0, road_wheel)
                          / self.params.drivetrain_efficiency)
            recuperated = (np.maximum(0.0, -road_wheel)
                           * self.params.regen_efficiency)
            auxiliary = self.params.auxiliary_power_kw * edge_time / 3600.0
            average_raw = propulsion + auxiliary - recuperated
            curtailed = np.maximum(0.0, -average_raw)
            stop_go = (self.params.stop_go_energy_kwh_per_kg
                       * mass * congestion_penalty)
            physical = average_raw + curtailed + stop_go
            sorted_cost = physical[order]
            group_min = np.minimum.reduceat(sorted_cost, starts)
            repeated_min = np.repeat(group_min, group_sizes)
            positions = np.arange(len(order))
            chosen_sorted = np.minimum.reduceat(
                np.where(sorted_cost == repeated_min, positions, len(order)),
                starts,
            )
            chosen_original = order[chosen_sorted]
            chosen_time = edge_time[chosen_original]
            graph = csr_matrix(
                (group_min, (group_u, group_v)),
                shape=(len(nodes), len(nodes)),
            )
            time_graph = csr_matrix(
                (chosen_time, (group_u, group_v)),
                shape=(len(nodes), len(nodes)),
            )
            distance, predecessor = dijkstra(
                graph, directed=True, indices=relevant_indices,
                return_predecessors=True,
            )
            energy_matrix = distance[:, relevant_indices]
            elapsed_matrix = _target_elapsed_from_predecessors(
                np.asarray(predecessor, dtype=np.int64),
                np.asarray(relevant_indices, dtype=np.int64),
                np.asarray(relevant_indices, dtype=np.int64),
                np.asarray(time_graph.indptr, dtype=np.int64),
                np.asarray(time_graph.indices, dtype=np.int64),
                np.asarray(time_graph.data, dtype=np.float64),
            )

            def accumulate_component(values: np.ndarray) -> np.ndarray:
                component_graph = csr_matrix(
                    (values[chosen_original], (group_u, group_v)),
                    shape=(len(nodes), len(nodes)),
                )
                return _target_elapsed_from_predecessors(
                    np.asarray(predecessor, dtype=np.int64),
                    np.asarray(relevant_indices, dtype=np.int64),
                    np.asarray(relevant_indices, dtype=np.int64),
                    np.asarray(component_graph.indptr, dtype=np.int64),
                    np.asarray(component_graph.indices, dtype=np.int64),
                    np.asarray(component_graph.data, dtype=np.float64),
                )

            components = {
                "rolling_wheel_kwh": mass * rolling_factor,
                "grade_wheel_kwh": mass * grade_factor,
                "aerodynamic_wheel_kwh": aerodynamic_wheel,
                "propulsion_kwh": propulsion,
                "recuperated_kwh": recuperated,
                "recuperation_curtailed_kwh": curtailed,
                "travel_auxiliary_kwh": auxiliary,
                "stop_go_kwh": stop_go,
                "total_kwh": physical,
                "congestion_penalty": congestion_penalty,
                "distance_km": length / 1000.0,
                "travel_time_s": edge_time,
            }
            component_matrices = {
                name: np.asarray(accumulate_component(values), float)
                for name, values in components.items()
            }

            self._relevant_cost_cache[(slot, float(payload))] = (
                relevant, relevant_local_index,
                np.asarray(energy_matrix, float),
                np.asarray(elapsed_matrix, float),
            )
            self._relevant_breakdown_cache[(slot, float(payload))] = (
                relevant, relevant_local_index, component_matrices)

    def cost_breakdown(self, source: str, target: str, sim_time_s: float,
                       payload_kg: float) -> dict[str, float]:
        """Return pre-accumulated SUMO energy terms on the chosen path."""

        source, target = str(source), str(target)
        self.register_relevant_targets((source, target))
        self.precompute_relevant_costs(sim_time_s)
        slot = self.slot(sim_time_s)
        payload = self.quantized_payload(payload_kg)
        relevant, node_index, matrices = self._relevant_breakdown_cache[
            (slot, payload)]
        del relevant
        source_index, target_index = node_index[source], node_index[target]
        return {
            name: float(matrix[source_index, target_index])
            for name, matrix in matrices.items()
        }

    def relevant_cost_tensor(self, sim_time_s: float, slot_count: int = 2
                             ) -> tuple[
                                 tuple[str, ...], dict[str, int],
                                 np.ndarray, np.ndarray]:
        """Return `[slot, payload, source, target]` energy/time tensors."""

        if slot_count < 1:
            raise ValueError("slot_count must be positive")
        first_slot = self.slot(sim_time_s)
        key = (first_slot, int(slot_count))
        cached = self._relevant_tensor_cache.get(key)
        if cached is not None:
            return cached
        resolution = float(self.params.payload_state_kg)
        payloads = np.arange(
            0.0, self.params.payload_capacity_kg + 0.5 * resolution,
            resolution, dtype=float)
        energy_slots = []
        elapsed_slots = []
        nodes = None
        node_index = None
        for offset in range(slot_count):
            local_time = sim_time_s + offset * 3600.0
            self.precompute_relevant_costs(local_time)
            slot = self.slot(local_time)
            values = [self._relevant_cost_cache[(slot, float(payload))]
                      for payload in payloads]
            if nodes is None:
                nodes, node_index = values[0][0], values[0][1]
            energy_slots.append(np.stack([value[2] for value in values]))
            elapsed_slots.append(np.stack([value[3] for value in values]))
        cached = (
            nodes, node_index,
            np.stack(energy_slots), np.stack(elapsed_slots),
        )
        self._relevant_tensor_cache[key] = cached
        return cached

    def clear_path_cache(self) -> None:
        self._path_cache.clear()
        self._dense_transition_cache.clear()
        self._relevant_cost_cache.clear()
        self._relevant_breakdown_cache.clear()
        self._relevant_tensor_cache.clear()

    def set_travel_time_multipliers(self, multipliers) -> None:
        """Apply one reproducible online traffic snapshot to all hourly slots.

        ``multipliers`` maps directed SUMO edge IDs to positive travel-time
        factors.  Every call first restores the immutable operational-v3
        travel times, so successive replanning epochs never compound shocks.
        Values below one are allowed for controlled recovery experiments, but
        an edge is never made faster than its physical free-flow travel time.
        All path and dense-oracle caches are invalidated atomically.
        """

        normalized = {
            str(edge_id): float(multiplier)
            for edge_id, multiplier in dict(multipliers).items()
        }
        unknown = set(normalized) - set(self.edge_data)
        if unknown:
            example = next(iter(unknown))
            raise KeyError(f"unknown operational edge {example!r}")
        if any(not np.isfinite(value) or value <= 0.0
               for value in normalized.values()):
            raise ValueError("travel-time multipliers must be finite and positive")
        if normalized == self._travel_time_multipliers:
            return

        for edge_id, values in tuple(self.edge_data.items()):
            target, length, grade, _, free_speed = values
            baseline = self._baseline_hourly_tt[edge_id]
            multiplier = normalized.get(edge_id, 1.0)
            freeflow_time = length / free_speed if free_speed > 0.0 else 0.0
            hourly_tt = tuple(
                max(freeflow_time, value * multiplier)
                for value in baseline
            )
            self.edge_data[edge_id] = (
                target, length, grade, hourly_tt, free_speed,
            )
        self._travel_time_multipliers = normalized
        self.clear_path_cache()

    @property
    def travel_time_multipliers(self) -> dict[str, float]:
        """Return a copy of the currently applied online traffic factors."""

        return dict(self._travel_time_multipliers)

    def register_relevant_targets(self, nodes) -> None:
        """Register experiment endpoints for batched one-to-many path caching."""
        before = len(self._relevant_targets)
        self._relevant_targets.update(str(node) for node in nodes)
        if len(self._relevant_targets) != before:
            self._dense_transition_cache.clear()
            self._relevant_cost_cache.clear()
            self._relevant_breakdown_cache.clear()
            self._relevant_tensor_cache.clear()

    def dense_transition_matrices(self, sources, destinations,
                                  sim_time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return shared energy/time tensors indexed by payload, source, target."""
        source_nodes = [str(node) for node in sources]
        destination_nodes = [str(node) for node in destinations]
        self.register_relevant_targets((*source_nodes, *destination_nodes))
        slot = self.slot(sim_time_s)
        cached = self._dense_transition_cache.get(slot)
        nodes = tuple(sorted(self._relevant_targets))
        if cached is None or cached[0] != nodes:
            resolution = self.params.payload_state_kg
            payload_count = int(round(self.params.payload_capacity_kg / resolution)) + 1
            energy = np.empty((payload_count, len(nodes), len(nodes)), dtype=float)
            elapsed = np.empty_like(energy)
            for payload_index in range(payload_count):
                payload = min(payload_index * resolution,
                              self.params.payload_capacity_kg)
                for source_index, source in enumerate(nodes):
                    values = self.shortest_paths(source, nodes, sim_time_s, payload)
                    for target_index, target in enumerate(nodes):
                        _, de, dt = values[target]
                        energy[payload_index, source_index, target_index] = de
                        elapsed[payload_index, source_index, target_index] = dt
            cached = (nodes, energy, elapsed)
            self._dense_transition_cache[slot] = cached
        node_index = {node: index for index, node in enumerate(cached[0])}
        source_indices = np.asarray([node_index[node] for node in source_nodes])
        destination_indices = np.asarray([node_index[node]
                                          for node in destination_nodes])
        local_energy = cached[1][:, source_indices][:, :, destination_indices]
        local_elapsed = cached[2][:, source_indices][:, :, destination_indices]
        return local_energy, local_elapsed
