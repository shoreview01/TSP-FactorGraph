"""Vehicle-specific state-aware min-sum trellis for EV collection routes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from simul.energy import OperationalEnergyNetwork, service_energy_kwh

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without the optional accelerator
    njit = None

# With the shared relevant-cost tensor, kernel setup is cheap enough that the
# compiled DP wins even on the small leave-one-out cubes used by pruned SOVA.
NUMBA_MIN_BINS = 4


@dataclass
class RouteResult:
    route: list[int]
    energy_kwh: float
    travel_time_s: float
    feasible: bool
    reason: str = ""


@dataclass
class FullTrellisResult:
    """Best depot-return route for every feasible terminal visited mask."""

    energy_by_mask: np.ndarray
    time_by_mask: np.ndarray
    route_by_mask: tuple[tuple[int, ...] | None, ...]


@dataclass
class _Label:
    energy_kwh: float
    travel_time_s: float
    last: int
    previous: "_Label | None"


def _state_aware_trellis_python(network: OperationalEnergyNetwork, depot_node: str,
                                service_nodes: list[str], demands_kg: list[float],
                                start_time_s: float, initial_payload_kg: float,
                                battery_kwh: float, reserve_kwh: float = 5.0,
                                start_node: str | None = None) -> RouteResult:
    """Exact subset DP with nondominated energy--time labels.

    Payload is fixed by the visited mask, while traffic depends on arrival time.
    Labels are therefore pruned only when another label at the same mask and
    location uses no more energy and arrives no later.
    """
    n = len(service_nodes)
    if n > 20:
        raise ValueError("state-aware trellis is limited to 20 bins per vehicle")
    if n == 0:
        return RouteResult([], 0.0, 0.0, True)
    origin = depot_node if start_node is None else start_node
    service_energy = service_energy_kwh(network.params)
    service_time = network.params.service_time_s
    labels: dict[tuple[int, int], list[_Label]] = {}

    def insert(key: tuple[int, int], candidate: _Label) -> None:
        current = labels.setdefault(key, [])
        if any(label.energy_kwh <= candidate.energy_kwh + 1e-12 and
               label.travel_time_s <= candidate.travel_time_s + 1e-9
               for label in current):
            return
        labels[key] = [label for label in current
                       if not (candidate.energy_kwh <= label.energy_kwh + 1e-12 and
                               candidate.travel_time_s <= label.travel_time_s + 1e-9)]
        labels[key].append(candidate)

    initial = network.shortest_paths(origin, service_nodes, start_time_s,
                                     initial_payload_kg)
    for j, node in enumerate(service_nodes):
        _, e, dt = initial[node]
        e += service_energy
        dt += service_time
        if e + reserve_kwh <= battery_kwh + 1e-9:
            insert((1 << j, j), _Label(e, dt, j, None))
    for size in range(1, n):
        states = [(key, tuple(state_labels)) for key, state_labels in labels.items()
                  if key[0].bit_count() == size]
        for (mask, last), state_labels in states:
            if mask.bit_count() != size:
                continue
            load = initial_payload_kg + sum(demands_kg[i] for i in range(n) if mask & (1 << i))
            next_indices = [i for i in range(n) if not mask & (1 << i)]
            for label in state_labels:
                transitions = network.shortest_paths(
                    service_nodes[last], [service_nodes[i] for i in next_indices],
                    start_time_s + label.travel_time_s, load)
                for nxt in next_indices:
                    _, de, ddt = transitions[service_nodes[nxt]]
                    new_energy = label.energy_kwh + de + service_energy
                    if new_energy + reserve_kwh > battery_kwh + 1e-9:
                        continue
                    insert((mask | (1 << nxt), nxt),
                           _Label(new_energy, label.travel_time_s + ddt + service_time,
                                  nxt, label))
    full = (1 << n) - 1
    best: tuple[float, float, _Label] | None = None
    load = initial_payload_kg + sum(demands_kg)
    for last in range(n):
        for label in labels.get((full, last), []):
            _, de, ddt = network.shortest_path(
                service_nodes[last], depot_node,
                start_time_s + label.travel_time_s, load)
            cand = (label.energy_kwh + de, label.travel_time_s + ddt, label)
            if cand[0] + reserve_kwh <= battery_kwh + 1e-9 and (
                    best is None or cand[:2] < best[:2]):
                best = cand
    if best is None:
        return RouteResult([], math.inf, math.inf, False,
                           "no battery-feasible depot-return route")
    energy, dt, label = best
    order: list[int] = []
    while label is not None:
        order.append(label.last)
        label = label.previous
    order.reverse()
    return RouteResult(order, energy, dt, True)


if njit is not None:
    @njit(cache=True)
    def _insert_label(head, next_label, label_energy, label_time,
                      label_state, label_previous, pool_size, pool_capacity,
                      state, energy, elapsed, previous):
        current = head[state]
        while current >= 0:
            if (label_energy[current] <= energy + 1e-12 and
                    label_time[current] <= elapsed + 1e-9):
                return pool_size, 0
            current = next_label[current]

        previous_in_list = -1
        current = head[state]
        while current >= 0:
            following = next_label[current]
            if (energy <= label_energy[current] + 1e-12 and
                    elapsed <= label_time[current] + 1e-9):
                if previous_in_list < 0:
                    head[state] = following
                else:
                    next_label[previous_in_list] = following
            else:
                previous_in_list = current
            current = following

        if pool_size >= pool_capacity:
            return pool_size, -1
        label_energy[pool_size] = energy
        label_time[pool_size] = elapsed
        label_state[pool_size] = state
        label_previous[pool_size] = previous
        next_label[pool_size] = head[state]
        head[state] = pool_size
        return pool_size + 1, 1


    @njit(cache=True)
    def _trellis_kernel(transition_energy, transition_time, available_slots,
                        demands, start_time_s, initial_payload_kg,
                        battery_kwh, reserve_kwh, payload_resolution,
                        payload_capacity, service_energy, service_time,
                        pool_capacity):
        n = demands.size
        empty_route = np.empty(0, dtype=np.int64)
        if n == 0:
            return 0, -1, empty_route, 0.0, 0.0, 0

        mask_count = 1 << n
        state_count = mask_count * n
        head = np.full(state_count, -1, dtype=np.int64)
        next_label = np.empty(pool_capacity, dtype=np.int64)
        label_energy = np.empty(pool_capacity, dtype=np.float64)
        label_time = np.empty(pool_capacity, dtype=np.float64)
        label_state = np.empty(pool_capacity, dtype=np.int64)
        label_previous = np.empty(pool_capacity, dtype=np.int64)
        mask_load = np.empty(mask_count, dtype=np.float64)
        mask_size = np.empty(mask_count, dtype=np.int16)
        mask_load[0] = initial_payload_kg
        mask_size[0] = 0
        for mask in range(1, mask_count):
            bit = mask & -mask
            index = 0
            value = bit
            while value > 1:
                value >>= 1
                index += 1
            previous_mask = mask ^ bit
            mask_load[mask] = mask_load[previous_mask] + demands[index]
            mask_size[mask] = mask_size[previous_mask] + 1

        slot = int(start_time_s // 3600.0) % 24
        if not available_slots[slot]:
            return 2, slot, empty_route, math.inf, math.inf, 0
        payload_index = int(round(initial_payload_kg / payload_resolution))
        payload_index = min(max(payload_index, 0),
                            int(round(payload_capacity / payload_resolution)))
        pool_size = 0
        for nxt in range(n):
            energy = transition_energy[slot, payload_index, 0, nxt] + service_energy
            elapsed = transition_time[slot, payload_index, 0, nxt] + service_time
            if energy + reserve_kwh <= battery_kwh + 1e-9:
                state = (1 << nxt) * n + nxt
                pool_size, inserted = _insert_label(
                    head, next_label, label_energy, label_time, label_state,
                    label_previous, pool_size, pool_capacity, state,
                    energy, elapsed, -1)
                if inserted < 0:
                    return 3, -1, empty_route, math.inf, math.inf, pool_size

        for size in range(1, n):
            for mask in range(1, mask_count):
                if mask_size[mask] != size:
                    continue
                load = mask_load[mask]
                payload_index = int(round(load / payload_resolution))
                payload_index = min(max(payload_index, 0),
                                    int(round(payload_capacity / payload_resolution)))
                for last in range(n):
                    if mask & (1 << last) == 0:
                        continue
                    state = mask * n + last
                    label = head[state]
                    while label >= 0:
                        departure = start_time_s + label_time[label]
                        slot = int(departure // 3600.0) % 24
                        if not available_slots[slot]:
                            return 2, slot, empty_route, math.inf, math.inf, pool_size
                        for nxt in range(n):
                            if mask & (1 << nxt):
                                continue
                            energy = (label_energy[label] +
                                      transition_energy[slot, payload_index,
                                                        last + 1, nxt] +
                                      service_energy)
                            if energy + reserve_kwh > battery_kwh + 1e-9:
                                continue
                            elapsed = (label_time[label] +
                                       transition_time[slot, payload_index,
                                                       last + 1, nxt] +
                                       service_time)
                            next_state = (mask | (1 << nxt)) * n + nxt
                            pool_size, inserted = _insert_label(
                                head, next_label, label_energy, label_time,
                                label_state, label_previous, pool_size,
                                pool_capacity, next_state, energy, elapsed, label)
                            if inserted < 0:
                                return 3, -1, empty_route, math.inf, math.inf, pool_size
                        label = next_label[label]

        full_mask = mask_count - 1
        payload_index = int(round(mask_load[full_mask] / payload_resolution))
        payload_index = min(max(payload_index, 0),
                            int(round(payload_capacity / payload_resolution)))
        best_energy = math.inf
        best_time = math.inf
        best_label = -1
        for last in range(n):
            label = head[full_mask * n + last]
            while label >= 0:
                departure = start_time_s + label_time[label]
                slot = int(departure // 3600.0) % 24
                if not available_slots[slot]:
                    return 2, slot, empty_route, math.inf, math.inf, pool_size
                energy = (label_energy[label] +
                          transition_energy[slot, payload_index, last + 1, n])
                elapsed = (label_time[label] +
                           transition_time[slot, payload_index, last + 1, n])
                if (energy + reserve_kwh <= battery_kwh + 1e-9 and
                        (energy < best_energy or
                         (energy == best_energy and elapsed < best_time))):
                    best_energy = energy
                    best_time = elapsed
                    best_label = label
                label = next_label[label]

        if best_label < 0:
            return 1, -1, empty_route, math.inf, math.inf, pool_size
        route = np.empty(n, dtype=np.int64)
        cursor = n - 1
        label = best_label
        while label >= 0:
            route[cursor] = label_state[label] % n
            cursor -= 1
            label = label_previous[label]
        return 0, -1, route, best_energy, best_time, pool_size


    @njit(cache=True)
    def _full_trellis_kernel(transition_energy, transition_time, available_slots,
                             demands, start_time_s, initial_payload_kg,
                             battery_kwh, reserve_kwh, payload_resolution,
                             payload_capacity, service_energy, service_time,
                             pool_capacity):
        """Compiled forward DP with an exact depot closure for every mask."""

        n = demands.size
        mask_count = 1 << n
        energy_by_mask = np.full(mask_count, math.inf, dtype=np.float64)
        time_by_mask = np.full(mask_count, math.inf, dtype=np.float64)
        best_label_by_mask = np.full(mask_count, -1, dtype=np.int64)
        state_count = mask_count * n
        head = np.full(state_count, -1, dtype=np.int64)
        next_label = np.empty(pool_capacity, dtype=np.int64)
        label_energy = np.empty(pool_capacity, dtype=np.float64)
        label_time = np.empty(pool_capacity, dtype=np.float64)
        label_state = np.empty(pool_capacity, dtype=np.int64)
        label_previous = np.empty(pool_capacity, dtype=np.int64)
        mask_load = np.empty(mask_count, dtype=np.float64)
        mask_size = np.empty(mask_count, dtype=np.int16)
        mask_load[0] = initial_payload_kg
        mask_size[0] = 0
        for mask in range(1, mask_count):
            bit = mask & -mask
            local = 0
            value = bit
            while value > 1:
                value >>= 1
                local += 1
            previous_mask = mask ^ bit
            mask_load[mask] = mask_load[previous_mask] + demands[local]
            mask_size[mask] = mask_size[previous_mask] + 1

        slot = int(start_time_s // 3600.0) % 24
        if not available_slots[slot]:
            return (2, slot, energy_by_mask, time_by_mask,
                    best_label_by_mask, label_state[:0], label_previous[:0], 0)
        payload_index = int(round(initial_payload_kg / payload_resolution))
        payload_index = min(max(payload_index, 0),
                            int(round(payload_capacity / payload_resolution)))
        empty_energy = transition_energy[slot, payload_index, 0, n]
        empty_time = transition_time[slot, payload_index, 0, n]
        if empty_energy + reserve_kwh <= battery_kwh + 1e-9:
            energy_by_mask[0] = empty_energy
            time_by_mask[0] = empty_time

        pool_size = 0
        for nxt in range(n):
            if mask_load[1 << nxt] > payload_capacity + 1e-9:
                continue
            energy = transition_energy[slot, payload_index, 0, nxt] + service_energy
            elapsed = transition_time[slot, payload_index, 0, nxt] + service_time
            if energy + reserve_kwh <= battery_kwh + 1e-9:
                state = (1 << nxt) * n + nxt
                pool_size, inserted = _insert_label(
                    head, next_label, label_energy, label_time, label_state,
                    label_previous, pool_size, pool_capacity, state,
                    energy, elapsed, -1)
                if inserted < 0:
                    return (3, -1, energy_by_mask, time_by_mask,
                            best_label_by_mask, label_state[:0],
                            label_previous[:0], pool_size)

        for size in range(1, n):
            for mask in range(1, mask_count):
                if mask_size[mask] != size or mask_load[mask] > payload_capacity + 1e-9:
                    continue
                payload_index = int(round(mask_load[mask] / payload_resolution))
                payload_index = min(max(payload_index, 0),
                                    int(round(payload_capacity / payload_resolution)))
                for last in range(n):
                    if mask & (1 << last) == 0:
                        continue
                    label = head[mask * n + last]
                    while label >= 0:
                        departure = start_time_s + label_time[label]
                        slot = int(departure // 3600.0) % 24
                        if not available_slots[slot]:
                            return (2, slot, energy_by_mask, time_by_mask,
                                    best_label_by_mask, label_state[:0],
                                    label_previous[:0], pool_size)
                        for nxt in range(n):
                            if mask & (1 << nxt):
                                continue
                            next_mask = mask | (1 << nxt)
                            if mask_load[next_mask] > payload_capacity + 1e-9:
                                continue
                            energy = (label_energy[label] +
                                      transition_energy[slot, payload_index,
                                                        last + 1, nxt] +
                                      service_energy)
                            if energy + reserve_kwh > battery_kwh + 1e-9:
                                continue
                            elapsed = (label_time[label] +
                                       transition_time[slot, payload_index,
                                                       last + 1, nxt] +
                                       service_time)
                            next_state = next_mask * n + nxt
                            pool_size, inserted = _insert_label(
                                head, next_label, label_energy, label_time,
                                label_state, label_previous, pool_size,
                                pool_capacity, next_state, energy, elapsed, label)
                            if inserted < 0:
                                return (3, -1, energy_by_mask, time_by_mask,
                                        best_label_by_mask, label_state[:0],
                                        label_previous[:0], pool_size)
                        label = next_label[label]

        for mask in range(1, mask_count):
            if mask_load[mask] > payload_capacity + 1e-9:
                continue
            payload_index = int(round(mask_load[mask] / payload_resolution))
            payload_index = min(max(payload_index, 0),
                                int(round(payload_capacity / payload_resolution)))
            for last in range(n):
                if mask & (1 << last) == 0:
                    continue
                label = head[mask * n + last]
                while label >= 0:
                    departure = start_time_s + label_time[label]
                    slot = int(departure // 3600.0) % 24
                    if not available_slots[slot]:
                        return (2, slot, energy_by_mask, time_by_mask,
                                best_label_by_mask, label_state[:0],
                                label_previous[:0], pool_size)
                    energy = (label_energy[label] +
                              transition_energy[slot, payload_index, last + 1, n])
                    elapsed = (label_time[label] +
                               transition_time[slot, payload_index, last + 1, n])
                    if (energy + reserve_kwh <= battery_kwh + 1e-9 and
                            (energy < energy_by_mask[mask] or
                             (energy == energy_by_mask[mask] and
                              elapsed < time_by_mask[mask]))):
                        energy_by_mask[mask] = energy
                        time_by_mask[mask] = elapsed
                        best_label_by_mask[mask] = label
                    label = next_label[label]

        return (0, -1, energy_by_mask, time_by_mask, best_label_by_mask,
                label_state[:pool_size].copy(),
                label_previous[:pool_size].copy(), pool_size)


def _fill_transition_slot(network: OperationalEnergyNetwork, depot_node: str,
                          origin: str, service_nodes: list[str], slot: int,
                          transition_energy: np.ndarray,
                          transition_time: np.ndarray) -> None:
    sources = [origin, *service_nodes]
    destinations = [*service_nodes, depot_node]
    sim_time_s = float(slot * 3600)
    # OperationalEnergyNetwork already builds the manuscript's shared
    # C_uv(t, l) tensor for every relevant endpoint and payload state.  Slice
    # that tensor for this assignment-pruned vehicle trellis instead of
    # rebuilding an increasingly large dense matrix for every decoded set.
    # Lightweight test networks do not expose this optimized API and retain
    # the generic dense-transition fallback.
    if hasattr(network, "relevant_cost_tensor"):
        if hasattr(network, "register_relevant_targets"):
            network.register_relevant_targets((*sources, *destinations))
        nodes, node_index, all_energy, all_elapsed = (
            network.relevant_cost_tensor(sim_time_s, slot_count=1))
        del nodes
        source_indices = np.asarray(
            [node_index[str(node)] for node in sources], dtype=np.int64)
        destination_indices = np.asarray(
            [node_index[str(node)] for node in destinations], dtype=np.int64)
        energy = all_energy[0][:, source_indices][:, :, destination_indices]
        elapsed = all_elapsed[0][:, source_indices][:, :, destination_indices]
    else:
        energy, elapsed = network.dense_transition_matrices(
            sources, destinations, sim_time_s)
    transition_energy[slot] = energy
    transition_time[slot] = elapsed


def _state_aware_trellis_numba(network: OperationalEnergyNetwork, depot_node: str,
                                service_nodes: list[str], demands_kg: list[float],
                                start_time_s: float, initial_payload_kg: float,
                                battery_kwh: float, reserve_kwh: float,
                                start_node: str | None) -> RouteResult:
    n = len(service_nodes)
    if n == 0:
        return RouteResult([], 0.0, 0.0, True)
    origin = depot_node if start_node is None else start_node
    resolution = network.params.payload_state_kg
    payload_capacity = network.params.payload_capacity_kg
    payload_count = int(round(payload_capacity / resolution)) + 1
    shape = (24, payload_count, n + 1, n + 1)
    transition_energy = np.full(shape, np.nan, dtype=np.float64)
    transition_time = np.full(shape, np.nan, dtype=np.float64)
    available_slots = np.zeros(24, dtype=np.bool_)
    demands = np.asarray(demands_kg, dtype=np.float64)
    service_energy = service_energy_kwh(network.params)
    service_time = network.params.service_time_s
    state_count = (1 << n) * n
    pool_capacity = max(1024, state_count * 4)
    maximum_pool = max(pool_capacity, state_count * 64)

    while True:
        status, needed_slot, route, energy, elapsed, _ = _trellis_kernel(
            transition_energy, transition_time, available_slots, demands,
            float(start_time_s), float(initial_payload_kg), float(battery_kwh),
            float(reserve_kwh), float(resolution), float(payload_capacity),
            float(service_energy), float(service_time),
            pool_capacity)
        if status == 2:
            _fill_transition_slot(network, depot_node, origin, service_nodes,
                                  int(needed_slot), transition_energy,
                                  transition_time)
            available_slots[int(needed_slot)] = True
            continue
        if status == 3 and pool_capacity < maximum_pool:
            pool_capacity = min(pool_capacity * 2, maximum_pool)
            continue
        if status == 3:
            raise MemoryError("Numba trellis label pool exceeded its safe limit")
        if status == 1:
            return RouteResult([], math.inf, math.inf, False,
                               "no battery-feasible depot-return route")
        return RouteResult(route.astype(int).tolist(), float(energy),
                           float(elapsed), True)


def warm_state_aware_trellis_numba() -> None:
    """Compile/load the Numba kernel before a measured solver run."""
    if njit is None:
        return
    transition_energy = np.zeros((24, 1, 2, 2), dtype=np.float64)
    transition_time = np.ones((24, 1, 2, 2), dtype=np.float64)
    available_slots = np.ones(24, dtype=np.bool_)
    _trellis_kernel(transition_energy, transition_time, available_slots,
                    np.array([0.0]), 0.0, 0.0, 10.0, 0.0,
                    1.0, 0.0, 0.0, 0.0, 1024)


def state_aware_full_table(network: OperationalEnergyNetwork, depot_node: str,
                           service_nodes: list[str], demands_kg: list[float],
                           start_time_s: float, initial_payload_kg: float,
                           battery_kwh: float, reserve_kwh: float = 5.0,
                           start_node: str | None = None) -> FullTrellisResult:
    """Return exact depot-return energy and route for every terminal mask."""

    if njit is None:
        raise RuntimeError("full terminal-mask trellis requires numba")
    n = len(service_nodes)
    if n > 20:
        raise ValueError("state-aware full trellis is limited to 20 bins")
    if n == 0:
        return FullTrellisResult(
            np.asarray([0.0]), np.asarray([0.0]), ((),))
    origin = depot_node if start_node is None else start_node
    resolution = network.params.payload_state_kg
    payload_capacity = network.params.payload_capacity_kg
    payload_count = int(round(payload_capacity / resolution)) + 1
    shape = (24, payload_count, n + 1, n + 1)
    transition_energy = np.full(shape, np.nan, dtype=np.float64)
    transition_time = np.full(shape, np.nan, dtype=np.float64)
    available_slots = np.zeros(24, dtype=np.bool_)
    demands = np.asarray(demands_kg, dtype=np.float64)
    service_energy = service_energy_kwh(network.params)
    service_time = network.params.service_time_s
    state_count = (1 << n) * n
    pool_capacity = max(1024, state_count * 4)
    maximum_pool = max(pool_capacity, state_count * 64)

    while True:
        result = _full_trellis_kernel(
            transition_energy, transition_time, available_slots, demands,
            float(start_time_s), float(initial_payload_kg), float(battery_kwh),
            float(reserve_kwh), float(resolution), float(payload_capacity),
            float(service_energy), float(service_time), pool_capacity)
        (status, needed_slot, energy_by_mask, time_by_mask,
         best_label_by_mask, label_state, label_previous, _) = result
        if status == 2:
            _fill_transition_slot(network, depot_node, origin, service_nodes,
                                  int(needed_slot), transition_energy,
                                  transition_time)
            available_slots[int(needed_slot)] = True
            continue
        if status == 3 and pool_capacity < maximum_pool:
            pool_capacity = min(pool_capacity * 2, maximum_pool)
            continue
        if status == 3:
            raise MemoryError("Numba full-trellis label pool exceeded its limit")
        routes: list[tuple[int, ...] | None] = [None] * (1 << n)
        if np.isfinite(energy_by_mask[0]):
            routes[0] = ()
        for mask in range(1, 1 << n):
            label = int(best_label_by_mask[mask])
            if label < 0:
                continue
            route: list[int] = []
            while label >= 0:
                route.append(int(label_state[label] % n))
                label = int(label_previous[label])
            route.reverse()
            routes[mask] = tuple(route)
        return FullTrellisResult(
            energy_by_mask, time_by_mask, tuple(routes))


def state_aware_trellis(network: OperationalEnergyNetwork, depot_node: str,
                        service_nodes: list[str], demands_kg: list[float],
                        start_time_s: float, initial_payload_kg: float,
                        battery_kwh: float, reserve_kwh: float = 5.0,
                        start_node: str | None = None, *,
                        use_numba: bool = True) -> RouteResult:
    """Exact state-aware trellis with an optional Numba-compiled DP kernel."""
    n = len(service_nodes)
    if n > 20:
        raise ValueError("state-aware trellis is limited to 20 bins per vehicle")
    if use_numba and njit is not None and n >= NUMBA_MIN_BINS:
        return _state_aware_trellis_numba(
            network, depot_node, service_nodes, demands_kg, start_time_s,
            initial_payload_kg, battery_kwh, reserve_kwh, start_node)
    return _state_aware_trellis_python(
        network, depot_node, service_nodes, demands_kg, start_time_s,
        initial_payload_kg, battery_kwh, reserve_kwh, start_node)
