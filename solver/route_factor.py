"""Full vehicle route-factor trellis and SOVA max-marginals.

For a vehicle scope of m bins, the table contains the best feasible complete
depot-return route for every one of the 2**m assignment masks.  Incoming rho
messages change only the mask utility, so the expensive physical trellis is
built once per replanning epoch and reused in every cooperative round.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from simul.energy import service_energy_kwh
from .trellis import state_aware_full_table

from .messages import HARD_NEGATIVE
from .model import PaperInstance, transition_cost, transition_costs


@dataclass
class _Label:
    energy_kwh: float
    elapsed_s: float
    last: int
    previous: "_Label | None"


@dataclass(frozen=True)
class RouteMarginals:
    delta: np.ndarray
    include_score: np.ndarray
    exclude_score: np.ndarray


@dataclass
class FullRouteTable:
    """Exact route energy E_k(mask) for every feasible mask in one scope."""

    vehicle: int
    scope: tuple[int, ...]
    energy_by_mask: np.ndarray
    time_by_mask: np.ndarray
    route_by_mask: tuple[tuple[int, ...] | None, ...]

    @property
    def masks(self) -> int:
        return len(self.energy_by_mask)

    @property
    def size(self) -> int:
        return len(self.scope)

    def mask_for_members(self, members: set[int] | list[int] | tuple[int, ...]) -> int:
        position = {global_index: local for local, global_index in enumerate(self.scope)}
        mask = 0
        for member in members:
            if int(member) not in position:
                raise KeyError(f"bin {member} is outside vehicle {self.vehicle}'s scope")
            mask |= 1 << position[int(member)]
        return mask

    def route_for_members(self, members: set[int] | list[int] | tuple[int, ...]
                          ) -> list[int] | None:
        mask = self.mask_for_members(members)
        route = self.route_by_mask[mask]
        return None if route is None else list(route)

    def energy_for_members(self, members: set[int] | list[int] | tuple[int, ...]
                           ) -> float:
        return float(self.energy_by_mask[self.mask_for_members(members)])

    def max_marginals(self, rho: np.ndarray) -> RouteMarginals:
        """Compute Eq. (32)-(33), excluding rho_i from message to b_ik."""

        rho = np.asarray(rho, dtype=float)
        if rho.shape != (self.size,):
            raise ValueError(f"rho must have shape {(self.size,)}, got {rho.shape}")
        feasible = np.isfinite(self.energy_by_mask)
        scores = np.full(self.masks, -np.inf)
        for mask_value in np.flatnonzero(feasible):
            mask = int(mask_value)
            utility = -float(self.energy_by_mask[mask])
            bits = mask
            while bits:
                bit = bits & -bits
                local = bit.bit_length() - 1
                utility += float(rho[local])
                bits ^= bit
            scores[mask] = utility

        include = np.full(self.size, -np.inf)
        exclude = np.full(self.size, -np.inf)
        for local in range(self.size):
            bit = 1 << local
            include_masks = [mask for mask in range(self.masks) if mask & bit]
            exclude_masks = [mask for mask in range(self.masks) if not mask & bit]
            if include_masks:
                # Factor-to-variable messages must omit the incoming rho_i.
                include[local] = float(np.max(scores[include_masks])) - rho[local]
            if exclude_masks:
                exclude[local] = float(np.max(scores[exclude_masks]))
        delta = include - exclude
        delta = np.where(np.isfinite(delta), delta, HARD_NEGATIVE)
        return RouteMarginals(delta, include, exclude)

    def max_marginal_for(self, rho: np.ndarray, target: int) -> float:
        """Return one exact extrinsic max-marginal from this route factor.

        Augmented cavity SOVA needs only the message for the dimension that
        was opened around the current assigned-set face.  Computing just that
        edge avoids forming every other marginal of the temporary cavity.
        """

        rho = np.asarray(rho, dtype=float)
        if rho.shape != (self.size,):
            raise ValueError(f"rho must have shape {(self.size,)}, got {rho.shape}")
        if not 0 <= int(target) < self.size:
            raise IndexError("target is outside the route-factor scope")

        masks = np.arange(self.masks, dtype=np.uint64)
        scores = -np.asarray(self.energy_by_mask, dtype=float).copy()
        scores[~np.isfinite(self.energy_by_mask)] = -np.inf
        for local, prize in enumerate(rho):
            if prize:
                scores += ((masks & np.uint64(1 << local)) != 0) * float(prize)

        target_bit = np.uint64(1 << int(target))
        included = (masks & target_bit) != 0
        include_score = float(np.max(scores[included])) - float(rho[target])
        exclude_score = float(np.max(scores[~included]))
        delta = include_score - exclude_score
        return float(delta) if np.isfinite(delta) else HARD_NEGATIVE

def _insert_label(labels: dict[tuple[int, int], list[_Label]],
                  key: tuple[int, int], candidate: _Label) -> None:
    current = labels.setdefault(key, [])
    if any(label.energy_kwh <= candidate.energy_kwh + 1e-12
           and label.elapsed_s <= candidate.elapsed_s + 1e-9
           for label in current):
        return
    labels[key] = [
        label for label in current
        if not (candidate.energy_kwh <= label.energy_kwh + 1e-12
                and candidate.elapsed_s <= label.elapsed_s + 1e-9)
    ]
    labels[key].append(candidate)


def _route_from_label(label: _Label | None, scope: tuple[int, ...]) -> tuple[int, ...]:
    local_route: list[int] = []
    while label is not None:
        local_route.append(label.last)
        label = label.previous
    local_route.reverse()
    return tuple(scope[local] for local in local_route)


def build_full_route_table(
    instance: PaperInstance,
    vehicle: int,
    scope: tuple[int, ...] | list[int],
    *,
    maximum_scope_size: int = 18,
) -> FullRouteTable:
    """Build the full forward trellis and best return route for every mask."""

    scope = tuple(int(index) for index in scope)
    if len(scope) != len(set(scope)):
        raise ValueError("route-factor scope contains duplicate bins")
    if any(not 0 <= index < instance.n_bins for index in scope):
        raise ValueError("route-factor scope contains an invalid bin")
    if len(scope) > maximum_scope_size:
        raise ValueError(
            f"exact full route factor is limited to {maximum_scope_size} bins; "
            f"vehicle {vehicle} received {len(scope)}")

    # Operational instances expose the shared C_uv(t,l) tensor used by the
    # compiled trellis.  One compiled forward pass now closes every feasible
    # terminal mask, replacing the former Python transition loop while
    # preserving the exact time/load-dependent route factor.
    if hasattr(instance.network, "relevant_cost_tensor"):
        state = instance.vehicle_states[vehicle]
        result = state_aware_full_table(
            instance.network,
            instance.depot_node,
            [instance.bin_nodes[index] for index in scope],
            [float(instance.demand_kg[index]) for index in scope],
            instance.start_time_s,
            state.payload_kg,
            state.battery_kwh,
            instance.reserve_kwh,
            start_node=state.node,
        )
        routes = tuple(
            None if route is None
            else tuple(scope[local] for local in route)
            for route in result.route_by_mask)
        return FullRouteTable(
            vehicle, scope, result.energy_by_mask,
            result.time_by_mask, routes)

    size = len(scope)
    mask_count = 1 << size
    energy_by_mask = np.full(mask_count, np.inf)
    time_by_mask = np.full(mask_count, np.inf)
    routes: list[tuple[int, ...] | None] = [None] * mask_count
    state = instance.vehicle_states[vehicle]
    remaining_capacity = instance.capacity_kg - state.payload_kg
    service_energy = service_energy_kwh(instance.network.params)
    service_time = float(instance.network.params.service_time_s)

    # Empty assignment still returns the current vehicle to the depot.
    try:
        empty_energy, empty_time = transition_cost(instance.network,
            state.node, instance.depot_node, instance.start_time_s,
            state.payload_kg)
        if empty_energy + instance.reserve_kwh <= state.battery_kwh + 1e-9:
            energy_by_mask[0] = float(empty_energy)
            time_by_mask[0] = float(empty_time)
            routes[0] = ()
    except ValueError:
        pass
    if not size:
        return FullRouteTable(vehicle, scope, energy_by_mask, time_by_mask,
                              tuple(routes))

    demands = np.asarray([instance.demand_kg[index] for index in scope], dtype=float)
    mask_load = np.zeros(mask_count)
    mask_size = np.zeros(mask_count, dtype=np.int16)
    for mask in range(1, mask_count):
        bit = mask & -mask
        local = bit.bit_length() - 1
        previous = mask ^ bit
        mask_load[mask] = mask_load[previous] + demands[local]
        mask_size[mask] = mask_size[previous] + 1

    labels: dict[tuple[int, int], list[_Label]] = {}
    initial = transition_costs(instance.network,
        state.node, [instance.bin_nodes[index] for index in scope],
        instance.start_time_s, state.payload_kg)
    for local, global_index in enumerate(scope):
        demand = demands[local]
        if demand > remaining_capacity + 1e-9:
            continue
        edge_energy, edge_time = initial[instance.bin_nodes[global_index]]
        energy = float(edge_energy) + service_energy
        elapsed = float(edge_time) + service_time
        if energy + instance.reserve_kwh <= state.battery_kwh + 1e-9:
            _insert_label(labels, (1 << local, local),
                          _Label(energy, elapsed, local, None))

    for cardinality in range(1, size + 1):
        states = [
            (mask, last, tuple(state_labels))
            for (mask, last), state_labels in labels.items()
            if mask_size[mask] == cardinality
        ]
        for mask, last, state_labels in states:
            load = state.payload_kg + mask_load[mask]
            # Close this assignment mask at the depot using every nondominated
            # arrival label; an earlier, slightly costlier partial route may
            # still yield the best time-dependent completion.
            for label in state_labels:
                try:
                    return_energy, return_time = transition_cost(instance.network,
                        instance.bin_nodes[scope[last]], instance.depot_node,
                        instance.start_time_s + label.elapsed_s, load)
                except ValueError:
                    continue
                total_energy = label.energy_kwh + float(return_energy)
                total_time = label.elapsed_s + float(return_time)
                if total_energy + instance.reserve_kwh > state.battery_kwh + 1e-9:
                    continue
                if (total_energy < energy_by_mask[mask] - 1e-12
                        or (abs(total_energy - energy_by_mask[mask]) <= 1e-12
                            and total_time < time_by_mask[mask])):
                    energy_by_mask[mask] = total_energy
                    time_by_mask[mask] = total_time
                    routes[mask] = _route_from_label(label, scope)

            if cardinality == size:
                continue
            if mask_load[mask] > remaining_capacity + 1e-9:
                continue
            next_locals = [local for local in range(size)
                           if not mask & (1 << local)]
            for label in state_labels:
                transitions = transition_costs(instance.network,
                    instance.bin_nodes[scope[last]],
                    [instance.bin_nodes[scope[local]] for local in next_locals],
                    instance.start_time_s + label.elapsed_s, load)
                for nxt in next_locals:
                    new_mask = mask | (1 << nxt)
                    if mask_load[new_mask] > remaining_capacity + 1e-9:
                        continue
                    edge_energy, edge_time = transitions[
                        instance.bin_nodes[scope[nxt]]]
                    new_energy = label.energy_kwh + float(edge_energy) + service_energy
                    if new_energy + instance.reserve_kwh > state.battery_kwh + 1e-9:
                        continue
                    _insert_label(
                        labels, (new_mask, nxt),
                        _Label(new_energy,
                               label.elapsed_s + float(edge_time) + service_time,
                               nxt, label))

    return FullRouteTable(vehicle, scope, energy_by_mask, time_by_mask,
                          tuple(routes))
