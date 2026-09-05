"""Determinism and cache-safety checks for online traffic replanning."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from simul.energy import OperationalEnergyNetwork
from scripts.online import (
    RuntimeVehicleState,
    _advance_vehicle,
    _candidate_improves_incumbent,
    _route_changes,
    _traffic_phase,
    generate_traffic_trace,
)
from scripts.online_multiseed import _align_terminal_epochs


class OnlineTrafficTests(unittest.TestCase):
    def test_travel_time_shocks_reset_instead_of_compounding(self):
        network = OperationalEnergyNetwork.__new__(OperationalEnergyNetwork)
        network.edge_data = {
            "e": ("v", 100.0, 0.0, (10.0,) * 24, 10.0),
        }
        network._baseline_hourly_tt = {"e": (10.0,) * 24}
        network._travel_time_multipliers = {}
        network._path_cache = {("x",): object()}
        network._dense_transition_cache = {1: object()}
        network._relevant_cost_cache = {(1, 0.0): object()}
        network._relevant_breakdown_cache = {(1, 0.0): object()}
        network._relevant_tensor_cache = {(1, 1): object()}

        network.set_travel_time_multipliers({"e": 2.0})
        self.assertEqual(network.edge_data["e"][3][0], 20.0)
        network.set_travel_time_multipliers({"e": 1.5})
        self.assertEqual(network.edge_data["e"][3][0], 15.0)
        network.set_travel_time_multipliers({})
        self.assertEqual(network.edge_data["e"][3][0], 10.0)
        self.assertFalse(network._path_cache)
        self.assertFalse(network._relevant_cost_cache)

    def test_seeded_trace_is_exactly_reproducible(self):
        edge_data = {
            "e0": ("n1", 100.0, 0.0, (10.0,) * 24, 10.0),
            "e1": ("n2", 100.0, 0.0, (10.0,) * 24, 10.0),
            "e2": ("n0", 100.0, 0.0, (10.0,) * 24, 10.0),
        }
        by_id = pd.DataFrame({
            "SUMO_EDGE_ID": ["e0", "e1", "e2"],
            "FROM_NODE": ["n0", "n1", "n2"],
        }).set_index("SUMO_EDGE_ID", drop=False)
        instance = SimpleNamespace(
            n_bins=3,
            bin_nodes=("n0", "n1", "n2"),
            network=SimpleNamespace(edge_data=edge_data, by_id=by_id),
        )
        first = generate_traffic_trace(
            instance, 61, seed=2026, disruption_radius_m=500.0)
        second = generate_traffic_trace(
            instance, 61, seed=2026, disruption_radius_m=500.0)
        self.assertEqual(first, second)
        self.assertEqual(first[0].state, "normal")
        self.assertEqual(first[0].multipliers, {})
        self.assertEqual(first[9].speed_multiplier, 1.0)
        degrading = first[10:21]
        self.assertEqual(degrading[0].state, "degrading")
        self.assertTrue(all(
            epoch.region_edges == degrading[0].region_edges
            for epoch in degrading))
        self.assertTrue(all(
            left.speed_multiplier > right.speed_multiplier
            for left, right in zip(degrading, degrading[1:])))
        self.assertAlmostEqual(first[20].speed_multiplier, 0.35)
        self.assertAlmostEqual(first[30].speed_multiplier, 0.35)
        self.assertAlmostEqual(first[55].speed_multiplier, 1.0)
        self.assertEqual(first[55].multipliers, {})

    def test_traffic_phase_is_continuous_at_phase_boundaries(self):
        values = [
            _traffic_phase(t, 0.35, 600.0, 600.0, 600.0, 1500.0)[0]
            for t in (599.999, 600.0, 1199.999, 1200.0,
                      1799.999, 1800.0, 3299.999, 3300.0)
        ]
        self.assertAlmostEqual(values[0], values[1], places=5)
        self.assertAlmostEqual(values[2], values[3], places=5)
        self.assertAlmostEqual(values[4], values[5], places=5)
        self.assertAlmostEqual(values[6], values[7], places=5)

    def test_partial_edge_progress_survives_a_traffic_change(self):
        class Network:
            def __init__(self):
                self.params = SimpleNamespace(
                    service_time_s=30.0,
                    service_auxiliary_power_kw=0.0,
                    lifting_compaction_energy_kwh=0.0,
                )
                self.edge_data = {"e": ("target", 100.0, 0.0, (), 1.0)}
                self.by_id = pd.DataFrame({
                    "SUMO_EDGE_ID": ["e"], "FROM_NODE": ["start"],
                }).set_index("SUMO_EDGE_ID", drop=False)
                self.energy = 1.0
                self.travel_time = 100.0

            def traversal(self, edge_id, sim_time_s, payload_kg):
                return self.energy, self.travel_time

            def shortest_path(self, source, target, sim_time_s, payload_kg):
                return ["e"], self.energy, self.travel_time

        network = Network()
        base = SimpleNamespace(
            network=network, bin_nodes=("target",),
            demand_kg=np.asarray([5.0]), capacity_kg=100.0,
            reserve_kwh=0.0,
        )
        state = RuntimeVehicleState("start", 0.0, 100.0)
        route = [0]
        remaining = {0}
        first = _advance_vehicle(
            base, 0, state, route, remaining, 0.0, 40.0, set(), None)
        self.assertAlmostEqual(first["energy_kwh"], 0.4)
        self.assertEqual(state.active_edge, "e")
        self.assertAlmostEqual(state.edge_remaining_fraction, 0.6)

        network.energy = 2.0
        network.travel_time = 200.0
        second = _advance_vehicle(
            base, 0, state, route, remaining, 40.0, 120.0, set(), None)
        self.assertAlmostEqual(second["energy_kwh"], 1.2)
        self.assertIsNone(state.active_edge)
        self.assertEqual(state.node, "target")

    def test_route_change_metrics_separate_next_choice_and_owner(self):
        previous = [[0, 1], [2, 3]]
        current = [[2, 1], [0, 3]]
        next_changed, reassigned = _route_changes(previous, current)
        self.assertEqual(next_changed, 2)
        self.assertEqual(reassigned, 2)

    def test_candidate_must_strictly_improve_incumbent_energy(self):
        incumbent = SimpleNamespace(feasible=True, energy_kwh=10.0)
        better = SimpleNamespace(feasible=True, energy_kwh=9.5)
        equal = SimpleNamespace(feasible=True, energy_kwh=10.0)
        infeasible = SimpleNamespace(feasible=False, energy_kwh=1.0)
        self.assertTrue(_candidate_improves_incumbent(
            incumbent, better, 1e-6))
        self.assertFalse(_candidate_improves_incumbent(
            incumbent, equal, 1e-6))
        self.assertFalse(_candidate_improves_incumbent(
            incumbent, infeasible, 1e-6))

    def test_completed_seed_is_carried_forward_for_epoch_average(self):
        frame = pd.DataFrame([
            {
                "traffic_seed": 1, "method": "nn", "policy": "adaptive",
                "epoch": 0, "cumulative_energy_kwh": 10.0,
                "elapsed_s": 0.0, "elapsed_min": 0.0,
                "epoch_energy_kwh": 10.0, "remaining_bins_before": 1,
                "remaining_bins_after": 0, "served_bins_epoch": 1,
                "traffic_state": "normal", "traffic_change": "initial",
            },
            {
                "traffic_seed": 2, "method": "nn", "policy": "adaptive",
                "epoch": 0, "cumulative_energy_kwh": 7.0,
                "elapsed_s": 0.0, "elapsed_min": 0.0,
                "epoch_energy_kwh": 7.0, "remaining_bins_before": 2,
                "remaining_bins_after": 1, "served_bins_epoch": 1,
                "traffic_state": "normal", "traffic_change": "initial",
            },
            {
                "traffic_seed": 2, "method": "nn", "policy": "adaptive",
                "epoch": 1, "cumulative_energy_kwh": 12.0,
                "elapsed_s": 60.0, "elapsed_min": 1.0,
                "epoch_energy_kwh": 5.0, "remaining_bins_before": 1,
                "remaining_bins_after": 0, "served_bins_epoch": 1,
                "traffic_state": "mild", "traffic_change": "worsened",
            },
        ])
        aligned = _align_terminal_epochs(frame)
        carried = aligned[
            (aligned["traffic_seed"] == 1) & (aligned["epoch"] == 1)
        ].iloc[0]
        self.assertEqual(len(aligned), 4)
        self.assertEqual(carried["cumulative_energy_kwh"], 10.0)
        self.assertEqual(carried["epoch_energy_kwh"], 0)
        self.assertEqual(carried["remaining_bins_after"], 0)
        self.assertEqual(carried["traffic_state"], "terminal")

if __name__ == "__main__":
    unittest.main()
