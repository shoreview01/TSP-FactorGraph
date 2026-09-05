"""Numerical verification for the bundled paper solvers."""

from __future__ import annotations

import itertools
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from simul.energy import EVParameters, edge_energy_breakdown
from solver.baselines import (
    solve_aco, solve_ga, solve_milp, solve_nn, solve_pso)
from solver.certified_route_factor import (
    certified_route_max_marginals)
from solver.messages import (
    canonical_exclusivity_messages, capacity_messages, decode_assignment,
    exclusivity_messages, initialize_messages, update_assignment_messages,
    update_assignment_messages_trw, with_route_messages,
    with_route_messages_trw)
from solver.model import (
    PaperInstance, VehicleState, evaluate_routes, evaluate_vehicle_route)
from solver.proposed import (
    ProposedConfig, _canonicalize_identical_vehicle_state,
    _two_sided_exchange_preference,
    solve_extrinsic_cavity, solve_proposed)
from solver.route_factor import build_full_route_table


class TinyNetwork:
    """Complete deterministic network implementing the production oracle API."""

    def __init__(self, capacity_kg: float):
        self.params = EVParameters(
            payload_capacity_kg=capacity_kg,
            auxiliary_power_kw=0.0,
            service_auxiliary_power_kw=0.0,
            service_time_s=10.0,
            lifting_compaction_energy_kwh=0.01,
            payload_state_kg=1.0,
        )
        self.position = {
            "d": 0.0, "b0": 1.0, "b1": 2.5, "b2": 4.0,
            "b3": 6.5, "b4": 9.0, "b5": 12.5,
        }

    def shortest_path(self, source, target, sim_time_s, payload_kg):
        distance = abs(self.position[str(target)] - self.position[str(source)])
        direction_penalty = (
            0.03 if self.position[str(target)] < self.position[str(source)] else 0.0)
        hour_factor = 1.0 + 0.01 * (int(sim_time_s // 3600) % 3)
        energy = (distance * (0.02 + 0.00001 * float(payload_kg))
                  + direction_penalty) * hour_factor
        elapsed = 5.0 + 3.0 * distance
        return ([] if source == target else [f"{source}->{target}"],
                energy, elapsed)

    def shortest_paths(self, source, targets, sim_time_s, payload_kg):
        return {
            str(target): self.shortest_path(
                source, target, sim_time_s, payload_kg)
            for target in targets
        }


def tiny_instance(n: int = 6, vehicles: int = 2,
                  capacity_kg: float = 300.0) -> PaperInstance:
    network = TinyNetwork(capacity_kg)
    return PaperInstance(
        network=network,
        depot_node="d",
        bin_nodes=tuple(f"b{i}" for i in range(n)),
        demand_kg=np.full(n, 100.0),
        urgency=np.linspace(1.0, 2.0, n),
        vehicle_states=tuple(
            VehicleState("d", 0.0, 100.0) for _ in range(vehicles)),
        start_time_s=8 * 3600.0,
        capacity_kg=capacity_kg,
        maximum_battery_kwh=100.0,
        reserve_kwh=5.0,
        seed=3,
    )


class SolverTests(unittest.TestCase):
    def test_certified_route_messages_match_full_route_table(self):
        instance = tiny_instance(n=6, vehicles=2, capacity_kg=300.0)
        table = build_full_route_table(instance, 0, tuple(range(6)))
        rho = np.random.default_rng(17).normal(size=6)

        def oracle(vehicle, members):
            self.assertEqual(vehicle, 0)
            route = table.route_for_members(set(members))
            energy = table.energy_for_members(set(members))
            return SimpleNamespace(
                feasible=route is not None and np.isfinite(energy),
                energy_kwh=energy,
            )

        expected = table.max_marginals(rho)
        actual = certified_route_max_marginals(
            instance, 0, rho, oracle, seed_sets=((0, 1, 2),))
        np.testing.assert_allclose(actual.delta, expected.delta, atol=1e-10)
        np.testing.assert_allclose(
            actual.include_score, expected.include_score, atol=1e-10)
        np.testing.assert_allclose(
            actual.exclude_score, expected.exclude_score, atol=1e-10)
        self.assertTrue(actual.statistics["certified"])

    def test_certified_full_cube_solver_mode_is_feasible(self):
        instance = tiny_instance(n=6, vehicles=2, capacity_kg=300.0)
        result = solve_proposed(
            instance,
            ProposedConfig(
                max_rounds=4,
                assignment_rounds=3,
                exact_global_limit=4,
                maximum_exact_trellis_bins=6,
                canonicalize_vehicle_symmetry=True,
                certified_route_messages=True,
            ),
        )
        self.assertTrue(result.evaluation.feasible)
        self.assertEqual(
            result.route_message_mode,
            "bound_certified_full_hypercube_sova",
        )
        self.assertTrue(result.diagnostics[0]["certified"])
        self.assertEqual(result.diagnostics[0]["sova_updated_edges"], 12)

    def test_canonical_exclusivity_messages_match_brute_force(self):
        rng = np.random.default_rng(91)
        omega = rng.normal(size=(5, 3))
        eligible = np.ones_like(omega, dtype=bool)
        actual = canonical_exclusivity_messages(omega, eligible)

        sequences = []
        for labels in itertools.product(range(3), repeat=5):
            introduced = 0
            feasible = True
            for label in labels:
                if label > introduced:
                    feasible = False
                    break
                introduced = max(introduced, label + 1)
            if feasible:
                sequences.append(labels)

        expected = np.empty_like(actual)
        for i in range(5):
            for vehicle in range(3):
                include = max((
                    sum(omega[j, labels[j]] for j in range(5) if j != i)
                    for labels in sequences if labels[i] == vehicle
                ), default=-np.inf)
                exclude = max((
                    sum(omega[j, labels[j]] for j in range(5))
                    for labels in sequences if labels[i] != vehicle
                ), default=-np.inf)
                if np.isneginf(include):
                    expected[i, vehicle] = -1e12
                elif np.isneginf(exclude):
                    expected[i, vehicle] = 1e12
                else:
                    expected[i, vehicle] = include - exclude
        np.testing.assert_allclose(actual, expected)

    def test_canonical_decoder_returns_restricted_growth_assignment(self):
        belief = np.array([
            [0.0, 10.0, 20.0],
            [0.0, 9.0, 19.0],
            [0.0, 8.0, 18.0],
            [0.0, 7.0, 17.0],
            [0.0, 6.0, 16.0],
        ])
        labels = decode_assignment(
            belief,
            np.ones(5),
            np.full(3, 5.0),
            np.ones((5, 3), dtype=bool),
            canonical_vehicle_symmetry=True,
        )
        introduced = 0
        for label in labels:
            self.assertLessEqual(int(label), introduced)
            introduced = max(introduced, int(label) + 1)
        self.assertEqual(set(labels.tolist()), {0, 1, 2})

    def test_congested_edge_includes_stop_go_and_auxiliary_energy(self):
        params = EVParameters(
            curb_mass_kg=7000.0, payload_capacity_kg=2000.0,
            auxiliary_power_kw=3.0, stop_go_tolerance_fraction=0.25,
            stop_go_energy_kwh_per_kg=4.93e-6)
        free_speed = 50.0 / 3.6
        free = edge_energy_breakdown(
            1000.0, 72.0, 0.0, 1000.0, params, free_speed)
        congested = edge_energy_breakdown(
            1000.0, 180.0, 0.0, 1000.0, params, free_speed)
        self.assertEqual(free.congestion_penalty, 0.0)
        self.assertEqual(free.stop_go_kwh, 0.0)
        self.assertGreater(congested.congestion_penalty, 0.0)
        self.assertGreater(congested.stop_go_kwh, 0.0)
        self.assertGreater(congested.auxiliary_kwh, free.auxiliary_kwh)
        self.assertGreater(congested.total_kwh, free.total_kwh)

    def test_exclusivity_and_capacity_messages(self):
        omega = np.array([[1.0, -2.0, 4.0], [3.0, 2.0, -1.0]])
        eligible = np.ones_like(omega, dtype=bool)
        expected = np.array([[-4.0, -4.0, -1.0], [-2.0, -3.0, -3.0]])
        np.testing.assert_allclose(
            exclusivity_messages(omega, eligible), expected)

        rng = np.random.default_rng(4)
        for n in range(1, 6):
            weights = rng.integers(1, 8, n).astype(float)
            gamma = rng.normal(size=(n, 2))
            capacities = np.array([10.0, 12.0])
            actual = capacity_messages(
                weights, gamma, capacities, np.ones_like(gamma, dtype=bool))
            for vehicle in range(2):
                for target in range(n):
                    if weights[target] > capacities[vehicle]:
                        self.assertLess(actual[target, vehicle], -1e11)
                        continue
                    conditioned = []
                    for fixed in (0, 1):
                        best = -np.inf
                        for rest in itertools.product((0, 1), repeat=n - 1):
                            choice = []
                            cursor = 0
                            for index in range(n):
                                if index == target:
                                    choice.append(fixed)
                                else:
                                    choice.append(rest[cursor]); cursor += 1
                            if (np.dot(weights, choice) <= capacities[vehicle]
                                    and sum(choice) >= 1):
                                utility = sum(
                                    gamma[index, vehicle] * choice[index]
                                    for index in range(n) if index != target)
                                best = max(best, utility)
                        conditioned.append(best)
                    if np.isneginf(conditioned[0]):
                        self.assertGreater(actual[target, vehicle], 1e11)
                    elif np.isneginf(conditioned[1]):
                        self.assertLess(actual[target, vehicle], -1e11)
                    else:
                        self.assertAlmostEqual(
                            actual[target, vehicle],
                            conditioned[1] - conditioned[0])

    def test_residual_nonempty_is_relaxed_after_vehicle_has_served(self):
        weights = np.array([2.0, 3.0])
        gamma = np.array([[-4.0], [-2.0]])
        eligible = np.ones((2, 1), dtype=bool)
        required = capacity_messages(
            weights, gamma, np.array([5.0]), eligible,
            require_nonempty=np.array([True]))
        relaxed = capacity_messages(
            weights, gamma, np.array([5.0]), eligible,
            require_nonempty=np.array([False]))
        self.assertTrue(np.all(required > 0.0))
        np.testing.assert_allclose(relaxed, 0.0)

        labels = decode_assignment(
            np.array([[3.0, 1.0], [2.0, 0.0]]),
            weights,
            np.array([5.0, 5.0]),
            np.ones((2, 2), dtype=bool),
            require_nonempty=np.array([False, False]),
        )
        self.assertEqual(labels.tolist(), [0, 0])

    def test_online_residual_solvers_allow_idle_vehicles(self):
        instance = replace(
            tiny_instance(n=2, vehicles=3, capacity_kg=300.0),
            require_nonempty=(False, False, False),
        )
        nn = solve_nn(instance)
        self.assertTrue(nn.evaluation.feasible)
        self.assertLess(sum(bool(route) for route in nn.routes), 3)

        proposed = solve_proposed(
            instance,
            ProposedConfig(
                max_rounds=3,
                assignment_rounds=2,
                exact_global_limit=4,
                maximum_exact_trellis_bins=4,
                canonicalize_vehicle_symmetry=False,
                symmetry_dual_path=False,
            ),
        )
        self.assertTrue(proposed.evaluation.feasible)
        self.assertLess(sum(bool(route) for route in proposed.routes), 3)

    def test_message_residual_is_independent_of_damping(self):
        weights = np.array([1.0, 1.0, 1.0])
        capacities = np.array([3.0, 3.0])
        eligible = np.ones((3, 2), dtype=bool)
        state = initialize_messages(3, 2, eligible)
        state.delta[:] = np.array([
            [0.3, -0.2], [0.1, 0.4], [-0.5, 0.2]])

        full, full_residual = update_assignment_messages(
            weights, capacities, state, eligible, 1.0)
        damped, damped_residual = update_assignment_messages(
            weights, capacities, state, eligible, 0.25)
        self.assertAlmostEqual(full_residual, damped_residual)
        self.assertFalse(np.allclose(full.eta, damped.eta))

        raw_delta = np.full((3, 2), -0.4)
        _, full_route_residual = with_route_messages(
            state, raw_delta, eligible, 1.0)
        _, damped_route_residual = with_route_messages(
            state, raw_delta, eligible, 0.25)
        self.assertAlmostEqual(full_route_residual, damped_route_residual)

    def test_exchange_message_uses_same_zero_and_one_face(self):
        # A large negative reverse message must occur in both conditioned
        # maxima.  The former one-sided expression returned 99 here; the
        # exact local-face max-marginal is -3 after cancellation.
        preference, zero_correction = _two_sided_exchange_preference(
            base_energy=10.0,
            removal_energy={0: 8.0, 1: 9.0},
            exchange_energy={0: 11.0, 1: 12.0},
            rho=np.array([-100.0, 0.0]))
        self.assertAlmostEqual(zero_correction, 102.0)
        self.assertAlmostEqual(preference, -3.0)

        zero_scores = [-10.0 - 100.0, -8.0, -9.0 - 100.0]
        one_scores = [-11.0, -12.0 - 100.0]
        brute = max(one_scores) - max(zero_scores)
        self.assertAlmostEqual(preference, brute)

    def test_vehicle_symmetry_canonicalization_swaps_message_columns(self):
        instance = tiny_instance(n=4, vehicles=2)
        eligible = np.ones((4, 2), dtype=bool)
        state = initialize_messages(4, 2, eligible)
        for name in ("eta", "phi", "delta", "omega", "gamma", "rho",
                     "belief"):
            values = getattr(state, name)
            values[:, 0] = 1.0
            values[:, 1] = 2.0
        labels = np.array([1, 1, 0, 0])

        canonical_labels, canonical_state, changed = (
            _canonicalize_identical_vehicle_state(instance, labels, state))

        self.assertTrue(changed)
        np.testing.assert_array_equal(
            canonical_labels, np.array([0, 0, 1, 1]))
        np.testing.assert_allclose(canonical_state.delta[:, 0], 2.0)
        np.testing.assert_allclose(canonical_state.delta[:, 1], 1.0)

    def test_vehicle_gauss_seidel_mode_is_feasible(self):
        instance = tiny_instance()
        result = solve_proposed(
            instance,
            ProposedConfig(
                max_rounds=3, assignment_rounds=1, damping=0.5,
                tolerance=1e-8, exact_global_limit=3,
                maximum_exact_trellis_bins=10,
                canonicalize_vehicle_symmetry=True,
                symmetry_dual_path=False,
                vehicle_gauss_seidel=True))
        self.assertTrue(result.evaluation.feasible)
        self.assertEqual(
            result.route_message_mode,
            "assignment_pruned_bundle_sova_gauss_seidel")
        outer = [row for row in result.diagnostics if row["round"] > 0]
        self.assertTrue(all(row["vehicle_gauss_seidel"] for row in outer))
        self.assertTrue(all(
            row["gauss_seidel_vehicle_updates"] == instance.vehicles
            for row in outer))
        self.assertTrue(all(
            row["sova_updated_edges"]
            == instance.n_bins * instance.vehicles
            for row in outer))

    def test_proposed_decoder_uses_every_vehicle(self):
        instance = tiny_instance(n=5, vehicles=3, capacity_kg=300.0)
        result = solve_proposed(
            instance,
            ProposedConfig(max_rounds=8, assignment_rounds=4,
                           exact_global_limit=3,
                           maximum_exact_trellis_bins=10))
        counts = np.bincount(result.labels, minlength=instance.vehicles)
        self.assertTrue(np.all(counts >= 1), counts)
        self.assertTrue(result.evaluation.feasible)

    def test_full_route_factor_matches_exhaustive_sova(self):
        instance = tiny_instance(n=5, vehicles=1, capacity_kg=1000.0)
        table = build_full_route_table(instance, 0, tuple(range(5)))
        brute_energy = np.full(1 << 5, np.inf)
        brute_energy[0] = evaluate_vehicle_route(instance, 0, []).energy_kwh
        for mask in range(1, 1 << 5):
            members = [index for index in range(5) if mask & (1 << index)]
            for order in itertools.permutations(members):
                evaluation = evaluate_vehicle_route(instance, 0, order)
                if evaluation.feasible:
                    brute_energy[mask] = min(
                        brute_energy[mask], evaluation.energy_kwh)
        np.testing.assert_allclose(table.energy_by_mask, brute_energy, atol=1e-12)

        rho = np.array([0.3, -0.2, 0.5, 0.1, -0.4])
        actual = table.max_marginals(rho).delta
        expected = np.empty(5)
        for target in range(5):
            conditioned = []
            for fixed in (0, 1):
                best = -np.inf
                for mask in range(1 << 5):
                    if bool(mask & (1 << target)) != bool(fixed):
                        continue
                    if not np.isfinite(brute_energy[mask]):
                        continue
                    score = -brute_energy[mask] + sum(
                        rho[index] for index in range(5)
                        if index != target and mask & (1 << index))
                    best = max(best, score)
                conditioned.append(best)
            expected[target] = conditioned[1] - conditioned[0]
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_proposed_exact_mode_is_feasible(self):
        instance = tiny_instance()
        result = solve_proposed(
            instance,
            ProposedConfig(max_rounds=20, damping=0.5, tolerance=1e-8,
                           exact_global_limit=10,
                           maximum_exact_trellis_bins=10))
        self.assertTrue(result.exact_global_factor_graph)
        self.assertTrue(result.evaluation.feasible)
        self.assertEqual(result.evaluation.served_bins, instance.n_bins)
        milp_result = solve_milp(instance, max_bins=6)
        self.assertAlmostEqual(
            result.evaluation.energy_kwh,
            milp_result.evaluation.energy_kwh, places=10)

    def test_large_mode_keeps_every_assignment_edge(self):
        instance = tiny_instance()
        result = solve_proposed(
            instance,
            ProposedConfig(max_rounds=8, damping=0.5, tolerance=1e-8,
                           assignment_rounds=1,
                           exact_global_limit=3,
                           maximum_exact_trellis_bins=10))
        self.assertTrue(result.full_assignment_graph)
        self.assertEqual(result.route_message_mode,
                         "dual_labeled_canonical_bundle_sova")
        self.assertEqual(
            {row["dual_path_branch"] for row in result.diagnostics},
            {"labeled", "canonical"})
        expected_scope = tuple(range(instance.n_bins))
        self.assertTrue(all(scope == expected_scope for scope in result.scopes))
        self.assertTrue(result.evaluation.feasible)
        self.assertTrue(all(
            row["assignment_edges"] == instance.n_bins * instance.vehicles
            for row in result.diagnostics))
        self.assertTrue(all(
            row["maximum_scope_size"] <= 10
            for row in result.diagnostics))
        self.assertTrue(all(
            "hypercube" in row["outer_phase"]
            for row in result.diagnostics))
        self.assertTrue(all(
            row["sova_updated_edges"] == instance.n_bins * instance.vehicles
            for row in result.diagnostics))
        self.assertTrue(all(
            row["full_route_tables_built"] >= 0
            for row in result.diagnostics))
        self.assertTrue(all(
            row["bundle_sova_updated_edges"] == instance.n_bins
            for row in result.diagnostics))
        self.assertTrue(all(
            row.get("unrestricted_decode_evaluated", True)
            for row in result.diagnostics))
        self.assertTrue(all(
            "path_insertion_branches" not in row
            for row in result.diagnostics))
        milp_result = solve_milp(instance, max_bins=6)
        self.assertAlmostEqual(
            result.evaluation.energy_kwh,
            milp_result.evaluation.energy_kwh, places=10)

    def test_dynamic_mode_does_not_treat_incumbent_stagnation_as_convergence(
            self):
        instance = tiny_instance()
        maximum_rounds = 5
        result = solve_proposed(
            instance,
            ProposedConfig(
                max_rounds=maximum_rounds, assignment_rounds=1,
                damping=0.5, tolerance=0.0, stagnation_patience=1,
                exact_global_limit=3, maximum_exact_trellis_bins=10,
                symmetry_dual_path=False))
        self.assertEqual(result.route_message_mode,
                         "assignment_pruned_bundle_sova")
        self.assertEqual(result.rounds, maximum_rounds)
        self.assertFalse(result.converged)
        self.assertTrue(result.evaluation.feasible)

    def test_symmetry_dual_path_keeps_best_incumbent(self):
        instance = tiny_instance()
        common = dict(
            max_rounds=4, assignment_rounds=2, damping=0.5,
            tolerance=1e-8, exact_global_limit=3,
            maximum_exact_trellis_bins=10)
        labeled = solve_proposed(
            instance,
            ProposedConfig(
                **common, canonicalize_vehicle_symmetry=False,
                symmetry_dual_path=False))
        canonical = solve_proposed(
            instance,
            ProposedConfig(
                **common, canonicalize_vehicle_symmetry=True,
                symmetry_dual_path=False))
        dual = solve_proposed(
            instance,
            ProposedConfig(
                **common, canonicalize_vehicle_symmetry=True,
                symmetry_dual_path=True))
        self.assertLessEqual(
            dual.evaluation.energy_kwh,
            labeled.evaluation.energy_kwh + 1e-12)
        self.assertLessEqual(
            dual.evaluation.energy_kwh,
            canonical.evaluation.energy_kwh + 1e-12)
        self.assertEqual(
            dual.route_message_mode,
            "dual_labeled_canonical_bundle_sova")
        self.assertEqual(
            {row["dual_path_branch"] for row in dual.diagnostics},
            {"labeled", "canonical"})

    def test_fixed_factor_epoch_holds_hypercube_until_decode(self):
        instance = tiny_instance()
        epoch_rounds = 3
        result = solve_proposed(
            instance,
            ProposedConfig(
                max_rounds=4, assignment_rounds=2,
                fixed_factor_epoch_rounds=epoch_rounds,
                damping=0.5, tolerance=0.0, exact_global_limit=3,
                maximum_exact_trellis_bins=10,
                symmetry_dual_path=False))
        self.assertEqual(
            result.route_message_mode, "fixed_factor_epoch_bundle_sova")
        self.assertTrue(result.evaluation.feasible)
        outer = [row for row in result.diagnostics if row["round"] > 0]
        self.assertTrue(all(row["fixed_factor_epoch"] for row in outer))
        self.assertTrue(all(
            row["fixed_factor_epoch_rounds"] == epoch_rounds
            for row in outer))
        self.assertTrue(all(
            "fixed_factor_epoch" in row["outer_phase"] for row in outer))

    def test_cavity_mode_refreshes_every_edge_and_keeps_best_plan(self):
        instance = tiny_instance()
        result = solve_extrinsic_cavity(
            instance,
            ProposedConfig(max_rounds=3, damping=0.5, tolerance=1e-8,
                           exact_global_limit=3,
                           maximum_exact_trellis_bins=10))
        self.assertEqual(result.route_message_mode,
                         "all_edge_augmented_cavity_sova")
        self.assertTrue(result.evaluation.feasible)
        self.assertEqual(result.evaluation.served_bins, instance.n_bins)
        self.assertTrue(all(
            row["sova_updated_edges"] == instance.n_bins * instance.vehicles
            for row in result.diagnostics))
        best_trace = [row["best_energy_kwh"] for row in result.diagnostics]
        self.assertTrue(all(
            right <= left + 1e-12
            for left, right in zip(best_trace, best_trace[1:])))

    def test_tree_reweighted_updates_subtract_reverse_messages(self):
        weights = np.array([1.0, 1.0, 1.0])
        eligible = np.ones((3, 2), dtype=bool)
        state = initialize_messages(3, 2, eligible)
        state.delta[:] = np.array([[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2]])
        state.phi[:] = 0.25
        state.eta[:] = -0.1
        state.rho = state.eta + state.phi
        state.omega = state.phi + state.delta
        state.gamma = state.eta + state.delta
        state.belief = state.rho + state.delta
        weight = 0.7
        raw_eta = exclusivity_messages(state.omega, eligible)
        raw_phi = capacity_messages(
            weights, state.gamma, np.array([3.0, 3.0]), eligible)
        updated, _ = update_assignment_messages_trw(
            weights, np.array([3.0, 3.0]), state, eligible, weight)
        np.testing.assert_allclose(
            updated.eta, weight * raw_eta - (1.0 - weight) * state.omega)
        np.testing.assert_allclose(
            updated.phi, weight * raw_phi - (1.0 - weight) * state.gamma)

        raw_delta = np.full((3, 2), -0.4)
        routed, _ = with_route_messages_trw(
            updated, raw_delta, eligible, weight)
        np.testing.assert_allclose(
            routed.delta,
            weight * raw_delta - (1.0 - weight) * updated.rho)

    def test_cavity_trw_mode_is_feasible(self):
        instance = tiny_instance()
        result = solve_extrinsic_cavity(
            instance,
            ProposedConfig(max_rounds=3, tolerance=1e-8,
                           exact_global_limit=3,
                           maximum_exact_trellis_bins=10,
                           tree_reweight=0.7))
        self.assertEqual(result.route_message_mode,
                         "all_edge_augmented_cavity_sova_trw")
        self.assertTrue(result.evaluation.feasible)

    def test_all_requested_baselines_are_feasible(self):
        instance = tiny_instance()
        results = [
            solve_nn(instance),
            solve_ga(instance, seed=3, population=16, generations=8),
            solve_pso(instance, seed=3, particles=12, iterations=8),
            solve_aco(instance, seed=3, ants=10, iterations=8),
            solve_milp(instance, max_bins=6),
        ]
        expected = list(range(instance.n_bins))
        for result in results:
            self.assertEqual(
                sorted(index for route in result.routes for index in route),
                expected)
            self.assertTrue(all(result.routes), result.method)
            self.assertTrue(result.evaluation.feasible)

    def test_incomplete_plan_is_not_feasible(self):
        instance = tiny_instance()
        evaluation = evaluate_routes(instance, [[0, 1, 2], [3, 4]])
        self.assertFalse(evaluation.exact_coverage)
        self.assertFalse(evaluation.feasible)


if __name__ == "__main__":
    unittest.main()
