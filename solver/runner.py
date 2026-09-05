"""Unified Table-I experiment runner for proposed and comparison solvers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from .baselines import (solve_aco, solve_ga, solve_milp, solve_nn, solve_pso)
from .model import load_table_i_instance, prepare_state_oracle
from .proposed import ProposedConfig, solve_proposed


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _result_record(result, *, n_bins: int, vehicles: int,
                   start_hour: int, seed: int, oracle_prep_s: float) -> dict:
    evaluation = result.evaluation
    remaining = [
        180.0 - float(energy)
        for energy in evaluation.vehicle_energy_kwh
    ]
    return {
        "run_id": f"{result.method}_n{n_bins}_k{vehicles}_h{start_hour:02d}_s{seed}",
        "method": result.method,
        "n_bins": n_bins,
        "vehicles": vehicles,
        "start_hour": start_hour,
        "seed": seed,
        "energy_kwh": evaluation.energy_kwh,
        "battery_remaining_mean_kwh": float(sum(remaining) / len(remaining)),
        "battery_remaining_min_kwh": float(min(remaining)),
        "battery_remaining_total_kwh": float(sum(remaining)),
        "oracle_prep_s": oracle_prep_s,
        "runtime_s": result.runtime_s,
        "makespan_s": evaluation.makespan_s,
        "served_bins": evaluation.served_bins,
        "served_demand_kg": evaluation.served_demand_kg,
        "active_vehicles": int(sum(bool(route) for route in result.routes)),
        "exact_coverage": evaluation.exact_coverage,
        "capacity_feasible": evaluation.capacity_feasible,
        "battery_feasible": evaluation.battery_feasible,
        "return_feasible": evaluation.return_feasible,
        "feasible": evaluation.feasible,
        "exact_global_factor_graph": getattr(
            result, "exact_global_factor_graph", None),
        "rounds": getattr(result, "rounds", None),
        "converged": getattr(result, "converged", None),
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    route_dir = output.parent / "routes"
    route_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for seed in args.seeds:
        for method in args.methods:
            instance = load_table_i_instance(
                args.n_bins, args.vehicles, args.start_hour, seed)
            oracle_prep_s = (prepare_state_oracle(instance)
                             if args.prepare_oracle else 0.0)
            if method == "proposed":
                config = ProposedConfig(
                    max_rounds=args.max_rounds,
                    assignment_rounds=args.assignment_rounds,
                    damping=args.damping,
                    tolerance=args.tolerance,
                    improvement_tolerance=args.improvement_tolerance,
                    exact_global_limit=args.exact_global_limit,
                    maximum_exact_trellis_bins=(
                        args.maximum_exact_trellis_bins),
                    hypercube_radii=tuple(args.hypercube_radii),
                    stagnation_patience=args.stagnation_patience,
                    canonicalize_vehicle_symmetry=(
                        args.canonicalize_vehicle_symmetry),
                    symmetry_dual_path=args.symmetry_dual_path,
                    vehicle_gauss_seidel=args.vehicle_gauss_seidel,
                    certified_route_messages=args.certified_route_messages,
                )
                result = solve_proposed(instance, config)
            elif method == "nn":
                result = solve_nn(instance)
            elif method == "ga":
                result = solve_ga(
                    instance, seed, population=args.ga_population,
                    generations=args.ga_generations)
            elif method == "pso":
                result = solve_pso(
                    instance, seed, particles=args.pso_particles,
                    iterations=args.pso_iterations)
            elif method == "aco":
                result = solve_aco(
                    instance, seed, ants=args.aco_ants,
                    iterations=args.aco_iterations)
            else:
                result = solve_milp(instance, max_bins=args.milp_max_bins)

            row = _result_record(
                result, n_bins=args.n_bins, vehicles=args.vehicles,
                start_hour=args.start_hour, seed=seed,
                oracle_prep_s=oracle_prep_s)
            rows.append(row)
            route_payload = {
                "run": row,
                "routes": result.routes,
                "vehicle_energy_kwh": result.evaluation.vehicle_energy_kwh,
                "vehicle_time_s": result.evaluation.vehicle_time_s,
                "metadata": getattr(result, "metadata", {}),
            }
            if method == "proposed":
                route_payload.update({
                    "config": asdict(config),
                    "scopes": result.scopes,
                    "full_assignment_graph": result.full_assignment_graph,
                    "route_message_mode": result.route_message_mode,
                    "diagnostics": result.diagnostics,
                })
            (route_dir / f"{row['run_id']}.json").write_text(
                json.dumps(route_payload, indent=2, ensure_ascii=False),
                encoding="utf-8")
            pd.DataFrame(rows).to_csv(output, index=False)
            print(pd.DataFrame([row]).to_string(index=False), flush=True)

    frame = pd.DataFrame(rows)
    summary = (frame.groupby("method", as_index=False)
               .agg(runs=("run_id", "count"),
                    energy_mean_kwh=("energy_kwh", "mean"),
                    energy_std_kwh=("energy_kwh", "std"),
                    runtime_mean_s=("runtime_s", "mean"),
                    feasible_rate=("feasible", "mean"),
                    served_bins_min=("served_bins", "min")))
    summary.to_csv(output.with_name(f"{output.stem}_summary.csv"), index=False)
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bins", type=int, default=84)
    parser.add_argument("--vehicles", type=int, default=10)
    parser.add_argument("--start-hour", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument(
        "--methods", nargs="+",
        choices=["proposed", "nn", "ga", "pso", "aco", "milp"],
        default=["proposed", "nn", "ga", "pso", "aco"])
    parser.add_argument("--max-rounds", type=int, default=32)
    parser.add_argument("--assignment-rounds", type=int, default=24)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--improvement-tolerance", type=float, default=1e-9)
    parser.add_argument("--exact-global-limit", type=int, default=14)
    parser.add_argument("--maximum-exact-trellis-bins", type=int, default=84)
    parser.add_argument(
        "--hypercube-radii", type=int, nargs="+",
        default=[1, 2, 4, 8, 12, 16, 24, 32],
        help="Hamming shells decoded between the separated I/V and trellis blocks")
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
    parser.add_argument("--ga-population", type=int, default=100)
    parser.add_argument("--ga-generations", type=int, default=500)
    parser.add_argument("--pso-particles", type=int, default=60)
    parser.add_argument("--pso-iterations", type=int, default=200)
    parser.add_argument("--aco-ants", type=int, default=60)
    parser.add_argument("--aco-iterations", type=int, default=150)
    parser.add_argument("--milp-max-bins", type=int, default=9)
    parser.add_argument(
        "--prepare-oracle", action="store_true",
        help="precompute the dense all-node/payload oracle outside solver timing")
    parser.add_argument(
        "--output", type=Path,
        default=PACKAGE_ROOT / "results" / "table_i_runs.csv")
    return parser


def main(arguments: list[str] | None = None) -> None:
    run(build_parser().parse_args(arguments))


if __name__ == "__main__":
    main()
