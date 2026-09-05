"""Run proposed, NN, ACO, PSO, and GA over all 24 traffic snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from .plotting.ieee import create_figures
from solver.baselines import solve_aco, solve_ga, solve_nn, solve_pso
from solver.model import load_table_i_instance, prepare_fast_cost_oracle
from solver.proposed import ProposedConfig, solve_proposed
from solver.runner import _result_record


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("proposed", "nn", "aco", "pso", "ga")


def _solve(method: str, instance, args):
    if method == "proposed":
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
                    args.canonicalize_vehicle_symmetry),
                symmetry_dual_path=args.symmetry_dual_path,
                vehicle_gauss_seidel=args.vehicle_gauss_seidel,
                certified_route_messages=args.certified_route_messages,
            ),
        )
    if method == "nn":
        return solve_nn(instance)
    if method == "aco":
        return solve_aco(
            instance, args.seed, ants=args.aco_ants,
            iterations=args.aco_iterations)
    if method == "pso":
        return solve_pso(
            instance, args.seed, particles=args.pso_particles,
            iterations=args.pso_iterations)
    return solve_ga(
        instance, args.seed, population=args.ga_population,
        generations=args.ga_generations)


def _write_outputs(rows: list[dict], output: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values(["start_hour", "method"])
    frame.to_csv(output, index=False)
    summary = (frame.groupby("method", as_index=False)
               .agg(runs=("run_id", "count"),
                    energy_mean_kwh=("energy_kwh", "mean"),
                    battery_remaining_mean_kwh=(
                        "battery_remaining_mean_kwh", "mean"),
                    battery_remaining_min_kwh=(
                        "battery_remaining_min_kwh", "min"),
                    makespan_mean_s=("makespan_s", "mean"),
                    runtime_total_s=("runtime_s", "sum"),
                    feasible_rate=("feasible", "mean")))
    summary.to_csv(output.with_name(f"{output.stem}_summary.csv"), index=False)
    return frame


def run(args: argparse.Namespace) -> pd.DataFrame:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    route_dir = output.parent / "hourly_routes"
    route_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if args.resume and output.exists():
        rows = pd.read_csv(output).to_dict("records")
    completed = {str(row["run_id"]) for row in rows}
    total = len(args.hours) * len(args.methods)
    completed_now = 0
    batch_started = time.perf_counter()

    for hour in args.hours:
        hour_run_ids = {
            method: (
                f"{method}_n{args.n_bins}_k{args.vehicles}_h{hour:02d}_s{args.seed}")
            for method in args.methods
        }
        if all(run_id in completed for run_id in hour_run_ids.values()):
            print(f"SKIP hour {hour:02d} (all methods complete)", flush=True)
            continue
        # One immutable instance is shared within an hour. Only its exact path
        # cache is warmed across methods; demands, vehicle states, and costs do
        # not change. This avoids recomputing identical network transitions.
        instance = load_table_i_instance(
            args.n_bins, args.vehicles, hour, args.seed,
            target_total_demand_kg=args.target_total_demand_kg)
        oracle_prep_s = prepare_fast_cost_oracle(
            instance, adjacent_slots=args.oracle_slots)
        for method in args.methods:
            run_id = hour_run_ids[method]
            if run_id in completed:
                print(f"SKIP {run_id}", flush=True)
                continue
            result = _solve(method, instance, args)
            if result.evaluation.makespan_s >= 3600.0 * args.oracle_slots:
                raise RuntimeError(
                    f"{run_id} crosses into the next traffic slot; "
                    "rerun this hour with a larger --oracle-slots value")
            row = _result_record(
                result, n_bins=args.n_bins, vehicles=args.vehicles,
                start_hour=hour, seed=args.seed,
                oracle_prep_s=oracle_prep_s)
            row["total_demand_kg"] = float(instance.demand_kg.sum())
            row["target_total_demand_kg"] = args.target_total_demand_kg
            rows.append(row)
            completed.add(run_id)
            completed_now += 1

            route_payload = {
                "run": row,
                "routes": result.routes,
                "vehicle_energy_kwh": result.evaluation.vehicle_energy_kwh,
                "vehicle_battery_remaining_kwh": [
                    instance.vehicle_states[vehicle].battery_kwh - energy
                    for vehicle, energy in enumerate(
                        result.evaluation.vehicle_energy_kwh)
                ],
                "vehicle_time_s": result.evaluation.vehicle_time_s,
                "metadata": getattr(result, "metadata", {}),
            }
            if method == "proposed":
                route_payload.update({
                    "config": {
                        "max_rounds": args.max_rounds,
                        "assignment_rounds": args.assignment_rounds,
                        "damping": args.damping,
                        "tolerance": args.tolerance,
                        "improvement_tolerance": args.improvement_tolerance,
                        "exact_global_limit": args.exact_global_limit,
                        "maximum_exact_trellis_bins": (
                            args.maximum_exact_trellis_bins),
                        "hypercube_radii": list(args.hypercube_radii),
                        "stagnation_patience": args.stagnation_patience,
                        "canonicalize_vehicle_symmetry": (
                            args.canonicalize_vehicle_symmetry),
                        "symmetry_dual_path": args.symmetry_dual_path,
                        "vehicle_gauss_seidel": args.vehicle_gauss_seidel,
                        "certified_route_messages": (
                            args.certified_route_messages),
                    },
                    "scopes": result.scopes,
                    "full_assignment_graph": result.full_assignment_graph,
                    "route_message_mode": result.route_message_mode,
                    "diagnostics": result.diagnostics,
                })
            (route_dir / f"{run_id}.json").write_text(
                json.dumps(route_payload, indent=2, ensure_ascii=False),
                encoding="utf-8")
            _write_outputs(rows, output)
            elapsed = time.perf_counter() - batch_started
            print(
                f"DONE {run_id} ({completed_now}/{total} new) "
                f"E={row['energy_kwh']:.6f} kWh, "
                f"Bmean={row['battery_remaining_mean_kwh']:.6f} kWh, "
                f"Tmax={row['makespan_s']:.3f} s, "
                f"solver={row['runtime_s']:.2f} s, batch={elapsed:.1f} s",
                flush=True,
            )

    frame = _write_outputs(rows, output)
    expected = {
        f"{method}_n{args.n_bins}_k{args.vehicles}_h{hour:02d}_s{args.seed}"
        for hour in args.hours for method in args.methods
    }
    if expected <= set(frame["run_id"].astype(str)) and (
            tuple(args.hours) == tuple(range(24))
            and tuple(args.methods) == METHODS):
        create_figures(output, args.figure_dir.resolve())
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bins", type=int, default=84)
    parser.add_argument("--vehicles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-total-demand-kg", type=float)
    parser.add_argument("--hours", type=int, nargs="+", default=list(range(24)))
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS))
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
    parser.add_argument("--oracle-slots", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=PACKAGE_ROOT / "results" / "hourly_24h" / "hourly.csv")
    parser.add_argument(
        "--figure-dir", type=Path,
        default=PACKAGE_ROOT / "figures" / "hourly_24h")
    return parser


def main(arguments: list[str] | None = None) -> None:
    run(build_parser().parse_args(arguments))


if __name__ == "__main__":
    main()
