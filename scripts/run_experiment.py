"""Run one repository experiment from its checked-in JSON configuration."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    if config.get("experiment") not in {"k_sweep", "hourly_24h", "online"}:
        raise ValueError("experiment must be k_sweep, hourly_24h, or online")
    return config


def _output_path(value: str, area: str) -> Path:
    candidate = (PROJECT_ROOT / value).resolve()
    base = (PROJECT_ROOT / area).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"{area} output must stay below {base}: {candidate}")
    return candidate


def _solver_arguments(config: dict) -> list[str]:
    solver = config.get("solver", {})
    arguments: list[str] = []
    scalar_flags = {
        "max_rounds": "--max-rounds",
        "assignment_rounds": "--assignment-rounds",
        "damping": "--damping",
        "tolerance": "--tolerance",
        "improvement_tolerance": "--improvement-tolerance",
        "exact_global_limit": "--exact-global-limit",
        "maximum_exact_trellis_bins": "--maximum-exact-trellis-bins",
        "stagnation_patience": "--stagnation-patience",
        "oracle_slots": "--oracle-slots",
    }
    for key, flag in scalar_flags.items():
        if key in solver:
            arguments.extend([flag, str(solver[key])])
    if "hypercube_radii" in solver:
        arguments.extend([
            "--hypercube-radii",
            *[str(value) for value in solver["hypercube_radii"]],
        ])
    if solver.get("canonicalize_vehicle_symmetry") is False:
        arguments.append("--no-canonicalize-vehicle-symmetry")
    if solver.get("symmetry_dual_path") is False:
        arguments.append("--no-symmetry-dual-path")
    if solver.get("vehicle_gauss_seidel"):
        arguments.append("--vehicle-gauss-seidel")
    if solver.get("certified_route_messages"):
        arguments.append("--certified-route-messages")
    return arguments


def _baseline_arguments(config: dict) -> list[str]:
    baselines = config.get("baselines", {})
    flags = {
        "ga_population": "--ga-population",
        "ga_generations": "--ga-generations",
        "pso_particles": "--pso-particles",
        "pso_iterations": "--pso-iterations",
        "aco_ants": "--aco-ants",
        "aco_iterations": "--aco-iterations",
    }
    arguments: list[str] = []
    for key, flag in flags.items():
        if key in baselines:
            arguments.extend([flag, str(baselines[key])])
    return arguments


def _hourly_command(config: dict, vehicles: int, hours: list[int],
                    output: Path, figure_dir: Path) -> list[str]:
    instance = config["instance"]
    command = [
        sys.executable, "-m", "scripts.hourly",
        "--n-bins", str(instance["n_bins"]),
        "--vehicles", str(vehicles),
        "--seed", str(instance["seed"]),
        "--hours", *map(str, hours),
        "--methods", *config["methods"],
        "--output", str(output),
        "--figure-dir", str(figure_dir),
        *_solver_arguments(config),
        *_baseline_arguments(config),
    ]
    target = instance.get("target_total_demand_kg")
    if target is not None:
        command.extend(["--target-total-demand-kg", str(target)])
    if config.get("resume", True):
        command.append("--resume")
    return command


def _k_sweep_pipeline(config: dict) -> tuple[list[list[str]], list[list[str]]]:
    instance = config["instance"]
    results = _output_path(config["output"]["results"], "results")
    figures = _output_path(config["output"]["figures"], "figures")
    hours = [int(value) for value in instance["hours"]]
    vehicles = [int(value) for value in instance["vehicles"]]
    primary = []
    for hour in hours:
        for vehicle_count in vehicles:
            output = results / f"h{hour:02d}" / f"k{vehicle_count:02d}" / "hourly.csv"
            primary.append(_hourly_command(
                config, vehicle_count, [hour], output, figures))
    aggregate = [
        sys.executable, "-m", "scripts.aggregate_k_sweep",
        "--root", str(results),
        "--hours", *map(str, hours),
        "--vehicles", *map(str, vehicles),
        "--target-total-demand-kg",
        str(instance["target_total_demand_kg"]),
    ]
    plot = [
        sys.executable, "-m", "scripts.plotting.k_sweep",
        "--input", str(results / "k_sweep_3h_all_methods.csv"),
        "--output-dir", str(figures),
        "--hours", *map(str, hours),
        "--vehicles", *map(str, vehicles),
    ]
    return primary, [aggregate, plot]


def _hourly_pipeline(config: dict) -> tuple[list[list[str]], list[list[str]]]:
    instance = config["instance"]
    results = _output_path(config["output"]["results"], "results")
    figures = _output_path(config["output"]["figures"], "figures")
    output = results / config["output"].get(
        "csv", "hourly_dual_path_with_baselines_seed7.csv")
    command = _hourly_command(
        config, int(instance["vehicles"]),
        [int(value) for value in instance["hours"]], output, figures)
    plot = [
        sys.executable, "-m", "scripts.plotting.ieee",
        "--input", str(output), "--output-dir", str(figures),
    ]
    return [command], [plot]


def _online_pipeline(config: dict) -> tuple[list[list[str]], list[list[str]]]:
    instance = config["instance"]
    traffic = config["traffic"]
    results = _output_path(config["output"]["results"], "results")
    figures = _output_path(config["output"]["figures"], "figures")
    command = [
        sys.executable, "-m", "scripts.online_multiseed",
        "--traffic-seeds", *map(str, instance["traffic_seeds"]),
        "--n-bins", str(instance["n_bins"]),
        "--vehicles", str(instance["vehicles"]),
        "--start-hour", str(instance["start_hour"]),
        "--demand-seed", str(instance["demand_seed"]),
        "--solver-seed", str(instance["solver_seed"]),
        "--methods", *config["methods"],
        "--max-simulation-minutes", str(traffic["max_simulation_minutes"]),
        "--traffic-step-s", str(traffic["traffic_step_s"]),
        "--replanning-interval-s", str(traffic["replanning_interval_s"]),
        "--disruption-fraction", str(traffic["disruption_fraction"]),
        "--disruption-radius-m", str(traffic["disruption_radius_m"]),
        "--minimum-speed-factor", str(traffic["minimum_speed_factor"]),
        "--disruption-delay-minutes", str(traffic["disruption_delay_minutes"]),
        "--degradation-minutes", str(traffic["degradation_minutes"]),
        "--disruption-hold-minutes", str(traffic["disruption_hold_minutes"]),
        "--recovery-minutes", str(traffic["recovery_minutes"]),
        "--replan-acceptance-tolerance-kwh",
        str(traffic["replan_acceptance_tolerance_kwh"]),
        "--output-dir", str(results),
        *_solver_arguments(config),
        *_baseline_arguments(config),
    ]
    cache = instance.get("initial_plan_cache")
    if cache:
        command.extend(["--initial-plan-cache", str((PROJECT_ROOT / cache).resolve())])
    if not config.get("resume", True):
        command.append("--rerun-complete-seeds")
    plot = [
        sys.executable, "-m", "scripts.plotting.online_final",
        "--aligned-epochs", str(results / "aligned_seed_epochs.csv"),
        "--final-stats", str(results / "final_statistics.csv"),
        "--output-dir", str(figures),
    ]
    return [command], [plot]


def _run(command: list[str], dry_run: bool) -> None:
    print(f"RUN {subprocess.list2cmdline(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run(config_path: Path, dry_run: bool = False,
        figures_only: bool = False) -> None:
    config_path = config_path.resolve()
    config = _load(config_path)
    builders = {
        "k_sweep": _k_sweep_pipeline,
        "hourly_24h": _hourly_pipeline,
        "online": _online_pipeline,
    }
    primary, post = builders[config["experiment"]](config)
    if not figures_only:
        jobs = max(1, int(config.get("execution", {}).get("parallel_jobs", 1)))
        if jobs == 1 or len(primary) == 1 or dry_run:
            for command in primary:
                _run(command, dry_run)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(_run, command, False)
                           for command in primary]
                for future in as_completed(futures):
                    future.result()
    for command in post:
        _run(command, dry_run)
    if not dry_run:
        results = _output_path(config["output"]["results"], "results")
        manifest = {
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "experiment": config["experiment"],
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "figures_only": figures_only,
            "parallel_jobs": int(config.get("execution", {}).get(
                "parallel_jobs", 1)),
        }
        (results / "last_run.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--figures-only", action="store_true",
        help="recreate figures from retained CSV files without running solvers")
    return parser


def main(arguments: list[str] | None = None) -> None:
    args = build_parser().parse_args(arguments)
    run(args.config, args.dry_run, args.figures_only)


if __name__ == "__main__":
    main()
