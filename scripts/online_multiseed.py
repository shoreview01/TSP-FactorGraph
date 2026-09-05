"""Run and aggregate paired static/adaptive online traffic experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .online import (
    METHODS,
    POLICIES,
    build_parser as build_online_parser,
    run as run_online,
)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _result_source(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return str(path)


def _result_is_complete(path: Path, seed: int, methods: list[str],
                        args: argparse.Namespace) -> bool:
    required = [
        path / "experiment_manifest.json",
        path / "online_epochs.csv",
        path / "online_summary.csv",
        path / "traffic_trace.json",
    ]
    if not all(item.is_file() for item in required):
        return False
    manifest = json.loads(required[0].read_text(encoding="utf-8"))
    expected = {
        "traffic_seed": seed,
        "n_bins": args.n_bins,
        "vehicles": args.vehicles,
        "start_hour": args.start_hour,
        "demand_seed": args.demand_seed,
        "solver_seed": args.solver_seed,
        "traffic_model": "wall_clock_delayed_gradual_disruption",
        "disruption_fraction": args.disruption_fraction,
        "disruption_radius_m": args.disruption_radius_m,
        "minimum_speed_factor": args.minimum_speed_factor,
        "max_simulation_minutes": args.max_simulation_minutes,
        "traffic_step_s": args.traffic_step_s,
        "replanning_interval_s": args.replanning_interval_s,
        "disruption_delay_s": args.disruption_delay_minutes * 60.0,
        "degradation_s": args.degradation_minutes * 60.0,
        "disruption_hold_s": args.disruption_hold_minutes * 60.0,
        "recovery_s": args.recovery_minutes * 60.0,
        "incumbent_gate": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    summary = pd.read_csv(required[2])
    combinations = set(zip(summary["method"], summary["policy"]))
    return all((method, policy) in combinations
               for method in methods for policy in POLICIES)


def _existing_map(values: list[str]) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(
                f"existing result must use SEED=PATH syntax: {value}")
        mapping[int(seed_text)] = Path(path_text).resolve()
    return mapping


def _online_arguments(args: argparse.Namespace, seed: int,
                      output_dir: Path) -> argparse.Namespace:
    values = [
        "--n-bins", str(args.n_bins),
        "--vehicles", str(args.vehicles),
        "--start-hour", str(args.start_hour),
        "--max-simulation-minutes", str(args.max_simulation_minutes),
        "--traffic-step-s", str(args.traffic_step_s),
        "--replanning-interval-s", str(args.replanning_interval_s),
        "--disruption-fraction", str(args.disruption_fraction),
        "--disruption-radius-m", str(args.disruption_radius_m),
        "--minimum-speed-factor", str(args.minimum_speed_factor),
        "--disruption-delay-minutes", str(args.disruption_delay_minutes),
        "--degradation-minutes", str(args.degradation_minutes),
        "--disruption-hold-minutes", str(args.disruption_hold_minutes),
        "--recovery-minutes", str(args.recovery_minutes),
        "--demand-seed", str(args.demand_seed),
        "--traffic-seed", str(seed),
        "--solver-seed", str(args.solver_seed),
        "--methods", *args.methods,
        "--oracle-slots", str(args.oracle_slots),
        "--ga-population", str(args.ga_population),
        "--ga-generations", str(args.ga_generations),
        "--pso-particles", str(args.pso_particles),
        "--pso-iterations", str(args.pso_iterations),
        "--aco-ants", str(args.aco_ants),
        "--aco-iterations", str(args.aco_iterations),
        "--max-rounds", str(args.max_rounds),
        "--assignment-rounds", str(args.assignment_rounds),
        "--damping", str(args.damping),
        "--tolerance", str(args.tolerance),
        "--improvement-tolerance", str(args.improvement_tolerance),
        "--exact-global-limit", str(args.exact_global_limit),
        "--maximum-exact-trellis-bins",
        str(args.maximum_exact_trellis_bins),
        "--hypercube-radii", *map(str, args.hypercube_radii),
        "--stagnation-patience", str(args.stagnation_patience),
        "--replan-acceptance-tolerance-kwh",
        str(args.replan_acceptance_tolerance_kwh),
        "--output-dir", str(output_dir),
        "--skip-figures",
    ]
    if not args.canonicalize_vehicle_symmetry:
        values.append("--no-canonicalize-vehicle-symmetry")
    if not args.symmetry_dual_path:
        values.append("--no-symmetry-dual-path")
    if args.vehicle_gauss_seidel:
        values.append("--vehicle-gauss-seidel")
    if args.certified_route_messages:
        values.append("--certified-route-messages")
    if args.initial_plan_cache is not None:
        values.extend(["--initial-plan-cache", str(args.initial_plan_cache)])
    return build_online_parser().parse_args(values)


def _align_terminal_epochs(frame: pd.DataFrame) -> pd.DataFrame:
    """Carry completed runs forward so late means are not survivor-biased."""

    max_epoch = int(frame["epoch"].max())
    rows: list[pd.Series] = []
    zero_columns = (
        "served_bins_epoch", "epoch_energy_kwh", "epoch_operating_s",
        "oracle_runtime_s", "solver_runtime_s", "planning_runtime_s",
        "compute_wait_aux_energy_kwh", "planned_remaining_energy_kwh",
        "planned_remaining_makespan_s", "next_choice_changes",
        "reassigned_bins", "traversed_edges", "disrupted_edges_traversed",
        "disrupted_edge_share", "epoch_energy_per_served_bin_kwh",
    )
    keys = ["traffic_seed", "method", "policy"]
    elapsed_by_epoch = (
        frame[["epoch", "elapsed_s", "elapsed_min"]]
        .drop_duplicates("epoch").set_index("epoch"))
    for _, part in frame.groupby(keys, sort=False):
        part = part.sort_values("epoch")
        rows.extend(row.copy() for _, row in part.iterrows())
        final = part.iloc[-1]
        for epoch in range(int(final["epoch"]) + 1, max_epoch + 1):
            row = final.copy()
            row["epoch"] = epoch
            if epoch in elapsed_by_epoch.index:
                row["elapsed_s"] = elapsed_by_epoch.at[epoch, "elapsed_s"]
                row["elapsed_min"] = elapsed_by_epoch.at[epoch, "elapsed_min"]
            row["traffic_state"] = "terminal"
            row["traffic_change"] = "complete"
            row["remaining_bins_before"] = 0
            row["remaining_bins_after"] = 0
            for column in zero_columns:
                if column in row.index:
                    row[column] = 0
            rows.append(row)
    return pd.DataFrame(rows).sort_values(keys + ["epoch"]).reset_index(drop=True)


def _aggregate_numeric(frame: pd.DataFrame, group_columns: list[str],
                       metrics: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, sort=False)
    pieces = [grouped.size().rename("n")]
    for metric in metrics:
        values = grouped[metric]
        pieces.extend([
            values.mean().rename(f"{metric}_mean"),
            values.std(ddof=1).rename(f"{metric}_std"),
            values.min().rename(f"{metric}_min"),
            values.max().rename(f"{metric}_max"),
        ])
    return pd.concat(pieces, axis=1).reset_index()


def aggregate(seed_dirs: dict[int, Path], output_dir: Path,
              methods: list[str], replanning_interval_s: float = 300.0,
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    epoch_frames = []
    summary_frames = []
    for seed, directory in sorted(seed_dirs.items()):
        epochs = pd.read_csv(directory / "online_epochs.csv")
        summary = pd.read_csv(directory / "online_summary.csv")
        epochs.insert(0, "traffic_seed", seed)
        summary.insert(0, "traffic_seed", seed)
        epoch_frames.append(epochs)
        summary_frames.append(summary)
    raw_epochs = pd.concat(epoch_frames, ignore_index=True)
    raw_summary = pd.concat(summary_frames, ignore_index=True)
    aligned = _align_terminal_epochs(raw_epochs)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_epochs.to_csv(output_dir / "all_seed_epochs.csv", index=False)
    aligned.to_csv(output_dir / "aligned_seed_epochs.csv", index=False)
    raw_summary.to_csv(output_dir / "all_seed_summary.csv", index=False)

    attempted = raw_epochs[
        (raw_epochs["policy"] == "adaptive")
        & raw_epochs["replan_attempted"].astype(bool)
    ].copy()
    attempted["deadline_met"] = (
        attempted["planning_runtime_s"] <= replanning_interval_s)
    deadline_rows = []
    for method, part in attempted.groupby("method", sort=False):
        deadline_rows.append({
            "method": method,
            "replanning_interval_s": replanning_interval_s,
            "attempts": len(part),
            "deadline_hits": int(part["deadline_met"].sum()),
            "deadline_misses": int((~part["deadline_met"]).sum()),
            "deadline_hit_rate": float(part["deadline_met"].mean()),
            "mean_single_replan_s": float(part["planning_runtime_s"].mean()),
            "max_single_replan_s": float(part["planning_runtime_s"].max()),
        })
    pd.DataFrame(deadline_rows).to_csv(
        output_dir / "replanning_deadline_statistics.csv", index=False)

    epoch_metrics = [
        "cumulative_energy_kwh", "cumulative_operating_s",
        "cumulative_planning_runtime_s", "latency_adjusted_energy_kwh",
    ]
    epoch_stats = _aggregate_numeric(
        aligned, ["method", "policy", "epoch", "elapsed_min"],
        epoch_metrics)
    epoch_stats.to_csv(output_dir / "epoch_statistics.csv", index=False)

    final_metrics = [
        "cumulative_energy_kwh", "cumulative_operating_s",
        "cumulative_planning_runtime_s", "latency_adjusted_energy_kwh",
        "latency_adjusted_time_s", "replanning_runtime_s",
        "post_departure_latency_adjusted_energy_kwh",
        "post_departure_latency_adjusted_time_s",
        "replan_attempts", "accepted_replans",
    ]
    final_stats = _aggregate_numeric(
        raw_summary, ["method", "policy"], final_metrics)
    final_stats.to_csv(output_dir / "final_statistics.csv", index=False)

    paired_rows = []
    for (seed, method), part in raw_summary.groupby(
            ["traffic_seed", "method"], sort=False):
        indexed = part.set_index("policy")
        if not set(POLICIES) <= set(indexed.index):
            continue
        for reference_policy in ("static_live", "static_frozen"):
            row = {
                "traffic_seed": seed,
                "method": method,
                "reference_policy": reference_policy,
            }
            for metric, label in (
                    ("cumulative_energy_kwh", "energy"),
                    ("cumulative_operating_s", "operating_time"),
                    ("post_departure_latency_adjusted_energy_kwh",
                     "compute_aware_energy"),
                    ("post_departure_latency_adjusted_time_s",
                     "compute_aware_time")):
                reference = float(indexed.at[reference_policy, metric])
                adaptive = float(indexed.at["adaptive", metric])
                row[f"reference_{label}"] = reference
                row[f"adaptive_{label}"] = adaptive
                row[f"{label}_delta"] = adaptive - reference
                row[f"{label}_gain_pct"] = (
                    100.0 * (reference - adaptive) / reference)
            paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    paired["adaptive_energy_win"] = (paired["energy_delta"] < 0).astype(int)
    paired.to_csv(output_dir / "paired_adaptation_by_seed.csv", index=False)
    paired_metrics = [
        column for column in paired.columns
        if column.endswith("_delta") or column.endswith("_gain_pct")
    ]
    paired_metrics.append("adaptive_energy_win")
    paired_stats = _aggregate_numeric(
        paired, ["method", "reference_policy"], paired_metrics)
    paired_stats.to_csv(
        output_dir / "paired_adaptation_statistics.csv", index=False)

    adaptive_energy = (raw_summary[raw_summary["policy"] == "adaptive"]
                       .pivot(index="traffic_seed", columns="method",
                              values="cumulative_energy_kwh"))
    comparison_columns = [
        "traffic_seed", "baseline", "proposed_energy_kwh",
        "baseline_energy_kwh", "proposed_minus_baseline_kwh",
        "proposed_reduction_pct", "proposed_win",
    ]
    proposed_comparisons = []
    if "proposed" in adaptive_energy.columns:
        for seed, values in adaptive_energy.iterrows():
            proposed = float(values["proposed"])
            for baseline in methods:
                if baseline == "proposed" or baseline not in values.index:
                    continue
                baseline_energy = float(values[baseline])
                proposed_comparisons.append({
                "traffic_seed": int(seed),
                "baseline": baseline,
                "proposed_energy_kwh": proposed,
                "baseline_energy_kwh": baseline_energy,
                "proposed_minus_baseline_kwh": proposed - baseline_energy,
                "proposed_reduction_pct": (
                    100.0 * (baseline_energy - proposed) / baseline_energy),
                    "proposed_win": int(proposed < baseline_energy),
                })
    comparisons = pd.DataFrame(
        proposed_comparisons, columns=comparison_columns)
    comparisons.to_csv(
        output_dir / "proposed_vs_baseline_by_seed.csv", index=False)
    comparison_metrics = [
        "proposed_minus_baseline_kwh", "proposed_reduction_pct",
        "proposed_win",
    ]
    if comparisons.empty:
        comparison_stats = pd.DataFrame(columns=["baseline"])
        for metric in comparison_metrics:
            comparison_stats[f"{metric}_mean"] = pd.Series(dtype=float)
            comparison_stats[f"{metric}_std"] = pd.Series(dtype=float)
            comparison_stats[f"{metric}_min"] = pd.Series(dtype=float)
            comparison_stats[f"{metric}_max"] = pd.Series(dtype=float)
    else:
        comparison_stats = _aggregate_numeric(
            comparisons, ["baseline"], comparison_metrics)
    comparison_stats.to_csv(
        output_dir / "proposed_vs_baseline_statistics.csv", index=False)

    return final_stats, paired_stats


def _write_results(output_dir: Path, seeds: list[int],
                   final_stats: pd.DataFrame,
                   paired_stats: pd.DataFrame) -> None:
    adaptive = final_stats[final_stats["policy"] == "adaptive"].set_index(
        "method")
    static_live = final_stats[
        final_stats["policy"] == "static_live"].set_index("method")
    static_frozen = final_stats[
        final_stats["policy"] == "static_frozen"].set_index("method")
    paired = paired_stats.set_index(["method", "reference_policy"])
    deadline = pd.read_csv(
        output_dir / "replanning_deadline_statistics.csv").set_index(
            "method")
    lines = [
        "# Multi-seed online replanning result",
        "",
        f"Traffic seeds: {', '.join(map(str, seeds))}. Demand and solver seeds "
        "are fixed, and all three policies receive an identical traffic trace "
        "within each seed.",
        "",
        "Approximately 5% of bins are selected as fixed disruption centers. "
        "Traffic remains normal for 10 minutes, speeds on edges within a "
        "500 m road-distance radius then decrease linearly from 100% to 35% "
        "over 10 minutes, remain at 35% for 10 minutes, and recover linearly "
        "to 100% over 25 minutes.",
        "",
        "Planning is modeled as asynchronous and nonblocking: vehicles do not "
        "stop, and planning wall time contributes neither auxiliary energy nor "
        "operating time. Energy trajectories use the idealized assumption that "
        "the new solution is available at the replanning epoch; measured wall "
        "time and 300 s deadline compliance are therefore reported separately. "
        "At every replanning epoch, a candidate is "
        "accepted only when exact evaluation under the current traffic state "
        "uses less residual energy than the incumbent route.",
        "",
        "The line in each tracking panel is the arithmetic mean. The shaded "
        "cloud and final error bars show the full minimum-to-maximum range. "
        "Standard deviations are retained in the statistics CSV files.",
        "",
        "| Method | Adaptive mean (kWh) | Live-navigation static (kWh) | Frozen-route static (kWh) | Gain vs. live static (%) | Wins vs. live | Gain vs. frozen static (%) | Wins vs. frozen | Cumulative replanning (s) | Mean / max single replan (s) | 300 s deadline hits | Accepted / attempted replans |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        if method not in adaptive.index:
            continue
        label = method.upper() if method != "proposed" else "Proposed"
        lines.append(
            f"| {label} "
            f"| {adaptive.at[method, 'cumulative_energy_kwh_mean']:.3f} "
            f"| {static_live.at[method, 'cumulative_energy_kwh_mean']:.3f} "
            f"| {static_frozen.at[method, 'cumulative_energy_kwh_mean']:.3f} "
            f"| {paired.at[(method, 'static_live'), 'energy_gain_pct_mean']:.2f} "
            f"| {int(round(paired.at[(method, 'static_live'), 'adaptive_energy_win_mean'] * len(seeds)))}/{len(seeds)} "
            f"| {paired.at[(method, 'static_frozen'), 'energy_gain_pct_mean']:.2f} "
            f"| {int(round(paired.at[(method, 'static_frozen'), 'adaptive_energy_win_mean'] * len(seeds)))}/{len(seeds)} "
            f"| {adaptive.at[method, 'replanning_runtime_s_mean']:.1f} "
            f"| {deadline.at[method, 'mean_single_replan_s']:.1f} / "
            f"{deadline.at[method, 'max_single_replan_s']:.1f} "
            f"| {int(deadline.at[method, 'deadline_hits'])}/"
            f"{int(deadline.at[method, 'attempts'])} "
            f"| {adaptive.at[method, 'accepted_replans_mean']:.1f} / "
            f"{adaptive.at[method, 'replan_attempts_mean']:.1f} |")
    comparison = pd.read_csv(
        output_dir / "proposed_vs_baseline_statistics.csv").set_index(
            "baseline")
    if not comparison.empty:
        lines.extend([
            "",
            "## Proposed versus adaptive baselines",
            "",
            "| Baseline | Mean Proposed difference (kWh) | Mean reduction (%) | Proposed wins |",
            "|---|---:|---:|---:|",
        ])
        for baseline in ("nn", "aco", "pso", "ga"):
            if baseline not in comparison.index:
                continue
            lines.append(
                f"| {baseline.upper()} "
                f"| {comparison.at[baseline, 'proposed_minus_baseline_kwh_mean']:.3f} "
                f"| {comparison.at[baseline, 'proposed_reduction_pct_mean']:.2f} "
                f"| {int(round(comparison.at[baseline, 'proposed_win_mean'] * len(seeds)))}/{len(seeds)} |")

    if "proposed" in adaptive.index:
        proposed_live_delta = (
            adaptive.at["proposed", "cumulative_energy_kwh_mean"]
            - static_live.at["proposed", "cumulative_energy_kwh_mean"])
        proposed_frozen_delta = (
            adaptive.at["proposed", "cumulative_energy_kwh_mean"]
            - static_frozen.at["proposed", "cumulative_energy_kwh_mean"])
        raw_summary = pd.read_csv(output_dir / "all_seed_summary.csv")
        proposed_seed = raw_summary[
            raw_summary["method"] == "proposed"].pivot(
                index="traffic_seed", columns="policy",
                values="cumulative_energy_kwh")
        adaptive_seed = raw_summary[
            (raw_summary["method"] == "proposed")
            & (raw_summary["policy"] == "adaptive")
        ].set_index("traffic_seed")
        proposed_seed = proposed_seed.join(
            adaptive_seed[["accepted_replans", "total_reassigned_bins"]])
        proposed_seed["adaptive_minus_live_kwh"] = (
            proposed_seed["adaptive"] - proposed_seed["static_live"])
        accepted_correlation = proposed_seed["accepted_replans"].corr(
            proposed_seed["adaptive_minus_live_kwh"])
        reassigned_correlation = proposed_seed["total_reassigned_bins"].corr(
            proposed_seed["adaptive_minus_live_kwh"])
        lines.extend([
            "",
            "## Interpretation",
            "",
            f"Proposed adaptive minus live-navigation static was "
            f"{proposed_live_delta:.3f} kWh, and Proposed adaptive minus "
            f"frozen-route static was {proposed_frozen_delta:.3f} kWh on "
            "average. The exact acceptance gate "
            "was evaluated only under the currently observed traffic snapshot; "
            "it did not forecast the scheduled recovery. Therefore, a route "
            "that was cheaper at acceptance could become more expensive after "
            "the disruption weakened.",
            "",
            "The seed-level Proposed difference relative to live-navigation "
            "static "
            f"had correlation {accepted_correlation:.3f} with the number of "
            "accepted replans and correlation "
            f"{reassigned_correlation:.3f} with the number of reassigned bins. "
            "With only ten seeds and weak correlations, these data do not "
            "support a monotonic claim that more accepted or larger replans "
            "alone caused the final differences. No "
            "traffic forecast or next-commit-horizon model was added, as required "
            "by the experiment scope.",
        ])
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    seed_root = output_dir / "seeds"
    existing = _existing_map(args.existing_seed_result)
    completed: dict[int, Path] = {}
    for seed in args.traffic_seeds:
        candidate = existing.get(seed, seed_root / f"seed{seed}")
        if (not args.rerun_complete_seeds
                and _result_is_complete(candidate, seed, args.methods, args)):
            print(f"REUSE traffic seed {seed}: {candidate}", flush=True)
        else:
            candidate = seed_root / f"seed{seed}"
            print(f"RUN traffic seed {seed}: {candidate}", flush=True)
            run_online(_online_arguments(args, seed, candidate))
        if not _result_is_complete(candidate, seed, args.methods, args):
            raise RuntimeError(f"traffic seed {seed} did not complete")
        completed[seed] = candidate.resolve()

    final_stats, paired_stats = aggregate(
        completed, output_dir, args.methods, args.replanning_interval_s)
    manifest = {
        "traffic_seeds": args.traffic_seeds,
        "seed_count": len(args.traffic_seeds),
        "demand_seed": args.demand_seed,
        "solver_seed": args.solver_seed,
        "paired_design": (
            "adaptive, frozen-route static, and live-navigation static "
            "policies share the exact traffic trace within each seed"),
        "traffic_model": "wall_clock_delayed_gradual_disruption",
        "disruption_fraction": args.disruption_fraction,
        "disruption_radius_m": args.disruption_radius_m,
        "minimum_speed_factor": args.minimum_speed_factor,
        "max_simulation_minutes": args.max_simulation_minutes,
        "traffic_step_s": args.traffic_step_s,
        "replanning_interval_s": args.replanning_interval_s,
        "disruption_delay_s": args.disruption_delay_minutes * 60.0,
        "degradation_s": args.degradation_minutes * 60.0,
        "disruption_hold_s": args.disruption_hold_minutes * 60.0,
        "recovery_s": args.recovery_minutes * 60.0,
        "cloud_definition": "minimum-to-maximum across traffic seeds",
        "execution_model": (
            "asynchronous nonblocking planning with exact incumbent-energy "
            "acceptance gate"),
        "solver_availability_assumption": (
            "energy trajectories idealize each solution as available at its "
            "replanning epoch; measured runtime and deadline compliance are "
            "reported separately and do not delay vehicle motion"),
        "deadline_statistics": "replanning_deadline_statistics.csv",
        "replan_acceptance_tolerance_kwh": (
            args.replan_acceptance_tolerance_kwh),
        "source_directories": {
            str(seed): _result_source(path, output_dir)
            for seed, path in completed.items()},
    }
    (output_dir / "multiseed_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    _write_results(output_dir, args.traffic_seeds, final_stats, paired_stats)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traffic-seeds", nargs="+", type=int,
        default=list(range(2026, 2036)))
    parser.add_argument("--n-bins", type=int, default=84)
    parser.add_argument("--vehicles", type=int, default=10)
    parser.add_argument("--start-hour", type=int, default=18)
    parser.add_argument("--max-simulation-minutes", type=float, default=120.0)
    parser.add_argument("--traffic-step-s", type=float, default=60.0)
    parser.add_argument("--replanning-interval-s", type=float, default=300.0)
    parser.add_argument("--disruption-fraction", type=float, default=0.05)
    parser.add_argument("--disruption-radius-m", type=float, default=500.0)
    parser.add_argument("--minimum-speed-factor", type=float, default=0.35)
    parser.add_argument("--disruption-delay-minutes", type=float, default=10.0)
    parser.add_argument("--degradation-minutes", type=float, default=10.0)
    parser.add_argument("--disruption-hold-minutes", type=float, default=10.0)
    parser.add_argument("--recovery-minutes", type=float, default=25.0)
    parser.add_argument("--demand-seed", type=int, default=7)
    parser.add_argument("--solver-seed", type=int, default=7)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--oracle-slots", type=int, default=2)
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
    parser.add_argument(
        "--replan-acceptance-tolerance-kwh", type=float, default=1e-6)
    parser.add_argument("--initial-plan-cache", type=Path)
    parser.add_argument(
        "--existing-seed-result", action="append", default=[],
        metavar="SEED=PATH")
    parser.add_argument(
        "--rerun-complete-seeds", action="store_true",
        help="ignore completed seed folders and recompute every seed")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PACKAGE_ROOT / "results" / "online")
    return parser


def main(arguments: list[str] | None = None) -> None:
    run(build_parser().parse_args(arguments))


if __name__ == "__main__":
    main()
