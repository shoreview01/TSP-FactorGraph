"""Validate and combine per-hour/per-K experiment shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .plotting.ieee import EXPECTED_METHODS


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def aggregate(
    root: Path,
    hours: tuple[int, ...],
    vehicles: tuple[int, ...],
    target_total_demand_kg: float,
) -> pd.DataFrame:
    shards: list[pd.DataFrame] = []
    missing_files: list[str] = []
    for hour in hours:
        for vehicle_count in vehicles:
            path = root / f"h{hour:02d}" / f"k{vehicle_count:02d}" / "hourly.csv"
            if not path.exists():
                missing_files.append(str(path))
                continue
            shard = pd.read_csv(path)
            shard = shard[
                (shard["start_hour"].astype(int) == hour)
                & (shard["vehicles"].astype(int) == vehicle_count)
                & shard["method"].isin(EXPECTED_METHODS)
            ].copy()
            shard["source_csv"] = path.relative_to(root).as_posix()
            shards.append(shard)
    if missing_files:
        raise ValueError(
            f"incomplete sweep: {len(missing_files)} shard files are missing; "
            f"first={missing_files[0]}"
        )

    frame = pd.concat(shards, ignore_index=True, sort=False)
    required = {
        "method", "start_hour", "vehicles", "energy_kwh", "makespan_s",
        "active_vehicles", "served_demand_kg", "feasible",
        "total_demand_kg", "target_total_demand_kg",
    }
    absent = required - set(frame.columns)
    if absent:
        raise ValueError(f"result columns are missing: {sorted(absent)}")

    expected = {
        (method, hour, vehicle_count)
        for method in EXPECTED_METHODS
        for hour in hours
        for vehicle_count in vehicles
    }
    keys = list(zip(
        frame["method"], frame["start_hour"].astype(int),
        frame["vehicles"].astype(int),
    ))
    observed = set(keys)
    if observed != expected or len(keys) != len(expected):
        raise ValueError(
            f"incomplete or duplicated observations: "
            f"missing={sorted(expected - observed)[:5]}, "
            f"extra={sorted(observed - expected)[:5]}, "
            f"rows={len(keys)}, expected_rows={len(expected)}"
        )
    if not _as_bool(frame["feasible"]).all():
        failed = frame.loc[
            ~_as_bool(frame["feasible"]),
            ["method", "start_hour", "vehicles"],
        ]
        raise ValueError(f"infeasible observations:\n{failed.to_string(index=False)}")
    if not (
        frame["active_vehicles"].astype(int) == frame["vehicles"].astype(int)
    ).all():
        raise ValueError("one or more observations do not use exact nonempty K")
    for column in ("served_demand_kg", "total_demand_kg", "target_total_demand_kg"):
        if not frame[column].astype(float).sub(target_total_demand_kg).abs().le(1e-6).all():
            raise ValueError(f"{column} does not equal {target_total_demand_kg:g} kg")

    frame = frame.sort_values(
        ["start_hour", "vehicles", "method"], kind="stable"
    ).reset_index(drop=True)
    frame.to_csv(root / "k_sweep_3h_all_methods.csv", index=False)

    winner_index = frame.groupby(["start_hour", "vehicles"])["energy_kwh"].idxmin()
    winners = frame.loc[
        winner_index,
        ["start_hour", "vehicles", "method", "energy_kwh"],
    ].sort_values(["start_hour", "vehicles"])
    winners.to_csv(root / "winner_by_hour_k.csv", index=False)

    optimum_index = frame.groupby(["method", "start_hour"])["energy_kwh"].idxmin()
    optima = frame.loc[
        optimum_index,
        ["method", "start_hour", "vehicles", "energy_kwh"],
    ].sort_values(["method", "start_hour"])
    optima.to_csv(root / "method_hour_energy_optima.csv", index=False)

    averages = frame.groupby(["method", "vehicles"], as_index=False).agg(
        mean_energy_kwh=("energy_kwh", "mean"),
        min_energy_kwh=("energy_kwh", "min"),
        max_energy_kwh=("energy_kwh", "max"),
        mean_makespan_s=("makespan_s", "mean"),
    )
    averages.to_csv(root / "method_k_average.csv", index=False)

    proposed = averages[averages["method"] == "proposed"]
    proposed_optimum = proposed.loc[proposed["mean_energy_kwh"].idxmin()]
    manifest = {
        "n_bins": int(frame["n_bins"].iloc[0]),
        "seed": int(frame["seed"].iloc[0]),
        "hours": list(hours),
        "vehicles": list(vehicles),
        "methods": list(EXPECTED_METHODS),
        "target_total_demand_kg": target_total_demand_kg,
        "observations": len(frame),
        "all_feasible": True,
        "all_exact_nonempty_k": True,
        "proposed_mean_energy_optimum_k": int(proposed_optimum["vehicles"]),
        "proposed_mean_energy_optimum_kwh": float(
            proposed_optimum["mean_energy_kwh"]
        ),
    }
    (root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return frame


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hours", type=int, nargs="+", required=True)
    parser.add_argument("--vehicles", type=int, nargs="+", required=True)
    parser.add_argument("--target-total-demand-kg", type=float, required=True)
    args = parser.parse_args(arguments)
    frame = aggregate(
        args.root.resolve(), tuple(args.hours), tuple(args.vehicles),
        args.target_total_demand_kg,
    )
    print(args.root.resolve() / "k_sweep_3h_all_methods.csv")
    print(f"validated observations: {len(frame)}")


if __name__ == "__main__":
    main()
