"""Create IEEE-style three-hour exact-K comparison figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .ieee import (
    EXPECTED_METHODS,
    METHOD_LABELS,
    METHOD_STYLES,
    _configure_ieee_style,
)


DEFAULT_HOURS = (4, 19, 11)
DEFAULT_VEHICLES = tuple(range(7, 15))
METRICS = {
    "energy": ("energy_kwh", "Energy consumption (kWh)"),
    "battery_remaining": (
        "battery_remaining_mean_kwh",
        "Mean remaining battery (kWh/vehicle)",
    ),
    "makespan": ("makespan_s", "Makespan (s)"),
}


def _validated_frame(
    path: Path,
    hours: tuple[int, ...],
    vehicles: tuple[int, ...],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "method", "start_hour", "vehicles", "energy_kwh",
        "battery_remaining_mean_kwh", "makespan_s", "feasible",
        "active_vehicles",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"K-sweep results are missing {sorted(missing)}")
    frame = frame[frame["method"].isin(EXPECTED_METHODS)].copy()
    expected = {
        (method, hour, vehicle_count)
        for method in EXPECTED_METHODS
        for hour in hours
        for vehicle_count in vehicles
    }
    observed = set(zip(
        frame["method"], frame["start_hour"].astype(int),
        frame["vehicles"].astype(int),
    ))
    if observed != expected:
        raise ValueError(
            f"incomplete K sweep: missing={sorted(expected-observed)[:5]}, "
            f"extra={sorted(observed-expected)[:5]}")
    if frame.duplicated(["method", "start_hour", "vehicles"]).any():
        raise ValueError("duplicate method/hour/K observations")
    if not frame["feasible"].astype(bool).all():
        raise ValueError("infeasible observation in K sweep")
    if not (frame["active_vehicles"].astype(int)
            == frame["vehicles"].astype(int)).all():
        raise ValueError("active vehicle count differs from exact K")
    return frame.sort_values(["start_hour", "vehicles", "method"])


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = []
    for suffix in ("pdf", "eps", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _draw_metric(
    frame: pd.DataFrame,
    column: str,
    ylabel: str,
    stem: Path,
    hours: tuple[int, ...],
    vehicles: tuple[int, ...],
) -> list[Path]:
    fig, axes = plt.subplots(
        1, len(hours), figsize=(7.16, 2.55), sharex=True,
    )
    if len(hours) == 1:
        axes = [axes]
    tick_values = list(vehicles)
    if len(tick_values) > 6:
        tick_values = tick_values[::2]
        if tick_values[-1] != vehicles[-1]:
            tick_values.append(vehicles[-1])
    for index, (axis, hour) in enumerate(zip(axes, hours)):
        for method in EXPECTED_METHODS:
            part = frame[
                (frame["start_hour"].astype(int) == hour)
                & (frame["method"] == method)
            ].sort_values("vehicles")
            axis.plot(
                part["vehicles"], part[column],
                label=METHOD_LABELS[method], markerfacecolor="white",
                markeredgewidth=0.75, **METHOD_STYLES[method],
            )
        axis.set_title(f"{hour:02d}:00")
        axis.set_xlabel("Exact active vehicles, K")
        axis.set_xticks(tick_values)
        axis.grid(True)
        axis.tick_params(direction="in", top=True, right=True, width=0.6)
        axis.text(0.03, 0.94, f"({chr(ord('a') + index)})",
                  transform=axis.transAxes, ha="left", va="top")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=5, loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False, handlelength=2.2, columnspacing=0.9,
    )
    fig.subplots_adjust(
        left=0.085, right=0.995, bottom=0.205, top=0.735, wspace=0.34)
    return _save(fig, stem)


def create_figures(
    input_csv: Path,
    output_dir: Path,
    hours: tuple[int, ...] = DEFAULT_HOURS,
    vehicles: tuple[int, ...] = DEFAULT_VEHICLES,
) -> list[Path]:
    _configure_ieee_style()
    frame = _validated_frame(input_csv, hours, vehicles)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, (column, ylabel) in METRICS.items():
        outputs.extend(_draw_metric(
            frame, column, ylabel, output_dir / f"k_sweep_3h_{name}_ieee",
            hours, vehicles))
    manifest = {
        "source": input_csv.resolve().relative_to(
            Path(__file__).resolve().parents[2]).as_posix(),
        "observations": int(len(frame)),
        "hours": list(hours),
        "vehicles": list(vehicles),
        "methods": list(EXPECTED_METHODS),
        "formats": ["PDF", "EPS", "PNG"],
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hours", type=int, nargs="+", default=DEFAULT_HOURS)
    parser.add_argument(
        "--vehicles", type=int, nargs="+", default=DEFAULT_VEHICLES)
    args = parser.parse_args(arguments)
    for path in create_figures(
        args.input, args.output_dir, tuple(args.hours), tuple(args.vehicles)):
        print(path)


if __name__ == "__main__":
    main()
