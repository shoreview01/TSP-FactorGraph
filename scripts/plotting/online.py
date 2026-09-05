"""IEEE-style tracking figures for rolling-horizon online experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ieee import (
    METHOD_LABELS,
    METHOD_STYLES,
    _configure_ieee_style,
)


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = []
    for suffix in ("pdf", "eps", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _validate(epoch_csv: Path, traffic_csv: Path
              ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frame = pd.read_csv(epoch_csv)
    traffic = pd.read_csv(traffic_csv)
    required = {
        "method", "policy", "epoch", "elapsed_min",
        "cumulative_energy_kwh",
        "cumulative_planning_runtime_s", "reassigned_bins",
        "latency_adjusted_energy_kwh", "latency_adjusted_time_s",
        "remaining_bins_before", "remaining_bins_after",
        "epoch_energy_kwh", "planned_remaining_energy_kwh",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"online epoch results are missing {sorted(missing)}")
    if not {"epoch", "elapsed_min", "disruption_region_speed_kmh",
            "affected_edges"} <= set(traffic):
        raise ValueError("traffic trace is missing required tracking columns")
    methods = [method for method in METHOD_LABELS
               if method in set(frame["method"])]
    if not methods:
        raise ValueError("online results contain no supported methods")
    return frame, traffic, methods


def _tracking_figure(frame: pd.DataFrame, traffic: pd.DataFrame,
                     methods: list[str], stem: Path) -> list[Path]:
    max_observed_minute = float(frame["elapsed_min"].max())
    traffic = traffic[
        traffic["elapsed_min"] <= max_observed_minute].copy()
    fig, axes = plt.subplots(
        4, 1, figsize=(3.5, 7.9), sharex=True,
        constrained_layout=True)

    axis = axes[0]
    axis.plot(
        traffic["elapsed_min"], traffic["disruption_region_speed_kmh"],
        color="0.10",
        label="Disruption-region speed")
    axis.set_ylabel("Region speed (km/h)")
    axis.grid(True)
    right = axis.twinx()
    right.plot(
        traffic["elapsed_min"], traffic["affected_edges"], color="0.55",
        linestyle="--", label="Affected edges")
    right.set_ylabel("Affected edges")
    axis.text(0.02, 0.92, "(a)", transform=axis.transAxes,
              ha="left", va="top")

    adaptive = frame[frame["policy"] == "adaptive"]
    for method in methods:
        part = adaptive[adaptive["method"] == method].sort_values("epoch")
        markevery = max(1, len(part) // 12)
        axes[1].plot(
            part["elapsed_min"], part["cumulative_energy_kwh"],
            label=METHOD_LABELS[method], markerfacecolor="white",
            markeredgewidth=0.75, markevery=markevery,
            **METHOD_STYLES[method])
        axes[2].plot(
            part["elapsed_min"], part["cumulative_planning_runtime_s"],
            label=METHOD_LABELS[method], markerfacecolor="white",
            markeredgewidth=0.75, markevery=markevery,
            **METHOD_STYLES[method])
        axes[3].plot(
            part["elapsed_min"], part["reassigned_bins"],
            label=METHOD_LABELS[method], markerfacecolor="white",
            markeredgewidth=0.75, markevery=markevery,
            **METHOD_STYLES[method])

    axes[1].set_ylabel("Cumulative energy (kWh)")
    axes[2].set_ylabel("Planning time (s)")
    axes[2].set_yscale("symlog", linthresh=1.0)
    axes[3].set_ylabel("Reassigned bins")
    axes[3].set_xlabel("Elapsed time (min)")
    for panel, axis in zip(("(b)", "(c)", "(d)"), axes[1:]):
        axis.grid(True)
        axis.tick_params(direction="in", top=True, right=True, width=0.6)
        panel_y = 0.10 if panel == "(c)" else 0.92
        panel_x = 0.97 if panel == "(c)" else 0.02
        panel_ha = "right" if panel == "(c)" else "left"
        axis.text(panel_x, panel_y, panel, transform=axis.transAxes,
                  ha=panel_ha, va="top")
    axes[1].legend(
        ncol=min(3, len(methods)), loc="lower center",
        bbox_to_anchor=(0.5, 1.015), frameon=False,
        handlelength=2.3, columnspacing=1.0)
    return _save(fig, stem)


def _progress_diagnostics_figure(frame: pd.DataFrame, methods: list[str],
                                 stem: Path) -> list[Path]:
    adaptive = frame[frame["policy"] == "adaptive"].copy()
    adaptive["planned_energy_per_remaining_bin_kwh"] = (
        adaptive["planned_remaining_energy_kwh"]
        / adaptive["remaining_bins_before"].clip(lower=1))
    # plan_evaluation is measured before executing the committed prefix.  Use
    # energy incurred before this epoch to avoid counting that prefix twice.
    adaptive["projected_completion_energy_kwh"] = (
        adaptive["cumulative_energy_kwh"]
        - adaptive["epoch_energy_kwh"]
        + adaptive["planned_remaining_energy_kwh"])

    fig, axes = plt.subplots(
        1, 3, figsize=(7.16, 2.55), constrained_layout=True)
    for method in methods:
        part = adaptive[adaptive["method"] == method].sort_values("epoch")
        if part.empty:
            continue
        initial_bins = int(part.iloc[0]["remaining_bins_before"])
        served = initial_bins - part["remaining_bins_after"].to_numpy()
        progress_x = np.r_[0, served]
        progress_y = np.r_[0.0, part["cumulative_energy_kwh"].to_numpy()]
        style = METHOD_STYLES[method]
        common = {
            "label": METHOD_LABELS[method],
            "markerfacecolor": "white",
            "markeredgewidth": 0.75,
            "markevery": max(1, len(part) // 12),
            **style,
        }
        axes[0].plot(progress_x, progress_y, **common)
        axes[1].plot(
            part["elapsed_min"],
            part["planned_energy_per_remaining_bin_kwh"],
            **common)
        axes[2].plot(
            part["elapsed_min"], part["projected_completion_energy_kwh"],
            **common)

    axes[0].set_xlabel("Cumulative served bins")
    axes[0].set_ylabel("Cumulative energy (kWh)")
    axes[1].set_xlabel("Elapsed time (min)")
    axes[1].set_ylabel("Planned residual energy\n(kWh/bin)")
    axes[2].set_xlabel("Elapsed time (min)")
    axes[2].set_ylabel("Snapshot-estimated total\nenergy (kWh)")
    for index, axis in enumerate(axes):
        axis.grid(True)
        axis.tick_params(direction="in", top=True, right=True, width=0.6)
        axis.text(0.03, 0.95, f"({chr(ord('a') + index)})",
                  transform=axis.transAxes, ha="left", va="top")
    axes[0].legend(
        ncol=min(3, len(methods)), loc="lower center",
        bbox_to_anchor=(1.62, 1.03), frameon=False,
        handlelength=2.3, columnspacing=1.0)
    return _save(fig, stem)


def _final_figure(frame: pd.DataFrame, methods: list[str],
                  stem: Path) -> list[Path]:
    final = (frame.sort_values("epoch")
             .groupby(["method", "policy"], as_index=False)
             .tail(1))
    fig, axes = plt.subplots(
        1, 2, figsize=(7.16, 2.55), constrained_layout=True)
    positions = np.arange(len(methods), dtype=float)
    width = 0.25
    for offset, policy, hatch, color, label in (
            (-width, "static_frozen", "//", "white", "Frozen static"),
            (0.0, "static_live", "..", "0.72", "Live-navigation static"),
            (width, "adaptive", "", "0.35", "Adaptive")):
        part = final[final["policy"] == policy].set_index("method")
        energy = [part.at[method, "cumulative_energy_kwh"]
                  for method in methods]
        planning = [part.at[method, "cumulative_planning_runtime_s"]
                    for method in methods]
        axes[0].bar(
            positions + offset, energy, width=width,
            facecolor=color,
            edgecolor="0.15", hatch=hatch, linewidth=0.7,
            label=label)
        axes[1].bar(
            positions + offset, planning, width=width,
            facecolor=color,
            edgecolor="0.15", hatch=hatch, linewidth=0.7,
            label=label)
    axes[0].set_ylabel("Realized energy (kWh)")
    axes[1].set_ylabel("Planning latency (s)")
    for index, axis in enumerate(axes):
        axis.set_xticks(positions, [METHOD_LABELS[method]
                                   for method in methods], rotation=20)
        axis.grid(True, axis="y")
        axis.tick_params(direction="in", top=True, right=True, width=0.6)
        axis.text(0.02, 0.94, f"({chr(ord('a') + index)})",
                  transform=axis.transAxes, ha="left", va="top")
    axes[0].legend(
        ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.015),
        frameon=False)
    return _save(fig, stem)


def create_online_figures(epoch_csv: Path, traffic_csv: Path,
                          output_dir: Path) -> list[Path]:
    _configure_ieee_style()
    frame, traffic, methods = _validate(epoch_csv, traffic_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.extend(_tracking_figure(
        frame, traffic, methods, output_dir / "online_tracking_ieee"))
    outputs.extend(_progress_diagnostics_figure(
        frame, methods, output_dir / "online_progress_diagnostics_ieee"))
    outputs.extend(_final_figure(
        frame, methods, output_dir / "online_final_comparison_ieee"))
    manifest = {
        "epoch_source": str(epoch_csv.resolve()),
        "traffic_source": str(traffic_csv.resolve()),
        "methods": methods,
        "formats": ["PDF", "EPS", "PNG"],
        "tracking_panels": [
            "traffic shock", "adaptive cumulative energy",
            "adaptive planning time", "adaptive reassignment count",
        ],
        "progress_diagnostic_panels": [
            "cumulative energy versus cumulative served bins",
            "planned residual energy per remaining bin",
            "snapshot-estimated total energy without current-step double count",
        ],
        "snapshot_estimated_total_energy_definition": (
            "cumulative_energy_kwh - epoch_energy_kwh + "
            "planned_remaining_energy_kwh"),
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=Path, required=True)
    parser.add_argument("--traffic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(arguments)
    for path in create_online_figures(
            args.epochs, args.traffic, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
