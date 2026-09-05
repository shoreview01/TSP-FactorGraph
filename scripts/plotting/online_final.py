"""Standalone IEEE-style figures for the final online experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ieee import METHOD_LABELS, METHOD_STYLES, _configure_ieee_style


POLICY_STYLES = (
    ("static_frozen", "", "#C1C9D2", "Frozen static"),
    ("static_live", "", "#788DA3", "Live-navigation static"),
    ("adaptive", "", "#2F4B61", "Adaptive"),
)


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = []
    for suffix in ("pdf", "eps", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _method_order(frame: pd.DataFrame) -> list[str]:
    methods = [
        method for method in METHOD_LABELS
        if method in set(frame["method"])
    ]
    if not methods:
        raise ValueError("no supported methods were found")
    return methods


def _lightened(color: str, amount: float = 0.84) -> tuple[float, ...]:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple(rgb + (1.0 - rgb) * amount)


def _boxed_legend(axis: plt.Axes, **kwargs: object) -> None:
    legend = axis.legend(
        frameon=True, fancybox=False, facecolor="white", edgecolor="0.15",
        framealpha=1.0, borderpad=0.45, **kwargs)
    legend.get_frame().set_linewidth(0.7)


def _snapshot_energy_figure(aligned: pd.DataFrame, methods: list[str],
                            stem: Path) -> list[Path]:
    required = {
        "traffic_seed", "method", "policy", "epoch", "elapsed_min",
        "cumulative_energy_kwh", "epoch_energy_kwh",
        "planned_remaining_energy_kwh",
    }
    missing = required - set(aligned)
    if missing:
        raise ValueError(f"aligned epoch data are missing {sorted(missing)}")
    adaptive = aligned[aligned["policy"] == "adaptive"].copy()
    # Route evaluation precedes the current one-minute execution step.  Remove
    # that step from cumulative energy before adding the residual plan energy.
    adaptive["snapshot_total_energy_kwh"] = (
        adaptive["cumulative_energy_kwh"]
        - adaptive["epoch_energy_kwh"]
        + adaptive["planned_remaining_energy_kwh"])
    stats = (adaptive.groupby(
        ["method", "epoch", "elapsed_min"], as_index=False)
        ["snapshot_total_energy_kwh"]
        .agg(["mean", "min", "max"])
        .reset_index())

    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    for method in methods:
        part = stats[stats["method"] == method].sort_values("epoch")
        if part.empty:
            continue
        style = METHOD_STYLES[method]
        band = axis.fill_between(
            part["elapsed_min"], part["min"], part["max"],
            facecolor=_lightened(style["color"]), edgecolor="none",
            alpha=0.38, zorder=1)
        band.set_rasterized(True)
        axis.plot(
            part["elapsed_min"], part["mean"],
            label=METHOD_LABELS[method], markerfacecolor="white",
            markeredgewidth=0.75, markevery=max(1, len(part) // 12),
            zorder=3, **style)
    for minute in (10.0, 20.0, 30.0, 55.0):
        axis.axvline(minute, color="0.72", linewidth=0.55,
                    linestyle=":", zorder=0)
    axis.set_xlabel("Elapsed time (min)")
    axis.set_ylabel("Snapshot-estimated total energy (kWh)")
    axis.grid(True)
    axis.tick_params(direction="in", top=True, right=True, width=0.6)
    top = max(float(stats["max"].max()) * 1.06,
              float(stats["mean"].max()) * 1.18)
    axis.set_ylim(top=top)
    _boxed_legend(
        axis, ncol=2, loc="upper left", handlelength=2.3,
        columnspacing=1.0)
    return _save(fig, stem)


def _energy_bar_figure(final_stats: pd.DataFrame, methods: list[str],
                       stem: Path) -> list[Path]:
    required = {"method", "policy"}
    for suffix in ("mean", "std"):
        required.add(f"cumulative_energy_kwh_{suffix}")
    missing = required - set(final_stats)
    if missing:
        raise ValueError(f"final statistics are missing {sorted(missing)}")

    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    positions = np.arange(len(methods), dtype=float)
    width = 0.25
    for policy_index, (policy, hatch, color, label) in enumerate(
            POLICY_STYLES):
        offset = (policy_index - 1) * width
        part = final_stats[final_stats["policy"] == policy].set_index(
            "method")
        means = np.asarray([
            part.at[method, "cumulative_energy_kwh_mean"]
            for method in methods], dtype=float)
        standard_deviations = np.asarray([
            part.at[method, "cumulative_energy_kwh_std"]
            for method in methods], dtype=float)
        axis.bar(
            positions + offset, means, width=width, facecolor=color,
            edgecolor="0.20", hatch=hatch, linewidth=0.65, label=label,
            zorder=2)
        axis.errorbar(
            positions + offset, means,
            yerr=standard_deviations,
            fmt="none", ecolor="0.15", elinewidth=0.7,
            capsize=2.0, capthick=0.7, zorder=3)
    axis.set_ylabel("Final operational energy (kWh)")
    axis.set_xticks(
        positions, [METHOD_LABELS[method] for method in methods])
    axis.set_axisbelow(True)
    axis.grid(True, axis="y")
    axis.tick_params(direction="in", top=True, right=True, width=0.6)
    highest = float((
        final_stats["cumulative_energy_kwh_mean"]
        + final_stats["cumulative_energy_kwh_std"]).max())
    axis.set_ylim(top=highest * 1.28)
    _boxed_legend(
        axis, ncol=3, loc="upper center", columnspacing=0.9,
        handlelength=2.0)
    return _save(fig, stem)


def _latency_figure(final_stats: pd.DataFrame, methods: list[str],
                    stem: Path) -> list[Path]:
    required = {"method", "policy"}
    for suffix in ("mean", "std"):
        required.add(f"replanning_runtime_s_{suffix}")
    missing = required - set(final_stats)
    if missing:
        raise ValueError(f"final statistics are missing {sorted(missing)}")
    adaptive = final_stats[final_stats["policy"] == "adaptive"].set_index(
        "method")
    means = np.asarray([
        adaptive.at[method, "replanning_runtime_s_mean"]
        for method in methods], dtype=float)
    standard_deviations = np.asarray([
        adaptive.at[method, "replanning_runtime_s_std"]
        for method in methods], dtype=float)

    fig, axis = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    positions = np.arange(len(methods), dtype=float)
    colors = [METHOD_STYLES[method]["color"] for method in methods]
    axis.bar(
        positions, means, width=0.62, color=colors,
        edgecolor="0.15", linewidth=0.7)
    axis.errorbar(
        positions, means, yerr=standard_deviations,
        fmt="none", ecolor="0.15", elinewidth=0.7,
        capsize=2.0, capthick=0.7)
    axis.set_ylabel("Cumulative replanning latency (s)")
    axis.set_xticks(positions, [METHOD_LABELS[method] for method in methods])
    axis.set_ylim(top=float((means + standard_deviations).max()) * 1.16)
    axis.set_axisbelow(True)
    axis.grid(True, axis="y")
    axis.tick_params(direction="in", top=True, right=True, width=0.6)

    zoom_count = min(3, len(methods))
    zoom = axis.inset_axes([0.075, 0.57, 0.47, 0.37])
    zoom_positions = np.arange(zoom_count, dtype=float)
    zoom.bar(
        zoom_positions, means[:zoom_count], width=0.62,
        color=colors[:zoom_count], edgecolor="0.15", linewidth=0.65,
        zorder=2)
    zoom.errorbar(
        zoom_positions, means[:zoom_count],
        yerr=standard_deviations[:zoom_count], fmt="none",
        ecolor="0.15", elinewidth=0.65, capsize=1.6, capthick=0.65,
        zorder=3)
    zoom.set_xlim(-0.5, zoom_count - 0.5)
    zoom.set_ylim(
        0.0, float((means[:zoom_count]
                    + standard_deviations[:zoom_count]).max()) * 1.12)
    zoom.set_xticks(
        zoom_positions,
        [METHOD_LABELS[method] for method in methods[:zoom_count]])
    zoom.text(
        0.04, 0.94, "Mean $\\pm$ SD (10 seeds)",
        transform=zoom.transAxes, ha="left", va="top", fontsize=6.0)
    zoom.set_axisbelow(True)
    zoom.grid(True, axis="y")
    zoom.tick_params(direction="in", top=True, right=True, width=0.5,
                     labelsize=6.0)
    for spine in zoom.spines.values():
        spine.set_linewidth(0.6)
    return _save(fig, stem)


def create_standalone_figures(aligned_epochs_csv: Path,
                              final_stats_csv: Path,
                              output_dir: Path) -> list[Path]:
    _configure_ieee_style()
    aligned = pd.read_csv(aligned_epochs_csv)
    final_stats = pd.read_csv(final_stats_csv)
    methods = _method_order(final_stats)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.extend(_snapshot_energy_figure(
        aligned, methods,
        output_dir / "online_snapshot_total_energy_ieee"))
    outputs.extend(_energy_bar_figure(
        final_stats, methods,
        output_dir / "online_final_energy_bar_ieee"))
    outputs.extend(_latency_figure(
        final_stats, methods,
        output_dir / "online_replanning_latency_ieee"))
    seed_count = int(aligned["traffic_seed"].nunique())
    manifest = {
        "seed_count": seed_count,
        "methods": methods,
        "formats": ["PDF", "EPS", "PNG"],
        "snapshot_total_energy_definition": (
            "cumulative_energy_kwh - epoch_energy_kwh + "
            "planned_remaining_energy_kwh"),
        "snapshot_line": "arithmetic mean across traffic seeds",
        "snapshot_cloud": "minimum-to-maximum across traffic seeds",
        "energy_bar": "mean with one-standard-deviation error bars",
        "latency_bar": (
            "mean cumulative adaptive replanning wall time per run with "
            "one-standard-deviation error bars; initial planning excluded; "
            "inset enlarges Proposed, NN, and ACO"),
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-epochs", type=Path, required=True)
    parser.add_argument("--final-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(arguments: list[str] | None = None) -> None:
    args = build_parser().parse_args(arguments)
    for path in create_standalone_figures(
            args.aligned_epochs, args.final_stats, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
