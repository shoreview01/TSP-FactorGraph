"""Create IEEE-style hourly proposed-versus-baseline comparison figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        return str(resolved)


EXPECTED_METHODS = ("proposed", "nn", "aco", "pso", "ga")
METHOD_LABELS = {
    "proposed": "Proposed",
    "nn": "NN",
    "aco": "ACO",
    "pso": "PSO",
    "ga": "GA",
}
METHOD_STYLES = {
    "proposed": {"marker": "*", "linestyle": "-", "color": "#000000"},
    "nn": {"marker": "o", "linestyle": "-", "color": "#0072B2"},
    "aco": {"marker": "s", "linestyle": "--", "color": "#D55E00"},
    "pso": {"marker": "^", "linestyle": "-.", "color": "#009E73"},
    "ga": {"marker": "D", "linestyle": ":", "color": "#CC79A7"},
}
METRICS = {
    "energy": ("energy_kwh", "Energy consumption (kWh)"),
    "battery_remaining": (
        "battery_remaining_mean_kwh", "Mean remaining battery (kWh/vehicle)"),
    "makespan": ("makespan_s", "Makespan (s)"),
}


def _configure_ieee_style() -> None:
    """Apply a compact IEEE-compatible serif style with embedded TrueType text."""

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "grid.linewidth": 0.4,
        "grid.alpha": 1.0,
        "grid.color": "0.88",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _validated_frame(input_csv: Path | list[Path]) -> pd.DataFrame:
    paths = [input_csv] if isinstance(input_csv, Path) else list(input_csv)
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {
        "method", "start_hour", "energy_kwh",
        "battery_remaining_mean_kwh", "makespan_s", "feasible",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"hourly results are missing columns: {sorted(missing)}")
    frame = frame[frame["method"].isin(EXPECTED_METHODS)].copy()
    duplicates = frame.duplicated(["method", "start_hour"], keep=False)
    if duplicates.any():
        rows = frame.loc[duplicates, ["method", "start_hour"]].to_dict("records")
        raise ValueError(f"duplicate method/hour observations: {rows[:4]}")
    expected = {(method, hour) for method in EXPECTED_METHODS for hour in range(24)}
    observed = set(zip(frame["method"], frame["start_hour"].astype(int)))
    absent = sorted(expected - observed)
    if absent:
        raise ValueError(f"hourly experiment is incomplete; missing {absent[:8]}")
    if not frame["feasible"].astype(bool).all():
        failed = frame.loc[~frame["feasible"].astype(bool),
                           ["method", "start_hour"]].to_dict("records")
        raise ValueError(f"infeasible runs cannot be plotted: {failed}")
    return frame.sort_values(["method", "start_hour"])


def _draw_metric(frame: pd.DataFrame, column: str, ylabel: str,
                 output_stem: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    for method in EXPECTED_METHODS:
        part = frame[frame["method"] == method].sort_values("start_hour")
        ax.plot(
            part["start_hour"], part[column], label=METHOD_LABELS[method],
            markevery=1, markerfacecolor="white", markeredgewidth=0.75,
            **METHOD_STYLES[method],
        )
    ax.set_xlabel("Starting hour")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax.grid(True, which="major", axis="both")
    ax.tick_params(direction="in", top=True, right=True, width=0.6)
    ax.legend(
        ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.015),
        frameon=False, handlelength=2.3, columnspacing=1.0,
    )
    paths = []
    for suffix in ("pdf", "eps", "png"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        paths.append(path)
    plt.close(fig)
    return paths


def _draw_combined(frame: pd.DataFrame, output_stem: Path) -> list[Path]:
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 6.7), sharex=True,
                             constrained_layout=True)
    panel_labels = ("(a)", "(b)", "(c)")
    for ax, (metric, (column, ylabel)), panel in zip(
            axes, METRICS.items(), panel_labels):
        del metric
        for method in EXPECTED_METHODS:
            part = frame[frame["method"] == method].sort_values("start_hour")
            ax.plot(
                part["start_hour"], part[column], label=METHOD_LABELS[method],
                markerfacecolor="white", markeredgewidth=0.75,
                **METHOD_STYLES[method],
            )
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 23)
        ax.grid(True, which="major", axis="both")
        ax.tick_params(direction="in", top=True, right=True, width=0.6)
        ax.text(0.02, 0.93, panel, transform=ax.transAxes,
                ha="left", va="top")
    axes[-1].set_xlabel("Starting hour")
    axes[-1].set_xticks([0, 4, 8, 12, 16, 20, 23])
    axes[0].legend(
        ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.015),
        frameon=False, handlelength=2.3, columnspacing=1.0,
    )
    paths = []
    for suffix in ("pdf", "eps", "png"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        paths.append(path)
    plt.close(fig)
    return paths


def create_figures(input_csv: Path | list[Path], output_dir: Path,
                   merged_output: Path | None = None) -> list[Path]:
    """Validate 120 hourly runs and export three figures plus a combined panel."""

    _configure_ieee_style()
    frame = _validated_frame(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    if merged_output is not None:
        merged_output.parent.mkdir(parents=True, exist_ok=True)
        frame.sort_values(["start_hour", "method"]).to_csv(
            merged_output, index=False)
    outputs: list[Path] = []
    for name, (column, ylabel) in METRICS.items():
        outputs.extend(_draw_metric(
            frame, column, ylabel, output_dir / f"hourly_{name}_ieee"))
    outputs.extend(_draw_combined(frame, output_dir / "hourly_comparison_ieee"))

    summary = (frame.groupby("method", as_index=False)
               .agg(energy_mean_kwh=("energy_kwh", "mean"),
                    energy_std_kwh=("energy_kwh", "std"),
                    battery_remaining_mean_kwh=(
                        "battery_remaining_mean_kwh", "mean"),
                    battery_remaining_min_kwh=(
                        "battery_remaining_min_kwh", "min"),
                    makespan_mean_s=("makespan_s", "mean"),
                    makespan_std_s=("makespan_s", "std"),
                    feasible_rate=("feasible", "mean")))
    summary["method"] = pd.Categorical(
        summary["method"], categories=EXPECTED_METHODS, ordered=True)
    summary = summary.sort_values("method")
    if merged_output is not None:
        summary.to_csv(
            merged_output.with_name(f"{merged_output.stem}_summary.csv"),
            index=False,
        )

    manifest = {
        "source": ([_portable_path(path) for path in input_csv]
                   if isinstance(input_csv, list)
                   else _portable_path(input_csv)),
        "observations": int(len(frame)),
        "hours": list(range(24)),
        "methods": list(EXPECTED_METHODS),
        "battery_metric": (
            "mean across vehicles of initial battery minus route energy"),
        "formats": ["PDF", "EPS", "PNG"],
        "figure_width_in": 3.5,
        "raster_dpi": 600,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, nargs="+",
        default=[PACKAGE_ROOT / "results" / "hourly_24h"
                 / "hourly_dual_path_with_baselines_seed7.csv"])
    parser.add_argument(
        "--output-dir", type=Path,
        default=PACKAGE_ROOT / "figures" / "hourly_24h")
    parser.add_argument("--merged-output", type=Path)
    return parser


def main(arguments: list[str] | None = None) -> None:
    args = build_parser().parse_args(arguments)
    for path in create_figures(
            args.input, args.output_dir, merged_output=args.merged_output):
        print(path)


if __name__ == "__main__":
    main()
