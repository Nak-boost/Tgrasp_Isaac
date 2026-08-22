#!/usr/bin/env python3
"""Plot comparable training curves for the tactile E1--E5 runs."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PLOT_GROUPS = {
    "optimization": [
        ("Loss/value_function", "Value function loss"),
        ("Loss/surrogate", "PPO surrogate loss"),
        ("Loss/upward_action_supervision", "Upward action supervision loss"),
        ("Policy/mean_noise_std", "Mean action noise std"),
    ],
    "task_performance": [
        ("Train/mean_reward", "Mean episode reward"),
        ("Train2/mean_reward/step", "Mean reward per step"),
        ("Train/mean_episode_length", "Mean episode length"),
        ("Train/mean_success", "Mean success rate (%)"),
    ],
    "tactile_performance": [
        ("Tactile/contact_found", "Contact discovery rate (%)"),
        ("Tactile/first_contact_step", "Mean first-contact step"),
        ("Tactile/multi_contact", "Multi-contact episode rate (%)"),
        ("Tactile/contact_step_ratio", "Contact-step ratio (%)"),
        ("Tactile/contact_losses", "Contact losses per episode"),
        ("Tactile/lifted", "Lift episode rate (%)"),
    ],
}

ALL_TAGS = [tag for metrics in PLOT_GROUPS.values() for tag, _ in metrics]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot E1--E5 TensorBoard scalars. TensorBoard event files are "
            "preferred because they preserve the real training iteration."
        )
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("dexgrasp/logs/tactile_0818"),
        help="Directory containing E1_seed*, ..., E5_seed* run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <log-root>/plots.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["E1", "E2", "E3", "E4", "E5"],
        help="Experiment prefixes to plot.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=101,
        help="Centered moving-average window in logged iterations; 1 disables smoothing.",
    )
    parser.add_argument(
        "--summary-window",
        type=int,
        default=500,
        help="Number of final iterations used for metrics_summary.csv.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png"],
        help="One or more output figure formats.",
    )
    return parser.parse_args()


def find_runs(log_root, experiments):
    runs = []
    for experiment in experiments:
        matches = sorted(path for path in log_root.glob(f"{experiment}_*") if path.is_dir())
        direct_path = log_root / experiment
        if not matches and direct_path.is_dir():
            matches = [direct_path]
        if not matches:
            print(f"Warning: no run directory found for {experiment} under {log_root}")
        runs.extend(matches)
    return runs


def load_tensorboard(run_dir):
    event_dir = run_dir / "logger"
    if not event_dir.is_dir() or not any(event_dir.glob("events.out.tfevents.*")):
        return None

    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available_tags = set(accumulator.Tags().get("scalars", []))
    data = {}

    for tag in ALL_TAGS:
        if tag not in available_tags:
            continue
        latest_by_step = {}
        for event in accumulator.Scalars(tag):
            previous = latest_by_step.get(event.step)
            if previous is None or event.wall_time >= previous.wall_time:
                latest_by_step[event.step] = event
        events = [latest_by_step[step] for step in sorted(latest_by_step)]
        steps = np.asarray([event.step for event in events], dtype=np.int64)
        values = np.asarray([event.value for event in events], dtype=np.float64)
        valid = np.isfinite(values)
        if np.any(valid):
            data[tag] = (steps[valid], values[valid])

    return data if data else None


def load_csv(run_dir):
    csv_path = run_dir / "log_train.csv"
    if not csv_path.is_file():
        return None

    frame = pd.read_csv(csv_path)
    unnamed_columns = [column for column in frame.columns if column.startswith("Unnamed:")]
    if unnamed_columns:
        steps = pd.to_numeric(frame[unnamed_columns[0]], errors="coerce").to_numpy()
    else:
        steps = np.arange(len(frame), dtype=np.int64)

    data = {}
    for tag in ALL_TAGS:
        if tag not in frame.columns:
            continue
        values = pd.to_numeric(frame[tag], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(steps) & np.isfinite(values)
        if np.any(valid):
            data[tag] = (steps[valid].astype(np.int64), values[valid])
    return data if data else None


def load_run(run_dir):
    data = load_tensorboard(run_dir)
    if data is not None:
        return data, "tensorboard"
    data = load_csv(run_dir)
    if data is not None:
        print(
            f"Warning: {run_dir.name} has no readable TensorBoard events; "
            "CSV row numbers are used as approximate iterations."
        )
        return data, "csv"
    raise RuntimeError(f"No readable training scalars found in {run_dir}")


def smooth_values(values, window):
    if window <= 1 or len(values) <= 1:
        return values
    window = min(window, len(values))
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def plot_group(group_name, metrics, runs, output_dir, smooth_window, formats, dpi):
    columns = 2
    rows = math.ceil(len(metrics) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(13, 3.8 * rows), squeeze=False)
    axes = axes.ravel()
    colors = plt.get_cmap("tab10")
    legend_handles = []
    legend_labels = []

    for metric_index, (tag, title) in enumerate(metrics):
        axis = axes[metric_index]
        plotted = False
        for run_index, run in enumerate(runs):
            if tag not in run["data"]:
                continue
            steps, values = run["data"][tag]
            color = colors(run_index % 10)
            if smooth_window > 1:
                axis.plot(steps, values, color=color, alpha=0.10, linewidth=0.6)
            line = axis.plot(
                steps,
                smooth_values(values, smooth_window),
                color=color,
                linewidth=1.8,
                label=run["label"],
            )[0]
            if run["label"] not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(run["label"])
            plotted = True

        axis.set_title(title)
        axis.set_xlabel("Learning iteration")
        axis.grid(True, alpha=0.25)
        if not plotted:
            axis.text(0.5, 0.5, "Metric not logged", ha="center", va="center")

    for axis in axes[len(metrics):]:
        axis.set_visible(False)

    if legend_handles:
        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(5, len(legend_labels)),
            frameon=False,
        )
    figure.suptitle(
        f"Tactile grasp training: {group_name.replace('_', ' ')}",
        y=0.995,
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    for output_format in formats:
        output_path = output_dir / f"{group_name}.{output_format}"
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {output_path}")
    plt.close(figure)


def write_summary(runs, output_path, final_window):
    rows = []
    for run in runs:
        for tag in ALL_TAGS:
            if tag not in run["data"]:
                continue
            steps, values = run["data"][tag]
            final_step = int(steps[-1])
            window_start = final_step - max(final_window - 1, 0)
            final_values = values[steps >= window_start]
            rows.append(
                {
                    "run": run["label"],
                    "source": run["source"],
                    "metric": tag,
                    "last_iteration": final_step,
                    "last_value": float(values[-1]),
                    "final_window_mean": float(np.mean(final_values)),
                    "final_window_std": float(np.std(final_values)),
                    "window_iterations": final_window,
                }
            )

    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    if args.smooth < 1 or args.summary_window < 1:
        raise ValueError("--smooth and --summary-window must be positive integers")

    log_root = args.log_root.resolve()
    output_dir = (args.output_dir or log_root / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_runs(log_root, args.experiments)
    if not run_dirs:
        raise RuntimeError(f"No experiment runs found under {log_root}")

    runs = []
    for run_dir in run_dirs:
        data, source = load_run(run_dir)
        last_step = max(int(steps[-1]) for steps, _ in data.values())
        print(f"Loaded {run_dir.name}: source={source}, last_iteration={last_step}")
        runs.append(
            {
                "label": run_dir.name,
                "data": data,
                "source": source,
            }
        )

    for group_name, metrics in PLOT_GROUPS.items():
        plot_group(
            group_name,
            metrics,
            runs,
            output_dir,
            args.smooth,
            args.formats,
            args.dpi,
        )
    write_summary(runs, output_dir / "metrics_summary.csv", args.summary_window)


if __name__ == "__main__":
    main()
