"""
Evaluate Task 3 predictions against manual t + 1 s endpoint annotations.

Annotation CSV format:
    frame, x_gt, y_gt, theta_gt

Here `frame` is the endpoint frame at t + 1 s. The prediction start
frame is inferred as frame - horizon_seconds * fps.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predictor import predict_all

ANNOTATION_COLUMNS = ["frame", "x_gt", "y_gt", "theta_gt"]

PREDICTOR_COLUMNS = {
    "stationary": ("stationary_x", "stationary_y", "stationary_theta"),
    "constant_velocity": (
        "constant_velocity_x",
        "constant_velocity_y",
        "constant_velocity_theta",
    ),
    "ekf": ("ekf_x", "ekf_y", "ekf_theta")
}


def load_annotations(path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Prediction annotation file not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in ANNOTATION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Prediction annotation CSV is missing columns: "
            + ", ".join(missing)
        )

    df["frame"] = pd.to_numeric(df["frame"], errors="raise").astype(int)
    df["x_gt"] = pd.to_numeric(df["x_gt"], errors="raise")
    df["y_gt"] = pd.to_numeric(df["y_gt"], errors="raise")
    df["theta_gt"] = pd.to_numeric(df["theta_gt"], errors="raise")

    return df


def euclidean_error(x_pred, y_pred, x_gt, y_gt):
    return float(np.hypot(x_pred - x_gt, y_pred - y_gt))


def angle_error(theta_pred, theta_gt):
    d = abs(theta_pred - theta_gt) % 180
    return min(d, 180 - d)


def summarize_errors(errors):
    values = np.asarray(errors, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {"count": 0, "mean": np.nan, "median": np.nan, "p90": np.nan}

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def evaluate_clip(
    trajectory,
    annotations,
    fps,
    predictors,
    horizon_seconds=1.0,
    velocity_window_seconds=0.5,
    arena_real_width_cm=30.5,
):
    start_offset = int(round(horizon_seconds * fps))
    rows = []

    for _, annotation in annotations.iterrows():
        target_frame = int(annotation["frame"])
        start_frame = target_frame - start_offset

        predictions = predict_all(
            trajectory=trajectory,
            start_frame=start_frame,
            fps=fps,
            horizon_seconds=horizon_seconds,
            velocity_window_seconds=velocity_window_seconds,
            arena_real_width_cm=arena_real_width_cm,
        )

        row = {
            "start_frame": start_frame,
            "frame": target_frame,
            "x_gt": float(annotation["x_gt"]),
            "y_gt": float(annotation["y_gt"]),
            "theta_gt": float(annotation["theta_gt"]),
        }

        for predictor in predictors:
            if predictor not in PREDICTOR_COLUMNS:
                raise ValueError(
                    f"Unknown predictor '{predictor}'. "
                    f"Available predictors: {', '.join(PREDICTOR_COLUMNS)}"
                )

            x_col, y_col, theta_col = PREDICTOR_COLUMNS[predictor]
            if any(c not in predictions for c in (x_col, y_col, theta_col)):
                raise ValueError(
                    f"predict_all() does not provide outputs for '{predictor}'."
                )

            row[f"{predictor}_x"] = predictions[x_col]
            row[f"{predictor}_y"] = predictions[y_col]
            row[f"{predictor}_theta"] = predictions[theta_col]
            row[f"{predictor}_position_error"] = euclidean_error(
                predictions[x_col],
                predictions[y_col],
                row["x_gt"],
                row["y_gt"],
            )
            row[f"{predictor}_angle_error"] = angle_error(
                predictions[theta_col],
                row["theta_gt"],
            )

        rows.append(row)

    return pd.DataFrame(rows)


def make_summary_table(results, predictors):
    summary = []

    for predictor in predictors:
        pos = summarize_errors(results[f"{predictor}_position_error"])
        angle = summarize_errors(results[f"{predictor}_angle_error"])

        summary.append({
            "method": predictor,
            "position_count": pos["count"],
            "position_mean_cm": pos["mean"],
            "position_median_cm": pos["median"],
            "position_p90_cm": pos["p90"],
            "orientation_count": angle["count"],
            "orientation_mean_deg": angle["mean"],
            "orientation_median_deg": angle["median"],
            "orientation_p90_deg": angle["p90"],
        })

    return pd.DataFrame(summary)


def save_summary(results, output_path, predictors):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    summary_path = output_path.parent / f"{output_path.stem}_summary.csv"
    make_summary_table(results, predictors).to_csv(summary_path, index=False)
    return summary_path


def plot_errors(results, predictors, metric, ylabel, title, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        results[f"{predictor}_{metric}_error"].dropna().to_numpy()
        for predictor in predictors
    ]

    x = np.arange(len(results))
    positions = np.arange(len(predictors))

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    # Evaluation-point plot
    ax = axes[0]

    for predictor in predictors:
        ax.plot(
            x,
            results[f"{predictor}_{metric}_error"],
            "o",
            label=predictor.replace("_", " ").title(),
        )

    ax.set_xlabel("Evaluation point")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — Evaluation Points")

    if metric == "angle":
        ax.set_ylim(0, 90)

    ax.grid(True, alpha=0.3)
    ax.legend()

    # Box plot
    ax = axes[1]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        showfliers=False,
    )

    # Use the boxplot's existing median line in the legend.
    bp["medians"][0].set_label("Median")

    rng = np.random.default_rng(0)

    for i, values in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, len(values))

        ax.scatter(
            positions[i] + jitter,
            values,
            alpha=0.7,
            s=20,
        )

        mean = np.mean(values)
        p90 = np.percentile(values, 90)

        ax.hlines(
            mean,
            positions[i] - 0.25,
            positions[i] + 0.25,
            linestyles="--",
            label="Mean" if i == 0 else None,
        )

        ax.hlines(
            p90,
            positions[i] - 0.25,
            positions[i] + 0.25,
            linestyles=":",
            label="90th percentile" if i == 0 else None,
        )

    ax.set_xticks(
        positions,
        [p.replace("_", " ").title() for p in predictors],
    )
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — Distribution")

    if metric == "angle":
        ax.set_ylim(0, 90)

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_trajectory_prediction(
    trajectory,
    result_row,
    predictors,
    output_path,
    history_seconds=3.0,
    fps=30.0,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_frame = int(result_row["start_frame"])
    history_start = max(0, start_frame - int(history_seconds * fps))
    history = trajectory[
        (trajectory["frame"] >= history_start)
        & (trajectory["frame"] <= start_frame)
    ]

    if history.empty:
        return

    plt.figure(figsize=(7, 7))
    plt.plot(history["x_cm"], history["y_cm"], label="Observed trajectory")

    current_x = float(history.iloc[-1]["x_cm"])
    current_y = float(history.iloc[-1]["y_cm"])

    plt.scatter([current_x], [current_y], label="Start t", zorder=5)
    plt.scatter(
        [result_row["x_gt"]],
        [result_row["y_gt"]],
        marker="x",
        s=80,
        label="Manual t + 1 s",
        zorder=6,
    )

    for predictor in predictors:
        plt.plot(
            [current_x, result_row[f"{predictor}_x"]],
            [current_y, result_row[f"{predictor}_y"]],
            linestyle="--",
            label=f"{predictor.replace('_', ' ').title()} prediction",
        )

    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title(f"Prediction from frame {start_frame}")
    plt.axis("equal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate(
    trajectory_csv,
    annotations_csv,
    output_directory,
    fps,
    predictors=("stationary", "constant_velocity"),
    horizon_seconds=1.0,
    velocity_window_seconds=0.5,
    arena_real_width_cm=30.5,
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    trajectory = pd.read_csv(trajectory_csv)
    annotations = load_annotations(annotations_csv)

    results = evaluate_clip(
        trajectory=trajectory,
        annotations=annotations,
        fps=fps,
        predictors=list(predictors),
        horizon_seconds=horizon_seconds,
        velocity_window_seconds=velocity_window_seconds,
        arena_real_width_cm=arena_real_width_cm,
    )

    results_path = output_directory / "prediction_results.csv"
    summary_path = save_summary(results, results_path, list(predictors))

    plot_errors(
        results,
        list(predictors),
        "position",
        "Position error (cm)",
        "Prediction Position Error",
        output_directory / "position_errors.png",
    )

    plot_errors(
        results,
        list(predictors),
        "angle",
        "Orientation error (degrees)",
        "Prediction Orientation Error",
        output_directory / "orientation_errors.png",
    )

    if not results.empty:
        plot_trajectory_prediction(
            trajectory=trajectory,
            result_row=results.iloc[0],
            predictors=list(predictors),
            output_path=output_directory / "prediction_example.png",
            fps=fps,
        )

    return results, pd.read_csv(summary_path)
