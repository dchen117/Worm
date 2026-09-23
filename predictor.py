"""
All predictors operate in centimetres and return:
    (x_pred_cm, y_pred_cm)

The predictor functions are deliberately independent of the evaluation
code so that additional prediction methods can be added later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _get_position_orientation_columns(df):
    """
    Return the position columns to use.

    The Task 2 trajectory CSV contains x_cm/y_cm.
    """
    if "x_cm" in df.columns and "y_cm" in df.columns and "theta_deg" in df.columns:
        return "x_cm", "y_cm", "theta_deg"

    raise ValueError("Trajectory must contain x_cm, y_cm, and theta_deg")


def _valid_history(
    trajectory: pd.DataFrame,
    start_frame: int,
    position_window_frames: int = 0,
) -> pd.DataFrame:
    """
    Return trajectory samples available at or before start_frame.

    Only rows <= start_frame are considered.
    """
    if "frame" not in trajectory.columns:
        raise ValueError("Trajectory CSV must contain a 'frame' column.")

    x_col, y_col, theta_deg = _get_position_orientation_columns(trajectory)

    history = trajectory[trajectory["frame"] <= start_frame].copy()
    history = history[
        np.isfinite(history[x_col])
        & np.isfinite(history[y_col])
        & np.isfinite(history[theta_deg])
    ]

    if position_window_frames > 0:
        min_frame = start_frame - position_window_frames
        history = history[history["frame"] >= min_frame]

    return history


def stationary_predictor(
    trajectory: pd.DataFrame,
    start_frame: int,
) -> tuple[float, float]:
    """
    Stationary baseline.

    Predicts that the robot remains at its position at t.
    """
    x_col, y_col, theta_deg = _get_position_orientation_columns(trajectory)
    history = _valid_history(trajectory, start_frame)
    if history.empty:
        raise ValueError(
            f"No valid trajectory position available at frame "
            f"{start_frame}."
        )
    row = history.iloc[-1]
    return float(row[x_col]), float(row[y_col]), float(row[theta_deg])


def constant_velocity_predictor(
    trajectory: pd.DataFrame,
    start_frame: int,
    fps: float,
    horizon_seconds: float = 1.0,
    velocity_window_seconds: float = 0.5,
    arena_real_width_cm=30.5,
) -> tuple[float, float]:
    """
    Constant-velocity baseline.

    Velocity is estimated from the displacement over the most recent
    velocity_window_seconds of available trajectory history.

    The position at t is then extrapolated forward by horizon_seconds.
    """
    if fps <= 0:
        raise ValueError("fps must be positive.")

    if horizon_seconds < 0:
        raise ValueError("horizon_seconds must be non-negative.")

    if velocity_window_seconds <= 0:
        raise ValueError(
            "velocity_window_seconds must be positive."
        )

    x_col, y_col, theta_deg = _get_position_orientation_columns(trajectory)
    window_frames = max(1, int(round(velocity_window_seconds * fps)))
    history = _valid_history(
        trajectory,
        start_frame,
        position_window_frames=window_frames,
    )

    if history.empty:
        raise ValueError(
            f"No valid trajectory position available at frame "
            f"{start_frame}."
        )

    current = history.iloc[-1]

    # We need an earlier point to estimate velocity.
    if len(history) < 2:
        return float(current[x_col]), float(current[y_col])

    previous = history.iloc[0]
    dt_frames = float(current["frame"] - previous["frame"])
    if dt_frames <= 0:
        return float(current[x_col]), float(current[y_col])
    dt_seconds = dt_frames / fps

    vx = (float(current[x_col]) - float(previous[x_col])) / dt_seconds
    vy = (float(current[y_col]) - float(previous[y_col])) / dt_seconds

    x_pred = np.clip(float(current[x_col]) + vx * horizon_seconds, 0, arena_real_width_cm)
    y_pred = np.clip(float(current[y_col]) + vy * horizon_seconds, 0, arena_real_width_cm)
    theta_pred = float(current[theta_deg])

    return x_pred, y_pred, theta_pred


def predict_all(
    trajectory: pd.DataFrame,
    start_frame: int,
    fps: float,
    horizon_seconds: float = 1.0,
    velocity_window_seconds: float = 0.5,
    arena_real_width_cm=30.5,
) -> dict:
    """
    Run all currently implemented predictors.
    """
    stationary_x, stationary_y, stationary_theta = stationary_predictor(
        trajectory,
        start_frame,
    )

    cv_x, cv_y, cv_theta = constant_velocity_predictor(
        trajectory,
        start_frame,
        fps=fps,
        horizon_seconds=horizon_seconds,
        velocity_window_seconds=velocity_window_seconds,
        arena_real_width_cm=arena_real_width_cm,
    )

    return {
        "stationary_x": stationary_x,
        "stationary_y": stationary_y,
        "stationary_theta": stationary_theta,
        "constant_velocity_x": cv_x,
        "constant_velocity_y": cv_y,
        "constant_velocity_theta": cv_theta,
    }