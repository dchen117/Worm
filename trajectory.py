import cv2
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter


# ============================================================
# Data Processing
# ============================================================

def load_detection_data(csv_path, meta_path):
    """
    Load detector CSV and metadata JSON.
    """

    df = pd.read_csv(csv_path)

    import json
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return df, meta


def interpolate_and_smooth_data(df, window_length=11, polyorder=2):
    """
    Interpolate missing values and smooth the trajectory.
    """

    df_clean = df.copy()

    # Convert to numeric.
    df_clean["x_cm"] = pd.to_numeric(df_clean["x_cm"], errors="coerce")
    df_clean["y_cm"] = pd.to_numeric(df_clean["y_cm"], errors="coerce")
    df_clean["theta_deg"] = pd.to_numeric(df_clean["theta_deg"], errors="coerce")

    # Interpolate.
    df_clean["x_cm"] = (
        df_clean["x_cm"]
        .interpolate(method="linear")
        .bfill()
        .ffill()
    )

    df_clean["y_cm"] = (
        df_clean["y_cm"]
        .interpolate(method="linear")
        .bfill()
        .ffill()
    )

    df_clean["theta_deg"] = (
        df_clean["theta_deg"]
        .interpolate(method="linear")
        .bfill()
        .ffill()
    )

    # If there were zero detections.
    df_clean[
        ["x_cm", "y_cm", "theta_deg"]
    ] = df_clean[
        ["x_cm", "y_cm", "theta_deg"]
    ].fillna(0.0)

    # Unwrap orientation.
    rad = np.radians(df_clean["theta_deg"].values)
    unwrapped_rad = (np.unwrap(2 * rad) / 2.0)
    num_samples = len(df_clean)

    if num_samples < window_length:
        window_length = num_samples if num_samples % 2 != 0 else num_samples - 1

    if window_length > polyorder and window_length >= 3:
        df_clean[
            "x_cm_smooth"
        ] = savgol_filter(
            df_clean["x_cm"].values,
            window_length,
            polyorder,
        )

        df_clean[
            "y_cm_smooth"
        ] = savgol_filter(
            df_clean["y_cm"].values,
            window_length,
            polyorder,
        )

        smoothed_rad = (
            savgol_filter(
                unwrapped_rad,
                window_length,
                polyorder,
            )
        )

        df_clean["theta_deg_smooth"] = np.degrees(smoothed_rad) % 180.0

    else:
        df_clean["x_cm_smooth"] = df_clean["x_cm"]
        df_clean["y_cm_smooth"] = df_clean["y_cm"]
        df_clean["theta_deg_smooth"] = df_clean["theta_deg"]

    return df_clean


# ============================================================
# Rendering
# ============================================================

def cm_to_pixel(
    x_cm,
    y_cm,
    arena_width_px,
    arena_real_width_cm,
):
    """
    Convert centimeters back to pixels.
    """

    pixels_per_cm = arena_width_px / arena_real_width_cm
    return x_cm * pixels_per_cm, y_cm * pixels_per_cm,


def draw_trailing_path(
    frame,
    history_pts,
    current_frame_idx,
    fps,
    arena_box,
    trailing_path_seconds,
):
    """
    Draw a fading trajectory trail.
    """

    x0, y0 = arena_box[:2]
    max_history_frames = int(trailing_path_seconds * fps)
    start_idx = max(0, current_frame_idx - max_history_frames)

    recent_pts = history_pts[start_idx: current_frame_idx + 1]

    if len(recent_pts) < 2:
        return

    num_segments = len(recent_pts) - 1

    for i in range(num_segments):

        pt1 = (
            int(recent_pts[i][0] + x0),
            int(recent_pts[i][1] + y0),
        )

        pt2 = (
            int(recent_pts[i + 1][0] + x0),
            int(recent_pts[i + 1][1] + y0),
        )

        alpha = (i + 1) / float(num_segments)
        thickness = max(1, int(1 + 3 * alpha))
        color = (0, int(255 * alpha), int(255 * (1 - alpha)))

        cv2.line(
            frame,
            pt1,
            pt2,
            color,
            thickness,
        )


# ============================================================
# Main Trajectory Function
# ============================================================

def generate_trajectory(
    input_video,
    input_csv,
    input_meta,
    output_video,
    output_csv,
    trajectory_config,
):
    """
    Generate the interpolated/smoothed trajectory and
    annotated trajectory video.
    """

    df, meta = load_detection_data(input_csv, input_meta)
    arena_box = meta["arena_box"]
    arena_width_px = meta["arena_width_px"]
    arena_real_width_cm = meta["arena_real_width_cm"]
    axis_line_length = meta.get("axis_line_length", 60)
    smoothing_config = trajectory_config["smoothing"]

    df_smooth = (
        interpolate_and_smooth_data(
            df,
            window_length=smoothing_config["window_length"],
            polyorder=smoothing_config["polyorder"],
        )
    )

    # Save final trajectory CSV.
    df_smooth.to_csv(output_csv, index=False)

    cap = cv2.VideoCapture(str(input_video))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{input_video}"
        )

    fps = meta.get("fps", cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create trajectory video: "
            f"{output_video}"
        )

    # Pre-compute pixel locations.
    pixel_centers = []

    for _, row in df_smooth.iterrows():
        cx, cy = cm_to_pixel(
            row["x_cm_smooth"],
            row["y_cm_smooth"],
            arena_width_px,
            arena_real_width_cm,
        )
        pixel_centers.append((cx, cy))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(df_smooth):
            break

        row = df_smooth.iloc[frame_idx]
        cx, cy = pixel_centers[frame_idx]
        theta = row["theta_deg_smooth"]
        was_detected = bool(row["detected"])

        # Draw trajectory.
        draw_trailing_path(
            frame,
            pixel_centers,
            frame_idx,
            fps,
            arena_box,
            trajectory_config["trailing_path_seconds"],
        )

        # Convert to frame coordinates.
        full_cx = int(cx + arena_box[0])

        full_cy = int(cy + arena_box[1])

        if was_detected:
            point_color = (0, 255, 0)
        else:
            point_color = (0, 165, 255)

        cv2.circle(
            frame,
            (full_cx, full_cy),
            5,
            point_color,
            -1,
        )

        # Orientation.
        angle_rad = np.radians(theta)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        x1 = int(full_cx - axis_line_length * dx)
        y1 = int(full_cy - axis_line_length * dy)
        x2 = int(full_cx + axis_line_length * dx)
        y2 = int(full_cy + axis_line_length * dy)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        # HUD.
        status_str = "DETECTED" if was_detected else "INTERPOLATED"

        cv2.putText(
            frame,
            f"Frame: {frame_idx} [{status_str}]",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            (
                f"Pos: "
                f"({row['x_cm_smooth']:.1f} cm, "
                f"{row['y_cm_smooth']:.1f} cm)"
            ),
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            point_color,
            2,
        )

        cv2.putText(
            frame,
            f"Theta: {theta:.1f} deg",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # Arena.
        cv2.rectangle(
            frame,
            (arena_box[0], arena_box[1]),
            (arena_box[2], arena_box[3]),
            (0, 255, 255),
            2,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(
        f"Trajectory video saved to: "
        f"{output_video}"
    )

    print(
        f"Trajectory CSV saved to: "
        f"{output_csv}"
    )


# ============================================================
# Optional Standalone CLI
# ============================================================

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--input-video", required=True)
#     parser.add_argument("-j", "--input-csv", required=True)
#     parser.add_argument("-m", "--input-meta", required=True)
#     parser.add_argument("-o", "--output-video", required=True)
#     parser.add_argument("-c", "--output-csv", required=True)
#     args = parser.parse_args()

#     config = {
#         "trailing_path_seconds": 3.0,
#         "smoothing": {
#             "window_length": 11,
#             "polyorder": 2,
#         },
#     }

#     generate_trajectory(
#         input_video=args.input_video,
#         input_csv=args.input_csv,
#         input_meta=args.input_meta,
#         output_video=args.output_video,
#         output_csv=args.output_csv,
#         trajectory_config=config,
#     )