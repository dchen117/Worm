import cv2
import numpy as np
import pandas as pd
import json
import argparse
from scipy.signal import savgol_filter


# Default trailing path window in seconds
TRAILING_PATH_SECONDS = 3.0


def load_detection_data(csv_path, meta_path):
    """ Load detection results CSV and metadata JSON file. """
    df = pd.read_csv(csv_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return df, meta


def interpolate_and_smooth_data(df, window_length=11, polyorder=2):
    """
    Interpolate missing (x_cm, y_cm, theta_deg) entries linearly,
    fill edge NaNs, and apply Savitzky-Golay filtering to smooth trajectories.
    """
    df_clean = df.copy()

    # 1. Convert columns to numeric, coercion replaces invalid/empty strings with NaN
    df_clean['x_cm'] = pd.to_numeric(df_clean['x_cm'], errors='coerce')
    df_clean['y_cm'] = pd.to_numeric(df_clean['y_cm'], errors='coerce')
    df_clean['theta_deg'] = pd.to_numeric(df_clean['theta_deg'], errors='coerce')

    # 2. Interpolate raw degrees FIRST so no NaNs are passed to np.radians/np.unwrap
    df_clean['x_cm'] = df_clean['x_cm'].interpolate(method='linear').bfill().ffill()
    df_clean['y_cm'] = df_clean['y_cm'].interpolate(method='linear').bfill().ffill()
    df_clean['theta_deg'] = df_clean['theta_deg'].interpolate(method='linear').bfill().ffill()

    # Fallback if the video has zero detections across all frames
    df_clean[['x_cm', 'y_cm', 'theta_deg']] = df_clean[['x_cm', 'y_cm', 'theta_deg']].fillna(0.0)

    # 3. Now unwrap clean, NaN-free angles
    rad = np.radians(df_clean['theta_deg'].values)
    unwrapped_rad = np.unwrap(2 * rad) / 2.0

    # 4. Adjust window length dynamically to guard against short video segments
    num_samples = len(df_clean)
    if num_samples < window_length:
        window_length = num_samples if num_samples % 2 != 0 else num_samples - 1

    # 5. Apply Savitzky-Golay filter safely
    if window_length > polyorder and window_length >= 3:
        df_clean['x_cm_smooth'] = savgol_filter(df_clean['x_cm'].values, window_length, polyorder)
        df_clean['y_cm_smooth'] = savgol_filter(df_clean['y_cm'].values, window_length, polyorder)
        
        smoothed_rad = savgol_filter(unwrapped_rad, window_length, polyorder)
        df_clean['theta_deg_smooth'] = np.degrees(smoothed_rad) % 180.0
    else:
        df_clean['x_cm_smooth'] = df_clean['x_cm']
        df_clean['y_cm_smooth'] = df_clean['y_cm']
        df_clean['theta_deg_smooth'] = df_clean['theta_deg']

    return df_clean


def cm_to_pixel(x_cm, y_cm, arena_width_px, arena_real_width_cm):
    """ Convert arena-relative cm coordinates back to frame pixel coordinates. """
    pixels_per_cm = arena_width_px / arena_real_width_cm
    return x_cm * pixels_per_cm, y_cm * pixels_per_cm


def draw_trailing_path(frame, history_pts, current_frame_idx, fps, arena_box):
    """ Draw a fading trajectory trail for the last few seconds. """
    x0, y0 = arena_box[0], arena_box[1]
    max_history_frames = int(TRAILING_PATH_SECONDS * fps)
    
    start_idx = max(0, current_frame_idx - max_history_frames)
    recent_pts = history_pts[start_idx:current_frame_idx + 1]

    if len(recent_pts) < 2:
        return

    num_segments = len(recent_pts) - 1
    for i in range(num_segments):
        pt1 = (int(recent_pts[i][0] + x0), int(recent_pts[i][1] + y0))
        pt2 = (int(recent_pts[i + 1][0] + x0), int(recent_pts[i + 1][1] + y0))

        alpha = (i + 1) / float(num_segments)
        thickness = max(1, int(1 + 3 * alpha))
        color = (0, int(255 * alpha), int(255 * (1 - alpha)))

        cv2.line(frame, pt1, pt2, color, thickness)


def main():
    parser = argparse.ArgumentParser(description="Interpolate missing positions using JSON metadata.")
    parser.add_argument("-i", "--input-video", help="Original input video", required=True)
    parser.add_argument("-j", "--input-csv", help="CSV from detection script", required=True)
    parser.add_argument("-m", "--input-meta", help="Metadata JSON file", required=True)
    parser.add_argument("-o", "--output-video", help="Annotated video output", required=True)
    parser.add_argument("-c", "--output-csv", help="Smoothed CSV output", required=True)
    args = parser.parse_args()

    # Read data and metadata automatically
    df, meta = load_detection_data(args.input_csv, args.input_meta)
    
    arena_box = meta["arena_box"]
    arena_width_px = meta["arena_width_px"]
    arena_real_width_cm = meta["arena_real_width_cm"]
    axis_line_length = meta.get("axis_line_length", 60)

    # Interpolate and smooth coordinates
    df_smooth = interpolate_and_smooth_data(df)
    df_smooth.to_csv(args.output_csv, index=False)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file {args.input_video}")
        return

    fps = meta.get("fps", cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    # Pre-compute pixel locations
    pixel_centers = []
    for _, row in df_smooth.iterrows():
        cx, cy = cm_to_pixel(row['x_cm_smooth'], row['y_cm_smooth'], arena_width_px, arena_real_width_cm)
        pixel_centers.append((cx, cy))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(df_smooth):
            break

        row = df_smooth.iloc[frame_idx]
        cx, cy = pixel_centers[frame_idx]
        theta = row['theta_deg_smooth']
        was_detected = bool(row['detected'])

        # Draw fading path
        draw_trailing_path(frame, pixel_centers, frame_idx, fps, arena_box)

        # Map arena coordinates to frame space
        full_cx = int(cx + arena_box[0])
        full_cy = int(cy + arena_box[1])

        point_color = (0, 255, 0) if was_detected else (0, 165, 255)
        cv2.circle(frame, (full_cx, full_cy), 5, point_color, -1)

        # Orientation vector
        angle_rad = np.radians(theta)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        x1 = int(full_cx - axis_line_length * dx)
        y1 = int(full_cy - axis_line_length * dy)
        x2 = int(full_cx + axis_line_length * dx)
        y2 = int(full_cy + axis_line_length * dy)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # HUD overlay
        status_str = "DETECTED" if was_detected else "INTERPOLATED"
        cv2.putText(frame, f"Frame: {frame_idx} [{status_str}]", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Pos: ({row['x_cm_smooth']:.1f} cm, {row['y_cm_smooth']:.1f} cm)", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, point_color, 2)
        cv2.putText(frame, f"Theta: {theta:.1f} deg", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw arena box
        cv2.rectangle(frame, (arena_box[0], arena_box[1]), (arena_box[2], arena_box[3]), (0, 255, 255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Annotated video saved to: {args.output_video}")


if __name__ == "__main__":
    main()