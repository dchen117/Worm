import cv2
import numpy as np
import csv
import argparse
import json


# Background subtractor settings
MOG_HISTORY = 300
MOG_VAR_THRESHOLD = 12
MOG_LEARNING_RATE = 0.002

# Blob filtering thresholds
MIN_BLOB_AREA = 200
MAX_BLOB_AREA = 20000

# Morphological filtering
MORPH_KERNEL_SIZE = 5

# Length of the orientation line in pixels
AXIS_LINE_LENGTH = 60

# Physical arena dimensions (cm)
ARENA_REAL_WIDTH_CM = 30.5

# Maximum frames to inspect searching for an arena boundary before falling back
MAX_ARENA_SEARCH_FRAMES = 150


# ============================================================
# Arena Detection
# ============================================================

def select_arena_manually(frame):
    """
    Allow manual selection with correct window scaling and clean window closure.
    """
    print("Opening ROI selector... Drag a box, press SPACE/ENTER to confirm, or 'c' to cancel.")
    window_name = "Select Arena Region"

    frame_h, frame_w = frame.shape[:2]

    # Calculate target window width and height to preserve aspect ratio within a standard screen size
    max_display_height = 720
    if frame_h > max_display_height:
        scale = max_display_height / float(frame_h)
        disp_w = int(frame_w * scale)
        disp_h = max_display_height
    else:
        disp_w, disp_h = frame_w, frame_h

    # Create resizable window with explicit dimensions
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)

    # selectROI returns (x, y, w, h) based on original image dimensions
    roi = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)

    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w > 0 and h > 0:
        return x, y, x + w, y + h, w

    print("Manual selection cancelled or invalid. Falling back to full frame.")
    return 0, 0, frame_w, frame_h, frame_w


def detect_arena_in_frame(frame):
    """
    Attempt to detect the arena boundary in a single frame automatically.
    Returns (x0, y0, x1, y1, width_px) if found, otherwise None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        frame_h, frame_w = frame.shape[:2]
        if w * h > 0.15 * (frame_w * frame_h):
            return x, y, x + w, y + h, w

    return None


def detect_arena_multi_frame(cap, max_frames=MAX_ARENA_SEARCH_FRAMES):
    """
    Iterate through frames until the arena is successfully detected.
    Falls back to full frame size if detection fails within max_frames.
    """
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    searched_frames = 0

    while searched_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        searched_frames += 1
        result = detect_arena_in_frame(frame)

        if result is not None:
            x0, y0, x1, y1, width_px = result
            print(f"Arena detected on frame {searched_frames}: X=[{x0}, {x1}], Y=[{y0}, {y1}], Width={width_px}px")
            return x0, y0, x1, y1, width_px

    print(f"Warning: Could not detect arena after {searched_frames} frames. Using full frame fallback.")
    return 0, 0, frame_w, frame_h, frame_w


# ============================================================
# Helper Functions
# ============================================================

def create_robot_mask(frame, background_subtractor, arena_box):
    """
    Apply MOG2 background subtraction and clean the resulting foreground mask.
    """
    x0, y0, x1, y1 = arena_box

    arena = frame[y0:y1, x0:x1]

    mask = background_subtractor.apply(arena, learningRate=MOG_LEARNING_RATE)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def find_robot_contour(mask):
    """
    Find the most likely robot contour.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if MIN_BLOB_AREA <= cv2.contourArea(c) <= MAX_BLOB_AREA]

    if not candidates:
        return None

    return max(candidates, key=cv2.contourArea)


def calculate_centroid(contour):
    """
    Calculate the center of the robot using contour moments.
    """
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    return cx, cy


def calculate_orientation(contour, mask):
    """
    Calculate the robot body axis using PCA.
    """
    robot_mask = np.zeros_like(mask)
    cv2.drawContours(robot_mask, [contour], -1, 255, thickness=cv2.FILLED)

    ys, xs = np.where(robot_mask == 255)

    if len(xs) < 2:
        return None

    points = np.column_stack((xs, ys))
    mean = np.mean(points, axis=0)
    centered = points - mean

    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    dx, dy = principal_axis[0], principal_axis[1]

    theta = np.degrees(np.arctan2(dy, dx)) % 180
    return theta


def pixel_to_cm(x, y, arena_width_px):
    """
    Convert arena-relative pixel coordinates to cm using dynamic scale.
    """
    cm_per_pixel = ARENA_REAL_WIDTH_CM / arena_width_px
    return x * cm_per_pixel, y * cm_per_pixel


def annotate_frame(frame, contour, cx, cy, x_cm, y_cm, theta, arena_box, frame_number):
    """
    Draw the robot outline, center point, orientation axis, frame number, and position in cm.
    """
    x0, y0 = arena_box[0], arena_box[1]

    # Draw contour
    cv2.drawContours(frame, [contour + np.array([[[x0, y0]]])], -1, (255, 0, 0), 2)

    full_cx = int(cx + x0)
    full_cy = int(cy + y0)

    # Draw center point
    cv2.circle(frame, (full_cx, full_cy), 5, (0, 255, 0), -1)

    # Draw orientation axis
    angle_rad = np.radians(theta)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    x1 = int(full_cx - AXIS_LINE_LENGTH * dx)
    y1 = int(full_cy - AXIS_LINE_LENGTH * dy)
    x2 = int(full_cx + AXIS_LINE_LENGTH * dx)
    y2 = int(full_cy + AXIS_LINE_LENGTH * dy)

    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Text overlays
    cv2.putText(frame, f"Frame: {frame_number}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Pos: ({x_cm:.1f} cm, {y_cm:.1f} cm)", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Theta: {theta:.1f} deg", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# ============================================================
# Main Execution
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Robot tracking with automated or manual arena selection.")
    parser.add_argument("-i", "--input", help="Path to input video file", required=True)
    parser.add_argument("-o", "--output-video", help="Path for annotated video output", required=True)
    parser.add_argument("-c", "--output-csv", help="Path for CSV results output", required=True)
    parser.add_argument("-m", "--output-meta", help="Path for JSON metadata output", required=True)
    parser.add_argument("-s", "--select", action="store_true", help="Manually select arena region using interactive GUI")
    return parser.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {args.input}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")

    if args.select:
        ret, first_frame = cap.read()
        if not ret:
            print("ERROR: Failed to read video frame for manual selection.")
            return
        x0, y0, x1, y1, arena_width_px = select_arena_manually(first_frame)
    else:
        x0, y0, x1, y1, arena_width_px = detect_arena_multi_frame(cap)

    arena_box = (x0, y0, x1, y1)

    print(f"Final Arena Bounding Box: X=[{x0}, {x1}], Y=[{y0}, {y1}]")
    print(f"Arena Width: {arena_width_px} px")

    # Rewind video back to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=MOG_HISTORY,
        varThreshold=MOG_VAR_THRESHOLD,
        detectShadows=False
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    csv_file = open(args.output_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "x_cm", "y_cm", "theta_deg", "detected"])

    detected_count = 0
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = create_robot_mask(frame, background_subtractor, arena_box)
        contour = find_robot_contour(mask)

        detected = False
        x_cm, y_cm, theta = "", "", ""

        if contour is not None:
            center = calculate_centroid(contour)

            if center is not None:
                cx, cy = center
                theta_result = calculate_orientation(contour, mask)

                if theta_result is not None:
                    theta = theta_result
                    x_cm_val, y_cm_val = pixel_to_cm(cx, cy, arena_width_px)
                    x_cm, y_cm = x_cm_val, y_cm_val
                    detected = True
                    detected_count += 1

                    annotate_frame(frame, contour, cx, cy, x_cm_val, y_cm_val, theta, arena_box, frame_number)

        csv_writer.writerow([frame_number, x_cm, y_cm, theta, int(detected)])

        if not detected:
            cv2.putText(frame, f"Frame: {frame_number}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "ROBOT NOT DETECTED", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Outline the selected arena
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

        writer.write(frame)
        frame_number += 1

        if frame_number % 100 == 0 and frame_count > 0:
            print(f"Processed {frame_number}/{frame_count} ({100 * frame_number / frame_count:.1f}%)")

    cap.release()
    writer.release()
    csv_file.close()

    detection_fraction = detected_count / frame_number if frame_number > 0 else 0

    metadata = {
        "arena_box": arena_box,  # [x0, y0, x1, y1]
        "arena_width_px": float(arena_width_px),
        "arena_real_width_cm": ARENA_REAL_WIDTH_CM,
        "fps": fps,
        "total_frames": frame_number,
        "detected_frames": detected_count,
        "detection_fraction": detection_fraction,
        "axis_line_length": AXIS_LINE_LENGTH,
    }

    # Save metadata to JSON
    with open(args.output_meta, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nFinished.")
    print(f"Frames processed: {frame_number}")
    print(f"Frames detected: {detected_count}")
    print(f"Detection fraction: {detection_fraction:.3f}")
    print(f"Annotated video: {args.output_video}")
    print(f"CSV: {args.output_csv}")
    print(f"Metadata JSON: {args.output_meta}")


if __name__ == "__main__":
    main()