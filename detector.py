import cv2
import numpy as np
import csv
import argparse
import json


# ============================================================
# Arena Detection
# ============================================================

def select_arena_manually(frame, display_height=720):
    """
    Allow the user to manually select the arena.

    The displayed image is scaled down to fit the screen.
    The returned coordinates are mapped back to the original
    frame resolution.

    Returns:
        (x0, y0, x1, y1, width_px)
        or None if selection was cancelled.
    """

    window_name = "Select Arena Region"
    frame_h, frame_w = frame.shape[:2]

    # Scale image for display.
    scale = min(1.0, display_height / float(frame_h))
    display_w = int(frame_w * scale)
    display_h = int(frame_h * scale)

    display_frame = cv2.resize(
        frame,
        (display_w, display_h),
        interpolation=cv2.INTER_AREA,
    )

    print()
    print("Manual arena selection")
    print("----------------------")
    print("Drag a rectangle around the arena.")
    print("Press ENTER or SPACE to confirm.")
    print("Press C or ESC to cancel.")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_w, display_h)

    roi = cv2.selectROI(
        window_name,
        display_frame,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(window_name)

    x, y, w, h = roi

    if w <= 0 or h <= 0:
        return None

    # Convert displayed coordinates back to original image.
    x0 = int(x / scale)
    y0 = int(y / scale)
    x1 = int((x + w) / scale)
    y1 = int((y + h) / scale)

    x0 = max(0, min(x0, frame_w - 1))
    y0 = max(0, min(y0, frame_h - 1))
    x1 = max(x0 + 1, min(x1, frame_w))
    y1 = max(y0 + 1, min(y1, frame_h))

    width_px = x1 - x0

    return x0, y0, x1, y1, width_px


def detect_arena_in_frame(frame):
    """
    Attempt to detect the arena boundary in a single frame.

    Returns:
        (x0, y0, x1, y1, width_px)
        or None if no suitable arena is found.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    frame_h, frame_w = frame.shape[:2]

    # Require the candidate to occupy a reasonable portion
    # of the image.
    if w * h > 0.15 * (frame_w * frame_h):
        return x, y, x + w, y + h, w

    return None


def detect_arena_multi_frame(cap, max_frames=150):
    """
    Search through the beginning of a video for the arena.

    IMPORTANT:
    This function does NOT fall back to the full frame.

    If detection fails, None is returned so that the pipeline
    can put the video into the manual-selection queue.
    """

    searched_frames = 0
    while searched_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        searched_frames += 1
        result = detect_arena_in_frame(frame)
        if result is not None:
            x0, y0, x1, y1, width_px = result
            print(
                f"Arena automatically detected on "
                f"frame {searched_frames}: "
                f"X=[{x0}, {x1}], "
                f"Y=[{y0}, {y1}], "
                f"Width={width_px}px"
            )
            return result
    return None


# ============================================================
# Detection Helpers
# ============================================================

def create_robot_mask(frame, background_subtractor, arena_box, config):
    """
    Apply MOG2 background subtraction and clean the mask.
    """

    x0, y0, x1, y1 = arena_box
    arena = frame[y0:y1, x0:x1]
    mask = background_subtractor.apply(arena, learningRate=config["mog_learning_rate"])
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    kernel_size = config["morph_kernel_size"]
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def find_robot_contour(mask, config):
    """
    Find the most likely robot contour.
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = [
        c for c in contours
        if config["min_blob_area"] <= cv2.contourArea(c) <= config["max_blob_area"]
    ]

    if not candidates:
        return None

    return max(candidates, key=cv2.contourArea)


def calculate_centroid(contour):
    """
    Calculate contour centroid.
    """

    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    return cx, cy


def calculate_orientation(contour, mask):
    """
    Calculate robot orientation using PCA.
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
    dx, dy = principal_axis

    theta = np.degrees(np.arctan2(dy, dx)) % 180
    return theta


def pixel_to_cm(x, y, arena_width_px, arena_real_width_cm):
    """
    Convert arena-relative pixels to centimeters.
    """

    cm_per_pixel = arena_real_width_cm / arena_width_px

    return x * cm_per_pixel, y * cm_per_pixel


def annotate_frame(
    frame,
    contour,
    cx,
    cy,
    x_cm,
    y_cm,
    theta,
    arena_box,
    frame_number,
    axis_line_length,
):
    """
    Draw detection information onto a frame.
    """

    x0, y0 = arena_box[:2]
    translated_contour = contour + np.array([[[x0, y0]]])

    cv2.drawContours(frame, [translated_contour], -1, (255, 0, 0), 2)

    full_cx = int(cx + x0)
    full_cy = int(cy + y0)

    cv2.circle(frame, (full_cx, full_cy), 5, (0, 255, 0), -1)

    angle_rad = np.radians(theta)

    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    x1 = int(full_cx - axis_line_length * dx)
    y1 = int(full_cy - axis_line_length * dy)
    x2 = int(full_cx + axis_line_length * dx)
    y2 = int(full_cy + axis_line_length * dy)

    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Pos: ({x_cm:.1f} cm, {y_cm:.1f} cm)",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
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


# ============================================================
# Main Detection Function
# ============================================================

def detect(
    input_video,
    output_video,
    output_csv,
    output_meta,
    arena_box,
    detector_config,
):
    """
    Run robot detection on a video.

    Parameters
    ----------
    input_video:
        Input video path.

    output_video:
        Detection visualization video path.
        May be None to disable video output.

    output_csv:
        Raw detection CSV path.

    output_meta:
        Arena/calibration metadata JSON path.

    arena_box:
        Tuple:
            (x0, y0, x1, y1)

    detector_config:
        Detector configuration dictionary.

    Returns
    -------
    metadata dictionary.
    """

    cap = cv2.VideoCapture(str(input_video))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print()
    print(f"Processing: {input_video}")
    print(
        f"Resolution: {width} x {height}"
    )
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")

    x0, y0, x1, y1 = arena_box
    arena_width_px = x1 - x0

    print(
        f"Arena: "
        f"X=[{x0}, {x1}], "
        f"Y=[{y0}, {y1}]"
    )

    # Background subtractor.
    background_subtractor = (
        cv2.createBackgroundSubtractorMOG2(
            history=detector_config["mog_history"],
            varThreshold=detector_config["mog_var_threshold"],
            detectShadows=False,
        )
    )

    writer = None

    if output_video is not None:
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
                f"Could not create output video: "
                f"{output_video}"
            )

    csv_file = open(output_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(
        [
            "frame",
            "x_cm",
            "y_cm",
            "theta_deg",
            "detected",
        ]
    )

    detected_count = 0
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = create_robot_mask(
            frame,
            background_subtractor,
            arena_box,
            detector_config,
        )

        contour = find_robot_contour(mask, detector_config)
        detected = False

        x_cm = ""
        y_cm = ""
        theta = ""

        if contour is not None:
            center = calculate_centroid(contour)

            if center is not None:
                cx, cy = center
                theta_result = calculate_orientation(contour, mask)

                if theta_result is not None:
                    theta = theta_result
                    x_cm_val, y_cm_val = (
                        pixel_to_cm(
                            cx,
                            cy,
                            arena_width_px,
                            detector_config["arena_real_width_cm"],
                        )
                    )

                    x_cm = x_cm_val
                    y_cm = y_cm_val

                    detected = True
                    detected_count += 1

                    if writer is not None:
                        annotate_frame(
                            frame,
                            contour,
                            cx,
                            cy,
                            x_cm_val,
                            y_cm_val,
                            theta,
                            arena_box,
                            frame_number,
                            detector_config[
                                "axis_line_length"
                            ],
                        )

        csv_writer.writerow(
            [
                frame_number,
                x_cm,
                y_cm,
                theta,
                int(detected),
            ]
        )

        if not detected and writer is not None:

            cv2.putText(
                frame,
                f"Frame: {frame_number}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "ROBOT NOT DETECTED",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if writer is not None:

            cv2.rectangle(
                frame,
                (x0, y0),
                (x1, y1),
                (0, 255, 255),
                2,
            )

            writer.write(frame)

        frame_number += 1

        if frame_number % 100 == 0 and frame_count > 0:
            progress = 100 * frame_number / frame_count
            print(
                f"Processed "
                f"{frame_number}/{frame_count} "
                f"({progress:.1f}%)"
            )

    cap.release()

    if writer is not None:
        writer.release()

    csv_file.close()

    detection_fraction = detected_count / frame_number if frame_number > 0 else 0

    metadata = {
        "arena_box": [
            int(x0),
            int(y0),
            int(x1),
            int(y1),
        ],
        "arena_width_px": float(arena_width_px),
        "arena_real_width_cm": float(detector_config["arena_real_width_cm"]),
        "fps": float(fps),
        "total_frames": int(frame_number),
        "detected_frames": int(
            detected_count
        ),
        "detection_fraction": float(
            detection_fraction
        ),
        "axis_line_length": int(
            detector_config[
                "axis_line_length"
            ]
        ),
    }

    with open(
        output_meta,
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
        )

    print()
    print("Detection complete.")
    print(
        f"Frames processed: "
        f"{frame_number}"
    )
    print(
        f"Frames detected: "
        f"{detected_count}"
    )
    print(
        f"Detection fraction: "
        f"{detection_fraction:.3f}"
    )

    return metadata


# ============================================================
# Optional Standalone CLI
# ============================================================

# def parse_args():
#     parser = argparse.ArgumentParser(description="Robot detector")
#     parser.add_argument("-i", "--input", required=True)
#     parser.add_argument("-o", "--output-video", required=True)
#     parser.add_argument("-c", "--output-csv", required=True)
#     parser.add_argument("-m", "--output-meta", required=True)
#     parser.add_argument("--x0", type=int, required=True)
#     parser.add_argument("--y0", type=int, required=True)
#     parser.add_argument("--x1", type=int, required=True)
#     parser.add_argument("--y1", type=int, required=True)
#     return parser.parse_args()


# def main():

#     args = parse_args()

#     config = {
#         "mog_history": 300,
#         "mog_var_threshold": 12,
#         "mog_learning_rate": 0.002,
#         "min_blob_area": 200,
#         "max_blob_area": 20000,
#         "morph_kernel_size": 5,
#         "axis_line_length": 60,
#         "arena_real_width_cm": 30.5,
#     }

#     detect(
#         input_video=args.input,
#         output_video=args.output_video,
#         output_csv=args.output_csv,
#         output_meta=args.output_meta,
#         arena_box=(
#             args.x0,
#             args.y0,
#             args.x1,
#             args.y1,
#         ),
#         detector_config=config,
#     )

# if __name__ == "__main__":
#     main()