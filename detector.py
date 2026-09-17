import cv2
import numpy as np
import csv


# ============================================================
# Configuration
# ============================================================

INPUT_VIDEO = "./data/20260915_082555.mp4"

OUTPUT_VIDEO = "task1_annotated.mp4"
OUTPUT_CSV = "task1_results.csv"


# Arena region in the ORIGINAL video.
#
# The arena occupies approximately this region. These 
# values can be adjusted later if necessary.
#
# Format:
# x from ARENA_X0 to ARENA_X1
# y from ARENA_Y0 to ARENA_Y1

ARENA_X0 = 70
ARENA_Y0 = 70
ARENA_X1 = 1010
ARENA_Y1 = 930


# IMPORTANT:
#
# Measure the arena width in pixels in the first frame.
# Replace this value with the measured width.
#
# cm_per_pixel = 30.5 / arena_width_pixels
#
ARENA_WIDTH_PX = 940


# Background subtractor settings

MOG_HISTORY = 300
MOG_VAR_THRESHOLD = 12

# How quickly the background model adapts.
#
# Smaller = slower adaptation.
#
# Start here and tune later if necessary.
MOG_LEARNING_RATE = 0.002


# Ignore blobs smaller/larger than these.
#
# These are deliberately loose for Version 1.
MIN_BLOB_AREA = 200
MAX_BLOB_AREA = 20000


# Morphological filtering
MORPH_KERNEL_SIZE = 5


# Length of the orientation line in pixels
AXIS_LINE_LENGTH = 60


# ============================================================
# Helper functions
# ============================================================

def create_robot_mask(frame, background_subtractor):
    """
    Apply MOG2 background subtraction and clean the resulting
    foreground mask.
    """

    # Only process the arena.
    arena = frame[
        ARENA_Y0:ARENA_Y1,
        ARENA_X0:ARENA_X1
    ]

    # Background subtraction
    mask = background_subtractor.apply(
        arena,
        learningRate=MOG_LEARNING_RATE
    )

    # MOG2 can produce grayscale values representing shadows.
    # We only want definite foreground pixels.
    _, mask = cv2.threshold(
        mask,
        200,
        255,
        cv2.THRESH_BINARY
    )

    # Morphological cleanup
    kernel = np.ones(
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
        np.uint8
    )

    # Remove small isolated noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def find_robot_contour(mask):
    """
    Find the most likely robot contour.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
            candidates.append(contour)

    if not candidates:
        return None

    # Version 1 assumption:
    # the robot is the largest foreground object.
    robot_contour = max(
        candidates,
        key=cv2.contourArea
    )

    return robot_contour


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

    The PCA direction corresponding to the largest eigenvalue
    is the robot's longest axis.
    """

    # Create a mask containing ONLY the selected contour.
    robot_mask = np.zeros_like(mask)

    cv2.drawContours(
        robot_mask,
        [contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # Get coordinates of all robot pixels.
    ys, xs = np.where(robot_mask == 255)

    if len(xs) < 2:
        return None

    points = np.column_stack((xs, ys))

    # Center the points.
    mean = np.mean(points, axis=0)

    centered = points - mean

    # Covariance matrix.
    covariance = np.cov(
        centered,
        rowvar=False
    )

    # Eigenvectors/eigenvalues.
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    # Eigenvector corresponding to largest eigenvalue.
    principal_axis = eigenvectors[
        :, np.argmax(eigenvalues)
    ]

    dx = principal_axis[0]
    dy = principal_axis[1]

    # Convert vector to angle.
    theta = np.degrees(
        np.arctan2(dy, dx)
    )

    # Body axis is defined modulo 180 degrees.
    theta = theta % 180

    return theta


def pixel_to_cm(x, y):
    """
    Convert arena-relative pixel coordinates to cm.
    """

    cm_per_pixel = 30.5 / ARENA_WIDTH_PX

    x_cm = x * cm_per_pixel
    y_cm = y * cm_per_pixel

    return x_cm, y_cm


def annotate_frame(
    frame,
    contour,
    cx,
    cy,
    theta
):
    """
    Draw the robot outline, center, and orientation axis.
    """

    # Draw contour.
    cv2.drawContours(
        frame,
        [
            contour + np.array(
                [[[ARENA_X0, ARENA_Y0]]]
            )
        ],
        -1,
        (255, 0, 0),
        2
    )

    # Convert arena-relative coordinates to full-frame coordinates.
    full_cx = int(cx + ARENA_X0)
    full_cy = int(cy + ARENA_Y0)

    # Draw center.
    cv2.circle(
        frame,
        (full_cx, full_cy),
        5,
        (0, 255, 0),
        -1
    )

    # Orientation axis.
    angle_rad = np.radians(theta)

    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    x1 = int(
        full_cx - AXIS_LINE_LENGTH * dx
    )
    y1 = int(
        full_cy - AXIS_LINE_LENGTH * dy
    )

    x2 = int(
        full_cx + AXIS_LINE_LENGTH * dx
    )
    y2 = int(
        full_cy + AXIS_LINE_LENGTH * dy
    )

    cv2.line(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2
    )

    # Text information.
    cv2.putText(
        frame,
        f"Center: ({full_cx}, {full_cy})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Theta: {theta:.1f} deg",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


# ============================================================
# Main
# ============================================================

def main():

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {INPUT_VIDEO}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")

    # Create background subtractor.
    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=MOG_HISTORY,
        varThreshold=MOG_VAR_THRESHOLD,
        detectShadows=False
    )

    # Video writer.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    # CSV output.
    csv_file = open(
        OUTPUT_CSV,
        "w",
        newline=""
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "frame",
        "x_cm",
        "y_cm",
        "theta_deg",
        "detected"
    ])

    detected_count = 0

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        mask = create_robot_mask(
            frame,
            background_subtractor
        )

        contour = find_robot_contour(mask)

        detected = False
        x_cm = ""
        y_cm = ""
        theta = ""

        if contour is not None:

            center = calculate_centroid(contour)

            if center is not None:

                cx, cy = center

                theta_result = calculate_orientation(
                    contour,
                    mask
                )

                if theta_result is not None:

                    theta = theta_result

                    # Convert arena-relative pixel position
                    # to centimeters.
                    x_cm, y_cm = pixel_to_cm(
                        cx,
                        cy
                    )

                    detected = True
                    detected_count += 1

                    # Draw result.
                    annotate_frame(
                        frame,
                        contour,
                        cx,
                        cy,
                        theta
                    )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        csv_writer.writerow([
            frame_number,
            x_cm,
            y_cm,
            theta,
            int(detected)
        ])

        # ----------------------------------------------------
        # Display status
        # ----------------------------------------------------

        if not detected:

            cv2.putText(
                frame,
                "ROBOT NOT DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Write annotated frame.
        writer.write(frame)

        frame_number += 1

        # Progress.
        if frame_number % 100 == 0:

            percentage = (
                100 * frame_number / frame_count
            )

            print(
                f"Processed {frame_number}/{frame_count} "
                f"({percentage:.1f}%)"
            )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()
    writer.release()
    csv_file.close()

    detection_fraction = (
        detected_count / frame_number
    )

    print()
    print("Finished.")
    print(f"Frames processed: {frame_number}")
    print(f"Frames detected: {detected_count}")
    print(
        f"Detection fraction: "
        f"{detection_fraction:.3f}"
    )
    print(f"Annotated video: {OUTPUT_VIDEO}")
    print(f"CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()