from pathlib import Path
import argparse
import yaml
import cv2

from detector import (
    detect,
    detect_arena_multi_frame,
    select_arena_manually,
)

from trajectory import generate_trajectory


# ============================================================
# Configuration
# ============================================================

def load_config(path):
    """
    Load YAML configuration.
    """

    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# Paths
# ============================================================

def ensure_directories(config):
    """
    Create configured output directories.
    """

    output_config = config["output"]

    directories = [
        output_config["trajectory_csv_directory"],
        output_config["trajectory_video_directory"],
    ]

    intermediate = output_config.get("intermediate", {})

    if intermediate.get("enabled", False):
        directories.extend(
            [
                intermediate["detection_csv_directory"],
                intermediate["detection_video_directory"],
                intermediate["metadata_directory"],
            ]
        )

    for directory in directories:

        Path(directory).mkdir(
            parents=True,
            exist_ok=True,
        )


def find_input_videos(config):
    """
    Find all supported videos in the configured input directory.
    """

    input_config = config["input"]
    input_directory = Path(input_config["directory"])

    extensions = {extension.lower() for extension in input_config["extensions"]}

    if not input_directory.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{input_directory}"
        )

    videos = [
        path
        for path in input_directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in extensions
        )
    ]

    return sorted(videos)


# ============================================================
# Arena Discovery
# ============================================================

def get_first_frame(video_path):
    """
    Get the first frame of a video.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    return frame


def automatically_find_arenas(videos, config):
    """
    First pass:

    Try to automatically determine the arena for every video.

    Returns:
        arenas
        manual_queue

    arenas:
        {Path: (x0, y0, x1, y1)}

    manual_queue:
        [Path, ...]
    """

    arenas = {}
    manual_queue = []
    arena_config = config["arena"]
    max_frames = arena_config["max_search_frames"]

    print()
    print("=" * 60)
    print("ARENA DETECTION")
    print("=" * 60)

    for index, video_path in enumerate(videos, start=1):
        print()
        print(
            f"[{index}/{len(videos)}] "
            f"{video_path.name}"
        )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("  ERROR: Could not open video.")
            manual_queue.append(video_path)
            continue

        result = None
        if arena_config["auto_detect"]:
            result = detect_arena_multi_frame(cap, max_frames=max_frames)

        cap.release()

        if result is not None:
            x0, y0, x1, y1, width = result
            arenas[video_path] = (x0, y0, x1, y1)
            print("  ✓ Automatic arena detection succeeded.")
        else:
            print("  ! Automatic arena detection failed.")
            manual_queue.append(video_path)

    return arenas, manual_queue


# ============================================================
# Manual Arena Selection
# ============================================================

def manually_select_failed_arenas(manual_queue, arenas, config):
    """
    Second pass:

    Open each video for which automatic detection failed.

    The user can manually select the arena.

    This happens AFTER all automatic detection attempts have
    finished.
    """

    arena_config = config["arena"]

    if not manual_queue:
        return

    if not arena_config["manual_selection_on_failure"]:
        print()
        print(
            "Manual selection disabled."
        )
        return

    print()
    print("=" * 60)
    print("MANUAL ARENA SELECTION")
    print("=" * 60)

    print(
        f"{len(manual_queue)} video(s) "
        f"require manual arena selection."
    )

    for index, video_path in enumerate(manual_queue, start=1):
        print()
        print(
            f"[{index}/{len(manual_queue)}] "
            f"{video_path.name}"
        )

        frame = get_first_frame(video_path)
        if frame is None:
            print("  ERROR: Could not read video.")
            continue

        result = select_arena_manually(
            frame,
            display_height=arena_config["manual_selection_display_height"],
        )

        if result is None:
            print("  Manual selection cancelled.")
            if arena_config["use_full_frame_on_manual_cancel"]:
                frame_h, frame_w = (frame.shape[:2])
                arenas[video_path] = (0, 0, frame_w, frame_h)
                print("  Using full frame.")
            else:
                print("  Video will be skipped.")
            continue

        x0, y0, x1, y1, width = result

        arenas[video_path] = (x0, y0, x1, y1)

        print(
            f"  ✓ Manual arena selected: "
            f"X=[{x0}, {x1}], "
            f"Y=[{y0}, {y1}]"
        )


# ============================================================
# Output Paths
# ============================================================

def output_paths(video_path, config):
    """
    Generate all output paths for one input video.
    """

    stem = video_path.stem
    output_config = config["output"]

    trajectory_csv = (
        Path(output_config["trajectory_csv_directory"])
        / f"{stem}_trajectory.csv"
    )

    trajectory_video = (
        Path(output_config["trajectory_video_directory"])
        / f"{stem}_trajectory.mp4"
    )

    detection_csv = None
    detection_video = None
    metadata = None

    intermediate = output_config.get("intermediate", {})
    if intermediate.get("enabled", False):
        detection_csv = (
            Path(intermediate["detection_csv_directory"])
            / f"{stem}_detection.csv"
        )

        detection_video = (
            Path(intermediate["detection_video_directory"])
            / f"{stem}_detection.mp4"
        )

        metadata = (
            Path(intermediate["metadata_directory"])
            / f"{stem}_metadata.json"
        )

    return {
        "trajectory_csv": trajectory_csv,
        "trajectory_video": trajectory_video,
        "detection_csv": detection_csv,
        "detection_video": detection_video,
        "metadata": metadata,
    }


# ============================================================
# Processing
# ============================================================

def process_video(video_path, arena_box, config):
    """
    Run detector and trajectory stages for one video.
    """

    paths = output_paths(video_path, config)
    pipeline_config = config["pipeline"]
    intermediate = config["output"].get("intermediate", {})
    skip_existing = pipeline_config["skip_existing"]
    detection_enabled = intermediate.get("enabled", False)

    print()
    print("=" * 60)
    print(
        f"PROCESSING: {video_path.name}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Detector
    # --------------------------------------------------------

    if pipeline_config["run_detection"]:
        if (
            skip_existing
            and paths["detection_csv"] is not None
            and paths["metadata"] is not None
            and paths["detection_csv"].exists()
            and paths["metadata"].exists()
        ):
            print("Detection outputs already exist.")
        else:
            if not detection_enabled:
                # We still need a temporary CSV and metadata
                # because trajectory depends on them.
                #
                # These are deleted after trajectory generation.
                temporary_directory = (Path(".pipeline_tmp"))
                temporary_directory.mkdir(exist_ok=True)
                detection_csv = temporary_directory / f"{video_path.stem}_detection.csv"
                metadata = temporary_directory / f"{video_path.stem}_metadata.json"
                detection_video = None
            else:
                detection_csv = paths["detection_csv"]
                metadata = paths["metadata"]
                detection_video = paths["detection_video"]

            detect(
                input_video=video_path,
                output_video=detection_video,
                output_csv=detection_csv,
                output_meta=metadata,
                arena_box=arena_box,
                detector_config=config["detector"],
            )
    else:
        # If detector is disabled, the pipeline expects
        # existing intermediate files.
        detection_csv = paths["detection_csv"]
        metadata = paths["metadata"]

        if detection_csv is None or metadata is None:

            raise RuntimeError(
                "run_detection is false, but "
                "intermediate detection outputs "
                "are disabled."
            )

        if not detection_csv.exists():
            raise FileNotFoundError(
                f"Detection CSV not found: "
                f"{detection_csv}"
            )

        if not metadata.exists():
            raise FileNotFoundError(
                f"Metadata not found: "
                f"{metadata}"
            )

    # --------------------------------------------------------
    # Trajectory
    # --------------------------------------------------------

    if pipeline_config["run_trajectory"]:
        if (
            skip_existing
            and paths["trajectory_csv"].exists()
            and paths["trajectory_video"].exists()
        ):
            print("Trajectory outputs already exist.")

        else:
            generate_trajectory(
                input_video=video_path,
                input_csv=detection_csv,
                input_meta=metadata,
                output_video=paths["trajectory_video"],
                output_csv=paths["trajectory_csv"],
                trajectory_config=config["trajectory"],
            )

    # --------------------------------------------------------
    # Cleanup temporary detection outputs.
    # --------------------------------------------------------

    if not detection_enabled and pipeline_config["run_detection"]:
        try:
            detection_csv.unlink(missing_ok=True)
            metadata.unlink(missing_ok=True)
        except Exception:
            pass


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(config_path="config.yaml"):
    """
    Execute the complete Worm pipeline.
    """

    config = load_config(config_path)
    ensure_directories(config)
    videos = find_input_videos(config)

    if not videos:
        print("No input videos found.")
        return

    print()
    print("=" * 60)
    print("WORM ROBOT TRACKING PIPELINE")
    print("=" * 60)

    print(f"Found {len(videos)} video(s).")

    # --------------------------------------------------------
    # Pass 1:
    # Automatically find arenas for every video.
    # --------------------------------------------------------

    arenas, manual_queue = automatically_find_arenas(videos, config)

    # --------------------------------------------------------
    # Pass 2:
    # Manually resolve failures.
    # --------------------------------------------------------

    manually_select_failed_arenas(manual_queue, arenas, config)

    # --------------------------------------------------------
    # Process videos whose arenas are known.
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PROCESSING VIDEOS")
    print("=" * 60)

    successful = 0
    skipped = 0

    for video_path in videos:
        if video_path not in arenas:
            print()
            print(
                f"Skipping {video_path.name}: "
                f"no arena available."
            )
            skipped += 1
            continue

        try:
            process_video(video_path, arenas[video_path], config)
            successful += 1
        except Exception as exc:
            print()
            print(
                f"ERROR processing "
                f"{video_path.name}:"
            )
            print(exc)
            skipped += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Skipped/failed: {skipped}")
    print(f"Total videos: {len(videos)}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Worm "
            "robot tracking pipeline."
        )
    )

    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help=(
            "Path to YAML configuration "
            "file."
        ),
    )

    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()