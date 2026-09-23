from pathlib import Path
import argparse
import cv2
import yaml

from detector import detect, detect_arena_multi_frame, select_arena_manually
from trajectory import generate_trajectory
from evaluation import evaluate


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_directories(config):
    output = config["output"]
    directories = [
        output["trajectory_csv_directory"],
        output["trajectory_video_directory"],
    ]

    intermediate = output.get("intermediate", {})
    if intermediate.get("enabled", False):
        directories += [
            intermediate["detection_csv_directory"],
            intermediate["detection_video_directory"],
            intermediate["metadata_directory"],
        ]

    prediction = config.get("prediction", {})
    if "prediction" in config.get("pipeline", {}).get("stages", []):
        directories.append(prediction["output_directory"])
        directories.append(prediction["annotation_directory"])

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def find_input_videos(config):
    input_config = config["input"]
    directory = Path(input_config["directory"])

    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")

    extensions = {x.lower() for x in input_config["extensions"]}
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def find_annotation_files(config):
    directory = Path(config["prediction"]["annotation_directory"])

    if not directory.exists():
        raise FileNotFoundError(
            f"Annotation directory does not exist: {directory}"
        )

    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".csv")


def get_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def automatically_find_arenas(videos, config):
    arenas = {}
    manual_queue = []
    arena_config = config["arena"]
    max_frames = arena_config["max_search_frames"]

    print("\n" + "=" * 60)
    print("ARENA DETECTION")
    print("=" * 60)

    for index, video_path in enumerate(videos, start=1):
        print(f"\n[{index}/{len(videos)}] {video_path.name}")
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
            x0, y0, x1, y1, _ = result
            arenas[video_path] = (x0, y0, x1, y1)
            print("  ✓ Automatic arena detection succeeded.")
        else:
            print("  ! Automatic arena detection failed.")
            manual_queue.append(video_path)

    return arenas, manual_queue


def manually_select_failed_arenas(manual_queue, arenas, config):
    arena_config = config["arena"]

    if not manual_queue or not arena_config["manual_selection_on_failure"]:
        return

    print("\n" + "=" * 60)
    print("MANUAL ARENA SELECTION")
    print("=" * 60)

    for index, video_path in enumerate(manual_queue, start=1):
        print(f"\n[{index}/{len(manual_queue)}] {video_path.name}")
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
                h, w = frame.shape[:2]
                arenas[video_path] = (0, 0, w, h)
                print("  Using full frame.")
            continue

        x0, y0, x1, y1, _ = result
        arenas[video_path] = (x0, y0, x1, y1)


def output_paths(video_path, config):
    stem = video_path.stem
    output = config["output"]

    paths = {
        "trajectory_csv": Path(output["trajectory_csv_directory"]) / f"{stem}_trajectory.csv",
        "trajectory_video": Path(output["trajectory_video_directory"]) / f"{stem}_trajectory.mp4",
        "detection_csv": None,
        "detection_video": None,
        "metadata": None,
    }

    intermediate = output.get("intermediate", {})
    if intermediate.get("enabled", False):
        paths["detection_csv"] = (
            Path(intermediate["detection_csv_directory"]) / f"{stem}_detection.csv"
        )
        paths["detection_video"] = (
            Path(intermediate["detection_video_directory"]) / f"{stem}_detection.mp4"
        )
        paths["metadata"] = (
            Path(intermediate["metadata_directory"]) / f"{stem}_metadata.json"
        )

    return paths


def run_detection(video_path, arena_box, config, paths):
    intermediate = config["output"].get("intermediate", {})
    persistent = intermediate.get("enabled", False)

    if persistent:
        detection_csv = paths["detection_csv"]
        metadata = paths["metadata"]
        detection_video = paths["detection_video"]
    else:
        temp = Path(".pipeline_tmp")
        temp.mkdir(exist_ok=True)
        detection_csv = temp / f"{video_path.stem}_detection.csv"
        metadata = temp / f"{video_path.stem}_metadata.json"
        detection_video = None

    detect(
        input_video=video_path,
        output_video=detection_video,
        output_csv=detection_csv,
        output_meta=metadata,
        arena_box=arena_box,
        detector_config=config["detector"],
    )

    return detection_csv, metadata, persistent

def run_predictions(config):
    prediction = config["prediction"]
    annotation_files = find_annotation_files(config)
    trajectory_dir = Path(config["output"]["trajectory_csv_directory"])
    arena_real_width_cm = config["detector"]["arena_real_width_cm"]

    successful = skipped = 0

    print("\n" + "=" * 60)
    print("PREDICTION / EVALUATION")
    print("=" * 60)

    for annotation_path in annotation_files:
        stem = annotation_path.stem.removesuffix("_test")
        trajectory_path = trajectory_dir / f"{stem}_trajectory.csv"

        if not trajectory_path.exists():
            print(
                f"Skipping {annotation_path.name}: "
                f"no matching trajectory CSV."
            )
            skipped += 1
            continue

        try:
            output_directory = Path(prediction["output_directory"]) / stem

            evaluate(
                trajectory_csv=trajectory_path,
                annotations_csv=annotation_path,
                output_directory=output_directory,
                fps=prediction["fps"],
                predictors=prediction.get(
                    "predictors",
                    ["stationary", "constant_velocity"],
                ),
                horizon_seconds=prediction.get("horizon_seconds", 1.0),
                velocity_window_seconds=prediction.get(
                    "velocity_window_seconds", 0.5
                ),
                arena_real_width_cm=arena_real_width_cm,
            )

            print(f"Evaluated: {annotation_path.name}")
            successful += 1

        except Exception as exc:
            print(f"ERROR evaluating {annotation_path.name}:")
            print(exc)
            skipped += 1

    print(f"\nSuccessful: {successful}")
    print(f"Skipped/failed: {skipped}")

def process_video(video_path, arena_box, config):
    stages = config["pipeline"]["stages"]
    paths = output_paths(video_path, config)
    skip_existing = config["pipeline"].get("skip_existing", False)

    print("\n" + "=" * 60)
    print(f"PROCESSING: {video_path.name}")
    print("=" * 60)

    detection_csv = paths["detection_csv"]
    metadata = paths["metadata"]
    persistent_detection = config["output"].get("intermediate", {}).get("enabled", False)

    if "detection" in stages:
        if (
            skip_existing
            and persistent_detection
            and paths["detection_csv"].exists()
            and paths["metadata"].exists()
        ):
            detection_csv, metadata = paths["detection_csv"], paths["metadata"]
            print("Detection outputs already exist.")
        else:
            detection_csv, metadata, persistent_detection = run_detection(
                video_path, arena_box, config, paths
            )

    elif "trajectory" in stages:
        if detection_csv is None or metadata is None:
            raise RuntimeError(
                "Trajectory stage requires existing detection CSV and metadata."
            )
        if not detection_csv.exists() or not metadata.exists():
            raise FileNotFoundError("Required detection outputs do not exist.")

    if "trajectory" in stages:
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

    if "detection" in stages and not persistent_detection:
        detection_csv.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)


def run_pipeline(config_path="config.yaml"):
    config = load_config(config_path)
    stages = config["pipeline"]["stages"]

    valid_stages = {"detection", "trajectory", "prediction"}
    invalid = set(stages) - valid_stages
    if invalid:
        raise ValueError(f"Unknown pipeline stages: {sorted(invalid)}")

    if "trajectory" in stages and "detection" not in stages:
        print("Using existing detection outputs.")
    if "prediction" in stages and "trajectory" not in stages:
        print("Using existing trajectory outputs.")

    ensure_directories(config)
    videos = find_input_videos(config)

    if not videos:
        print("No input videos found.")
        return

    arenas = {}
    if "detection" in stages:
        arenas, manual_queue = automatically_find_arenas(videos, config)
        manually_select_failed_arenas(manual_queue, arenas, config)

    successful = skipped = 0

    if "detection" in stages or "trajectory" in stages:
        print("\n" + "=" * 60)
        print("PROCESSING VIDEOS")
        print("=" * 60)

        for video_path in videos:
            if "detection" in stages and video_path not in arenas:
                print(f"\nSkipping {video_path.name}: no arena available.")
                skipped += 1
                continue

            try:
                arena = arenas.get(video_path)
                process_video(video_path, arena, config)
                successful += 1
            except Exception as exc:
                print(f"\nERROR processing {video_path.name}:")
                print(exc)
                skipped += 1

        print("\n" + "=" * 60)
        print("VIDEO PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Successful: {successful}")
        print(f"Skipped/failed: {skipped}")
        print(f"Total videos: {len(videos)}")

    if "prediction" in stages:
        run_predictions(config)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run the Worm robot tracking pipeline.")
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
