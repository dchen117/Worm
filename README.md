# Worm Robot Tracking

A computer-vision pipeline for detecting and tracking a robot moving inside an arena.

The pipeline processes all supported videos in an input directory and can run the detection and trajectory stages automatically. Configuration is handled through a YAML file rather than separate command-line arguments for each stage.

## Project Structure

```text
Worm/
├── pipeline.py
├── detector.py
├── trajectory.py
├── config.yaml
├── requirements.txt
└── data/
    ├── input/
    ├── detection/
    │   ├── csv/
    │   ├── videos/
    │   └── metadata/
    ├── trajectory/
    │   ├── csv/
    │   └── videos/
    └── evaluation/
        └── annotations/
```

The output directories can be changed in `config.yaml`.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/dchen117/Worm.git
cd Worm
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the pipeline

Edit `config.yaml` to specify the input and output directories and pipeline settings.

For example:

```yaml
input:
  directory: "./data/input"
  extensions:
    - ".mp4"
    - ".avi"
    - ".mov"
    - ".mkv"

output:
  trajectory_csv_directory: "./data/trajectory/csv"
  trajectory_video_directory: "./data/trajectory/videos"

  intermediate:
    enabled: true
    detection_csv_directory: "./data/detection/csv"
    detection_video_directory: "./data/detection/videos"
    metadata_directory: "./data/detection/metadata"

pipeline:
    stages:
      - detection
      - trajectory
      - prediction

  skip_existing: false
```

Arena detection and detector/trajectory parameters can also be configured in this file.

## Running the Pipeline

Place your input videos in the configured input directory and run:

```bash
python pipeline.py
```

To use a different configuration file:

```bash
python pipeline.py --config experiment.yaml
```

The pipeline will:

1. Find all supported videos in the input directory.
2. Attempt to automatically detect the arena for each video.
3. Prompt for manual arena selection for videos where automatic detection fails.
4. Run the configured pipeline stages.
5. Save the resulting files to the configured output directories.

### Running Specific Stages

The stages are controlled by:

```yaml
pipeline:
  stages:
    - detection
    - trajectory
    - prediction
```

Run both detection and trajectory processing:

```yaml
pipeline:
  stages:
    - detection
    - trajectory
    # - prediction
```

Run trajectory processing using existing detection results:

```yaml
pipeline:
  stages:
    # - detection
    - trajectory
    # - prediction
```

When running the trajectory stage by itself, the corresponding detection outputs must already exist.

Run prediction using trajectory results and manually annotated endpoints:

```yaml
pipeline:
  stages:
    # - detection
    # - trajectory
    - prediction
```

When running prediction stage by itself, the trajectory results and manually annotated endpoints must already exist. The manually annotated endpoints can be collected with annotate.py.

## Arena Selection

By default, the pipeline can use automatic arena detection with manual fallback:

```yaml
arena:
  auto_detect: true
  manual_selection_on_failure: true
```

The manual selection frame and display settings can also be configured in `config.yaml`.

## Output

Depending on the enabled stages, the pipeline can produce:

- Detection CSV files
- Detection/annotated videos
- Detection metadata
- Trajectory CSV files
- Trajectory videos

All output locations are controlled by `config.yaml`.

## Summary

The pipeline is intended to be run through a single entry point:

```bash
python pipeline.py
```

`pipeline.py` handles configuration, video discovery, arena selection, and orchestration, while `detector.py` and `trajectory.py` provide the individual processing stages.
