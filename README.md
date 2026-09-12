# Intelligent Traffic Surveillance & Vehicle Analytics

A command-line Computer Vision project for traffic-video analysis. The system detects vehicles, tracks them across frames, records trajectories and motion, supports perspective correction with homography, and exports an annotated video plus CSV/JSON analytics.

![Traffic Vision Analytics Demo](docs/demo_preview.jpg)

## Course alignment
The implementation connects directly to the CSE3010 Computer Vision syllabus:

- **Module 1:** image/video processing and geometric image representation.
- **Module 2:** perspective geometry and homography for road-plane mapping.
- **Module 3:** object detection and feature-analysis context.
- **Module 4:** background subtraction, tracking and motion analysis.

The project is designed as an application rather than a collection of unrelated algorithm demos.

## Main functional modules
1. **Vehicle Detection** — YOLO detects car, motorcycle, bus and truck classes. MOG2 is retained as a motion-detection baseline.
2. **Multi-Object Tracking** — class-aware centroid/IoU matching maintains vehicle IDs and trajectories.
3. **Perspective & Speed Module** — optional four-point homography maps road pixels to measured world coordinates. Without calibration, motion is reported in pixels/s.
4. **Traffic Analytics** — class counts, active vehicles, track counts and speed statistics are exported.
5. **Visualization & Reporting** — annotated MP4, CSV observations and JSON summary are generated automatically.

## Requirements
- Windows, Linux or macOS
- Python 3.10–3.14
- Terminal/PowerShell
- A CPU is sufficient for testing; a CUDA-capable GPU can be used if supported by the local PyTorch installation.

## Setup
Open a terminal in the repository root.

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows CMD:

```bat
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first YOLO run downloads the selected pretrained model automatically.

## Input video
Put a traffic video inside `data/`. Example:

```text
data/traffic.mp4
```

Do not commit large videos to GitHub. Keep the dataset/video source and license information in your report.

## Run the complete pipeline

```bash
python main.py --input data/traffic.mp4
```

The default configuration uses YOLO on CPU at 640px inference size.

### Faster development test

For a short test:

```bash
python main.py --input data/traffic.mp4 --max-frames 300 --imgsz 640
```

For a 4K video, this is the recommended first run.

### Process every second frame

```bash
python main.py --input data/traffic.mp4 --skip 1 --imgsz 640
```

### Optional preview window

```bash
python main.py --input data/traffic.mp4 --show
```

The project remains fully usable without `--show`; command-line execution is the normal mode.

## Outputs
After execution:

```text
outputs/
├── annotated_video.mp4
├── vehicle_data.csv
└── traffic_summary.json
```

### annotated_video.mp4
Contains bounding boxes, vehicle class, confidence, persistent ID, trajectory and motion value.

### vehicle_data.csv
Contains one row per visible tracked vehicle observation:

```text
frame, vehicle_id, class, confidence, x, y, width, height, speed, speed_unit
```

### traffic_summary.json
Contains frame count, average active vehicles, number of tracks, class statistics and speed statistics.

## Speed calibration
Do **not** interpret pixel/s as km/h. For real-world speed, measured road geometry is required.

Copy:

```text
config/calibration.example.json
```

to a project-specific calibration file and replace the four image points with four road-plane points from your actual video. Replace the corresponding world points with measured distances in metres.

Then run:

```bash
python main.py --input data/traffic.mp4 --calibration config/calibration.json
```

The transformation follows:

\[
\mathbf{x}' \sim H\mathbf{x}
\]

where `H` is the 3×3 planar homography. Only after measured calibration should the resulting displacement be interpreted in metres/s and converted to km/h:

\[
v_{km/h}=3.6v_{m/s}
\]

## Detector comparison
The course-aligned MOG2 baseline can be run with:

```bash
python main.py --input data/traffic.mp4 --detector mog2 --max-frames 300
```

This is useful for discussing the difference between **motion-based detection** and **semantic object detection** in the report.

## Testing
Run:

```bash
pytest -q
```

The tests cover tracking persistence, IoU, homography mapping, speed calculation and analytics aggregation.

## Limitations
- A fixed-camera view is assumed.
- YOLO detections can be affected by severe occlusion, blur and unusual viewpoints.
- A centroid/IoU tracker is intentionally lightweight and is not equivalent to a production-grade MOT tracker.
- Real-world speed requires measured camera/road calibration.
- This project is an academic prototype and should not be used as an enforcement system without validated calibration and evaluation.

## Suggested evaluation
For the final report, evaluate:

- detection precision/recall on a manually labelled sample,
- track continuity and ID switches,
- processing FPS,
- speed MAE if ground-truth speed is available,
- comparison of YOLO and MOG2 under moving/stationary vehicles.

## Repository structure

```text
traffic-vision-analytics/
├── main.py
├── requirements.txt
├── README.md
├── statement.md
├── config/
├── src/
│   ├── detector.py
│   ├── tracker.py
│   ├── homography.py
│   ├── speed_estimator.py
│   ├── traffic_analyzer.py
│   ├── visualizer.py
│   └── video_processor.py
├── tests/
├── data/
├── outputs/
├── docs/
└── report/
```

## License
MIT. See `LICENSE`.
