# System Architecture & Technical Specification

## Intelligent Traffic Surveillance & Vehicle Analytics

---

## 1. Architectural Overview

The **Traffic Vision Analytics** system is an end-to-end, modular computer vision pipeline designed for automated traffic surveillance, vehicle localization, trajectory tracking, planar perspective correction, and quantitative kinematic reporting from fixed-view camera streams.

The architecture is explicitly decoupled into independent functional units adhering to single-responsibility design:
```
                               +-----------------------------+
                               |     Input Video Stream      |
                               |  (MP4 / AVI / 4K UHD 50FPS) |
                               +--------------+--------------+
                                              |
                                              v
                               +-----------------------------+
                               |       VideoProcessor        |
                               | (Frame Ingestion & Control) |
                               +--------------+--------------+
                                              |
                                              v
                               +-----------------------------+
                               |       VehicleDetector       |
                               |   (YOLO11n / OpenCV MOG2)   |
                               +--------------+--------------+
                                              | Detections [bbox, centroid, class, conf]
                                              v
                               +-----------------------------+
                               |     MultiObjectTracker      |
                               | (Centroid Distance + IoU)   |
                               +--------------+--------------+
                                              | Persistent Tracks [track_id, history, bbox]
                                              v
                       +----------------------+----------------------+
                       |                                             |
                       v                                             v
        +-----------------------------+               +-----------------------------+
        |    HomographyTransformer    |               |       SpeedEstimator        |
        |  (Planar 4-Point Homography)| ------------> | (Smoothed Temporal Velocity)|
        +-----------------------------+ (World Coord) +--------------+--------------+
                                                                     |
                                                                     v
                                                      +-----------------------------+
                                                      |       TrafficAnalyzer       |
                                                      | (Statistical Aggregation)   |
                                                      +--------------+--------------+
                                                                     |
                                      +------------------------------+------------------------------+
                                      |                              |                              |
                                      v                              v                              v
                        +---------------------------+  +---------------------------+  +---------------------------+
                        |    Annotated MP4 Video    |  |     vehicle_data.csv      |  |   traffic_summary.json    |
                        | (HUD, BBoxes, Trajectories|  | (Frame-by-frame kinematics|  | (Overall counts, avg/max  |
                        |   & Speed Overlays)       |  |   & vehicle observations) |  |   speeds, congestion lvl) |
                        +---------------------------+  +---------------------------+  +---------------------------+
```

---

## 2. Component Specifications

### 2.1 VideoProcessor (`src/video_processor.py`)
- **Role**: Pipeline orchestrator and I/O controller.
- **Responsibilities**:
  - Ingests video streams using `cv2.VideoCapture`.
  - Controls inference pacing via optional frame skipping (`--skip N`).
  - Pipes decoded frames through detection, tracking, speed calculation, and visual rendering.
  - Emits real-time progress diagnostics to stdout for headless monitoring.
  - Concurrently streams annotated frames to a `cv2.VideoWriter` container (MP4V) and tabular records to CSV.

### 2.2 VehicleDetector (`src/detector.py`)
- **Role**: Object detection subsystem with dual-backend support.
- **Backends**:
  1. **YOLO (Primary Semantic Engine)**:
     - Leverages Ultralytics YOLO11n for real-time deep semantic detection.
     - Constrained to relevant COCO vehicle classes: Class 2 (Car), Class 3 (Motorcycle), Class 5 (Bus), Class 7 (Truck).
     - Configurable input resolution (`--imgsz`, default 640px) and confidence threshold (`--conf`, default 0.35).
  2. **MOG2 (Course-Aligned Baseline)**:
     - Implements Gaussian Mixture-based Background/Foreground Subtraction (`cv2.createBackgroundSubtractorMOG2`).
     - Includes morphological opening (erosion followed by dilation) and closing to suppress shadows and noise.
     - Retained for direct alignment with CSE3010 Module 4 curriculum, illustrating the trade-offs between motion-based foreground segmentation and deep semantic classification.

### 2.3 MultiObjectTracker (`src/tracker.py`)
- **Role**: Inter-frame temporal association and trajectory maintenance.
- **Algorithm**:
  - Predicts associations using a dual-metric cost function: Euclidean centroid proximity bounded by `--max-distance` and bounding-box Intersection-over-Union (IoU) with class consistency.
  - Maintains track state (`active`, `missed`, `total_hits`).
  - Supports occlusion tolerance: tracks persist across `--max-missed` frames before retirement, preventing premature ID fragmentation when vehicles are temporarily obscured.

### 2.4 HomographyTransformer (`src/homography.py`)
- **Role**: Geometric perspective correction and road-plane coordinate transformation.
- **Mathematical Principle**:
  - Models the camera-to-ground projection as a 2D planar projective transformation:
    $$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
  - Computes $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ using 4 corresponding coplanar landmarks: 4 image coordinates on the road surface $\leftrightarrow$ 4 real-world metric coordinates (metres).

### 2.5 SpeedEstimator (`src/speed_estimator.py`)
- **Role**: Discrete kinematic velocity estimation.
- **Formulation**:
  - Maintains a sliding temporal window of past centroids (size $N=5$).
  - Evaluates displacement between window endpoints:
    $$d = \sqrt{(X_t - X_{t - \Delta t})^2 + (Y_t - Y_{t - \Delta t})^2}$$
    $$\Delta t = \frac{\Delta \text{frames}}{\text{FPS}}$$
    $$v = \frac{d}{\Delta t}$$
  - When homography is active, $d$ is in metres, yielding instantaneous velocity in $\text{m/s}$.
  - Real-world velocity is converted to physical road units:
    $$v_{\text{km/h}} = 3.6 \times v_{\text{m/s}}$$
  - If uncalibrated, $d$ represents pixel displacement, and speed is strictly tagged as `pixels/s`.

### 2.6 TrafficAnalyzer (`src/traffic_analyzer.py`)
- **Role**: Statistical aggregation and traffic flow indexing.
- **Metrics Computed**:
  - Total frame count and unique vehicle track count.
  - Frame-average active vehicle occupancy.
  - Categorical distribution across vehicle classes (`car`, `bus`, `truck`, `motorcycle`).
  - Mean and peak velocity across all vehicles.
  - Congestion Level indexing: classified into `Light`, `Moderate`, or `Heavy` based on active lane saturation.

### 2.7 Visualizer (`src/visualizer.py`)
- **Role**: Dynamic graphical overlay and heads-up display (HUD).
- **Features**:
  - Resolution-adaptive typography and bounding-box scaling (supporting 720p through 4K UHD).
  - Multi-line semi-transparent HUD banner with live frame count, active vehicle counts, class breakdown, and calibration indicator.
  - Trajectory trail rendering depicting past vehicle motion paths.

---

## 3. Data Interfaces & Schemas

### 3.1 Road Plane Calibration Schema (`config/calibration.json`)
```json
{
  "camera": "Fixed traffic surveillance camera",
  "resolution": "3840x2160",
  "road_plane_description": "Road surface plane between near crosswalk and stop line",
  "image_points": [[1450, 1450], [2900, 1450], [3500, 2160], [600, 2160]],
  "world_points": [[0.0, 40.0], [14.0, 40.0], [14.0, 0.0], [0.0, 0.0]],
  "units": { "world_coordinates": "metres", "speed": "m/s and km/h" }
}
```

### 3.2 Vehicle Observation Schema (`outputs/vehicle_data.csv`)
| Column | Type | Description |
|---|---|---|
| `frame` | Integer | Zero-indexed frame number |
| `vehicle_id` | Integer | Persistent tracker identity |
| `class` | String | Semantic class label (`car`, `bus`, `truck`) |
| `confidence` | Float | Detection confidence score $[0.0, 1.0]$ |
| `x`, `y` | Integer | Centroid coordinates in image space |
| `width`, `height` | Integer | Bounding box dimensions |
| `speed` | Float | Calculated velocity ($m/s$ if calibrated, else $px/s$) |
| `speed_unit` | String | Measurement unit (`m/s` or `pixels/s`) |
| `speed_kmh` | Float/String | Physical speed in $\text{km/h}$ (or `N/A`) |

### 3.3 Analytics Summary Schema (`outputs/traffic_summary.json`)
```json
{
  "frames_processed": 300,
  "unique_vehicle_tracks": 29,
  "average_active_vehicles_per_frame": 7.62,
  "congestion_level": "Moderate",
  "total_vehicle_detections": 2285,
  "vehicle_detections_by_class": { "car": 1915, "bus": 351, "truck": 19 },
  "average_speed": 10.42,
  "maximum_speed": 18.85,
  "average_speed_kmh": 37.51,
  "maximum_speed_kmh": 67.86,
  "speed_unit": "m/s",
  "speed_calibrated": true,
  "input_resolution": "3840x2160",
  "video_fps": 50.0,
  "output_video": "outputs/annotated_video.mp4"
}
```

---

## 4. Error Handling & Robustness

1. **Input Validation**: `main.py` explicitly tests for input video presence, valid codec decompression, and directory write permissions before initiating processing.
2. **Missing Video Grace**: Terminates with a structured stderr diagnostic and returncode `1` rather than an unhandled Python traceback.
3. **Calibration Fail-Safe**: Validates 4-point quadrilateral geometry prior to computing $\mathbf{H}$; if omitted, gracefully falls back to uncalibrated pixel kinematics with clear HUD tagging.
4. **Headless Resilience**: Defaults to non-GUI processing, ensuring flawless execution on remote Linux servers, Docker containers, and automated grading sandboxes without `DISPLAY` environment variables.
