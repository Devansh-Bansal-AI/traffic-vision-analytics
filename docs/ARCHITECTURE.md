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
- **Role**: Statistical aggregation, kinematic distribution, and transportation engineering analysis.
- **Metrics Computed**:
  - **Highway Capacity Manual (HCM) Level of Service (LOS)**: Evaluates road operating conditions (LOS A through F) derived from physical vehicle density per lane-kilometer and operating speed.
  - **Estimated Hourly Flow Rate ($Q$)**: Extrapolates observed vehicle throughput to standard vehicles-per-hour ($\text{vph}$) equivalent.
  - **85th-Percentile Operating Velocity ($V_{85}$)**: The fundamental metric used by transportation engineers worldwide to assess speed compliance and design consistency.
  - **15th-Percentile Velocity ($V_{15}$)**: Lower-tail distribution identifying impeded or slow-moving vehicles.
  - **Speed Dispersion**: Measures standard deviation ($\sigma_v$) and Coefficient of Variation ($CV = \sigma_v / \mu_v$) to characterize turbulence in the traffic stream.
  - **Fleet Modal Split**: Quantifies proportional volume of passenger cars vs. heavy commercial transport (buses, trucks) alongside the Heavy Vehicle Percentage ($P_{HV}$).
  - **Congestion Level**: Qualitative traffic density index (`Light`, `Moderate`, `Heavy`).

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
  "peak_active_vehicles": 10,
  "traffic_density_veh_per_lane_km": 43.87,
  "level_of_service": "LOS F (Forced/Breakdown Flow - Heavy congestion)",
  "congestion_level": "Moderate",
  "estimated_hourly_flow_rate_vph": 27000,
  "total_vehicle_detections": 2285,
  "vehicle_detections_by_class": { "car": 1915, "bus": 351, "truck": 19 },
  "fleet_modal_split_percentage": { "car": 83.8, "bus": 15.4, "truck": 0.8 },
  "heavy_vehicle_percentage": 16.2,
  "average_speed": 9.084,
  "maximum_speed": 37.552,
  "speed_std_dev": 7.587,
  "speed_coeff_variation": 1.01,
  "speed_percentile_15": 1.138,
  "speed_percentile_85": 15.213,
  "speed_unit": "m/s",
  "speed_calibrated": true,
  "average_speed_kmh": 32.70,
  "maximum_speed_kmh": 135.19,
  "speed_percentile_85_kmh": 54.77,
  "speed_percentile_15_kmh": 4.10,
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
