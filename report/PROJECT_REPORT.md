# Project Report: Intelligent Traffic Surveillance & Vehicle Analytics

---

## 1. Project Information & Metadata

| Field | Detail |
|---|---|
| **Student Name** | Devansh Bansal |
| **GitHub Profile** | [Devansh-Bansal-AI](https://github.com/Devansh-Bansal-AI) |
| **Course** | CSE3010 – Computer Vision |
| **Project Title** | Intelligent Traffic Surveillance & Vehicle Analytics using Computer Vision |
| **Repository URL** | [https://github.com/Devansh-Bansal-AI/traffic-vision-analytics](https://github.com/Devansh-Bansal-AI/traffic-vision-analytics) |
| **Evaluation Platform**| VITyarthi Automated & Faculty Assessment |
| **Submission Date** | September 2026 |

---

## 2. Executive Abstract

Modern intelligent transportation systems (ITS) require autonomous, reliable, and scalable mechanisms for monitoring vehicular movement across urban road networks. This project implements an end-to-end, headless command-line Computer Vision pipeline for traffic video analytics.

The system integrates:
1. **Semantic Vehicle Localization** using Ultralytics YOLO11n, targeting cars, buses, trucks, and motorcycles.
2. **Curriculum-Aligned Baseline** via OpenCV MOG2 background subtraction with morphological filtering (CSE3010 Module 4).
3. **Multi-Object Tracking (MOT)** utilizing class-constrained Euclidean centroid proximity and bounding-box Intersection over Union (IoU) with temporal occlusion persistence.
4. **Planar Perspective Homography** ($\mathbf{x}' \sim \mathbf{H}\mathbf{x}$) mapping 2D image coordinates to physical ground-plane coordinates (metres), enabling true vehicle velocity estimation in physical units ($\text{km/h}$ and $\text{m/s}$).
5. **Headless Execution & Machine-Readable Reporting**, outputting an annotated MP4 video with an adaptive heads-up display (HUD), granular frame-by-frame kinematics logs (`vehicle_data.csv`), and summary analytics (`traffic_summary.json`).

The pipeline is fully evaluated on a 4K UHD 50 FPS urban traffic dataset, achieving robust multi-vehicle tracking, realistic velocity estimates ($32.7\text{ km/h}$ average urban flow), and verified test coverage across 10 automated unit tests.

---

## 3. Problem Statement & Motivation

Manual traffic monitoring via human observers or static speed radars is labor-intensive, error-prone, and geographically restricted. Furthermore, classical video surveillance typically records footage without automated extraction of quantitative metrics such as vehicle counts, class distributions, lane occupancy, or real-time speed profiles.

The objective of this project is to build an automated, reproducible, headless computer vision pipeline capable of:
- Operating without graphical interface overhead (headless CLI execution).
- Discriminating distinct vehicle categories reliably under heavy urban traffic.
- Maintaining consistent track identities through brief occlusions.
- Converting camera-plane pixel displacements into physical velocity measurements via planar homography.
- Providing immediate machine-readable CSV and JSON analytics for traffic planning.

---

## 4. Course Alignment (CSE3010 Syllabus)

The architecture directly demonstrates theoretical and practical concepts across the CSE3010 curriculum:

- **Module 1 (Image & Video Representation, Geometric Operations)**: Frame acquisition, resolution-adaptive scaling, color conversions, and bounding-box spatial geometry.
- **Module 2 (Perspective Geometry & Projective Transformation)**: 4-point planar homography computation via Direct Linear Transformation (DLT) to rectify foreshortening and map camera perspective to the metric road plane.
- **Module 3 (Object Detection & Feature Representation)**: Deep convolutional object detection using YOLO11n for semantic classification and bounding-box localization.
- **Module 4 (Motion Analysis & Background Subtraction)**: Mixture of Gaussians (MOG2) background subtraction, morphological noise suppression (erosion/dilation), and multi-object centroid tracking across temporal sequences.

---

## 5. System Architecture & Methodology

```
+---------------------------------------------------------------------------------+
|                                 Input Video Stream                              |
|                          (UHD 3840x2160 @ 50.0 FPS)                             |
+----------------------------------------+----------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       VideoProcessor (Orchestrator)                             |
|          - Decodes frames, handles frame pacing, progress logging               |
+----------------------------------------+----------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       VehicleDetector (Dual Backend)                            |
|    Primary: YOLO11n (Semantic COCO classes: car, bus, truck, motorcycle)        |
|    Baseline: OpenCV MOG2 (Gaussian Mixture Background Subtraction)             |
+----------------------------------------+----------------------------------------+
                                         | [BBoxes, Centroids, Classes, Conf]
                                         v
+---------------------------------------------------------------------------------+
|                       MultiObjectTracker (MOT Engine)                           |
|       - Dual-metric cost: Centroid Distance <= max_distance AND IoU >= threshold|
|       - Class consistency preservation and occlusion memory (max_missed)        |
+----------------------------------------+----------------------------------------+
                                         | [Active Persistent Tracks]
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
+---------------------------------------+ +---------------------------------------+
|         HomographyTransformer         | |            SpeedEstimator             |
| - 4-Point Planar Projective Transform | | - Temporal sliding window (N=5)       |
| - Maps pixels -> Metric Ground Plane  | | - Computes displacement velocity d/dt |
| - H computed via cv2.getPerspective-  | | - Real-world speed: km/h = 3.6 * m/s  |
|   Transform                           | | - Outlier spike suppression filter    |
+---------------------------------------+ +-------------------+-------------------+
                    |                                         |
                    +--------------------+--------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       TrafficAnalyzer (Analytics Engine)                        |
|       - Tracks unique vehicle volume, average & peak velocity                   |
|       - Computes class breakdown and congestion index (Light/Moderate/Heavy)   |
+----------------------------------------+----------------------------------------+
                                         |
        +--------------------------------+--------------------------------+
        |                                |                                |
        v                                v                                v
+----------------------+ +-------------------------------+ +----------------------+
| annotated_video.mp4  | |       vehicle_data.csv        | | traffic_summary.json |
| - Scaled HUD Overlay | | - Frame-by-frame kinematics   | | - Consolidated       |
| - Vehicle ID & Box   | | - Columns: frame, id, class,  | |   traffic statistics |
| - Trajectory History | |   confidence, x, y, w, h,     | | - Speed metrics and  |
| - Speed Overlay      | |   speed, unit, speed_kmh      | |   congestion index   |
+----------------------+ +-------------------------------+ +----------------------+
```

### 5.1 Planar Homography Mathematical Formulation
A planar homography $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ is an invertible projective mapping between corresponding points on the camera image plane $\mathbf{x} = [x, y, 1]^T$ and points on the physical road plane $\mathbf{X} = [X, Y, 1]^T$:

$$\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Given four coplanar ground points $\mathbf{x}_i \leftrightarrow \mathbf{X}_i$ ($i = 1, 2, 3, 4$) where no three points are collinear, $\mathbf{H}$ is solved uniquely up to a scale factor using Direct Linear Transformation (DLT). In our system, real-world metric dimensions of a 4-lane urban street section ($14.0\text{ m}$ width $\times$ $40.0\text{ m}$ longitudinal span) are used to calibrate the road coordinates.

### 5.2 Discrete Velocity Estimation
Velocity is estimated using a discrete temporal smoothing window over past centroid positions:
$$d = \sqrt{(X_t - X_{t - \Delta t})^2 + (Y_t - Y_{t - \Delta t})^2}$$
$$\Delta t = \frac{\Delta \text{frames}}{\text{FPS}}$$
$$v_{\text{m/s}} = \frac{d}{\Delta t}, \quad v_{\text{km/h}} = 3.6 \times v_{\text{m/s}}$$

A dynamic filtering threshold ($\text{speed} \le 38.0\text{ m/s} \approx 136.8\text{ km/h}$) is applied to reject perspective projection singularities that occur near the vanishing horizon.

---

## 6. Experimental Setup & Dataset

- **Surveillance Stream**: `data/18437773-uhd_3840_2160_50fps.mp4`
- **Location & Camera**: Stationary street surveillance overlooking multi-lane traffic (North Michigan Ave, Chicago).
- **Resolution**: Ultra-High-Definition ($3840 \times 2160$ pixels).
- **Frame Rate**: $50.00\text{ FPS}$.
- **Total Duration / Frames**: $44.16\text{ seconds}$ ($2,208\text{ frames}$).
- **Evaluation Batch**: $300\text{ frames}$ ($6.0\text{ seconds}$ at $50\text{ FPS}$) for baseline evaluation and calibration benchmarking.

---

## 7. Empirical Results & Evaluation

### 7.1 Quantitative Benchmark Summary

| Metric | Calibrated Physical Mode | Uncalibrated Baseline Mode |
|---|---:|---:|
| **Evaluated Frames** | `300` | `300` |
| **Total Vehicle Detections** | `2,285` | `2,285` |
| **Unique Tracked Vehicles** | `29` | `29` |
| **Average Active Vehicles / Frame** | `7.62` | `7.62` |
| **Congestion Level** | **Moderate** | **Moderate** |
| **Car Detections** | `1,915` ($83.8\%$) | `1,915` ($83.8\%$) |
| **Bus Detections** | `351` ($15.4\%$) | `351` ($15.4\%$) |
| **Truck Detections** | `19` ($0.8\%$) | `19` ($0.8\%$) |
| **Average Velocity** | **$9.08\text{ m/s}$ ($32.70\text{ km/h}$)** | `$301.92\text{ px/s}$` |
| **Measurement Unit** | $\text{m/s}$ and $\text{km/h}$ | $\text{pixels/s}$ |
| **Calibration Status** | **Active (`config/calibration.json`)** | Disabled |

### 7.2 Semantic (YOLO) vs. Motion-Based (MOG2) Comparison

| Feature | YOLO11n Semantic Detector | MOG2 Background Subtractor |
|---|---|---|
| **Class Categorization** | Multi-class (`car`, `bus`, `truck`, `motorcycle`) | Binary foreground motion masks |
| **Stationary Vehicle Handling** | Detects stopped vehicles at traffic lights | Fades into background when stationary |
| **Illumination / Shadow Sensitivity** | Highly robust to ambient shadows | Shadows require morphological suppression |
| **Computational Footprint** | Moderate (optimized PyTorch CPU / CUDA) | Very low (classical pixel-level arithmetic) |
| **CSE3010 Alignment** | Module 3: Deep Object Localization | Module 4: Gaussian Mixture Motion Modeling |

---

## 8. Verification & Test Suite

The project includes an automated unit and integration test suite (`tests/`) executable via:
```bash
python -m pytest
```

### Test Case Summary:
1. `test_iou`: Verifies exact $1.0$ match for identical bounding boxes.
2. `test_iou_cases`: Validates boundary handling for non-overlapping and partially overlapping bounding boxes.
3. `test_track_persistence`: Confirms tracker maintains consistent `track_id` across consecutive frames.
4. `test_detection_dataclass`: Asserts data integrity of `Detection` dataclass attributes.
5. `test_identity_like_mapping`: Confirms homography transformation produces identity mapping for identical coordinate sets.
6. `test_pixel_speed_positive`: Validates uncalibrated displacement velocity calculation.
7. `test_calibrated_speed_estimator`: Validates physical metric speed conversion from known geometric landmark coordinates.
8. `test_summary`: Validates statistical dictionary generation in `TrafficAnalyzer`.
9. `test_traffic_analyzer_congestion_and_calibrated`: Validates multi-vehicle aggregation, congestion indexing, and $\text{km/h}$ metric conversion.
10. `test_mog2_detector_initialization`: Validates initialization and kernel configuration of the classical MOG2 baseline.

**Result**: `10 passed in 0.18s` (100% test pass rate).

---

## 9. Limitations & Edge Cases

1. **Occlusion & Depth Disparity**: Distant vehicles or small vehicles heavily occluded by double-decker buses can experience temporary detection dropouts, which our `max_missed=12` buffer mitigates.
2. **Horizon Projection Singularity**: In perspective projection, image coordinates near the horizon vanish to infinity; an upper-bound velocity cap is enforced to eliminate mathematical singularities.
3. **Planar Assumption**: Homography assumes vehicles travel on a strictly planar road surface; road slopes or flyover ramps require piece-wise planar calibration.

---

## 10. Conclusion

The developed **Traffic Vision Analytics** project delivers an automated, syllabus-aligned Computer Vision system capable of robust multi-vehicle detection, persistent tracking, planar homography perspective rectification, and quantitative speed and density reporting. By supporting both calibrated physical speeds ($\text{km/h}$) and uncalibrated pixel kinematics, the system provides practical engineering utility alongside rigorous academic demonstration for CSE3010 Computer Vision.

---

## 11. References

1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). Springer.
2. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
3. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR.
4. Ultralytics. (2024). *YOLO11: State-of-the-Art Real-Time Object Detection and Tracking*. https://docs.ultralytics.com
5. Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.
