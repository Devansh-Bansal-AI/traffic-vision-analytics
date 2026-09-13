# Project Statement

## Title
**Intelligent Traffic Surveillance & Vehicle Analytics using Computer Vision**

## Course
**CSE3010 - Computer Vision**

---

## 1. Problem Statement
Manual traffic monitoring and conventional radar speed traps cannot scale to continuous, multi-lane urban surveillance. They suffer from high labor overhead, susceptibility to human error, blind spots, and an absence of standardized, machine-readable kinematic records. 

This project addresses these challenges by developing an automated, reproducible, headless command-line computer vision system capable of real-time multi-vehicle detection, trajectory association, road-plane projective rectification, physical velocity estimation, and structured traffic density reporting from stationary surveillance camera footage.

---

## 2. Scope of the Project
- **Target Video Feeds**: Stationary, fixed-view urban traffic surveillance cameras (supporting standard 720p/1080p up to 4K UHD @ 50 FPS).
- **Vehicle Classification Scope**: Detects and discriminates primary road vehicle classes (`car`, `bus`, `truck`, `motorcycle`).
- **Kinematic Estimation Scope**: Supports both:
  1. *Calibrated Mode*: Employs 4-point planar homography ($\mathbf{x}' \sim \mathbf{H}\mathbf{x}$) mapping 2D image coordinates to physical ground-plane coordinates to derive real velocities in $\text{km/h}$ and $\text{m/s}$.
  2. *Uncalibrated Mode*: Computes displacement rates in $\text{pixels/s}$ when field calibration is unavailable, explicitly tagging outputs to avoid false interpretations.
- **Evaluation Environment Scope**: Optimized for headless command-line execution without GUI dependencies (no display servers required), enabling seamless execution in automated testing sandboxes, CI/CD pipelines, and grading servers.
- **Exclusions**: Mobile/drone dashcam tracking (dynamic ego-motion compensation is beyond the scope of a planar homography stationary model).

---

## 3. Target Users
1. **Municipal Traffic Authorities & City Planners**: For macroscopic congestion analysis, Highway Capacity Manual (HCM) Level of Service (LOS) indexing, and lane capacity planning.
2. **Traffic Law Enforcement Agencies**: For automated velocity compliance screening and peak-hour violation monitoring.
3. **Intelligent Transportation System (ITS) Engineers**: For integrating edge computer vision streams into adaptive signal control networks.
4. **Academic Evaluators & Researchers**: For reproducible benchmarking of deep learning detectors (YOLO11n) against classical background subtraction baselines (OpenCV MOG2).

---

## 4. High-Level Features
- **Dual-Engine Detection Subsystem**: Ultralytics YOLO11n for deep semantic localization with an OpenCV MOG2 background subtraction baseline (aligned with CSE3010 Module 4).
- **Multi-Object Tracking (MOT) Engine**: Class-aware centroid Euclidean distance gating combined with bounding-box Intersection-over-Union (IoU) and occlusion buffering (`max_missed`).
- **Planar Perspective Homography**: Rectifies foreshortening distortions and maps road pixels to real-world ground distances.
- **Physical Kinematics & Speed Outlier Rejection**: Sliding temporal window velocity calculation ($v = \Delta d / \Delta t$) with horizon singularity suppression.
- **Transportation Engineering Analytics**: Automated calculation of HCM Level of Service (LOS A–F), 85th-percentile speed ($V_{85}$), traffic density (veh/lane-km), hourly flow rate ($Q$), and fleet modal split.
- **Automated Multi-Artifact Generation**: Produces full-resolution annotated video (`annotated_video.mp4`), tabular per-frame observation logs (`vehicle_data.csv`), and consolidated JSON summaries (`traffic_summary.json`).
- **Fault-Tolerant Headless Operation**: Dynamic calibration synthesis, missing file fallbacks, and zero GUI window dependencies.
