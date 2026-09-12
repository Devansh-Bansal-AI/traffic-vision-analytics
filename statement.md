# Project Statement

## Title
**Intelligent Traffic Surveillance & Vehicle Analytics using Computer Vision**

## Course
**CSE3010 - Computer Vision**

## Problem Statement
Manual traffic observation and conventional speed traps cannot scale to continuous, multi-lane urban surveillance. They suffer from high labor overhead, susceptibility to human error, and a lack of standardized, machine-readable kinematic records. This project addresses the challenge by developing an automated, reproducible, headless command-line computer vision system capable of real-time multi-vehicle detection, trajectory association, road-plane projective rectification, physical velocity estimation, and structured traffic density reporting from stationary surveillance camera footage.

## Objectives
1. **Semantic Vehicle Localization**: Detect multiple vehicle categories (`car`, `bus`, `truck`, `motorcycle`) in high-resolution video streams using Ultralytics YOLO11n.
2. **Motion-Based Baseline**: Maintain an OpenCV MOG2 background subtraction baseline with morphological filtering to directly evaluate semantic vs. motion-based paradigms (aligned with CSE3010 Module 4).
3. **Multi-Object Tracking (MOT)**: Maintain persistent vehicle identities, active/missed states, and continuous trajectory histories across successive frames using class-constrained Euclidean centroid proximity and Intersection-over-Union (IoU) association.
4. **Planar Perspective Homography**: Model the projective geometry between 2D image coordinates and the 3D ground plane via a 4-point homography transformation ($\mathbf{x}' \sim \mathbf{H}\mathbf{x}$), converting pixel displacement into real-world physical velocities ($\text{m/s}$ and $\text{km/h}$).
5. **Headless Execution & Artifact Generation**: Execute entirely via command line without GUI dependencies, generating full annotated 4K/1080p MP4 videos, granular frame-by-frame observation logs (`vehicle_data.csv`), and consolidated traffic summaries (`traffic_summary.json`).
6. **Robustness & Error Resilience**: Provide structured error handling for absent media, unreadable codecs, and invalid calibration definitions, backed by an automated unit test suite.

## Scope & Operating Modes
The pipeline targets stationary traffic surveillance camera views and operates in two well-defined modes:
- **Calibrated Physical Mode (Default Recommended)**: Utilizes a 4-point road calibration configuration (`config/calibration.json`) to compute actual ground-plane vehicle speeds in $\text{km/h}$ and $\text{m/s}$.
- **Uncalibrated Baseline Mode**: Evaluates motion as raw pixel displacement ($\text{pixels/s}$) when field measurements are unavailable, explicitly tagging the absence of calibration on the HUD, CSV logs, and JSON summaries.
