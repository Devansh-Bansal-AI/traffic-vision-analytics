# Project Statement

## Title
Intelligent Traffic Surveillance & Vehicle Analytics using Computer Vision

## Problem
Manual traffic observation is time-consuming and does not provide consistent vehicle-level measurements. The project develops a command-line computer vision pipeline that detects road vehicles, assigns persistent track IDs, measures image-plane motion, optionally maps road-plane coordinates using homography, and produces machine-readable traffic analytics.

## Objectives
1. Detect cars, motorcycles, buses and trucks in traffic video.
2. Track detected vehicles across frames.
3. Record trajectories and motion measurements.
4. Support planar road calibration using a four-point homography.
5. Generate annotated video, CSV observations and JSON summary statistics.
6. Provide a syllabus-aligned MOG2 background-subtraction baseline for comparison.

## Scope
The system is intended for fixed-camera traffic videos. Real-world speed requires measured road calibration; without calibration the system reports pixel displacement speed and explicitly labels it as such.
