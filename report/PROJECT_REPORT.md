# Project Report
## Intelligent Traffic Surveillance & Vehicle Analytics using Computer Vision

> Replace bracketed fields with your actual details and measured experimental results before submission.

### 1. Cover Page
- Student name: [Name]
- Registration number: [Reg. No.]
- Course: CSE3010 Computer Vision
- Project title: Intelligent Traffic Surveillance & Vehicle Analytics using Computer Vision
- Faculty: [Faculty]
- Date: [Date]

### 2. Abstract
This project presents a command-line computer vision system for analysing fixed-camera traffic video. It detects road vehicles, assigns persistent identities across frames, records trajectories and motion, and produces annotated video and structured analytics. The system also supports planar homography so that image coordinates can be mapped to measured road coordinates when a valid calibration is supplied. A MOG2 background-subtraction baseline is included to compare motion-based detection with semantic object detection.

### 3. Problem Statement
Manual traffic monitoring is labour-intensive and difficult to scale. The proposed system automates vehicle-level observation from video while keeping the processing reproducible from a terminal.

### 4. Objectives
1. Detect common road-vehicle classes.
2. Track vehicles over time.
3. Record trajectories and motion measurements.
4. Demonstrate perspective/homography-based road mapping.
5. Generate machine-readable results.
6. Evaluate reliability and limitations.

### 5. Functional Requirements
- FR1: Accept a local video path from the command line.
- FR2: Detect road vehicles.
- FR3: Track detected vehicles and assign IDs.
- FR4: Calculate image-plane or calibrated motion.
- FR5: Generate annotated video.
- FR6: Generate CSV and JSON reports.

### 6. Non-Functional Requirements
- Reproducibility
- Maintainability
- Error handling
- CPU-compatible execution
- Clear CLI operation
- Reasonable processing performance

### 7. System Architecture
```text
Video → Detector → Tracker → Motion/Calibration → Analytics → Outputs
          │            │             │                │
        YOLO/MOG2   Track IDs    Homography       CSV/JSON
                                      │
                                Annotated Video
```

### 8. Algorithms
#### Vehicle detection
YOLO is used for semantic detection of car, motorcycle, bus and truck. MOG2 provides a motion-based baseline.

#### Tracking
The tracker combines predicted centroid distance and bounding-box IoU with class consistency.

#### Homography
For four corresponding road-plane points:

\[
\mathbf{x}' \sim H\mathbf{x}
\]

The transformation is valid for points lying on the calibrated plane.

#### Motion
For displacement \(d\) over time \(t\):

\[
v=\frac{d}{t}
\]

If calibrated world coordinates are in metres, \(v\) is in m/s and:

\[
v_{km/h}=3.6v_{m/s}
\]

### 9. Implementation
Describe each source file and include screenshots of the terminal and annotated output from your own run.

### 10. Dataset
- Video source: [source URL/name]
- License: [license]
- Resolution: [resolution]
- FPS: [FPS]
- Duration: [duration]
- Number of evaluated frames: [N]

### 11. Experimental Results
Fill with your actual measurements:

| Metric | Result |
|---|---:|
| Frames processed | [ ] |
| Processing FPS | [ ] |
| Unique tracks | [ ] |
| Average active vehicles/frame | [ ] |
| Detection precision | [ ] |
| Detection recall | [ ] |
| ID switches | [ ] |
| Speed MAE | [ ] |

### 12. Testing
Run `pytest -q` and paste the actual output. Explain each test briefly.

### 13. Limitations
Discuss occlusion, camera angle, detection errors, tracking failures and calibration sensitivity.

### 14. Future Enhancements
- stronger multi-object tracking,
- automatic camera calibration,
- lane-level analytics,
- congestion classification,
- GPU acceleration,
- larger benchmark evaluation.

### 15. Conclusion
Summarize what was implemented and what the measured results demonstrate.

### 16. References
Use the actual versions and sources used in your implementation. Include the CSE3010 prescribed texts and the documentation for any external software/model used.
