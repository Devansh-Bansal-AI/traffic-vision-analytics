"""
Headless Video Processor Pipeline Orchestrator.
Processes traffic video streams in strict CLI mode:
  - Decodes frames without GUI display server dependencies
  - Runs detection, multi-object tracking, and kinematic estimation
  - Streams annotated video to MP4 container and observations to CSV/JSON
"""

from collections import Counter
import csv
import json
import os
from pathlib import Path
from typing import Dict, Optional

# Enforce strict offscreen/headless execution for OpenCV/Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
from .visualizer import draw_hud, draw_tracks


class VideoProcessor:
    """
    Coordinates batch frame execution across detector, tracker, speed estimator,
    and statistical analyzer subsystems in a headless environment.
    """

    def __init__(self, detector, tracker, estimator, analyzer):
        self.detector = detector
        self.tracker = tracker
        self.estimator = estimator
        self.analyzer = analyzer

    def run(
        self,
        input_path: str,
        video_output: str,
        csv_output: str,
        json_output: str,
        max_frames: Optional[int] = None,
        frame_skip: int = 0,
    ) -> Dict:
        """
        Executes the video processing pipeline in strict headless CLI mode.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video input: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ensure target directories exist
        Path(video_output).parent.mkdir(parents=True, exist_ok=True)
        Path(csv_output).parent.mkdir(parents=True, exist_ok=True)
        Path(json_output).parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            video_output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps / (frame_skip + 1),
            (width, height),
        )

        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "frame", "vehicle_id", "class", "confidence", "x", "y", "width", "height",
                "speed", "speed_unit", "speed_kmh"
            ])

            frame_no = 0
            processed = 0

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    # Frame decimation for performance acceleration if requested
                    if frame_skip and frame_no % (frame_skip + 1) != 0:
                        frame_no += 1
                        continue

                    # 1. Semantic / Motion Detection
                    detections = self.detector.detect(frame)

                    # 2. Multi-Object Tracking Association
                    tracks = self.tracker.update(detections)

                    # 3. Kinematic Displacement & Speed Estimation
                    speeds = {}
                    for t in tracks:
                        if t.missed == 0:
                            speeds[t.track_id] = self.estimator.update(t.track_id, t.centroid)

                    # 4. Statistical Aggregation
                    self.analyzer.update(tracks, speeds)
                    active_tracks = [t for t in tracks if t.missed == 0]
                    counts = Counter(t.class_name for t in active_tracks)

                    # 5. Graphical Annotation Rendering
                    calibrated = self.estimator.transformer is not None
                    annotated = draw_tracks(frame, tracks, speeds, calibrated)
                    annotated = draw_hud(
                        annotated, frame_no, len(active_tracks), counts, calibrated
                    )
                    writer.write(annotated)

                    # 6. Tabular Observation Persistence
                    unit = "m/s" if calibrated else "pixels/s"
                    for t in active_tracks:
                        x, y, w, h = t.bbox
                        speed_val = speeds.get(t.track_id, 0.0)
                        speed_kmh_str = f"{speed_val * 3.6:.2f}" if calibrated else "N/A"
                        csv_writer.writerow([
                            frame_no, t.track_id, t.class_name, f"{t.confidence:.4f}",
                            t.centroid[0], t.centroid[1], w, h,
                            f"{speed_val:.4f}", unit, speed_kmh_str
                        ])

                    processed += 1
                    frame_no += 1

                    # Live headless progress feedback to stdout
                    if processed % 50 == 0 or (max_frames is not None and processed >= max_frames):
                        target_str = f"/{max_frames}" if max_frames else f"/{total_frames}"
                        print(f"  [CLI Engine] Processed {processed}{target_str} frames | Active vehicles: {len(active_tracks)}")

                    if max_frames is not None and processed >= max_frames:
                        break
            finally:
                cap.release()
                writer.release()

        # Compile and export comprehensive JSON analytics summary
        summary = self.analyzer.summary(calibrated=self.estimator.transformer is not None)
        summary["input_resolution"] = f"{width}x{height}"
        summary["video_fps"] = round(fps, 3)
        summary["output_video"] = str(Path(video_output).resolve())
        summary["output_csv"] = str(Path(csv_output).resolve())

        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
