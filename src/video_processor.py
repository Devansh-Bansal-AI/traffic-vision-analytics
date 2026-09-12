import csv
import json
from collections import Counter
from pathlib import Path

import cv2

from .visualizer import draw_hud, draw_tracks


class VideoProcessor:
    def __init__(self, detector, tracker, estimator, analyzer):
        self.detector = detector
        self.tracker = tracker
        self.estimator = estimator
        self.analyzer = analyzer

    def run(self, input_path, video_output, csv_output, json_output,
            max_frames=None, frame_skip=0, show=False):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            video_output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps / (frame_skip + 1),
            (width, height),
        )

        Path(csv_output).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "frame", "vehicle_id", "class", "confidence", "x", "y", "width", "height",
                "speed", "speed_unit"
            ])
            frame_no = 0
            processed = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_skip and frame_no % (frame_skip + 1) != 0:
                        frame_no += 1
                        continue

                    detections = self.detector.detect(frame)
                    tracks = self.tracker.update(detections)
                    speeds = {}
                    for t in tracks:
                        if t.missed == 0:
                            speeds[t.track_id] = self.estimator.update(t.track_id, t.centroid)

                    self.analyzer.update(tracks, speeds)
                    counts = Counter(t.class_name for t in tracks if t.missed == 0)

                    calibrated = self.estimator.transformer is not None
                    annotated = draw_tracks(frame, tracks, speeds, calibrated)
                    annotated = draw_hud(
                        annotated, frame_no,
                        sum(1 for t in tracks if t.missed == 0), counts, calibrated
                    )
                    writer.write(annotated)
                    if show:
                        cv2.imshow("Traffic Vision Analytics", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    unit = "m/s" if calibrated else "pixels/s"
                    for t in tracks:
                        if t.missed == 0:
                            x, y, w, h = t.bbox
                            csv_writer.writerow([
                                frame_no, t.track_id, t.class_name, f"{t.confidence:.4f}",
                                t.centroid[0], t.centroid[1], w, h,
                                f"{speeds.get(t.track_id, 0.0):.4f}", unit
                            ])
                    processed += 1
                    frame_no += 1
                    if max_frames is not None and processed >= max_frames:
                        break
            finally:
                cap.release()
                writer.release()
                if show:
                    cv2.destroyAllWindows()

        summary = self.analyzer.summary(calibrated=self.estimator.transformer is not None)
        summary["input_resolution"] = f"{width}x{height}"
        summary["video_fps"] = round(fps, 3)
        summary["output_video"] = str(Path(video_output).resolve())
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary
