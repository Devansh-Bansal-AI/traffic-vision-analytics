import argparse
import json
from pathlib import Path

from src.detector import VehicleDetector
from src.tracker import MultiObjectTracker
from src.homography import HomographyTransformer
from src.speed_estimator import SpeedEstimator
from src.traffic_analyzer import TrafficAnalyzer
from src.video_processor import VideoProcessor


def parse_args():
    p = argparse.ArgumentParser(
        description="Intelligent Traffic Surveillance & Vehicle Analytics"
    )
    p.add_argument("--input", required=True, help="Input traffic video")
    p.add_argument("--output-dir", default="outputs", help="Output directory")
    p.add_argument("--detector", choices=["yolo", "mog2"], default="yolo")
    p.add_argument("--model", default="yolo11n.pt", help="Ultralytics YOLO model")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu", help="cpu, 0, or another Ultralytics device")
    p.add_argument("--max-distance", type=float, default=90.0)
    p.add_argument("--max-missed", type=int, default=12)
    p.add_argument("--iou-threshold", type=float, default=0.05)
    p.add_argument("--min-area", type=int, default=1800)
    p.add_argument("--calibration", help="JSON road calibration file")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--skip", type=int, default=0, help="Process every N+1th frame")
    p.add_argument("--show", action="store_true", help="Show preview window (optional)")
    return p.parse_args()


import sys


def load_transformer(path):
    if not path:
        return None
    cal_path = Path(path)
    if not cal_path.is_file():
        print(f"Error: Calibration file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(cal_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "image_points" not in cfg or "world_points" not in cfg:
            raise ValueError("Calibration file must contain 'image_points' and 'world_points'")
        return HomographyTransformer(cfg["image_points"], cfg["world_points"])
    except Exception as exc:
        print(f"Error reading calibration file: {exc}", file=sys.stderr)
        sys.exit(1)


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: Input video file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Error creating output directory {args.output_dir}: {exc}", file=sys.stderr)
        sys.exit(1)

    import cv2
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: OpenCV could not open video: {args.input}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    transformer = load_transformer(args.calibration)
    try:
        detector = VehicleDetector(
            backend=args.detector,
            model_path=args.model,
            confidence=args.conf,
            image_size=args.imgsz,
            device=args.device,
            min_area=args.min_area,
        )
    except Exception as exc:
        print(f"Error initializing detector: {exc}", file=sys.stderr)
        sys.exit(1)

    tracker = MultiObjectTracker(
        max_distance=args.max_distance,
        max_missed=args.max_missed,
        iou_threshold=args.iou_threshold,
    )
    max_physical_speed = 38.0 if transformer else None
    estimator = SpeedEstimator(fps=fps, transformer=transformer, max_speed=max_physical_speed)
    analyzer = TrafficAnalyzer(fps=fps)
    processor = VideoProcessor(detector, tracker, estimator, analyzer)

    calib_status = (
        f"ENABLED via {args.calibration} (Real-world physical speed in m/s & km/h)"
        if transformer
        else "NOT ENABLED (Displacement reported in pixels/s; use --calibration for km/h)"
    )

    print("=" * 68)
    print("  INTELLIGENT TRAFFIC SURVEILLANCE & VEHICLE ANALYTICS (CSE3010)")
    print("=" * 68)
    print(f" Input Video : {args.input}")
    print(f" Resolution  : {width}x{height}")
    print(f" Native FPS  : {fps:.2f}")
    print(f" Frames      : {frame_count}")
    print(f" Mode        : Headless Command-Line Execution")
    print(f" Detector    : {args.detector.upper()}")
    if args.detector == "yolo":
        print(f" Model       : {args.model}")
        print(f" Confidence  : {args.conf:.2f}")
        print(f" Inference   : {args.imgsz}px on {args.device}")
    print(f" Calibration : {calib_status}")
    print(f" Frame Skip  : {args.skip}")
    print("-" * 68)
    print("Processing pipeline started...\n")

    summary = processor.run(
        input_path=args.input,
        video_output=str(out / "annotated_video.mp4"),
        csv_output=str(out / "vehicle_data.csv"),
        json_output=str(out / "traffic_summary.json"),
        max_frames=args.max_frames,
        frame_skip=args.skip,
        show=args.show,
    )

    print("\n" + "=" * 68)
    print("  PIPELINE EXECUTION SUMMARY")
    print("=" * 68)
    for k, v in summary.items():
        print(f"  {k:34s}: {v}")
    print("-" * 68)
    print(f"  All outputs generated successfully in: {out.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()
