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


def load_transformer(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return HomographyTransformer(cfg["image_points"], cfg["world_points"])


def main():
    args = parse_args()
    if not Path(args.input).is_file():
        raise FileNotFoundError(f"Input video not found: {args.input}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    import cv2
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    transformer = load_transformer(args.calibration)
    detector = VehicleDetector(
        backend=args.detector,
        model_path=args.model,
        confidence=args.conf,
        image_size=args.imgsz,
        device=args.device,
        min_area=args.min_area,
    )
    tracker = MultiObjectTracker(
        max_distance=args.max_distance,
        max_missed=args.max_missed,
        iou_threshold=args.iou_threshold,
    )
    estimator = SpeedEstimator(fps=fps, transformer=transformer)
    analyzer = TrafficAnalyzer(fps=fps)
    processor = VideoProcessor(detector, tracker, estimator, analyzer)

    print("=" * 64)
    print(" Intelligent Traffic Surveillance & Vehicle Analytics")
    print("=" * 64)
    print(f"Input       : {args.input}")
    print(f"Resolution  : {width}x{height}")
    print(f"FPS         : {fps:.2f}")
    print(f"Frames      : {frame_count}")
    print(f"Detector    : {args.detector}")
    if args.detector == "yolo":
        print(f"Model       : {args.model}")
        print(f"Confidence  : {args.conf:.2f}")
        print(f"Inference   : {args.imgsz}px on {args.device}")
    print(f"Calibration : {'enabled' if transformer else 'not enabled (speed reported in pixels/s)'}")
    print(f"Frame skip  : {args.skip}")
    print("-" * 64)
    print("Processing...\n")

    summary = processor.run(
        input_path=args.input,
        video_output=str(out / "annotated_video.mp4"),
        csv_output=str(out / "vehicle_data.csv"),
        json_output=str(out / "traffic_summary.json"),
        max_frames=args.max_frames,
        frame_skip=args.skip,
        show=args.show,
    )

    print("\nProcessing complete.")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nResults saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
