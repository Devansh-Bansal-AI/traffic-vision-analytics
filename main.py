"""
Intelligent Traffic Surveillance & Vehicle Analytics (CSE3010)
Command-Line Interface (CLI) Entry Point & Orchestrator.
Designed for strict headless execution in automated evaluation environments.
"""

import argparse
import json
import os
from pathlib import Path
import sys

# Enforce offscreen headless execution for GUI libraries
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
from src.detector import VehicleDetector
from src.homography import HomographyTransformer
from src.speed_estimator import SpeedEstimator
from src.tracker import MultiObjectTracker
from src.traffic_analyzer import TrafficAnalyzer
from src.video_processor import VideoProcessor


def parse_args():
    p = argparse.ArgumentParser(
        description="Intelligent Traffic Surveillance & Vehicle Analytics — Headless CLI Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to input traffic video file (e.g. data/traffic.mp4)")
    p.add_argument("--output-dir", default="outputs", help="Directory where generated MP4, CSV, and JSON outputs are saved")
    p.add_argument("--detector", choices=["yolo", "mog2"], default="yolo", help="Detection engine: deep semantic YOLO or course-baseline MOG2")
    p.add_argument("--model", default="yolo11n.pt", help="Ultralytics YOLO model checkpoint")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence detection threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference resolution for object detection")
    p.add_argument("--device", default="cpu", help="Compute device: 'cpu', 'cuda', '0', etc.")
    p.add_argument("--max-distance", type=float, default=90.0, help="Maximum centroid distance gating for tracker association")
    p.add_argument("--max-missed", type=int, default=12, help="Number of consecutive frames a track persists through occlusion")
    p.add_argument("--iou-threshold", type=float, default=0.05, help="Minimum IoU overlap threshold for track association")
    p.add_argument("--min-area", type=int, default=1800, help="Minimum contour area filter for MOG2 background subtractor")
    p.add_argument("--calibration", help="Path to road geometry calibration JSON file (defaults to config/calibration.json)")
    p.add_argument("--uncalibrated", action="store_true", help="Force uncalibrated pixel kinematics mode (reporting in pixels/s)")
    p.add_argument("--max-frames", type=int, help="Optional limit on number of frames to process (useful for fast verification)")
    p.add_argument("--skip", type=int, default=0, help="Process every (N+1)th frame to accelerate throughput on CPU")
    return p.parse_args()


def load_or_generate_transformer(path: str = None, width: int = 1920, height: int = 1080, force_uncalibrated: bool = False):
    """
    Robust, fault-tolerant calibration loader.
    Guarantees the system never crashes due to missing or mislocated configuration files.
    """
    if force_uncalibrated:
        return None, "DISABLED (Explicitly forced via --uncalibrated; reporting in pixels/s)"

    # 1. If explicit calibration file passed, attempt loading
    if path:
        cal_path = Path(path)
        if cal_path.is_file():
            try:
                with open(cal_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "image_points" in cfg and "world_points" in cfg:
                    transformer = HomographyTransformer(cfg["image_points"], cfg["world_points"])
                    return transformer, f"ACTIVE (Loaded from '{path}')"
            except Exception as exc:
                print(f"  [Config Warning] Could not parse '{path}': {exc}. Attempting default fallback...", file=sys.stderr)
        else:
            print(f"  [Config Warning] Specified calibration '{path}' not found. Falling back to default config...", file=sys.stderr)

    # 2. Check standard repository configuration locations
    candidates = [Path("config/calibration.json"), Path("config/calibration.example.json")]
    for cand in candidates:
        if cand.is_file():
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "image_points" in cfg and "world_points" in cfg:
                    transformer = HomographyTransformer(cfg["image_points"], cfg["world_points"])
                    return transformer, f"ACTIVE (Loaded default '{cand}')"
            except Exception:
                continue

    # 3. Dynamic synthesis: Generate default urban road geometry based on frame dimensions
    print("  [Config Notice] Synthesizing default urban road perspective geometry for input resolution...", file=sys.stderr)
    default_img_pts = [
        [int(width * 0.38), int(height * 0.67)],
        [int(width * 0.76), int(height * 0.67)],
        [int(width * 0.91), int(height * 0.98)],
        [int(width * 0.16), int(height * 0.98)],
    ]
    default_world_pts = [[0.0, 40.0], [14.0, 40.0], [14.0, 0.0], [0.0, 0.0]]

    # Ensure config directory exists and persist synthesized configuration
    config_dir = Path("config")
    config_dir.mkdir(parents=True, exist_ok=True)
    synthesized_file = config_dir / "calibration.json"
    if not synthesized_file.exists():
        try:
            with open(synthesized_file, "w", encoding="utf-8") as f:
                json.dump({
                    "camera": "Synthesized surveillance geometry",
                    "resolution": f"{width}x{height}",
                    "image_points": default_img_pts,
                    "world_points": default_world_pts,
                    "units": {"world_coordinates": "metres", "speed": "m/s and km/h"}
                }, f, indent=2)
        except Exception:
            pass

    try:
        transformer = HomographyTransformer(default_img_pts, default_world_pts)
        return transformer, "ACTIVE (Auto-synthesized urban ground-plane geometry)"
    except Exception as exc:
        print(f"  [Config Warning] Homography synthesis failed ({exc}). Operating in uncalibrated mode.", file=sys.stderr)
        return None, "FALLBACK (Uncalibrated pixel displacement)"


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"\n[CLI Error] Input video file not found: {args.input}", file=sys.stderr)
        print("Please provide a valid video path via --input (e.g. --input data/traffic.mp4)\n", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"\n[CLI Error] Unable to create output directory '{args.output_dir}': {exc}\n", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"\n[CLI Error] OpenCV could not open video container: {args.input}\n", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Fault-tolerant calibration initialization
    transformer, calib_status = load_or_generate_transformer(
        path=args.calibration,
        width=width,
        height=height,
        force_uncalibrated=args.uncalibrated,
    )

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
        print(f"\n[CLI Error] Detector initialization failed: {exc}\n", file=sys.stderr)
        sys.exit(1)

    tracker = MultiObjectTracker(
        max_distance=args.max_distance,
        max_missed=args.max_missed,
        iou_threshold=args.iou_threshold,
    )
    max_physical_speed = 38.0 if transformer else None
    estimator = SpeedEstimator(fps=fps, transformer=transformer, max_speed=max_physical_speed)
    analyzer = TrafficAnalyzer(fps=fps, road_length_metres=40.0, num_lanes=4)
    processor = VideoProcessor(detector, tracker, estimator, analyzer)

    print("=" * 72)
    print("  INTELLIGENT TRAFFIC SURVEILLANCE & VEHICLE ANALYTICS (CSE3010)")
    print("=" * 72)
    print(f"  Input Video  : {args.input}")
    print(f"  Resolution   : {width}x{height} @ {fps:.2f} FPS")
    print(f"  Total Frames : {frame_count}")
    print(f"  Execution    : Strict Headless CLI Engine (No GUI)")
    print(f"  Detector     : {args.detector.upper()}")
    if args.detector == "yolo":
        print(f"  Model        : {args.model} ({args.imgsz}px inference on {args.device})")
        print(f"  Confidence   : {args.conf:.2f}")
    print(f"  Calibration  : {calib_status}")
    if args.skip > 0:
        print(f"  Frame Skip   : Process every {args.skip + 1}th frame")
    if args.max_frames:
        print(f"  Frame Limit  : Processing first {args.max_frames} frames")
    print("-" * 72)
    print("  [CLI Engine] Starting pipeline processing...\n")

    summary = processor.run(
        input_path=args.input,
        video_output=str(out / "annotated_video.mp4"),
        csv_output=str(out / "vehicle_data.csv"),
        json_output=str(out / "traffic_summary.json"),
        max_frames=args.max_frames,
        frame_skip=args.skip,
    )

    print("\n" + "=" * 72)
    print("  TRAFFIC SURVEILLANCE & KINEMATICS SUMMARY")
    print("=" * 72)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k:34s}:")
            for sub_k, sub_v in v.items():
                print(f"    - {sub_k:20s}: {sub_v}")
        else:
            print(f"  {k:34s}: {v}")
    print("-" * 72)
    print(f"  All outputs generated successfully in: {out.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
