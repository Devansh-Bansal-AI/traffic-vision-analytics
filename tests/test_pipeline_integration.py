import pytest
from pathlib import Path
from src.detector import Detection, VehicleDetector
from src.homography import HomographyTransformer
from src.speed_estimator import SpeedEstimator
from src.traffic_analyzer import TrafficAnalyzer
from src.tracker import MultiObjectTracker, Track, iou


def test_iou_cases():
    # Identical boxes
    assert iou((10, 10, 20, 20), (10, 10, 20, 20)) == 1.0
    # Non-overlapping boxes
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    # Partial overlap (half width overlap)
    val = iou((0, 0, 20, 10), (10, 0, 20, 10))
    assert 0.0 < val < 1.0


def test_detection_dataclass():
    det = Detection(
        bbox=(100, 200, 50, 60),
        centroid=(125, 230),
        area=3000.0,
        class_name="car",
        confidence=0.88,
    )
    assert det.class_name == "car"
    assert det.confidence == 0.88
    assert det.centroid == (125, 230)
    assert det.area == 3000.0


def test_calibrated_speed_estimator():
    # Map a 100x100 pixel square to a 10x10 metre ground plane
    transformer = HomographyTransformer(
        [[0, 0], [100, 0], [100, 100], [0, 100]],
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
    )
    estimator = SpeedEstimator(fps=10.0, transformer=transformer, smoothing=2)
    # First point
    assert estimator.update(1, (0, 0)) == 0.0
    # Second point 100 pixels in x = 10 metres in world. Elapsed dt = 1/10 = 0.1s -> speed = 100 m/s
    speed_mps = estimator.update(1, (100, 0))
    assert abs(speed_mps - 100.0) < 1e-2


def test_traffic_analyzer_congestion_and_calibrated():
    analyzer = TrafficAnalyzer(fps=30.0)
    # Add tracks for multiple frames to test active counts and congestion
    tracks = [
        Track(1, (100, 100), (90, 90, 20, 20), "car", 0.9),
        Track(2, (200, 200), (190, 190, 20, 20), "bus", 0.85),
    ]
    analyzer.update(tracks, {1: 10.0, 2: 8.0})

    # Calibrated summary
    summary = analyzer.summary(calibrated=True)
    assert summary["frames_processed"] == 1
    assert summary["unique_vehicle_tracks"] == 2
    assert summary["average_active_vehicles_per_frame"] == 2.0
    assert summary["congestion_level"] == "Light"
    assert summary["speed_calibrated"] is True
    assert summary["speed_unit"] == "m/s"
    assert "average_speed_kmh" in summary
    assert "maximum_speed_kmh" in summary
    assert summary["average_speed_kmh"] == round(9.0 * 3.6, 2)


def test_mog2_detector_initialization():
    detector = VehicleDetector(backend="mog2", min_area=500)
    assert detector.backend == "mog2"
    assert detector.bg is not None
