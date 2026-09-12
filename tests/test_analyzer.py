from src.traffic_analyzer import TrafficAnalyzer
from src.tracker import Track


def test_summary():
    a = TrafficAnalyzer()
    t = Track(1, (10, 10), (0, 0, 10, 10), "car", 0.9)
    a.update([t], {1: 5.0})
    s = a.summary(False)
    assert s["unique_vehicle_tracks"] == 1
    assert s["vehicle_detections_by_class"]["car"] == 1
