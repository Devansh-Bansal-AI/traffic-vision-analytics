from src.detector import Detection
from src.tracker import MultiObjectTracker, iou


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_track_persistence():
    tr = MultiObjectTracker(max_distance=50)
    a = Detection((10, 10, 20, 20), (20, 20), 400, "car", 0.9)
    b = Detection((15, 12, 20, 20), (25, 22), 400, "car", 0.9)
    first = tr.update([a])
    second = tr.update([b])
    assert first[0].track_id == second[0].track_id
