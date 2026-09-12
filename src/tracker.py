from dataclasses import dataclass, field
from typing import Dict, List
import math
from .detector import Detection


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


@dataclass
class Track:
    track_id: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float
    age: int = 1
    missed: int = 0
    history: List[tuple[int, int]] = field(default_factory=list)
    velocity: tuple[float, float] = (0.0, 0.0)


class MultiObjectTracker:
    """Lightweight class-aware tracker using predicted centroid + IoU matching."""

    def __init__(self, max_distance=90.0, max_missed=12, iou_threshold=0.05):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _register(self, d):
        t = Track(self.next_id, d.centroid, d.bbox, d.class_name, d.confidence,
                  history=[d.centroid])
        self.tracks[self.next_id] = t
        self.next_id += 1

    def update(self, detections: List[Detection]) -> List[Track]:
        if not self.tracks:
            for d in detections:
                self._register(d)
            return list(self.tracks.values())

        candidates = []
        for tid, t in self.tracks.items():
            predicted = (t.centroid[0] + t.velocity[0], t.centroid[1] + t.velocity[1])
            for idx, d in enumerate(detections):
                if t.class_name != d.class_name and t.class_name != "moving_object":
                    continue
                dist = self._dist(predicted, d.centroid)
                overlap = iou(t.bbox, d.bbox)
                # Distance gate grows slightly for older tracks.
                gate = self.max_distance * (1.0 + min(t.missed, 5) * 0.15)
                if dist <= gate or overlap >= self.iou_threshold:
                    cost = dist - 50.0 * overlap
                    candidates.append((cost, tid, idx, dist, overlap))

        used_t, used_d = set(), set()
        for _, tid, idx, _, _ in sorted(candidates):
            if tid in used_t or idx in used_d:
                continue
            t = self.tracks[tid]
            d = detections[idx]
            old = t.centroid
            t.velocity = (d.centroid[0] - old[0], d.centroid[1] - old[1])
            t.centroid = d.centroid
            t.bbox = d.bbox
            t.class_name = d.class_name
            t.confidence = d.confidence
            t.age += 1
            t.missed = 0
            t.history.append(d.centroid)
            t.history = t.history[-60:]
            used_t.add(tid)
            used_d.add(idx)

        for tid, t in list(self.tracks.items()):
            if tid not in used_t:
                t.missed += 1
                t.age += 1
                t.centroid = (
                    int(t.centroid[0] + t.velocity[0]),
                    int(t.centroid[1] + t.velocity[1]),
                )
                if t.missed > self.max_missed:
                    del self.tracks[tid]

        for idx, d in enumerate(detections):
            if idx not in used_d:
                self._register(d)
        return list(self.tracks.values())
