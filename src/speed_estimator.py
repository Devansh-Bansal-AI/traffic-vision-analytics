from collections import defaultdict, deque
import math


class SpeedEstimator:
    """Computes displacement per second. With homography, displacement is metres/s."""

    def __init__(self, fps, transformer=None, smoothing=5, max_speed=None):
        self.fps = fps
        self.transformer = transformer
        self.points = defaultdict(lambda: deque(maxlen=smoothing))
        self.max_speed = max_speed

    def update(self, track_id, image_point):
        point = self.transformer.transform_point(image_point) if self.transformer else image_point
        history = self.points[track_id]
        history.append(point)
        if len(history) < 2:
            return 0.0
        # Use oldest point in the smoothing window to reduce frame-to-frame jitter.
        a, b = history[0], history[-1]
        dt = (len(history) - 1) / self.fps
        if dt <= 0:
            return 0.0
        distance = math.hypot(b[0] - a[0], b[1] - a[1])
        speed = distance / dt
        if self.max_speed is not None and speed > self.max_speed:
            return 0.0
        return speed
