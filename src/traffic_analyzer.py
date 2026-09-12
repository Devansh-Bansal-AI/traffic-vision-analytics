from collections import Counter, defaultdict


class TrafficAnalyzer:
    def __init__(self, fps=30.0):
        self.fps = fps
        self.frames_processed = 0
        self.active_counts = []
        self.class_counts = Counter()
        self.track_ids = set()
        self.speed_values = []
        self.max_speed = 0.0
        self.total_detections = 0

    def update(self, tracks, speeds):
        self.frames_processed += 1
        active = 0
        for t in tracks:
            if t.missed == 0:
                active += 1
                self.track_ids.add(t.track_id)
                self.class_counts[t.class_name] += 1
                self.total_detections += 1
                s = speeds.get(t.track_id, 0.0)
                if s >= 0:
                    self.speed_values.append(s)
                    self.max_speed = max(self.max_speed, s)
        self.active_counts.append(active)

    def summary(self, calibrated=False):
        avg_speed = sum(self.speed_values) / len(self.speed_values) if self.speed_values else 0.0
        avg_active = (
            round(sum(self.active_counts) / len(self.active_counts), 2)
            if self.active_counts
            else 0.0
        )
        congestion = (
            "Light" if avg_active < 4.0 else "Moderate" if avg_active < 10.0 else "Heavy"
        )
        result = {
            "frames_processed": self.frames_processed,
            "unique_vehicle_tracks": len(self.track_ids),
            "average_active_vehicles_per_frame": avg_active,
            "congestion_level": congestion,
            "total_vehicle_detections": self.total_detections,
            "vehicle_detections_by_class": dict(self.class_counts),
            "average_speed": round(avg_speed, 3),
            "maximum_speed": round(self.max_speed, 3),
            "speed_unit": "m/s" if calibrated else "pixels/s",
            "speed_calibrated": calibrated,
        }
        if calibrated:
            result["average_speed_kmh"] = round(avg_speed * 3.6, 2)
            result["maximum_speed_kmh"] = round(self.max_speed * 3.6, 2)
        return result
