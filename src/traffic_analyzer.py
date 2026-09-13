"""
Traffic Flow Analyzer & Kinematic Aggregation Subsystem.
Implements empirical transportation engineering metrics aligned with
Highway Capacity Manual (HCM) standards:
  - Level of Service (LOS A-F) based on vehicular density and mean operating speed
  - 85th-Percentile Operating Velocity (V85) and 15th-Percentile Velocity (V15)
  - Speed dispersion (Variance, Standard Deviation, Coefficient of Variation)
  - Equivalent Hourly Flow Rate (vehicles/hour)
  - Fleet composition and Heavy Vehicle Percentage (P_HV)
"""

from collections import Counter
import math
from typing import Dict, List, Optional, Set


class TrafficAnalyzer:
    """
    Aggregates multi-object tracking data over temporal video sequences to derive
    macroscopic and microscopic traffic stream metrics.
    """

    def __init__(self, fps: float = 30.0, road_length_metres: float = 40.0, num_lanes: int = 4):
        self.fps = float(fps)
        self.road_length_km = max(0.01, road_length_metres / 1000.0)
        self.num_lanes = max(1, num_lanes)

        self.frames_processed: int = 0
        self.active_counts: List[int] = []
        self.class_counts: Counter = Counter()
        self.track_ids: Set[int] = set()
        self.speed_samples: List[float] = []
        self.max_speed: float = 0.0
        self.total_detections: int = 0

    def update(self, tracks, speeds: Dict[int, float]) -> None:
        """
        Ingest current frame track associations and instantaneous velocities.
        """
        self.frames_processed += 1
        active_in_frame = 0

        for t in tracks:
            if t.missed == 0:
                active_in_frame += 1
                self.track_ids.add(t.track_id)
                self.class_counts[t.class_name] += 1
                self.total_detections += 1

                inst_speed = speeds.get(t.track_id, 0.0)
                if inst_speed > 0.0:
                    self.speed_samples.append(inst_speed)
                    if inst_speed > self.max_speed:
                        self.max_speed = inst_speed

        self.active_counts.append(active_in_frame)

    def _compute_percentile(self, sorted_data: List[float], percentile: float) -> float:
        """Linear interpolation of empirical sample percentiles."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * (percentile / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return d0 + d1

    def _evaluate_hcm_los(self, density_veh_km_lane: float, mean_speed_kmh: float) -> str:
        """
        Evaluates Level of Service (LOS) using criteria adapted from Highway Capacity
        Manual (HCM) multi-lane arterial standards.
        """
        if density_veh_km_lane <= 7.0 and mean_speed_kmh >= 45.0:
            return "LOS A (Free Flow - Unrestricted maneuverability)"
        elif density_veh_km_lane <= 11.0 and mean_speed_kmh >= 35.0:
            return "LOS B (Reasonably Free Flow - Stable operating speed)"
        elif density_veh_km_lane <= 16.0 and mean_speed_kmh >= 28.0:
            return "LOS C (Stable Flow - Moderate vehicle interactions)"
        elif density_veh_km_lane <= 22.0 and mean_speed_kmh >= 20.0:
            return "LOS D (Approaching Unstable - Noticeable congestion)"
        elif density_veh_km_lane <= 28.0:
            return "LOS E (Unstable Flow - Near capacity boundary)"
        else:
            return "LOS F (Forced/Breakdown Flow - Heavy congestion)"

    def summary(self, calibrated: bool = False) -> Dict:
        """
        Compiles the complete traffic surveillance and kinematic summary ledger.
        """
        sorted_speeds = sorted(self.speed_samples) if self.speed_samples else []
        n_samples = len(sorted_speeds)

        # Kinematic Central Tendencies
        if n_samples > 0:
            mean_speed = sum(sorted_speeds) / n_samples
            variance = sum((x - mean_speed) ** 2 for x in sorted_speeds) / n_samples
            std_dev = math.sqrt(variance)
            coeff_var = (std_dev / mean_speed) if mean_speed > 0 else 0.0
            p15 = self._compute_percentile(sorted_speeds, 15.0)
            p85 = self._compute_percentile(sorted_speeds, 85.0)
        else:
            mean_speed = variance = std_dev = coeff_var = p15 = p85 = 0.0

        # Occupancy & Density
        avg_active = (
            round(sum(self.active_counts) / len(self.active_counts), 2)
            if self.active_counts
            else 0.0
        )
        max_active = max(self.active_counts) if self.active_counts else 0

        # Traffic Density (k = active_vehicles / (length_km * lanes))
        density_lane_km = round(avg_active / (self.road_length_km * self.num_lanes), 2)

        # Elapsed observation time in hours
        elapsed_hours = (self.frames_processed / self.fps) / 3600.0 if self.fps > 0 else 0.0
        hourly_flow_rate = (
            round(len(self.track_ids) / elapsed_hours)
            if elapsed_hours > 0
            else len(self.track_ids)
        )

        # Fleet Modal Composition
        total_class_vol = sum(self.class_counts.values()) or 1
        fleet_pct = {
            cls_name: round((cnt / total_class_vol) * 100.0, 1)
            for cls_name, cnt in self.class_counts.items()
        }
        heavy_veh_count = self.class_counts.get("bus", 0) + self.class_counts.get("truck", 0)
        heavy_veh_pct = round((heavy_veh_count / total_class_vol) * 100.0, 1)

        # Congestion Index & Level of Service
        if calibrated:
            mean_speed_kmh = mean_speed * 3.6
            los_rating = self._evaluate_hcm_los(density_lane_km, mean_speed_kmh)
        else:
            mean_speed_kmh = 0.0
            los_rating = (
                "Light Density"
                if avg_active < 4.0
                else "Moderate Density"
                if avg_active < 10.0
                else "Heavy Density"
            )

        congestion = (
            "Light" if avg_active < 4.0 else "Moderate" if avg_active < 10.0 else "Heavy"
        )

        result = {
            "frames_processed": self.frames_processed,
            "unique_vehicle_tracks": len(self.track_ids),
            "average_active_vehicles_per_frame": avg_active,
            "peak_active_vehicles": max_active,
            "traffic_density_veh_per_lane_km": density_lane_km,
            "level_of_service": los_rating,
            "congestion_level": congestion,
            "estimated_hourly_flow_rate_vph": hourly_flow_rate,
            "total_vehicle_detections": self.total_detections,
            "vehicle_detections_by_class": dict(self.class_counts),
            "fleet_modal_split_percentage": fleet_pct,
            "heavy_vehicle_percentage": heavy_veh_pct,
            "average_speed": round(mean_speed, 3),
            "maximum_speed": round(self.max_speed, 3),
            "speed_std_dev": round(std_dev, 3),
            "speed_coeff_variation": round(coeff_var, 3),
            "speed_percentile_15": round(p15, 3),
            "speed_percentile_85": round(p85, 3),
            "speed_unit": "m/s" if calibrated else "pixels/s",
            "speed_calibrated": calibrated,
        }

        if calibrated:
            result["average_speed_kmh"] = round(mean_speed * 3.6, 2)
            result["maximum_speed_kmh"] = round(self.max_speed * 3.6, 2)
            result["speed_percentile_85_kmh"] = round(p85 * 3.6, 2)
            result["speed_percentile_15_kmh"] = round(p15 * 3.6, 2)

        return result
