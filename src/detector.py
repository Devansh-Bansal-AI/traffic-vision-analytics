from dataclasses import dataclass
from typing import List
import cv2
import numpy as np


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[int, int]
    area: float
    class_name: str = "vehicle"
    confidence: float = 1.0


class VehicleDetector:
    """Semantic vehicle detector (YOLO) with MOG2 as a course-aligned baseline."""

    COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, backend="yolo", model_path="yolo11n.pt", confidence=0.35,
                 image_size=640, device="cpu", min_area=1800):
        self.backend = backend.lower()
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.min_area = min_area
        self.kernel = np.ones((5, 5), np.uint8)

        if self.backend == "yolo":
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(
                    "YOLO requires ultralytics. Run: pip install -r requirements.txt"
                ) from exc
            self.model = YOLO(model_path)
            self.bg = None
        elif self.backend == "mog2":
            self.model = None
            self.bg = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=32, detectShadows=True
            )
        else:
            raise ValueError("backend must be 'yolo' or 'mog2'")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return self._detect_yolo(frame) if self.backend == "yolo" else self._detect_mog2(frame)

    def _detect_yolo(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            classes=list(self.COCO_VEHICLE_CLASSES),
            device=self.device,
            verbose=False,
        )
        detections = []
        if not results or results[0].boxes is None:
            return detections
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            w, h = x2 - x1, y2 - y1
            if w < 2 or h < 2:
                continue
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = self.COCO_VEHICLE_CLASSES.get(cls_id)
            if name is None:
                continue
            detections.append(
                Detection((x1, y1, w, h), (x1 + w // 2, y1 + h // 2), w * h, name, conf)
            )
        return detections

    def _detect_mog2(self, frame):
        mask = self.bg.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue
            ratio = w / h
            if not 0.2 <= ratio <= 5.0:
                continue
            detections.append(
                Detection((x, y, w, h), (x + w // 2, y + h // 2), area, "moving_object", 1.0)
            )
        return detections
