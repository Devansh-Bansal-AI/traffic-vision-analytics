import cv2
import numpy as np


class HomographyTransformer:
    """Maps image points on the road plane to real-world planar coordinates."""

    def __init__(self, image_points, world_points):
        self.image_points = np.asarray(image_points, dtype=np.float32)
        self.world_points = np.asarray(world_points, dtype=np.float32)
        if self.image_points.shape != (4, 2) or self.world_points.shape != (4, 2):
            raise ValueError("image_points and world_points must each contain four (x,y) points")
        self.H = cv2.getPerspectiveTransform(self.image_points, self.world_points)

    def transform_point(self, point):
        pts = np.array([[point]], dtype=np.float32)
        out = cv2.perspectiveTransform(pts, self.H)[0, 0]
        return float(out[0]), float(out[1])
