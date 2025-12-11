from __future__ import annotations

import numpy as np

from cv_vlm.mapping.object_store import MapObject


def compute_goal(object_: MapObject, offset_m: float = 0.8) -> np.ndarray:
    """Compute a naive goal point offset from the object centroid.

    The goal is placed on the line from the world origin to the object, stepping
    back by ``offset_m`` so the robot stops before colliding with the object.
    """

    centroid = object_.centroid
    distance = np.linalg.norm(centroid)
    if distance < 1e-6:
        return centroid
    direction = centroid / distance
    goal = centroid - direction * offset_m
    return goal
