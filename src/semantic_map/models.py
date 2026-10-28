"""Core types: observations are immutable evidence; tracks are mutable hypotheses."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


def vec3(a):
    a = np.asarray(a, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("Expected finite 3-vector")
    return a


def emb(a):
    if a is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all():
        raise ValueError("Invalid appearance embedding")
    n = np.linalg.norm(a)
    if n < 1e-9:
        raise ValueError("Zero appearance embedding")
    return a / n


@dataclass
class Observation:
    source: str
    frame_id: str
    detection_id: str
    label: str
    center: np.ndarray
    low: np.ndarray
    high: np.ndarray
    timestamp: float
    appearance: Optional[np.ndarray] = None
    description: str = ""
    center_cam: Optional[np.ndarray] = None
    T_map_cam: Optional[np.ndarray] = None
    map_version: int = 0
    confidence: float = 1.0
    map_id: str = "default"

    def __post_init__(self):
        self.center, self.low, self.high = map(vec3, (self.center, self.low, self.high))
        if np.any(self.high <= self.low):
            raise ValueError("Invalid 3D bounding box")
        self.appearance = emb(self.appearance)
        if self.center_cam is not None:
            self.center_cam = vec3(self.center_cam)
        if self.T_map_cam is not None:
            self.T_map_cam = np.asarray(self.T_map_cam, dtype=np.float64)
            if self.T_map_cam.shape != (4,4) or not np.isfinite(self.T_map_cam).all():
                raise ValueError("Invalid camera pose")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Invalid confidence")

    @property
    def key(self):
        return (self.map_id, self.source, self.frame_id, self.detection_id)


@dataclass
class Track:
    object_id: str
    label: str
    center: np.ndarray
    low: np.ndarray
    high: np.ndarray
    appearance: Optional[np.ndarray]
    hits: int
    first_seen: float
    last_seen: float
    state: str = "tentative"
    description: str = ""
    map_version: int = 0
    attributes: dict = field(default_factory=dict)
    map_id: str = "default"
