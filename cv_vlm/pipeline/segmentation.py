from __future__ import annotations

import dataclasses
from typing import List

import cv2
import numpy as np


@dataclasses.dataclass
class MaskRegion:
    mask: np.ndarray  # bool mask, shape HxW
    class_name: str
    score: float


class DepthMaskGenerator:
    """Lightweight mask extractor using depth proximity.

    This is a placeholder that prefers near-field regions and splits them into
    connected components to mimic instance masks without running a heavy model.
    """

    def __init__(self, near_percentile: float = 15.0, far_percentile: float = 70.0, min_area: int = 200):
        self.near_percentile = near_percentile
        self.far_percentile = far_percentile
        self.min_area = min_area

    def __call__(self, depth: np.ndarray, default_class: str = "object") -> List[MaskRegion]:
        finite_depth = depth[np.isfinite(depth) & (depth > 0)]
        if finite_depth.size == 0:
            return []
        near = np.percentile(finite_depth, self.near_percentile)
        far = np.percentile(finite_depth, self.far_percentile)
        window_mask = (depth >= near) & (depth <= far)
        window_mask = window_mask.astype(np.uint8)
        # remove noise
        window_mask = cv2.morphologyEx(window_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        num_labels, labels = cv2.connectedComponents(window_mask)
        regions: List[MaskRegion] = []
        for label in range(1, num_labels):
            region = labels == label
            area = int(region.sum())
            if area < self.min_area:
                continue
            score = float(area) / depth.size
            regions.append(MaskRegion(mask=region, class_name=default_class, score=score))
        return regions
