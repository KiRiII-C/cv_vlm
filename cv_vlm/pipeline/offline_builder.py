from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from tqdm import tqdm

from cv_vlm.datasets.tum import TUMDataset, depth_to_points, transform_points
from cv_vlm.mapping.object_store import ObjectStore
from cv_vlm.pipeline.segmentation import DepthMaskGenerator


@dataclasses.dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def as_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)


@dataclasses.dataclass
class BuilderConfig:
    dataset_root: Path
    output_dir: Path
    intrinsics: Intrinsics
    depth_scale: float = 5000.0
    default_class: str = "object"
    max_points_per_mask: int = 8000
    merge_distance: float = 0.6
    near_percentile: float = 10.0
    far_percentile: float = 70.0
    min_area: int = 300


class OfflineMapBuilder:
    def __init__(self, config: BuilderConfig):
        self.config = config
        self.segmenter = DepthMaskGenerator(
            near_percentile=config.near_percentile,
            far_percentile=config.far_percentile,
            min_area=config.min_area,
        )
        self.store = ObjectStore(merge_distance=config.merge_distance)

    def _load_depth(self, path: Path) -> np.ndarray:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Cannot read depth image: {path}")
        return raw.astype(np.float32) / float(self.config.depth_scale)

    def process_frame(self, depth: np.ndarray, pose, timestamp: float) -> None:
        masks = self.segmenter(depth, default_class=self.config.default_class)
        if not masks:
            return
        camera_points = depth_to_points(depth, self.config.intrinsics.as_matrix())
        for region in masks:
            indices = region.mask.reshape(-1)
            region_points = camera_points[indices]
            region_points = region_points[np.isfinite(region_points).all(axis=1)]
            if len(region_points) == 0:
                continue
            if len(region_points) > self.config.max_points_per_mask:
                choice = np.random.choice(len(region_points), self.config.max_points_per_mask, replace=False)
                region_points = region_points[choice]
            world_points = transform_points(region_points, pose)
            self.store.add_observation(region.class_name, world_points, timestamp)

    def run(self) -> ObjectStore:
        dataset = TUMDataset(self.config.dataset_root)
        frames = list(dataset.iter_frames())
        for frame in tqdm(frames, desc="Building map"):
            depth = self._load_depth(frame.depth_path)
            self.process_frame(depth, frame.pose, frame.timestamp)
        self.store.save(self.config.output_dir)
        return self.store
