from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class Pose:
    """SE3 pose expressed as translation + quaternion (x, y, z, qx, qy, qz, qw)."""

    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float

    def as_matrix(self) -> np.ndarray:
        x, y, z, qx, qy, qz, qw = (
            self.tx,
            self.ty,
            self.tz,
            self.qx,
            self.qy,
            self.qz,
            self.qw,
        )
        # Quaternion to rotation matrix
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm == 0:
            raise ValueError("Quaternion norm is zero")
        qx, qy, qz, qw = (qx / norm, qy / norm, qz / norm, qw / norm)
        R = np.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
            ]
        )
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.array([x, y, z], dtype=np.float64)
        return T


@dataclasses.dataclass(frozen=True)
class TUMFrame:
    timestamp: float
    rgb_path: Path
    depth_path: Path
    pose: Pose


@dataclasses.dataclass
class TUMDataset:
    root: Path
    time_tolerance: float = 0.02

    def _read_list(self, file_name: str) -> List[Tuple[float, str]]:
        file_path = self.root / file_name
        pairs: List[Tuple[float, str]] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                timestamp = float(parts[0])
                filename = parts[1]
                pairs.append((timestamp, filename))
        return pairs

    def _read_groundtruth(self) -> Dict[float, Pose]:
        poses: Dict[float, Pose] = {}
        gt_path = self.root / "groundtruth.txt"
        with gt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ts, tx, ty, tz, qx, qy, qz, qw = map(float, line.split())
                poses[ts] = Pose(tx, ty, tz, qx, qy, qz, qw)
        return poses

    def _associate(
        self, rgb: List[Tuple[float, str]], depth: List[Tuple[float, str]], poses: Dict[float, Pose]
    ) -> List[TUMFrame]:
        depth_index = 0
        frames: List[TUMFrame] = []
        for ts_rgb, rgb_file in rgb:
            while depth_index + 1 < len(depth) and depth[depth_index + 1][0] <= ts_rgb:
                depth_index += 1
            best_depth = depth[depth_index]
            if abs(best_depth[0] - ts_rgb) > self.time_tolerance:
                continue
            pose_key = min(poses.keys(), key=lambda t: abs(t - ts_rgb)) if poses else None
            if pose_key is None or abs(pose_key - ts_rgb) > self.time_tolerance:
                continue
            frames.append(
                TUMFrame(
                    timestamp=ts_rgb,
                    rgb_path=self.root / rgb_file,
                    depth_path=self.root / best_depth[1],
                    pose=poses[pose_key],
                )
            )
        return frames

    def load(self) -> List[TUMFrame]:
        rgb_list = self._read_list("rgb.txt")
        depth_list = self._read_list("depth.txt")
        poses = self._read_groundtruth()
        return self._associate(rgb_list, depth_list, poses)

    def iter_frames(self) -> Iterator[TUMFrame]:
        for frame in self.load():
            yield frame


def depth_to_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert a depth map (in meters) to a Nx3 point cloud using intrinsics K."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    height, width = depth.shape
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    zs = depth
    xs = (xs - cx) * zs / fx
    ys = (ys - cy) * zs / fy
    points = np.stack((xs, ys, zs), axis=-1)
    return points.reshape(-1, 3)


def transform_points(points: np.ndarray, pose: Pose) -> np.ndarray:
    T = pose.as_matrix()
    homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    world = homo @ T.T
    return world[:, :3]
