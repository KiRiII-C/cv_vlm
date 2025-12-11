from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclasses.dataclass
class MapObject:
    object_id: int
    class_name: str
    centroid: np.ndarray
    points_count: int
    observations: int
    last_seen: float

    def as_dict(self) -> Dict:
        return {
            "object_id": self.object_id,
            "class_name": self.class_name,
            "centroid": self.centroid.tolist(),
            "points_count": self.points_count,
            "observations": self.observations,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MapObject":
        return cls(
            object_id=int(data["object_id"]),
            class_name=str(data["class_name"]),
            centroid=np.array(data["centroid"], dtype=float),
            points_count=int(data.get("points_count", 0)),
            observations=int(data.get("observations", 0)),
            last_seen=float(data.get("last_seen", 0.0)),
        )


class ObjectStore:
    def __init__(self, merge_distance: float = 0.6):
        self.merge_distance = merge_distance
        self._objects: Dict[int, MapObject] = {}
        self._next_id = 0

    def _find_match(self, class_name: str, centroid: np.ndarray) -> Optional[int]:
        for object_id, obj in self._objects.items():
            if obj.class_name != class_name:
                continue
            if np.linalg.norm(obj.centroid - centroid) <= self.merge_distance:
                return object_id
        return None

    def add_observation(self, class_name: str, points: np.ndarray, timestamp: float) -> MapObject:
        centroid = points.mean(axis=0)
        object_id = self._find_match(class_name, centroid)
        if object_id is None:
            object_id = self._next_id
            self._next_id += 1
            self._objects[object_id] = MapObject(
                object_id=object_id,
                class_name=class_name,
                centroid=centroid,
                points_count=len(points),
                observations=1,
                last_seen=timestamp,
            )
        else:
            obj = self._objects[object_id]
            updated_points = obj.points_count + len(points)
            obj.centroid = (obj.centroid * obj.points_count + centroid * len(points)) / updated_points
            obj.points_count = updated_points
            obj.observations += 1
            obj.last_seen = timestamp
            self._objects[object_id] = obj
        return self._objects[object_id]

    def objects(self) -> List[MapObject]:
        return list(self._objects.values())

    def query(self, class_name: str) -> List[MapObject]:
        class_name_lower = class_name.lower()
        return [obj for obj in self._objects.values() if class_name_lower in obj.class_name.lower()]

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        data = [obj.as_dict() for obj in self._objects.values()]
        with (output_dir / "objects.json").open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, output_dir: Path) -> "ObjectStore":
        store = cls()
        obj_path = output_dir / "objects.json"
        if not obj_path.exists():
            return store
        with obj_path.open("r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            obj = MapObject.from_dict(entry)
            store._objects[obj.object_id] = obj
            store._next_id = max(store._next_id, obj.object_id + 1)
        return store
