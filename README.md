# Semantic Memory Mapping — offline RGB-D (experimental baseline)

**Primary objective:** construct a world-aligned 3D instance map with **persistent IDs** and
semantic descriptions, without silently duplicating objects when they leave and re-enter
the camera field of view. The aim is not depth completion.

## Implemented and tested

- Camera-calibrated, metric masked RGB-D → 3D points using `T_map_cam`.
- Per-observation robust 3D centroid and AABB.
- Durable SQLite object memory with source evidence and aliases.
- Global spatial + semantic-class + appearance data association, one-to-one Hungarian matching.
- Re-identification across disappearance, process restarts and frame replay.
- Provisional identities requiring two corroborating observations before semantic confirmation.
- Ambiguous/conflicting assignments **not** inserted as new canonical objects.
- Conservative duplicate merge proposals with co-visibility protection; human/validation approval before merging.
- Descriptions/attributes stored per object, allowing later on-demand VLM updates.
- Offline JSONL adapter accepting actual depth maps, segmentation masks and feature vectors.
- Lightweight observed-surface voxel store with PLY export (NOT a TSDF, ESDF or navigation occupancy grid).
- CPU unit/integration tests and GitHub Actions CI; ROS 2 Jazzy Dockerfile with RTAB-Map dependencies.

## Important limitations

**This is an infrastructure and algorithmic baseline, not a completed robot stack.** No
segmentation model, VLM, learned cross-attention encoder, RTAB-Map launch/ROS bag adapter,
loop-closure map correction, object-level accuracy benchmark, or Nav2 goal execution
is implemented yet. Embeddings must come from a real feature encoder and masks from a
real detector/segmenter. We do not claim zero duplicate objects or error-free re-ID.

The tests use deterministic constructed examples, *not* benchmark re-ID scores.
Cross-attention must be evaluated on persistent instances rather than recovering depth.

## Quick start (Ubuntu/runner)

```bash
python -m pip install -e '.[test]'
pytest -q
python -m semantic_map.demo --db demo.sqlite
```

## ROS 2 environment (Docker on a host)

```bash
docker build -t semantic-map:jazzy .
docker run --rm -it -v "$PWD:/workspace" semantic-map:jazzy
```

Docker is optional for CI; the main Python core works without ROS.

## Offline manifest schema (one JSON object per line)

```json
{
  "source":"tum_freiburg1_room", "frame_id":"123.450", "timestamp":123.45,
  "map_version":0,
  "camera":{"fx":525,"fy":525,"cx":319.5,"cy":239.5,
            "width":640,"height":480,"depth_unit_m":0.0002},
  "depth_path":"frames/depth/123.png",
  "T_map_cam":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "detections":[{"id":"det_0","label":"chair",
                  "mask_path":"masks/123_chair.png", "confidence":0.92,
                  "embedding":[0.31,0.18,0.84],
                  "description":"dark chair near wooden table"}]
}
```

**Camera depth scaling** is dataset-specific: for standard TUM Freiburg depth PNGs,
`depth_unit_m` is 1/5000; for the TUM Scribble dataset it is 1/1000.
`T_map_cam` must contain the pose of the **optical** camera frame in the persistent map
frame, including optical-axis convention. The JSON sample only illustrates the schema.

```bash
python -m semantic_map.manifest frames.jsonl --db world.sqlite --report report.json
```

If RTAB-Map loop closure changes the pose graph, old object positions **must be
reprojected** from their original observations before doing cross-session matching.
The MVP currently refuses association across different `map_version` values.

## Architecture and milestones

See [`docs/architecture.md`](docs/architecture.md). CI runs automatically on changes
within branch `feature/semantic-memory-mapping`. Main branch is unchanged.
