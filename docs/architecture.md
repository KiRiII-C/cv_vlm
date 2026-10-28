# Multimodal persistent semantic mapping: architecture and acceptance criteria

## Data flow

1. RGB-D `color`, `registered_depth`, camera `K`, timestamps, `T_map_camera`.
2. RTAB-Map (ROS 2 Jazzy): global pose graph, loop closures, geometric map and occupancy.
3. Open-vocabulary instance proposals: Grounding DINO / YOLOE + SAM 2 as alternatives.
4. Appearance encoder (DINOv2, SigLIP/CLIP) + metric object shape from depth masks.
5. Object association: one-to-one assignment in map frame, geometry/embedding/class gates,
   uncertainty hold, tentative tracks, global re-entry retrieval, duplicate candidates.
6. Persistent object graph and semantic memory: SQLite (baseline), optional FAISS for
   nearest-neighbor retrieval, source evidence, descriptions and object relations.
7. VLM on *confirmed instance crops and selected views*, rather than on every video frame.
8. Natural-language grounding to an existing 3D object ID, spatial relation constraints,
   then Nav2 goal with clearance and reachable-space validation.

## Identity invariants (required)

- Never use only a 2D tracker ID for global object identity.
- Maintain `map_uuid`, `map_version`, global frame and persistent `object_uuid`.
- A single track may receive at most one observation per synchronized frame.
- New observations that conflict with known objects remain unresolved, not immediately
  instantiated as duplicate confirmed objects.
- Provisional tracks require repeated supporting evidence; a single-frame detection is
  not a confirmed landmark.
- Retain all observation evidence to audit, split, merge and reassociate IDs.
- A loop closure invalidating camera poses triggers replay/reprojection and reassociation;
  never compare stale XYZ across coordinate systems.
- Conflicting same-class objects visible simultaneously cannot be merged.
- Track state and semantic text are separate: renaming/re-description cannot change identity.
- Memory survives process restarts and new sessions within the same map frame.
- Reject unknown registration or low-confidence relocalization; never force a match.

## Scientific evaluation

Use sequences with full trajectories and instance annotations (Replica, ScanNet / ScanNet++,
3RScan, RGB-D Scenes where licensing and annotations permit). Dataset selection and split
must prevent train/test scene leakage. Exact dataset adapters remain to be implemented.

### Primary metrics

- 3D instance segmentation: mAP@0.25/0.5 IoU, PQ and class-agnostic mask AP.
- Multi-view object tracking: IDF1, ID switches, HOTA / AssA, fragmentation.
- Map-level duplicates: false split rate (# physical objects with >1 canonical ID),
  false merge rate (# distinct physical objects sharing one canonical ID), precision/recall.
- Re-entry tests: identification accuracy after N hidden frames, changed viewpoint,
  full camera excursion, distinct similar-class objects and cross-session relocalization.
- Mapping: ATE/RPE for pose, mesh/point accuracy & completeness, occupancy precision/recall.
- Semantics: open-vocabulary retrieval Recall@K, description correctness, relation accuracy.
- Navigation: goal localization error, reachable-goal rate, collision-free plan success.

### Controlled ablations

1. Geometry-only vs appearance-only vs combined + class priors.
2. DINOv2 / SigLIP features vs learned RGB–3D Cross-Attention embeddings.
3. 2D mask + projected 3D cluster vs learned point/voxel encoder.
4. Always-new vs global re-ID vs ambiguity hold vs duplicate merge/split.
5. Frozen poses vs pose-graph correction / reprojected historical observations.
6. VLM single view vs multi-view aggregate semantic descriptors.

## Milestones

- **M0 — done as a baseline**: Python, offline schema, persistent store, global association,
  re-entry/idempotence/co-visibility tests, CI; Docker config for Jazzy.
- **M1 — pending**: Real benchmark dataset reader/GT, 3D evaluation harness and metrics.
- **M2 — pending**: Actual detector/segmenter and appearance embedding extraction.
- **M3 — pending**: RTAB-Map pose graph + loop-closure-aware map correction and 3D surfaces.
- **M4 — pending**: Train RGB–3D fusion on instance association/segmentation, not depth completion.
- **M5 — pending**: VLM semantic descriptions, relations, retrieval, NL goal selection and Nav2.
- **M6 — pending**: End-to-end evaluation on withheld scenes and physical robot integration.

## Architectural precedent

OVI-MAP (CVPR 2026) separates class-agnostic 3D instance reconstruction from
on-demand semantic inference. Its design is a relevant baseline; we are not claiming
its published performance for this project.
