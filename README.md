# Semantic Memory Mapping — offline RGB-D (experimental baseline)

## Приёмка (обязательное исправление статуса)

**На 28.10.2026 полная система НЕ принята.**
Существующая карта из 15 763 вокселей — наблюдаемые поверхности с внешними
эталонными позами, не RTAB-Map SLAM, не TSDF, не подтверждённая Nav2 occupancy.
На реальных кадрах обнаружены ошибки/фрагментация объектных треков.
Снижение unresolved не доказывает стабильность ID.
Синтетический IDF1=1.0 не является real-world метрикой.

См. [реестр приёмки](docs/acceptance_status.md): нарушения имеют статус FAIL,
непроверенные компоненты — NOT_TESTED/IMPLEMENTED_UNVALIDATED. Они **не являются
фиксированным достижением или лучшей архитектурой**.

Новая реализация геометрического слоя: offline_mapping.py, TSDF +
лучевое свободное/занятое пространство, новый export PLY/PGM/YAML.
Она подлежит отдельному CI на всей TUM последовательности и использует GT poses.
**Реконструкция по GT не измеряет ATE/RPE или качество loop closure**. Для
полного SLAM RTAB-Map и реального Re-ID нужны отдельные проверки.

Команда геометрической реконструкции (Open3D требуется):
python -m semantic_map.offline_mapping --dataset data/tum --output results/full_geometry


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

## Extended reproducible research (2026-10-28)

Now includes the following **separately scoped components**:

- `python -m semantic_map.benchmark` — identity association ablation on **synthetic** 3D object observations.
- `python -m semantic_map.benchmark_extensions` — simulated multiview semantic voting and selective VLM call policy.
- `python -m semantic_map.dashboard` — interactive 3D synthetic object map, 2D occupancy, A* and Russian commands.
- `python -m semantic_map.tum_yolo_pipeline` — default YOLOE-26s-seg with 12 text-prompted categories and 2-stage temporal RGB-D tracker, followed by persistent UUID fusion; first run downloads YOLOE weights and MobileCLIP embeddings.
- `python -m semantic_map.visualize_real` — real observed surface point cloud and detected object centers visualization.

**Real processing and GT evaluation are different**: TUM Freiburg1 Desk has RGB-D/poses, but no comprehensive physical-object instance ID annotations, so real-world IDF1/mAP cannot be claimed from this data alone.

Research methodology and the verified focused bibliography are in `docs/research_2026.md`. See `results/` for synthetic experiment files; actual TUM run outputs are generated by GitHub Actions artifacts when the workflow completes.

### Completed real-sequence benchmark and replay ablations

- [Full RGB-D + YOLO sequence run](https://github.com/KiRiII-C/cv_vlm/actions/runs/40700640646): 596 frames / 2541 observations / 15763 surface voxels.
- [Real observation-level association ablations](https://github.com/KiRiII-C/cv_vlm/actions/runs/40701631314): geometry, appearance, fused and adaptive modes.
- See `results/project_report.html`, `results/tum_real/real_map_interactive.html`, and `docs/research_2026.md`.

**Scientific status:** full object uniqueness, learned VLM/Cross-Attention, loop-closure reconciliation and on-robot Nav2 have **not** passed real-world validation.


## One-command offline RGB-D end-to-end (Ubuntu 24.04)

Dependencies (CPU by default, CUDA optional for YOLO on a local RTX):
~~~bash
sudo apt-get install -y libegl1 libgl1 libgomp1
python3.11 -m pip install -e '.[test]' 'open3d==0.19.0' ultralytics matplotlib plotly 'transformers>=4.55,<6' pillow
bash scripts/run_offline_e2e.sh
~~~

To use CUDA for YOLO, set YOLO_DEVICE=0. Maps, trajectory, object memories,
semantic descriptions and operator command results are saved in
results/offline_full/ (override OUTPUT_DIR). The frame source is the public
TUM fr1_desk sequence, with benchmark ground-truth ONLY for final pose
evaluation, not pose estimation.

**Latest independently reproduced scoped results (2026-10-28):**
- 596 RGB-D frames, 9 nonlocal pose constraints; custom ORB+PnP posegraph
  ATE RMSE 0.05498 m, RPE@30 0.04777 m. [E2E run](https://github.com/KiRiII-C/cv_vlm/actions/runs/40748057181).
- 298 training RGB-D frames -> TSDF; 298 withheld frames:
  468,722 depth points, median depth-to-map distance 0.03103 m,
  coverage <=0.10m 98.40%. [Holdout CI](https://github.com/KiRiII-C/cv_vlm/actions/runs/40746269127).
- 2,541 real pretrained YOLO instance observations in persistent SQLite;
  54 confirmed tracks, **not proven 54 physical identities**.
- Real SmolVLM-256M inference on 24 confirmed object proposals
  (up to 2 views each): 12 lexically consistent descriptions, 12 quarantined.
- Actual ROS2 Jazzy Nav2 ComputePathToPose action on reconstructed
  offline occupancy: SUCCEEDED, 30 poses to object UUID. [Nav2 CI](https://github.com/KiRiII-C/cv_vlm/actions/runs/40748419765).

**Not accepted:** cross-scene physical identity persistence without duplicates,
GT instance-ID accuracy, real robot localization and navigation, independent
multi-scene surface reconstruction GT accuracy, RTAB-Map integration. The
custom VO is not RTAB-Map. See docs/acceptance_status.md and
config/acceptance.json; a green technical CI is NOT full project acceptance.



## Validated labeled 3D evaluation and denser instance map (2026-10-28)

**3D semantic GT (14 actual RGB-D Scenes v2 reconstructed scans):**
Train on scenes 01–10; evaluate only scenes 11–14 with official per-point
semantic labels. Point-level foreground mIoU: RGB-only 16.33%,
geometry-only 13.56%, RGB+geometry 22.96%, multiscale 21.39%.
See [3D GT benchmark methodology](docs/gt_3d_semantic_benchmark.md)
and [executed workflow](https://github.com/KiRiII-C/cv_vlm/actions/runs/40752415584).

**Full actual RGB-D input → YOLO masks → 3D instance candidates → GT:**
For scenes 11 and 14 we downloaded original 3DMatch RGB-D frames and
calibration and processed 120 frames each. Foreground 3D mIoU of the
projected semantic map = 18.51% (scene 11) and 40.79% (scene 14).
The GT supplies semantic categories but not physical instance IDs.
See [two-scene end-to-end GT workflow](https://github.com/KiRiII-C/cv_vlm/actions/runs/40753853627).

**More dense full TUM map and colored object surfaces:**
TSDF integration at 640px and voxel size 3cm yields 90,058 triangles
(up from 20,532). The predicted masked RGB-D points are fused at 1.5cm
and assigned a persistent color per physical track UUID:
108,562 color-coded object 3D points across 54 *candidate* track IDs.
Each detected object appears as a colored set of actual observed 3D
surface points (not just a marker). 3D Plotly map and PLY are exported.
See [full dense e2e workflow](https://github.com/KiRiII-C/cv_vlm/actions/runs/40752675350).

**Blocker remains:** Neither the TUM nor the public RGB-D Scenes v2 GT
provides physically unique object IDs. Therefore real GT IDF1,
false-split/false-merge and physical re-ID correctness are not established.
A separate physical-instance-labeled dataset is necessary. Do not
interpret colored UUIDs or scene-specific category mIoU as proof of
correct physical object identification.


## YOLOE-26 migration and tracking (current)

- Default **YOLOE-26s-seg.pt** (official open-vocabulary instance masks),
  **12 fixed classes** in config/yoloe26_classes.json.
- TemporalTracker tracks 3D geometry + 2D bbox/mask overlap +
  masked spatial HSV (optional MobileNetV3 features). ByteTrack-like
  low-confidence rescue, ambiguity abstention, max 40 missing frames.
- Persistent global SemanticMapper uses 3D world geometry, SQLite UUIDs,
  class and pose-version gates, conservative short-term hints, and
  up to eight distinct multi-view descriptors per object.
- Run: bash scripts/run_offline_e2e.sh (YOLO_DEVICE=0 to use RTX CUDA).
- Direct detector smoke: 40 actual TUM frames, 330 segmentation masks,
  20 confirmed track *candidates*, 15,538 colored 3D points:
  [GitHub Actions](https://github.com/KiRiII-C/cv_vlm/actions/runs/40759845090).
- Physical GT evaluation: Hypersim, selected target-only scenes. Initial
  ai_001_001 contains only one target-class GT physical object; no broad
  conclusions are valid from that scene. Expanded tests compare ai_002_001
  (12 target physical IDs) and ai_005_001 (128 crowded chair IDs).
  Current tracker has NOT met strict physical ID accuracy requirements.
- Technical methods and evaluated upstream projects:
  [docs/yoloe26_reid_tracking.md](docs/yoloe26_reid_tracking.md).


## Долговременная память физических объектов ObjectMemoryV2 (28.10.2026)

**По умолчанию в сквозном YOLOE-26 пайплайне:** объектная память
ObjectMemoryV2 с 12 контролируемыми текстовыми классами, упор на физический
UUID, а не на класс детектора или краткосрочный tracklet.

Новая память включает:
- Долговременные UUID в SQLite, неизменяемые исходные измерения и provenance;
- multi-view appearance prototypes с quality gate, class+world geometry;
- 1:1 Hungarian, право НЕ идентифицировать неоднозначную маску,
  явный журнал review и запрет неконтролируемых merge;
- физически осмысленные class-specific пределы смещения статических
  объектов (без расширения радиуса из-за раздутого AABB);
- 1,5 см цветные UUID-воксели с support/negative evidence и хранением
  физически измеренных RGB-D поверхностей;
- атомарную коррекцию координат объектов после pose-graph revision.
  После loop closure voxel layer **обязательно перестроить** из RGB-D:
  stale layer запрещено экспортировать; ID сохраняется, но требуется
  повторная ассоциация для подтверждения физической правильности;
- безопасный экспериментальный relocation gate: три наблюдения предмета
  на новом месте + две независимые, разнесённые по позиции RGB-D
  проверки освобождения старого. Проверен тестами, не на реальном роботе.

**Важные подтверждённые IDF1:** две development сцены Hypersim при
одинаковых детекциях YOLOE-26 и измерениях GT физического ID:
разные объекты 30,93% → 32,69%; похожие стулья 2,59% → 4,77%,
без false merge. Но **на заранее отложенной ai_017_001** IDF1
не улучшился: 35,18% → 34,75%; исходный FAILED статус
полного physical-ID ReID остаётся неизменным.

Измерения:
[fixed-input regression](https://github.com/KiRiII-C/cv_vlm/actions/runs/40766015906),
[independent unseen scene](https://github.com/KiRiII-C/cv_vlm/actions/runs/40766680560).
[Полный научный отчёт, ограничения, источники и методы](docs/object_memory_v2_research.md).

Внутри Docker/Ubuntu запустить:
~~~bash
YOLO_DEVICE=0 bash scripts/run_offline_e2e.sh
~~~
