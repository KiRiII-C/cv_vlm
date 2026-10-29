# Physical ID root causes: YOLOE-26x over all 300 Hypersim frames

Date 2026-10-29. Scene cameras ai_002_001, ai_005_001, ai_017_001,
100/100 consecutive RGB-D frames each, official physical-instance GT.

**Evaluation status: FAIL for persistent physical-ID stability.**
GT 3D world positions are provided to ALL runs, so this is not SLAM.
GT physical IDs are exclusively for metrics/explicit oracle interventions.

## What has the largest effect?

There are 6,612 target-physical GT instance-frame appearances. Only
1,553 (23.49%) have an aligned YOLOE-26x mask observation under the
benchmark's IoU >= 0.10 one-to-one Hungarian mapping. Thus 5,059
(76.51%) lack a matched predicted object observation.

Pooled across three separate scene-specific physical-ID matching tables
(sum IDTP, IDFP, IDFN; not arithmetic scene average):

| Intervention, all 300 frames | Pooled IDF1 | Difference vs raw |
|---|---:|---:|
| Raw YOLOE-26x predictions | 21.613% | baseline |
| GT labels on existing matched detections, same masks | 24.166% | +2.553 pp |
| GT-mask-derived 3D geometry + appearance, same detection presence | 29.465% | +7.852 pp |
| Correct both classes and GT 3D observation | 31.009% | +9.396 pp |
| Add missing objects with PERFECT GT mask and class | 85.281% | +63.668 pp |
| Only oracle GT observations | 95.821% | +74.208 pp |
| Suppress appearance descriptor, same detections | 21.600% | -0.013 pp |

**Conclusion: missing object observations dominate pooled IDF1**, with
inaccurate mask geometry second and class label mistakes third. This
ordering is scene-dependent: ai_017_001 gains +15.09 pp from oracle
mask geometry, +16.19 pp from missing-object insertion, +7.26 pp
from corrected classes. On ai_005_001 missing-object oracle gains
+89.18 pp, dwarfing geometry +4.82 pp.

The missing-object intervention inserts oracle GT labels and PERFECT
GT masks for wholly missed instances. It is a diagnostic upper bound,
not a deployable recovery algorithm. Deltas are not additive.

## Does a new ID appear after loss of view or during continuous visibility?

Exactly **270** confirmed observed changes between different UUIDs
for the SAME GT object while running ObjectMemoryV2:

| Visibility between the two assigned observations | Events |
|---|---:|
| Consecutive frames, object detected in both | 122 (45.2%) |
| Object continuously GT-visible; detector missed >=1 intermediate frame | 53 (19.6%) |
| GT-visible, detector aligned intermediate frames but global UUID was NULL | 29 (10.7%) |
| Object truly left field of view and reappeared | 66 (24.4%) |

204/270 (75.6%) changes occurred with *continuous GT visibility*.
**Only 76/270 (28.1%) were creation of a completely new world UUID**,
grounded in the actual per-observation SQLite status created_tentative.
The other 194/270 were switches to a previously created UUID (which
might be a previously created ID of the same GT object or another track).
Among the 76 new-UUID creations following an older ID of the same GT:
53 happened while GT remained visible and 23 after real FOV absence.

## What prevented matching to the previous UUID?

An instrumented pre-frame hard-gate check of the previously assigned
world UUID found:

- 125/270 (46.3%): predicted class changed, hard category gate fails.
- 127/270 (47.0%): static class-specific 3D radius fails, often because
  partial/contaminated masks shift the measured centroid.
  39/41 UUID switches in crowded-chair scene exceed the chair cap 0.45m.
- 18/270 (6.7%): candidate competition, thresholds, remaining geometry.

This gate check is diagnostic, not a formal causal proof: earlier bad
associations may have polluted the old track's center or appearance.
Physical GT IDs were NEVER supplied to ObjectMemoryV2 scoring.

## Exact three ID layers: not the same IDs

1. YOLOE-26x outputs per-frame masks in an array and class indexes
   from boxes.cls. The class index is not a physical instance ID.
   RGBD ingestion assigns a per-frame detection_id such as det0.
   The unique key of an observation is (map_id, source, frame_id,
   detection_id). Reusing det0 in the next frame is normal.
2. The optional TemporalTracker assigns short_0000001-like temporary
   short_id using 3D geometry, image bbox, mask IoU and appearance.
   It removes tracks absent for more than 40 frames. This ID is
   ephemeral, not a persistent physical UUID, and is not stored as
   the primary key of semantic memory.
3. ObjectMemoryV2 assigns a persistent obj_<16 hex> world UUID from a
   deterministic uuid5 of the initial observation key. Many per-frame
   masks and different short tracklets MAY map to the same world UUID.
   On ambiguity the observation has NULL object_id and is retained
   as unassigned evidence. Permanent objects are stored in SQLite
   objects table; observation-to-world-ID links in observations and
   object_evidence_v2; colored 3D voxels keyed by the world UUID.

In the main TUM pipeline: YOLOE26x -> RGB-D observation -> TemporalTracker
-> soft prior from short_id -> ObjectMemoryV2 global Hungarian and class,
geometry, appearance gates -> persistent obj_* UUID / NULL ->
semantic memory and colored object clouds.

**Crucial limitation**: the full 300-frame Hypersim ID benchmark and
this error-attribution replay did NOT run TemporalTracker. They invoked
ObjectMemoryV2 directly on frozen real YOLOE26x observations.
Hence no evidence here establishes that TemporalTracker is incorrect;
270 events are the global memory's associations, not temporal track
switches. To benchmark the local tracker, save mask/bbox and short_id
on a GT-annotated full scene and evaluate both identity layers.

Within-frame same-GT duplicate observations cannot be counted as same
GT in this benchmark because GT-to-predicted matching is 1:1 Hungarian;
they can instead contribute to FP counts.

## Verified experiments

- 300 real YOLOE26x frame inference vs both memories and GT masks:
  https://github.com/KiRiII-C/cv_vlm/actions/runs/40820074555
- Exact pre-frame gate, SQLite creation/match and visibility classification:
  https://github.com/KiRiII-C/cv_vlm/actions/runs/40822126640
- Full 100-frame-per-scene, seven-condition causal oracle diagnostic:
  https://github.com/KiRiII-C/cv_vlm/actions/runs/40822432945

Source: src/semantic_map/identity_failure_attribution.py and
src/semantic_map/full_sequence_oracle_attribution.py.

**Engineering implications**: improve candidate recall and
multi-frame mask support; decouple stable geometry identity from
potentially changing YOLO class; use robust multi-view object geometry
rather than one-view centroid radius; resolve ambiguous hypotheses
without duplicating world UUID; add direct GT benchmarking of
TemporalTracker vs ObjectMemoryV2. No unvalidated changes to production
matching thresholds are claimed as improvements.
