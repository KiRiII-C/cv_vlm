# YOLOE-26X: controlled experimental isolation of perception errors

Date: 2026-10-29.

Model: official Ultralytics yoloe-26x-seg.pt, text-prompted 12 categories from config/yoloe26_classes.json. Hypersim synthetic scenes ai_002_001 and ai_005_001 (20 RGB frames each), 126 and 1340 GT physical target-object appearances.

All 3D coordinates are Hypersim GT world XYZ (SLAM NOT TESTED), held equal in EVERY variant. Only predefined NYU40 target categories with unambiguous prompt mappings are scored: 3 cabinet, 4 bed, 5 chair, 6 sofa, 7 table, 10 bookshelf, 14 desk -> table, 25 television. Other model prompts are not individually labeled by NYU40 GT. No physical GT UUID is passed to ObjectMemoryV2 when matching. Perfect GT instance masks/classes are used only inside explicitly identified oracle interventions.

## Measured four hypotheses

| Metric | Diverse furniture | Many similar chairs |
|---|---:|---:|
| Ground truth physical appearance occurrences | 126 | 1340 |
| Actual model instance masks at conf=0.10 | 197 | 186 |
| Recall at 2D mask IoU>=0.50 | 38.10% | 6.94% |
| Correct class among diagnostic 2D-IoU>=0.10 matches | 64.10% | 100.00% |
| Mean IoU of corresponding masks | 65.10% | 49.96% |
| Mean fraction of predicted mask outside same GT instance | 24.72% | 44.20% |
| GT instances with >=2 overlapping predicted masks, frame events | 17 | 45 |
| Predicted masks overlapping >=2 different physical GT instances, frame events | 17 | 78 |

The split/merge numbers are **mask-overlap events**, not permanent physical ID false merges. Low IoU and off-object pixel contamination affect 3D object geometry and appearance.

## Same-input controlled ObjectMemoryV2 IDF1

| Variant | Diverse | Similar chairs |
|---|---:|---:|
| Raw YOLOE26x masks, classes, detections | 0.39858 | 0.19174 |
| Correct GT class for aligned existing detections only | 0.46528 | 0.19174 |
| Correct GT mask for aligned existing detections only | 0.43902 | 0.24115 |
| Correct mask and class, presence fixed | 0.48611 | 0.24115 |
| Remove GT-identified duplicate detections only | 0.40569 | 0.19668 |
| Add missing GT instances (GT mask + class for additions) | 0.55215 | 0.95994 |
| All GT masks, classes and physical detections | 0.89600 | 0.99776 |

Matched pairs for replacement are found by label-agnostic 2D IoU Hungarian assignment (IoU >=0.10). The assignment and physical identity are never used for raw YOLOE26x inference or raw ObjectMemoryV2 association.

The GT-recall intervention is a diagnostic upper bound that inserts entirely missing objects with a correct class and a perfect mask. This is not an automated error corrector. Variant differences are NOT independent or additive.

## False-merges versus confidence threshold

| Confidence cut-off | Diverse scene IDF1 | Crowded scene IDF1 | Crowded global false merges |
|---:|---:|---:|---:|
| 0.10 | 0.39858 | 0.19174 | 8 |
| 0.15 | 0.38376 | 0.12265 | 3 |
| 0.20 | 0.36220 | 0.08280 | 1 |
| 0.25 | 0.34959 | 0.05652 | 0 |
| 0.35 | 0.34862 | 0.03804 | 0 |

Increasing threshold reduces wrong global matches, but recall and IDF1 collapse. A more useful follow-up is to retain low-confidence candidates for evidence association while requiring high reliability for new permanent object creation and Nav2 goals. That policy is not yet GT validated.

## Executed source and provenance

- Actual YOLOE-26x 2-scene 7-condition GT CI: https://github.com/KiRiII-C/cv_vlm/actions/runs/40769982699
- Same frozen detection confidence replays: https://github.com/KiRiII-C/cv_vlm/actions/runs/40770506902
- Script: src/semantic_map/yoloe26x_factorial.py
- Output: per-scene factorial_results.json, fixed raw.json.gz and GT control caches.
- Exact model documentation: https://docs.ultralytics.com/models/yoloe
- Official Hypersim physical instance and world XYZ benchmark:
  https://github.com/apple/ml-hypersim

Current acceptance: detector runtime and experimental measurement PASS; stable long-term real physical instance ID FAIL. 20-frame synthetic Hypersim with GT world poses is NOT equivalent to real on-robot SLAM+Nav2.
