#!/usr/bin/env bash
# Run after installing: pip install -e '.[test]' 'open3d==0.19.0' ultralytics matplotlib plotly 'transformers>=4.55,<6' pillow
# Ubuntu runtime: libegl1 libgl1 libgomp1. GPU optional via YOLO_DEVICE=0.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=2
DATA="${TUM_DATA:-data/tum}"
OUTPUT="${OUTPUT_DIR:-results/offline_full}"
YOLO_MODEL="${YOLO_MODEL:-yoloe-26s-seg.pt}"
YOLO_DEVICE="${YOLO_DEVICE:-cpu}"
mkdir -p "$OUTPUT"/{vo,geometry,gravity,semantic,vlm,operator}
echo '[1/9] RGB-D ORB+PnP camera pose estimation'
python -m semantic_map.visual_odometry --data "$DATA" --output "$OUTPUT/vo"
echo '[2/9] Verified loop-closure pose graph'
python -m semantic_map.pose_graph --poses "$OUTPUT/vo/estimated_poses.json" \
  --vo "$OUTPUT/vo/vo_evaluation.json" --out "$OUTPUT/vo" --data "$DATA"
echo '[3/9] All-frame RGB-D TSDF and occupancy'
python -m semantic_map.offline_mapping --dataset "$DATA" \
  --pose-json "$OUTPUT/vo/optimized_poses.json" --output "$OUTPUT/geometry" \
  --max-frames 0 --voxel .06 --tsdf-side 320
echo '[4/9] Conservative gravity-aligned occupancy'
python -m semantic_map.floor_alignment --geometry "$OUTPUT/geometry" --out "$OUTPUT/gravity"
echo '[5/9] Real instance segmentation plus persisted object memory'
python -m semantic_map.tum_yolo_pipeline --data "$DATA" \
  --output "$OUTPUT/semantic" --model "$YOLO_MODEL" \
  --device "$YOLO_DEVICE" --classes-json config/yoloe26_classes.json --conf .23 --max-frames 0 --memory-engine v2 --persist-object-surfaces --pose-json "$OUTPUT/vo/optimized_poses.json"
echo '[6/9] Conservative reconciliation and cross-visit identity audit'
python -m semantic_map.offline_reconcile --db "$OUTPUT/semantic/semantic_memory.sqlite" \
  --out "$OUTPUT/semantic/reconciliation.json"
python -m semantic_map.identity_audit --root "$OUTPUT/semantic" \
  --out "$OUTPUT/semantic/identity_audit.json"
echo '[7/9] Visualize real 3D scene'
python -m semantic_map.visualize_real --root "$OUTPUT/semantic"
echo '[8/9] VLM on selected object views'
python -m semantic_map.vlm_enrich --root "$OUTPUT/semantic" \
  --out "$OUTPUT/vlm" --max-objects 40
echo '[9/9] Offline goal grounding and navigation candidate paths'
python -m semantic_map.operator_e2e --memory "$OUTPUT/semantic" \
  --geometry "$OUTPUT/geometry" --gravity "$OUTPUT/gravity" --output "$OUTPUT/operator"
echo 'OFFLINE_E2E_COMPLETED (no physical-ID GT or actual Nav2 motion claim)'
echo "Results: $OUTPUT"
