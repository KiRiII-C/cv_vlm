#!/usr/bin/env bash
# A8: run one independent full-dataset RTAB-Map localization arm.
# Each mode loads a FRESH COPY of the same mapping SQLite database.
set -euo pipefail
: "${A8_MODE:?A8_MODE must be first_node, saved_last, or wrong_8m}"
A8_MAP_DB="${A8_MAP_DB:-results/a8_map/rtabmap.db}"
A8_QUERY_DATA="${A8_QUERY_DATA:-data/tum_a7}"
case "$A8_MODE" in
  first_node|saved_last|wrong_8m) ;;
  *) echo "Unexpected A8 mode: $A8_MODE" >&2; exit 2 ;;
esac
out="results/a8/$A8_MODE"
mkdir -p "$out"
test -s "$A8_MAP_DB"
test -s "$A8_QUERY_DATA/rgbd_dataset_freiburg1_desk2/rgb.txt"
cp "$A8_MAP_DB" "$out/rtabmap.db"
sha256sum "$A8_MAP_DB" "$out/rtabmap.db" > "$out/input_db_sha256.txt"
if ! cmp -s "$A8_MAP_DB" "$out/rtabmap.db"; then
  echo "A8 map copy differs before launch" >&2
  exit 1
fi
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
# Do not post /initialpose in ANY experimental arm.
map_args=(-p 'Mem/IncrementalMemory:="false"'
          -p 'Mem/LocalizationReadOnly:="true"')
case "$A8_MODE" in
  first_node)
    map_args+=(-p 'RGBD/StartAtOrigin:="true"')
    ;;
  saved_last)
    map_args+=(-p 'RGBD/StartAtOrigin:="false"')
    ;;
  wrong_8m)
    map_args+=(-p 'RGBD/StartAtOrigin:="true"')
    map_args+=(-p 'initial_pose:=8 0 0 0 0 3.141593')
    ;;
esac
python3 -m py_compile scripts/relocalize_independent_tum_a7.py \
  scripts/monitor_relocalization_a6.py
ros2 run rtabmap_odom rgbd_odometry --ros-args \
  -p frame_id:=camera_color_optical_frame \
  -p subscribe_rgbd:=true -p publish_tf:=true \
  -p always_process_most_recent_frame:=false \
  -r rgbd_image:=/camera/rgbd_image -r odom:=/odom \
  -r odom_info_lite:=/odom_info_lite \
  > "$out/odom.log" 2>&1 &
ODOM_PID=$!
ros2 run rtabmap_slam rtabmap --ros-args \
  -p frame_id:=camera_color_optical_frame \
  -p subscribe_rgbd:=true -p subscribe_depth:=true \
  -p subscribe_odom_info:=false -p publish_tf:=true \
  -p pub_loc_pose_only_when_localizing:=true \
  -p database_path:="/workspace/$out/rtabmap.db" \
  "${map_args[@]}" \
  -r rgbd_image:=/camera/rgbd_image -r odom:=/odom \
  > "$out/slam.log" 2>&1 &
SLAM_PID=$!
python3 -u scripts/monitor_relocalization_a6.py --out "$out" > "$out/monitor.log" 2>&1 &
MON_PID=$!
cleanup(){
  kill -TERM "$MON_PID" 2>/dev/null || true
  kill -INT "$SLAM_PID" "$ODOM_PID" 2>/dev/null || true
  sleep 4
  kill -TERM "$SLAM_PID" "$ODOM_PID" 2>/dev/null || true
  wait "$MON_PID" "$SLAM_PID" "$ODOM_PID" 2>/dev/null || true
}
trap cleanup EXIT
sleep 8
if ! kill -0 "$SLAM_PID" 2>/dev/null; then
  tail -100 "$out/slam.log"
  echo "RTAB-Map died before A8 query, invalid trial" >&2
  exit 1
fi
# No subsampling; start at original first query frame and preserve time order.
python3 -u scripts/relocalize_independent_tum_a7.py \
  --data "$A8_QUERY_DATA" --output "$out/full_desk2.json" \
  --max-rate-hz 15 --ack-timeout-s 5 \
  > "$out/replay.log" 2>&1 || {
    tail -70 "$out/replay.log"
    tail -70 "$out/slam.log"
    exit 1
  }
sleep 8
kill -TERM "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true
# Explicit record of which initialization was actually configured.
grep -E 'Setting initial pose|initial_pose|StartAtOrigin|last localization|Update map correction' "$out/slam.log" \
  > "$out/initialization_evidence.txt" || true
if [[ "$A8_MODE" == wrong_8m ]]; then
  if ! grep -q 'Setting initial pose' "$out/slam.log"; then
    echo "Wrong initial_pose injection was NOT confirmed; trial invalid" >&2
    exit 1
  fi
fi
echo "A8_ARM_COMPLETE $A8_MODE"
tail -6 "$out/replay.log"
tail -3 "$out/monitor.log"
