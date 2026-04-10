#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-batch}"

DATA_DIR="/home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260405_161057"
OUTPUT_ROOT="/home/wycaihyj/Documents/WYC/Depth_SLAM/FoundationStereo/auto_detection_weld/outputs"

COMMON_ARGS=(
  --data_dir "${DATA_DIR}"
  --depth_source foundation_stereo
  --runtime_mode save_only
  --enable_sam 1
  --enable_qwen 1
  --qwen_max_new_tokens 128
)

if [[ "${MODE}" == "single" ]]; then
  conda run -n foundation_stereo python auto_detection_weld/cli.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "${OUTPUT_ROOT}/weld_hybrid_single_00060" \
    --frame_id 000060 \
    --max_color_frame_delta 0
elif [[ "${MODE}" == "batch" ]]; then
  conda run -n foundation_stereo python auto_detection_weld/cli.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "${OUTPUT_ROOT}/weld_hybrid_batch" \
    --start_frame_id 000030 \
    --max_frames 10 \
    --max_color_frame_delta 2
else
  echo "Usage: bash run_weld_hybrid.sh [single|batch]"
  exit 1
fi

