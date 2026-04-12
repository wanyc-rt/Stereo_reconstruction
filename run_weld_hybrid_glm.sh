#!/usr/bin/env bash
set -euo pipefail

if [[ -f "${HOME}/.bashrc" ]]; then
  # Load user-defined API key exports for non-interactive shell execution.
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi

MODE="${1:-single_glm}"

DATA_DIR="/home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260405_161057"
OUTPUT_ROOT="/home/wycaihyj/Documents/WYC/Depth_SLAM/FoundationStereo/auto_detection_weld/outputs"
PYTHON_BIN="conda run -n foundation_stereo python"
API_ENV_NAME="${ZAI_API_KEY_ENV:-ZAI_API_KEY}"
GLM_MODEL_NAME="${GLM_MODEL_NAME:-glm-5v-turbo}"

if [[ -z "${!API_ENV_NAME:-}" && -z "${BIGMODEL_API_KEY:-}" ]]; then
  echo "Missing API key."
  echo "Set ${API_ENV_NAME} or BIGMODEL_API_KEY before running."
  exit 1
fi

COMMON_ARGS=(
  --data_dir "${DATA_DIR}"
  --depth_source foundation_stereo
  --runtime_mode save_only
  --enable_sam 1
  --enable_glm 1
  --reasoner_backend glm
  --glm_model_name "${GLM_MODEL_NAME}"
  --glm_api_key_env "${API_ENV_NAME}"
)

if [[ "${MODE}" == "single_glm" ]]; then
  OUTPUT_DIR="${OUTPUT_ROOT}/weld_hybrid_single_00060_glm"
  ${PYTHON_BIN} auto_detection_weld/cli.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --frame_id 000060 \
    --max_color_frame_delta 0
elif [[ "${MODE}" == "batch_glm" ]]; then
  OUTPUT_DIR="${OUTPUT_ROOT}/weld_hybrid_batch_glm"
  ${PYTHON_BIN} auto_detection_weld/cli.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --start_frame_id 000060 \
    --max_frames 10 \
    --max_color_frame_delta 2
else
  echo "Usage: bash run_weld_hybrid_glm.sh [single_glm|batch_glm]"
  exit 1
fi

echo
echo "Output dir: ${OUTPUT_DIR}"
echo "JSON summary:"
${PYTHON_BIN} - <<PY
import json, glob, os
files = sorted(glob.glob(os.path.join("${OUTPUT_DIR}", "json", "*_weld_detection.json")))
if not files:
    print("No weld_detection.json files found.")
    raise SystemExit(0)
for path in files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(
        os.path.basename(path),
        "backend=", data.get("detector_backend"),
        "seam_shape=", data.get("seam_shape"),
        "straightness=", round(float(data.get("centerline_straightness", 0.0)), 4),
        "smoothness=", round(float(data.get("centerline_smoothness", 0.0)), 4),
    )
PY

echo
echo "GLM artifacts:"
${PYTHON_BIN} - <<PY
import glob, os
base = "${OUTPUT_DIR}"
sam_dir = os.path.join(base, "sam")
color_imgs = sorted(glob.glob(os.path.join(sam_dir, "*_glm_color.png")))
overlay_imgs = sorted(glob.glob(os.path.join(sam_dir, "*_glm_candidates.png")))
reasonings = sorted(glob.glob(os.path.join(sam_dir, "*_glm_reasoning.json")))
if not color_imgs and not overlay_imgs and not reasonings:
    print("No GLM artifacts found.")
    raise SystemExit(0)
for label, paths in (
    ("raw_color", color_imgs),
    ("candidate_overlay", overlay_imgs),
    ("reasoning_json", reasonings),
):
    print(f"[{label}]")
    if not paths:
        print("  none")
        continue
    for p in paths:
        print(" ", p)
PY
