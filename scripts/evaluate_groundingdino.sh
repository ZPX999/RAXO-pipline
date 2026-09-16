#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate an RGB-pretrained or fine-tuned Grounding DINO checkpoint on SIXray-D.
# Set CHECKPOINT to a local fine-tuned checkpoint to override the official
# RGB-pretrained Swin-T weight downloaded by install/groundingdino/install.sh.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MMDET_DIR="${MMDET_DIR:-${PROJECT_ROOT}/mmdetection}"
CONFIG="${CONFIG:-${MMDET_DIR}/configs/grounding_dino/grounding_dino_swin_t_si_xray.py}"
CHECKPOINT="${CHECKPOINT:-${MMDET_DIR}/checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth}"
TEST_JSON="${TEST_JSON:-${PROJECT_ROOT}/annotations/si_xray_d/test.json}"
TEST_IMAGES="${TEST_IMAGES:-${PROJECT_ROOT}/dataset/SIXray-D/images/test}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/outputs/groundingdino/sixray_d_rgb_eval}"

for path in "${MMDET_DIR}/tools/test.py" "${CONFIG}" "${CHECKPOINT}" "${TEST_JSON}" "${TEST_IMAGES}"; do
  if [[ ! -e "${path}" ]]; then
    echo "error: required path does not exist: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${WORK_DIR}"
"${PYTHON_BIN}" "${MMDET_DIR}/tools/test.py" "${CONFIG}" "${CHECKPOINT}" \
  --work-dir "${WORK_DIR}" \
  --cfg-options \
    "test_dataloader.dataset.ann_file=${TEST_JSON}" \
    "test_dataloader.dataset.data_prefix.img=${TEST_IMAGES}" \
    "test_evaluator.ann_file=${TEST_JSON}" \
    "test_evaluator.outfile_prefix=${WORK_DIR}/predictions"

