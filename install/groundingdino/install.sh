#!/usr/bin/env bash
set -Eeuo pipefail

# Install the MMDetection implementation of Grounding DINO used by this project.
# Run this script from anywhere. By default files are installed under the project
# root inferred from this script's location.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MMDET_DIR="${MMDET_DIR:-${PROJECT_ROOT}/mmdetection}"
MMDET_REPO="${MMDET_REPO:-https://github.com/open-mmlab/mmdetection.git}"
MMDET_REF="${MMDET_REF:-cfd5d3a985b0249de009b67d04f37263e11cdf3d}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-1}"
GDINO_WEIGHTS_URL="${GDINO_WEIGHTS_URL:-https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth}"

command -v git >/dev/null 2>&1 || { echo "error: git is required" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 9), "Python >= 3.9 is required"'

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

if ! "${PYTHON_BIN}" -c \
  'import torch, torchvision, sys; sys.exit(0 if torch.__version__.split("+")[0] == sys.argv[1] and torchvision.__version__.split("+")[0] == sys.argv[2] else 1)' \
  "${TORCH_VERSION}" "${TORCHVISION_VERSION}" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
fi

"${PYTHON_BIN}" -m pip install --upgrade openmim
"${PYTHON_BIN}" -m mim install "mmengine==0.10.7"
"${PYTHON_BIN}" -m mim install "mmcv==2.1.0"

if [[ ! -d "${MMDET_DIR}/.git" ]]; then
  if [[ -e "${MMDET_DIR}" ]]; then
    echo "error: ${MMDET_DIR} exists but is not a Git checkout" >&2
    exit 1
  fi
  git clone "${MMDET_REPO}" "${MMDET_DIR}"
  git -C "${MMDET_DIR}" checkout --detach "${MMDET_REF}"
else
  actual_ref="$(git -C "${MMDET_DIR}" rev-parse HEAD)"
  if [[ "${actual_ref}" != "${MMDET_REF}" ]]; then
    echo "warning: existing MMDetection checkout is ${actual_ref}; expected ${MMDET_REF}" >&2
  fi
fi

"${PYTHON_BIN}" -m pip install -r "${MMDET_DIR}/requirements/multimodal.txt"
"${PYTHON_BIN}" -m pip install "transformers==4.46.3"
"${PYTHON_BIN}" -m pip install -v -e "${MMDET_DIR}"

# Keep project-specific configs outside the vendored repository and copy them in
# after each clean installation.
if compgen -G "${PROJECT_ROOT}/configs/grounding_dino/*.py" >/dev/null; then
  cp "${PROJECT_ROOT}"/configs/grounding_dino/*.py \
    "${MMDET_DIR}/configs/grounding_dino/"
fi

if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
  checkpoint_dir="${MMDET_DIR}/checkpoints"
  checkpoint_file="${checkpoint_dir}/groundingdino_swint_ogc_mmdet-822d7e9d.pth"
  mkdir -p "${checkpoint_dir}"
  if [[ ! -s "${checkpoint_file}" ]]; then
    "${PYTHON_BIN}" -c \
      'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' \
      "${GDINO_WEIGHTS_URL}" "${checkpoint_file}"
  fi
fi

"${PYTHON_BIN}" - <<'PY'
import mmcv
import mmengine
import mmdet
import torch
print(f"torch={torch.__version__}")
print(f"mmcv={mmcv.__version__}")
print(f"mmengine={mmengine.__version__}")
print(f"mmdet={mmdet.__version__}")
PY

echo "Grounding DINO installation completed: ${MMDET_DIR}"
