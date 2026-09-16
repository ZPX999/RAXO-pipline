#!/usr/bin/env bash
set -Eeuo pipefail

# Install RAXO and the SAM2/DINOv3 components used by the local pipeline.
# DINOv3 weights require accepting Meta's license. Supply the authorized URL as
# DINOV3_WEIGHTS_URL, or copy the file to the printed destination afterwards.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-${PROJECT_ROOT}/third_party}"
RAXO_DIR="${RAXO_DIR:-${THIRD_PARTY_DIR}/RAXO}"
SAM2_DIR="${SAM2_DIR:-${THIRD_PARTY_DIR}/sam2}"
DINOV3_DIR="${DINOV3_DIR:-${THIRD_PARTY_DIR}/dinov3}"

RAXO_REPO="${RAXO_REPO:-https://github.com/PAGF188/RAXO.git}"
RAXO_REF="${RAXO_REF:-main}"
SAM2_REPO="${SAM2_REPO:-https://github.com/facebookresearch/sam2.git}"
SAM2_REF="${SAM2_REF:-2b90b9f5ceec907a1c18123530e92e794ad901a4}"
DINOV3_REPO="${DINOV3_REPO:-https://github.com/facebookresearch/dinov3.git}"
DINOV3_REF="${DINOV3_REF:-6876159a11b4df116f30f667f8c9888617df0751}"

TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
SAM2_WEIGHTS_URL="${SAM2_WEIGHTS_URL:-https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt}"
DINOV3_WEIGHTS_URL="${DINOV3_WEIGHTS_URL:-}"
DINOV3_WEIGHTS_NAME="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

command -v git >/dev/null 2>&1 || { echo "error: git is required" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 11), "Python >= 3.11 is required by the pinned DINOv3 source"'

clone_pinned() {
  local repo="$1"
  local ref="$2"
  local destination="$3"
  local label="$4"

  if [[ ! -d "${destination}/.git" ]]; then
    if [[ -e "${destination}" ]]; then
      echo "error: ${destination} exists but is not a Git checkout" >&2
      exit 1
    fi
    git clone "${repo}" "${destination}"
    git -C "${destination}" checkout --detach "${ref}"
  else
    echo "Using existing ${label} checkout: ${destination}"
  fi
}

download_file() {
  local url="$1"
  local destination="$2"
  mkdir -p "$(dirname -- "${destination}")"
  if [[ ! -s "${destination}" ]]; then
    "${PYTHON_BIN}" -c \
      'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' \
      "${url}" "${destination}"
  fi
}

mkdir -p "${THIRD_PARTY_DIR}"
if [[ -d "${RAXO_DIR}/raxo" && ! -d "${RAXO_DIR}/.git" ]]; then
  echo "Using bundled project-specific RAXO adaptation: ${RAXO_DIR}"
else
  clone_pinned "${RAXO_REPO}" "${RAXO_REF}" "${RAXO_DIR}" "RAXO"
fi
clone_pinned "${SAM2_REPO}" "${SAM2_REF}" "${SAM2_DIR}" "SAM2"
clone_pinned "${DINOV3_REPO}" "${DINOV3_REF}" "${DINOV3_DIR}" "DINOv3"

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
if ! "${PYTHON_BIN}" -c \
  'import torch, torchvision, sys; sys.exit(0 if torch.__version__.split("+")[0] == sys.argv[1] and torchvision.__version__.split("+")[0] == sys.argv[2] else 1)' \
  "${TORCH_VERSION}" "${TORCHVISION_VERSION}" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
fi

"${PYTHON_BIN}" -m pip install \
  numpy pillow opencv-python pycocotools scipy scikit-learn \
  matplotlib networkx nltk tabulate tqdm supervision torchmetrics \
  python-dotenv google-images-search openai tenacity tidecv

"${PYTHON_BIN}" -m pip install -e "${SAM2_DIR}"
"${PYTHON_BIN}" -m pip install -e "${DINOV3_DIR}"

sam2_checkpoint="${SAM2_DIR}/checkpoints/sam2.1_hiera_large.pt"
dinov3_checkpoint="${DINOV3_DIR}/weights/${DINOV3_WEIGHTS_NAME}"
download_file "${SAM2_WEIGHTS_URL}" "${sam2_checkpoint}"

if [[ -n "${DINOV3_WEIGHTS_URL}" ]]; then
  download_file "${DINOV3_WEIGHTS_URL}" "${dinov3_checkpoint}"
else
  mkdir -p "$(dirname -- "${dinov3_checkpoint}")"
  echo "DINOv3 weight download needs an authorized Meta URL."
  echo "Re-run with DINOV3_WEIGHTS_URL=<authorized-url>, or copy the file to:"
  echo "  ${dinov3_checkpoint}"
fi

"${PYTHON_BIN}" - <<PY
import sys
sys.path.insert(0, "${SAM2_DIR}")
import sam2
import torch
print(f"torch={torch.__version__}")
print(f"sam2={sam2.__file__}")
PY

echo "RAXO installation completed: ${RAXO_DIR}"
echo "export SAM2_CHECKPOINT=${sam2_checkpoint}"
