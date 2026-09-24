#!/bin/bash
# ==============================================================================
# apply_patch.sh -- overlay this project's verl customizations onto an
# official verl repository
# ==============================================================================
# This project (PS-Pipeline) does not ship a full copy of verl -- it only
# contains the files that are new or modified relative to official verl
# (located under patch/). Before use, clone an official verl repo and run
# this script to overlay patch/ on top of it.
#
# Usage:
#   git clone https://github.com/volcengine/verl.git /path/to/verl
#   bash scripts/apply_patch.sh /path/to/verl
#
# Then:
#   cd /path/to/verl && pip install -e .
#   export VERL_REPO_DIR=/path/to/verl
#   bash run_train.sh --algorithm grpo ...
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_DIR="${RELEASE_ROOT}/patch"

VERL_REPO_DIR="$1"

if [ -z "$VERL_REPO_DIR" ]; then
    echo "Usage: bash apply_patch.sh <verl repo root>"
    echo "Example: bash apply_patch.sh /path/to/verl"
    exit 1
fi

if [ ! -d "${VERL_REPO_DIR}/verl" ]; then
    echo "Error: ${VERL_REPO_DIR} does not look like a valid verl repo root"
    echo "(${VERL_REPO_DIR}/verl subdirectory not found -- please git clone official verl first)"
    exit 1
fi

echo "=============================================="
echo "  Applying the PS-Pipeline patch to an official verl repo"
echo "=============================================="
echo "Patch source : ${PATCH_DIR}"
echo "Target repo  : ${VERL_REPO_DIR}"
echo ""

# Overlay/add files inside the verl package
if [ -d "${PATCH_DIR}/verl" ]; then
    echo "[1/2] Copying patch/verl/* -> ${VERL_REPO_DIR}/verl/"
    cp -rv "${PATCH_DIR}/verl/." "${VERL_REPO_DIR}/verl/"
fi

# Overlay/add recipe files (e.g. DAPO's main_dapo.py / dapo_ray_trainer.py)
if [ -d "${PATCH_DIR}/recipe" ]; then
    echo "[2/2] Copying patch/recipe/* -> ${VERL_REPO_DIR}/recipe/"
    mkdir -p "${VERL_REPO_DIR}/recipe"
    cp -rv "${PATCH_DIR}/recipe/." "${VERL_REPO_DIR}/recipe/"
fi

echo ""
echo "Done. Next steps:"
echo "  cd ${VERL_REPO_DIR} && pip install -e ."
echo "  export VERL_REPO_DIR=${VERL_REPO_DIR}"
echo "  bash ${RELEASE_ROOT}/run_train.sh --algorithm grpo ..."
