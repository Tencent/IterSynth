#!/bin/bash
# ==============================================================================
# init_node_env.sh -- per-node initialization script for multi-node training
# ==============================================================================
# Single-node training does not need this script (run_train.sh will
# automatically `ray start --head`).
#
# Multi-node usage:
#   1. On the head node:
#        ROLE=head bash scripts/init_node_env.sh
#      Note the printed head node IP.
#   2. On every worker node:
#        ROLE=worker RAY_HEAD_ADDR=<head_ip>:6379 bash scripts/init_node_env.sh
#   3. Back on the head node, set AUTO_START_RAY=false and run run_train.sh:
#        AUTO_START_RAY=false NNODES=<N> bash run_train.sh --algorithm grpo ...
#
# This script also exports a set of optional training-stability /
# exploration-control environment variables (rubric scoring, EntroPIC
# entropy control, FlexEC entropy control, etc.) -- see the "Advanced
# Training Techniques" section in README.md for details. All of these
# have safe defaults and can be ignored if you don't need them.
# ==============================================================================

set -e

ROLE="${ROLE:-head}"          # head | worker
RAY_HEAD_ADDR="${RAY_HEAD_ADDR:-}"   # required in worker mode, e.g. 1.2.3.4:6379
RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-0.0.0.0}"

# ==============================================================================
# 1. Start Ray
# ==============================================================================

if [ "$ROLE" = "head" ]; then
    echo "Starting Ray head node..."
    ray start --head --dashboard-host="${RAY_DASHBOARD_HOST}"
else
    if [ -z "$RAY_HEAD_ADDR" ]; then
        echo "Error: RAY_HEAD_ADDR must be set in worker mode (e.g. 1.2.3.4:6379)"
        exit 1
    fi
    echo "Starting Ray worker node, connecting to ${RAY_HEAD_ADDR} ..."
    ray start --address="${RAY_HEAD_ADDR}"
fi

# ==============================================================================
# 2. Install required Python dependencies (must be consistent across nodes)
# ==============================================================================

echo "Installing Python dependencies..."

# json_repair: used to parse LLM output that may contain malformed JSON
pip install json_repair

# Ray and newer protobuf versions can have a gencode/runtime version
# mismatch (gencode 7.x requires runtime 7.x). Uncomment if you hit
# related errors:
# pip install "protobuf>=7.34.1"

# wandb is disabled by default to avoid extra dependency conflicts. To
# enable it, `pip install wandb` and add
# trainer.logger=['console','tensorboard','wandb'] to TRAINER_PARAMS in
# run_train.sh.
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# ==============================================================================
# 3. Tool backend credentials (search_tool.py / visit_tool.py)
# ==============================================================================
# These are required for the multi-turn tool-calling rollout to actually
# retrieve search results and webpage content. See README.md for how to
# obtain each key.

# Serper.dev API key, used by the "search" tool (https://serper.dev)
export SERPER_API_KEY="${SERPER_API_KEY:-}"
# Jina Reader API key, used by the "visit" tool (optional -- anonymous
# access works with lower rate limits; https://jina.ai/reader)
export JINA_API_KEY="${JINA_API_KEY:-}"
# OpenAI-compatible LLM used by the "visit" tool to extract/summarize
# webpage content relevant to the search goal
export VISIT_SUMMARY_API_KEY="${VISIT_SUMMARY_API_KEY:-}"
export VISIT_SUMMARY_API_BASE="${VISIT_SUMMARY_API_BASE:-https://api.openai.com/v1}"
export VISIT_SUMMARY_MODEL_NAME="${VISIT_SUMMARY_MODEL_NAME:-gpt-4o-mini}"

if [ -z "$SERPER_API_KEY" ]; then
    echo "[Warning] SERPER_API_KEY is not set; the 'search' tool will return an error at runtime."
fi
if [ -z "$VISIT_SUMMARY_API_KEY" ]; then
    echo "[Warning] VISIT_SUMMARY_API_KEY is not set; the 'visit' tool will not be able to summarize page content."
fi

# ==============================================================================
# 4. Rubric scoring configuration (PS-Pipeline / Rubric Reward Manager)
# ==============================================================================

export ROLLOUT_RUBRIC_ENABLED="${ROLLOUT_RUBRIC_ENABLED:-true}"
export RUBRIC_MAX_CONCURRENT_CALLS="${RUBRIC_MAX_CONCURRENT_CALLS:-16}"
export RUBRIC_MAX_QPM="${RUBRIC_MAX_QPM:-200}"

# Important (multi-node only): jobs submitted via `ray job submit` run
# inside the Ray node process environment, not the shell that launched
# run_train.sh. For single-node training, run_train.sh starts Ray itself
# so the environments are naturally consistent; for multi-node training,
# make sure the two rubric file paths below are identical and reachable
# on every node (use a shared filesystem with an absolute path, or deploy
# an identical copy of this repo on every node):
# export PS_PLANNER_RUBRIC_PATH=/shared/ps-pipeline-release/rubrics/planner_rubric.json
# export PS_SYNTHESIZER_RUBRIC_PATH=/shared/ps-pipeline-release/rubrics/synthesizer_rubric.json

# LLM Judge credentials used for rubric scoring and answer verification.
# Please provide your own via environment variables -- do not hardcode them.
export LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY:-}"
export LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-https://api.openai.com/v1}"

# ==============================================================================
# 5. EntroPIC entropy-stabilization control (optional; PI-controller-based
#    dynamic entropy stabilization)
# ==============================================================================
# TARGET_ENTROPY >= 0 enables it, < 0 disables it. EntroPIC and FlexEC are
# mutually exclusive -- do not enable both at the same time.

export ENTROPIC_TARGET_ENTROPY="${ENTROPIC_TARGET_ENTROPY:--1}"
export ENTROPIC_KP="${ENTROPIC_KP:-3.0}"
export ENTROPIC_KI="${ENTROPIC_KI:-0.002}"
export ENTROPIC_HIGH_PROB_THRESH="${ENTROPIC_HIGH_PROB_THRESH:-0.95}"
export ENTROPIC_INTEGRAL_LIMIT="${ENTROPIC_INTEGRAL_LIMIT:-20.0}"
export ENTROPIC_INTEGRAL_DECAY="${ENTROPIC_INTEGRAL_DECAY:-0.01}"
export ENTROPIC_DEADBAND="${ENTROPIC_DEADBAND:-0.02}"
export ENTROPIC_RESET_ON_FLIP="${ENTROPIC_RESET_ON_FLIP:-true}"

# EntroPIC requires an additional entropic_dp_actor.py implementation
# (not included in this open-source release). If you have your own
# implementation, place it at
# $VERL_REPO_DIR/verl/workers/actor/entropic_dp_actor.py and swap the
# import in fsdp_workers.py manually (see the commented-out sed command
# below) to enable it.
if [ "$(python3 -c "print(1 if float('${ENTROPIC_TARGET_ENTROPY}') >= 0 else 0)" 2>/dev/null || echo 0)" = "1" ]; then
    echo "[EntroPIC] target_entropy=${ENTROPIC_TARGET_ENTROPY} >= 0, EntroPIC was requested"
    echo "[EntroPIC] Note: this open-source release does not include entropic_dp_actor.py; implement your own and wire it up per README"
    # sed -i 's/from verl\.workers\.actor import DataParallelPPOActor/from verl.workers.actor.entropic_dp_actor import EntroPICDataParallelPPOActor as DataParallelPPOActor/' \
    #     "${VERL_REPO_DIR}/verl/workers/fsdp_workers.py"
fi

# ==============================================================================
# 6. FlexEC entropy control (optional; dynamic per-token clip range based
#    on gradient-preserving clipping)
# ==============================================================================
# Paper: arXiv:2602.09782 "Flexible Entropy Control in RLVR with
# Gradient-Preserving Perspective"
# Usage: append actor_rollout_ref.actor.policy_loss.loss_mode=flexec to
# run_train.sh, and make sure EntroPIC is disabled (ENTROPIC_TARGET_ENTROPY=-1).

export FLEXEC_STRATEGY="${FLEXEC_STRATEGY:-id}"          # id | did | od | static
export FLEXEC_PHASE_SPLIT="${FLEXEC_PHASE_SPLIT:-0.5}"
export FLEXEC_UPPER_SLOPE="${FLEXEC_UPPER_SLOPE:--0.25}"
export FLEXEC_UPPER_INTERCEPT="${FLEXEC_UPPER_INTERCEPT:-0.5}"
export FLEXEC_LOWER_SLOPE="${FLEXEC_LOWER_SLOPE:--0.13}"
export FLEXEC_LOWER_INTERCEPT="${FLEXEC_LOWER_INTERCEPT:-0.3}"
export FLEXEC_CLIP_BOUND_MIN="${FLEXEC_CLIP_BOUND_MIN:-0.05}"
export FLEXEC_CLIP_BOUND_MAX="${FLEXEC_CLIP_BOUND_MAX:-0.6}"
export FLEXEC_OD_H_INIT="${FLEXEC_OD_H_INIT:-0.5}"
export FLEXEC_OD_H_MIN_RATIO="${FLEXEC_OD_H_MIN_RATIO:-0.2}"

# Like EntroPIC, FlexEC's implementation (core_algos_flexec.py) is not
# included in this open-source release. If you have your own
# implementation, place it at
# $VERL_REPO_DIR/verl/trainer/ppo/core_algos_flexec.py -- it self-registers
# via the @register_policy_loss decorator, so it only needs to be
# imported once inside core_algos.py (see below).
if [ -n "$VERL_REPO_DIR" ] && [ -f "${VERL_REPO_DIR}/verl/trainer/ppo/core_algos_flexec.py" ]; then
    CORE_ALGOS_FILE="${VERL_REPO_DIR}/verl/trainer/ppo/core_algos.py"
    if ! grep -q "core_algos_flexec" "$CORE_ALGOS_FILE" 2>/dev/null; then
        {
            echo ""
            echo "# [FlexEC] trigger flexec policy loss registration"
            echo "try:"
            echo "    from verl.trainer.ppo import core_algos_flexec  # noqa: F401"
            echo "except Exception as _e:"
            echo "    print(f'[FlexEC] import failed: {_e}')"
        } >> "$CORE_ALGOS_FILE"
        echo "[FlexEC] Appended import to core_algos.py"
    fi
else
    echo "[FlexEC] core_algos_flexec.py not found, skipping auto-registration (implement your own if needed)"
fi

# ==============================================================================
# 7. Other training behavior switches
# ==============================================================================

# exceed_max_turns sample handling policy:
#   True  - included in advantage computation (as a reward=0 contrastive
#           sample within the group), but excluded from the loss
#   False - included in both advantage computation and the loss
#           (reward=0, the model learns to avoid running out of turns)
export EXCLUDE_MAX_TURNS_FROM_LOSS="${EXCLUDE_MAX_TURNS_FROM_LOSS:-False}"

# Distributed training timeouts (increase for long-context / slow-network setups)
export GLOO_SOCKET_TIMEOUT_MS="${GLOO_SOCKET_TIMEOUT_MS:-7200000}"
export GLOO_TIMEOUT_MS="${GLOO_TIMEOUT_MS:-7200000}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-WARN}"

echo ""
echo "Node initialization complete (role=${ROLE})."
