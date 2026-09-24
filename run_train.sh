#!/bin/bash
# ==============================================================================
# PS-Pipeline (Planner-Synthesizer) RL Training Launcher
# ==============================================================================
# This script assumes:
#   1. You have applied patch/verl/ on top of an official verl installation,
#      following the "Installation" section in README.md (via
#      scripts/apply_patch.sh, then `pip install -e .`).
#   2. A Ray cluster is already running (single node: this script will
#      automatically `ray start --head`; multi-node: run
#      scripts/init_node_env.sh on every node first, then run this script
#      on the head node).
#   3. Your training/validation data has been converted to the parquet
#      format expected by verl, and the `tools_kwargs` field's tool name
#      matches `name: search` in patch/verl/tools/tool_config.yaml.
#
# Supported algorithms (--algorithm):
#   dapo   - Dynamic sAmpling Policy Optimization, with filter_groups
#   grpo   - Standard GRPO (default, recommended starting point)
#   gspo   - Group Sequence Policy Optimization (seq-mean-token-mean loss)
#   drgrpo - Dr.GRPO (std-normalized advantage)
#
# Example:
#   export MODEL_PATH=/path/to/your/qwen3-8b
#   export TRAIN_FILES=/path/to/train.parquet
#   export VAL_FILES=/path/to/val.parquet
#   export CKPT_DIR=./checkpoints/my_run
#   bash run_train.sh --algorithm grpo --n-resp 8 --train-bsz 16
#
# See --help for the full parameter list.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# Required environment variables (no defaults; must be provided by the user)
# ==============================================================================
: "${MODEL_PATH:?Please set MODEL_PATH (base model path used by Actor/Rollout, HF format directory)}"
: "${TRAIN_FILES:?Please set TRAIN_FILES (training set parquet path, comma-separated for multiple files)}"
: "${VAL_FILES:?Please set VAL_FILES (validation set parquet path)}"
# VERL_REPO_DIR: local path to the official verl repo, after applying this
# project's patch/verl and patch/recipe via scripts/apply_patch.sh
# (see the "Installation" section in README.md). The `ray job submit`
# --working-dir flag points here, matching the `pip install -e .` directory.
: "${VERL_REPO_DIR:?Please set VERL_REPO_DIR (verl repo root with the patch applied, see README)}"

# ==============================================================================
# Paths / cluster configuration overridable via environment variables
# (all have sensible defaults)
# ==============================================================================
CKPT_DIR="${CKPT_DIR:-./checkpoints/ps_pipeline}"
LOG_DIR="${LOG_DIR:-./logs}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${CKPT_DIR}/rollout_data}"
TOOL_CONFIG="${TOOL_CONFIG:-${VERL_REPO_DIR}/verl/tools/tool_config.yaml}"
# Rubric definition files used by the Rubric Reward Manager
# (apiprimedapopspipelinerubric); default to the rubrics/*.json shipped
# with this repo (one scoring rubric each for the Planner and Synthesizer roles).
export PS_PLANNER_RUBRIC_PATH="${PS_PLANNER_RUBRIC_PATH:-${SCRIPT_DIR}/rubrics/planner_rubric.json}"
export PS_SYNTHESIZER_RUBRIC_PATH="${PS_SYNTHESIZER_RUBRIC_PATH:-${SCRIPT_DIR}/rubrics/synthesizer_rubric.json}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
# Single-node runs auto-start a local Ray cluster by default. For
# multi-node training, set AUTO_START_RAY=false and bootstrap the cluster
# manually via scripts/init_node_env.sh on each node beforehand.
AUTO_START_RAY="${AUTO_START_RAY:-true}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"

# ==============================================================================
# Default training configuration (overridable via CLI flags, see --help)
# ==============================================================================

ALGORITHM="grpo"  # dapo, grpo, gspo, drgrpo

# Context length (32K by default, suitable for most single-GPU 80GB setups;
# increase for long-context scenarios).
MAX_MODEL_LEN=$((1024 * 32))
MAX_PROMPT_LENGTH=$((1024 * 24))
MAX_RESPONSE_LENGTH=$((1024 * 8))
OVERLONG_BUFFER_LEN=$((1024 * 2))

TRAIN_BATCH_SIZE=16
N_RESP_PER_PROMPT=8
MINI_BATCH_SIZE=16

TEMPERATURE=1.0
TOP_P=0.95
TOP_K=20
VAL_TEMPERATURE=0.6
VAL_TOP_P=0.95
OVER_SAMPLE_RATE=0.2

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28

TP=8
SP=8
USE_DYNAMIC_BSZ="True"
OFFLOAD="True"
GPU_MEMORY_UTIL=0.75

LEARNING_RATE=1e-6
LR_WARMUP_RATIO=0.1
TOTAL_EPOCHS=3
SAVE_FREQ=5
TEST_FREQ=5

ENABLE_MULTI_TURN="True"
MAX_ASSISTANT_TURNS=20
MAX_USER_TURNS=20
MAX_TOOL_RESPONSE_LENGTH=8000

# DAPO-specific
ENABLE_FILTER_GROUPS="True"
MAX_NUM_GEN_BATCHES=10
FILTER_GROUPS_METRIC="acc"

EXPERIMENT_NAME=""
PROJECT_NAME="ps-pipeline"
DRY_RUN=false

# reward_model.reward_manager is auto-selected based on --algorithm, but
# can be explicitly overridden with --reward-manager (see REWARD_MANAGER
# below). Available registered managers:
#   apiprimedapopspipelinerubric  - PS-Pipeline + rubric group-centered reward (recommended)
#   apiprimedapopspipeline        - PS-Pipeline baseline (outcome reward only)
#   apiprimedapotrajectoryv2cgrpo - Single-agent ReAct + C-GRPO (Citation-aware Rubric)
#   apiprimedapotrajectoryv1naive - Single-agent ReAct + naive intermediate-fact reward
#   apiprimedapoendtoendonly      - Single-agent ReAct, end-to-end correctness reward only
REWARD_MANAGER=""

# ==============================================================================
# Help text
# ==============================================================================

show_help() {
    cat << EOF
PS-Pipeline RL Training Launcher

USAGE:
    bash run_train.sh --algorithm <ALGORITHM> [OPTIONS]

REQUIRED ENVIRONMENT VARIABLES:
    MODEL_PATH        Base model path (HF format directory)
    TRAIN_FILES       Training set parquet path
    VAL_FILES         Validation set parquet path
    VERL_REPO_DIR     verl repo root with the patch applied

ALGORITHM OPTIONS:
    --algorithm <name>            dapo | grpo | gspo | drgrpo (default: grpo)
    --reward-manager <name>       explicitly override the reward manager registry name

TRAINING OPTIONS:
    --train-bsz <int>             training batch size (default: 16)
    --n-resp <int>                samples per prompt / GRPO group size (default: 8)
    --mini-batch <int>            PPO mini batch size (default: 16)
    --experiment <name>           experiment name (default: auto-generated)
    --project <name>              tensorboard/W&B project name

CONTEXT LENGTH:
    --max-prompt-len <int>        max prompt length (default: 24576)
    --max-response-len <int>      max response length (default: 8192)
    --max-model-len <int>         max model context length (default: 32768)

SAMPLING PARAMETERS:
    --temperature <float>         sampling temperature (default: 1.0)
    --top-p <float>               default: 0.95
    --top-k <int>                 default: 20
    --over-sample-rate <float>    over-sampling ratio (default: 0.2)
    --clip-ratio-low <float>      default: 0.2
    --clip-ratio-high <float>     default: 0.28

PERFORMANCE:
    --tp <int>                    tensor parallel size (default: 8)
    --sp <int>                    Ulysses sequence parallel size (default: 8)
    --no-dynamic-bsz              disable dynamic batch size
    --no-offload                  disable parameter / optimizer offload
    --gpu-mem-util <float>        SGLang KV cache memory fraction (default: 0.75)

MULTI-TURN TOOL CALLING (PS-Pipeline / ReAct search scenarios):
    --disable-multi-turn          disable multi-turn tool calling
    --max-turns <int>             max assistant turns (default: 20)
    --max-tool-resp-len <int>     max characters per tool response (default: 8000)
    --tool-config <path>          custom tool config yaml (default: built-in tool_config.yaml)

TRAINING CONTROL:
    --lr <float>                  learning rate (default: 1e-6)
    --epochs <int>                total epochs (default: 3)
    --save-freq <int>             checkpoint save frequency (default: 5)
    --test-freq <int>             validation frequency (default: 5)

CLUSTER:
    --nnodes <int>                number of nodes (default: 1)
    --gpus-per-node <int>         GPUs per node (default: 8)
    --ray-address <url>           Ray Dashboard address (default: http://127.0.0.1:8265)
    --no-auto-ray                 don't auto-start local Ray (for multi-node training)

OTHER:
    --dry-run                     only print the config and final command, don't run
    --help, -h                    show this help message

EXAMPLES:
    # Single node, 8 GPUs, standard GRPO
    MODEL_PATH=/models/qwen3-8b TRAIN_FILES=./data/train.parquet VAL_FILES=./data/val.parquet \\
        bash run_train.sh --algorithm grpo

    # DAPO with dynamic filtering
    bash run_train.sh --algorithm dapo --train-bsz 32 --n-resp 16

    # Just preview the command that would run
    bash run_train.sh --algorithm grpo --dry-run
EOF
}

# ==============================================================================
# Parse command-line arguments
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --algorithm) ALGORITHM="$2"; shift 2 ;;
        --reward-manager) REWARD_MANAGER="$2"; shift 2 ;;
        --train-bsz) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --n-resp) N_RESP_PER_PROMPT="$2"; shift 2 ;;
        --mini-batch) MINI_BATCH_SIZE="$2"; shift 2 ;;
        --experiment) EXPERIMENT_NAME="$2"; shift 2 ;;
        --project) PROJECT_NAME="$2"; shift 2 ;;
        --max-prompt-len) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
        --max-response-len) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --over-sample-rate) OVER_SAMPLE_RATE="$2"; shift 2 ;;
        --clip-ratio-low) CLIP_RATIO_LOW="$2"; shift 2 ;;
        --clip-ratio-high) CLIP_RATIO_HIGH="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --sp) SP="$2"; shift 2 ;;
        --no-dynamic-bsz) USE_DYNAMIC_BSZ="False"; shift ;;
        --no-offload) OFFLOAD="False"; shift ;;
        --gpu-mem-util) GPU_MEMORY_UTIL="$2"; shift 2 ;;
        --disable-multi-turn) ENABLE_MULTI_TURN="False"; shift ;;
        --max-turns) MAX_ASSISTANT_TURNS="$2"; MAX_USER_TURNS="$2"; shift 2 ;;
        --max-tool-resp-len) MAX_TOOL_RESPONSE_LENGTH="$2"; shift 2 ;;
        --tool-config) TOOL_CONFIG="$2"; shift 2 ;;
        --lr) LEARNING_RATE="$2"; shift 2 ;;
        --epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
        --save-freq) SAVE_FREQ="$2"; shift 2 ;;
        --test-freq) TEST_FREQ="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --gpus-per-node) N_GPUS_PER_NODE="$2"; shift 2 ;;
        --ray-address) RAY_ADDRESS="$2"; shift 2 ;;
        --no-auto-ray) AUTO_START_RAY="false"; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) show_help; exit 0 ;;
        *) echo "Error: Unknown option: $1"; echo "Use --help for usage information"; exit 1 ;;
    esac
done

case $ALGORITHM in
    dapo|grpo|gspo|drgrpo) ;;
    *)
        echo "Error: Invalid algorithm: $ALGORITHM (supported: dapo, grpo, gspo, drgrpo)"
        exit 1
        ;;
esac

if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="ps_pipeline_${ALGORITHM}_$(date +%Y%m%d_%H%M%S)"
fi

# Default reward manager: pick a sensible default per algorithm if not
# explicitly specified.
if [ -z "$REWARD_MANAGER" ]; then
    REWARD_MANAGER="apiprimedapopspipelinerubric"
fi

ACTOR_PPO_MAX_TOKEN_LEN=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) / SP))
INFER_PPO_MAX_TOKEN_LEN=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) / SP))

case $ALGORITHM in
    dapo)  ADV_ESTIMATOR="grpo" ;;
    grpo)  ADV_ESTIMATOR="grpo"; ENABLE_FILTER_GROUPS="False" ;;
    gspo)  ADV_ESTIMATOR="grpo"; ENABLE_FILTER_GROUPS="False"; POLICY_LOSS_MODE="gspo"; LOSS_AGG_MODE="seq-mean-token-mean" ;;
    drgrpo) ADV_ESTIMATOR="grpo"; ENABLE_FILTER_GROUPS="False"; NORM_ADV_BY_STD="False"; LOSS_AGG_MODE="seq-mean-token-sum-norm" ;;
esac

# ==============================================================================
# Print configuration
# ==============================================================================

echo ""
echo "=============================================="
echo "  PS-Pipeline RL Training"
echo "=============================================="
echo "Algorithm       : $ALGORITHM"
echo "Reward Manager  : $REWARD_MANAGER"
echo "Experiment      : $EXPERIMENT_NAME"
echo "Model Path      : $MODEL_PATH"
echo "Verl Repo Dir   : $VERL_REPO_DIR"
echo "Train Files     : $TRAIN_FILES"
echo "Val Files       : $VAL_FILES"
echo "Checkpoint Dir  : $CKPT_DIR"
echo "Cluster         : ${NNODES} node(s) x ${N_GPUS_PER_NODE} GPU(s)"
echo "Context         : prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH} model_len=${MAX_MODEL_LEN}"
echo "Batch           : train_bsz=${TRAIN_BATCH_SIZE} n_resp=${N_RESP_PER_PROMPT} mini_bsz=${MINI_BATCH_SIZE}"
echo "Multi-turn Tool : enable=${ENABLE_MULTI_TURN} max_turns=${MAX_ASSISTANT_TURNS} tool_config=${TOOL_CONFIG}"
echo "=============================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    DRY_RUN_ONLY=true
else
    DRY_RUN_ONLY=false
fi

# ==============================================================================
# Start a local Ray cluster (single-node scenario)
# ==============================================================================

if [ "$AUTO_START_RAY" = "true" ] && [ "$DRY_RUN_ONLY" = false ]; then
    if ! ray status >/dev/null 2>&1; then
        echo "[Ray] No running local Ray cluster detected, starting one (ray start --head)..."
        ray start --head --dashboard-host=0.0.0.0
    else
        echo "[Ray] A running Ray cluster was detected, skipping ray start --head"
    fi
fi

mkdir -p "${LOG_DIR}" "${ROLLOUT_DATA_DIR}" "${CKPT_DIR}"
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
CURRENT_LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${CURRENT_TIME}.log"
echo "Log file: ${CURRENT_LOG_FILE}"

# ==============================================================================
# Assemble the training command
# ==============================================================================

COMMON_PARAMS="data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=False \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001"

MODEL_PARAMS="actor_rollout_ref.model.path=\"${MODEL_PATH}\" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.nccl_timeout=7200"

ACTOR_PARAMS="actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP_RATIO} \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP} \
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.entropy_checkpointing=True"

# Note: if you hit an "illegal memory access" during SGLang's
# release_memory_occupation phase, try lowering gpu_memory_utilization /
# max_num_batched_tokens, or set the environment variable SGLANG_SAFE_MODE=1
# (see the "Troubleshooting" section in README.md).
SGLANG_EXTRA_PARAMS=""
if [ "${SGLANG_SAFE_MODE:-0}" = "1" ] || [ "${SGLANG_DISABLE_CUDA_GRAPH:-0}" = "1" ]; then
    SGLANG_EXTRA_PARAMS="${SGLANG_EXTRA_PARAMS} +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=True"
fi
if [ "${SGLANG_SAFE_MODE:-0}" = "1" ] && [ -z "${SGLANG_MEM_FRACTION:-}" ]; then
    SGLANG_MEM_FRACTION=0.65
fi
if [ -n "${SGLANG_MEM_FRACTION:-}" ]; then
    SGLANG_EXTRA_PARAMS="${SGLANG_EXTRA_PARAMS} +actor_rollout_ref.rollout.engine_kwargs.sglang.mem_fraction_static=${SGLANG_MEM_FRACTION}"
fi
EFFECTIVE_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}

ROLLOUT_PARAMS="actor_rollout_ref.rollout.over_sample_rate=${OVER_SAMPLE_RATE} \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_k=${TOP_K} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTIL} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${EFFECTIVE_MAX_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.name=sglang \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.allow_auto_truncate=True \
    ${SGLANG_EXTRA_PARAMS}"

if [ "$ENABLE_MULTI_TURN" = "True" ]; then
    MULTITURN_PARAMS="actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_USER_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=\"left\" \
    actor_rollout_ref.rollout.multi_turn.format=qwen \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=\"${TOOL_CONFIG}\" \
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode='disable'"
else
    MULTITURN_PARAMS=""
fi

REF_PARAMS="actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP} \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN}"

REWARD_PARAMS_BASE="reward_model.reward_manager=${REWARD_MANAGER} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=True \
    +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}"

TRAINER_PARAMS="trainer.val_before_train=False \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='${PROJECT_NAME}' \
    trainer.experiment_name=\"${EXPERIMENT_NAME}\" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.resume_mode=auto \
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR} \
    trainer.default_local_dir=\"${CKPT_DIR}\" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.critic_warmup=0"

case $ALGORITHM in
    dapo)
        ALGO_SPECIFIC_PARAMS="actor_rollout_ref.hybrid_engine=True \
        algorithm.filter_groups.enable=${ENABLE_FILTER_GROUPS} \
        algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES} \
        algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC} \
        actor_rollout_ref.actor.loss_agg_mode=\"token-mean\" \
        data.gen_batch_size=$((TRAIN_BATCH_SIZE * 2))"

        TRAINING_CMD="ray job submit --address=\"${RAY_ADDRESS}\" \
            --working-dir \"${VERL_REPO_DIR}\" \
            -- \
            python3 -m recipe.dapo.main_dapo \
            ${COMMON_PARAMS} \
            ${ALGO_SPECIFIC_PARAMS} \
            ${MODEL_PARAMS} \
            ${ACTOR_PARAMS} \
            ${ROLLOUT_PARAMS} \
            ${MULTITURN_PARAMS} \
            ${REF_PARAMS} \
            ${REWARD_PARAMS_BASE} \
            ${TRAINER_PARAMS}"
        ;;

    grpo)
        ALGO_SPECIFIC_PARAMS="actor_rollout_ref.hybrid_engine=True \
        actor_rollout_ref.actor.loss_agg_mode=\"token-mean\""

        TRAINING_CMD="ray job submit --address=\"${RAY_ADDRESS}\" \
            --working-dir \"${VERL_REPO_DIR}\" \
            -- \
            python3 -m verl.trainer.main_ppo \
            ${COMMON_PARAMS} \
            ${ALGO_SPECIFIC_PARAMS} \
            ${MODEL_PARAMS} \
            ${ACTOR_PARAMS} \
            ${ROLLOUT_PARAMS} \
            ${MULTITURN_PARAMS} \
            ${REF_PARAMS} \
            ${REWARD_PARAMS_BASE} \
            ${TRAINER_PARAMS}"
        ;;

    gspo)
        ALGO_SPECIFIC_PARAMS="actor_rollout_ref.hybrid_engine=True \
        actor_rollout_ref.actor.policy_loss.loss_mode=${POLICY_LOSS_MODE} \
        actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}"

        TRAINING_CMD="ray job submit --address=\"${RAY_ADDRESS}\" \
            --working-dir \"${VERL_REPO_DIR}\" \
            -- \
            python3 -m verl.trainer.main_ppo \
            ${COMMON_PARAMS} \
            ${ALGO_SPECIFIC_PARAMS} \
            ${MODEL_PARAMS} \
            ${ACTOR_PARAMS} \
            ${ROLLOUT_PARAMS} \
            ${MULTITURN_PARAMS} \
            ${REF_PARAMS} \
            ${REWARD_PARAMS_BASE} \
            ${TRAINER_PARAMS}"
        ;;

    drgrpo)
        ALGO_SPECIFIC_PARAMS="actor_rollout_ref.hybrid_engine=True \
        algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD} \
        actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}"

        TRAINING_CMD="ray job submit --address=\"${RAY_ADDRESS}\" \
            --working-dir \"${VERL_REPO_DIR}\" \
            -- \
            python3 -m verl.trainer.main_ppo \
            ${COMMON_PARAMS} \
            ${ALGO_SPECIFIC_PARAMS} \
            ${MODEL_PARAMS} \
            ${ACTOR_PARAMS} \
            ${ROLLOUT_PARAMS} \
            ${MULTITURN_PARAMS} \
            ${REF_PARAMS} \
            ${REWARD_PARAMS_BASE} \
            ${TRAINER_PARAMS}"
        ;;
esac

echo "Final training command:"
echo "$TRAINING_CMD"
echo ""

if [ "$DRY_RUN_ONLY" = true ]; then
    echo "[Dry Run] Training was not actually executed."
    exit 0
fi

eval $TRAINING_CMD 2>&1 | tee "${CURRENT_LOG_FILE}"
exit_code=${PIPESTATUS[0]:-$?}

echo ""
echo "=============================================="
echo "Training finished, exit code: ${exit_code}"
echo "Checkpoint directory: ${CKPT_DIR}"
echo "=============================================="

exit ${exit_code}
