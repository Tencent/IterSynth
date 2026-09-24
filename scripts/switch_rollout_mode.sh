#!/bin/bash
# ==============================================================================
# switch_rollout_mode.sh -- switch the SGLang rollout implementation mode
# ==============================================================================
# verl's fsdp_workers.py / megatron_workers.py always import the
# SGLangRollout class from a single fixed module path:
#   verl.workers.rollout.sglang_rollout.sglang_rollout
#
# This project ships two different implementations under that same file
# name:
#
#   sglang_rollout.py     - Single-agent ReAct search mode (active by
#                            default). Pairs with reward managers:
#                              apiprimedapotrajectoryv2cgrpo (C-GRPO, recommended)
#                              apiprimedapotrajectoryv1naive
#                              apiprimedapoendtoendonly
#
#   sglang_rollout_lf.py   - Planner-Synthesizer dual-agent pipeline mode
#                            (alternates rollout between a Planner that
#                            produces the search plan and a Synthesizer
#                            that synthesizes evidence). Pairs with reward
#                            managers:
#                              apiprimedapopspipelinerubric (PS-Pipeline + Rubric, recommended)
#                              apiprimedapopspipeline (PS-Pipeline baseline)
#
# Since both files define a class with the same name
# (`class SGLangRollout(BaseRollout)`), only one can be active at a time.
# Use this script to switch between the two.
#
# Usage:
#   bash scripts/switch_rollout_mode.sh react       # switch to single-agent ReAct mode
#   bash scripts/switch_rollout_mode.sh ps_pipeline  # switch to PS-Pipeline dual-agent mode
#
# Note: this script operates on the files under patch/verl/. If you have
# already run apply_patch.sh to apply the patch to a separate verl repo,
# re-run apply_patch.sh after switching modes to sync the new
# sglang_rollout.py over.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROLLOUT_DIR="${RELEASE_ROOT}/patch/verl/workers/rollout/sglang_rollout"

MODE="$1"

if [ -z "$MODE" ]; then
    echo "Usage: bash switch_rollout_mode.sh <react|ps_pipeline>"
    exit 1
fi

case "$MODE" in
    react)
        SRC="${ROLLOUT_DIR}/sglang_rollout_react.py.bak"
        if [ -f "$SRC" ]; then
            cp -v "$SRC" "${ROLLOUT_DIR}/sglang_rollout.py"
        else
            echo "Note: no backed-up react version found; assuming sglang_rollout.py is already in react mode (the default on first checkout). Nothing to do."
        fi
        echo "Switched to: single-agent ReAct search mode"
        echo "Recommended reward_manager: apiprimedapotrajectoryv2cgrpo"
        ;;
    ps_pipeline)
        # Back up the current (react) version so it can be restored later
        if [ ! -f "${ROLLOUT_DIR}/sglang_rollout_react.py.bak" ]; then
            cp -v "${ROLLOUT_DIR}/sglang_rollout.py" "${ROLLOUT_DIR}/sglang_rollout_react.py.bak"
        fi
        cp -v "${ROLLOUT_DIR}/sglang_rollout_lf.py" "${ROLLOUT_DIR}/sglang_rollout.py"
        echo "Switched to: Planner-Synthesizer dual-agent pipeline mode"
        echo "Recommended reward_manager: apiprimedapopspipelinerubric"
        ;;
    *)
        echo "Error: unknown mode '$MODE', supported values: react | ps_pipeline"
        exit 1
        ;;
esac

echo ""
echo "If you have already run apply_patch.sh, re-run it now to sync to \$VERL_REPO_DIR:"
echo "  bash scripts/apply_patch.sh \$VERL_REPO_DIR"
