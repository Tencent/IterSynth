#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS Pipeline Rubric Reward Manager

Builds on the base PS Pipeline Reward Manager (api_prime_dapo_ps_pipeline.py)
by adding a rubric-based, per-round process reward.

Core mechanism:
- Rubric Reward computation:
  For every normal trajectory (correct / incorrect), each round is scored
  using the Planner/Synthesizer rubric, producing a rubric score in
  [-1, 1] per round.
  **Trajectory-level aggregation**: per-role, per-round scores within the
  same trajectory are aggregated into two scalars
  (R_rubric_planner_traj, R_rubric_synth_traj), which are then broadcast
  back to every sample according to its role.
  Motivation: eliminates the "short trajectories get an automatic high
  score" loophole, and aligns naturally with GRPO's uid-by-role grouping.
- Final reward formula (controlled by RUBRIC_REWARD_MODE):
  - "combined" mode (default, scheme A + trajectory aggregation + group-centering):
    - When correct (R_o = 1): R = R_o + alpha * (R_rubric_traj - mean_{correct in group}(R_rubric_traj))
    - When incorrect (R_o = 0): R = 0   # identical to the baseline, guarantees a floor
    - R_o: Outcome Reward (0 or 1, judged by the browsecomp_zh LLM Judge), broadcast to every round in the trajectory
    - R_rubric_traj: the aggregated rubric score (mean / median) for the corresponding role across the whole trajectory
    - Group-centering: subtract the mean R_rubric_traj of correct samples within the same group (same prompt + same role)
      Motivation: forces the rubric to strictly degrade into a tie-breaker, immune to absolute drift in the judge's scoring.
    - alpha: bonus strength (default 0.5, configurable via env RUBRIC_ALPHA)
    - Aggregation method: env RUBRIC_TRAJ_AGG in {mean, median}, default mean
    - Group-centering toggle: env RUBRIC_GROUP_CENTER (enabled by default)
    Design motivation:
    - Avoids the failure mode of "(1-alpha)*R_o + alpha*R_r", where the model
      could learn to optimize for "wrong answer but high rubric" trajectories
    - Only uses the rubric for quality fine-tuning (a tie-breaker) when the
      trajectory is correct, preserving GRPO's within-group discrimination signal

  - "rubric_only" mode (ablation, env RUBRIC_REWARD_MODE=rubric_only):
    - Abnormal rounds still get R = 0 (a data-filtering concern)
    - Other rounds: R = R_rubric_traj (in [-1, +1], per-role trajectory-aggregated score)
    - Does not use the outcome reward at all; alpha and group_center are ignored in this mode
    - Used to isolate the experiment "can the model learn a signal from the rubric alone"

- UID rewrite: concatenates a role-type suffix onto the uid, so GRPO
  normalizes within groups of (prompt, role type). Two groups: Planner / Synthesizer.

- Error-handling policy:
  - Discarded (excluded from training): search_error / tool_parse_error / exceed_max_tokens / exceed_max_turns
  - Offending round zeroed: excessive_tool_calls (trajectory kept, only that round's reward is zeroed, other rounds proceed normally)
  - Handled normally: repeated_query
  - Search returns 0 results: the model sees a "No results found" hint and continues reasoning, following the normal correct/incorrect flow
  - No answer / no ground_truth: reward=0, rubric scoring is skipped

- Configurable judge model:
  Set via the RUBRIC_JUDGE_MODEL environment variable (default gemini-2.5-flash-lite)
  Set API credentials via the RUBRIC_API_KEY / RUBRIC_BASE_URL environment variables
"""

import asyncio
import json
import os
import re
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Callable, Optional, List, Dict, Any, Tuple

import torch
import numpy as np
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score.browsecomp_zh import compute_score as browsecomp_compute_score
from verl.utils.reward_score.trajectory_reward_v2_cgrpo import classify_abnormal_trajectory
from verl.utils.reward_score.ps_rubric_reward import (
    PSRubricRewardCalculator,
    build_trajectory_rounds_from_samples,
)
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Group constants
# ============================================================

GROUP_PLANNER = "planner"             # Planner (includes search and answer rounds)
GROUP_SYNTHESIZER = "synthesizer"     # Synthesizer, updates the summary


def classify_sample_group(role: str, action_type: str) -> str:
    """Classify which group a sample belongs to, based on its role.

    Args:
        role: "planner" or "synthesizer"
        action_type: "tool_call" (search) or "answer" (final answer) -- no longer used for grouping

    Returns:
        The group name: GROUP_PLANNER / GROUP_SYNTHESIZER
    """
    if role == "synthesizer":
        return GROUP_SYNTHESIZER
    else:
        # planner and unknown roles are both grouped under planner
        return GROUP_PLANNER



def extract_answer_from_text(text: str) -> Optional[str]:
    """Extract the content inside <answer></answer> tags from text."""
    if not text or not isinstance(text, str):
        return None

    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

    if matches:
        answer = matches[-1].strip()
        return answer if answer else None

    return None


async def single_compute_ps_end_to_end_score(
    sequences_str: str,
    ground_truth: Dict[str, Any],
    task_extra_info: Dict[str, Any],
    executor: ThreadPoolExecutor,
    timeout: float = 6000.0,
) -> float:
    """Compute the end-to-end reward for a single sample (browsecomp_zh LLM Judge)."""
    loop = asyncio.get_running_loop()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                partial(
                    browsecomp_compute_score,
                    sequences_str,
                    ground_truth,
                    task_extra_info,
                ),
            ),
            timeout=timeout,
        )
        return float(result) if result is not None else 0.0
    except asyncio.TimeoutError:
        logger.warning("PS Pipeline end-to-end reward computation timed out")
        return 0.0
    except Exception as e:
        logger.error(f"PS Pipeline end-to-end reward computation failed: {e}")
        logger.error(traceback.format_exc())
        return 0.0


async def parallel_compute_ps_end_to_end_score_async(
    sequences_strs: List[str],
    ground_truths: List[Dict[str, Any]],
    task_extra_infos: List[Dict[str, Any]],
    num_processes: int = 64,
) -> List[float]:
    """Compute end-to-end rewards concurrently for a batch."""
    results = []
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        tasks_async = [
            single_compute_ps_end_to_end_score(
                sequences_str, ground_truth, task_extra_info,
                executor, timeout=6000.0
            )
            for sequences_str, ground_truth, task_extra_info in zip(
                sequences_strs, ground_truths, task_extra_infos
            )
        ]
        results = await asyncio.gather(*tasks_async, return_exceptions=True)

    processed_results = []
    for result in results:
        if isinstance(result, (Exception, BaseException)) or result is None:
            processed_results.append(0.0)
        else:
            processed_results.append(float(result))

    return processed_results


@register("apiprimedapopspipelinerubric")
class ApiPrimeDapoPSPipelineRubricRewardManager(AbstractRewardManager):
    """
    PS Pipeline Rubric Reward Manager

    Reward formula (scheme A: bonus-only):
    - R_outcome = 1: R = R_o + alpha * R_rubric   # rubric acts as a bonus when correct
    - R_outcome = 0: R = 0                         # degrades to baseline when incorrect, rubric has no effect
    - R_outcome: answer correctness (0/1, browsecomp_zh LLM Judge), broadcast to every round in the trajectory
    - R_rubric: per-round rubric score (range [-1, 1], scored by an external LLM against the rubric dimensions)
    - alpha: mixing coefficient (default 0.5)

    UID rewrite: concatenates the role type (Planner / Synthesizer) onto
    the uid, so GRPO normalizes within groups of (prompt, role type).

    Rubric scoring scope:
    - Normal trajectories (correct / incorrect): scored with the rubric
    - Other abnormalities/errors (including exceed_max_turns): not scored with the rubric; reward is set directly based on the error type
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        max_resp_len: Optional[int] = None,
        overlong_buffer_cfg: Optional[Any] = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
        )
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        self.config = {
            "num_workers": 64,
        }

        if self.overlong_buffer_cfg is not None:
            assert (
                self.max_resp_len is not None
            ), f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            assert (
                self.max_resp_len >= self.overlong_buffer_cfg.len
            ), "max_resp_len must be larger than overlong_buffer.len"

        # ---- Rubric Reward initialization ----
        self.rubric_enabled = True
        # After centering, alpha=0.5 is effectively equivalent to
        # "alpha ~= 0.25 under the old (uncentered) formula", because the
        # variance of (R_r - mean_correct) is roughly 60% narrower than R_r itself.
        self.rubric_alpha = float(os.environ.get("RUBRIC_ALPHA", "0.7"))
        # Trajectory-level aggregation method: mean | median (aggregates a
        # trajectory's per-round scores separately for each role)
        self.rubric_traj_agg = os.environ.get("RUBRIC_TRAJ_AGG", "mean").lower()
        # Whether to enable "within-group centering" -- subtract the mean
        # rubric score of correct samples in the same group (same
        # prompt+role) from R_rubric, forcing the rubric to strictly
        # degrade into a tie-breaker, immune to absolute drift in the judge's scoring
        self.rubric_group_center = os.environ.get("RUBRIC_GROUP_CENTER", "1").lower() not in ("0", "false", "")
        # Ablation toggle: whether to split the uid by role (default 1 =
        # split; 0 = ablation, uses the original prompt-uid grouping, with
        # planner / synthesizer mixed into the same GRPO normalization group).
        # When disabled, the "group-centering" group key also degrades to
        # use only the uid (no longer by role), to keep the ablated
        # variable isolated (only removing the "group by role" dimension).
        self.rubric_uid_rewrite_by_role = (
            os.environ.get("RUBRIC_UID_REWRITE_BY_ROLE", "1").lower() not in ("0", "false", "")
        )
        # Reward formula mode (for ablation):
        #   "combined" (default): R = R_o + alpha * (R_rubric_traj - mean_correct_in_group) if R_o > 0.5 else 0
        #   "rubric_only":        R = R_rubric_traj (after per-role aggregation; ignores R_o, alpha, group_center)
        #                         Pure ablation of the "outcome reward" -- each round uses only the rubric score.
        #                         Abnormal rounds are still zeroed (a data-filtering concern).
        self.rubric_reward_mode = os.environ.get("RUBRIC_REWARD_MODE", "combined").lower()
        if self.rubric_reward_mode not in ("combined", "rubric_only"):
            logger.warning(
                f"[Rubric Init] Unknown RUBRIC_REWARD_MODE={self.rubric_reward_mode!r}, falling back to combined"
            )
            self.rubric_reward_mode = "combined"
        self.rubric_calculator = None

        # Judge model configuration (overridable via environment variables;
        # do not hardcode LLM Judge credentials)
        rubric_judge_model = os.environ.get("RUBRIC_JUDGE_MODEL", "gemini-2.5-flash-lite")
        rubric_api_key = os.environ.get("RUBRIC_API_KEY") or os.environ.get("LLM_JUDGE_API_KEY", "")
        rubric_base_url = os.environ.get(
            "RUBRIC_BASE_URL",
            "https://api.openai.com/v1",
        )

        planner_rubric_path = os.environ.get(
            "PS_PLANNER_RUBRIC_PATH",
            "./rubrics/planner_rubric.json",
        )
        synthesizer_rubric_path = os.environ.get(
            "PS_SYNTHESIZER_RUBRIC_PATH",
            "./rubrics/synthesizer_rubric.json",
        )

        try:
            rubric_max_concurrent = int(os.environ.get("RUBRIC_MAX_CONCURRENT_CALLS", "16"))
            rubric_max_qpm = int(os.environ.get("RUBRIC_MAX_QPM", "200"))
            if os.path.exists(planner_rubric_path) and os.path.exists(synthesizer_rubric_path):
                self.rubric_calculator = PSRubricRewardCalculator(
                    planner_rubric_path=planner_rubric_path,
                    synthesizer_rubric_path=synthesizer_rubric_path,
                    judge_model=rubric_judge_model,
                    api_key=rubric_api_key,
                    base_url=rubric_base_url,
                    max_concurrent_calls=rubric_max_concurrent,
                    max_qpm=rubric_max_qpm,
                )
                logger.info(
                    f"[OK] Rubric Reward enabled: "
                    f"planner={planner_rubric_path}, "
                    f"synthesizer={synthesizer_rubric_path}, "
                    f"alpha={self.rubric_alpha}, "
                    f"judge_model={rubric_judge_model}, "
                    f"base_url={rubric_base_url}"
                )
            else:
                self.rubric_enabled = False
                logger.warning(
                    f"Rubric Reward disabled: rubric file(s) not found "
                    f"(planner={planner_rubric_path}, synthesizer={synthesizer_rubric_path})"
                )
        except Exception as e:
            self.rubric_enabled = False
            logger.warning(f"Rubric Reward initialization failed, disabled: {e}")

        logger.info(
            f"[OK] ApiPrimeDapoPSPipelineRubricRewardManager initialized "
            f"[reward_mode={self.rubric_reward_mode}, alpha={self.rubric_alpha}, "
            f"traj_agg={self.rubric_traj_agg}, group_center={self.rubric_group_center}, "
            f"uid_by_role={self.rubric_uid_rewrite_by_role}]"
        )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute the PS Pipeline reward tensor (including Rubric Reward + UID rewrite).

        Processing flow:
        1. Extract rollout_extra_info and request_id
        2. Group by request_id -- each group is one complete trajectory
        3. For each trajectory: check for abnormalities -> extract the answer -> collect trajectories to verify
        4. Batch, concurrently call the LLM Judge to verify answers -> Outcome Reward (R_o)
        5. Batch, concurrently score with the rubric -> Rubric Reward (R_rubric per-round)
        6. Scheme A mixing: when R_o=1, R = R_o + alpha*R_rubric; when R_o=0, R = 0
        7. Overwrite uid so GRPO groups by role type
        8. Fill extra_info, write JSONL logs
        """
        # If rm_scores already exists, return it directly
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)

        batch_size = len(data)

        # ========================================
        # === Step 1: extract rollout_extra_info and request_id ===
        # ========================================
        rollout_extra_info_list = data.non_tensor_batch.get("rollout_extra_info", [{}] * batch_size)
        request_id_list = data.non_tensor_batch.get("request_id", [None] * batch_size)

        # Decode responses
        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # ========================================
        # === Step 2: group by request_id ===
        # ========================================
        trajectory_groups = defaultdict(list)
        for i in range(batch_size):
            req_id = request_id_list[i]
            if req_id is None:
                req_id = f"__standalone_{i}"
            trajectory_groups[req_id].append(i)

        logger.info(
            f"PS Pipeline Rubric Reward: batch_size={batch_size}, "
            f"num_trajectories={len(trajectory_groups)}"
        )

        # ========================================
        # === Step 3: extract ground_truth and question ===
        # ========================================
        reward_model_list = data.non_tensor_batch.get("reward_model", None)
        ground_truth_top = data.non_tensor_batch.get("ground_truth", None)

        ground_truth_per_sample = []
        for i in range(batch_size):
            gt = None
            if reward_model_list is not None:
                try:
                    rm = reward_model_list[i]
                    if isinstance(rm, dict):
                        gt = rm.get("ground_truth", None)
                except (IndexError, TypeError, KeyError):
                    pass
            if gt is None and ground_truth_top is not None:
                try:
                    gt = ground_truth_top[i]
                except (IndexError, TypeError):
                    pass
            ground_truth_per_sample.append(gt)

        gt_missing_count = sum(1 for gt in ground_truth_per_sample if gt is None)
        if gt_missing_count > 0:
            logger.warning(
                f"[PSPipeline Rubric] {gt_missing_count}/{batch_size} samples are missing ground_truth!"
                f" non_tensor_batch keys: {list(data.non_tensor_batch.keys())}"
            )

        question_ntb = data.non_tensor_batch.get("question", None)
        extra_info_ntb = data.non_tensor_batch.get("extra_info", None)
        messages_ntb = data.non_tensor_batch.get("messages", None)

        question_list = []
        missing_question_count = 0
        for i in range(batch_size):
            question = None

            if question is None and question_ntb is not None:
                try:
                    q = question_ntb[i]
                    if q is not None and isinstance(q, str) and q.strip():
                        question = q
                except (IndexError, TypeError):
                    pass

            if question is None and extra_info_ntb is not None:
                try:
                    ei = extra_info_ntb[i]
                    if isinstance(ei, dict):
                        q = ei.get("question", None)
                        if q is not None and isinstance(q, str) and q.strip():
                            question = q
                except (IndexError, TypeError):
                    pass

            if question is None and reward_model_list is not None:
                try:
                    rm = reward_model_list[i]
                    if isinstance(rm, dict):
                        q = rm.get("question", None)
                        if q is not None and isinstance(q, str) and q.strip():
                            question = q
                except (IndexError, TypeError):
                    pass

            if question is None and messages_ntb is not None:
                try:
                    msg_data = messages_ntb[i]
                    msgs = None
                    if isinstance(msg_data, dict):
                        msgs = msg_data.get("messages", None)
                    elif isinstance(msg_data, list):
                        msgs = msg_data

                    if msgs is not None:
                        for msg in msgs:
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                content = msg.get("content", "")
                                if isinstance(content, str) and content.strip():
                                    question = content.strip()
                                    break
                except (IndexError, TypeError):
                    pass

            if question is None:
                missing_question_count += 1

            question_list.append(question)

        if missing_question_count > 0:
            logger.warning(
                f"[PSPipeline Rubric] {missing_question_count}/{batch_size} samples are missing question!"
                f" non_tensor_batch keys: {list(data.non_tensor_batch.keys())}"
            )

        # ========================================
        # === Step 4: process each trajectory, collecting the ones that need LLM Judge verification ===
        # ========================================
        #
        # Summary of error-handling policy:
        #
        # +-----------------------------------------+-----------+-----------+------------+
        # | Case                                     | Reward    | Rubric    | Trained?   |
        # +-----------------------------------------+-----------+-----------+------------+
        # | search_error (API retries exhausted, env)| 0         | no        | discarded  |
        # | exceed_max_tokens (truncated)             | 0         | no        | discarded  |
        # | exceed_max_turns (too many turns)         | 0         | no        | included   |
        # | tool_parse_error (JSON parse failure)     | 0         | no        | included   |
        # | excessive_tool_calls (over limit in 1 turn)| offending round | *scored* | included |
        # |   trajectory kept, only offending round's | zeroed    |           |            |
        # |   reward is zeroed                        |           |           |            |
        # | repeated_query (duplicate query)          | normal    | *scored*  | included   |
        # | no_answer_extracted (no answer found)     | 0         | no        | included   |
        # | no_ground_truth (missing reference answer)| 0         | no        | included   |
        # | correct (correct answer)                  | 1         | *scored*  | included   |
        # | incorrect (wrong answer)                  | 0         | *scored*  | included   |
        # | exception (processing exception)          | 0         | no        | included   |
        # +-----------------------------------------+-----------+-----------+------------+
        #
        all_rewards = [0.0] * batch_size
        all_end_to_end_scores = [0.0] * batch_size
        all_extracted_answers = [None] * batch_size
        all_is_correct = [False] * batch_size
        all_abnormal_types = [[] for _ in range(batch_size)]
        all_reward_reason = [""] * batch_size
        all_rubric_scores = [0.0] * batch_size
        # Marks whether a sample was actually scored with the rubric (used
        # for accurate metric reporting, avoiding a legitimate score of 0
        # being mistaken for "not scored" and filtered out)
        all_rubric_scored = [0] * batch_size
        # Which group each sample belongs to
        all_sample_groups = [""] * batch_size
        already_print_data_sources = {}

        trajectories_to_verify = []
        # Trajectories that need rubric scoring (normal trajectories only: correct / incorrect)
        trajectories_for_rubric = []  # [(req_id, sample_indices), ...]
        # Set of exceed_max_turns trajectory ids (for stats)
        exceed_max_turns_req_ids = set()

        for req_id, sample_indices in trajectory_groups.items():
            try:
                first_idx = sample_indices[0]
                extra_info = rollout_extra_info_list[first_idx]
                abnormal_flags = extra_info.get("abnormal_flags", {}) if isinstance(extra_info, dict) else {}

                should_give_zero_reward, should_discard, abnormal_types = classify_abnormal_trajectory(abnormal_flags)

                if should_discard:
                    for idx in sample_indices:
                        all_abnormal_types[idx] = abnormal_types
                        all_reward_reason[idx] = f"discarded: {abnormal_types}"
                    logger.info(
                        f"[PS Pipeline Rubric] Trajectory {req_id} discarded: {abnormal_types}"
                    )
                    continue

                # excessive_tool_calls_per_turn: the trajectory is not
                # discarded; reward is computed normally afterward, and
                # only the offending round's reward is zeroed out when the
                # final reward is assigned (abnormal_flags["excessive_rounds"]
                # records which rounds exceeded the limit)

                if should_give_zero_reward:
                    # exceed_max_turns / tool_parse_error: reward=0, no rubric scoring, skip directly
                    for idx in sample_indices:
                        all_abnormal_types[idx] = abnormal_types
                        all_reward_reason[idx] = f"abnormal: {abnormal_types}"
                    continue

                # Find the P(answer) sample and extract the answer
                answer_text = None

                for idx in sample_indices:
                    ei = rollout_extra_info_list[idx]
                    if not isinstance(ei, dict):
                        continue
                    role = ei.get("role", "")
                    answer_field = ei.get("answer", None)
                    if role == "planner" and answer_field is not None and answer_field != "":
                        answer_text = answer_field

                if answer_text is None:
                    for idx in reversed(sample_indices):
                        ei = rollout_extra_info_list[idx]
                        if isinstance(ei, dict) and ei.get("role", "") == "planner":
                            resp_text = sequences_str[idx]
                            extracted = extract_answer_from_text(resp_text)
                            if extracted is not None:
                                answer_text = extracted
                                break

                if answer_text is None:
                    for idx in sample_indices:
                        all_reward_reason[idx] = "no_answer_extracted"
                    continue

                extracted_answer = answer_text.strip()
                for idx in sample_indices:
                    all_extracted_answers[idx] = extracted_answer

                gt = ground_truth_per_sample[first_idx]
                if gt is None:
                    for idx in sample_indices:
                        all_reward_reason[idx] = "no_ground_truth"
                    continue

                answer_sequences_str = f"<answer>{extracted_answer}</answer>"

                task_extra_info = {}
                if question_list[first_idx] is not None:
                    task_extra_info["question"] = question_list[first_idx]

                if task_extra_info.get("question", None) is None:
                    logger.warning(
                        f"Trajectory {req_id[:8]}... is missing the question field, "
                        f"browsecomp_zh.compute_score will return 0"
                    )

                trajectories_to_verify.append(
                    (req_id, sample_indices, answer_sequences_str, gt, task_extra_info)
                )

            except Exception as e:
                logger.error(f"Failed to process trajectory {req_id}: {e}")
                logger.error(traceback.format_exc())
                for idx in sample_indices:
                    all_rewards[idx] = 0.0
                    all_reward_reason[idx] = f"exception: {str(e)[:100]}"

        # ========================================
        # === Step 5: batch, concurrently call the LLM Judge to verify answers (Outcome Reward) ===
        # ========================================
        if len(trajectories_to_verify) > 0:
            verify_sequences = [t[2] for t in trajectories_to_verify]
            verify_ground_truths = [t[3] for t in trajectories_to_verify]
            verify_extra_infos = [t[4] for t in trajectories_to_verify]

            end_to_end_scores = self._verify_answers_batch(
                verify_sequences, verify_ground_truths, verify_extra_infos
            )

            for traj_idx, (req_id, sample_indices, _, _, _) in enumerate(trajectories_to_verify):
                score = end_to_end_scores[traj_idx]
                is_correct = score >= 1.0
                trajectory_reward = 1.0 if is_correct else 0.0

                # Get the list of offending rounds (for per-round zeroing)
                first_ei = rollout_extra_info_list[sample_indices[0]]
                excessive_rounds = set()
                if isinstance(first_ei, dict):
                    af = first_ei.get("abnormal_flags", {})
                    excessive_rounds = set(af.get("excessive_rounds", []))

                for idx in sample_indices:
                    ei = rollout_extra_info_list[idx]
                    sample_round = ei.get("round", -1) if isinstance(ei, dict) else -1

                    # Samples in an offending round: reward is zeroed
                    if sample_round in excessive_rounds:
                        sample_reward = 0.0
                        all_reward_reason[idx] = f"excessive_round_{sample_round}"
                    else:
                        sample_reward = trajectory_reward
                        all_reward_reason[idx] = "correct" if is_correct else "incorrect"

                    all_rewards[idx] = sample_reward
                    all_end_to_end_scores[idx] = score
                    all_is_correct[idx] = is_correct

                    valid_len = valid_response_length[idx].item()
                    if valid_len > 0:
                        reward_tensor[idx, valid_len - 1] = sample_reward

                # Normal trajectories (correct/incorrect) also need rubric scoring
                trajectories_for_rubric.append((req_id, sample_indices))

        # ========================================
        # === Step 5.5: Rubric Reward computation ===
        # ========================================
        #
        # Only the following trajectories are scored with the rubric:
        # 1. Normal trajectories (correct / incorrect)
        #
        # Not scored (skipped directly, rubric_score stays 0.0):
        # - search_error / tool_parse_error / exceed_max_tokens / exceed_max_turns (discarded)
        # - no_answer_extracted
        # - no_ground_truth
        # - exception
        #
        # Detect whether this is the validation set: no rubric scoring
        # needed for validation (only the outcome reward is needed to compute accuracy)
        is_validate = data.meta_info.get("validate", False) if hasattr(data, "meta_info") and data.meta_info else False

        if self.rubric_enabled and self.rubric_calculator is not None and trajectories_for_rubric and not is_validate:
            try:
                # ============================================
                # Decide whether to reuse precomputed scores at the
                # trajectory granularity:
                # - all samples in a trajectory have a precomputed_rubric_score -> the whole trajectory takes the "reuse" path
                # - any sample is missing one -> the whole trajectory takes the "recompute" path
                # (Why trajectory granularity rather than sample granularity:
                #  rubric scoring is independent per round, but reward
                #  aggregation/logging is done per trajectory, so mixing
                #  the two sources within the same trajectory would
                #  complicate trajectory-level stats and logging logic.
                #  Splitting by trajectory keeps scoring consistent on both
                #  ends and makes the code easier to reuse.)
                # ============================================
                cached_trajectories = []      # [(req_id, sample_indices), ...] all samples have a precomputed score
                recompute_trajectories = []   # [(req_id, sample_indices), ...] need to be rescored

                n_total_samples = 0
                n_precomputed_samples = 0
                for req_id, sample_indices in trajectories_for_rubric:
                    all_have_precomp = True
                    for idx in sample_indices:
                        n_total_samples += 1
                        ei = rollout_extra_info_list[idx]
                        if isinstance(ei, dict) and ei.get("precomputed_rubric_score") is not None:
                            n_precomputed_samples += 1
                        else:
                            all_have_precomp = False

                    if all_have_precomp:
                        cached_trajectories.append((req_id, sample_indices))
                    else:
                        recompute_trajectories.append((req_id, sample_indices))

                logger.info(
                    f"[Rubric Reward] precomputed hits: "
                    f"{n_precomputed_samples}/{n_total_samples} samples, "
                    f"{len(cached_trajectories)}/{len(trajectories_for_rubric)} trajectories reused; "
                    f"{len(recompute_trajectories)} trajectories need rescoring"
                )

                # ============================================
                # Step 1: batch-call the LLM Judge for the trajectories
                # that need rescoring
                # ============================================
                rubric_results = []           # results for recompute_trajectories only
                rubric_traj_mapping = []      # mapping for recompute_trajectories only (used for logging)

                if recompute_trajectories:
                    rubric_trajectories = []
                    for req_id, sample_indices in recompute_trajectories:
                        first_idx = sample_indices[0]
                        q = question_list[first_idx] if first_idx < len(question_list) else None
                        if not q:
                            continue
                        try:
                            trajectory_rounds = build_trajectory_rounds_from_samples(
                                sample_indices=sample_indices,
                                rollout_extra_info_list=rollout_extra_info_list,
                                sequences_str=sequences_str,
                                messages_list=messages_ntb,
                            )
                            if trajectory_rounds:
                                # GT extraction consistent with the rollout side: dict -> target; otherwise -> str
                                gt = ground_truth_per_sample[first_idx] if first_idx < len(ground_truth_per_sample) else None
                                gt_str = ""
                                if gt is not None:
                                    if isinstance(gt, dict):
                                        gt_str = str(gt.get("target", "") or "")
                                    else:
                                        gt_str = str(gt)
                                rubric_trajectories.append({
                                    "user_query": q,
                                    "ground_truth": gt_str,
                                    "rounds": trajectory_rounds,
                                })
                                rubric_traj_mapping.append((req_id, sample_indices))
                        except Exception as e:
                            logger.warning(
                                f"Trajectory {req_id[:8]}... failed to build rubric input: {e}"
                            )

                    if rubric_trajectories:
                        n_normal = sum(1 for r, _ in rubric_traj_mapping if r not in exceed_max_turns_req_ids)
                        n_exceed = sum(1 for r, _ in rubric_traj_mapping if r in exceed_max_turns_req_ids)
                        logger.info(
                            f"[Rubric Reward] Starting to score {len(rubric_trajectories)} trajectories with the rubric "
                            f"(normal={n_normal}, exceed_max_turns={n_exceed})..."
                        )
                        rubric_results = self.rubric_calculator.batch_compute_rubric_rewards(
                            rubric_trajectories
                        )

                # ============================================
                # Step 2: apply the reward formula uniformly (cached +
                # recomputed use the same formula)
                # ============================================
                #
                # Improvements (vs. the old per-round rubric):
                # (1) Trajectory-level aggregation: for each trajectory,
                #     per-round rubric scores are aggregated separately per
                #     role into a single scalar
                #     traj_role_rubric[(req_id, role)], then broadcast back
                #     to every sample. Eliminates the "short trajectories
                #     get an automatic high score" loophole, and aligns
                #     naturally with GRPO's uid-by-role grouping.
                # (2) Within-group standardization (optional, enabled by
                #     default): subtract the mean of correct samples within
                #     the same group ((orig_uid, role)) from R_rubric,
                #     forcing the rubric to strictly degrade into a
                #     tie-breaker, immune to absolute drift in the judge's scoring.
                # ============================================

                # ---- 2.0 collect the per-role rubric score list for each trajectory ----
                # traj_role_scores[(req_id, role)] = [per-round scores...]
                traj_role_scores: Dict[Tuple[str, str], List[float]] = defaultdict(list)
                # raw_per_sample_rubric[idx] = the raw rubric score of the round corresponding to this sample (for logging)
                raw_per_sample_rubric: Dict[int, float] = {}

                # 2.0a. cached trajectories: collect from precomputed_rubric_score
                for req_id, sample_indices in cached_trajectories:
                    for idx in sample_indices:
                        ei = rollout_extra_info_list[idx]
                        if not isinstance(ei, dict):
                            continue
                        rb = ei.get("precomputed_rubric_score")
                        if rb is None:
                            continue
                        role = ei.get("role", "") or "planner"
                        traj_role_scores[(req_id, role)].append(float(rb))
                        raw_per_sample_rubric[idx] = float(rb)

                # 2.0b. recomputed trajectories: collect from per_round_rewards
                #       per_round_rewards: dict[(role, round)] -> score
                for traj_idx, (req_id, sample_indices) in enumerate(rubric_traj_mapping):
                    result = rubric_results[traj_idx]
                    per_round_rewards = result.get("per_round_rewards", {})
                    trajectory_rubric_fallback = float(result.get("trajectory_rubric_reward", 0.0))

                    for idx in sample_indices:
                        ei = rollout_extra_info_list[idx]
                        if not isinstance(ei, dict):
                            continue
                        role = ei.get("role", "") or "planner"
                        rnd = ei.get("round", 0)
                        rb = per_round_rewards.get((role, rnd), trajectory_rubric_fallback)
                        try:
                            rb_f = float(rb)
                        except (TypeError, ValueError):
                            rb_f = trajectory_rubric_fallback
                        traj_role_scores[(req_id, role)].append(rb_f)
                        raw_per_sample_rubric[idx] = rb_f

                # ---- 2.1 aggregate each trajectory's per-role score (mean / median) ----
                def _aggregate(scores: List[float]) -> float:
                    if not scores:
                        return 0.0
                    if self.rubric_traj_agg == "median":
                        sorted_s = sorted(scores)
                        n = len(sorted_s)
                        return sorted_s[n // 2] if n % 2 == 1 else 0.5 * (sorted_s[n // 2 - 1] + sorted_s[n // 2])
                    return sum(scores) / len(scores)

                traj_role_rubric: Dict[Tuple[str, str], float] = {
                    key: _aggregate(scores) for key, scores in traj_role_scores.items()
                }

                # ---- 2.2 compute the trajectory-level rubric score for each sample ----
                # sample_traj_rubric[idx] = traj_role_rubric[(req_id, role_of_sample)]
                # Also collects the rubric scores of "correct samples" keyed
                # by group, for within-group centering.
                # The group key depends on self.rubric_uid_rewrite_by_role:
                #   - True (default): (orig_uid, role) -- aligns with the by-role GRPO grouping
                #   - False (ablation): (orig_uid, "_all_") -- corresponds to the "not split by role" GRPO grouping
                sample_traj_rubric: Dict[int, float] = {}
                group_correct_rubrics: Dict[Tuple[str, str], List[float]] = defaultdict(list)

                orig_uid_list = data.non_tensor_batch.get("uid", None)

                # Use a set for dedup: the same trajectory within the same
                # group should only contribute once to the group mean
                _counted_traj_role_in_group: set = set()

                def _group_key_of(orig_uid: str, role: str) -> Tuple[str, str]:
                    if self.rubric_uid_rewrite_by_role:
                        return (orig_uid, role)
                    return (orig_uid, "_all_")

                for req_id, sample_indices in trajectories_for_rubric:
                    for idx in sample_indices:
                        ei = rollout_extra_info_list[idx]
                        if not isinstance(ei, dict):
                            continue
                        role = ei.get("role", "") or "planner"
                        traj_rub = traj_role_rubric.get((req_id, role), 0.0)
                        sample_traj_rubric[idx] = traj_rub

                        # Only correct samples participate in the "correct group mean"
                        if all_is_correct[idx] and orig_uid_list is not None and idx < len(orig_uid_list):
                            orig_uid = str(orig_uid_list[idx])
                            group_key = _group_key_of(orig_uid, role)
                            # The dedup key still carries (req_id, role),
                            # because the planner-traj and synth-traj are
                            # two different trajectories (aggregated scores
                            # for different roles); even in ablation mode
                            # where they merge into the same group, each
                            # should still enter the group mean as an
                            # independent observation.
                            dedup_key = (req_id, role, group_key)
                            if dedup_key not in _counted_traj_role_in_group:
                                _counted_traj_role_in_group.add(dedup_key)
                                group_correct_rubrics[group_key].append(traj_rub)

                # ---- 2.3 compute the mean correct-rubric per group (the centering baseline) ----
                group_correct_rubric_mean: Dict[Tuple[str, str], float] = {
                    k: (sum(v) / len(v)) for k, v in group_correct_rubrics.items() if v
                }

                # ---- 2.4 apply to reward_tensor ----
                def _apply_rubric_to_sample(idx: int) -> None:
                    """Decide the reward formula based on self.rubric_reward_mode:

                    - "combined" (default):
                        R_o=0 -> R = 0 (baseline)
                        R_o=1 -> R = R_o + alpha * (R_rubric_traj - group_correct_mean)
                    - "rubric_only" (ablation): does not use the outcome signal
                        abnormal round -> R = 0
                        other rounds -> R = R_rubric_traj (in [-1, +1])
                    """
                    rb_traj = sample_traj_rubric.get(idx, 0.0)
                    all_rubric_scores[idx] = rb_traj  # the trajectory-level aggregated score, used for logging/stats
                    all_rubric_scored[idx] = 1

                    # Abnormal rounds were already flagged as
                    # excessive_round_* in step 5; both modes keep
                    # reward=0 for these, purely as data filtering
                    reason = all_reward_reason[idx] or ""
                    is_excessive_round = reason.startswith("excessive_round_")

                    if self.rubric_reward_mode == "rubric_only":
                        if is_excessive_round:
                            combined_reward = 0.0
                        else:
                            combined_reward = rb_traj
                    else:
                        # combined mode
                        R_o = all_rewards[idx]
                        if R_o > 0.5:
                            if self.rubric_group_center and orig_uid_list is not None and idx < len(orig_uid_list):
                                ei = rollout_extra_info_list[idx]
                                role = (ei.get("role", "") or "planner") if isinstance(ei, dict) else "planner"
                                orig_uid = str(orig_uid_list[idx])
                                mean_ref = group_correct_rubric_mean.get(_group_key_of(orig_uid, role), rb_traj)
                                centered = rb_traj - mean_ref
                            else:
                                centered = rb_traj
                            combined_reward = R_o + self.rubric_alpha * centered
                        else:
                            combined_reward = R_o

                    all_rewards[idx] = combined_reward
                    valid_len = valid_response_length[idx].item()
                    if valid_len > 0:
                        reward_tensor[idx, valid_len - 1] = combined_reward

                # Collector (used for the "scoring summary" log)
                cached_score_collector: List[float] = []

                # 2a. trajectories that reused a precomputed score
                for req_id, sample_indices in cached_trajectories:
                    for idx in sample_indices:
                        ei = rollout_extra_info_list[idx]
                        if not isinstance(ei, dict):
                            continue
                        if idx not in raw_per_sample_rubric:
                            # Fallback (should not happen in theory: on the
                            # cached path, the whole trajectory has a precomputed score)
                            sample_traj_rubric.setdefault(idx, 0.0)
                        cached_score_collector.append(raw_per_sample_rubric.get(idx, 0.0))
                        _apply_rubric_to_sample(idx)

                # 2b. trajectories that were rescored
                for traj_idx, (req_id, sample_indices) in enumerate(rubric_traj_mapping):
                    for idx in sample_indices:
                        ei = rollout_extra_info_list[idx]
                        if not isinstance(ei, dict):
                            continue
                        _apply_rubric_to_sample(idx)

                # ============================================
                # Step 3: scoring summary
                # ============================================
                # Trajectory-level aggregated score distribution (stats separately per role)
                planner_traj_scores = [v for (rid, role), v in traj_role_rubric.items() if role == "planner"]
                synth_traj_scores = [v for (rid, role), v in traj_role_rubric.items() if role == "synthesizer"]
                p_mean = (sum(planner_traj_scores) / len(planner_traj_scores)) if planner_traj_scores else 0.0
                s_mean = (sum(synth_traj_scores) / len(synth_traj_scores)) if synth_traj_scores else 0.0
                # Mean |R_r - mean| after group-centering (measures tie-breaker signal strength)
                centered_abs_sum, centered_n = 0.0, 0
                if self.rubric_group_center:
                    for idx, rb in sample_traj_rubric.items():
                        if not all_is_correct[idx]:
                            continue
                        ei = rollout_extra_info_list[idx]
                        if not isinstance(ei, dict):
                            continue
                        role = ei.get("role", "") or "planner"
                        if orig_uid_list is None or idx >= len(orig_uid_list):
                            continue
                        orig_uid = str(orig_uid_list[idx])
                        mean_ref = group_correct_rubric_mean.get(_group_key_of(orig_uid, role))
                        if mean_ref is None:
                            continue
                        centered_abs_sum += abs(rb - mean_ref)
                        centered_n += 1
                centered_abs_mean = (centered_abs_sum / centered_n) if centered_n else 0.0
                logger.info(
                    f"[Rubric Reward Summary] num_trajectories={len(trajectories_for_rubric)} "
                    f"(cached={len(cached_trajectories)}, recompute={len(rubric_results)}), "
                    f"mode={self.rubric_reward_mode}, "
                    f"agg={self.rubric_traj_agg}, "
                    f"planner_traj_mean={p_mean:.3f} (n={len(planner_traj_scores)}), "
                    f"synth_traj_mean={s_mean:.3f} (n={len(synth_traj_scores)}), "
                    f"group_center={'on' if self.rubric_group_center else 'off'}, "
                    f"uid_by_role={'on' if self.rubric_uid_rewrite_by_role else 'off (ABLATION)'}, "
                    f"|centered|_mean(correct)={centered_abs_mean:.3f}, "
                    f"alpha={self.rubric_alpha}"
                )

                # ========================================
                # === Write the rubric scoring detail JSONL log ===
                # ========================================
                # Note: this only records trajectories that were rescored
                # on the reward-manager side (rubric_traj_mapping), since
                # cached trajectories only have a precomputed scalar and
                # lack per-round rubric_scores detail. If you need detail
                # for cached trajectories, add logging on the rollout side.
                if rubric_traj_mapping and rubric_results:
                    try:
                        rubric_log_dir = os.environ.get(
                            "RUBRIC_LOG_DIR",
                            "./logs/rubric_logs"
                        )
                        os.makedirs(rubric_log_dir, exist_ok=True)

                        step = (
                            data.meta_info.get("global_steps", 0)
                            if hasattr(data, "meta_info") and data.meta_info
                            else 0
                        )
                        worker_rank = os.environ.get(
                            "RANK", os.environ.get("LOCAL_RANK", "0")
                        )
                        rubric_log_path = os.path.join(
                            rubric_log_dir,
                            f"rubric_step_{step}_worker_{worker_rank}.jsonl"
                        )

                        with open(rubric_log_path, "a", encoding="utf-8") as f_log:
                            for traj_idx, (req_id, sample_indices) in enumerate(rubric_traj_mapping):
                                result = rubric_results[traj_idx]
                                first_idx = sample_indices[0]

                                _uid_list = data.non_tensor_batch.get("uid", None)
                                traj_uid = (
                                    str(_uid_list[first_idx])
                                    if _uid_list is not None
                                    and first_idx < len(_uid_list)
                                    and _uid_list[first_idx] is not None
                                    else "N/A"
                                )

                                q = (
                                    question_list[first_idx]
                                    if first_idx < len(question_list) and question_list[first_idx]
                                    else None
                                )

                                # Build a lookup table for rubric_io_samples: (role, round) -> {prompt, llm_output, ...}
                                io_lookup = {}
                                for io_s in result.get("rubric_io_samples", []):
                                    key = (io_s.get("role", ""), io_s.get("round", 0))
                                    io_lookup[key] = io_s

                                # Sample rubric IO: log prompt/llm_output for the first few trajectories, skip the rest
                                MAX_RUBRIC_IO_TRAJECTORIES = 3

                                # Build the per-round rubric scoring detail
                                planner_round_details = []
                                for pr in result.get("planner_scores", []):
                                    round_detail = {
                                        "round": pr["round"],
                                        "role": "planner",
                                        "normalized_score": round(pr["normalized_score"], 4),
                                        "rubric_scores": {},
                                    }
                                    for rid, score_info in pr.get("scores", {}).items():
                                        round_detail["rubric_scores"][rid] = {
                                            "score": score_info.get("score", 0),
                                            "reason": score_info.get("reason", ""),
                                        }
                                    # Sample the prompt/llm_output for the first few trajectories
                                    if traj_idx < MAX_RUBRIC_IO_TRAJECTORIES:
                                        io_match = io_lookup.get(("planner", pr["round"]))
                                        if io_match:
                                            round_detail["prompt"] = io_match.get("prompt", "")[:3000]
                                            round_detail["llm_output"] = io_match.get("llm_output", "")[:2000]
                                    planner_round_details.append(round_detail)

                                synthesizer_round_details = []
                                for sr in result.get("synthesizer_scores", []):
                                    round_detail = {
                                        "round": sr["round"],
                                        "role": "synthesizer",
                                        "normalized_score": round(sr["normalized_score"], 4),
                                        "rubric_scores": {},
                                    }
                                    for rid, score_info in sr.get("scores", {}).items():
                                        round_detail["rubric_scores"][rid] = {
                                            "score": score_info.get("score", 0),
                                            "reason": score_info.get("reason", ""),
                                        }
                                    # Sample the prompt/llm_output for the first few trajectories
                                    if traj_idx < MAX_RUBRIC_IO_TRAJECTORIES:
                                        io_match = io_lookup.get(("synthesizer", sr["round"]))
                                        if io_match:
                                            round_detail["prompt"] = io_match.get("prompt", "")[:3000]
                                            round_detail["llm_output"] = io_match.get("llm_output", "")[:2000]
                                    synthesizer_round_details.append(round_detail)

                                # Merge and sort by round
                                all_round_details = sorted(
                                    planner_round_details + synthesizer_round_details,
                                    key=lambda x: (x["round"], 0 if x["role"] == "planner" else 1),
                                )

                                traj_log = {
                                    "timestamp": datetime.now().isoformat(),
                                    "step": step,
                                    "request_id": req_id,
                                    "uid": traj_uid,
                                    "question": str(q)[:300] if q else None,
                                    "is_correct": all_is_correct[first_idx],
                                    "outcome_reward": 1.0 if all_is_correct[first_idx] else 0.0,
                                    "trajectory_rubric_reward": round(
                                        result.get("trajectory_rubric_reward", 0.0), 4
                                    ),
                                    "planner_mean_score": round(
                                        result.get("planner_mean_score", 0.0), 4
                                    ),
                                    "synthesizer_mean_score": round(
                                        result.get("synthesizer_mean_score", 0.0), 4
                                    ),
                                    "n_planner_rounds": len(planner_round_details),
                                    "n_synthesizer_rounds": len(synthesizer_round_details),
                                    "rounds": all_round_details,
                                }
                                f_log.write(
                                    json.dumps(traj_log, ensure_ascii=False) + "\n"
                                )

                        logger.info(
                            f"[Rubric Log] Wrote rubric scoring detail for {len(rubric_traj_mapping)} trajectories -> {rubric_log_path}"
                        )
                    except Exception as log_e:
                        logger.warning(f"[Rubric Log] Failed to write rubric log: {log_e}")

            except Exception as e:
                logger.error(f"[Rubric Reward] Error during scoring, skipping rubric reward: {e}")
                logger.error(traceback.format_exc())

        # ========================================
        # === Step 6: sample grouping (for UID rewriting) ===
        # ========================================
        # Split all samples into two groups by role:
        # 1. Planner: planner (includes search rounds and answer rounds)
        # 2. Synthesizer: synthesizer (summary-update rounds)
        #
        group_indices = {
            GROUP_PLANNER: [],
            GROUP_SYNTHESIZER: [],
        }

        for idx in range(batch_size):
            ei = rollout_extra_info_list[idx]
            if not isinstance(ei, dict):
                all_sample_groups[idx] = GROUP_PLANNER
                group_indices[GROUP_PLANNER].append(idx)
                continue

            role = ei.get("role", "")
            group = classify_sample_group(role, "")
            all_sample_groups[idx] = group
            group_indices[group].append(idx)

        # ========================================
        # === Step 6.5: overwrite uid so GRPO groups by role type ===
        # ========================================
        #
        # The original uid is one UUID per prompt, shared by all round
        # samples under that prompt. GRPO advantage normalization does a
        # within-group z-score per uid. If the uid is left unmodified,
        # planner and synthesizer samples get normalized together, even
        # though their reward magnitude and meaning differ substantially
        # by role type.
        #
        # Default scheme: append the sample_group onto the uid, forming
        # "{uid}_{group}". This makes GRPO normalize within groups of
        # (prompt, role type), ensuring like-for-like comparisons.
        #
        # Ablation mode (RUBRIC_UID_REWRITE_BY_ROLE=0): keep the original
        # prompt-uid without the role suffix; planner / synthesizer samples
        # get mixed into the same GRPO normalization group.
        #
        # Example:
        #   Original: uid="abc123" -> planner sample uid="abc123"
        #   Default:  -> planner sample uid="abc123_planner"
        #             -> synthesizer sample uid="abc123_synthesizer"
        #   Ablation: -> uid stays "abc123" (planner / synth mixed grouping)
        #
        original_uid_list = data.non_tensor_batch.get("uid", None)
        if original_uid_list is not None:
            if self.rubric_uid_rewrite_by_role:
                new_uid_list = []
                for idx in range(batch_size):
                    orig_uid = str(original_uid_list[idx]) if idx < len(original_uid_list) else str(idx)
                    group = all_sample_groups[idx] if all_sample_groups[idx] else GROUP_PLANNER
                    new_uid_list.append(f"{orig_uid}_{group}")
                data.non_tensor_batch["uid"] = np.array(new_uid_list, dtype=object)

                new_uid_counts = Counter(new_uid_list)
                n_p = sum(1 for u in new_uid_counts if u.endswith(GROUP_PLANNER))
                n_s = sum(1 for u in new_uid_counts if u.endswith(GROUP_SYNTHESIZER))
                logger.info(
                    f"[UID Rewrite for GRPO Grouping] "
                    f"original uid group count={len(set(str(u) for u in original_uid_list))}, "
                    f"new uid group count={len(new_uid_counts)} "
                    f"({GROUP_PLANNER}={n_p}, "
                    f"{GROUP_SYNTHESIZER}={n_s})"
                )
            else:
                # Ablation: do not rewrite uid, GRPO groups by the original prompt-uid (planner+synth mixed)
                logger.info(
                    f"[UID Rewrite ABLATION: by-role disabled] "
                    f"keeping original uid group count={len(set(str(u) for u in original_uid_list))}, "
                    f"planner/synth mixed GRPO normalization"
                )

        # ========================================
        # === Step 7: print debug info (enhanced) ===
        # ========================================
        data_sources = data.non_tensor_batch.get("data_source", None)
        uid_list = data.non_tensor_batch.get("uid", [None] * batch_size)

        for req_id, sample_indices in trajectory_groups.items():
            first_idx = sample_indices[0]
            ds = "unknown"
            if data_sources is not None:
                try:
                    ds = data_sources[first_idx] if first_idx < len(data_sources) else "unknown"
                except (IndexError, TypeError):
                    ds = "unknown"

            if ds not in already_print_data_sources:
                already_print_data_sources[ds] = 0

            if already_print_data_sources[ds] < self.num_examine:
                already_print_data_sources[ds] += 1

                traj_uid = uid_list[first_idx] if first_idx < len(uid_list) else "N/A"

                logger.info(
                    f"[Trajectory {req_id[:8]}...] "
                    f"uid={str(traj_uid)[:8]}..., "
                    f"samples={len(sample_indices)}, "
                    f"reward={all_rewards[first_idx]:.3f}, "
                    f"end_to_end={all_end_to_end_scores[first_idx]:.3f}, "
                    f"rubric={all_rubric_scores[first_idx]:.3f}, "
                    f"reason={all_reward_reason[first_idx]}, "
                    f"answer={str(all_extracted_answers[first_idx])[:100]}"
                )

                roles = []
                for idx in sample_indices:
                    ei = rollout_extra_info_list[idx]
                    if isinstance(ei, dict):
                        role = ei.get("role", "?")
                        rnd = ei.get("round", "?")
                        r = all_rewards[idx]
                        rb = all_rubric_scores[idx]
                        grp = all_sample_groups[idx]
                        roles.append(
                            f"{role}(r{rnd},grp={grp},rew={r:.3f},rub={rb:.3f})"
                        )
                    else:
                        roles.append("?")
                logger.info(f"  Trajectory structure: {' -> '.join(roles)}")

        # GRPO grouping stats (used by downstream logging)
        uid_to_rewards = defaultdict(list)
        for i in range(batch_size):
            uid = uid_list[i] if i < len(uid_list) else None
            if uid is not None:
                uid_to_rewards[str(uid)].append(all_rewards[i])

        # ========================================
        # === Step 8: fill reward_extra_info ===
        # ========================================
        reward_extra_info['rubric_score'] = all_rubric_scores
        reward_extra_info['rubric_scored'] = all_rubric_scored  # 0/1 flag for whether it was scored
        reward_extra_info['sample_group'] = all_sample_groups
        # The acc field is used by process_validation_metrics for the
        # validation set (core_var = "acc" if "acc" in var2metric2val else "reward").
        # Without this field, val-core on the validation set would use the
        # mixed reward instead of the 0/1 accuracy.
        reward_extra_info['acc'] = [
            1.0 if c else 0.0 for c in all_is_correct
        ]
        reward_extra_info['reward_reason'] = all_reward_reason
        reward_extra_info['abnormal_types'] = all_abnormal_types

        # Trajectory-level stats
        # Note: these lists have length = number of trajectories (!= batch_size),
        # so they cannot go directly into non_tensor_batch (DataProto
        # requires all fields to have equal length). After being placed
        # into reward_extra_info, safe_convert_reward_extra_info_to_numpy
        # in ray_trainer pops them into meta_info beforehand.
        #
        # Compatibility warning:
        # - RayPPOTrainer: pops ps_trajectory_* into meta_info (safe)
        # - RayDAPOTrainer: directly converts all fields with np.array(v) (unsafe!)
        # To prevent the DAPO trainer from crashing, these are not placed
        # into reward_extra_info here; instead they are written directly
        # into data.meta_info (if available).
        trajectory_rewards = []
        trajectory_rounds = []
        for req_id, sample_indices in trajectory_groups.items():
            first_idx = sample_indices[0]
            # Use the outcome reward (0/1) rather than the mixed reward,
            # since trajectory_acc's threshold is >= 0.999, and a mixed
            # reward (e.g. 0.94) would be misclassified as incorrect.
            trajectory_rewards.append(1.0 if all_is_correct[first_idx] else 0.0)
            ei = rollout_extra_info_list[first_idx]
            if isinstance(ei, dict):
                traj_metrics = ei.get("trajectory_metrics", {})
                rounds = traj_metrics.get("search_rounds", 0)
            else:
                rounds = 0
            trajectory_rounds.append(float(rounds))

        # Write directly into meta_info, bypassing the
        # reward_extra_info -> non_tensor_batch conversion path
        if hasattr(data, "meta_info") and data.meta_info is not None:
            data.meta_info['ps_trajectory_reward'] = trajectory_rewards
            data.meta_info['ps_trajectory_rounds'] = trajectory_rounds
        else:
            # Fallback: place into reward_extra_info, relying on the trainer to handle it correctly
            reward_extra_info['ps_trajectory_reward'] = trajectory_rewards
            reward_extra_info['ps_trajectory_rounds'] = trajectory_rounds

        # Group stats -> write into meta_info (scalar values cannot go into non_tensor_batch)
        group_stats = {}
        for group_name in [GROUP_PLANNER, GROUP_SYNTHESIZER]:
            indices = group_indices[group_name]
            if indices:
                group_rewards = [all_rewards[i] for i in indices]
                group_stats[f'group_{group_name}_count'] = len(indices)
                group_stats[f'group_{group_name}_reward_mean'] = sum(group_rewards) / len(group_rewards)
            else:
                group_stats[f'group_{group_name}_count'] = 0
                group_stats[f'group_{group_name}_reward_mean'] = 0.0

        if hasattr(data, "meta_info") and data.meta_info is not None:
            data.meta_info.update(group_stats)

        # acc is used for logging trajectory accuracy on tensorboard, and
        # should use the 0/1 is_correct rather than the mixed all_rewards
        # (e.g. 0.94), otherwise reward/acc_eq_1_ratio would be severely underestimated
        data.batch["acc"] = torch.tensor(
            [1.0 if c else 0.0 for c in all_is_correct],
            dtype=torch.float32, device=prompt_ids.device,
        )

        # Summary logging (trajectory-level stats, to avoid confusing
        # sample-level counts with trajectory counts)
        total_trajectories = len(trajectory_groups)
        avg_reward = sum(all_rewards) / batch_size if batch_size > 0 else 0.0
        avg_rubric = sum(all_rubric_scores) / batch_size if batch_size > 0 else 0.0

        # Trajectory-level stats: iterate over trajectory groups, using first_idx to check
        traj_correct_count = 0
        traj_abnormal_count = 0
        traj_no_answer_count = 0
        traj_exceed_max_turns_count = 0
        abnormal_type_counter = defaultdict(int)
        for req_id, sample_indices in trajectory_groups.items():
            first_idx = sample_indices[0]
            if all_is_correct[first_idx]:
                traj_correct_count += 1
            if len(all_abnormal_types[first_idx]) > 0:
                traj_abnormal_count += 1
                for t in all_abnormal_types[first_idx]:
                    abnormal_type_counter[t] += 1
            if all_reward_reason[first_idx] == "no_answer_extracted":
                traj_no_answer_count += 1
            if all_reward_reason[first_idx] == "exceed_max_turns":
                traj_exceed_max_turns_count += 1

        unique_uids = len(uid_to_rewards) if uid_to_rewards else 0
        samples_per_group = [len(v) for v in uid_to_rewards.values()] if uid_to_rewards else []
        avg_samples_per_group = sum(samples_per_group) / len(samples_per_group) if samples_per_group else 0

        group_reward_stds = []
        for rewards in uid_to_rewards.values():
            if len(rewards) > 1:
                r_mean = sum(rewards) / len(rewards)
                r_var = sum((r - r_mean) ** 2 for r in rewards) / len(rewards)
                group_reward_stds.append(r_var ** 0.5)
        avg_group_reward_std = sum(group_reward_stds) / len(group_reward_stds) if group_reward_stds else 0

        traj_acc = traj_correct_count / total_trajectories if total_trajectories > 0 else 0.0
        logger.info(
            f"[PS Pipeline Rubric Reward Summary]\n"
            f"  Total samples: {batch_size}, trajectories: {total_trajectories}\n"
            f"  Correct trajectories: {traj_correct_count}, incorrect trajectories: {total_trajectories - traj_correct_count - traj_abnormal_count - traj_no_answer_count}\n"
            f"  Abnormal trajectories: {traj_abnormal_count} (of which exceed_max_turns={traj_exceed_max_turns_count})\n"
            f"  No-answer trajectories: {traj_no_answer_count}\n"
            f"  Trajectory accuracy: {traj_acc:.3f}\n"
            f"  Abnormality type distribution: {dict(abnormal_type_counter)}\n"
            f"  Mean reward (mixed): {avg_reward:.3f}, mean rubric: {avg_rubric:.3f}, alpha: {self.rubric_alpha}\n"
            f"  Groups: Planner={len(group_indices[GROUP_PLANNER])}, "
            f"Synthesizer={len(group_indices[GROUP_SYNTHESIZER])}\n"
            f"  GRPO groups: num_uids={unique_uids}, "
            f"avg samples per group={avg_samples_per_group:.1f}, "
            f"within-group reward std={avg_group_reward_std:.3f}"
        )

        logger.info(f"[Reward Summary] Mean reward: {reward_tensor.sum(dim=-1).mean().item():.3f}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        else:
            return reward_tensor

    def _verify_answers_batch(
        self,
        sequences_strs: List[str],
        ground_truths: List[Dict[str, Any]],
        task_extra_infos: List[Dict[str, Any]],
    ) -> List[float]:
        """Batch, concurrently call browsecomp_zh.compute_score to verify answers."""
        try:
            results = asyncio.run(
                parallel_compute_ps_end_to_end_score_async(
                    sequences_strs,
                    ground_truths,
                    task_extra_infos,
                    num_processes=self.config.get("num_workers", 64),
                )
            )
        except asyncio.TimeoutError:
            logger.error("PS Pipeline answer verification timed out globally! Setting all rewards to 0")
            results = [0.0] * len(sequences_strs)
        except RuntimeError as e:
            if "asyncio.run()" in str(e):
                logger.error(f"asyncio.run() exception: {e}, trying a fallback method")
                try:
                    loop = asyncio.get_event_loop()
                    results = loop.run_until_complete(
                        parallel_compute_ps_end_to_end_score_async(
                            sequences_strs,
                            ground_truths,
                            task_extra_infos,
                            num_processes=self.config.get("num_workers", 64),
                        )
                    )
                except Exception as e2:
                    logger.error(f"Fallback method also failed: {e2}")
                    logger.error(traceback.format_exc())
                    results = [0.0] * len(sequences_strs)
            else:
                logger.error(f"RuntimeError: {e}")
                logger.error(traceback.format_exc())
                results = [0.0] * len(sequences_strs)
        except Exception as e:
            logger.error(f"PS Pipeline answer verification exception, setting all rewards to 0: {e}")
            logger.error(traceback.format_exc())
            results = [0.0] * len(sequences_strs)

        return results
