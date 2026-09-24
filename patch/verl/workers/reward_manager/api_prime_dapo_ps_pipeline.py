#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS Pipeline Reward Manager

A reward manager designed for the Planner-Synthesizer (PS) Pipeline.

Core logic:
1. Regroup the flattened samples into trajectories by request_id
2. Find the last Planner's P(answer) within each trajectory, extract the content inside <answer></answer>
3. Verify answer correctness with browsecomp_zh.compute_score (LLM Judge)
4. Broadcast the trajectory's reward to every sample of that trajectory (planner and synthesizer)
5. Abnormal trajectories (based on abnormal_flags from the rollout stage) get a reward of 0 directly

Answer verification approach:
- Identical to the End-to-End Only / V2 C-GRPO and other reward managers
- Uses browsecomp_zh.compute_score -> GPT-4o LLM Judge to determine answer correctness
- Requires question + ground_truth['target'] + extracted_answer

Abnormality classification policy:
- Uses classify_abnormal_trajectory()
- Discarded (excluded from training): search_error / tool_parse_error / exceed_max_tokens / exceed_max_turns
- Offending round zeroed: excessive_tool_calls_per_turn (trajectory kept, only that round's reward is zeroed)
- Handled normally: repeated_query

Data flow (specific to PS Pipeline):
  sglang_rollout (PS Pipeline)
    -> DataProto (flattened round-level samples)
    -> ray_trainer: batch.repeat(N).union(gen_batch_output)
    -> this Reward Manager
    -> reward_tensor (assigned at each sample's last valid token position)

non_tensor_batch fields output by the PS Pipeline rollout:
  - messages: each sample's conversation context [{role, content}, ...]
  - reward_scores: tool reward dict
  - request_id: request UUID (shared within the same trajectory)
  - rollout_extra_info: {abnormal_flags, trajectory_metrics, role, round,
                         data_sample_index, think, answer, summary}
  - batch_statistics: batch-level stats

Note: fields like reward_model/ground_truth/question/data_source/extra_info
      come from the original dataset and are merged back in by
      ray_trainer's batch.union(). However, because PS Pipeline expands
      one-to-many (one prompt -> multiple round samples), these fields may
      or may not be present after union, or may be misaligned due to
      batch_size mismatches. This reward manager handles this defensively:
      - question: prefers non_tensor_batch, falls back to extraction from messages
      - ground_truth: prefers reward_model, falls back to the top level of non_tensor_batch

Version: 3.0.0
"""

import asyncio
import json
import os
import re
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Callable, Optional, List, Dict, Any

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score.browsecomp_zh import compute_score as browsecomp_compute_score
from verl.utils.reward_score.trajectory_reward_v2_cgrpo import classify_abnormal_trajectory
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_answer_from_text(text: str) -> Optional[str]:
    """Extract the content inside <answer></answer> tags from text.

    Supports multiple formats:
    - <answer>the answer</answer>
    - <Answer>the answer</Answer>
    - <ANSWER>the answer</ANSWER>

    Args:
        text: the raw model output text

    Returns:
        The extracted answer string, or None if extraction fails.
    """
    if not text or not isinstance(text, str):
        return None

    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

    if matches:
        # Take the last match (the final answer is usually the last one)
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
    """Compute the end-to-end reward for a single sample (browsecomp_zh LLM Judge).

    Uses the exact same scoring function as the End-to-End Only reward manager.

    Args:
        sequences_str: the generated sequence string (containing an <answer> tag)
        ground_truth: ground-truth data (must contain a 'target' field)
        task_extra_info: extra task info (must contain a 'question' field)
        executor: the thread pool executor
        timeout: timeout in seconds

    Returns:
        The end-to-end reward score (0.0 or 1.0).
    """
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
    """Compute end-to-end rewards concurrently for a batch.

    Args:
        sequences_strs: list of generated sequence strings
        ground_truths: list of ground-truth entries
        task_extra_infos: list of extra task info dicts
        num_processes: number of concurrent workers

    Returns:
        List of end-to-end reward scores.
    """
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


@register("apiprimedapopspipeline")
class ApiPrimeDapoPSPipelineRewardManager(AbstractRewardManager):
    """
    PS Pipeline Reward Manager

    Reward computation designed for the Planner-Synthesizer dual-role pipeline.

    Core logic:
    - Group by request_id -> find P(answer) -> extract <answer>
    - Verify the answer with browsecomp_zh.compute_score (LLM Judge)
    - Correct = 1, incorrect = 0, abnormal = 0
    - All samples in the same trajectory share the same reward

    Answer verification:
    - Identical to End-to-End Only / V2 C-GRPO
    - browsecomp_zh.compute_score -> GPT-4o Judge -> correct: yes/no -> 1.0/0.0

    Data requirements (from sglang_rollout PS pipeline + ray_trainer union):
    Required fields (output directly by the PS Pipeline rollout):
    - non_tensor_batch["rollout_extra_info"]: {abnormal_flags, role, round, answer, ...}
    - non_tensor_batch["request_id"]: used for grouping (shared within the same trajectory)
    - non_tensor_batch["messages"]: each sample's conversation context (includes the user question)
    Optional fields (from the original dataset, merged back via ray_trainer union):
    - non_tensor_batch["reward_model"]["ground_truth"]: the reference answer (must contain a target field)
    - non_tensor_batch["question"]: the question text
    - non_tensor_batch["data_source"]: the data source identifier
    - non_tensor_batch["extra_info"]: original extra info
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

        # browsecomp_zh.compute_score's API config is read from
        # environment variables inside browsecomp_zh.py; here we only need
        # to configure the concurrency level (same as End-to-End Only)
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

        logger.info("[OK] ApiPrimeDapoPSPipelineRewardManager initialized")

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute the PS Pipeline reward tensor.

        Processing flow:
        1. Extract rollout_extra_info and request_id
        2. Group by request_id -- each group is one complete trajectory
        3. For each trajectory:
           a. Check abnormal_flags (using the classify_abnormal_trajectory tiered policy)
           b. Find the P(answer) sample -> extract the <answer> content
           c. If extraction fails -> reward = 0
           d. Verify the answer with browsecomp_zh.compute_score (LLM Judge)
        4. Broadcast the trajectory reward to every sample of that trajectory

        Args:
            data: DataProto, from sglang_rollout PS pipeline
            return_dict: whether to return a dict

        Returns:
            reward_tensor, or {"reward_tensor": ..., "reward_extra_info": ...}
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
        trajectory_groups = defaultdict(list)  # request_id -> [sample_index, ...]
        for i in range(batch_size):
            req_id = request_id_list[i]
            if req_id is None:
                req_id = f"__standalone_{i}"
            trajectory_groups[req_id].append(i)

        logger.info(
            f"PS Pipeline Reward: batch_size={batch_size}, "
            f"num_trajectories={len(trajectory_groups)}"
        )

        # ========================================
        # === Step 3: extract ground_truth and question ===
        # ========================================
        #
        # PS Pipeline data flow note:
        # - the rollout output's non_tensor_batch only contains
        #   messages/request_id/rollout_extra_info etc.
        # - reward_model/question/data_source/extra_info come from the
        #   original dataset and are merged back by ray_trainer's batch.union()
        # - but because PS Pipeline expands one-to-many, these fields may
        #   be missing or misaligned after union
        # - hence a defensive, multi-level fallback extraction is needed
        #

        # --- extract ground_truth ---
        # Priority: reward_model.ground_truth > ground_truth (top level)
        reward_model_list = data.non_tensor_batch.get("reward_model", None)
        ground_truth_top = data.non_tensor_batch.get("ground_truth", None)

        ground_truth_per_sample = []
        for i in range(batch_size):
            gt = None
            # Prefer reward_model
            if reward_model_list is not None:
                try:
                    rm = reward_model_list[i]
                    if isinstance(rm, dict):
                        gt = rm.get("ground_truth", None)
                except (IndexError, TypeError, KeyError):
                    pass
            # Fall back to the top-level ground_truth
            if gt is None and ground_truth_top is not None:
                try:
                    gt = ground_truth_top[i]
                except (IndexError, TypeError):
                    pass
            ground_truth_per_sample.append(gt)

        gt_missing_count = sum(1 for gt in ground_truth_per_sample if gt is None)
        if gt_missing_count > 0:
            logger.warning(
                f"[PSPipeline] {gt_missing_count}/{batch_size} samples are missing ground_truth!"
                f" non_tensor_batch keys: {list(data.non_tensor_batch.keys())}"
            )

        # --- extract question ---
        # Priority: non_tensor_batch["question"] > extra_info["question"]
        #        > reward_model["question"] > the first user message in messages
        #
        # In PS Pipeline, the messages field is guaranteed to exist
        # (output directly by the rollout) and contains that sample's
        # (round's) conversation context. Its first user message usually
        # contains the original question. Note however: in PS Pipeline,
        # each round's messages are that round's prompt/response, not the
        # original user question -- the original user question needs to
        # come from a field merged in by ray_trainer's union.
        #
        question_ntb = data.non_tensor_batch.get("question", None)
        extra_info_ntb = data.non_tensor_batch.get("extra_info", None)
        messages_ntb = data.non_tensor_batch.get("messages", None)

        question_list = []
        missing_question_count = 0
        for i in range(batch_size):
            question = None

            # Prefer non_tensor_batch["question"] (merged in by ray_trainer union)
            if question is None and question_ntb is not None:
                try:
                    q = question_ntb[i]
                    if q is not None and isinstance(q, str) and q.strip():
                        question = q
                except (IndexError, TypeError):
                    pass

            # Try extra_info
            if question is None and extra_info_ntb is not None:
                try:
                    ei = extra_info_ntb[i]
                    if isinstance(ei, dict):
                        q = ei.get("question", None)
                        if q is not None and isinstance(q, str) and q.strip():
                            question = q
                except (IndexError, TypeError):
                    pass

            # Try reward_model
            if question is None and reward_model_list is not None:
                try:
                    rm = reward_model_list[i]
                    if isinstance(rm, dict):
                        q = rm.get("question", None)
                        if q is not None and isinstance(q, str) and q.strip():
                            question = q
                except (IndexError, TypeError):
                    pass

            # Last resort: extract the first user message from messages
            # PS Pipeline's messages format: {"messages": [{"role": "user", "content": ...}, ...]}
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
                f"[PSPipeline] {missing_question_count}/{batch_size} samples are missing question!"
                f" non_tensor_batch keys: {list(data.non_tensor_batch.keys())}"
            )

        # ========================================
        # === Step 4: process each trajectory, collecting the ones that need LLM Judge verification ===
        # ========================================
        #
        # Initialize results for all samples
        #
        all_rewards = [0.0] * batch_size
        all_end_to_end_scores = [0.0] * batch_size
        all_extracted_answers = [None] * batch_size
        all_is_correct = [False] * batch_size
        all_abnormal_types = [[] for _ in range(batch_size)]
        all_reward_reason = [""] * batch_size
        already_print_data_sources = {}

        # Collect the trajectories that need LLM Judge verification
        # Format: [(req_id, sample_indices, answer_sequences_str, ground_truth, task_extra_info), ...]
        trajectories_to_verify = []

        for req_id, sample_indices in trajectory_groups.items():
            try:
                first_idx = sample_indices[0]
                extra_info = rollout_extra_info_list[first_idx]
                abnormal_flags = extra_info.get("abnormal_flags", {}) if isinstance(extra_info, dict) else {}

                # --- 4a. classify abnormalities ---
                # Discard: search_error / exceed_max_tokens / degenerate_output
                # Zero reward: tool_parse_error / exceed_max_turns
                # Offending round zeroed: excessive_tool_calls_per_turn
                # Handled normally: repeated_query
                should_give_zero_reward, should_discard, abnormal_types = classify_abnormal_trajectory(abnormal_flags)

                if should_discard:
                    for idx in sample_indices:
                        all_abnormal_types[idx] = abnormal_types
                        all_reward_reason[idx] = f"discarded: {abnormal_types}"
                    logger.info(
                        f"[PS Pipeline] Trajectory {req_id} discarded: {abnormal_types}"
                    )
                    continue

                if should_give_zero_reward:
                    # Kept for backward compatibility (current logic never reaches here)
                    for idx in sample_indices:
                        all_abnormal_types[idx] = abnormal_types
                        all_reward_reason[idx] = f"abnormal: {abnormal_types}"
                    continue

                # --- 4b. find the P(answer) sample and extract the answer ---
                answer_text = None

                # Method 1: from the pre-extracted answer field in rollout_extra_info
                for idx in sample_indices:
                    ei = rollout_extra_info_list[idx]
                    if not isinstance(ei, dict):
                        continue
                    role = ei.get("role", "")
                    answer_field = ei.get("answer", None)
                    if role == "planner" and answer_field is not None and answer_field != "":
                        answer_text = answer_field

                # Method 2: fall back to extracting <answer> from the response text
                if answer_text is None:
                    for idx in reversed(sample_indices):
                        ei = rollout_extra_info_list[idx]
                        if isinstance(ei, dict) and ei.get("role", "") == "planner":
                            resp_text = sequences_str[idx]
                            extracted = extract_answer_from_text(resp_text)
                            if extracted is not None:
                                answer_text = extracted
                                break

                # --- 4c. check the answer extraction result ---
                if answer_text is None:
                    for idx in sample_indices:
                        all_reward_reason[idx] = "no_answer_extracted"
                    continue

                extracted_answer = answer_text.strip()
                for idx in sample_indices:
                    all_extracted_answers[idx] = extracted_answer

                # Get ground_truth
                gt = ground_truth_per_sample[first_idx]
                if gt is None:
                    for idx in sample_indices:
                        all_reward_reason[idx] = "no_ground_truth"
                    continue

                # --- 4d. build the input required by the LLM Judge ---
                #
                # browsecomp_zh.compute_score requires:
                # 1. sequences_str: text containing <answer>...</answer>
                # 2. ground_truth: dict, must contain a 'target' field
                # 3. extra_info: dict, must contain a 'question' field
                #
                # For PS Pipeline, we need to build text with an <answer>
                # tag, since browsecomp_zh extracts the answer from it
                # itself via extract_solution()
                #
                answer_sequences_str = f"<answer>{extracted_answer}</answer>"

                # Build task_extra_info
                # In PS Pipeline, question comes from the multi-level
                # fallback extraction (see step 3)
                task_extra_info = {}
                if question_list[first_idx] is not None:
                    task_extra_info["question"] = question_list[first_idx]

                # Check whether question is present (required by browsecomp_zh.compute_score)
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
        # === Step 5: batch, concurrently call the LLM Judge to verify answers ===
        # ========================================
        #
        # Consistent with End-to-End Only: uses asyncio + ThreadPoolExecutor for concurrency
        #
        if len(trajectories_to_verify) > 0:
            verify_sequences = [t[2] for t in trajectories_to_verify]
            verify_ground_truths = [t[3] for t in trajectories_to_verify]
            verify_extra_infos = [t[4] for t in trajectories_to_verify]

            end_to_end_scores = self._verify_answers_batch(
                verify_sequences, verify_ground_truths, verify_extra_infos
            )

            # Broadcast the verification result to every sample of each trajectory
            # Every round receives the same trajectory reward (no time discounting)
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

        # ========================================
        # === Step 6: print debug info (enhanced) ===
        # ========================================
        # data_source may not exist in the PS Pipeline output (from the original dataset via union)
        data_sources = data.non_tensor_batch.get("data_source", None)
        uid_list = data.non_tensor_batch.get("uid", [None] * batch_size)

        # --- 6a. print detail for each trajectory ---
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

                # Get this trajectory's uid (for verifying GRPO grouping)
                traj_uid = uid_list[first_idx] if first_idx < len(uid_list) else "N/A"

                logger.info(
                    f"[Trajectory {req_id[:8]}...] "
                    f"uid={str(traj_uid)[:8]}..., "
                    f"samples={len(sample_indices)}, "
                    f"reward={all_rewards[first_idx]:.1f}, "
                    f"end_to_end={all_end_to_end_scores[first_idx]:.3f}, "
                    f"reason={all_reward_reason[first_idx]}, "
                    f"answer={str(all_extracted_answers[first_idx])[:100]}"
                )

                # Print the trajectory structure, with each sample's reward and uid
                roles = []
                for idx in sample_indices:
                    ei = rollout_extra_info_list[idx]
                    if isinstance(ei, dict):
                        role = ei.get("role", "?")
                        rnd = ei.get("round", "?")
                        sample_uid = str(uid_list[idx])[:8] if idx < len(uid_list) else "?"
                        r = all_rewards[idx]
                        roles.append(f"{role}(r{rnd},uid={sample_uid},rew={r:.1f})")
                    else:
                        roles.append("?")
                logger.info(f"  Trajectory structure: {' -> '.join(roles)}")

        # --- 6b. GRPO grouping verification log ---
        # Group by uid, and check the reward distribution within the same uid
        uid_to_rewards = defaultdict(list)
        uid_to_request_ids = defaultdict(set)
        for i in range(batch_size):
            uid = uid_list[i] if i < len(uid_list) else None
            if uid is not None:
                uid_to_rewards[str(uid)].append(all_rewards[i])
                req_id = request_id_list[i] if i < len(request_id_list) else "?"
                uid_to_request_ids[str(uid)].add(str(req_id)[:8])

        logger.info(
            f"[GRPO Grouping] total_samples={batch_size}, "
            f"num_uid_groups={len(uid_to_rewards)}, "
            f"num_trajectories={len(trajectory_groups)}"
        )

        # Print detail for the first few uid groups (to verify grouping correctness)
        printed_groups = 0
        for uid_str, rewards in uid_to_rewards.items():
            if printed_groups >= min(5, self.num_examine):
                break
            req_ids = uid_to_request_ids[uid_str]
            reward_mean = sum(rewards) / len(rewards) if rewards else 0
            reward_values = [f"{r:.1f}" for r in rewards]
            logger.info(
                f"  [GRPO Group uid={uid_str[:8]}...] "
                f"samples={len(rewards)}, "
                f"request_ids={req_ids}, "
                f"rewards=[{','.join(reward_values)}], "
                f"mean={reward_mean:.3f}"
            )
            printed_groups += 1

        # ========================================
        # === Step 7: fill reward_extra_info ===
        # ========================================
        #
        # An extra_info format consistent with other reward managers
        #
        reward_extra_info['end_to_end_score'] = all_end_to_end_scores
        reward_extra_info['final_reward'] = all_rewards
        reward_extra_info['extracted_answer'] = [
            a if a is not None else "" for a in all_extracted_answers
        ]
        reward_extra_info['is_correct'] = [
            1.0 if c else 0.0 for c in all_is_correct
        ]
        # The acc field is used by process_validation_metrics for the
        # validation set (core_var = "acc" if "acc" in var2metric2val else "reward").
        # Without this field, val-core on the validation set would use the
        # discounted reward instead of the 0/1 accuracy.
        reward_extra_info['acc'] = [
            1.0 if c else 0.0 for c in all_is_correct
        ]
        reward_extra_info['reward_reason'] = all_reward_reason
        reward_extra_info['abnormal_types'] = all_abnormal_types

        # ---- trajectory-level reward and round-count stats ----
        # A sample-level reward mean would be diluted by the number of
        # samples per trajectory (trajectories with more rounds contribute
        # more samples), so it doesn't reflect true trajectory accuracy.
        # We compute an aggregated value here for tensorboard.
        # Note: these lists have length != batch_size, so they can't go
        # directly into non_tensor_batch. Once placed into
        # reward_extra_info, ray_trainer extracts them before injecting
        # into non_tensor_batch.
        trajectory_rewards = []  # one reward per trajectory (0 or 1)
        trajectory_rounds = []   # number of search rounds per trajectory
        for req_id, sample_indices in trajectory_groups.items():
            first_idx = sample_indices[0]
            # Use the outcome reward (0/1) rather than the discounted
            # reward, since trajectory_acc's threshold is >= 0.999, and a
            # discounted reward (e.g. 0.98) would be misclassified as incorrect.
            trajectory_rewards.append(1.0 if all_is_correct[first_idx] else 0.0)
            # Extract search_rounds from rollout_extra_info
            ei = rollout_extra_info_list[first_idx]
            if isinstance(ei, dict):
                traj_metrics = ei.get("trajectory_metrics", {})
                rounds = traj_metrics.get("search_rounds", 0)
            else:
                rounds = 0
            trajectory_rounds.append(float(rounds))

        reward_extra_info['ps_trajectory_reward'] = trajectory_rewards
        reward_extra_info['ps_trajectory_rounds'] = trajectory_rounds

        # acc is used for logging trajectory accuracy on tensorboard, and
        # should use the 0/1 is_correct rather than the discounted
        # all_rewards (e.g. 0.98), otherwise reward/acc_eq_1_ratio would be severely underestimated
        data.batch["acc"] = torch.tensor(
            [1.0 if c else 0.0 for c in all_is_correct],
            dtype=torch.float32, device=prompt_ids.device,
        )

        # Summary logging (trajectory-level stats, to avoid confusing
        # sample-level counts with trajectory counts)
        total_trajectories = len(trajectory_groups)
        avg_reward = sum(all_rewards) / batch_size if batch_size > 0 else 0.0

        # Trajectory-level stats: iterate over trajectory groups, using first_idx to check
        traj_correct_count = 0
        traj_abnormal_count = 0
        traj_no_answer_count = 0
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

        # Group-level stats for GRPO
        unique_uids = len(uid_to_rewards) if uid_to_rewards else 0
        samples_per_group = [len(v) for v in uid_to_rewards.values()] if uid_to_rewards else []
        avg_samples_per_group = sum(samples_per_group) / len(samples_per_group) if samples_per_group else 0

        # Compute within-group reward variance (to check whether GRPO is effective)
        group_reward_stds = []
        for rewards in uid_to_rewards.values():
            if len(rewards) > 1:
                r_mean = sum(rewards) / len(rewards)
                r_var = sum((r - r_mean) ** 2 for r in rewards) / len(rewards)
                group_reward_stds.append(r_var ** 0.5)
        avg_group_reward_std = sum(group_reward_stds) / len(group_reward_stds) if group_reward_stds else 0

        traj_acc = traj_correct_count / total_trajectories if total_trajectories > 0 else 0.0
        logger.info(
            f"[PS Pipeline Reward Summary]\n"
            f"  Total samples: {batch_size}, trajectories: {total_trajectories}\n"
            f"  Correct trajectories: {traj_correct_count}, incorrect trajectories: {total_trajectories - traj_correct_count - traj_abnormal_count - traj_no_answer_count}\n"
            f"  Abnormal trajectories: {traj_abnormal_count}, no-answer trajectories: {traj_no_answer_count}\n"
            f"  Trajectory accuracy: {traj_acc:.3f}\n"
            f"  Abnormality type distribution: {dict(abnormal_type_counter)}\n"
            f"  Mean reward (discounted): {avg_reward:.3f}\n"
            f"  GRPO groups: num_uids={unique_uids}, "
            f"avg samples per group={avg_samples_per_group:.1f}, "
            f"within-group reward std={avg_group_reward_std:.3f}"
        )

        logger.info(f"[Reward Summary] Mean reward: {reward_tensor.sum(dim=-1).mean().item():.3f}")

        # ========================================
        # === Step 8: write the JSONL file log ===
        # ========================================
        # Similar to the rollout's ps_pipeline_logs, writes detailed reward info to a file
        try:
            reward_log_dir = os.environ.get("REWARD_LOG_DIR", "./logs/reward_logs")
            os.makedirs(reward_log_dir, exist_ok=True)

            # Get step from meta_info; identify the worker via the RANK / LOCAL_RANK environment variable
            step = data.meta_info.get("global_steps", 0) if hasattr(data, "meta_info") and data.meta_info else 0
            worker_rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
            reward_log_path = os.path.join(
                reward_log_dir,
                f"step_{step}_worker_{worker_rank}.jsonl"
            )

            # Build detailed reward records for each trajectory
            trajectory_records = []
            for req_id, sample_indices in trajectory_groups.items():
                first_idx = sample_indices[0]
                traj_uid = str(uid_list[first_idx]) if first_idx < len(uid_list) and uid_list[first_idx] is not None else "N/A"

                # Build info for each round/sample
                round_details = []
                for idx in sample_indices:
                    ei = rollout_extra_info_list[idx]
                    sample_info = {
                        "sample_index": idx,
                        "reward": all_rewards[idx],
                        "end_to_end_score": all_end_to_end_scores[idx],
                        "reward_reason": all_reward_reason[idx],
                        "uid": str(uid_list[idx]) if idx < len(uid_list) and uid_list[idx] is not None else "N/A",
                    }
                    if isinstance(ei, dict):
                        sample_info["role"] = ei.get("role", "?")
                        sample_info["round"] = ei.get("round", "?")
                        sample_info["abnormal_flags"] = ei.get("abnormal_flags", {})
                    round_details.append(sample_info)

                traj_record = {
                    "timestamp": datetime.now().isoformat(),
                    "step": step,
                    "request_id": req_id,
                    "uid": traj_uid,
                    "n_samples": len(sample_indices),
                    "reward": all_rewards[first_idx],
                    "end_to_end_score": all_end_to_end_scores[first_idx],
                    "is_correct": all_is_correct[first_idx],
                    "reward_reason": all_reward_reason[first_idx],
                    "extracted_answer": str(all_extracted_answers[first_idx])[:200] if all_extracted_answers[first_idx] else None,
                    "question": str(question_list[first_idx])[:200] if first_idx < len(question_list) and question_list[first_idx] else None,
                    "abnormal_types": all_abnormal_types[first_idx],
                    "rounds": round_details,
                }
                trajectory_records.append(traj_record)

            # Build a summary of the GRPO groups
            grpo_groups_summary = []
            for uid_str, rewards in uid_to_rewards.items():
                req_ids = list(uid_to_request_ids[uid_str])
                reward_mean = sum(rewards) / len(rewards) if rewards else 0
                reward_std = 0
                if len(rewards) > 1:
                    r_var = sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)
                    reward_std = r_var ** 0.5
                grpo_groups_summary.append({
                    "uid": uid_str,
                    "n_samples": len(rewards),
                    "n_trajectories": len(req_ids),
                    "request_ids": req_ids,
                    "rewards": rewards,
                    "reward_mean": round(reward_mean, 4),
                    "reward_std": round(reward_std, 4),
                })

            # Write the full batch reward log
            batch_log = {
                "timestamp": datetime.now().isoformat(),
                "step": step,
                "worker_rank": worker_rank,
                "batch_size": batch_size,
                "n_trajectories": len(trajectory_groups),
                "n_correct": traj_correct_count,
                "n_abnormal": traj_abnormal_count,
                "n_no_answer": traj_no_answer_count,
                "avg_reward": round(avg_reward, 4),
                "abnormal_type_distribution": dict(abnormal_type_counter),
                "grpo_summary": {
                    "n_uid_groups": unique_uids,
                    "avg_samples_per_group": round(avg_samples_per_group, 2),
                    "avg_group_reward_std": round(avg_group_reward_std, 4),
                },
                "grpo_groups": grpo_groups_summary,
                "trajectories": trajectory_records,
            }

            with open(reward_log_path, "a", buffering=1) as f:
                f.write(json.dumps(batch_log, ensure_ascii=False) + "\n")

            logger.info(f"[Reward Log] Wrote to {reward_log_path}")

        except Exception as e:
            logger.warning(f"[Reward Log] Failed to write log file: {type(e).__name__}: {e}")

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
        """Batch, concurrently call browsecomp_zh.compute_score to verify answers.

        Uses the exact same verification logic as the End-to-End Only
        reward manager's verify_normal_samples.

        Args:
            sequences_strs: list of texts containing an <answer> tag (one per trajectory)
            ground_truths: list of ground-truth entries
            task_extra_infos: list of extra task info dicts (must contain a question field)

        Returns:
            List of end-to-end reward scores (0.0 or 1.0).
        """
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
