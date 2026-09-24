#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Only Reward Manager

A reward manager based purely on the end-to-end reward (browsecomp_zh.evaluate),
with no trajectory-coverage reward.

Core features:
- Computes only the end-to-end reward (answer correctness)
- Handles formatting errors and overlong rollouts (abnormal flags read from sglang rollout)
- Abnormal samples are given a reward of 0 directly

Differences from Trajectory Reward V1 Naive:
- Does NOT compute trajectory-coverage reward
- Does NOT perform intermediate-fact identification
- Does NOT need within-group normalization
- DOES keep all abnormal-handling logic (identical to trajectory_reward_v1_naive)

Author: PS-Pipeline Contributors
Version: 1.0.0
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Optional, List, Dict, Any
import traceback
from collections import defaultdict

import torch
import numpy as np
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score.browsecomp_zh import compute_score as browsecomp_compute_score
from verl.utils.reward_score.trajectory_reward_v2_cgrpo import classify_abnormal_trajectory
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def single_compute_end_to_end_score(
    config: Dict[str, Any],
    prompts_str: str,
    sequences_str: str,
    ground_truth: Dict[str, Any],
    data_source: str,
    task_extra_info: Dict[str, Any],
    executor: ThreadPoolExecutor,
    timeout: float = 6000.0,
) -> float:
    """Compute the end-to-end reward for a single sample.

    Note (2026-01-01):
    - Only computes the end-to-end reward (browsecomp_zh.evaluate)
    - Does not compute trajectory-coverage reward

    Args:
        config: config dict
        prompts_str: the prompt string
        sequences_str: the generated sequence string
        ground_truth: ground-truth data
        data_source: the data source
        task_extra_info: extra task info
        executor: the thread pool executor
        timeout: timeout in seconds

    Returns:
        The end-to-end reward score (0.0 to 1.0).
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
        logger.warning("End-to-end reward computation timed out")
        return 0.0
    except Exception as e:
        logger.error(f"End-to-end reward computation failed: {e}")
        logger.error(traceback.format_exc())
        return 0.0


async def parallel_compute_end_to_end_score_async(
    config: Dict[str, Any],
    prompts_strs: List[str],
    sequences_strs: List[str],
    ground_truths: List[Dict[str, Any]],
    data_sources: List[str],
    task_extra_infos: Optional[List[Dict[str, Any]]] = None,
    num_processes: int = 64,
) -> List[float]:
    """Compute end-to-end rewards concurrently for a batch.

    Note (2026-01-01):
    - Only computes the end-to-end reward, not coverage

    Args:
        config: config dict
        prompts_strs: list of prompt strings
        sequences_strs: list of generated sequence strings
        ground_truths: list of ground-truth entries
        data_sources: list of data sources
        task_extra_infos: list of extra task info dicts
        num_processes: number of concurrent workers

    Returns:
        List of end-to-end reward scores.
    """
    results = []
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        if task_extra_infos is None:
            task_extra_infos = [{}] * len(data_sources)

        # Create all tasks
        tasks_async = [
            single_compute_end_to_end_score(
                config, prompts_str, sequences_str, ground_truth, 
                data_source, task_extra_info, executor, timeout=6000.0
            )
            for prompts_str, sequences_str, ground_truth, data_source, task_extra_info in zip(
                prompts_strs, sequences_strs, ground_truths, 
                data_sources, task_extra_infos
            )
        ]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks_async, return_exceptions=False)

    # Process results
    processed_results = []
    for result in results:
        if isinstance(result, (Exception, BaseException)) or result is None:
            # Handle failed or timed-out tasks
            processed_results.append(0.0)
        else:
            processed_results.append(float(result))

    return processed_results


@register("apiprimedapoendtoendonly")
class ApiPrimeDapoEndToEndOnlyRewardManager(AbstractRewardManager):
    """
    End-to-End Only Reward Manager

    Features:
    - Computes only the end-to-end reward (browsecomp_zh.evaluate)
    - Handles formatting errors and overlong rollouts
    - Does not compute trajectory-coverage reward

    Differences from Trajectory Reward V1 Naive:
    - Final reward formula: reward = end_to_end_score (no coverage involved)
    - Does not save trajectory_reward_info
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

        # Note (2026-01-01):
        # browsecomp_zh.compute_score does not need to read config from a
        # file -- its API config (api_key, base_url, model) is read from
        # environment variables inside browsecomp_zh.py. We only need to
        # configure the concurrency level here.
        #
        # compute_score parameter: accepted for interface compatibility,
        # but this reward manager always uses browsecomp_compute_score and
        # ignores the passed-in compute_score.
        self.config = {
            "num_workers": 64,  # concurrency level
        }

        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert (
                self.max_resp_len is not None
            ), f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            assert (
                self.max_resp_len >= self.overlong_buffer_cfg.len
            ), "max_resp_len must be larger than overlong_buffer.len"

        logger.info("[OK] ApiPrimeDapoEndToEndOnlyRewardManager initialized")

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute the final reward tensor.

        Logic (2026-01-01):
        1. **Upfront abnormality detection**: read abnormal flags produced
           by sglang rollout.
           - Any sample with an abnormal flag (tool parse error, repeated
             query, overlong, etc.) is given a reward of 0 directly.
           - Overlong / tool-abnormality detection is no longer done inside
             this reward manager.

        2. **Reward computation for normal samples**:
           - Only samples without abnormal flags go through reward computation.
           - Only the end-to-end reward is computed, not coverage.
           - End-to-End Only reward formula:
             * reward = end_to_end_score (used directly)

        Args:
            data: the data batch
            return_dict: whether to return a dict

        Returns:
            The reward tensor, or a dict containing reward info.
        """
        # If rm_scores already exists, return it directly
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # Batched reward computation
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        # ========================================
        # === Step 1: extract abnormal flags from rollout_extra_info ===
        # ========================================
        #
        # rollout_extra_info structure (produced by sglang rollout):
        # rollout_extra_info = [
        #     {
        #         "abnormal_flags": {
        #             "excessive_tool_calls_per_turn": bool,
        #             "tool_parse_error": bool,
        #             "repeated_query": bool,
        #             "search_error": bool,
        #             "exceed_max_turns": bool,
        #             "exceed_max_tokens": bool,
        #         },
        #         "trajectory_metrics": {
        #             "search_steps": int,
        #             "subq_calls": int,
        #             ...,
        #         }
        #     },
        #     ...,
        # ]
        #
        rollout_extra_info_list = data.non_tensor_batch.get("rollout_extra_info", [{}] * len(data))

        # Extract abnormal flags
        abnormal_flags_list = []
        trajectory_metrics_list = []
        for rollout_extra_info in rollout_extra_info_list:
            if isinstance(rollout_extra_info, dict):
                abnormal_flags_list.append(rollout_extra_info.get("abnormal_flags", {}))
                trajectory_metrics_list.append(rollout_extra_info.get("trajectory_metrics", {}))
            else:
                # Compatibility with legacy data (no rollout_extra_info)
                abnormal_flags_list.append({})
                trajectory_metrics_list.append({})

        # ========================================
        # === Step 2: check abnormalities + <answer> tag -> identify normal samples ===
        # ========================================
        #
        # Abnormality classification policy:
        # - Discard: search_error / tool_parse_error / exceed_max_tokens / exceed_max_turns
        # - Zero the offending round: excessive_tool_calls_per_turn (PS-Pipeline only)
        # - Compute normally: repeated_query
        #
        normal_indices = []
        for i, abnormal_flags in enumerate(abnormal_flags_list):
            should_give_zero_reward, should_discard, abnormal_types = classify_abnormal_trajectory(abnormal_flags)

            # Discard: search_error / tool_parse_error / exceed_max_tokens / exceed_max_turns
            if should_discard:
                continue

            # Check the answer format: is there an <answer> tag
            has_answer_tag = False
            try:
                trajectory = sequences_str[i]
                has_answer_tag = '<answer>' in trajectory.lower()
            except Exception as e:
                logger.warning(f"Failed to check <answer> tag for sample {i}: {e}")

            # Only compute reward for samples that should NOT get zero
            # reward AND have an answer tag
            if not should_give_zero_reward and has_answer_tag:
                normal_indices.append(i)
            else:
                # Record the abnormality type
                if not has_answer_tag:
                    abnormal_flags_list[i]["no_answer_tag"] = True

        logger.info(f"Batch size: {len(data)}, normal samples: {len(normal_indices)}, abnormal samples: {len(data) - len(normal_indices)}")

        # ========================================
        # === Pre-initialize reward_extra_info for all samples ===
        # ========================================
        #
        # Important fix (2026-01-01):
        # The previous code appended normal samples' extra_info first, then
        # abnormal samples' extra_info, which made the order of
        # reward_extra_info inconsistent with reward_tensor!
        #
        # Correct approach: initialize every sample's extra_info in the
        # original data order.
        #
        batch_size = len(data)
        end_to_end_scores_all = [0.0] * batch_size  # end-to-end score for every sample (0 for abnormal ones)
        final_rewards_all = [0.0] * batch_size  # final reward for every sample (0 for abnormal ones)
        abnormal_types_all = [[] for _ in range(batch_size)]  # abnormality types for every sample

        # First, tag the abnormality types for abnormal samples
        for i, abnormal_flags in enumerate(abnormal_flags_list):
            if i not in normal_indices:
                abnormal_types_all[i] = [k for k, v in abnormal_flags.items() if v]

        # Only compute rewards if there are normal samples
        if len(normal_indices) > 0:
            # Extract data for normal samples
            normal_prompts_str = [self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True) for i in normal_indices]
            normal_sequences_str = [sequences_str[i] for i in normal_indices]

            # Extract ground_truth (compatible with both old and new data schemas)
            ground_truth_list = []
            for data_item in data:
                if "reward_model" in data_item.non_tensor_batch:
                    ground_truth_list.append(data_item.non_tensor_batch["reward_model"]["ground_truth"])
                else:
                    # Legacy compatibility
                    ground_truth_list.append({})

            normal_ground_truth = [ground_truth_list[i] for i in normal_indices]
            normal_data_sources = [data_sources[i] for i in normal_indices]

            # Extract task_extra_infos and the question field
            #
            # Note (2026-01-01):
            # browsecomp_zh.compute_score requires extra_info to contain a
            # 'question' field. Possible sources:
            #   1. data.non_tensor_batch["extra_info"] (the original data's extra_info, contains question)
            #   2. data.non_tensor_batch["question"] (question stored directly as its own field)
            #
            task_extra_infos = []

            # Extract the original data's extra_info (contains question)
            original_extra_info_list = data.non_tensor_batch.get("extra_info", [{}] * len(data))

            # Extract the question field (compatible with both data formats)
            question_list = []
            missing_question_count = 0  # count of samples missing a question
            for i, data_item in enumerate(data):
                # Prefer non_tensor_batch["question"]
                question = data_item.non_tensor_batch.get("question", None)

                # Otherwise, try the original extra_info
                if question is None and isinstance(original_extra_info_list[i], dict):
                    question = original_extra_info_list[i].get("question", None)

                # Otherwise, try reward_model
                if question is None and "reward_model" in data_item.non_tensor_batch:
                    question = data_item.non_tensor_batch["reward_model"].get("question", None)

                if question is None:
                    missing_question_count += 1

                question_list.append(question)

            # Print debug info
            if missing_question_count > 0:
                logger.warning(f"[ApiPrimeDapoEndToEndOnlyRewardManager] {missing_question_count}/{len(data)} samples are missing the question field!")
                logger.warning(f"[ApiPrimeDapoEndToEndOnlyRewardManager] data.non_tensor_batch.keys(): {list(data.non_tensor_batch.keys())}")
                if len(data) > 0:
                    logger.warning(f"[ApiPrimeDapoEndToEndOnlyRewardManager] data[0].non_tensor_batch.keys(): {list(data[0].non_tensor_batch.keys())}")
                    if "reward_model" in data[0].non_tensor_batch:
                        logger.warning(f"[ApiPrimeDapoEndToEndOnlyRewardManager] data[0].non_tensor_batch['reward_model'].keys(): {list(data[0].non_tensor_batch['reward_model'].keys())}")

            for i in normal_indices:
                # Build task_extra_info, ensuring it contains a question field
                task_extra_info = {}

                # Copy over anything from the original extra_info first
                if isinstance(original_extra_info_list[i], dict):
                    task_extra_info.update(original_extra_info_list[i])

                # Ensure the question field is present (required by browsecomp_zh.compute_score)
                if question_list[i] is not None:
                    task_extra_info["question"] = question_list[i]

                task_extra_infos.append(task_extra_info)

            # Compute the end-to-end reward for normal samples
            end_to_end_scores = self.verify_normal_samples(
                normal_prompts_str,
                normal_sequences_str,
                normal_ground_truth,
                normal_data_sources,
                task_extra_infos
            )

            # ========================================
            # === Step 3: compute the final reward (End-to-End Only) ===
            # ========================================
            #
            # End-to-End Only reward computation:
            # - reward = end_to_end_score (used directly)
            #
            for idx, i in enumerate(normal_indices):
                # End-to-End Only reward computation
                final_reward = end_to_end_scores[idx]

                # Set the reward tensor
                reward_tensor[i, valid_response_length[i].item() - 1] = final_reward

                # Update the extra_info at the corresponding position (preserve original order)
                end_to_end_scores_all[i] = end_to_end_scores[idx]
                final_rewards_all[i] = final_reward
                # abnormal_types_all[i] is already initialized to an empty list

                # Print debug info
                data_source = data_sources[i]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    logger.info(
                        f"[Normal Sample {i}] "
                        f"end_to_end={end_to_end_scores[idx]:.3f}, "
                        f"final_reward={final_reward:.3f}"
                    )

        # ========================================
        # === Step 4: log abnormal-sample info (reward is already 0) ===
        # ========================================
        for i in range(batch_size):
            if i not in normal_indices:
                # Extract the abnormality types (including no_answer_tag)
                abnormal_types = [k for k, v in abnormal_flags_list[i].items() if v]
                abnormal_types_all[i] = abnormal_types

                # Print debug info
                data_source = data_sources[i]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    logger.info(
                        f"[Abnormal Sample {i}] "
                        f"abnormal_types={abnormal_types}, "
                        f"final_reward=0.0"
                    )

        # ========================================
        # === Step 5: fill reward_extra_info in original order ===
        # ========================================
        #
        # Important: extra_info is filled here in the original data order
        # (0, 1, 2, ...) to stay consistent with reward_tensor's order.
        #
        reward_extra_info['end_to_end_score'] = end_to_end_scores_all
        reward_extra_info['final_reward'] = final_rewards_all
        reward_extra_info['abnormal_types'] = abnormal_types_all

        logger.info(f"[Reward Summary] Mean reward: {reward_tensor.sum(dim=-1).mean().item():.3f}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        else:
            return reward_tensor

    def verify_normal_samples(
        self, 
        prompts_strs: List[str],
        sequences_strs: List[str],
        ground_truths: List[Dict[str, Any]],
        data_sources: List[str],
        task_extra_infos: List[Dict[str, Any]]
    ) -> List[float]:
        """Compute the end-to-end reward for normal samples.

        A helper method that only processes normal samples (those without
        abnormal flags).

        Args:
            prompts_strs: list of prompt strings
            sequences_strs: list of generated sequence strings
            ground_truths: list of ground-truth entries
            data_sources: list of data sources
            task_extra_infos: list of extra task info dicts

        Returns:
            List of end-to-end reward scores.
        """
        try:
            results = asyncio.run(
                parallel_compute_end_to_end_score_async(
                    self.config,
                    prompts_strs,
                    sequences_strs,
                    ground_truths,
                    data_sources,
                    task_extra_infos=task_extra_infos,
                    num_processes=self.config.get("num_workers", 64),
                )
            )
        except asyncio.TimeoutError:
            logger.error("Global timeout while computing rewards for normal samples! Setting all rewards to 0")
            results = [0.0] * len(sequences_strs)
        except Exception as e:
            logger.error(f"Exception while computing rewards for normal samples, setting all rewards to 0: {e}")
            logger.error(traceback.format_exc())
            results = [0.0] * len(sequences_strs)

        return results
