#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Reward Manager - V1 Naive version

A reward manager based on intermediate-fact coverage within a trajectory
(the Naive version), using within-group normalization.

Core features:
- Concurrently computes the end-to-end reward (browsecomp_zh.evaluate) and the trajectory coverage reward
- Uses within-group normalization (GRPO-style) to remove the effect of question difficulty
- Handles format errors and overlong rollouts
- Overlong rollouts participate in advantage normalization but not in the gradient backward pass

Version: 1.0.0
"""

import asyncio
import os
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
from verl.utils.reward_score.trajectory_reward_v1_naive import (
    TrajectoryFactIdentifierV1,
    calculate_coverage_rate,
    normalize_coverage_within_group,
    extract_mid_facts_from_ground_truth,
)
from verl.utils.reward_score.rewards import EvalItemBuilder
import yaml
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def single_compute_combined_score(
    config: Dict[str, Any],
    prompts_str: str,
    sequences_str: str,
    ground_truth: Dict[str, Any],
    data_source: str,
    task_extra_info: Dict[str, Any],
    executor: ThreadPoolExecutor,
    timeout: float = 6000.0,
) -> Dict[str, Any]:
    """Concurrently compute the end-to-end reward and the trajectory coverage reward for a single sample.

    Note (2026-01-06):
    - Format-error checking is no longer done here (moved earlier, into sglang)
    - Only the end-to-end reward and coverage rate are computed

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
        A dict containing end_to_end_score, coverage_rate, mid_facts_identified.
    """
    loop = asyncio.get_running_loop()

    try:
        # Concurrent task 1: end-to-end reward (browsecomp_zh.evaluate)
        async def compute_end_to_end():
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
                logger.warning(f"End-to-end reward computation timed out")
                return 0.0
            except Exception as e:
                logger.error(f"End-to-end reward computation failed: {e}")
                return 0.0

        # Concurrent task 2: trajectory coverage reward
        async def compute_coverage():
            try:
                # Extract the trajectory and query
                item = EvalItemBuilder.build_from_sequences(sequences_str, prompts_str)
                query = item.get('user_question', '')
                trajectory = item.get('readable_trajectory', sequences_str)

                # Extract the list of intermediate facts
                mid_facts_ground_truth = extract_mid_facts_from_ground_truth(ground_truth)
                if not mid_facts_ground_truth:
                    return {
                        'coverage_rate': 0.0,
                        'mid_facts_identified': [],
                    }

                # Identify intermediate facts within the trajectory
                identifier = TrajectoryFactIdentifierV1(config)
                mid_facts_identified = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        identifier.identify_facts,
                        trajectory,
                        query,
                        mid_facts_ground_truth,
                    ),
                    timeout=timeout,
                )

                # Compute the coverage rate
                coverage_rate = calculate_coverage_rate(mid_facts_identified, mid_facts_ground_truth)

                return {
                    'coverage_rate': coverage_rate,
                    'mid_facts_identified': mid_facts_identified,
                }
            except asyncio.TimeoutError:
                logger.warning(f"Trajectory coverage computation timed out")
                return {
                    'coverage_rate': 0.0,
                    'mid_facts_identified': [],
                }
            except Exception as e:
                logger.error(f"Trajectory coverage computation failed: {e}")
                return {
                    'coverage_rate': 0.0,
                    'mid_facts_identified': [],
                }

        # Run both tasks concurrently
        end_to_end_task = compute_end_to_end()
        coverage_task = compute_coverage()

        end_to_end_score, coverage_result = await asyncio.gather(
            end_to_end_task, coverage_task, return_exceptions=True
        )

        # Handle exceptions
        if isinstance(end_to_end_score, BaseException):
            logger.error(f"End-to-end reward exception: {end_to_end_score}")
            end_to_end_score = 0.0

        if isinstance(coverage_result, BaseException):
            logger.error(f"Coverage computation exception: {coverage_result}")
            coverage_result = {
                'coverage_rate': 0.0,
                'mid_facts_identified': [],
            }

        return {
            'end_to_end_score': end_to_end_score,
            'coverage_rate': coverage_result['coverage_rate'],
            'mid_facts_identified': coverage_result['mid_facts_identified'],
        }

    except Exception as e:
        logger.error(f"Combined reward computation failed: {e}")
        logger.error(traceback.format_exc())
        return {
            'end_to_end_score': 0.0,
            'coverage_rate': 0.0,
            'mid_facts_identified': [],
        }


async def parallel_compute_combined_score_async(
    config: Dict[str, Any],
    prompts_strs: List[str],
    sequences_strs: List[str],
    ground_truths: List[Dict[str, Any]],
    data_sources: List[str],
    task_extra_infos: Optional[List[Dict[str, Any]]] = None,
    num_processes: int = 64,
) -> List[Dict[str, Any]]:
    """Compute combined rewards concurrently for a batch.

    Note (2026-01-06):
    - No longer returns has_format_error (moved earlier, into sglang)

    Args:
        config: config dict
        prompts_strs: list of prompt strings
        sequences_strs: list of generated sequence strings
        ground_truths: list of ground-truth entries
        data_sources: list of data sources
        task_extra_infos: list of extra task info dicts
        num_processes: number of concurrent workers

    Returns:
        List of reward info dicts.
    """
    results = []
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        if task_extra_infos is None:
            task_extra_infos = [{}] * len(data_sources)

        # Create all tasks
        tasks_async = [
            single_compute_combined_score(
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
            # Handle a failed or timed-out task
            processed_results.append({
                'end_to_end_score': 0.0,
                'coverage_rate': 0.0,
                'mid_facts_identified': [],
            })
        else:
            processed_results.append(result)

    return processed_results


@register("apiprimedapotrajectoryv1naive")
class ApiPrimeDapoTrajectoryV1NaiveRewardManager(AbstractRewardManager):
    """
    Trajectory Reward Manager V1 Naive version

    Features:
    - Concurrently computes the end-to-end reward and the trajectory coverage reward
    - Uses within-group normalization (GRPO-style) to remove the effect of question difficulty
    - Handles format errors and overlong rollouts

    Where data is stored:
    - mid_facts_ground_truth_list: ground_truth['mid_facts']
    - Identified intermediate entities and coverage: DataProto.non_tensor_batch['trajectory_reward_info']
        - mid_facts_identified: list of identified intermediate facts
        - coverage_rate_raw: raw coverage rate
        - coverage_rate_normalized: coverage rate after normalization
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

        # By default loads verl/trainer/config/reward_dapo.yaml at a fixed
        # location relative to this file; overridable via the
        # REWARD_DAPO_CONFIG_PATH environment variable
        config_path = os.environ.get(
            "REWARD_DAPO_CONFIG_PATH",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "trainer", "config", "reward_dapo.yaml",
            ),
        )
        self.config = yaml.safe_load(open(config_path, "r"))

        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert (
                self.max_resp_len is not None
            ), f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            assert (
                self.max_resp_len >= self.overlong_buffer_cfg.len
            ), "max_resp_len must be larger than overlong_buffer.len"

        logger.info("[OK] ApiPrimeDapoTrajectoryV1NaiveRewardManager initialized")

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute the final reward tensor.

        New logic (2026-01-06):
        1. **Upfront abnormality detection**: reads abnormality flags from the sglang rollout
           - If a sample has any abnormality (tool parse error, repeated query, overlong, etc.), it gets reward = 0 directly
           - Overlong / tool-abnormality detection is no longer done inside the reward manager

        2. **Reward computation for normal samples**:
           - Only samples without abnormalities go through reward computation
           - Within-group coverage normalization (GRPO-style)
           - Naive-version reward formula:
             * Correct answer (end_to_end_score >= 1): reward = 1.0
             * Incorrect answer (end_to_end_score < 1): reward = alpha * coverage_rate_normed

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

        # Batch reward computation
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        # ========================================
        # === Step 1: extract abnormality flags from rollout_extra_info ===
        # ========================================
        #
        # rollout_extra_info structure (from the sglang rollout):
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
        #             ...
        #         }
        #     },
        #     ...
        # ]
        #
        rollout_extra_info_list = data.non_tensor_batch.get("rollout_extra_info", [{}] * len(data))

        # Extract abnormality flags
        abnormal_flags_list = []
        trajectory_metrics_list = []
        for rollout_extra_info in rollout_extra_info_list:
            if isinstance(rollout_extra_info, dict):
                abnormal_flags_list.append(rollout_extra_info.get("abnormal_flags", {}))
                trajectory_metrics_list.append(rollout_extra_info.get("trajectory_metrics", {}))
            else:
                # Backward compatibility with old-format data (no rollout_extra_info)
                abnormal_flags_list.append({})
                trajectory_metrics_list.append({})

        # ========================================
        # === Step 2: only compute rewards for normal samples ===
        # ========================================
        #
        # Abnormality classification policy:
        # - Discard: search_error / tool_parse_error / exceed_max_tokens / exceed_max_turns
        # - Handled normally: excessive_tool_calls_per_turn / repeated_query
        #
        from verl.utils.reward_score.trajectory_reward_v2_cgrpo import classify_abnormal_trajectory

        normal_indices = []
        for i, abnormal_flags in enumerate(abnormal_flags_list):
            should_give_zero_reward, should_discard, abnormal_types = classify_abnormal_trajectory(abnormal_flags)

            if should_discard:
                continue

            if not should_give_zero_reward:
                normal_indices.append(i)

        logger.info(f"Batch size: {len(data)}, normal samples: {len(normal_indices)}, abnormal samples: {len(data) - len(normal_indices)}")

        # ========================================
        # === Pre-initialize reward_extra_info for all samples ===
        # ========================================
        #
        # Important fix (2026-01-01):
        # The old code appended extra_info for normal samples first, then
        # for abnormal samples, which made reward_extra_info's order
        # inconsistent with reward_tensor's order!
        #
        # The correct approach: initialize extra_info for all samples in
        # the original data order.
        #
        batch_size = len(data)
        end_to_end_scores_all = [0.0] * batch_size
        coverage_rates_all = [0.0] * batch_size
        normalized_coverage_rates_all = [0.0] * batch_size
        final_rewards_all = [0.0] * batch_size
        abnormal_types_all = [[] for _ in range(batch_size)]
        mid_facts_identified_list_all = [[] for _ in range(batch_size)]

        # First mark the abnormality types for abnormal samples
        for i, abnormal_flags in enumerate(abnormal_flags_list):
            if i not in normal_indices:
                abnormal_types_all[i] = [k for k, v in abnormal_flags.items() if v]

        # Only compute rewards if there are normal samples
        if len(normal_indices) > 0:
            # Extract data for normal samples
            normal_prompts_str = [self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True) for i in normal_indices]
            normal_sequences_str = [sequences_str[i] for i in normal_indices]

            # Extract ground_truth (compatible with old and new data structures)
            ground_truth_list = []
            for data_item in data:
                if "reward_model" in data_item.non_tensor_batch:
                    ground_truth_list.append(data_item.non_tensor_batch["reward_model"]["ground_truth"])
                else:
                    # Backward compatibility
                    ground_truth_list.append({})

            normal_ground_truth = [ground_truth_list[i] for i in normal_indices]
            normal_data_sources = [data_sources[i] for i in normal_indices]

            # Extract task_extra_infos and the question field
            #
            # Note (2026-01-01):
            # browsecomp_zh.compute_score requires extra_info to contain a
            # 'question' field. Data sources:
            #   1. data.non_tensor_batch["extra_info"] (the original data's extra_info, contains question)
            #   2. data.non_tensor_batch["question"] (a directly stored question field)
            #
            task_extra_infos = []

            # Extract the original data's extra_info (contains question)
            original_extra_info_list = data.non_tensor_batch.get("extra_info", [{}] * len(data))

            # Extract the question field (compatible with both data formats)
            question_list = []
            for i, data_item in enumerate(data):
                # Prefer non_tensor_batch["question"]
                question = data_item.non_tensor_batch.get("question", None)

                # If missing, try the original extra_info
                if question is None and isinstance(original_extra_info_list[i], dict):
                    question = original_extra_info_list[i].get("question", None)

                question_list.append(question)

            for i in normal_indices:
                # Build task_extra_info, ensuring it contains a question field
                task_extra_info = {}

                # If the original extra_info has info, copy it over first
                if isinstance(original_extra_info_list[i], dict):
                    task_extra_info.update(original_extra_info_list[i])

                # Ensure a question field is present (required by browsecomp_zh.compute_score)
                if question_list[i] is not None:
                    task_extra_info["question"] = question_list[i]

                task_extra_infos.append(task_extra_info)

            # Compute the combined reward for normal samples
            results = self.verify_normal_samples(
                normal_prompts_str,
                normal_sequences_str,
                normal_ground_truth,
                normal_data_sources,
                task_extra_infos
            )

            # Extract the end-to-end reward and coverage rate
            end_to_end_scores = [r['end_to_end_score'] for r in results]
            coverage_rates = [r['coverage_rate'] for r in results]
            mid_facts_identified_list = [r['mid_facts_identified'] for r in results]

            # ========================================
            # === Step 3: within-group coverage normalization (GRPO-style) ===
            # ========================================
            #
            # Note (2026-01-01):
            # Normalization must be done grouped by uid (similar to GRPO advantage computation)
            # - Same uid = different rollouts of the same query
            # - Normalization only happens within the same uid group; different uids don't affect each other
            #
            # Extract the uid corresponding to normal samples
            uid_list_all = data.non_tensor_batch.get("uid", None)
            if uid_list_all is None:
                raise ValueError("data.non_tensor_batch is missing the 'uid' field, cannot perform within-group normalization")

            normal_uid_list = [uid_list_all[i] for i in normal_indices]

            # Call the normalization function (passing uid_list)
            normalized_coverage_rates = normalize_coverage_within_group(coverage_rates, normal_uid_list)

            # ========================================
            # === Step 4: compute the final reward (Naive version) ===
            # ========================================
            alpha = self.config.get('trajectory_reward_alpha', 0.5)

            for idx, i in enumerate(normal_indices):
                # Naive-version reward computation
                if end_to_end_scores[idx] >= 1.0:
                    # Correct answer
                    final_reward = 1.0
                else:
                    # Incorrect answer, use coverage as partial credit
                    final_reward = alpha * normalized_coverage_rates[idx]

                # Set the reward tensor
                reward_tensor[i, valid_response_length[i].item() - 1] = final_reward

                # Update the corresponding position's extra_info (preserving the original order)
                end_to_end_scores_all[i] = end_to_end_scores[idx]
                coverage_rates_all[i] = coverage_rates[idx]
                normalized_coverage_rates_all[i] = normalized_coverage_rates[idx]
                final_rewards_all[i] = final_reward
                mid_facts_identified_list_all[i] = mid_facts_identified_list[idx]
                # abnormal_types_all[i] was already initialized to an empty list

                # Print debug info
                data_source = data_sources[i]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    logger.info(
                        f"[Normal Sample {i}] "
                        f"end_to_end={end_to_end_scores[idx]:.3f}, "
                        f"coverage_raw={coverage_rates[idx]:.3f}, "
                        f"coverage_norm={normalized_coverage_rates[idx]:.3f}, "
                        f"final_reward={final_reward:.3f}, "
                        f"mid_facts_identified={len(mid_facts_identified_list[idx])}"
                    )

        # ========================================
        # === Step 5: print info for abnormal samples (reward is already 0) ===
        # ========================================
        for i in range(batch_size):
            if i not in normal_indices:
                # Print debug info
                data_source = data_sources[i]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    logger.info(
                        f"[Abnormal Sample {i}] "
                        f"abnormal_types={abnormal_types_all[i]}, "
                        f"final_reward=0.0"
                    )

        # ========================================
        # === Step 6: fill reward_extra_info in the original order ===
        # ========================================
        #
        # Important: fill extra_info here in the original data order (0, 1, 2, ...)
        # to keep it consistent with reward_tensor's order
        #
        reward_extra_info['end_to_end_score'] = end_to_end_scores_all
        reward_extra_info['coverage_rate_raw'] = coverage_rates_all
        reward_extra_info['coverage_rate_normalized'] = normalized_coverage_rates_all
        reward_extra_info['final_reward'] = final_rewards_all
        reward_extra_info['abnormal_types'] = abnormal_types_all

        # Save the trajectory reward info to non_tensor_batch (for logging)
        trajectory_reward_info = []
        for i in range(batch_size):
            trajectory_reward_info.append({
                'mid_facts_identified': mid_facts_identified_list_all[i],
                'coverage_rate_raw': coverage_rates_all[i],
                'coverage_rate_normalized': normalized_coverage_rates_all[i],
            })

        data.non_tensor_batch['trajectory_reward_info'] = np.array(trajectory_reward_info, dtype=object)

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
    ) -> List[Dict[str, Any]]:
        """Compute the combined reward for normal samples.

        This is a helper method that only processes normal samples (those
        without abnormality flags).

        Args:
            prompts_strs: list of prompt strings
            sequences_strs: list of generated sequence strings
            ground_truths: list of ground-truth entries
            data_sources: list of data sources
            task_extra_infos: list of extra task info dicts

        Returns:
            List of reward info dicts.
        """
        try:
            results = asyncio.run(
                parallel_compute_combined_score_async(
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
            logger.error("Normal-sample reward computation timed out globally! Setting all rewards to 0")
            results = [{
                'end_to_end_score': 0.0,
                'coverage_rate': 0.0,
                'mid_facts_identified': [],
            }] * len(sequences_strs)
        except Exception as e:
            logger.error(f"Normal-sample reward computation exception, setting all rewards to 0: {e}")
            logger.error(traceback.format_exc())
            results = [{
                'end_to_end_score': 0.0,
                'coverage_rate': 0.0,
                'mid_facts_identified': [],
            }] * len(sequences_strs)

        return results
