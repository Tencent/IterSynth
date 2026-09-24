#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Reward Manager - V2 C-GRPO version

A reward manager based on the CaRR (Citation-aware Rubric Reward) framework.

Core features (three-step evaluation):
1. Step 1: Hidden Entity Identification - identify hidden entities in the Agent's answer
2. Step 2: Citation-based Rubric Judgment - verify whether constraints are backed by citations
3. Step 3: Evidence Connectivity - check the connectivity of the evidence chain (BFS starting from the answer entity)

Final reward formula (C-GRPO):
R_i = (1 - alpha) * R_o^(H_i) + alpha * R_o^(H_i) * R_hat_r^(H_i)

Where:
- R_o: Outcome Reward (whether the correct answer was found, 0 or 1)
- R_r: Rubric Reward (reasoning quality, |R_connect| / |R_q|)
- R_hat_r: within-group normalized Rubric Reward

Version: 2.0.0
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Optional, List, Dict, Any
import traceback
from collections import defaultdict
import os

import torch
import numpy as np
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score.browsecomp_zh import compute_score as browsecomp_compute_score
from verl.utils.reward_score.trajectory_reward_v2_cgrpo import (
    CGRPORewardCalculator,
    extract_constraints_and_entities_from_ground_truth,
    normalize_rubric_reward_within_group,
    calculate_cgrpo_mixed_reward,
    RubricRewardError,
)
import yaml
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def single_compute_cgrpo_score(
    config: Dict[str, Any],
    prompts_str: str,
    sequences_str: str,
    ground_truth: Dict[str, Any],
    data_source: str,
    task_extra_info: Dict[str, Any],
    executor: ThreadPoolExecutor,
    timeout: float = 6000.0,
) -> Dict[str, Any]:
    """Compute the C-GRPO reward for a single sample.

    Computed concurrently:
    1. End-to-end reward (Outcome Reward)
    2. Rubric Reward (three-step evaluation)

    Note:
    - sequences_str should be the full trajectory decoded from response_ids
    - <answer> tags and <tool_response> tags will be extracted from sequences_str

    Fault tolerance:
    - All exceptions are caught and return a score of 0
    - Timeouts are caught and return a score of 0
    - Ensures a single sample's exception doesn't fail the whole batch

    Args:
        config: config dict
        prompts_str: the prompt string
        sequences_str: the sequence string decoded from response_ids (the full trajectory)
        ground_truth: ground-truth data
        data_source: the data source
        task_extra_info: extra task info
        executor: the thread pool executor
        timeout: timeout in seconds

    Returns:
        A dict containing end_to_end_score, rubric_reward, rubric_details, error_code.
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
                # Make sure the returned value is a valid float
                try:
                    score = float(result) if result is not None else 0.0
                    # Check for NaN or Inf
                    if not np.isfinite(score):
                        logger.warning(f"End-to-end reward computation returned an invalid value: {score}, setting to 0")
                        return 0.0
                    return score
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert the end-to-end reward return value: {result}, error: {e}")
                    return 0.0

            except asyncio.TimeoutError:
                logger.warning(f"End-to-end reward computation timed out (timeout={timeout}s)")
                return 0.0
            except Exception as e:
                logger.error(f"End-to-end reward computation failed: {type(e).__name__}: {e}")
                logger.error(traceback.format_exc())
                return 0.0

        # Concurrent task 2: Rubric Reward (three-step evaluation)
        async def compute_rubric_reward():
            try:
                # Extract the query (fault-tolerant)
                try:
                    query = task_extra_info.get('question', '') if isinstance(task_extra_info, dict) else ''
                except Exception as e:
                    logger.warning(f"Failed to extract question: {e}")
                    query = ''

                # sequences_str is the full trajectory decoded from response_ids
                # It contains <tool_response> and <answer> tags
                trajectory = sequences_str
                response = sequences_str  # response also uses the full sequences_str; step2 extracts <answer> from it

                # Extract constraints and entities from task_extra_info
                # (i.e. extra_info) (fault-tolerant)
                # Note: constraints and entities live in the extra_info
                # field, not in ground_truth
                try:
                    constraints, entities = extract_constraints_and_entities_from_ground_truth(task_extra_info)
                except Exception as e:
                    logger.error(f"Failed to extract constraints and entities: {e}")
                    logger.error(traceback.format_exc())
                    constraints, entities = [], {}

                # Important optimization (2026-01-28):
                # When constraints or entities are empty, degrade to
                # computing only the Outcome Reward. In this case the
                # Rubric Reward should be 1.0, equivalent to no penalty
                # (i.e. equivalent to looking only at the end-to-end reward).
                #
                # Rationale:
                # 1. Empty constraints mean reasoning quality can't be
                #    evaluated, so the Agent shouldn't be penalized.
                # 2. Rubric Reward = 1.0 makes the C-GRPO formula degrade to:
                #    R = (1-alpha)*R_o + alpha*R_o*1 = (1-alpha+alpha)*R_o = R_o
                # 3. Avoids a division-by-zero problem (originally
                #    rubric_reward = n_connected / n_total)
                if not constraints or not entities:
                    logger.info(
                        f"Constraints or entities are empty (constraints={len(constraints)}, entities={len(entities)}), "
                        f"degrading to computing only the Outcome Reward, setting Rubric Reward = 1.0"
                    )
                    return {
                        'rubric_reward': 1.0,  # Key change: from 0.0 to 1.0
                        'rubric_details': {
                            'r_identify': 1.0,  # Degraded mode: all ratios are 1.0
                            'r_support': 1.0,
                            'r_connect': 1.0,
                            'n_total': 0,
                            'n_identified': 0,
                            'n_supported': 0,
                            'n_connected': 0,
                            'error_code': RubricRewardError.NO_CONSTRAINTS,
                        }
                    }

                # Create the C-GRPO calculator (fault-tolerant)
                try:
                    calculator = CGRPORewardCalculator(config)
                except Exception as e:
                    logger.error(f"Failed to create CGRPORewardCalculator: {e}")
                    logger.error(traceback.format_exc())
                    return {
                        'rubric_reward': 0.0,
                        'rubric_details': {'error_code': 'calculator_init_failed', 'error_msg': str(e)},
                    }

                # Compute the Rubric Reward (fault-tolerant)
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            executor,
                            partial(
                                calculator.calculate_rubric_reward,
                                query,
                                constraints,
                                entities,
                                trajectory,
                                response,
                            ),
                        ),
                        timeout=timeout,
                    )

                    # Validate the returned result
                    if not isinstance(result, dict):
                        logger.error(f"Rubric Reward returned a value of the wrong type: {type(result)}")
                        return {
                            'rubric_reward': 0.0,
                            'rubric_details': {'error_code': 'invalid_result_type'},
                        }

                    # Check that rubric_reward is a valid float
                    rubric_reward = result.get('rubric_reward', 0.0)
                    try:
                        rubric_reward = float(rubric_reward)
                        if not np.isfinite(rubric_reward):
                            logger.warning(f"Rubric Reward returned an invalid value: {rubric_reward}, setting to 0")
                            rubric_reward = 0.0
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to convert Rubric Reward: {rubric_reward}, error: {e}")
                        rubric_reward = 0.0

                    return {
                        'rubric_reward': rubric_reward,
                        'rubric_details': result,
                    }

                except asyncio.TimeoutError:
                    logger.warning(f"Rubric Reward computation timed out (timeout={timeout}s)")
                    return {
                        'rubric_reward': 0.0,
                        'rubric_details': {'error_code': 'timeout'},
                    }

            except Exception as e:
                logger.error(f"Rubric Reward computation failed: {type(e).__name__}: {e}")
                logger.error(traceback.format_exc())
                return {
                    'rubric_reward': 0.0,
                    'rubric_details': {'error_code': 'exception', 'error_msg': str(e)},
                }

        # Run both tasks concurrently (fault-tolerant)
        try:
            end_to_end_task = compute_end_to_end()
            rubric_task = compute_rubric_reward()

            end_to_end_score, rubric_result = await asyncio.gather(
                end_to_end_task, rubric_task, return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Concurrent task execution failed: {e}")
            logger.error(traceback.format_exc())
            end_to_end_score = 0.0
            rubric_result = {
                'rubric_reward': 0.0,
                'rubric_details': {'error_code': 'gather_failed'},
            }

        # Handle exceptions (fault-tolerant)
        if isinstance(end_to_end_score, BaseException):
            logger.error(f"End-to-end reward exception: {type(end_to_end_score).__name__}: {end_to_end_score}")
            logger.error(traceback.format_exc())
            end_to_end_score = 0.0

        if isinstance(rubric_result, BaseException):
            logger.error(f"Rubric Reward exception: {type(rubric_result).__name__}: {rubric_result}")
            logger.error(traceback.format_exc())
            rubric_result = {
                'rubric_reward': 0.0,
                'rubric_details': {'error_code': 'exception'},
            }

        # Make sure the return value has the correct format
        try:
            return {
                'end_to_end_score': float(end_to_end_score) if end_to_end_score is not None else 0.0,
                'rubric_reward': float(rubric_result.get('rubric_reward', 0.0)) if isinstance(rubric_result, dict) else 0.0,
                'rubric_details': rubric_result.get('rubric_details', {}) if isinstance(rubric_result, dict) else {},
            }
        except Exception as e:
            logger.error(f"Failed to construct the return result: {e}")
            logger.error(traceback.format_exc())
            return {
                'end_to_end_score': 0.0,
                'rubric_reward': 0.0,
                'rubric_details': {'error_code': 'result_construction_failed'},
            }

    except Exception as e:
        logger.error(f"C-GRPO reward computation top-level exception: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        return {
            'end_to_end_score': 0.0,
            'rubric_reward': 0.0,
            'rubric_details': {'error_code': 'top_level_exception', 'error_msg': str(e)},
        }


async def parallel_compute_cgrpo_score_async(
    config: Dict[str, Any],
    prompts_strs: List[str],
    sequences_strs: List[str],
    ground_truths: List[Dict[str, Any]],
    data_sources: List[str],
    task_extra_infos: Optional[List[Dict[str, Any]]] = None,
    num_processes: int = 64,
) -> List[Dict[str, Any]]:
    """Compute C-GRPO rewards concurrently for a batch.

    Fault tolerance:
    - Each sample's exceptions are handled independently, without affecting other samples
    - All exceptions result in a score of 0
    - Ensures the returned list length matches the input length

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
    try:
        # Validate parameters
        batch_size = len(prompts_strs)
        if not (len(sequences_strs) == len(ground_truths) == len(data_sources) == batch_size):
            logger.error(
                f"Input list lengths are inconsistent: prompts={len(prompts_strs)}, "
                f"sequences={len(sequences_strs)}, ground_truths={len(ground_truths)}, "
                f"data_sources={len(data_sources)}"
            )
            # Return an all-zero result
            return [{
                'end_to_end_score': 0.0,
                'rubric_reward': 0.0,
                'rubric_details': {'error_code': 'input_length_mismatch'},
            }] * batch_size

        if task_extra_infos is None:
            task_extra_infos = [{}] * batch_size
        elif len(task_extra_infos) != batch_size:
            logger.warning(
                f"task_extra_infos length mismatch: {len(task_extra_infos)} vs {batch_size}, padding with empty dicts"
            )
            task_extra_infos = list(task_extra_infos) + [{}] * (batch_size - len(task_extra_infos))

        results = []
        try:
            with ThreadPoolExecutor(max_workers=num_processes) as executor:
                # Create all tasks (fault-tolerant)
                try:
                    tasks_async = [
                        single_compute_cgrpo_score(
                            config, prompts_str, sequences_str, ground_truth,
                            data_source, task_extra_info, executor, timeout=6000.0
                        )
                        for prompts_str, sequences_str, ground_truth, data_source, task_extra_info in zip(
                            prompts_strs, sequences_strs, ground_truths,
                            data_sources, task_extra_infos
                        )
                    ]
                except Exception as e:
                    logger.error(f"Failed to create the task list: {e}")
                    logger.error(traceback.format_exc())
                    return [{
                        'end_to_end_score': 0.0,
                        'rubric_reward': 0.0,
                        'rubric_details': {'error_code': 'task_creation_failed'},
                    }] * batch_size

                # Run all tasks concurrently (fault-tolerant)
                try:
                    # return_exceptions=True ensures a single task's
                    # failure doesn't affect the others
                    results = await asyncio.gather(*tasks_async, return_exceptions=True)
                except Exception as e:
                    logger.error(f"Failed to run tasks concurrently: {e}")
                    logger.error(traceback.format_exc())
                    return [{
                        'end_to_end_score': 0.0,
                        'rubric_reward': 0.0,
                        'rubric_details': {'error_code': 'gather_failed'},
                    }] * batch_size

        except Exception as e:
            logger.error(f"Thread pool execution failed: {e}")
            logger.error(traceback.format_exc())
            return [{
                'end_to_end_score': 0.0,
                'rubric_reward': 0.0,
                'rubric_details': {'error_code': 'executor_failed'},
            }] * batch_size

        # Process results (fault-tolerant)
        processed_results = []
        for i, result in enumerate(results):
            try:
                if isinstance(result, (Exception, BaseException)):
                    logger.error(f"Sample {i} returned an exception: {type(result).__name__}: {result}")
                    logger.error(traceback.format_exc())
                    processed_results.append({
                        'end_to_end_score': 0.0,
                        'rubric_reward': 0.0,
                        'rubric_details': {'error_code': 'sample_exception', 'error_msg': str(result)},
                    })
                elif result is None:
                    logger.warning(f"Sample {i} returned None")
                    processed_results.append({
                        'end_to_end_score': 0.0,
                        'rubric_reward': 0.0,
                        'rubric_details': {'error_code': 'result_is_none'},
                    })
                elif not isinstance(result, dict):
                    logger.error(f"Sample {i} returned the wrong type: {type(result)}")
                    processed_results.append({
                        'end_to_end_score': 0.0,
                        'rubric_reward': 0.0,
                        'rubric_details': {'error_code': 'invalid_result_type'},
                    })
                else:
                    # Validate required fields
                    if 'end_to_end_score' not in result or 'rubric_reward' not in result:
                        logger.warning(f"Sample {i} is missing required fields, padding with 0")
                        result.setdefault('end_to_end_score', 0.0)
                        result.setdefault('rubric_reward', 0.0)
                        result.setdefault('rubric_details', {'error_code': 'missing_fields'})
                    processed_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process the result for sample {i}: {e}")
                logger.error(traceback.format_exc())
                processed_results.append({
                    'end_to_end_score': 0.0,
                    'rubric_reward': 0.0,
                    'rubric_details': {'error_code': 'result_processing_failed'},
                })

        # Make sure the returned list has the correct length
        if len(processed_results) != batch_size:
            logger.error(
                f"Returned result count mismatch: expected {batch_size}, got {len(processed_results)}"
            )
            # Pad or truncate
            if len(processed_results) < batch_size:
                processed_results += [{
                    'end_to_end_score': 0.0,
                    'rubric_reward': 0.0,
                    'rubric_details': {'error_code': 'result_padding'},
                }] * (batch_size - len(processed_results))
            else:
                processed_results = processed_results[:batch_size]

        return processed_results

    except Exception as e:
        logger.error(f"Batch computation top-level exception: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        # Return an all-zero result
        batch_size = len(prompts_strs) if prompts_strs else 1
        return [{
            'end_to_end_score': 0.0,
            'rubric_reward': 0.0,
            'rubric_details': {'error_code': 'top_level_exception', 'error_msg': str(e)},
        }] * batch_size


@register("apiprimedapotrajectoryv2cgrpo")
class ApiPrimeDapoTrajectoryV2CGRPORewardManager(AbstractRewardManager):
    """
    Trajectory Reward Manager V2 C-GRPO version

    Features:
    - Three-step evaluation: Entity Identification -> Citation Judgment -> Evidence Connectivity
    - C-GRPO mixed-reward formula
    - Within-group normalization (GRPO-style)

    Data source (ground_truth fields):
    - constraints: list of constraints (e.g. ["C1. <E1>graduated from a music conservatory<E2>", ...])
    - entities: entity mapping (e.g. {"E0": "Li Ronghao", "E1": "Shan Yichun", ...})
    - target: the correct answer (used for the Outcome Reward)

    Where output is stored:
    - DataProto.non_tensor_batch['cgrpo_reward_info']:
        - rubric_reward_raw: the raw Rubric Reward
        - rubric_reward_normalized: the Rubric Reward after normalization
        - rubric_details: detailed results of the three-step evaluation
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

        # By default loads
        # verl/trainer/config/reward_dapo_trajectory_v2_cgrpo.yaml at a
        # fixed location relative to this file; overridable via the
        # REWARD_CGRPO_CONFIG_PATH environment variable
        config_path = os.environ.get(
            "REWARD_CGRPO_CONFIG_PATH",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "trainer", "config", "reward_dapo_trajectory_v2_cgrpo.yaml",
            ),
        )
        logger.info(f"Loading C-GRPO config file: {config_path}")
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

        logger.info("[OK] ApiPrimeDapoTrajectoryV2CGRPORewardManager initialized")

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute the final reward tensor.

        C-GRPO reward computation flow:
        1. Extract abnormality flags from rollout_extra_info
        2. Only compute rewards for normal samples
        3. Compute the Outcome Reward (end-to-end) and the Rubric Reward (three-step evaluation)
        4. Normalize the Rubric Reward within groups
        5. Compute the C-GRPO mixed reward

        C-GRPO mixed-reward formula:
        R_i = (1 - alpha) * R_o + alpha * R_o * R_hat_r

        Where:
        - R_o: Outcome Reward (whether the answer is correct, 0 or 1)
        - R_hat_r: the normalized Rubric Reward
        - alpha: the balancing parameter (default 0.3)

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

        # Decode from response_ids to get the full trajectory string
        # This sequences_str contains the full Agent trajectory, including:
        # - <tool_call> and <tool_response> tags
        # - the final answer inside <answer> tags
        # - citations in [citation:x] format
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        # ========================================
        # === Step 1: extract abnormality flags from rollout_extra_info ===
        # ========================================
        rollout_extra_info_list = data.non_tensor_batch.get("rollout_extra_info", [{}] * len(data))

        abnormal_flags_list = []
        trajectory_metrics_list = []
        for rollout_extra_info in rollout_extra_info_list:
            if isinstance(rollout_extra_info, dict):
                abnormal_flags_list.append(rollout_extra_info.get("abnormal_flags", {}))
                trajectory_metrics_list.append(rollout_extra_info.get("trajectory_metrics", {}))
            else:
                abnormal_flags_list.append({})
                trajectory_metrics_list.append({})

        # ========================================
        # === Step 2: check the answer format + identify abnormal samples ===
        # ========================================
        #
        # Abnormality classification policy:
        # - Discard: search_error / tool_parse_error / exceed_max_tokens / exceed_max_turns
        # - Offending round zeroed: excessive_tool_calls_per_turn (only applies to the PS Pipeline path)
        # - Handled normally: repeated_query
        #
        # Note: the no_answer_tag abnormality is handled separately here,
        # not through classify_abnormal_trajectory()
        #
        from verl.utils.reward_score.trajectory_reward_v2_cgrpo import classify_abnormal_trajectory

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
                logger.warning(f"Sample {i} failed to check for the <answer> tag: {e}")

            # Only compute the Rubric Reward for samples that shouldn't
            # get reward=0 and that have an answer tag
            if not should_give_zero_reward and has_answer_tag:
                normal_indices.append(i)
            else:
                if not has_answer_tag:
                    abnormal_flags_list[i]["no_answer_tag"] = True

        logger.info(
            f"Batch size: {len(data)}, "
            f"normal samples: {len(normal_indices)}, "
            f"abnormal samples: {len(data) - len(normal_indices)}"
        )

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

        # Initialize the result storage for all samples
        all_end_to_end_scores = [0.0] * batch_size
        all_outcome_rewards = [0.0] * batch_size
        all_rubric_rewards_raw = [0.0] * batch_size
        all_rubric_rewards_normalized = [0.0] * batch_size
        all_final_rewards = [0.0] * batch_size
        all_abnormal_types = [[] for _ in range(batch_size)]
        all_rubric_details = [{} for _ in range(batch_size)]
        all_r_identify = [0.0] * batch_size
        all_r_support = [0.0] * batch_size
        all_r_connect = [0.0] * batch_size
        all_n_connected = [0] * batch_size
        all_n_total = [0] * batch_size

        # First mark the abnormality types for abnormal samples
        for i, abnormal_flags in enumerate(abnormal_flags_list):
            if i not in normal_indices:
                all_abnormal_types[i] = [k for k, v in abnormal_flags.items() if v]

        # Only compute rewards if there are normal samples
        if len(normal_indices) > 0:
            try:
                # Extract data for normal samples (fault-tolerant)
                try:
                    normal_prompts_str = [self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True) for i in normal_indices]
                    normal_sequences_str = [sequences_str[i] for i in normal_indices]
                except Exception as e:
                    logger.error(f"Failed to extract data for normal samples: {e}")
                    logger.error(traceback.format_exc())
                    normal_prompts_str = [""] * len(normal_indices)
                    normal_sequences_str = [sequences_str[i] if i < len(sequences_str) else "" for i in normal_indices]

                # Extract ground_truth (fault-tolerant)
                try:
                    ground_truth_list = []
                    for data_item in data:
                        try:
                            if "reward_model" in data_item.non_tensor_batch:
                                ground_truth_list.append(data_item.non_tensor_batch["reward_model"]["ground_truth"])
                            else:
                                ground_truth_list.append({})
                        except Exception as e:
                            logger.warning(f"Failed to extract a single ground_truth: {e}")
                            ground_truth_list.append({})

                    normal_ground_truth = [ground_truth_list[i] for i in normal_indices]
                    normal_data_sources = [data_sources[i] for i in normal_indices]
                except Exception as e:
                    logger.error(f"Failed to extract ground_truth: {e}")
                    logger.error(traceback.format_exc())
                    normal_ground_truth = [{}] * len(normal_indices)
                    normal_data_sources = [data_sources[i] if i < len(data_sources) else "" for i in normal_indices]

                # Extract task_extra_infos (fault-tolerant)
                try:
                    task_extra_infos = []
                    original_extra_info_list = data.non_tensor_batch.get("extra_info", [{}] * len(data))

                    question_list = []
                    for i, data_item in enumerate(data):
                        try:
                            question = data_item.non_tensor_batch.get("question", None)
                            if question is None and isinstance(original_extra_info_list[i], dict):
                                question = original_extra_info_list[i].get("question", None)
                            question_list.append(question)
                        except Exception as e:
                            logger.warning(f"Failed to extract question for sample {i}: {e}")
                            question_list.append(None)

                    for i in normal_indices:
                        try:
                            task_extra_info = {}
                            if i < len(original_extra_info_list) and isinstance(original_extra_info_list[i], dict):
                                task_extra_info.update(original_extra_info_list[i])
                            if i < len(question_list) and question_list[i] is not None:
                                task_extra_info["question"] = question_list[i]
                            task_extra_infos.append(task_extra_info)
                        except Exception as e:
                            logger.warning(f"Failed to extract task_extra_info for sample {i}: {e}")
                            task_extra_infos.append({})
                except Exception as e:
                    logger.error(f"Failed to extract task_extra_infos: {e}")
                    logger.error(traceback.format_exc())
                    task_extra_infos = [{}] * len(normal_indices)

                # Compute the C-GRPO reward for normal samples (fault-tolerant)
                try:
                    results = self.verify_normal_samples(
                        normal_prompts_str,
                        normal_sequences_str,
                        normal_ground_truth,
                        normal_data_sources,
                        task_extra_infos
                    )
                except Exception as e:
                    logger.error(f"Failed to compute rewards for normal samples: {e}")
                    logger.error(traceback.format_exc())
                    results = [{
                        'end_to_end_score': 0.0,
                        'rubric_reward': 0.0,
                        'rubric_details': {},
                    }] * len(normal_indices)

                # Extract the results (fault-tolerant)
                try:
                    normal_end_to_end_scores = [r.get('end_to_end_score', 0.0) for r in results]
                    normal_rubric_rewards = [r.get('rubric_reward', 0.0) for r in results]
                    normal_rubric_details = [r.get('rubric_details', {}) for r in results]
                except Exception as e:
                    logger.error(f"Failed to extract reward results: {e}")
                    logger.error(traceback.format_exc())
                    normal_end_to_end_scores = [0.0] * len(normal_indices)
                    normal_rubric_rewards = [0.0] * len(normal_indices)
                    normal_rubric_details = [{}] * len(normal_indices)

                # ========================================
                # === Stats: Rubric Reward computation results ===
                # ========================================
                rubric_success_count = 0
                rubric_error_count = 0
                rubric_success_scores = []
                rubric_error_types = defaultdict(int)

                for detail in normal_rubric_details:
                    error_code = detail.get('error_code', 'SUCCESS')
                    if error_code == RubricRewardError.SUCCESS or error_code == 'SUCCESS':
                        rubric_success_count += 1
                        rubric_success_scores.append(detail.get('rubric_reward', 0.0))
                    else:
                        rubric_error_count += 1
                        rubric_error_types[error_code] += 1

                # Compute the average score among successfully computed rubrics
                avg_rubric_score = np.mean(rubric_success_scores) if rubric_success_scores else 0.0

                # Print stats
                total_normal = len(normal_indices)
                logger.info(
                    f"=== Rubric Reward Stats (normal trajectories) ===\n"
                    f"  Computed successfully: {rubric_success_count}/{total_normal} ({100*rubric_success_count/total_normal:.1f}%), "
                    f"avg score: {avg_rubric_score:.3f}\n"
                    f"  Computation failed: {rubric_error_count}/{total_normal} ({100*rubric_error_count/total_normal:.1f}%)"
                )

                # If there are failures, print the failure-type distribution
                if rubric_error_count > 0:
                    error_breakdown = ", ".join([f"{k}: {v}" for k, v in rubric_error_types.items()])
                    logger.info(f"  Failure type distribution: {error_breakdown}")


                # ========================================
                # === Step 3: within-group Rubric Reward normalization (GRPO-style) ===
                # ========================================
                try:
                    uid_list_all = data.non_tensor_batch.get("uid", None)
                    if uid_list_all is None:
                        logger.error("data.non_tensor_batch is missing the 'uid' field, cannot perform within-group normalization; using the raw Rubric Reward")
                        normalized_rubric_rewards = normal_rubric_rewards
                    else:
                        normal_uid_list = [uid_list_all[i] for i in normal_indices]

                        # Normalize the Rubric Reward (fault-tolerant)
                        try:
                            normalized_rubric_rewards = normalize_rubric_reward_within_group(
                                normal_rubric_rewards, normal_uid_list
                            )
                        except Exception as e:
                            logger.error(f"Failed to normalize the Rubric Reward: {e}")
                            logger.error(traceback.format_exc())
                            normalized_rubric_rewards = normal_rubric_rewards
                except Exception as e:
                    logger.error(f"Failed to extract uid or normalize: {e}")
                    logger.error(traceback.format_exc())
                    normalized_rubric_rewards = normal_rubric_rewards

                # ========================================
                # === Step 4: compute the C-GRPO mixed reward ===
                # ========================================
                try:
                    alpha = self.config.get('cgrpo_alpha', 0.3)
                except Exception as e:
                    logger.warning(f"Failed to get the alpha parameter: {e}, using the default value 0.3")
                    alpha = 0.3

                for idx, i in enumerate(normal_indices):
                    try:
                        outcome_reward = 1.0 if normal_end_to_end_scores[idx] >= 1.0 else 0.0
                        rubric_reward_normalized = normalized_rubric_rewards[idx]

                        # C-GRPO mixed-reward formula (fault-tolerant)
                        try:
                            final_reward = calculate_cgrpo_mixed_reward(
                                outcome_reward, rubric_reward_normalized, alpha
                            )
                        except Exception as e:
                            logger.error(f"Sample {i} failed to compute the mixed reward: {e}")
                            final_reward = 0.0

                        # Set the reward tensor (fault-tolerant)
                        try:
                            valid_len = valid_response_length[i].item()
                            if valid_len > 0:
                                reward_tensor[i, valid_len - 1] = final_reward
                        except Exception as e:
                            logger.error(f"Sample {i} failed to set the reward tensor: {e}")

                        # Update the corresponding position's extra_info
                        # (preserving the original order, fault-tolerant)
                        try:
                            all_end_to_end_scores[i] = normal_end_to_end_scores[idx]
                            all_outcome_rewards[i] = outcome_reward
                            all_rubric_rewards_raw[i] = normal_rubric_rewards[idx]
                            all_rubric_rewards_normalized[i] = rubric_reward_normalized
                            all_final_rewards[i] = final_reward
                            all_rubric_details[i] = normal_rubric_details[idx]

                            # Update the three-step evaluation details
                            details = normal_rubric_details[idx]
                            all_r_identify[i] = details.get('r_identify', 0.0)
                            all_r_support[i] = details.get('r_support', 0.0)
                            all_r_connect[i] = details.get('r_connect', 0.0)
                            all_n_connected[i] = details.get('n_connected', 0)
                            all_n_total[i] = details.get('n_total', 0)
                        except Exception as e:
                            logger.error(f"Sample {i} failed to update extra_info: {e}")

                        # Print debug info (fault-tolerant)
                        try:
                            data_source = data_sources[i]
                            if data_source not in already_print_data_sources:
                                already_print_data_sources[data_source] = 0

                            if already_print_data_sources[data_source] < self.num_examine:
                                already_print_data_sources[data_source] += 1

                                # ====== Basic score info ======
                                logger.info(
                                    f"[Normal Sample {i}] "
                                    f"outcome={outcome_reward:.0f}, "
                                    f"rubric_raw={normal_rubric_rewards[idx]:.3f}, "
                                    f"rubric_norm={rubric_reward_normalized:.3f}, "
                                    f"final={final_reward:.3f}, "
                                    f"r_identify={details.get('r_identify', 0):.3f}, "
                                    f"r_support={details.get('r_support', 0):.3f}, "
                                    f"r_connect={details.get('r_connect', 0):.3f}, "
                                    f"n_connected={details.get('n_connected', 0)}/{details.get('n_total', 0)}"
                                )

                                # ====== Trajectory content (first 500 chars) ======
                                trajectory_preview = normal_sequences_str[idx][:500] if len(normal_sequences_str[idx]) > 500 else normal_sequences_str[idx]
                                logger.info(
                                    f"[Sample {i} Trajectory] {trajectory_preview}..."
                                )

                                # ====== Ground Truth: constraints and entities ======
                                try:
                                    gt = normal_ground_truth[idx]
                                    task_info = task_extra_infos[idx]

                                    # Extract constraints and entities from
                                    # task_extra_info or ground_truth
                                    constraints_gt = task_info.get('constraints', gt.get('constraints', []))
                                    entities_gt = task_info.get('entities', gt.get('entities', {}))

                                    logger.info(
                                        f"[Sample {i} Ground Truth] "
                                        f"total_constraints={len(constraints_gt)}, "
                                        f"total_entities={len(entities_gt)}"
                                    )

                                    # Print entities (E0 is the query entity)
                                    if entities_gt:
                                        logger.info(
                                            f"[Sample {i} Entities] {entities_gt}"
                                        )

                                    # Print the constraint list (first 3 as an example)
                                    if constraints_gt:
                                        constraints_preview = constraints_gt[:3]
                                        logger.info(
                                            f"[Sample {i} Constraints (first 3)] {constraints_preview}"
                                        )
                                except Exception as e:
                                    logger.warning(f"Sample {i} failed to extract ground truth: {e}")

                                # ====== Rubric computation intermediate results ======
                                try:
                                    # Step 1: identified entities
                                    identified_entities = details.get('identified_entities', {})
                                    if identified_entities:
                                        logger.info(
                                            f"[Sample {i} Step1 Identified] "
                                            f"n_identified={details.get('n_identified', 0)}/{details.get('n_total', 0)}, "
                                            f"entities={identified_entities}"
                                        )

                                    # Step 2: supported constraints
                                    supported_constraints = details.get('supported_constraints', [])
                                    if supported_constraints:
                                        logger.info(
                                            f"[Sample {i} Step2 Supported] "
                                            f"n_supported={details.get('n_supported', 0)}/{details.get('n_identified', 0)}, "
                                            f"constraint_indices={supported_constraints}"
                                        )

                                    # Step 3: connected constraints
                                    connected_constraints = details.get('connected_constraints', [])
                                    if connected_constraints:
                                        logger.info(
                                            f"[Sample {i} Step3 Connected] "
                                            f"n_connected={details.get('n_connected', 0)}/{details.get('n_supported', 0)}, "
                                            f"constraint_indices={connected_constraints}"
                                        )

                                    # Error code (if any)
                                    error_code = details.get('error_code', 'SUCCESS')
                                    if error_code != 'SUCCESS':
                                        logger.warning(
                                            f"[Sample {i} Error] error_code={error_code}"
                                        )
                                except Exception as e:
                                    logger.warning(f"Sample {i} failed to print rubric intermediate results: {e}")

                        except Exception as e:
                            logger.warning(f"Sample {i} failed to print debug info: {e}")

                    except Exception as e:
                        logger.error(f"Failed to process sample {i}: {e}")
                        logger.error(traceback.format_exc())
                        # Make sure this sample gets a score of 0
                        all_final_rewards[i] = 0.0

            except Exception as e:
                logger.error(f"Failed to process normal samples: {e}")
                logger.error(traceback.format_exc())
                # All normal samples get a score of 0
                for i in normal_indices:
                    all_final_rewards[i] = 0.0

        # ========================================
        # === Step 5: print info for abnormal samples (reward is already 0) ===
        # ========================================
        for i in range(batch_size):
            if i not in normal_indices:
                # Extract the abnormality types
                abnormal_types = [k for k, v in abnormal_flags_list[i].items() if v]
                all_abnormal_types[i] = abnormal_types

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

                    # If it's due to a missing <answer> tag, print a trajectory preview
                    if "no_answer_tag" in abnormal_types:
                        try:
                            trajectory_preview = sequences_str[i] if len(sequences_str[i]) > 300 else sequences_str[i]
                            logger.info(
                                f"[Sample {i} Abnormal Trajectory] {trajectory_preview}..."
                            )
                        except Exception as e:
                            logger.warning(f"Sample {i} failed to print the abnormal trajectory: {e}")

        # ========================================
        # === Step 6: fill reward_extra_info in the original order ===
        # ========================================
        #
        # Important: fill extra_info here in the original data order (0, 1, 2, ...)
        # to keep it consistent with reward_tensor's order
        #
        # Data flow note:
        # 1. Reward Manager layer: reward_extra_info is a defaultdict(list), every field is a Python list
        # 2. Ray Trainer layer: converted to numpy arrays via safe_convert_reward_extra_info_to_numpy()
        # 3. DataProto layer: batch.non_tensor_batch stores numpy arrays
        #
        # Conversion rules:
        # - List[float] -> np.array(dtype=float64)
        # - List[int] -> np.array(dtype=int64)
        # - List[str] -> np.array(dtype='<U...')  (Unicode strings)
        # - List[List[str]] -> np.array(dtype=object)  (nested structure, each element remains a Python list)
        #
        reward_extra_info['end_to_end_score'] = all_end_to_end_scores  # List[float] -> np.ndarray
        reward_extra_info['outcome_reward'] = all_outcome_rewards  # List[float] -> np.ndarray
        reward_extra_info['rubric_reward_raw'] = all_rubric_rewards_raw  # List[float] -> np.ndarray
        reward_extra_info['rubric_reward_normalized'] = all_rubric_rewards_normalized  # List[float] -> np.ndarray
        reward_extra_info['final_reward'] = all_final_rewards  # List[float] -> np.ndarray
        reward_extra_info['abnormal_types'] = all_abnormal_types  # List[List[str]] -> np.array(dtype=object)
        reward_extra_info['r_identify'] = all_r_identify  # List[float] -> np.ndarray
        reward_extra_info['r_support'] = all_r_support  # List[float] -> np.ndarray
        reward_extra_info['r_connect'] = all_r_connect  # List[float] -> np.ndarray
        reward_extra_info['n_connected'] = all_n_connected  # List[int] -> np.ndarray
        reward_extra_info['n_total'] = all_n_total  # List[int] -> np.ndarray

        # Collect rubric_error_types (added)
        all_rubric_error_types = []
        for i in range(batch_size):
            error_code = all_rubric_details[i].get('error_code', RubricRewardError.SUCCESS)
            all_rubric_error_types.append(error_code)
        reward_extra_info['rubric_error_types'] = all_rubric_error_types  # List[str] -> np.ndarray

        # Save the C-GRPO reward info to non_tensor_batch
        cgrpo_reward_info = []
        for i in range(batch_size):
            cgrpo_reward_info.append({
                'rubric_reward_raw': all_rubric_rewards_raw[i],
                'rubric_details': all_rubric_details[i],
            })

        data.non_tensor_batch['cgrpo_reward_info'] = np.array(cgrpo_reward_info, dtype=object)

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
        """Compute the C-GRPO reward for normal samples.

        Fault tolerance:
        - All exceptions result in a score of 0
        - Timeouts result in a score of 0
        - Ensures the returned list length matches the input length

        Args:
            prompts_strs: list of prompt strings
            sequences_strs: list of generated sequence strings
            ground_truths: list of ground-truth entries
            data_sources: list of data sources
            task_extra_infos: list of extra task info dicts

        Returns:
            List of reward info dicts.
        """
        batch_size = len(sequences_strs)

        # Default return value (all zero)
        default_result = [{
            'end_to_end_score': 0.0,
            'rubric_reward': 0.0,
            'rubric_details': {},
        }] * batch_size

        try:
            # Validate parameters
            if not (len(prompts_strs) == len(ground_truths) == len(data_sources) == len(task_extra_infos) == batch_size):
                logger.error(
                    f"verify_normal_samples input lengths are inconsistent: "
                    f"prompts={len(prompts_strs)}, sequences={batch_size}, "
                    f"ground_truths={len(ground_truths)}, data_sources={len(data_sources)}, "
                    f"task_extra_infos={len(task_extra_infos)}"
                )
                return default_result

            # Run the async computation (fault-tolerant)
            try:
                results = asyncio.run(
                    parallel_compute_cgrpo_score_async(
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
                logger.error(traceback.format_exc())
                return default_result
            except RuntimeError as e:
                if "asyncio.run()" in str(e):
                    logger.error(f"asyncio.run() exception: {e}, trying a fallback method")
                    logger.error(traceback.format_exc())
                    # Try using the existing event loop
                    try:
                        loop = asyncio.get_event_loop()
                        results = loop.run_until_complete(
                            parallel_compute_cgrpo_score_async(
                                self.config,
                                prompts_strs,
                                sequences_strs,
                                ground_truths,
                                data_sources,
                                task_extra_infos=task_extra_infos,
                                num_processes=self.config.get("num_workers", 64),
                            )
                        )
                    except Exception as e2:
                        logger.error(f"Fallback method also failed: {e2}")
                        logger.error(traceback.format_exc())
                        return default_result
                else:
                    logger.error(f"RuntimeError: {e}")
                    logger.error(traceback.format_exc())
                    return default_result
            except Exception as e:
                logger.error(f"Normal-sample reward computation exception, setting all rewards to 0: {type(e).__name__}: {e}")
                logger.error(traceback.format_exc())
                return default_result

            # Validate the returned result (fault-tolerant)
            try:
                if not isinstance(results, list):
                    logger.error(f"Returned result type is wrong: {type(results)}, expected list")
                    return default_result

                if len(results) != batch_size:
                    logger.error(
                        f"Returned result count mismatch: expected {batch_size}, got {len(results)}"
                    )
                    # Pad or truncate
                    if len(results) < batch_size:
                        results += default_result[0:1] * (batch_size - len(results))
                    else:
                        results = results[:batch_size]

                # Validate the format of each result
                for i, result in enumerate(results):
                    if not isinstance(result, dict):
                        logger.warning(f"Sample {i} has the wrong result type: {type(result)}, replacing with a default value")
                        results[i] = default_result[0]
                    elif 'end_to_end_score' not in result or 'rubric_reward' not in result:
                        logger.warning(f"Sample {i} is missing required fields, padding with 0")
                        results[i].setdefault('end_to_end_score', 0.0)
                        results[i].setdefault('rubric_reward', 0.0)
                        results[i].setdefault('rubric_details', {})

                return results

            except Exception as e:
                logger.error(f"Failed to validate the returned result: {e}")
                logger.error(traceback.format_exc())
                return default_result

        except Exception as e:
            logger.error(f"verify_normal_samples top-level exception: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            return default_result
