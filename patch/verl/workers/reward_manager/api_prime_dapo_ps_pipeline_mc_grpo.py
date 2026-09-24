#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS Pipeline Multi-Context GRPO Reward Manager

Aligned with the Multi-Context GRPO method from the MemSearcher paper:
- Reward: trajectory-level outcome reward (0/1), broadcast uniformly to all rounds (no time discounting)
- Advantage: computed by compute_mc_grpo_advantage inside ray_trainer's compute_advantage
  - Grouped by uid (the G trajectories for the same question)
  - Trajectory-level z-score: A_i = (R_i - mu) / sigma
  - Broadcast to every round in the trajectory: A_{i,j} = A_i
- Loss: normalized by 1/sum(n_i) (controlled by the actor's loss_agg_mode)

Differences from api_prime_dapo_ps_pipeline.py:
1. Removes time discounting (gamma=0.995) -> every round receives the same trajectory reward
2. ps_trajectory_reward is written into meta_info (consistent with the rubric version, to prevent the DAPO trainer from crashing)
3. reward_extra_info gains a trajectory_reward field (used for advantage computation)

Registry name: apiprimedapopspipelinemcgrpo
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
                partial(browsecomp_compute_score, sequences_str, ground_truth, task_extra_info),
            ),
            timeout=timeout,
        )
        return float(result) if result is not None else 0.0
    except asyncio.TimeoutError:
        logger.warning("PS Pipeline MC-GRPO end-to-end reward computation timed out")
        return 0.0
    except Exception as e:
        logger.error(f"PS Pipeline MC-GRPO end-to-end reward computation failed: {e}")
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
            single_compute_ps_end_to_end_score(s, g, e, executor, timeout=6000.0)
            for s, g, e in zip(sequences_strs, ground_truths, task_extra_infos)
        ]
        results = await asyncio.gather(*tasks_async, return_exceptions=True)

    return [float(r) if not isinstance(r, (Exception, BaseException)) and r is not None else 0.0 for r in results]


@register("apiprimedapopspipelinemcgrpo")
class ApiPrimeDapoPSPipelineMCGRPORewardManager(AbstractRewardManager):
    """
    PS Pipeline Multi-Context GRPO Reward Manager

    Aligned with the paper's Multi-Context GRPO:
    - Each trajectory receives one outcome reward R_i (0 or 1)
    - R_i is broadcast uniformly to every round sample in the trajectory (no time discounting)
    - The advantage is computed via a trajectory-level z-score inside compute_advantage, then broadcast

    Key difference from standard GRPO:
    - Standard GRPO: token-level reward -> token-level advantage (each token independent)
    - MC-GRPO: trajectory-level reward -> trajectory-level advantage -> broadcast to every round
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

        self.config = {"num_workers": 64}

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None
            assert self.max_resp_len >= self.overlong_buffer_cfg.len

        logger.info("[OK] ApiPrimeDapoPSPipelineMCGRPORewardManager initialized")

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute the PS Pipeline MC-GRPO reward tensor.

        Core logic (aligned with the paper's Eq. 5-8):
        1. Group into trajectories by request_id
        2. For each trajectory: find P(answer) -> extract the answer -> LLM Judge -> R_i in {0, 1}
        3. Broadcast R_i uniformly to every round sample in the trajectory (A_{i,j} = A_i, no time discounting)
        4. Abnormal trajectories: reward = 0; offending rounds are zeroed
        """
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

        # === Step 1: extract rollout_extra_info and request_id ===
        rollout_extra_info_list = data.non_tensor_batch.get("rollout_extra_info", [{}] * batch_size)
        request_id_list = data.non_tensor_batch.get("request_id", [None] * batch_size)

        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # === Step 2: group by request_id ===
        trajectory_groups = defaultdict(list)
        for i in range(batch_size):
            req_id = request_id_list[i]
            if req_id is None:
                req_id = f"__standalone_{i}"
            trajectory_groups[req_id].append(i)

        logger.info(
            f"PS Pipeline MC-GRPO Reward: batch_size={batch_size}, "
            f"num_trajectories={len(trajectory_groups)}"
        )

        # === Step 3: extract ground_truth and question ===
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

        question_ntb = data.non_tensor_batch.get("question", None)
        extra_info_ntb = data.non_tensor_batch.get("extra_info", None)
        messages_ntb = data.non_tensor_batch.get("messages", None)

        question_list = []
        for i in range(batch_size):
            question = None
            if question_ntb is not None:
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
                    msgs = msg_data.get("messages", None) if isinstance(msg_data, dict) else (msg_data if isinstance(msg_data, list) else None)
                    if msgs is not None:
                        for msg in msgs:
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                content = msg.get("content", "")
                                if isinstance(content, str) and content.strip():
                                    question = content.strip()
                                    break
                except (IndexError, TypeError):
                    pass
            question_list.append(question)

        # === Step 4: process each trajectory ===
        all_rewards = [0.0] * batch_size
        all_end_to_end_scores = [0.0] * batch_size
        all_extracted_answers = [None] * batch_size
        all_is_correct = [False] * batch_size
        all_abnormal_types = [[] for _ in range(batch_size)]
        all_reward_reason = [""] * batch_size
        # The trajectory-level reward of the trajectory each sample belongs to (used for MC-GRPO advantage computation)
        all_trajectory_reward = [0.0] * batch_size

        trajectories_to_verify = []

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
                    continue

                if should_give_zero_reward:
                    for idx in sample_indices:
                        all_abnormal_types[idx] = abnormal_types
                        all_reward_reason[idx] = f"abnormal: {abnormal_types}"
                    continue

                # Find the P(answer) sample and extract the answer
                answer_text = None
                for idx in sample_indices:
                    ei = rollout_extra_info_list[idx]
                    if isinstance(ei, dict) and ei.get("role") == "planner" and ei.get("answer"):
                        answer_text = ei["answer"]

                if answer_text is None:
                    for idx in reversed(sample_indices):
                        ei = rollout_extra_info_list[idx]
                        if isinstance(ei, dict) and ei.get("role") == "planner":
                            extracted = extract_answer_from_text(sequences_str[idx])
                            if extracted:
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

                task_extra_info = {}
                if question_list[first_idx] is not None:
                    task_extra_info["question"] = question_list[first_idx]

                trajectories_to_verify.append(
                    (req_id, sample_indices, f"<answer>{extracted_answer}</answer>", gt, task_extra_info)
                )

            except Exception as e:
                logger.error(f"Failed to process trajectory {req_id}: {e}")
                traceback.print_exc()
                for idx in sample_indices:
                    all_reward_reason[idx] = f"exception: {str(e)[:100]}"

        # === Step 5: batch LLM Judge ===
        if trajectories_to_verify:
            verify_sequences = [t[2] for t in trajectories_to_verify]
            verify_ground_truths = [t[3] for t in trajectories_to_verify]
            verify_extra_infos = [t[4] for t in trajectories_to_verify]

            end_to_end_scores = self._verify_answers_batch(
                verify_sequences, verify_ground_truths, verify_extra_infos
            )

            for traj_idx, (req_id, sample_indices, _, _, _) in enumerate(trajectories_to_verify):
                score = end_to_end_scores[traj_idx]
                is_correct = score >= 1.0
                # Paper Eq. 5: trajectory-level reward R_i
                trajectory_reward = 1.0 if is_correct else 0.0

                # Get the list of offending rounds and the forced_answer flag
                first_ei = rollout_extra_info_list[sample_indices[0]]
                excessive_rounds = set()
                is_forced_answer = False
                if isinstance(first_ei, dict):
                    af = first_ei.get("abnormal_flags", {})
                    excessive_rounds = set(af.get("excessive_rounds", []))
                    is_forced_answer = af.get("forced_answer_generated", False)

                # For trajectories that were forced to generate an answer
                # after exceeding max_rounds: if the answer is correct,
                # apply a discounted reward (penalizing the failure to
                # answer proactively within the limit)
                FORCED_ANSWER_REWARD_DISCOUNT = 0.5
                if is_forced_answer and is_correct:
                    trajectory_reward = trajectory_reward * FORCED_ANSWER_REWARD_DISCOUNT
                    logger.info(
                        f"[MC-GRPO] Trajectory {req_id[:8]}...: forced answer correct, "
                        f"applying discount {FORCED_ANSWER_REWARD_DISCOUNT} -> reward={trajectory_reward}"
                    )

                # Paper Eq. 8: A_{i,j} = A_i, every round receives the same
                # trajectory reward, with no time discounting! (the key
                # difference from api_prime_dapo_ps_pipeline.py)
                for idx in sample_indices:
                    ei = rollout_extra_info_list[idx]
                    sample_round = ei.get("round", -1) if isinstance(ei, dict) else -1

                    if sample_round in excessive_rounds:
                        sample_reward = 0.0
                        all_reward_reason[idx] = f"excessive_round_{sample_round}"
                    else:
                        # Every round receives the same trajectory_reward (no discounting)
                        sample_reward = trajectory_reward
                        if is_forced_answer:
                            all_reward_reason[idx] = "forced_correct_discounted" if is_correct else "forced_incorrect"
                        else:
                            all_reward_reason[idx] = "correct" if is_correct else "incorrect"

                    all_rewards[idx] = sample_reward
                    all_trajectory_reward[idx] = trajectory_reward  # record the trajectory-level reward
                    all_end_to_end_scores[idx] = score
                    all_is_correct[idx] = is_correct

                    valid_len = valid_response_length[idx].item()
                    if valid_len > 0:
                        reward_tensor[idx, valid_len - 1] = sample_reward

        # === Step 6: debug logging ===
        data_sources = data.non_tensor_batch.get("data_source", None)
        uid_list = data.non_tensor_batch.get("uid", [None] * batch_size)
        already_print_data_sources = {}

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
                    f"[MC-GRPO Trajectory {req_id[:8]}...] "
                    f"uid={str(traj_uid)[:8]}..., "
                    f"samples={len(sample_indices)}, "
                    f"reward={all_rewards[first_idx]:.1f}, "
                    f"reason={all_reward_reason[first_idx]}, "
                    f"answer={str(all_extracted_answers[first_idx])[:100]}"
                )

        # GRPO grouping stats
        uid_to_rewards = defaultdict(list)
        for i in range(batch_size):
            uid = uid_list[i] if i < len(uid_list) else None
            if uid is not None:
                uid_to_rewards[str(uid)].append(all_rewards[i])

        # === Step 7: fill reward_extra_info ===
        reward_extra_info['end_to_end_score'] = all_end_to_end_scores
        reward_extra_info['final_reward'] = all_rewards
        reward_extra_info['extracted_answer'] = [
            a if a is not None else "" for a in all_extracted_answers
        ]
        reward_extra_info['is_correct'] = [1.0 if c else 0.0 for c in all_is_correct]
        reward_extra_info['acc'] = [1.0 if c else 0.0 for c in all_is_correct]
        reward_extra_info['reward_reason'] = all_reward_reason
        reward_extra_info['abnormal_types'] = all_abnormal_types
        # MC-GRPO specific: the trajectory-level reward for each sample (used for advantage computation)
        reward_extra_info['trajectory_reward'] = all_trajectory_reward

        # Trajectory-level stats -> write into meta_info (length != batch_size)
        trajectory_rewards = []
        trajectory_rounds = []
        for req_id, sample_indices in trajectory_groups.items():
            first_idx = sample_indices[0]
            trajectory_rewards.append(1.0 if all_is_correct[first_idx] else 0.0)
            ei = rollout_extra_info_list[first_idx]
            rounds = ei.get("trajectory_metrics", {}).get("search_rounds", 0) if isinstance(ei, dict) else 0
            trajectory_rounds.append(float(rounds))

        if hasattr(data, "meta_info") and data.meta_info is not None:
            data.meta_info['ps_trajectory_reward'] = trajectory_rewards
            data.meta_info['ps_trajectory_rounds'] = trajectory_rounds
        else:
            reward_extra_info['ps_trajectory_reward'] = trajectory_rewards
            reward_extra_info['ps_trajectory_rounds'] = trajectory_rounds

        data.batch["acc"] = torch.tensor(
            [1.0 if c else 0.0 for c in all_is_correct],
            dtype=torch.float32, device=prompt_ids.device,
        )

        # Summary logging
        total_trajectories = len(trajectory_groups)
        traj_correct_count = sum(1 for _, si in trajectory_groups.items() if all_is_correct[si[0]])
        traj_acc = traj_correct_count / total_trajectories if total_trajectories > 0 else 0.0

        unique_uids = len(uid_to_rewards)
        samples_per_group = [len(v) for v in uid_to_rewards.values()] if uid_to_rewards else []
        avg_samples_per_group = sum(samples_per_group) / len(samples_per_group) if samples_per_group else 0

        logger.info(
            f"[PS Pipeline MC-GRPO Reward Summary]\n"
            f"  Total samples: {batch_size}, trajectories: {total_trajectories}\n"
            f"  Correct trajectories: {traj_correct_count}, trajectory accuracy: {traj_acc:.3f}\n"
            f"  GRPO groups: num_uids={unique_uids}, avg samples per group={avg_samples_per_group:.1f}"
        )

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(reward_extra_info)}
        return reward_tensor

    def _verify_answers_batch(self, sequences_strs, ground_truths, task_extra_infos):
        """Batch, concurrently call browsecomp_zh.compute_score to verify answers."""
        try:
            results = asyncio.run(
                parallel_compute_ps_end_to_end_score_async(
                    sequences_strs, ground_truths, task_extra_infos,
                    num_processes=self.config.get("num_workers", 64),
                )
            )
        except RuntimeError as e:
            if "asyncio.run()" in str(e):
                try:
                    loop = asyncio.get_event_loop()
                    results = loop.run_until_complete(
                        parallel_compute_ps_end_to_end_score_async(
                            sequences_strs, ground_truths, task_extra_infos,
                            num_processes=self.config.get("num_workers", 64),
                        )
                    )
                except Exception:
                    results = [0.0] * len(sequences_strs)
            else:
                results = [0.0] * len(sequences_strs)
        except Exception:
            results = [0.0] * len(sequences_strs)
        return results
