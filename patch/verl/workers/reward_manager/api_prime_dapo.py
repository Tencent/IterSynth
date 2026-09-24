# -*- coding: utf-8 -*-
# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Optional
import traceback
from collections import defaultdict
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
import yaml
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


async def single_compute_score(
    config,
    compute_score,
    prompts_str,
    sequences_str,
    ground_truth,
    data_source,
    task_extra_info,
    executor,
    timeout=6000.0,
):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(
                        compute_score,
                        config,
                        prompts_str,
                        sequences_str,
                        ground_truth,
                        data_source,
                        task_extra_info,
                    ),
                    # Ensure synchronous
                ),
                timeout=timeout,
            )
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {sequences_str}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"Error processing completion: {sequences_str[:10]}, Error: {e}")
        print(traceback.format_exc())
        return None  # Default value for failed rows


async def parallel_compute_score_async(
    config,
    compute_score,
    prompts_strs,
    sequences_strs,
    ground_truths,
    data_sources,
    task_extra_infos=None,
    num_processes=64,
):
    scores = []
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        if task_extra_infos is None:
            task_extra_infos = [None] * len(data_sources)
        # Create tasks for all rows
        tasks_async = [
            single_compute_score(
                config,
                compute_score,
                prompts_str,
                sequences_str,
                ground_truth,
                data_source,
                task_extra_info,
                executor,
                timeout=6000.0,
            )
            for prompts_str, sequences_str, ground_truth, data_source, task_extra_info in zip(
                prompts_strs,
                sequences_strs,
                ground_truths,
                data_sources,
                task_extra_infos,
            )
        ]
        results = await asyncio.gather(*tasks_async, return_exceptions=False)

    # Process results
    for result, prompt, completion, reference, task in zip(
        results, prompts_strs, sequences_strs, ground_truths, data_sources
    ):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores


@register("apiprimedapo")
class ApiPrimeDapoRewardManager(AbstractRewardManager):
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        max_resp_len: Optional[Callable] = None,
        overlong_buffer_cfg: Optional[Callable] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        # Defaults to verl/trainer/config/reward_dapo.yaml relative to this
        # file's location; override with the REWARD_DAPO_CONFIG_PATH environment variable.
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

    def verify(self, data):
        """
        verify the batch and save as ``acc`` tensor
        """
        prompt_ids = data.batch["prompts"]
        prompts_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)

        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )
        ground_truth = [
            data_item.non_tensor_batch["reward_model"]["ground_truth"]
            for data_item in data
        ]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", {})
        # print(f"[DEBUG] extra_info: {extra_info}{type(extra_info)}")

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = asyncio.run(
                parallel_compute_score_async(
                    self.config,
                    self.compute_score,
                    prompts_str,
                    sequences_str,
                    ground_truth,
                    data_sources,
                    task_extra_infos=extra_info,
                    num_processes=self.config["num_workers"],
                )
            )
        except asyncio.TimeoutError:
            print("Global timeout in reward computing! Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(
                f"Unexpected error in batched reward computing. Setting all as 0.: {e}"
            )
            scores = [0.0 for _ in range(len(sequences_str))]
        # data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(
            dim=-1
        )
        sequences_str = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )
        data_sources = data.non_tensor_batch["data_source"]
        # data["valid_response_length"] = valid_response_length

        scores = self.verify(data)
        reward_extra_info["acc"] = scores

        # overlong
        if self.overlong_buffer_cfg.enable:
            new_scores = []
            for i in range(len(data)):
                _valid_response_length = valid_response_length[i]
                _response_length = len(sequences_str[i])

                reward = scores[i]
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = _valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(
                    -exceed_len / overlong_buffer_len * overlong_penalty_factor, 0
                )
                reward += overlong_reward
                print(
                    f"_response_length: {_response_length}, "
                    f"_valid_response_length: {_valid_response_length}"
                    f"expected_len: {expected_len}"
                    f"exceed_len: {exceed_len}"
                    f"overlong_reward: {overlong_reward}"
                    f"reward: {reward}"
                )
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)
                new_scores.append(reward)
            print(f"scores: {scores}, after overlong: {new_scores}")
            scores = new_scores

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
            # reward_extra_info[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("sequences_str", sequences_str)

        print(f"[DEBUG]rw reward_tensor {reward_tensor}")
        print(f"[DEBUG]rw reward_extra_info {reward_extra_info}")
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
