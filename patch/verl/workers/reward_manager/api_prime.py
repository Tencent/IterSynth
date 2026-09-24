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
    messages,
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
                        messages,
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
        print(f"[ERROR] Timeout occurred for completion: {sequences_str}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"[ERROR] Error processing completion: {sequences_str[:10]}, Error: {e}")
        print(traceback.format_exc())
        return None  # Default value for failed rows


async def parallel_compute_score_async(
    config,
    compute_score,
    prompts_strs,
    sequences_strs,
    ground_truths,
    data_sources,
    messages=None,
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
                message,
                task_extra_info,
                executor,
                timeout=6000.0,
            )
            for prompts_str, sequences_str, ground_truth, data_source, message, task_extra_info in zip(
                prompts_strs,
                sequences_strs,
                ground_truths,
                data_sources,
                messages,
                task_extra_infos,
            )
        ]
        results = await asyncio.gather(*tasks_async, return_exceptions=False)

    # Process results
    for result, prompt, completion, reference, task in zip(
        results, prompts_strs, sequences_strs, ground_truths, data_sources
    ):
        # print("[DEBUG] result type", type(result), result)
        # result type <class 'list'> [{'train_final_score': 0.0, 'val_final_score': 0.0}]
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append({"train_final_score": 0.0, "val_final_score": 0.0})
        else:
            scores.append(result[0])
    return scores


@register("apiprime")
class ApiPrimeRewardManager(AbstractRewardManager):
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        # Defaults to verl/trainer/config/reward.yaml relative to this file's
        # location; override with the REWARD_CONFIG_PATH environment variable.
        config_path = os.environ.get(
            "REWARD_CONFIG_PATH",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "trainer", "config", "reward.yaml",
            ),
        )
        self.config = yaml.safe_load(open(config_path, "r"))

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
        messages = data.non_tensor_batch.get("messages", [])
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
                    messages=messages,
                    task_extra_infos=extra_info,
                    num_processes=self.config["num_workers"],
                )
            )
        except asyncio.TimeoutError:
            print("[ERROR] Global timeout in reward computing! Setting all as 0.")
            scores = [
                {"train_final_score": 0.0, "val_final_score": 0.0}
                for _ in range(len(sequences_str))
            ]
        except Exception as e:
            print(
                f"[ERROR] Unexpected error in batched reward computing. Setting all as 0.: {e}"
            )
            traceback.print_exc()
            scores = [
                {"train_final_score": 0.0, "val_final_score": 0.0}
                for _ in range(len(sequences_str))
            ]
        # NOTE: unclear where this is consumed downstream; left commented out.
        # data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        val_reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32
        )

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompts_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)

        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(
            dim=-1
        )
        sequences_str = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )
        data_sources = data.non_tensor_batch["data_source"]

        scores = self.verify(data)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i][
                "train_final_score"
            ]
            val_reward_tensor[i, valid_response_length[i].item() - 1] = scores[i][
                "val_final_score"
            ]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(
                    f"[DEBUG] prompt_str: {prompts_str}, sequences_str: {sequences_str}, score: {scores[i]['train_final_score']}, val_score: {scores[i]['val_final_score']}"
                )

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "val_reward_tensor": val_reward_tensor,
            }
        else:
            return reward_tensor
