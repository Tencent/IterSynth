# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging  # Added by PS-Pipeline Contributors: used for logging
import os
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def safe_convert_reward_extra_info_to_numpy(reward_extra_infos_dict: dict) -> dict:
    """Safely convert reward_extra_info into numpy arrays, adapting to different reward managers.
    
    This function automatically detects the data type and applies the correct conversion strategy:
    - For a list of scalars (e.g. float, int), convert directly into a numpy array
    - For a nested list (e.g. abnormal_types), convert with dtype=object
    - For other complex objects, keep dtype=object
    
    This stays compatible with the custom fields returned by different reward managers,
    without needing to modify ray_trainer.
    
    Args:
        reward_extra_infos_dict: the reward_extra_info dict returned by the reward manager
        
    Returns:
        The converted dict, where every value is a numpy array
        
    Added by: PS-Pipeline Contributors
    Date: 2026-01-01
    """
    if not reward_extra_infos_dict:
        return {}
    
    converted_dict = {}
    for key, value in reward_extra_infos_dict.items():
        if not isinstance(value, list):
            # If it's not a list, try converting directly (preserves the original behavior)
            converted_dict[key] = np.array(value)
            continue
        
        if len(value) == 0:
            # Empty list, use the object dtype
            converted_dict[key] = np.array(value, dtype=object)
            continue
        
        # Check the type of the first element
        first_elem = value[0]
        
        # Case 1: scalar types (int, float, bool, str)
        if isinstance(first_elem, (int, float, bool, str, np.integer, np.floating)):
            try:
                # Try converting directly (fastest)
                converted_dict[key] = np.array(value)
            except (ValueError, TypeError):
                # If that fails, use dtype=object
                converted_dict[key] = np.array(value, dtype=object)
        
        # Case 2: nested structures (list, dict, tuple, etc.)
        elif isinstance(first_elem, (list, dict, tuple)):
            # For nested structures, always use dtype=object
            converted_dict[key] = np.array(value, dtype=object)
        
        # Case 3: other types (custom objects, etc.)
        else:
            # Handle with dtype=object
            converted_dict[key] = np.array(value, dtype=object)
    
    return converted_dict


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _compute_mc_grpo_advantage(
    data: DataProto,
    is_discarded: Optional[np.ndarray],
    norm_adv_by_std: bool = True,
) -> tuple:
    """Multi-Context GRPO advantage computation (aligned with the MemSearcher paper, Eq. 5-8).

    Core logic (two-level grouping):
    1. Group by uid (all trajectories and all rounds of the same question)
    2. Within each uid group, further group by request_id into trajectories
    3. Take one trajectory_reward R_i per trajectory (identical across all rounds of that trajectory)
    4. Apply a z-score at the trajectory level: A_i = (R_i - mean({R_1,...,R_G})) / std({R_1,...,R_G})
    5. Broadcast A_i to every token position of every round inside the trajectory

    Key point: the z-score is computed across the G trajectories (each trajectory
    contributes only one R_i),
    so it is unaffected by differences in the number of rounds per trajectory.

    Args:
        data: DataProto with batch and non_tensor_batch
        is_discarded: a boolean array marking the abnormal samples to be excluded
        norm_adv_by_std: whether to normalize by std

    Returns:
        Two tensors, (advantages, returns), with the same shape as token_level_rewards
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    uid_array = data.non_tensor_batch["uid"]
    trajectory_reward_array = data.non_tensor_batch["trajectory_reward"]
    request_id_array = data.non_tensor_batch.get("request_id", None)
    batch_size = token_level_rewards.shape[0]

    # Build the valid-sample mask
    if is_discarded is not None:
        valid_mask = ~np.array(is_discarded, dtype=bool)
    else:
        valid_mask = np.ones(batch_size, dtype=bool)

    # === Two-level grouping ===
    # Level 1: group by uid (all rounds of all trajectories of the same question)
    # Level 2: within each uid group, group by request_id into trajectories
    #
    # Structure: uid -> {request_id -> [sample_indices]}
    uid_to_traj_groups = {}  # uid -> {request_id -> [sample_idx, ...]}

    for i in range(batch_size):
        if not valid_mask[i]:
            continue
        uid = str(uid_array[i])
        # request_id identifies the trajectory; if absent, dedup via trajectory_reward
        req_id = str(request_id_array[i]) if request_id_array is not None else f"__sample_{i}"

        if uid not in uid_to_traj_groups:
            uid_to_traj_groups[uid] = {}
        if req_id not in uid_to_traj_groups[uid]:
            uid_to_traj_groups[uid][req_id] = []
        uid_to_traj_groups[uid][req_id].append(i)

    # === Trajectory-level z-score ===
    advantages = torch.zeros_like(token_level_rewards)
    returns = torch.zeros_like(token_level_rewards)

    n_trajectories_total = 0
    n_valid_samples = 0
    all_traj_level_rewards = []

    for uid, traj_groups in uid_to_traj_groups.items():
        # Take one trajectory_reward per trajectory (identical across all rounds)
        traj_rewards = []  # trajectory-level reward list, length = G
        traj_sample_indices = []  # all sample indices belonging to each trajectory

        for req_id, indices in traj_groups.items():
            # Take the trajectory_reward of any sample of this trajectory (identical across all rounds)
            r_i = float(trajectory_reward_array[indices[0]])
            traj_rewards.append(r_i)
            traj_sample_indices.append(indices)
            all_traj_level_rewards.append(r_i)

        n_trajectories_total += len(traj_rewards)

        # Paper Eq. 5: A_i = (R_i - mean) / std, computed across the G trajectories
        mean_r = np.mean(traj_rewards)
        std_r = np.std(traj_rewards)

        for traj_idx, indices in enumerate(traj_sample_indices):
            r_i = traj_rewards[traj_idx]

            if norm_adv_by_std and std_r > 1e-8:
                adv_value = (r_i - mean_r) / std_r
            elif std_r <= 1e-8:
                adv_value = 0.0
            else:
                adv_value = r_i - mean_r

            # Paper Eq. 8: A_{i,j} = A_i, broadcast to all valid tokens of all rounds in the trajectory
            for sample_idx in indices:
                mask = response_mask[sample_idx]
                advantages[sample_idx] = adv_value * mask.float()
                returns[sample_idx] = token_level_rewards[sample_idx]
                n_valid_samples += 1

    n_discarded = batch_size - n_valid_samples

    # Statistics
    n_uid_groups = len(uid_to_traj_groups)
    avg_traj_per_uid = n_trajectories_total / n_uid_groups if n_uid_groups > 0 else 0
    avg_rounds_per_traj = n_valid_samples / n_trajectories_total if n_trajectories_total > 0 else 0
    traj_reward_mean = np.mean(all_traj_level_rewards) if all_traj_level_rewards else 0
    traj_reward_std = np.std(all_traj_level_rewards) if all_traj_level_rewards else 0

    pprint(
        f"[MC-GRPO Advantage] batch_size={batch_size}, valid_samples={n_valid_samples}, "
        f"discarded={n_discarded}, uid_groups={n_uid_groups}, "
        f"total_trajectories={n_trajectories_total}, avg_traj_per_uid={avg_traj_per_uid:.1f}, "
        f"avg_rounds_per_traj={avg_rounds_per_traj:.1f}, "
        f"traj_reward_mean={traj_reward_mean:.3f}, traj_reward_std={traj_reward_std:.3f}"
    )

    return advantages, returns


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # is_discarded: abnormal samples (search_error/exceed_max_tokens/degenerate_output)
        # Must be excluded from the group stats, so that reward=0 abnormal samples don't drag
        # down the group mean and distort the advantage normalization
        is_discarded = data.non_tensor_batch.get("is_discarded", None)

        # ================================================================
        # Multi-Context GRPO (MC-GRPO) detection
        # If non_tensor_batch has a trajectory_reward field (written by the MC-GRPO reward manager),
        # use trajectory-level advantage computation (aligned with the MemSearcher paper, Eq. 5-8):
        #   A_i = (R_i - mu) / sigma  (trajectory-level z-score)
        #   A_{i,j} = A_i         (broadcast to all rounds in the trajectory)
        # Otherwise use standard GRPO (token-level advantage)
        # ================================================================
        use_mc_grpo = "trajectory_reward" in data.non_tensor_batch

        if use_mc_grpo:
            advantages, returns = _compute_mc_grpo_advantage(
                data=data,
                is_discarded=is_discarded,
                norm_adv_by_std=norm_adv_by_std_in_grpo,
            )
        elif is_discarded is not None and np.any(is_discarded):
            # Build a subset containing only valid samples, to compute the advantage
            valid_mask = ~np.array(is_discarded, dtype=bool)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) > 0:
                # Run the GRPO advantage computation on the valid-sample subset
                valid_rewards = data.batch["token_level_rewards"][valid_indices]
                valid_response_mask = grpo_calculation_mask[valid_indices]
                valid_uid = data.non_tensor_batch["uid"][valid_indices]

                valid_advantages, valid_returns = core_algos.compute_grpo_outcome_advantage(
                    token_level_rewards=valid_rewards,
                    response_mask=valid_response_mask,
                    index=valid_uid,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                )

                # Write the result back into the full batch; discarded samples get advantage/returns = 0
                advantages = torch.zeros_like(data.batch["token_level_rewards"])
                returns = torch.zeros_like(data.batch["token_level_rewards"])
                advantages[valid_indices] = valid_advantages
                returns[valid_indices] = valid_returns
            else:
                # All samples were discarded, set all advantages to 0
                advantages = torch.zeros_like(data.batch["token_level_rewards"])
                returns = torch.zeros_like(data.batch["token_level_rewards"])
        else:
            # No discarded samples, compute normally
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            warnings.warn(
                "Disabled critic as algorithm.adv_estimator != gae. "
                "If it is not intended, please set critic.enable=True",
                stacklevel=2,
            )
            self.use_critic = False

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        # Actor validation done in ActorConfig.__post_init__ and validate()
        actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.data.train_batch_size)

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # Inject uid (aligned with the fit() training path: multiple rollouts of the same
            # prompt share the same uid,
            # so the uid-grouped normalization in the reward manager and the GRPO advantage
            # computation etc. all work correctly).
            # Note: this must be injected before repeat, otherwise every sample gets a different uid
            # and the N rollouts of the same prompt won't land in the same group.
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # Inject global data index for PS Pipeline (same as training path)
            test_gen_batch.non_tensor_batch["__global_data_idx__"] = np.arange(
                len(test_gen_batch), dtype=np.int64
            )

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Detect PS Pipeline mode in validation
            is_val_ps_pipeline = (
                "batch_statistics" in test_output_gen_batch.non_tensor_batch
                and isinstance(test_output_gen_batch.non_tensor_batch["batch_statistics"][0], dict)
                and test_output_gen_batch.non_tensor_batch["batch_statistics"][0].get("pipeline_mode") == "ps_pipeline"
            )

            if is_val_ps_pipeline:
                # PS Pipeline: test_output_gen_batch has more samples (round-level) than test_batch.
                # We need to map test_batch metadata onto the expanded samples, similar to
                # _merge_ps_pipeline_output but without downsampling.
                val_N = self.config.actor_rollout_ref.rollout.val_kwargs.n
                val_M = len(test_batch)
                val_batch_data_ids = test_output_gen_batch.non_tensor_batch["batch_data_id"]
                val_original_prompt_indices = val_batch_data_ids // val_N

                print(
                    f"[Validation PS Pipeline] {len(test_output_gen_batch)} round-level samples "
                    f"from {val_M} prompts (val_N={val_N})"
                )

                # Copy metadata fields from test_batch to each round sample
                for field in list(test_batch.non_tensor_batch.keys()):
                    if field not in test_output_gen_batch.non_tensor_batch:
                        test_output_gen_batch.non_tensor_batch[field] = np.array(
                            [test_batch.non_tensor_batch[field][idx] for idx in val_original_prompt_indices],
                            dtype=object,
                        )

                # In PS Pipeline mode, test_output_gen_batch IS the merged batch (no union needed)
                test_batch = test_output_gen_batch
                test_batch.meta_info["validate"] = True

                # Fix sample_inputs/sample_gts: they were collected based on prompt-level test_batch,
                # but now we have round-level samples. Remove the previously added entries and
                # re-add based on the expanded sample count.
                n_prev_inputs = len(sample_inputs) - len(input_texts)
                sample_inputs = sample_inputs[:n_prev_inputs]
                # For PS Pipeline validation, we use the expanded count for inputs
                expanded_input_ids = test_output_gen_batch.batch.get("input_ids", None)
                if expanded_input_ids is not None:
                    expanded_input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in expanded_input_ids]
                else:
                    # input_ids were popped for rollout; re-derive from prompt indices
                    expanded_input_texts = [input_texts[idx] for idx in val_original_prompt_indices]
                sample_inputs.extend(expanded_input_texts)

                n_prev_gts = len(sample_gts) - len(ground_truths)
                sample_gts = sample_gts[:n_prev_gts]
                expanded_gts = [ground_truths[idx] for idx in val_original_prompt_indices]
                sample_gts.extend(expanded_gts)

                # Store generated outputs
                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)
            else:
                # Standard path: test_output_gen_batch has same batch size as test_batch
                # Store generated outputs
                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)

                test_batch = test_batch.union(test_output_gen_batch)
                test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            
            # Added by PS-Pipeline Contributors: wrap in try-except to capture the specific error
            try:
                result = self.val_reward_fn(test_batch, return_dict=True)
                # update1017: use val_reward_tensor for the validation-set reward
                reward_tensor = result["reward_tensor"]
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)

                reward_extra_infos_dict["reward"].extend(scores)
                print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
                if "reward_extra_info" in result:
                    # Extract the trajectory-level aggregated data (length != batch_size);
                    # it must not go into reward_extra_infos_dict (a length-consistency check follows).
                    _ps_agg_keys = ["ps_trajectory_reward", "ps_trajectory_rounds"]
                    for _k in _ps_agg_keys:
                        if _k in result["reward_extra_info"]:
                            _agg_data = result["reward_extra_info"].pop(_k)
                            reward_extra_infos_dict[_k].extend(_agg_data)

                    for key, lst in result["reward_extra_info"].items():
                        reward_extra_infos_dict[key].extend(lst)
                        print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

                # The rubric reward manager writes ps_trajectory_* directly into data.meta_info
                # rather than reward_extra_info, so it must also be extracted from test_batch.meta_info
                for _k in ["ps_trajectory_reward", "ps_trajectory_rounds"]:
                    if _k not in reward_extra_infos_dict or len(reward_extra_infos_dict[_k]) == 0:
                        _meta_val = test_batch.meta_info.get(_k, None) if test_batch.meta_info else None
                        if _meta_val is not None and len(_meta_val) > 0:
                            reward_extra_infos_dict[_k].extend(_meta_val)
            except Exception as e:
                import traceback
                logging.error(f"[Validation] val_reward_fn call failed!")
                logging.error(f"[Validation] error type: {type(e).__name__}")
                logging.error(f"[Validation] error message: {str(e)}")
                logging.error(f"[Validation] full stack trace:\n{traceback.format_exc()}")
                # Print the key info of test_batch to aid debugging
                logging.error(f"[Validation] test_batch.batch.keys(): {list(test_batch.batch.keys())}")
                logging.error(f"[Validation] test_batch.non_tensor_batch.keys(): {list(test_batch.non_tensor_batch.keys())}")
                if "rollout_extra_info" in test_batch.non_tensor_batch:
                    logging.error(f"[Validation] rollout_extra_info exists, length: {len(test_batch.non_tensor_batch['rollout_extra_info'])}")
                else:
                    logging.error(f"[Validation] rollout_extra_info is missing! This may be why validation failed.")
                raise  # re-raise the exception, preserving the original behavior

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        _ps_agg_keys_set = {"ps_trajectory_reward", "ps_trajectory_rounds"}
        for key_info, lst in reward_extra_infos_dict.items():
            if key_info in _ps_agg_keys_set:
                continue  # trajectory-level aggregated data; its length != sample count, so skip the check
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        # Remove the trajectory-level data from reward_extra_infos_dict, so the subsequent
        # process_validation_metrics doesn't error out
        _ps_val_agg_data = {}
        for _k in _ps_agg_keys_set:
            if _k in reward_extra_infos_dict:
                _ps_val_agg_data[_k] = reward_extra_infos_dict.pop(_k)

        data_sources = np.concatenate(data_source_lst, axis=0)

        # Filter out non-numeric fields (e.g. abnormal_types is a list of strings)
        # Keep only the numeric fields for which statistics can be computed
        # This prevents weird values returned by a custom reward_manager (strings, lists, dicts, etc.)
        # from causing errors
        numeric_reward_extra_infos_dict = {}
        for key, values in reward_extra_infos_dict.items():
            if len(values) == 0:
                continue
            
            # Check the type of the first non-None element
            first_valid = next((v for v in values if v is not None), None)
            if first_valid is None:
                continue
            
            # Keep only numeric types (int, float, np.number)
            # Exclude strings, lists, dicts, etc.
            if isinstance(first_valid, (int, float, np.number)):
                numeric_reward_extra_infos_dict[key] = values
            elif isinstance(first_valid, (list, dict, str)):
                # Skip non-numeric types (e.g. abnormal_types is a list)
            
                continue
            else:
                # Unknown type, try converting to float
                try:
                    float(first_valid)
                    numeric_reward_extra_infos_dict[key] = values
                except (TypeError, ValueError):
                    # Conversion failed, skip
        
                    continue
        
        # Wrap process_validation_metrics in try-except to guard against any potential error
        try:
            data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, numeric_reward_extra_infos_dict)
        except Exception as e:
        
            # If it fails, return empty metrics
            data_src2var2metric2val = {}
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        # PS Pipeline validation-set trajectory-level metrics
        ps_val_traj_rewards = _ps_val_agg_data.get("ps_trajectory_reward", None)
        ps_val_traj_rounds = _ps_val_agg_data.get("ps_trajectory_rounds", None)
        if ps_val_traj_rewards is not None and len(ps_val_traj_rewards) > 0:
            try:
                metric_dict["val-core/ps_pipeline/trajectory_acc"] = float(
                    np.mean([1.0 if r >= 0.999 else 0.0 for r in ps_val_traj_rewards])
                )
                metric_dict["val-aux/ps_pipeline/trajectory_reward_mean"] = float(np.mean(ps_val_traj_rewards))
                metric_dict["val-aux/ps_pipeline/trajectory_count"] = float(len(ps_val_traj_rewards))
                if ps_val_traj_rounds is not None and len(ps_val_traj_rounds) > 0:
                    metric_dict["val-aux/ps_pipeline/avg_rounds"] = float(np.mean(ps_val_traj_rounds))
            except Exception:
                pass

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            assert (
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                is not None
            ), "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _merge_ps_pipeline_output(self, batch: DataProto, gen_batch_output: DataProto) -> DataProto:
        """Merge PS Pipeline rollout output with original batch metadata.

        PS Pipeline produces a dynamic number of round-level samples per prompt
        (one trajectory with T_i rounds yields T_i independent training samples).
        This means gen_batch_output.batch_size != M*N, so we cannot use the
        standard batch.repeat(N).union(gen_batch_output) path.

        Instead, we use batch_data_id (stored in gen_batch_output by PS Pipeline)
        to map each round sample back to the original prompt and copy over
        the metadata fields (reward_model, question, data_source, uid, etc.).

        Following the paper's approach, we also perform adaptive downsampling
        to reduce the training corpus to the largest multiple of DP size.

        Args:
            batch: Original DataProto (batch_size=M) with metadata fields
                   (reward_model, question, data_source, extra_info, uid, etc.)
            gen_batch_output: PS Pipeline output (batch_size=sum of all rounds)
                   with tensor fields and rollout-specific non_tensor_batch fields.

        Returns:
            Merged DataProto ready for reward computation and training.
        """
        N = self.config.actor_rollout_ref.rollout.n  # G trajectories per prompt
        M = len(batch)  # original batch size (number of unique prompts)
        original_batch = batch  # batch_size = M, contains reward_model, question, uid, etc.

        # batch_data_id maps each round sample to its GLOBAL index in gen_batch (after repeat).
        # gen_batch was repeat(N, interleave=True): prompt i -> indices [i*N, i*N+1, ..., i*N+N-1]
        # So batch_data_id // N gives the original prompt index in batch (before repeat).
        #
        # IMPORTANT: batch_data_id must be a GLOBAL index (0..M*N-1), not a worker-local index.
        # The __global_data_idx__ field injected before dispatch ensures this.
        batch_data_ids = gen_batch_output.non_tensor_batch["batch_data_id"]
        original_prompt_indices = batch_data_ids // N

        # Sanity check: original_prompt_indices should be in [0, M)
        max_idx = int(np.max(original_prompt_indices)) if len(original_prompt_indices) > 0 else 0
        min_idx = int(np.min(original_prompt_indices)) if len(original_prompt_indices) > 0 else 0
        if max_idx >= M or min_idx < 0:
            pprint(
                f"[PS Pipeline WARNING] batch_data_id mapping out of range! "
                f"original_prompt_indices range=[{min_idx}, {max_idx}], but M={M}, N={N}. "
                f"batch_data_ids sample: {batch_data_ids[:20].tolist()}"
            )
        else:
            n_unique_prompts = len(np.unique(original_prompt_indices))
            pprint(
                f"[PS Pipeline] batch_data_id mapping OK: {len(batch_data_ids)} samples → "
                f"{n_unique_prompts}/{M} unique prompts (N={N})"
            )

        # Copy metadata fields from original batch to each round sample
        for field in list(original_batch.non_tensor_batch.keys()):
            if field not in gen_batch_output.non_tensor_batch:
                gen_batch_output.non_tensor_batch[field] = np.array(
                    [original_batch.non_tensor_batch[field][idx] for idx in original_prompt_indices],
                    dtype=object,
                )

        # Assign uid: all rounds from the same prompt share the same uid (for GRPO grouping)
        gen_batch_output.non_tensor_batch["uid"] = np.array(
            [original_batch.non_tensor_batch["uid"][idx] for idx in original_prompt_indices],
            dtype=object,
        )

        total_samples = len(gen_batch_output)
        dp_size = self.actor_rollout_wg.world_size

        # Adaptive downsampling: reduce to largest multiple of DP size (paper Eq. 6)
        # CRITICAL: We must drop WHOLE trajectories, not individual samples.
        # If we naively truncate (gen_batch_output[:usable_samples]), we may cut
        # a trajectory in the middle, losing the P(answer) sample. The remaining
        # partial samples would get reward=0 ("no_answer_extracted") incorrectly.
        # Instead, we group samples by request_id (trajectory), then greedily
        # include whole trajectories until adding the next one would exceed the
        # largest DP-divisible count.
        usable_samples = (total_samples // dp_size) * dp_size
        if usable_samples < total_samples and usable_samples > 0:
            # Group sample indices by trajectory (request_id), preserving order
            request_ids = gen_batch_output.non_tensor_batch["request_id"]
            from collections import OrderedDict
            traj_to_indices = OrderedDict()  # request_id -> [sample_indices]
            for i, rid in enumerate(request_ids):
                traj_to_indices.setdefault(rid, []).append(i)

            # Greedily include whole trajectories
            kept_indices = []
            for rid, indices in traj_to_indices.items():
                if len(kept_indices) + len(indices) <= usable_samples:
                    kept_indices.extend(indices)

            # Re-align to DP size (the greedy result may undershoot usable_samples)
            final_count = (len(kept_indices) // dp_size) * dp_size
            if final_count == 0:
                raise ValueError(
                    f"[PS Pipeline] Cannot form a DP-divisible batch from {total_samples} samples "
                    f"across {len(traj_to_indices)} trajectories with DP size={dp_size}. "
                    f"Increase batch size or reduce DP parallelism."
                )
            kept_indices = sorted(kept_indices[:final_count])
            gen_batch_output = gen_batch_output[kept_indices]

            # Count how many complete trajectories survived
            surviving_rids = set(gen_batch_output.non_tensor_batch["request_id"])
            pprint(
                f"[PS Pipeline] Adaptive downsampling: {total_samples} → {len(gen_batch_output)} samples "
                f"(DP size={dp_size}, dropped {total_samples - len(gen_batch_output)} samples from "
                f"{len(traj_to_indices) - len(surviving_rids)} trajectories, "
                f"< {(total_samples - len(gen_batch_output)) / total_samples * 100:.1f}% loss)"
            )
        elif usable_samples == 0:
            raise ValueError(
                f"[PS Pipeline] Total samples ({total_samples}) < DP size ({dp_size}). "
                f"Increase batch size or reduce DP parallelism."
            )

        # Log PS pipeline statistics
        batch_stats = gen_batch_output.non_tensor_batch["batch_statistics"][0]
        surviving_request_ids = set(rid for rid in gen_batch_output.non_tensor_batch["request_id"])
        pprint(
            f"[PS Pipeline] Merged batch: {len(gen_batch_output)} samples from "
            f"{len(surviving_request_ids)} trajectories, "
            f"planners={batch_stats.get('planner_count', '?')}, "
            f"synthesizers={batch_stats.get('synthesizer_count', '?')}"
        )

        return gen_batch_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate() 
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                # Inject global data index for PS Pipeline: after repeat, each item needs
                # a globally unique index so that workers can map batch_data_id back to the
                # correct original prompt. Without this, each worker's enumerate() produces
                # LOCAL indices (0..shard_size-1) which collide after collect/concat.
                gen_batch.non_tensor_batch["__global_data_idx__"] = np.arange(
                    len(gen_batch), dtype=np.int64
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # ============================================================
                    # Detect PS Pipeline mode: gen_batch_output has dynamic batch_size
                    # (one prompt -> multiple round-level samples), so we cannot use
                    # the standard batch.repeat(N).union(gen_batch_output) path.
                    # ============================================================
                    is_ps_pipeline = (
                        "batch_statistics" in gen_batch_output.non_tensor_batch
                        and isinstance(gen_batch_output.non_tensor_batch["batch_statistics"][0], dict)
                        and gen_batch_output.non_tensor_batch["batch_statistics"][0].get("pipeline_mode") == "ps_pipeline"
                    )

                    if is_ps_pipeline:
                        batch = self._merge_ps_pipeline_output(batch, gen_batch_output)
                    else:
                        # Standard path: repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)

                        # ========================================
                        # [PS Pipeline] Log entropy grouped by role (planner vs synthesizer vs forced_answer)
                        # Makes it easy to analyze how policy entropy evolves per role
                        # (planner usually has higher entropy, synthesizer lower)
                        # ========================================
                        try:
                            if "rollout_extra_info" in batch.non_tensor_batch:
                                _rei_list = batch.non_tensor_batch["rollout_extra_info"]
                                _role_groups = {}  # role -> list[sample_idx]
                                for _i, _ei in enumerate(_rei_list):
                                    if not isinstance(_ei, dict):
                                        continue
                                    _role = _ei.get("role", "unknown")
                                    # Distinguish whether this is a forced_answer (a validation-only path,
                                    # flagged in trajectory_metrics or meta)
                                    _is_forced = bool(_ei.get("is_forced_answer", False))
                                    _key = f"{_role}_forced" if _is_forced else _role
                                    _role_groups.setdefault(_key, []).append(_i)

                                # Aggregate entropy for each role subset
                                # Use the token-mean method (weighted average): sum(entropy * mask) / sum(mask)
                                for _role_key, _indices in _role_groups.items():
                                    if len(_indices) == 0:
                                        continue
                                    _idx_t = torch.tensor(_indices, dtype=torch.long, device=entropys.device)
                                    _sub_entropy = entropys.index_select(0, _idx_t)  # (n_role, T)
                                    _sub_mask = response_masks.index_select(0, _idx_t)  # (n_role, T)
                                    _valid_tokens = _sub_mask.sum()
                                    if _valid_tokens.item() > 0:
                                        _role_ent_mean = (_sub_entropy * _sub_mask).sum() / _valid_tokens
                                        metrics[f"actor/entropy_by_role/{_role_key}"] = _role_ent_mean.detach().item()
                                        metrics[f"actor/entropy_by_role/{_role_key}_sample_count"] = len(_indices)
                                        metrics[f"actor/entropy_by_role/{_role_key}_valid_tokens"] = int(_valid_tokens.item())

                                        # Additionally log the quantiles of per-sample sequence entropy,
                                        # to see whether a few samples are inflating the overall entropy
                                        _seq_ent = (_sub_entropy * _sub_mask).sum(dim=-1) / _sub_mask.sum(dim=-1).clamp(min=1)
                                        if _seq_ent.numel() > 1:
                                            _seq_sorted = torch.sort(_seq_ent)[0]
                                            _n = _seq_sorted.numel()
                                            metrics[f"actor/entropy_by_role/{_role_key}_p5"] = _seq_sorted[int(_n * 0.05)].item()
                                            metrics[f"actor/entropy_by_role/{_role_key}_p50"] = _seq_sorted[int(_n * 0.50)].item()
                                            metrics[f"actor/entropy_by_role/{_role_key}_p95"] = _seq_sorted[int(min(_n * 0.95, _n - 1))].item()
                                            metrics[f"actor/entropy_by_role/{_role_key}_std"] = _seq_ent.std().item()

                                # Diagnostic log on first execution
                                if self.global_steps <= 2:
                                    _role_summary = {k: len(v) for k, v in _role_groups.items()}
                                    print(f"[EntropyByRole step={self.global_steps}] role distribution: {_role_summary}")
                        except Exception as _e_role_ent:
                            print(f"[EntropyByRole] FAILED: {type(_e_role_ent).__name__}: {_e_role_ent}")

                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            # Extract the trajectory-level aggregated data (length != batch_size)
                            # and store it in meta_info rather than non_tensor_batch, to avoid a shape mismatch.
                            _ps_agg_keys = [
                                "ps_trajectory_reward", "ps_trajectory_rounds",
                                "ps_trajectory_uid", "ps_trajectory_end_to_end_score",
                                "per_trajectory_round_details", "frcr_trajectory_data",
                            ]
                            for _k in _ps_agg_keys:
                                if _k in reward_extra_infos_dict:
                                    batch.meta_info[_k] = reward_extra_infos_dict.pop(_k)

                            # Use the safe conversion function, which auto-adapts to the custom fields
                            # of different reward managers
                            converted_extra_info = safe_convert_reward_extra_info_to_numpy(reward_extra_infos_dict)
                            batch.non_tensor_batch.update(converted_extra_info)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # ========================================
                        # === Discard abnormal samples (before the advantage computation) ===
                        # ========================================
                        # Only exclude environment failures and data truncation (excluded from both
                        # the GRPO group stats and the loss):
                        #   - search_error: search API failure, an environment issue
                        #   - exceed_max_tokens: data truncated / incomplete
                        #
                        # Other abnormalities (tool_parse_error, exceed_max_turns):
                        #   - reward=0 but still participate in training; the model learns to avoid them
                        #     through the reward signal
                        #   - not excluded from the GRPO grouping
                        #
                        if "rollout_extra_info" in batch.non_tensor_batch:
                            rollout_extra_info_list = batch.non_tensor_batch["rollout_extra_info"]
                            
                            # ============================================================
                            # Abnormal-sample exclusion policy (two separate layers):
                            #
                            # 1. is_discarded=True (excluded from the advantage computation AND the loss):
                            #    - search_error: search API failure, an environment issue
                            #    - exceed_max_tokens: data truncated / incomplete
                            #    - degenerate_output: degenerate output (repetition/garbage), harmful to training
                            #
                            # 2. Excluded from the loss only (response_mask=0, but participates in the
                            #    advantage computation):
                            #    - exceed_max_turns (when EXCLUDE_MAX_TURNS_FROM_LOSS=True):
                            #      These samples have reward=0 and join the GRPO group as a contrast baseline,
                            #      but don't contribute to the loss, so the model doesn't waste gradient on
                            #      invalid trajectories
                            #
                            # 3. Participate in both the advantage computation and the loss (reward=0, the
                            #    model learns to avoid them):
                            #    - tool_parse_error, exceed_max_turns (when the switch is off)
                            # ============================================================
                            exclude_max_turns_from_loss = os.environ.get(
                                "EXCLUDE_MAX_TURNS_FROM_LOSS", "False"
                            ).lower() in ("true", "1", "yes")

                            exceed_token_indices = []
                            search_error_indices = []
                            degenerate_indices = []
                            exceed_max_turns_indices = []
                            for i, rollout_extra_info in enumerate(rollout_extra_info_list):
                                abnormal_flags = rollout_extra_info.get("abnormal_flags", {})
                                if abnormal_flags.get("exceed_max_tokens", False):
                                    exceed_token_indices.append(i)
                                if abnormal_flags.get("search_error", False):
                                    search_error_indices.append(i)
                                if abnormal_flags.get("degenerate_output", False):
                                    degenerate_indices.append(i)
                                if abnormal_flags.get("exceed_max_turns", False):
                                    exceed_max_turns_indices.append(i)
                            
                            # --- Layer 1: excluded from both the advantage computation and the loss
                            #     (is_discarded=True + response_mask=0) ---
                            discarded_indices = list(set(
                                exceed_token_indices + search_error_indices + degenerate_indices
                            ))
                            
                            total_samples = len(rollout_extra_info_list)
                            
                            is_discarded = np.zeros(total_samples, dtype=bool)
                            for idx in discarded_indices:
                                is_discarded[idx] = True
                            batch.non_tensor_batch["is_discarded"] = is_discarded
                            
                            # --- Layer 2: excluded from the loss only (response_mask=0, but is_discarded=False) ---
                            # Note: these samples' response_mask is NOT modified here!
                            # response_mask=0 is only set after compute_advantage,
                            # to ensure they participate in the advantage computation with their real reward.
                            loss_only_excluded_indices = []
                            num_max_turns_excluded = 0
                            if exclude_max_turns_from_loss and exceed_max_turns_indices:
                                loss_only_excluded_indices = [
                                    idx for idx in exceed_max_turns_indices
                                    if idx not in set(discarded_indices)
                                ]
                                num_max_turns_excluded = len(loss_only_excluded_indices)
                            
                            # === Layer 1: set response_mask=0 only for is_discarded samples (before the advantage) ===
                            # These samples are also excluded from the GRPO grouping via is_discarded=True,
                            # so response_mask=0 does not affect the advantage computation.
                            if len(discarded_indices) > 0:
                                for idx in discarded_indices:
                                    batch.batch["response_mask"][idx] = 0.0
                            
                            # Record the exclusion stats (metrics and print are deferred until after
                            # the layer-2 exclusion, then emitted together)
                            num_exceed = len(exceed_token_indices)
                            num_search_error = len(search_error_indices)
                            num_degenerate = len(degenerate_indices)
                            num_discarded = len(discarded_indices)
                            
                            # Record the number of samples actually participating in the loss
                            all_loss_excluded = list(set(discarded_indices + loss_only_excluded_indices))
                            num_total_loss_excluded = len(all_loss_excluded)
                            num_effective = total_samples - num_total_loss_excluded
                            
                            if num_total_loss_excluded > 0:
                                print(
                                    f"[Loss Exclusion] Will exclude {num_total_loss_excluded}/{total_samples} samples from loss "
                                    f"(exceed_max_tokens={num_exceed}, "
                                    f"search_error={num_search_error}, "
                                    f"degenerate_output={num_degenerate}, "
                                    f"exceed_max_turns_loss_only={num_max_turns_excluded})"
                                )
                            
                            metrics.update({
                                "training/samples_excluded_from_loss": num_total_loss_excluded,
                                "training/samples_excluded_ratio": num_total_loss_excluded / total_samples if total_samples > 0 else 0.0,
                                "training/samples_excluded_exceed_tokens": num_exceed,
                                "training/samples_excluded_search_error": num_search_error,
                                "training/samples_excluded_degenerate": num_degenerate,
                                "training/samples_excluded_max_turns_loss_only": num_max_turns_excluded,
                                "training/samples_discarded_from_advantage": num_discarded,
                                "training/effective_samples": num_effective,
                                "training/effective_ratio": num_effective / total_samples if total_samples > 0 else 0.0,
                            })
                            print(
                                f"[Training] Effective samples for loss: "
                                f"{num_effective}/{total_samples} "
                                f"({num_effective / total_samples * 100:.1f}%)"
                            )
                        # === End of Added by PS-Pipeline Contributors ===

                        # ========================================
                        # === Entropy monitor 1: within-group reward variance ===
                        # ========================================
                        # When an entropy spike occurs, check whether the within-group variance approaches 0.
                        # If many groups have variance 0, the model has fallen into an all-right/all-wrong dead end,
                        # and GRPO's advantage normalization will amplify the noise.
                        if "uid" in batch.non_tensor_batch and "token_level_rewards" in batch.batch:
                            try:
                                _uid_arr = batch.non_tensor_batch["uid"]
                                _rew_sum = batch.batch["token_level_rewards"].sum(dim=-1).cpu().tolist()

                                from collections import defaultdict as _dd_grv
                                _group_rewards = _dd_grv(list)
                                for _i, _u in enumerate(_uid_arr):
                                    _group_rewards[str(_u)].append(_rew_sum[_i])

                                _group_stds = []
                                _group_means = []
                                _n_zero_var = 0
                                _n_all_zero = 0
                                _n_all_one = 0
                                for _u, _rs in _group_rewards.items():
                                    if len(_rs) <= 1:
                                        continue
                                    _std = float(np.std(_rs))
                                    _mean = float(np.mean(_rs))
                                    _group_stds.append(_std)
                                    _group_means.append(_mean)
                                    if _std < 1e-6:
                                        _n_zero_var += 1
                                        if abs(_mean) < 1e-6:
                                            _n_all_zero += 1
                                        elif abs(_mean - 1.0) < 1e-6:
                                            _n_all_one += 1

                                _n_groups = len(_group_stds)
                                if _n_groups > 0:
                                    metrics.update({
                                        "entropy_diag/group_reward_std_mean": float(np.mean(_group_stds)),
                                        "entropy_diag/group_reward_std_max": float(np.max(_group_stds)),
                                        "entropy_diag/group_reward_std_min": float(np.min(_group_stds)),
                                        "entropy_diag/zero_variance_group_ratio": _n_zero_var / _n_groups,
                                        "entropy_diag/all_zero_group_ratio": _n_all_zero / _n_groups,
                                        "entropy_diag/all_one_group_ratio": _n_all_one / _n_groups,
                                        "entropy_diag/group_count": _n_groups,
                                    })
                                    if self.global_steps <= 2:
                                        print(
                                            f"[EntropyDiag step={self.global_steps}] Monitor-1 (GroupVar) OK: "
                                            f"n_groups={_n_groups}, zero_var_ratio={_n_zero_var/_n_groups:.3f}"
                                        )
                            except Exception as _e_diag1:
                                print(f"[EntropyDiag] Monitor 1 (GroupVar) FAILED: {type(_e_diag1).__name__}: {_e_diag1}")
                        elif self.global_steps <= 2:
                            print(
                                f"[EntropyDiag step={self.global_steps}] Monitor-1 SKIPPED: "
                                f"uid_in_batch={'uid' in batch.non_tensor_batch}, "
                                f"token_level_rewards_in_batch={'token_level_rewards' in batch.batch}"
                            )

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # === Layer-2 loss exclusion: response_mask=0 for exceed_max_turns ===
                        # Must be set after compute_advantage, to ensure these samples participate in the
                        # advantage computation with their real reward
                        if "rollout_extra_info" in batch.non_tensor_batch and 'loss_only_excluded_indices' in dir():
                            if loss_only_excluded_indices and len(loss_only_excluded_indices) > 0:
                                for idx in loss_only_excluded_indices:
                                    batch.batch["response_mask"][idx] = 0.0
                                print(
                                    f"[Loss Exclusion] Applied response_mask=0 for {len(loss_only_excluded_indices)} "
                                    f"exceed_max_turns samples (after advantage computation)"
                                )

                        # ========================================
                        # === Gradient-dilution guard: filter out query groups with no valid advantage signal ===
                        # ========================================
                        # If all trajectories of a query have reward all-0 or all-1 (no within-group variance),
                        # then after GRPO normalization the advantages are all 0, providing no useful signal.
                        # Letting these samples into the gradient update would dilute the useful gradient,
                        # so they are excluded by setting response_mask=0.
                        #
                        # Note: this logic mainly serves PS Pipeline long-horizon multi-turn training;
                        # for react single-turn training,
                        # the cold-start phase often sees "all prompts answered wrong", which triggers
                        # all-group zero-var and would mask the whole batch to 0,
                        # causing the training to run empty. Disable it via env DISABLE_GRADIENT_DILUTION_FILTER=1.
                        _disable_dilution = os.environ.get(
                            "DISABLE_GRADIENT_DILUTION_FILTER", "1"
                        ).lower() in ("1", "true", "yes")
                        if _disable_dilution:
                            # react mode: skip the dilution filter, all samples join the gradient (even if
                            # the advantage is 0)
                            pass
                        elif "uid" in batch.non_tensor_batch:
                            try:
                                uid_arr = batch.non_tensor_batch["uid"]
                                rewards_sum = batch.batch["token_level_rewards"].sum(dim=-1)

                                from collections import defaultdict as _dd_filter
                                query_groups = _dd_filter(list)
                                for i in range(len(uid_arr)):
                                    query_groups[str(uid_arr[i])].append(i)

                                dilution_indices = []
                                num_all_zero_groups = 0
                                num_all_one_groups = 0
                                for uid_str, indices in query_groups.items():
                                    if len(indices) <= 1:
                                        continue
                                    group_rewards = [rewards_sum[i].item() for i in indices]
                                    all_zero = all(abs(r) < 1e-6 for r in group_rewards)
                                    all_one = all(abs(r - 1.0) < 1e-6 for r in group_rewards)

                                    if all_zero:
                                        dilution_indices.extend(indices)
                                        num_all_zero_groups += 1
                                    elif all_one:
                                        dilution_indices.extend(indices)
                                        num_all_one_groups += 1

                                total_samples = len(uid_arr)
                                total_groups = len(query_groups)

                                # Safeguard: if all groups are zero-var, skip the dilution (otherwise the
                                # whole batch gets mask=0,
                                # which makes the advantage computation hit an empty tensor and the metric
                                # function crash; instead let this step run through with all
                                # 0 rewards, and recover on the next step once exploration produces output).
                                all_groups_filtered = (
                                    total_groups > 0
                                    and (num_all_zero_groups + num_all_one_groups) == total_groups
                                )
                                if all_groups_filtered:
                                    print(
                                        f"[Gradient Dilution Filter] SKIPPED: all {total_groups} groups are zero-variance "
                                        f"(zero={num_all_zero_groups}, one={num_all_one_groups}). "
                                        f"Keeping original response_mask to avoid empty-batch gradient computation. "
                                        f"This step's gradient will be ~0 but training will not crash."
                                    )
                                    metrics.update({
                                        "training/gradient_dilution_samples_excluded": 0,
                                        "training/gradient_dilution_ratio": 0.0,
                                        "training/gradient_dilution_all_zero_groups": num_all_zero_groups,
                                        "training/gradient_dilution_all_one_groups": num_all_one_groups,
                                        "training/gradient_dilution_skipped_all_filtered": 1.0,
                                    })
                                elif len(dilution_indices) > 0:
                                    for idx in dilution_indices:
                                        batch.batch["response_mask"][idx] = 0.0

                                    print(
                                        f"[Gradient Dilution Filter] Excluded {len(dilution_indices)}/{total_samples} samples "
                                        f"from {num_all_zero_groups} all-zero groups and {num_all_one_groups} all-one groups "
                                        f"(total {num_all_zero_groups + num_all_one_groups}/{total_groups} groups filtered)"
                                    )

                                    metrics.update({
                                        "training/gradient_dilution_samples_excluded": len(dilution_indices),
                                        "training/gradient_dilution_ratio": len(dilution_indices) / total_samples if total_samples > 0 else 0.0,
                                        "training/gradient_dilution_all_zero_groups": num_all_zero_groups,
                                        "training/gradient_dilution_all_one_groups": num_all_one_groups,
                                    })
                            except Exception as e:
                                print(f"[Gradient Dilution Filter] Error: {e}")

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        # ========================================
                        # === Entropy monitor 3: combined gradient norm / KL / entropy diagnostics ===
                        # ========================================
                        # An entropy spike is usually accompanied by an abnormal gradient-norm spike (exploding gradients).
                        # Check whether the approximate KL goes negative or extremely large, to tell whether
                        # PPO/GRPO ratio clipping has stopped working.
                        try:
                            def _to_float(v):
                                """Convert a metric value that may be a list/tensor/ndarray into a float"""
                                if v is None:
                                    return None
                                if isinstance(v, (list, tuple)):
                                    if len(v) == 0:
                                        return None
                                    v = v[-1] if len(v) > 0 else None
                                if hasattr(v, "item"):
                                    try:
                                        v = v.item()
                                    except Exception:
                                        pass
                                try:
                                    return float(v)
                                except (TypeError, ValueError):
                                    return None

                            _grad_norm = _to_float(actor_output_metrics.get("actor/grad_norm", None))
                            _ppo_kl = _to_float(actor_output_metrics.get("actor/ppo_kl", None))
                            _entropy_loss = _to_float(actor_output_metrics.get(
                                "actor/entropy_loss",
                                actor_output_metrics.get(
                                    "actor/entropy_token_mean_loss",
                                    actor_output_metrics.get("actor/entropy_token_mean", None)
                                )
                            ))
                            _pg_clipfrac = _to_float(actor_output_metrics.get("actor/pg_clipfrac", None))
                            _pg_clipfrac_lower = _to_float(actor_output_metrics.get("actor/pg_clipfrac_lower", None))

                            # On first execution, print the available actor keys to aid debugging
                            if self.global_steps <= 2:
                                print(
                                    f"[EntropyDiag step={self.global_steps}] "
                                    f"actor_output_metrics keys = {sorted(actor_output_metrics.keys())}"
                                )
                                print(
                                    f"[EntropyDiag step={self.global_steps}] "
                                    f"parsed: grad_norm={_grad_norm}, ppo_kl={_ppo_kl}, "
                                    f"entropy_loss={_entropy_loss}, pg_clipfrac={_pg_clipfrac}"
                                )

                            # Record the raw values (always recorded, independent of any threshold)
                            if _grad_norm is not None:
                                metrics["entropy_diag/grad_norm"] = _grad_norm
                                metrics["entropy_diag/grad_norm_gt_10"] = 1.0 if _grad_norm > 10.0 else 0.0
                                metrics["entropy_diag/grad_norm_gt_100"] = 1.0 if _grad_norm > 100.0 else 0.0
                                metrics["entropy_diag/grad_norm_nan"] = 1.0 if (
                                    _grad_norm != _grad_norm or _grad_norm == float("inf")
                                    or _grad_norm == float("-inf")
                                ) else 0.0

                            if _ppo_kl is not None:
                                metrics["entropy_diag/ppo_kl_raw"] = _ppo_kl
                                metrics["entropy_diag/ppo_kl_abs"] = abs(_ppo_kl)
                                metrics["entropy_diag/ppo_kl_negative"] = 1.0 if _ppo_kl < -0.01 else 0.0
                                metrics["entropy_diag/ppo_kl_extreme"] = 1.0 if abs(_ppo_kl) > 0.5 else 0.0

                            if _entropy_loss is not None and _ppo_kl is not None:
                                if abs(_entropy_loss) > 1e-6:
                                    metrics["entropy_diag/kl_over_entropy"] = abs(_ppo_kl) / abs(_entropy_loss)

                            if _pg_clipfrac is not None:
                                metrics["entropy_diag/pg_clipfrac_raw"] = _pg_clipfrac
                                metrics["entropy_diag/pg_clipfrac_high"] = 1.0 if _pg_clipfrac > 0.3 else 0.0
                            if _pg_clipfrac_lower is not None:
                                metrics["entropy_diag/pg_clipfrac_lower_raw"] = _pg_clipfrac_lower
                        except Exception as _e_diag3:
                            print(f"[EntropyDiag] Monitor 3 (grad/kl) FAILED: {type(_e_diag3).__name__}: {_e_diag3}")

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                # Build the reward_extra_infos_dict used for the dump
                                # Automatically extract all fields coming from the reward manager
                                if 'reward_extra_infos_dict' not in locals():
                                    reward_extra_infos_dict = {}
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )
                            
                            # Automatically extract the reward manager's custom fields from batch.non_tensor_batch
                            # These fields have already been converted and stored by
                            # safe_convert_reward_extra_info_to_numpy
                            if 'reward_extra_infos_dict' not in locals():
                                reward_extra_infos_dict = {}
                            
                            known_reward_fields = [
                                'end_to_end_score', 'final_reward', 'abnormal_types',
                                'coverage_rate_raw', 'coverage_rate_normalized'
                            ]
                            for field_name in known_reward_fields:
                                if field_name in batch.non_tensor_batch:
                                    reward_extra_infos_dict.setdefault(
                                        field_name,
                                        batch.non_tensor_batch[field_name].tolist()
                                    )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                try:
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                except RuntimeError as e:
                    err_str = str(e)
                    # Two known trigger scenarios:
                    # 1. all samples aborted (rollout failed entirely)
                    # 2. all samples zero-var -> response_mask all set to 0 by the dilution filter
                    #    -> valid_adv is empty
                    if "numel() == 0" in err_str or "Expected reduction dim" in err_str:
                        print(
                            f"[Step {self.global_steps}] compute_data_metrics failed: "
                            f"no valid samples for advantage computation "
                            f"(all aborted, or all groups zero-variance and filtered). "
                            f"Skipping data metrics for this step. Error: {e}"
                        )
                        metrics.update({
                            "data/all_samples_aborted": 1.0,
                            "training/no_effective_samples_step": 1.0,
                        })
                    else:
                        raise
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                
                # ========================================
                # === Extract monitoring metrics from batch_statistics and extra_info ===
                # ========================================
                
                # 1. Extract the batch-level stats (batch_statistics)
                if "batch_statistics" in batch.non_tensor_batch:
                    # batch_statistics is an np.array containing batch_size identical dicts
                    # Taking the first element is enough (all elements are identical)
                    batch_stats = batch.non_tensor_batch["batch_statistics"][0]
                    
                    # 1.1 Metrics related to abnormal samples
                    metrics.update({
                        "abnormal/normal_rate": batch_stats.get("normal_rate", 0.0),
                        "abnormal/repeated_query_rate": batch_stats.get("repeated_query_rate", 0.0),
                        "abnormal/tool_parse_error_rate": batch_stats.get("tool_parse_error_rate", 0.0),
                        "abnormal/exceed_max_turns_rate": batch_stats.get("exceed_max_turns_rate", 0.0),
                        "abnormal/exceed_max_tokens_rate": batch_stats.get("exceed_max_tokens_rate", 0.0),
                        "abnormal/excessive_tool_calls_rate": batch_stats.get("excessive_tool_calls_rate", 0.0),
                        "abnormal/search_error_rate": batch_stats.get("search_error_rate", 0.0),
                        "abnormal/unknown_tool_rate": batch_stats.get("unknown_tool_rate", 0.0),
                    })
                    
                    # 1.2 Search steps and subq calls across all trajectories
                    metrics.update({
                        "trajectory_all/search_steps_avg": batch_stats.get("search_steps_avg", 0.0),
                        "trajectory_all/search_steps_median": batch_stats.get("search_steps_median", 0.0),
                        "trajectory_all/search_steps_max": batch_stats.get("search_steps_max", 0),
                        "trajectory_all/subq_calls_avg": batch_stats.get("subq_calls_avg", 0.0),
                        "trajectory_all/subq_calls_median": batch_stats.get("subq_calls_median", 0.0),
                        "trajectory_all/subq_calls_max": batch_stats.get("subq_calls_max", 0),
                    })
                
                # 2. Extract each sample's trajectory metrics (rollout_extra_info) and compute
                #    the statistics over the normal samples
                if "rollout_extra_info" in batch.non_tensor_batch:
                    rollout_extra_info_list = batch.non_tensor_batch["rollout_extra_info"]
                    
                    # 2.1 Filter for normal samples
                    normal_search_steps = []
                    normal_subq_calls = []
                    normal_tool_call_attempts = []
                    
                    for rollout_extra_info in rollout_extra_info_list:
                        abnormal_flags = rollout_extra_info.get("abnormal_flags", {})
                        trajectory_metrics = rollout_extra_info.get("trajectory_metrics", {})
                        
                        # Determine whether this is a normal sample
                        is_abnormal = any([
                            abnormal_flags.get("excessive_tool_calls_per_turn", False),
                            abnormal_flags.get("tool_parse_error", False),
                            abnormal_flags.get("repeated_query", False),
                            abnormal_flags.get("exceed_max_turns", False),
                            abnormal_flags.get("exceed_max_tokens", False),
                            abnormal_flags.get("search_error", False),
                            abnormal_flags.get("unknown_tool_in_trajectory", False),
                        ])
                        
                        if not is_abnormal:
                            normal_search_steps.append(trajectory_metrics.get("search_steps", 0))
                            normal_subq_calls.append(trajectory_metrics.get("subq_calls", 0))
                            normal_tool_call_attempts.append(trajectory_metrics.get("tool_call_attempts", 0))
                    
                    # 2.2 Compute the statistics over the normal samples
                    if len(normal_search_steps) > 0:
                        metrics.update({
                            "trajectory_normal/search_steps_avg": float(np.mean(normal_search_steps)),
                            "trajectory_normal/search_steps_median": float(np.median(normal_search_steps)),
                            "trajectory_normal/search_steps_max": int(np.max(normal_search_steps)),
                            "trajectory_normal/subq_calls_avg": float(np.mean(normal_subq_calls)),
                            "trajectory_normal/subq_calls_median": float(np.median(normal_subq_calls)),
                            "trajectory_normal/subq_calls_max": int(np.max(normal_subq_calls)),
                            "trajectory_normal/tool_call_attempts_avg": float(np.mean(normal_tool_call_attempts)),
                            "trajectory_normal/tool_call_attempts_median": float(np.median(normal_tool_call_attempts)),
                            "trajectory_normal/tool_call_attempts_max": int(np.max(normal_tool_call_attempts)),
                        })
                    
                
                # 3. Compute reject-sampling-related metrics (the ratio of samples with reward = 1)
                if "token_level_rewards" in batch.batch:
                    reward_tensor = batch.batch["token_level_rewards"]
                    response_mask = batch.batch.get("response_mask", None)
                    
                    # Compute the sequence-level reward (sum over the whole sequence, consistent
                    # with compute_data_metrics)
                    if response_mask is not None:
                        # Use sum(-1) for the sequence reward, consistent with compute_data_metrics
                        # in metric_utils.py
                        sequence_rewards = reward_tensor.sum(-1)
                        
                        # Compute the ratio of samples with reward = 1 (acc = 1)
                        reward_eq_1_mask = (sequence_rewards >= 0.999)  # use 0.999 to avoid float-precision issues
                        reward_eq_1_ratio = torch.mean(reward_eq_1_mask.float()).item()
                        
                        metrics.update({
                            "reward/acc_eq_1_ratio": reward_eq_1_ratio,
                            "reward/acc_eq_1_count": int(torch.sum(reward_eq_1_mask).item()),
                        })
                        
                        # Compute the reward distribution
                        metrics.update({
                            "reward/sequence_reward_mean": torch.mean(sequence_rewards).item(),
                            "reward/sequence_reward_std": torch.std(sequence_rewards).item(),
                            "reward/sequence_reward_min": torch.min(sequence_rewards).item(),
                            "reward/sequence_reward_max": torch.max(sequence_rewards).item(),
                        })
                
                # 4. PPO-training-related metrics: quantiles of the importance ratio and the TIS ratio
                # === Preliminary diagnostics: check whether the required fields exist ===
                if self.global_steps <= 2:
                    _has_old = "old_log_probs" in batch.batch
                    _has_rollout = "rollout_log_probs" in batch.batch
                    _has_rmask = "response_mask" in batch.batch
                    _has_adv = "advantages" in batch.batch
                    print(
                        f"[EntropyDiag step={self.global_steps}] batch.batch keys: "
                        f"old_log_probs={_has_old}, rollout_log_probs={_has_rollout}, "
                        f"response_mask={_has_rmask}, advantages={_has_adv}"
                    )
                    print(
                        f"[EntropyDiag step={self.global_steps}] full batch.batch keys: "
                        f"{sorted(batch.batch.keys())}"
                    )

                if "old_log_probs" in batch.batch and "rollout_log_probs" in batch.batch:
                    old_log_probs = batch.batch["old_log_probs"]
                    rollout_log_probs = batch.batch["rollout_log_probs"]
                    response_mask = batch.batch.get("response_mask", None)
                    
                    if response_mask is not None:
                        # Compute the importance ratio: pi_theta / pi_old
                        log_ratio = old_log_probs - rollout_log_probs
                        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)  # guard against numerical overflow
                        importance_ratio = torch.exp(log_ratio)
                        
                        # Only count the ratio over valid tokens
                        valid_importance_ratio = importance_ratio[response_mask.bool()]
                        
                        if valid_importance_ratio.numel() > 0:
                            # Compute the quantiles
                            importance_ratio_sorted = torch.sort(valid_importance_ratio)[0]
                            n = importance_ratio_sorted.numel()
                            
                            metrics.update({
                                "ppo/importance_ratio_p5": importance_ratio_sorted[int(n * 0.05)].item(),
                                "ppo/importance_ratio_p50": importance_ratio_sorted[int(n * 0.50)].item(),
                                "ppo/importance_ratio_p95": importance_ratio_sorted[int(n * 0.95)].item(),
                                "ppo/importance_ratio_mean": torch.mean(valid_importance_ratio).item(),
                                "ppo/importance_ratio_max": torch.max(valid_importance_ratio).item(),
                                "ppo/importance_ratio_min": torch.min(valid_importance_ratio).item(),
                            })
                            
                            # Compute the ratio of tokens falling outside the clip range (using the
                            # asymmetric clip-higher strategy)
                            clip_ratio = self.config.actor_rollout_ref.actor.get("clip_ratio", 0.2)
                            # clip_higher strategy: a looser upper bound, a stricter lower bound
                            clip_lower = 1 - clip_ratio  # e.g. 0.8
                            clip_upper = 1 + 2 * clip_ratio  # e.g. 1.4 (looser)
                            
                            clipped_lower_mask = valid_importance_ratio < clip_lower
                            clipped_upper_mask = valid_importance_ratio > clip_upper
                            clipped_mask = clipped_lower_mask | clipped_upper_mask
                            
                            metrics.update({
                                "ppo/importance_ratio_clipped_ratio": torch.mean(clipped_mask.float()).item(),
                                "ppo/importance_ratio_clipped_lower_ratio": torch.mean(clipped_lower_mask.float()).item(),
                                "ppo/importance_ratio_clipped_upper_ratio": torch.mean(clipped_upper_mask.float()).item(),
                                "ppo/clip_lower_bound": clip_lower,
                                "ppo/clip_upper_bound": clip_upper,
                            })

                            # ========================================
                            # === Entropy monitor 2: Likelihood Drift (LLD) detection ===
                            # ========================================
                            # Separately tally the log-ratio distribution of positive samples
                            # (advantage > 0) and negative samples (advantage < 0)
                            # If the likelihood of both positive and negative samples keeps decreasing
                            # (mean log_ratio significantly < 0),
                            # that indicates Lazy Likelihood Displacement (LLD).
                            if "advantages" in batch.batch:
                                try:
                                    _adv = batch.batch["advantages"]
                                    _mask_bool = response_mask.bool()

                                    # advantages is usually (B, T) and needs to be broadcast to the token dim
                                    if _adv.dim() == 1:
                                        _adv = _adv.unsqueeze(-1).expand_as(response_mask)

                                    # log_ratio here is old_log_probs - rollout_log_probs
                                    # a proxy for pi_theta(old) / pi_rollout, reflecting train/inference consistency
                                    _log_ratio_valid = log_ratio[_mask_bool]
                                    _adv_valid = _adv[_mask_bool]

                                    _pos_mask = _adv_valid > 1e-6
                                    _neg_mask = _adv_valid < -1e-6

                                    if self.global_steps <= 2:
                                        print(
                                            f"[EntropyDiag step={self.global_steps}] LLD: "
                                            f"log_ratio_valid shape={_log_ratio_valid.shape}, "
                                            f"pos_count={_pos_mask.sum().item()}, "
                                            f"neg_count={_neg_mask.sum().item()}"
                                        )

                                    if _pos_mask.sum() > 0:
                                        _pos_lr = _log_ratio_valid[_pos_mask]
                                        metrics.update({
                                            "entropy_diag/log_ratio_pos_mean": _pos_lr.mean().item(),
                                            "entropy_diag/log_ratio_pos_std": _pos_lr.std().item() if _pos_lr.numel() > 1 else 0.0,
                                            "entropy_diag/log_ratio_pos_p50": torch.median(_pos_lr).item(),
                                            "entropy_diag/pos_sample_ratio": (_pos_mask.float().mean()).item(),
                                        })

                                    if _neg_mask.sum() > 0:
                                        _neg_lr = _log_ratio_valid[_neg_mask]
                                        metrics.update({
                                            "entropy_diag/log_ratio_neg_mean": _neg_lr.mean().item(),
                                            "entropy_diag/log_ratio_neg_std": _neg_lr.std().item() if _neg_lr.numel() > 1 else 0.0,
                                            "entropy_diag/log_ratio_neg_p50": torch.median(_neg_lr).item(),
                                            "entropy_diag/neg_sample_ratio": (_neg_mask.float().mean()).item(),
                                        })

                                    # LLD indicator: the difference between the mean log_ratio of positive and
                                    # negative samples (should be near 0; persistently negative indicates LLD)
                                    if _pos_mask.sum() > 0 and _neg_mask.sum() > 0:
                                        _pos_mean = _log_ratio_valid[_pos_mask].mean().item()
                                        _neg_mean = _log_ratio_valid[_neg_mask].mean().item()
                                        metrics["entropy_diag/log_ratio_pos_minus_neg"] = _pos_mean - _neg_mean
                                except Exception as _e_diag2:
                                    print(f"[EntropyDiag] Monitor 2 (LLD) FAILED: {type(_e_diag2).__name__}: {_e_diag2}")
                
                # 5. TIS (Token Importance Sampling) ratio quantiles
                # TIS ratio = π_rollout / π_ref
                # Used to detect token mismatch caused by train/inference inconsistency:
                # - the rollout backend uses sglang/vllm
                # - the training backend uses MSDP/Megatron
                # - even with identical parameters, different backends can make pi_old / pi_old != 1
                # - monitoring this metric detects training instability from train/inference mismatch
                if self.use_reference_policy and "ref_log_prob" in batch.batch and "rollout_log_probs" in batch.batch:
                    ref_log_prob = batch.batch["ref_log_prob"]
                    rollout_log_probs = batch.batch["rollout_log_probs"]
                    response_mask = batch.batch.get("response_mask", None)
                    
                    if response_mask is not None:
                        # Compute the TIS ratio: pi_rollout / pi_ref
                        log_tis_ratio = rollout_log_probs - ref_log_prob
                        log_tis_ratio = torch.clamp(log_tis_ratio, min=-50.0, max=50.0)  # a wider clamp range
                        tis_ratio = torch.exp(log_tis_ratio)
                        
                        # Only count the TIS ratio over valid tokens
                        valid_tis_ratio = tis_ratio[response_mask.bool()]
                        
                        if valid_tis_ratio.numel() > 0:
                            # Compute the quantiles
                            tis_ratio_sorted = torch.sort(valid_tis_ratio)[0]
                            n = tis_ratio_sorted.numel()
                            
                            metrics.update({
                                "ppo/tis_ratio_p5": tis_ratio_sorted[int(n * 0.05)].item(),
                                "ppo/tis_ratio_p50": tis_ratio_sorted[int(n * 0.50)].item(),
                                "ppo/tis_ratio_p95": tis_ratio_sorted[int(n * 0.95)].item(),
                                "ppo/tis_ratio_p100": torch.max(valid_tis_ratio).item(),
                                "ppo/tis_ratio_mean": torch.mean(valid_tis_ratio).item(),
                                "ppo/tis_ratio_min": torch.min(valid_tis_ratio).item(),
                            })
                            
                            # Compute the ratio of tokens exceeding the threshold
                            # - the minimum can reach 1e-15 (extremely low-probability tokens)
                            # - a maximum significantly above 10 usually warrants checking token mismatch first
                            high_tis_mask = valid_tis_ratio > 10.0
                            low_tis_mask = valid_tis_ratio < 1e-10  # monitor the extremely small values
                            
                            metrics.update({
                                "ppo/tis_ratio_over_10_ratio": torch.mean(high_tis_mask.float()).item(),
                                "ppo/tis_ratio_under_1e10_ratio": torch.mean(low_tis_mask.float()).item(),
                                "ppo/tis_ratio_over_10_count": int(torch.sum(high_tis_mask).item()),
                            })
                
                # ========================================
                # === Entropy-spike diagnostic metrics ===
                # ========================================
                # Used to locate the root cause of an abnormal entropy spike

                # --- 1. Group reward variance: detect whether all rewards in a group are
                #     identical (all correct / all wrong) ---
                if "uid" in batch.non_tensor_batch and "token_level_rewards" in batch.batch:
                    try:
                        _uid_arr = batch.non_tensor_batch["uid"]
                        _rewards_sum = batch.batch["token_level_rewards"].sum(dim=-1).cpu()

                        from collections import defaultdict as _dd_grv
                        _uid_groups = _dd_grv(list)
                        for _i in range(len(_uid_arr)):
                            _uid_groups[str(_uid_arr[_i])].append(_rewards_sum[_i].item())

                        _group_vars = []
                        _n_zero_var_groups = 0
                        for _uid_str, _rewards in _uid_groups.items():
                            if len(_rewards) <= 1:
                                continue
                            _var = float(np.var(_rewards))
                            _group_vars.append(_var)
                            if _var < 1e-8:
                                _n_zero_var_groups += 1

                        if _group_vars:
                            metrics.update({
                                "entropy_diag/group_reward_var_mean": float(np.mean(_group_vars)),
                                "entropy_diag/group_reward_var_median": float(np.median(_group_vars)),
                                "entropy_diag/zero_var_group_ratio": _n_zero_var_groups / len(_group_vars),
                                "entropy_diag/zero_var_group_count": _n_zero_var_groups,
                                "entropy_diag/total_groups": len(_group_vars),
                            })
                    except Exception as _e_d1:
                        print(f"[EntropyDiag] Spike-1 (GroupVar) FAILED: {type(_e_d1).__name__}: {_e_d1}")

                # --- 2. Likelihood Drift: tally the ratio separately for positive vs negative samples ---
                if "old_log_probs" in batch.batch and "advantages" in batch.batch:
                    try:
                        _old_lp = batch.batch["old_log_probs"]
                        _adv = batch.batch["advantages"]
                        _resp_mask = batch.batch.get("response_mask", None)

                        if _resp_mask is not None and "rollout_log_probs" in batch.batch:
                            _rollout_lp = batch.batch["rollout_log_probs"]
                            _log_ratio = _old_lp - _rollout_lp
                            _log_ratio = torch.clamp(_log_ratio, min=-20.0, max=20.0)
                            _ratio = torch.exp(_log_ratio)

                            # If advantages is (B,), broadcast it to (B, T) to match response_mask
                            if _adv.dim() == 1:
                                _adv_exp = _adv.unsqueeze(-1).expand_as(_resp_mask)
                            else:
                                _adv_exp = _adv

                            # Group by the sign of the advantage (token level)
                            _valid = _resp_mask.bool()
                            _pos_mask = _valid & (_adv_exp > 0)
                            _neg_mask = _valid & (_adv_exp < 0)

                            _pos_ratios = _ratio[_pos_mask]
                            _neg_ratios = _ratio[_neg_mask]

                            if self.global_steps <= 2:
                                print(
                                    f"[EntropyDiag step={self.global_steps}] Spike-2 LLD: "
                                    f"adv shape={tuple(_adv.shape)}, "
                                    f"pos_ratios.numel={_pos_ratios.numel()}, "
                                    f"neg_ratios.numel={_neg_ratios.numel()}"
                                )

                            if _pos_ratios.numel() > 0:
                                metrics["entropy_diag/pos_adv_ratio_mean"] = _pos_ratios.mean().item()
                                metrics["entropy_diag/pos_adv_ratio_std"] = _pos_ratios.std().item() if _pos_ratios.numel() > 1 else 0.0
                            if _neg_ratios.numel() > 0:
                                metrics["entropy_diag/neg_adv_ratio_mean"] = _neg_ratios.mean().item()
                                metrics["entropy_diag/neg_adv_ratio_std"] = _neg_ratios.std().item() if _neg_ratios.numel() > 1 else 0.0

                            # LLD detection: whether the ratio of both positive and negative samples is
                            # decreasing (< 1)
                            if _pos_ratios.numel() > 0 and _neg_ratios.numel() > 0:
                                metrics["entropy_diag/pos_ratio_below_1_frac"] = (_pos_ratios < 1.0).float().mean().item()
                                metrics["entropy_diag/neg_ratio_below_1_frac"] = (_neg_ratios < 1.0).float().mean().item()
                    except Exception as _e_d2:
                        print(f"[EntropyDiag] Spike-2 (LLD) FAILED: {type(_e_d2).__name__}: {_e_d2}")

                # --- 3. Approximate-KL extreme-value detection ---
                if "old_log_probs" in batch.batch and "rollout_log_probs" in batch.batch:
                    try:
                        _old_lp = batch.batch["old_log_probs"]
                        _rollout_lp = batch.batch["rollout_log_probs"]
                        _resp_mask = batch.batch.get("response_mask", None)

                        if _resp_mask is not None:
                            _approx_kl = _rollout_lp - _old_lp  # KL ≈ E[log π_old - log π_θ]
                            _valid_kl = _approx_kl[_resp_mask.bool()]

                            if _valid_kl.numel() > 0:
                                _kl_sorted = torch.sort(_valid_kl)[0]
                                _n_kl = _kl_sorted.numel()
                                metrics.update({
                                    "entropy_diag/approx_kl_mean": _valid_kl.mean().item(),
                                    "entropy_diag/approx_kl_std": _valid_kl.std().item() if _valid_kl.numel() > 1 else 0.0,
                                    "entropy_diag/approx_kl_min": _kl_sorted[0].item(),
                                    "entropy_diag/approx_kl_max": _kl_sorted[-1].item(),
                                    "entropy_diag/approx_kl_p5": _kl_sorted[int(_n_kl * 0.05)].item(),
                                    "entropy_diag/approx_kl_p95": _kl_sorted[int(min(_n_kl * 0.95, _n_kl - 1))].item(),
                                    # Ratio of negative KL (if many are negative, the policy has drifted badly)
                                    "entropy_diag/negative_kl_frac": (_valid_kl < 0).float().mean().item(),
                                })
                    except Exception as _e_d3:
                        print(f"[EntropyDiag] Spike-3 (ApproxKL) FAILED: {type(_e_d3).__name__}: {_e_d3}")

                # --- 4. Precise clip-ratio stats (by clip_ratio_low / clip_ratio_high) ---
                if "old_log_probs" in batch.batch and "rollout_log_probs" in batch.batch:
                    try:
                        _clip_low = self.config.actor_rollout_ref.actor.get("clip_ratio_low", 0.2)
                        _clip_high = self.config.actor_rollout_ref.actor.get("clip_ratio_high", 0.28)
                        _old_lp = batch.batch["old_log_probs"]
                        _rollout_lp = batch.batch["rollout_log_probs"]
                        _resp_mask = batch.batch.get("response_mask", None)

                        if _resp_mask is not None:
                            _log_r = _old_lp - _rollout_lp
                            _log_r = torch.clamp(_log_r, min=-20.0, max=20.0)
                            _r = torch.exp(_log_r)
                            _valid_r = _r[_resp_mask.bool()]

                            if _valid_r.numel() > 0:
                                _below_low = (_valid_r < (1 - _clip_low)).float().mean().item()
                                _above_high = (_valid_r > (1 + _clip_high)).float().mean().item()
                                metrics.update({
                                    "entropy_diag/ratio_below_clip_low": _below_low,
                                    "entropy_diag/ratio_above_clip_high": _above_high,
                                    "entropy_diag/ratio_out_of_clip_total": _below_low + _above_high,
                                    "entropy_diag/clip_ratio_low": _clip_low,
                                    "entropy_diag/clip_ratio_high": _clip_high,
                                })
                    except Exception as _e_d4:
                        print(f"[EntropyDiag] Spike-4 (ClipRatio) FAILED: {type(_e_d4).__name__}: {_e_d4}")

                # === End of Entropy Spike Diagnostics ===

                # ========================================
                # === C-GRPO-specific metric stats ===
                # === Added by PS-Pipeline Contributors ===
                # ========================================
                # 
                # This section tallies the key metrics of the C-GRPO training process, including:
                # 1. normal_trajectory_rate: the ratio of normal trajectories
                # 2. end_to_end_reward: end-to-end reward stats
                # 3. rubric_raw/rubric_norm: rubric reward stats (computed only over trajectories
                #    that got the answer right)
                # 4. r_identify/r_support/r_connect: the three-step evaluation metrics
                # 5. rubric_error_types: rubric error-type stats
                #
                # Data-type notes:
                # - all fields in batch.non_tensor_batch are numpy arrays (converted by
                #   safe_convert_reward_extra_info_to_numpy)
                # - end_to_end_score, rubric_reward_raw, etc.: np.ndarray[float]
                # - abnormal_types: np.ndarray[object], each element is a List[str]
                # - rubric_error_types：np.ndarray[str]
                #
                if "end_to_end_score" in batch.non_tensor_batch:
                    try:
                        from verl.utils.reward_score.trajectory_reward_v2_cgrpo import compute_cgrpo_metrics
                        
                        # Extract the required data (using .get() for robustness)
                        # Note: these fields are already numpy arrays and need to be converted to Python lists
                        end_to_end_scores = batch.non_tensor_batch.get("end_to_end_score", [])
                        rubric_rewards_raw = batch.non_tensor_batch.get("rubric_reward_raw", [])
                        rubric_rewards_norm = batch.non_tensor_batch.get("rubric_reward_normalized", [])
                        r_identify_list = batch.non_tensor_batch.get("r_identify", [])
                        r_support_list = batch.non_tensor_batch.get("r_support", [])
                        r_connect_list = batch.non_tensor_batch.get("r_connect", [])
                        rubric_error_types = batch.non_tensor_batch.get("rubric_error_types", [])
                        abnormal_types_list = batch.non_tensor_batch.get("abnormal_types", [])
                        
                        # Determine whether this is a normal trajectory (samples whose abnormal_types is
                        # an empty list)
                        # Note: abnormal_types_list is an np.ndarray(dtype=object), each element is a List[str]
                        is_normal_trajectory = []
                        for abnormal_types in abnormal_types_list:
                            # abnormal_types may be a list, a numpy array, or a tensor
                            if isinstance(abnormal_types, list):
                                is_normal_trajectory.append(len(abnormal_types) == 0)
                            elif isinstance(abnormal_types, (np.ndarray, torch.Tensor)):
                                # If it's a numpy array or tensor, convert it to a list
                                is_normal_trajectory.append(len(list(abnormal_types)) == 0)
                            else:
                                # Otherwise, treat it as abnormal
                                is_normal_trajectory.append(False)
                        
                        # Convert to a Python list (to ensure the type is correct)
                        def safe_to_list(data):
                            """Safely convert into a Python list
                            
                             Handles the numpy arrays extracted from DataProto.non_tensor_batch,
                             converting them into Python lists for compute_cgrpo_metrics.
                            """
                            if isinstance(data, (list, tuple)):
                                return list(data)
                            elif isinstance(data, np.ndarray):
                                return data.tolist()
                            elif isinstance(data, torch.Tensor):
                                return data.cpu().tolist()
                            else:
                                return list(data) if hasattr(data, '__iter__') else []
                        
                        end_to_end_scores = safe_to_list(end_to_end_scores)
                        rubric_rewards_raw = safe_to_list(rubric_rewards_raw)
                        rubric_rewards_norm = safe_to_list(rubric_rewards_norm)
                        r_identify_list = safe_to_list(r_identify_list)
                        r_support_list = safe_to_list(r_support_list)
                        r_connect_list = safe_to_list(r_connect_list)
                        rubric_error_types = safe_to_list(rubric_error_types)
                        
                        # Compute the C-GRPO metrics
                        cgrpo_metrics = compute_cgrpo_metrics(
                            end_to_end_rewards=end_to_end_scores,
                            rubric_rewards_raw=rubric_rewards_raw,
                            rubric_rewards_norm=rubric_rewards_norm,
                            r_identify_list=r_identify_list,
                            r_support_list=r_support_list,
                            r_connect_list=r_connect_list,
                            is_normal_trajectory=is_normal_trajectory,
                            rubric_error_types=rubric_error_types
                        )
                        
                        # Add to metrics
                        # Note: only metrics.update() is needed here, no extra print or log output
                        # All metrics are later logged together via logger.log(metrics, step) to
                        # wandb/tensorboard/console
                        # This is the standard pattern of the official code (see
                        # verl/verl/trainer/ppo/ray_trainer.py:1204, 1244, 1278, etc.)
                        metrics.update(cgrpo_metrics)
                        
                    except Exception as e:
                        # On an exception, record no metrics, to avoid polluting the metrics dict
                        # For debugging, you can temporarily uncomment the line below
                        # import traceback
                        # print(f"[C-GRPO Metrics] computation failed: {e}")
                        # print(traceback.format_exc())
                        pass
                # === End of C-GRPO Metrics ===

                # === PS Pipeline trajectory-level metrics ===
                # A sample-level reward_mean is affected by the number of rounds per trajectory
                # (trajectories with more rounds contribute more samples),
                # so it doesn't reflect true trajectory accuracy. Here we use trajectory-level
                # statistics (obtained from meta_info).
                ps_traj_rewards = batch.meta_info.get("ps_trajectory_reward", None) if batch.meta_info else None
                ps_traj_rounds = batch.meta_info.get("ps_trajectory_rounds", None) if batch.meta_info else None

                if ps_traj_rewards is not None and len(ps_traj_rewards) > 0:
                    try:
                        if isinstance(ps_traj_rewards, np.ndarray):
                            ps_traj_rewards = ps_traj_rewards.tolist()
                        if ps_traj_rounds is not None and isinstance(ps_traj_rounds, np.ndarray):
                            ps_traj_rounds = ps_traj_rounds.tolist()

                        metrics["ps_pipeline/trajectory_reward_mean"] = float(np.mean(ps_traj_rewards))
                        metrics["ps_pipeline/trajectory_reward_std"] = float(np.std(ps_traj_rewards))
                        metrics["ps_pipeline/trajectory_acc"] = float(
                            np.mean([1.0 if r >= 0.999 else 0.0 for r in ps_traj_rewards])
                        )
                        metrics["ps_pipeline/trajectory_count"] = float(len(ps_traj_rewards))

                        if ps_traj_rounds is not None and len(ps_traj_rounds) > 0:
                            metrics["ps_pipeline/avg_rounds"] = float(np.mean(ps_traj_rounds))
                            metrics["ps_pipeline/median_rounds"] = float(np.median(ps_traj_rounds))
                            metrics["ps_pipeline/max_rounds"] = float(np.max(ps_traj_rounds))

                    except Exception:
                        pass
                # === End of PS Pipeline Metrics ===

                # ========================================
                # === Rubric & Exceed Max Turns Metrics ===
                # ========================================

                # --- Requirement 1: exceed_max_turns sample count / total sample count ---
                if "rollout_extra_info" in batch.non_tensor_batch:
                    _rei_list = batch.non_tensor_batch["rollout_extra_info"]
                    _n_exceed_max_turns = sum(
                        1 for ei in _rei_list
                        if isinstance(ei, dict) and ei.get("abnormal_flags", {}).get("exceed_max_turns", False)
                    )
                    _n_tool_parse_error = sum(
                        1 for ei in _rei_list
                        if isinstance(ei, dict) and ei.get("abnormal_flags", {}).get("tool_parse_error", False)
                    )
                    _n_degenerate = sum(
                        1 for ei in _rei_list
                        if isinstance(ei, dict) and ei.get("abnormal_flags", {}).get("degenerate_output", False)
                    )
                    _total = len(_rei_list)
                    metrics.update({
                        "abnormal/exceed_max_turns_count": _n_exceed_max_turns,
                        "abnormal/exceed_max_turns_ratio": _n_exceed_max_turns / _total if _total > 0 else 0.0,
                        "abnormal/tool_parse_error_count": _n_tool_parse_error,
                        "abnormal/degenerate_output_count": _n_degenerate,
                        "training/total_samples": _total,
                    })

                # --- Requirement 2: stats of the rubric reward values ---
                if "rubric_score" in batch.non_tensor_batch:
                    try:
                        _rubric_scores = batch.non_tensor_batch["rubric_score"]
                        if isinstance(_rubric_scores, np.ndarray):
                            _rubric_scores = _rubric_scores.tolist()

                        # Use the rubric_scored flag (0/1) to precisely distinguish scored / unscored samples;
                        # Fallback: when the field is absent, fall back to a "non-zero" test (backward compatible)
                        _rubric_scored_flags = batch.non_tensor_batch.get("rubric_scored", None)
                        if _rubric_scored_flags is not None:
                            if isinstance(_rubric_scored_flags, np.ndarray):
                                _rubric_scored_flags = _rubric_scored_flags.tolist()
                            _scored_mask = [int(f) > 0 for f in _rubric_scored_flags]
                        else:
                            # Backward compatibility with the old logic
                            _scored_mask = [abs(s) > 1e-6 for s in _rubric_scores]

                        _scored_rubric = [s for s, m in zip(_rubric_scores, _scored_mask) if m]
                        _all_rubric = list(_rubric_scores)

                        if _all_rubric:
                            metrics["rubric/score_mean_all"] = float(np.mean(_all_rubric))
                        if _scored_rubric:
                            # These are the samples actually scored by the LLM Judge, so they correctly
                            # reflect the judge's distribution
                            # Even a normalized_score of 0 (all 3s) is counted
                            metrics["rubric/score_mean_scored"] = float(np.mean(_scored_rubric))
                            metrics["rubric/score_std_scored"] = float(np.std(_scored_rubric))
                            metrics["rubric/score_min_scored"] = float(np.min(_scored_rubric))
                            metrics["rubric/score_max_scored"] = float(np.max(_scored_rubric))
                            metrics["rubric/scored_sample_count"] = len(_scored_rubric)
                            metrics["rubric/scored_sample_ratio"] = len(_scored_rubric) / len(_all_rubric)

                        # Separate the rubric scores of correct / incorrect trajectories (using scored_mask)
                        if "acc" in batch.non_tensor_batch:
                            _acc = batch.non_tensor_batch["acc"]
                            if isinstance(_acc, np.ndarray):
                                _acc = _acc.tolist()
                            _correct_rubric = [
                                s for s, a, m in zip(_rubric_scores, _acc, _scored_mask)
                                if m and abs(a - 1.0) < 1e-6
                            ]
                            _incorrect_rubric = [
                                s for s, a, m in zip(_rubric_scores, _acc, _scored_mask)
                                if m and abs(a) < 1e-6
                            ]
                            if _correct_rubric:
                                metrics["rubric/correct_traj_score_mean"] = float(np.mean(_correct_rubric))
                                metrics["rubric/correct_traj_score_count"] = len(_correct_rubric)
                            if _incorrect_rubric:
                                metrics["rubric/incorrect_traj_score_mean"] = float(np.mean(_incorrect_rubric))
                                metrics["rubric/incorrect_traj_score_count"] = len(_incorrect_rubric)
                    except Exception:
                        pass

                # --- Requirement 3: each step, log ~10 fully rubric-scored trajectories to a JSON file ---
                # Including each round's full content (prompt + response decoded text) and rubric score
                if "rubric_score" in batch.non_tensor_batch and "rollout_extra_info" in batch.non_tensor_batch:
                    try:
                        import json as _json_log

                        _rei_list = batch.non_tensor_batch["rollout_extra_info"]
                        _rubric_scores = batch.non_tensor_batch["rubric_score"]
                        if isinstance(_rubric_scores, np.ndarray):
                            _rubric_scores = _rubric_scores.tolist()
                        _request_ids = batch.non_tensor_batch.get("request_id", [None] * len(_rei_list))
                        _acc_list = batch.non_tensor_batch.get("acc", [None] * len(_rei_list))
                        if isinstance(_acc_list, np.ndarray):
                            _acc_list = _acc_list.tolist()
                        _reward_reasons = batch.non_tensor_batch.get("reward_reason", [""] * len(_rei_list))

                        # Decode the prompt and response text
                        _prompts_decoded = self.tokenizer.batch_decode(
                            batch.batch["prompts"], skip_special_tokens=True
                        )
                        _responses_decoded = self.tokenizer.batch_decode(
                            batch.batch["responses"], skip_special_tokens=True
                        )

                        # Group by request_id
                        from collections import defaultdict as _dd_rubric_log
                        _traj_groups = _dd_rubric_log(list)
                        for i in range(len(_rei_list)):
                            req_id = str(_request_ids[i]) if _request_ids[i] is not None else f"__s_{i}"
                            _traj_groups[req_id].append(i)

                        # Only log trajectories that were successfully scored by the rubric (all rounds
                        # have a non-zero score), at most 10
                        _rubric_trace_entries = []
                        for req_id, indices in _traj_groups.items():
                            if len(_rubric_trace_entries) >= 10:
                                break
                            # Only log trajectories whose every round was scored successfully
                            all_scored = all(abs(_rubric_scores[i]) > 1e-6 for i in indices)
                            if not all_scored:
                                continue

                            first_idx = indices[0]
                            is_correct = abs(_acc_list[first_idx] - 1.0) < 1e-6 if _acc_list[first_idx] is not None else None

                            question = ""
                            first_ei = _rei_list[first_idx]
                            if isinstance(first_ei, dict):
                                msgs = first_ei.get("messages", [])
                                if msgs and isinstance(msgs, list):
                                    for m in msgs:
                                        if isinstance(m, dict) and m.get("role") == "user":
                                            question = str(m.get("content", ""))
                                            break

                            rounds_detail = []
                            for idx in indices:
                                ei = _rei_list[idx]
                                if not isinstance(ei, dict):
                                    continue

                                prompt_text = _prompts_decoded[idx] if idx < len(_prompts_decoded) else ""
                                response_text = _responses_decoded[idx] if idx < len(_responses_decoded) else ""

                                rounds_detail.append({
                                    "role": ei.get("role", "?"),
                                    "round": ei.get("round", "?"),
                                    "rubric_score": round(_rubric_scores[idx], 4),
                                    "precomputed_rubric": round(ei.get("precomputed_rubric_score", 0.0) or 0.0, 4),
                                    "prompt": prompt_text,
                                    "response": response_text,
                                })

                            _rubric_trace_entries.append({
                                "step": self.global_steps,
                                "request_id": req_id[:24],
                                "is_correct": is_correct,
                                "reward_reason": str(_reward_reasons[first_idx]) if first_idx < len(_reward_reasons) else "",
                                "question": question,
                                "num_rounds": len(rounds_detail),
                                "rounds": rounds_detail,
                            })

                        # Write to the JSON file
                        if _rubric_trace_entries:
                            _trace_dir = os.environ.get(
                                "RUBRIC_TRACE_DIR",
                                "./logs/rubric_traces"
                            )
                            os.makedirs(_trace_dir, exist_ok=True)
                            _trace_path = os.path.join(
                                _trace_dir, f"rubric_trace_step_{self.global_steps}.json"
                            )
                            with open(_trace_path, "w", encoding="utf-8") as _f_trace:
                                _json_log.dump(
                                    _rubric_trace_entries, _f_trace,
                                    ensure_ascii=False, indent=2
                                )
                            print(
                                f"[Rubric Trace] Wrote {len(_rubric_trace_entries)} trajectories "
                                f"to {_trace_path}"
                            )
                    except Exception as _trace_e:
                        print(f"[Rubric Trace] Failed to write trace: {_trace_e}")
                # === End of Rubric & Exceed Max Turns Metrics ===

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
