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

# Note(haibin.lin): no need to include all reward managers here in case of complicated dependencies
__all__ = [
    "BatchRewardManager",
    "DAPORewardManager",
    "NaiveRewardManager",
    "PrimeRewardManager",
    "ApiPrimeRewardManager",
    "ApiPrimeDapoRewardManager",
    "ApiPrimeDapoEndToEndOnlyRewardManager",
    "ApiPrimeDapoTrajectoryV1NaiveRewardManager",
    "ApiPrimeDapoTrajectoryV2CGRPORewardManager",
    "ApiPrimeDapoPSPipelineRewardManager",
    "ApiPrimeDapoPSPipelineRubricRewardManager",
    "ApiPrimeDapoPSPipelineMCGRPORewardManager",
    "register",
    "get_reward_manager_cls",
]

from .registry import get_reward_manager_cls, register  # noqa: I001
from .batch import BatchRewardManager
from .dapo import DAPORewardManager
from .naive import NaiveRewardManager
from .prime import PrimeRewardManager
from .api_prime import ApiPrimeRewardManager
from .api_prime_dapo import ApiPrimeDapoRewardManager
from .api_prime_dapo_end_to_end_only import ApiPrimeDapoEndToEndOnlyRewardManager
from .api_prime_dapo_trajectory_v1_naive import ApiPrimeDapoTrajectoryV1NaiveRewardManager
from .api_prime_dapo_trajectory_v2_cgrpo import ApiPrimeDapoTrajectoryV2CGRPORewardManager
from .api_prime_dapo_ps_pipeline import ApiPrimeDapoPSPipelineRewardManager
from .api_prime_dapo_ps_pipeline_rubric import ApiPrimeDapoPSPipelineRubricRewardManager
from .api_prime_dapo_ps_pipeline_mc_grpo import ApiPrimeDapoPSPipelineMCGRPORewardManager