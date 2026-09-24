#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intermediate-process reward - V1 Naive version

This module implements a reward computation based on intermediate-fact
identification within a trajectory (naive version), using within-group
normalization to remove the influence of question difficulty.

Core features:
- Concurrent computation of end-to-end reward and trajectory-coverage reward
- Within-group normalization to remove the effect of question difficulty on gradients
- Handles formatting errors and overlong rollouts
- Early-signal amplification to encourage exploration

Version: 1.0.0
"""

import json
import logging
import os
import re
import traceback
from typing import List, Dict, Any, Optional

import json_repair
import numpy as np
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrajectoryFactIdentifierV1:
    """Trajectory intermediate-fact identifier, V1.

    Identifies which intermediate entities/facts have already been
    discovered from a search agent's reasoning trajectory.
    """

    # Intermediate-fact identification prompt template.
    # NOTE: intentionally kept in Chinese, since it is applied to
    # Chinese-language search-agent trajectories in this pipeline; this is
    # functional prompt content sent to the LLM, not a code comment.
    FACT_IDENTIFICATION_PROMPT = """你是一个专业的事实识别助手。

给定一个复杂的 search agent 查询、轨迹和中间事实列表，请判断轨迹中识别出了哪些中间事实。

**原始查询**：
{query}

**中间事实列表（Ground Truth）**：
{mid_facts_ground_truth}

**轨迹内容**：
{trajectory}

**任务**：
请仔细分析轨迹内容，判断轨迹中通过搜索和推理识别出了哪些中间事实。
对于每个中间事实，判断是否在轨迹中被发现（即通过搜索结果或推理过程中提及或验证）。

**输出格式**：
请严格按照以下 JSON 格式输出（不要添加任何其他文字）：
{{
    "identified_facts": [
        "事实1描述",
        "事实2描述"
    ],
    "analysis": "简要说明识别到这些事实的理由"
}}

注意：
1. 只列出在轨迹中明确识别或验证的事实
2. 如果某个事实在轨迹中没有被提及或验证，不要包含在列表中
3. 确保输出是有效的 JSON 格式
"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the trajectory fact identifier.

        Args:
            config: configuration dict containing the parameters needed for
                the API call (including model_name, etc.)
        """
        self.config = config
        self.model_name = config.get('fact_identifier_model', 'gpt-4o')

        # LLM Judge API credentials: provide via environment variables, do not hardcode
        self.client = OpenAI(
            api_key=os.environ.get("LLM_JUDGE_API_KEY", ""),
            base_url=os.environ.get("LLM_JUDGE_BASE_URL", "https://api.openai.com/v1"),
        )

    def identify_facts(
        self, 
        trajectory: str, 
        query: str, 
        mid_facts_ground_truth_list: List[str]
    ) -> List[str]:
        """Identify the intermediate facts discovered within a trajectory.

        Args:
            trajectory: the reasoning trajectory string
            query: the original query
            mid_facts_ground_truth_list: list of ground-truth intermediate facts

        Returns:
            The list of identified intermediate facts.
        """
        try:
            # Build the identification prompt
            mid_facts_str = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(mid_facts_ground_truth_list)])
            
            prompt = self.FACT_IDENTIFICATION_PROMPT.format(
                query=query,
                mid_facts_ground_truth=mid_facts_str,
                trajectory=trajectory
            )
            
            # Call the LLM the same way as browsecomp_zh
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=1000
                    )
                    content = response.choices[0].message.content.strip()

                    result = self._parse_identification_result(content)
                    identified_facts = result.get('identified_facts', [])
                    logger.info(f"Identified {len(identified_facts)} intermediate facts: {identified_facts}")

                    return identified_facts

                except Exception as e:
                    logger.error(f"Intermediate-fact identification API call failed (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2 ** attempt)
                    else:
                        logger.error("All retries failed")
                        return []
                
        except Exception as e:
            logger.error(f"Intermediate-fact identification failed: {e}")
            logger.error(traceback.format_exc())
            return []

    def _parse_identification_result(self, result_text: str) -> Dict[str, Any]:
        """Parse the intermediate-fact identification result.

        Args:
            result_text: the raw text returned by the API

        Returns:
            The parsed result dict.
        """
        try:
            # Try parsing as JSON directly
            result = json_repair.loads(result_text)
            return result
        except Exception as e:
            logger.warning(f"JSON parsing failed, falling back to regex extraction: {e}")

            # Fallback: extract via regex
            try:
                # Look for a fenced JSON code block
                json_pattern = r'```json\s*(\{.*?\})\s*```'
                match = re.search(json_pattern, result_text, re.DOTALL)
                if match:
                    result = json_repair.loads(match.group(1))
                    return result

                # Look for a plain JSON object
                json_pattern = r'\{[^}]*"identified_facts"[^}]*\}'
                match = re.search(json_pattern, result_text, re.DOTALL)
                if match:
                    result = json_repair.loads(match.group(0))
                    return result

                logger.error(f"Could not extract JSON from result: {result_text[:200]}")
                return {'identified_facts': []}

            except Exception as e2:
                logger.error(f"Regex extraction also failed: {e2}")
                return {'identified_facts': []}


def calculate_coverage_rate(
    mid_facts_identified: List[str],
    mid_facts_ground_truth: List[str]
) -> float:
    """Compute the intermediate-fact coverage rate.

    Args:
        mid_facts_identified: list of identified intermediate facts
        mid_facts_ground_truth: list of ground-truth intermediate facts

    Returns:
        Coverage rate, a float in [0, 1].
    """
    if len(mid_facts_ground_truth) == 0:
        logger.warning("Ground-truth intermediate-fact list is empty")
        return 0.0

    # Coverage rate = number of identified facts / number of ground-truth facts
    coverage_rate = len(mid_facts_identified) / len(mid_facts_ground_truth)

    # Clamp to [0, 1]
    coverage_rate = max(0.0, min(1.0, coverage_rate))

    logger.info(
        f"Coverage rate computation: {len(mid_facts_identified)}/{len(mid_facts_ground_truth)} = {coverage_rate:.3f}"
    )

    return coverage_rate


def normalize_coverage_within_group(
    coverage_rates: List[float], 
    uid_list: List[str]
) -> List[float]:
    """Normalize coverage rates within each group (mirrors GRPO's advantage
    normalization implementation).

    Purpose of within-group normalization:
    1. Remove the effect of question difficulty on gradient magnitude
    2. Prevent reward collapse; amplify early signal to encourage exploration

    Normalization method (fully analogous to GRPO):
    - Group by uid (the same uid = different rollouts of the same query)
    - Within each uid group, normalized coverage = raw coverage / max coverage in the group
    - If the group's max coverage is 0, all normalized coverages in that group are 0
    - If the group has only 1 sample, normalized coverage = 1.0 (if raw coverage > 0) or 0.0

    This ensures that:
    - The best rollout within each uid group gets a normalized coverage of 1.0
    - Other rollouts are scaled relative to the group's best rollout, into [0, 1]
    - Different uids (different queries) do not affect each other

    Args:
        coverage_rates: list of coverage rates (possibly from different queries)
        uid_list: list of uids identifying which query each sample belongs to
            (the same uid = different rollouts of the same query)

    Returns:
        The list of normalized coverage rates, each in [0, 1].
    """
    if len(coverage_rates) == 0:
        return []

    if len(coverage_rates) != len(uid_list):
        raise ValueError(f"coverage_rates and uid_list length mismatch: {len(coverage_rates)} vs {len(uid_list)}")

    # Group by uid (mirrors GRPO's id2score)
    from collections import defaultdict
    uid2coverage = defaultdict(list)
    uid2indices = defaultdict(list)

    for i, (coverage, uid) in enumerate(zip(coverage_rates, uid_list)):
        uid2coverage[uid].append(coverage)
        uid2indices[uid].append(i)

    # Compute the max coverage per uid group (mirrors GRPO's id2mean/id2std)
    uid2max_coverage = {}
    for uid, coverage_list in uid2coverage.items():
        if len(coverage_list) == 1:
            # With only one trajectory: normalize to 1.0 if coverage > 0, else 0.0
            uid2max_coverage[uid] = coverage_list[0] if coverage_list[0] > 0 else 1.0  # avoid division by zero
        else:
            # With multiple trajectories: normalize using the group's max coverage
            uid2max_coverage[uid] = max(coverage_list)

    # For each sample, normalize using its uid group's max coverage
    normalized_coverage_rates = [0.0] * len(coverage_rates)

    for i, (coverage, uid) in enumerate(zip(coverage_rates, uid_list)):
        max_coverage = uid2max_coverage[uid]

        if max_coverage == 0:
            # Max coverage in the group is 0, normalize to 0
            normalized_coverage_rates[i] = 0.0
        else:
            # Normalized coverage = raw coverage / max coverage in the group
            normalized_coverage_rates[i] = coverage / max_coverage

    # Log the normalization details for each uid group
    for uid in uid2coverage:
        coverage_list = uid2coverage[uid]
        max_coverage = uid2max_coverage[uid]
        logger.info(
            f"Within-group normalization (GRPO-style) [uid={uid[:8]}...]: "
            f"n_rollouts={len(coverage_list)}, max_coverage={max_coverage:.3f}"
        )

    logger.info(
        f"Overall normalization stats: n_samples={len(coverage_rates)}, "
        f"raw range=[{min(coverage_rates):.3f}, {max(coverage_rates):.3f}], "
        f"normalized range=[{min(normalized_coverage_rates):.3f}, {max(normalized_coverage_rates):.3f}]"
    )

    return normalized_coverage_rates


def extract_mid_facts_from_ground_truth(ground_truth: Optional[Dict[str, Any]]) -> List[str]:
    """Extract the list of intermediate facts from ground truth.

    Args:
        ground_truth: the ground-truth data dict

    Returns:
        The list of intermediate facts.
    """
    try:
        if not ground_truth:
            logger.warning("ground_truth is empty")
            return []

        # Try extracting the mid_facts field from ground_truth
        mid_facts = ground_truth.get('mid_facts', [])

        if isinstance(mid_facts, list):
            return mid_facts
        elif isinstance(mid_facts, str):
            # If it's a string, try splitting by line
            return [fact.strip() for fact in mid_facts.split('\n') if fact.strip()]
        else:
            logger.warning(f"Unsupported mid_facts field type: {type(mid_facts)}")
            return []

    except Exception as e:
        logger.error(f"Failed to extract intermediate facts: {e}")
        logger.error(traceback.format_exc())
        return []
