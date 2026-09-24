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
Web Search Tool (Serper.dev backend)

A verl-compatible search tool that queries the web via the Serper.dev
Google Search API (https://serper.dev). The tool itself is backend-agnostic:
all Serper-specific logic lives in `serper_search()` below, so swapping in a
different search provider (Tavily, Bing, your own retrieval service, etc.)
only requires rewriting that one function.

Execution model:
    Tool calls run inside a Ray actor pool (`SearchExecutionWorker`) so that
    blocking HTTP requests never block the SGLang rollout event loop. An
    optional token-bucket rate limiter (`TokenBucketWorker`) caps the number
    of concurrent outbound requests across the whole training job.

Required environment variable:
    SERPER_API_KEY - your Serper.dev API key (https://serper.dev)

Usage example:
    from verl.tools.search_tool import SearchTool, init_search_execution_pool

    search_tool = SearchTool(
        config={"num_workers": 16, "rate_limit": 16, "timeout": 30},
        tool_schema=openai_tool_schema,
    )
    tool_response, reward, metrics = await search_tool.execute(
        instance_id="traj-0",
        parameters={"query": ["reinforcement learning from human feedback"]},
    )
"""

import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor
import requests

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_API_URL = "https://google.serper.dev/search"


def _contains_chinese(text: str) -> bool:
    """Rough heuristic to detect Chinese characters, used to pick a
    reasonable search locale (zh-CN vs en-US)."""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def serper_search(query: str, timeout: int = 30, max_retries: int = 5) -> str:
    """Query the Serper.dev Google Search API for a single query string and
    format the organic results as a numbered list of citations.

    Args:
        query: The search query string.
        timeout: Per-request timeout in seconds.
        max_retries: Number of retries on network failure.

    Returns:
        A human-readable, LLM-friendly text block listing the search results,
        or a short natural-language message if the search failed / returned
        nothing.
    """
    if not SERPER_API_KEY:
        return "[Search] SERPER_API_KEY is not set. Please export SERPER_API_KEY to use this tool."

    if _contains_chinese(query):
        payload = {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
    else:
        payload = {"q": query, "location": "United States", "gl": "us", "hl": "en"}
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    response = None
    for attempt in range(max_retries):
        try:
            response = requests.post(SERPER_API_URL, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            break
        except Exception as e:
            logger.warning(f"[Search] Serper request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return f"Search request timed out for query '{query}'. Please try again later."

    try:
        results = response.json()
    except Exception:
        return f"No results found for query: '{query}'. Use a less specific query."

    if "organic" not in results or not results["organic"]:
        return f"No results found for query: '{query}'. Use a less specific query."

    web_snippets = []
    for idx, page in enumerate(results["organic"], start=1):
        date_published = f"\nDate published: {page['date']}" if "date" in page else ""
        source = f"\nSource: {page['source']}" if "source" in page else ""
        snippet = f"\n{page['snippet']}" if "snippet" in page else ""
        entry = f"{idx}. [{page.get('title', 'Untitled')}]({page.get('link', '')}){date_published}{source}\n{snippet}"
        entry = entry.replace("Your browser can't play this video.", "")
        web_snippets.append(entry)

    return (
        f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
        + "\n\n".join(web_snippets)
    )


# ==============================================================================
# Ray-based execution pool (backend-agnostic; also reused by visit_tool.py)
# ==============================================================================
# Adapted from verl/tools/sandbox_fusion_tools.py


class PoolMode(Enum):
    """Execution pool mode: ThreadMode is I/O-bound friendly; ProcessMode is
    reserved for future CPU-bound tool implementations and not implemented."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """A simple token-bucket rate limiter implemented as a Ray actor.

    The `acquire` concurrency group is capped at 1 to serialize token
    acquisition, while `release` allows higher concurrency since releasing
    a token never blocks.
    """

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # for observability only
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


class SearchExecutionWorker:
    """Executes a blocking function, optionally applying a global rate
    limit backed by a named `TokenBucketWorker` Ray actor."""

    def __init__(self, enable_global_rate_limit: bool = True, rate_limit: int = 10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit: int):
        # `get_if_exists=True` ensures there is only ever one rate limiter
        # actor per Ray cluster, shared across all tool instances.
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"[Search] Error during rate-limited execution: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_search_execution_pool(
    num_workers: int,
    enable_global_rate_limit: bool = True,
    rate_limit: int = 10,
    mode: PoolMode = PoolMode.ThreadMode,
):
    """Create a Ray actor pool for executing tool calls with bounded
    concurrency and an optional global rate limit."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(SearchExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    raise NotImplementedError("Process mode is not implemented")


# ==============================================================================
# SearchTool
# ==============================================================================


class SearchTool(BaseTool):
    """Web search tool backed by the Serper.dev Google Search API.

    Config options (all optional):
        num_workers (int): Ray worker thread count. Default 16.
        rate_limit (int): max concurrent outbound Serper requests. Default 16.
        timeout (int): per-request timeout in seconds. Default 30.
        enable_global_rate_limit (bool): whether to rate-limit globally. Default True.
        max_tool_response_length (int | None): optional response truncation length.
        tool_response_truncate_side (str): "left" | "right" | "both". Default "left".
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict = {}

        self.num_workers = config.get("num_workers", 16)
        self.rate_limit = config.get("rate_limit", 16)
        self.timeout = config.get("timeout", 30)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        self.max_tool_response_length = config.get("max_tool_response_length", None)
        self.tool_response_truncate_side = config.get("tool_response_truncate_side", "left")

        logger.info(f"SearchTool initialized with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"reward": []}
        return instance_id, ToolResponse()

    def execute_search(self, query_list: List[str]) -> str:
        """Run every query in `query_list` and join the formatted results.
        Executed inside the Ray worker, so it may block synchronously."""
        responses = [serper_search(q, timeout=self.timeout) for q in query_list]
        return "\n=======\n".join(responses)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        # Accept a few different parameter names for compatibility with
        # different prompt/tool-schema conventions.
        query_list = parameters.get("query") or parameters.get("query_list") or parameters.get("queries")

        if isinstance(query_list, str):
            query_list = [query_list]

        if not query_list or not isinstance(query_list, list):
            error_msg = "Invalid request: parameters must contain a 'query' field with a list of strings."
            return (
                ToolResponse(text=json.dumps({"result": error_msg}, ensure_ascii=False)),
                0.0,
                {},
            )

        if instance_id not in self._instance_dict:
            await self.create(instance_id)

        try:
            result_text = await self.execution_pool.execute.remote(self.execute_search, query_list)

            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            metrics = {
                "query_count": len(query_list),
                "status": "success",
                "result_length": len(result_text),
            }

            if self.max_tool_response_length and len(result_text) > self.max_tool_response_length:
                if self.tool_response_truncate_side == "left":
                    result_text = result_text[: self.max_tool_response_length] + "...(truncated)"
                elif self.tool_response_truncate_side == "right":
                    result_text = "(truncated)..." + result_text[-self.max_tool_response_length :]
                else:
                    length = self.max_tool_response_length // 2
                    result_text = result_text[:length] + "...(truncated)..." + result_text[-length:]

            return ToolResponse(text=json.dumps({"result": result_text}, ensure_ascii=False)), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"}, ensure_ascii=False)
            logger.error(f"[SearchTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> list:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
