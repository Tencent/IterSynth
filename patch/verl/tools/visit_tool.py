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
Web Visit Tool (Jina Reader + LLM extraction backend)

A verl-compatible "visit" tool that fetches the full text of a webpage via
the Jina Reader API (https://jina.ai/reader) and then asks an OpenAI-
compatible LLM to extract the information relevant to a given goal.

Execution model:
    Shares the same Ray-based execution pool as `search_tool.py`
    (`init_search_execution_pool` / `PoolMode`), so blocking HTTP/LLM calls
    never block the SGLang rollout event loop.

Environment variables:
    JINA_API_KEY            - optional; Jina Reader API key (higher rate
                              limits). Anonymous access works with lower
                              limits. See https://jina.ai/reader.
    VISIT_SUMMARY_API_KEY   - API key for the OpenAI-compatible summarization
                              LLM (required for content extraction to work).
    VISIT_SUMMARY_API_BASE  - Base URL of the summarization LLM endpoint.
                              Default: https://api.openai.com/v1
    VISIT_SUMMARY_MODEL_NAME - Model name for the summarization LLM.
                              Default: gpt-4o-mini

Usage example:
    from verl.tools.visit_tool import VisitTool

    visit_tool = VisitTool(
        config={"num_workers": 16, "rate_limit": 16},
        tool_schema=openai_tool_schema,
    )
    tool_response, reward, metrics = await visit_tool.execute(
        instance_id="traj-0",
        parameters={"url": "https://example.com", "goal": "find the pricing"},
    )
"""

import json
import logging
import os
import time
from typing import Any, List, Optional
from uuid import uuid4

import requests
import tiktoken
from openai import OpenAI

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from .search_tool import PoolMode, init_search_execution_pool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
VISIT_SUMMARY_API_KEY = os.environ.get("VISIT_SUMMARY_API_KEY", "")
VISIT_SUMMARY_API_BASE = os.environ.get("VISIT_SUMMARY_API_BASE", "https://api.openai.com/v1")
VISIT_SUMMARY_MODEL_NAME = os.environ.get("VISIT_SUMMARY_MODEL_NAME", "gpt-4o-mini")

# JSON schema hint embedded in the extraction prompt, describing the exact
# fields the LLM must return.
OSS_JSON_FORMAT = (
    '# Response Format\n'
    '## visit_content\n'
    '{"properties": {'
    '"rational": {"type": "string", "description": '
    '"Locate the specific section(s)/data directly related to the goal within the webpage content"}, '
    '"evidence": {"type": "string", "description": '
    '"Identify and extract the most relevant information from the content. '
    'Never omit important details; preserve the original wording as much as possible."}, '
    '"summary": {"type": "string", "description": '
    '"A concise, logically organized paragraph summarizing how the content contributes to the goal."}'
    '}}'
)


def _truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    """Truncate `text` to at most `max_tokens` tokens (cl100k_base
    encoding), falling back to a rough character-based cut if tiktoken is
    unavailable for some reason."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoding.decode(tokens[:max_tokens])
    except Exception:
        return text[: max_tokens * 4]


def _build_extractor_prompt(webpage_content: str, goal: str) -> str:
    """Build the extraction prompt sent to the summarization LLM. Uses plain
    string concatenation (not str.format) because `OSS_JSON_FORMAT` itself
    contains literal `{`/`}` characters."""
    return (
        "Extract the information from the webpage content below that helps achieve the goal.\n\n"
        + OSS_JSON_FORMAT
        + "\n\nWebpage content:\n"
        + webpage_content
        + "\n\nGoal: "
        + goal
        + '\n\nRespond with a single JSON object matching the schema above, for example:\n'
        + '{"rational": "...", "evidence": "...", "summary": "..."}'
    )


def _jina_read_page(url: str, timeout: int = 50, max_retries: int = 3) -> str:
    """Fetch the readable text content of a webpage via the Jina Reader API."""
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"} if JINA_API_KEY else {}
    for attempt in range(max_retries):
        try:
            response = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.text
            logger.warning(f"[Visit] Jina Reader returned status {response.status_code} for {url}")
        except Exception as e:
            logger.warning(f"[Visit] Jina Reader request failed (attempt {attempt + 1}/{max_retries}): {e}")
        time.sleep(0.5)
    return "[Visit] Failed to read page."


def _call_summary_llm(messages: list, max_retries: int = 2) -> str:
    """Call an OpenAI-compatible chat completion endpoint to summarize /
    extract information from webpage content."""
    if not VISIT_SUMMARY_API_KEY:
        logger.warning("[Visit] VISIT_SUMMARY_API_KEY is not set; skipping content extraction.")
        return ""

    client = OpenAI(api_key=VISIT_SUMMARY_API_KEY, base_url=VISIT_SUMMARY_API_BASE)
    for attempt in range(max_retries):
        try:
            chat_response = client.chat.completions.create(
                model=VISIT_SUMMARY_MODEL_NAME,
                messages=messages,
                temperature=0.7,
            )
            content = chat_response.choices[0].message.content
            if content:
                try:
                    json.loads(content)
                except Exception:
                    # Try to salvage a JSON object embedded in extra text
                    left, right = content.find("{"), content.rfind("}")
                    if left != -1 and right != -1 and left <= right:
                        content = content[left : right + 1]
                return content
        except Exception as e:
            logger.warning(f"[Visit] Summary LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return ""
    return ""


def _failure_message(url: str, goal: str) -> str:
    return (
        f"The useful information in {url} for user goal {goal} as follows:\n\n"
        "Evidence in page: \nThe provided webpage content could not be accessed. "
        "Please check the URL or file format.\n\n"
        "Summary: \nThe webpage content could not be processed, and therefore, "
        "no information is available.\n\n"
    )


def visit_and_summarize(url: str, goal: str) -> str:
    """Fetch a webpage via Jina Reader, then extract goal-relevant evidence
    and a summary via an LLM call. Falls back to a graceful failure message
    if either step fails.

    Retries the LLM extraction step up to 3 times with progressively more
    aggressive content truncation if the response is too short or the JSON
    cannot be parsed, mirroring common failure modes of long-context LLM
    calls (context overflow, truncated generation, etc.).
    """
    content = _jina_read_page(url)
    if content.startswith("[Visit] Failed to read page."):
        return _failure_message(url, goal)

    content = _truncate_to_tokens(content, max_tokens=95000)
    messages = [{"role": "user", "content": _build_extractor_prompt(content, goal)}]

    raw = _call_summary_llm(messages)
    retries_left = 3
    while len(raw) < 10 and retries_left >= 0:
        truncate_length = int(0.7 * len(content)) if retries_left > 0 else 25000
        content = content[:truncate_length]
        messages = [{"role": "user", "content": _build_extractor_prompt(content, goal)}]
        raw = _call_summary_llm(messages)
        retries_left -= 1

    if isinstance(raw, str):
        raw = raw.replace("```json", "").replace("```", "").strip()

    parsed = None
    for _ in range(3):
        try:
            parsed = json.loads(raw)
            break
        except Exception:
            raw = _call_summary_llm(messages)

    if parsed is None:
        return _failure_message(url, goal)

    return (
        f"The useful information in {url} for user goal {goal} as follows:\n\n"
        f"Evidence in page: \n{parsed.get('evidence', '')}\n\n"
        f"Summary: \n{parsed.get('summary', '')}\n\n"
    )


class VisitTool(BaseTool):
    """Webpage visit tool: fetches page content via Jina Reader and
    extracts goal-relevant evidence/summary via an LLM call.

    Config options (all optional):
        num_workers (int): Ray worker thread count. Default 16.
        rate_limit (int): max concurrent visit requests. Default 16.
        enable_global_rate_limit (bool): whether to rate-limit globally. Default True.
        max_tool_response_length (int | None): optional response truncation length.
        max_batch_seconds (int): total time budget (seconds) across all URLs
            in a single call. Default 900 (15 minutes).
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict = {}

        self.num_workers = config.get("num_workers", 16)
        self.rate_limit = config.get("rate_limit", 16)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        self.max_tool_response_length = config.get("max_tool_response_length", None)
        self.max_batch_seconds = config.get("max_batch_seconds", 900)

        logger.info(f"VisitTool initialized with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"reward": []}
        return instance_id, ToolResponse()

    @staticmethod
    def _clean_urls(urls: List[str]) -> List[str]:
        cleaned = []
        for url in urls:
            if isinstance(url, str):
                u = url.strip().strip('"').strip("'")
                if u.startswith("http://") or u.startswith("https://"):
                    cleaned.append(u)
        return cleaned

    def execute_visit(self, urls: List[str], goal: str) -> str:
        """Visit every URL in `urls` sequentially, respecting the overall
        `max_batch_seconds` time budget. Executed inside the Ray worker."""
        start_time = time.time()
        responses = []
        for url in urls:
            if time.time() - start_time > self.max_batch_seconds:
                responses.append(_failure_message(url, goal))
                continue
            try:
                responses.append(visit_and_summarize(url, goal))
            except Exception as e:
                responses.append(f"Error fetching {url}: {e}")
        return "\n=======\n".join(responses)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        urls = parameters.get("url", [])
        goal = parameters.get("goal", "")

        if isinstance(urls, str):
            urls = [urls]

        cleaned_urls = self._clean_urls(urls) if isinstance(urls, list) else []

        if not cleaned_urls:
            error_msg = "Invalid request: 'url' is missing, empty, or contains no valid http(s) URLs."
            return ToolResponse(text=json.dumps({"result": error_msg}, ensure_ascii=False)), 0.0, {}

        if instance_id not in self._instance_dict:
            await self.create(instance_id)

        try:
            result_text = await self.execution_pool.execute.remote(self.execute_visit, cleaned_urls, goal)

            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            metrics = {
                "url_count": len(cleaned_urls),
                "status": "success",
                "result_length": len(result_text),
            }

            if self.max_tool_response_length and len(result_text) > self.max_tool_response_length:
                result_text = result_text[: self.max_tool_response_length] + "...(truncated)"

            return ToolResponse(text=json.dumps({"result": result_text}, ensure_ascii=False)), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Visit execution failed: {e}"}, ensure_ascii=False)
            logger.error(f"[VisitTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> list:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
