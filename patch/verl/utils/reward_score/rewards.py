#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reinforcement-learning reward scoring system

This module provides multi-dimensional reward scoring, supporting both
answer-quality evaluation and trajectory-quality evaluation, for a
comprehensive quality assessment and reward computation of
model-generated sequences.

Main features:
- Multi-dimensional answer-quality evaluation (content organization, intent understanding, refusal, reading experience, etc.)
- Trajectory-quality evaluation (logical coherence, query redundancy)
- Reference-answer comparison evaluation
- Search-turn reward computation
- Concurrent evaluation and score normalization
- Multiple reward-function strategies (mean score, min score, reference comparison, etc.)

Core characteristics:
- **Multi-dimensional evaluation**: supports 8 answer-quality dimensions and 2 trajectory-quality dimensions
- **Flexible configuration**: configurable dimension weights, evaluation model, prompts, etc.
- **Concurrent processing**: uses a thread pool to evaluate multiple dimensions concurrently, for efficiency
- **Multiple strategies**: provides mean, min, ref_v1, ref_v2, trajectory and other reward-computation strategies
- **Two-stage evaluation**: supports generating an answer first, then evaluating it
- **Error recovery**: thorough error handling and graceful degradation

Evaluation dimensions:
    Answer-quality dimensions:
    - Content organization: whether the answer's structure and organization are reasonable
    - Intent understanding: how accurately the user's intent is understood
    - Refusal: whether the refusal is reasonable and polite
    - Reading experience: the answer's readability and user experience
    - Content quality - faithfulness: how faithful the answer is to the reference documents
    - Content quality - timeliness: the timeliness of the answer's information
    - Content quality - credibility: the credibility of the answer's content
    - Content quality - other: other content-quality factors

    Trajectory-quality dimensions:
    - Logical coherence: the logical coherence of the reasoning process
    - Query redundancy: how repetitive and redundant the search queries are

Usage example:
    # 1. Use the mean-score strategy
    from rewards import my_reward_function_mean

    config = {
        'url': 'http://api.example.com',
        'model': 'your-model',
        'max_retries': 3,
        'max_assistant_turns': 5
    }

    result = my_reward_function_mean(
        config=config,
        prompt_str=prompt,
        sequences_str=sequence,
        ground_truth=None,
        data_source='test',
        messages=messages,
        task_extra_info={}
    )
    print(f"train score: {result['train_final_score']}")
    print(f"val score: {result['val_final_score']}")

    # 2. Use the reference-comparison strategy
    from rewards import my_reward_function_ref_v1

    result = my_reward_function_ref_v1(
        config=config,
        prompt_str=prompt,
        sequences_str=sequence,
        ground_truth={'rm_scores': ['{...}']},
        data_source='test',
        messages=messages,
        task_extra_info={}
    )

    # 3. Use the trajectory-evaluation strategy
    from rewards import my_reward_function_trajectory

    result = my_reward_function_trajectory(
        config=config,
        prompt_str=prompt,
        sequences_str=sequence,
        ground_truth=None,
        data_source='test',
        messages=messages,
        task_extra_info={}
    )

Architecture:
    - RewardConfig: config-management class, centralizes all config items
    - TextExtractor: text-extraction utility, extracts info from prompts and responses
    - EvalItemBuilder: evaluation-item builder, builds the data structures needed for evaluation
    - ModelAPIClient: model API client, with retry and error handling
    - AnswerEvaluator: answer evaluator, evaluates each answer-quality dimension
    - TrajectoryEvaluator: trajectory evaluator, evaluates the quality of the reasoning trajectory
    - ScoreCalculator: score calculator, normalizes and aggregates per-dimension scores
    - RewardOrchestrator: reward-computation orchestrator, coordinates the whole evaluation flow

NOTE: the dimension names below are kept in Chinese on purpose -- they are the
external data contract used for the score-dict keys and for matching against
the prompt template files under prompts/ (e.g. "内容组织" ->
prompt_neirongzuzhi.txt). Translating them would break compatibility with
existing ground-truth data and prompt files.

Version: 1.0.0
"""

import json
import logging
import os
import random
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import json_repair
import requests
from requests.exceptions import Timeout, ConnectionError, RequestException

from verl.utils.reward_score.api_generative import prime_query_openai_async

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardConfig:
    """Reward-evaluation config management class.

    Centralizes all evaluation-related config items, including evaluation
    dimensions, dimension mappings, weights, and API settings.
    """

    # Answer-quality evaluation dimensions (8 dimensions).
    # NOTE: kept in Chinese -- these are the score-dict keys / prompt-file contract.
    DIMENSIONS: List[str] = [
        "内容组织", "意图理解", "拒答", "阅读体验",
        "内容质量-忠实度", "内容质量-时效性", "内容质量-可信度", "内容质量-其他"
    ]

    # Trajectory-quality evaluation dimensions (2 dimensions).
    TRAJECTORY_DIMENSIONS: List[str] = ["逻辑性", "检索词重复冗余性"]

    # Mapping from dimension name to file name (used to load the corresponding prompt file)
    DIM_MAP: Dict[str, str] = {
        "内容组织": "neirongzuzhi",
        "意图理解": "yitulijie",
        "拒答": "juda",
        "阅读体验": "yuedutiyan",
        "多轮体验": "duoluntiyan",
        "内容质量": "neirongzhiliang",
        "COT": "cot",
        "COT一致性": "cotyizhixing",
        "逻辑性": "trajectory_reasoning",
        "检索词重复冗余性": "trajectory_subq",
        "内容质量-忠实度": "neirongzhiliang_zhongshidu",
        "内容质量-时效性": "neirongzhiliang_shixiaoxing",
        "内容质量-可信度": "neirongzhiliang_kexindu",
        "内容质量-其他": "neirongzhiliang_other",
    }

    # Per-dimension weight config (used for the weighted average that yields the final score)
    DIM_WEIGHT: Dict[str, float] = {
        "内容组织": 0.1,
        "意图理解": 0.1,
        "拒答": 0.1,
        "阅读体验": 0.1,
        "内容质量-忠实度": 0.2,
        "内容质量-时效性": 0.2,
        "内容质量-可信度": 0.2,
        "内容质量-其他": 0.2
    }

    # API auth token pool (for load balancing and fault tolerance)
    TOKENS: List[str] = []

    # Default config path (defaults to the prompts/ directory next to this
    # file; overridable via REWARD_PROMPT_DIR)
    DEFAULT_PROMPT_DIR: str = os.environ.get(
        "REWARD_PROMPT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts"),
    )
    DEFAULT_MODEL_NAME: str = "rm_0918sp_fix_qw_8b_v1_step130"

    # Scoring range constants
    MIN_SCORE: float = 0.0
    MAX_SCORE: float = 4.0
    VALID_SCORES: List[float] = [0.0, 1.0, 2.0, 3.0, 4.0]
    NORMALIZATION_FACTOR: float = 3.0  # converts a 0-4 score into 0-1

    # Document length limit (avoids exceeding the model's context limit)
    MAX_DOCS_LENGTH: int = 95000


class TextExtractor:
    """Text-extraction utility class.

    Provides static methods for extracting structured information from
    prompt and response strings, including the query, conversation
    history, and metadata.
    """

    @staticmethod
    def extract_prompt_info(prompt_str: str) -> Tuple[Optional[str], List[Dict[str, str]], str]:
        """Extract the query, conversation history, and timestamp from a prompt string.

        Args:
            prompt_str: the raw prompt string to parse.

        Returns:
            A tuple containing:
                - the current user query (Optional[str])
                - the conversation history list (List[Dict[str, str]])
                - the current timestamp (str)
        """
        # Extract the conversation history
        history: List[Dict[str, str]] = []
        try:
            # Match the multi-turn dialogue pattern: user\n...\nassistant\n...\nuser\n
            qa_pattern = r"user\n(.*?)\nassistant\n(.*?)\nuser\n"
            matches = re.findall(qa_pattern, prompt_str, re.DOTALL)
            if matches:
                for qa in matches:
                    query = qa[0].strip()
                    answer = qa[1].strip()
                    history.append({"query": query, "answer": answer})
        except Exception as e:
            logger.error(f"Error while extracting the conversation history: {e}")

        # Extract the current user question (supports multiple formats)
        q: Optional[str] = None
        try:
            # Try the first format: Question: ...
            q_pattern = r"Question: (.*?)\nassistant"
            matches = re.findall(q_pattern, prompt_str, re.DOTALL)
            if matches:
                q = matches[0].strip()
            else:
                # Try the second format: The user's message is: ...
                q_pattern = r"The user's message is:(.*?)assistant"
                matches = re.findall(q_pattern, prompt_str, re.DOTALL)
                if matches:
                    q = matches[0].strip()
        except Exception as e:
            logger.error(f"Error while extracting the query: {e}")

        # Extract the time info (if the prompt doesn't have it, use the current time)
        current_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            time_pattern = r"user\nThe current date is (.*?). Your primary task is to solve the user\\'s questions"
            matches = re.findall(time_pattern, prompt_str, re.DOTALL)
            if matches:
                current_time = matches[-1].strip()
        except Exception as e:
            logger.error(f"Error while extracting the current time: {e}")

        return q, history, current_time

    @staticmethod
    def extract_response_info(response_str: str) -> Tuple[str, str, str, int, str]:
        """Extract the documents, reasoning process, and answer from a model response.

        Args:
            response_str: the model-generated response string.

        Returns:
            A tuple containing:
                - the search documents (str)
                - the thinking process (str)
                - the final answer (str)
                - the number of search turns (int)
                - the readable trajectory (str)
        """
        # Extract the search results of every round (from the <tool_response> tags)
        doc_pattern = r"<tool_response>(.*?)</tool_response>"
        matches = re.findall(doc_pattern, response_str, re.DOTALL)

        if matches:
            docs = [m.strip() for m in matches]
            doc_str = "\n\n".join(docs).strip()
            num_turns = len(matches)  # the number of search turns equals the number of tool_responses
        else:
            doc_str = ""
            num_turns = 0

        # Extract the last round's thinking process (from the <think> tags)
        think_pattern = r"<think>(.*?)</think>"
        matches = re.findall(think_pattern, response_str, re.DOTALL)
        think = matches[-1].strip() if matches else ""

        # Extract the final answer (prefers the <answer> tag, otherwise takes whatever follows </think>)
        answer = ""
        try:
            answer_pattern = r"<answer>(.*?)</answer>"
            matches = re.findall(answer_pattern, response_str, re.DOTALL)
            if matches:
                answer = matches[-1].strip()
            else:
                # If there is no answer tag, try extracting from after </think>
                answer = response_str.split("</think>")[-1].strip()
                # If it contains a tool_call tag, we're still in the search phase: there is no final answer yet
                if "<tool_call>" in answer:
                    answer = ""
        except Exception as e:
            logger.error(f"Error while extracting the answer: {e}")
            answer = ""

        # Build a readable reasoning trajectory (each round's thinking + search query)
        readable_trajectory = ""
        trajectory_pattern = r"assistant\n<think>(.*?)</think>\d*<tool_call>(.*?)</tool_call>"
        matches = re.findall(trajectory_pattern, response_str, re.DOTALL)
        if matches:
            for m in matches:
                readable_trajectory += f"<think>{m[0]}</think>\n"
                readable_trajectory += f"<search_query>{m[1]}</search_query>\n"
        # Append the last round's thinking
        readable_trajectory += f"<think>{think}</think>\n"

        return doc_str, think, answer, num_turns, readable_trajectory

    @staticmethod
    def history_to_str(history: List[Dict[str, str]]) -> str:
        """Convert a conversation history into a formatted string.

        Args:
            history: the list of dialogue turns, each with a query and an answer.

        Returns:
            The formatted conversation-history string.
        """
        qa_str = ""
        for i, qa in enumerate(history):
            qa_str += f"第{i+1}轮用户prompt：\n{qa['query']}\n第{i+1}轮回复：\n{qa['answer']}\n"
        return qa_str


class EvalItemBuilder:
    """Evaluation-item builder.

    Builds a structured evaluation data item from the raw prompt and
    sequence strings, for downstream evaluation tasks.
    """

    @staticmethod
    def build_from_sequences(sequence_str: str, prompt_str: str) -> Dict[str, Any]:
        """Build an evaluation data item from a sequence and a prompt.

        Args:
            sequence_str: the model-generated sequence string.
            prompt_str: the input prompt string.

        Returns:
            A dict containing all fields needed for evaluation, including
            user_question, answer, reference_doc, think, num_turns, etc.
        """
        # Extract info from the prompt
        query, history, current_time = TextExtractor.extract_prompt_info(prompt_str)

        # Extract info from the response
        reference_doc, think, answer, num_turns, readable_trajectory = TextExtractor.extract_response_info(
            sequence_str
        )

        # Build the combined documents (conversation history + current search results)
        combined_docs = ""
        if not history:
            combined_docs += "当前为第一轮，无历史对话信息\n\n"
        else:
            history_str = TextExtractor.history_to_str(history)
            combined_docs += f"# 多轮对话历史：\n{history_str}\n\n"

        if reference_doc:
            combined_docs += f"# 当前轮次搜索结果:\n{reference_doc}"
        else:
            combined_docs += "# 没有找到相关搜索结果"

        return {
            "user_question": query,
            "answer": answer,
            "reference_doc": reference_doc,
            "combined_docs": combined_docs,
            "think": think,
            "current_time": current_time,
            "num_turns": num_turns,
            "readable_trajectory": readable_trajectory,
            "history": TextExtractor.history_to_str(history) if history else ""
        }


class BaseAPIClient:
    """API client base class, providing generic HTTP request and retry logic.

    Encapsulates common functionality such as HTTP requests, a retry
    mechanism, and error handling, serving as the base for specific API clients.
    """

    def __init__(self, base_url: str, default_config: Dict[str, Any] = None):
        """Initialize the API client.

        Args:
            base_url: the API base URL.
            default_config: the default config dict.
        """
        self.base_url = base_url
        self.default_config = default_config or {}

    def _handle_response_status(self, response: requests.Response, attempt: int) -> Optional[bool]:
        """Uniformly handle the HTTP response status code.

        Args:
            response: the HTTP response object.
            attempt: the current retry count.

        Returns:
            True on success, False on an unrecoverable error, None when a retry is needed.
        """
        status_code = response.status_code

        if status_code == 200:
            # Request succeeded
            return True
        elif status_code == 400:
            # Bad request parameters, unrecoverable
            logger.error(f"Bad Request (400): {response.text[:500]}")
            return False
        elif status_code == 401:
            # Auth failure, likely a token issue, retry
            logger.error("Unauthorized (401): Invalid token")
            return None
        elif status_code == 429:
            # Rate limited, wait according to the Retry-After header
            retry_after = int(response.headers.get('Retry-After', 30))
            logger.warning(f"Rate limited (429), waiting {retry_after}s")
            time.sleep(retry_after)
            return None
        elif status_code in [500, 503]:
            # Server error, retry after exponential backoff
            logger.warning(f"Server error ({status_code})")
            time.sleep(5 * (attempt + 1))
            return None
        else:
            # Other errors, wait and retry
            logger.error(f"HTTP error {status_code}: {response.text[:200]}")
            time.sleep(5)
            return None

    def _call_api_with_retry(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        auth_token: Optional[str] = None,
        max_retries: int = 20,
        base_timeout: int = 60,
        use_token_pool: bool = False,
        adaptive_timeout: bool = False
    ) -> Optional[Dict[str, Any]]:
        """The core API-calling function, with retry logic and error handling.

        Args:
            endpoint: the API endpoint path.
            payload: the API request payload dict.
            auth_token: the auth token (used when use_token_pool=False).
            max_retries: the max number of retries.
            base_timeout: the base timeout (seconds).
            use_token_pool: whether to pick a token at random from the configured token pool.
            adaptive_timeout: whether to use an adaptive timeout (exponential growth).

        Returns:
            The API response's JSON data, or None if all retries fail.
        """
        for attempt in range(max_retries):
            # Pick the auth token according to the config (supports token-pool load balancing)
            if use_token_pool:
                token = random.choice(RewardConfig.TOKENS)
            else:
                token = auth_token

            # Build the request headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }

            # Compute the timeout (supports the adaptive-timeout strategy)
            if adaptive_timeout:
                # Double the timeout every 5 retries, up to 600 seconds
                timeout = min(base_timeout * (2 ** (attempt // 5)), 600)
            else:
                timeout = 600

            try:
                # Send the HTTP POST request
                full_url = f"{self.base_url}/{endpoint.lstrip('/')}"
                response = requests.post(
                    full_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=timeout
                )

                # Handle the response status code
                status_result = self._handle_response_status(response, attempt)

                if status_result is True:
                    # Request succeeded, parse the JSON response
                    try:
                        return response.json()
                    except json.JSONDecodeError as e:
                        logger.error(f"Response parsing error: {e}")
                        time.sleep(3)
                        continue
                elif status_result is False:
                    # Unrecoverable error (e.g. 400), return directly
                    return None
                else:
                    # status_result is None, meaning a retry is needed
                    continue

            except (Timeout, ConnectionError, RequestException) as e:
                # Network-related exception, retry after exponential backoff
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                sleep_time = min(2 ** attempt, 60)  # exponential backoff, at most 60 seconds
                time.sleep(sleep_time)
                continue
            except Exception as e:
                # Other unexpected exception
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                logger.error(traceback.format_exc())
                time.sleep(5)
                continue

        # All retries failed
        logger.error(f"All {max_retries} attempts failed")
        return None


class ModelAPIClient(BaseAPIClient):
    """Model API client.

    Encapsulates the interaction logic with an external model API, with
    retry mechanism, error handling, and load balancing. Provides a
    unified calling interface for multiple model types (OpenAI format, etc.).

    Core characteristics:
    - Supports multiple model formats
    - Smart retry mechanism and error recovery
    - Load balancing and token-pool management
    - Adaptive timeout strategy
    - Config validation and safety checks
    - Backward-compatible static methods
    - Caching support
    - Detailed logging

    Usage example:
        ```python
        # Basic usage
        client = ModelAPIClient()

        # Call an OpenAI-format model
        messages = [
            {"role": "user", "content": "hello, please introduce reinforcement learning"}
        ]
        result = client.call_openai_format(messages, model='gpt-5')

        # Call a thinking-enabled model
        content, thinking = client.call_deepseek_v31(messages)

        # Custom config
        config = {
            "max_retries": 10,
            "base_timeout": 30,
            "enable_cache": True,
            "cache_ttl": 600
        }
        client = ModelAPIClient(default_config=config)

        # Get performance stats
        metrics = client.get_metrics()
        print(f"total calls: {metrics['total_calls']}")
        print(f"success rate: {metrics['successful_calls'] / metrics['total_calls'] * 100:.1f}%")
        ```

    Config options:
        - max_retries: the max number of retries (default 20)
        - base_timeout: the base timeout in seconds (default 60)
        - use_token_pool: whether to use token-pool load balancing (default True)
        - adaptive_timeout: whether to use an adaptive timeout (default True)
        - enable_cache: whether to enable caching (default False)
        - cache_ttl: the cache TTL in seconds (default 300)
        - enable_metrics: whether to enable performance monitoring (default True)
    """

    # Supported model types and their default config
    SUPPORTED_MODELS = {
        'gpt-5': {'type': 'openai', 'max_tokens': 4096},
        'deepseek-v3.1-terminus': {'type': 'deepseek', 'max_tokens': 8192},
        'qwen-7b': {'type': 'openai', 'max_tokens': 2048}
    }

    def __init__(self, base_url: str = None, default_config: Dict[str, Any] = None):
        """Initialize the model API client.

        Args:
            base_url: the API base URL; uses the default when None.
            default_config: the default config dict.

        Raises:
            ValueError: when a config parameter is invalid.
        """
        base_url = base_url or "https://api.openai.com/v1/chat/completions"
        default_config = default_config or {
            "max_retries": 20,
            "base_timeout": 60,
            "use_token_pool": True,
            "adaptive_timeout": True,
            "enable_cache": False,
            "cache_ttl": 300,
            "enable_metrics": True
        }

        # Validate the config parameters
        self._validate_config(default_config)

        super().__init__(base_url, default_config)

        # Initialize the performance stats
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'average_response_time': 0.0,
            'last_call_timestamp': None
        }

        # Initialize the cache (if enabled)
        self._cache = {}
        logger.info("[OK] ModelAPIClient initialized")

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate that the config parameters are legal.

        Args:
            config: the config dict.

        Raises:
            ValueError: when a config parameter is invalid.
        """
        if config.get("max_retries", 0) < 0:
            raise ValueError("max_retries must be >= 0")
        if config.get("base_timeout", 0) <= 0:
            raise ValueError("base_timeout must be > 0")
        if config.get("cache_ttl", 0) < 0:
            raise ValueError("cache_ttl must be >= 0")

    def _update_metrics(self, success: bool, response_time: float) -> None:
        """Update the performance stats.

        Args:
            success: whether the call succeeded.
            response_time: the response time (seconds).
        """
        if not self.default_config.get("enable_metrics", True):
            return

        self.metrics['total_calls'] += 1
        if success:
            self.metrics['successful_calls'] += 1
        else:
            self.metrics['failed_calls'] += 1

        # Compute the average response time (moving average)
        if self.metrics['total_calls'] == 1:
            self.metrics['average_response_time'] = response_time
        else:
            alpha = 0.1  # smoothing factor
            self.metrics['average_response_time'] = (
                alpha * response_time +
                (1 - alpha) * self.metrics['average_response_time']
            )

        self.metrics['last_call_timestamp'] = time.time()

    def get_metrics(self) -> Dict[str, Any]:
        """Get the current performance stats.

        Returns:
            A dict containing the various performance metrics.
        """
        return self.metrics.copy()

    def _get_cache_key(self, messages: List[Dict[str, str]], model: str) -> str:
        """Generate a cache key.

        Args:
            messages: the list of conversation messages.
            model: the model name.

        Returns:
            The cache key string.
        """
        import hashlib
        content = json.dumps(messages, sort_keys=True) + model
        return hashlib.md5(content.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check whether data for the given key exists in the cache.

        Args:
            cache_key: the cache key.

        Returns:
            The cached data, or None if it doesn't exist.
        """
        if not self.default_config.get("enable_cache", False):
            return None

        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            cache_ttl = self.default_config.get("cache_ttl", 300)

            if time.time() - timestamp < cache_ttl:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_data
            else:
                # Cache expired, delete it
                del self._cache[cache_key]

        return None

    def _set_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Store data into the cache.

        Args:
            cache_key: the cache key.
            data: the data to cache.
        """
        if self.default_config.get("enable_cache", False):
            self._cache[cache_key] = (data, time.time())
            logger.debug(f"Cache set: {cache_key}")

    def _call_model_api(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0,
        thinking_enabled: bool = False,
        max_retries: int = None,
        base_timeout: int = None,
        use_cache: bool = None
    ) -> Optional[Dict[str, Any]]:
        """Generic model API calling method.

        Args:
            messages: the list of conversation messages.
            model: the model name.
            temperature: the temperature parameter.
            thinking_enabled: whether to enable the thinking-chain feature.
            max_retries: the max number of retries (uses the default when None).
            base_timeout: the base timeout (uses the default when None).
            use_cache: whether to use the cache (uses the default when None).

        Returns:
            The API response data, or None on failure.
        """
        # Validate model support
        if model not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model} is not in the supported list, it may not work correctly")

        # Use the default config or the passed-in parameters
        max_retries = max_retries or self.default_config.get("max_retries", 20)
        base_timeout = base_timeout or self.default_config.get("base_timeout", 60)
        use_cache = use_cache if use_cache is not None else self.default_config.get("enable_cache", False)

        # Check the cache (if enabled)
        cache_key = self._get_cache_key(messages, model)
        if use_cache:
            cached_response = self._check_cache(cache_key)
            if cached_response:
                self._update_metrics(True, 0.0)  # cache hit, response time is 0
                return cached_response

        # Build the payload
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "stream": False,
        }

        # If the thinking-chain feature is enabled, add the corresponding parameter
        if thinking_enabled:
            payload["thinking_enabled"] = True

        # Call the API and record performance
        start_time = time.time()
        try:
            response_data = self._call_api_with_retry(
                endpoint="chat/completions",
                payload=payload,
                auth_token=None,
                max_retries=max_retries,
                base_timeout=base_timeout,
                use_token_pool=self.default_config.get("use_token_pool", True),
                adaptive_timeout=self.default_config.get("adaptive_timeout", True)
            )

            response_time = time.time() - start_time

            if response_data:
                # Update the performance stats
                self._update_metrics(True, response_time)

                # Cache the response data (if enabled)
                if use_cache:
                    self._set_cache(cache_key, response_data)

                logger.info(f"[OK] API call succeeded: model={model}, response_time={response_time:.2f}s")
                return response_data
            else:
                # The call failed
                self._update_metrics(False, response_time)
                logger.error(f"[FAIL] API call failed: model={model}")
                return None

        except Exception as e:
            response_time = time.time() - start_time
            self._update_metrics(False, response_time)
            logger.error(f"[ERROR] API call exception: model={model}, error={str(e)}")
            logger.error(traceback.format_exc())
            return None

    def call_openai_format(
        self,
        messages: List[Dict[str, str]],
        model: str = 'gpt-5',
        max_retries: int = None,
        base_timeout: int = None
    ) -> Optional[str]:
        """Call an OpenAI-format model API.

        Args:
            messages: the list of conversation messages, with role and content.
            model: the model name.
            max_retries: the max number of retries.
            base_timeout: the base timeout (seconds).

        Returns:
            The model-generated text content, or None on failure.
        """
        response_data = self._call_model_api(
            messages=messages,
            model=model,
            max_retries=max_retries,
            base_timeout=base_timeout
        )

        if response_data:
            try:
                return response_data['choices'][0]['message']['content']
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to extract content from response: {e}")
                return None
        return None

    def call_deepseek_v31(
        self,
        messages: List[Dict[str, str]],
        model: str = 'deepseek-v3.1-terminus',
        max_retries: int = None,
        base_timeout: int = None
    ) -> Tuple[str, str]:
        """Call a thinking-enabled model (DeepSeek V3.1 style).

        Args:
            messages: the list of conversation messages.
            model: the model name.
            max_retries: the max number of retries.
            base_timeout: the base timeout (seconds).

        Returns:
            A tuple of (generated content, thinking process); returns ("", "") on failure.
        """
        response_data = self._call_model_api(
            messages=messages,
            model=model,
            thinking_enabled=True,
            max_retries=max_retries,
            base_timeout=base_timeout
        )

        if response_data:
            try:
                message = response_data.get("choices", [{}])[0].get("message", {})
                thinking_content = message.get("reasoning_content", "")
                content = message.get("content", "")
                return content, thinking_content
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to extract content from response: {e}")
                return "", ""
        return "", ""


class AnswerEvaluator:
    """Answer-quality evaluator.

    Evaluates answer quality across multiple dimensions, including content
    organization, intent understanding, reading experience, etc.
    """

    def __init__(self, model_name: str, prompt_dir: str):
        """Initialize the answer evaluator.

        Args:
            model_name: the evaluation model name.
            prompt_dir: the directory containing the prompt template files.
        """
        self.model_name = model_name
        self.prompt_dir = prompt_dir
        self.dimensions = RewardConfig.DIMENSIONS
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """Load the prompt templates for all evaluation dimensions."""
        prompts = {}
        for dimension in self.dimensions:
            dim = RewardConfig.DIM_MAP.get(dimension)
            if not dim:
                logger.warning(f"[FAIL] No mapping found in DIM_MAP for dimension {dimension}")
                prompts[dimension] = ""
                continue

            prompt_file = os.path.join(self.prompt_dir, f"prompt_{dim}.txt")
            if os.path.exists(prompt_file):
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        prompts[dimension] = f.read()
                        logger.info(f"[OK] Loaded the prompt file for dimension {dimension}")
                except IOError as e:
                    logger.error(f"[FAIL] Failed to read file {prompt_file}: {e}")
                    prompts[dimension] = ""
            else:
                logger.warning(f"[FAIL] Prompt file not found for dimension {dimension}: {prompt_file}")
                prompts[dimension] = ""
        return prompts

    def construct_eval_prompt(self, item: Dict[str, Any], dimension: str) -> Optional[str]:
        """Build the evaluation prompt for a specific dimension.

        Args:
            item: the data item containing the info needed for evaluation.
            dimension: the evaluation dimension name.

        Returns:
            The built prompt string, or None on failure.
        """
        try:
            query = item.get('user_question', '')
            answer = item.get('answer', '')
            docs = item.get('combined_docs', '')
            current_time = item.get('current_time', '')

            if not query or not answer:
                logger.error(f"Dimension {dimension}: query or answer is empty")
                return None

            prompt_template = self.prompts.get(dimension, "")
            if not prompt_template:
                logger.error(f"Dimension {dimension} has no prompt template")
                return None

            # Truncate the string (to avoid exceeding the model's context limit)
            if len(docs) > RewardConfig.MAX_DOCS_LENGTH:
                docs = docs[:RewardConfig.MAX_DOCS_LENGTH]
                logger.warning(f"Document content is too long, truncated to {RewardConfig.MAX_DOCS_LENGTH} chars")

            prompt_content = prompt_template.format(
                docs=docs,
                query=query,
                date_str=current_time,
                answer=answer
            )

            return prompt_content

        except KeyError as e:
            logger.error(f"Missing required field while building the prompt (dimension={dimension}): {e}")
            return None
        except Exception as e:
            logger.error(f"Error while building the prompt (dimension={dimension}): {e}")
            logger.error(traceback.format_exc())
            return None

    def extract_score(self, model_output_text: str, dimension: str) -> float:
        """Extract the score from the model's output text.

        Args:
            model_output_text: the model's output text.
            dimension: the evaluation dimension name.

        Returns:
            The extracted score (0-4); returns 0.0 if extraction fails.
        """
        try:
            # NOTE: this regex must match the score marker emitted by the judge model,
            # so it is intentionally kept in its original (Chinese) form.
            pattern_score = re.compile(r"维度整体得分：(\d)\n", re.DOTALL)
            score_match = pattern_score.search(model_output_text)
            if score_match:
                score = float(score_match.group(1))
                # Validate the score range
                if score in RewardConfig.VALID_SCORES:
                    return score
                else:
                    logger.warning(f"Dimension {dimension}: extracted score {score} is out of the valid range")
                    return 0.0

            logger.warning(f"Dimension {dimension}: could not extract a score from the output")
            return 0.0
        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to extract the score ({dimension}): {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Unknown error while extracting the score ({dimension}): {e}")
            return 0.0

    def evaluate_dimension(self, config: Dict[str, Any], item: Dict[str, Any], dimension: str) -> float:
        """Evaluate a single dimension and return its score.

        Args:
            config: the config dict.
            item: the evaluation data item.
            dimension: the evaluation dimension name.

        Returns:
            The evaluation score (0-4); returns 0.0 on failure.
        """
        eval_prompt = self.construct_eval_prompt(item, dimension)
        if eval_prompt is None:
            logger.warning(f"Dimension {dimension}: failed to build the prompt")
            return config.get('default_score', 0.0)

        resp = prime_query_openai_async(config, eval_prompt)
        if resp is None:
            logger.warning(f"Dimension {dimension}: API call failed")
            return config.get('default_score', 0.0)

        try:
            content = resp.json()['choices'][0]['message']['content']
            score = self.extract_score(content, dimension)

            # Validate that the score is legal (extract_score already does this; this is a double check)
            if score not in RewardConfig.VALID_SCORES:
                logger.warning(f"Dimension {dimension} returned an invalid score: {score}")
                return 0.0
            return score
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Failed to parse the response while evaluating dimension {dimension}: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to evaluate dimension {dimension}: {e}")
            logger.error(traceback.format_exc())
            return 0.0


class TrajectoryEvaluator:
    """Trajectory-quality evaluator.

    Evaluates the quality of the reasoning trajectory, including logical
    coherence and query redundancy.
    """

    def __init__(self, model: str, prompt_dir: str):
        self.model = model
        self.prompt_dir = prompt_dir
        self.dimensions = RewardConfig.TRAJECTORY_DIMENSIONS
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict:
        """Load the prompt templates for all evaluation dimensions."""
        prompts = {}
        for dimension in self.dimensions:
            dim = RewardConfig.DIM_MAP[dimension]
            prompt_file = os.path.join(self.prompt_dir, f"prompt_{dim}.txt")
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompts[dimension] = f.read()
                    logger.info(f"[OK] Loaded the prompt file for dimension {dimension}")
            else:
                logger.warning(f"[FAIL] Prompt file not found for dimension {dimension}: {prompt_file}")
                prompts[dimension] = ""
        return prompts

    def build_prompt(self, data_item: Dict[str, Any], dimension: str) -> str:
        """Build the trajectory-evaluation prompt for a specific dimension.

        Args:
            data_item: the data item containing the info needed for evaluation.
            dimension: the evaluation dimension name.

        Returns:
            The built prompt string; returns an empty string on failure.
        """
        try:
            query = data_item.get('user_question', '')
            history = data_item.get('history', '')
            trajectory = data_item.get('readable_trajectory', '')
            time_str = data_item.get('current_time', '')

            prompt_template = self.prompts.get(dimension, "")
            format_params = {
                'history': history,
                'query': query,
                'trajectory': trajectory,
                'time_str': time_str,
            }

            formatted_prompt = prompt_template.format(**format_params)
            return formatted_prompt.strip()
        except Exception as e:
            logger.error(f"Error while building the prompt: {e}")
            return ""

    def evaluate_dimension(self, config: Dict[str, Any], item: Dict[str, Any], dimension: str) -> float:
        """Evaluate a single trajectory dimension and return its score.

        Args:
            config: the config dict.
            item: the evaluation data item.
            dimension: the evaluation dimension name.

        Returns:
            The evaluation score (0-4); returns 0.0 on failure.
        """
        try:
            # Build the prompt
            formatted_prompt = self.build_prompt(item, dimension)
            if not formatted_prompt:
                logger.error(f"Dimension {dimension}: failed to build the prompt")
                return config.get('default_score', 0.0)

            # Build the messages
            messages = [
                {
                    'role': 'system',
                    # NOTE: this is the system prompt actually sent to the judge model.
                    'content': '你是全领域的专家，需要根据问题和提供的文档来判断答案是否优质。'
                },
                {
                    'role': 'user',
                    'content': formatted_prompt
                }
            ]

            # Call the API
            resp = ModelAPIClient.call_openai_format(messages)
            if resp is None:
                logger.warning(f"Dimension {dimension}: API call failed")
                return config.get('default_score', 0.0)

            try:
                content = resp.json()['choices'][0]['message']['content']
            except (KeyError, IndexError) as e:
                logger.error(f"Dimension {dimension}: failed to parse the response - {e}")
                return 0.0

            # Parse the result
            parsed_result = self._parse_result(content)
            if not parsed_result or 'result' not in parsed_result:
                logger.error(f"Dimension {dimension}: the parsed result is empty")
                return 0.0

            # Get the score for this dimension from the result dict
            score = parsed_result['result'].get(dimension, 0.0)

            # Check that the score is legal
            if score not in RewardConfig.VALID_SCORES:
                logger.warning(f"Dimension {dimension}: invalid score {score}")
                return 0.0
            return float(score)

        except Exception as e:
            logger.error(f"Failed to evaluate dimension {dimension}: {e}")
            logger.error(traceback.format_exc())
            return 0.0

    def _parse_result(self, result_text: str) -> Dict[str, Any]:
        """Parse the trajectory-quality score from the LLM's output.

        Args:
            result_text: the LLM's output text.

        Returns:
            A dict containing result, raw_result, and possibly an error field.
        """
        if result_text == "safety reason error":
            parsed_result = {
                'result': {},
                'raw_result': result_text,
                'error': "safety reason error",
            }
            return parsed_result
        try:
            import re
            parsed_result = {
                'result': {},
                'raw_result': result_text
            }

            # 1. Look for the final scoring dict (in the final-scoring section).
            # NOTE: the marker below must match what the judge model emits,
            # so it is intentionally kept in its original (Chinese) form.
            final_score_pattern = r'【最终打分】[：:]?\s*({[^}]*})'
            final_match = re.search(final_score_pattern, result_text)

            if final_match:
                dict_str = final_match.group(1)

                # Try parsing the dict
                try:
                    scores = json_repair.loads(dict_str)
                    parsed_result['result'] = scores

                except Exception as e:
                    logger.warning(f"Failed to parse the dict: {e}, trying the fallback method")
                    # Fallback parsing method: extract each dimension's score via regex
                    scores = self._extract_trajectory_scores_by_regex(dict_str)
                    parsed_result['result'] = scores
            else:
                # If no final-scoring section was found, try looking for any dict format
                dict_patterns = []
                # Generate a matching pattern for each dimension
                for dimension in self.dimensions:
                    pattern = r'\{{[^}}]*["\']' + re.escape(dimension) + r'["\'][^}}]*\}}'
                    dict_patterns.append(pattern)
                # Add a generic pattern
                dict_patterns.append(r'\{[^}]*:\s*\d+[^}]*\}')

                for pattern in dict_patterns:
                    dict_match = re.search(pattern, result_text)
                    if dict_match:
                        dict_str = dict_match.group(0)
                        try:
                            scores = json_repair.loads(dict_str)
                            parsed_result['result'] = scores
                            break
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.debug(f"Failed to parse the dict: {e}")
                            continue

                # If still nothing was found, use regex extraction
                if not parsed_result['result']:
                    parsed_result['result'] = self._extract_trajectory_scores_by_regex(result_text)

            return parsed_result

        except Exception as e:
            logger.error(f"Error while parsing the result: {e}")
            logger.error(f"Raw text: {result_text[:500]}...")  # only log the first 500 chars
            logger.error(traceback.format_exc())
            return {
                'result': {},
                'raw_result': result_text,
                'error': str(e)
            }

    def _extract_trajectory_scores_by_regex(self, text: str) -> Dict[str, float]:
        """Extract trajectory scores using regex (fallback method).

        Args:
            text: the text to parse.

        Returns:
            A dict mapping dimension name to score.
        """
        scores = {}

        # Use the configured evaluation dimensions
        dimensions = self.dimensions

        # Generic score-extraction patterns.
        # NOTE: these must match the judge model's output format, so they are
        # intentionally kept in their original (Chinese) form.
        score_patterns = [
            r'["\']([^"\':]+)["\']\s*:\s*(\d+)',  # "dimension name": score
            r'([^：:]+)[：:]\s*(\d+)分',  # dimension name：X points
            r'([^：:]+)[：:]\s*评分[：:]\s*(\d+)',  # dimension name：score：X
            r'(\d+)\s*[.、]\s*\[([^\]]+)\]\s*分数[：:]\s*(\d+)',  # 1. [logical coherence] score：3
        ]

        for pattern in score_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 2:
                    dimension = match[0].strip()
                    try:
                        score = int(match[1])
                        # Only keep the trajectory-quality evaluation dimensions
                        if dimension in dimensions:
                            scores[dimension] = score
                    except ValueError:
                        continue
                elif len(match) == 3:  # handle the numbered format
                    dimension = match[1].strip()
                    try:
                        score = int(match[2])
                        if dimension in dimensions:
                            scores[dimension] = score
                    except ValueError:
                        continue

        # Specially handle the scores in the problem-detail section
        problem_detail_pattern = r'(\d+)\. \[([^\]]+)\]\s*分数[：:]([^\s]+)'
        problem_matches = re.findall(problem_detail_pattern, text)
        for match in problem_matches:
            if len(match) == 3:
                dimension = match[1].strip()
                score_text = match[2].strip()
                # Extract the numeric score
                score_match = re.search(r'(\d+)', score_text)
                if score_match and dimension in dimensions:
                    try:
                        score = int(score_match.group(1))
                        scores[dimension] = score
                    except ValueError:
                        continue

        return scores


class ScoreCalculator:
    """Score calculator.

    Provides multiple score-normalization and aggregation strategies,
    converting raw scores (0-4) into normalized scores (0-1).
    """

    @staticmethod
    def normalize_scores(all_scores: Dict[str, float]) -> Dict[str, Any]:
        """Normalize the scores and compute the weighted average.

        Args:
            all_scores: the dict of raw per-dimension scores (0-4).

        Returns:
            A dict containing the normalized scores, the weighted average, and the min score.
        """
        # Normalize: convert 0-4 into 0-1
        normalized = {
            dim: score / RewardConfig.NORMALIZATION_FACTOR
            for dim, score in all_scores.items()
        }

        # Compute the min score (used for a conservative evaluation)
        rm_final_score_min = float(min(normalized.values())) if normalized else 0.0

        # Compute the weighted average (according to the dimension weights)
        rm_final_score = 0.0
        for dim, score in normalized.items():
            weight = RewardConfig.DIM_WEIGHT.get(dim, 0.0)
            rm_final_score += score * weight

        return {
            "all_scores": normalized,
            "rm_final_score": rm_final_score,
            "rm_final_score_min": rm_final_score_min
        }

    @staticmethod
    def normalize_scores_min(all_scores: Dict[str, float]) -> Dict[str, Any]:
        """Normalize the scores using the min value (conservative strategy).

        Args:
            all_scores: the dict of raw per-dimension scores (0-4).

        Returns:
            A dict containing the normalized scores and the min score.
        """
        # Normalize: convert 0-4 into 0-1
        normalized = {
            dim: score / RewardConfig.NORMALIZATION_FACTOR
            for dim, score in all_scores.items()
        }

        # Use the min score as the final score (weakest-link rule)
        rm_final_score_min = float(min(normalized.values())) if normalized else 0.0

        return {
            "all_scores": normalized,
            "rm_final_score": rm_final_score_min,
            "rm_final_score_min": rm_final_score_min
        }

    @staticmethod
    def normalize_scores_mean(all_scores: Dict[str, float]) -> Dict[str, Any]:
        """Normalize the scores using the arithmetic mean.

        Args:
            all_scores: the dict of raw per-dimension scores (0-4).

        Returns:
            A dict containing the normalized scores, the mean score, and the min score.
        """
        # Normalize: convert 0-4 into 0-1
        normalized = {
            dim: score / RewardConfig.NORMALIZATION_FACTOR
            for dim, score in all_scores.items()
        }

        # Compute the min score
        rm_final_score_min = float(min(normalized.values())) if normalized else 0.0

        # Compute the arithmetic mean
        rm_final_score = (
            float(sum(normalized.values()) / len(normalized))
            if normalized else 0.0
        )

        return {
            "all_scores": normalized,
            "rm_final_score": rm_final_score,
            "rm_final_score_min": rm_final_score_min
        }

    @staticmethod
    def calculate_turns_reward(
        rm_final_score: float,
        num_turns: int,
        max_turns: Optional[int]
    ) -> float:
        """Compute the search-turn reward (encourages reaching a perfect score in fewer turns).

        Args:
            rm_final_score: the RM final score.
            num_turns: the number of search turns actually used.
            max_turns: the max number of search turns allowed.

        Returns:
            The turn reward score (between 0 and 1).
        """
        # Only award a turn reward when a perfect score (1.0) is reached and max_turns is set
        if max_turns and rm_final_score == 1.0:
            return 1.0 - num_turns / max_turns
        return 0.0


class RewardOrchestrator:
    """Reward-computation orchestrator.

    Coordinates the whole reward-evaluation flow, including answer-quality
    evaluation, trajectory-quality evaluation, and score aggregation. Uses
    a thread pool to run the multi-dimensional evaluation concurrently, for efficiency.
    """

    def __init__(self, prompt_dir: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize the reward-computation orchestrator.

        Args:
            prompt_dir: the directory containing the prompt files (uses the default when None).
            model_name: the evaluation model name (uses the default when None).
        """
        self.prompt_dir = prompt_dir or RewardConfig.DEFAULT_PROMPT_DIR
        self.model_name = model_name or RewardConfig.DEFAULT_MODEL_NAME

        # Initialize the answer evaluator and the trajectory evaluator
        self.answer_evaluator = AnswerEvaluator(self.model_name, self.prompt_dir)
        self.trajectory_evaluator = TrajectoryEvaluator("gpt-5", self.prompt_dir)

        # Load the summary prompt (used for the two-stage evaluation)
        self.dsv31_summary_prompt = self._load_prompt("prompt_dsv31_summary.txt")

    def _load_prompt(self, filename: str) -> str:
        """Load a prompt template from a file.

        Args:
            filename: the prompt template file name.

        Returns:
            The prompt template content; returns an empty string on failure.
        """
        try:
            filepath = os.path.join(self.prompt_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load the prompt file {filename}: {e}")
            return ""

    def get_rm_scores(self, config: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, float]:
        """Concurrently obtain the RM scores for all answer-quality dimensions.

        Uses a thread pool to evaluate multiple dimensions concurrently, improving efficiency.

        Args:
            config: the config dict.
            item: the evaluation data item.

        Returns:
            The dict of raw per-dimension scores (0-4).

        Raises:
            ValueError: when config or item is empty, or item is missing required fields.
        """
        # Validate the arguments
        if not config:
            raise ValueError("config cannot be empty")
        if not item:
            raise ValueError("item cannot be empty")
        if 'user_question' not in item or 'answer' not in item:
            raise ValueError("item is missing required fields: user_question or answer")

        with ThreadPoolExecutor() as executor:
            # Submit the evaluation tasks for all dimensions
            future_to_dim = {
                executor.submit(
                    self.answer_evaluator.evaluate_dimension,
                    config, item, dim
                ): dim
                for dim in RewardConfig.DIMENSIONS
            }

            # Initialize the score dict
            all_scores = {dim: 0.0 for dim in RewardConfig.DIMENSIONS}

            # Collect the evaluation results
            for future in as_completed(future_to_dim):
                dim = future_to_dim[future]
                try:
                    score = future.result()
                    all_scores[dim] = score
                    logger.debug(f"{dim} evaluation complete: score={score}")
                except Exception as e:
                    logger.error(f"{dim} evaluation failed: {str(e)}")
                    logger.error(traceback.format_exc())
                    all_scores[dim] = 0.0

        return all_scores

    def get_trajectory_scores(
        self,
        config: Dict[str, Any],
        item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Concurrently obtain the scores for all trajectory-quality dimensions.

        Args:
            config: the config dict.
            item: the evaluation data item.

        Returns:
            A dict containing the normalized scores and the mean trajectory score.

        Raises:
            ValueError: when config or item is empty.
        """
        # Validate the arguments
        if not config:
            raise ValueError("config cannot be empty")
        if not item:
            raise ValueError("item cannot be empty")

        with ThreadPoolExecutor() as executor:
            # Submit the evaluation tasks for all trajectory dimensions
            future_to_dim = {
                executor.submit(
                    self.trajectory_evaluator.evaluate_dimension,
                    config, item, dim
                ): dim
                for dim in RewardConfig.TRAJECTORY_DIMENSIONS
            }

            # Initialize the score dict
            all_scores = {dim: 0.0 for dim in RewardConfig.TRAJECTORY_DIMENSIONS}

            # Collect the evaluation results
            for future in as_completed(future_to_dim):
                dim = future_to_dim[future]
                try:
                    score = future.result()
                    all_scores[dim] = score
                    logger.debug(f"{dim} evaluation complete: score={score}")
                except Exception as e:
                    logger.error(f"{dim} evaluation failed: {str(e)}")
                    logger.error(traceback.format_exc())
                    all_scores[dim] = 0.0

        # Normalize and compute the mean trajectory score
        normalized = {
            dim: score / RewardConfig.NORMALIZATION_FACTOR
            for dim, score in all_scores.items()
        }
        trajectory_score = (
            float(sum(normalized.values()) / len(normalized))
            if normalized else 0.0
        )

        return {
            "all_scores": normalized,
            "trajectory_score": trajectory_score
        }

    def parse_ground_truth_scores(self, ground_truth: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Parse the reference-answer RM scores from the ground truth.

        Args:
            ground_truth: the ground-truth data dict.

        Returns:
            The dict of per-dimension reference scores.
        """
        try:
            if not ground_truth:
                return {dim: 0.0 for dim in RewardConfig.DIMENSIONS}

            rm_scores = ground_truth.get("rm_scores")
            if rm_scores is None or len(rm_scores) == 0:
                return {dim: 0.0 for dim in RewardConfig.DIMENSIONS}

            return json.loads(rm_scores[0])
        except Exception as e:
            logger.error(f"Failed to parse ground_truth rm_scores: {str(e)}")
            logger.error(traceback.format_exc())
            return {dim: 0.0 for dim in RewardConfig.DIMENSIONS}


def my_reward_function_mean(
    config: Dict[str, Any],
    prompt_str: str,
    sequences_str: str,
    ground_truth: Optional[Dict[str, Any]],
    data_source: str,
    messages: Dict[str, Any],
    task_extra_info: Dict[str, Any]
) -> Dict[str, float]:
    """A reward function using the mean score (recommended for training).

    Evaluation strategy:
    - Uses the arithmetic mean of all dimensions as the main reward
    - Awards a turn reward to samples that reach a perfect score
    - Returns the min score for validation-set evaluation

    Args:
        config: the config dict, containing model and API settings.
        prompt_str: the input prompt string.
        sequences_str: the model-generated sequence string.
        ground_truth: the reference-answer data (optional).
        data_source: the data-source identifier.
        messages: the list of conversation messages.
        task_extra_info: extra task info.

    Returns:
        A dict containing train_final_score and val_final_score.
    """
    try:
        # Initialize the orchestrator
        orchestrator = RewardOrchestrator()

        # Build the evaluation data item
        item = EvalItemBuilder.build_from_sequences(sequences_str, prompt_str)
        item["sequences_str"] = sequences_str
        item["prompt_str"] = prompt_str
        item["sequences_str_len"] = len(sequences_str)
        item["prompt_str_len"] = len(prompt_str)

        # Get the raw RM scores (0-4)
        raw_scores = orchestrator.get_rm_scores(config, item)

        # Normalize and compute the mean score
        model_rm_scores = ScoreCalculator.normalize_scores_mean(raw_scores)

        all_scores = model_rm_scores["all_scores"]
        rm_final_score = model_rm_scores["rm_final_score"]
        rm_final_score_min = model_rm_scores["rm_final_score_min"]

        # Compute the search-turn reward (encourages reaching a high score in fewer turns)
        max_turns = config.get("max_assistant_turns")
        turns_reward = ScoreCalculator.calculate_turns_reward(
            rm_final_score, item.get("num_turns", 0), max_turns
        )

        # Train score = RM mean score + turn reward
        final_score = rm_final_score + turns_reward
        # Validation score = RM min score (conservative evaluation)
        val_final_score = rm_final_score_min

        logger.info(
            f"[REWARD_MEAN] all_scores: {all_scores}, "
            f"rm_final_score: {rm_final_score}, "
            f"turns_reward: {turns_reward}, final_score: {final_score}"
        )

    except Exception as e:
        logger.error(f"Reward function execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        final_score = 0.0
        val_final_score = 0.0

    return {
        "train_final_score": float(final_score),
        "val_final_score": float(val_final_score)
    }


def my_reward_function_min(
    config: Dict[str, Any],
    prompt_str: str,
    sequences_str: str,
    ground_truth: Optional[Dict[str, Any]],
    data_source: str,
    messages: Dict[str, Any],
    task_extra_info: Dict[str, Any]
) -> Dict[str, float]:
    """A reward function using the min score (conservative strategy).

    Evaluation strategy:
    - Uses the min score across all dimensions as the main reward (weakest-link rule)
    - Ensures the answer reaches a certain quality on every dimension
    - Suitable for scenarios with strict answer-quality requirements

    Args:
        config: the config dict.
        prompt_str: the input prompt.
        sequences_str: the generated sequence.
        ground_truth: the reference answer (optional).
        data_source: the data source.
        messages: the conversation messages.
        task_extra_info: extra info.

    Returns:
        A dict containing the train and validation scores.
    """
    try:
        orchestrator = RewardOrchestrator()
        item = EvalItemBuilder.build_from_sequences(sequences_str, prompt_str)
        item["sequences_str"] = sequences_str
        item["prompt_str"] = prompt_str

        # Get the RM scores and apply the min strategy
        raw_scores = orchestrator.get_rm_scores(config, item)
        model_rm_scores = ScoreCalculator.normalize_scores_min(raw_scores)

        rm_final_score = model_rm_scores["rm_final_score"]
        rm_final_score_min = model_rm_scores["rm_final_score_min"]

        # Compute the turn reward
        max_turns = config.get("max_assistant_turns")
        turns_reward = ScoreCalculator.calculate_turns_reward(
            rm_final_score, item.get("num_turns", 0), max_turns
        )

        final_score = rm_final_score + turns_reward
        val_final_score = rm_final_score_min

        logger.info(
            f"[REWARD_MIN] rm_final_score: {rm_final_score}, "
            f"turns_reward: {turns_reward}, "
            f"final_score: {final_score}"
        )

    except Exception as e:
        logger.error(f"Reward function execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        final_score = 0.0
        val_final_score = 0.0

    return {
        "train_final_score": float(final_score),
        "val_final_score": float(val_final_score)
    }


def my_reward_function_ref_v1(
    config: Dict[str, Any],
    prompt_str: str,
    sequences_str: str,
    ground_truth: Optional[Dict[str, Any]],
    data_source: str,
    messages: Dict[str, Any],
    task_extra_info: Dict[str, Any]
) -> Dict[str, float]:
    """Reference-answer reward function v1.

    Evaluation strategy:
    - Compares the model output against the reference answer
    - Awards an extra reward ratio when the model score exceeds the reference score
    - Combines the search-turn reward for a comprehensive evaluation
    - Suitable for scenarios requiring comparison against a reference answer

    Args:
        config: the config dict.
        prompt_str: the input prompt.
        sequences_str: the generated sequence.
        ground_truth: the reference-answer data (optional).
        data_source: the data source.
        messages: the conversation messages.
        task_extra_info: extra info.

    Returns:
        A dict containing the train and validation scores.
    """
    try:
        orchestrator = RewardOrchestrator()
        item = EvalItemBuilder.build_from_sequences(sequences_str, prompt_str)
        item["sequences_str"] = sequences_str
        item["prompt_str"] = prompt_str

        # Get the RM scores for the model's predicted answer
        raw_scores = orchestrator.get_rm_scores(config, item)
        model_rm_scores = ScoreCalculator.normalize_scores(raw_scores)

        # Get the RM scores for the reference answer
        ground_truth_rm_score = orchestrator.parse_ground_truth_scores(ground_truth)
        ref_rm_scores = ScoreCalculator.normalize_scores(ground_truth_rm_score)

        # Compute the reference-answer reward ratio
        if ref_rm_scores["rm_final_score"] != 0 and model_rm_scores["rm_final_score"] > ref_rm_scores["rm_final_score"]:
            ref_reward_ratio = model_rm_scores["rm_final_score"] / ref_rm_scores["rm_final_score"]
        else:
            ref_reward_ratio = 1.0

        # Compute the search-turn reward
        max_turns = config.get("max_assistant_turns")
        turns_reward = ScoreCalculator.calculate_turns_reward(
            model_rm_scores["rm_final_score"], item.get("num_turns", 0), max_turns
        )

        # Final score = model score * reference ratio + turn reward
        final_score = model_rm_scores["rm_final_score"] * ref_reward_ratio + turns_reward
        val_final_score = model_rm_scores["rm_final_score_min"]

        logger.info(
            f"[REWARD_REF_V1] model score: {model_rm_scores['rm_final_score']}, "
            f"reference score: {ref_rm_scores['rm_final_score']}, "
            f"reward ratio: {ref_reward_ratio}, "
            f"turn reward: {turns_reward}, "
            f"final score: {final_score}"
        )

    except Exception as e:
        logger.error(f"Reference-answer reward function v1 execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        final_score = 0.0
        val_final_score = 0.0

    return {
        "train_final_score": float(final_score),
        "val_final_score": float(val_final_score)
    }


def my_reward_function_ref_v1_2stage(
    config: Dict[str, Any],
    prompt_str: str,
    sequences_str: str,
    ground_truth: Optional[Dict[str, Any]],
    data_source: str,
    messages: Dict[str, Any],
    task_extra_info: Dict[str, Any]
) -> Dict[str, float]:
    """Reference-answer reward function v1 (two-stage version).

    Evaluation strategy:
    - Stage 1: generate the answer with a strong generation model
    - Stage 2: compare the generated answer against the reference answer
    - Awards an extra reward ratio when the model score exceeds the reference score
    - Combines the search-turn reward for a comprehensive evaluation
    - Suitable for scenarios requiring high-quality answer generation

    Args:
        config: the config dict.
        prompt_str: the input prompt.
        sequences_str: the generated sequence.
        ground_truth: the reference-answer data (optional).
        data_source: the data source.
        messages: the conversation messages.
        task_extra_info: extra info.

    Returns:
        A dict containing the train and validation scores.
    """
    try:
        orchestrator = RewardOrchestrator()
        item = EvalItemBuilder.build_from_sequences(sequences_str, prompt_str)
        item["sequences_str"] = sequences_str
        item["prompt_str"] = prompt_str

        # Stage 1: generate the answer with a strong generation model
        try:
            dsv31_prompt_formatted = orchestrator.dsv31_summary_prompt.format(
                TIME=item["current_time"],
                DOCS=item["reference_doc"],
                QUERY=item["user_question"]
            )

            ds_messages = _trans_rollout_messages_to_ds(messages)
            ds_messages = ds_messages[:-1]
            ds_messages.append({"role": "user", "content": dsv31_prompt_formatted})

            v31_answer, _ = ModelAPIClient.call_deepseek_v31(ds_messages)
            item["answer"] = v31_answer
            logger.info(f"[REWARD_REF_V1_2STAGE] generated answer: {v31_answer[:100]}...")
        except Exception as e:
            logger.error(f"Generation-model call failed: {str(e)}")

        # Stage 2: reference-answer comparison evaluation
        raw_scores = orchestrator.get_rm_scores(config, item)
        model_rm_scores = ScoreCalculator.normalize_scores(raw_scores)

        ground_truth_rm_score = orchestrator.parse_ground_truth_scores(ground_truth)
        ref_rm_scores = ScoreCalculator.normalize_scores(ground_truth_rm_score)

        # Compute the reference-answer reward ratio
        if ref_rm_scores["rm_final_score"] != 0 and model_rm_scores["rm_final_score"] > ref_rm_scores["rm_final_score"]:
            ref_reward_ratio = model_rm_scores["rm_final_score"] / ref_rm_scores["rm_final_score"]
        else:
            ref_reward_ratio = 1.0

        # Compute the search-turn reward
        max_turns = config.get("max_assistant_turns")
        turns_reward = ScoreCalculator.calculate_turns_reward(
            model_rm_scores["rm_final_score"], item.get("num_turns", 0), max_turns
        )

        # Final score = model score * reference ratio + turn reward
        final_score = model_rm_scores["rm_final_score"] * ref_reward_ratio + turns_reward
        val_final_score = model_rm_scores["rm_final_score_min"]

        logger.info(
            f"[REWARD_REF_V1_2STAGE] model score: {model_rm_scores['rm_final_score']}, "
            f"reference score: {ref_rm_scores['rm_final_score']}, "
            f"reward ratio: {ref_reward_ratio}, "
            f"turn reward: {turns_reward}, "
            f"final score: {final_score}"
        )

    except Exception as e:
        logger.error(f"Reference-answer reward function v1 (two-stage) execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        final_score = 0.0
        val_final_score = 0.0

    return {
        "train_final_score": float(final_score),
        "val_final_score": float(val_final_score)
    }


def my_reward_function_ref_v2(
    config: Dict[str, Any],
    prompt_str: str,
    sequences_str: str,
    ground_truth: Optional[Dict[str, Any]],
    data_source: str,
    messages: Dict[str, Any],
    task_extra_info: Dict[str, Any]
) -> Dict[str, float]:
    """Reference-answer reward function v2.

    Evaluation strategy:
    - When the model score exceeds the reference score, awards a turn-related reward
    - When the model score is below the reference score, applies a turn-related penalty
    - The reward/penalty is inversely proportional to the number of search turns,
      encouraging better results in fewer turns
    - Suitable for scenarios that need to balance quality and efficiency

    Args:
        config: the config dict.
        prompt_str: the input prompt.
        sequences_str: the generated sequence.
        ground_truth: the reference-answer data (optional).
        data_source: the data source.
        messages: the conversation messages.
        task_extra_info: extra info.

    Returns:
        A dict containing the train and validation scores.
    """
    try:
        orchestrator = RewardOrchestrator()
        item = EvalItemBuilder.build_from_sequences(sequences_str, prompt_str)
        item["sequences_str"] = sequences_str
        item["prompt_str"] = prompt_str

        # Get the RM scores for the model's predicted answer
        raw_scores = orchestrator.get_rm_scores(config, item)
        model_rm_scores = ScoreCalculator.normalize_scores(raw_scores)

        # Get the RM scores for the reference answer
        ground_truth_rm_score = orchestrator.parse_ground_truth_scores(ground_truth)
        ref_rm_scores = ScoreCalculator.normalize_scores(ground_truth_rm_score)

        # v2 reward computation logic
        max_turns = config.get("max_assistant_turns", 10)
        num_turns = item.get("num_turns", 1)

        # When the model score exceeds the reference: reward = (model - reference) / turns
        # When the model score is below the reference: reward = (model - reference) + (1 - turns) / max_turns
        if model_rm_scores["rm_final_score"] > ref_rm_scores["rm_final_score"]:
            ref_reward = (model_rm_scores["rm_final_score"] - ref_rm_scores["rm_final_score"]) / num_turns
        else:
            ref_reward = (model_rm_scores["rm_final_score"] - ref_rm_scores["rm_final_score"]) \
                        + (1 - num_turns) / max_turns

        final_score = ref_reward
        val_final_score = model_rm_scores["rm_final_score_min"]

        logger.info(
            f"[REWARD_REF_V2] model score: {model_rm_scores['rm_final_score']}, "
            f"reference score: {ref_rm_scores['rm_final_score']}, "
            f"search turns: {num_turns}, "
            f"max turns: {max_turns}, "
            f"final score: {final_score}"
        )

    except Exception as e:
        logger.error(f"Reference-answer reward function v2 execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        final_score = 0.0
        val_final_score = 0.0

    return {
        "train_final_score": float(final_score),
        "val_final_score": float(val_final_score)
    }


def my_reward_function_ref_v2_2stage(
    config: Dict[str, Any],
    prompt_str: str,
    sequences_str: str,
    ground_truth: Optional[Dict[str, Any]],
    data_source: str,
    messages: Dict[str, Any],
    task_extra_info: Dict[str, Any]
) -> Dict[str, float]:
    """Reference-answer reward function v2 (two-stage version).

    Evaluation strategy:
    - Stage 1: generate a high-quality answer with a strong generation model
    - Stage 2: compare against the reference answer using the v2 strategy
    - When the model score exceeds the reference score, awards a turn-related reward
    - When the model score is below the reference score, applies a turn-related penalty
    - Suitable for scenarios requiring high-quality answer generation plus reference comparison

    Args:
        config: the config dict.
        prompt_str: the input prompt.
        sequences_str: the generated sequence.
        ground_truth: the reference-answer data (optional).
        data_source: the data source.
        messages: the conversation messages.
        task_extra_info: extra info.

    Returns:
        A dict containing the train and validation scores.
    """
    try:
        orchestrator = RewardOrchestrator()
        item = EvalItemBuilder.build_from_sequences(sequences_str, prompt_str)
        item["sequences_str"] = sequences_str
        item["prompt_str"] = prompt_str

        # Stage 1: generate the answer with a strong generation model
        try:
            dsv31_prompt_formatted = orchestrator.dsv31_summary_prompt.format(
                TIME=item["current_time"],
                DOCS=item["reference_doc"],
                QUERY=item["user_question"]
            )

            ds_messages = _trans_rollout_messages_to_ds(messages)
            ds_messages = ds_messages[:-1]
            ds_messages.append({"role": "user", "content": dsv31_prompt_formatted})

            v31_answer, _ = ModelAPIClient.call_deepseek_v31(ds_messages)
            item["answer"] = v31_answer
            logger.info(f"[REWARD_REF_V2_2STAGE] generated answer: {v31_answer[:100]}...")
        except Exception as e:
            logger.error(f"Generation-model call failed: {str(e)}")

        # Stage 2: v2-strategy reference-answer comparison evaluation
        raw_scores = orchestrator.get_rm_scores(config, item)
        model_rm_scores = ScoreCalculator.normalize_scores(raw_scores)

        ground_truth_rm_score = orchestrator.parse_ground_truth_scores(ground_truth)
        ref_rm_scores = ScoreCalculator.normalize_scores(ground_truth_rm_score)

        max_turns = config.get("max_assistant_turns", 10)
        num_turns = item.get("num_turns", 1)

        # v2 reward computation logic
        if model_rm_scores["rm_final_score"] > ref_rm_scores["rm_final_score"]:
            ref_reward = (model_rm_scores["rm_final_score"] - ref_rm_scores["rm_final_score"]) / num_turns
        else:
            ref_reward = (model_rm_scores["rm_final_score"] - ref_rm_scores["rm_final_score"]) \
                        + (1 - num_turns) / max_turns

        final_score = ref_reward
        val_final_score = model_rm_scores["rm_final_score_min"]

        logger.info(
            f"[REWARD_REF_V2_2STAGE] model score: {model_rm_scores['rm_final_score']}, "
            f"reference score: {ref_rm_scores['rm_final_score']}, "
            f"search turns: {num_turns}, "
            f"max turns: {max_turns}, "
            f"final score: {final_score}"
        )

    except Exception as e:
        logger.error(f"Reference-answer reward function v2 (two-stage) execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        final_score = 0.0
        val_final_score = 0.0

    return {
        "train_final_score": float(final_score),
        "val_final_score": float(val_final_score)
    }


def my_reward_function_trajectory(
    config: Dict[str, Any],
    prompt_str: str,
    sequences_str: str,
    ground_truth: Optional[Dict[str, Any]],
    data_source: str,
    messages: Dict[str, Any],
    task_extra_info: Dict[str, Any]
) -> Dict[str, float]:
    """A reward function that includes trajectory-quality evaluation.

    Evaluation strategy:
    - Evaluates answer quality (8 dimensions)
    - Evaluates trajectory quality (2 dimensions)
    - Combines the answer score, the trajectory score, and the turn reward
    - Suitable for scenarios that need to optimize both the reasoning process and the answer quality

    Args:
        config: the config dict.
        prompt_str: the input prompt string.
        sequences_str: the model-generated sequence string.
        ground_truth: the reference-answer data (optional).
        data_source: the data-source identifier.
        messages: the list of conversation messages.
        task_extra_info: extra task info.

    Returns:
        A dict containing the train and validation scores.
    """
    try:
        orchestrator = RewardOrchestrator()
        item = EvalItemBuilder.build_from_sequences(sequences_str, prompt_str)
        item["sequences_str"] = sequences_str
        item["prompt_str"] = prompt_str

        # Get the answer-quality RM scores (8 dimensions)
        raw_scores = orchestrator.get_rm_scores(config, item)
        model_rm_scores = ScoreCalculator.normalize_scores(raw_scores)

        # Get the trajectory-quality scores (2 dimensions)
        trajectory_scores = orchestrator.get_trajectory_scores(config, item)

        rm_final_score = model_rm_scores["rm_final_score"]
        rm_final_score_min = model_rm_scores["rm_final_score_min"]
        trajectory_score = trajectory_scores["trajectory_score"]

        # Compute the search-turn reward (encourages reaching a high score in fewer turns)
        max_turns = config.get("max_assistant_turns", 1)
        turns_reward = ScoreCalculator.calculate_turns_reward(
            rm_final_score, item.get("num_turns", 0), max_turns
        )

        # Final score = answer-quality score + trajectory-quality score + turn reward
        final_score = rm_final_score + trajectory_score + turns_reward
        val_final_score = rm_final_score_min

        logger.info(
            f"[REWARD_TRAJECTORY] answer quality score: {rm_final_score}, "
            f"trajectory quality score: {trajectory_score}, "
            f"turn reward: {turns_reward}, "
            f"final score: {final_score}"
        )

    except Exception as e:
        logger.error(f"Trajectory-quality reward function execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        final_score = 0.0
        val_final_score = 0.0

    return {
        "train_final_score": float(final_score),
        "val_final_score": float(val_final_score)
    }


def _trans_rollout_messages_to_ds(messages: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert the rollout message format into the chat-completion API format.

    Converts internal message objects into the standard dict format,
    handling special fields such as tool_calls.

    Args:
        messages: a dict containing the message list {"messages": [msg1, msg2, ...]}.

    Returns:
        The list of messages in the chat-completion API format.
    """
    new_messages = []

    for msg in messages.get("messages", []):
        # Build the basic message structure
        new_msg = {
            "role": msg.role,
            "content": msg.content
        }

        # Handle tool calls (if present)
        if msg.tool_calls is not None:
            tool_calls = []
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                })
            if tool_calls:
                new_msg["tool_calls"] = tool_calls

        new_messages.append(new_msg)

    return new_messages
