#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generative reward-scoring utility for an OpenAI-compatible API

This module provides generative reward-scoring functionality backed by
any OpenAI-compatible chat completion API, supporting both streaming and
non-streaming response handling, for quality assessment and scoring of
model-generated sequences.

Main features:
- Text output extraction and formatting
- Streaming response handling
- Non-streaming response handling
- Asynchronous API querying with a retry mechanism
- Error handling and graceful degradation

Core characteristics:
- **Multi-format support**: supports JSON, plain text, and other response formats
- **Streaming processing**: processes streaming API responses in real time
- **Async querying**: supports concurrent API calls
- **Retry mechanism**: built-in exponential-backoff retry strategy
- **Error recovery**: thorough error handling and graceful degradation

Usage example:
    from api_generative import prime_query_openai_async, extract_output

    # Configure the API parameters
    config = {
        'url': 'http://api.example.com',
        'model': 'your-model',
        'wsid': 'your-wsid',
        'api_key': 'your-token',
        'stream': False,
        'temperature': 0.1,
        'max_retries': 3
    }

    # Query a sequence
    sequence = "the model-generated text sequence"
    response = prime_query_openai_async(config, sequence)

    # Extract the scoring result
    score = extract_output(response.text)
    print(f"score: {score}")

Version: 1.0.0
"""
import re
import asyncio
import json
import logging
import traceback
from typing import List, Any, Optional, Dict, Generator

import torch
import requests
from transformers import AutoTokenizer

class GenericOpenAIChatClient:
    """A generic OpenAI-compatible Chat Completions client (non-streaming).

    This open-source version replaces the original client that was bound
    to an internal service-discovery mechanism, connecting directly
    instead to any OpenAI-compatible HTTP endpoint (a self-hosted
    vLLM/SGLang OpenAI server, the official OpenAI API, or any other
    cloud vendor's compatible gateway all work).

    The interface stays compatible with callers: `chat_completion(messages=...)`
    returns the raw `requests.Response` object (callers access
    `.ok` / `.status_code` / `.json()`).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        auth_token: str = "",
        wsid: str = "",
        stream: bool = False,
        temperature: float = 0.1,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        output_seq_len: int = 64,
        max_input_seq_len: int = 10240,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        # auth_token may be either "Bearer xxx" or a bare token; both formats are supported
        self.auth_token = auth_token or ""
        self.stream = stream
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.output_seq_len = output_seq_len
        self.max_input_seq_len = max_input_seq_len
        self.timeout = timeout

    def chat_completion(self, messages: List[Dict[str, Any]]):
        auth_header = self.auth_token
        if auth_header and not auth_header.lower().startswith("bearer "):
            auth_header = f"Bearer {auth_header}"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.output_seq_len,
        }
        headers = {"Content-Type": "application/json"}
        if auth_header:
            headers["Authorization"] = auth_header

        # url may be either a full endpoint (including /chat/completions) or just the base_url
        url = self.base_url if self.base_url.endswith("/chat/completions") else f"{self.base_url}/chat/completions"
        return requests.post(url, headers=headers, json=payload, timeout=self.timeout)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_output(solution_text: str) -> Optional[str]:
    """
    Extract the formatted output from a solution text.

    This function uses a regular expression to match the formatting
    marker in the text (e.g. \\bold{}), extracting the content of the
    last match as the output result.

    Args:
        solution_text (str): the solution text containing a formatting marker
            - e.g.: "The answer is \\bold{42}" or "Final result: \\bold{correct}"

    Returns:
        Optional[str]: the extracted output content, or None if there is no match
            - e.g.: "42", "correct"

    Raises:
        TypeError: if the input argument is not a string

    Example:
        >>> text = "Based on the calculation, the answer is \\bold{42}"
        >>> result = extract_output(text)
        >>> print(result)  # "42"

        >>> text = "text with no formatting marker"
        >>> result = extract_output(text)
        >>> print(result)  # None
    """
    # Validate the argument
    if not isinstance(solution_text, str):
        raise TypeError(f"solution_text must be a string, got: {type(solution_text)}")

    if not solution_text.strip():
        print("[WARNING] input text is empty, cannot extract output")
        return None

    # Regex pattern: matches content in \\bold{} format
    boxed_pattern = r"\\bold{(.*)}"

    try:
        # Find all matches
        matches = re.findall(boxed_pattern, solution_text)

        if matches:
            # Extract the content of the last match
            last_match = matches[-1].strip()
            print(f"[DEBUG] successfully extracted output: '{last_match}'")
            return last_match
        else:
            # No match found
            print(f"[DEBUG] no formatting marker found, text length: {len(solution_text)}")
            return None

    except re.error as e:
        print(f"[ERROR] regex matching failed: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] unknown error while extracting output: {e}")
        return None


def process_stream_response(response: requests.Response) -> Generator[str, None, None]:
    """
    Process a streaming API response, extracting content line by line.

    This function processes server-pushed streaming response data,
    extracting and yielding text content in real time. Supports SSE
    (Server-Sent Events)-format data streams.

    Args:
        response (requests.Response): the HTTP response object
            - must be a streaming response, containing an iter_lines() method
            - the response format should be SSE format (lines starting with "data:")

    Yields:
        str: a chunk of text content extracted from the streaming response
            - each chunk is a contiguous block of text
            - may contain partial words or complete sentences

    Raises:
        ValueError: if the response object is invalid or not a streaming response
        UnicodeDecodeError: if the response encoding is not UTF-8

    Example:
        >>> response = requests.post(url, stream=True)
        >>> for content_chunk in process_stream_response(response):
        ...     print(f"received content: {content_chunk}")
    """
    # Validate the argument
    if not response or not hasattr(response, "iter_lines"):
        raise ValueError("Invalid response object, must support streaming")

    if not response.ok:
        raise ValueError(f"Abnormal HTTP status code: {response.status_code}")

    line_count = 0
    content_count = 0

    print("[INFO] starting to process the streaming response...")

    for line in response.iter_lines():
        line_count += 1

        # Skip empty lines
        if not line:
            continue

        try:
            # Decode the UTF-8-encoded byte stream
            line_str = line.decode("utf-8")

            # Check whether this is an SSE-format data line
            if line_str.startswith("data:"):
                try:
                    # Parse the JSON data (stripping the 'data:' prefix)
                    json_data = json.loads(line_str[5:])

                    # Safely extract the content
                    choices = json_data.get("choices", [{}])
                    if choices:
                        first_choice = choices[0]
                        delta = first_choice.get("delta", {})
                        content = delta.get("content", "").strip()

                        # Only return non-empty content
                        if content:
                            content_count += 1
                            print(
                                f"[DEBUG] extracted content chunk #{content_count}: {content[:50]}..."
                            )
                            yield content

                except json.JSONDecodeError as e:
                    print(f"[WARNING] JSON parse failed (line {line_count}): {e}")
                    print(f"[DEBUG] raw data: {line_str[:100]}")
                    continue

        except UnicodeDecodeError as e:
            print(f"[ERROR] UTF-8 decode failed (line {line_count}): {e}")
            continue
        except Exception as e:
            print(f"[ERROR] error while processing stream data (line {line_count}): {e}")
            continue

    print(
        f"[INFO] finished processing the streaming response, lines processed: {line_count}, content chunks extracted: {content_count}"
    )


def process_non_stream_response(response: requests.Response) -> str:
    """
    Process a non-streaming API response, extracting the full content.

    This function processes a standard JSON-format response, extracting
    the full text content generated by the model. Supports an
    OpenAI-compatible API response format.

    Args:
        response (requests.Response): the HTTP response object
            - must be a successful HTTP response (status code 200)
            - the response body should be valid JSON

    Returns:
        str: the text content extracted from the response
            - leading/trailing whitespace already stripped
            - returns an empty string if the content is empty

    Raises:
        ValueError: if the response format doesn't match expectations or is missing required fields
        json.JSONDecodeError: if the response body is not valid JSON
        requests.RequestException: if the HTTP request failed

    Example:
        >>> response = requests.post(url, json=data)
        >>> content = process_non_stream_response(response)
        >>> print(f"model-generated content: {content}")
    """
    # Validate the argument
    if not response or not response.ok:
        raise ValueError(
            f"Abnormal HTTP status code: {response.status_code if response else 'no response'}"
        )

    try:
        # Parse the JSON response
        response_data = response.json()

        # Validate the response structure
        if "choices" not in response_data:
            raise ValueError("Response is missing the 'choices' field")

        choices = response_data["choices"]
        if not choices or not isinstance(choices, list):
            raise ValueError("'choices' is empty or not a list")

        first_choice = choices[0]
        if "message" not in first_choice:
            raise ValueError("choice is missing the 'message' field")

        message = first_choice["message"]
        if "content" not in message:
            raise ValueError("message is missing the 'content' field")

        # Extract and clean the content
        content = message["content"].strip()

        print(f"[INFO] successfully extracted content, length: {len(content)} chars")
        if len(content) > 100:
            print(f"[DEBUG] content preview: {content[:100]}...")
        else:
            print(f"[DEBUG] full content: {content}")

        return content

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON response: {e}"
        print(f"[ERROR] {error_msg}")
        print(f"[DEBUG] response text: {response.text[:200]}")
        raise ValueError(error_msg)

    except KeyError as e:
        error_msg = f"Abnormal response format, missing required field: {e}"
        print(f"[ERROR] {error_msg}")
        print(
            f"[DEBUG] response structure: {list(response_data.keys()) if 'response_data' in locals() else 'not parsed'}"
        )
        raise ValueError(error_msg)

    except Exception as e:
        error_msg = f"Unknown error while processing the non-streaming response: {e}"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)


def prime_query_openai_async(
    config: Dict[str, Any], sequence_str: str
) -> Optional[requests.Response]:
    """
    Query an OpenAI-compatible API asynchronously for scoring.

    This function is the core API-calling interface, supporting a retry
    mechanism and error handling. Builds the full conversation message
    based on the config parameters, calling the API to obtain the scoring result.

    Args:
        config (Dict[str, Any]): the API config dict
            - url (str): the API service address
            - model (str): the model name
            - wsid (str): the service-group identifier
            - api_key (str): the auth token
            - stream (bool): whether to use a streaming response
            - temperature (float): the temperature parameter
            - top_p (float): the nucleus-sampling parameter
            - top_k (int): the Top-K sampling parameter
            - repetition_penalty (float): the repetition-penalty coefficient
            - output_seq_len (int): the output sequence length
            - max_input_seq_len (int): the max input sequence length
            - max_retries (int): the max number of retries
            - system_prompt (str): the system prompt
            - scoring_prompt (str): the scoring prompt
            - tokenizer (str, optional): the tokenizer path
            - apply_chat_template (bool, optional): whether to apply the chat template

        sequence_str (str): the sequence text to be scored
            - e.g.: "the model-generated text content"

    Returns:
        Optional[requests.Response]: the API response object
            - returns the Response object on success
            - returns None on failure

    Raises:
        ValueError: config parameter error or missing
        ConnectionError: network connection failure
        TimeoutError: request timeout

    Example:
        >>> config = {
        ...     'url': 'http://api.example.com',
        ...     'model': 'your-model',
        ...     'max_retries': 3,
        ...     'system_prompt': 'You are a scoring assistant',
        ...     'scoring_prompt': 'Please score the following text:'
        ... }
        >>> sequence = "a piece of text to be scored"
        >>> response = prime_query_openai_async(config, sequence)
        >>> if response:
        ...     score = extract_output(response.text)
        ...     print(f"score: {score}")
    """
    # 1. Validate parameters
    required_config_keys = [
        "url",
        "model",
        "wsid",
        "api_key",
        "max_retries",
        "system_prompt",
        "scoring_prompt",
    ]

    for key in required_config_keys:
        if key not in config:
            raise ValueError(f"Missing config parameter: {key}")

    if not sequence_str or not sequence_str.strip():
        raise ValueError("sequence_str cannot be empty")

    print(f"[INFO] starting the API query, sequence length: {len(sequence_str)} chars")

    # 2. Initialize the API client
    try:
        api_client = GenericOpenAIChatClient(
            base_url=config["url"],
            model=config["model"],
            wsid=config["wsid"],
            auth_token=config["api_key"],
            stream=config.get("stream", False),
            temperature=config.get("temperature", 0.1),
            top_p=config.get("top_p", 0.8),
            top_k=config.get("top_k", 20),
            repetition_penalty=config.get("repetition_penalty", 1.0),
            output_seq_len=config.get("output_seq_len", 64),
            max_input_seq_len=config.get("max_input_seq_len", 10240),
        )
    except Exception as e:
        print(f"[ERROR] API client initialization failed: {e}")
        return None

    # 3. Retry mechanism
    max_retries = config["max_retries"]

    for attempt in range(max_retries):
        print(f"[INFO] attempting API call {attempt + 1}/{max_retries}...")

        try:
            # 4. Build the message structure
            messages = [
                {"role": "system", "content": config["system_prompt"]},
                {
                    "role": "user",
                    "content": f"{config['scoring_prompt']}\n{sequence_str}",
                },
            ]

            print(f"[DEBUG] system prompt length: {len(config['system_prompt'])}")
            print(f"[DEBUG] user prompt length: {len(config['scoring_prompt'])}")
            print(f"[DEBUG] sequence text length: {len(sequence_str)}")

            # 5. Apply the chat template (if configured)
            if config.get("tokenizer") and config.get("apply_chat_template"):
                print("[INFO] applying the chat template...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
                    messages = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    print("[INFO] chat template applied successfully")
                except Exception as e:
                    print(f"[WARNING] failed to apply the chat template: {e}")
                    print("[INFO] using the raw message format")

            # 6. Perform the API call
            print("[INFO] starting the API call...")
            response = api_client.chat_completion(messages=messages)

            # 7. Validate the response
            if not response:
                raise ValueError("The API response is empty")

            if not response.ok:
                raise ValueError(f"Abnormal HTTP status code: {response.status_code}")

            print(f"[SUCCESS] API call succeeded, status code: {response.status_code}")
            return response

        except Exception as e:
            print(f"[ERROR] attempt {attempt + 1} failed: {str(e)}")
            traceback.print_exc()

            # The last attempt failed
            if attempt == max_retries - 1:
                print(f"[ERROR] all {max_retries} attempts failed")
                return None

            # Exponential-backoff wait
            wait_time = 2**attempt
            print(f"[INFO] waiting {wait_time} seconds before retrying...")
            import time

            time.sleep(wait_time)

    return None


async def query_openai_async(
    client: GenericOpenAIChatClient,
    sequence_str: str,
    config: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    index: int,
) -> tuple[int, float]:
    """
    Query an OpenAI-compatible API asynchronously to obtain a scoring result.

    This function is the async version of the core API-calling interface,
    using a semaphore to control concurrency, with a retry mechanism and
    graceful error degradation.

    Args:
        client (GenericOpenAIChatClient): the already-initialized API client
        sequence_str (str): the sequence text to be scored
        config (Dict[str, Any]): the API config parameters
        semaphore (asyncio.Semaphore): the concurrency-control semaphore
        index (int): the task index, used for result ordering

    Returns:
        tuple[int, float]: a tuple containing the index and the score
            - index: the original task index
            - score: the scoring result (0.0-1.0)

    Raises:
        asyncio.CancelledError: the task was cancelled
        ValueError: config parameter error

    Note:
        - Uses a semaphore to control concurrency, avoiding API rate limiting
        - Has a built-in retry mechanism, returning a default score on failure
        - Supports chat-template application and result extraction
    """
    max_retries = config["max_retries"]
    scoring_prompt = config["scoring_prompt"]

    async with semaphore:
        for attempt in range(max_retries):
            print(
                f"[ASYNC] starting the async query, index: {index}, attempt: {attempt + 1}/{max_retries}"
            )

            try:
                # Build the message structure
                messages = [
                    {
                        "role": "system",
                        "content": "You are a professional evaluation assistant. Please evaluate the content according to the given evaluation criteria, and strictly output the evaluation result in the required JSON format.",
                    },
                    {"role": "user", "content": f"{scoring_prompt}\n{sequence_str}"},
                ]

                # Apply the chat template (if configured)
                if config.get("tokenizer") and config.get("apply_chat_template"):
                    print(f"[ASYNC] applying the chat template, index: {index}")
                    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
                    messages = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                # Async API call
                print(f"[ASYNC] starting the API call, index: {index}")
                response = await asyncio.to_thread(
                    client.chat_completion, messages=messages
                )

                # Validate the response
                if not response or not response.ok:
                    raise ValueError(
                        f"Abnormal API response, status code: {response.status_code if response else 'no response'}"
                    )

                # Extract the content
                content = response.json()["choices"][0]["message"]["content"]
                print(f"[ASYNC] API call succeeded, index: {index}, content length: {len(content)}")

                # Extract the scoring result
                extracted = extract_output(content)
                print(f"[ASYNC] scoring extraction result, index: {index}, result: {extracted}")

                if extracted is not None:
                    score = float(extracted)
                    # Validate the score range
                    if 0.0 <= score <= 1.0:
                        print(f"[SUCCESS] async query complete, index: {index}, score: {score}")
                        return index, score
                    else:
                        print(f"[WARNING] score out of range, index: {index}, score: {score}")
                        raise ValueError("Score out of the valid range (0.0-1.0)")
                else:
                    raise ValueError("Could not extract a valid score from the response")

            except Exception as e:
                print(
                    f"[ERROR] async query failed, index: {index}, attempt: {attempt + 1}, error: {str(e)}"
                )

                # The last attempt failed, return the default score
                if attempt == max_retries - 1:
                    default_score = config.get("default_score", 0.5)
                    print(
                        f"[WARNING] all attempts failed, index: {index}, using the default score: {default_score}"
                    )
                    return index, default_score

                # Exponential-backoff wait
                wait_time = 2**attempt
                print(f"[INFO] waiting {wait_time} seconds before retrying, index: {index}")
                await asyncio.sleep(wait_time)

        # Should never reach here in theory
        default_score = config.get("default_score", 0.5)
        return index, default_score


async def process_data_async(
    data_source: List[str],
    solution_str: List[str],
    ground_truth: List[str],
    extra_info: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> torch.Tensor:
    """
    Asynchronously batch-process data and obtain scores.

    This function supports asynchronously batch-processing multiple data
    samples, using concurrency control to optimize performance. Each
    sample calls the API to obtain a score, ultimately returning a score tensor.

    Args:
        data_source (List[str]): the list of data-source identifiers
        solution_str (List[str]): the list of solutions to be scored
        ground_truth (List[str]): the list of reference answers
        extra_info (List[Dict[str, Any]]): the list of extra info
        config (Dict[str, Any]): the API config parameters

    Returns:
        torch.Tensor: the score-result tensor
            - shape: (len(solution_str),)
            - dtype: torch.float32
            - value range: 0.0-1.0

    Raises:
        ValueError: input parameter lengths don't match
        RuntimeError: async task execution failed

    Example:
        >>> solutions = ["answer1", "answer2", "answer3"]
        >>> references = ["reference1", "reference2", "reference3"]
        >>> scores = await process_data_async(["source1", "source2", "source3"], solutions, references, [{}]*3, config)
        >>> print(f"scores: {scores}")
    """
    # Validate parameters
    if len(solution_str) != len(ground_truth):
        raise ValueError("solution_str and ground_truth must have the same length")

    if extra_info and len(extra_info) != len(solution_str):
        raise ValueError("extra_info must have the same length as solution_str")

    print(f"[INFO] starting the async batch processing, sample count: {len(solution_str)}")

    # Initialize the score tensor
    reward_tensor = torch.zeros(len(solution_str), dtype=torch.float32)

    # Initialize the API client
    try:
        api_client = GenericOpenAIChatClient(
            base_url=config["url"],
            model=config["model"],
            wsid=config["wsid"],
            auth_token=config["api_key"],
            stream=config.get("stream", False),
            temperature=config.get("temperature", 0.1),
            top_p=config.get("top_p", 0.8),
            top_k=config.get("top_k", 20),
            repetition_penalty=config.get("repetition_penalty", 1.0),
            output_seq_len=config.get("output_seq_len", 64),
            max_input_seq_len=config.get("max_input_seq_len", 10240),
        )
    except Exception as e:
        raise RuntimeError(f"API client initialization failed: {e}")

    # Create a semaphore to control concurrency
    max_concurrency = config.get("max_concurrency", 5)
    semaphore = asyncio.Semaphore(max_concurrency)
    print(f"[INFO] concurrency control: max concurrency {max_concurrency}")

    # Create the list of async tasks
    tasks = []
    for i in range(len(solution_str)):
        prompt = solution_str[i]
        response = ground_truth[i]

        # Build the query sequence
        if response is None:
            sequence_str = prompt
        else:
            sequence_str = f"{prompt}\nReference:\n{response}"

        # Create the async task
        task = asyncio.create_task(
            query_openai_async(api_client, sequence_str, config, semaphore, i)
        )
        tasks.append(task)

        print(f"[DEBUG] created async task, index: {i}, sequence length: {len(sequence_str)}")

    # Wait for all tasks to complete
    print("[INFO] waiting for all async tasks to complete...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process the results
    success_count = 0
    for result in results:
        if isinstance(result, Exception):
            print(f"[ERROR] async task execution failed: {result}")
            continue

        if isinstance(result, tuple) and len(result) == 2:
            index, score = result
            reward_tensor[index] = score
            success_count += 1
            print(f"[DEBUG] processed result, index: {index}, score: {score}")

    print(f"[INFO] async batch processing complete, success: {success_count}/{len(solution_str)}")
    return reward_tensor
