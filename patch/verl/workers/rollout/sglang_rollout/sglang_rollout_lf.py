# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import time
import traceback
from copy import deepcopy
from json import JSONDecodeError
from typing import Any, Optional
from uuid import uuid4
import atexit
from datetime import datetime
import json

import numpy as np
import sglang.srt.entrypoints.engine
import torch
import torch.distributed as dist
from sglang.srt.managers.tokenizer_manager import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    get_ip,
    get_open_port,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from verl import DataProto
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionParsedSchema, OpenAIFunctionToolCall
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.net_utils import is_ipv6
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.config import RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
)
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser

try:
    from sglang.srt.entrypoints.openai.protocol import Tool
except ImportError:
    from sglang.srt.openai_api.protocol import Tool


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ============================================================
# Degenerate-output detection (repetition / garbage / abnormal repeats)
# ============================================================

def detect_degenerate_output(text: str, min_length: int = 500) -> tuple:
    """Detect whether the model output is degenerate (repetition, garbage, abnormal repeats).

    Detection rules:
    1. High n-gram repetition: split the text into n-grams; if the repetition ratio is too
       high, it is judged to be repetition
    2. High single-character repetition: if one character accounts for too large a share of
       the total length, it is judged to be garbage
    3. Short-sentence looping repetition: split the text into sentences; if the ratio of
       repeated sentences is too high, it is judged to be repetition

    Args:
        text: the model-generated text
        min_length: outputs shorter than this are not checked (short outputs rarely degenerate)

    Returns:
        (is_degenerate: bool, reason: str)
    """
    if not text or len(text) < min_length:
        return False, ""

    # ---- Check 1: an excessive share of a single character (garbage / repeated char) ----
    from collections import Counter
    char_counts = Counter(text)
    most_common_char, most_common_count = char_counts.most_common(1)[0]
    char_ratio = most_common_count / len(text)
    # Exclude spaces and newlines (they may naturally account for a large share)
    if most_common_char not in (' ', '\n', '\t') and char_ratio > 0.4:
        return True, f"single_char_repeat: '{most_common_char}' ratio={char_ratio:.2f}"

    # ---- Check 2: n-gram repetition ratio (detects phrase-level repetition) ----
    # Uses 10-grams; if repeated 10-grams exceed 50% of all 10-grams, it is judged to be
    # repetition
    ngram_size = 10
    words = text.split()
    if len(words) >= ngram_size * 2:
        ngrams = [tuple(words[i:i+ngram_size]) for i in range(len(words) - ngram_size + 1)]
        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        repeated_ngrams = sum(c - 1 for c in ngram_counts.values() if c > 1)
        repeat_ratio = repeated_ngrams / total_ngrams if total_ngrams > 0 else 0
        if repeat_ratio > 0.5:
            # Find the most-repeated n-gram for the log
            top_ngram, top_count = ngram_counts.most_common(1)[0]
            return True, f"ngram_repeat: ratio={repeat_ratio:.2f}, top='{' '.join(top_ngram[:5])}...' x{top_count}"

    # ---- Check 3: sentence-level looping repetition ----
    # Split into sentences by punctuation or newlines; if repeated sentences exceed 60%
    import re
    sentences = [s.strip() for s in re.split(r'[。！？\n.!?]', text) if len(s.strip()) > 5]
    if len(sentences) >= 4:
        sentence_counts = Counter(sentences)
        total_sentences = len(sentences)
        unique_sentences = len(sentence_counts)
        repeat_sentence_ratio = 1 - (unique_sentences / total_sentences)
        if repeat_sentence_ratio > 0.6:
            top_sentence, top_count = sentence_counts.most_common(1)[0]
            return True, (
                f"sentence_repeat: ratio={repeat_sentence_ratio:.2f}, "
                f"unique={unique_sentences}/{total_sentences}, "
                f"top='{top_sentence[:50]}...' x{top_count}"
            )

    return False, ""


# logging tool for sglang multi-turn rollout
class SGLangLogManager:
    def __init__(self):
        self.file_handles = {}
        atexit.register(self.close_all)

    def get_handle(self, log_path):
        if log_path not in self.file_handles:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.file_handles[log_path] = open(log_path, 'a', buffering=1)
        return self.file_handles[log_path]

    def log(self, log_path, event, duration=None, extra=None, workid=None, step=None,**extra_keys):
        handle = self.get_handle(log_path)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
        }
        if duration is not None:
            log_entry["duration_sec"] = duration
        if extra is not None:
            log_entry["extra"] = extra
        if workid is not None:
            log_entry["workid"] = workid
        if step is not None:
            log_entry["step"] = step
        if extra_keys is not None:
            for key in extra_keys:
                log_entry[key] = extra_keys[key]
        ordered_keys = ["timestamp", "event", "duration_sec"] + [k for k in log_entry.keys() if k not in ("timestamp", "event", "duration_sec")]
        ordered_entry = {k: log_entry[k] for k in ordered_keys if k in log_entry}
        try:
            handle.write(json.dumps(ordered_entry) + '\n')
            handle.flush()
        except OSError as e:
            # On IO errors such as an exhausted disk quota, degrade to stderr output and
            # don't crash the rollout
            import sys
            print(f"[SGLangLogManager] failed to write the log ({e}), event={event}", file=sys.stderr)

    def close_all(self):
        for handle in self.file_handles.values():
            handle.close()


# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class
class AsyncEngine(sglang.srt.entrypoints.engine.Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # default to use dummy load format, which need to reload weights in first time
        self._need_reload = True

    async def release_memory_occupation(self, tags: Optional[list[str]] = None):
        """Release GPU occupation temporarily."""
        if tags is None:
            obj = ReleaseMemoryOccupationReqInput()
        else:
            obj = ReleaseMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def resume_memory_occupation(self, tags: Optional[list[str]] = None):
        """Resume GPU occupation."""
        # because __init__ is a sync method, it can not call the async release_memory_occupation
        # have to move release_memory_occupation from __init__ to here
        # For multi-stage awake, we run release weight and kv_cache when we resume weights for the first time.
        if self._need_reload:
            await self.release_memory_occupation()
            self._need_reload = False

        if tags is None:
            obj = ResumeMemoryOccupationReqInput()
        else:
            obj = ResumeMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.resume_memory_occupation(obj, None)

    async def update_weights_from_tensor(self, update_weights_request: UpdateWeightsFromTensorReqInput):
        return await self.tokenizer_manager.update_weights_from_tensor(update_weights_request, None)

    async def flush_cache(self):
        return await self.tokenizer_manager.flush_cache()

    async def abort_request(self, rid: str = "", abort_all: bool = False):
        """Abort a specific request or all requests.

        Args:
            rid: The request ID to abort. If empty and abort_all is False, no action is taken.
            abort_all: If True, abort all running requests regardless of rid.
        """
        return self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)


# NOTE(sgm): add for verl. We can optimize it by making
#  the dataloader yield List[int] without padding.
def _pre_process_inputs(
    pad_token_id,
    prompt_token_ids: torch.Tensor,
) -> torch.Tensor:
    # remove the left padding in the prompt token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    return prompt_token_ids[non_pad_index:]


def _extract_logprob_from_output(output):
    """
    extract log_prob from single sglang inference output
    """

    def _map_each_response(resp):
        input_token_logprobs = resp["meta_info"]["input_token_logprobs"]
        log_probs, output_token_ids = zip(
            *[(log_prob, token_ids) for log_prob, token_ids, _ in input_token_logprobs[1:]], strict=False
        )
        return torch.tensor(output_token_ids), torch.tensor(log_probs)

    output_token_ids, log_probs = _map_each_response(output)
    return output_token_ids, log_probs


# NOTE(linjunrong): adhoc
def _post_process_outputs(processing_class, output):
    try:
        # This is when processing_class is a processor
        tokenizer = processing_class.tokenizer
    except AttributeError:
        try:
            # This is when processing_class is a tokenizer
            tokenizer = processing_class
        except AttributeError as e:
            raise ValueError(f"Cannot get tokenizer from processing_class {processing_class}") from e

    def _map_each_response(resp):
        output_token_logprobs = resp["meta_info"]["output_token_logprobs"]
        log_probs, output_token_ids = zip(
            *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True
        )
        return torch.tensor(output_token_ids), torch.tensor(log_probs)

    out_map = map(lambda x: _map_each_response(x), output)
    batched_output_token_ids = []
    batched_logprobs = []
    for output_token_ids, log_probs in out_map:
        batched_output_token_ids.append(output_token_ids)
        batched_logprobs.append(log_probs)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batched_output_token_ids = pad_sequence(batched_output_token_ids, batch_first=True, padding_value=pad_token_id)
    if len(batched_logprobs) > 0:
        batched_logprobs = pad_sequence(batched_logprobs, batch_first=True, padding_value=pad_token_id)
    return batched_output_token_ids, batched_logprobs


def get_tool_call_parser_type(
    processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
) -> str:
    items = FunctionCallParser.ToolCallParserEnum.items()
    for parser_type, parser_cls in items:
        parser = parser_cls()
        try:
            # This is when processing_class is a tokenizer
            tokenizer_vocab = processing_class.get_vocab()
        except AttributeError:
            try:
                # This is when processing_class is a processor
                tokenizer_vocab = processing_class.tokenizer.get_vocab()
            except AttributeError as e:
                raise ValueError(f"Cannot get vocab from processing_class {processing_class}") from e

        if parser.bot_token.strip() in tokenizer_vocab and (
            parser.eot_token == "" or parser.eot_token.strip() in tokenizer_vocab
        ):
            return parser_type
    else:
        raise ValueError(f"No tool call parser found for processing_class {processing_class}")


class SGLangRollout(BaseRollout):
    def __init__(
        self,
        actor_module: str,
        config: RolloutConfig,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        """Synchronized SGLang rollout engine.

        Args:
            actor_module: Huggingface model name or path to the model. The
                model should be supported by SGLang.
            config: A DictConfig object containing SGLang-specific operational
                parameters and rollout settings.
                Refer to https://docs.sglang.ai/backend/server_arguments.html
            processing_class: The tokenizer or processor instance compatible with the actor_module.
            model_hf_config: The Hugging Face model's configuration (e.g.,
                `transformers.PretrainedConfig`). It provides architectural
                details and hyperparameters like `max_position_embeddings`,
                used by SGLang for correct model initialization. This is
                the model's inherent design, not SGLang's runtime behavior.
            port: Optional port for multi-node initialization when nnodes > 1.
            trust_remote_code: Whether or not to allow for custom models
                defined on the Hub in their own modeling files.
            device_mesh: Optional `DeviceMesh` object for distributed setup.
            **kwargs: Additional keyword arguments, primarily `train_tp` for
                Megatron Backend integration to initialize hybrid engine
                process groups.
        """
        super().__init__()
        # self.step = 0
        self.config = config
        self._device_mesh_cpu = device_mesh
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        # The log_manager that was added
        self.log_manager = SGLangLogManager()
        # The log_dir that was added; if your experiment doesn't set EXPERIMENT_NAME,
        # multiturn_log_dir is used by default
        logdir = self.config.get("engine_kwargs", {}).get("logdir", "logs")
        self.log_dir = f"{logdir}/multiturn_log_dir"
        self._rank = None  # wsill be set in _init_distributed_env

        (
            self._tool_schemas,
            self._tool_map,
            self._tool_call_parser_type,
            self._sgl_tools,
            self._function_call_parser,
        ) = self._initialize_tools(config, processing_class)
        self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(config)
        # If turn on `free_cache_engine`, SGLang engine's KV cache
        # will be freed after each `generate_sequences` call.
        logger.info(
            f"tool_schemas: {self._tool_schemas}, tool_map: {self._tool_map}, tool_call_parser_type: "
            f"{self._tool_call_parser_type}, sgl_tools: {self._sgl_tools}, function_call_parser: "
            f"{self._function_call_parser}"
        )
        print(f"tool_schemas: {self._tool_schemas}, tool_map: {self._tool_map}, tool_call_parser_type: "
            f"{self._tool_call_parser_type}, sgl_tools: {self._sgl_tools}, function_call_parser: "
            f"{self._function_call_parser}")

        self._init_distributed_env(device_mesh_cpu=device_mesh, **kwargs)

        self._verify_config(model_hf_config=model_hf_config)
        # initialize the inference engine
        self._init_inference_engine(trust_remote_code, actor_module, port)

        self._init_sampling_params(**kwargs)

        self.processing_class = processing_class

        print(f"[DEBUG] chat_template: {self.processing_class.chat_template}")

        try:
            # This is when processing_class is a tokenizer
            self.pad_token_id = self.processing_class.pad_token_id
        except AttributeError:
            try:
                # This is when processing_class is a processor
                self.pad_token_id = self.processing_class.tokenizer.pad_token_id
            except AttributeError as e:
                raise ValueError(f"Cannot get pad_token_id from processing_class {self.processing_class}") from e

        # ---- Rubric scorer initialization (optional, for streaming parallel scoring during the rollout) ----
        # Enabled via the environment variable ROLLOUT_RUBRIC_ENABLED=true
        # Once enabled, as soon as a P+S round completes, rubric scoring is launched
        # asynchronously,
        # and the results are stored in round_data and passed to the reward manager, avoiding
        # duplicate calls during the reward stage.
        self._rubric_calculator = None
        self._rubric_enabled = os.environ.get("ROLLOUT_RUBRIC_ENABLED", "false").lower() == "true"
        if self._rubric_enabled:
            try:
                from verl.utils.reward_score.ps_rubric_reward import PSRubricRewardCalculator
                planner_rubric_path = os.environ.get(
                    "PS_PLANNER_RUBRIC_PATH", "./rubrics/planner_rubric.json"
                )
                synthesizer_rubric_path = os.environ.get(
                    "PS_SYNTHESIZER_RUBRIC_PATH", "./rubrics/synthesizer_rubric.json"
                )
                rubric_judge_model = os.environ.get("RUBRIC_JUDGE_MODEL", "gemini-2.5-flash-lite")
                # LLM Judge API credentials: provide them via environment variables, don't hardcode
                rubric_api_key = os.environ.get("RUBRIC_API_KEY") or os.environ.get("LLM_JUDGE_API_KEY", "")
                rubric_base_url = os.environ.get(
                    "RUBRIC_BASE_URL", "https://api.openai.com/v1"
                )
                if os.path.exists(planner_rubric_path) and os.path.exists(synthesizer_rubric_path):
                    rubric_max_concurrent = int(os.environ.get("RUBRIC_MAX_CONCURRENT_CALLS", "16"))
                    rubric_max_qpm = int(os.environ.get("RUBRIC_MAX_QPM", "200"))
                    self._rubric_calculator = PSRubricRewardCalculator(
                        planner_rubric_path=planner_rubric_path,
                        synthesizer_rubric_path=synthesizer_rubric_path,
                        judge_model=rubric_judge_model,
                        api_key=rubric_api_key,
                        base_url=rubric_base_url,
                        max_concurrent_calls=rubric_max_concurrent,
                        max_qpm=rubric_max_qpm,
                    )
                    logger.info(
                        f"[Rollout Rubric] enabled in-rollout async rubric scoring: "
                        f"planner={planner_rubric_path}, synthesizer={synthesizer_rubric_path}, "
                        f"judge_model={rubric_judge_model}"
                    )
                else:
                    self._rubric_enabled = False
                    logger.warning(
                        f"[Rollout Rubric] rubric file not found, disabled: "
                        f"planner={planner_rubric_path}, synthesizer={synthesizer_rubric_path}"
                    )
            except Exception as e:
                self._rubric_enabled = False
                logger.warning(f"[Rollout Rubric] initialization failed, disabled: {e}")

    def _init_distributed_env(self, device_mesh_cpu, **kwargs):
        self._device_mesh_cpu = device_mesh_cpu
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert self.tensor_parallel_size <= dist.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        self.train_tp = kwargs.get("train_tp", None)
        if self.train_tp is not None:
            # deployed with megatron
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // self.tensor_parallel_size
            sglang_ps.initialize_parallel_state(
                tensor_model_parallel_size=self.tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        tp_size = self.tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        if self._device_mesh_cpu is None:
            device_mesh_kwargs = dict(
                mesh_shape=(world_size // tp_size, tp_size, 1),
                mesh_dim_names=["dp", "tp", "pp"],
            )

            self._device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

        self._rank = self._device_mesh_cpu.get_rank()
        self._tp_rank = self._device_mesh_cpu["tp"].get_local_rank()
        self._tp_size = self._device_mesh_cpu["tp"].size()
        if self._rank == 0:
            logger.info(f"_init_distributed_env: :tp_world: {self._tp_size}, global_world: {world_size}")
        # get tp_rank of this process in this tp group
        visible_devices = [None] * self._device_mesh_cpu.size(1)

        torch.distributed.all_gather_object(
            visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], self._device_mesh_cpu.get_group("tp")
        )
        self.visible_devices_set = set(",".join(visible_devices).split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))

    def _verify_config(self, model_hf_config):
        if not self.config.get("max_model_len", None):
            self.config.max_model_len = self.config.prompt_length + self.config.response_length
        assert (
            self.config.max_model_len >= self.config.prompt_length + self.config.response_length
        ), f"""max_model_len should be greater than total sequence length (prompt_length + response_length):
            {self.config.max_model_len} >= {self.config.prompt_length} + {self.config.response_length}"""
        max_position_embeddings = None
        if hasattr(model_hf_config, "max_position_embeddings"):
            max_position_embeddings = model_hf_config.max_position_embeddings
        elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
            max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
        elif hasattr(model_hf_config, "text_config") and hasattr(
            model_hf_config.text_config, "max_position_embeddings"
        ):
            max_position_embeddings = model_hf_config.text_config.max_position_embeddings
        if max_position_embeddings is None:
            raise ValueError("max_position_embeddings not found in model_hf_config")
        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            assert max_position_embeddings >= self.config.prompt_length + self.config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= self.config.prompt_length + self.config.response_length
            ), (
                f"model context length should be greater than total sequence length, "
                f"got rope_scaling_factor={rope_scaling_factor} and "
                f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        # currently max_assistant_turns stand for max number of tool calls
        if self.config.multi_turn.max_assistant_turns is None:
            self.config.multi_turn.max_assistant_turns = self.config.max_model_len // 3
        if self.config.multi_turn.max_user_turns is None:
            self.config.multi_turn.max_user_turns = self.config.max_model_len // 3

        print(f"[DEBUG] max_assistant_turns: {self.config.multi_turn.max_assistant_turns}")
        print(f"[DEBUG] max_user_turns: {self.config.multi_turn.max_user_turns}")

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        # initialize the inference engine
        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group("tp"),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0
        engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}

        # attention backend will be changed to fa3 if not specified
        attention_backend = engine_kwargs.pop("attention_backend", None)

        if first_rank_in_node:
            rank = dist.get_rank()
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = AsyncEngine(
                model_path=actor_module,
                dtype=self.config.dtype,
                mem_fraction_static=self.config.gpu_memory_utilization,
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                trust_remote_code=trust_remote_code,
                # NOTE(linjunrong): add rank to prevent SGLang generate same port inside PortArgs.init_new
                # when random.seed is being set during training
                port=30000 + rank,
                # NOTE(Chenyang): turn on log_level to see the decoding speed of SGLang Engine
                # log_level="INFO",
                # NOTE(Chenyang): turn the following lines to see the input and output of each request
                # log_requests=True,
                # log_requests_level=2,
                # NOTE(Chenyang): turn on max_running_requests to set the max concurrent running requests
                # max_running_requests=self.config.max_running_requests,
                mm_attention_backend="fa3",
                attention_backend=attention_backend if attention_backend is not None else "fa3",
                # In async mode for AgentLoop, SGLang support token in token out to avoid the tokenizer
                # inconsistency issue.
                skip_tokenizer_init=self.config.mode == "async",
                **engine_kwargs,
            )
        else:
            self._engine = None

        self.sharding_manager = None
        self.is_sleep = True

    def _init_sampling_params(self, **kwargs):
        kwargs = dict(
            n=1,
            max_new_tokens=self.config.response_length,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.05,
        )
        # supporting adding any sampling params from the config file
        for k in self.config.keys():
            if hasattr(SamplingParams(), str(k)) or "stop" in str(k):
                kwargs[k] = self.config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        self.sampling_params = kwargs

    def _initialize_tools(self, config, processing_class):
        """Initialize tools from configuration.

        Args:
            config: Configuration object containing tool-related settings,
                    specifically `config.multi_turn.tool_config_path`.
            tokenizer: The tokenizer instance used for parsing tool calls from
                       the model's generated text.

        Returns:
            tuple: A tuple containing:
                - tool_schemas (list[dict]): OpenAI-formatted JSON schemas
                  defining each tool's capabilities.
                - tool_map (dict[str, BaseTool]): A dictionary mapping tool
                  names to their executable `BaseTool` objects.
                - tool_call_parser_type (str): The identifier for the specific
                  parser type (e.g., 'json_mode', 'tool_code') used to extract
                  tool calls.
                - sgl_tools (list[sglang.srt.openai_api.protocol.Tool]): Tool
                  definitions optimized for SGLang's internal engine.
                - function_call_parser (sglang.srt.function_call_parser.FunctionCallParser):
                  The active parser instance responsible for extracting
                  structured tool calls from model outputs.
        """
        if config.multi_turn.tool_config_path is None:
            return [], {}, None, [], None

        tools_config_file = config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tools_config_file)

        logger.info(f"Initialize tools from configuration.: tool_list: {tool_list}")
        print(f"Initialize tools from configuration.: tool_list: {tool_list}")
        tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
        tool_map = {tool.name: tool for tool in tool_list}
        tool_call_parser_type = get_tool_call_parser_type(processing_class)
        sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
        function_call_parser = FunctionCallParser(
            sgl_tools,
            tool_call_parser_type,
        )

        return (
            tool_schemas,
            tool_map,
            tool_call_parser_type,
            sgl_tools,
            function_call_parser,
        )

    def _initialize_interactions(self, config):
        """Initialize interactions from configuration.

        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if config.multi_turn.interaction_config_path is None:
            return {}

        interaction_config_file = config.multi_turn.interaction_config_path
        interaction_map = initialize_interactions_from_config(interaction_config_file)

        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        # Get step from meta_info, using get() to avoid a KeyError
        step = prompts.meta_info.get("global_steps", 0)  # default value is 0
        self.step = step  # preserve the original logic

        # PS Pipeline mode: Planner-Synthesizer dual-agent pipeline
        ps_pipeline_enabled = False
        if ps_pipeline_enabled and self.config.multi_turn.enable:
            try:
                return self._req_level_generate_sequences_ps(prompts, step=step, **kwargs)
            except Exception as e:
                error_msg = (
                    f"[rank={self._rank}, tp_rank={self._tp_rank}, step={step}] "
                    f"exception while calling _req_level_generate_sequences_ps: {type(e).__name__}: {str(e)}"
                )
                logger.error(error_msg)
                print(error_msg)
                import traceback
                tb = traceback.format_exc()
                logger.error(tb)
                print(tb)
                raise
        
        if self.config.multi_turn.enable:
            try:
                return self._req_level_generate_sequences(prompts, step=step, **kwargs)
            except Exception as e:
                error_msg = f"[rank={self._rank}, tp_rank={self._tp_rank}, step={step}] exception while calling _req_level_generate_sequences: {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                print(error_msg)
                import traceback
                tb = traceback.format_exc()
                logger.error(tb)
                print(tb)
                raise
        try:
            return self._batch_level_generate_sequences(prompts, **kwargs)
        except Exception as e:
            error_msg = f"[rank={self._rank}, tp_rank={self._tp_rank}, step={step}] exception while calling _batch_level_generate_sequences: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
            import traceback
            tb = traceback.format_exc()
            logger.error(tb)
            print(tb)
            raise

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _batch_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generates single-turn sequences for a batch of prompts.
        For single-turn generation, all prompts are processed in one request.
        `_batch_level_generate_sequences` involves:
        1.  Extracting and pre-processing prompt token IDs from the input
            `prompts`. This includes handling padding and preparing raw
            token ID lists.
        2.  Preparing inputs for the SGLang engine, including multi-modal
            data if present.
        3.  Invoking the SGLang engine (`self._engine.async_generate`,
            an async coroutine) with the batch of processed inputs and
            specified sampling parameters on the master TP rank.
        4.  Broadcasting the results from the master TP rank to all
            other TP ranks.
        5.  Post-processing the engine's output to format the generated
            token IDs and (if applicable) log probabilities.
        6.  Constructing the final sequences by concatenating original
            prompts with the generated responses.
        7.  Updating attention masks and position IDs to reflect the full
            concatenated sequences.
        8.  If `self.config.free_cache_engine` is true, the SGLang engine's
            KV cache is flushed after generation on the master TP rank.
        Args:
            prompts: A `DataProto` object containing the batch of
              input prompts, including tensor data (like `input_ids`,
              `attention_mask`) and meta-information (like `eos_token_id`,
              `do_sample`).
            **kwargs: Additional keyword arguments that can override the
              default sampling parameters (e.g., `temperature`, `top_p`,
              `max_new_tokens`). These are temporarily applied using
              `update_sampling_params`.
        Returns:
            DataProto: A `DataProto` object containing the batch of
              generated sequences. This includes tensors for `prompts`
              (original input IDs), `responses` (generated token IDs),
              `input_ids` (concatenated prompt and response),
              `attention_mask`, and `position_ids` for the full
              sequences.
        Note that in GRPO, if the prompts are validated, we repeat the prompts for rollout.n times in ray_trainer.
        Thus we do not need to repeat the prompts here and set the sampling parameter n to 1.
        """
        # input ids: (bs, prompt_length), left-padded
        idx = prompts.batch["input_ids"]
        # attention_mask: (bs, seq_length), left-padded
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to generate attention mask for the
        # response based on EOS token position
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        # Extract non-tensor data
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]).tolist() for i in range(batch_size)],
                dtype=object,
            )

        if "multi_modal_data" in non_tensor_batch:
            sglang_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
                strict=True,
            ):
                sglang_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                        "image_data": (
                            multi_modal_data.get("image", None) if isinstance(multi_modal_data, dict) else None
                        ),
                    }
                )
        else:
            sglang_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in sglang_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        # Extract token IDs and image data for SGLang Engine
        idx_list = [input_data["prompt_token_ids"] for input_data in sglang_inputs]
        image_list = [input_data.get("image_data", None) for input_data in sglang_inputs]

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        # Create request-level sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update(
                {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.05,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "ignore_eos": False,
                    "min_new_tokens": 0,
                    "max_new_tokens": self.config.response_length,
                    "skip_special_tokens": True,
                    "spaces_between_special_tokens": True,
                }
            )
        elif is_validate:
            request_sampling_params.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            )

        # Update with any additional kwargs
        request_sampling_params.update(kwargs)
        torch.cuda.synchronize()
        engine_call_start_time = time.time()
        if self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            output = loop.run_until_complete(
                self._engine.async_generate(
                    prompt=None,  # because we have already convert it to prompt token id
                    sampling_params=request_sampling_params,
                    return_logprob=True,
                    input_ids=idx_list,
                    image_data=image_list,
                )
            )
            torch.cuda.synchronize()
            generate_end_time = time.time()
            log_path = os.path.join(
                self.log_dir,
                f"step_{self.step}",
                f"worker_{self._rank}.jsonl"
            )
            self.log_manager.log(
                log_path,
                event="engine_async_generate",
                duration=generate_end_time - engine_call_start_time,
                workid=self._rank,
                step=self.step
            )
        else:
            output = None

        torch.cuda.synchronize()
        barrier_start_time = time.time()
        # Most naive implementation, can extract tensor and send via gloo if too slow
        dist.barrier()
        torch.cuda.synchronize()
        barrier_end_time = time.time()
        if self._tp_rank == 0:
            log_path = os.path.join(
                self.log_dir,
                f"step_{self.step}",
                f"worker_{self._rank}.jsonl"
            )
            self.log_manager.log(
                log_path,
                event="barrier",
                duration=barrier_end_time - barrier_start_time,
                workid=self._rank,
                step=self.step
            )
        [output] = broadcast_pyobj(
            data=[output],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )
        out = _post_process_outputs(self.processing_class, output)

        response = out[0].to(idx.device)
        rollout_log_probs = None
        if self.config.calculate_log_probs:
            rollout_log_probs = out[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_sequence_to_length(
                    rollout_log_probs, self.config.response_length, self.pad_token_id
                )

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        # free cache engine
        if self._engine is not None and self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def _async_rollout_a_request(
        self,
        req: AsyncRolloutRequest,
        do_sample: bool = True,
        is_validate: bool = False,
        **kwargs,
    ) -> AsyncRolloutRequest:
        """Asynchronously run the rollout of a single request.

        Key variables:
        - req: AsyncRolloutRequest - the original request object, containing the initial prompt_ids, prompt_mask, etc.
        - _req: AsyncRolloutRequest - a deep copy of req, modified in this function (messages added, state updated, etc.)
        - finish_reason_type: FinishReasonTypeEnum - the finish reason (STOP/LENGTH/TOOL_CALL)
        - output: dict - a single generation output from the engine, with fields such as text, output_ids, meta_info

        Counter variables:
        - current_turns: int - the number of completed assistant turns (one per LLM generation)
        - current_docs: int - the cumulative number of retrieved documents (tallied during tool_call)
        - user_turns: int - the number of user interaction turns (interaction-related)
        - user_turn_rewards: list[float] - the reward of each user turn

        Messages and tool calls:
        - _req.messages: list[Message] - the full conversation history; each Message has role, content, tool_calls
        - _req.messages[-1].tool_calls: list[OpenAIFunctionToolCall] - the tool calls of the last message

        === Relationship between the three token sequences (multi-turn tool use) ===

        Below is how the three variables evolve in a multi-turn tool-use scenario.

        Example message sequence:
        [sys, user, assistant(think+tool_call), tool(response), assistant(think+tool_call),
         tool(response), assistant(think+answer)]

        _req.prompt_ids: torch.Tensor [1, prompt_len]
        ├─ Definition: the token IDs of the initial conversation (system prompt + user question)
        ├─ Initialized: set when AsyncRolloutRequest is initialized (initialize_request)
        ├─ Content: tokenize([sys, user]) + generation_prompt (e.g. "<im_start>assistant\n")
        └─ Changes: **stays constant throughout the rollout**, always the original prompt

        _req.input_ids: torch.Tensor [1, current_seq_len]
        ├─ Definition: the current full token sequence (prompt + everything generated so far)
        ├─ Initialized: equal to prompt_ids at the start
        ├─ Grows whenever the following methods are called
        │   ├─ add_assistant_message(): appends the assistant's generated think + tool_call
        │   ├─ add_tool_response_messages(): appends the observation returned by the tool
        │   └─ add_user_message(): appends a user interaction message (interaction scenario)
        ├─ Update method: _update_input_ids() -> torch.cat([input_ids, new_content_ids], dim=-1)
        └─ Content evolution example:
            initial: [sys, user, <im_start>assistant\n]
            Turn1: [sys, user, assistant(think), assistant(tool_call), tool(obs1)]
            Turn2: [sys, user, assistant(think), assistant(tool_call), tool(obs1),
                   assistant(think), assistant(tool_call), tool(obs2)]
            TurnN: [sys, user, ..., assistant(final_answer)]

        _req.response_ids: torch.Tensor [1, response_len]
        ├─ Definition: only the generated token IDs (excluding the original prompt)
        ├─ Initialized: None (at the start of the rollout)
        ├─ Computed: in the finalize() method
        │   └─ response_ids = input_ids[:, prompt_ids.shape[-1]:]
        └─ Content: all tokens from the end of the prompt to the end of input_ids
            e.g.: [assistant(think), assistant(tool_call), tool(obs1),
                   assistant(think), assistant(tool_call), tool(obs2),
                   assistant(final_answer)]

        Key points:
        1. prompt_ids is fixed and is the "input" part during training
        2. input_ids grows dynamically and is the complete conversation sequence
        3. response_ids = input_ids - prompt_ids, and is the "output" part during training
        4. In the multi-turn case, response_ids interleaves multiple rounds of assistant
           generation and tool observations

        Other key fields:
        - _req.request_id: str - the unique identifier of the request
        - _req.state: AsyncRolloutRequestStateEnum - the request state
          (PENDING/RUNNING/TOOL_CALLING/INTERACTING/COMPLETED)
        """
        assert self._tp_rank == 0, "only the master process can call this function"
        _req = deepcopy(req)
        finish_reason_type = None  # finish-reason type (STOP/LENGTH/TOOL_CALL)
        output = None  # engine output, dict: {"text": str, "output_ids": list[int], "meta_info": {...}}

        # === Counter variables ===
        current_turns = 0  # int: assistant turn counter, +1 after each LLM generation
        current_docs = 0  # int: cumulative retrieved-doc count, summed from tool_metrics["total_results"]
        user_turns = 0  # int: user interaction turns (interaction scenario)
        user_turn_rewards = []  # list[float]: the reward of each user turn

        # ===================================
        # Example of the complete multi-turn tool-use flow
        # ===================================

        # Initial state (after initialize_request):
        # ├─ messages: [{"role": "system", ...}, {"role": "user", "content": "query"}]
        # ├─ prompt_ids: [1, 50] = tokenize([sys, user, <im_start>assistant\n])
        # ├─ input_ids: [1, 50] = prompt_ids
        # └─ response_ids: None

        # Turn 1 - RUNNING -> TOOL_CALLING:
        # ├─ Engine generates -> output["text"] = "<think>analyze the problem</think>\n<tool_call>search(...)</tool_call>"
        # ├─ after add_assistant_message():
        # │   ├─ messages: [..., {"role": "assistant", "content": "...", "tool_calls": [...]}]
        # │   ├─ prompt_ids: [1, 50] (unchanged)
        # │   ├─ input_ids: [1, 200] (assistant tokens appended)
        # │   └─ response_ids: None
        # ├─ Tool executes -> tool_call_results = [("observation 1", 0.0, {"total_results": 3})]
        # ├─ after add_tool_response_messages():
        # │   ├─ messages: [..., {"role": "tool", "content": "observation 1"}]
        # │   ├─ prompt_ids: [1, 50] (unchanged)
        # │   ├─ input_ids: [1, 350] (tool response tokens appended)
        # │   └─ response_ids: None
        # └─ current_docs: 3

        # Turn 2 - RUNNING -> TOOL_CALLING:
        # ├─ Engine generates -> output["text"] = "<think>keep analyzing</think>\n<tool_call>search(...)</tool_call>"
        # ├─ after add_assistant_message():
        # │   ├─ messages: [..., {"role": "assistant", "content": "...", "tool_calls": [...]}]
        # │   ├─ prompt_ids: [1, 50] (unchanged)
        # │   ├─ input_ids: [1, 500] (assistant tokens appended)
        # │   └─ response_ids: None
        # ├─ Tool executes -> tool_call_results = [("observation 2", 0.0, {"total_results": 2})]
        # ├─ after add_tool_response_messages():
        # │   ├─ messages: [..., {"role": "tool", "content": "observation 2"}]
        # │   ├─ prompt_ids: [1, 50] (unchanged)
        # │   ├─ input_ids: [1, 650] (tool response tokens appended)
        # │   └─ response_ids: None
        # └─ current_docs: 5

        # Turn 3 - RUNNING -> STOP (final answer):
        # ├─ Engine generates -> output["text"] = "<think>synthesize the information</think>\nthe final answer is..."
        # ├─ after add_assistant_message():
        # │   ├─ messages: [..., {"role": "assistant", "content": "the final answer is..."}]
        # │   ├─ prompt_ids: [1, 50] (unchanged)
        # │   ├─ input_ids: [1, 800] (assistant final-answer tokens appended)
        # │   └─ response_ids: None
        # └─ current_turns: 3, finish_reason_type: STOP

        # After finalize:
        # ├─ messages: [sys, user, assistant1, tool1, assistant2, tool2, assistant3]
        # ├─ prompt_ids: [1, 50] (always unchanged)
        # ├─ input_ids: [1, 800] (the complete sequence)
        # ├─ response_ids: [1, 750] = input_ids[:, 50:] (all generated content)
        # ├─ response_loss_mask: [1, 750]
        # │   └─ [1,1,...,1, 0,0,...,0, 1,1,...,1, 0,0,...,0, 1,1,...,1]
        # │       ^assistant1  ^tool1     ^assistant2  ^tool2     ^assistant3
        # └─ reward_scores: {"search_tool": [0.5, 0.3], ...}
        # ==================================


        # Turn-level timing tracking
        turn_timings = []

        # Full request-level timing record
        torch.cuda.synchronize()
        request_start_time = time.time()
        log_path = os.path.join(
            self.log_dir,
            f"step_{self.step}",
            f"worker_{self._rank}.jsonl"
        )

        self.log_manager.log(
            log_path,
            event="request_start",
            extra={
                "request_id": req.request_id,
                "initial_prompt_length": len(req.get_generation_prompt_ids(self.processing_class))
            },
            workid=self._rank,
            step=self.step
        )
        # Create request-level sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update(
                {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.05,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "ignore_eos": False,
                    "min_new_tokens": 0,
                    "max_new_tokens": self.config.response_length,
                    "skip_special_tokens": True,
                    "spaces_between_special_tokens": True,
                }
            )
        elif is_validate:
            request_sampling_params.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            )

        # Update with any additional kwargs
        request_sampling_params.update(kwargs)
        torch.cuda.synchronize()
        main_loop_start_time = time.time()
        while current_turns < self.config.multi_turn.max_assistant_turns:
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                torch.cuda.synchronize()
                pending_start_time = time.time()
                await self._handle_pending_state(_req)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
                torch.cuda.synchronize()
                pending_end_time = time.time()
                self.log_manager.log(
                    log_path,
                    event="pending_state_handling",
                    duration=pending_end_time - pending_start_time,
                    extra={"request_id": _req.request_id},
                    workid=self._rank,
                    step=self.step
                )
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                torch.cuda.synchronize()
                tool_calling_start_time = time.time()
                if _req.messages[-1].tool_calls is not None:
                    # === Tool-call related variables ===
                    # parsed_tool_calls: list[OpenAIFunctionToolCall] - the parsed tool-call list
                    #   each OpenAIFunctionToolCall contains:
                    #     - id: str - the unique ID of the tool call
                    #     - function.name: str - the tool name (e.g. "search_tool")
                    #     - function.arguments: dict - the tool arguments (JSON format)
                    parsed_tool_calls = _req.messages[-1].tool_calls
                    torch.cuda.synchronize()
                    tool_execution_start_time = time.time()

                    # 1017update: add the original query field to the tool_call
                    # print("[DEBUG] tool_calling all: ", _req)
                    
                    # === Prepare the tool-call arguments ===
                    # prompt_str: str - the initial prompt token sequence decoded to text, used to
                    #   extract the user's original question
                    #   purpose: extract the user question in search_tool via extract_prompt_str()
                    #   note: use prompt_ids rather than input_ids, since we want the original
                    #   question, not the full conversation history
                    prompt_str = self.processing_class.decode(_req.prompt_ids[0], skip_special_tokens=True)
                    # print("[DEBUG] tool_calling_prompt_str: ", prompt_str)
                    
                    # === Tool execution results ===
                    # tool_call_results: list[tuple[ToolResponse, float, dict]]
                    #   each tuple contains three elements:
                    #     - resp: ToolResponse - the response object returned by the tool
                    #     - reward: float - the immediate reward of the tool execution (currently unused)
                    #     - tool_metrics: dict - the tool execution metrics
                    #         e.g.: {"total_results": 5, "status": "success", "api_request_error": None, ...}
                    try:
                        tool_call_results = await asyncio.gather(
                            *[
                                self._tool_map[tool_call.function.name].execute(
                                    _req.request_id,
                                    tool_call.function.arguments,
                                    **_req.tools_kwargs[tool_call.function.name].get("execute_kwargs", {"prompt_str": prompt_str}),
                                )
                                for tool_call in parsed_tool_calls
                            ]
                        )
                    except Exception as e:
                        # Tool execution failed - handled uniformly
                        logger.error(
                            f"[sglang_rollout._async_rollout_a_request] Request {_req.request_id} turn {current_turns}: "
                            f"Tool execution failed - {e}"
                        )
                        traceback.print_exc()
                        _req.search_error = True
                        _req.state = AsyncRolloutRequestStateEnum.FAILED
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    
                    # ========================================
                    # === Detect search_error (environment failure) ===
                    # ========================================
                    # Detect search abnormalities from tool_metrics:
                    #
                    # Two cases, handled differently:
                    # 1. API error (api_request_error is set or status == "api_error"):
                    #    this means the underlying call_search_api already retried MAX_RETRIES
                    #    times and all attempts failed,
                    #    only then do we mark search_error and discard the whole trajectory
                    # 2. The search returned 0 results (total_results == 0, but the API itself succeeded):
                    #    do not mark search_error; the result text already contains hints such as
                    #    "No results found...",
                    #    so the model can keep reasoning based on the hint and retry with different keywords
                    #
                    for tool_result_idx, (_, _, tool_metrics) in enumerate(tool_call_results):
                        # doc_num: int - the number of documents returned by this tool call
                        doc_num = tool_metrics.get("total_results", 0)
                        current_docs += doc_num  # accumulate into the total document count
                        
                        api_error = tool_metrics.get("api_request_error")
                        status = tool_metrics.get("status")
                        
                        # Only an error after the API retries are exhausted is marked search_error
                        # (an environment failure, discarded)
                        if api_error or status == "api_error":
                            _req.search_error = True
                            logger.warning(
                                f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                f"Search API error detected - total_results={doc_num}, status={status}, "
                                f"api_error={api_error}"
                            )
                        elif doc_num == 0:
                            # Search returned 0 results: do not mark search_error, the model can keep reasoning
                            logger.info(
                                f"[Search] Request {_req.request_id} turn {current_turns}: "
                                f"Search returned 0 results (status={status}), "
                                f"model will see 'No results found' hint and continue reasoning"
                            )


                    torch.cuda.synchronize()
                    tool_execution_end_time = time.time()
                    self.log_manager.log(
                        log_path,
                        event="tool_execution",
                        duration=tool_execution_end_time - tool_execution_start_time,
                        extra={"request_id": _req.request_id, "turn": current_turns},
                        workid=self._rank,
                        step=self.step
                    )
                    torch.cuda.synchronize()
                    tool_response_start_time = time.time()
                    # === Append the tool response to the conversation history ===
                    # Calling add_tool_response_messages() will:
                    # 1. append the tool response to _req.messages
                    # 2. update _req.input_ids via _update_input_ids()
                    # 
                    # Token sequence change:
                    # before: [sys, user, assistant(think+tool_call)]
                    # after:  [sys, user, assistant(think+tool_call), tool(obs)]
                    # 
                    # State of the three:
                    # - prompt_ids: unchanged, still [sys, user, <im_start>assistant\n]
                    # - input_ids: grows, the tool(obs) tokens are appended
                    # - response_ids: still None (not finalized yet)
                    try:
                        _req.add_tool_response_messages(self.processing_class, [resp for resp, _, _ in tool_call_results])
                        for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results, strict=True):
                            _req.update_metrics(metrics, tool_call.function.name)
                    except Exception as e:
                        # add_tool_response_messages or update_metrics failed - handled uniformly
                        logger.error(
                            f"[sglang_rollout._async_rollout_a_request] Request {_req.request_id} turn {current_turns}: "
                            f"Failed to process tool responses - {e}"
                        )
                        traceback.print_exc()
                        _req.state = AsyncRolloutRequestStateEnum.FAILED
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        _req.tool_parse_error = True
                        break
                    torch.cuda.synchronize()
                    tool_response_end_time = time.time()
                    self.log_manager.log(
                        log_path,
                        event="tool_response_processing",
                        duration=tool_response_end_time - tool_response_start_time,
                        extra={"request_id": _req.request_id, "turn": current_turns},
                        workid=self._rank,
                        step=self.step
                    )
                    # ========================================
                    # === Length check after the tool call (checkpoint 1) ===
                    # ========================================
                    # 
                    # After appending the tool response to input_ids, check whether the sequence
                    # length exceeds the limit
                    # 
                    # Why this check is needed:
                    # - the tool response can be very long (e.g. a search returning lots of text)
                    # - appending it may push the sequence past max_model_len
                    # - it must be checked before the next generation round, to avoid an SGLang error
                    # 
                    # Check condition: len(_req.input_ids) >= self.config.max_model_len
                    # 
                    # Example trigger scenario:
                    # - max_model_len = 8000
                    # - current input_ids = 7900 tokens
                    # - tool response = 150 tokens
                    # - new input_ids = 8050 tokens > 8000
                    # 
                    # How it is handled:
                    # - set finish_reason = STOP
                    # - break out of the loop and stop generating
                    # - do not attempt another generation round
                    # 
                    # Logging:
                    # - print a DEBUG message: "FinishReasonTypeEnum.STOP current_turns={}, prompt_length={}"
                    # - event="async_rollout_request_complete"
                    # - extra={"finish_reason": "STOP", "total_sequence_length": len(input_ids)}
                    # print(f"[DEBUG]toolcall: {_req.messages}")
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        print(f"[DEBUG] FinishReasonTypeEnum.STOP current_turns{current_turns}, prompt_length{len(_req.input_ids)}")
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    # ========================================
                    # === Exception handling: inconsistent state ===
                    # ========================================
                    # Trigger condition:
                    # - the state is TOOL_CALLING, but messages[-1].tool_calls is None
                    # - this is an inconsistent state, possibly caused by:
                    #   1. a logic bug in the code
                    #   2. the state not being updated correctly on an exception path
                    #
                    # How it is handled:
                    # - record detailed debug information
                    # - mark it as tool_parse_error
                    # - set the state to COMPLETED and break
                    logger.error(
                        f"[sglang_rollout._async_rollout_a_request] Request {_req.request_id} turn {current_turns}: "
                        f"Unexpected state: TOOL_CALLING but messages[-1].tool_calls is None. "
                        f"Last message: {_req.messages[-1]}"
                    )
                    _req.tool_parse_error = True
                    _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                    finish_reason_type = FinishReasonTypeEnum.STOP
                    break
                torch.cuda.synchronize()
                tool_calling_end_time = time.time()
                self.log_manager.log(
                    log_path,
                    event="tool_calling_state",
                    duration=tool_calling_end_time - tool_calling_start_time,
                    extra={"request_id": _req.request_id, "turn": current_turns},
                    workid=self._rank,
                    step=self.step
                )
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                # === TURN START ===
                torch.cuda.synchronize()
                turn_start_time = time.time()
                self.log_manager.log(
                    log_path,
                    event="turn_start",
                    extra={
                        "request_id": _req.request_id,
                        "turn": current_turns,
                        "current_sequence_length": len(_req.get_generation_prompt_ids(self.processing_class))
                    },
                    workid=self._rank,
                    step=self.step
                )

                # === TURN PRE-PROCESS ===
                torch.cuda.synchronize()
                turn_pre_process_start_time = time.time()

                # ========================================
                # === Length check before generation (checkpoint 2) ===
                # ========================================
                # 
                # Before calling the engine to generate, check the current sequence length
                # 
                # Why this check is needed:
                # - SGLang automatically appends an EOS token when generating
                # - if prompt_length + max_new_tokens + 1 > max_model_len, it errors out
                # - one token slot must be reserved for EOS
                # 
                # Check condition: prompt_length + 1 >= max_model_len
                # 
                # Meaning of prompt_length:
                # - obtained via _req.get_generation_prompt_ids(self.processing_class)
                # - contains the full conversation history (prompt + everything generated so far)
                # - this is the actual token sequence passed to the engine
                # 
                # Example trigger scenario:
                # - max_model_len = 8000
                # - prompt_length = 7999
                # - 7999 + 1 >= 8000 -> the check triggers
                # 
                # How it is handled:
                # - set finish_reason = LENGTH
                # - break out of the loop and stop generating
                # - do not call the engine to generate
                # 
                # Difference from checkpoint 1:
                # - checkpoint 1: checks len(_req.input_ids) (the complete sequence)
                # - checkpoint 2: checks len(generation_prompt_ids) (the prompt used for generation)
                # - the two are usually identical, but may differ slightly in some cases
                # 
                # Logging:
                # - print a DEBUG message: "FinishReasonTypeEnum.LENGTH current_turns={}, prompt_length={}"
                # - event="async_rollout_request_complete"
                # - extra={"finish_reason": "LENGTH"}
                # 
                # Only continue the conversation if the prompt length is not greater than max_model_len - 1,
                # since SGLang raises an error when max_new_tokens + 1 is greater to max_model_len (the extra
                # token accounts for the EOS token).
                prompt_length = len(_req.get_generation_prompt_ids(self.processing_class))

                if prompt_length + 1 >= self.config.max_model_len:
                    finish_reason_type = FinishReasonTypeEnum.LENGTH
                    print(f"[DEBUG] FinishReasonTypeEnum.LENGTH current_turns{current_turns}, prompt_length{prompt_length}")
                    break

                # Video support is not implemented yet
                image_data = (
                    _req.multi_modal_data["image"]
                    if _req.multi_modal_data and "image" in _req.multi_modal_data
                    else None
                )
                video_data = (
                    _req.multi_modal_data["video"]
                    if _req.multi_modal_data and "video" in _req.multi_modal_data
                    else None
                )
                if video_data:
                    logger.warning(
                        "video support is not implemented yet, current length of video data is %d", len(video_data)
                    )
                torch.cuda.synchronize()
                turn_pre_process_end_time = time.time()
                turn_pre_process_duration = turn_pre_process_end_time - turn_pre_process_start_time
                self.log_manager.log(
                    log_path,
                    event="turn_pre_process",
                    duration=turn_pre_process_duration,
                    extra={
                        "request_id": _req.request_id,
                        "turn": current_turns,
                        "generation_prompt_length": len(_req.get_generation_prompt_ids(self.processing_class))
                    },
                    workid=self._rank,
                    step=self.step
                )

                # === TURN ENGINE CALL ===
                torch.cuda.synchronize()
                turn_engine_call_start_time = time.time()
                # output: dict - the engine generation output, containing these key fields:
                #   {
                #     "text": str - the generated text (including think + tool_call, or the final answer)
                #     "output_ids": list[int] - the generated token IDs
                #     "meta_info": {
                #       "finish_reason": {"type": str} - the finish reason ("stop"/"length")
                #       "output_token_logprobs": list[tuple] - the log probability of each token
                #       ...
                #     }
                #   }
                output = await self._handle_engine_call(_req, request_sampling_params, image_data=image_data)
                torch.cuda.synchronize()
                turn_engine_call_end_time = time.time()
                turn_engine_call_duration = turn_engine_call_end_time - turn_engine_call_start_time
                self.log_manager.log(
                    log_path,
                    event="turn_engine_call",
                    duration=turn_engine_call_duration,
                    extra={
                        "request_id": _req.request_id,
                        "turn": current_turns,
                        "generated_tokens": len(output.get("output_ids", [])) if output else 0,
                        "output_text_length": len(output.get("text", "")) if output else 0
                    },
                    workid=self._rank,
                    step=self.step
                )

                # === TURN POST-PROCESS ===
                torch.cuda.synchronize()
                turn_post_process_start_time = time.time()

                # content: str - the generated text extracted from output
                #   format example: "<think>reasoning</think>\n<tool_call>...</tool_call>"
                #   or a plain-text answer
                content = output["text"]
                # finish_reason_type: FinishReasonTypeEnum - the finish reason returned by the engine
                #   possible values: STOP (normal end) / LENGTH (too long) /
                #   TOOL_CALL (a tool call is required)
                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])

                # Initialize tool parsing timing
                tool_parsing_duration = 0
                has_tool_calls = False
                parsed_tool_calls_count = 0

                # Update the assistant-turns counter (after each engine generation)
                current_turns += 1
                _req.total_assistant_turns = current_turns
                
                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(self.processing_class, content)
                    print(f"[DEBUG] FinishReasonTypeEnum.LENGTH current_turns{current_turns}")
                    break
                else:
                    if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                        finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                        _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                        torch.cuda.synchronize()
                        tool_parsing_start_time = time.time()
                        
                        # ========================================
                        # === Count the occurrences of the <tool_call> tag ===
                        # ========================================
                        # Count how many <tool_call> tags the current round's generation contains
                        # Count them even if parsing fails (used to detect exceeding the max allowed count)
                        tool_call_tag_count = content.count("<tool_call>")
                        _req.total_tool_call_attempts += tool_call_tag_count
                        
                        # Initialize the parse-error flag
                        has_parse_error = False
                        parse_error_reason = None
                        
                        try:
                            # ========================================
                            # === Tool-call parsing (a format error may occur) ===
                            # ========================================
                            # 
                            # self._function_call_parser.parse_non_stream(content) tries to parse the
                            # tool calls out of the generated text
                            # 
                            # Input: content - the raw text generated by the model
                            #   example: "<think>reasoning process</think>\n<tool_call>{\"name\": \"search\","arguments":{"query_list": }}</tool_call>"
                            # 
                            # Output:
                            #   - normed_content: str - the normalized content (tool-call tags removed,
                            #     think preserved)
                            #   - tool_calls: list[ToolCallItem] - the raw tool-call list
                            #       each ToolCallItem contains:
                            #         - name: str - the tool name (e.g. "search")
                            #         - parameters: str - the JSON-format parameter string
                            #         - tool_index: int - the tool-call index
                            normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
                        except JSONDecodeError as e:
                            # Exception handling 1: JSON parsing failed
                            # 
                            # Trigger condition:
                            # 1. The <tool_call> tag is not properly closed
                            #    example: "<tool_call>{\"query\": \"test\"</tool_call>" (JSON not closed)
                            # 2. Malformed JSON
                            #    example: "<tool_call>{query: test}</tool_call>" (missing quotes)
                            # 3. The <think> tag is not closed
                            #    example: "<think>reasoning..." (missing </think>)
                            # 
                            # How it is handled:
                            # - keep the original content (normed_content = content)
                            # - clear the tool-call list (tool_calls = [])
                            # - this content is later appended to messages as plain text
                            # - set finish_reason = STOP and end the generation
                            # 
                            # Logging:
                            # - event="turn_tool_parsing"
                            # - extra={"raw_tool_calls": 0, "parsed_tool_calls": 0}
                            normed_content = content
                            tool_calls = []
                            has_parse_error = True
                            parse_error_reason = f"JSONDecodeError: {str(e)}"
                            _req.tool_parse_error = True
                            logger.warning(f"[Abnormal] Request {_req.request_id} turn {current_turns}: JSON parse error - {str(e)}")
                        except AttributeError as e:
                            # Exception handling 2: parser attribute error
                            # 
                            # Trigger condition:
                            # - the parser object lacks a required attribute or method (very rare)
                            # 
                            # Handled the same way as JSONDecodeError
                            normed_content = content
                            tool_calls = []
                            has_parse_error = True
                            parse_error_reason = f"AttributeError: {str(e)}"
                            _req.tool_parse_error = True
                            logger.warning(f"[Abnormal] Request {_req.request_id} turn {current_turns}: Parser attribute error - {str(e)}")
                        # ========================================
                        # === Tool-call argument validation and filtering ===
                        # ========================================
                        # 
                        # Tool format: <tool_call>{"name": "search", "arguments": {"query_list": ["q1","q2","q3"]}}</tool_call>
                        # 
                        # Filter the valid tool calls out of the raw tool_calls list
                        # Filter conditions:
                        # 1. the tool name must be "search" (present in tool_map)
                        # 2. the arguments must parse correctly into a dict
                        # 3. the arguments must contain a query_list field
                        
                        parsed_tool_calls = []
                        invalid_tool_calls_count = 0  # number of tool calls whose arguments failed to parse
                        unknown_tool_calls_count = 0  # number of calls to unknown tools
                        unknown_tool_names = []  # record the names of unknown tools (for monitoring)
                        
                        # Extract all queries, for duplicate detection and statistics
                        current_turn_queries = []  # all queries of the current round
                        current_turn_query_count = 0  # total queries this round (total length of query_list)
                        
                        for tool_call in tool_calls:
                            # ========================================
                            # === Detect unknown tool calls ===
                            # ========================================
                            # Check whether the tool is in the tool map (should only be "search")
                            if tool_call.name not in self._tool_map:
                                unknown_tool_calls_count += 1
                                unknown_tool_names.append(tool_call.name)
                                has_parse_error = True
                                if parse_error_reason is None:
                                    parse_error_reason = f"Unknown tool: {tool_call.name}"
                                _req.tool_parse_error = True
                                logger.warning(
                                    f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                    f"Unknown tool '{tool_call.name}' called (expected 'search')"
                                )
                                continue  # skip unknown tools
                            
                            # ========================================
                            # === Argument validation: parse the JSON string ===
                            # ========================================
                            # Try converting the JSON string into a dict
                            # 
                            # OpenAIFunctionCallSchema.from_openai_function_parsed_schema() does:
                            # 1. try json.loads(tool_call.parameters) to parse the JSON
                            # 2. check whether the parsed result is a dict
                            # 3. return (function, has_decode_error)
                            function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                                OpenAIFunctionParsedSchema(
                                    name=tool_call.name,
                                    arguments=tool_call.parameters,  # JSON string, e.g. '{"query_list": ["q1","q2"]}'
                                )
                            )
                            
                            # Exception handling: argument decode error
                            # Trigger condition: has_decode_error = True
                            # - JSON parsing failed (malformed, unclosed brackets, etc.)
                            # - the parsed result is not a dict
                            # - the arguments are empty or malformed
                            if has_decode_error:
                                invalid_tool_calls_count += 1
                                has_parse_error = True
                                if parse_error_reason is None:
                                    parse_error_reason = f"Parameter decode error for tool: {tool_call.name}"
                                _req.tool_parse_error = True
                                logger.warning(
                                    f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                    f"Parameter decode error for tool '{tool_call.name}'"
                                )
                                continue  # discard tool calls that have errors
                            
                            # ========================================
                            # === Extract query_list for statistics and duplicate detection ===
                            # ========================================
                            # Tool argument format: {"query_list": ["subq1", "subq2", "subq3"]}
                            try:
                                if isinstance(function.arguments, dict) and 'query_list' in function.arguments:
                                    query_list = function.arguments['query_list']
                                    if isinstance(query_list, list):
                                        # Use extend to add all queries to the current round's query list
                                        valid_queries = [q for q in query_list if isinstance(q, str) and q.strip()]
                                        current_turn_queries.extend(valid_queries)
                                        current_turn_query_count += len(valid_queries)
                                    else:
                                        logger.warning(
                                            f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                            f"query_list is not a list: {type(query_list)}"
                                        )
                                else:
                                    logger.warning(
                                        f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                        f"query_list not found in tool arguments"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                    f"Failed to extract query_list: {e}"
                                )
                            
                            parsed_tool_calls.append(
                                OpenAIFunctionToolCall(
                                    id=str(tool_call.tool_index),
                                    function=function,
                                )
                            )
                        
                        parsed_tool_calls_count = len(parsed_tool_calls)  # int: number of successfully parsed tool calls
                        
                        # ========================================
                        # === Check 1: too many queries in a single round ===
                        # ========================================
                        # Check whether the total length of query_list exceeds the limit (5 by default)
                        max_queries_per_turn = self.config.multi_turn.get('max_tool_calls_per_turn', 5)
                        if current_turn_query_count > max_queries_per_turn:
                            _req.excessive_tool_calls_per_turn = True
                            logger.warning(
                                f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                f"Excessive queries {current_turn_query_count} > {max_queries_per_turn}"
                            )
                        
                        # ========================================
                        # === Check 2: duplicate-query detection ===
                        # ========================================
                        has_repeated_query = False
                        repeated_queries = []
                        for query in current_turn_queries:
                            if _req.check_repeated_query(query):
                                _req.repeated_query = True
                                has_repeated_query = True
                                repeated_queries.append(query)
                                logger.warning(
                                    f"[Abnormal] Request {_req.request_id} turn {current_turns}: "
                                    f"Repeated query detected: '{query}'"
                                )
                        
                        # Add all of this round's queries to the history (using extend)
                        _req.query_history.extend(current_turn_queries)
                        
                        # Add the unknown tool names to _req (for later monitoring and analysis)
                        if unknown_tool_names:
                            _req.unknown_tool_names.extend(unknown_tool_names)
                        
                        # ========================================
                        # === Update the statistics ===
                        # ========================================
                        # Record this round's query count (total length of query_list)
                        _req.tool_calls_per_turn.append(current_turn_query_count)
                        # Update the total query count
                        _req.total_tool_calls += current_turn_query_count
                        # Update the max query count in a single round
                        if current_turn_query_count > _req.max_tool_calls_in_single_turn:
                            _req.max_tool_calls_in_single_turn = current_turn_query_count
                        
                        torch.cuda.synchronize()
                        tool_parsing_end_time = time.time()
                        tool_parsing_duration = tool_parsing_end_time - tool_parsing_start_time
                        self.log_manager.log(
                            log_path,
                            event="turn_tool_parsing",
                            duration=tool_parsing_duration,
                            extra={
                                "request_id": _req.request_id,
                                "turn": current_turns,
                                "raw_tool_calls": len(tool_calls),  # number of raw parsed tool calls
                                "parsed_tool_calls": parsed_tool_calls_count,  # number of successfully parsed tool calls
                                "invalid_tool_calls": invalid_tool_calls_count,  # number of argument-parse failures
                                "unknown_tool_calls": unknown_tool_calls_count,  # number of unknown tools
                                "unknown_tool_names": unknown_tool_names,  # list of unknown tool names
                                "current_turn_query_count": current_turn_query_count,  # total queries this round
                                "current_turn_queries": current_turn_queries,  # all queries of this round
                                "has_parse_error": has_parse_error,
                                "parse_error_reason": parse_error_reason,
                                "has_repeated_query": has_repeated_query,
                                "repeated_queries": repeated_queries,  # list of duplicated queries
                                "excessive_queries": _req.excessive_tool_calls_per_turn,
                            },
                            workid=self._rank,
                            step=self.step
                        )

                        # ========================================
                        # === Break + 0-reward abnormality detection ===
                        # ========================================
                        # Core logic:
                        # As long as the model emitted a <tool_call> (i.e. has_tool_call(content)
                        # returned True),
                        # it means the model wants to call a tool.
                        # 
                        # If at this point no valid tool call was parsed, or some other abnormality occurred,
                        # we should Break and give a 0 reward.
                        # 
                        # The abnormal cases include:
                        # 1. parsed_tool_calls_count == 0: no tool call was parsed successfully
                        #    - JSON parsing failed (tool_parse_error)
                        #    - all tools are unknown (unknown_tool_calls)
                        #    - the arguments failed to decode (invalid_tool_calls)
                        # 2. excessive_tool_calls_per_turn: query_list exceeds the length limit
                        # 3. repeated_query: there are duplicate queries
                        
                        # Check whether no tool call was parsed successfully
                        no_valid_tool_calls = (parsed_tool_calls_count == 0)
                        
                        # Decide whether to Break + discard
                        # tool_parse_error / no_valid_tool_calls -> no complete trajectory, discard
                        # excessive_tool_calls_per_turn -> no break, keep going; only the
                        # offending round's reward is zeroed
                        # repeated_query -> no break, compute normally
                        should_break_with_zero_reward = (
                            no_valid_tool_calls or  # no tool call was parsed successfully (the key condition)
                            _req.tool_parse_error  # parse error -> discard
                        )
                        
                        if should_break_with_zero_reward:
                            # Append the content as plain text (without tool_calls)
                            _req.add_assistant_message(self.processing_class, content)
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            
                            # Print the reason for the abnormality
                            abnormal_reasons = []
                            if no_valid_tool_calls:
                                abnormal_reasons.append(f"no_valid_tool_calls(parsed={parsed_tool_calls_count})")
                            if _req.tool_parse_error:
                                abnormal_reasons.append(f"tool_parse_error({parse_error_reason})")
                            if _req.excessive_tool_calls_per_turn:
                                abnormal_reasons.append(f"excessive_queries({current_turn_query_count}>{max_queries_per_turn})")
                            if _req.repeated_query:
                                abnormal_reasons.append(f"repeated_query({len(repeated_queries)} duplicates)")
                            
                            logger.warning(
                                f"[Abnormal Break] Request {_req.request_id} turn {current_turns}: "
                                f"Breaking with zero reward due to: {', '.join(abnormal_reasons)}"
                            )
                            print(
                                f"[DEBUG] FinishReasonTypeEnum.STOP (Abnormal Break): {', '.join(abnormal_reasons)}"
                            )
                            break
                        
                        # Reaching here means there is a valid tool call and no abnormality
                        if len(parsed_tool_calls) > 0:
                            # ===  There is a valid tool call ===
                            # 
                            # Append the assistant message (containing tool_calls) to the conversation history
                            # Token sequence change:
                            # before: [..., tool(obs_n-1)] or [sys, user]
                            # after:  [..., assistant(think+tool_call)]
                            # 
                            # State of the three:
                            # - prompt_ids: unchanged
                            # - input_ids: grows, the assistant(think+tool_call) tokens are appended
                            # - response_ids: still None
                            _req.add_assistant_message(
                                self.processing_class, normed_content, tool_calls=parsed_tool_calls
                            )
                        else:
                            # Exception handling 4: no valid tool call
                            # 
                            # Trigger condition:
                            # 1. the model emitted an empty tool call
                            #    example: "<tool_call></tool_call>"
                            # 2. all tool calls were filtered out (has_decode_error = True)
                            #    example: all 3 tool calls have argument errors and were discarded
                            # 3. parse_non_stream() returned an empty list (after a JSONDecodeError)
                            # 
                            # How it is handled:
                            # - append the original content to messages as plain text (without tool_calls)
                            # - set finish_reason = STOP
                            # - set state = COMPLETED
                            # - break out of the loop and stop generating
                            # 
                            # Token sequence change:
                            # before: [..., tool(obs_n-1)]
                            # after:  [..., assistant(content)]  # as plain text
                            # 
                            # Consequences:
                            # - this round's generation is treated as the final answer
                            # - even if the content contains a malformed tool call, it is kept in the conversation
                            # - the model cannot call tools anymore
                            # 
                            # Logging:
                            # - print a DEBUG message: "FinishReasonTypeEnum.STOP no tool_calls"
                            # - event="turn_end"
                            # - extra={"finish_reason": "STOP", "has_tool_calls": False}
                            _req.add_assistant_message(self.processing_class, content)
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            print(f"[DEBUG] FinishReasonTypeEnum.STOP no tool_calls")
                            break
                    else:
                        # === Append the assistant message (final answer, no tool_calls) ===
                        # This is the last generation round; the assistant gives the final answer
                        # Token sequence change:
                        # before: [..., tool(obs_n)]
                        # after:  [..., tool(obs_n), assistant(final_answer)]
                        # 
                        # State of the three:
                        # - prompt_ids: unchanged
                        # - input_ids: grows, the assistant(final_answer) tokens are appended
                        # - response_ids: still None (computed at finalize time)
                        _req.add_assistant_message(
                            self.processing_class,
                            content,
                        )
                        if (
                            _req.interaction_kwargs
                            and self.interaction_map
                            and user_turns < self.config.multi_turn.max_user_turns
                            and current_turns < self.config.multi_turn.max_assistant_turns
                        ):
                            _req.state = AsyncRolloutRequestStateEnum.INTERACTING
                        else:
                            break


                torch.cuda.synchronize()
                turn_post_process_end_time = time.time()
                turn_post_process_duration = turn_post_process_end_time - turn_post_process_start_time
                self.log_manager.log(
                    log_path,
                    event="turn_post_process",
                    duration=turn_post_process_duration,
                    extra={
                        "request_id": _req.request_id,
                        "turn": current_turns,
                        "has_tool_calls": has_tool_calls,
                        "parsed_tool_calls": parsed_tool_calls_count,
                        "finish_reason": str(finish_reason_type),
                        "tool_parsing_duration": tool_parsing_duration
                    },
                    workid=self._rank,
                    step=self.step
                )

                # === TURN END ===
                current_turns += 1
                torch.cuda.synchronize()
                turn_end_time = time.time()
                turn_total_duration = turn_end_time - turn_start_time

                # Record turn timing details
                turn_timing = {
                    "turn": current_turns - 1,
                    "total_duration": turn_total_duration,
                    "pre_process_duration": turn_pre_process_duration,
                    "engine_call_duration": turn_engine_call_duration,
                    "post_process_duration": turn_post_process_duration,
                    "tool_parsing_duration": tool_parsing_duration,
                    "has_tool_calls": has_tool_calls,
                    "finish_reason": str(finish_reason_type)
                }
                turn_timings.append(turn_timing)

                self.log_manager.log(
                    log_path,
                    event="turn_end",
                    duration=turn_total_duration,
                    extra={
                        "request_id": _req.request_id,
                        "turn": current_turns - 1,
                        "pre_process_duration": turn_pre_process_duration,
                        "engine_call_duration": turn_engine_call_duration,
                        "post_process_duration": turn_post_process_duration,
                        "tool_parsing_duration": tool_parsing_duration,
                        "has_tool_calls": has_tool_calls,
                        "finish_reason": str(finish_reason_type),
                        "engine_call_pct": round(turn_engine_call_duration / turn_total_duration * 100, 2),
                        "post_process_pct": round(turn_post_process_duration / turn_total_duration * 100, 2)
                    },
                    workid=self._rank,
                    step=self.step
                )




            elif _req.state == AsyncRolloutRequestStateEnum.INTERACTING:
                torch.cuda.synchronize()
                interacting_start_time = time.time()
                user_turns += 1
                messages = [{"role": x.role, "content": x.content} for x in _req.messages]

                # Get interaction by name from interaction_kwargs
                interaction_name = _req.interaction_kwargs.get(
                    "name", "gsm8k"
                )  # Default to gsm8k for backward compatibility
                if interaction_name not in self.interaction_map:
                    raise ValueError(
                        f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                        f"{list(self.interaction_map.keys())}"
                    )

                interaction = self.interaction_map[interaction_name]
                torch.cuda.synchronize()
                interaction_response_start_time = time.time()
                should_terminate_sequence, content, reward, metrics = await interaction.generate_response(
                    _req.request_id, messages, **_req.interaction_kwargs
                )
                torch.cuda.synchronize()
                interaction_response_end_time = time.time()
                self.log_manager.log(
                    log_path,
                    event="interaction_response",
                    duration=interaction_response_end_time - interaction_response_start_time,
                    extra={
                        "request_id": _req.request_id,
                        "user_turn": user_turns,
                        "assistant_turn": current_turns
                    },
                    workid=self._rank,
                    step=self.step
                )
                user_turn_rewards.append(reward)
                if should_terminate_sequence:
                    finish_reason_type = FinishReasonTypeEnum.STOP
                    _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                    break
                else:
                    _req.add_user_message(self.processing_class, content)
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    else:
                        _req.state = AsyncRolloutRequestStateEnum.RUNNING
                torch.cuda.synchronize()
                interacting_end_time = time.time()
                self.log_manager.log(
                    log_path,
                    event="interacting_state",
                    duration=interacting_end_time - interacting_start_time,
                    extra={
                        "request_id": _req.request_id,
                        "user_turn": user_turns,
                        "assistant_turn": current_turns
                    },
                    workid=self._rank,
                    step=self.step
                )
        torch.cuda.synchronize()
        main_loop_end_time = time.time()

        self.log_manager.log(
            log_path,
            event="main_loop",
            duration=main_loop_end_time - main_loop_start_time,
            extra={"request_id": _req.request_id},
            workid=self._rank,
            step=self.step
        )

        # ========================================
        # === Max-turns exceeded check (after the loop) ===
        # ========================================
        # 
        # After the main loop ends, check whether it exited because the max turns were reached
        # 
        # Trigger condition: current_turns >= self.config.multi_turn.max_assistant_turns
        # 
        # Default config:
        # - max_assistant_turns = max_model_len // 3 (if not configured)
        # - e.g.: max_model_len=8000 -> max_assistant_turns=2666
        # 
        # Loop mechanism:
        # - loop condition: while current_turns < max_assistant_turns
        # - after each generation: current_turns += 1
        # - when current_turns >= max_assistant_turns, the loop exits naturally
        # 
        # How it is handled:
        # - set finish_reason = STOP
        # - set the exceed_max_turns flag
        # - continue with the subsequent finalize flow (a 0 reward is given later)
        # 
        # Logging:
        # - print a DEBUG message: "FinishReasonTypeEnum.STOP current_turns={} >= max_assistant_turns={}"
        # - event="async_rollout_request_complete"
        # - extra={"finish_reason": "STOP", "turns": current_turns, "exceed_max_turns": True}
        if _req.total_tool_call_attempts > 100:  # hard-coded max allowed tool-call attempts: 100
            finish_reason_type = FinishReasonTypeEnum.STOP
            _req.exceed_max_turns = True
            logger.warning(
                f"[Abnormal] Request {_req.request_id}: "
                f"Exceeded max tool call attempts {_req.total_tool_call_attempts} > 100"
            )
            print(
                f"[DEBUG] FinishReasonTypeEnum.STOP (Exceed Max Tool Calls) "
                f"total_tool_call_attempts={_req.total_tool_call_attempts} > 100"
            )
        
        # ========================================
        # === Check whether the token count exceeds the limit ===
        # ========================================
        # Check whether the current sequence length exceeds the max limit
        # If so, mark exceed_max_tokens=True
        # These samples:
        # - are still used to compute the within-group advantage (reducing statistical bias)
        # - but don't participate in the loss / gradient backward (handled later by the Trainer)
        current_seq_len = len(_req.input_ids.squeeze(0)) if _req.input_ids is not None else 0
        if current_seq_len > self.config.max_model_len:
            _req.exceed_max_tokens = True
            logger.warning(
                f"[Abnormal] Request {_req.request_id}: "
                f"Exceeded max tokens {current_seq_len} > {self.config.max_model_len}"
            )
            print(
                f"[DEBUG] exceed_max_tokens: current_seq_len={current_seq_len} > "
                f"max_model_len={self.config.max_model_len}"
            )

        # Calculate the reward for each tool
        torch.cuda.synchronize()
        reward_calculation_start_time = time.time()
        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        tool_reward_tasks = []
        for name in _req.tools_kwargs.keys():
            tool = self._tool_map[name]
            tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
        tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = dict(tool_reward_scores)
        all_rewards = {**tool_reward_scores, **{"user_turn_rewards": user_turn_rewards}}
        torch.cuda.synchronize()
        reward_calculation_end_time = time.time()
        self.log_manager.log(
            log_path,
            event="reward_calculation",
            duration=reward_calculation_end_time - reward_calculation_start_time,
            extra={"request_id": _req.request_id},
            workid=self._rank,
            step=self.step
        )

        torch.cuda.synchronize()
        finalization_start_time = time.time()
        
        # Save finish_reason_type onto the request object
        _req.finish_reason_type = finish_reason_type
        
        # === Finalize the request: compute response_ids ===
        # Calling finalize() will:
        # 1. set state = COMPLETED
        # 2. save reward_scores and finish_reason_type
        # 3. **compute response_ids = input_ids[:, prompt_ids.shape[-1]:]**
        # 4. run a tokenization sanity check (comparing full tokenization against
        #    incremental tokenization)
        # 
        # Final state of the three (complete multi-turn tool-use example):
        # 
        # prompt_ids: [1, 50]
        #   content: [sys, user, <im_start>assistant\n]
        #   note: the original prompt, always unchanged
        # 
        # input_ids: [1, 800]
        #   content: [sys, user, 
        #          assistant(think1), assistant(tool_call1), tool(obs1),
        #          assistant(think2), assistant(tool_call2), tool(obs2),
        #          assistant(think3), assistant(final_answer)]
        #   note: the complete conversation sequence, including all rounds
        # 
        # response_ids: [1, 750]  
        #   content: [assistant(think1), assistant(tool_call1), tool(obs1),
        #          assistant(think2), assistant(tool_call2), tool(obs2),
        #          assistant(think3), assistant(final_answer)]
        #   note: response_ids = input_ids[:, 50:] (sliced from the end of the prompt)
        #         contains all of the agent's reasoning, tool calls, and observations
        # 
        # The corresponding loss_mask:
        # - prompt_loss_mask: [1, 50], all zeros (no loss)
        # - response_loss_mask: [1, 750]
        #     of which: tokens generated by the assistant are 1 (loss computed)
        #              tokens from the tool observation are 0 (no loss)
        #   example: [1,1,1,...,1, 0,0,0,...,0, 1,1,1,...,1, 0,0,0,...,0, 1,1,1,...,1]
        #         ^assistant1   ^tool1       ^assistant2   ^tool2       ^assistant3
        _req.finalize(self.processing_class, all_rewards, finish_reason_type)
        torch.cuda.synchronize()
        finalization_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="finalization",
                duration=finalization_end_time - finalization_start_time,
                extra={"request_id": _req.request_id},
                workid=self._rank,
                step=self.step
            )

        torch.cuda.synchronize()
        request_end_time = time.time()
        total_request_time = request_end_time - request_start_time
        if self.config.calculate_log_probs:
            debug_sampling_params = {**self.sampling_params}
            debug_sampling_params["max_new_tokens"] = 0
            output = await self._engine.async_generate(
                prompt=None,
                input_ids=_req.input_ids,
                sampling_params=debug_sampling_params,
                return_logprob=True,
                logprob_start_len=0,
            )
            # len(input_token_logprobs) = len(input_tokens)-1，because logprob of 1st token is None
            _req.output_token_ids, _req.rollout_log_probs = _extract_logprob_from_output(output)


        response_length = len(_req.response_ids.squeeze(0)) if _req.response_ids is not None else 0
        actual_response_tokens = torch.sum(_req.response_loss_mask.squeeze(0)).item() if _req.response_loss_mask is not None else 0

        # Calculate turn statistics
        if turn_timings:
            avg_turn_duration = np.mean([t["total_duration"] for t in turn_timings])
            avg_engine_call_duration = np.mean([t["engine_call_duration"] for t in turn_timings])
            avg_post_process_duration = np.mean([t["post_process_duration"] for t in turn_timings])
            total_engine_time = sum([t["engine_call_duration"] for t in turn_timings])
            total_post_process_time = sum([t["post_process_duration"] for t in turn_timings])
            turns_with_tools = sum([1 for t in turn_timings if t["has_tool_calls"]])
        else:
            avg_turn_duration = 0
            avg_engine_call_duration = 0
            avg_post_process_duration = 0
            total_engine_time = 0
            total_post_process_time = 0
            turns_with_tools = 0
        
        # ========================================
        # === Compute behavior-metric statistics ===
        # ========================================
        # Search-step statistics (number of assistant messages)
        search_steps = _req.total_assistant_turns
        
        # Search-call statistics
        total_tool_calls = _req.total_tool_calls
        
        # Statistics of tool calls per round
        if _req.tool_calls_per_turn:
            avg_tool_calls_per_turn = np.mean(_req.tool_calls_per_turn)
            median_tool_calls_per_turn = np.median(_req.tool_calls_per_turn)
            max_tool_calls_per_turn = np.max(_req.tool_calls_per_turn)
        else:
            avg_tool_calls_per_turn = 0
            median_tool_calls_per_turn = 0
            max_tool_calls_per_turn = 0

        self.log_manager.log(
            log_path,
            event="async_rollout_request_complete",
            duration=total_request_time,
            extra={
                "request_id": _req.request_id,
                "batch_data_id": _req.batch_data_id,
                "finish_reason": str(finish_reason_type),
                "turns": current_turns,
                "user_turns": user_turns,
                "turns_with_tools": turns_with_tools,
                "response_length": response_length,
                "actual_response_tokens": actual_response_tokens,
                "total_sequence_length": len(_req.input_ids.squeeze(0)) if _req.input_ids is not None else 0,
                "avg_turn_duration": round(avg_turn_duration, 4),
                "avg_engine_call_duration": round(avg_engine_call_duration, 4),
                "avg_post_process_duration": round(avg_post_process_duration, 4),
                "total_engine_time": round(total_engine_time, 4),
                "total_post_process_time": round(total_post_process_time, 4),
                "engine_time_pct": round(total_engine_time / total_request_time * 100, 2) if total_request_time > 0 else 0,
                "post_process_time_pct": round(total_post_process_time / total_request_time * 100, 2) if total_request_time > 0 else 0,
                "turn_timings": turn_timings,
                # ========================================
                # === Abnormality flag fields ===
                # ========================================
                "excessive_tool_calls_per_turn": _req.excessive_tool_calls_per_turn,
                "tool_parse_error": _req.tool_parse_error,
                "repeated_query": _req.repeated_query,
                "search_error": _req.search_error,
                "exceed_max_turns": _req.exceed_max_turns,
                "exceed_max_tokens": _req.exceed_max_tokens,
                # ========================================
                # === Behavior-metric statistics ===
                # ========================================
                "search_steps": search_steps,  # total number of assistant messages
                "total_tool_calls": total_tool_calls,  # total number of tool calls
                "avg_tool_calls_per_turn": round(avg_tool_calls_per_turn, 2),
                "median_tool_calls_per_turn": median_tool_calls_per_turn,
                "max_tool_calls_in_single_turn": _req.max_tool_calls_in_single_turn,
                "tool_calls_per_turn": _req.tool_calls_per_turn,  # list of tool calls per round
                "query_history_length": len(_req.query_history),  # number of queries in the history
            },
            workid=self._rank,
            step=self.step
        )
        return _req

    async def _handle_engine_call(
        self, _req: AsyncRolloutRequest, sampling_params: dict, image_data: Optional[list[Any]] = None
    ) -> dict:
        generation_prompt_ids = _req.get_generation_prompt_ids(self.processing_class)
        return await self._handle_engine_generate(generation_prompt_ids, sampling_params, image_data)

    async def _handle_engine_generate(
        self, generation_prompt_ids: list[int], sampling_params: dict, image_data: Optional[list[Any]] = None, request_id: Optional[str] = None, max_new_tokens_override: int | None = None, max_model_len_override: int | None = None
    ) -> dict:
        torch.cuda.synchronize()
        setup_start_time = time.time()
        response_length = max_new_tokens_override if max_new_tokens_override is not None else self.config.response_length
        effective_max_model_len = max_model_len_override if max_model_len_override is not None else self.config.max_model_len
        max_new_tokens = min(response_length, effective_max_model_len - len(generation_prompt_ids) - 1)
        # Fallback: avoids a negative max_new_tokens when the prompt is too long,
        # which would make SGLang error out
        if max_new_tokens < 0:
            logger.warning(
                f"[Engine Generate] Prompt too long: {len(generation_prompt_ids)} tokens, "
                f"max_model_len={self.config.max_model_len}, computed max_new_tokens={max_new_tokens}. "
                f"Clamping to 1."
            )
            max_new_tokens = 1

        kwargs = sampling_params.copy()
        kwargs["max_new_tokens"] = max_new_tokens
        kwargs["n"] = 1  # group size is supported in preprocess
        torch.cuda.synchronize()
        setup_end_time = time.time()
        log_path = os.path.join(
            self.log_dir,
            f"step_{self.step}",
            f"worker_{self._rank}.jsonl"
        )
        self.log_manager.log(
            log_path,
            event="engine_generate_setup",
            duration=setup_end_time - setup_start_time,
            extra={"request_id": request_id} if request_id else None,
            workid=self._rank,
            step=self.step
        )
        torch.cuda.synchronize()
        engine_call_start_time = time.time()
        output = await self._engine.async_generate(
            input_ids=generation_prompt_ids,
            sampling_params=kwargs,
            return_logprob=False,
            image_data=image_data,
        )
        torch.cuda.synchronize()
        engine_call_end_time = time.time()
        self.log_manager.log(
            log_path,
            event="engine_async_generate_actual",
            duration=engine_call_end_time - engine_call_start_time,
            extra={"request_id": request_id} if request_id else None,
            workid=self._rank,
            step=self.step
        )
        return output

    async def _handle_pending_state(self, _req: AsyncRolloutRequest) -> AsyncRolloutRequest:
        log_path = os.path.join(
            self.log_dir,
            f"step_{self.step}",
            f"worker_{self._rank}.jsonl"
        )
        if _req.tool_schemas is not None:
            torch.cuda.synchronize()
            tool_creation_start_time = time.time()
            tool_creation_coroutines = []
            for tool_schema in _req.tool_schemas:
                tool = self._tool_map[tool_schema.function.name]
                create_kwargs = _req.tools_kwargs[tool.name].get("create_kwargs", {})
                tool_creation_coroutines.append(tool.create(_req.request_id, **create_kwargs))
            tool_creation_results = await asyncio.gather(*tool_creation_coroutines)
            _req.add_tool_response_messages(
                self.processing_class, [tool_result for _, tool_result in tool_creation_results]
            )
            torch.cuda.synchronize()
            tool_creation_end_time = time.time()
            self.log_manager.log(
                log_path,
                event="tool_creation_pending_state",
                duration=tool_creation_end_time - tool_creation_start_time,
                extra={"request_id": _req.request_id},
                workid=self._rank,
                step=self.step
            )
        if _req.interaction_kwargs and self.interaction_map:
            torch.cuda.synchronize()
            interaction_start_time = time.time()
            interaction_kwargs = _req.interaction_kwargs
            # Get interaction by name from interaction_kwargs
            interaction_name = interaction_kwargs.get("name", "gsm8k")  # Default to gsm8k for backward compatibility
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )

            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(_req.request_id, **interaction_kwargs)
            torch.cuda.synchronize()
            interaction_end_time = time.time()
            self.log_manager.log(
                log_path,
                event="interaction_start_pending_state",
                duration=interaction_end_time - interaction_start_time,
                extra={"request_id": _req.request_id},
                workid=self._rank,
                step=self.step
            )

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _req_level_generate_sequences(self, prompts: DataProto, step: int = 0, **kwargs) -> DataProto:
        """Generates multi-turn sequences for a batch of prompts.
        For multi-turn generation, each prompt is processed separately via
        `_req_level_generate_sequences` for better tool calling control.
        Note that in multi-turn generation, we repeat the prompts for rollout.n times in ray_trainer.
        Thus we do not need to repeat the prompts here and set the sampling parameter n to 1.
        
        Args:
            prompts: DataProto containing input prompts
            step: Global training step (passed from generate_sequences)
            **kwargs: Additional arguments
        """
        # Save step into self.step for use by later methods
        self.step = step
        
        # Generate the log file name (unique per step)
        log_path = None
        if self._tp_rank == 0:
            log_path = os.path.join(
                self.log_dir,
                f"step_{step}",
                f"worker_{self._rank}.jsonl"
            )

        # Async rollout with tools support
        torch.cuda.synchronize()
        start_time = time.time()
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device

        if self._tp_rank == 0:
            torch.cuda.synchronize()
            preprocess_start_time = time.time()
            req_list = self._preprocess_prompt_to_async_rollout_requests(
                prompts,
            )
            torch.cuda.synchronize()
            preprocess_end_time = time.time()
            self.log_manager.log(
                log_path,
                event="preprocessing_duration",
                duration=preprocess_end_time - preprocess_start_time,
                workid=self._rank,
                step=self.step
            )
            # distinguish training and validation
            if is_validate:
                # Validation mode: process all requests without abort
                loop = asyncio.get_event_loop()
                output_req_list = loop.run_until_complete(
                    asyncio.gather(
                        *[self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) for req in req_list],
                    )
                )
            else:
                # add progress monitoring and abort function
                total_requests = len(req_list)
                target_completion = int(total_requests * (1 - self.config.over_sample_rate))
                # abort when target_completion of requests are completed

                completed_count = 0
                aborted_requests = []
                all_tasks = []

                async def rollout_a_request_with_cancellation_handler(req):
                    try:
                        result = await self._async_rollout_a_request(req, do_sample, is_validate, **kwargs)
                        # print(f"[DEBUG] _async_rollout_a_request result: {result}")

                        return result
                    except asyncio.CancelledError:
                        # request is cancelled, return padding
                        logger.info(f"Request {req.request_id} was cancelled, creating padding")
                        aborted_requests.append(req.request_id)
                        return self._create_padding_request(req)

                async def run_with_cancellation():
                    nonlocal all_tasks
                    nonlocal completed_count
                    all_tasks = [
                        asyncio.create_task(rollout_a_request_with_cancellation_handler(req)) for req in req_list
                    ]

                    # Wait for target_completion tasks to complete
                    try:
                        for completed_task in asyncio.as_completed(all_tasks):
                            await completed_task
                            completed_count += 1
                            if completed_count >= target_completion:
                                break
                    finally:
                        # Cancel remaining tasks
                        for t in all_tasks:
                            if not t.done():
                                t.cancel()

                        # Wait for all tasks to finish (including cancelled ones)
                        final_results = await asyncio.gather(*all_tasks, return_exceptions=True)
                        # Abort all requests in SGLang engine
                        await self._engine.abort_request(abort_all=True)
                    return final_results

                torch.cuda.synchronize()
                async_generate_start_time = time.time()
                loop = asyncio.get_event_loop()
                output_req_list = loop.run_until_complete(run_with_cancellation())
                # print(f"[DEBUG]output_req_list: {output_req_list}")
                torch.cuda.synchronize()
                async_generate_end_time = time.time()
                self.log_manager.log(
                    log_path,
                    event="async_generate_duration",
                    duration=async_generate_end_time - async_generate_start_time,
                    workid=self._rank,
                    step=self.step
                )
            torch.cuda.synchronize()
            sort_start_time = time.time()
            sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))
            torch.cuda.synchronize()
            sort_end_time = time.time()
            self.log_manager.log(
                log_path,
                event="sorting_duration",
                duration=sort_end_time - sort_start_time,
                workid=self._rank,
                step=self.step
            )
        else:
            sorted_output_req_list = None

        torch.cuda.synchronize()
        barrier_start_time = time.time()
        dist.barrier()
        torch.cuda.synchronize()
        barrier_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="barrier_wait_duration",
                duration=barrier_end_time - barrier_start_time,
                workid=self._rank,
                step=self.step
            )

        torch.cuda.synchronize()
        broadcast_start_time = time.time()
        [sorted_output_req_list] = broadcast_pyobj(
            data=[sorted_output_req_list],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )
        torch.cuda.synchronize()
        broadcast_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="broadcast_duration",
                duration=broadcast_end_time - broadcast_start_time,
                workid=self._rank,
                step=self.step
            )
        # ========================================
        # === Step 1: extract data from AsyncRolloutRequest ===
        # ========================================
        # 
        # After _async_rollout_a_request finishes, each req has been finalized() and contains:
        # - prompt_ids/response_ids: the token sequences
        # - attention_mask/position_ids/loss_mask: the corresponding masks and position info
        # - messages: the full conversation history (system, user, assistant, tool messages)
        # - reward_scores: the reward scores of each tool
        # - multi_modal_inputs: multi-modal inputs (if any)
        # 
        # Now we need to extract this information from all reqs and build a batched DataProto
        
        torch.cuda.synchronize()
        postprocess_start_time = time.time()

        # === Initialize the lists used to collect the data ===
        # These lists collect data from each AsyncRolloutRequest
        prompt_ids, response_ids = [], []  # list[Tensor[seq_len]]: prompt and response token IDs of each req
        prompt_attention_mask, response_attention_mask = [], []  # list[Tensor[seq_len]]: attention masks
        prompt_position_ids, response_position_ids = [], []  # list[Tensor[seq_len]]: position IDs
        prompt_loss_mask, response_loss_mask = [], []  # list[Tensor[seq_len]]: loss masks (which tokens get a loss)
        messages = []  # list[dict]: conversation history, each element is {"messages": [...]}
        reward_scores = []  # list[dict]: reward scores, each element is {"tool_name": [score1, score2, ...], ...}
        multi_modal_inputs = []  # list[dict]: multi-modal inputs (images, videos, etc.)
        request_ids = []  # list[str]: request IDs
        if self.config.calculate_log_probs:
            output_logprobs = []  # list[Tensor]: log probabilities
            rollout_output_token_ids = []  # list[Tensor]: output token IDs from the rollout
        
        # ========================================
        # === Lists for collecting abnormality flags and metrics ===
        # ========================================
        # Used to collect each request's abnormality flags and statistics
        abnormal_flags_list = []  # list[dict]: each request's abnormality flags
        metrics_list = []  # list[dict]: each request's statistics

        # === Iterate over all completed requests and extract the data ===
        for req in sorted_output_req_list:
            # === Validate the request state ===
            # Every req must be in the COMPLETED state, and all sequence lengths must match
            assert req.state == AsyncRolloutRequestStateEnum.COMPLETED, f"Request {req.request_id} is not completed"
            assert (
                req.input_ids.shape[-1]
                == req.attention_mask.shape[-1]
                == req.position_ids.shape[-1]
                == req.loss_mask.shape[-1]
            ), f"""Request {req.request_id} has different length of
                {req.input_ids.shape[-1]=}, {req.attention_mask.shape[-1]=},
                {req.position_ids.shape[-1]=}, {req.loss_mask.shape[-1]=}"""
            error_message_lines = [
                f"""Request {req.request_id} has input_ids length {req.input_ids.shape[-1]}
                    greater than max_model_len {self.config.max_model_len}""",
                f"Decoded input_ids: {self.processing_class.decode(req.input_ids.squeeze(0))}",
                f"Decoded prompt_ids: {self.processing_class.decode(req.prompt_ids.squeeze(0))}",
                f"Decoded response_ids: {self.processing_class.decode(req.response_ids.squeeze(0))}",
                f"Messages: {req.messages}",
                f"Max model length: {req.max_model_len}",
            ]
            error_message = "\n".join(error_message_lines)
            assert req.input_ids.shape[-1] <= self.config.max_model_len, error_message

            # === Extract the token sequences ===
            # req.prompt_ids: [1, prompt_len] - the original prompt (fixed)
            # req.response_ids: [1, response_len] - all generated content (assistant + tool observations)
            #   computed in finalize(): response_ids = input_ids[:, prompt_ids.shape[-1]:]
            # 
            # Example (multi-turn tool use):
            # prompt_ids:   [sys, user, <im_start>assistant\n]  # length 50
            # response_ids: [assistant(think1), tool_call1, tool(obs1), 
            #                assistant(think2), tool_call2, tool(obs2),
            #                assistant(final_answer)]  # length 750
            # 
            # squeeze(0) removes the batch dim, turning [1, seq_len] into [seq_len]
            prompt_ids.append(req.prompt_ids.to(tgt_device).squeeze(0))
            response_ids.append(req.response_ids.to(tgt_device).squeeze(0))


            # debug_message_lines = [
            #     f"[DEBUG]0input_ids: {req.input_ids[0].tolist()}",
            #     f"[DEBUG]0prompt_ids: {req.prompt_ids[0].tolist()}",
            #     f"[DEBUG]0response_ids: {req.response_ids[0].tolist()}",
            #     f"[DEBUG]0Decoded input_ids: {self.processing_class.decode(req.input_ids[0])}",
            #     f"[DEBUG]0Decoded prompt_ids: {self.processing_class.decode(req.prompt_ids[0])}",
            #     f"[DEBUG]0Decoded response_ids: {self.processing_class.decode(req.response_ids[0])}",
            # ]
            # debug_message = "\n".join(debug_message_lines)
            # logger.debug(f"logger[DEBUG]0debug_message: {debug_message}")
            # print(f"print[DEBUG]0debug_message: {debug_message}")


            # === Check the length ===
            # Check whether response_ids exceeds max_response_len
            # If so, the exceed_max_tokens flag must be set, since the response will be
            # truncated in truncate_output_ids
            if req.response_ids.shape[-1] > self.config.response_length:
                # Important fix (2026-01-01):
                # previously this only printed a warning without setting the abnormality flag
                # but response_ids was already truncated to max_response_len in truncate_output_ids
                # this makes the saved rollout_data's output incomplete (e.g. missing the <answer> part)
                # 
                # Fix: set the exceed_max_tokens flag so the reward manager knows this is an
                # abnormal sample
                req.exceed_max_tokens = True
                logger.warning(
                    f"""{req.request_id=} has response_ids length {req.response_ids.shape[-1]}
                    greater than max_response_len {self.config.response_length}, marking as exceed_max_tokens"""
                )
            
            # === Extract the attention masks ===
            # attention_mask: marks which positions are real tokens (1) and which are padding (0)
            # - prompt_attention_mask: [prompt_len] - the mask of the prompt part
            # - response_attention_mask: [response_len] - the mask of the response part
            prompt_attention_mask.append(req.prompt_attention_mask.to(tgt_device).squeeze(0))
            response_attention_mask.append(req.response_attention_mask.to(tgt_device).squeeze(0))
            
            # === Extract the position IDs ===
            # position_ids: the position index of each token (used for positional encoding)
            # - prompt_position_ids: [prompt_len] - the position indices of the prompt part
            # - response_position_ids: [response_len] - the position indices of the response part
            prompt_position_ids.append(req.prompt_position_ids.to(tgt_device).squeeze(0))
            response_position_ids.append(req.response_position_ids.to(tgt_device).squeeze(0))
            
            # === Extract the loss masks ===
            # loss_mask: marks which positions get a loss (1) and which don't (0)
            # - prompt_loss_mask: [prompt_len] - usually all zeros (the prompt gets no loss)
            # - response_loss_mask: [response_len] - assistant-generated tokens are 1, tool-observation tokens are 0
            # 
            # Example (multi-turn tool use):
            # response_loss_mask: [1,1,1,...,1, 0,0,0,...,0, 1,1,1,...,1, 0,0,0,...,0, 1,1,1,...,1]
            #                      ^assistant1   ^tool1       ^assistant2   ^tool2       ^assistant3
            # 
            # This ensures the loss is computed only over model-generated parts, not over the
            # observations returned by tools
            prompt_loss_mask.append(req.prompt_loss_mask.to(tgt_device).squeeze(0))
            response_loss_mask.append(req.response_loss_mask.to(tgt_device).squeeze(0))
            
            # === Extract the non-tensor data ===
            # messages: list[Message] - the full conversation history
            #   example: [
            #     {"role": "system", "content": "..."},
            #     {"role": "user", "content": "..."},
            #     {"role": "assistant", "content": "...", "tool_calls": [...]},
            #     {"role": "tool", "content": "observation 1"},
            #     {"role": "assistant", "content": "...", "tool_calls": [...]},
            #     {"role": "tool", "content": "observation 2"},
            #     {"role": "assistant", "content": "final answer"}
            #   ]
            messages.append({"messages": req.messages})
            
            # reward_scores: dict[str, list[float]] - the reward scores of each tool
            #   example: {"search_tool": [0.5, 0.3], "user_turn_rewards": [0.8]}
            reward_scores.append(req.reward_scores)
            
            # multi_modal_inputs: dict - multi-modal inputs (if any)
            multi_modal_inputs.append(req.multi_modal_inputs)
            
            # request_id: str - the unique identifier of the request
            request_ids.append(req.request_id)
            
            # === Extract the log probabilities (if needed) ===
            if self.config.calculate_log_probs:
                # output_logprobs: Tensor[response_len] - the log probability of each token in the response
                # rollout_output_token_ids: Tensor[response_len] - the token IDs generated during the rollout
                # 
                # Note: req.rollout_log_probs are the log_probs of the entire input_ids
                # here we only extract the response part (the last len(req.response_ids) entries)
                output_logprobs.append(req.rollout_log_probs[-len(req.response_ids) :])
                rollout_output_token_ids.append(req.output_token_ids[-len(req.response_ids) :])
            
            # ========================================
            # === Collect the abnormality flags ===
            # ========================================
            # Detect whether there was an unknown tool call (unknown_tool_names is non-empty)
            has_unknown_tool = len(req.unknown_tool_names) > 0
            
            abnormal_flags = {
                "excessive_tool_calls_per_turn": req.excessive_tool_calls_per_turn,
                "tool_parse_error": req.tool_parse_error,
                "repeated_query": req.repeated_query,
                "search_error": req.search_error,
                "exceed_max_turns": req.exceed_max_turns,
                "exceed_max_tokens": req.exceed_max_tokens,
                "unknown_tool_in_trajectory": has_unknown_tool,  # new: unknown-tool-call flag
            }
            abnormal_flags_list.append(abnormal_flags)
            
            # ========================================
            # === Collect trajectory-level features (for per-trajectory reward computation) ===
            # ========================================
            # 
            # These are the raw features of each trajectory; they are saved to DataProto
            # directly, without aggregation
            # Purpose:
            # - the reward manager can compute a different reward per trajectory from these features
            # - analysis tools can inspect each trajectory's details
            # 
            trajectory_features = {
                # === Core behavior features ===
                "search_steps": req.total_assistant_turns,  # number of search steps (how many assistant turns)
                "subq_calls": req.total_tool_calls,  # total sub-query calls (total length of all query_lists)
                "tool_call_attempts": req.total_tool_call_attempts,  # occurrences of the <tool_call> tag
                
                # === Abnormalities (boolean flags) ===
                # These fields are copied directly from abnormal_flags, but listed explicitly here
                # for convenience
                "excessive_tool_calls_per_turn": req.excessive_tool_calls_per_turn,
                "tool_parse_error": req.tool_parse_error,
                "repeated_query": req.repeated_query,
                "search_error": req.search_error,
                "exceed_max_turns": req.exceed_max_turns,
                "exceed_max_tokens": req.exceed_max_tokens,
                
                # === Detailed statistics (for in-depth analysis) ===
                "max_tool_calls_in_single_turn": req.max_tool_calls_in_single_turn,
                "tool_calls_per_turn": req.tool_calls_per_turn,  # list of query counts per round
                "query_history": req.query_history,  # history of all queries
                "unknown_tool_names": req.unknown_tool_names,  # list of unknown tool names
                "finish_reason_type": str(req.finish_reason_type) if req.finish_reason_type else None,
            }
            metrics_list.append(trajectory_features)

        # debug_message_lines = [
        #     f"[DEBUG]1Decoded prompt_ids: {self.processing_class.decode(prompt_ids[0])}",
        #     f"[DEBUG]1Decoded response_ids: {self.processing_class.decode(response_ids[0])}",
        # ]
        # debug_message = "\n".join(debug_message_lines)
        # print(f"[DEBUG]1debug_message: {debug_message}")

        torch.cuda.synchronize()
        data_extraction_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="data_extraction_duration",
                duration=data_extraction_end_time - postprocess_start_time,
                workid=self._rank,
                step=self.step
            )

        # ========================================
        # === Step 2: pad the sequences to a common length ===
        # ========================================
        # 
        # Since each req's prompt and response may have different lengths, they must be padded
        # to a common length
        # - prompt: left padding (pad on the left side)
        # - response: right padding (pad on the right side)
        # 
        # Why pad this way?
        # 1. the prompt uses left padding: since the LLM generates left-to-right, the real
        #    content should be aligned to the right
        # 2. the response uses right padding: since generation starts from the left, the real
        #    content should be aligned to the left
        
        torch.cuda.synchronize()
        padding_start_time = time.time()

        # === Padding Prompt IDs ===
        # input:  list[Tensor[prompt_len_i]] - each req's prompt length differs
        # output: Tensor[batch_size, max_prompt_len] - aligned to the longest prompt
        # 
        # pad_sequence finds the longest sequence and pads all the others to that length
        # padding_side="left" means pad_token_id is filled in on the left side
        # 
        # Example:
        # input:  [[1,2,3], [4,5,6,7,8]]
        # output: [[pad,pad,1,2,3], [4,5,6,7,8]]
        prompt_ids = pad_sequence(
            prompt_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        # If the padded length is still below the configured prompt_length, keep padding up to
        # the configured length
        if prompt_ids.shape[-1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True)
        
        # === Padding Response IDs ===
        # input:  list[Tensor[response_len_i]] - each req's response length differs
        # output: Tensor[batch_size, max_response_len] - aligned to the longest response
        # 
        # the default padding_side="right" fills pad_token_id on the right side
        # 
        # Example:
        # input:  [[10,11,12,13], [20,21]]
        # output: [[10,11,12,13], [20,21,pad,pad]]
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[-1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        # === Padding Attention Masks ===
        # attention_mask is filled with 0 (0 means a padding position, excluded from attention)
        # 
        # Example (prompt_attention_mask, left padding):
        # input:  [[1,1,1], [1,1,1,1,1]]
        # output: [[0,0,1,1,1], [1,1,1,1,1]]
        prompt_attention_mask = pad_sequence(
            prompt_attention_mask,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )
        if prompt_attention_mask.shape[-1] < self.config.prompt_length:
            prompt_attention_mask = pad_sequence_to_length(
                prompt_attention_mask, self.config.prompt_length, 0, left_pad=True
            )
        
        # Example (response_attention_mask, right padding):
        # input:  [[1,1,1,1], [1,1]]
        # output: [[1,1,1,1], [1,1,0,0]]
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[-1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)

        # padding prompt_position_ids
        if prompt_position_ids[0].dim() == 2:
            # if prompt_position_ids is a 2D tensor
            # e.g. from qwen2vl, prompt_position_ids.shape = (3, seq_len)
            transposed_prompt_position_ids = [p.transpose(0, 1) for p in prompt_position_ids]
            prompt_position_ids = pad_sequence(
                transposed_prompt_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
            prompt_position_ids = prompt_position_ids.transpose(1, 2)
        else:
            prompt_position_ids = pad_sequence(
                prompt_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
        if prompt_position_ids.shape[-1] < self.config.prompt_length:
            prompt_position_ids = pad_sequence_to_length(
                prompt_position_ids, self.config.prompt_length, 0, left_pad=True
            )

        # padding response_position_ids
        if response_position_ids[0].dim() == 2:
            # if response_position_ids is a 2D tensor
            # e.g. from qwen2vl, response_position_ids.shape = (3, seq_len)
            transposed_response_position_ids = [p.transpose(0, 1) for p in response_position_ids]
            response_position_ids = pad_sequence(
                transposed_response_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
            response_position_ids = response_position_ids.transpose(1, 2)
        else:
            response_position_ids = pad_sequence(response_position_ids, batch_first=True, padding_value=0)
        if response_position_ids.shape[-1] < self.config.response_length:
            response_position_ids = pad_sequence_to_length(response_position_ids, self.config.response_length, 0)

        # === Padding Loss Masks ===
        # loss_mask is filled with 0 (0 means no loss is computed at that position)
        # 
        # Key point: loss_mask decides which tokens participate in the loss during training
        # - prompt_loss_mask: usually all zeros (no loss on the prompt)
        # - response_loss_mask: assistant-generated tokens are 1, tool-observation tokens are 0
        # 
        # Example (response_loss_mask):
        # original: [1,1,1,1, 0,0,0,0, 1,1,1,1]  # assistant1, tool1, assistant2
        #        ^loss      ^no loss  ^loss
        # After padding: [1,1,1,1, 0,0,0,0, 1,1,1,1, 0,0,0,0]  # zeros padded on the right
        prompt_loss_mask = pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")
        if prompt_loss_mask.shape[1] < self.config.prompt_length:
            prompt_loss_mask = pad_sequence_to_length(prompt_loss_mask, self.config.prompt_length, 0, left_pad=True)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
        if self.config.calculate_log_probs:
            output_logprobs = pad_sequence(output_logprobs, padding_value=0.0, batch_first=True)
            output_logprobs = pad_sequence_to_length(
                output_logprobs, pad_token_id=0.0, max_seq_len=response_ids.shape[-1]
            ).to(tgt_device)
            rollout_output_token_ids = pad_sequence(
                rollout_output_token_ids, padding_value=self.pad_token_id, batch_first=True
            )
            rollout_output_token_ids = pad_sequence_to_length(
                rollout_output_token_ids, pad_token_id=self.pad_token_id, max_seq_len=response_ids.shape[-1]
            ).to(tgt_device)

        torch.cuda.synchronize()
        padding_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="padding_duration",
                duration=padding_end_time - padding_start_time,
                workid=self._rank,
                step=self.step
            )

        # ========================================
        # === Step 3: concatenate the prompt and the response ===
        # ========================================
        # 
        # Concatenate the prompt and the response into the complete sequence
        # 
        # Structure after concatenation:
        # input_ids: [batch_size, prompt_length + response_length]
        # = [pad, pad, ..., sys, user, gen_prompt, assistant1, tool1, assistant2, tool2, ..., pad, pad]
        #     ^left padding  ^prompt part        ^response part                                ^right padding

        # The corresponding attention_mask:
        # = [0, 0, ..., 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ..., 0, 0]
        #     ^left padding ^real tokens                        ^right padding

        # The corresponding loss_mask (passed through response_mask):
        # = [0, 0, ..., 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, ..., 0, 0]
        #     ^prompt: no loss         ^ass1  ^tool1 ^ass2     ^padding

        torch.cuda.synchronize()
        concatenation_start_time = time.time()

        # Concatenate along the last dimension (the sequence dimension)
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        torch.cuda.synchronize()
        concatenation_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="concatenation_duration",
                duration=concatenation_end_time - concatenation_start_time,
                workid=self._rank,
                step=self.step
            )
        # ========================================
        # === Step 4: build the TensorDict (batch) ===
        # ========================================
        # 
        # Pack all tensor data into a TensorDict; this is the content of DataProto.batch
        # 
        # TensorDict field reference:
        # - prompts: [batch_size, prompt_length] - the original prompt token IDs
        # - responses: [batch_size, response_length] - the generated response token IDs
        # - response_mask: [batch_size, response_length] - the response loss mask
        #     * positions with value 1: loss is computed (assistant-generated tokens)
        #     * positions with value 0: no loss (tool observations, padding)
        # - input_ids: [batch_size, prompt_length + response_length] - the complete sequence
        # - attention_mask: [batch_size, prompt_length + response_length] - attention mask
        # - position_ids: [batch_size, prompt_length + response_length] - position IDs
        # 
        # If calculate_log_probs is enabled, it also contains:
        # - rollout_log_probs: [batch_size, response_length] - the response log probabilities
        # - rollout_output_token_ids: [batch_size, response_length] - the token IDs from the rollout
        
        torch.cuda.synchronize()
        batch_construction_start_time = time.time()
        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [batch_size, prompt_length]
                "responses": response_ids,  # [batch_size, response_length]
                "response_mask": response_loss_mask,  # [batch_size, response_length] - key! marks which tokens get a loss
                "input_ids": input_ids,  # [batch_size, total_length] - the complete sequence
                "attention_mask": attention_mask,  # [batch_size, total_length]
                "position_ids": position_ids,  # [batch_size, total_length]
            },
            batch_size=len(sorted_output_req_list),
        )

        # debug_message_lines = [
        #     f"[DEBUG]2input_ids: {input_ids[0].tolist()}",
        #     f"[DEBUG]2Decoded input_ids: {self.processing_class.decode(input_ids[0])}",
        #     f"[DEBUG]2Decoded prompt_ids: {self.processing_class.decode(prompt_ids[0])}",
        #     f"[DEBUG]2Decoded response_ids: {self.processing_class.decode(response_ids[0])}",
        # ]
        # debug_message = "\n".join(debug_message_lines)
        # print(f"[DEBUG]2debug_message: {debug_message}")

        torch.cuda.synchronize()
        batch_construction_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="batch_construction_duration",
                duration=batch_construction_end_time - batch_construction_start_time,
                workid=self._rank,
                step=self.step
            )
        if self.config.calculate_log_probs:
            batch["rollout_log_probs"] = output_logprobs
            batch["rollout_output_token_ids"] = rollout_output_token_ids

        # free cache engine
        torch.cuda.synchronize()
        cache_flush_start_time = time.time()
        if self._engine is not None and self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())
        torch.cuda.synchronize()
        cache_flush_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="cache_flush_duration",
                duration=cache_flush_end_time - cache_flush_start_time,
                workid=self._rank,
                step=self.step
            )

        # ========================================
        # === Step 5: build non_tensor_batch ===
        # ========================================
        # 
        # non_tensor_batch holds data that can't be turned into tensors (dicts, lists, strings, etc.)
        # This data maps to DataProto.non_tensor_batch
        # 
        # non_tensor_batch field reference:
        # - messages: np.array[dict] - each sample's conversation history
        #     example: [{"messages": [
        #              {"role": "system", "content": "..."},
        #              {"role": "user", "content": "..."},
        #              {"role": "assistant", "content": "...", "tool_calls": [...]},
        #              {"role": "tool", "content": "observation 1"},
        #              ...
        #            ]}]
        # 
        # - reward_scores: np.array[dict] - each sample's reward scores
        #     example: [{"search_tool": [0.5, 0.3], "user_turn_rewards": [0.8]}]
        #     these scores are used by the subsequent reward computation and policy optimization
        # 
        # - request_id: np.array[str] - each sample's request ID
        #     used for tracking and debugging
        # 
        # - multi_modal_inputs (if any): np.array[dict] - multi-modal inputs
        #     contains non-text data such as images and videos
        
        torch.cuda.synchronize()
        final_construction_start_time = time.time()
        
        # ========================================
        # === Compute batch-level aggregate metrics (for monitoring and analysis) ===
        # ========================================
        # 
        # These are batch-level aggregate statistics: ratios, means, medians, etc.
        # Purpose:
        # - monitor behavior trends during model training
        # - spot abnormal batches early and adjust in time
        # 
        batch_size = len(abnormal_flags_list)
        
        # === 1. Compute the ratio of each type of abnormal trajectory ===
        # 
        # Abnormality categories:
        # - repeated_query: duplicate queries (the model is stuck in a loop)
        # - tool_parse_error: tool parsing failed (malformed JSON, etc.)
        # - exceed_max_turns: hit the max_steps limit (tool-call attempts > 100)
        # - exceed_max_tokens: exceeded the token limit and was truncated
        # - excessive_tool_calls_per_turn: a large number of sub-queries in a single round
        # - search_error: search error (an environment issue, e.g. a timeout)
        # 
        repeated_query_count = sum(1 for f in abnormal_flags_list if f["repeated_query"])
        tool_parse_error_count = sum(1 for f in abnormal_flags_list if f["tool_parse_error"])
        exceed_max_turns_count = sum(1 for f in abnormal_flags_list if f["exceed_max_turns"])
        exceed_max_tokens_count = sum(1 for f in abnormal_flags_list if f["exceed_max_tokens"])
        excessive_tool_calls_count = sum(1 for f in abnormal_flags_list if f["excessive_tool_calls_per_turn"])
        search_error_count = sum(1 for f in abnormal_flags_list if f["search_error"])
        
        # Count the trajectories that used an unknown tool (using the
        # unknown_tool_in_trajectory field in abnormal_flags)
        unknown_tool_count = sum(1 for f in abnormal_flags_list if f.get("unknown_tool_in_trajectory", False))
        
        # Count the normal samples (no abnormality flag at all, including
        # unknown_tool_in_trajectory)
        normal_count = sum(
            1 for f in abnormal_flags_list
            if not any([
                f["excessive_tool_calls_per_turn"],
                f["tool_parse_error"],
                f["repeated_query"],
                f["search_error"],
                f["exceed_max_turns"],
                f["exceed_max_tokens"],
                f.get("unknown_tool_in_trajectory", False),  # new: an unknown tool also counts as abnormal
            ])
        )
        
        # === 2. Compute the distribution of search steps ===
        all_search_steps = [m["search_steps"] for m in metrics_list]
        
        # === 3. Compute the distribution of sub-query call counts ===
        all_subq_calls = [m["subq_calls"] for m in metrics_list]
        
        # Collect all unknown tool names (deduplicated)
        all_unknown_tool_names = set()
        for m in metrics_list:
            all_unknown_tool_names.update(m["unknown_tool_names"])
        
        # Batch statistics summary (aggregate metrics only)
        batch_statistics = {
            # === Basic batch info ===
            "batch_size": batch_size,
            "normal_count": normal_count,
            "normal_rate": round(normal_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            # === Ratio of abnormal trajectories ===
            "repeated_query_count": repeated_query_count,
            "repeated_query_rate": round(repeated_query_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            "tool_parse_error_count": tool_parse_error_count,
            "tool_parse_error_rate": round(tool_parse_error_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            "exceed_max_turns_count": exceed_max_turns_count,
            "exceed_max_turns_rate": round(exceed_max_turns_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            "exceed_max_tokens_count": exceed_max_tokens_count,
            "exceed_max_tokens_rate": round(exceed_max_tokens_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            "excessive_tool_calls_count": excessive_tool_calls_count,
            "excessive_tool_calls_rate": round(excessive_tool_calls_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            "search_error_count": search_error_count,
            "search_error_rate": round(search_error_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            "unknown_tool_count": unknown_tool_count,
            "unknown_tool_rate": round(unknown_tool_count / batch_size, 4) if batch_size > 0 else 0.0,
            
            # === Search-step statistics (mean, max, median) ===
            "search_steps_avg": round(np.mean(all_search_steps), 2) if all_search_steps else 0.0,
            "search_steps_median": round(np.median(all_search_steps), 2) if all_search_steps else 0.0,
            "search_steps_max": int(np.max(all_search_steps)) if all_search_steps else 0,
            "search_steps_min": int(np.min(all_search_steps)) if all_search_steps else 0,
            
            # === Sub-query count statistics (mean, max, median) ===
            "subq_calls_avg": round(np.mean(all_subq_calls), 2) if all_subq_calls else 0.0,
            "subq_calls_median": round(np.median(all_subq_calls), 2) if all_subq_calls else 0.0,
            "subq_calls_max": int(np.max(all_subq_calls)) if all_subq_calls else 0,
            "subq_calls_min": int(np.min(all_subq_calls)) if all_subq_calls else 0,
            
            # === List of unknown tool names (the full deduplicated list) ===
            "unique_unknown_tool_names": list(all_unknown_tool_names) if all_unknown_tool_names else [],
            "unique_unknown_tool_count": len(all_unknown_tool_names),
        }
        
        # Record the batch statistics to the log
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="batch_statistics",
                extra=batch_statistics,
                workid=self._rank,
                step=self.step
            )
            
            # Also print to the console (for easy monitoring)
            logger.info(
                f"[Batch Statistics] Step {self.step}: "
                f"Size={batch_size}, Normal={normal_count}({batch_statistics['normal_rate']:.2%}), "
                f"Repeated={repeated_query_count}({batch_statistics['repeated_query_rate']:.2%}), "
                f"ParseErr={tool_parse_error_count}({batch_statistics['tool_parse_error_rate']:.2%}), "
                f"MaxTurns={exceed_max_turns_count}({batch_statistics['exceed_max_turns_rate']:.2%}), "
                f"MaxTokens={exceed_max_tokens_count}({batch_statistics['exceed_max_tokens_rate']:.2%}), "
                f"ExcessiveCalls={excessive_tool_calls_count}({batch_statistics['excessive_tool_calls_rate']:.2%})"
            )
            
            logger.info(
                f"[Batch Metrics] Step {self.step}: "
                f"SearchSteps(avg={batch_statistics['search_steps_avg']:.2f}, "
                f"median={batch_statistics['search_steps_median']:.2f}, "
                f"max={batch_statistics['search_steps_max']}), "
                f"SubqCalls(avg={batch_statistics['subq_calls_avg']:.2f}, "
                f"median={batch_statistics['subq_calls_median']:.2f}, "
                f"max={batch_statistics['subq_calls_max']})"
            )
            
            # If there are unknown tools, print a separate warning
            if all_unknown_tool_names:
                logger.warning(
                    f"[Unknown Tools] Step {self.step}: "
                    f"Found in {unknown_tool_count} trajectories ({batch_statistics['unknown_tool_rate']:.2%}), "
                    f"Unique tools: {list(all_unknown_tool_names)}"
                )
        
        # ========================================
        # === Build extra_info (compatible with the existing DataProto structure) ===
        # ========================================
        # 
        # extra_info holds each sample's extra information, used by the subsequent reward
        # computation and analysis
        # Structure: extra_info = [sample1_info, sample2_info, ...]
        # 
        # Each sample's extra_info contains:
        # - abnormal_flags: dict - abnormality flags (booleans)
        # - trajectory_metrics: dict - trajectory features (search_steps, subq_calls, etc.)
        # 
        extra_info_list = []
        for i in range(len(abnormal_flags_list)):
            sample_extra_info = {
                # === Abnormality flags (used to decide whether to give a 0 reward) ===
                "abnormal_flags": abnormal_flags_list[i],
                
                # === Trajectory features (used for per-trajectory reward computation) ===
                "trajectory_metrics": metrics_list[i],
            }
            extra_info_list.append(sample_extra_info)
        
        non_tensor_batch = {
            "messages": np.array(messages, dtype=object),  # conversation history
            "reward_scores": np.array(reward_scores, dtype=object),  # reward scores
            "request_id": np.array(request_ids, dtype=object),  # request ID
            
            # === rollout_extra_info: each sample's extra information (produced during rollout) ===
            # Renamed to rollout_extra_info to avoid clashing with the raw data's extra_info
            "rollout_extra_info": np.array(extra_info_list, dtype=object),
            
            # === batch_statistics: batch-level aggregate statistics (for monitoring) ===
            # Note: batch_statistics is a dict and must be wrapped into an array of batch_size
            # identical elements
            # This satisfies DataProto.check_consistency() (the first dimension of all
            # non-tensor data must equal batch_size)
            "batch_statistics": np.array([batch_statistics] * batch_size, dtype=object),
        }

        is_multimodal = isinstance(self.processing_class, ProcessorMixin) and (
            hasattr(self.processing_class, "image_processor") or hasattr(self.model_hf_config, "vision_config")
        )

        if is_multimodal:
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs, dtype=object)

        # ========================================
        # === Step 6: build the final DataProto ===
        # ========================================
        # 
        # DataProto is the output format of the whole rollout, with two parts:
        # 1. batch (TensorDict): all tensor data
        # 2. non_tensor_batch (dict): all non-tensor data
        # 
        # The complete DataProto structure:
        # {
        #   "batch": {
        #     "prompts": Tensor[batch_size, prompt_length],
        #     "responses": Tensor[batch_size, response_length],
        #     "response_mask": Tensor[batch_size, response_length],  # marks where the loss is computed
        #     "input_ids": Tensor[batch_size, total_length],
        #     "attention_mask": Tensor[batch_size, total_length],
        #     "position_ids": Tensor[batch_size, total_length],
        #     "rollout_log_probs": Tensor[batch_size, response_length],  # if present
        #     "rollout_output_token_ids": Tensor[batch_size, response_length],  # if present
        #   },
        #   "non_tensor_batch": {
        #     "messages": np.array[dict],  # conversation history
        #     "reward_scores": np.array[dict],  # reward scores
        #     "request_id": np.array[str],  # request ID
        #     "multi_modal_inputs": np.array[dict],  # multi-modal inputs (if any)
        #   },
        #   "meta_info": {}  # meta information (added by the caller)
        # }
        # 
        # This DataProto is passed on to the subsequent training flow (critic, actor, etc.)
        result = DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
        )
        torch.cuda.synchronize()
        final_construction_end_time = time.time()
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="final_construction_duration",
                duration=final_construction_end_time - final_construction_start_time,
                workid=self._rank,
                step=self.step
            )

        torch.cuda.synchronize()
        total_end_time = time.time()
        if self._tp_rank == 0:
            all_response_lengths = [len(req.response_ids.squeeze(0)) for req in sorted_output_req_list if req.response_ids is not None]
            all_actual_response_tokens = [torch.sum(req.response_loss_mask.squeeze(0)).item() for req in sorted_output_req_list if req.response_loss_mask is not None]


            self.log_manager.log(
                log_path,
                event="step_response_length_stats",
                extra={
                    "response_lengths": all_response_lengths,
                    "actual_response_tokens": all_actual_response_tokens,
                    "response_length_mean": np.mean(all_response_lengths),
                    "response_length_median": np.median(all_response_lengths),
                    "response_length_p80": np.percentile(all_response_lengths, 80),
                    "response_length_p95": np.percentile(all_response_lengths, 95),
                    "response_length_max": max(all_response_lengths),
                    "response_length_min": min(all_response_lengths),
                    "requests_over_500": sum(1 for x in all_response_lengths if x > 500),
                    "requests_over_1000": sum(1 for x in all_response_lengths if x > 1000),
                    "batch_size": len(sorted_output_req_list)
                },
                workid=self._rank,
                step=self.step
            )
        if self._tp_rank == 0:
            self.log_manager.log(
                log_path,
                event="total_step_duration",
                duration=total_end_time - start_time,
                workid=self._rank,
                step=self.step
            )

        return result

    def _create_padding_request(self, original_req: AsyncRolloutRequest) -> AsyncRolloutRequest:
        # create a padding request to replace the aborted request
        # the padding request has the following characteristics:
        # 1. state is COMPLETED, but contains empty response
        # 2. response_loss_mask is all 0, ensuring it is ignored in loss calculation
        # 3. keep the original request structure, but the content is empty
        # create padding response_ids (all pad_token_id)
        padding_response_length = self.config.response_length
        padding_response_ids = torch.full(
            (1, padding_response_length),
            self.pad_token_id,
            dtype=torch.long,
            device=original_req.input_ids.device if original_req.input_ids is not None else "cpu",
        )

        # create padding attention_mask (all 0)
        padding_response_attention_mask = torch.zeros(
            (1, padding_response_length),
            dtype=torch.long,
            device=original_req.attention_mask.device if original_req.attention_mask is not None else "cpu",
        )

        # create padding position_ids
        if original_req.position_ids is not None:
            prompt_length = original_req.prompt_ids.shape[-1] if original_req.prompt_ids is not None else 0
            padding_response_position_ids = torch.arange(
                prompt_length, prompt_length + padding_response_length, dtype=torch.long
            ).unsqueeze(0)
            if original_req.position_ids.dim() == 2:
                # if it is a 2D tensor (e.g. qwen2vl)
                padding_response_position_ids = padding_response_position_ids.repeat(
                    original_req.position_ids.shape[0], 1
                )
        else:
            padding_response_position_ids = None

        # create padding loss_mask (all 0, ensuring it is ignored)
        padding_response_loss_mask = torch.zeros(
            (1, padding_response_length),
            dtype=torch.long,
            device=original_req.loss_mask.device if original_req.loss_mask is not None else "cpu",
        )

        padding_req = AsyncRolloutRequest(
            batch_data_id=original_req.batch_data_id,
            rollout_offset=original_req.rollout_offset,
            request_id=original_req.request_id,
            state=AsyncRolloutRequestStateEnum.COMPLETED,
            messages=original_req.messages,
            multi_modal_keys=original_req.multi_modal_keys,
            multi_modal_data=original_req.multi_modal_data,
            multi_modal_inputs=original_req.multi_modal_inputs,
            tool_schemas=original_req.tool_schemas,
            tools_kwargs=original_req.tools_kwargs,
            interaction_kwargs=original_req.interaction_kwargs,
            input_ids=original_req.input_ids,
            prompt_ids=original_req.prompt_ids,
            response_ids=padding_response_ids,
            attention_mask=original_req.attention_mask,
            prompt_attention_mask=original_req.prompt_attention_mask,
            response_attention_mask=padding_response_attention_mask,
            position_ids=original_req.position_ids,
            prompt_position_ids=original_req.prompt_position_ids,
            response_position_ids=padding_response_position_ids,
            loss_mask=original_req.loss_mask,
            prompt_loss_mask=original_req.prompt_loss_mask,
            response_loss_mask=padding_response_loss_mask,
            reward_scores={},
            max_prompt_len=original_req.max_prompt_len,
            max_response_len=original_req.max_response_len,
            metrics={},
            output_token_ids=None,
            rollout_log_probs=None,
            use_inference_chat_template=original_req.use_inference_chat_template,
            tokenization_sanity_check_mode=original_req.tokenization_sanity_check_mode,
            generation_prompt_ids=original_req.generation_prompt_ids,
            base_conv_wo_gen_prompt_end_pos=original_req.base_conv_wo_gen_prompt_end_pos,
            base_conv_with_gen_prompt_end_pos=original_req.base_conv_with_gen_prompt_end_pos,
            processing_class=self.processing_class,
        )
        return padding_req

    def _preprocess_prompt_to_async_rollout_requests(self, prompts: DataProto, n: int = 1) -> list[AsyncRolloutRequest]:
        assert "raw_prompt" in prompts.non_tensor_batch, (
            "need data.return_raw_chat=True, due to no official way do parse_messages"
        )
        logger.info(
            "n is deprecated for SGLang rollout since ray ppo trainer will repeat the prompts for rollout.n times"
        )
        req_list = []
        multi_modal_data_list = prompts.non_tensor_batch.get(
            "multi_modal_data", [None] * len(prompts.non_tensor_batch["raw_prompt"])
        )

        # Use global data index if available (injected by ray_trainer before dispatch).
        # This is critical for PS Pipeline: without it, each worker's enumerate()
        # produces local indices (0..shard_size-1) which collide after concat,
        # causing batch_data_id // N to map different prompts to the same uid.
        global_data_idx_arr = prompts.non_tensor_batch.get("__global_data_idx__", None)

        # Extract reward_model / ground_truth list (used by rollout-side rubric scoring
        # to call the LLM Judge with the same GT that reward manager will use).
        reward_model_arr = prompts.non_tensor_batch.get("reward_model", None)
        ground_truth_arr = prompts.non_tensor_batch.get("ground_truth", None)

        for data_idx, (raw_prompt, multi_modal_data) in enumerate(
            zip(prompts.non_tensor_batch["raw_prompt"], multi_modal_data_list, strict=True)
        ):
            # Use global index when available, fall back to local index
            effective_data_idx = int(global_data_idx_arr[data_idx]) if global_data_idx_arr is not None else data_idx

            if self._tool_schemas:
                _tools_kwargs = prompts.non_tensor_batch["tools_kwargs"][data_idx]
                _tool_schemas = [self._tool_map[k].get_openai_tool_schema() for k in _tools_kwargs.keys()]
                _input_ids = None
                _attention_mask = None
            else:
                _input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][data_idx])
                _attention_mask = _pre_process_inputs(0, prompts.batch["attention_mask"][data_idx])
                _tools_kwargs = {}
                _tool_schemas = None

            if self.interaction_map:
                _interaction_kwargs = prompts.non_tensor_batch["interaction_kwargs"][data_idx]
            else:
                _interaction_kwargs = {}

            if not isinstance(raw_prompt, list | np.ndarray):
                raise TypeError(f"raw_prompt must be a list or numpy array, got {type(raw_prompt)}")

            # Extract ground_truth (passed to the LLM Judge during rollout-side rubric scoring)
            # Priority: reward_model.ground_truth > ground_truth (top level), consistent with
            # the reward manager's extraction logic
            _gt_for_rubric = None
            try:
                if reward_model_arr is not None:
                    _rm = reward_model_arr[data_idx]
                    if isinstance(_rm, dict):
                        _gt_for_rubric = _rm.get("ground_truth", None)
            except (IndexError, TypeError, KeyError):
                pass
            if _gt_for_rubric is None and ground_truth_arr is not None:
                try:
                    _gt_for_rubric = ground_truth_arr[data_idx]
                except (IndexError, TypeError):
                    pass
            # If it's a dict, extract target; otherwise convert to str
            if isinstance(_gt_for_rubric, dict):
                _gt_for_rubric = str(_gt_for_rubric.get("target", "") or "")
            elif _gt_for_rubric is not None:
                _gt_for_rubric = str(_gt_for_rubric)
            else:
                _gt_for_rubric = ""

            req = AsyncRolloutRequest(
                batch_data_id=effective_data_idx,
                rollout_offset=0,
                request_id=str(uuid4()),
                state=AsyncRolloutRequestStateEnum.PENDING,
                messages=list(raw_prompt),
                multi_modal_data=multi_modal_data,
                tool_schemas=_tool_schemas,
                tools_kwargs=_tools_kwargs,
                interaction_kwargs=_interaction_kwargs,
                input_ids=_input_ids,
                response_ids=None,
                attention_mask=_attention_mask,
                response_attention_mask=None,
                response_position_ids=None,
                response_loss_mask=None,
                reward_scores={},
                max_prompt_len=self.config.prompt_length,
                max_response_len=self.config.response_length,
                max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
                use_inference_chat_template=self.config.multi_turn.use_inference_chat_template,
                tokenization_sanity_check_mode=self.config.multi_turn.tokenization_sanity_check_mode,
                processing_class=self.processing_class,
            )
            error_message = f"""Request {req.request_id} has mismatched lengths:
            input_ids={req.input_ids.shape[-1]},
            attention_mask={req.attention_mask.shape[-1]},
            position_ids={req.position_ids.shape[-1]},
            loss_mask={req.loss_mask.shape[-1]}"""
            assert (
                req.input_ids.shape[-1]
                == req.attention_mask.shape[-1]
                == req.position_ids.shape[-1]
                == req.loss_mask.shape[-1]
            ), error_message
            # Record ground_truth for use by the rollout-side rubric scoring (keeping the input
            # consistent with the reward manager side)
            if not hasattr(self, "_rubric_gt_map"):
                self._rubric_gt_map = {}
            self._rubric_gt_map[req.request_id] = _gt_for_rubric
            req_list.append(req)

        return req_list

    # ==================== server mode public methods ====================

    async def chat_completion(self, json_request):
        """OpenAI chat completion API."""
        assert self._tp_rank == 0, "only called in tp rank 0"
        _input_ids = None
        _attention_mask = None
        _position_ids = None
        _tool_schemas = []
        _tools_kwargs = {}

        req = AsyncRolloutRequest(
            request_id=str(uuid4()),
            state=AsyncRolloutRequestStateEnum.PENDING,
            messages=[Message.model_validate(msg) for msg in json_request["messages"]],
            tool_schemas=_tool_schemas,
            tools_kwargs=_tools_kwargs,
            input_ids=_input_ids,
            prompt_ids=_input_ids,
            response_ids=None,
            attention_mask=_attention_mask,
            prompt_attention_mask=_attention_mask,
            response_attention_mask=None,
            position_ids=_position_ids,
            prompt_position_ids=_position_ids,
            response_position_ids=None,
            loss_mask=None,
            prompt_loss_mask=None,
            response_loss_mask=None,
            reward_scores={},
            max_prompt_len=self.config.prompt_length,
            max_response_len=self.config.response_length,
            max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
            use_inference_chat_template=self.config.multi_turn.use_inference_chat_template,
            tokenization_sanity_check_mode=self.config.multi_turn.tokenization_sanity_check_mode,
            processing_class=self.processing_class,
        )

        # json_request already contains sampling_params
        # Filter only valid SamplingParams arguments
        valid_sampling_params = {}
        temp_sampling_params = SamplingParams()  # Create temporary instance to check valid attributes
        for k, v in json_request.items():
            if k not in ["messages", "model", "tools"] and hasattr(temp_sampling_params, k):
                valid_sampling_params[k] = v
        output = await self._handle_engine_call(req, valid_sampling_params)
        # it can be Dict or AsyncIterator[Dict]
        if isinstance(output, dict):
            outputs = [output]
        else:
            outputs = output

        # build openai chat completion format
        choices = []
        id = None
        for i, content in enumerate(outputs):
            choices.append(
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": content["text"],
                    },
                    "finish_reason": content["meta_info"]["finish_reason"]["type"],
                }
            )
            id = content["meta_info"]["id"]

        return {
            "id": "chatcmpl-" + id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": json_request.get("model", "sglang_model"),
            "choices": choices,
        }

        # this function is left for uniform train-inference resharding

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> torch.Tensor:
        """Generate sequence with token-in-token-out."""
        request_sampling_params = self.sampling_params.copy()
        request_sampling_params.update(sampling_params)
        output = await self._handle_engine_generate(prompt_ids, request_sampling_params, image_data=image_data)
        return output["output_ids"]

    async def wake_up(self):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        await self.sharding_manager.wake_up()  # pylint: disable=C2801
        self.is_sleep = False

    async def sleep(self):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        await self.sharding_manager.sleep()
        self.is_sleep = True

    # ====================================================================
    # ===================== PS Pipeline (Planner-Synthesizer) =============
    # ====================================================================
    #
    # The new Planner-Synthesizer Pipeline:
    #
    # Architecture:
    #   ┌──────────────────────────────────────────────────────────────────┐
    #   │  Round 1:                                                        │
    #   │    Planner(question, summary="") -> think + tool_call(search)    │
    #   │    Search(queries) -> tool_response (documents)                   │
    #   │    Synthesizer(question, summary="", P_think, queries, docs)     │
    #   │      -> new_summary                                              │
    #   │                                                                  │
    #   │  Round 2:                                                        │
    #   │    Planner(question, summary) -> think + tool_call(search)       │
    #   │    Search(queries) -> tool_response (documents)                   │
    #   │    Synthesizer(question, summary, P_think, queries, docs)        │
    #   │      -> updated_summary                                          │
    #   │                                                                  │
    #   │  Round N (final):                                                │
    #   │    Planner(question, summary) -> think + answer                  │
    #   │                                                                  │
    #   └──────────────────────────────────────────────────────────────────┘
    #
    # Output data: each round produces an independent data sample
    #   the trajectory P(search)-S-P(search)-S-P(answer) produces 5 samples:
    #     [P_round1, S_round1, P_round2, S_round2, P_round3(answer)]
    #
    # Structure of each data sample:
    #   - prompt_ids: this round's input prompt (the Planner or Synthesizer prompt)
    #   - response_ids: the model-generated response (excluding tool_response)
    #   - loss_mask: all ones (the response contains no tool output, so everything gets a loss)
    #
    # ====================================================================

    # ---- Final Answer prompt template ----
    # When max_search_rounds is reached but the Planner still emits a <tool_call>,
    # this prompt is used to force the generation of a final answer.
    # Note: during the rollout this prompt generates the response, but during update
    # the prompt is replaced with the normal PLANNER_PROMPT_TEMPLATE (flagged via forced_answer).
    FINAL_ANSWER_PROMPT_TEMPLATE = """The current date is {TIME}.

[Core Instruction: Language Consistency]
You MUST write your entire response in the same language as the user's question.

[Core Instruction: Citation Formatting]
If you are referencing a specific webpage in your final answer, you must use the format "[citation:x]", where x is the corresponding webpage number.

You have reached the maximum number of search rounds ({max_rounds} rounds). Based on ALL the information gathered, you MUST now synthesize a final answer.

# User's Original Question
{question}

# Research Summary (All Information Gathered)
<summary>
{summary}
</summary>

**Important Guidelines:**
1. **Absolute Source Fidelity**: Your answer must be forged *only* from the information in the summary. Do not introduce any external information.
2. **Synthesis Over Summarization**: Intelligently weave together facts from multiple sources to build a holistic narrative.
3. **Structured Clarity**: Your answer must be impeccably structured, logical, and address all parts of the user's question.
4. **Handle Uncertainty**: If some aspects of the question remain unanswered, acknowledge this explicitly while providing the best possible answer based on available evidence.

# Output Format
First write your reasoning process, then output your final answer enclosed in <answer> tags:

<answer>
Your final answer here...
</answer>
"""

    # ---- Planner prompt template ----
    # Aligned with Q_AGENT_USER_PROMPT_TEMPLATE in data_generation.py
    PLANNER_PROMPT_TEMPLATE = """You are a professional problem-solving agent with rigorous information verification capabilities and deep analytical thinking.

## CRITICAL OUTPUT FORMAT REQUIREMENTS
You MUST follow this exact format. Every response must contain:
1. <think>...</think> (always required)
2. Either <answer>...</answer> OR <tool_call>...</tool_call> (never both)

## Input Format
- **Current Date**: Current Date
- **Question**: The problem posed by the user that needs to be solved
- **Last Status Report and Deep Analysis**: A summary overview of current work progress

## Output Format

<think>
Your reasoning here, reasoning process should be comprehensive and demonstrate deep analytical thinking with at least 200 words of detailed analysis covering multiple perspectives and potential implications.
</think> You MUST output this section enclosed with <think></think> tags!

**Decision Point**: Are you certain that no further verification or information gathering is needed to provide the final answer?

**If YES - Information is sufficient:**
<answer>
Answer Format:
1. **Language**: Your answer should be in the same language as the question. If the question uses English, answer in English. If the question uses Chinese, answer in Chinese.
2. The answer should include as much relevant content as possible. Organize the content into separate paragraphs to avoid overly long sections. Avoid content duplication in the answer.
3. Do not include any non-text elements such as URLs, images, or tables that appeared in the reasoning.
4. Output only the answer text. Do not use any additional symbols or start with phrases like 'Here is my answer'.
5. First, output a direct answer to the question.
6. Do not just output the answer to the question; provide a rich and lengthy response by synthesizing all relevant information, and format it using markdown.
7. For statistical data with at least 3 items, use a markdown table to present the results, ensuring the table description is clear. For less than 3 items, describe them directly in text.
8. For research-type questions, try to generate a report of over 1000 words, using subheadings and other elements to improve readability and logic.
</answer>
You MUST output this section enclosed with <answer></answer> tags!

**If NO - Further action needed:**
<tool_call>
{{"name": "tool name here", "arguments": {{"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}}}
</tool_call>
You MUST output this section enclosed with <tool_call></tool_call> tags!

## Working Principles
1. **Rigorous Verification**: Critically evaluate all information sources
2. **Deep Thinking**: Pursue essential understanding, not satisfied with surface phenomena
3. **Evidence-Driven**: Make reasoning decisions based on reliable evidence through deep thinking

## Special Requirements
- All tools in the tool list are real and functional - as long as you make correct tool calls, you will receive their returned results.
- Clearly distinguish between "confirmed facts," "highly credible inferences," and "hypotheses to be verified"
- Clearly indicate uncertainty when information is insufficient
- Always focus on the original question
- **When further action is needed, you must select an appropriate tool from your available tool list and carefully configure the tool call parameters based on the tool's specific characteristics and requirements**
- **When the current status is sufficient to answer the question, must provide the final answer enclosed with <answer></answer> tags rather than continue with actions**

## FORMAT REMINDER
- Start with <think>...</think> section
- Then choose: <answer>...</answer> if sufficient info, OR <tool_call>...</tool_call> if need more action
- Never output both answer and tool call tags in same response

## Input
- Current Date: {current_date}
- Question: {question}
- Available Tools
{{
    "type": "function",
    "function": {{
        "name": "search",
        "description": "Perform web searches then returns a string of the top search results. Accepts multiple queries.",
        "parameters": {{
            "type": "object",
            "properties": {{
                "query": {{
                    "type": "array",
                    "items": {{"type": "string"}},
                    "minItems": 1,
                    "maxItems": 5,
                    "description": "The list of search queries."
                }}
            }},
            "required": ["query"]
        }}
    }}
}}

- Last Status Summary and Deep Analysis:
Below is the summary from the previous attempt:

<summary>
{summary}
</summary>

**Instructions for Utilizing the Summary:**
1. Critical Evaluation:
The summary provided above is a *suggested* synthesis of past efforts, not an absolute truth. It may contain hallucinations, unverified assumptions, or premature conclusions. You are encouraged to:
- Identify and ignore any parts of the summary that seem illogical or poorly supported.
- If the summary's quality is low or its direction feels like a dead-end, you have the full autonomy to completely disregard it and initiate a fresh search strategy.

2. Expand the search space:
If the task remains unsolved, it means the current search space is insufficient. Do not get trapped in the "logic loop" of the summary. Use the [Uncertainties, Limitations, Gaps] section as a springboard to:
- Pivot to entirely different keyword clusters or tool-call strategies.
- Cross-verify "Facts" that were marked as "unverified" or "partial" in the summary.

3. Autonomous re-planning:
The summary provides the *memory*, but you provide the *reasoning*. Based on the existing facts and your own judgment of the current situation, determine your own next steps.

Now please begin your deep analytical work. The language of your output must be consistent with the language of the question. If the question is in Chinese, output in Chinese; if the question is in English, output in English.
"""

    # ---- Synthesizer prompt template ----
    # Aligned with R_AGENT_USER_PROMPT_TEMPLATE_GLOBAL in data_generation.py
    SYNTHESIZER_PROMPT_TEMPLATE = """You are a professional information analysis agent responsible for updating and maintaining a global summary based on search results.

## CRITICAL OUTPUT FORMAT REQUIREMENTS
You MUST follow this exact format. Every response must contain:
- <summary>...</summary> (always required)

## Input Format
- **Question**: The problem posed by the user that needs to be solved
- **Current Global Summary**: Accumulated information from previous searches
- **Latest Reasoning**: The Q-Agent's thinking process for the latest search
- **Search Queries**: The queries used in the latest search
- **Search Results**: Documents retrieved from the latest search

## Output Format

Output your UPDATED global summary enclosed in <summary> tags:

<summary>
## 0) Latest Search
- **Search Queries**: List the queries used in the latest search. For each query, briefly explain its intended purpose (what specific information it was trying to find). If a specific query yielded no search results, you MUST explicitly note "NO RESULTS" next to it.
- **Key Evidence Found**: Extract the most important evidence/facts discovered
- **Source Documents**: List documents that provided useful information [Title | What it contributed]

## 1) Facts & Evidence Collected
- List every factual item discovered across ALL searches
- Attach a source annotation: [Source: Title | Verified: yes/no/partial]
- Keep ALL previously collected facts that remain relevant
- ADD new facts from the latest search
- DELETE facts that are proven wrong, irrelevant, or hallucinatory by new evidence

## 2) Analysis & Conclusions
- State all logical conclusions derived, explicitly linking each to the supporting evidence
- Re-evaluate past conclusions dialectically: if new data contradicts old conclusions, boldly rewrite them.

## 3) Source Inventory & Verification Status
- Enumerate ONLY the sources that provided USEFUL information and evaluate their current verification status. (do NOT list unused/irrelevant documents).

## 4) Uncertainties, Limitations, Gaps
- List all unknown variables, data ambiguities, and failure modes currently blocking a final decision.
- This section should shrink as more information is gathered.
</summary>
You MUST output this section enclosed with <summary></summary> tags!

## Working Principles
1. **PRESERVE**: Keep all valuable information that is strictly relevant to solving the problem.
2. **REFINE**: Improve any information with new findings.
3. **ADD**: Include new facts and evidence from the latest search documents.
4. **UPDATE**: Revise conclusions if new evidence provides better answers or exposes previous logical flaws.
5. **PRUNE & PURGE (CRITICAL)**: Actively delete incorrect, outdated, or useless information. If previous facts, inferences, or summary sections are contradicted by new evidence, proven to be hallucinations, or lead to dead-ends, REMOVE them. Keep the summary concise and highly relevant; do not let it become bloated with noise or obsolete data.

## Special Requirements
- **Dialectical Evaluation**: Treat the 'Current Global Summary' with healthy skepticism. Cross-verify its contents against the latest search results. Do not blindly inherit past errors, unverified assumptions, or illogical leaps.
- **Aggressive Condensation**: Delete verbose descriptions of failed paths once they are briefly noted in "Failed Attempts". Remove redundant data. Your goal is a high-signal, low-noise summary.
- You MUST write your entire response in the same language as the user's question.
- Do NOT use placeholders like "see above", "as mentioned", "ibid.", "refer to earlier", or "same as before".
- Do NOT omit any important information that appears in the input.
- Be exhaustive over the input regarding *useful* facts, but merciless in deleting *useless/proven false* facts.
- Keep claims tightly tied to evidence; if you infer, label it "Inference" and explain why.
- If sources conflict, present both and state which you trust more (and why).

## Input
- Question: {question}
- Current Global Summary:
<summary>
{global_summary}
</summary>
- Latest Reasoning:
<think>
{q_agent_think}
</think>
- Latest Search Queries:
<tool_call>
{tool_call}
</tool_call>
- Latest Search Results ({doc_count} documents):
<tool_response>
{documents_formatted}
</tool_response>

Now please analyze the search results and update the global summary. The language of your output must be consistent with the language of the question. If the question is in Chinese, output in Chinese; if the question is in English, output in English.
"""

    def _build_planner_prompt(
        self,
        question: str,
        summary: str,
        current_date: str | None = None,
    ) -> str:
        """Build the Planner agent prompt.

        Args:
            question: The user's question.
            summary: The current global summary from previous rounds.
            current_date: Current date string. Defaults to today.

        Returns:
            The formatted Planner prompt string.
        """
        if current_date is None:
            current_date = datetime.now().strftime("%Y-%m-%d")
        if not summary:
            summary = "No previous search has been conducted yet."

        return self.PLANNER_PROMPT_TEMPLATE.format(
            current_date=current_date,
            question=question,
            summary=summary,
        )

    def _build_synthesizer_prompt(
        self,
        question: str,
        global_summary: str,
        q_agent_think: str,
        tool_call: str,
        doc_count: int,
        documents_formatted: str,
    ) -> str:
        """Build the Synthesizer agent prompt.

        Args:
            question: The user's question.
            global_summary: The current accumulated global summary.
            q_agent_think: The Planner's <think> content from the latest round.
            tool_call: The tool_call content from the Planner's output.
            doc_count: Number of documents returned.
            documents_formatted: Formatted search results.

        Returns:
            The formatted Synthesizer prompt string.
        """
        if not global_summary:
            global_summary = "No previous search has been conducted yet."

        return self.SYNTHESIZER_PROMPT_TEMPLATE.format(
            question=question,
            global_summary=global_summary,
            q_agent_think=q_agent_think,
            tool_call=tool_call,
            doc_count=doc_count,
            documents_formatted=documents_formatted,
        )

    def _parse_planner_output(self, content: str) -> dict:
        """Parse Planner output to extract think, answer, and tool_call.

        The Planner output format:
            <think>...</think>
            <answer>...</answer>  OR  <tool_call>...</tool_call>

        Args:
            content: Raw generated text from the Planner.

        Returns:
            dict with keys:
                - "think": str - content inside <think> tags
                - "action_type": "answer" | "tool_call" | "unknown"
                - "answer": str | None - content inside <answer> tags
                - "tool_call_raw": str | None - raw content inside <tool_call> tags
                - "tool_call_parsed": dict | None - parsed tool call JSON
                - "queries": list[str] | None - extracted search queries
                - "parse_error": str | None - error message if parsing failed
        """
        import re

        result = {
            "think": "",
            "action_type": "unknown",
            "answer": None,
            "tool_call_raw": None,
            "tool_call_parsed": None,
            "queries": None,
            "parse_error": None,
        }

        # Extract <think> content
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match:
            result["think"] = think_match.group(1).strip()

        # Check for <answer>
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if answer_match:
            result["action_type"] = "answer"
            result["answer"] = answer_match.group(1).strip()
            return result

        # Check for <tool_call>
        tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        if tool_call_match:
            result["action_type"] = "tool_call"
            result["tool_call_raw"] = tool_call_match.group(1).strip()

            try:
                parsed = json.loads(result["tool_call_raw"])
                result["tool_call_parsed"] = parsed

                # Extract queries - supports both the "query" and "queries" argument names
                if isinstance(parsed, dict):
                    args = parsed.get("arguments", {})
                    if isinstance(args, dict):
                        queries = args.get("query") or args.get("queries", [])
                        if isinstance(queries, list):
                            result["queries"] = [q for q in queries if isinstance(q, str) and q.strip()]
            except (json.JSONDecodeError, TypeError) as e:
                result["parse_error"] = f"JSON parse error: {str(e)}"
            return result

        # Neither answer nor tool_call found
        result["action_type"] = "unknown"
        result["parse_error"] = "No <answer> or <tool_call> tags found in output"
        return result

    def _parse_synthesizer_output(self, content: str) -> dict:
        """Parse Synthesizer output to extract the updated summary.

        Args:
            content: Raw generated text from the Synthesizer.

        Returns:
            dict with keys:
                - "summary": str - content inside <summary> tags
                - "parse_error": str | None - error message if parsing failed
        """
        import re

        result = {
            "summary": "",
            "parse_error": None,
        }

        summary_match = re.search(r"<summary>(.*?)</summary>", content, re.DOTALL)
        if summary_match:
            result["summary"] = summary_match.group(1).strip()
        else:
            result["parse_error"] = "No <summary> tags found in Synthesizer output"
            # Fallback: use the entire content as summary
            result["summary"] = content.strip()

        return result

    async def _ps_generate_single_turn(
        self,
        prompt_text: str,
        request_sampling_params: dict,
        request_id: str = "",
        max_new_tokens_override: int | None = None,
        max_model_len_override: int | None = None,
    ) -> dict:
        """Generate a single turn response for a given prompt text.

        This method applies chat template to the prompt, tokenizes it, calls the engine,
        and returns the output. Used by both Planner and Synthesizer in the PS pipeline.

        The prompt_text is wrapped as a user message and processed through the model's
        chat template (e.g., <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n)
        to ensure the model sees the same format as during SFT/pretraining.

        Args:
            prompt_text: The formatted prompt string (will be wrapped as user message).
            request_sampling_params: Sampling parameters for generation.
            request_id: Request ID for logging.

        Returns:
            dict: Engine output with "text", "output_ids", "meta_info", and
                  "prompt_ids" (the tokenized prompt tensor).
        """
        # Get tokenizer
        try:
            tokenizer = self.processing_class.tokenizer
        except AttributeError:
            tokenizer = self.processing_class

        # Apply chat template: wrap prompt_text as a user message
        # This produces format like: <|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n
        # For Qwen3 models with thinking mode enabled during SFT, we must pass enable_thinking=True
        # to ensure the chat template includes the thinking-mode special tokens.
        messages = [{"role": "user", "content": prompt_text}]
        templated_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )

        # Tokenize the templated prompt
        prompt_ids = tokenizer.encode(templated_prompt, return_tensors="pt")
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids_list = prompt_ids.squeeze(0).tolist()
        else:
            prompt_ids_list = prompt_ids

        # Guard: reject prompt if it exceeds max_model_len to prevent SGLang
        # from raising "input is longer than context length".
        # The caller's except-block will catch this and break with search_error=True.
        effective_max_model_len = max_model_len_override if max_model_len_override is not None else self.config.max_model_len
        if len(prompt_ids_list) >= effective_max_model_len:
            raise ValueError(
                f"[PS Pipeline] request_id={request_id}: Prompt too long "
                f"({len(prompt_ids_list)} tokens >= max_model_len={effective_max_model_len}), "
                f"skipping generation."
            )

        # Call engine
        output = await self._handle_engine_generate(
            prompt_ids_list,
            request_sampling_params,
            request_id=request_id,
            max_new_tokens_override=max_new_tokens_override,
            max_model_len_override=max_model_len_override,
        )

        # Attach prompt_ids tensor for later use
        if isinstance(prompt_ids, torch.Tensor):
            output["prompt_ids"] = prompt_ids
        else:
            output["prompt_ids"] = torch.tensor([prompt_ids_list], dtype=torch.long)

        # Compatibility: SGLang may return "output_token_ids" or "token_ids" instead of "output_ids"
        if "output_ids" not in output:
            for alt_key in ("output_token_ids", "token_ids"):
                if alt_key in output:
                    output["output_ids"] = output[alt_key]
                    break
            else:
                # Fallback: try to get from meta_info
                meta = output.get("meta_info", {})
                if "output_token_ids" in meta:
                    output["output_ids"] = meta["output_token_ids"]
                elif "token_ids" in meta:
                    output["output_ids"] = meta["token_ids"]
                elif output.get("text"):
                    # Fallback: encode text back to token ids using tokenizer
                    output["output_ids"] = self.processing_class.encode(
                        output["text"], add_special_tokens=False
                    )
                else:
                    logger.error(
                        f"[PS Pipeline] Engine output missing 'output_ids'. "
                        f"Available keys: {list(output.keys())}, "
                        f"meta_info keys: {list(meta.keys())}"
                    )
                    output["output_ids"] = []

        return output

    async def _async_rollout_ps_pipeline(
        self,
        req: AsyncRolloutRequest,
        do_sample: bool = True,
        is_validate: bool = False,
        **kwargs,
    ) -> list[dict]:
        """Execute the Planner-Synthesizer pipeline for a single request.

        This is the core function of the PS pipeline. It alternates between
        Planner and Synthesizer agents, producing independent data samples
        for each round.

        A trajectory like P(search)-S-P(search)-S-P(answer) produces 5 data samples:
          [P_round1_data, S_round1_data, P_round2_data, S_round2_data, P_round3_data]

        Each data sample contains:
          - role: "planner" or "synthesizer"
          - prompt_ids: tokenized prompt for this round
          - response_ids: model-generated response tokens (no tool_response)
          - loss_mask: all 1s (every token participates in loss)
          - messages: conversation context for this round

        Args:
            req: The initial AsyncRolloutRequest.
            do_sample: Whether to sample (True) or use greedy decoding.
            is_validate: Whether this is a validation run.

        Returns:
            list[dict]: List of per-round data samples. Each dict contains:
                - "role": str ("planner" or "synthesizer")
                - "prompt_ids": Tensor [1, prompt_len]
                - "response_ids": Tensor [1, response_len]
                - "attention_mask": Tensor [1, prompt_len + response_len]
                - "position_ids": Tensor [1, prompt_len + response_len]
                - "loss_mask": Tensor [1, prompt_len + response_len]
                - "batch_data_id": int
                - "request_id": str
                - "messages": list - context messages
                - "think": str - Planner's think content (if planner)
                - "answer": str | None - Planner's answer (if final planner round)
                - "summary": str - current global summary
                - "abnormal_flags": dict - error/anomaly flags
                - "trajectory_metrics": dict - per-sample metrics
        """
        assert self._tp_rank == 0, "only the master process can call this function"

        log_path = os.path.join(
            self.log_dir,
            f"step_{self.step}",
            f"worker_{self._rank}.jsonl"
        )

        # Extract the user's question from the original messages
        question = ""
        for msg in req.messages:
            if isinstance(msg, Message):
                if msg.role == "user":
                    question = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break
            elif isinstance(msg, dict):
                if msg.get("role") == "user":
                    question = msg.get("content", "")
                    break

        if not question:
            logger.warning(f"[PS Pipeline] Request {req.request_id}: No user question found in messages")

        # Initialize state
        global_summary = ""
        round_data_list = []  # Collect per-round data samples
        search_round_count = 0  # search-round counter: P(search)+S counts as one round
        # ps_max_rounds: the max number of search rounds (P(search)+S counts as one round);
        # P(answer) is not counted
        max_search_rounds = 30

        # ---- Parameter overrides used during validation ----
        # During validation, a larger max_model_len is used, allowing a longer context
        # Planner max_new_tokens=4096, Synthesizer max_new_tokens=8192
        if is_validate:
            effective_max_model_len = 40960
            planner_max_new_tokens = 4096
            synth_max_new_tokens = 8192
        else:
            effective_max_model_len = self.config.max_model_len
            planner_max_new_tokens = 4096  # also 4096 during training
            synth_max_new_tokens = None  # during training, the default response_length is used

        # ---- Async rubric scoring task list ----
        # Once a trajectory finishes, check whether it is normal (no search_error /
        # tool_parse_error / exceed_max_turns / exceed_max_tokens);
        # rubric scoring is only launched for normal trajectories, to avoid wasting API calls
        # on abnormal ones
        # format: [(asyncio.Task, role, round_num, data_sample_index), ...]
        rubric_pending_tasks = []

        # Prepare sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update({
                "n": 1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.05,
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "ignore_eos": False,
                "min_new_tokens": 0,
                "max_new_tokens": self.config.response_length,
                "skip_special_tokens": True,
                "spaces_between_special_tokens": True,
            })
        elif is_validate:
            request_sampling_params.update({
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            })
        request_sampling_params.update(kwargs)

        # Prepare for tool execution
        prompt_str = ""
        if req.prompt_ids is not None:
            prompt_str = self.processing_class.decode(req.prompt_ids[0], skip_special_tokens=True)

        # Abnormal flags for the entire trajectory
        # Abnormality categories:
        # 1. Discard (excluded from training):
        #    - search_error: the search tool failed / the API retries were exhausted
        #    - exceed_max_tokens: a single prompt+response exceeded max_model_len
        #    - degenerate_output: degenerate output (repetition/garbage/abnormal repeats)
        # 2. Give a 0 reward (participates in training, the model learns to avoid it):
        #    - tool_parse_error: the Planner's output format is abnormal
        #    - exceed_max_turns: the search rounds hit the limit
        # 3. Offending round zeroed (trajectory kept, only that round's reward is zeroed):
        #    - excessive_tool_calls_per_turn: too many queries in a single round
        # 4. Handled normally (does not affect the reward):
        #    - repeated_query: duplicate search query
        trajectory_flags = {
            "excessive_tool_calls_per_turn": False,
            "tool_parse_error": False,
            "repeated_query": False,
            "search_error": False,
            "exceed_max_turns": False,
            "exceed_max_tokens": False,
            "degenerate_output": False,
            "excessive_rounds": [],  # records which rounds exceeded the limit (to zero their reward)
            "forced_answer_generated": False,  # whether the forced prompt successfully produced an answer
        }
        query_history = []
        total_tool_calls = 0
        total_tool_call_attempts = 0

        torch.cuda.synchronize()
        request_start_time = time.time()

        self.log_manager.log(
            log_path,
            event="ps_pipeline_request_start",
            extra={
                "request_id": req.request_id,
                "question_length": len(question),
            },
            workid=self._rank,
            step=self.step,
        )

        # ==========================================
        # === Initialize tools (PENDING state) ===
        # ==========================================
        if req.tool_schemas is not None:
            tool_creation_coroutines = []
            for tool_schema in req.tool_schemas:
                tool = self._tool_map[tool_schema.function.name]
                create_kwargs = req.tools_kwargs[tool.name].get("create_kwargs", {})
                tool_creation_coroutines.append(tool.create(req.request_id, **create_kwargs))
            await asyncio.gather(*tool_creation_coroutines)

        # ==========================================
        # === Main Loop: Planner-Synthesizer ===
        # ==========================================
        #
        # Round-counting rules:
        #   - search_round_count: search rounds (+1 after P(search)+S completes)
        #   - P(answer) is the terminating action and is not counted in search_round_count
        #   - loop condition: search_round_count < max_search_rounds
        #   - when search_round_count >= max_search_rounds, force the Planner to make one last
        #     answer (no more tool_call); if it still emits a tool_call, set
        #     exceed_max_turns=True and break
        #
        # Abnormal exit conditions (following the original Break + 0 reward logic):
        #   - the Planner's output failed to parse (tool_parse_error)
        #   - the model called a tool when it shouldn't (still tool_call after exceed_max_turns)
        #   - the search tool raised an exception (search_error, an environment failure ->
        #     discarded directly)
        #   - an unknown tool name (tool_parse_error)
        #
        data_sample_index = 0  # the global index of the data sample

        while True:
            torch.cuda.synchronize()
            round_start_time = time.time()

            # ---- Step 1: Planner ----
            planner_prompt: str = self._build_planner_prompt(
                question=question,
                summary=global_summary,
            )

            self.log_manager.log(
                log_path,
                event="ps_planner_start",
                extra={
                    "request_id": req.request_id,
                    "search_round": search_round_count,
                    "data_sample_index": data_sample_index,
                    "prompt_length": len(planner_prompt),
                },
                workid=self._rank,
                step=self.step,
            )

            torch.cuda.synchronize()
            planner_gen_start = time.time()
            try:
                planner_output = await self._ps_generate_single_turn(
                    prompt_text=planner_prompt,
                    request_sampling_params=request_sampling_params,
                    request_id=f"{req.request_id}_P_{data_sample_index}",
                    max_new_tokens_override=planner_max_new_tokens,
                    max_model_len_override=effective_max_model_len,
                )
            except Exception as e:
                # Engine generation exception (environment failure): record the error and break
                logger.error(
                    f"[PS Pipeline] Request {req.request_id} Planner generation failed "
                    f"at search_round={search_round_count}: {type(e).__name__}: {e}"
                )
                traceback.print_exc()
                trajectory_flags["search_error"] = True
                break
            torch.cuda.synchronize()
            planner_gen_end = time.time()

            planner_content = planner_output["text"]
            planner_prompt_ids = planner_output["prompt_ids"]  # [1, prompt_len]

            planner_response_ids = torch.tensor(
                [planner_output["output_ids"]], dtype=torch.long
            )  # [1, response_len]

            # Build Planner data sample
            p_prompt_len = planner_prompt_ids.shape[-1]
            p_response_len = planner_response_ids.shape[-1]
            p_total_len = p_prompt_len + p_response_len
            p_attention_mask = torch.ones(1, p_total_len, dtype=torch.long)
            p_position_ids = torch.arange(p_total_len, dtype=torch.long).unsqueeze(0)
            p_loss_mask = torch.cat([
                torch.zeros(1, p_prompt_len, dtype=torch.long),  # prompt: no loss
                torch.ones(1, p_response_len, dtype=torch.long),  # response: all loss
            ], dim=-1)

            # Check the token length limit (only flagged during training, not limited during validation)
            if not is_validate and p_total_len > effective_max_model_len:
                trajectory_flags["exceed_max_tokens"] = True
                logger.warning(
                    f"[PS Pipeline] Request {req.request_id}: Planner exceed_max_tokens "
                    f"{p_total_len} > {effective_max_model_len}"
                )

            # Check for degenerate output (only detected and broken on during training, not
            # interrupted during validation)
            is_degenerate, degenerate_reason = detect_degenerate_output(planner_content)
            if not is_validate and is_degenerate:
                trajectory_flags["degenerate_output"] = True
                logger.warning(
                    f"[PS Abnormal] Request {req.request_id} search_round={search_round_count}: "
                    f"Planner degenerate output detected - {degenerate_reason}, "
                    f"content_len={len(planner_content)}"
                )
                break

            # Parse Planner output
            parsed_planner = self._parse_planner_output(planner_content)

            planner_data = {
                "role": "planner",
                "prompt_ids": planner_prompt_ids,
                "response_ids": planner_response_ids,
                "attention_mask": p_attention_mask,
                "position_ids": p_position_ids,
                "loss_mask": p_loss_mask,
                "batch_data_id": req.batch_data_id,
                "request_id": req.request_id,
                "round": search_round_count,
                "data_sample_index": data_sample_index,
                "messages": [
                    {"role": "user", "content": planner_prompt},
                    {"role": "assistant", "content": planner_content},
                ],
                "think": parsed_planner["think"],
                "answer": parsed_planner.get("answer"),
                "summary": global_summary,
                "raw_content": planner_content,
                "abnormal_flags": dict(trajectory_flags),
                "trajectory_metrics": {
                    "search_rounds": search_round_count,
                    "total_tool_calls": total_tool_calls,
                    "tool_call_attempts": total_tool_call_attempts,
                },
            }

            self.log_manager.log(
                log_path,
                event="ps_planner_complete",
                duration=planner_gen_end - planner_gen_start,
                extra={
                    "request_id": req.request_id,
                    "search_round": search_round_count,
                    "action_type": parsed_planner["action_type"],
                    "response_tokens": p_response_len,
                    "parse_error": parsed_planner.get("parse_error"),
                },
                workid=self._rank,
                step=self.step,
            )

            round_data_list.append(planner_data)
            data_sample_index += 1

            # ---- Decision: answer / tool_call / unknown ----

            # Case 1: the Planner emitted <answer> -> normal end
            if parsed_planner["action_type"] == "answer":
                self.log_manager.log(
                    log_path,
                    event="ps_planner_answer",
                    extra={
                        "request_id": req.request_id,
                        "search_round": search_round_count,
                    },
                    workid=self._rank,
                    step=self.step,
                )

                break

            # Case 2: the Planner's output format is abnormal -> Break + 0 reward (model abnormality)
            if parsed_planner["action_type"] == "unknown":
                trajectory_flags["tool_parse_error"] = True
                logger.warning(
                    f"[PS Abnormal Break] Request {req.request_id} search_round={search_round_count}: "
                    f"Planner output parse failed - {parsed_planner.get('parse_error', 'unknown')}"
                )
                print(
                    f"[DEBUG PS] Abnormal Break (tool_parse_error): "
                    f"Planner output has no <answer> or <tool_call> tags"
                )
                break

            # Case 3: the Planner emitted <tool_call> -> a search must be executed
            assert parsed_planner["action_type"] == "tool_call"

            # ---- Check whether the max search rounds have been exceeded ----
            # P(search)+S counts as one round. If search_round_count >= max_search_rounds,
            # enough search rounds have been done and we should not continue.
            # At this point the Planner emitted a <tool_call>, so we force the final answer.
            if search_round_count >= max_search_rounds:
                # Remove the useless tool_call planner_data that was just added:
                # this response is a <tool_call> emitted when the model should have answered,
                # the search will not be executed; training on it would encourage the model
                # to keep searching after exceeding the limit.
                round_data_list.pop()
                data_sample_index -= 1

                # ---- Training vs validation branch ----
                # During training: break directly, set exceed_max_turns=True, and let the
                # reward=0 sample participate in training
                # During validation: force the generation of a final answer, so every question
                # gets an answer and evaluation stays fair
                if not is_validate:
                    # Training mode: don't force an answer, just mark exceed_max_turns
                    trajectory_flags["exceed_max_turns"] = True
                    logger.info(
                        f"[PS Pipeline] Request {req.request_id}: "
                        f"Reached max search rounds ({search_round_count} >= {max_search_rounds}), "
                        f"training mode - marking exceed_max_turns (reward=0)"
                    )
                    break

                # ---- Validation mode: force the generation of a final answer ----
                logger.info(
                    f"[PS Pipeline] Request {req.request_id}: "
                    f"Reached max search rounds ({search_round_count} >= {max_search_rounds}), "
                    f"validation mode - forcing final answer generation"
                )

                # ---- Force the generation of a final answer ----
                # Use a dedicated final-answer prompt in place of the normal Planner prompt
                forced_answer_prompt = self.FINAL_ANSWER_PROMPT_TEMPLATE.format(
                    TIME=datetime.now().strftime("%Y-%m-%d"),
                    max_rounds=max_search_rounds,
                    question=question,
                    summary=global_summary if global_summary else "No information gathered.",
                )

                try:
                    forced_output = await self._ps_generate_single_turn(
                        prompt_text=forced_answer_prompt,
                        request_sampling_params=request_sampling_params,
                        request_id=f"{req.request_id}_P_forced_answer",
                        max_model_len_override=effective_max_model_len,
                    )
                except Exception as e:
                    logger.error(
                        f"[PS Pipeline] Request {req.request_id} forced answer generation failed: "
                        f"{type(e).__name__}: {e}"
                    )
                    trajectory_flags["exceed_max_turns"] = True
                    break

                forced_content = forced_output["text"]
                forced_prompt_ids = forced_output["prompt_ids"]
                forced_response_ids = torch.tensor(
                    [forced_output["output_ids"]], dtype=torch.long
                )

                # Extract the answer from the forced generation
                forced_parsed = self._parse_planner_output(forced_content)
                forced_answer = forced_parsed.get("answer")

                if not forced_answer:
                    # The model still didn't emit <answer>; mark it as exceed_max_turns
                    trajectory_flags["exceed_max_turns"] = True
                    logger.warning(
                        f"[PS Pipeline] Request {req.request_id}: "
                        f"Forced answer generation failed to produce <answer> tag"
                    )
                    break

                # Build the planner_data for the forced answer
                # Key point: during the rollout the forced_answer_prompt is used to generate,
                # but during update it must be replaced with the normal Planner prompt.
                # The "forced_answer" flag tells the downstream flow to swap the prompt.
                fa_prompt_len = forced_prompt_ids.shape[-1]
                fa_response_len = forced_response_ids.shape[-1]
                fa_total_len = fa_prompt_len + fa_response_len

                # Check the token length limit
                if fa_total_len > effective_max_model_len:
                    trajectory_flags["exceed_max_tokens"] = True
                    logger.warning(
                        f"[PS Pipeline] Request {req.request_id}: forced answer exceed_max_tokens "
                        f"{fa_total_len} > {effective_max_model_len}"
                    )

                fa_attention_mask = torch.ones(1, fa_total_len, dtype=torch.long)
                fa_position_ids = torch.arange(fa_total_len, dtype=torch.long).unsqueeze(0)
                fa_loss_mask = torch.cat([
                    torch.zeros(1, fa_prompt_len, dtype=torch.long),
                    torch.ones(1, fa_response_len, dtype=torch.long),
                ], dim=-1)

                # Build the normal Planner prompt used as a replacement during update
                # Replace forced_answer_prompt with the normal Planner prompt (only the summary
                # is replaced with the latest one)
                normal_planner_prompt = self._build_planner_prompt(
                    question=question,
                    summary=global_summary,
                )

                forced_planner_data = {
                    "role": "planner",
                    "prompt_ids": forced_prompt_ids,
                    "response_ids": forced_response_ids,
                    "attention_mask": fa_attention_mask,
                    "position_ids": fa_position_ids,
                    "loss_mask": fa_loss_mask,
                    "batch_data_id": req.batch_data_id,
                    "request_id": req.request_id,
                    "round": search_round_count,
                    "data_sample_index": data_sample_index,
                    "messages": [
                        {"role": "user", "content": forced_answer_prompt},
                        {"role": "assistant", "content": forced_content},
                    ],
                    "think": forced_parsed.get("think", ""),
                    "answer": forced_answer,
                    "summary": global_summary,
                    "raw_content": forced_content,
                    "abnormal_flags": dict(trajectory_flags),
                    "trajectory_metrics": {
                        "search_rounds": search_round_count,
                        "total_tool_calls": total_tool_calls,
                        "tool_call_attempts": total_tool_call_attempts,
                    },
                    # Flag: this is a forced answer; the prompt must be swapped during update
                    "forced_answer": True,
                    # Store the normal Planner prompt, used as a replacement during update
                    "normal_planner_prompt": normal_planner_prompt,
                }

                round_data_list.append(forced_planner_data)
                data_sample_index += 1

                # Mark exceed_max_turns but **do not discard** (there is an answer now)
                trajectory_flags["exceed_max_turns"] = True
                # New flag: exceed_max_turns, but a forced answer was produced successfully
                trajectory_flags["forced_answer_generated"] = True

                self.log_manager.log(
                    log_path,
                    event="ps_forced_answer_generated",
                    extra={
                        "request_id": req.request_id,
                        "search_round": search_round_count,
                        "answer_length": len(forced_answer),
                    },
                    workid=self._rank,
                    step=self.step,
                )
                break

            # ---- Parse the queries ----
            queries = parsed_planner.get("queries", [])

            # No queries were parsed -> Break + 0 reward (model abnormality)
            if not queries:
                trajectory_flags["tool_parse_error"] = True
                parse_err = parsed_planner.get("parse_error", "no queries extracted")
                logger.warning(
                    f"[PS Abnormal Break] Request {req.request_id} search_round={search_round_count}: "
                    f"tool_call parsed but no valid queries - {parse_err}"
                )
                print(
                    f"[DEBUG PS] Abnormal Break (tool_parse_error): no queries in tool_call"
                )
                break

            total_tool_call_attempts += 1
            total_tool_calls += len(queries)

            # ---- Duplicate-query detection ----
            repeated_queries = []
            for q in queries:
                normalized_q = q.strip().lower()
                if normalized_q in [h.strip().lower() for h in query_history]:
                    repeated_queries.append(q)
                    trajectory_flags["repeated_query"] = True
            query_history.extend(queries)

            if repeated_queries:
                logger.warning(
                    f"[PS Pipeline] Request {req.request_id} search_round={search_round_count}: "
                    f"Repeated queries detected: {repeated_queries}"
                )

            # ---- Detect too many queries in a single round ----
            max_queries_per_turn = self.config.multi_turn.get('max_tool_calls_per_turn', 5)
            if len(queries) > max_queries_per_turn:
                trajectory_flags["excessive_tool_calls_per_turn"] = True
                trajectory_flags["excessive_rounds"].append(search_round_count)  # record the round that exceeded the limit
                logger.warning(
                    f"[PS Pipeline] Request {req.request_id} search_round={search_round_count}: "
                    f"Excessive queries: {len(queries)} > {max_queries_per_turn}"
                )

            # ---- Break + discard abnormality decision ----
            # Core logic: as long as the model emitted a <tool_call>, it must have valid queries.
            # tool_parse_error -> no complete trajectory, discard directly
            # excessive_tool_calls_per_turn -> no break, keep going; only the offending round's
            # reward is zeroed at the reward layer
            should_break_with_zero_reward = (
                trajectory_flags["tool_parse_error"]  # parse error -> discard
                # excessive_tool_calls_per_turn: no longer breaks, let the model keep reasoning
                # repeated_query: no break
            )

            if should_break_with_zero_reward:
                abnormal_reasons = []
                if trajectory_flags["tool_parse_error"]:
                    abnormal_reasons.append("tool_parse_error")
                logger.warning(
                    f"[PS Abnormal Break] Request {req.request_id} search_round={search_round_count}: "
                    f"Breaking due to: {', '.join(abnormal_reasons)} (trajectory will be discarded)"
                )
                print(
                    f"[DEBUG PS] Abnormal Break (discard): {', '.join(abnormal_reasons)}"
                )
                break

            # ---- Step 2: Execute search tool ----
            tool_call_raw_str = parsed_planner.get("tool_call_raw", "")
            tool_call_parsed = parsed_planner.get("tool_call_parsed", {})

            torch.cuda.synchronize()
            tool_exec_start = time.time()

            tool_response_text = ""
            doc_count = 0
            try:
                # Find the search tool
                search_tool_name = None
                if tool_call_parsed and isinstance(tool_call_parsed, dict):
                    search_tool_name = tool_call_parsed.get("name", "search")

                if search_tool_name and search_tool_name in self._tool_map:
                    tool = self._tool_map[search_tool_name]
                    arguments = tool_call_parsed.get("arguments", {})

                    tool_result = await tool.execute(
                        req.request_id,
                        arguments,
                        **req.tools_kwargs.get(search_tool_name, {}).get("execute_kwargs", {"prompt_str": prompt_str}),
                    )
                    resp, reward, tool_metrics = tool_result
                    doc_count = tool_metrics.get("total_results", 0)

                    # Only an API-level error is marked search_error (an environment failure;
                    # the trajectory is discarded)
                    # doc_count==0 is a query-quality issue, not an environment failure; the model
                    # should learn naturally from the "No results found" hint
                    api_error = tool_metrics.get("api_request_error")
                    api_status = tool_metrics.get("status", "")
                    if api_error or api_status == "api_error":
                        trajectory_flags["search_error"] = True
                        logger.warning(
                            f"[PS Pipeline] Request {req.request_id} search_round={search_round_count}: "
                            f"Search API error - doc_count={doc_count}, "
                            f"status={api_status}, api_error={api_error}"
                        )
                    elif doc_count == 0:
                        logger.info(
                            f"[PS Pipeline] Request {req.request_id} search_round={search_round_count}: "
                            f"Search returned 0 results (status={api_status}), "
                            f"model will see 'No results found' prompt"
                        )

                    # Extract text from ToolResponse
                    if resp and resp.text:
                        tool_response_text = resp.text
                    else:
                        tool_response_text = ""
                else:
                    # Unknown tool name -> model abnormality, Break + 0 reward
                    trajectory_flags["tool_parse_error"] = True
                    logger.warning(
                        f"[PS Abnormal Break] Request {req.request_id} search_round={search_round_count}: "
                        f"Unknown tool name '{search_tool_name}', available: {list(self._tool_map.keys())}"
                    )
                    print(
                        f"[DEBUG PS] Abnormal Break (tool_parse_error): unknown tool '{search_tool_name}'"
                    )
                    break
            except Exception as e:
                # The tool raised an exception -> environment failure (discarded directly,
                # not the model's fault)
                logger.error(
                    f"[PS Pipeline] Request {req.request_id} search_round={search_round_count}: "
                    f"Tool execution failed - {type(e).__name__}: {e}"
                )
                traceback.print_exc()
                trajectory_flags["search_error"] = True
                break

            torch.cuda.synchronize()
            tool_exec_end = time.time()

            self.log_manager.log(
                log_path,
                event="ps_tool_execution",
                duration=tool_exec_end - tool_exec_start,
                extra={
                    "request_id": req.request_id,
                    "search_round": search_round_count,
                    "queries": queries,
                    "doc_count": doc_count,
                    "search_error": trajectory_flags["search_error"],
                },
                workid=self._rank,
                step=self.step,
            )

            # ---- Step 3: Synthesizer ----
            synthesizer_prompt = self._build_synthesizer_prompt(
                question=question,
                global_summary=global_summary,
                q_agent_think=parsed_planner["think"],
                tool_call=tool_call_raw_str,
                doc_count=doc_count,
                documents_formatted=tool_response_text,
            )

            self.log_manager.log(
                log_path,
                event="ps_synthesizer_start",
                extra={
                    "request_id": req.request_id,
                    "search_round": search_round_count,
                    "data_sample_index": data_sample_index,
                    "prompt_length": len(synthesizer_prompt),
                },
                workid=self._rank,
                step=self.step,
            )

            torch.cuda.synchronize()
            synth_gen_start = time.time()
            try:
                synth_output = await self._ps_generate_single_turn(
                    prompt_text=synthesizer_prompt,
                    request_sampling_params=request_sampling_params,
                    request_id=f"{req.request_id}_S_{search_round_count}",
                    max_new_tokens_override=synth_max_new_tokens,
                    max_model_len_override=effective_max_model_len,
                )
            except Exception as e:
                # Engine generation exception (environment failure)
                logger.error(
                    f"[PS Pipeline] Request {req.request_id} Synthesizer generation failed "
                    f"at search_round={search_round_count}: {type(e).__name__}: {e}"
                )
                traceback.print_exc()
                trajectory_flags["search_error"] = True
                break
            torch.cuda.synchronize()
            synth_gen_end = time.time()

            synth_content = synth_output["text"]
            synth_prompt_ids = synth_output["prompt_ids"]
            synth_response_ids = torch.tensor(
                [synth_output["output_ids"]], dtype=torch.long
            )

            # Build Synthesizer data sample
            s_prompt_len = synth_prompt_ids.shape[-1]
            s_response_len = synth_response_ids.shape[-1]
            s_total_len = s_prompt_len + s_response_len
            s_attention_mask = torch.ones(1, s_total_len, dtype=torch.long)
            s_position_ids = torch.arange(s_total_len, dtype=torch.long).unsqueeze(0)
            s_loss_mask = torch.cat([
                torch.zeros(1, s_prompt_len, dtype=torch.long),
                torch.ones(1, s_response_len, dtype=torch.long),
            ], dim=-1)

            # Check the token length limit (only flagged and broken on during training, not
            # limited during validation)
            if not is_validate and s_total_len > effective_max_model_len:
                trajectory_flags["exceed_max_tokens"] = True
                logger.warning(
                    f"[PS Pipeline] Request {req.request_id}: Synthesizer exceed_max_tokens "
                    f"{s_total_len} > {effective_max_model_len}"
                )
                break

            # Check for degenerate output (only detected and broken on during training, not
            # interrupted during validation)
            is_degenerate_s, degenerate_reason_s = detect_degenerate_output(synth_content)
            if not is_validate and is_degenerate_s:
                trajectory_flags["degenerate_output"] = True
                logger.warning(
                    f"[PS Abnormal] Request {req.request_id} search_round={search_round_count}: "
                    f"Synthesizer degenerate output detected - {degenerate_reason_s}, "
                    f"content_len={len(synth_content)}"
                )
                break

            # Parse Synthesizer output
            parsed_synth = self._parse_synthesizer_output(synth_content)
            global_summary = parsed_synth["summary"]

            if parsed_synth.get("parse_error"):
                logger.warning(
                    f"[PS Pipeline] Request {req.request_id} search_round={search_round_count}: "
                    f"Synthesizer parse warning - {parsed_synth['parse_error']}"
                )

            synthesizer_data = {
                "role": "synthesizer",
                "prompt_ids": synth_prompt_ids,
                "response_ids": synth_response_ids,
                "attention_mask": s_attention_mask,
                "position_ids": s_position_ids,
                "loss_mask": s_loss_mask,
                "batch_data_id": req.batch_data_id,
                "request_id": req.request_id,
                "round": search_round_count,
                "data_sample_index": data_sample_index,
                "messages": [
                    {"role": "user", "content": synthesizer_prompt},
                    {"role": "assistant", "content": synth_content},
                ],
                "think": parsed_planner["think"],
                "answer": None,
                "summary": planner_data["summary"],
                "synthesizer_output": global_summary,
                "raw_content": synth_content,
                "tool_response": tool_response_text,
                "doc_count": doc_count,
                "tool_call_raw": tool_call_raw_str,
                "abnormal_flags": dict(trajectory_flags),
                "trajectory_metrics": {
                    "search_rounds": search_round_count,
                    "total_tool_calls": total_tool_calls,
                    "tool_call_attempts": total_tool_call_attempts,
                },
            }

            self.log_manager.log(
                log_path,
                event="ps_synthesizer_complete",
                duration=synth_gen_end - synth_gen_start,
                extra={
                    "request_id": req.request_id,
                    "search_round": search_round_count,
                    "response_tokens": s_response_len,
                    "summary_length": len(global_summary),
                    "parse_error": parsed_synth.get("parse_error"),
                },
                workid=self._rank,
                step=self.step,
            )

            round_data_list.append(synthesizer_data)
            data_sample_index += 1

            # ---- A full search round (P(search)+S) completed, increment the counter ----
            search_round_count += 1

            torch.cuda.synchronize()
            round_end_time = time.time()
            self.log_manager.log(
                log_path,
                event="ps_round_complete",
                duration=round_end_time - round_start_time,
                extra={
                    "request_id": req.request_id,
                    "search_round": search_round_count - 1,
                    "total_samples_so_far": data_sample_index,
                },
                workid=self._rank,
                step=self.step,
            )

            # ---- Safety net: a hard cap on tool_call_attempts (consistent with the original code) ----
            if total_tool_call_attempts > 100:
                trajectory_flags["exceed_max_turns"] = True
                logger.warning(
                    f"[PS Abnormal] Request {req.request_id}: "
                    f"Exceeded hard limit of tool call attempts ({total_tool_call_attempts} > 100)"
                )
                break

        # ==========================================
        # === Post-loop abnormality check ===
        # ==========================================
        # Note: exceed_max_turns is already set inside the loop (search_round_count >=
        # max_search_rounds)
        # This is a final consistency check
        if search_round_count >= max_search_rounds and not trajectory_flags["exceed_max_turns"]:
            # If the loop exited because search_round_count >= max_search_rounds,
            # and the Planner gave a normal answer on the last round, then exceed_max_turns
            # doesn't need to be set.
            # But if the last round_data isn't a planner-answer, it exited abnormally.
            last_data = round_data_list[-1] if round_data_list else None
            if last_data and not (last_data["role"] == "planner" and last_data.get("answer")):
                trajectory_flags["exceed_max_turns"] = True

        # ==========================================
        # === Release tools and collect reward scores ===
        # ==========================================
        # Note: sglang_rollout.py doesn't compute the final reward value;
        # it only passes the abnormality flags downstream to the Reward Manager.
        # The Reward Manager decides, based on these flags:
        #   - discarded (excluded from training): search_error / tool_parse_error /
        #     exceed_max_tokens / exceed_max_turns
        #   - offending round zeroed: excessive_tool_calls_per_turn (the specific round is
        #     recorded in excessive_rounds)
        #   - computed normally: repeated_query
        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(req.request_id, **req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(req.request_id, **req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        tool_reward_tasks = []
        for name in req.tools_kwargs.keys():
            tool = self._tool_map[name]
            tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
        tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = dict(tool_reward_scores)

        # ---- After a trajectory completes: launch async rubric scoring for normal trajectories ----
        # Only trajectories without abnormality flags are scored, to avoid wasting API calls on
        # exceed_max_turns/search_error etc.
        is_normal_trajectory = not any([
            trajectory_flags.get("search_error", False),
            trajectory_flags.get("tool_parse_error", False),
            trajectory_flags.get("exceed_max_tokens", False),
            trajectory_flags.get("exceed_max_turns", False),
            trajectory_flags.get("degenerate_output", False),
        ])

        if self._rubric_enabled and self._rubric_calculator is not None and not is_validate and is_normal_trajectory and round_data_list:
            loop = asyncio.get_event_loop()
            # Take this request's ground_truth (the same GT used on the reward manager side, so
            # the Judge input stays consistent)
            _rubric_gt = ""
            if hasattr(self, "_rubric_gt_map"):
                _rubric_gt = self._rubric_gt_map.get(req.request_id, "") or ""
            # Compute last_synth_round (consistent with the logic in
            # compute_trajectory_rubric_reward on the reward manager side)
            _last_synth_round = -1
            for _rd in round_data_list:
                if _rd.get("role", "") == "synthesizer":
                    _r = _rd.get("round", -1)
                    if isinstance(_r, int) and _r > _last_synth_round:
                        _last_synth_round = _r

            for rd in round_data_list:
                role = rd.get("role", "")
                rnd = rd.get("round", 0)
                sample_idx = rd.get("data_sample_index", -1)

                if role == "planner":
                    action_type = "answer" if rd.get("answer") else "tool_call"
                    planner_output = rd.get("answer", "") or rd.get("raw_content", "")
                    task = loop.run_in_executor(
                        None,
                        lambda _rnd=rnd, _summary=rd.get("summary", ""),
                               _think=rd.get("think", ""),
                               _output=planner_output,
                               _action=action_type, _q=question,
                               _gt=_rubric_gt: (
                            self._rubric_calculator.score_planner_round(
                                user_query=_q,
                                round_number=_rnd,
                                historical_context=_summary,
                                planner_think=_think,
                                planner_output=_output,
                                action_type=_action,
                                ground_truth=_gt,
                            )
                        ),
                    )
                    rubric_pending_tasks.append((task, "planner", rnd, sample_idx))

                elif role == "synthesizer":
                    _is_last = (rnd == _last_synth_round)
                    task = loop.run_in_executor(
                        None,
                        lambda _rnd=rnd, _prev_summary=rd.get("summary", ""),
                               _think=rd.get("think", ""),
                               _queries=rd.get("tool_call_raw", ""),
                               _results=rd.get("tool_response", ""),
                               _doc_count=rd.get("doc_count", 0),
                               _synth_out=rd.get("synthesizer_output", ""),
                               _q=question, _gt=_rubric_gt,
                               _last=_is_last: (
                            self._rubric_calculator.score_synthesizer_round(
                                user_query=_q,
                                round_number=_rnd,
                                previous_summary=_prev_summary,
                                planner_think=_think,
                                search_queries=_queries,
                                search_results=_results,
                                doc_count=_doc_count,
                                synthesizer_output=_synth_out,
                                ground_truth=_gt,
                                is_last_synth_round=_last,
                            )
                        ),
                    )
                    rubric_pending_tasks.append((task, "synthesizer", rnd, sample_idx))

            logger.info(
                f"[Rollout Rubric] Request {req.request_id}: "
                f"trajectory is normal, launching {len(rubric_pending_tasks)} rubric scoring tasks"
            )
        elif self._rubric_enabled and not is_validate and not is_normal_trajectory:
            logger.info(
                f"[Rollout Rubric] Request {req.request_id}: "
                f"trajectory is abnormal ({[k for k,v in trajectory_flags.items() if v]}), skipping rubric scoring"
            )

        # ---- Collect the async rubric scoring results ----
        # rubric_pending_tasks: [(asyncio.Task, role, round, data_sample_index), ...]
        # Build the data_sample_index -> rubric_score mapping
        rubric_scores_by_sample_idx = {}  # data_sample_index -> float
        if rubric_pending_tasks:
            rubric_start = time.time()
            n_success = 0
            n_fail = 0
            for task_future, role, rnd, sample_idx in rubric_pending_tasks:
                try:
                    scores, normalized_score, _prompt, _llm_output = await asyncio.wait_for(task_future, timeout=120.0)
                    rubric_scores_by_sample_idx[sample_idx] = normalized_score
                    n_success += 1
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[Rollout Rubric] Request {req.request_id} {role} round {rnd} "
                        f"rubric scoring timed out"
                    )
                    rubric_scores_by_sample_idx[sample_idx] = 0.2  # give a low score on timeout
                    n_fail += 1
                except Exception as e:
                    logger.warning(
                        f"[Rollout Rubric] Request {req.request_id} {role} round {rnd} "
                        f"rubric scoring failed: {e}"
                    )
                    rubric_scores_by_sample_idx[sample_idx] = 0.2
                    n_fail += 1

            rubric_duration = time.time() - rubric_start
            logger.info(
                f"[Rollout Rubric] Request {req.request_id}: "
                f"collected {n_success + n_fail} rubric scores "
                f"(success={n_success}, fail={n_fail}) in {rubric_duration:.1f}s"
            )

        # Clean up this request's GT cache, to prevent a cumulative memory leak
        if hasattr(self, "_rubric_gt_map"):
            self._rubric_gt_map.pop(req.request_id, None)

        # ---- Finally update the flags and metrics of all data samples ----
        for rd in round_data_list:
            rd["abnormal_flags"] = dict(trajectory_flags)
            rd["reward_scores"] = tool_reward_scores
            rd["trajectory_metrics"].update({
                "search_rounds": search_round_count,
                "total_tool_calls": total_tool_calls,
                "tool_call_attempts": total_tool_call_attempts,
                "query_history": query_history,
                "max_search_rounds": max_search_rounds,
            })
            # Store the precomputed rubric score
            sample_idx = rd.get("data_sample_index", -1)
            if sample_idx in rubric_scores_by_sample_idx:
                rd["precomputed_rubric_score"] = rubric_scores_by_sample_idx[sample_idx]

        torch.cuda.synchronize()
        request_end_time = time.time()
        self.log_manager.log(
            log_path,
            event="ps_pipeline_request_complete",
            duration=request_end_time - request_start_time,
            extra={
                "request_id": req.request_id,
                "search_rounds": search_round_count,
                "total_samples": len(round_data_list),
                "trajectory_flags": trajectory_flags,
            },
            workid=self._rank,
            step=self.step,
        )

        return round_data_list

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _req_level_generate_sequences_ps(self, prompts: DataProto, step: int = 0, **kwargs) -> DataProto:
        """PS Pipeline batch-level entry point.

        Generates multi-turn sequences using the Planner-Synthesizer pipeline.
        Each trajectory produces multiple independent data samples (one per round).

        The output DataProto contains all round-level samples from all requests,
        flattened into a single batch. Each sample is an independent training example
        with its own prompt and response.

        Args:
            prompts: DataProto containing input prompts.
            step: Global training step.

        Returns:
            DataProto with all round-level samples.
        """
        self.step = step

        log_path = None
        if self._tp_rank == 0:
            log_path = os.path.join(
                self.log_dir,
                f"step_{step}",
                f"worker_{self._rank}.jsonl"
            )

        torch.cuda.synchronize()
        start_time = time.time()
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device

        if self._tp_rank == 0:
            # Preprocess prompts to AsyncRolloutRequests
            torch.cuda.synchronize()
            preprocess_start_time = time.time()
            req_list = self._preprocess_prompt_to_async_rollout_requests(prompts)
            torch.cuda.synchronize()
            preprocess_end_time = time.time()

            if log_path:
                self.log_manager.log(
                    log_path,
                    event="ps_preprocessing_duration",
                    duration=preprocess_end_time - preprocess_start_time,
                    workid=self._rank,
                    step=self.step,
                )

            # Run PS pipeline for all requests
            loop = asyncio.get_event_loop()

            if is_validate:
                all_round_data = loop.run_until_complete(
                    asyncio.gather(
                        *[self._async_rollout_ps_pipeline(req, do_sample, is_validate, **kwargs) for req in req_list]
                    )
                )
            else:
                # Training mode with over-sample + cancel
                total_requests = len(req_list)
                target_completion = int(total_requests * (1 - self.config.over_sample_rate))
                completed_count = 0
                aborted_req_ids = set()

                async def ps_rollout_with_cancellation(req):
                    try:
                        return await self._async_rollout_ps_pipeline(req, do_sample, is_validate, **kwargs)
                    except asyncio.CancelledError:
                        aborted_req_ids.add(req.request_id)
                        return []  # Return empty list for cancelled requests

                async def run_with_cancellation():
                    nonlocal completed_count
                    all_tasks = [
                        asyncio.create_task(ps_rollout_with_cancellation(req)) for req in req_list
                    ]

                    try:
                        for completed_task in asyncio.as_completed(all_tasks):
                            await completed_task
                            completed_count += 1
                            if completed_count >= target_completion:
                                break
                    finally:
                        for t in all_tasks:
                            if not t.done():
                                t.cancel()
                        final_results = await asyncio.gather(*all_tasks, return_exceptions=True)
                        await self._engine.abort_request(abort_all=True)
                    return final_results

                torch.cuda.synchronize()
                async_start = time.time()
                all_round_data = loop.run_until_complete(run_with_cancellation())
                torch.cuda.synchronize()
                async_end = time.time()

                if log_path:
                    self.log_manager.log(
                        log_path,
                        event="ps_async_generate_duration",
                        duration=async_end - async_start,
                        workid=self._rank,
                        step=self.step,
                    )

            # Flatten all round data from all requests
            flattened_data = []
            for req_rounds in all_round_data:
                if isinstance(req_rounds, list):
                    flattened_data.extend(req_rounds)
                elif isinstance(req_rounds, Exception):
                    # Log non-CancelledError exceptions that were silently caught by return_exceptions=True
                    logger.warning(
                        f"[PS Pipeline] Task returned exception (silently skipped): "
                        f"{type(req_rounds).__name__}: {req_rounds}"
                    )
                # Skip exceptions from cancelled tasks

            # Sort by (batch_data_id, data_sample_index) to preserve trajectory order
            flattened_data.sort(key=lambda x: (x["batch_data_id"], x.get("data_sample_index", 0)))

        else:
            flattened_data = None

        # Barrier and broadcast
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()

        [flattened_data] = broadcast_pyobj(
            data=[flattened_data],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )

        if not flattened_data:
            # Edge case: no data produced, return empty DataProto
            logger.warning(f"[PS Pipeline] Step {step}: No data produced, returning empty DataProto")
            empty_batch = TensorDict(
                {
                    "prompts": torch.zeros(0, self.config.prompt_length, dtype=torch.long),
                    "responses": torch.zeros(0, self.config.response_length, dtype=torch.long),
                    "response_mask": torch.zeros(0, self.config.response_length, dtype=torch.long),
                    "input_ids": torch.zeros(0, self.config.prompt_length + self.config.response_length, dtype=torch.long),
                    "attention_mask": torch.zeros(0, self.config.prompt_length + self.config.response_length, dtype=torch.long),
                    "position_ids": torch.zeros(0, self.config.prompt_length + self.config.response_length, dtype=torch.long),
                },
                batch_size=0,
            )
            return DataProto(batch=empty_batch, non_tensor_batch={})

        # ==========================================
        # === Extract and pad data ===
        # ==========================================
        batch_size = len(flattened_data)
        prompt_ids_list = []
        response_ids_list = []
        messages_list = []
        reward_scores_list = []
        request_ids_list = []
        extra_info_list = []
        roles_list = []

        for rd in flattened_data:
            # Key point: for forced_answer samples, replace prompt_ids with the normal Planner prompt
            # This way, during update the model sees the same prompt as in a normal answer round
            if rd.get("forced_answer") and rd.get("normal_planner_prompt"):
                normal_prompt = rd["normal_planner_prompt"]
                # Re-tokenize using the chat template (consistent with _ps_generate_single_turn)
                chat_messages = [{"role": "user", "content": normal_prompt}]
                normal_prompt_text = self.processing_class.apply_chat_template(
                    chat_messages, tokenize=False, add_generation_prompt=True
                )
                normal_prompt_ids_raw = self.processing_class.encode(
                    normal_prompt_text, return_tensors="pt"
                )
                if isinstance(normal_prompt_ids_raw, torch.Tensor):
                    normal_prompt_ids_tensor = normal_prompt_ids_raw.squeeze(0).to(tgt_device)
                else:
                    normal_prompt_ids_tensor = torch.tensor(normal_prompt_ids_raw, dtype=torch.long, device=tgt_device)
                prompt_ids_list.append(normal_prompt_ids_tensor)
            else:
                prompt_ids_list.append(rd["prompt_ids"].to(tgt_device).squeeze(0))
            response_ids_list.append(rd["response_ids"].to(tgt_device).squeeze(0))
            messages_list.append({"messages": rd["messages"]})
            reward_scores_list.append(rd.get("reward_scores", {}))
            request_ids_list.append(rd["request_id"])
            roles_list.append(rd["role"])
            extra_info_list.append({
                "abnormal_flags": rd["abnormal_flags"],
                "trajectory_metrics": rd["trajectory_metrics"],
                "role": rd["role"],
                "round": rd["round"],
                "data_sample_index": rd.get("data_sample_index", 0),
                "think": rd.get("think", ""),
                "answer": rd.get("answer"),
                "summary": rd.get("summary", ""),
                "raw_content": rd.get("raw_content", ""),
                # The precomputed rubric score (from the async scoring during the rollout)
                "precomputed_rubric_score": rd.get("precomputed_rubric_score"),
            })

        # Pad prompt_ids (left padding)
        # First truncate an over-long prompt_ids (keep the right side, i.e. the most recent tokens)
        prompt_ids_list = [p[-self.config.prompt_length:] for p in prompt_ids_list]
        prompt_ids = pad_sequence(
            prompt_ids_list, batch_first=True, padding_value=self.pad_token_id, padding_side="left"
        )
        if prompt_ids.shape[-1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True)

        # Pad response_ids (right padding)
        # First truncate an over-long response_ids (keep the left side, i.e. the earliest tokens)
        response_ids_list = [r[:self.config.response_length] for r in response_ids_list]
        response_ids = pad_sequence(response_ids_list, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[-1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)

        # Build attention masks
        prompt_attention_mask = (prompt_ids != self.pad_token_id).long()
        response_attention_mask = (response_ids != self.pad_token_id).long()

        # Build position_ids
        prompt_position_ids = torch.zeros_like(prompt_ids, dtype=torch.long)
        for i in range(batch_size):
            valid_positions = prompt_attention_mask[i].sum().item()
            start_pos = prompt_ids.shape[-1] - valid_positions
            prompt_position_ids[i, start_pos:] = torch.arange(valid_positions, dtype=torch.long)

        response_position_ids = torch.zeros_like(response_ids, dtype=torch.long)
        for i in range(batch_size):
            last_prompt_pos = prompt_position_ids[i].max().item()
            valid_resp_positions = response_attention_mask[i].sum().item()
            response_position_ids[i, :valid_resp_positions] = torch.arange(
                last_prompt_pos + 1,
                last_prompt_pos + 1 + valid_resp_positions,
                dtype=torch.long,
            )

        # Build loss mask: all 1s for response (no tool_response in PS pipeline)
        response_loss_mask = response_attention_mask.clone()  # 1 for real tokens, 0 for padding

        # Concatenate prompt and response
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)

        # Build TensorDict
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_loss_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # Build non_tensor_batch
        batch_statistics = {
            "batch_size": batch_size,
            "planner_count": sum(1 for r in roles_list if r == "planner"),
            "synthesizer_count": sum(1 for r in roles_list if r == "synthesizer"),
            "pipeline_mode": "ps_pipeline",
        }

        # Extract batch_data_id for each round sample (maps back to the index in repeated gen_batch)
        batch_data_id_list = [rd["batch_data_id"] for rd in flattened_data]

        non_tensor_batch = {
            "messages": np.array(messages_list, dtype=object),
            "reward_scores": np.array(reward_scores_list, dtype=object),
            "request_id": np.array(request_ids_list, dtype=object),
            "rollout_extra_info": np.array(extra_info_list, dtype=object),
            "batch_statistics": np.array([batch_statistics] * batch_size, dtype=object),
            "batch_data_id": np.array(batch_data_id_list, dtype=np.int64),
        }

        # Flush cache
        if self._engine is not None and self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())

        result = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

        torch.cuda.synchronize()
        total_end_time = time.time()

        if self._tp_rank == 0 and log_path:
            self.log_manager.log(
                log_path,
                event="ps_total_step_duration",
                duration=total_end_time - start_time,
                extra={
                    "total_samples": batch_size,
                    "planner_samples": batch_statistics["planner_count"],
                    "synthesizer_samples": batch_statistics["synthesizer_count"],
                },
                workid=self._rank,
                step=self.step,
            )

        return result
