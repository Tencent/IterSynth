#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intermediate-process reward - V2 C-GRPO version

This module implements reward computation based on the CaRR
(Citation-aware Rubric Reward) framework.

Core features (three-step evaluation):
1. Step 1: Hidden Entity Identification - identify hidden entities in the Agent's answer
2. Step 2: Citation-based Rubric Judgment - verify whether constraints are backed by citations
3. Step 3: Evidence Connectivity - check the connectivity of the evidence chain (BFS starting from the answer entity)

Final reward formula (C-GRPO):
R_i = (1 - alpha) * R_o^(H_i) + alpha * R_o^(H_i) * R_hat_r^(H_i)

Where:
- R_o: Outcome Reward (whether the correct answer was found, 0 or 1)
- R_r: Rubric Reward (reasoning quality, |R_connect| / |R_q|)
- R_hat_r: within-group normalized Rubric Reward

Version: 2.0.0
"""

import logging
import os
import re
import traceback
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Tuple, Set

import json_repair
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Prompt templates
# ============================================================

ENTITY_IDENTIFICATION_PROMPT = """Task Description
You will receive the following inputs:
1. A complex multi-hop question.
2. A list of single-hop constraints, decomposed from the original question.
• The final answer entity is labeled as <E0>.
• Intermediate entities are labeled as <E1>, <E2>, <E3>, and so on.
3. An AI assistant's response to the multi-hop question.
Your Task
Extract the explicitly stated real identities of <E0>, <E1>, <E2>, . . . from the assistant's response. Provide an
analysis first, then return a JSON object with one key per entity label.
Output Format
## Analysis
{{Explain, for each entity label, whether the assistant's response clearly
and explicitly provides its real identity. State the actual name or value
if it is explicitly mentioned; otherwise, indicate that the identity is
not clearly stated.}}
## Final JSON-format Summary
```json
{{
"E0": {{actual identity from assistant's response or null}},
"E1": {{actual identity from assistant's response or null}},
"E2": {{actual identity from assistant's response or null}},
...
}}
```
Important Rules
• Only use information that is explicitly and unambiguously stated in the assistant's response.
• Do not infer, guess, or deduce entity identities beyond what is explicitly provided.
• If the assistant's response does not clearly identify an entity, set its value to null.
• Follow the output format exactly.
[Begin of Question]
{question}
[End of Question]
[Begin of Constraints]
{constraints}
[End of Constraints]
[Begin of Assistant's Response]
{response}
[End of Assistant's Response]"""


CITATION_RUBRIC_JUDGMENT_PROMPT = """You will receive:
1. The contents of several webpages.
2. Several single-hop factual statements: S1, S2, ..., Sn.
Your task is to determine whether each statement is fully supported by the provided webpage contents.
For each statement:
• Find the exact evidence from the webpage contents that supports or contradicts it.
• Explain clearly why the statement is or is not fully supported, citing relevant parts of the provided text.
• List the URLs of webpages where the supporting evidence was found.
• Conclude with a judgment: Fully Supported: yes or Fully Supported: no.
At the end, summarize your results in a JSON object mapping each statement label (S1, S2, ...) to a boolean value (true
for fully supported, false for not fully supported).
Output Format:
## Supportness Analysis of S1
Explanation: {{Clearly explain why S1 is or is not fully supported
according to the given webpage contents}}
Evidence URLs: {{List of URLs containing evidence used in your explanation}}
Fully Supported: {{yes/no}}
## Supportness Analysis of S2
Explanation: {{Clearly explain why S2 is or is not fully supported
according to the given webpage contents}}
Evidence URLs: {{List of URLs containing evidence used in your explanation}}
Fully Supported: {{yes/no}}
...
## Final JSON-format Summary
**CRITICAL: You MUST output a complete valid JSON object at the end. Include ALL statements (S1, S2, S3, ...) in the JSON.**
```json
{{
"S1": {{true/false}},
"S2": {{true/false}},
"S3": {{true/false}},
...
}}
```
—
[Begin of Webpage Contents]
{context}
[End of Webpage Contents]
[Begin of Statements]
{statements}
[End of Statements]"""


# ============================================================
# Constants
# ============================================================

# Max citation count limit (prevents the Agent from hacking the reward via excessive citations)
MAX_CITATION_COUNT = 20


# ============================================================
# Error code definitions (for the fault-tolerance mechanism)
# ============================================================

class RubricRewardError:
    """Error codes for Rubric Reward computation."""
    SUCCESS = "success"
    NO_ANSWER_TAG = "no_answer_tag"  # No <answer> tag in the answer
    NO_CITATION = "no_citation"  # No [citation:x]-format citation in the answer
    NO_TOOL_RESPONSE = "no_tool_response"  # No tool_response in the trajectory
    CITATION_NOT_FOUND = "citation_not_found"  # The cited citation could not be found in tool_response
    NO_CONSTRAINTS = "no_constraints"  # No constraints in the ground truth
    LLM_CALL_FAILED = "llm_call_failed"  # LLM call failed
    PARSE_ERROR = "parse_error"  # Parse error


class AbnormalTrajectoryType:
    """Abnormal trajectory type definitions.

    Abnormalities are handled in three tiers:
    1. Discard (excluded from training):
       - SEARCH_ERROR: search API failure (environment issue)
       - EXCEED_MAX_TOKENS: exceeded the token limit (data truncated / incomplete)
       - DEGENERATE_OUTPUT: degenerate output (repetition/garbage/abnormal repeats, harmful to training)
    2. Give reward 0 (participates in training, the model learns to avoid it):
       - EXCEED_MAX_TURNS: exceeded the max search rounds without answering
       - TOOL_PARSE_ERROR: output-format parsing failure
    3. Offending round zeroed (trajectory kept, only that round's reward is zeroed):
       - EXCESSIVE_TOOL_CALLS: too many queries in a single round
    4. Handled normally (does not affect reward):
       - REPEATED_QUERY: duplicate query
    """
    TOOL_PARSE_ERROR = "tool_parse_error"  # Output-format parsing failure (reward 0, participates in training)
    EXCEED_MAX_TURNS = "exceed_max_turns"  # Exceeded the max search rounds (reward 0, participates in training)
    EXCEED_MAX_TOKENS = "exceed_max_tokens"  # Exceeded the max token count (discarded, data truncated)
    EXCESSIVE_TOOL_CALLS = "excessive_tool_calls_per_turn"  # Too many queries in one round (that round's reward is zeroed)
    REPEATED_QUERY = "repeated_query"  # Duplicate query (reward computed normally)
    SEARCH_ERROR = "search_error"  # Search API failure (discarded, environment issue)
    DEGENERATE_OUTPUT = "degenerate_output"  # Degenerate output: repetition/garbage/abnormal repeats (discarded, harmful to training)


# ============================================================
# Helper functions
# ============================================================

def extract_answer_from_trajectory(trajectory: str) -> Tuple[str, bool]:
    """Extract the final answer inside the <answer> tag from a trajectory.

    Args:
        trajectory: the Agent's full trajectory

    Returns:
        (the extracted answer text, whether extraction succeeded).
    """
    # Match the <answer>...</answer> tag
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, trajectory, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip(), True

    # If there is no <answer> tag, return failure
    return "", False


def extract_citation_indexes(answer_text: str, unique: bool = False, sorted_result: bool = False) -> Tuple[List[int], bool]:
    """Parse citation document indexes from the final answer.

    Per the paper's requirement, only extracts [citation:x]-format
    citations from the answer, and takes at most the first
    MAX_CITATION_COUNT (20 by default).

    Args:
        answer_text: answer text containing citation markers
        unique: whether to deduplicate; default False (preserves the order of appearance, used to take the first 20)
        sorted_result: whether to sort the result; default False

    Returns:
        (the list of citation indexes, whether any citation was successfully extracted)
        e.g. ([1, 2, 3, 6, 8, 13, 18], True)
    """
    # Match the [citation:number] pattern
    citation_pattern = r'\[citation:(\d+)\]'

    # Find all matches (in order of appearance)
    matches = re.findall(citation_pattern, answer_text)

    if not matches:
        return [], False

    # Convert to integers
    citation_indexes = [int(idx) for idx in matches]

    # Take the first MAX_CITATION_COUNT (before deduplication, preserving order of appearance)
    citation_indexes = citation_indexes[:MAX_CITATION_COUNT]

    # Process the result based on the parameters
    if unique:
        # Deduplicate while preserving order
        seen = set()
        unique_indexes = []
        for idx in citation_indexes:
            if idx not in seen:
                seen.add(idx)
                unique_indexes.append(idx)
        citation_indexes = unique_indexes

    if sorted_result:
        citation_indexes = sorted(citation_indexes)

    return citation_indexes, True


def extract_tool_responses_from_trajectory(trajectory: str) -> Tuple[str, bool]:
    """Extract and concatenate the contents of all <tool_response> tags from a trajectory.

    Args:
        trajectory: the Agent's full trajectory

    Returns:
        (the concatenated tool_response content, whether extraction succeeded).
    """
    # Match several possible tool_response formats
    patterns = [
        r'<tool_response>(.*?)</tool_response>',
        r'<\|tool_response\|>(.*?)<\|/tool_response\|>',
        r'<\|observation\|>(.*?)<\|/observation\|>',
    ]

    all_responses = []
    for pattern in patterns:
        matches = re.findall(pattern, trajectory, re.DOTALL)
        all_responses.extend(matches)

    if not all_responses:
        return "", False

    return '\n'.join(all_responses), True


def parse_webpage_from_web_tpl(tool_response_text: str, citation_index: int) -> Optional[Dict[str, str]]:
    """Parse the webpage content for a given index from tool_response text.

    WEB_TPL format:
    [webpage {idx} begin]
    [webpage title]
    {title}
    [webpage url]
    {domain}
    [webpage date published]
    {date}
    [webpage authoritativeness]
    {auth}
    [webpage content begin]
    {passage}
    [webpage content end]
    [webpage {idx} end]

    Note: the idx in WEB_TPL corresponds directly to the x in [citation:x].

    Args:
        tool_response_text: the concatenated tool_response text
        citation_index: the citation index (the x extracted from [citation:x], corresponds directly to WEB_TPL's idx)

    Returns:
        A dict containing idx, title, url, date, auth, content, or None if not found.
    """
    # Build a regex to match the webpage for the specific index
    # citation_index corresponds directly to the idx in WEB_TPL
    webpage_pattern = rf'\[webpage {citation_index} begin\](.*?)\[webpage {citation_index} end\]'
    match = re.search(webpage_pattern, tool_response_text, re.DOTALL)

    if not match:
        return None

    webpage_content = match.group(1)

    # Parse each field
    result: Dict[str, str] = {
        'idx': str(citation_index),
        'title': '',
        'url': '',
        'date': '',
        'auth': '',
        'content': ''
    }

    # Extract title
    title_pattern = r'\[webpage title\]\s*(.*?)\s*\[webpage url\]'
    title_match = re.search(title_pattern, webpage_content, re.DOTALL)
    if title_match:
        result['title'] = title_match.group(1).strip()

    # Extract url
    url_pattern = r'\[webpage url\]\s*(.*?)\s*\[webpage date published\]'
    url_match = re.search(url_pattern, webpage_content, re.DOTALL)
    if url_match:
        result['url'] = url_match.group(1).strip()

    # Extract date
    date_pattern = r'\[webpage date published\]\s*(.*?)\s*\[webpage authoritativeness\]'
    date_match = re.search(date_pattern, webpage_content, re.DOTALL)
    if date_match:
        result['date'] = date_match.group(1).strip()

    # Extract authoritativeness
    auth_pattern = r'\[webpage authoritativeness\]\s*(.*?)\s*\[webpage content begin\]'
    auth_match = re.search(auth_pattern, webpage_content, re.DOTALL)
    if auth_match:
        result['auth'] = auth_match.group(1).strip()

    # Extract content
    content_pattern = r'\[webpage content begin\]\s*(.*?)\s*\[webpage content end\]'
    content_match = re.search(content_pattern, webpage_content, re.DOTALL)
    if content_match:
        result['content'] = content_match.group(1).strip()

    return result


def collect_cited_webpage_contents(
    trajectory: str,
    citation_indexes: List[int]
) -> Tuple[str, int, int]:
    """Collect the corresponding webpage contents from a trajectory based on citation indexes.

    Implements the paper's ExtractCitation + CollectContent logic:
    1. Extract citation indexes from <answer> (already done by extract_citation_indexes)
    2. Match WEB_TPL-format webpage contents from <tool_response>

    Args:
        trajectory: the Agent's full trajectory
        citation_indexes: the list of citation indexes (already truncated to the first 20)

    Returns:
        (the collected webpage content string, the number of citations successfully found, the total number of citations).
    """
    if not citation_indexes:
        return "", 0, 0

    # Extract all tool_response content
    tool_response_text, has_tool_response = extract_tool_responses_from_trajectory(trajectory)

    if not has_tool_response:
        logger.warning("No tool_response content found")
        return "", 0, len(citation_indexes)

    # Collect the webpage content for each citation index
    collected_webpages = []
    found_indexes = set()
    unique_citation_indexes = list(dict.fromkeys(citation_indexes))  # deduplicate while preserving order

    for idx in unique_citation_indexes:
        webpage = parse_webpage_from_web_tpl(tool_response_text, idx)
        if webpage:
            found_indexes.add(idx)
            # Format into readable webpage content
            formatted = f"""[Cited Webpage {idx}]
Title: {webpage['title']}
URL: {webpage['url']}
Date: {webpage['date']}
Authoritativeness: {webpage['auth']}
Content:
{webpage['content']}
"""
            collected_webpages.append(formatted)

    logger.info(
        f"Successfully extracted {len(found_indexes)} webpage contents out of {len(unique_citation_indexes)} unique citations"
    )

    if not collected_webpages:
        logger.warning(f"Could not find any cited webpage content in the trajectory, citation_indexes={unique_citation_indexes}")
        return "", 0, len(unique_citation_indexes)

    return '\n---\n'.join(collected_webpages), len(found_indexes), len(unique_citation_indexes)


def instantiate_constraint_with_entities(
    constraint: str,
    identified_entities: Dict[str, Optional[str]]
) -> Tuple[str, bool]:
    """Replace entity labels in a constraint with the identified entities.

    Args:
        constraint: the original constraint (e.g. "C1. <E1>graduated from a music conservatory<E2>")
        identified_entities: the identified entity mapping (e.g. {"E1": "Shan Yichun", "E2": "Zhejiang Conservatory of Music"})

    Returns:
        (the instantiated constraint, whether it was fully identified).
    """
    # Extract all entity labels from the constraint
    entity_pattern = r'<(E\d+)>'
    entity_labels = re.findall(entity_pattern, constraint)

    # Check whether all entities were identified
    all_identified = True
    instantiated = constraint

    for label in entity_labels:
        entity_value = identified_entities.get(label)
        if entity_value is None:
            all_identified = False
            # Keep the original label
        else:
            # Replace the entity label with the actual value
            instantiated = instantiated.replace(f'<{label}>', entity_value)

    return instantiated, all_identified


def build_bipartite_graph(
    constraints: List[str],
    supported_constraints: Set[int],
    constraint_entities_map: Dict[int, List[str]]
) -> Dict[str, Any]:
    """Build an entity-constraint bipartite graph.

    Args:
        constraints: the list of constraints
        supported_constraints: the set of supported constraint indexes
        constraint_entities_map: mapping from constraint to entities {constraint index: [list of entity labels]}

    Returns:
        The bipartite graph structure: {
            'entities': {entity label: [list of constraint indexes]},
            'constraints': {constraint index: [list of entity labels]}
        }
    """
    graph = {
        'entities': defaultdict(list),
        'constraints': {}
    }

    for constraint_idx in supported_constraints:
        if constraint_idx in constraint_entities_map:
            entities = constraint_entities_map[constraint_idx]
            graph['constraints'][constraint_idx] = entities

            for entity in entities:
                graph['entities'][entity].append(constraint_idx)

    return graph


def bfs_connected_constraints(
    graph: Dict[str, Any],
    answer_entity: str = 'E0'
) -> Set[int]:
    """BFS starting from the answer entity to find all connected constraints.

    Args:
        graph: the bipartite graph structure
        answer_entity: the answer entity label (default 'E0')

    Returns:
        The set of constraint indexes connected to the answer entity.
    """
    visited_entities = set()
    visited_constraints = set()
    queue = deque([answer_entity])
    visited_entities.add(answer_entity)

    while queue:
        entity = queue.popleft()

        # Visit all constraints connected to this entity
        if entity in graph['entities']:
            for constraint_idx in graph['entities'][entity]:
                if constraint_idx not in visited_constraints:
                    visited_constraints.add(constraint_idx)

                    # Visit all entities involved in this constraint
                    if constraint_idx in graph['constraints']:
                        for new_entity in graph['constraints'][constraint_idx]:
                            if new_entity not in visited_entities:
                                visited_entities.add(new_entity)
                                queue.append(new_entity)

    return visited_constraints


def extract_entities_from_constraint(constraint: str) -> List[str]:
    """Extract all entity labels from a constraint.

    Args:
        constraint: the constraint string (e.g. "C1. <E1>graduated from a music conservatory<E2>")

    Returns:
        The list of entity labels (e.g. ["E1", "E2"]).
    """
    entity_pattern = r'<(E\d+)>'
    return re.findall(entity_pattern, constraint)


# ============================================================
# Core class
# ============================================================

class CGRPORewardCalculator:
    """C-GRPO reward calculator.

    Implements the three-step evaluation based on the CaRR framework:
    1. Step 1: Hidden Entity Identification
    2. Step 2: Citation-based Rubric Judgment
    3. Step 3: Evidence Connectivity
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the C-GRPO reward calculator.

        Args:
            config: the config dict
        """
        self.config = config
        self.model_name = config.get('cgrpo_judge_model', 'gpt-4o')

        # LLM Judge API credentials: prefer config, then fall back to
        # environment variables, do not hardcode
        self.client = OpenAI(
            api_key=config.get('api_key') or os.environ.get("LLM_JUDGE_API_KEY", ""),
            base_url=config.get('api_base_url') or os.environ.get("LLM_JUDGE_BASE_URL", "https://api.openai.com/v1"),
        )

    def _call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call the LLM API.

        Args:
            prompt: the input prompt
            max_tokens: the max number of output tokens

        Returns:
            The LLM output text.
        """
        max_retries = 3
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                logger.error(f"LLM API call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)

        # Raise the last exception after all retries fail
        raise RuntimeError(f"LLM API call still failed after {max_retries} retries") from last_error

    def step1_entity_identification(
        self,
        question: str,
        constraints: List[str],
        response: str,
        ground_truth_entities: Dict[str, str]
    ) -> Tuple[Dict[str, Optional[str]], List[int]]:
        """Step 1: Hidden Entity Identification.

        Identify hidden entities in the Agent's answer.

        Args:
            question: the original question
            constraints: the list of constraints
            response: the Agent's answer
            ground_truth_entities: the ground-truth entity mapping

        Returns:
            (the identified entity mapping, the list of fully-identified constraint indexes).
        """
        try:
            # Build the prompt
            constraints_str = '\n'.join(constraints)
            prompt = ENTITY_IDENTIFICATION_PROMPT.format(
                question=question,
                constraints=constraints_str,
                response=response
            )

            # Call the LLM
            result_text = self._call_llm(prompt)

            # Parse the result
            identified_entities = self._parse_entity_identification_result(result_text, ground_truth_entities)

            # Compute the fully-identified constraints
            fully_identified_constraints = []
            for i, constraint in enumerate(constraints):
                entity_labels = extract_entities_from_constraint(constraint)
                if all(identified_entities.get(label) is not None for label in entity_labels):
                    fully_identified_constraints.append(i)

            logger.info(
                f"Step 1 complete: identified {sum(1 for v in identified_entities.values() if v is not None)}/{len(ground_truth_entities)} entities, "
                f"fully identified {len(fully_identified_constraints)}/{len(constraints)} constraints"
            )

            return identified_entities, fully_identified_constraints

        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
            logger.error(traceback.format_exc())
            return {}, []

    def _parse_entity_identification_result(
        self,
        result_text: str,
        ground_truth_entities: Dict[str, str]
    ) -> Dict[str, Optional[str]]:
        """Parse the entity identification result.

        Args:
            result_text: the LLM output text
            ground_truth_entities: the ground-truth entity mapping

        Returns:
            The identified entity mapping.
        """
        identified = {}

        # Initialize all entities to None
        for entity_label in ground_truth_entities.keys():
            identified[entity_label] = None

        try:
            # Try extracting JSON
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, result_text, re.DOTALL)

            if match:
                parsed = json_repair.loads(match.group(1))
            else:
                # Try parsing directly
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    parsed = json_repair.loads(result_text[json_start:json_end])
                else:
                    logger.warning(f"Could not extract JSON from the result: {result_text}")
                    return identified

            # Update the identification result
            for key, value in parsed.items():
                if key in identified and value is not None and value != 'null':
                    identified[key] = str(value)

        except Exception as e:
            logger.error(f"Failed to parse the entity identification result: {e}")

        return identified

    def step2_citation_rubric_judgment(
        self,
        constraints: List[str],
        fully_identified_indices: List[int],
        identified_entities: Dict[str, Optional[str]],
        trajectory: str,
        response: str
    ) -> Tuple[Set[int], Dict[int, str], str]:
        """Step 2: Citation-based Rubric Judgment.

        Verifies whether the identified constraints are supported by cited webpage content.

        Implements the paper's logic:
        1. Extract [citation:x]-format citation indexes from the <answer> tag
        2. Truncate to the first MAX_CITATION_COUNT (20) citations, to prevent the Agent from hacking the reward via excessive citations
        3. Match WEB_TPL-format webpage content from <tool_response>
        4. Call the Judge LLM to determine whether each constraint is supported by the cited content

        Fault tolerance:
        - If there is no [citation:x]-format citation in the answer, returns error code NO_CITATION
        - If a cited citation cannot be found in tool_response, returns error code CITATION_NOT_FOUND

        Args:
            constraints: the list of constraints
            fully_identified_indices: the list of fully-identified constraint indexes
            identified_entities: the identified entity mapping
            trajectory: the Agent's full trajectory
            response: the Agent's final answer

        Returns:
            (the set of supported constraint indexes, the instantiated constraint mapping, an error code).
        """
        if not fully_identified_indices:
            logger.info("Step 2: no fully-identified constraints, skipping")
            return set(), {}, RubricRewardError.SUCCESS

        try:
            # Step 2.1: extract citation indexes from the answer
            # response should be the sequence_str decoded from response_ids
            answer_text, has_answer = extract_answer_from_trajectory(response)

            if not has_answer:
                # Debug log: print the first 500 chars of the raw response,
                # to help check whether there really is no <answer> tag
                response_preview = response if len(response) > 500 else response
                logger.warning(
                    f"[DEBUG] NO_ANSWER_TAG error - raw response preview:\n"
                    f"{response_preview}\n"
                    f"response length: {len(response)}"
                )
                return set(), {}, RubricRewardError.NO_ANSWER_TAG

            # Extract citation indexes (take the first MAX_CITATION_COUNT, preserving order of appearance)
            citation_indexes, has_citation = extract_citation_indexes(answer_text, unique=False, sorted_result=False)

            if not has_citation:
                # Debug log: print the parsed answer_text, to help check whether there really is no citation
                logger.warning(
                    f"[DEBUG] NO_CITATION error - parsed answer_text:\n"
                    f"{answer_text}\n"
                    f"answer length: {len(answer_text)}"
                )
                return set(), {}, RubricRewardError.NO_CITATION

            logger.info(f"Step 2: extracted {len(citation_indexes)} citation indexes from the answer (limited to the first {MAX_CITATION_COUNT})")

            # Step 2.2: collect the cited webpage content from the trajectory
            webpage_content, found_count, total_count = collect_cited_webpage_contents(trajectory, citation_indexes)

            if found_count == 0:
                # Debug log: print the citation indexes and a trajectory preview, to help check why no citations were found
                trajectory_preview = trajectory[:1000] if len(trajectory) > 1000 else trajectory
                logger.warning(
                    f"[DEBUG] CITATION_NOT_FOUND error - none of the citations were found:\n"
                    f"extracted citation indexes: {citation_indexes[:10]} (first 10 shown)\n"
                    f"trajectory length: {len(trajectory)}\n"
                    f"trajectory preview:\n{trajectory_preview}\n"
                    f"{'...' if len(trajectory) > 1000 else ''}"
                )
                return set(), {}, RubricRewardError.CITATION_NOT_FOUND

            if found_count < total_count:
                # Some citations couldn't be found, still continue processing (log at info level only)
                logger.info(f"Step 2: {total_count - found_count}/{total_count} citations could not be found in tool_response")

            # Step 2.3: instantiate constraints (replace entity labels with the identified entities)
            instantiated_constraints = {}
            statements = []
            statement_to_constraint_idx = {}

            for i, constraint_idx in enumerate(fully_identified_indices):
                constraint = constraints[constraint_idx]
                instantiated, _ = instantiate_constraint_with_entities(constraint, identified_entities)
                instantiated_constraints[constraint_idx] = instantiated

                # Strip the constraint number prefix (e.g. "C1. ")
                clean_statement = re.sub(r'^C\d+\.\s*', '', instantiated)
                statement_label = f"S{i+1}"
                statements.append(f"{statement_label}. {clean_statement}")
                statement_to_constraint_idx[statement_label] = constraint_idx

            # Step 2.4: build the prompt and call the Judge LLM
            statements_str = '\n'.join(statements)
            prompt = CITATION_RUBRIC_JUDGMENT_PROMPT.format(
                context=webpage_content[:15000],  # limit the context length
                statements=statements_str
            )

            # Step 2.5: call the LLM and retry until the JSON parses successfully
            max_parse_retries = 5
            supported_constraints = set()

            for parse_attempt in range(max_parse_retries):
                try:
                    result_text = self._call_llm(prompt, max_tokens=4000)

                    # Try parsing the result
                    supported_constraints = self._parse_citation_judgment_result(
                        result_text, statement_to_constraint_idx
                    )

                    # Check whether all statements were parsed (validate JSON completeness)
                    expected_statements = set(statement_to_constraint_idx.keys())
                    parsed_statements = self._get_parsed_statements_from_result(result_text)
                    missing_statements = expected_statements - parsed_statements

                    if not missing_statements:
                        # Parsed successfully and completely
                        logger.info(f"Step 2: JSON parsed successfully (attempt {parse_attempt+1}/{max_parse_retries})")
                        break
                    else:
                        logger.warning(
                            f"Step 2: incomplete JSON (attempt {parse_attempt+1}/{max_parse_retries}), "
                            f"missing statements: {missing_statements}"
                        )
                        if parse_attempt == max_parse_retries - 1:
                            # The last attempt also failed, return an error
                            logger.error(f"Step 2: still could not parse complete JSON after {max_parse_retries} retries: {result_text}")
                            return set(), instantiated_constraints, RubricRewardError.PARSE_ERROR

                except Exception as e:
                    logger.error(f"Step 2: LLM call or parsing failed (attempt {parse_attempt+1}/{max_parse_retries}): {e}")
                    if parse_attempt == max_parse_retries - 1:
                        # The last attempt also failed
                        return set(), instantiated_constraints, RubricRewardError.LLM_CALL_FAILED

            logger.info(
                f"Step 2 complete: {len(supported_constraints)}/{len(fully_identified_indices)} constraints are supported by citations, "
                f"using {found_count}/{total_count} valid citations"
            )

            return supported_constraints, instantiated_constraints, RubricRewardError.SUCCESS

        except Exception as e:
            logger.error(f"Step 2 failed: {e}")
            logger.error(traceback.format_exc())
            return set(), {}, RubricRewardError.PARSE_ERROR

    def _parse_citation_judgment_result(
        self,
        result_text: str,
        statement_to_constraint_idx: Dict[str, int]
    ) -> Set[int]:
        """Parse the citation judgment result.

        Args:
            result_text: the LLM output text
            statement_to_constraint_idx: mapping from statement label to constraint index

        Returns:
            The set of supported constraint indexes.
        """
        supported = set()

        try:
            # Try extracting JSON
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, result_text, re.DOTALL)

            if match:
                parsed = json_repair.loads(match.group(1))
            else:
                # Try parsing directly
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    parsed = json_repair.loads(result_text[json_start:json_end])
                else:
                    logger.warning(f"Could not extract JSON from the result: {result_text[:200]}")
                    return supported

            # Parse the supported status
            for statement_label, is_supported in parsed.items():
                if statement_label in statement_to_constraint_idx:
                    if is_supported is True or str(is_supported).lower() == 'true':
                        constraint_idx = statement_to_constraint_idx[statement_label]
                        supported.add(constraint_idx)

        except Exception as e:
            logger.error(f"Failed to parse the citation judgment result: {e}")

        return supported

    def _get_parsed_statements_from_result(self, result_text: str) -> Set[str]:
        """Extract the set of parsed statement labels from the parsing result.

        Used to check whether the JSON contains all the expected statements.

        Args:
            result_text: the LLM output text

        Returns:
            The set of parsed statement labels (e.g. {"S1", "S2", "S3"}).
        """
        parsed_statements = set()

        try:
            # Try extracting JSON
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, result_text, re.DOTALL)

            if match:
                parsed = json_repair.loads(match.group(1))
            else:
                # Try parsing directly
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    parsed = json_repair.loads(result_text[json_start:json_end])
                else:
                    return parsed_statements

            # Extract all statement labels
            parsed_statements = set(parsed.keys())

        except Exception as e:
            logger.error(f"Failed to extract parsed statements: {e}")

        return parsed_statements

    def step3_evidence_connectivity(
        self,
        constraints: List[str],
        supported_constraints: Set[int]
    ) -> Set[int]:
        """Step 3: Evidence Connectivity Check.

        Checks whether the supported constraints are connected to the answer entity.

        Args:
            constraints: the list of constraints
            supported_constraints: the set of supported constraint indexes

        Returns:
            The set of constraint indexes connected to the answer entity.
        """
        if not supported_constraints:
            logger.info("Step 3: no supported constraints, skipping")
            return set()

        try:
            # Build the constraint-to-entities mapping
            constraint_entities_map = {}
            for constraint_idx in supported_constraints:
                constraint = constraints[constraint_idx]
                entities = extract_entities_from_constraint(constraint)
                constraint_entities_map[constraint_idx] = entities

            # Build the bipartite graph
            graph = build_bipartite_graph(
                constraints, supported_constraints, constraint_entities_map
            )

            # BFS to find the connected constraints
            connected_constraints = bfs_connected_constraints(graph, 'E0')

            logger.info(
                f"Step 3 complete: {len(connected_constraints)}/{len(supported_constraints)} constraints are connected to the answer entity"
            )

            return connected_constraints

        except Exception as e:
            logger.error(f"Step 3 failed: {e}")
            logger.error(traceback.format_exc())
            return set()

    def calculate_rubric_reward(
        self,
        question: str,
        constraints: List[str],
        ground_truth_entities: Dict[str, str],
        trajectory: str,
        response: str
    ) -> Dict[str, Any]:
        """Compute the complete Rubric Reward.

        Runs the three-step evaluation and computes the final Rubric Reward.

        Fault tolerance:
        - If the constraint list is empty, returns reward 0 with error code NO_CONSTRAINTS
        - If Step 2 returns an error code (e.g. NO_CITATION, CITATION_NOT_FOUND), the Rubric Reward is directly 0
        - Other parsing errors also result in a Rubric Reward of 0

        Args:
            question: the original question
            constraints: the list of constraints
            ground_truth_entities: the ground-truth entity mapping
            trajectory: the Agent's full trajectory (used to extract tool_response)
            response: the Agent's final answer (the sequence_str decoded from response_ids)

        Returns:
            A dict containing the results of each step, the final reward, and the error code.
        """
        total_constraints = len(constraints)

        # Important optimization (2026-01-28):
        # When constraints is empty, degrade to computing only the Outcome
        # Reward. In this case the Rubric Reward should be 1.0, equivalent
        # to no penalty (i.e. equivalent to looking only at the end-to-end reward).
        #
        # Rationale:
        # 1. Empty constraints mean reasoning quality can't be evaluated, so the Agent shouldn't be penalized.
        # 2. Rubric Reward = 1.0 makes the C-GRPO formula degrade to:
        #    R = (1-alpha)*R_o + alpha*R_o*1 = (1-alpha+alpha)*R_o = R_o
        # 3. Avoids a division-by-zero problem (originally rubric_reward = n_connected / n_total)
        if total_constraints == 0:
            logger.info("Constraint list is empty, degrading to computing only the Outcome Reward, setting Rubric Reward = 1.0")
            return {
                'rubric_reward': 1.0,  # Key change: from 0.0 to 1.0
                'r_identify': 1.0,  # Degraded mode: all ratios are 1.0
                'r_support': 1.0,
                'r_connect': 1.0,
                'n_total': 0,
                'n_identified': 0,
                'n_supported': 0,
                'n_connected': 0,
                'identified_entities': {},
                'supported_constraints': [],
                'connected_constraints': [],
                'error_code': RubricRewardError.NO_CONSTRAINTS,
            }

        # Check whether entities is empty (added 2026-01-28)
        # If entities is empty, this should also degrade to computing only the Outcome Reward
        if not ground_truth_entities or len(ground_truth_entities) == 0:
            logger.info("Entity list is empty, degrading to computing only the Outcome Reward, setting Rubric Reward = 1.0")
            return {
                'rubric_reward': 1.0,  # Degraded mode
                'r_identify': 1.0,
                'r_support': 1.0,
                'r_connect': 1.0,
                'n_total': total_constraints,
                'n_identified': 0,
                'n_supported': 0,
                'n_connected': 0,
                'identified_entities': {},
                'supported_constraints': [],
                'connected_constraints': [],
                'error_code': RubricRewardError.NO_CONSTRAINTS,
            }

        # Define the default result template (for error cases)
        default_result = {
            'rubric_reward': 0.0,
            'r_identify': 0.0,
            'r_support': 0.0,
            'r_connect': 0.0,
            'n_total': total_constraints,
            'n_identified': 0,
            'n_supported': 0,
            'n_connected': 0,
            'identified_entities': {},
            'supported_constraints': [],
            'connected_constraints': [],
            'error_code': RubricRewardError.SUCCESS,
        }

        # Step 1: Entity Identification
        identified_entities, fully_identified_indices = self.step1_entity_identification(
            question, constraints, response, ground_truth_entities
        )

        # Step 2: Citation-based Rubric Judgment
        supported_constraints, instantiated_constraints, step2_error = self.step2_citation_rubric_judgment(
            constraints, fully_identified_indices, identified_entities, trajectory, response
        )

        # Check the Step 2 error code
        if step2_error != RubricRewardError.SUCCESS:
            # No longer logging a separate WARNING here; summarized in the final stats instead
            default_result['n_identified'] = len(fully_identified_indices)
            default_result['r_identify'] = len(fully_identified_indices) / total_constraints if total_constraints > 0 else 0.0
            default_result['identified_entities'] = identified_entities
            default_result['error_code'] = step2_error
            return default_result

        # Step 3: Evidence Connectivity
        connected_constraints = self.step3_evidence_connectivity(
            constraints, supported_constraints
        )

        # Compute the ratio for each step
        n_identified = len(fully_identified_indices)
        n_supported = len(supported_constraints)
        n_connected = len(connected_constraints)

        r_identify = n_identified / total_constraints if total_constraints > 0 else 0.0
        r_support = n_supported / n_identified if n_identified > 0 else 0.0
        r_connect = n_connected / n_supported if n_supported > 0 else 0.0

        # Final Rubric Reward = |R_connect| / |R_q|
        rubric_reward = n_connected / total_constraints if total_constraints > 0 else 0.0

        logger.info(
            f"Rubric Reward computation complete: "
            f"r_identify={r_identify:.3f} ({n_identified}/{total_constraints}), "
            f"r_support={r_support:.3f} ({n_supported}/{n_identified}), "
            f"r_connect={r_connect:.3f} ({n_connected}/{n_supported}), "
            f"final={rubric_reward:.3f} ({n_connected}/{total_constraints})"
        )

        return {
            'rubric_reward': rubric_reward,
            'r_identify': r_identify,
            'r_support': r_support,
            'r_connect': r_connect,
            'n_total': total_constraints,
            'n_identified': n_identified,
            'n_supported': n_supported,
            'n_connected': n_connected,
            'identified_entities': identified_entities,
            'supported_constraints': list(supported_constraints),
            'connected_constraints': list(connected_constraints),
            'error_code': RubricRewardError.SUCCESS,
        }


# ============================================================
# Helper functions (used by the Reward Manager)
# ============================================================

def extract_constraints_and_entities_from_ground_truth(
    extra_info: Dict[str, Any]
) -> Tuple[List[str], Dict[str, str]]:
    """Extract constraints and entities from extra_info.

    Note: constraints and entities live in the data's extra_info field,
    not in ground_truth. ground_truth only contains target (the reference answer).

    Args:
        extra_info: the data's extra_info field, containing:
            - constraints: the list of constraints (e.g. ["C1. <E1>graduated from a music conservatory<E2>", ...])
            - entities: the entity mapping (e.g. {"E0": "Li Ronghao", "E1": "Shan Yichun", ...})

    Returns:
        (the list of constraints, the entity mapping).
    """
    constraints = extra_info.get('constraints', [])
    entities = extra_info.get('entities', {})

    # Ensure constraints is a list
    if isinstance(constraints, str):
        constraints = [c.strip() for c in constraints.split('\n') if c.strip()]

    # Ensure entities is a dict
    if not isinstance(entities, dict):
        entities = {}

    return constraints, entities


def normalize_rubric_reward_within_group(
    rubric_rewards: List[float],
    uid_list: List[str]
) -> List[float]:
    """Normalize the Rubric Reward within each group (GRPO-style).

    Normalization method:
    - Group by uid (the same uid = different rollouts of the same query)
    - Within each uid group, normalized = raw value / the group's max value

    Args:
        rubric_rewards: the list of Rubric Rewards
        uid_list: the list of uids

    Returns:
        The list of normalized Rubric Rewards.
    """
    if len(rubric_rewards) == 0:
        return []

    if len(rubric_rewards) != len(uid_list):
        raise ValueError(f"rubric_rewards and uid_list length mismatch: {len(rubric_rewards)} vs {len(uid_list)}")

    # Group by uid
    uid2rewards = defaultdict(list)
    uid2indices = defaultdict(list)

    for i, (reward, uid) in enumerate(zip(rubric_rewards, uid_list)):
        uid2rewards[uid].append(reward)
        uid2indices[uid].append(i)

    # Compute the max value for each uid group
    uid2max = {}
    for uid, rewards in uid2rewards.items():
        max_reward = max(rewards) if rewards else 0.0
        uid2max[uid] = max_reward if max_reward > 0 else 1.0  # avoid division by zero

    # Normalize
    normalized = [0.0] * len(rubric_rewards)
    for i, (reward, uid) in enumerate(zip(rubric_rewards, uid_list)):
        max_reward = uid2max[uid]
        normalized[i] = reward / max_reward if max_reward > 0 else 0.0

    logger.info(
        f"Rubric Reward within-group normalization: n_samples={len(rubric_rewards)}, "
        f"raw range=[{min(rubric_rewards):.3f}, {max(rubric_rewards):.3f}], "
        f"normalized range=[{min(normalized):.3f}, {max(normalized):.3f}]"
    )

    return normalized


def calculate_cgrpo_mixed_reward(
    outcome_reward: float,
    rubric_reward_normalized: float,
    alpha: float = 0.3
) -> float:
    """Compute the C-GRPO mixed reward.

    Formula: R_i = (1 - alpha) * R_o + alpha * R_o * R_hat_r

    Args:
        outcome_reward: the Outcome Reward (0 or 1)
        rubric_reward_normalized: the normalized Rubric Reward
        alpha: the balancing parameter

    Returns:
        The mixed reward.
    """
    return (1 - alpha) * outcome_reward + alpha * outcome_reward * rubric_reward_normalized


def classify_abnormal_trajectory(abnormal_flags: Dict[str, bool]) -> Tuple[bool, bool, List[str]]:
    """Classify an abnormal trajectory, determining whether it should be discarded or given reward 0.

    Abnormalities are handled in three tiers:

    1. Discard (excluded from training, excluded from GRPO grouping and the loss):
       - search_error: search API failure, an environment issue that shouldn't penalize the model
       - exceed_max_tokens: exceeded the token limit, the trajectory is truncated / incomplete

    2. Give reward 0 (participates in training, the model learns to avoid it via the reward signal):
       - exceed_max_turns: exceeded the max search rounds without answering
       - tool_parse_error: output-format parsing failure

    3. Offending round zeroed (trajectory kept, only that round's reward is zeroed):
       - excessive_tool_calls_per_turn: too many queries in a single round

    4. Handled normally (does not affect reward):
       - repeated_query: duplicate query

    Args:
        abnormal_flags: the abnormality-flag dict, whose keys match the AbnormalTrajectoryType constants

    Returns:
        (should_give_zero_reward, should_discard, abnormal_types).
    """
    abnormal_types = []

    # === Discard tier: environment issues or data truncation, excluded from training ===
    # Only these cases are discarded:
    # - search_error: search API failure, an environment problem, not the model's fault
    # - exceed_max_tokens: data truncated / incomplete, harmful to training
    # - degenerate_output: degenerate output (repetition/garbage), training on these samples would reinforce degenerate behavior
    discard_types = [
        AbnormalTrajectoryType.SEARCH_ERROR,
        AbnormalTrajectoryType.EXCEED_MAX_TOKENS,
        AbnormalTrajectoryType.DEGENERATE_OUTPUT,
    ]

    should_discard = False
    for abnormal_type in discard_types:
        if abnormal_flags.get(abnormal_type, False):
            abnormal_types.append(abnormal_type)
            should_discard = True

    # === Reward-0 tier: model-behavior abnormality, participates in training so the model learns to avoid it ===
    # - exceed_max_turns: exceeded the max search rounds without answering
    # - tool_parse_error: output-format parsing failure
    # These trajectories get reward=0 but are not discarded, so the model can still learn from them
    zero_reward_types = [
        AbnormalTrajectoryType.EXCEED_MAX_TURNS,
        AbnormalTrajectoryType.TOOL_PARSE_ERROR,
    ]

    should_give_zero_reward = False
    for abnormal_type in zero_reward_types:
        if abnormal_flags.get(abnormal_type, False):
            abnormal_types.append(abnormal_type)
            should_give_zero_reward = True

    # === Offending-round-zeroed tier: trajectory kept, only that round's reward is zeroed ===
    if abnormal_flags.get(AbnormalTrajectoryType.EXCESSIVE_TOOL_CALLS, False):
        abnormal_types.append(AbnormalTrajectoryType.EXCESSIVE_TOOL_CALLS)

    # === Handled-normally tier: does not affect reward ===
    normal_types = [
        AbnormalTrajectoryType.REPEATED_QUERY,
    ]
    for abnormal_type in normal_types:
        if abnormal_flags.get(abnormal_type, False):
            abnormal_types.append(abnormal_type)

    return should_give_zero_reward, should_discard, abnormal_types


def compute_cgrpo_metrics(
    end_to_end_rewards: List[float],
    rubric_rewards_raw: List[float],
    rubric_rewards_norm: List[float],
    r_identify_list: List[float],
    r_support_list: List[float],
    r_connect_list: List[float],
    is_normal_trajectory: List[bool],
    rubric_error_types: List[str]
) -> Dict[str, float]:
    """Compute monitoring metrics for the C-GRPO training process.

    This function computes the following metrics:
    1. Normal-trajectory rate (normal_rate)
    2. End-to-end reward stats (end_to_end_reward): mean, std
    3. Rubric reward stats (computed only over trajectories that got the answer right):
       - mean of rubric_raw
       - mean of rubric_norm
       - r_i/r_q (r_identify), r_s/r_q (r_support), r_c/r_q (r_connect)
    4. Rubric error-type stats

    Args:
        end_to_end_rewards: list of end-to-end rewards (0 or 1)
        rubric_rewards_raw: list of raw rubric rewards
        rubric_rewards_norm: list of normalized rubric rewards
        r_identify_list: list of r_identify values
        r_support_list: list of r_support values
        r_connect_list: list of r_connect values
        is_normal_trajectory: list of whether each trajectory is normal
        rubric_error_types: list of rubric error types

    Returns:
        A dict of monitoring metrics.
    """
    import numpy as np

    metrics = {}

    # 1. Normal-trajectory rate
    if len(is_normal_trajectory) > 0:
        normal_count = sum(is_normal_trajectory)
        total_count = len(is_normal_trajectory)
        metrics['cgrpo/normal_trajectory_rate'] = normal_count / total_count
        metrics['cgrpo/normal_trajectory_count'] = float(normal_count)
        metrics['cgrpo/abnormal_trajectory_count'] = float(total_count - normal_count)

    # 2. End-to-end reward stats (computed only over normal trajectories)
    normal_end_to_end_rewards = [
        r for r, is_normal in zip(end_to_end_rewards, is_normal_trajectory) if is_normal
    ]

    if len(normal_end_to_end_rewards) > 0:
        metrics['cgrpo/end_to_end_reward_mean'] = float(np.mean(normal_end_to_end_rewards))
        metrics['cgrpo/end_to_end_reward_std'] = float(np.std(normal_end_to_end_rewards))
        metrics['cgrpo/end_to_end_success_rate'] = float(np.mean([r >= 0.999 for r in normal_end_to_end_rewards]))

    # 3. Rubric reward stats (computed only over normal trajectories that got the answer right)
    # "Got the answer right" is defined as end_to_end_reward == 1
    correct_normal_indices = [
        i for i, (r, is_normal) in enumerate(zip(end_to_end_rewards, is_normal_trajectory))
        if r >= 0.999 and is_normal
    ]

    if len(correct_normal_indices) > 0:
        correct_rubric_raw = [rubric_rewards_raw[i] for i in correct_normal_indices]
        correct_rubric_norm = [rubric_rewards_norm[i] for i in correct_normal_indices]
        correct_r_identify = [r_identify_list[i] for i in correct_normal_indices]
        correct_r_support = [r_support_list[i] for i in correct_normal_indices]
        correct_r_connect = [r_connect_list[i] for i in correct_normal_indices]

        metrics['cgrpo/rubric_raw_mean'] = float(np.mean(correct_rubric_raw))
        metrics['cgrpo/rubric_norm_mean'] = float(np.mean(correct_rubric_norm))
        metrics['cgrpo/r_identify_mean'] = float(np.mean(correct_r_identify))
        metrics['cgrpo/r_support_mean'] = float(np.mean(correct_r_support))
        metrics['cgrpo/r_connect_mean'] = float(np.mean(correct_r_connect))

        # Compute standard deviation
        metrics['cgrpo/rubric_raw_std'] = float(np.std(correct_rubric_raw))
        metrics['cgrpo/rubric_norm_std'] = float(np.std(correct_rubric_norm))

        # Record the number of samples that got the answer right
        metrics['cgrpo/correct_sample_count'] = float(len(correct_normal_indices))

    # 4. Rubric error-type stats
    error_type_counts = defaultdict(int)
    for error_type in rubric_error_types:
        if error_type != RubricRewardError.SUCCESS:
            error_type_counts[error_type] += 1

    total_errors = sum(error_type_counts.values())
    if total_errors > 0:
        for error_type, count in error_type_counts.items():
            metrics[f'cgrpo/rubric_error/{error_type}'] = float(count)
        metrics['cgrpo/rubric_error/total'] = float(total_errors)

    return metrics
