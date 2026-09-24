#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PS Pipeline Rubric Reward - a rubric-based, per-round process reward.

Core idea:
- For each round of the PS Pipeline, the Planner and Synthesizer are each
  scored against a rubric by an external LLM.
- Planner rubric evaluates: anchor prioritization, hypothesis management,
  targeted verification search, constraint satisfaction, termination judgment.
- Synthesizer rubric evaluates: cross-clue consistency, uncertainty
  management, source faithfulness, relevance filtering, actionable summary.

Usage:
    calculator = PSRubricRewardCalculator(
        planner_rubric_path="planner_rubric.json",
        synthesizer_rubric_path="synthesizer_rubric.json",
    )
    rubric_scores = calculator.compute_trajectory_rubric_reward(trajectory_rounds)
"""

import json
import logging
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Prompt templates
# ============================================================

PLANNER_RUBRIC_EVAL_PROMPT = '''You are a professional evaluator for a Planner-Synthesizer (PS) multi-agent deep search system. Your task is to objectively and rigorously assess the quality of the **Planner's current-round action** against the provided rubrics, using the trajectory context as evidence.

## System Context

The system has two agents; you evaluate only the Planner.

- **Planner**: Each round produces `<think>...</think>` reasoning, then EITHER a `<tool_call>` (issuing one or more search queries) OR an `<answer>` (final termination). The Planner only sees the global summary, never raw retrieved documents.
- **Synthesizer** (not evaluated here): After each search round, integrates retrieved documents into the updated global summary the Planner will see next.

Surface-level queries often look similar across trajectories, so what actually distinguishes a good Planner from a bad one is the reasoning behind those queries — which sub-constraint is the current critical gap, whether candidates are grounded in workspace evidence, and whether the emitted queries / answer causally follow from that diagnosis.

## Input Components
1. **User Query**: The user's original question (contains the sub-constraints you will check against).
2. **Ground Truth**: Reference answer. It is used ONLY when the current round's action is `answer` (to check whether termination was warranted). NEVER use it to reward or punish search-round queries that happen to (mis)match the gold — well-reasoned exploration is allowed to be wrong.
3. **Round Number**: Current round index (0-based).
4. **Historical Context**: The global summary available to the Planner at the start of this round — the accumulated evidence state from all previous rounds.
5. **Current Round Planner Action**:
   - **Action Type**: `"tool_call"` (search) or `"answer"` (termination).
   - **Planner Think**: verbatim `<think>` content.
   - **Planner Output**: either the list of search queries OR the final answer text.
6. **Rubrics**: Evaluation criteria. **ALL rubrics are always applicable — every rubric must receive a score of 5 / 3 / 1 in every round.** Each rubric's description explains how it applies to both tool_call and answer rounds.

## Evaluation Rules

### Rule 1: Every Rubric Scored Every Round (STRICT)
All rubrics MUST be evaluated. For each rubric, you MUST either output a score (5 / 3 / 1) or mark it as N/A (not applicable).

When a rubric's description says "N/A" for a specific scenario (e.g. "no conflict triggered", "no contested slot", "no new claims"), you MUST output **"score": "N/A"** for that rubric. Do NOT score 3 for these cases.

When a rubric is applicable but the round doesn't exhibit the positive indicator AND doesn't exhibit a clear violation either, score **3** and briefly note in the reason.

### Rule 2: Mandatory Evidence Quote
Every rubric's `reason` MUST contain at least ONE verbatim quote (≤30 words, word-for-word) taken from the CURRENT round, using one of these labeled prefixes:
- `think_quote: "..."` — from this round's `<think>` (preferred default anchor).
- `query_quote: "..."` — from this round's issued queries.
- `branch_justification: "..."` — a think snippet that explicitly justifies the chosen direction (branch or answer) against stem qualifiers (use for `branch_direction_justification`).
- `constraint_closure: "..."` — a think snippet that revisits a stem sub-constraint with its satisfied / still-open status (use for `constraint_closure_tracking`).
You MAY additionally add `ctx_quote: "..."` (from historical context) or `prior_think_quote: "..."` / `prior_query_quote: "..."` (from a prior round) as SUPPORTING evidence, but they do NOT replace the primary anchor above.
A reason missing a valid labeled quote is treated downstream as score = 1 regardless of what number you wrote. Do not paraphrase inside the quote.

### Rule 3: Scoring
Score each applicable rubric on the hard three-tier scale (5 / 3 / 1). For rubrics marked as N/A in the description, output "score": "N/A". See Rule 1 and Rule 5 for details.

### Rule 4: Anti-Inflation & Strict Score Ceilings (CRITICAL — apply FIRST)
Before you even consider a 5, verify the rubric's "5-requires ALL THREE" checklist in its description. If ANY element of the checklist is missing, you MUST cap at 3. Additionally, the following universal ceilings apply:
- **Generic Query Ceiling**: If the Planner issues broad, generic queries (e.g. "Entity X" rather than "Entity X specific date/feature"), cap at **3**.
- **Repetitive/Lazy Thinking Ceiling**: If `<think>` merely summarizes the historical context without providing genuinely NEW diagnostic insight, cap at **3**.
- **Premature Answer Ceiling**: If the Planner outputs `<answer>` but the `<think>` fails to explicitly map every user sub-constraint to the collected evidence, cap at **3**.
- **Query/Think Mismatch Ceiling**: If the `<think>` identifies a gap, but the actual emitted `tool_call` queries fail to explicitly address that exact gap, cap at **3**.

### Rule 5: Scoring Scale (HARD three-tier + N/A)
**3 is the default score. 5 is rare and reserved for verifiably exceptional rounds.**

**Calibration target — internalize this distribution:**
- On ~70% of applicable rubrics: **3** (default for competent execution).
- On ~15% of applicable rubrics: **5** (only when ALL elements of the rubric's 5-checklist are verifiably satisfied with verbatim quotes).
- On ~15% of applicable rubrics: **1** (clear failure).

**Score definitions:**
- **5 (Exceptional)**: All elements of the rubric's "requires ALL THREE" checklist are verifiable. The round is demonstrably beyond a median competent baseline.
- **3 (Acceptable / Good — DEFAULT)**: Round functions correctly. Has clear intent. May have generic queries, template-style reasoning, or partial constraint tracking. **This is the expected score for most rounds.**
- **1 (Failure)**: A concrete 1-anchor violation applies — outright error, hallucination, skipped constraint, or derailed logic.
- **N/A**: Only when the rubric's description explicitly allows N/A.

**Anti-default discipline (MANDATORY):**
- "Query is reasonable and targets the gap" → **3** (not 5).
- "Reasoning is clean, structured, and actions follow through" → **3** (not 5).
- "The Planner enumerated constraints and chose a direction" → **3** (not 5).
- "Query is uniquely designed to falsify candidate A by testing attribute X, with A and B both named in `<think>`" → **5** (only if all three rubric-5 elements verifiable).

### Rule 6: Ground Truth Usage
Ground truth is ONLY allowed to influence the scoring of answer rounds (where the Planner is committing to a final answer). For tool_call rounds it must be ignored.

## Evaluation Process

Execute in order:
1. Parse the User Query → enumerate its sub-constraints (stem qualifiers, entities, limiters).
2. Read Historical Context → form your own view of each sub-constraint's state at round start (resolved / unresolved / conflicting / not-addressed).
3. Read the current `<think>` → identify (a) the branch/candidate it leans toward, (b) how each sub-constraint is treated, (c) what genuinely new reasoning this round contributes.
4. Read the Planner Output → check causal alignment with the `<think>` diagnosis.
5. For EVERY rubric in the Rubrics input, score 5 / 3 / 1 per Rule 5, with at least one Rule 2 labeled quote in `reason`. Do not omit any rubric.
6. Emit the JSON object described below.

## Output Format (CRITICAL)

Output ONLY a single JSON object. No preamble, no code fences around the outer object. Use the rubric IDs from the Rubrics input as top-level keys. Each entry has EXACTLY two fields: `reason` (string, includes a Rule 2 labeled quote) and `score` (5, 3, 1, or "N/A"). **Every rubric in the Rubrics input must appear exactly once in the output** — do NOT omit any.

For N/A rubrics: output `"score": "N/A"` and explain in `reason` why it's not applicable (e.g. "no conflict triggered this round").

**IMPORTANT: Keep each `reason` concise — one labeled quote + one sentence of judgment (≤50 words total). Do NOT write multi-paragraph explanations.**

```json
{{
  "{rubric_id_1}": {{
    "reason": "think_quote: \\"...\\" — brief judgment.",
    "score": 5
  }},
  "{rubric_id_2}": {{
    "reason": "query_quote: \\"...\\" — brief judgment.",
    "score": "N/A"
  }}
}}
```

The output JSON's keys must match the Rubrics input 1-to-1. Missing a key will be treated downstream as an error.

---

Now evaluate the following content.

**User Query:**
```
{user_query}
```

**Ground Truth:**
```
{ground_truth}
```

**Round Number:** {round_number}

**Historical Context (Global Summary at round start):**
```
{historical_context}
```

**Current Round Planner Action:**
```
Action Type: {action_type}

Planner Think:
{planner_think}

Planner Output:
{planner_output}
```

**Rubrics:**
```json
{rubrics_json}
```

Your JSON evaluation:'''


SYNTHESIZER_RUBRIC_EVAL_PROMPT = '''You are a professional evaluator for a Planner-Synthesizer (PS) multi-agent deep search system. Your task is to objectively and rigorously assess the quality of the **Synthesizer's current-round updated global summary** along two axes — **faithfulness** (claims are evidence-grounded) and **strategic value** (the update actually moves the investigation forward).

## System Context

The Synthesizer runs after each search round. It receives (a) the previous global summary and (b) the documents retrieved by the Planner's most recent search, and must emit an updated global summary that the Planner will read at the START of the next round.

A strong Synthesizer:
1. Keeps every claim grounded in either the prior summary or this round's retrieved documents (faithfulness).
2. Advances the investigation — surfaces new discriminative evidence, narrows open gaps, flags conflicts, re-tags tentative candidates (strategic value).

A summary can be perfectly faithful yet strategically useless; both axes matter.

## Input Components
1. **User Query**: The user's original question; its sub-constraints define what "clue-relevant" means.
2. **Ground Truth**: Reference answer, used ONLY to help you identify which retrieved facts are clue-relevant. NEVER reward the summary for "happening to mention the right entity" if that mention is not grounded in this round's evidence, and NEVER penalize well-grounded claims that happen not to match the gold.
3. **Round Number**: Current round index (0-based).
4. **Is Last Synth Round**: Boolean flag. `true` means this is the LAST synthesizer round of the trajectory (the next Planner step is expected to be `<answer>`); its job shifts from surfacing new evidence to consolidation. The Gap-delta ceiling in Rule 4 does NOT apply when this flag is true.
5. **Previous Global Summary**: The summary at the START of this round.
6. **Planner Context** (read-only, for disambiguation):
   - Planner `<think>` for this round.
   - Search queries issued this round.
7. **Search Results**: Documents retrieved this round (listed as `[passage N] ...`).
8. **Synthesizer Output**: The updated global summary — **THE THING YOU EVALUATE**.
9. **Rubrics**: Evaluation criteria.

## Evaluation Rules

### Rule 1: Pre-Scoring Internal Analysis (THINK — do NOT output)
Before scoring any rubric, internally work out the following analysis. Do NOT include this analysis in the output JSON; use it only to inform your scores:
- `new_claims`: factual statements present in UPDATED but NOT in PREVIOUS summary.
- `modified_or_removed`: prior-summary claims that were altered or dropped.
- `gap_changes`: which sub-constraints were resolved / narrowed / newly-surfaced / unchanged.
- `verbatim_copy_ratio_estimate`: float in [0.0, 1.0] — what fraction of the updated summary is verbatim from the previous summary.
- `retrieval_utilization`: for each retrieved passage, whether it was incorporated-with-citation / noted-as-irrelevant / silently-dropped.
Use this analysis to apply the Rule 4 ceilings below.

### Rule 2: Mandatory Evidence Quote
Every rubric's `reason` MUST include at least ONE verbatim quote (≤30 words) using one of these labeled prefixes:
- `quoted: "..."` — the generic, always-valid anchor (from either the updated summary, the previous summary, or a retrieved passage).
- `anchor: "..."` — a "uniqueness anchor" quote showing how a candidate entity is uniquely distinguished (use for `entity_anchor_integrity`).
- `gold_passage_quote: "..."` — a verbatim excerpt of a strongly stem-matching passage that the updated summary should / should not have surfaced (use for `gold_evidence_salience`).
- `CONFLICT: "..."` — the marker + surrounding quoted evidence when flagging cross-source contradiction (use for `conflict_flagging_required`).
A reason missing a valid labeled quote is downstream treated as score = 1 regardless of what number you wrote. Do not paraphrase inside the quote.

### Rule 3: Every Rubric Scored Every Round (STRICT)
All rubrics MUST be evaluated. For each rubric, you MUST either output a score (5 / 3 / 1) or mark it as N/A (not applicable).

When a rubric's description says "N/A" for a specific scenario (e.g. "no conflict triggered", "no contested slot", "no new claims"), you MUST output **"score": "N/A"** for that rubric. Do NOT score 3 for these cases.

When a rubric is applicable but the round doesn't exhibit the positive indicator AND doesn't exhibit a clear violation either, score **3** and briefly note in the reason.

### Rule 4: Anti-Inflation Ceilings (CRITICAL - Apply BEFORE finalizing a score)
Before you even consider a 5, verify the rubric's "5-requires ALL THREE" checklist in its description. If ANY element is missing, cap at 3. Additionally, the following universal ceilings apply:

- **Copy-paste / Bloat Ceiling**: If `verbatim_copy_ratio_estimate > 0.7`, OR if the Synthesizer simply appends new facts to the bottom of the summary without actively rewriting/integrating them, cap ANY progress-oriented rubric at **≤ 3**.
- **Gap-delta Ceiling**: If `len(resolved) + len(narrowed) + len(newly_surfaced) − len(unchanged) ≤ 0` AND the retrieval contained clearly clue-relevant content (meaning the Synthesizer missed it), cap the progress rubric at **≤ 3**.
- **Signal-vs-noise / Information Density Ceiling**: If the updated summary contains generic background facts, irrelevant fluff from the retrieved docs, or fails to actively prune dead-end candidates from previous rounds, cap `gold_evidence_salience` and overall scores at **≤ 3**. A 5-point summary must be dense and strictly focused on the user's constraints.
- **Anchor missing Ceiling**: If a candidate entity is upgraded to confirmed without a uniqueness-anchor citation in the updated summary, `entity_anchor_integrity` caps at **1**.

#### Consolidation-round exemptions (apply when `is_last_synth_round == true`)
The final Synth round's job is CONSOLIDATION (re-confirming the leading candidate, tightening language), NOT surfacing new evidence:
- **Gap-delta ceiling** is lifted. 
- **Anchor missing ceiling** is lifted IF the uniqueness anchor for the leading candidate was already cited in the Previous Global Summary.
- **Copy-paste ceiling** and **Signal-vs-noise ceiling** still STRICTLY apply.

### Rule 4.5: Peer-LLM Halo Defense (LIGHT GUIDANCE)
The summary you are evaluating was produced by another LLM that tends to produce structured, authoritative-looking summaries. Surface polish alone is not evidence of 5-tier quality — but a clean, faithful, well-cited summary IS the standard 3, which is the expected default. Use the rubric checklists to decide 5 vs 3, not stylistic suspicion.

### Rule 5: Scoring Scale (HARD three-tier + N/A)
**3 is the default score. 5 is rare and reserved for verifiably exceptional synthesis.**

**Calibration target — internalize this distribution:**
- On ~70% of applicable rubrics: **3** (default for competent synthesis).
- On ~15% of applicable rubrics: **5** (only when ALL elements of the rubric's 5-checklist are verifiably satisfied).
- On ~15% of applicable rubrics: **1** (clear failure).

**Score definitions:**
- **5 (Exceptional)**: All elements of the rubric's "requires ALL THREE" checklist are verifiable. Summary is demonstrably beyond a median competent baseline.
- **3 (Acceptable / Good — DEFAULT)**: Summary is faithful and captures the new evidence, but may read like a list of facts, retain minor irrelevant details, or fail to deeply integrate new facts with the old narrative. **This is the expected score for most rounds.**
- **1 (Failure)**: Concrete 1-anchor violation — hallucination, missing crucial retrieved clues, or severe logical contradiction.
- **N/A**: Only when the rubric's description explicitly allows N/A.

**Anti-default discipline (MANDATORY):**
- "Didn't hallucinate and added the new info" → **3** (not 5).
- "Summary is well-structured with citations" → **3** (not 5).
- "Every candidate carries some standard anchor (date/location)" → **3** (not 5).
- "Summary explicitly re-flagged candidate X as dead-end by citing retrieved fact Y, and reordered to push unresolved constraint Z to the top" → **5** (only if all three rubric-5 elements verifiable).

### Rule 6: Ground Truth Usage
Use ground truth ONLY to decide which retrieved facts are clue-relevant. Never use it to score an ungrounded "lucky mention," and never penalize a faithfully grounded claim simply because the overall summary has not yet reached the gold.

## Evaluation Process

Execute in order:
1. Read User Query → enumerate sub-constraints & clue-relevant patterns.
2. Read Previous Summary → note current gap states & committed candidates.
3. Read Retrieval → identify clue-relevant passages (esp. those matching rare stem tokens / limiters).
4. Read Synthesizer Output → internally compute the diff analysis described in Rule 1 (do NOT output it).
5. Apply the Rule 4 ceilings where applicable.
6. For EVERY rubric in the Rubrics input, score 5 / 3 / 1 per Rule 5, with at least one Rule 2 labeled quote in `reason`. Do not omit any rubric.
7. Emit the JSON object.

## Output Format (CRITICAL)

Output ONLY a single JSON object. No preamble, no code fences around the outer object. Use the rubric IDs from the Rubrics input as top-level keys. Each entry has EXACTLY two fields: `reason` (string, includes a Rule 2 labeled quote) and `score` (5, 3, 1, or "N/A"). **Every rubric in the Rubrics input must appear exactly once in the output** — do NOT omit any.

For N/A rubrics: output `"score": "N/A"` and explain in `reason` why it's not applicable (e.g. "no conflict triggered this round").

**IMPORTANT: Keep each `reason` concise — one labeled quote + one sentence of judgment (≤50 words total). Do NOT write multi-paragraph explanations.**

```json
{{
  "{rubric_id_1}": {{
    "reason": "quoted: \\"...\\" — brief judgment.",
    "score": 5
  }},
  "{rubric_id_2}": {{
    "reason": "anchor: \\"...\\" — brief judgment.",
    "score": "N/A"
  }}
}}
```

The output JSON's keys must match the Rubrics input 1-to-1. Missing a key will be treated downstream as an error.

---

Now evaluate the following content.

**User Query:**
```
{user_query}
```

**Ground Truth:**
```
{ground_truth}
```

**Round Number:** {round_number}

**Is Last Synth Round:** {is_last_synth_round}

**Previous Global Summary:**
```
{previous_summary}
```

**Planner's Reasoning (context only):**
```
{planner_think}
```

**Search Queries (context only):**
```
{search_queries}
```

**Search Results (document count: {doc_count}):**
```
{search_results}
```

**Synthesizer Output (Updated Summary — THE THING YOU EVALUATE):**
```
{synthesizer_output}
```

**Rubrics:**
```json
{rubrics_json}
```

Your JSON evaluation:'''


# ============================================================
# ============================================================
# Round reward computation
# ============================================================

def compute_round_reward(
    normalized_score: float,
    round_scores: Dict[str, Dict[str, Any]],
    rubrics: List[Dict[str, Any]],
    is_critical_round: bool = False,
) -> float:
    """
    Compute the reward for a single round.

    Notes:
    - Directly returns the rubric-weighted, normalized score (range [-1, 1]).
    - No longer applies critical-aware nonlinear amplification, to avoid
      reward hacking (e.g. the model could just keep the critical rubric
      at score 3 to sidestep the amplified penalty).
    - `round_scores`/`rubrics`/`is_critical_round` are kept for call-site
      compatibility but are not currently used in the computation.

    Args:
        normalized_score: the rubric-weighted, normalized score, range [-1, 1]
        round_scores: (kept for compatibility) scoring details
        rubrics: (kept for compatibility) the rubric list
        is_critical_round: (kept for compatibility) whether this is a critical round

    Returns:
        reward in [-1, 1]
    """
    # Clamp to guard against numerical error pushing it out of range
    return max(-1.0, min(1.0, float(normalized_score)))


# ============================================================
# Rate limiter
# ============================================================

class TokenBucketRateLimiter:
    """A token-bucket rate limiter, controlling the max requests per minute (QPM).

    Unlike a Semaphore-based concurrency limit, the token bucket ensures
    the request count within any sliding window does not exceed the
    configured cap, preventing a burst of requests from exceeding QPM.

    Args:
        max_per_minute: the max number of requests per minute
    """

    def __init__(self, max_per_minute: int = 200):
        self._max_tokens = max_per_minute
        self._tokens = float(max_per_minute)
        self._refill_rate = max_per_minute / 60.0  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        """Acquire one token; block and wait until a token is available if the bucket is empty."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._max_tokens,
                    self._tokens + elapsed * self._refill_rate
                )
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Compute the wait time needed
                wait_time = (1.0 - self._tokens) / self._refill_rate

            time.sleep(wait_time)


# ============================================================
# Core class
# ============================================================

class PSRubricRewardCalculator:
    """PS Pipeline Rubric Reward calculator.

    Loads the Planner / Synthesizer rubric JSON, calls an external LLM to
    score each round, and computes the weighted, normalized rubric reward.

    Attributes:
        planner_rubrics: the Planner rubric list
        synthesizer_rubrics: the Synthesizer rubric list
        planner_total_weight: the Planner rubrics' total weight
        synthesizer_total_weight: the Synthesizer rubrics' total weight
    """

    def __init__(
        self,
        planner_rubric_path: Optional[str] = None,
        synthesizer_rubric_path: Optional[str] = None,
        planner_rubric_data: Optional[Dict[str, Any]] = None,
        synthesizer_rubric_data: Optional[Dict[str, Any]] = None,
        judge_model: str = "gemini-2.5-flash-lite",
        api_key: str = None,
        base_url: str = "https://api.openai.com/v1",
        max_workers: int = 32,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_concurrent_calls: int = 16,
        max_qpm: int = 200,
    ):
        """Initialize PSRubricRewardCalculator.

        Args:
            planner_rubric_path: path to the Planner rubric JSON file
            synthesizer_rubric_path: path to the Synthesizer rubric JSON file
            planner_rubric_data: pass the Planner rubric dict directly (takes precedence over the file path)
            synthesizer_rubric_data: pass the Synthesizer rubric dict directly
            judge_model: the LLM model name used for scoring
            api_key: the API key
            base_url: the API base URL
            max_workers: max number of threads for concurrent scoring
            timeout: timeout (seconds) for a single API call
            max_retries: max number of API call retries
        """
        # Load Planner rubrics
        if planner_rubric_data is not None:
            p_data = planner_rubric_data
        elif planner_rubric_path is not None:
            with open(planner_rubric_path, "r", encoding="utf-8") as f:
                p_data = json.load(f)
        else:
            raise ValueError("Must provide either planner_rubric_path or planner_rubric_data")

        # Load Synthesizer rubrics
        if synthesizer_rubric_data is not None:
            s_data = synthesizer_rubric_data
        elif synthesizer_rubric_path is not None:
            with open(synthesizer_rubric_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
        else:
            raise ValueError("Must provide either synthesizer_rubric_path or synthesizer_rubric_data")

        self.planner_rubrics = p_data["rubrics"]
        self.synthesizer_rubrics = s_data["rubrics"]
        self.planner_total_weight = sum(r["weight"] for r in self.planner_rubrics)
        self.synthesizer_total_weight = sum(r["weight"] for r in self.synthesizer_rubrics)

        self.judge_model = judge_model
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = OpenAI(api_key=api_key or os.environ.get("LLM_JUDGE_API_KEY", ""), base_url=base_url)

        # Global concurrency limit: caps the number of simultaneously in-flight API requests
        self._max_concurrent_calls = max_concurrent_calls
        self._api_semaphore = threading.Semaphore(max_concurrent_calls)

        # Rate limit: caps the max requests per minute, preventing QPM overrun
        self._rate_limiter = TokenBucketRateLimiter(max_per_minute=max_qpm)

        logger.info(
            f"PSRubricRewardCalculator initialized: "
            f"planner_rubrics={len(self.planner_rubrics)} (total_weight={self.planner_total_weight}), "
            f"synthesizer_rubrics={len(self.synthesizer_rubrics)} (total_weight={self.synthesizer_total_weight}), "
            f"judge_model={judge_model}, "
            f"max_concurrent_calls={max_concurrent_calls}, "
            f"max_qpm={max_qpm}"
        )

    # --------------------------------------------------------
    # LLM call
    # --------------------------------------------------------

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the external LLM API to perform scoring.

        Two-tier limiting:
        1. Rate limit (TokenBucket): ensures the request count per minute never exceeds max_qpm
        2. Concurrency limit (Semaphore): caps the number of simultaneously in-flight requests

        Args:
            prompt: the full evaluation prompt

        Returns:
            The model's output string, or None on failure.
        """
        for attempt in range(self.max_retries):
            try:
                self._rate_limiter.acquire()  # Rate limit: wait for a token
                with self._api_semaphore:     # Concurrency limit
                    response = self.client.chat.completions.create(
                        model=self.judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=2000,
                    )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def _parse_rubric_scores(self, llm_output: Optional[str], rubric_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Parse the rubric scoring result from the LLM output.

        Args:
            llm_output: the LLM's raw output (may be None)
            rubric_ids: the list of expected rubric IDs

        Returns:
            {rubric_id: {"reason": str, "score": int or None}}
            score of None means this rubric is N/A (not applicable).
        """
        if not llm_output:
            return {rid: {"reason": "LLM call failed", "score": 1} for rid in rubric_ids}

        # Try extracting JSON
        try:
            # Try parsing directly
            result = json.loads(llm_output)
        except json.JSONDecodeError:
            # Try extracting from ```json ... ```
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_output, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    result = {}
            else:
                # Try finding the first { and the last }
                start = llm_output.find('{')
                end = llm_output.rfind('}')
                if start != -1 and end != -1 and end > start:
                    try:
                        result = json.loads(llm_output[start:end + 1])
                    except json.JSONDecodeError:
                        result = {}
                else:
                    result = {}

        # Validate and fill in any missing rubrics
        parsed = {}
        for rid in rubric_ids:
            if rid in result and isinstance(result[rid], dict):
                score_raw = result[rid].get("score", 1)
                reason_raw = str(result[rid].get("reason", ""))

                # Check whether this is N/A
                if isinstance(score_raw, str) and score_raw.strip().upper() == "N/A":
                    # N/A: not applicable, excluded from scoring
                    parsed[rid] = {
                        "reason": reason_raw or "N/A",
                        "score": None,  # None means N/A
                    }
                else:
                    # Ensure score is within 1-5
                    try:
                        score = max(1, min(5, int(score_raw)))
                    except (ValueError, TypeError):
                        score = 1

                    parsed[rid] = {
                        "reason": reason_raw,
                        "score": score,
                    }
            else:
                parsed[rid] = {"reason": "Missing in LLM output", "score": 1}

        return parsed

    # --------------------------------------------------------
    # Rubric formatting
    # --------------------------------------------------------

    def _format_rubrics_for_prompt(self, rubrics: List[Dict[str, Any]]) -> str:
        """Format the rubric list into the JSON string used inside the prompt.

        Note: the critical field is excluded, to ensure the judge's
        scoring isn't influenced by criticality information. Criticality
        is only used during reward computation.
        """
        formatted = []
        for r in rubrics:
            entry = {
                "id": r["id"],
                "name": r["name"],
                "description": r["description"],
                "weight": r["weight"],
            }
            # Optional fields: include these if present on the rubric
            if "positive_indicators" in r:
                entry["positive_indicators"] = r["positive_indicators"]
            if "negative_indicators" in r:
                entry["negative_indicators"] = r["negative_indicators"]
            formatted.append(entry)
        return json.dumps(formatted, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # Per-round scoring
    # --------------------------------------------------------

    def score_planner_round(
        self,
        user_query: str,
        round_number: int,
        historical_context: str,
        planner_think: str,
        planner_output: str,
        action_type: str,
        ground_truth: str = "",
    ) -> Tuple[Dict[str, Dict[str, Any]], float, str, Optional[str]]:
        """Score one round of the Planner's output against the rubric.

        Args:
            user_query: the user's question
            round_number: the round number
            historical_context: the global summary at the start of this round
            planner_think: the Planner's <think> content
            planner_output: the Planner's output (queries or answer)
            action_type: "tool_call" or "answer"
            ground_truth: the reference answer, used to evaluate answer rounds

        Returns:
            (the scoring-detail dict, the weighted normalized score [-1, 1], the prompt text, the raw LLM output).
        """
        rubric_ids = [r["id"] for r in self.planner_rubrics]
        rubrics_json = self._format_rubrics_for_prompt(self.planner_rubrics)

        prompt = PLANNER_RUBRIC_EVAL_PROMPT.format(
            user_query=user_query,
            ground_truth=ground_truth if ground_truth else "(not provided)",
            round_number=round_number,
            historical_context=historical_context if historical_context else "No previous search has been conducted yet.",
            planner_think=planner_think if planner_think else "(empty)",
            planner_output=planner_output if planner_output else "(empty)",
            action_type=action_type,
            rubrics_json=rubrics_json,
            rubric_id_1=rubric_ids[0] if len(rubric_ids) > 0 else "r1",
            rubric_id_2=rubric_ids[1] if len(rubric_ids) > 1 else "r2",
        )

        llm_output = self._call_llm(prompt)
        scores = self._parse_rubric_scores(llm_output, rubric_ids)

        # Score mapping: 1->-1, 3->0, 5->1
        # N/A handling: excluded from both the numerator and denominator (dynamic denominator)
        weighted_sum = 0.0
        total_weight = 0.0
        
        for rubric in self.planner_rubrics:
            rid = rubric["id"]
            w = rubric["weight"]
            s = scores[rid]["score"]
            
            # Skip N/A rubrics
            if s is None:
                continue
            
            # Score mapping: 1->-1, 3->0, 5->1
            mapped_score = (s - 3) / 2  # 1->-1, 3->0, 5->1
            
            weighted_sum += w * mapped_score
            total_weight += w  # Only include the weight of applicable rubrics
        
        # Normalize to [-1, 1]: use the mapped score directly
        normalized_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return scores, normalized_score, prompt, llm_output

    def score_synthesizer_round(
        self,
        user_query: str,
        round_number: int,
        previous_summary: str,
        planner_think: str,
        search_queries: str,
        search_results: str,
        doc_count: int,
        synthesizer_output: str,
        ground_truth: str = "",
        is_last_synth_round: bool = False,
    ) -> Tuple[Dict[str, Dict[str, Any]], float, str, Optional[str]]:
        """Score one round of the Synthesizer's output against the rubric.

        Args:
            user_query: the user's question
            round_number: the round number
            previous_summary: the global summary before this round
            planner_think: the corresponding Planner's <think> content
            search_queries: the search queries
            search_results: the search results text (may be truncated)
            doc_count: the document count
            synthesizer_output: the Synthesizer's updated summary output
            ground_truth: the reference answer, used to determine which facts are clue-relevant
            is_last_synth_round: whether this is the last synthesizer round

        Returns:
            (the scoring-detail dict, the weighted normalized score [-1, 1], the prompt text, the raw LLM output).
        """
        rubric_ids = [r["id"] for r in self.synthesizer_rubrics]
        rubrics_json = self._format_rubrics_for_prompt(self.synthesizer_rubrics)

        prompt = SYNTHESIZER_RUBRIC_EVAL_PROMPT.format(
            user_query=user_query,
            ground_truth=ground_truth if ground_truth else "(not provided)",
            round_number=round_number,
            is_last_synth_round=is_last_synth_round,
            previous_summary=previous_summary if previous_summary else "No previous summary.",
            planner_think=planner_think if planner_think else "(empty)",
            search_queries=search_queries if search_queries else "(empty)",
            search_results=search_results[:8000] if search_results else "(no results)",
            doc_count=doc_count,
            synthesizer_output=synthesizer_output if synthesizer_output else "(empty)",
            rubrics_json=rubrics_json,
            rubric_id_1=rubric_ids[0] if len(rubric_ids) > 0 else "r1",
            rubric_id_2=rubric_ids[1] if len(rubric_ids) > 1 else "r2",
        )

        llm_output = self._call_llm(prompt)
        scores = self._parse_rubric_scores(llm_output, rubric_ids)

        # Score mapping: 1->-1, 3->0, 5->1
        # N/A handling: excluded from both the numerator and denominator (dynamic denominator)
        weighted_sum = 0.0
        total_weight = 0.0
        
        for rubric in self.synthesizer_rubrics:
            rid = rubric["id"]
            w = rubric["weight"]
            s = scores[rid]["score"]
            
            # Skip N/A rubrics
            if s is None:
                continue
            
            # Score mapping: 1->-1, 3->0, 5->1
            mapped_score = (s - 3) / 2  # 1->-1, 3->0, 5->1
            
            weighted_sum += w * mapped_score
            total_weight += w  # Only include the weight of applicable rubrics
        
        # Normalize to [-1, 1]: use the mapped score directly
        normalized_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return scores, normalized_score, prompt, llm_output

    # --------------------------------------------------------
    # Trajectory-level scoring
    # --------------------------------------------------------

    def compute_trajectory_rubric_reward(
        self,
        trajectory_rounds: List[Dict[str, Any]],
        user_query: str,
        ground_truth: str = "",
    ) -> Dict[str, Any]:
        """Score every round of a full trajectory against the rubric.

        trajectory_rounds format (built by the reward manager):
        [
            {
                "round": int,
                "planner": {
                    "think": str,
                    "output": str,         # the queries JSON or the answer text
                    "action_type": str,    # "tool_call" or "answer"
                    "summary_before": str, # the global summary at the start of this round
                },
                "synthesizer": {           # optional, absent for a P(answer) round
                    "output": str,         # the updated summary
                    "summary_before": str, # equal to planner.summary_before
                    "search_queries": str,
                    "search_results": str,
                    "doc_count": int,
                } | None,
            },
            ...
        ]

        Args:
            trajectory_rounds: the trajectory's per-round data
            user_query: the user's question
            ground_truth: the reference answer, used for evaluation

        Returns:
            {
                "planner_scores": [{round, scores, normalized_score, reward}, ...],
                "synthesizer_scores": [{round, scores, normalized_score, reward}, ...],
                "planner_mean_score": float,       # the mean normalized score across all Planner rounds
                "synthesizer_mean_score": float,   # the mean normalized score across all Synthesizer rounds
                "trajectory_rubric_reward": float, # the combined rubric reward (may fall outside [0, 1])
                "per_round_rewards": {             # the rubric reward for each sample (nonlinear transform applied)
                    (role, round): float,
                },
                "rubric_io_samples": [{           # a sample of prompt/response for each round
                    "role": str, "round": int, "is_critical": bool,
                    "prompt": str, "llm_output": str,
                    "normalized_score": float, "reward": float,
                }, ...],
            }
        """
        # Determine critical rounds
        # 1. Find the last synthesizer round
        last_synth_round = -1
        for rd in trajectory_rounds:
            if rd.get("synthesizer") is not None:
                last_synth_round = max(last_synth_round, rd["round"])

        # 2. For each round, determine whether it is a critical round
        def is_critical_round(role: str, rnd: int, action_type: Optional[str] = None) -> bool:
            """
            Determine whether this is a critical round.
            - round <= 1 (the first two rounds)
            - Planner: action_type == "answer"
            - Synthesizer: is the last synthesizer round
            """
            if rnd <= 1:
                return True
            # if role == "planner" and action_type == "answer":
            #     return True
            # if role == "synthesizer" and rnd == last_synth_round:
            #     return True
            return False

        planner_tasks = []
        synthesizer_tasks = []

        for rd in trajectory_rounds:
            rnd = rd["round"]
            p = rd["planner"]
            planner_tasks.append({
                "round": rnd,
                "user_query": user_query,
                "historical_context": p["summary_before"],
                "planner_think": p["think"],
                "planner_output": p["output"],
                "action_type": p["action_type"],
                "is_critical": is_critical_round("planner", rnd, p["action_type"]),
                "ground_truth": ground_truth,
            })

            if rd.get("synthesizer") is not None:
                s = rd["synthesizer"]
                synthesizer_tasks.append({
                    "round": rnd,
                    "user_query": user_query,
                    "previous_summary": s["summary_before"],
                    "planner_think": p["think"],
                    "search_queries": s["search_queries"],
                    "search_results": s["search_results"],
                    "doc_count": s["doc_count"],
                    "synthesizer_output": s["output"],
                    "is_critical": is_critical_round("synthesizer", rnd),
                    "ground_truth": ground_truth,
                    "is_last_synth_round": (rnd == last_synth_round),
                })

        # Run all scoring tasks concurrently
        planner_results = []
        synthesizer_results = []
        per_round_rewards = {}
        rubric_io_samples = []  # sampled prompt/response logs

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit Planner scoring tasks
            p_futures = {}
            for task in planner_tasks:
                future = executor.submit(
                    self.score_planner_round,
                    user_query=task["user_query"],
                    round_number=task["round"],
                    historical_context=task["historical_context"],
                    planner_think=task["planner_think"],
                    planner_output=task["planner_output"],
                    action_type=task["action_type"],
                    ground_truth=task["ground_truth"],
                )
                p_futures[future] = task

            # Submit Synthesizer scoring tasks
            s_futures = {}
            for task in synthesizer_tasks:
                future = executor.submit(
                    self.score_synthesizer_round,
                    user_query=task["user_query"],
                    round_number=task["round"],
                    previous_summary=task["previous_summary"],
                    planner_think=task["planner_think"],
                    search_queries=task["search_queries"],
                    search_results=task["search_results"],
                    doc_count=task["doc_count"],
                    synthesizer_output=task["synthesizer_output"],
                    ground_truth=task["ground_truth"],
                    is_last_synth_round=task["is_last_synth_round"],
                )
                s_futures[future] = task

            # Collect Planner results and apply the critical-aware reward
            for future in as_completed(p_futures):
                task = p_futures[future]
                rnd = task["round"]
                is_critical = task["is_critical"]
                try:
                    scores, normalized, prompt, llm_output = future.result(timeout=self.timeout)
                    # Apply the critical-aware nonlinear transform
                    reward = compute_round_reward(
                        normalized_score=normalized,
                        round_scores=scores,
                        rubrics=self.planner_rubrics,
                        is_critical_round=is_critical,
                    )
                    planner_results.append({
                        "round": rnd,
                        "scores": scores,
                        "normalized_score": normalized,
                        "reward": reward,
                        "is_critical": is_critical,
                    })
                    per_round_rewards[("planner", rnd)] = reward
                    # Collect the prompt/response sample
                    rubric_io_samples.append({
                        "role": "planner",
                        "round": rnd,
                        "is_critical": is_critical,
                        "prompt": prompt,
                        "llm_output": llm_output,
                        "normalized_score": normalized,
                        "reward": reward,
                    })
                except Exception as e:
                    logger.error(f"Failed to score Planner round {rnd}: {e}")
                    planner_results.append({
                        "round": rnd,
                        "scores": {},
                        "normalized_score": -0.5,  # give a negative value on scoring failure
                        "reward": -0.5,
                        "is_critical": is_critical,
                    })
                    per_round_rewards[("planner", rnd)] = -0.5

            # Collect Synthesizer results and apply the critical-aware reward
            for future in as_completed(s_futures):
                task = s_futures[future]
                rnd = task["round"]
                is_critical = task["is_critical"]
                try:
                    scores, normalized, prompt, llm_output = future.result(timeout=self.timeout)
                    # Apply the critical-aware nonlinear transform
                    reward = compute_round_reward(
                        normalized_score=normalized,
                        round_scores=scores,
                        rubrics=self.synthesizer_rubrics,
                        is_critical_round=is_critical,
                    )
                    synthesizer_results.append({
                        "round": rnd,
                        "scores": scores,
                        "normalized_score": normalized,
                        "reward": reward,
                        "is_critical": is_critical,
                    })
                    per_round_rewards[("synthesizer", rnd)] = reward
                    # Collect the prompt/response sample
                    rubric_io_samples.append({
                        "role": "synthesizer",
                        "round": rnd,
                        "is_critical": is_critical,
                        "prompt": prompt,
                        "llm_output": llm_output,
                        "normalized_score": normalized,
                        "reward": reward,
                    })
                except Exception as e:
                    logger.error(f"Failed to score Synthesizer round {rnd}: {e}")
                    synthesizer_results.append({
                        "round": rnd,
                        "scores": {},
                        "normalized_score": -0.5,  # give a negative value on scoring failure
                        "reward": -0.5,
                        "is_critical": is_critical,
                    })
                    per_round_rewards[("synthesizer", rnd)] = -0.5

        # Sort by round
        planner_results.sort(key=lambda x: x["round"])
        synthesizer_results.sort(key=lambda x: x["round"])

        # Compute the mean score (using the transformed reward)
        p_rewards = [r["reward"] for r in planner_results]
        s_rewards = [r["reward"] for r in synthesizer_results]

        planner_mean = sum(p_rewards) / len(p_rewards) if p_rewards else 0.0
        synthesizer_mean = sum(s_rewards) / len(s_rewards) if s_rewards else 0.0

        # Combined rubric reward: a weighted average of Planner and Synthesizer
        # Planner has a larger weight (since the search strategy is more critical)
        planner_weight = 0.6
        synthesizer_weight = 0.4
        if not s_rewards:
            # If there are no Synthesizer rounds (e.g. P(answer) only), give all weight to the Planner
            trajectory_rubric_reward = planner_mean
        else:
            trajectory_rubric_reward = (
                planner_weight * planner_mean + synthesizer_weight * synthesizer_mean
            )

        return {
            "planner_scores": planner_results,
            "synthesizer_scores": synthesizer_results,
            "planner_mean_score": planner_mean,
            "synthesizer_mean_score": synthesizer_mean,
            "trajectory_rubric_reward": trajectory_rubric_reward,
            "per_round_rewards": per_round_rewards,
            "rubric_io_samples": rubric_io_samples,
        }

    # --------------------------------------------------------
    # Batch scoring (called by the reward manager)
    # --------------------------------------------------------

    def batch_compute_rubric_rewards(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute the rubric reward for multiple trajectories in a batch.

        Args:
            trajectories: the list of trajectories, each element containing:
                - "user_query": str
                - "ground_truth": str (required, used to evaluate each round)
                - "rounds": List[Dict] (same as trajectory_rounds in compute_trajectory_rubric_reward)

        Returns:
            A list of rubric reward results the same length as the input.
        """
        results = []

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(trajectories))) as executor:
            futures = {}
            for i, traj in enumerate(trajectories):
                ground_truth = traj.get("ground_truth", "")
                if not ground_truth:
                    logger.warning(f"Trajectory {i} has an empty ground_truth, which will affect scoring quality")

                future = executor.submit(
                    self.compute_trajectory_rubric_reward,
                    trajectory_rounds=traj["rounds"],
                    user_query=traj["user_query"],
                    ground_truth=ground_truth,
                )
                futures[future] = i

            idx_results = {}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result(timeout=self.timeout * 20)
                    idx_results[idx] = result
                except Exception as e:
                    logger.error(f"Failed to score rubric for trajectory {idx}: {e}")
                    idx_results[idx] = {
                        "planner_scores": [],
                        "synthesizer_scores": [],
                        "planner_mean_score": 0.0,
                        "synthesizer_mean_score": 0.0,
                        "trajectory_rubric_reward": 0.0,
                        "per_round_rewards": {},
                        "rubric_io_samples": [],
                    }

            for i in range(len(trajectories)):
                results.append(idx_results.get(i, {
                    "planner_scores": [],
                    "synthesizer_scores": [],
                    "planner_mean_score": 0.0,
                    "synthesizer_mean_score": 0.0,
                    "trajectory_rubric_reward": 0.0,
                    "per_round_rewards": {},
                    "rubric_io_samples": [],
                }))

        return results


# ============================================================
# Utility function: build trajectory_rounds from PS Pipeline reward manager data
# ============================================================

def build_trajectory_rounds_from_samples(
    sample_indices: List[int],
    rollout_extra_info_list: List[Any],
    sequences_str: List[str],
    messages_list: List[Any],
) -> List[Dict[str, Any]]:
    """Build a per-round structure from the flattened PS Pipeline sample data.

    In the reward manager, samples sharing the same request_id are already
    ordered by data_sample_index. These samples alternate as
    planner / synthesizer / planner / synthesizer / ... / planner(answer).

    Args:
        sample_indices: the list of sample indexes sharing the same request_id
        rollout_extra_info_list: the global rollout_extra_info list
        sequences_str: the global list of decoded response text
        messages_list: the global messages list

    Returns:
        A trajectory_rounds list, in the same format as
        compute_trajectory_rubric_reward's input.
    """
    # First sort by data_sample_index
    sorted_indices = sorted(sample_indices, key=lambda i: (
        rollout_extra_info_list[i].get("data_sample_index", i)
        if isinstance(rollout_extra_info_list[i], dict) else i
    ))

    # Separate the planner and synthesizer samples
    rounds_data = {}  # round_num -> {"planner_idx": int, "synthesizer_idx": int | None}

    for idx in sorted_indices:
        ei = rollout_extra_info_list[idx]
        if not isinstance(ei, dict):
            continue
        role = ei.get("role", "")
        rnd = ei.get("round", 0)

        if rnd not in rounds_data:
            rounds_data[rnd] = {"planner_idx": None, "synthesizer_idx": None}

        if role == "planner":
            rounds_data[rnd]["planner_idx"] = idx
        elif role == "synthesizer":
            rounds_data[rnd]["synthesizer_idx"] = idx

    # Build trajectory_rounds
    trajectory_rounds = []
    prev_summary = ""

    for rnd in sorted(rounds_data.keys()):
        rd = rounds_data[rnd]
        p_idx = rd["planner_idx"]

        if p_idx is None:
            continue

        p_ei = rollout_extra_info_list[p_idx]
        p_think = p_ei.get("think", "")
        p_answer = p_ei.get("answer", None)
        p_summary_before = prev_summary  # the summary at the start of this round

        # Determine action_type
        if p_answer is not None and p_answer != "":
            action_type = "answer"
            p_output = p_answer
        else:
            action_type = "tool_call"
            # Extract the queries from messages or raw_content
            p_raw = p_ei.get("raw_content", "")
            if not p_raw:
                p_raw = sequences_str[p_idx] if p_idx < len(sequences_str) else ""
            p_output = p_raw  # the raw output containing <tool_call>

        round_entry = {
            "round": rnd,
            "planner": {
                "think": p_think,
                "output": p_output,
                "action_type": action_type,
                "summary_before": p_summary_before,
            },
            "synthesizer": None,
        }

        # Process the Synthesizer (if present)
        s_idx = rd["synthesizer_idx"]
        if s_idx is not None:
            s_ei = rollout_extra_info_list[s_idx]
            s_output = s_ei.get("summary", "")
            if not s_output:
                s_raw = s_ei.get("raw_content", "")
                if not s_raw:
                    s_raw = sequences_str[s_idx] if s_idx < len(sequences_str) else ""
                s_output = s_raw

            # Extract the search queries and results (from messages)
            search_queries_str = ""
            search_results_str = ""
            doc_count = 0

            # The Synthesizer's messages contain the search info
            if messages_list is not None and s_idx < len(messages_list):
                s_msgs = messages_list[s_idx]
                if isinstance(s_msgs, list):
                    for msg in s_msgs:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            content = msg.get("content", "")
                            # Extract the search queries and results from the
                            # synthesizer prompt (format follows _build_synthesizer_prompt)
                            tq_match = re.search(
                                r'<tool_call>\s*(.*?)\s*</tool_call>',
                                content, re.DOTALL
                            )
                            if tq_match:
                                search_queries_str = tq_match.group(1).strip()

                            tr_match = re.search(
                                r'<tool_response>\s*(.*?)\s*</tool_response>',
                                content, re.DOTALL
                            )
                            if tr_match:
                                search_results_str = tr_match.group(1).strip()

                            dc_match = re.search(
                                r'(\d+)\s*documents?\)',
                                content
                            )
                            if dc_match:
                                try:
                                    doc_count = int(dc_match.group(1))
                                except ValueError:
                                    pass
                            break

            round_entry["synthesizer"] = {
                "output": s_output,
                "summary_before": p_summary_before,
                "search_queries": search_queries_str,
                "search_results": search_results_str,
                "doc_count": doc_count,
            }

            # Update prev_summary to the new summary produced by the synthesizer
            prev_summary = s_output
        else:
            # No synthesizer (a P(answer) round), the summary stays unchanged
            pass

        trajectory_rounds.append(round_entry)

    return trajectory_rounds
