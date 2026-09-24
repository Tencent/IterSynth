# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import os
import random
import re
from datetime import datetime
from typing import List, Dict, Tuple
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Grading prompt templates (English and Chinese variants). Kept in both
# languages intentionally: JUDGE_PROMPT_CN is used when language="zh" so the
# LLM judge reasons in the same language as the underlying QA data.
JUDGE_PROMPT_EN = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].  Put the extracted answer as  'None'  if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer],
        focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer.
        Do not comment on any background to the problem, do not attempt to solve the problem,
        do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer]  given  above,
        or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e.
        if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
"""

JUDGE_PROMPT_CN = """根据以下精确且明确的[response]，判断以下对[question]的[correct_answer]是否正确。

[question]:  {question}

[response]:  {response}

您的判断必须符合以下指定的格式和标准：

extracted_final_answer: 从[response]中提取的最终准确答案。如果无法从答案中提取出准确的最终答案，则将提取的答案填写为"None"。

[correct_answer]: {correct_answer}

reasoning: 根据[correct_answer]解释提取的最终答案正确或错误的原因， 仅关注[correct_answer]和提取的最终答案之间是否存在有意义的差异。请勿评论问题的任何背景，请勿尝试解决问题，请勿争论任何与[correct_answer]不同的答案，仅关注答案是否匹配。

correct: 如果提取的最终答案与上面给出的[correct_answer]相符，或者在数值问题的误差范围内，则回答"yes"。否则，例如，如果存在任何不一致、歧义、不等同，或者提取的答案不正确，则回答"no"。

confidence: 从[response]中提取的置信度分数，介于0% 到100% 之间。如果没有可用的置信度分数，则填写100%。
"""

class BrowseCompZHEvaluator:
    def __init__(self, grader_model_name: str = "gpt-4o", language: str = "zh"):
        self.grader_model_name = grader_model_name
        self.language = language.lower()
        # LLM Judge API credentials: provide via environment variables, do not hardcode
        self.client = OpenAI(
            api_key=os.environ.get("LLM_JUDGE_API_KEY", ""),
            base_url=os.environ.get("LLM_JUDGE_BASE_URL", "https://api.openai.com/v1"),
        )

    def get_prompt_template(self):
        if self.language == "zh":
            return JUDGE_PROMPT_CN
        elif self.language == "en":
            return JUDGE_PROMPT_EN
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    def grade_answer(self, question: str, correct_answer: str, response: str) -> Tuple[Dict, str]:
        prompt = self.get_prompt_template().format(
            question=question,
            response=response,
            correct_answer=correct_answer
        )
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.grader_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=500
                )
                model_output = response.choices[0].message.content.strip()

                # Parse the model output to extract each field
                result = self.parse_judge_output(model_output)
                return result, model_output
            except Exception as e:
                print(f"Grading failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                else:
                    return {
                        'extracted_final_answer': 'None',
                        'correct': 'no',
                        'confidence': '100',
                        'reasoning': f"ERROR: {str(e)}"
                    }, f"ERROR: {str(e)}"

    def parse_judge_output(self, output: str) -> Dict:
        """Parse the judge model's output and extract each field."""
        result = {
            'extracted_final_answer': 'None',
            'correct': 'no',
            'confidence': '100',
            'reasoning': ''
        }

        # Extract extracted_final_answer
        extracted_match = re.search(r'extracted_final_answer:\s*(.+?)(?=\n|$)', output, re.IGNORECASE | re.DOTALL)
        if extracted_match:
            result['extracted_final_answer'] = extracted_match.group(1).strip()

        # Extract correct
        correct_match = re.search(r'correct:\s*(yes|no)', output, re.IGNORECASE)
        if correct_match:
            result['correct'] = correct_match.group(1).lower()

        # Extract confidence
        confidence_match = re.search(r'confidence:\s*(\d+(?:\.\d+)?)', output, re.IGNORECASE)
        if confidence_match:
            result['confidence'] = confidence_match.group(1)

        # Extract reasoning
        reasoning_match = re.search(r'reasoning:\s*(.+?)(?=\n|$)', output, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1).strip()

        return result

evaluator = BrowseCompZHEvaluator()



# def normalize_answer(s):
#     def remove_articles(text):
#         return re.sub(r"\b(a|an|the)\b", " ", text)

#     def white_space_fix(text):
#         return " ".join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return "".join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))


# def em_check(prediction, golden_answers):
#     if isinstance(golden_answers, str):
#         golden_answers = [golden_answers]
#     normalized_prediction = normalize_answer(prediction)
#     score = 0
#     for golden_answer in golden_answers:
#         golden_answer = normalize_answer(golden_answer)
#         if golden_answer == normalized_prediction:
#             score = 1
#             break
#     return score


# def subem_check(prediction, golden_answers):
#     if isinstance(golden_answers, str):
#         golden_answers = [golden_answers]
#     normalized_prediction = normalize_answer(prediction)
#     score = 0
#     for golden_answer in golden_answers:
#         golden_answer = normalize_answer(golden_answer)
#         if golden_answer in normalized_prediction:
#             score = 1
#             break
#     return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Handle None and empty-string inputs
    if solution_str is None or solution_str.strip() == "":
        return None

    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_question(solution_str):
    """Extract the question from the solution string."""
    # Extract the current user question (supports multiple formats)
    q = None
    try:
        # Try format 1: Question: ...
        q_pattern = r"Question: (.*?)\nassistant"
        matches = re.findall(q_pattern, solution_str, re.DOTALL)
        if matches:
            q = matches[0].strip()
        else:
            # Try format 2: The user's message is: ...
            q_pattern = r"The user's message is:(.*?)assistant"
            matches = re.findall(q_pattern, solution_str, re.DOTALL)
            if matches:
                q = matches[0].strip()
    except Exception as e:
        logger.error(f"Error extracting query: {e}")
    return q


def compute_score(solution_str, ground_truth, extra_info):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    question = extra_info.get('question', None)

    if question is None or answer is None:
        return 0

    judge_result, _ = evaluator.grade_answer(question, ground_truth['target'], answer)
    if judge_result['correct'] == 'yes':
        score = 1.0
    else:
        score = 0.0

    # do_print = random.randint(1, 64) == 1
    do_print = True

    if do_print:
        import json
        log_info = {
            "golden_answers": ground_truth['target'],
            "extracted_answer": answer,
            "solution_string": solution_str,
            "question": question,
            "judge_result": judge_result,
            "score": score,
        }
        # print(json.dumps(log_info, ensure_ascii=False))

    return score
