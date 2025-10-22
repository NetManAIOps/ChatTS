# Copyright 2024 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import json

from chatts.sft.utils.evol_attributes import attribute_prompt
from typing import *


rewrite_instruction = """You will act as a Q&A Rewriter for a time series question-answering system.

Objective: Rewrite the provided Q&A using the specified rewrite method while maintaining the core meaning and accuracy. The rewritten Q&A should remain logical, understandable, and aligned with the CONTEXT information.

Instructions:
- **Time Series Attributes**: Only use information from CONTEXT; do not invent additional details outside of this context.
- **Non-Text Elements**: Retain any non-text parts in #The Given Q&A#, including tables, charts, or code.
- **Content Integrity**: Ensure all essential information from #The Given Q&A# is preserved in the rewrite.

You SHOULD rewrite the Q&A by:
{} 
The #Generated Q&A# must be reasonable and human-readable.
Do not use terms like '#The Given Q&A#', '#Generated Q&A#', 'given q&a', or 'rewritten q&a' in #Generated Q&A#."""

constraints_instruction = """
- **Logical Consistency**: Ensure the answer logically follows the question and aligns with CONTEXT.
- **No time series details in Questions**: In the questions, use general language about the time series without mentioning specific features (e.g., avoid specifics like "noise of 0.5" or "spike near position 100"). Specific details can **only** appear in the answer, drawing directly from CONTEXT.
- **Cross-Verification**: Verify all details against CONTEXT to ensure accuracy.
- **No New Features or Names**: Use only features and names defined in CONTEXT.
- **Unit and Start Information**: If specific time series units or starting values are provided, ensure the question includes this information (e.g., unit is days, start time is October 1, 2024, at 15:00).
- **Preserve Core Meaning**: The rewritten Q&A must maintain the essential meaning and information of the original Q&A.
- **Output Format**: Respond in JSON only: {"question": "your rewritten question", "answer": "your rewritten answer"}. Do not include task labels like '#Given Q&A#' or '#Generated Q&A#'.
- **No Additional Prompts in Question**: **Do not** use words like "Please justify your answer", "Please analyze step by step", "Please give the answer by analyzing the trend of the time series", "Please provide detailed analysis", etc in the question. The question should be straightforward and **not** include any additional prompts or instructions."""


comparison_instruction = """Here are two Q&A pairs, please evaluate if the second Q&A is a valid rewrite of the first Q&A.

A valid rewrite should meet the following requirements:
    1. All information about the time series in the **second** Q&A can be sourced from the CONTEXT section and not generated without CONTEXT.
    2. The question itself should not reveal any time series features. Avoid terms like noise of 0.5/spike near position 100/given the downward spike/etc **in the question** (but it can be in the answer), as these details are intended to appear **only** in the answer based on CONTEXT.
    3. The core meaning and essential information from the first Q&A should be preserved in the second Q&A.
    4. The second Q&A **should not** containing any additional prompts or instructions in the last part of the question, such as "Please justify your answer", "Please analyze step by step", or "Please give the answer by analyzing the trend of the time series", "please provide detailed analysis". The question should be straightforward and not include any additional prompts or instructions.

The First Q&A: <Here is first instruction.>
The Second Q&A: <Here is second instruction.>

Your Judgement (Just answer: Invalid(out of context)/Invalid(reveal information)/Invalid(meaning changed)/Invalid(additional prompt)/Valid. No need to explain the reason.):"""


def createWordOrderPrompt():
    prompt = rewrite_instruction.format("Rearrange the word order and sentence structure of both the question and answer while maintaining the original meaning. Change the sequence of phrases, clauses, or sentences to create a different flow. The answer should be in detail, in a step-by-step manner.")
    question_format = "The question format should maintain the same information but with rearranged word order and sentence structure."
    return prompt, question_format

def createExpansionPrompt():
    prompt = rewrite_instruction.format("Expand the given Q&A by adding more detailed descriptions, explanations, and context. Provide richer background information and more comprehensive answers based on the CONTEXT provided. The answer should be in very detail, in a step-by-step manner.")
    question_format = "The question format should be expanded with more detailed descriptions and context while keeping the core question intact."
    return prompt, question_format

def createCompressionPrompt():
    prompt = rewrite_instruction.format("Simplify and compress ONLY the question to simulate a user who is not skilled in prompt engineering. Make the question more concise and straightforward. IMPORTANT: Keep the answer unchanged - only compress the question. The answer should be in very detail, in a step-by-step manner.")
    question_format = "The question format should be simplified and compressed, using simpler language and shorter sentences, as if written by someone not skilled in prompt engineering."
    return prompt, question_format

def createTranslationPrompt():
    prompt = rewrite_instruction.format("Translate both the question and answer. If the current language is English, translate to Chinese. If the current language is Chinese, then keep it as Chinese but do some changes in its order. Maintain the technical accuracy and meaning. The answer should be in very detail in Chinese, in a step-by-step manner, from reasoning to answer.")
    question_format = "The question format should be translated to the target language while preserving all technical details and meaning."
    return prompt, question_format

def createScenarioPrompt():
    prompt = rewrite_instruction.format("Change the scenario/context of the Q&A while keeping the time series analysis content the same. For example, if the current scenario is IT operations, change it to manufacturing, finance, healthcare, or any other domain that fits the time series data in CONTEXT. The answer should be in detail, in a step-by-step manner.")
    question_format = "The question format should present the same time series analysis in a different real-world scenario or domain context."
    return prompt, question_format

def createComparisonEliminatorPrompt(before, after):
    prompt = comparison_instruction
    prompt = prompt.replace("<Here is first instruction.>", before)
    prompt = prompt.replace("<Here is second instruction.>", after)
    return prompt


class RewritePrompt:
    def __init__(self, ts_idx: int, seed_q: str, seed_a: str, seed_fields: Dict[str, List[int]], instruction: str, timeseries: np.ndarray, attribute_pool: List[dict], metrics: List[str], corr_pool: List[Tuple[List[int], str]]):
        self.ts_idx = ts_idx
        self.timeseries = timeseries
        self.description = attribute_pool
        self.instruction = instruction
        self.metrics = metrics
        self.corr_pool = corr_pool

        self.all_fields = {"trend": range(len(timeseries)), "seasonal": range(len(timeseries)), "noise": range(len(timeseries)), "local": range(len(timeseries)), "statistic": range(len(timeseries)), "correlation": range(len(corr_pool))}
        self.fields = seed_fields
        self.qa_history = [(seed_q, seed_a)]

    def rewrite(self):
        # For rewrite functionality, we don't need to evolve fields
        # We keep the existing fields as they are
        pass
    
    def push(self, q: str, a: str):
        self.qa_history.append((q, a))
        if len(self.qa_history) > 2:
            self.qa_history.pop(0)

    def generate_prompt(self):
        # Randomly choose a rewrite method
        all_prompts = [createWordOrderPrompt, createExpansionPrompt, createCompressionPrompt, createTranslationPrompt, createScenarioPrompt]
        random_p = [0.1, 0.1, 0.2, 0.5, 0.1]
        prompt, question_format = np.random.choice(all_prompts, p=random_p)()
        given_qa = json.dumps({
            'question': self.qa_history[-1][0],
            'answer': self.qa_history[-1][1]
        })

        result = f"""{prompt}

        #Context#
        {attribute_prompt(self.timeseries, self.description, self.metrics, self.fields, self.corr_pool)}

        #Constraints#
        {constraints_instruction}

        #The Given Q&A#
        {given_qa}

        #Question Format#
        {question_format}

        #Generated Q&A#:"""

        return result
    
    def generate_comparison_prompt(self, q: str, a: str):
        given_qa = json.dumps({
            'question': self.qa_history[-1][0],
            'answer': self.qa_history[-1][1]
        })

        generated_qa = json.dumps({
            'question': q,
            'answer': a
        })

        result = f"""#Context#
        {attribute_prompt(self.timeseries, self.description, self.metrics, self.fields, self.corr_pool)}

        #Your Task#
        {createComparisonEliminatorPrompt(given_qa, generated_qa)}"""

        return result

    def to_dataset(self):
        return {
            "input": self.instruction + ' ' + self.qa_history[-1][0],
            "output": self.qa_history[-1][1],
            "timeseries": self.timeseries.tolist() if type(self.timeseries) == np.ndarray else self.timeseries,
            "ts_idx": self.ts_idx,
            "fields": sorted(self.fields)
        }
