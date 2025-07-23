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


depth_instruction = """You will act as a Q&A Rewriter for a time series question-answering system.

Objective: Rewrite the provided Q&A to increase complexity and nuance, creating a challenge for advanced AI systems by using CONTEXT information related to the time series. The rewritten Q&A should remain logical, understandable, and respondable by humans.

Instructions:
- **Time Series Attributes**: Only use information from CONTEXT; do not invent additional details outside of this context.
- **Non-Text Elements**: Retain any non-text parts in #The Given Q&A#, including tables, charts, or code.
- **Input Integrity**: Ensure all inputs from #The Given Q&A# are included in the rewrite.

You SHOULD add complexity by:
{} 
Limit additional words in #Generated Q&A# to 10-20 beyond #The Given Q&A#.
Do not use terms like '#The Given Q&A#', '#Generated Q&A#', 'given q&a', or 'rewritten q&a' in #Generated Q&A#."""


breadth_instruction = """You will act as a Q&A Creator for a time series question-answering system.

Objective: Create a new Q&A inspired by #Given Q&A#, aligning with the CONTEXT section. This new Q&A should explore the same domain but focus on a rarer, distinctive aspect within that context.

Instructions:
- **Time Series Attributes**: Reference only CONTEXT attributes; avoid adding details not explicitly provided.
- **Domain Consistency with Rarity**: Keep the new Q&A in the same domain, but make it unique by focusing on rare scenarios, events, or relationships.

You SHOULD create the Q&A by:
{} 
The #Generated Q&A# must be reasonable and human-readable.
Do not use terms like '#Given Q&A#', '#Generated Q&A#', 'given q&a', or 'created q&a' in #Generated Q&A#."""

constraints_instruction = """
- **Logical Consistency**: Ensure the answer logically follows the question and aligns with CONTEXT.
- **No time series details in Questions**: In the questions, use general language about the time series without mentioning specific attributes (e.g., avoid specifics like "noise of 0.5" or "spike near position 100"). Specific details can **only** appear in the answer, drawing directly from CONTEXT.
- **Cross-Verification**: Verify all details against CONTEXT to ensure accuracy.
- **No New Features or Names**: Use only attributes and names defined in CONTEXT.
- **One Question, One Answer**: Limit to a single question and answer; keep it clear and concise.
- **Unit and Start Information**: If specific time series units or starting values are provided, ensure the question includes this information (e.g., unit is days, start time is October 1, 2024, at 15:00).
- **Output Format**: Respond in JSON only: {"question": "your question (strictly following the format in the question format)", "answer": "your answer"}. Do not include task labels like '#Given Q&A#' or '#Generated Q&A#'."""


comparison_instruction = """Here are two Instructions to ChatGPT AI, do you think they are equal to each other, which meet any one of the following requirements:
    1. Their questions and answers are almost the same, with only minor modification in terms of the order of the sequences.
    2. The second QA is an simple and obvious inference from the first QA
    3. No difference between the breadths or depths of the two QAs

If you think they are equal, then just answer Equal.
If they are not equal, then do you think it is a valid Q&A that meets all of the below requirements:
    1. All information about the time series in the **second** Q&A can be sourced from the CONTEXT section and not generated without CONTEXT.
    2. The question should not reveal specific time series attributes (e.g., avoid terms like "noise of 0.5" or "spike near position 100"), as these details are intended to appear **only** in the answer based on CONTEXT.


The First Q&A: <Here is first instruction.>
The Second Q&A: <Here is second instruction.>

Your Judgement (Just answer: Equal/Invalid/Valid. No need to explain the reason.):"""


def createSituationPrompt():
    prompt = breadth_instruction.format("""Based on the time series data in CONTEXT, create a virtual scenario with:

Real-World Context: Set a realistic scenario (e.g., specific industry, system, environment) relevant to the time series data.
Detailed Questions: Generate questions (multiple-choice or Q&A) about one of the timeseries or compare many of them.
                                        
Requirements:
Only use time series attributes provided in CONTEXT.
Make questions concrete and specific to the scenario.
Output Format: JSON only: {"question": "your question", "answer": "your answer"}

Examples:
E-commerce Holiday Sales:
Question: "The sales data starts from June 7th, and each point represent a day. A E-commerce Holiday Sales is happend every year. During this time, the sales will be higher than the normal times. How many holiday sales peaks are present?"
Answer: "In the timeseries, I've found 3 upward spikes compared to the original timeseries. Therefore, the time series shows 3 additional peaks in May, November, and December."
Manufacturing Plant Energy Usage:
Question: "The energy usage data starts from Jan 1, and each point is a day. During this period, energy saving may conduct in some of the days, when the energy consumption may drop. In the time series, on how many days did energy consumption drop by over 20%?"
Answer: "In the timeseries, I've found that there are lower values near point xxx, xxx, xxx. Therefore, Energy consumption decreased by over 20% on 5 days: January 15, February 20, March 5, April 18, and May 12."
""")
    question_format = "The question format: a description of the current situation (the generated virtual scenario, like a event or holiday or some special things, etc), specifying its unit and start point (e.g. The energy usage data starts from Jan 1, and each point is a day), along with background context relevant to the question (some special settings in your virtual scenario, like there are sales / holiday / weekends, etc), as shown in the example. Finally, the question should directly relate to the timeseries, and all elements must be included exactly as instructed, with no omissions or deviations. The question could be a multiple-choice question (preferred), or just a general q&a question with detailed explaination. "
    return prompt, question_format

def createDeductivePrompt():
    prompt = breadth_instruction.format("""
""")
    question_format = """"""
    return prompt, question_format

def createConstraintsPrompt():
    prompt = depth_instruction.format("Please add one more constraints/requirements into #The Given Q&A# according to the time series attributes provided in CONTEXT.")
    question_format = "The question format should be like: Your generated condition, the question about timeseries."
    return prompt, question_format

def createDeepenPrompt():
    prompt = depth_instruction.format(
        "If #The Given Q&A# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.")
    question_format = "The question format should be similar with the given qa, but the depth and breadth of the inquiry can be increased."
    return prompt, question_format

def createConcretizingPrompt():
    prompt = depth_instruction.format("Please replace general concepts with more specific concepts.")
    question_format = "The question format should be similar with the given qa, but should replace general concepts with more specific concepts."
    return prompt, question_format

def createComplexReasoningPrompt():
    prompt = breadth_instruction.format(
        "If #The Given Q&A# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.")
    question_format = "The question format should be like a hard math problem or other complex reasoning problem, which may be different from the original Q&A. "
    return prompt, question_format

def createDeductiveReasoningPrompt():
    prompt = """
Based on the time series in CONTEXT, create deductive reasoning Q&A. Each question includes a condition and asks if the behavior aligns with it. Use Yes/No format and provide reasoning.

Key Points:
- Conditions can be rules (e.g., thresholds), contexts, or patterns.
- Avoid specific details from the time series (e.g., "spike at point X").
- Provide both positive (Yes) and negative (No) examples.

Example:
Question: If CPU usage above 50 is abnormal, should the current usage be treated as abnormal?
Answer: No, because the usage did not exceed the threshold of 50.
"""
    question_format = """
- Yes/No question based on a condition.
- Avoid specific time series details.
- Answer starts with Yes/No, followed by reasoning.
"""
    return prompt, question_format

def createCausalReasoningPrompt():
    prompt = """
Create causal reasoning Q&A for time series. Questions prompt the model to infer causes or effects using multiple-choice answers.

Key Points:
- Categories: Cause Identification, Effect Prediction, Anomaly Explanation, Temporal Correlation.
- Avoid specific details from the time series (e.g., "spike at point X").
- Include detailed reasoning for the answers.

Example:
Question: What likely caused the observed pattern? Choose from: load increase, maintenance, stable operations.
Answer: Load increase. The spikes suggest unexpected surges in system load.
"""
    question_format = """
- Multiple-choice question about causes or effects.
- Categories: Cause Identification, Effect Prediction, Anomaly Explanation, Temporal Correlation.
- Answer includes choice and reasoning.
"""
    return prompt, question_format

def createComparisonEliminatorPrompt(before, after):
    prompt = comparison_instruction
    prompt = prompt.replace("<Here is first instruction.>", before)
    prompt = prompt.replace("<Here is second instruction.>", after)
    return prompt


class EvolPrompt:
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

    def evol(self):
        # Get diff between fields and seed_fields
        diff_fields = {}
        for field in self.all_fields:
            if field not in self.fields:
                if len(self.all_fields[field]) > 0:
                    diff_fields[field] = self.all_fields[field]
            elif len(set(self.all_fields[field]) - set(self.fields[field])) > 0:
                diff_fields[field] = list(set(self.all_fields[field]) - set(self.fields[field]))
        
        # Random choose a field not in self.fields and add it
        if len(diff_fields) > 0:
            field = np.random.choice(list(diff_fields.keys()))
            self.fields.setdefault(field, [])
            self.fields[field].append(np.random.choice(diff_fields[field]))
    
    def push(self, q: str, a: str):
        self.qa_history.append((q, a))
        if len(self.qa_history) > 2:
            self.qa_history.pop(0)

    def generate_prompt(self):
        # Randomly choose a prompt
        all_prompts = [createSituationPrompt, createConstraintsPrompt, createDeepenPrompt, createConcretizingPrompt, createComplexReasoningPrompt, createDeductiveReasoningPrompt, createCausalReasoningPrompt]
        prompt, question_format = np.random.choice(all_prompts)()
        given_qa = json.dumps({
            'question': self.qa_history[-1][0],
            'answer': self.qa_history[-1][1]
        })

        result = f"""{prompt}

        #Context#
        {attribute_prompt(self.timeseries, self.description, self.metrics, self.fields, self.corr_pool)}

        #Contraints#
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
