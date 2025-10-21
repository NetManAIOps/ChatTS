# Copyright 2025 Tsinghua University and ByteDance.
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
"""
    This script is used to generate the stage-2 training data with LLMs for multivariate time series, which can be further used as seed QAs for TSEvol.
    Usage:
        python3 -m chatts.sft.generate_mts_reason
"""

import numpy as np
import random
from tqdm import tqdm
import json
import yaml
from typing import *
from chatts.ts_generator.generate import generate_time_series, generate_controlled_attributes, attribute_to_text, generate_random_attributes, all_attribute_set
from chatts.utils.llm_utils import LLMClient, parse_llm_json
from chatts.utils.encoding_utils import timeseries_encoding, timeseries_to_list
from chatts.utils.attribute_utils import metric_to_controlled_attributes
from chatts.sft.generate_uts_reason import REASONING_TEMPLATES as UTS_REASONING_TEMPLATES
import os
import copy


# CONFIG
TOTAL_CNT = yaml.safe_load(open("config/datagen_config.yaml"))['num_data_llm_qa']
SEQ_LEN = yaml.safe_load(open("config/datagen_config.yaml"))['seq_len']  # Set to None to enable random sequence length selection
ENCODING_METHOD = yaml.safe_load(open("config/datagen_config.yaml"))['encoding_method']
OUTPUT_BASE_DIR = yaml.safe_load(open("config/datagen_config.yaml"))['data_output_dir']
OUTPUT_DATASET = f'{OUTPUT_BASE_DIR}/llm_mts_reason_{TOTAL_CNT}_{ENCODING_METHOD}.jsonl'
OUTPUT_LABEL = f'{OUTPUT_BASE_DIR}/evol_labels/llm_mts_reason_{TOTAL_CNT}_{ENCODING_METHOD}.json'
DRYRUN = yaml.safe_load(open("config/datagen_config.yaml"))['dryrun']
LOCAL_LLM_PATH = yaml.safe_load(open("config/datagen_config.yaml"))['local_llm_path']
DISABLE_METRIC_CONFIG = yaml.safe_load(open("config/datagen_config.yaml"))["disable_metric_config"]
DISABLE_EXTREME_LENGTHS = yaml.safe_load(open("config/datagen_config.yaml"))["disable_extreme_lengths"]
MAX_SERIES_NUM = 6 # Max number of series in a multivariate set

# All Config for TS Attributes (type & probability)
metric_config = json.load(open('config/metric_set.json', 'rt'))


# NEW: Named reasoning templates for Multivariate Time Series (MTS) analysis
MTS_REASONING_TEMPLATES = {
    # --- Comparative Judgment Questions ---
    "Correlated Trend Judgment": {
        "example": "A 'healthy system scale-up' is defined as a scenario where an 'increase' in '{metric_A}' is matched by a corresponding 'increase' in '{metric_B}'. Analyze both time series. Based on this definition, is the system exhibiting a healthy scale-up?",
        "question_type": "judgment"
    },
    "Divergent Trend Judgment": {
        "example": "An 'inefficient process' is suspected if '{metric_A}' shows a 'steep increase' while '{metric_B}' remains 'steady' or 'decreases'. Analyze the provided data for both metrics. Does the time series indicate an inefficient process?",
        "question_type": "judgment"
    },
    "Causal Event Judgment": {
        "example": "A 'network-induced CPU spike' is defined as an 'upward spike' in '{metric_B}' that occurs within 15 time points *after* a 'sudden increase' in '{metric_A}'. Analyze both time series. Is there evidence of a network-induced CPU spike?",
        "question_type": "judgment"
    },
    "System-Wide Anomaly Judgment": {
        "example": "A 'critical failure state' is defined by the simultaneous occurrence of: '{metric_A}' is above 90, '{metric_B}' shows a 'sudden increase' of over 20, and '{metric_C}' drops to near zero. Analyze all provided time series. Does the system enter a critical failure state?",
        "question_type": "judgment"
    },

    # --- Comparative Multiple Choice Questions ---
    "Best-Fit Scenario Identification in Real-World Applications": {
        "example": "Given the time series for '{metric_A}' and '{metric_B}', which of the following scenarios is the most likely explanation for the observed patterns? A) A large-scale data backup operation. B) A DDoS attack. C) A user-facing application experiencing a viral traffic surge. D) A hardware failure in the network card.",
        "question_type": "multiple_choice"
    },
    # "Root Cause Analysis": {
    #     "example": "The '{metric_A}' shows a significant degradation (sharp increase). After analyzing the accompanying '{metric_B}' and '{metric_C}' series, what is the most probable root cause? A) The database is overloaded with slow queries. B) The application server is CPU-bound. C) High network latency. D) A recent code deployment introduced a bug. Explain your reasoning by linking the primary issue in {metric_A} to one of the other metrics.",
    #     "question_type": "multiple_choice"
    # },

    # --- Holistic Open-Ended Analytical Questions ---
    "Overall Assessment in Real-Wolrd Applications": {
        "example": "As a DevOps engineer, you are presented with the time series for '{metric_A}', '{metric_B}', and '{metric_C}'. According to the information from all three metrics to provide an overall assessment of the server's health and performance. Are there any emerging issues or risks you would report?",
        "question_type": "open_ended"
    },
    # "Inter-Metric Relationship Analysis": {
    #     "example": "Analyze the relationship between '{metric_A}' and '{metric_B}'. Does an increase in one consistently lead to a change in the other? Are there any periods where this relationship breaks down? Discuss the potential business implications of your findings.",
    #     "question_type": "open_ended"
    # },
}

# Example answer prompt for UTS questions
UTS_EXAMPLE_ANSWER_PROMPT = """
---
**Example Answer (for a similar question):**

**Answer Example (Yes):**
Looking at the time series data, I can identify two key components: First, the baseline response time shows a gradual upward trend from around 200ms to 350ms throughout the observation period, satisfying the 'increase' trend requirement. Second, there's a prominent spike reaching 950ms at position 210, which exceeds the 800ms threshold specified in the definition. Since both conditions are met - an increasing baseline trend and a spike above 800ms. Therefore, the answer should be: Yes, a critical server overload is present according to the given definition. 

**Answer Example (No):**
Looking at the time series data, I can identify two key components: First, the baseline response time remains relatively stable around 250-280ms throughout the observation period, showing a 'keep steady' trend rather than the required 'increase' trend. Second, while there is a notable spike reaching 920ms at position 180, this spike occurs during a stable baseline period, not during an increasing trend. Since the first condition is not met - the baseline trend is not increasing as required by the definition. Therefore, the answer should be: No, a critical server overload is not present according to the given definition.

**Answer Example (Open-Ended):**
Looking at the time series data, the server exhibits a concerning performance degradation pattern. The baseline response time shows a gradual upward trend from around 200ms to 350ms, indicating the system is under increasing stress or resource constraints. More critically, there's a severe spike reaching 950ms at position 210, which represents a nearly 3x increase from the baseline. This pattern suggests the server is approaching its capacity limits and experiencing intermittent overload conditions. My primary concerns would be user experience degradation and potential system instability. I would recommend immediate actions including: monitoring resource utilization to identify bottlenecks, implementing load balancing or scaling solutions, and setting up alerts for response times exceeding 500ms to prevent future incidents. \n\n"""

# NEW: Example answer prompt for MTS questions
MTS_EXAMPLE_ANSWER_PROMPT = """
---
**Example Answer (for a similar comparative question):**

**Answer Example (Judgment - Yes):**
After analyzing both time series, I can confirm that the system is exhibiting a healthy scale-up. The CPU usage data shows a clear 'increase' trend, rising from 20 to 60 over the period. Concurrently, the Memory usage also shows a corresponding 'increase' trend, moving from 4GB to 12GB. The trends are positively correlated.

Because both metrics are increasing together as per the definition, my conclusion is: Yes, the system is demonstrating a healthy scale-up.

**Answer Example (Judgment - No):**
After analyzing both time series, I see a clear divergence. The CPU usage data shows a steep 'increase' trend, climbing from 1,000 to 8,000 requests per second. However, the Memory usage remains 'steady' at around 500 transactions per second, failing to keep pace.

Since the throughput does not increase with the request load, this signals a bottleneck. Therefore, my conclusion is: No, this pattern indicates an inefficient process, not a healthy one.

**Answer Example (Open-Ended):**
Based on a holistic analysis of the provided metrics, the server's health is poor and at immediate risk. While CPU usage is fluctuating within a normal range, Memory usage shows a steady, non-stop increase, which is a classic symptom of a memory leak. This is corroborated by the API Latency, which shows intermittent, severe spikes that align with periods of high memory consumption.

My primary concern is an imminent application crash due to memory exhaustion. I would recommend an immediate application restart to temporarily resolve the issue, followed by a thorough investigation of recent code changes to identify and fix the memory leak. \n\n"""


def generate_prompt_data():
    # Determine sequence length
    if SEQ_LEN is None:
        p = random.random()
        if p > 0.4:
            current_seq_len = 256
        elif p > 0.1 or DISABLE_EXTREME_LENGTHS:
            current_seq_len = random.randint(64, 1024)
        elif p > 0.05:
            current_seq_len = random.randint(5, 64)
        else:
            current_seq_len = random.randint(1024, 4096)
    else:
        current_seq_len = SEQ_LEN

    # Randomly choose a category with enough metrics for MTS
    possible_categories = [s for s in metric_config if len(s['metrics']) >= 2]
    if not possible_categories:
        raise ValueError("No categories found with 2 or more metrics for MTS generation.")
    sample_category = random.choice(possible_categories)
    category = sample_category['category']
    
    # Determine number of series to generate and sample metrics
    max_num = min(MAX_SERIES_NUM, len(sample_category['metrics']))
    num_series = random.randint(2, max_num)
    metrics = list(np.random.choice(sample_category['metrics'], size=num_series, replace=False))

    all_timeseries, all_attribute_pools, all_scaled_timeseries, all_ts_prompts = [], [], [], []

    # Generate each time series
    for metric in metrics:
        # Generate attribute_pool and time series
        if DISABLE_METRIC_CONFIG:
            attribute_pool = generate_random_attributes(all_attribute_set['overall_attribute'], all_attribute_set['change'], seq_len=current_seq_len)
        else:
            attribute_pool = generate_controlled_attributes(metric_to_controlled_attributes(metric), seq_len=current_seq_len)

        attribute_pool['metric_name'] = metric
        attribute_pool['situation'] = category

        timeseries, attribute_pool = generate_time_series(attribute_pool, current_seq_len)
        scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(timeseries, ENCODING_METHOD)
        
        all_timeseries.append(timeseries)
        all_attribute_pools.append(attribute_pool)
        all_scaled_timeseries.append(scaled_timeseries)
        all_ts_prompts.append(f"'{metric}': {cur_ts_prompt}")

    # Combine prompts for all time series
    metric_names = ", ".join([f"'{m}'" for m in metrics])
    combined_ts_prompt = "\n - ".join(all_ts_prompts)
    
    # Decide whether to ask a UTS or MTS question
    # Ask a UTS question if there's only one series or by random chance
    is_uts_question = (num_series == 1) or (random.random() < 0.4) 

    instruction = f"I have a set of {num_series} metrics from {category}: \n - {combined_ts_prompt}\n\n"
    analyzed_idx_list = []

    if is_uts_question:
        # --- Generate a UTS-style question about one of the series ---
        idx = random.randint(0, num_series - 1)
        analyzed_idx_list.append(idx)
        chosen_metric = metrics[idx]        
        template_name = random.choice(list(UTS_REASONING_TEMPLATES.keys()))
        template_info = UTS_REASONING_TEMPLATES[template_name]
        example_question = template_info["example"]
        example_answer_prompt = UTS_EXAMPLE_ANSWER_PROMPT
        
        prompt = f"I am creating a dataset for a time series analysis model. Based on the provided time series, generate rich QA pairs. Now, I have a time series named '{chosen_metric}' from the {category} domain."
        prompt += f"The features of the '{chosen_metric}' series are as follows: "
        prompt += attribute_to_text(
            all_timeseries[idx],
            all_attribute_pools[idx],
            include_attributes=['length', 'trend', 'periodicity', 'frequency', 'noise', 'local', 'statistic'],
            generate_values=False
        )
        prompt += f"Generate QA pairs about the reasoning concept: **{template_name}**. Here is a high-quality example question: '{example_question}'"
    
    else:
        # --- Generate an MTS-style question comparing multiple series ---
        template_name = random.choice(list(MTS_REASONING_TEMPLATES.keys()))
        template_info = copy.deepcopy(MTS_REASONING_TEMPLATES[template_name])
        
        # Dynamically select metrics for the question and fill placeholders
        metric_placeholders = ['{metric_A}', '{metric_B}', '{metric_C}']
        num_metrics_needed = sum(1 for p in metric_placeholders if p in template_info['example'])
        
        if num_series < num_metrics_needed:
            # Fallback to a simpler template if not enough metrics are available
            simple_templates = {k: v for k, v in MTS_REASONING_TEMPLATES.items() if '{metric_C}' not in v['example']}
            template_name = random.choice(list(simple_templates.keys()))
            template_info = copy.deepcopy(MTS_REASONING_TEMPLATES[template_name])
            num_metrics_needed = sum(1 for p in metric_placeholders if p in template_info['example'])

        selected_metrics = random.sample(metrics, k=num_metrics_needed)
        analyzed_idx_list.extend([metrics.index(m) for m in selected_metrics])
        
        format_dict = {}
        for i, placeholder in enumerate(metric_placeholders):
            if i < len(selected_metrics):
                format_dict[placeholder.strip('{}')] = selected_metrics[i]
        
        example_question = template_info["example"].format(**format_dict)
        example_answer_prompt = MTS_EXAMPLE_ANSWER_PROMPT

        prompt = f"I am creating a dataset for a time series analysis model. Based on the provided multivariate time series, generate rich QA pairs that compare or synthesize information across them. Now, I have a set of time series from the {category} domain: {metric_names}."
        prompt += f"Generate QA pairs about the reasoning concept: **{template_name}**. Here is a high-quality example question: '{example_question}'"

    # Common part of the prompt
    prompt += example_answer_prompt
    prompt += f"""
Now, create new, diverse reasoning questions about the given time series. The questions must be self-contained, set in a realistic scenario, and provide all necessary definitions for the user to make a judgment.

**Key Requirements:**
0. **Explicitly give the target metric name**: Always include the target metric name in the question, e.g., "the CPU usage of the server" or "the memory usage of the application". Do not use generic terms like "the time series" or "the data".
1.  **Question Diversity**: Create a mix of questions with different question formats and expressions. You should **rewrite** the initial question and answers into **multiple questions and answers with different format, word orders and expressions**. Some are simple questions, while some questions are not in well-formed format. For example, some questions may use only simple words with no requirements about the answer format, and some questions should be as detailed as possible, raising lots of requirements about the answer. The output list must include Q&As in **different question formats and types.**
2.  **Self-Contained & Realistic**: Ensure every question is set in a real-world context (e.g., IT, finance, e-commerce) and clearly defines the criteria for judgment.
3.  **Focus on Core Scenarios**: For single-series questions, focus on trend/local feature interactions. For multi-series questions, focus on correlation, causation, and divergence.
4.  **Demand Deep Reasoning**: Answers must follow the format of deep reasoning in the example, which is as detailed as possible.
5. **QAs with different answers:** For yes/no and multi-choice questions, generate questions and answers with different final answers (e.g., "Yes", "No", "A", "B"). For Yes/No questions, ensure you have clearly conclude the answer and explictly state the "Yes" or "No" at the end of the reasoning part. For multi-choice questions, ensure the answer is one of the options (e.g., "A", "B", "C", "D") and clearly state the answer at the end of the reasoning part.

**CRITICAL: MAXIMIZE FORMAT DIVERSITY**
Create varied question and answer formats (Direct, Scenario-based, Conversational, Diagnostic, etc.). Mix confident/uncertain tones, and technical/business language.

**Question Style Examples:**
- "As a network security engineer monitoring this traffic data over the past 6 hours. The company defines a 'coordinated attack' as sustained traffic increases of 300 above baseline lasting more than 15 minutes, combined with connection timeout spikes. Based on this definition and the observed data patterns, is the system currently under a coordinated attack?"
- "You're consulting for an e-commerce platform experiencing customer complaints about checkout delays. The platform considers 'critical performance degradation' as any period where both: (1) response times exceed 2 seconds for more than 5 consecutive minutes, AND (2) the baseline shows deteriorating trends. What's your professional assessment of the current system health?"

The answer should be **as detailed and long as possible**, following this structure:
[Initial observation of data] → [Specific evidence from patterns] → [Technical reasoning] → [Clear conclusion]

Note that you should **rewrite** the initial question and answers into **multiple questions and answers with different format, word orders and expressions**! For Yes/No questions, ensure you have clearly conclude the answer and explictly state the "Yes" or "No" at the end of the reasoning part. For multi-choice questions, ensure the answer is one of the options (e.g., "A", "B", "C", "D") and clearly state the answer at the end of the reasoning part.

**Note:** If the given time series cannot support the targeted QA pairs, return an empty list.\n\n"""
    prompt += """Now, please strictly follow the above requirements to generate as many QA pairs as possible (if can), and include the reference text for the answers. Output in JSON format, for example: [{"question": "Strictly follow the task question 1 of different format and expressions", "answer": "Answer 1 found from the data, which is in reasoning format and as detailed as possible, with different expressions", "reference": "Precise original text fragment for answer 1"}, ...]. The included attributes in answers **must be found** from the given time series, and the answers must be accurate. The generated QA pairs should not be repetitive. Specific time series features must **not** be mentioned in the question (e.g., "the spike of amplitude 50") as we will provide them. Just use words like "according to the time series" or "based on the provided data"."""

    # Generate final result
    result = {
        'instruction': instruction,
        'prompt': prompt,
        'fields': {'trend': analyzed_idx_list, 'seasonal': analyzed_idx_list, 'noise': analyzed_idx_list, 'local': analyzed_idx_list, 'statistic': analyzed_idx_list},
        'timeseries': all_scaled_timeseries,
        'original_timeseries': all_timeseries,
        'metrics': metrics,
        'attribute_pool': all_attribute_pools,
        'corr_pool': []
    }

    return [result]


def generate_dataset():
    """Main function to generate the entire dataset."""
    result_data = []
    prompts = []
    num_cnt = 0
    with tqdm(total=TOTAL_CNT, desc='Generating MTS prompts...') as t:
        while len(result_data) < TOTAL_CNT:
            try:
                cur_data_list = generate_prompt_data()
            except (ValueError, IndexError) as err:
                # print(f"Skipping due to data generation error: {err}")
                continue
            
            for item in cur_data_list:
                item['ts_idx'] = num_cnt
                result_data.append(item)
                prompts.append(item['prompt'])
                t.update(1)
                if len(result_data) >= TOTAL_CNT:
                    break
            num_cnt += 1
            if len(result_data) >= TOTAL_CNT:
                break

    # Use LLM to generate answers
    if DRYRUN:
        llm_answers = ['[{"question": "This is a test MTS question.", "answer": "This is a test MTS answer."}]'] * len(prompts)
    else:
        llm_client = LLMClient(model_path=LOCAL_LLM_PATH, engine='vllm', batch_size=8)
        llm_answers = llm_client.llm_batch_generate(prompts, use_chat_template=True)
        llm_client.kill()

    # Parse JSON and create final dataset
    dataset = []
    labels = []
    failed_cnt = 0
    for i in tqdm(range(len(result_data)), desc='Parsing LLM outputs'):
        try:
            cur_qa_list = parse_llm_json(llm_answers[i])
            for j, qa in enumerate(cur_qa_list):
                if 'question' in qa and 'answer' in qa:
                    dataset.append({
                        'input': result_data[i]['instruction'] + qa['question'],
                        'output': qa['answer'],
                        'timeseries': timeseries_to_list(result_data[i]['timeseries'])
                    })
                    labels.append({
                        'instruction': result_data[i]['instruction'],
                        'question': qa['question'],
                        'fields': result_data[i]['fields'],
                        'ts_idx': result_data[i]['ts_idx'],
                        'metrics': result_data[i]['metrics'],
                        'corr_pool': result_data[i]['corr_pool'],
                        'attribute_pool': result_data[i]['attribute_pool']
                    })
                else:
                    failed_cnt += 1
        except Exception as err:
            # print(f"Failed to parse LLM output at index {i}: {err}")
            failed_cnt += 1
            continue
    print(f"Parse finished. Failed count: {failed_cnt}, Success count: {len(dataset)}.")

    return dataset, labels


if __name__ == '__main__':
    # Create output directory if not exists
    os.makedirs(os.path.dirname(OUTPUT_DATASET), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_LABEL), exist_ok=True)

    result, labels = generate_dataset()
    with open(OUTPUT_DATASET, 'wt', encoding='utf-8') as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(OUTPUT_LABEL, 'wt', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

    print(f"Finished! Saved {len(result)} QA pairs to {OUTPUT_DATASET} and {OUTPUT_LABEL}.")
