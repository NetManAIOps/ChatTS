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
    This script is used to generate the stage-2 training data with LLMs for univariate time series, which can be further used as seed QAs for TSEvol.
    Usage:
        python3 -m chatts.sft.generate_uts_reason
"""

import numpy as np
import random
from tqdm import tqdm
import json
import yaml
from typing import *
from chatts.ts_generator.generate import generate_random_attributes, generate_time_series, generate_controlled_attributes, attribute_to_text, all_attribute_set
from chatts.utils.llm_utils import LLMClient, parse_llm_json
from chatts.utils.encoding_utils import timeseries_encoding, timeseries_to_list
from chatts.utils.attribute_utils import metric_to_controlled_attributes
import os


# CONFIG
TOTAL_CNT = yaml.safe_load(open("config/datagen_config.yaml"))['num_data_llm_qa']
SEQ_LEN = yaml.safe_load(open("config/datagen_config.yaml"))['seq_len']  # Set to None to enable random sequence length selection
ENCODING_METHOD = yaml.safe_load(open("config/datagen_config.yaml"))['encoding_method']
OUTPUT_BASE_DIR = yaml.safe_load(open("config/datagen_config.yaml"))['data_output_dir']
OUTPUT_DATASET = f'{OUTPUT_BASE_DIR}/llm_uts_reason_{TOTAL_CNT}_{ENCODING_METHOD}.jsonl'
OUTPUT_LABEL = f'{OUTPUT_BASE_DIR}/evol_labels/llm_uts_reason_{TOTAL_CNT}_{ENCODING_METHOD}.json'
DRYRUN = yaml.safe_load(open("config/datagen_config.yaml"))['dryrun']
LOCAL_LLM_PATH = yaml.safe_load(open("config/datagen_config.yaml"))['local_llm_path']
DISABLE_METRIC_CONFIG = yaml.safe_load(open("config/datagen_config.yaml"))["disable_metric_config"]
DISABLE_EXTREME_LENGTHS = yaml.safe_load(open("config/datagen_config.yaml"))["disable_extreme_lengths"]

# All Config for TS Attributes (type & probability)
metric_config = json.load(open('config/metric_set.json', 'rt'))

# Named reasoning templates with example questions
# Named reasoning templates with placeholders
REASONING_TEMPLATES = {
    # --- Judgment Questions (If-Then) ---
    # These questions are self-contained and require detailed explanation.
    "Multi-Trend Anomaly Judgment": {
        "example": "If an anomaly is defined as a time series that first shows an 'increase' trend and is immediately followed by a 'decrease' trend, analyze this time series. Based on this definition, does the segment from point 0 to 256 contain an anomaly?",
        "question_type": "judgment"
    },
    "Trend-Local Interaction Judgment": {
        "example": "A 'critical event' is defined as an 'upward spike' with an amplitude greater than 30 occurring during an overall 'increase' trend. Analyze this time series. Based on this rule, is there a critical event present?",
        "question_type": "judgment"
    },
    "Sequential Local Feature Judgment": {
        "example": "If a 'system fault' is defined as a 'sudden increase' with amplitude > 20 followed within 15 points by a 'downward spike' with amplitude > 15, does this time series exhibit a system fault?",
        "question_type": "judgment"
    },
    "Stable Trend Disruption Judgment": {
        "example": "A 'destabilization event' is defined as a 'sudden decrease' with an amplitude of 25 or more that occurs within a long-term 'keep steady' trend. Analyze the time series from point 0 to 200. Does it contain a destabilization event according to this definition?",
        "question_type": "judgment"
    },
    "Multi-Phase Trend Progression Judgment": {
        "example": "If a 'normal growth cycle' is defined by three phases in order: 'increase', then 'keep steady', then 'increase' again, does this time series follow the normal growth cycle pattern?",
        "question_type": "judgment"
    },
    "Contradictory Signal Judgment": {
        "example": "If a 'conflicting signal' is defined as observing a 'continuous downward spike' with total amplitude change > 40 during a period otherwise identified as a general 'increase' trend, is there a conflicting signal in this data?",
        "question_type": "judgment"
    },
    "Noise Threshold Judgment": {
        "example": "If a time series is classified as having a 'noisy' environment (noise std > 0.2), while the trend is labeled as 'keep steady', does the noise level invalidate the steady trend classification?",
        "question_type": "judgment"
    },
    "Seasonal Stability Judgment": {
        "example": "A 'stable seasonal pattern' is defined as a 'sin periodic fluctuation' with amplitude > 1.0 over time. Analyze the time series. Does the seasonal pattern meet the stability criteria?",
        "question_type": "judgment"
    },
    "Statistical Judgment": {
        "example": "If any data point lower than 10 is considered an 'outlier'. Should this time series be considered anomalous?",
        "question_type": "judgment"
    },
    "Long-Term Statistical Judgment": {
        "example": "If lower than 10 and last for more than 20 data points is considered as anomalous. Should this time series be considered anomalous?",
        "question_type": "judgment"
    },
    "Long-Term Statistical with Recovery Judgment": {
        "example": "If data point lower than 10 and not recovery to the original state is considered as anomalous. Is there any anomalies in the time series?",
        "question_type": "judgment"
    },
    "Multi-Trend Anomaly Judgment in Real-World Application": {
        "example": "For a new mobile app, an 'unsuccessful launch' is defined as user engagement that initially increases but then begins to decline within the first week. Based on this definition, does the app's user engagement data from the past 7 days indicate an unsuccessful launch?",
        "question_type": "judgment"
    },
    "Trend-Local Interaction Judgment in Real-World Application": {
        "example": "A 'critical server overload' is defined as an API response time 'upward spike' exceeding 800ms occurring during a period where the baseline response time is already showing a gradual 'increase' trend. Analyze the server's performance data. Based on this rule, is a critical server overload present?",
        "question_type": "judgment"
    },
    "Sequential Local Feature Judgment in Real-World Application": {
        "example": "A 'memory leak crash' in an application is identified by its memory usage 'suddenly increasing' by over 200MB, followed within 10 minutes by a sharp 'downward spike' of over 150MB (indicating a crash and restart). Does this application's memory usage data exhibit a memory leak crash?",
        "question_type": "judgment"
    },
    "Stable Trend Disruption Judgment in Real-World Application": {
        "example": "A 'supply chain disruption' for an e-commerce product is defined as a 'sudden decrease' in hourly sales of 50 units or more that occurs during a period where sales have otherwise been stable ('keep steady' trend). Analyze the sales data from the last 24 hours. Does it contain a supply chain disruption according to this definition?",
        "question_type": "judgment"
    },
    "Multi-Phase Trend Progression Judgment in Real-World Application": {
        "example": "A 'standard market adoption cycle' for a new financial product is defined by three phases in order: initial slow 'increase' in trading volume, followed by a period of 'keep steady' consolidation, and then another phase of 'increase'. Does the trading volume for this product follow the standard market adoption cycle?",
        "question_type": "judgment"
    },
    "Contradictory Signal Judgment in Real-World Application": {
        "example": "For a factory production line, a 'machine health alert' is triggered if a 'continuous downward spike' in output (total drop > 40 units/hour) is observed during a shift that is otherwise showing a general 'increase' in production. Is there a machine health alert in this data?",
        "question_type": "judgment"
    },
    "Noise Threshold Judgment in Real-World Application": {
        "example": "An IoT temperature sensor for a chemical process should remain steady. If 'unreliable data' is defined as sensor readings having a standard deviation greater than 2°C due to environmental noise, does this condition invalidate the conclusion that the underlying process temperature is stable ('keep steady')?",
        "question_type": "judgment"
    },
    "Seasonal Stability Judgment in Real-World Application": {
        "example": "A 'stable daily traffic pattern' for an e-commerce site is defined as having a predictable 'sin periodic fluctuation' where the peak traffic consistently exceeds 1000 users per hour. Analyze the site's traffic data. Does the daily pattern meet this criterion for being considered stable and significant?",
        "question_type": "judgment"
    },

    # --- Multiple Choice Questions ---
    # These questions offer clear options and require justification.

    "Multi-Trend Pattern Identification": {
        "example": "A server's CPU utilization data is provided. Analyze the time series and determine: This pattern is most indicative of which scenario? A) A critical system failure. B) The server reaching its processing capacity limit under heavy load. C) A normal daily cycle. D) A software bug causing random spikes. Explain your choice based on the observed trend progression.",
        "question_type": "multiple_choice"
    },
    "Local Feature Interpretation in Context": {
        "example": "A stock's price data over one month is provided. The stock has been in a general decline. If you observe any significant upward price movements, what is the most likely interpretation? A) A fundamental reversal of the stock's downward trend. B) A brief, speculative event, possibly due to a news announcement, with no long-term impact. C) The start of a seasonal rally. D) A data reporting error. Justify your selection based on your analysis.",
        "question_type": "multiple_choice"
    },
    "Best-Fit Scenario for Combined Features": {
        "example": "A system's network traffic data at 2:00 AM is provided. Analyze the overall patterns and any anomalous events. Which scenario does this pattern best represent? A) A planned data migration or system backup. B) A Distributed Denial-of-Service (DDoS) attack. C) Normal user growth. D) A network hardware malfunction. Provide reasoning based on your analysis.",
        "question_type": "multiple_choice"
    },
    "Multi-Trend Pattern Identification in Real-World Application": {
        "example": "A server's CPU utilization data over several hours is provided. Analyze the pattern and determine: This behavior is most indicative of which scenario? A) A critical system failure. B) The server reaching its processing capacity limit under heavy load. C) A normal daily cycle. D) A software bug causing random spikes. Explain your choice.",
        "question_type": "multiple_choice"
    },
    "Local Feature Interpretation in Context in Real-World Application": {
        "example": "A stock's price data over one month is provided. Analyze the time series for any significant events or anomalies. If you find notable price movements against the general trend, what is the most likely interpretation? A) A fundamental reversal of the trend. B) A brief, speculative event with no long-term impact. C) The start of a seasonal pattern. D) A data reporting error. Justify your selection.",
        "question_type": "multiple_choice"
    },
    "Best-Fit Scenario for Combined Features in Real-World Application": {
        "example": "A system's network traffic data during early morning hours is provided. Analyze the patterns and identify any significant events. Which scenario does this pattern best represent? A) A planned data migration or system backup. B) A Distributed Denial-of-Service (DDoS) attack. C) Normal user growth. D) A network hardware malfunction. Provide reasoning.",
        "question_type": "multiple_choice"
    },

    # --- Open-Ended Analytical Questions ---

    "Multi-Trend Impact Analysis": {
        "example": "A patient's blood sugar monitoring data over 4 hours is provided. Analyze the pattern and discuss what this behavior could signify for the patient's health. What would be your primary concern as a healthcare provider?",
        "question_type": "open_ended"
    },
    "Trend and Local Feature Synthesis": {
        "example": "Network latency monitoring data is provided. Analyze the overall trend and any fluctuations. How do any observed variations affect your confidence in the network's stability? Explain your analytical thinking.",
        "question_type": "open_ended"
    },
    "Positional Significance Analysis": {
        "example": "A power grid's output data throughout a day is provided. Analyze any significant events and their timing. Why is the timing of any major events particularly important for your overall assessment of the grid's stability?",
        "question_type": "open_ended"
    },
    "Multi-Trend Impact Analysis in Real-World Application": {
        "example": "A patient's blood sugar level monitoring data over several hours is provided. Analyze the pattern and discuss what this behavior could signify for the patient's health. What would be the primary concern for a healthcare provider?",
        "question_type": "open_ended"
    },
    "Trend and Local Feature Synthesis in Real-World Application": {
        "example": "Network latency monitoring data over time is provided. Analyze the overall patterns and any fluctuations. How do these observations affect your confidence in the network's stability? Explain your analytical thinking.",
        "question_type": "open_ended"
    },
    "Positional Significance Analysis in Real-World Application": {
        "example": "A power grid's output data throughout the day is provided, with focus on evening hours. Analyze the data and explain why the timing of any significant events is particularly important for assessing grid stability.",
        "question_type": "open_ended"
    }
}


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

    # Random choose a category and metric name
    sample = random.choice(list(metric_config))
    category = sample['category']
    metric = random.choice(sample['metrics'])
    fields = []

    # Choose a metric and generate
    # Generate attribute_pool and time series
    if DISABLE_METRIC_CONFIG:
        attribute_pool = generate_random_attributes(all_attribute_set['overall_attribute'], all_attribute_set['change'], seq_len=current_seq_len)
    else:
        attribute_pool = generate_controlled_attributes(metric_to_controlled_attributes(metric), seq_len=current_seq_len)

    attribute_pool['metric_name'] = metric
    attribute_pool['situation'] = category

    timeseries, attribute_pool = generate_time_series(attribute_pool, current_seq_len)

    # Scalar
    scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(timeseries, ENCODING_METHOD)

    # Generate QA
    instruction = f"This is a metric called {metric} collected from {category} with length of {current_seq_len}: {cur_ts_prompt}. "
    prompts = []
    fields = []

    # Generate random task
    task_candidates = ['reason']
    tasks = list(np.random.choice(task_candidates, size=1, replace=False))
    
    for task in tasks:
        prompt = f"I am creating a dataset for a time series analysis large language model. Based on the information I provide about the time series, I need you to generate as many rich QA pairs as possible according to the specified task requirements. This will be used as training data for the large language model. Now, I have a time series named {metric} from the {category} domain."

        if task == 'reason':
            fields.append({'trend': [0], 'seasonal': [0], 'noise': [0], 'local': [0], 'statistic': [0]})
            prompt += "The features of the given time series are as follows: "
            prompt += attribute_to_text(
                timeseries,
                attribute_pool,
                include_attributes=['length', 'trend', 'periodicity', 'frequency', 'noise', 'local', 'statistic'],
                generate_values=False
            )
            
            # Randomly select a named reasoning template.
            template_name = random.choice(list(REASONING_TEMPLATES.keys()))
            template_info = REASONING_TEMPLATES[template_name]
            
            # The example is a complete, self-contained, high-quality real-world question.
            example_question = template_info["example"]

            # This is the new part: a high-quality example of a QA pair.
            example_answer_prompt = """
---
**Example Answer (for a similar question):**

**Answer Example (Yes):**
Looking at the time series data, I can identify two key components: First, the baseline response time shows a gradual upward trend from around 200ms to 350ms throughout the observation period, satisfying the 'increase' trend requirement. Second, there's a prominent spike reaching 950ms at position 210, which exceeds the 800ms threshold specified in the definition. Since both conditions are met - an increasing baseline trend and a spike above 800ms. Therefore, the answer should be: Yes, a critical server overload is present according to the given definition. 

**Answer Example (No):**
Looking at the time series data, I can identify two key components: First, the baseline response time remains relatively stable around 250-280ms throughout the observation period, showing a 'keep steady' trend rather than the required 'increase' trend. Second, while there is a notable spike reaching 920ms at position 180, this spike occurs during a stable baseline period, not during an increasing trend. Since the first condition is not met - the baseline trend is not increasing as required by the definition. Therefore, the answer should be: No, a critical server overload is not present according to the given definition.

**Answer Example (Open-Ended):**

Looking at the time series data, the server exhibits a concerning performance degradation pattern. The baseline response time shows a gradual upward trend from around 200ms to 350ms, indicating the system is under increasing stress or resource constraints. More critically, there's a severe spike reaching 950ms at position 210, which represents a nearly 3x increase from the baseline. This pattern suggests the server is approaching its capacity limits and experiencing intermittent overload conditions. My primary concerns would be user experience degradation and potential system instability. I would recommend immediate actions including: monitoring resource utilization to identify bottlenecks, implementing load balancing or scaling solutions, and setting up alerts for response times exceeding 500ms to prevent future incidents. \n\n"""

            prompt += f"Generate QA pairs about the reasoning concept: **{template_name}**. Here is a high-quality example question: '{example_question}'"
            prompt += example_answer_prompt
            prompt += f"""
Now, create new, diverse reasoning questions about the given time series. The questions must be self-contained, set in a realistic scenario, and provide all necessary definitions for the user to make a judgment.

**Key Requirements:**
1.  **Question Diversity**: Create a mix of questions with different question formats and expressions. Some are simple questions, while some questions are not in well-formed format. For example, some questions may use only simple words with no requirements about the answer format, and some questions should be as detailed as possible, raising lots of requirements about the answer. The output list must include Q&As in **different question formats and types.**
2.  **Self-Contained & Realistic**: Ensure every question is set in a real-world context (e.g., IT, finance, e-commerce) and clearly defines the criteria for judgment (what constitutes an "anomaly", "overload", etc.). For multiple choice questions, provide **clear options** that reflect realistic scenarios. For open-ended questions, ensure they require deep analysis and reasoning.
3.  **Focus on Core Scenarios**: Questions should primarily explore interactions between overall trends (including multi-phase trends) and local features (like spikes, drops, or sudden changes).
4.  **Demand Deep Reasoning**: Answers must follow the structured format shown in the example answer above: **Analysis** -> **Evidence** -> **Reasoning** -> **Conclusion**. They must explain the 'why' behind the conclusion, not just state it. The answer should be rich and detailed, providing a comprehensive analysis of the time series data.
5. **QAs with different answers:** For yes/no questions and multi-choice questions, you should generate questions and answers with different answers, such as "Yes", "No", "A", "B", etc. For open-ended questions, you should generate questions and answers with different perspectives and conclusions. For Yes/No questions, ensure you have clearly conclude the answer and explictly state the "Yes" or "No" at the end of the reasoning part. For multi-choice questions, ensure the answer is one of the options (e.g., "A", "B", "C", "D") and clearly state the answer at the end of the reasoning part.

**CRITICAL: MAXIMIZE FORMAT DIVERSITY**
Create varied question and answer formats:

**Question Style Examples:**
- "As a network security engineer monitoring this traffic data over the past 6 hours. The company defines a 'coordinated attack' as sustained traffic increases of 300 above baseline lasting more than 15 minutes, combined with connection timeout spikes. Based on this definition and the observed data patterns, is the system currently under a coordinated attack?"
- "You're consulting for an e-commerce platform experiencing customer complaints about checkout delays. The platform considers 'critical performance degradation' as any period where both: (1) response times exceed 2 seconds for more than 5 consecutive minutes, AND (2) the baseline shows deteriorating trends. What's your professional assessment of the current system health?"

**Answer Format Guide** (use natural language, don't include these labels):
The answer should be **as detailed and long as possible**, following this structure:
[Initial observation of data] → [Specific evidence from patterns] → [Technical reasoning] → [Clear conclusion]

Mix confident/uncertain tones, quantitative/qualitative reasoning, technical/business language, brief/comprehensive responses. The responses should be very rich and detailed, providing a comprehensive analysis of the time series data.

Note that you should **rewrite** the initial question and answers into **multiple questions and answers with different format, word orders and expressions**! For Yes/No questions, ensure you have clearly conclude the answer and explictly state the "Yes" or "No" at the end of the reasoning part. For multi-choice questions, ensure the answer is one of the options (e.g., "A", "B", "C", "D") and clearly state the answer at the end of the reasoning part.

**Note:** If the given time series is not able to generate the targeted QA pairs, just return a empty list.\n\n"""
        else:
            raise ValueError(f"Unknown task: {task}")

        prompt += """Now, please strictly follow the above requirements to generate as many QA pairs as possible (if can), and include the reference text for the answers. Output in JSON format, for example: [{"question": "Strictly follow the task question 1", "answer": "Answer 1 found from the data", "reference": "Precise original text fragment for answer 1"}, {"question": "Strictly follow the task question 2", "answer": "Answer 2 found from the data", "reference": "Precise original text fragment for answer 2"}]. The included attributes in answers **must be found** from the given time series, and the answers must be accurate. The generated QA pairs should not be repetitive, and the answers can be **very detailed**. Specific time series feature must **not** be mentioned in the question (e.g., using words like "the spike of amplitude 50", "the sudden increase in the time series") as we will provide them. Just use words like "according to the time series" or "according to the values near point 50"."""

        prompts.append(prompt)

    # Generate final result
    result = []
    for prompt, field in zip(prompts, fields):
        result.append({
            'instruction': instruction,
            'prompt': prompt,
            'fields': field,
            'timeseries': [scaled_timeseries],
            'original_timeseries': [timeseries],
            'metrics': [metric],
            'attribute_pool': [attribute_pool],
            'corr_pool': []
        })

    return result

def generate_dataset():
    result = []
    prompts = []
    num_cnt = 0
    with tqdm(total=TOTAL_CNT, desc='Generating prompt...') as t:
        cnt = 0
        while True:
            try:
                cur_data = generate_prompt_data()
            except ValueError as err:
                continue
            except IndexError as err:
                continue
            for item in cur_data:
                item['ts_idx'] = num_cnt
                result.append(item)
                prompts.append(item['prompt'])
                t.update()
                cnt += 1
            if cnt >= TOTAL_CNT:
                break
            num_cnt += 1

    # Use LLM to generate answer
    if DRYRUN:
        llm_answers = ['[{"question": "This is a test question.", "answer": "This is a test answer."}]'] * len(prompts)
    else:
        llm_client = LLMClient(model_path=LOCAL_LLM_PATH, engine='vllm', batch_size=8)
        llm_answers = llm_client.llm_batch_generate(prompts, use_chat_template=True)
        llm_client.kill()

    # Parse json
    dataset = []
    labels = []
    failed_cnt = 0
    for i in tqdm(range(len(result)), desc='Parsing'):
        try:
            cur_qa_list = parse_llm_json(llm_answers[i])
            for j, qa in enumerate(cur_qa_list):
                dataset.append({
                    'input': result[i]['instruction'] + qa['question'],
                    'output': qa['answer'],
                    'timeseries': timeseries_to_list(result[i]['timeseries'])
                })
                labels.append({
                    'instruction': result[i]['instruction'],
                    'question': qa['question'],
                    'fields': result[i]['fields'],
                    'ts_idx': result[i]['ts_idx'],
                    'metrics': result[i]['metrics'],
                    'corr_pool': result[i]['corr_pool'],
                    'attribute_pool': result[i]['attribute_pool']
                })
        except Exception as err:
            failed_cnt += 1
            continue
    print(f"Parse finished. Failed count: {failed_cnt}, Success count: {len(dataset)}.")

    return dataset, labels


if __name__ == '__main__':
    # Create output directory if not exists
    os.makedirs(os.path.dirname(OUTPUT_DATASET), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_LABEL), exist_ok=True)

    result, labels = generate_dataset()
    with open(OUTPUT_DATASET, 'wt') as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(OUTPUT_LABEL, 'wt') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

    print(f"Finished! Saved to {OUTPUT_DATASET} and {OUTPUT_LABEL}.")
