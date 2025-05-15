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
"""
    This script is used to generate the stage-2 training data with LLMs for univariate time series, which can be further used as seed QAs for TSEvol.
    Usage:
        python3 -m chatts.generate_llm_qa
"""

import numpy as np
import random
from tqdm import tqdm
import json
from typing import *
from chatts.ts_generator import generate_time_series, generate_controlled_attributes, attribute_to_text
from chatts.llm_utils import llm_batch_generate, parse_llm_json
from chatts.encoding_utils import timeseries_encoding, timeseries_to_list
from chatts.attribute_utils import metric_to_controlled_attributes
import os


# CONFIG
TOTAL_CNT = 1000
SEQ_LEN = 256  # Set to None to enable random sequence length selection
ENCODING_METHOD = 'sp'
OUTPUT_BASE_DIR = json.load(open("config/datagen_config.json"))["data_output_dir"]
OUTPUT_DATASET = f'{OUTPUT_BASE_DIR}/llm_qa_{TOTAL_CNT}_{ENCODING_METHOD}.jsonl'
OUTPUT_LABEL = f'{OUTPUT_BASE_DIR}/evol_labels/llm_qa_{TOTAL_CNT}_{ENCODING_METHOD}.json'
DRYRUN = False


# All Config for TS Attributes (type & probability)
metric_config = json.load(open('config/metric_set.json', 'rt'))


def generate_prompt_data():
    # Determine the current sequence length
    if SEQ_LEN is None:
        current_seq_len = random.randint(64, 1024)
    else:
        current_seq_len = SEQ_LEN

    # Random choose a category and metric name
    sample = random.choice(list(metric_config))
    category = sample['category']
    metric = random.choice(sample['metrics'])
    fields = []

    # Choose a metric and generate
    attribute_pool = generate_controlled_attributes(metric_to_controlled_attributes(metric))
    timeseries, attribute_pool = generate_time_series(attribute_pool, current_seq_len)

    # Scalar
    scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(timeseries, ENCODING_METHOD)

    # Generate QA
    instruction = f"You are a time series analysis expert. This is a metric called {metric} collected from {category} with length of {current_seq_len}: {cur_ts_prompt}."
    prompts = []
    fields = []

    # Generate random task
    task_candidates = ['stl', 'local-all', 'statistic-all', 'statistic-part']
    tasks = list(np.random.choice(task_candidates, size=1, replace=False))
    
    for task in tasks:
        prompt = f"I am creating a dataset for a time series analysis large language model. Based on the information I provide about the time series, I need you to generate as many rich QA pairs as possible according to the specified task requirements. This will be used as training data for the large language model. Now, I have a time series named {metric} from the {category} domain."

        if task == 'stl':
            fields.append({'trend': [0], 'seasonal': [0], 'noise': [0]})
            prompt += "From the overall trend, periodicity, and noise characteristics, the description of this time series is as follows:"
            prompt += attribute_to_text(
                timeseries,
                attribute_pool,
                include_attributes=['length', 'trend', 'periodicity', 'frequency', 'noise'],
                generate_values=False
            )
            prompt += "Now, I need you to generate some questions and answers about this time series based on the information provided above. Some optional questions include: asking about the trend, periodicity, noise, etc., in different ways, and trying to combine the metric with the environment to ask comprehensive questions (e.g., asking about the trend of CPU Usage to explain what problem it is experiencing)."
        elif task == 'local-all':
            fields.append({'local': [0]})
            if len(attribute_pool['local']) == 0:
                continue
            prompt += "From the local fluctuations in ths time series, the description is as follows:"
            prompt += attribute_to_text(
                timeseries,
                attribute_pool,
                include_attributes=['local'],
                generate_values=False
            )
            prompt += "Now, I need you to generate some questions and answers about this time series based on the information provided above. Some optional questions include: asking about the characteristics of different local features of the time series, or asking what kind of feature fluctuations occurred in a certain time interval (from point X to point Y), or asking whether a certain type of local fluctuation occurred, etc., in different ways, and trying to combine the metric with the environment to ask comprehensive questions (e.g., asking about if there is a spike in CPU Usage to explain what problem it is experiencing)."
        elif task == 'statistic-all':
            fields.append({'statistic': [0]})
            prompt += "From the perspective of statistic, the information about this time series is:"
            prompt += attribute_to_text(
                timeseries,
                attribute_pool,
                include_attributes=['length'],
                generate_values=True
            )
            prompt += "Some other information fyi: " + ";".join([f"The value of point {int(i)} is {float(timeseries[i]):.2f}" for i in np.random.choice(range(current_seq_len), 5)]) + '. '
            
            prompt += "Now, I need you to generate some questions and answers about this time series based on the information provided above. Some optional questions include: asking about the max/min values, period values, the value of some of the data points above, etc., in different ways, and trying to combine the metric with the environment to ask comprehensive questions."
        elif task == 'statistic-part':
            fields.append({'statistic': [0]})
            prompt += "From the perspective of period statistic, the information about this time series is:"
            start_idx = random.choice(range(current_seq_len - 10))
            end_idx = min(start_idx + random.choice(range(5, 20)), current_seq_len)
            period_ts = timeseries[start_idx:end_idx]
            prompt += f"In the time series data points from {start_idx} to {end_idx}, the values are: {', '.join([f'{float(i):.2f}' for i in period_ts])}. During this period, the difference between the rightmost and leftmost values is {float(period_ts[-1] - period_ts[0]):.2f}, the maximum value is {float(np.max(period_ts)):.2f}, the minimum value is {float(np.min(period_ts)):.2f}, the average value is {float(np.mean(period_ts)):.2f}, and the standard deviation is {float(np.std(period_ts)):.2f}."
            
            prompt += f"Now, I need you to generate some questions and answers about this period of time series (between point {start_idx} to point {end_idx}) based on the information provided above. Some optional questions include: asking about the max/min values, the shape of this period, the trend of this period, the value of some of the data points above, etc., in different ways, and trying to combine the metric with the environment to ask comprehensive questions."

        prompt += """Now, please strictly follow the above requirements to generate as many QA pairs as possible, and include the reference text for the answers. Output in JSON format, for example: [{"question": "Strictly follow the task question 1", "answer": "Answer 1 found from the data", "reference": "Precise original text fragment for answer 1"}, {"question": "Strictly follow the task question 2", "answer": "Answer 2 found from the data", "reference": "Precise original text fragment for answer 2"}]. Please note that you need to ask questions in as many forms as possible, such as active-passive conversion, logical reasoning, multiple-choice questions, search questions, etc. However, the features in answers must be found from the original data, and the answers must be accurate. The generated QA pairs should not be repetitive, and the answers can be relatively long and rich, leaning towards human preferences. Specific time series feature must **not** be mentioned in the question (e.g., using words like "the spike of amplitude 50", "the sudden increase in the time series") as we will provide them. Just use words like "according to the time series" or "according to the values near point 50". I hope you can ask questions by combining the physical meaning and scenarios of the metrics as much as possible, just like a professional analysis expert. """

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
        llm_answers = llm_batch_generate(prompts, use_chat_template=True)

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
