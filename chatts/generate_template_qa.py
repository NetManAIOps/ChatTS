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
    This script is used to generate the stage-1 data with a set of prompts, which can be further used as seed QAs for TSEvol.
    We only show some simple templates here for demonstration. You can modify the code to generate your own QA dataset.
    Usage:
        python3 -m chatts.generate_template_qa
"""

import random
from tqdm import tqdm
import json
import os
from typing import *
from chatts.ts_generator import generate_controlled_attributes, generate_time_series, attribute_to_text
from chatts.encoding_utils import timeseries_encoding, timeseries_to_list
from chatts.attribute_utils import metric_to_controlled_attributes


# CONFIG
ENCODING_METHOD = 'sp'
SEQ_LEN = None  # Set to none for random seq_len
TOTAL_CNT = 1000
OUTPUT_DATASET = f'result/template_qa_{TOTAL_CNT}_{ENCODING_METHOD}.jsonl'
OUTPUT_LABEL = f'labels/template_qa_{TOTAL_CNT}_{ENCODING_METHOD}.json'

# All Config for TS Attributes (type & probability)
metric_config = json.load(open('config/metric_set.json', 'rt'))


def univariate_seed_qa():
    if SEQ_LEN is None:
        current_seq_len = random.randint(64, 1024)
    else:
        current_seq_len = SEQ_LEN

    # Randomly choose a type and metric name
    sample = random.choice(list(metric_config))
    category = sample['category']
    metric = random.choice(sample['metrics'])

    # Choose a metric and generate
    attribute_pool = generate_controlled_attributes(metric_to_controlled_attributes(metric))
    timeseries, attribute_pool = generate_time_series(attribute_pool, current_seq_len)

    # Scalar
    scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(timeseries, ENCODING_METHOD)

    # Generate QA
    instruction = f"You are a time series analysis expert. This is a metric called {metric} collected from {category} with length of {current_seq_len}: {cur_ts_prompt}."
    questions, answers, fields = [], [], []
    # (Step 1) general shape
    questions.append("Now, please analyze the characteristics of this time series from the perspectives of periodicity, trend, local characteristics, frequency characteristics, and noise.")
    answers.append(attribute_to_text(timeseries, attribute_pool, generate_values=False))
    cur_fields = {"trend": [0], "seasonal": [0], "noise": [0], "local": [0]}
    fields.append(cur_fields)

    # (Step 2) shape and values
    questions.append("Now, please analyze the characteristics of this time series from the perspectives of periodicity, trend, local characteristics, frequency characteristics, and noise. Also include the approximate mean values for every 16 points, as well as the maximum and minimum values of the time series (rounded to 2 decimal places).")
    answers.append(attribute_to_text(timeseries, attribute_pool, generate_values=True))
    cur_fields = {"trend": [0], "seasonal": [0], "noise": [0], "local": [0], "statistic": [0]}
    fields.append(cur_fields)

    # (Step 3) local change
    for local_char in attribute_pool['local']:
        question_position = local_char['position_start'] + random.randint(-5, 5)
        questions.append(f"Is there a local characteristic fluctuation starting around point {question_position} in this time series?")
        answers.append(f"Yes, this time series " + local_char['detail'])
        cur_fields = {"local": [0]}
        fields.append(cur_fields)

    # Generate final result
    result = []
    for q, a, f in zip(questions, answers, fields):
        result.append({
            'instruction': instruction,
            'question': q,
            'answer': a,
            'fields': f,
            'metrics': [metric],
            'attribute_pool': [attribute_pool],
            'timeseries': [scaled_timeseries],
            'original_timeseries': [timeseries],
            'corr_pool': []
        })

    return result


def multivariate_seed_qa():
    if SEQ_LEN is None:
        current_seq_len = random.randint(64, 1024)
    else:
        current_seq_len = SEQ_LEN

    # Randomly choose the number of time series (variables)
    num_series = random.randint(2, 10)

    # Randomly select a category from metric_config
    sample = random.choice(metric_config)
    category = sample['category']

    # Get all metrics from the selected category
    metrics_in_category = []
    for s in metric_config:
        if s['category'] == category:
            metrics_in_category.extend(s['metrics'])

    # Ensure we have enough metrics to sample
    if len(metrics_in_category) < num_series:
        num_series = len(metrics_in_category)

    # Randomly select metrics for the time series
    metrics = random.sample(metrics_in_category, num_series)

    timeseries_list = []
    attribute_pool_list = []

    for metric in metrics:
        # Generate attribute_pool and time series for each metric
        attribute_pool = generate_controlled_attributes(metric_to_controlled_attributes(metric))
        timeseries, attribute_pool = generate_time_series(attribute_pool, current_seq_len)

        # Append to lists
        timeseries_list.append(timeseries)
        attribute_pool_list.append(attribute_pool)

    # Scale and encode the time series
    scaled_timeseries_list = []
    cur_ts_prompts = []

    for timeseries in timeseries_list:
        scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(timeseries, ENCODING_METHOD)
        scaled_timeseries_list.append(scaled_timeseries)
        cur_ts_prompts.append(cur_ts_prompt)

    # Generate instruction
    instruction = f"You are a time series analysis expert. There are {num_series} metrics collected from {category} with length of {current_seq_len}."

    # List the metrics and their data
    for i in range(num_series):
        instruction += f"\nMetric {i+1}: {metrics[i]}. Time series data: {cur_ts_prompts[i]}"

    questions, answers, fields = [], [], []

    # (Task 1) Detailed analysis of a specific time series
    selected_index = random.randint(0, num_series - 1)
    questions.append(f"Please analyze the characteristics of time series {selected_index+1} ({metrics[selected_index]}), including its periodicity, trend, local characteristics, frequency characteristics, and noise.")

    desc_text = attribute_to_text(
        timeseries_list[selected_index],
        attribute_pool_list[selected_index],
        generate_values=False
    )
    answers.append(desc_text)
    cur_fields = {"trend": [selected_index], "seasonal": [selected_index], "noise": [selected_index], "local": [selected_index]}
    fields.append(cur_fields)

    # (Task 2) Describe a specific attribute for all time series
    available_attributes = ['trend', 'periodicity', 'frequency', 'noise', 'local']
    selected_attribute = random.choice(available_attributes)
    questions.append(f"Please describe the {selected_attribute} characteristics of all the time series.")            

    # Build the answer by extracting the selected attribute from each time series
    attribute_text = ""
    for i in range(num_series):
        desc_text = attribute_to_text(
            timeseries_list[i],
            attribute_pool_list[i],
            generate_values=False,
            include_attributes=[selected_attribute]
        )
        attribute_text += f"Time series {i+1} ({metrics[i]}):\n{desc_text}\n"
    attribute_fields_map = {
        'trend': 'trend',
        'periodicity': 'seasonal',
        'frequency': 'seasonal',
        'noise': 'noise',
        'local': 'local'
    }
    cur_fields = {
        attribute_fields_map[selected_attribute]: list(range(num_series))
    }
    fields.append(cur_fields)
    answers.append(attribute_text.strip())

    # (Task 3) Searching for time series with similar trend
    visited = set()
    corr_pool = []
    for i in range(num_series):
        if i in visited:
            continue
        visited.add(i)
        cur_result = {i}
        for j in range(i + 1, num_series):
            if j in visited:
                continue
            if attribute_pool_list[i]['trend']['type'] == attribute_pool_list[j]['trend']['type'] and attribute_pool_list[i]['trend']['type'] != 'multiple':
                # Similar trend found
                visited.add(j)
                cur_result.add(j)

        if len(cur_result) > 1:
            # Add a question
            question = f"Please find the time series with similar trend characteristics with {metrics[random.choice(list(cur_result))]}."
            questions.append(question)
            cur_answer = f"Time series with similar trend characteristics: {', '.join([metrics[i] for i in cur_result])}, because their trend are all {attribute_pool_list[i]['trend']['type']}."
            answers.append(cur_answer)
            cur_fields = {
                "trend": list(cur_result),
                "correlation": [len(corr_pool)]
            }
            corr_pool.append((sorted(cur_result), cur_answer))

    # (Task 4) Searching for time series with similar local fluctuation
    visited = set()
    for i in range(num_series):
        if i in visited:
            continue
        visited.add(i)
        cur_result = {i}
        for j in range(i + 1, num_series):
            if j in visited:
                continue
            if len(attribute_pool_list[i]['local']) != len(attribute_pool_list[j]['local']):
                continue
            for l1, l2 in zip(attribute_pool_list[i]['local'], attribute_pool_list[j]['local']):
                if abs(l1['position_start'] - l2['position_start']) > 15:
                    break
            else:
                # Similar local found
                visited.add(j)
                cur_result.add(j)

        if len(cur_result) > 1:
            # Add a question
            question = f"Please find the time series with similar local fluctuations characteristics with {metrics[random.choice(list(cur_result))]} in terms of their fluctuation positions."
            questions.append(question)
            cur_answer = f"Time series with similar local fluctuations characteristics: {', '.join([metrics[i] for i in cur_result])}, because they all have local fluctuations near points: {','.join([l['position_start'] for l in attribute_pool_list[i]['local']])}."
            answers.append(cur_answer)
            cur_fields = {
                "local": list(cur_result),
                "correlation": [len(corr_pool)]
            }
            corr_pool.append((sorted(cur_result), cur_answer))

    # Generate final result
    result = []
    for q, a, f in zip(questions, answers, fields):
        result.append({
            'instruction': instruction,
            'question': q,
            'answer': a,
            'fields': f,
            'metrics': metrics,
            'attribute_pool': attribute_pool_list,
            'timeseries': scaled_timeseries_list,
            'original_timeseries': timeseries_list,
            'corr_pool': corr_pool
        })

    return result

def generate_seed_qa_dataset():
    generated_cnt = 0
    ts_idx = 0
    labels = []
 
    # Create output directory
    os.makedirs("result", exist_ok=True)
    os.makedirs("labels", exist_ok=True)

    with tqdm(total=TOTAL_CNT, desc='Generating seed qa') as t, open(OUTPUT_DATASET, 'wt') as f:
        while True:
            # Generate seed qa
            try:
                if random.random() > 0.5:
                    seed_qa = univariate_seed_qa()
                else:
                    seed_qa = multivariate_seed_qa()
            except Exception as err:
                # Error when generating random time series, just try again
                continue
            
            # Process seed qa
            for item in seed_qa:
                cur_label = {
                    'fields': item['fields'],
                    'metrics': item['metrics'],
                    'corr_pool': item['corr_pool'],
                    'attribute_pool': item['attribute_pool'],
                    'instruction': item['instruction'],
                    'question': item['question'],
                    'ts_idx': ts_idx
                }
                cur_data = {
                    'input': item['instruction'] + item['question'],
                    'output': item['answer'],
                    'timeseries': timeseries_to_list(item['timeseries'])
                }
                labels.append(cur_label)
                f.write(json.dumps(cur_data, ensure_ascii=False) + '\n')
                generated_cnt += 1
                t.update()
            ts_idx += 1

            if generated_cnt >= TOTAL_CNT:
                break
    json.dump(labels, open(OUTPUT_LABEL, 'wt'), ensure_ascii=False, indent=4)

    print(f"Finished! Total {generated_cnt} samples generated. Saved to {OUTPUT_DATASET} and {OUTPUT_LABEL}.")

if __name__ == '__main__':
    generate_seed_qa_dataset()
