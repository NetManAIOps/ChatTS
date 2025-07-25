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

import numpy as np
import random
from tqdm import tqdm
import re
import json
import copy
from typing import *
from chatts.ts_generator.generate import generate_random_attributes, generate_time_series, attribute_to_text, all_attribute_set
from chatts.utils.encoding_utils import timeseries_encoding, timeseries_to_list
import yaml
import os


# CONFIG
NUM_DATA = yaml.safe_load(open("config/datagen_config.yaml"))["num_data_template_qa"]
SEQ_LEN = yaml.safe_load(open("config/datagen_config.yaml"))["seq_len"]  # Set to None for random length
ENCODING_METHOD = yaml.safe_load(open("config/datagen_config.yaml"))["encoding_method"]
OUTPUT_BASE_DIR = yaml.safe_load(open("config/datagen_config.yaml"))["data_output_dir"]
OUTPUT_PATH = f'{OUTPUT_BASE_DIR}/uts_template_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.jsonl'
DISABLE_EXTREME_LENGTHS = yaml.safe_load(open("config/datagen_config.yaml"))["disable_extreme_lengths"]


def attribute_pool_to_json(attribute_pool: dict) -> str:
    result = copy.deepcopy(attribute_pool)
    for i in range(len(result['local'])):
        result["local"][i]['amplitude'] = round(result["local"][i]['amplitude'], 2)
    if 'overall_amplitude' in result:
        del result['overall_amplitude']
    if 'overall_bias' in result:
        del result['overall_bias']
    if 'statistics' in result:
        del result['statistics']
    if 'trend_list' in result.get('trend', {}):
        del result['trend']['trend_list']
    return json.dumps(result, ensure_ascii=False)

def generate_single_dataset():
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

    # Choose a metric and generate
    attribute_pool = generate_random_attributes(all_attribute_set['overall_attribute'], all_attribute_set['change'], seq_len=current_seq_len)
    timeseries, attribute_pool = generate_time_series(attribute_pool, current_seq_len)

    # Scalar
    scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(timeseries, ENCODING_METHOD)

    # Generate QA
    instruction = f"There is a time series of length {current_seq_len}: {cur_ts_prompt}."
    questions, answers = [], []
    # (Step 1) general shape
    questions.append("Now, please analyze the characteristics of this time series from the perspectives of periodicity, trend, local characteristics, frequency characteristics, and noise.")
    answers.append(attribute_to_text(timeseries, attribute_pool, generate_values=False))

    # (Step 2) general shape and values
    questions.append("Now, please analyze the characteristics of this time series from the perspectives of periodicity, trend, local characteristics, frequency characteristics, and noise. Also include the approximate mean values for every 16 points, as well as the maximum and minimum values of the time series (rounded to 2 decimal places).")
    answers.append(attribute_to_text(timeseries, attribute_pool, generate_values=True))

    # (Step 3) generate the reason of each change
    for local_char in attribute_pool['local']:
        question_position = local_char['position_start'] + random.randint(-5, 5)
        questions.append(f"Is there a local characteristic fluctuation starting around point {question_position} in this time series?")
        answers.append(f"Yes, this time series " + local_char['detail'])

    # (Step 4) randomly generate a non-change point and ask it
    all_change_positions = [local_char['position_start'] for local_char in attribute_pool['local']]
    for _ in range(3):
        point = random.randint(0, current_seq_len - 1)
        if all([abs(point - i) >= 50 for i in all_change_positions]):
            questions.append(f"Is there a local characteristic fluctuation starting around point {point} in this time series?")
            answers.append(f"I did not find any local characteristic fluctuation starting around point {point} in this time series.")

    # (Step 5) Jsonize
    questions.append("Please output the characteristics of the current time series in JSON format, including periodicity, trend, local characteristics, frequency characteristics, and noise fields.")
    answers.append(attribute_pool_to_json(attribute_pool))

    # Generate final result
    result = []
    for q, a in zip(questions, answers):
        result.append({
            'instruction': instruction,
            'question': q,
            'answer': a,
            'timeseries': [scaled_timeseries],
            'original_timeseries': [timeseries]
        })

    return result


if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wt') as f:
        with tqdm(total=NUM_DATA, desc='Generating') as t:
            cnt = 0
            while True:
                try:
                    result = generate_single_dataset()
                except ValueError as err:
                    continue
                except IndexError as err:
                    continue
                for item in result:
                    item = {
                        'input': item['instruction'][:-1] + '. ' + item['question'],
                        'output': item['answer'],
                        'timeseries': timeseries_to_list(item['timeseries']),
                        # 'original_timeseries': [i.tolist() for i in item['original_timeseries']]
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    t.update()
                    cnt += 1
                if cnt >= NUM_DATA:
                    break
