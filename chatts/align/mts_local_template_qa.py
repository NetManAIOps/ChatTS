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
from typing import *
from chatts.ts_generator.generate import generate_random_attributes, generate_time_series, attribute_to_text
from chatts.encoding_utils import timeseries_encoding, timeseries_to_list
import copy
import os


# CONFIG
NUM_DATA = 20000
SEQ_LEN = 256
ENCODING_METHOD = 'sp'
OUTPUT_BASE_DIR = json.load(open("config/datagen_config.json"))["data_output_dir"]
OUTPUT_PATH = f'{OUTPUT_BASE_DIR}/mts_local_template_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.jsonl'


# All Config for TS Features
all_config = {
    "overall_attribute": {
        "seasonal": {
            "no periodic fluctuation": 0.7,
            "periodic fluctuation": 0.3
        },
        "trend": {
            "decrease": 0.2,
            "increase": 0.2,
            "keep steady": 0.6
        },
        "frequency": {
            "high frequency": 0.5,
            "low frequency": 0.5
        },
        "noise": {
            "noisy": 0.3,
            "almost no noise": 0.7
        }
    },
    "change": {
        "shake": 2,
        "upward spike": 10,
        "downward spike": 6,
        "continuous upward spike": 4,
        "continuous downward spike": 2,
        "upward convex": 2,
        "downward convex": 2,
        "sudden increase": 2,
        "sudden decrease": 2,
        "rapid rise followed by slow decline": 2,
        "slow rise followed by rapid decline": 2,
        "rapid decline followed by slow rise": 2,
        "slow decline followed by rapid rise": 2,
        "decrease after upward spike": 3,
        "increase after downward spike": 3,
        "increase after upward spike": 3,
        "decrease after downward spike": 3,
        "wide upward spike": 3,
        "wide downward spike": 3
    }
}

def attribute_pool_to_json(attribute_pool: dict) -> str:
    for i in range(len(attribute_pool['local'])):
        attribute_pool["local"][i]['amplitude'] = round(attribute_pool["local"][i]['amplitude'], 2)
    return json.dumps(attribute_pool, ensure_ascii=False)

def generate_positive_timeseries(cnt: int, seq_len: int = 256) -> Tuple[List[np.ndarray], int, List[dict]]:
    """
    Generate several time series with one change in the same position
    Return: List of timeseries, the change position, and list of attributes.
    """
    change_position = random.randint(int(0.02 * seq_len), int(0.95 * seq_len))

    timeseries = []
    attributes = []
    for _ in range(cnt):
        changes = {(int(change_position + random.uniform(-10, 10)), None)}
        attribute_pool = generate_random_attributes(all_config['overall_attribute'], all_config['change'], changes.copy(), seq_len)
        ts, attribute_pool = generate_time_series(attribute_pool, seq_len)
        timeseries.append(ts)
        attributes.append(attribute_pool)
    
    return timeseries, attributes, change_position

def generate_negative_timeseries(cnt: int, positive_positions: List[int], seq_len: int = 256) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Generate several time series with random changes (their change positions should not overlap)
    Return: List of timeseries and list of attributes.
    """
    min_interval = seq_len // 8  # Minimum interval to ensure changes do not overlap significantly
    negative_positions = set()
    
    timeseries = []
    attributes = []
    for _ in range(cnt):
        if random.random() > 0.8:
            candidate_position = random.randint(int(0.02 * seq_len), int(0.95 * seq_len))
            try_cnt = 0
            flag = True
            while any(abs(candidate_position - pos) <= min_interval for pos in positive_positions + list(negative_positions)):
                candidate_position = random.randint(int(0.02 * seq_len), int(0.95 * seq_len))
                try_cnt += 1
                if try_cnt >= 10000:
                    flag = False
                    break
            if flag:
                negative_positions.add(candidate_position)
                changes = {(candidate_position, None)}
            else:
                changes = set()
        else:
            changes = set()
        
        attribute_pool = generate_random_attributes(all_config['overall_attribute'], all_config['change'], changes, seq_len)
        ts, attribute_pool = generate_time_series(attribute_pool, seq_len)
        timeseries.append(ts)
        attributes.append(attribute_pool)
    
    return timeseries, attributes

def generate_fluctuation_naive_mts(seq_len: int=256):
    if SEQ_LEN is None:
        if random.random() > 0.4:
            current_seq_len = 256
        else:
            current_seq_len = random.randint(64, 1024)
    else:
        current_seq_len = SEQ_LEN

    # Number of generated items
    num_positive_items = random.randint(2, 4)
    num_negative_items = random.randint(0, 5)
    num_same_items = random.randint(0, 2)
    
    # Generate time series
    positive_timeseries, positive_attributes, positive_change_position = generate_positive_timeseries(num_positive_items, current_seq_len)
    negative_timeseries, negative_attributes = generate_negative_timeseries(num_negative_items, [positive_change_position], current_seq_len)
    same_timeseries, same_attributes = [], []
    for _ in range(num_same_items):
        ts, desc = generate_time_series(positive_attributes[0], current_seq_len)
        same_timeseries.append(ts)
        same_attributes.append(desc)
    
    # Shuffle
    shuffle_indices = np.random.permutation(num_positive_items + num_negative_items + num_same_items)
    combined_timeseries = positive_timeseries + negative_timeseries + same_timeseries
    combined_attributes = positive_attributes + negative_attributes + same_attributes
    combined_timeseries = [combined_timeseries[i] for i in shuffle_indices]
    combined_attributes = [combined_attributes[i] for i in shuffle_indices]
    shuffle_argsort = np.argsort(shuffle_indices)
    positive_indices, negative_indices, same_indices = shuffle_argsort[:num_positive_items], shuffle_argsort[num_positive_items:num_positive_items + num_negative_items], shuffle_argsort[num_positive_items + num_negative_items:]
    original_timeseries = copy.deepcopy(combined_timeseries)

    # Generate attribute_pool
    prompt = f'There are {len(shuffle_indices)} time series:'
    question_list = []
    answer_list = []
    for i in range(len(shuffle_indices)):
        # Scalar
        scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(combined_timeseries[i], ENCODING_METHOD)
        
        # Generate prompt
        prompt += f" Time series {i+1} is of length {current_seq_len}: {cur_ts_prompt};"

        # Generate QAs
        # (Task 1) Describe the characteristic of one timeseries
        question_list.append(f"Based on the characteristics of the time series, please describe the characteristics of time series {i+1} from the aspects of periodicity, trend, local characteristics, frequency characteristics, and noise. Also include the approximate mean values for every 16 points, as well as the maximum and minimum values of the time series (rounded to 2 decimal places).")
        answer_list.append(f"The {i+1}th " + attribute_to_text(combined_timeseries[i], combined_attributes[i]))
        
        # (Task 2) Describe two time series
        for j in range(len(shuffle_indices)):
            # Control the number
            if random.random() < 0.7:
                continue
            if i == j:
                continue
            question_list.append(f"Based on the characteristics of the time series, please describe the characteristics of time series {i+1} and time series {j+1} from the aspects of periodicity, trend, local characteristics, frequency characteristics, and noise, and analyze whether there may be a correlation of fluctuation between them.")
            cur_answer = f" Time series {i+1} " + attribute_to_text(combined_timeseries[i], combined_attributes[i], generate_values=False) + f" Time series {j+1} " + attribute_to_text(combined_timeseries[j], combined_attributes[j], generate_values=False)
            if (i in same_indices and shuffle_indices[j] == 0) or (j in same_indices and shuffle_indices[i] == 0) or (i in same_indices and j in same_indices):
                # Same
                cur_answer += f" Both time series have very similar periodicity, trend, local characteristics, frequency characteristics, and noise characteristics, and both show sudden changes around point {positive_change_position}, indicating a strong correlation in terms of both overall trend and fluctuations."
            elif (i in positive_indices and j in positive_indices) or (i in positive_indices and j in same_indices) or (j in positive_indices and i in same_indices):
                # Similar
                cur_answer += f" Both time series show sudden changes around point {positive_change_position}, indicating a possible correlation in terms of fluctuation."
            else:
                cur_answer += " These two time series do not seem to have much correlation in terms of fluctuation."
            answer_list.append(cur_answer)
        
        # (Task 3) Find similar time series
        question_list.append(f"Based on the fluctuations in the time series, please find other time series that may be related to time series {i+1}, output their numbers, and explain the reasons. If no related time series are found, output that no related time series were found.")
        if i in negative_indices:
            cur_answer = f"Among these time series, I did not find any other time series that may be related to time series {i+1} in terms of fluctuation."
        else:
            cur_answer = f'I found the following time series that may be related to time series {i+1} in terms of fluctuation:'
            i_change = combined_attributes[i]['local'][0]['type']
            for j in range(len(shuffle_indices)):
                if i == j:
                    continue
                if j in negative_indices:
                    continue
                j_change = combined_attributes[j]['local'][0]['type']
                if i_change == j_change:
                    cur_answer += f" Time series {i+1} and time series {j+1} both show {i_change} around point {positive_change_position}, indicating a possible correlation in terms of fluctuation."
                else:
                    cur_answer += f" Time series {i+1} shows {i_change} around point {positive_change_position}, while time series {j+1} shows {j_change} around this point, indicating a possible correlation in terms of fluctuation."
        answer_list.append(cur_answer)
        combined_timeseries[i] = scaled_timeseries
    
    return original_timeseries, combined_timeseries, combined_attributes, prompt, question_list, answer_list


if __name__ == '__main__':
    print('Generating...')
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wt') as f:
        with tqdm(total=NUM_DATA, desc='Generating') as t:
            cnt = 0
            while True:
                try:
                    original_timeseries, combined_timeseries, combined_attributes, prompt, question_list, answer_list = generate_fluctuation_naive_mts(SEQ_LEN)
                except ValueError as err:
                    continue
                except IndexError as err:
                    continue
                for i in range(len(question_list)):
                    result = {
                        'input': prompt[:-1] + '. ' + question_list[i],
                        'output': answer_list[i],
                        'timeseries': timeseries_to_list(combined_timeseries),
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    t.update()
                    cnt += 1
                if cnt >= NUM_DATA:
                    break
