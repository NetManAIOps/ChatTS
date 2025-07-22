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
from typing import *
from chatts.ts_generator.generate import generate_random_attributes, generate_time_series
from chatts.ts_generator.trend_utils import generate_random_points, generate_trend_prompt, generate_trend_curve
from chatts.encoding_utils import timeseries_encoding, timeseries_to_list
from tqdm import tqdm
import json
import copy
import os


# CONFIG
NUM_DATA = 20000
SEQ_LEN = 256
ENCODING_METHOD = 'sp'
OUTPUT_BASE_DIR = json.load(open("config/datagen_config.json"))["data_output_dir"]
OUTPUT_PATH = f'{OUTPUT_BASE_DIR}/mts_shape_template_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.jsonl'


# All Config for TS Features
all_config = {
    "overall_attribute": {
        "seasonal": {
            "no periodic fluctuation": 0.9,
            "periodic fluctuation": 0.1
        },
        "trend": {
            "keep steady": 0.6
        },
        "frequency": {
            "high frequency": 0.5,
            "low frequency": 0.5
        },
        "noise": {
            "noisy": 0.1,
            "almost no noise": 0.9
        }
    },
    "change": {
        "shake": 2,
        "upward spike": 10,
        "downward spike": 6,
        "continuous upward spike": 4,
        "continuous downward spike": 2,
        "wide upward spike": 3,
        "wide downward spike": 3
    }
}


def generate_positive_timeseries(cnt: int, seq_len: int = 256) -> Tuple[List[np.ndarray], List[dict], List[Tuple[int, float]]]:
    timeseries = []
    attributes = []

    points, curve_type = generate_random_points(seq_len)

    for _ in range(cnt):
        if random.random() > 0.8:
            changes = {(None, None)}
        else:
            changes = {}
        attribute_pool = generate_random_attributes(all_config['overall_attribute'], all_config['change'], changes.copy(), seq_len)
        ts, attribute_pool = generate_time_series(attribute_pool, seq_len)

        # Add some changes to points
        for i in range(len(points)):
            new_x = min(max(0, points[i][0] + random.randint(-5, 5)), seq_len - 1)
            new_y = points[i][1] + random.uniform(-0.05, 0.05) * (max([point[1] for point in points]) - min([point[1] for point in points]))
            points[i] = (new_x, new_y)

        # Generate trend
        curve_x, curve_y, curve_type = generate_trend_curve(seq_len, points)
        if curve_y.max() - curve_y.min() > 1e-3:
            ts += curve_y / (curve_y.max() - curve_y.min()) * attribute_pool['overall_amplitude'] * random.uniform(3.0, 15.0)

        timeseries.append(ts)
        attributes.append(attribute_pool)
    
    return timeseries, attributes, points


def generate_negative_timeseries(cnt: int, positive_points: List[Tuple[int, float]], seq_len: int = 256) -> Tuple[List[np.ndarray], List[dict], List[Optional[Tuple[int, float]]], List[List[Tuple[int, float]]]]:
    timeseries = []
    attributes = []
    different_type = []
    result_points = []

    for _ in range(cnt):
        if random.random() > 0.8:
            changes = {(None, None)}
        else:
            changes = {}
        attribute_pool = generate_random_attributes(all_config['overall_attribute'], all_config['change'], changes.copy(), seq_len)
        ts, attribute_pool = generate_time_series(attribute_pool, seq_len)

        if random.random() > 0.7 or len(positive_points) <= 3:
            # Type 1: Totally different
            points, curve_type = generate_random_points(seq_len)
            different_type.append(None)
        else:
            # Type 2: Change one point in positive_points
            points = copy.deepcopy(positive_points)
            i = random.choice(range(len(points)))
            new_x = min(max(0, points[i][0] + random.randint(-5, 5)), seq_len - 1)
            diff = random.choice([-1, 1]) * random.uniform(0.5, 1.0) * (max([point[1] for point in points]) - min([point[1] for point in points]))
            new_y = points[i][1] + diff
            points[i] = (new_x, new_y)
            different_type.append((i, float(diff)))

        # Generate trend
        curve_x, curve_y, curve_type = generate_trend_curve(seq_len, points)
        result_points.append(points)

        # Add to ts
        if curve_y.max() - curve_y.min() > 1e-3:
            ts += curve_y / (curve_y.max() - curve_y.min()) * attribute_pool['overall_amplitude'] * random.uniform(3.0, 15.0)

        timeseries.append(ts)
        attributes.append(attribute_pool)
    
    return timeseries, attributes, different_type, result_points


def generate_trend_naive_mts(seq_len: Optional[int]=256):
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
    
    # Generate time series
    positive_timeseries, positive_attributes, positive_points = generate_positive_timeseries(num_positive_items, current_seq_len)
    negative_timeseries, negative_attributes, negative_different_types, negative_points_list = generate_negative_timeseries(num_negative_items, positive_points, current_seq_len)
    
    # Shuffle
    shuffle_indices = np.random.permutation(num_positive_items + num_negative_items)
    combined_timeseries = positive_timeseries + negative_timeseries
    combined_attributes = positive_attributes + negative_attributes
    combined_different_types = [None] * len(positive_timeseries) + negative_different_types
    combined_points = [positive_points] * len(positive_attributes) + negative_points_list
    combined_timeseries = [combined_timeseries[i] for i in shuffle_indices]
    combined_attributes = [combined_attributes[i] for i in shuffle_indices]
    combined_different_types = [combined_different_types[i] for i in shuffle_indices]
    combined_points = [combined_points[i] for i in shuffle_indices]
    shuffle_argsort = np.argsort(shuffle_indices)
    positive_indices, negative_indices = shuffle_argsort[:num_positive_items], shuffle_argsort[num_positive_items:num_positive_items + num_negative_items]

    # Generate attribute_pool
    prompt = f'There are {len(shuffle_indices)} time series:'
    question_list = []
    answer_list = []
    original_timeseries = copy.deepcopy(combined_timeseries)
    for i in range(len(shuffle_indices)):
        # Scalar
        scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(combined_timeseries[i], ENCODING_METHOD)
        combined_timeseries[i] = scaled_timeseries
        
        # Generate prompt
        prompt += f" Time series {i+1} is of length {current_seq_len}: {cur_ts_prompt};"

        # Generate QAs
        # (Task 1) Describe the trend of each time series
        question_list.append(f"Analyze the trend of time series {i+1} in detail.")
        answer_list.append(generate_trend_prompt(combined_points[i]))

        # (Task 2) Describe the trend of two time series
        for j in range(len(shuffle_indices)):
            # Control the number
            if random.random() > 0.6:
                continue
            if i not in positive_indices and j not in positive_indices:
                continue
            if i == j:
                continue
            question_list.append(f"Based on the **trend** characteristics analyze whether there may be a correlation of trend between time series {i+1} and time series {j+1}.")
            cur_answer = ''
            if i in positive_indices and j in positive_indices:
                # Similar
                cur_answer += f" Both time series are showing similar trends, indicating a possible correlation in terms of trend: " + generate_trend_prompt(combined_points[i])
            else:
                if i in positive_indices:
                    if combined_different_types[j] is None:
                        cur_answer += f"Time series {j+1} is totally different from time series {i+1} in terms of trend. The trend of time series {i+1} is: " + generate_trend_prompt(combined_points[i]) + f". While the trend of time series {j+1} is: " + generate_trend_prompt(combined_points[j])
                    else:
                        cur_answer += f"Time series {j+1} is different from time series {i+1} in terms of trend. Although the trend between may be similar in some parts, their trends near point {int(combined_points[j][combined_different_types[j][0]][0])} are different, "
                        if combined_different_types[j][1] > 0:
                            cur_answer += f"where time series {j+1} has a higher trend than time series {i+1}."
                        else:
                            cur_answer += f"where time series {j+1} has a lower trend than time series {i+1}."
                else:
                    if combined_different_types[i] is None:
                        cur_answer += f"Time series {j+1} is totally different from time series {i+1} in terms of trend. The trend of time series {i+1} is: " + generate_trend_prompt(combined_points[i]) + f". While the trend of time series {j+1} is: " + generate_trend_prompt(positive_points)
                    else:
                        cur_answer += f"Time series {j+1} is different from time series {i+1} in terms of trend. Although the trend between may be similar in some parts, their trends near point {int(combined_points[i][combined_different_types[i][0]][0])} are different, "
                        if combined_different_types[i][1] > 0:
                            cur_answer += f"where time series {i+1} has a higher trend than time series {j+1}."
                        else:
                            cur_answer += f"where time series {i+1} has a lower trend than time series {j+1}."
            answer_list.append(cur_answer)
        
    return original_timeseries, combined_timeseries, combined_attributes, prompt, question_list, answer_list


if __name__ == "__main__":
    print('Generating...')
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wt') as f:
        with tqdm(total=NUM_DATA, desc='Generating') as t:
            cnt = 0
            while True:
                try:
                    original_timeseries, combined_timeseries, combined_attributes, prompt, question_list, answer_list = generate_trend_naive_mts(SEQ_LEN)
                except ValueError as err:
                    continue
                except IndexError as err:
                    continue
                for i in range(len(question_list)):
                    result = {
                        'input': prompt[:-1] + '. ' + question_list[i],
                        'output': answer_list[i],
                        'timeseries': timeseries_to_list(combined_timeseries),
                        # 'original_timeseries': [ts.tolist() for ts in original_timeseries]
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    t.update()
                    cnt += 1
                if cnt >= NUM_DATA:
                    break
