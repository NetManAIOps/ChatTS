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
import random
from tqdm import tqdm
import re
import json
import copy
from typing import *
from chatts.ts_generator.generate import generate_random_attributes, generate_time_series
from sktime.classification.kernel_based import RocketClassifier


# CONFIG
NUM_DATA = 3000
SEQ_LEN = 256
WINDOW_SIZE = 64
OUTPUT_PATH = "result/rocket"

# All Config for TS Features
all_config = {
    "overall_feature": {
        "seasonal": {
            "no periodic fluctuation": 1.0
        },
        "trend": {
            "keep steady": 1.0
        },
        "frequency": {
            "low frequency": 1.0
        },
        "noise": {
            "almost no noise": 1.0
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

# Change type list
change_type_list = sorted(all_config['change'].keys())

def description_to_json(description: dict) -> str:
    result = copy.deepcopy(description)
    for i in range(len(result['local'])):
        result["local"][i]['amplitude'] = round(result["local"][i]['amplitude'], 2)
    if 'overall_amplitude' in result:
        del result['overall_amplitude']
    if 'overall_bias' in result:
        del result['overall_bias']
    return json.dumps(result, ensure_ascii=False)

def generate_single_dataset(window_size=64):
    # Choose a metric and generate
    description = generate_random_attributes(all_config['overall_feature'], all_config['change'], seq_len=SEQ_LEN, change_positions=[(None, None)])
    timeseries, description = generate_time_series(description, SEQ_LEN)
    
    # Pad around
    local = description['local'][0]
    start_pad = (window_size - local['position_start'] + local['position_end']) // 2

    if start_pad < 0:
        raise ValueError('Invalid start pad')

    start_position = local['position_start'] - start_pad
    end_position = start_position + window_size

    if start_position < 0 or end_position > SEQ_LEN:
        raise IndexError('Out of range')

    # Return result
    value = timeseries[start_position:end_position]
    class_label = change_type_list.index(local['type'])

    return value, class_label


if __name__ == '__main__':
    # Generate data
    print("Generating training data for rocket...")
    timeseries = []
    labels = []

    for _ in tqdm(range(NUM_DATA)):
        try:
            ts, label = generate_single_dataset(window_size=WINDOW_SIZE)
            timeseries.append(ts)
            labels.append(label)
        except Exception as err:
            continue

    print(f"Generated {len(timeseries)} samples")
    timeseries = np.stack(timeseries, axis=0)[:, np.newaxis, :]
    labels = np.array(labels)

    # Train a model
    print("Training classifier...")
    clf = RocketClassifier()
    clf.fit(timeseries, labels)

    # Save model
    print("Saving model...")
    clf.save(OUTPUT_PATH)
