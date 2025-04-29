import numpy as np
import random
from tqdm import tqdm
import re
import json
import copy
from typing import *
from chatts.ts_generator import generate_random_attributes, generate_time_series, attribute_to_text
from chatts.encoding_utils import timeseries_encoding, timeseries_to_list
import os


# CONFIG
NUM_DATA = 20000
SEQ_LEN = None
ENCODING_METHOD = 'sp'
OUTPUT_BASE_DIR = json.load(open("config/datagen_config.json"))["data_output_dir"]
OUTPUT_PATH = f'{OUTPUT_BASE_DIR}/uts_template_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.jsonl'


# All Config for TS Features
all_config = {
    "overall_attributes": {
        "seasonal": {
            "no periodic fluctuation": 0.7,
            "sin periodic fluctuation": 0.25,
            "square periodic fluctuation": 0.02,
            "triangle periodic fluctuation": 0.03
        },
        "trend": {
            "decrease": 0.15,
            "increase": 0.15,
            "keep steady": 0.2,
            "multiple": 0.5
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
        "upward spike": 6,
        "downward spike": 4,
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
        if random.random() > 0.4:
            current_seq_len = 256
        else:
            current_seq_len = random.randint(64, 1024)
    else:
        current_seq_len = SEQ_LEN

    # Choose a metric and generate
    attribute_pool = generate_random_attributes(all_config['overall_attributes'], all_config['change'], seq_len=current_seq_len)
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
