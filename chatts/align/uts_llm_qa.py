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
from chatts.ts_generator import generate_time_series, generate_controlled_attributes, attribute_to_text, generate_random_attributes
from chatts.llm_utils import llm_batch_generate
from chatts.encoding_utils import timeseries_encoding, timeseries_to_list
from chatts.attribute_utils import metric_to_controlled_attributes
import os


# CONFIG
NUM_DATA = 15000
SEQ_LEN = 256  # Set to None for random length
ENCODING_METHOD = 'sp'
OUTPUT_BASE_DIR = json.load(open("config/datagen_config.json"))["data_output_dir"]
OUTPUT_PATH = f'{OUTPUT_BASE_DIR}/uts_llm_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.jsonl'
LABEL_PATH = f'{OUTPUT_BASE_DIR}/labels/uts_llm_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.json'
DISABLE_METRIC_CONFIG = False
DRYRUN = False

# All Config for TS Features
all_config = {
    "overall_attribute": {
        "seasonal": {"no periodic fluctuation": 0.7, "periodic fluctuation": 0.3},
        "trend": {"decrease": 0.2, "increase": 0.2, "keep steady": 0.6},
        "frequency": {"high frequency": 0.5, "low frequency": 0.5},
        "noise": {"noisy": 0.3, "almost no noise": 0.7}
    },
    "change": {
        "shake": 2, "upward spike": 10, "downward spike": 6, "continuous upward spike": 4,
        "continuous downward spike": 2, "upward convex": 2, "downward convex": 2,
        "sudden increase": 2, "sudden decrease": 2, "rapid rise followed by slow decline": 2,
        "slow rise followed by rapid decline": 2, "rapid decline followed by slow rise": 2,
        "slow decline followed by rapid rise": 2, "decrease after upward spike": 3,
        "increase after downward spike": 3, "increase after upward spike": 3, "decrease after downward spike": 3,
        "wide upward spike": 3, "wide downward spike": 3
    }
}

metric_config = json.load(open('config/metric_set.json', 'rt'))
all_prompt_idx = 0


def replace_prompts(data, obj):
    pattern = re.compile(r"<\|prompt(\d+)\|>")
    if isinstance(obj, dict):
        return {k: replace_prompts(data, v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_prompts(data, item) for item in obj]
    elif isinstance(obj, str):
        def repl(m): return data[int(m.group(1))]
        return pattern.sub(repl, obj)
    else:
        return obj


def generate_prompt_data():
    global all_prompt_idx
    # Determine sequence length
    if SEQ_LEN is None:
        current_seq_len = 256 if random.random() > 0.4 else random.randint(64, 1024)
    else:
        current_seq_len = SEQ_LEN

    # Randomly pick category and metric
    sample = random.choice(list(metric_config))
    category = sample['category']
    metric = random.choice(sample['metrics'])

    # Generate attribute_pool and time series
    if DISABLE_METRIC_CONFIG:
        attribute_pool = generate_random_attributes(all_config['overall_attribute'], all_config['change'])
    else:
        attribute_pool = generate_controlled_attributes(metric_to_controlled_attributes(metric))
    timeseries, attribute_pool = generate_time_series(attribute_pool, current_seq_len)

    # Encode series
    scaled_ts, ts_prompt, _ = timeseries_encoding(timeseries, ENCODING_METHOD)

    instruction = (
        f"You are a time series analysis expert. This is a metric called {metric}"
        f" collected from {category} with length of {current_seq_len}: {ts_prompt}."
    )
    questions, answers, prompts, fields = [], [], [], []

    # Step 1: periodicity
    questions.append(
        "Now, please analyze the characteristics of this metric from the perspectives of periodicity,"
        " and conclude the physical meaning of the periodicity in one sentence."
    )
    fields.append({'seasonal': [0]})
    answers.append(
        attribute_to_text(timeseries, attribute_pool, generate_values=False,
                             include_attributes=['periodicity', 'frequency'])
        + f'<|prompt{all_prompt_idx}|>'
    )
    prompts.append([
        f"There is a metric called {metric} collected from {category} with length of {current_seq_len}. "
        "The periodicity of this metric is as follow: "
        + attribute_to_text(timeseries, attribute_pool, generate_values=False,
                              include_attributes=['periodicity'])
        + " Please analyze the physical meaning of this kind of periodicity in one sentence (xxx indicates that xxx):"
    ])
    all_prompt_idx += 1

    # Step 2: trend
    questions.append(
        "Now, please analyze the characteristics of this metric from the perspectives of trend,"
        " and conclude the physical meaning of the trend in one sentence."
    )
    fields.append({'trend': [0]})
    answers.append(
        attribute_to_text(timeseries, attribute_pool, generate_values=False,
                             include_attributes=['trend'])
        + f'<|prompt{all_prompt_idx}|>'
    )
    prompts.append([
        f"There is a metric called {metric} collected from {category} with length of {current_seq_len}. "
        f"The trend of this metric is {attribute_pool['trend']['type']}. "
        "Please analyze the physical meaning of this kind of trend in one sentence."
    ])
    all_prompt_idx += 1

    # Step 3: local fluctuations
    if attribute_pool.get('local'):
        questions.append(
            "Now, please analyze the characteristics of this metric from the perspectives of local fluctuations, and conclude the physical meaning of each in one sentence. Answer format: shake, position around point 125, amplitude 135.03. A sudden surge in public interest, likely due to significant news, a major event, or a trending topic related to the platform that rapidly captured user attention; small sudden decrease, position around point 102, amplitude 31.05. A slight increase in interest, possibly driven by minor news, promotions, or social media discussions that briefly captured attention without indicating a significant trend."
        )
        fields.append({'local': [0]})
        # combine multiple local explanations
        local_texts = []
        for local_char in attribute_pool['local']:
            local_texts.append(
                f"{local_char['type']}, position around point {local_char['position_start']},"
                f" amplitude {local_char['amplitude']:.2f}. <|prompt{all_prompt_idx}|>"
            )
            all_prompt_idx += 1
        answers.append(';'.join(local_texts))

        # build individual prompts
        local_prompts = []
        for local_char in attribute_pool['local']:
            local_prompts.append(
                f"There is a metric called {metric} collected from {category} with length of {current_seq_len}. "
                f"A local fluctuation of this metric is found. The type is {local_char['type']}. "
                "Please analyze the physical meaning of this fluctuation in one sentence (keep it simple, just output the physical meaning itself, do not output any description words like `the fluctuation of this metric`. Output Example: indicates that there are many computational extensive programs using CPU):"
            )
        prompts.append(local_prompts)

    # Compile results
    result = []
    for q, a, p, f in zip(questions, answers, prompts, fields):
        result.append({
            'instruction': instruction,
            'question': q,
            'answer': a,
            'fields': f,
            'prompt': p,
            'timeseries': [scaled_ts],
            'original_timeseries': [timeseries],
            'metrics': [metric],
            'attribute_pool': [attribute_pool],
            'corr_pool': []  # no correlation in single-var dataset
        })
    return result


def generate_dataset():
    result, prompts = [], []
    with tqdm(total=NUM_DATA, desc='Generating prompt...') as t:
        cnt = 0
        while cnt < NUM_DATA:
            try:
                items = generate_prompt_data()
            except (ValueError, IndexError):
                continue
            for item in items:
                item['ts_idx'] = len(result)
                result.append(item)
                prompts.extend(item['prompt'])
                cnt += 1
                t.update()

    llm_answers = (
        ['This is a test answer.'] * len(prompts)
        if DRYRUN else llm_batch_generate(prompts, use_chat_template=True)
    )

    # Replace placeholder tokens with LLM outputs
    idx = 0
    for item in result:
        for _ in item['prompt']:
            item['answer'] = item['answer'].replace(f'<|prompt{idx}|>', llm_answers[idx])
            idx += 1

    # Build labels matching original format
    labels_out = []
    for item in result:
        labels_out.append({
            'fields': item['fields'],
            'metrics': item['metrics'],
            'corr_pool': item['corr_pool'],
            'attribute_pool': item['attribute_pool'],
            'instruction': item['instruction'],
            'question': item['question'],
            'ts_idx': item['ts_idx']
        })
    return result, labels_out


if __name__ == '__main__':
    result, labels = generate_dataset()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LABEL_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wt') as f:
        for item in result:
            out = {
                'input': item['instruction'].rstrip('.') + '. ' + item['question'],
                'output': item['answer'],
                'timeseries': timeseries_to_list(item['timeseries']),
                'ts_idx': item['ts_idx'],
                'fields': item['fields']
            }
            f.write(json.dumps(out, ensure_ascii=False) + '\n')
    with open(LABEL_PATH, 'wt') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)
