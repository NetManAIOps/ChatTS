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
from chatts.ts_generator.generate import generate_time_series, generate_controlled_attributes, attribute_to_text, generate_random_attributes, all_attribute_set
from chatts.utils.llm_utils import LLMClient
from chatts.utils.encoding_utils import timeseries_encoding, timeseries_to_list
from chatts.utils.attribute_utils import metric_to_controlled_attributes
import yaml
import os


# CONFIG
NUM_DATA = yaml.safe_load(open("config/datagen_config.yaml"))["num_data_llm_qa"]
SEQ_LEN = yaml.safe_load(open("config/datagen_config.yaml"))["seq_len"]  # Set to None for random length
ENCODING_METHOD = yaml.safe_load(open("config/datagen_config.yaml"))["encoding_method"]
OUTPUT_BASE_DIR = yaml.safe_load(open("config/datagen_config.yaml"))["data_output_dir"]
OUTPUT_PATH = f'{OUTPUT_BASE_DIR}/uts_llm_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.jsonl'
LABEL_PATH = f'{OUTPUT_BASE_DIR}/labels/uts_llm_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.json'
EVOL_LABEL_PATH = f'{OUTPUT_BASE_DIR}/evol_labels/uts_llm_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.json'
DISABLE_METRIC_CONFIG = yaml.safe_load(open("config/datagen_config.yaml"))["disable_metric_config"]
DISABLE_EXTREME_LENGTHS = yaml.safe_load(open("config/datagen_config.yaml"))["disable_extreme_lengths"]
DRYRUN = yaml.safe_load(open("config/datagen_config.yaml"))["dryrun"]
LOCAL_LLM_PATH = yaml.safe_load(open("config/datagen_config.yaml"))["local_llm_path"]

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

    # Randomly pick category and metric
    sample = random.choice(list(metric_config))
    category = sample['category']
    metric = random.choice(sample['metrics'])

    # Generate attribute_pool and time series
    if DISABLE_METRIC_CONFIG:
        attribute_pool = generate_random_attributes(all_attribute_set['overall_attribute'], all_attribute_set['change'], seq_len=current_seq_len)
    else:
        attribute_pool = generate_controlled_attributes(metric_to_controlled_attributes(metric), seq_len=current_seq_len)

    attribute_pool['metric_name'] = metric
    attribute_pool['situation'] = category

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

    if DRYRUN:
        llm_answers = ['This is a test answer.'] * len(prompts)
    else:
        llm_client = LLMClient(model_path=LOCAL_LLM_PATH, engine='vllm')
        llm_answers = llm_client.llm_batch_generate(prompts, use_chat_template=True)
        llm_client.kill()

    # Replace placeholder tokens with LLM outputs
    idx = 0
    for item in result:
        for _ in item['prompt']:
            item['answer'] = item['answer'].replace(f'<|prompt{idx}|>', llm_answers[idx])
            idx += 1

    # Build labels matching original format
    labels_out, evol_labels_out = [], []

    for item in result:
        evol_labels_out.append({
            'fields': item['fields'],
            'metrics': item['metrics'],
            'corr_pool': item['corr_pool'],
            'attribute_pool': item['attribute_pool'],
            'instruction': item['instruction'],
            'question': item['question'],
            'ts_idx': item['ts_idx'],
        })
        
        labels_out.append({
            'label': item['attribute_pool'][0],
            'ts_idx': item['ts_idx'],
            'timeseries': timeseries_to_list(item['original_timeseries'][0]),
        })
    return result, evol_labels_out, labels_out


if __name__ == '__main__':
    result, evol_labels, labels = generate_dataset()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(EVOL_LABEL_PATH), exist_ok=True)
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
    with open(EVOL_LABEL_PATH, 'wt') as f:
        json.dump(evol_labels, f, ensure_ascii=False, indent=4)
    with open(LABEL_PATH, 'wt') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)
