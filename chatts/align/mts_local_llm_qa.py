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
import copy
import os


# CONFIG
NUM_DATA = 15000
SEQ_LEN = None  # Set to None for random length
ENCODING_METHOD = 'sp'
OUTPUT_BASE_DIR = json.load(open("config/datagen_config.json"))["data_output_dir"]
OUTPUT_PATH = f'{OUTPUT_BASE_DIR}/mts_local_llm_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.jsonl'
EVOL_LABEL_PATH = f'{OUTPUT_BASE_DIR}/labels/mts_local_llm_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.json'
CLUSTER_LABEL_PATH = f'{OUTPUT_BASE_DIR}/cluster_labels/mts_local_llm_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.json'
DISABLE_METRIC_CONFIG = False
DRYRUN = False

# All Config for TS Features
all_config = {
    "overall_feature": {
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

metric_config = json.load(open('config/metric_set.json', 'rt'))
all_prompt_idx = 0

def replace_prompts(data, obj):
    pattern = re.compile(r"<\|prompt(\d+)\|>")
    
    if isinstance(obj, dict):
        return {k: replace_prompts(data, v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_prompts(data, item) for item in obj]
    elif isinstance(obj, str):
        def prompt_replacer(match):
            i = int(match.group(1))
            return data[i]
        
        return pattern.sub(prompt_replacer, obj)
    else:
        return obj

def attribute_pool_to_json(attribute_pool: dict) -> str:
    for i in range(len(attribute_pool['local'])):
        attribute_pool["local"][i]['amplitude'] = round(attribute_pool["local"][i]['amplitude'], 2)
    return json.dumps(attribute_pool, ensure_ascii=False)

def generate_positive_timeseries(cnt: int, change_position: int=None, seq_len: int = 256) -> Tuple[List[np.ndarray], int, List[dict]]:
    """
    Generate several time series with one change in the same position
    Return: List of timeseries, the change position, and list of attributes.
    """
    if change_position is None:
        change_position = random.randint(int(0.02 * seq_len), int(0.95 * seq_len))

    timeseries = []
    attributes = []
    for _ in range(cnt):
        changes = {(int(change_position + random.uniform(-10, 10)), None)}
        attribute_pool = generate_random_attributes(all_config['overall_feature'], all_config['change'], changes.copy())
        ts, attribute_pool = generate_time_series(attribute_pool, seq_len)
        timeseries.append(ts)
        attributes.append(attribute_pool)
    
    return timeseries, attributes, change_position

def generate_negative_timeseries(cnt: int, positive_positions: List[int], seq_len: int = 256) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Generate several time series with random changes (their change positions should not overlap)
    Return: List of timeseries and list of attributes.
    """
    min_interval = seq_len // 6  # Minimum interval to ensure changes do not overlap significantly
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
        
        attribute_pool = generate_random_attributes(all_config['overall_feature'], all_config['change'], changes)
        ts, attribute_pool = generate_time_series(attribute_pool, seq_len)
        timeseries.append(ts)
        attributes.append(attribute_pool)
    
    return timeseries, attributes

def generate_prompt_data(seq_len: int=256):
    global all_prompt_idx

    if SEQ_LEN is None:
        if random.random() > 0.4:
            current_seq_len = 256
        else:
            current_seq_len = random.randint(64, 1024)
    else:
        current_seq_len = SEQ_LEN

    # Random choose a category and metric name
    sample = random.choice(list(metric_config))
    situation = sample['category']
    cluster: Dict[str, List[str]] = sample['cluster']
    metric_to_cluster = {metric: cluster_name for cluster_name, metrics in cluster.items() for metric in metrics}

    # Positive metrics groups
    num_positive_clusters = random.randint(1, 3)
    visited_metrics, visited_clusters = set(), set()
    positive_cluster, positive_metrics = [], []
    for _ in range(num_positive_clusters):
        if random.random() > 0.5:
            # Generate from a cluster
            candidate_clusters = [i for i in cluster if len(set(cluster[i]) - visited_metrics) > 1 and i not in visited_clusters]
            if len(candidate_clusters) == 0:
                continue
            current_cluster = random.choice(candidate_clusters)
            candidate_metrics = list(set(cluster[current_cluster]) - visited_metrics)
            cur_positive_metrics = list(np.random.choice(candidate_metrics, size=random.randint(2, len(candidate_metrics)), replace=False))
            visited_clusters.add(current_cluster)
            visited_metrics.update(cur_positive_metrics)
            positive_metrics.extend(cur_positive_metrics)
            positive_cluster.append(cur_positive_metrics)
        else:
            # Random selection
            candidate_metrics = [i for i in metric_to_cluster if i not in visited_metrics]
            if len(candidate_metrics) < 2:
                continue
            cur_positive_metrics = list(np.random.choice(candidate_metrics, size=random.randint(2, min(len(candidate_metrics), 5)), replace=False))
            visited_metrics.update(cur_positive_metrics)
            positive_metrics.extend(cur_positive_metrics)
            positive_cluster.append(cur_positive_metrics)
        
    # Negative metrics
    negative_metrics = random.sample(sorted(set(metric_to_cluster) - set(positive_metrics)), random.randint(0, 5))

    # Number of generated items
    num_negative_items = len(negative_metrics)
    
    # Generate time series
    positive_timeseries, positive_attributes, positive_idx_list, positive_change_position_list = [], [], [], []
    for i in range(len(positive_cluster)):
        while True:
            positive_change_position = random.randint(int(0.02 * current_seq_len), int(0.95 * current_seq_len))
            if all(abs(positive_change_position - pos) > current_seq_len // 5 for pos in positive_change_position_list):
                break
        
        cur_positive_timeseries, cur_positive_attributes, positive_change_position = generate_positive_timeseries(len(positive_cluster[i]), positive_change_position, current_seq_len)
        positive_timeseries.extend(cur_positive_timeseries)
        positive_attributes.extend(cur_positive_attributes)
        positive_idx_list.extend([i] * len(positive_cluster[i]))
        positive_change_position_list.append(positive_change_position)
    negative_timeseries, negative_attributes = generate_negative_timeseries(num_negative_items, positive_change_position_list, current_seq_len)
     
    # Shuffle
    shuffle_indices = np.random.permutation(len(positive_metrics) + num_negative_items)
    combined_timeseries = positive_timeseries + negative_timeseries
    combined_attributes = positive_attributes + negative_attributes
    combined_metrics = positive_metrics + negative_metrics
    combined_cluster_idx = positive_idx_list + [None] * num_negative_items

    combined_timeseries = [combined_timeseries[i] for i in shuffle_indices]
    combined_attributes = [combined_attributes[i] for i in shuffle_indices]
    combined_metrics = [combined_metrics[i] for i in shuffle_indices]
    combined_cluster_idx = [combined_cluster_idx[i] for i in shuffle_indices]

    label = {
        'timeseries': [i.tolist() for i in combined_timeseries],
        'label': {
            'clusters': [],
            'position': int(positive_change_position_list[0]),
            'correlations': [],
            'cols': combined_metrics,
            'situation': situation
        },
        'attribute_pool': combined_attributes
    }

    # Generate attribute_pool
    prompt = f'In a {situation} system, there are {len(shuffle_indices)} metrics:'
    question_list = []
    answer_list = []
    llm_prompt_list = []
    fields_list = []
    corr_pool_list = [None] * len(shuffle_indices)
    original_timeseries = copy.deepcopy(combined_timeseries)

    for i in range(len(shuffle_indices)):
        # Scalar
        scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(combined_timeseries[i], ENCODING_METHOD)
        combined_timeseries[i] = scaled_timeseries

        # Generate prompt
        prompt += f"\n {combined_metrics[i]} is of length {current_seq_len}: {cur_ts_prompt};"

        # Generate QAs
        # (Task 1) Describe two time series
        cur_positive_idx = 0
        positive_indicies = [i for i in range(len(combined_metrics)) if combined_metrics[i] in positive_cluster[cur_positive_idx]]
        
        for j in range(len(shuffle_indices)):
            # Control the number
            if random.random() < 0.8 and not (i in positive_indicies and j in positive_indicies):
                continue
            if i == j:
                continue
            # Random select a positive change position
            positive_change_position = positive_change_position_list[cur_positive_idx]
            question_list.append(f"Based on the characteristics of the time series, please describe the characteristics of {combined_metrics[i]} and {combined_metrics[j]} from the aspects of periodicity, trend, local characteristics, frequency characteristics, and noise. And analyze whether there may be a correlation of fluctuation between them around point {positive_change_position}. Conclude the physical meaning of the fluctuation correlation (or no correlation) in one sentence.")
            fields_list.append({
                "local": [i, j],
                "seasonal": [i, j],
                "trend": [i, j],
                "noise": [i, j],
                "statistic": [i, j]
            })
            cur_answer = f"{combined_metrics[i]}: " + attribute_to_text(combined_timeseries[i], combined_attributes[i], generate_values=False) + f"; {combined_metrics[j]}: " + attribute_to_text(combined_timeseries[j], combined_attributes[j], generate_values=False)
            if i in positive_indicies and j in positive_indicies:
                # Similar
                cur_answer += f" Both metrics show sudden changes around point {positive_change_position}, indicating a possible correlation in terms of fluctuation. <|prompt{all_prompt_idx}|>"
                label["label"]["correlations"].append({
                    "pair": [combined_metrics[i], combined_metrics[j]],
                    "explain": f"<|prompt{all_prompt_idx}|>",
                    "label": True
                })
                all_prompt_idx += 1
                cur_llm_prompt = f"In a {situation} system, there are many monitoring metrics. Near a timestamp (maybe during a failure), we found there are fluctuations in {combined_metrics[i]} and {combined_metrics[j]} that happens together. Please explain why {combined_metrics[i]} and {combined_metrics[j]} fluctuates together in their physical meaning in English in one sentence (e.g. both a and b are xxx-related metrics and xxx may cause their fluctuations / a may cause b). Make sure to keep it simple. "
                if metric_to_cluster[combined_metrics[i]] == metric_to_cluster[combined_metrics[j]]:
                    cur_llm_prompt += f"(Hint: These two metrics are both {metric_to_cluster[combined_metrics[i]]}-related.)"
                llm_prompt_list.append([cur_llm_prompt])
            elif combined_cluster_idx[i] is not None and combined_cluster_idx[i] == combined_cluster_idx[j]:
                # Similar but not around point
                cur_answer += f" No. Both metrics show sudden changes around point {positive_change_position_list[combined_cluster_idx[i]]}, but no sudden changes around point {positive_change_position}. <|prompt{all_prompt_idx}|>"
                label["label"]["correlations"].append({
                    "pair": [combined_metrics[i], combined_metrics[j]],
                    "explain": f"<|prompt{all_prompt_idx}|>",
                    "label": False
                })
                all_prompt_idx += 1
                cur_llm_prompt = f"In a {situation} system, there are many monitoring metrics. Near a timestamp (maybe during a failure), we found there are **no** fluctuations in both {combined_metrics[i]} and {combined_metrics[j]}, but they fluctuated together in another time (before or after the failure). Please explain why {combined_metrics[i]} and {combined_metrics[j]} are not fluctuating together at this time in their physical meaning in English in one sentence (e.g. both a and b are xxx-related metrics and xxx may cause their fluctuations / a may cause b). Make sure to keep it simple. "
                if metric_to_cluster[combined_metrics[i]] == metric_to_cluster[combined_metrics[j]]:
                    cur_llm_prompt += f"(Hint: These two metrics are both {metric_to_cluster[combined_metrics[i]]}-related.)"
                llm_prompt_list.append([cur_llm_prompt])
            elif i in positive_indicies or j in positive_indicies:
                cur_answer += f" These two time series do not seem to have much correlation in terms of fluctuation around point {positive_change_position}. <|prompt{all_prompt_idx}|>"
                label["label"]["correlations"].append({
                    "pair": [combined_metrics[i], combined_metrics[j]],
                    "explain": f"<|prompt{all_prompt_idx}|>",
                    "label": False
                })
                all_prompt_idx += 1
                if i in positive_indicies:
                    a, b = i, j
                else:
                    a, b = j, i
                cur_llm_prompt = f"In a {situation} system, there are many monitoring metrics. Near a timestamp (maybe during a failure), we found there are fluctuations in {combined_metrics[a]}, but no fluctuations in {combined_metrics[b]}. Please explain why {combined_metrics[a]} and {combined_metrics[b]} are **not** fluctuating together in their physical meaning in English in one simple sentence (e.g. a is xxx-related, so xxx. But b is xxx-related, which may not affected by xxx). Make sure to keep it simple:"
                llm_prompt_list.append([cur_llm_prompt])
            else:
                cur_answer += f" These two time series do not seem to have much correlation in terms of fluctuation around point {positive_change_position}. <|prompt{all_prompt_idx}|>"
                label["label"]["correlations"].append({
                    "pair": [combined_metrics[i], combined_metrics[j]],
                    "explain": f"<|prompt{all_prompt_idx}|>",
                    "label": False
                })
                all_prompt_idx += 1
                cur_llm_prompt = f"In a {situation} system, there are many monitoring metrics. Near a timestamp (during a failure), we found there are fluctuations in some of the metrics, but no fluctuations in both {combined_metrics[i]} and {combined_metrics[j]}. Please explain why {combined_metrics[i]} and {combined_metrics[j]} are **not** fluctuating in their physical meaning in English in one simple sentence. Make sure to keep it simple:"
                llm_prompt_list.append([cur_llm_prompt])
            answer_list.append(cur_answer)
        
        # (Task 3) Find similar time series
        question_list.append(f"Based on the fluctuations in the metrics around point {positive_change_position}, please find other metric(s) that may be related to {combined_metrics[i]}, output their numbers, and explain the reasons. If related metrics are found, explain why they have similar local fluctuations considering their physical meaning in one sentence. If no related metrics are found, output that no related metrics were found.")
        negative_indicies = [i for i in range(len(combined_cluster_idx)) if combined_cluster_idx[i] is None]
        if i in negative_indicies:
            cur_answer = f"Among these metrics, I did not find any other metrics that may be related to {combined_metrics[i]} in terms of fluctuation around point {positive_change_position}. It seems that {combined_metrics[i]} shows no significant fluctuation around this point."
            fields_list.append({
                "local": [i]
            })
            llm_prompt_list.append([])
        elif i not in positive_indicies:
            cur_answer = f"Among these metrics, I did not find any other metrics that may be related to {combined_metrics[i]} in terms of fluctuation around point {positive_change_position}. It seems that {combined_metrics[i]} shows no significant fluctuation around this point."
            fields_list.append({
                "local": [i]
            })
            llm_prompt_list.append([])
        else:
            cur_answer = f'I found the following metrics that may be related to {combined_metrics[i]} in terms of fluctuation:'
            i_change = combined_attributes[i]['local'][0]['type']
            for j in range(len(shuffle_indices)):
                if i == j:
                    continue
                if j not in positive_indicies:
                    continue
                j_change = combined_attributes[j]['local'][0]['type']
                if i_change == j_change:
                    cur_answer += f" {combined_metrics[i]} and {combined_metrics[j]} both show {i_change} around point {positive_change_position}, indicating a possible correlation in terms of fluctuation."
                else:
                    cur_answer += f" {combined_metrics[i]} shows {i_change} around point {positive_change_position}, while {combined_metrics[j]} shows {j_change} around this point, indicating a possible correlation in terms of fluctuation."
            cur_answer += f' <|prompt{all_prompt_idx}|>'
            fields_list.append({
                "local": [j for j in positive_indicies]
            })
            corr_pool_list[i] = [[j for j in positive_indicies], cur_answer]

            if len(label["label"]["clusters"]) == 0:
                label["label"]["clusters"].append({
                    'col_idx': [[int(j), combined_attributes[j]['local'][0]['type']] for j in positive_indicies],
                    'cols': [combined_metrics[j] for j in positive_indicies],
                    'explain': f"<|prompt{all_prompt_idx}|>",
                })

            all_prompt_idx += 1
            llm_prompt_list.append([f"In a {situation} system, there are many monitoring metrics. Near a timestamp (maybe during a failure), we found there are fluctuations in " + ', '.join(combined_metrics[j] for j in positive_indicies) + f". Please explain their relationship in physical meaning and simply describe what's may happening in the {situation} system in English in 1 sentence, like `these metrics are all xxx-related or xxx. {situation} may xxx.` (the format may be different, but keep simple): "])
        answer_list.append(cur_answer)
    return original_timeseries, combined_timeseries, combined_metrics, combined_attributes, prompt, question_list, answer_list, llm_prompt_list, fields_list, corr_pool_list, label

def generate_dataset():
    result = []
    prompts = []
    labels = []
    with tqdm(total=NUM_DATA, desc='Generating prompt...') as t:
        cnt = 0
        while True:
            try:
                original_timeseries, combined_timeseries, combined_metrics, combined_attributes, prompt, question_list, answer_list, llm_prompt_list, fields_list, corr_pool_list, label = generate_prompt_data(SEQ_LEN)
            except ValueError as err:
                continue
            except IndexError as err:
                continue
            result.append((original_timeseries, combined_timeseries, combined_metrics, combined_attributes, prompt, question_list, answer_list, llm_prompt_list, fields_list, corr_pool_list))
            labels.append(label)
            for item in llm_prompt_list:
                prompts.extend(item)
                t.update()
                cnt += 1
            if cnt >= NUM_DATA:
                break

    # Use LLM to generate answer
    if DRYRUN:
        llm_answers = ['This is a test answer.'] * len(prompts)
    else:
        llm_answers = llm_batch_generate(prompts, use_chat_template=True)

    idx = 0
    for original_timeseries, combined_timeseries, combined_metrics, combined_attributes, prompt, question_list, answer_list, llm_prompt_list, fields_list, corr_pool_list in result:
        for i in range(len(question_list)):
            for j in range(len(llm_prompt_list[i])):
                answer_list[i] = answer_list[i].replace(f'<|prompt{idx}|>', llm_answers[idx])
                idx += 1
        for i in range(len(corr_pool_list)):
            if corr_pool_list[i] is not None:
                corr_pool_list[i][1] = replace_prompts(llm_answers, corr_pool_list[i][1])

    # Replace <|prompti|> with llm_answers in label
    labels = replace_prompts(llm_answers, labels)

    return result, labels


if __name__ == '__main__':
    print('Generating...')
    result, cluster_labels = generate_dataset()
    evol_labels = []

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(EVOL_LABEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CLUSTER_LABEL_PATH), exist_ok=True)

    with open(OUTPUT_PATH, 'wt') as f:
        for ts_idx, (original_timeseries, combined_timeseries, combined_metrics, combined_attributes, prompt, question_list, answer_list, llm_prompt_list, fields_list, corr_pool) in enumerate(result):
            for i in range(len(question_list)):
                result = {
                    'input': prompt[:-1] + '. ' + question_list[i],
                    'output': answer_list[i],
                    'timeseries': timeseries_to_list(combined_timeseries),
                }
                cur_label = {
                    "fields": fields_list[i],
                    "metrics": combined_metrics,
                    "corr_pool": corr_pool,
                    "attribute_pool": combined_attributes,
                    "instruction": prompt,
                    "question": question_list[i],
                    "ts_idx": ts_idx
                }

                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                evol_labels.append(cur_label)

    with open(CLUSTER_LABEL_PATH, 'wt') as f:
        json.dump(cluster_labels, f, ensure_ascii=False, indent=4)
    with open(EVOL_LABEL_PATH, 'wt') as f:
        json.dump(evol_labels, f, ensure_ascii=False, indent=4)
