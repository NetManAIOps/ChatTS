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
from chatts.trend_utils import generate_trend_curve, generate_random_points, generate_trend_prompt
import copy
import os


# CONFIG
NUM_DATA = 15000
SEQ_LEN = 256  # Set to None for random length
ENCODING_METHOD = 'sp'
OUTPUT_BASE_DIR = json.load(open("config/datagen_config.json"))["data_output_dir"]
OUTPUT_PATH = f'{OUTPUT_BASE_DIR}/mts_shape_llm_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.jsonl'
EVOL_LABEL_PATH = f'{OUTPUT_BASE_DIR}/labels/mts_shape_llm_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.json'
CLUSTER_LABEL_PATH = f'{OUTPUT_BASE_DIR}/cluster_labels/mts_shape_llm_{SEQ_LEN}_{NUM_DATA}_{ENCODING_METHOD}.json'
DISABLE_METRIC_CONFIG = False
DRYRUN = False


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

def generate_positive_timeseries(cnt: int, seq_len: int = 256) -> Tuple[List[np.ndarray], int, List[dict]]:
    timeseries = []
    attributes = []

    points, curve_type = generate_random_points(seq_len)

    for _ in range(cnt):
        if random.random() > 0.8:
            changes = {(None, None)}
        else:
            changes = {}
        attribute_pool = generate_random_attributes(all_config['overall_attribute'], all_config['change'], changes.copy())
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

def generate_negative_timeseries(cnt: int, positive_points: List[Tuple[int, float]], seq_len: int = 256) -> Tuple[List[np.ndarray], List[dict]]:
    timeseries = []
    attributes = []
    different_type = []
    result_points = []

    for _ in range(cnt):
        if random.random() > 0.8:
            changes = {(None, None)}
        else:
            changes = {}
        attribute_pool = generate_random_attributes(all_config['overall_attribute'], all_config['change'], changes.copy())
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
    positive_timeseries, positive_attributes, positive_idx_list, cluster_points_list, positive_points_list = [], [], [], [], []
    for i in range(len(positive_cluster)):
        cur_positive_timeseries, cur_positive_attributes, cur_positive_points = generate_positive_timeseries(len(positive_cluster[i]), current_seq_len)
        positive_timeseries.extend(cur_positive_timeseries)
        positive_attributes.extend(cur_positive_attributes)
        positive_idx_list.extend([i] * len(positive_cluster[i]))
        cluster_points_list.append(cur_positive_points)
        positive_points_list.extend([cur_positive_points] * len(positive_cluster[i]))
    negative_timeseries, negative_attributes, negative_different_types, negative_points_list = generate_negative_timeseries(num_negative_items, cluster_points_list[0], current_seq_len)
    
    # Shuffle
    shuffle_indices = np.random.permutation(len(positive_metrics) + num_negative_items)
    combined_timeseries = positive_timeseries + negative_timeseries
    combined_attributes = positive_attributes + negative_attributes
    combined_metrics = positive_metrics + negative_metrics
    combined_different_types = [None] * len(positive_timeseries) + negative_different_types
    combined_points = positive_points_list + negative_points_list
    combined_cluster_idx = positive_idx_list + [None] * num_negative_items

    combined_timeseries = [combined_timeseries[i] for i in shuffle_indices]
    combined_attributes = [combined_attributes[i] for i in shuffle_indices]
    combined_metrics = [combined_metrics[i] for i in shuffle_indices]
    combined_different_types = [combined_different_types[i] for i in shuffle_indices]
    combined_points = [combined_points[i] for i in shuffle_indices]
    combined_cluster_idx = [combined_cluster_idx[i] for i in shuffle_indices]

    label = {
        'timeseries': [i.tolist() for i in combined_timeseries],
        'label': {
            'clusters': [],
            'correlations': [],
            'cols': combined_metrics,
            'situation': situation
        },
        'attribute_pool': combined_attributes
    }
    appended_clusters = set()

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
        # (Task 1) Describe the trend of each time series
        question_list.append(f"Analyze the trend of {combined_metrics[i]} in detail.")
        fields_list.append({'trend': [i]})
        answer_list.append(generate_trend_prompt(combined_points[i]))
        llm_prompt_list.append([])

        # (Task 2) Describe the trend of two time series
        for j in range(len(shuffle_indices)):            
            if combined_cluster_idx[i] is None and combined_cluster_idx[j] is None:
                continue
            if i == j:
                continue
            cur_answer = ''
            if combined_cluster_idx[i] == combined_cluster_idx[j]:
                # Control the number
                if random.random() > 0.7:
                    continue
                # Similar
                cur_answer += f" Both time series are showing similar trends, indicating a possible correlation in terms of trend: " + generate_trend_prompt(combined_points[i])
                cur_llm_prompt = f"In a {situation} system, there are many monitoring metrics. We found that the trends in {combined_metrics[i]} and {combined_metrics[j]} are very similar. Please explain why {combined_metrics[i]} and {combined_metrics[j]} have very similar trends in their physical meaning in English in one simle sentence (e.g. both a and b are xxx-related metrics, so xxx). Make sure to keep the sentence simple. "
                if metric_to_cluster[combined_metrics[i]] == metric_to_cluster[combined_metrics[j]]:
                    cur_llm_prompt += f"(Hint: These two metrics are both {metric_to_cluster[combined_metrics[i]]}-related.)"
                llm_prompt_list.append([cur_llm_prompt])
                label["label"]["correlations"].append({
                    "pair": [combined_metrics[i], combined_metrics[j]],
                    "explain": f"<|prompt{all_prompt_idx}|>",
                    "label": True
                })
            else:
                if random.random() > 0.25:
                    continue

                if combined_cluster_idx[i] is not None:
                    a, b = i, j
                else:
                    a, b = j, i
                
                if combined_different_types[b] is None:
                    cur_answer += f"{combined_metrics[b]} is totally different from {combined_metrics[a]} in terms of trend. The trend of {combined_metrics[a]} is: " + generate_trend_prompt(combined_points[a]) + f" While the trend of {combined_metrics[b]} is: " + generate_trend_prompt(combined_points[b])
                else:
                    cur_answer += f"{combined_metrics[b]} is different from {combined_metrics[a]} in terms of trend. Although the trend between may be similar in some parts, their trends near point {int(combined_points[b][combined_different_types[b][0]][0])} are different, "
                    if combined_different_types[b][1] > 0:
                        cur_answer += f"where {combined_metrics[b]} has a higher trend than {combined_metrics[a]}."
                    else:
                        cur_answer += f"where {combined_metrics[b]} has a lower trend than {combined_metrics[a]}."
            
                cur_llm_prompt = f"In a {situation} system, there are many monitoring metrics. We found that the trends in {combined_metrics[i]} and {combined_metrics[j]} are not similar. Please explain why the trends {combined_metrics[i]} and {combined_metrics[j]} are not similar in their physical meaning in English in one simle sentence (e.g. both a and b are xxx-related metrics, so xxx). Make sure to keep the sentence simple. "
                llm_prompt_list.append([cur_llm_prompt])
                label["label"]["correlations"].append({
                    "pair": [combined_metrics[i], combined_metrics[j]],
                    "explain": f"<|prompt{all_prompt_idx}|>",
                    "label": False
                })

            question_list.append(f"Based on the **trend** characteristics analyze whether there may be a correlation of trend between {combined_metrics[i]} and {combined_metrics[j]}. Conclude the physical meaning of the trend correlation (or no correlation) in one sentence.")
            fields_list.append({'trend': [i, j]})
            cur_answer += f" <|prompt{all_prompt_idx}|>"
            all_prompt_idx += 1
            answer_list.append(cur_answer)
        
        # (Task 3) Find similar time series (cluster)
        question_list.append(f"Based on the **trends** in the time series, please find time series (including itself) that may be related to {combined_metrics[i]}, output their numbers, and explain the reasons. If related metrics are found, explain why they have similar trends considering their physical meaning in one sentence. If no related time series are found, output that no related time series were found.")
        if combined_cluster_idx[i] is None:
            cur_answer = f"Among these time series, I did not find any other time series that may be related to {combined_metrics[i]} in terms of trend."
            llm_prompt_list.append([])
            fields_list.append({'trend': [i]})
        else:
            cur_answer = f"Among these time series, " + ", ".join([combined_metrics[j] for j in range(len(shuffle_indices)) if combined_cluster_idx[i] == combined_cluster_idx[j]]) + f" may be related to {combined_metrics[i]} in terms of trend. All the time series have similar trends: " + generate_trend_prompt(combined_points[i])
            fields_list.append({'trend': [j for j in range(len(shuffle_indices)) if combined_cluster_idx[i] == combined_cluster_idx[j]]})

            if combined_cluster_idx[i] not in appended_clusters:
                label["label"]["clusters"].append({
                    'col_idx': [int(j) for j in range(len(shuffle_indices)) if combined_cluster_idx[i] == combined_cluster_idx[j]],
                    'cols': [combined_metrics[j] for j in range(len(shuffle_indices)) if combined_cluster_idx[i] == combined_cluster_idx[j]],
                    'explain': f"<|prompt{all_prompt_idx}|>",
                })
                appended_clusters.add(combined_cluster_idx[i])

            corr_pool_list[i] = [[j for j in range(len(shuffle_indices)) if combined_cluster_idx[i] == combined_cluster_idx[j]], cur_answer + f" <|prompt{all_prompt_idx}|>"]
            cur_answer += f' <|prompt{all_prompt_idx}|>'
            all_prompt_idx += 1
            llm_prompt_list.append([f"In a {situation} system, there are many monitoring metrics. We found the overall trends in " + ', '.join([combined_metrics[j] for j in range(len(shuffle_indices)) if combined_cluster_idx[i] == combined_cluster_idx[j]]) + f" are very similar. Please explain their relationship in physical meaning and simply describe why they can be similar in English in 1 sentence, like `these metrics are all xxx-related or xxx.` (the format may be different, but keep simple): "])
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
