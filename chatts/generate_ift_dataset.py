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
import json
import os
import random
from chatts.encoding_utils import timeseries_encoding, timeseries_to_list
from tqdm import tqdm
import copy
import traceback

# Config
ENCODING_METHOD = 'sp'
TARGET_CNT = 10000
OUTPUT_BASE_DIR = json.load(open("config/datagen_config.json"))["data_output_dir"]
OUTPUT_PATH = f'{OUTPUT_BASE_DIR}/ift_256_{ENCODING_METHOD}.jsonl'
LABEL_FILES = [
    f'{OUTPUT_BASE_DIR}/labels/mts_local_llm_256_15000_{ENCODING_METHOD}.json',
    f'{OUTPUT_BASE_DIR}/labels/mts_shape_llm_256_15000_{ENCODING_METHOD}.json',
    f'{OUTPUT_BASE_DIR}/labels/uts_llm_256_15000_{ENCODING_METHOD}.json'
]
ALL_LOCAL_TYPES = {'increase after downward spike', 'increase after upward spike', 'upward spike', 'rapid decline followed by slow rise', 'slow rise followed by rapid decline', 'continuous upward spike', 'wide downward spike', 'slow decline followed by rapid rise', 'upward convex', 'shake', 'rapid rise followed by slow decline', 'sudden increase', 'downward spike', 'sudden decrease', 'continuous downward spike', 'decrease after upward spike', 'wide upward spike', 'decrease after downward spike', 'downward convex'}

# Return: question, answer
# L0: basic feature: STL shape + statistic
def generate_trend(sample):
    question = 'What is the trend of this time series? Please choose from ["steady", "decreasing", "increasing"] and describe the value trend change. Answer format: steady, the starting point value is around 32.10, and the trend change value from left to right is around 0.12.'
    answer = f"{sample['label']['trend']['type']}, the starting point value is around {sample['label']['trend']['start']:.2f}, and the trend change value from left to right is around {sample['label']['trend']['amplitude']:.2f}."
    if sample['label']['trend']['type'] == 'multiple':
        raise NotImplementedError("ift not implemented for multiple trend")
    return question, answer

def generate_trend_llm(sample):
    question = f'What is the trend of this time series? Please choose from ["steady", "decreasing", "increasing"], describe the value trend change, and conclude the llm meaning of this trend change in one sentence. Answer format: steady, the starting point value is around 32.10, and the trend change value from left to right is around 0.12. The trend indicates that the temperature is stable during the period.'
    answer = f"{sample['label']['trend']['type']}, the starting point value is around {sample['label']['trend']['start']:.2f}, and the trend change value from left to right is around {sample['label']['trend']['amplitude']:.2f}. The trend indicates that {sample['label']['trend']['detail']}"
    if sample['label']['trend']['type'] == 'multiple':
        raise NotImplementedError("ift not implemented for multiple trend")
    return question, answer

def generate_season(sample):
    question = 'What is the periodicity of this time series? Please choose from ["no periodic fluctuation", "periodic fluctuation"]. If there is periodic fluctuation, describe the fluctuation frequency and amplitude. Answer format: periodic fluctuation, each period is around 20.58 points, and the amplitude of the periodic fluctuation is around 31.51.'
    if 'no' in sample['label']['seasonal']['type']:
        answer = "no periodic fluctuation"
    else:
        answer = f"periodic fluctuation, each period is around {sample['label']['frequency']['period']:.2f} points, and the amplitude of the periodic fluctuation is around {sample['label']['seasonal']['amplitude']:.2f}."
    return question, answer

def generate_season_llm(sample):
    question = f'What is the periodicity of this time series? Please choose from ["no periodic fluctuation", "periodic fluctuation"], and conclude the llm meaning of the periodicity in one sentence. If there is periodic fluctuation, also describe the fluctuation frequency and amplitude. Answer format: periodic fluctuation, each period is around 20.58 points, and the amplitude of the periodic fluctuation is around 31.51. The periodic fluctuation indicates that the temperature is periodically changing in a day.'
    if 'no' in sample['label']['seasonal']['type']:
        answer = f'no periodic fluctuation. It indicates that {sample["label"]["seasonal"]["detail"]}'
    else:
        answer = f"periodic fluctuation, each period is around {sample['label']['frequency']['period']:.2f} points, and the amplitude of the periodic fluctuation is around {sample['label']['seasonal']['amplitude']:.2f}. It indicates that {sample['label']['seasonal']['detail']}"
    return question, answer

def generate_noise(sample):
    question = 'What are the noise characteristics of this time series? Please choose from ["noisy", "almost no noise"]. Answer format: noisy, the overall noise standard deviation is around 1.5.'
    answer = f"{sample['label']['noise']['type']}, the overall noise standard deviation is around {sample['label']['noise']['std']:.2f}."
    return question, answer

# L1: local feature: local change type + statistic
def generate_local(sample):
    question = 'What are the local characteristic fluctuations of this time series? The optional types of local characteristic fluctuations include: ["' + '", "'.join(sorted(ALL_LOCAL_TYPES)) + '"]. You need to analyze all the characteristic fluctuations that appear in this time series and answer each type, position, and amplitude in the format. Different local characteristic fluctuations should be separated by semicolons. Answer format: shake, position around point 125, amplitude 135.03; small sudden decrease, position around point 102, amplitude 31.05.'
    answer = '; '.join([f"{i['type'] if type(i['type']) == str else i['type'][0]}, position around point {i['position_start']}, amplitude {i['amplitude']:.2f}" for i in sample['label']['local']])

    if len(sample['label']['local']) == 0:
        answer = 'No local characteristic fluctuations found.'

    features = []
    for i in sample['label']['local']:
        features.append({
            'type': i['type'] if type(i['type']) == str else [j for j in i['type']],
            'position': i['position_start'],
            'amplitude': round(i['amplitude'], 2)
        })
    return question, answer

def generate_local_llm(sample):
    question = f'What are the local characteristic fluctuations of this time series? The optional types of local characteristic fluctuations include: ["' + '", "'.join(sorted(ALL_LOCAL_TYPES)) + '"]. You need to analyze all the characteristic fluctuations that appear in this time series and answer each type, position, and amplitude in the format, and conclude the llm meaning of **each** fluctuation in one sentence. Different local characteristic fluctuations should be separated by semicolons. Answer format: shake, position around point 125, amplitude 135.03. A sudden surge in public interest, likely due to significant news, a major event, or a trending topic related to the platform that rapidly captured user attention; small sudden decrease, position around point 102, amplitude 31.05. A slight increase in interest, possibly driven by minor news, promotions, or social media discussions that briefly captured attention without indicating a significant trend.'
    answer = '; '.join([f"{i['type'] if type(i['type']) == str else i['type'][0]}, position around point {i['position_start']}, amplitude {i['amplitude']:.2f}. {i['detail'] if not i['detail'].endswith('.') else i['detail'][:-1]}" for i in sample['label']['local']])

    if len(sample['label']['local']) == 0:
        answer = 'No local characteristic fluctuations found.'

    features = []
    for i in sample['label']['local']:
        features.append({
            'type': i['type'],
            'position': i['position_start'],
            'amplitude': round(i['amplitude'], 2),
            'explain': i['detail']
        })
    return question, answer

# L2: correlation and cluster
def generate_shape_correlation_llm(sample):
    if len(sample['label']['correlations']) == 0:
        raise NotImplementedError("ift not implemented for shape correlation with empty correlations")
    pairs = random.choice(sample['label']['correlations'])
    question = f'From the perspective of the overall trend, do {pairs["pair"][0]} and {pairs["pair"][1]} have very similar trend characteristics? Just answer yes or no, and explain why they are correlated/no correlated considering their llm meaning in one sentence. Answer format: Yes. Both metrics are related to the same system component, so they are highly correlated.'
    if pairs['label']:
        answer = 'Yes. ' + pairs['explain']
    else:
        answer = 'No. ' + pairs['explain']
    return question, answer

def generate_fluctuation_correlation_llm(sample):
    # Choice and balance the label
    positive_pairs = [p for p in sample['label']['correlations'] if p['label']]
    negative_pairs = [p for p in sample['label']['correlations'] if not p['label']]

    if len(positive_pairs) and (random.random() > 0.5 or len(negative_pairs) == 0):
        pairs = random.choice(positive_pairs)
    else:
        pairs = random.choice(negative_pairs)
    question = f'From the perspective of local fluctuations, do {pairs["pair"][0]} and {pairs["pair"][1]} both have fluctuations near point {sample["label"]["position"]}? Answer yes or no, the types of their correlated fluctuations (if yes), and explain why they are correlated/no correlated considering their llm meaning in one sentence. Answer format: Yes. [metric 1], shake; [metric 2], upward spike. Both [metric 1] and [metric 2] are related to CPU metrics, so they may have near fluctuations when the CPU usage is high.'
    if pairs['label']:
        answer = 'Yes. '
        # Find fluctuation type label
        def get_fluctuation_type(metric: str):
            for cluster in sample['label']['clusters']:
                if metric in cluster['cols']:
                    return cluster['col_idx'][cluster['cols'].index(metric)][1]
        answer += '; '.join([f"{m}, {get_fluctuation_type(m)}" for m in pairs['pair']])
        answer += '. ' + pairs['explain']
    else:
        answer = 'No. ' + pairs['explain']
    features = copy.deepcopy(pairs)

    if pairs['label']:
        features['pair'] = [[m, get_fluctuation_type(m)] for m in pairs['pair']]
    return question, answer

def generate_shape_cluster_llm(sample):
    cluster = random.choice(sample['label']['clusters'])
    question = f'From the perspective of the overall trend, which metric(s) have very similar trend characteristics with {random.choice(cluster["cols"])}? List the metrics (including itself) and explain why they have similar overall trend considering their llm meaning in one sentence. Answer format: A, B, C. All metrics are related to the same system component, so they may have similar overall trend.'
    answer = ', '.join(cluster['cols']) + '. ' + cluster['explain']
    return question, answer

def generate_fluctuation_cluster_llm(sample):
    cluster = random.choice(sample['label']['clusters'])
    question = f'From the perspective of the position of local fluctuations, which metric(s) have very similar local fluctuation characteristics with {random.choice(cluster["cols"])}? The optional types of local characteristic fluctuations include: ["' + '", "'.join(sorted(ALL_LOCAL_TYPES)) + '"]. List the metrics (including itself), the types of fluctuations, and explain why they have similar local fluctuations considering their llm meaning in one sentence. Answer format: [metric 1], shake; [metric 2], upward spike; [metric 3], downward spike. All metrics are related to the same system component, so they may have correlated local fluctuations near the same point.'
    answer = '; '.join([f"{i}, {cluster['col_idx'][idx][1]}" for idx, i in enumerate(cluster['cols'])]) + '. ' + cluster['explain']
    return question, answer


# Generate dataset
def generate_qa(sample, filename: str):
    # Step 1. Check data type
    candidate_funcs = []
    mts_flag = False
    llm_flag = False
    if 'uts' in filename:
        candidate_funcs += [generate_trend, generate_season, generate_noise, generate_local, generate_trend_llm, generate_season_llm, generate_local_llm]
        llm_flag = True
    if 'shape' in filename:
        candidate_funcs += [generate_shape_correlation_llm, generate_shape_cluster_llm]
        mts_flag = True
        llm_flag = True
    if 'local' in filename:
        candidate_funcs += [generate_fluctuation_correlation_llm, generate_fluctuation_cluster_llm]
        mts_flag = True
        llm_flag = True

    # Step 2. Randomly choose a data type
    funcs = np.random.choice(candidate_funcs, size=min(len(candidate_funcs), random.randint(3, 4)), replace=False)
    
    # Step 3. Augmentation
    original_timeseries = copy.deepcopy(sample['timeseries'])
    if mts_flag:
        timeseries = sample['timeseries']

        if llm_flag:
            cols = sample['label']['cols']
            question = f"You are a time series analysis expert. In a monitoring system of {sample['label']['situation']}, there are {len(timeseries)} metrics collected."
            for i in range(len(timeseries)):
                # Scalar
                cur_timeseries = np.array(timeseries[i])
                scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(cur_timeseries, ENCODING_METHOD)
                timeseries[i] = scaled_timeseries

                question += f"""\n "{sample['label']['cols'][i]}" is a time series with length of {len(timeseries[i])}: {cur_ts_prompt}"""
            question += ', please analyze the time series features and answer the following questions:'
    else:
        # Scalar
        timeseries = sample['timeseries']
        scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(timeseries, ENCODING_METHOD)
        if llm_flag:
            timeseries = [scaled_timeseries]
            cols = [sample['label']['metric_name']]
            question = f"""You are a time series analysis expert. This time series is "{sample['label']['metric_name']}" from {sample['label']['situation']} with length of {len(timeseries[0])}: {cur_ts_prompt}, please analyze the time series features and answer the following questions:"""
        else:
            timeseries = [scaled_timeseries]
            cols = ['Time Series']
            question = f'You are a time series analysis expert. Here is a time series of length {len(timeseries[0])}: {cur_ts_prompt}, please analyze the time series features and answer the following questions:'

    answer = ''
    
    # Step 3. Generate QAs
    for idx, func in enumerate(funcs):
        cur_question, cur_answer = func(sample)
        question += f'\n{idx+1}. {cur_question}'
        answer += f'{idx+1}. {cur_answer}\n'
    question += '\nNow, based on the above questions, please strictly follow the output format requirements and provide the answers. Each line corresponds to an answer to a question, formatted as:'
    question += '\n'.join([f"{i+1}. Strictly formatted answer {i+1}" for i in range(len(funcs))])

    # Step 4. Return result
    return {
        'timeseries': timeseries,
        'original_timeseries': original_timeseries,
        'cols': cols,
        'question': question,
        'answer': answer
    }

def generate_dataset():
    print("Start generation...")
    samples = []
    filenames = []
    for file in LABEL_FILES:
        label = json.load(open(file))
        for sample in label:
            samples.append(sample)
            filenames.append(file)
    samples = samples
    filenames = filenames

    result = []
    with tqdm(total=TARGET_CNT, desc='Generating samples') as pbar:
        while len(result) < TARGET_CNT:
            idx = random.randint(0, len(samples) - 1)
            sample = copy.deepcopy(samples[idx])
            try:
                qa = generate_qa(sample, filenames[idx])
            except NotImplementedError as err:
                continue
            except Exception as err:
                traceback.print_exc()
                continue
            if qa is not None:
                result.append(qa)
                pbar.update(1)
    
    print("Saving dataset...")
    with open(OUTPUT_PATH, 'wt') as f:
        for item in result:
            item = {
                'input': item['question'],
                'output': item['answer'],
                'timeseries': timeseries_to_list(item['timeseries']),
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'Finished! {len(result)} samples saved to {OUTPUT_PATH}.')


if __name__ == '__main__':
    generate_dataset()
