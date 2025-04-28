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

from chatts.ts_generator import attribute_to_text
import numpy as np

from typing import *


def trend_attribute_prompt(timeseries: np.ndarray, attribute_pool: List[dict], metrics: List[str], idx: List[int], *kargs):
    result = 'The trend information of some time series are as follows: '
    
    for i in idx:
        result += f"In {metrics[i]}, " + attribute_to_text(timeseries[i], attribute_pool[i], generate_values=False, include_attributes=['trend']).rstrip().rstrip('.') + '; '
    result = result[:-1]

    return result

def seasonal_attribute_prompt(timeseries: np.ndarray, attribute_pool: List[dict], metrics: List[str], idx: List[int], *kargs):
    result = 'The seasonal information of some time series are as follows: '
    
    for i in idx:
        result += f"In {metrics[i]}, " + attribute_to_text(timeseries[i], attribute_pool[i], generate_values=False, include_attributes=['periodicity', 'frequency']).rstrip().rstrip('.') + '; '
    result = result[:-1]

    return result

def noise_attribute_prompt(timeseries: np.ndarray, attribute_pool: List[dict], metrics: List[str], idx: List[int], *kargs):
    result = 'The noise information of some time series are as follows: '
    
    for i in idx:
        result += f"In {metrics[i]}, " + attribute_to_text(timeseries[i], attribute_pool[i], generate_values=False, include_attributes=['noise']).rstrip().rstrip('.') + '; '
    result = result[:-1]

    return result

def local_attribute_prompt(timeseries: np.ndarray, attribute_pool: List[dict], metrics: List[str], idx: List[int], *kargs):
    result = 'The local change information of some time series are as follows: '
    
    for i in idx:
        result += f"\n - In {metrics[i]}: " + attribute_to_text(timeseries[i], attribute_pool[i], generate_values=False, include_attributes=['local']).rstrip().rstrip('.')

    return result

def statistic_attribute_prompt(timeseries: np.ndarray, attribute_pool: List[dict], metrics: List[str], idx: List[int], *kargs):
    result = 'The statistic information of some time series are as follows: '
    
    for i in idx:
        cur_mean = round(float(attribute_pool[i]['statistics']['mean']), 2)
        cur_min = round(float(np.min(attribute_pool[i]['statistics']['max'])), 2)
        cur_max = round(float(np.max(attribute_pool[i]['statistics']['min'])), 2)
        cur_min_pos = attribute_pool[i]['statistics']['min_pos']
        cur_max_pos = attribute_pool[i]['statistics']['max_pos']
        
        result += f"In {metrics[i]}, the mean value is {cur_mean}, the minimum value is {cur_min} (around point {cur_min_pos}), and the maximum value is {cur_max} (around point {cur_max_pos}); "
    result = result[:-1]

    return result

def correlation_attribute_prompt(timeseries: np.ndarray, attribute_pool: List[dict], metrics: List[str], idx: List[int], corr_pool: List[Tuple[List[int], str]], *kargs):
    result = f'The correlation information of some time series are as follows: '
    corr_attribute_pool = []
    for i in idx:
        if i >= len(corr_pool):
            break
        if corr_pool[i] is None:
            continue
        corr_attribute_pool.append(" - " + corr_pool[i][1])
    result += '\n'.join(corr_attribute_pool)
    return result

def attribute_prompt(timeseries: np.ndarray, attribute_pool: List[dict], metrics: List[str], required_fields: Dict[str, List[int]], corr_pool: List[Tuple[List[int], str]]):
    cur_func_dict = {
        'trend': trend_attribute_prompt,
        'seasonal': seasonal_attribute_prompt,
        'noise': noise_attribute_prompt,
        'local': local_attribute_prompt,
        'statistic': statistic_attribute_prompt,
        'correlation': correlation_attribute_prompt,
    }
    
    result = f'There are {len(timeseries)} timeseries with length of {len(timeseries[0])}: '
    
    for i in range(len(timeseries)):
        result += f"the {i + 1}-th timeseries is {metrics[i]}; "
    result = result[:-2] + '.\n'
    
    for field, idx in required_fields.items():
        if field == 'correlation':
            result += cur_func_dict[field](timeseries, attribute_pool, metrics, idx, corr_pool) + '\n'
        else:
            result += cur_func_dict[field](timeseries, attribute_pool, metrics, idx) + '\n'

    return result
