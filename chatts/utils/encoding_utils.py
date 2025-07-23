"""
    Scripts for encoding time series into TS-MLLM Format.
"""
import numpy as np
import re
import copy
from typing import *


def minmax_scale_encoding(timeseries: np.ndarray):
    # Scalar
    mean = np.mean(timeseries)
    scaled_timeseries = timeseries - mean
    scale_factor = 1.0
    if np.any(np.abs(scaled_timeseries) >= 3.0):
        # Scale
        scale_factor = np.max(np.abs(scaled_timeseries)) / 3.0
        scaled_timeseries /= scale_factor
    prompt = f"[Offset: {-mean:.4f}|Scaled by: {scale_factor:.4f}]<ts><ts/>"

    return scaled_timeseries[:, np.newaxis], prompt, {'offset': float(-mean), 'scale_factor': float(scale_factor)}

def sp_encoding(timeseries: np.ndarray):
    # Scalar
    mean = np.mean(timeseries)
    scaled_timeseries = timeseries - mean
    scale_factor = 1.0
    if np.any(np.abs(scaled_timeseries) >= 3.0):
        # Scale
        scale_factor = np.max(np.abs(scaled_timeseries)) / 3.0
        scaled_timeseries /= scale_factor

    prompt = f"[Value Offset: {-mean:.4f}|Value Scaling: {scale_factor:.4f}]<ts><ts/>"

    result_timeseries = np.stack([scaled_timeseries, np.ones_like(scaled_timeseries)], axis=-1).reshape(-1, 1)

    return result_timeseries, prompt, {'offset': float(-mean), 'scale_factor': float(scale_factor)}

def no_encoding(timeseries: np.ndarray):
    return np.array(timeseries), "<ts><ts/>", {}

def timeseries_encoding(timeseries: np.ndarray, method: str):
    if method == 'minmax_scale':
        return minmax_scale_encoding(timeseries)
    elif method == 'sp':
        return sp_encoding(timeseries)
    elif method == 'no':
        return no_encoding(timeseries)
    else:
        raise NotImplementedError(f"Timeseries encoding method: {method} not implemented!")

def timeseries_prompt(prompt: str, timeseries: np.ndarray):
    if type(timeseries) == np.ndarray:
        timeseries = timeseries.tolist()

    prompt_list = prompt.split('<ts><ts/>')
    result = prompt_list[0]
    assert len(timeseries) == len(prompt_list) - 1

    for i in range(len(timeseries)):
        result += f"<ts>{[[round(k, 3) for k in j] for j in list(timeseries[i])]}<ts/>" + prompt_list[i + 1]

    return result

def eval_prompt_to_encoding(prompt: str, timeseries: list, method: str) -> Tuple[str, np.ndarray]:
    prompt_list = prompt.split('<ts><ts/>')
    result_prompt = prompt_list[0]
    assert len(timeseries) == len(prompt_list) - 1

    # Convert time series and pad
    scaled_timeseries: List[np.ndarray] = []
    
    for i in range(len(timeseries)):
        cur_ts, cur_prompt, _ = timeseries_encoding(np.array(timeseries[i]), method)
        result_prompt += f"{cur_prompt}" + prompt_list[i + 1]
        scaled_timeseries.append(np.array([cur_ts]))

    # Pad batch ts
    max_length = max(arr.shape[1] for arr in scaled_timeseries)
    padded_time_series_attributes = [
        np.pad(arr, ((0, 0), (0, max_length - arr.shape[1]), (0, 0)), mode='constant', constant_values=0)
        for arr in scaled_timeseries
    ]
    concatenated_time_series = np.concatenate(padded_time_series_attributes, axis=0)

    return result_prompt, concatenated_time_series

def timeseries_to_list(timeseries, digits: int=6, cp=True):
    if cp:
        result = copy.deepcopy(timeseries)
    else:
        result = timeseries
    if type(result) == np.ndarray:
        result = result.tolist()
    
    if type(result[0]) == float:
        for i in range(len(result)):
            result[i] = round(float(result[i]), 6)
    else:
        for i in range(len(result)):
            result[i] = timeseries_to_list(result[i], digits, cp=False)

    return result
