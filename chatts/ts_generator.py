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
from typing import *
from chatts.trend_utils import generate_random_points, generate_trend_prompt, generate_trend_curve, generate_trend_list


# Config
ENABLE_MULTIPLE_TREND = True
ENABLE_DROP_PROMPT = True
ENABLE_MULTIPLE_SEASONAL = False
ENABLE_MULTIPLE_NOISE = False

# All Config for TS Attributes
# Notes
# 1. Seaonal and Frequency can be combined (e.g., high frequency-sin periodic fluctuation, so there should be 7 types of seasonal attributes)
# 2. We implemented 2 types of noise types (e.g., random noise, sin noise), see the codes below for more information
# 3. The value following the overall attribute is the probability of the attribute to be selected
# All Attribute Set
all_attribute_set = {
    "overall_attribute": {
        "seasonal": {
            "no periodic fluctuation": 0.6,
            "sin periodic fluctuation": 0.35,
            "square periodic fluctuation": 0.02,
            "triangle periodic fluctuation": 0.03
        },
        "trend": {
            "decrease": 0.3,
            "increase": 0.3,
            "keep steady": 0.3,
            "multiple": 0.1
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
        "upward spike": 12,
        "downward spike": 10,
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

def generate_random_attributes(overall_attribute: Dict[str, Dict[str, float]], change_attribute: Dict[str, float], change_positions: Optional[List[Tuple[Optional[int], Optional[float]]]] = None, seq_len: int = 512):
    if change_positions is None:
        change_positions = [(None, None) for _ in range(random.randint(0, 3))]
    description = {}

    description["seasonal"] = {
        "type": np.random.choice(list(overall_attribute['seasonal']), p=np.array(list(overall_attribute['seasonal'].values()))/sum(list(overall_attribute['seasonal'].values())))
    }

    trend_candidates = overall_attribute['trend'].copy()
    if not ENABLE_MULTIPLE_TREND and "multiple" in trend_candidates:
        trend_candidates.pop("multiple")
    trend_char = np.random.choice(list(trend_candidates), p=np.array(list(trend_candidates.values()))/sum(list(trend_candidates.values())))
    
    description["trend"] = {
        "type": trend_char
    }

    num_local_chars = len(change_positions)
    local_chars = list(np.random.choice(list(change_attribute), size=num_local_chars, p=np.array(list(change_attribute.values()))/sum(list(change_attribute.values()))))
    
    description["local"] = []
    for char in local_chars:
        local_position, local_amplitude = change_positions.pop()
        description["local"].append({
            "type": char,
            "position_start": local_position,
            "amplitude": local_amplitude
        })

    if 'no periodic fluctuation' not in description["seasonal"]['type']:
        description["frequency"] = {'type': np.random.choice(list(overall_attribute['frequency']), p=np.array(list(overall_attribute['frequency'].values()))/sum(list(overall_attribute['frequency'].values())))}
    else:
        description["frequency"] = {'type': 'no periodicity'}
        
    description["noise"] = {'type': np.random.choice(list(overall_attribute['noise']), p=np.array(list(overall_attribute['noise'].values()))/sum(list(overall_attribute['noise'].values())))}

    return description

def generate_controlled_attributes(attribute_set, change_positions: Optional[List[Tuple[Optional[int], Optional[float]]]] = None):
    if change_positions is None:
        change_positions = [(None, None) for _ in range(random.randint(0, 3))]
    description = {}

    seasonal_p = [all_attribute_set['overall_attribute']['seasonal'][i] for i in attribute_set['seasonal']['attributes']]
    description["seasonal"] = {
        "type": np.random.choice(list(attribute_set['seasonal']['attributes']), p=np.array(seasonal_p)/sum(seasonal_p)),
        "amplitude": random.uniform(attribute_set['seasonal']['amplitude']['min'], attribute_set['seasonal']['amplitude']['max'])
    }
    
    if not ENABLE_MULTIPLE_TREND:
        if "multiple" in attribute_set['trend']['attributes']:
            attribute_set['trend']['attributes'].remove("multiple")
            if len(attribute_set['trend']['attributes']) == 0:
                attribute_set['trend']['attributes'] = ['increase', 'decrease', 'keep steady']
    trend_p = [all_attribute_set['overall_attribute']['trend'][i] for i in attribute_set['trend']['attributes']]
    description["trend"] = {
        "type": np.random.choice(list(attribute_set['trend']['attributes']), p=np.array(trend_p)/sum(trend_p)),
        "start": random.uniform(attribute_set['trend']['start']['min'], attribute_set['trend']['start']['max']),
        "amplitude": random.uniform(attribute_set['trend']['amplitude']['min'], attribute_set['trend']['amplitude']['max'])
    }

    num_local_chars = len(change_positions)
    change_p = [all_attribute_set['change'][i] for i in attribute_set['change']['attributes']]
    local_chars = list(np.random.choice(list(attribute_set['change']['attributes']), size=num_local_chars, p=np.array(change_p)/sum(change_p)))
    
    description["local"] = []
    for char in local_chars:
        description["local"].append({
            "type": char,
            "position_start": None,
            "amplitude": random.uniform(attribute_set['change']['amplitude']['min'], attribute_set['change']['amplitude']['max'])
        })

    if 'no periodic fluctuation' not in description["seasonal"]['type']:
        # Generate period and then determine type
        period = random.uniform(attribute_set['seasonal']['period']['min'], attribute_set['seasonal']['period']['max'])
        if period < 30.0:
            description["frequency"] = {'type': 'high frequency', 'period': round(period, 1)}
        else:
            description["frequency"] = {'type': 'low frequency', 'period': round(period, 1)}
    else:
        description["frequency"] = {'type': 'no periodicity'}
        
    noise_p = [all_attribute_set['overall_attribute']['noise'][i] for i in attribute_set['noise']['attributes']]
    description["noise"] = {'type': np.random.choice(list(attribute_set['noise']['attributes']), p=np.array(noise_p)/sum(noise_p))}

    return description

def generate_seasonal_wave(period, amplitude_list, split_points, seq_len, wave_type=None):
    # Time array
    t = np.linspace(0, seq_len, seq_len)
    data = np.zeros(seq_len)
    base_frequency = 1 / period

    # Amplitude series
    amplitude_series = np.zeros(seq_len)
    for i in range(len(amplitude_list)):
        amplitude_series[split_points[i]:split_points[i + 1]] = amplitude_list[i]

    # Smoothing amplitude_series with window size
    sliding_window = 5
    for i in range(seq_len - sliding_window):
        amplitude_series[i + sliding_window // 2] = np.mean(amplitude_series[i:i + sliding_window])

    if wave_type is None:
        wave_type = str(np.random.choice(['sin', 'square', 'triangle'], p=[0.7, 0.15, 0.15]))

    if wave_type == 'sin':
        num_harmonics = np.random.randint(1, max(1, min(period // 6, 10)))
        for n in range(1, num_harmonics + 1):
            phase = np.random.uniform(0, 2 * np.pi)
            harmonic_amplitude = amplitude_series / n * (1 + np.random.uniform(0, 0.05) * np.sin(np.random.uniform(1, 3) * np.pi * t / seq_len + np.random.uniform(0, 2 * np.pi)))
            data += harmonic_amplitude * np.sin(2 * np.pi * base_frequency * n * t + phase)
    elif wave_type == 'square':
        start = np.random.uniform(0, 0.3)
        duration = np.random.uniform(0.1, 0.3)
        for i in range(seq_len):
            cycle_pos = (t[i] % period) / period
            if start <= cycle_pos < start + duration:
                data[i] = amplitude_series[i]
            else:
                data[i] = 0.0
    else:
        start = np.random.uniform(0, 0.3)
        duration = np.random.uniform(0.1, 0.6)
        end = start + duration
        for i in range(seq_len):
            cycle_pos = (t[i] % period) / period
            if start <= cycle_pos < end:
                if cycle_pos < (start + end) / 2:
                    data[i] = amplitude_series[i] * 2 * (cycle_pos - start) / duration
                else:
                    data[i] = amplitude_series[i] * 2 * (end - cycle_pos) / duration
            else:
                data[i] = 0.0

    # normalize to amplitude
    data = data / (data.max() - data.min() + 1e-7) * max(amplitude_list)
    data -= np.mean(data)

    return data

def generate_sin_noise(amplitude, seq_len):
    # Time array
    t = np.linspace(0, seq_len, seq_len)
    data = np.zeros(seq_len)

    num_harmonics = 200
    for n in range(1, num_harmonics + 1):
        phase = np.random.uniform(0, 2 * np.pi)
        cur_freq = np.random.uniform(50 / seq_len, 200 / seq_len)
        data += np.sin(cur_freq * t + phase) * np.random.uniform(0.3, 1.0)

    # normalize to amplitude
    data = data / (data.max() - data.min() + 1e-7) * amplitude
    data -= np.mean(data)

    return data

def generate_ts_change(length: int, amplitude: float, add_random_noise: bool=True):
    x = np.arange(length) / length
    func = random.choice([
        lambda x: x ** 2,
        lambda x: np.sin(x * np.pi / 2),
        lambda x: x,
        lambda x: 1.0 - (1.0 - x) ** 2
    ])
    cur_value = func(x)

    if add_random_noise:
        # Randomly add noise
        if random.random() > 0.8 and length > 3:
            cur_value += np.random.uniform(-1.0, 1.0, length) * np.random.uniform(0.1, 0.3)

    cur_value = cur_value / (cur_value.max() - cur_value.min() + 1e-7) * amplitude

    return cur_value

def generate_spike(amplitude: float):
    rise_length = np.random.choice([1, 2, 3], p = [0.8, 0.15, 0.05])
    fall_length = np.random.choice([1, 2, 3], p = [0.8, 0.15, 0.05])
    peak_length = np.random.choice([0, 1, 2], p = [0.96, 0.03, 0.01])

    result = np.zeros(rise_length + fall_length + peak_length, dtype=np.float32)
    result[:rise_length] += generate_ts_change(rise_length, amplitude)
    result[rise_length:] += amplitude
    result[rise_length + peak_length:] += generate_ts_change(fall_length, -amplitude)
    
    return result

def generate_noise(attribute_pool, y, overall_amplitude, seq_len):
    max_change = np.abs(np.max(y) - np.min(y))
    noise_level = attribute_pool["noise"]['type']
    if noise_level == "noisy":
        if random.random() > 0.5 and max_change > overall_amplitude / 2 and attribute_pool["frequency"]['type'] == "no periodicity":
            # Generate a sin type noise
            noise = generate_sin_noise(0.2 * overall_amplitude, seq_len)
            noise += np.random.normal(0, 0.03 * overall_amplitude, seq_len)
            std = round(float(np.std(noise)), 3)
            attribute_pool["noise"]["detail"] = f"There is a irregular fluctuating noise, indicating a noisy curve: "
        else:
            # Generate random type noise
            std = np.random.uniform(0.03, 0.15) * overall_amplitude
            noise = np.random.normal(0, std, seq_len)
            attribute_pool["noise"]["detail"] = f"There is a random noise, indicating a noisy curve: "

        # Apply noise segments
        num_noise_segments = 1
        if ENABLE_MULTIPLE_NOISE:
            num_noise_segments = random.randint(1, 3)
        
        # Choose segments to apply noise
        attribute_pool["noise"]["segments"] = []
        noise_segments = generate_split_points(seq_len, num_noise_segments)
        for i in range(num_noise_segments):
            noise_start = noise_segments[i]
            noise_end = noise_segments[i + 1]
            noise_std_amp = np.random.uniform(0.1, 5.0)
            noise[noise_start:noise_end] *= noise_std_amp
            attribute_pool["noise"]["segments"].append({
                "position_start": noise_start,
                "position_end": noise_end,
                "amplitude": round(noise_std_amp * std, 2),
                "description": f"the noise std is {noise_std_amp * std:.2f} between point {noise_start} and point {noise_end}"
            })
            attribute_pool["noise"]["detail"] += f"the noise std is {noise_std_amp * std:.2f} between point {noise_start} and point {noise_end}, "
        attribute_pool["noise"]["detail"] = attribute_pool["noise"]["detail"][:-2] + ". "
    elif noise_level == "almost no noise":
        if max_change > overall_amplitude / 2:
            std = np.random.uniform(0.0, 0.001) * overall_amplitude
        else:
            std = 0.0
        noise = np.random.normal(0, std, seq_len)
        attribute_pool["noise"]["std"] = round(std, 3)
        attribute_pool["noise"]["detail"] = f"The overall noise standard deviation is around {std:.2f}, very small compared the overall change of the curve. The curve is overall smooth with almost no noise. "
    
    return noise

def generate_seasonal(attribute_pool, overall_amplitude, seq_len):
        y = np.zeros(seq_len)
        if "no period" not in attribute_pool["seasonal"]['type']:
            if attribute_pool["seasonal"]['type'] == "periodic fluctuation":
                wave_type = None
            else:
                wave_type = attribute_pool["seasonal"]["type"].split(" ")[0]
            if 'amplitude' not in attribute_pool['seasonal']:
                # Many periods of seasonal amplitudes
                num_seasonal = 1
                if ENABLE_MULTIPLE_SEASONAL:
                    num_seasonal = random.randint(1, 3)
                amp = []
                for _ in range(num_seasonal):
                    amp.append(random.uniform(1.0, 2.0) * overall_amplitude)
                split_points = generate_split_points(seq_len, num_seasonal)
            else:
                # Only one period of seasonal amplitudes
                amp = [attribute_pool['seasonal']['amplitude']]
                split_points = [0, seq_len]
            y += generate_seasonal_wave(attribute_pool['frequency']['period'], amp, split_points, seq_len, wave_type)

            attribute_pool["seasonal"]['detail'] = f"The time series is showing {attribute_pool['seasonal']['type']}: "
            attribute_pool["seasonal"]["segments"] = []
            for i in range(len(amp)):
                attribute_pool["seasonal"]["segments"].append({
                    "amplitude": round(amp[i], 2),
                    "position_start": split_points[i],
                    "position_end": split_points[i + 1],
                    "description": f"the amplitude of the periodic fluctuation is {amp[i]:.1f} between point {split_points[i]} and point {split_points[i + 1]}"
                })
                attribute_pool["seasonal"]['detail'] += f"the amplitude of the periodic fluctuation is {amp[i]:.1f} between point {split_points[i]} and point {split_points[i + 1]}, "
            attribute_pool["seasonal"]['detail'] = attribute_pool["seasonal"]['detail'][:-2] + ". "
        elif attribute_pool["seasonal"]['type'] == "no periodic fluctuation":
            y += 0.0
            attribute_pool["seasonal"]["segments"] = []
            attribute_pool["seasonal"]['detail'] = f"No periodic fluctuations observed, showing {attribute_pool['seasonal']['type']}. "
        return y

def generate_local_chars(attribute_pool, overall_amplitude, seq_len):
    """
    Generate a time series with local characteristics based on the given attribute pool.
    Parameters:
    attribute_pool (dict): A dictionary containing local characteristics to be applied to the time series.
                           Each local characteristic should have a "type", "position_start", and "amplitude".
    overall_amplitude (float): The overall amplitude to scale the local characteristics.
    seq_len (int): The length of the time series to be generated.
    Returns:
    np.ndarray: A numpy array representing the generated time series with local characteristics applied.
    Notes:
    - The function applies various types of local characteristics such as "shake", "upward spike", "downward spike",
      "continuous upward spike", "continuous downward spike", "upward convex", "downward convex", "sudden increase",
      "sudden decrease", "rapid rise followed by slow decline", "slow rise followed by rapid decline",
      "rapid decline followed by slow rise", "slow decline followed by rapid rise", "decrease after upward spike",
      "increase after downward spike", "increase after upward spike", "decrease after downward spike",
      "wide upward spike", and "wide downward spike".
    - The function uses several magic numbers for parameters such as peak lengths, amplitude scaling factors, and
      random ranges. These values are set empirically based on the desired characteristics of the generated time series.
    """
    x = np.arange(seq_len)
    y = np.zeros(seq_len)

    # Apply local attribute
    for local_char in attribute_pool["local"]:
        local_position = local_char["position_start"]
        local_amplitude = local_char["amplitude"]
        if local_position is None:
            while True:
                local_position = random.randint(1, seq_len - 5)

                if all([abs(local_position - i['position_start']) > seq_len / 8 for i in attribute_pool['local'] if i['position_start'] is not None]):
                    local_char['position_start'] = local_position
                    break
        if local_char["type"] == "shake":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            peak_start = local_position
            peak_length = random.randint(6, int(seq_len * 0.15))
            func = random.choice([
                lambda x: np.random.uniform(-1, 1, peak_length) * local_amplitude / 2,
                lambda x: np.sin(x[peak_start:peak_start + peak_length] * 5.0) * local_amplitude / 2
            ])
            y[peak_start:peak_start + peak_length] += func(x)
            local_char["detail"] = f"shake with an amplitude of about {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}"
            local_char['position_end'] = peak_start + peak_length
        elif local_char["type"] == "upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            peak_start = local_position
            spike = generate_spike(local_amplitude)
            peak_length = len(spike)
            y[peak_start:peak_start + peak_length] += spike
            spike_top_idx = peak_start + np.argmax(np.abs(spike))
            local_char["position_end"] = peak_start + peak_length
            local_char["detail"] = f"an upward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}, with the time series value rapidly rising from around <|{peak_start}|> to around <|{spike_top_idx}|> and then quickly falling back to around <|{peak_start + peak_length}|>"
        elif local_char["type"] == "downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            peak_start = local_position
            spike = generate_spike(-local_amplitude)
            peak_length = len(spike)
            y[peak_start:peak_start + peak_length] += spike
            local_char["position_end"] = peak_start + peak_length
            spike_top_idx = peak_start + np.argmax(np.abs(spike))
            local_char["detail"] = f"a downward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}, with the time series value rapidly falling from around <|{peak_start}|> to around <|{spike_top_idx}|> and then quickly rising back to around <|{peak_start + peak_length}|>"
        elif local_char["type"] == "continuous upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            peak_region_start = local_position
            num_peaks = random.randint(2, 5)
            peaks = []
            spike_top_ids = []
            all_amplitudes = []
            for _ in range(num_peaks):
                peak_start = random.randint(peak_region_start + 0, peak_region_start + 5)
                cur_amplitude = random.uniform(local_amplitude * 0.6, local_amplitude * 1.5)
                all_amplitudes.append(cur_amplitude)
                peaks.append(f"point {peak_start}")
                spike = generate_spike(cur_amplitude)
                peak_length = len(spike)
                y[peak_start:peak_start + peak_length] += spike
                peak_region_start = peak_start + peak_length
                spike_top_ids.append(peak_start + np.argmax(np.abs(spike)))
            
            local_char["position_end"] = peak_start + peak_length
            local_amplitude = float(np.mean(all_amplitudes))
            local_char["detail"] = f"at {' and '.join(peaks)}, there were {num_peaks} consecutive upward spikes with amplitudes ranging from {min(all_amplitudes):.2f} to {max(all_amplitudes):.2f}, with the time series value repeatedly rising sharply from around <|{local_position}|> to around <|" + '|> and <|'.join(map(str, spike_top_ids)) + f"|>, and then quickly falling back to around <|{peak_start + peak_length}|>"
        elif local_char["type"] == "continuous downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            peak_region_start = local_position
            num_peaks = random.randint(2, 5)
            peaks = []
            all_amplitudes = []
            spike_top_ids = []
            for _ in range(num_peaks):
                peak_start = random.randint(peak_region_start + 0, peak_region_start + 5)
                cur_amplitude = random.uniform(local_amplitude * 0.6, local_amplitude * 1.5)
                all_amplitudes.append(cur_amplitude)
                peaks.append(f"point {peak_start}")
                spike = generate_spike(-cur_amplitude)
                peak_length = len(spike)
                y[peak_start:peak_start + peak_length] += spike
                peak_region_start = peak_start + peak_length
                spike_top_ids.append(peak_start + np.argmax(np.abs(spike)))
            
            local_char["position_end"] = peak_start + peak_length
            local_amplitude = float(np.mean(all_amplitudes))
            local_char["detail"] = f"at {' and '.join(peaks)}, there were {num_peaks} consecutive downward spikes with amplitudes ranging from {min(all_amplitudes):.2f} to {max(all_amplitudes):.2f}, with the time series value repeatedly falling sharply from around <|{local_position - 1}|> to around <|" + '|> and <|'.join(map(str, spike_top_ids)) + f"|>, and then quickly rising back to around <|{peak_start + peak_length}|>"
        elif local_char["type"] == "upward convex":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            convex_start = local_position
            start_length, end_length = random.randint(1, 5), random.randint(1, 5)
            convex_length = random.randint(int(seq_len * 0.03), int(seq_len * 0.2))
            y[convex_start:convex_start + start_length] += generate_ts_change(start_length, local_amplitude)
            y[convex_start + start_length:convex_start + start_length + convex_length + end_length] += local_amplitude
            y[convex_start + start_length + convex_length: convex_start + start_length + convex_length + end_length] += generate_ts_change(end_length, -local_amplitude)
            local_char["position_end"] = convex_start + start_length + convex_length + end_length

            # Add some noise
            if random.random() > 0.7:
                y[convex_start + start_length:convex_start + start_length + convex_length] += np.sin((0.8 + np.abs(random.normalvariate(0.0, 2.0))) * x)[convex_start + start_length:convex_start + start_length + convex_length]
            if random.random() > 0.7:
                y[convex_start + start_length:convex_start + start_length + convex_length] += np.random.uniform(-1.0, 1.0, convex_length) * np.random.uniform(0.1, 0.5) * local_amplitude

            local_char["detail"] = f"starting from point {convex_start}, the time series value rises from around <|{convex_start}|> to around <|{convex_start + start_length}|>, forms an upward convex with an amplitude of about {local_amplitude:.2f}, and then falls back to around <|{convex_start + start_length + convex_length + end_length}|>"
        elif local_char["type"] == "downward convex":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            convex_start = local_position
            start_length, end_length = random.randint(1, 5), random.randint(1, 5)
            convex_length = random.randint(int(seq_len * 0.03), int(seq_len * 0.2))
            y[convex_start:convex_start + start_length] += generate_ts_change(start_length, -local_amplitude)
            y[convex_start + start_length:convex_start + start_length + convex_length + end_length] -= local_amplitude
            y[convex_start + start_length + convex_length: convex_start + start_length + convex_length + end_length] += generate_ts_change(end_length, local_amplitude)
            local_char["position_end"] = convex_start + start_length + convex_length + end_length

            # Add some noise
            if random.random() > 0.7:
                y[convex_start + start_length:convex_start + start_length + convex_length] += np.sin((0.8 + np.abs(random.normalvariate(0.0, 2.0))) * x)[convex_start + start_length:convex_start + start_length + convex_length]
            if random.random() > 0.7:
                y[convex_start + start_length:convex_start + start_length + convex_length] += np.random.uniform(-1.0, 1.0, convex_length) * np.random.uniform(0.1, 0.5) * local_amplitude

            local_char["detail"] = f"starting from point {convex_start}, the time series value falls from around <|{convex_start}|> to around <|{convex_start + start_length}|>, forms a downward convex with an amplitude of about {local_amplitude:.2f}, and then rises back to around <|{convex_start + start_length + convex_length + end_length}|>"
        elif local_char["type"] == "sudden increase":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            drop_length = random.randint(1, 10)
            y[local_position: local_position + drop_length] += generate_ts_change(drop_length, local_amplitude)
            y[local_position + drop_length:] += local_amplitude
            local_char["position_end"] = local_position + drop_length
            local_char["detail"] = f"a sudden increase with an amplitude of {local_amplitude:.2f} occurred between point {local_position} and point {local_position + drop_length}, with the time series value rising from around <|{local_position - 1}|> to around <|{local_position + drop_length}|>"
            if random.random() < 0.5:
                recover_length = random.randint(1, 10.0)
                recover_amplitude = random.uniform(0, local_amplitude / 3)
                y[local_position + drop_length: local_position + drop_length + recover_length] += generate_ts_change(recover_length, -recover_amplitude)
                y[local_position + drop_length + recover_length:] -= recover_amplitude

                if ENABLE_DROP_PROMPT:
                    local_char["detail"] += f", then a drop with an amplitude of {recover_amplitude:.2f} occurred between point {local_position + drop_length} and point {local_position + drop_length + recover_length}, with the time series value falling back to around <|{local_position + drop_length + recover_length + 1}|>"
        elif local_char["type"] == "sudden decrease":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            drop_length = random.randint(1, 10)
            y[local_position: local_position + drop_length] += generate_ts_change(drop_length, -local_amplitude)
            y[local_position + drop_length:] += -local_amplitude
            local_char["position_end"] = local_position + drop_length
            local_char["detail"] = f"a sudden decrease with an amplitude of {local_amplitude:.2f} occurred between point {local_position} and point {local_position + drop_length}, with the time series value falling from around <|{local_position - 1}|> to around <|{local_position + drop_length}|>"
            if random.random() < 0.5:
                recover_length = random.randint(1, 10.0)
                recover_amplitude = random.uniform(0, local_amplitude / 3)
                y[local_position + drop_length: local_position + drop_length + recover_length] += generate_ts_change(recover_length, recover_amplitude)
                y[local_position + drop_length + recover_length:] += recover_amplitude

                if ENABLE_DROP_PROMPT:
                    local_char["detail"] += f", then a rise with an amplitude of {recover_amplitude:.2f} occurred between point {local_position + drop_length} and point {local_position + drop_length + recover_length}, with the time series value rising back to around <|{local_position + drop_length + recover_length + 1}|>"
        elif local_char["type"] == "rapid rise followed by slow decline":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            rise_length = random.randint(1, 5)  # Length of rapid rise
            fall_length = random.randint(int(seq_len * 0.05), int(seq_len * 0.15))  # Length of slow decline
            y[local_position: local_position + rise_length] += generate_ts_change(rise_length, local_amplitude)
            y[local_position + rise_length: local_position + rise_length + fall_length] += generate_ts_change(fall_length, -local_amplitude) + local_amplitude
            local_char["position_end"] = local_position + rise_length + fall_length
            local_char["detail"] = (
                f"a rapid rise with an amplitude of {local_amplitude:.2f} occurred between point {local_position} and point {local_position + rise_length}, "
                f"with the time series value rising from around <|{local_position - 1}|> to around <|{local_position + rise_length}|>, "
                f"followed by a slow decline between point {local_position + rise_length} and point {local_position + rise_length + fall_length} back to around <|{local_position + rise_length + fall_length}|>"
            )
        elif local_char["type"] == "slow rise followed by rapid decline":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            rise_length = random.randint(int(seq_len * 0.05), int(seq_len * 0.15))  # Length of slow rise
            fall_length = random.randint(1, 5)  # Length of rapid decline
            y[local_position: local_position + rise_length] += generate_ts_change(rise_length, local_amplitude)
            y[local_position + rise_length: local_position + rise_length + fall_length] += generate_ts_change(fall_length, -local_amplitude) + local_amplitude
            local_char["position_end"] = local_position + rise_length + fall_length
            local_char["detail"] = (
                f"starting from point {local_position}, the time series value slowly rises, "
                f"reaching a peak at point {local_position + rise_length}, followed by a rapid decline between point {local_position + rise_length} and point {local_position + rise_length + fall_length} back to around <|{local_position + rise_length + fall_length}|>"
            )
        elif local_char["type"] == "rapid decline followed by slow rise":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            drop_length = random.randint(1, 5)  # Length of rapid decline
            rise_length = random.randint(int(seq_len * 0.05), int(seq_len * 0.15))  # Length of slow rise
            y[local_position: local_position + drop_length] += generate_ts_change(drop_length, -local_amplitude)
            y[local_position + drop_length: local_position + drop_length + rise_length] += generate_ts_change(rise_length, local_amplitude) - local_amplitude
            local_char["position_end"] = local_position + drop_length + rise_length
            local_char["detail"] = (
                f"a rapid decline with an amplitude of {local_amplitude:.2f} occurred between point {local_position} and point {local_position + drop_length}, "
                f"with the time series value falling from around <|{local_position - 1}|> to around <|{local_position + drop_length}|>, "
                f"followed by a slow rise between point {local_position + drop_length} and point {local_position + drop_length + rise_length} back to around <|{local_position + drop_length + rise_length}|>"
            )
        elif local_char["type"] == "slow decline followed by rapid rise":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            drop_length = random.randint(int(seq_len * 0.05), int(seq_len * 0.15))  # Length of slow decline
            rise_length = random.randint(1, 5)  # Length of rapid rise
            y[local_position: local_position + drop_length] += generate_ts_change(drop_length, -local_amplitude)
            y[local_position + drop_length: local_position + drop_length + rise_length] += generate_ts_change(rise_length, local_amplitude) - local_amplitude
            local_char["position_end"] = local_position + drop_length + rise_length
            local_char["detail"] = (
                f"starting from point {local_position}, the time series value slowly declines, "
                f"reaching a low point at point {local_position + drop_length}, followed by a rapid rise between point {local_position + drop_length} and point {local_position + drop_length + rise_length} back to around <|{local_position + drop_length + rise_length}|>"
            )
        elif local_char["type"] == "decrease after upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            fall_amplitude = random.uniform(0.1, 0.7) * local_amplitude
            peak_start = local_position
            spike = generate_spike(local_amplitude)
            peak_length = len(spike)
            fall_length = random.randint(2, int(seq_len * 0.05))  # Length of slow decline
            y[peak_start: peak_start + peak_length] += spike
            spike_top_idx = peak_start + np.argmax(np.abs(spike))
            y[peak_start + peak_length: peak_start + peak_length + fall_length] += generate_ts_change(fall_length, -fall_amplitude)
            y[peak_start + peak_length + fall_length: ] -= fall_amplitude
            local_char["position_end"] = peak_start + peak_length + fall_length
            local_char["detail"] = (
                f"an upward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}, "
                f"with the time series value rapidly rising from around <|{peak_start - 1}|> to around <|{spike_top_idx}|> and quickly falling back, "
                f"followed by a further decline between point {peak_start + peak_length} and point {peak_start + peak_length + fall_length} to around <|{peak_start + peak_length + fall_length}|>"
            )
        elif local_char["type"] == "increase after downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            fall_amplitude = random.uniform(0.1, 0.7) * local_amplitude
            peak_start = local_position
            spike = generate_spike(-local_amplitude)
            peak_length = len(spike)
            rise_length = random.randint(2, int(seq_len * 0.05))  # Length of slow rise
            y[peak_start: peak_start + peak_length] += spike
            spike_top_idx = peak_start + np.argmax(np.abs(spike))
            y[peak_start + peak_length: peak_start + peak_length + rise_length] += generate_ts_change(rise_length, fall_amplitude)
            y[peak_start + peak_length + rise_length: ] += fall_amplitude
            local_char["position_end"] = peak_start + peak_length + rise_length
            local_char["detail"] = (
                f"a downward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}, "
                f"with the time series value rapidly falling from around <|{peak_start}|> to around <|{spike_top_idx}|> and quickly rising back, "
                f"followed by a further rise between point {peak_start + peak_length} and point {peak_start + peak_length + rise_length} to around <|{peak_start + peak_length + rise_length}|>"
            )
        elif local_char["type"] == "increase after upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            fall_amplitude = random.uniform(0.1, 0.7) * local_amplitude
            peak_start = local_position
            spike = generate_spike(local_amplitude)
            peak_length = len(spike)
            fall_length = random.randint(2, int(seq_len * 0.05))  # Length of slow decline
            y[peak_start: peak_start + peak_length] += spike
            spike_top_idx = peak_start + np.argmax(np.abs(spike))
            y[peak_start + peak_length: peak_start + peak_length + fall_length] += generate_ts_change(fall_length, fall_amplitude)
            y[peak_start + peak_length + fall_length: ] += fall_amplitude
            local_char["position_end"] = peak_start + peak_length + fall_length
            local_char["detail"] = (
                f"an upward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}, "
                f"with the time series value rapidly rising from around <|{peak_start - 1}|> to around <|{spike_top_idx}|> and quickly falling back, "
                f"followed by a further rise between point {peak_start + peak_length} and point {peak_start + peak_length + fall_length} to around <|{peak_start + peak_length + fall_length}|>"
            )
        elif local_char["type"] == "decrease after downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            fall_amplitude = random.uniform(0.1, 0.7) * local_amplitude
            peak_start = local_position
            spike = generate_spike(-local_amplitude)
            peak_length = len(spike)
            rise_length = random.randint(2, int(seq_len * 0.05))  # Length of slow rise
            y[peak_start: peak_start + peak_length] += spike
            spike_top_idx = peak_start + np.argmax(np.abs(spike))
            y[peak_start + peak_length: peak_start + peak_length + rise_length] += generate_ts_change(rise_length, -fall_amplitude)
            y[peak_start + peak_length + rise_length: ] -= fall_amplitude
            local_char["position_end"] = peak_start + peak_length + rise_length
            local_char["detail"] = (
                f"a downward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}, "
                f"with the time series value rapidly falling from around <|{peak_start}|> to around <|{spike_top_idx}|> and quickly rising back, "
                f"followed by a further decline between point {peak_start + peak_length} and point {peak_start + peak_length + rise_length} to around <|{peak_start + peak_length + rise_length}|>"
            )
        elif local_char["type"] == "wide upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            # Define longer rise and fall lengths
            rise_length = random.randint(int(seq_len * 0.02), int(seq_len * 0.08))  # Length of slow rise
            peak_length = random.randint(1, 4)  # Length of short peak
            fall_length = random.randint(int(seq_len * 0.02), int(seq_len * 0.08))  # Length of slow decline

            # Slow rise
            y[local_position: local_position + rise_length] += generate_ts_change(rise_length, local_amplitude)
            # Short peak
            y[local_position + rise_length: local_position + rise_length + peak_length + fall_length] += local_amplitude
            # Slow decline
            y[local_position + rise_length + peak_length: local_position + rise_length + peak_length + fall_length] += generate_ts_change(fall_length, -local_amplitude)
            local_char["position_end"] = local_position + rise_length + peak_length + fall_length
            local_char["detail"] = (
                f"a slow rise from around <|{local_position - 1}|> to around <|{local_position + rise_length}|> occurred between point {local_position} and point {local_position + rise_length}, "
                f"forming a short peak with an amplitude of {local_amplitude:.2f}, "
                f"followed by a slow decline between point {local_position + rise_length + peak_length} and point {local_position + rise_length + peak_length + fall_length} back to around <|{local_position + rise_length + peak_length + fall_length + 1}|>"
            )
        elif local_char["type"] == "wide downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            # Define longer decline and rise lengths
            drop_length = random.randint(int(seq_len * 0.02), int(seq_len * 0.08))  # Length of slow decline
            peak_length = random.randint(1, 4)  # Length of short trough
            rise_length = random.randint(int(seq_len * 0.02), int(seq_len * 0.08))  # Length of slow rise

            # Slow decline
            y[local_position: local_position + drop_length] += generate_ts_change(drop_length, -local_amplitude)
            # Short trough
            y[local_position + drop_length: local_position + drop_length + peak_length + rise_length] -= local_amplitude
            # Slow rise
            y[local_position + drop_length + peak_length: local_position + drop_length + peak_length + rise_length] += generate_ts_change(rise_length, local_amplitude)
            local_char["position_end"] = local_position + drop_length + peak_length + rise_length
            local_char["detail"] = (
                f"a slow decline from around <|{local_position - 1}|> to around <|{local_position + drop_length}|> occurred between point {local_position} and point {local_position + drop_length}, "
                f"forming a short trough with an amplitude of {local_amplitude:.2f}, "
                f"followed by a slow rise between point {local_position + drop_length + peak_length} and point {local_position + drop_length + peak_length + rise_length} back to around <|{local_position + drop_length + peak_length + rise_length + 1}|>"
            )

        local_char['amplitude'] = local_amplitude

    # Sort the local char
    attribute_pool["local"] = sorted(attribute_pool["local"], key=lambda x: x['position_start'])

    return y

def generate_trend(attribute_pool, y, overall_amplitude, overall_bias, seq_len):
    # Apply trend attribute
    trend = attribute_pool["trend"]["type"]

    if 'amplitude' in attribute_pool['trend']:
        amplitude = attribute_pool['trend']['amplitude']
    else:
        amplitude = random.uniform(0.8, 3.0) * overall_amplitude
    if 'start' in attribute_pool['trend']:
        bias = attribute_pool['trend']['start']
    else:
        bias = overall_bias

    if trend == "decrease":
        cur_value = generate_ts_change(seq_len, -amplitude, add_random_noise=False) + bias
        y += cur_value
        attribute_pool["trend"]["detail"] = f"From the perspective of the slope, the overall trend is decreasing. "
        attribute_pool["trend"]["trend_list"] = [("decrease", 0, seq_len - 1)]
    elif trend == "increase":
        cur_value = generate_ts_change(seq_len, amplitude, add_random_noise=False) + bias
        y += cur_value
        attribute_pool["trend"]["detail"] = f"From the perspective of the slope, the overall trend is increasing. "
        attribute_pool["trend"]["trend_list"] = [("increase", 0, seq_len - 1)]
    elif trend == "multiple":
        # Ensure the generated trend has more than one type
        while True:
            points = generate_random_points(seq_len=seq_len)[0]
            if len(generate_trend_list(points, seq_len)) > 1:
                break
        trend_ts = generate_trend_curve(seq_len=seq_len, points=points)[1]
        y += trend_ts * amplitude
        attribute_pool["trend"]["detail"] = "From the perspective of the slope, the overall trend contains multiple different segments: " + generate_trend_prompt(points)
        attribute_pool["trend"]["trend_list"] = generate_trend_list(points, seq_len)
    elif trend == "keep steady":
        y += bias
        attribute_pool["trend"]["detail"] = f"From the perspective of the slope, the overall trend is steady. "
        attribute_pool["trend"]["trend_list"] = [("keep steady", 0, seq_len - 1)]

    # Find increase or decrease in local char
    local_phase_change = [i['type'] for i in attribute_pool["local"] if 'increase' in i['type'] or 'decrease' in i['type']]
    if len(local_phase_change):
        attribute_pool["trend"]["detail"] += f"However, local phase changes were observed, including: {', '.join(local_phase_change)}. "
    attribute_pool["trend"]["start"] = round(float(y[0]), 2)
    attribute_pool["trend"]["amplitude"] = round(float(y[-1] - y[0]), 2)
    attribute_pool["trend"]["detail"] += f"The value of time series starts from around {float(y[0]):.2f} and ends at around {float(y[-1]):.2f}, with an overall amplitude of {float(y[-1] - y[0]):.2f}. "
    return y

def generate_split_points(seq_len: int, num_segments: int) -> list:
    if num_segments < 1:
        raise ValueError("Number of segments must be at least 1.")
    if seq_len < num_segments:
        raise ValueError("Sequence length must be at least equal to the number of segments.")

    min_segment_len = seq_len / num_segments / 2  # Minimum segment length
    split_points = [0]  # Start with the first point
    
    for _ in range(num_segments - 1):
        # Determine the valid range for the next split point
        min_point = split_points[-1] + min_segment_len
        max_point = seq_len - (num_segments - len(split_points)) * min_segment_len
        if min_point >= max_point:
            raise ValueError("Cannot generate split points satisfying the constraints.")
        
        # Randomly select a split point within the valid range
        split_points.append(int(random.uniform(min_point, max_point)))
    split_points.append(seq_len)
    
    return split_points

def generate_time_series(attribute_pool, seq_len=512):
    """
    Generate a time series based on the given attribute pool and sequence length.
    Parameters:
    attribute_pool (dict): A dictionary containing attributes that define the characteristics of the time series.
    seq_len (int, optional): The length of the generated time series. Default is 512.
    Returns:
    tuple: A tuple containing the generated time series (numpy array) and the updated attribute pool (dict).
    The attribute_pool dictionary can contain the following keys:
    - "seasonal": A dictionary with a "type" key that defines the type of seasonal pattern.
    - "trend": A dictionary with a "type" key that defines the type of trend.
    - "frequency": A dictionary with "type" and "period" keys that define the frequency characteristics.
    - "overall_amplitude": A float that defines the overall amplitude of the time series.
    - "overall_bias": A float that defines the overall bias of the time series.
    - "local": A list of dictionaries that define local characteristics of the time series.
    - "statistics": A dictionary that will be populated with statistical information about the generated time series.
    The function performs the following steps:
    1. Adapts to legacy behavior by modifying the attribute pool based on certain conditions.
    2. Generates the base time series using a linear space.
    3. Adjusts the period based on the frequency attribute.
    4. Sets an overall amplitude and bias for the time series.
    5. Applies seasonal attributes to the time series.
    6. Applies local changes to the time series.
    7. Applies trend attributes to the time series.
    8. Replaces details in local characteristics with actual values from the time series.
    9. Adds noise to the time series.
    10. Adds statistical information to the attribute pool.
    Returns the generated time series and the updated attribute pool.
    """
    # Adapt to legacy behavior
    if not ENABLE_MULTIPLE_TREND:
        # (Step 1) Remove seasonal type
        if "no period" not in attribute_pool["seasonal"]['type']:
            attribute_pool["seasonal"]["type"] = "periodic fluctuation"

        # (Step 2) Remove multiple trend
        if attribute_pool["trend"]["type"] == "multiple":
            attribute_pool["trend"]["type"] = random.choice(["increase", "decrease", "keep steady"])

    # Generate timeseries
    x = np.linspace(0, 10 * np.pi, seq_len)
    y = np.zeros_like(x)
    
    period = seq_len
    if "frequency" in attribute_pool:
        if period not in attribute_pool["frequency"]:
            if attribute_pool["frequency"]['type'] == "high frequency":
                period = random.uniform(20.0, 30.0)
            elif attribute_pool["frequency"]['type'] == "low frequency":
                period = random.uniform(30.0, 80.0)

        if attribute_pool["frequency"]['type'] == "no periodicity":
            attribute_pool["frequency"]['period'] = 0.0
            attribute_pool["frequency"]['detail'] = "No significant periodic fluctuations observed, overall almost no periodicity. "
        else:
            attribute_pool["frequency"]['period'] = round(period, 1)
            attribute_pool["frequency"]['detail'] = f"Each fluctuation period is approximately {period:.1f} points, thus the overall fluctuation is {attribute_pool['frequency']['type']}. "

    # Set an overall amplitude for all attributes (to ensure the time series is not too flat)
    if 'overall_amplitude' in attribute_pool and 'overall_bias' in attribute_pool:
        overall_amplitude = attribute_pool['overall_amplitude']
        overall_bias = attribute_pool['overall_bias']
    else:
        overall_amplitude_e = np.random.choice([-1, 0, 1, 2, 3, 4, 5, 6, 7], p=[0.2, 0.3, 0.3, 0.05, 0.05, 0.03, 0.03, 0.03, 0.01])
        overall_amplitude = np.random.uniform(10.0 ** (overall_amplitude_e - 1), 10.0 ** (overall_amplitude_e + 1))
        overall_bias = np.random.uniform(-(10.0 ** (overall_amplitude_e + 1)), 10.0 ** (overall_amplitude_e + 1))
        attribute_pool['overall_amplitude'] = overall_amplitude
        attribute_pool['overall_bias'] = overall_bias

    # Apply Seasonal Feature
    y += generate_seasonal(attribute_pool, overall_amplitude, seq_len)

    # Apply local changes
    y += generate_local_chars(attribute_pool, overall_amplitude, seq_len)

    # Apply trend attribute
    y = generate_trend(attribute_pool, y, overall_amplitude, overall_bias, seq_len)

    # Replace detail in local_chars
    for local_char in attribute_pool["local"]:
        pattern = re.compile(r'<\|(\d+)\|>')

        def replacer(match):
            n = int(match.group(1))
            return f"{y[n]:.2f}"
        local_char['detail'] = pattern.sub(replacer, local_char['detail'])

    # Add noise
    y += generate_noise(attribute_pool, y, overall_amplitude, seq_len)

    # Add statistic information to description
    attribute_pool["statistics"] = {
        "mean": round(float(np.mean(y)), 2),
        "std": round(float(np.std(y)), 2),
        "max": round(float(np.max(y)), 2),
        "min": round(float(np.min(y)), 2),
        "max_pos": int(np.argmax(y)),
        "min_pos": int(np.argmin(y))
    }

    return y, attribute_pool

def attribute_to_text(time_series: np.ndarray, attribute_pool: dict, generate_values: bool=True, include_attributes: List[str] = ['length', 'trend', 'periodicity', 'frequency', 'noise', 'local', 'statistic']) -> str:
    """
    Generates a textual description of a time series based on various attributes and attributes.
    Args:
        time_series (np.ndarray): The time series data as a numpy array.
        attribute_pool (dict): A dictionary containing attribute details for the time series.
        generate_values (bool, optional): Deprecated. Use 'statistic' in include_attributes instead. Defaults to True.
        include_attributes (List[str], optional): A list of attributes to include in the description. Defaults to ['length', 'trend', 'periodicity', 'frequency', 'noise', 'local', 'statistic'].
    Returns:
        str: A detailed textual description of the time series.
    """
    # Adapt to legacy parameters
    if not generate_values and 'statistic' in include_attributes:
        include_attributes.remove('statistic')
    elif generate_values and 'statistic' not in include_attributes:
        include_attributes.append('statistic')

    seq_len = len(time_series)
    segment_mean = [round(np.mean(time_series[i:i+seq_len//32]), 2) for i in range(0, seq_len, seq_len // 32)]
    max_value = round(np.max(time_series), 2)
    min_value = round(np.min(time_series), 2)

    detailed_description = ''
    if 'length' in include_attributes:
        detailed_description += f"The length of the time series is {seq_len}. "
    if 'trend' in include_attributes:
        detailed_description += f"{attribute_pool['trend']['detail']}"
    if 'periodicity' in include_attributes:
        detailed_description += attribute_pool['seasonal']['detail']
    if "no" not in attribute_pool['seasonal']['type'] and 'frequency' in include_attributes:
        detailed_description += attribute_pool['frequency']['detail']
    if 'noise' in include_attributes:
        detailed_description += attribute_pool['noise']['detail']
    if 'local' in include_attributes:
        if len(attribute_pool["local"]):
            detailed_description += 'In terms of local characteristics, ' + ";".join([f"{i['detail']}, forming a {i['type']}" for i in attribute_pool['local']]) + '. '
        else:
            detailed_description += 'No local characteristics are found. '
    if 'statistic' in include_attributes:
        detailed_description += f"Specific data details: The time series is divided into 32 segments, with the approximate mean values for each {seq_len // 32}-point interval being: {segment_mean}. The maximum value of the entire series is {max_value}, and the minimum value is {min_value}."

    return detailed_description


def attribute_to_caption(time_series: np.ndarray, attribute_pool: dict, generate_values: bool=True) -> str:
    """
        Compared with text, caption is in a more natural and fluent way that combines the trend with the local flucations.
    """
    seq_len = len(time_series)
    segment_mean = [round(np.mean(time_series[i:i+seq_len//32]), 2) for i in range(0, seq_len, seq_len // 32)]
    max_value = round(np.max(time_series), 2)
    min_value = round(np.min(time_series), 2)

    # Some basic attribute_pool
    detailed_description = ''
    detailed_description += f"The length of the time series is {seq_len}. "
    detailed_description += attribute_pool['seasonal']['detail']
    if "no" not in attribute_pool['seasonal']['type']:
        detailed_description += attribute_pool['frequency']['detail']
    detailed_description += attribute_pool['noise']['detail']

    # Combine the multiple attribute_pool
    detailed_description += "In terms of the trend and changes of this time series: At the beginning, "
    all_local_changes = dict((int(v['position_start']), v) for v in attribute_pool['local'])
    cur_pos = 0
    while True:
        if cur_pos >= seq_len - 1:
            break

        # Find the next local change
        later_changes = sorted(k for k in all_local_changes if k >= cur_pos)
        later_trend = sorted(k[1] for k in attribute_pool["trend"]["trend_list"] if k[1] > cur_pos)
        cur_trend = [k for k in attribute_pool["trend"]["trend_list"] if (k[1] <= cur_pos < k[2])][0]

        if (len(later_changes) > 0 and len(later_trend) > 0 and later_changes[0] < later_trend[0]) or (len(later_changes) > 0 and len(later_trend) == 0):
            # Later is a change
            nxt_pos = later_changes[0]
            cur_change = [k for k in attribute_pool["local"] if k['position_start'] == nxt_pos][0]
            if nxt_pos > cur_pos:
                detailed_description += f"from point {cur_pos} to {nxt_pos}, the time series {cur_trend[0]} with values from {float(time_series[cur_pos]):.2f} to {float(time_series[nxt_pos]):.2f}; "
            detailed_description += f"from point {cur_change['position_start']} to point {cur_change['position_end']}, {cur_change['detail']}, forming a {cur_change['type']}; "
            cur_pos = cur_change['position_end']
        elif (len(later_changes) > 0 and len(later_trend) > 0 and later_changes[0] >= later_trend[0]) or (len(later_trend) > 0 and len(later_changes) == 0):
            # Later is a trend
            nxt_pos = later_trend[0]
            nxt_trend = [k for k in attribute_pool["trend"]["trend_list"] if k[1] == nxt_pos]
            if nxt_pos > cur_pos:
                detailed_description += f"from point {cur_pos} to {nxt_pos}, the time series {cur_trend[0]} with values from {float(time_series[cur_pos]):.2f} to {float(time_series[nxt_pos]):.2f}, and then the trend of the time series changes to {nxt_trend[0][0]}; "
            cur_pos = nxt_pos
        else:
            # Later is the end
            nxt_pos = seq_len - 1
            if nxt_pos > cur_pos:
                detailed_description += f"finally, from point {cur_pos} to {nxt_pos}, the time series {cur_trend[0]} with values from {float(time_series[cur_pos]):.2f} to {float(time_series[nxt_pos]):.2f}. "
            break
    
    if generate_values:
        detailed_description += f"Specific data details: The time series is divided into 32 segments, with the approximate mean values for each {seq_len // 32}-point interval being: {segment_mean}. The maximum value of the entire series is {max_value}, and the minimum value is {min_value}, the start value is {float(time_series[0]):.2f}, the end value if {float(time_series[-1]):.2f}. "
        
        # Random choose some points
        for _ in range(5):
            cur_pos = random.choice(list(range(seq_len)))
            detailed_description += f"The value of point {cur_pos} is {float(time_series[cur_pos]):.2f}. "

    return detailed_description

def prompt_to_inference(timeseries: np.ndarray, prompt: str) -> str:
    prompt_list = prompt.split("<ts><ts/>")
    result = prompt_list[0]

    for i in range(len(prompt_list) - 1):
        cur_ts = timeseries[i]
        if type(cur_ts) == np.ndarray:
            cur_ts = cur_ts.tolist()
        cur_ts = [[round(float(v), 4) for v in item] for item in cur_ts]
        result += f"<ts>{cur_ts}<ts/>" + prompt_list[i + 1]

    return result
