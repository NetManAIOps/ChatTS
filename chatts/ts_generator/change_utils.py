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

def generate_spike(amplitude: float, max_length: int=None):
    assert max_length is None or max_length >= 2
    while True:
        rise_length = np.random.choice([1, 2, 3], p = [0.96, 0.03, 0.01])
        fall_length = np.random.choice([1, 2, 3], p = [0.96, 0.03, 0.01])
        peak_length = np.random.choice([0, 1, 2], p = [0.99, 0.005, 0.005])

        if max_length is None or (rise_length + fall_length + peak_length) <= max_length:
            break

    result = np.zeros(rise_length + fall_length + peak_length, dtype=np.float32)
    result[:rise_length] += generate_ts_change(rise_length, amplitude)
    result[rise_length:] += amplitude
    result[rise_length + peak_length:] += generate_ts_change(fall_length, -amplitude)
    
    return result
