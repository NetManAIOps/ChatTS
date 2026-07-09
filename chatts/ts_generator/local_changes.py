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

from abc import abstractmethod
import numpy as np
import random
from chatts.ts_generator.change_utils import generate_ts_change, generate_spike
import traceback
from typing import List
import yaml


# Basic Config
ENABLE_DROP_PROMPT = yaml.safe_load(open("config/datagen_config.yaml"))["enable_drop_prompt"]  # Enable or disable drop prompt in SuddenChange
LOCAL_CHANGE_VERBOSE = yaml.safe_load(open("config/datagen_config.yaml"))["local_change_verbose"]


def _clamp_raw_point(point: int, seq_len: int) -> int:
    return max(0, min(int(point), seq_len - 1))


def _previous_raw_point(point: int) -> int:
    return max(0, int(point) - 1)


def _inclusive_end_point(exclusive_end: int, seq_len: int) -> int:
    return _clamp_raw_point(int(exclusive_end) - 1, seq_len)


def _value_ref(point: int, seq_len: int) -> str:
    return f"<|{_clamp_raw_point(point, seq_len)}|>"


class BaseChange:
    """Base class for all local changes in time series"""
    
    def __init__(self, change_type: str, position_start: int = None, amplitude: float = None):
        self.change_type = change_type
        self.position_start = position_start
        self.amplitude = amplitude
        self.position_end = None
        self.detail = ""
    
    @abstractmethod
    def get_min_length(self) -> int:
        """Return minimum length required for this change type"""
        pass
    
    @abstractmethod
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        """Apply the change to the time series and return modified array"""
        pass
    
    def set_position_if_none(self, seq_len: int, existing_objs: List['BaseChange']):
        """Set position if not provided, ensuring minimum length requirement"""
        min_length = self.get_min_length()
        max_start_pos = seq_len - min_length
        if max_start_pos < 0:
            raise KeyError(f"Cannot set position for {self.change_type} with sequence length {seq_len} as it is shorter than minimum length {min_length}.")
        if self.position_start is not None:
            self.position_start = max(0, min(int(self.position_start), max_start_pos))
            return

        if self.position_start is None:
            min_interval = max(seq_len / 8, min_length, 20)

            cnt = 0
            while True:
                self.position_start = random.randint(0, max_start_pos)
                cnt += 1

                if cnt > 1000:
                    raise KeyError(f"Cannot find a valid position for {self.change_type} after 1000 attempts.")
                
                flag = True
                for obj in existing_objs:
                    if self.position_start >= obj.position_start and self.position_start < obj.position_end:
                        flag = False
                        break
                    if abs(self.position_start + min_length - obj.position_start) < min_interval or abs(self.position_start - obj.position_end) < min_interval:
                        flag = False
                        break
                    if self.position_start + min_length > seq_len:
                        flag = False
                        break
    
                if flag:
                    break

    def get_remaining_length(self, seq_len: int) -> int:
        """Get remaining length from current position to end of sequence"""
        return seq_len - self.position_start
    
    def set_amplitude_if_none(self, overall_amplitude: float, base_factor: float = 0.8, variance: float = 2.0):
        """Set amplitude if not provided"""
        if self.amplitude is None:
            self.amplitude = (base_factor + np.abs(random.normalvariate(0.0, variance))) * overall_amplitude


class ShakeChange(BaseChange):
    """Represents a shake/vibration change"""
    
    def get_min_length(self) -> int:
        return 8
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        
        peak_start = self.position_start
        remaining_length = self.get_remaining_length(seq_len)
        peak_length = min(random.randint(8, max(int(seq_len * 0.15), 16)), remaining_length)
        
        func = random.choice([
            lambda: np.random.uniform(-1, 1, peak_length) * self.amplitude / 2,
            # lambda: np.sin(np.linspace(0, 5.0, peak_length)) * self.amplitude / 2
        ])
        
        y[peak_start:peak_start + peak_length] += func()
        self.position_end = peak_start + peak_length
        peak_end = _inclusive_end_point(self.position_end, seq_len)
        self.detail = f"shake with an amplitude of about {self.amplitude:.2f} occurred between point {peak_start} and point {peak_end}"
        
        return y


class SpikeChange(BaseChange):
    """Base class for spike changes"""
    
    def get_min_length(self) -> int:
        return 3
    
    def _generate_spike_detail(self, peak_start: int, peak_end: int, spike_top_idx: int, direction: str, seq_len: int):
        """Generate detail description for spike"""
        public_peak_end = _inclusive_end_point(peak_end, seq_len)
        if direction == "upward":
            self.detail = f"an upward spike with an amplitude of {self.amplitude:.2f} occurred between point {peak_start} and point {public_peak_end}, with the time series value rapidly rising from around {_value_ref(peak_start, seq_len)} to around {_value_ref(spike_top_idx, seq_len)} and then quickly falling back to around {_value_ref(public_peak_end, seq_len)}"
        else:
            self.detail = f"a downward spike with an amplitude of {self.amplitude:.2f} occurred between point {peak_start} and point {public_peak_end}, with the time series value rapidly falling from around {_value_ref(peak_start, seq_len)} to around {_value_ref(spike_top_idx, seq_len)} and then quickly rising back to around {_value_ref(public_peak_end, seq_len)}"


class UpwardSpikeChange(SpikeChange):
    """Represents an upward spike change"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        
        peak_start = self.position_start
        remaining_length = self.get_remaining_length(seq_len)
        spike = generate_spike(self.amplitude, remaining_length)
        actual_length = min(len(spike), remaining_length)
        
        y[peak_start:peak_start + actual_length] += spike[:actual_length]
        spike_top_idx = peak_start + np.argmax(np.abs(spike[:actual_length]))
        self.position_end = peak_start + actual_length
        self._generate_spike_detail(peak_start, self.position_end, spike_top_idx, "upward", seq_len)
        
        return y


class DownwardSpikeChange(SpikeChange):
    """Represents a downward spike change"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        
        peak_start = self.position_start
        remaining_length = self.get_remaining_length(seq_len)
        spike = generate_spike(-self.amplitude, remaining_length)
        actual_length = min(len(spike), remaining_length)
        
        y[peak_start:peak_start + actual_length] += spike[:actual_length]
        spike_top_idx = peak_start + np.argmax(np.abs(spike[:actual_length]))
        self.position_end = peak_start + actual_length
        self._generate_spike_detail(peak_start, self.position_end, spike_top_idx, "downward", seq_len)
        
        return y


class ContinuousSpikeChange(BaseChange):
    """Base class for continuous spike changes"""
    
    def get_min_length(self) -> int:
        return 10
    
    def _apply_continuous_spikes(self, y: np.ndarray, seq_len: int, direction: int):
        """Apply multiple consecutive spikes"""
        peak_region_start = self.position_start
        remaining_length = self.get_remaining_length(seq_len)
        num_peaks = min(random.randint(2, 5), remaining_length // 3)
        
        peaks = []
        spike_top_ids = []
        all_amplitudes = []
        current_pos = peak_region_start
        
        for i in range(num_peaks):
            if remaining_length - (current_pos - self.position_start) < 3:
                break
            
            peak_start = current_pos + random.randint(0, min(3, remaining_length - (current_pos - self.position_start) - 3))
            cur_amplitude = random.uniform(self.amplitude * 0.6, self.amplitude * 1.5)
            all_amplitudes.append(cur_amplitude)
            peaks.append(f"point {peak_start}")
            
            spike = generate_spike(direction * cur_amplitude, self.get_remaining_length(seq_len) - (peak_start - self.position_start))
            actual_length = min(len(spike), seq_len - peak_start)
            y[peak_start:peak_start + actual_length] += spike[:actual_length]
            current_pos = peak_start + actual_length
            spike_top_ids.append(peak_start + np.argmax(np.abs(spike[:actual_length])))
        
        self.position_end = current_pos
        self.amplitude = float(np.mean(all_amplitudes)) if all_amplitudes else self.amplitude
        
        direction_word = "upward" if direction > 0 else "downward"
        action_word = "rising" if direction > 0 else "falling"
        public_end = _inclusive_end_point(current_pos, seq_len)
        
        self.detail = f"at {' and '.join(peaks)}, there were {len(all_amplitudes)} consecutive {direction_word} spikes with amplitudes ranging from {min(all_amplitudes):.2f} to {max(all_amplitudes):.2f}, with the time series value repeatedly {action_word} sharply from around {_value_ref(self.position_start, seq_len)} to around " + ' and '.join(_value_ref(i, seq_len) for i in spike_top_ids) + f", and then quickly falling back to around {_value_ref(public_end, seq_len)}"
        
        return y


class ContinuousUpwardSpikeChange(ContinuousSpikeChange):
    """Represents continuous upward spikes"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        return self._apply_continuous_spikes(y, seq_len, 1)


class ContinuousDownwardSpikeChange(ContinuousSpikeChange):
    """Represents continuous downward spikes"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        return self._apply_continuous_spikes(y, seq_len, -1)


class ConvexChange(BaseChange):
    """Base class for convex changes"""
    
    def get_min_length(self) -> int:
        return 15
    
    def _apply_convex(self, y: np.ndarray, seq_len: int, direction: int):
        """Apply convex change (upward or downward)"""
        convex_start = self.position_start
        remaining_length = self.get_remaining_length(seq_len)
        
        start_length = min(random.randint(1, 4), remaining_length // 3)
        end_length = min(random.randint(1, 4), (remaining_length - start_length) // 2)
        convex_length = min(random.randint(max(int(seq_len * 0.03), 6), max(int(seq_len * 0.2), 16)), 
                          remaining_length - start_length - end_length)
        convex_length = max(1, convex_length)
        
        # Apply changes
        y[convex_start:convex_start + start_length] += generate_ts_change(start_length, direction * self.amplitude)
        y[convex_start + start_length:convex_start + start_length + convex_length] += direction * self.amplitude
        y[convex_start + start_length + convex_length:convex_start + start_length + convex_length + end_length] += generate_ts_change(end_length, -direction * self.amplitude) + direction * self.amplitude
        
        self.position_end = convex_start + start_length + convex_length + end_length
        
        # Add some noise occasionally
        x = np.arange(seq_len)
        if random.random() > 0.7:
            y[convex_start + start_length:convex_start + start_length + convex_length] += np.sin((0.8 + np.abs(random.normalvariate(0.0, 2.0))) * x)[convex_start + start_length:convex_start + start_length + convex_length]
        if random.random() > 0.7:
            y[convex_start + start_length:convex_start + start_length + convex_length] += np.random.uniform(-1.0, 1.0, convex_length) * np.random.uniform(0.1, 0.5) * self.amplitude
        
        direction_word = "upward" if direction > 0 else "downward"
        action_words = ("rises", "falls") if direction > 0 else ("falls", "rises")
        public_end = _inclusive_end_point(self.position_end, seq_len)
        
        self.detail = f"starting from point {convex_start}, the time series value {action_words[0]} from around {_value_ref(convex_start, seq_len)} to around {_value_ref(convex_start + start_length - 1, seq_len)}, forms a {direction_word} convex with an amplitude of about {self.amplitude:.2f}, and then {action_words[1]} back to around {_value_ref(public_end, seq_len)}"
        
        return y


class UpwardConvexChange(ConvexChange):
    """Represents an upward convex change"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        return self._apply_convex(y, seq_len, 1)


class DownwardConvexChange(ConvexChange):
    """Represents a downward convex change"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        return self._apply_convex(y, seq_len, -1)


class SuddenChange(BaseChange):
    """Base class for sudden changes"""
    
    def get_min_length(self) -> int:
        return 2
    
    def _apply_sudden_change(self, y: np.ndarray, seq_len: int, direction: int):
        """Apply sudden increase or decrease"""
        remaining_length = self.get_remaining_length(seq_len)
        drop_length = min(random.randint(1, 10), remaining_length)
        
        y[self.position_start:self.position_start + drop_length] += generate_ts_change(drop_length, direction * self.amplitude)
        y[self.position_start + drop_length:] += direction * self.amplitude
        self.position_end = self.position_start + drop_length
        
        action_word = "increase" if direction > 0 else "decrease"
        movement_word = "rising" if direction > 0 else "falling"
        public_end = _inclusive_end_point(self.position_end, seq_len)
        start_value_point = _previous_raw_point(self.position_start)
        
        self.detail = f"a sudden {action_word} with an amplitude of {self.amplitude:.2f} occurred between point {self.position_start} and point {public_end}, with the time series value {movement_word} from around {_value_ref(start_value_point, seq_len)} to around {_value_ref(public_end, seq_len)}"
        
        # Add recovery with some probability
        if random.random() < 0.5:
            recover_length = min(random.randint(1, 10), seq_len - self.position_start - drop_length)
            if recover_length > 0:
                recover_amplitude = random.uniform(0, self.amplitude / 3)
                y[self.position_start + drop_length:self.position_start + drop_length + recover_length] += generate_ts_change(recover_length, -direction * recover_amplitude)
                y[self.position_start + drop_length + recover_length:] -= direction * recover_amplitude
                
                if ENABLE_DROP_PROMPT:
                    recovery_word = "drop" if direction > 0 else "rise"
                    recovery_movement = "falling" if direction > 0 else "rising"
                    recovery_start = self.position_start + drop_length
                    recovery_end = _inclusive_end_point(recovery_start + recover_length, seq_len)
                    self.detail += f", then a {recovery_word} with an amplitude of {recover_amplitude:.2f} occurred between point {recovery_start} and point {recovery_end}, with the time series value {recovery_movement} back to around {_value_ref(recovery_end, seq_len)}"
        
        return y


class SuddenIncreaseChange(SuddenChange):
    """Represents a sudden increase change"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        return self._apply_sudden_change(y, seq_len, 1)


class SuddenDecreaseChange(SuddenChange):
    """Represents a sudden decrease change"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        return self._apply_sudden_change(y, seq_len, -1)


class TwoPhaseChange(BaseChange):
    """Base class for two-phase changes (e.g., rapid rise followed by slow decline)"""
    
    def get_min_length(self) -> int:
        return 10


class RapidRiseSlowDeclineChange(TwoPhaseChange):
    """Represents rapid rise followed by slow decline"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        
        remaining_length = self.get_remaining_length(seq_len)
        rise_length = min(random.randint(1, 5), remaining_length // 2)
        fall_length = min(random.randint(max(int(seq_len * 0.05), 8), max(int(seq_len * 0.15), 20)), remaining_length - rise_length)
        
        y[self.position_start:self.position_start + rise_length] += generate_ts_change(rise_length, self.amplitude)
        y[self.position_start + rise_length:self.position_start + rise_length + fall_length] += generate_ts_change(fall_length, -self.amplitude) + self.amplitude
        
        self.position_end = self.position_start + rise_length + fall_length
        rise_end = _inclusive_end_point(self.position_start + rise_length, seq_len)
        public_end = _inclusive_end_point(self.position_end, seq_len)
        self.detail = (
            f"a rapid rise with an amplitude of {self.amplitude:.2f} occurred between point {self.position_start} and point {rise_end}, "
            f"with the time series value rising from around {_value_ref(_previous_raw_point(self.position_start), seq_len)} to around {_value_ref(rise_end, seq_len)}, "
            f"followed by a slow decline between point {self.position_start + rise_length} and point {public_end} back to around {_value_ref(public_end, seq_len)}"
        )
        
        return y


class SlowRiseRapidDeclineChange(TwoPhaseChange):
    """Represents slow rise followed by rapid decline"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        
        remaining_length = self.get_remaining_length(seq_len)
        rise_length = min(random.randint(max(int(seq_len * 0.05), 8), max(int(seq_len * 0.15), 20)), remaining_length // 2)
        fall_length = min(random.randint(1, 5), remaining_length - rise_length)
        
        y[self.position_start:self.position_start + rise_length] += generate_ts_change(rise_length, self.amplitude)
        y[self.position_start + rise_length:self.position_start + rise_length + fall_length] += generate_ts_change(fall_length, -self.amplitude) + self.amplitude
        
        self.position_end = self.position_start + rise_length + fall_length
        rise_end = _inclusive_end_point(self.position_start + rise_length, seq_len)
        public_end = _inclusive_end_point(self.position_end, seq_len)
        self.detail = (
            f"starting from point {self.position_start}, the time series value slowly rises, "
            f"reaching a peak at point {rise_end}, followed by a rapid decline between point {self.position_start + rise_length} and point {public_end} back to around {_value_ref(public_end, seq_len)}"
        )
        
        return y


class RapidDeclineSlowRiseChange(TwoPhaseChange):
    """Represents rapid decline followed by slow rise"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        
        remaining_length = self.get_remaining_length(seq_len)
        drop_length = min(random.randint(1, 5), remaining_length // 2)
        rise_length = min(random.randint(max(int(seq_len * 0.05), 8), max(int(seq_len * 0.15), 20)), remaining_length - drop_length)
        
        y[self.position_start:self.position_start + drop_length] += generate_ts_change(drop_length, -self.amplitude)
        y[self.position_start + drop_length:self.position_start + drop_length + rise_length] += generate_ts_change(rise_length, self.amplitude) - self.amplitude
        
        self.position_end = self.position_start + drop_length + rise_length
        drop_end = _inclusive_end_point(self.position_start + drop_length, seq_len)
        public_end = _inclusive_end_point(self.position_end, seq_len)
        self.detail = (
            f"a rapid decline with an amplitude of {self.amplitude:.2f} occurred between point {self.position_start} and point {drop_end}, "
            f"with the time series value falling from around {_value_ref(_previous_raw_point(self.position_start), seq_len)} to around {_value_ref(drop_end, seq_len)}, "
            f"followed by a slow rise between point {self.position_start + drop_length} and point {public_end} back to around {_value_ref(public_end, seq_len)}"
        )
        
        return y


class SlowDeclineRapidRiseChange(TwoPhaseChange):
    """Represents slow decline followed by rapid rise"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        
        remaining_length = self.get_remaining_length(seq_len)
        drop_length = min(random.randint(max(int(seq_len * 0.05), 8), max(int(seq_len * 0.15), 20)), remaining_length // 2)
        rise_length = min(random.randint(1, 5), remaining_length - drop_length)
        
        y[self.position_start:self.position_start + drop_length] += generate_ts_change(drop_length, -self.amplitude)
        y[self.position_start + drop_length:self.position_start + drop_length + rise_length] += generate_ts_change(rise_length, self.amplitude) - self.amplitude
        
        self.position_end = self.position_start + drop_length + rise_length
        drop_end = _inclusive_end_point(self.position_start + drop_length, seq_len)
        public_end = _inclusive_end_point(self.position_end, seq_len)
        self.detail = (
            f"starting from point {self.position_start}, the time series value slowly declines, "
            f"reaching a low point at point {drop_end}, followed by a rapid rise between point {self.position_start + drop_length} and point {public_end} back to around {_value_ref(public_end, seq_len)}"
        )
        
        return y


class SpikeFollowedByChange(BaseChange):
    """Base class for spike followed by another change"""
    
    def get_min_length(self) -> int:
        return 8


class DecreaseAfterUpwardSpikeChange(SpikeFollowedByChange):
    """Represents decrease after upward spike"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        
        remaining_length = self.get_remaining_length(seq_len)
        fall_amplitude = random.uniform(0.1, 0.7) * self.amplitude
        peak_start = self.position_start
        
        spike = generate_spike(self.amplitude, remaining_length)
        peak_length = min(len(spike), remaining_length // 2)
        fall_length = min(random.randint(2, max(int(seq_len * 0.05), 12)), remaining_length - peak_length)
        
        y[peak_start:peak_start + peak_length] += spike[:peak_length]
        spike_top_idx = peak_start + np.argmax(np.abs(spike[:peak_length]))
        y[peak_start + peak_length:peak_start + peak_length + fall_length] += generate_ts_change(fall_length, -fall_amplitude)
        y[peak_start + peak_length + fall_length:] -= fall_amplitude
        
        self.position_end = peak_start + peak_length + fall_length
        peak_end = _inclusive_end_point(peak_start + peak_length, seq_len)
        public_end = _inclusive_end_point(self.position_end, seq_len)
        self.detail = (
            f"an upward spike with an amplitude of {self.amplitude:.2f} occurred between point {peak_start} and point {peak_end}, "
            f"with the time series value rapidly rising from around {_value_ref(_previous_raw_point(peak_start), seq_len)} to around {_value_ref(spike_top_idx, seq_len)} and quickly falling back, "
            f"followed by a further decline between point {peak_start + peak_length} and point {public_end} to around {_value_ref(public_end, seq_len)}"
        )
        
        return y


class IncreaseAfterDownwardSpikeChange(SpikeFollowedByChange):
    """Represents increase after downward spike"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        
        remaining_length = self.get_remaining_length(seq_len)
        rise_amplitude = random.uniform(0.1, 0.7) * self.amplitude
        peak_start = self.position_start
        
        spike = generate_spike(-self.amplitude, remaining_length)
        peak_length = min(len(spike), remaining_length // 2)
        rise_length = min(random.randint(2, max(int(seq_len * 0.05), 12)), remaining_length - peak_length)
        
        y[peak_start:peak_start + peak_length] += spike[:peak_length]
        spike_top_idx = peak_start + np.argmax(np.abs(spike[:peak_length]))
        y[peak_start + peak_length:peak_start + peak_length + rise_length] += generate_ts_change(rise_length, rise_amplitude)
        y[peak_start + peak_length + rise_length:] += rise_amplitude
        
        self.position_end = peak_start + peak_length + rise_length
        peak_end = _inclusive_end_point(peak_start + peak_length, seq_len)
        public_end = _inclusive_end_point(self.position_end, seq_len)
        self.detail = (
            f"a downward spike with an amplitude of {self.amplitude:.2f} occurred between point {peak_start} and point {peak_end}, "
            f"with the time series value rapidly falling from around {_value_ref(peak_start, seq_len)} to around {_value_ref(spike_top_idx, seq_len)} and quickly rising back, "
            f"followed by a further rise between point {peak_start + peak_length} and point {public_end} to around {_value_ref(public_end, seq_len)}"
        )
        
        return y


class IncreaseAfterUpwardSpikeChange(SpikeFollowedByChange):
    """Represents increase after upward spike"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        
        remaining_length = self.get_remaining_length(seq_len)
        rise_amplitude = random.uniform(0.1, 0.7) * self.amplitude
        peak_start = self.position_start
        
        spike = generate_spike(self.amplitude, remaining_length)
        peak_length = min(len(spike), remaining_length // 2)
        rise_length = min(random.randint(2, max(int(seq_len * 0.05), 12)), remaining_length - peak_length)
        
        y[peak_start:peak_start + peak_length] += spike[:peak_length]
        spike_top_idx = peak_start + np.argmax(np.abs(spike[:peak_length]))
        y[peak_start + peak_length:peak_start + peak_length + rise_length] += generate_ts_change(rise_length, rise_amplitude)
        y[peak_start + peak_length + rise_length:] += rise_amplitude
        
        self.position_end = peak_start + peak_length + rise_length
        peak_end = _inclusive_end_point(peak_start + peak_length, seq_len)
        public_end = _inclusive_end_point(self.position_end, seq_len)
        self.detail = (
            f"an upward spike with an amplitude of {self.amplitude:.2f} occurred between point {peak_start} and point {peak_end}, "
            f"with the time series value rapidly rising from around {_value_ref(_previous_raw_point(peak_start), seq_len)} to around {_value_ref(spike_top_idx, seq_len)} and quickly falling back, "
            f"followed by a further rise between point {peak_start + peak_length} and point {public_end} to around {_value_ref(public_end, seq_len)}"
        )
        
        return y


class DecreaseAfterDownwardSpikeChange(SpikeFollowedByChange):
    """Represents decrease after downward spike"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        
        remaining_length = self.get_remaining_length(seq_len)
        fall_amplitude = random.uniform(0.1, 0.7) * self.amplitude
        peak_start = self.position_start
        
        spike = generate_spike(-self.amplitude, remaining_length)
        peak_length = min(len(spike), remaining_length // 2)
        fall_length = min(random.randint(2, max(int(seq_len * 0.05), 12)), remaining_length - peak_length)
        
        y[peak_start:peak_start + peak_length] += spike[:peak_length]
        spike_top_idx = peak_start + np.argmax(np.abs(spike[:peak_length]))
        y[peak_start + peak_length:peak_start + peak_length + fall_length] += generate_ts_change(fall_length, -fall_amplitude)
        y[peak_start + peak_length + fall_length:] -= fall_amplitude
        
        self.position_end = peak_start + peak_length + fall_length
        peak_end = _inclusive_end_point(peak_start + peak_length, seq_len)
        public_end = _inclusive_end_point(self.position_end, seq_len)
        self.detail = (
            f"a downward spike with an amplitude of {self.amplitude:.2f} occurred between point {peak_start} and point {peak_end}, "
            f"with the time series value rapidly falling from around {_value_ref(peak_start, seq_len)} to around {_value_ref(spike_top_idx, seq_len)} and quickly rising back, "
            f"followed by a further decline between point {peak_start + peak_length} and point {public_end} to around {_value_ref(public_end, seq_len)}"
        )
        
        return y


class WideSpikeChange(BaseChange):
    """Base class for wide spike changes"""
    
    def get_min_length(self) -> int:
        return 16


class WideUpwardSpikeChange(WideSpikeChange):
    """Represents a wide upward spike"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        
        remaining_length = self.get_remaining_length(seq_len)
        
        # Define longer rise and fall lengths
        rise_length = min(random.randint(max(int(seq_len * 0.02), 4), max(int(seq_len * 0.08), 8)), remaining_length // 3)
        peak_length = min(random.randint(1, 3), (remaining_length - rise_length) // 2)
        fall_length = min(random.randint(max(int(seq_len * 0.02), 4), max(int(seq_len * 0.08), 8)), remaining_length - rise_length - peak_length)

        # Slow rise
        y[self.position_start:self.position_start + rise_length] += generate_ts_change(rise_length, self.amplitude)
        # Short peak
        y[self.position_start + rise_length:self.position_start + rise_length + peak_length] += self.amplitude
        # Slow decline
        y[self.position_start + rise_length + peak_length:self.position_start + rise_length + peak_length + fall_length] += generate_ts_change(fall_length, -self.amplitude) + self.amplitude
        
        self.position_end = self.position_start + rise_length + peak_length + fall_length
        rise_end = _inclusive_end_point(self.position_start + rise_length, seq_len)
        public_end = _inclusive_end_point(self.position_end, seq_len)
        self.detail = (
            f"a slow rise from around {_value_ref(self.position_start, seq_len)} to around {_value_ref(rise_end, seq_len)} occurred between point {self.position_start} and point {rise_end}, "
            f"forming a short peak with an amplitude of {self.amplitude:.2f}, "
            f"followed by a slow decline between point {self.position_start + rise_length + peak_length} and point {public_end} back to around {_value_ref(public_end, seq_len)}"
        )
        
        return y


class WideDownwardSpikeChange(WideSpikeChange):
    """Represents a wide downward spike"""
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        
        remaining_length = self.get_remaining_length(seq_len)
        
        # Define longer decline and rise lengths
        drop_length = min(random.randint(max(int(seq_len * 0.02), 4), max(int(seq_len * 0.08), 8)), remaining_length // 3)
        peak_length = min(random.randint(1, 3), (remaining_length - drop_length) // 2)
        rise_length = min(random.randint(max(int(seq_len * 0.02), 4), max(int(seq_len * 0.08), 8)), remaining_length - drop_length - peak_length)

        # Slow decline
        y[self.position_start:self.position_start + drop_length] += generate_ts_change(drop_length, -self.amplitude)
        # Short trough
        y[self.position_start + drop_length:self.position_start + drop_length + peak_length] -= self.amplitude
        # Slow rise
        y[self.position_start + drop_length + peak_length:self.position_start + drop_length + peak_length + rise_length] += generate_ts_change(rise_length, self.amplitude) - self.amplitude
        
        self.position_end = self.position_start + drop_length + peak_length + rise_length
        drop_end = _inclusive_end_point(self.position_start + drop_length, seq_len)
        public_end = _inclusive_end_point(self.position_end, seq_len)
        self.detail = (
            f"a slow decline from around {_value_ref(self.position_start, seq_len)} to around {_value_ref(drop_end, seq_len)} occurred between point {self.position_start} and point {drop_end}, "
            f"forming a short trough with an amplitude of {self.amplitude:.2f}, "
            f"followed by a slow rise between point {self.position_start + drop_length + peak_length} and point {public_end} back to around {_value_ref(public_end, seq_len)}"
        )
        
        return y


# Factory class to create appropriate change objects
class ChangeFactory:
    """Factory class to create change objects based on change type"""
    
    _change_classes = {
        "shake": ShakeChange,
        "upward spike": UpwardSpikeChange,
        "downward spike": DownwardSpikeChange,
        "continuous upward spike": ContinuousUpwardSpikeChange,
        "continuous downward spike": ContinuousDownwardSpikeChange,
        "upward convex": UpwardConvexChange,
        "downward convex": DownwardConvexChange,
        "sudden increase": SuddenIncreaseChange,
        "sudden decrease": SuddenDecreaseChange,
        "rapid rise followed by slow decline": RapidRiseSlowDeclineChange,
        "slow rise followed by rapid decline": SlowRiseRapidDeclineChange,
        "rapid decline followed by slow rise": RapidDeclineSlowRiseChange,
        "slow decline followed by rapid rise": SlowDeclineRapidRiseChange,
        "decrease after upward spike": DecreaseAfterUpwardSpikeChange,
        "increase after downward spike": IncreaseAfterDownwardSpikeChange,
        "increase after upward spike": IncreaseAfterUpwardSpikeChange,
        "decrease after downward spike": DecreaseAfterDownwardSpikeChange,
        "wide upward spike": WideUpwardSpikeChange,
        "wide downward spike": WideDownwardSpikeChange,
    }
    
    @classmethod
    def create_change(self, change_type: str, position_start: int = None, amplitude: float = None) -> BaseChange:
        """Create a change object based on change type"""
        if change_type not in self._change_classes:
            raise ValueError(f"Unknown change type: {change_type}")
        
        return self._change_classes[change_type](change_type, position_start, amplitude)
    
    @classmethod
    def get_supported_types(self) -> list:
        """Get list of supported change types"""
        return list(self._change_classes.keys())


def generate_local_chars(attribute_pool, overall_amplitude, seq_len):
    """
    Generate a time series with local characteristics using object-oriented approach.
    
    Args:
        attribute_pool (dict): Pool of attributes containing local characteristics
        overall_amplitude (float): Overall amplitude for scaling
        seq_len (int): Length of the time series
    
    Returns:
        np.ndarray: Modified time series with local changes applied
    """
    y = np.zeros(seq_len)
    existing_objs = []
    
    # Create change objects and set positions
    updated_local = []
    for local_char in attribute_pool["local"]:
        try:
            change_obj = ChangeFactory.create_change(
                local_char["type"], 
                local_char.get("position_start"), 
                local_char.get("amplitude")
            )
            change_obj.set_position_if_none(seq_len, existing_objs)
            existing_objs.append(change_obj)

            # Apply the current change
            y = change_obj.apply_change(y, seq_len, overall_amplitude)
            if change_obj.position_end > seq_len:
                raise ValueError(f"Change exceeds sequence length: {change_obj.position_end} >= {seq_len}. This should never happend! ({change_obj.type=}, {change_obj.position_start=}, {change_obj.amplitude=})")
            local_char.update({
                "position_start": change_obj.position_start,
                "position_end": _inclusive_end_point(change_obj.position_end, seq_len),
                "amplitude": change_obj.amplitude,
                "detail": change_obj.detail
            })
            updated_local.append(local_char)
        except KeyError as e:
            if LOCAL_CHANGE_VERBOSE:
                print(f"Warning ({seq_len=}): {e}. Skipping this change.")
            continue
        except Exception as e:
            if LOCAL_CHANGE_VERBOSE:
                traceback.print_exc()
                print(f"Error ({seq_len=}): {e}. Skipping this change.")
            continue
    
    # Order of position
    updated_local.sort(key=lambda x: x["position_start"])
    attribute_pool["local"] = updated_local
    
    return y
