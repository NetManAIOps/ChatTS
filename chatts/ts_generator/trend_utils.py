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

import math
import random
import numpy as np
from scipy.interpolate import make_interp_spline, PchipInterpolator


def generate_random_points(seq_len):
    """
    Generates random key points for the curve based on the sequence length.

    Parameters:
        seq_len (int): The total number of points on the x-axis (from 0 to seq_len-1).

    Returns:
        points (list of tuples): The list of generated key points as (x, y) tuples.
        curve_type (str): The type of curve used ('Bezier' or 'Straight Line').
    """
    min_distance = math.ceil(seq_len / 8)
    
    num_turning_points = random.randint(0, 3)
    
    total_key_points = 2 + num_turning_points
    
    total_min_distance = (total_key_points - 1) * min_distance
    
    total_distance = seq_len - 1
    
    extra_distance = total_distance - total_min_distance
    
    while extra_distance < 0 and num_turning_points > 0:
        num_turning_points -= 1
        total_key_points = 2 + num_turning_points
        total_min_distance = (total_key_points - 1) * min_distance
        extra_distance = total_distance - total_min_distance
    
    if extra_distance < 0:
        raise ValueError("seq_len is too small")
    
    gaps = [min_distance] * (total_key_points - 1)
    
    for _ in range(extra_distance):
        idx = random.randint(0, total_key_points - 2)
        gaps[idx] += 1
    
    key_x = [0]
    for gap in gaps:
        key_x.append(key_x[-1] + gap)
    
    y_positions = np.random.uniform(-1, 1, total_key_points)
    
    points = list(zip(key_x, y_positions))
    
    if random.random() < 0.99:
        curve_type = "Bezier"
    else:
        curve_type = "Straight Line"
    
    return points, curve_type

def generate_trend_curve(seq_len, points):
    """
    Generates the curve based on the key points.

    Parameters:
        seq_len (int): The total number of points on the x-axis (from 0 to seq_len-1).
        points (list of tuples): The list of generated key points as (x, y) tuples.

    Returns:
        curve_x (np.ndarray): The x-coordinates of the generated curve (0 to seq_len-1).
        curve_y (np.ndarray): The y-coordinates of the generated curve.
        curve_type (str): The type of curve used ('Bezier' or 'Straight Line').
    """
    # Extract x and y from points
    key_x = [point[0] for point in points]
    key_y = [point[1] for point in points]
    
    # Decide whether to use Bezier curves or straight lines
    curve_x = np.arange(seq_len)
    if random.random() < 0.99:
        curve_type = "Bezier"
        interpolator = PchipInterpolator(key_x, key_y)
        curve_y = interpolator(curve_x)
    else:
        curve_type = "Straight Line"
        curve_y = np.interp(np.arange(seq_len), key_x, key_y)
    
    return curve_x, curve_y, curve_type

def generate_trend_prompt(points):
    """
    Generates an English prompt describing the trend between each pair of adjacent points.
    If consecutive trends are the same, merges them into a single description with a note
    about some variation in the trend.

    Parameters:
        points (list of tuples): The list of key points as (x, y) tuples.

    Returns:
        prompt (str): The generated English prompt describing the trends.
    """
    if not points or len(points) < 2:
        return "Insufficient points to determine trends."

    # Extract y-values
    y_values = [y for _, y in points]

    # Calculate the curve range
    curve_range = max(y_values) - min(y_values)

    if curve_range == 0:
        # All y-values are the same
        curve_range = 1  # To avoid division by zero

    # Determine trends between each pair of points
    trends = []
    for i in range(len(points) - 1):
        _, y_left = points[i]
        _, y_right = points[i + 1]
        delta_y = y_right - y_left

        if delta_y > 0.1 * curve_range:
            trend = "increasing"
        elif delta_y < -0.1 * curve_range:
            trend = "decreasing"
        else:
            trend = "stable"

        trends.append(trend)

    # Merge consecutive same trends
    merged_trends = []
    current_trend = trends[0]
    start_idx = 0

    for i in range(1, len(trends)):
        if trends[i] != current_trend:
            merged_trends.append((current_trend, start_idx, i))
            current_trend = trends[i]
            start_idx = i
    # Append the last trend
    merged_trends.append((current_trend, start_idx, len(trends)))

    # Generate the prompt
    prompt_segments = []
    for trend, start, end in merged_trends:
        # Determine the starting and ending points of the merged segment
        point_start = points[start]
        point_end = points[end]

        # Determine the article based on the trend
        if trend == "increasing":
            article = "an increasing trend"
        elif trend == "decreasing":
            article = "a decreasing trend"
        else:
            article = "a stable trend"

        # If the segment spans multiple trends of the same type, mention variation
        if end - start > 1:
            variation_note = "with some variation in slope"
        else:
            variation_note = ""

        # Construct the sentence
        if variation_note:
            sentence = (f"From point {point_start[0]} "
                        f"to point {point_end[0]}, "
                        f"there is {article} {variation_note}.")
        else:
            sentence = (f"From point {point_start[0]} "
                        f"to point {point_end[0]}, "
                        f"there is {article}.")

        prompt_segments.append(sentence)

    # Combine all segments into a single prompt
    prompt = " ".join(prompt_segments)

    return prompt

def generate_trend_list(points, seq_len):
    """
    Generates an English trend list describing the trend between each pair of adjacent points.
    If consecutive trends are the same, merges them into a single description with a note
    about some variation in the trend.

    Parameters:
        points (list of tuples): The list of key points as (x, y) tuples.

    Returns:
        trend_list: [(increase/decrease, start_point, end_point)]
    """
    if not points or len(points) < 2:
        return "Insufficient points to determine trends."

    # Extract y-values
    y_values = [y for _, y in points]

    # Calculate the curve range
    curve_range = max(y_values) - min(y_values)

    if curve_range == 0:
        # All y-values are the same
        curve_range = 1  # To avoid division by zero

    # Determine trends between each pair of points
    trends = []
    for i in range(len(points) - 1):
        _, y_left = points[i]
        _, y_right = points[i + 1]
        delta_y = y_right - y_left

        if delta_y > 0.1 * curve_range:
            trend = "increase"
        elif delta_y < -0.1 * curve_range:
            trend = "decrease"
        else:
            trend = "steady"

        trends.append(trend)

    # Merge consecutive same trends
    merged_trends = []
    current_trend = trends[0]
    start_idx = 0

    for i in range(1, len(trends)):
        if trends[i] != current_trend:
            merged_trends.append((current_trend, points[start_idx][0], points[i][0]))
            current_trend = trends[i]
            start_idx = i
    # Append the last trend
    merged_trends.append((current_trend, points[start_idx][0], seq_len - 1))

    return merged_trends
