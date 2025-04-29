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

import openai
import numpy as np
import os
import json
from statsmodels.tsa.seasonal import STL
from scipy.signal import find_peaks, correlate
from evaluation.evaluate_qa import match_metric_name
from typing import *
from sktime.classification.kernel_based import RocketClassifier
from evaluation.train_rocket_tsc import change_type_list
from adtk.detector import AutoregressionAD
import pandas as pd


def parse_latest_plugin_call(text: str):
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    k = text.rfind('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
    if 0 <= i < j < k:
        plugin_name = text[i + len('\nAction:'):j].strip()
        plugin_args = text[j + len('\nAction Input:'):k].strip()
        return plugin_name, plugin_args
    return '', ''

def parse_json(plugin_args: str):
    plugin_args = plugin_args.replace('```json', '').replace('```', '')
    return json.loads(plugin_args)

def stl_tool(timeseries: np.ndarray) -> str:
    def find_period(signal, min_period=10, amplitude_threshold=0.1):
        acf = np.correlate(signal, signal, 'full')[-len(signal):]
        inflection = np.diff(np.sign(np.diff(acf)))
        peaks = (inflection < 0).nonzero()[0] + 1

        if len(peaks) == 0:
            return None

        max_acf_value = acf[peaks].max()
        valid_peaks = [p for p in peaks if acf[p] >= amplitude_threshold * max_acf_value]

        valid_peaks = [p for p in valid_peaks if p >= min_period]

        if len(valid_peaks) == 0:
            return None

        return valid_peaks[acf[valid_peaks].argmax()]

    period = find_period(timeseries)
    if period is None:
        stl_period = 20
    else:
        stl_period = period

    stl = STL(timeseries, period=stl_period, robust=True)
    result = stl.fit()

    # Generate prompt
    prompt = f"The STL decomposition of the time series is as follows: "
    prompt += f"From the trend perspective, the difference between the rightmost and leftmost points is {round(float(result.trend[-1] - result.trend[0]), 2)}, the start point of the trend is around {round(float(result.trend[0]), 2)}."
    prompt += f"The 10 equal parts of the trend are as follows: {[round(float(result.trend[i]), 2) for i in range(0, len(result.trend), len(result.trend) // 10)]}. "
    prompt += "Please analyze the trend characteristics of the time series considering the overall changes in the time series. "
    if period is not None:
        prompt += f"From the seasonal perspective, the period is around {period}, the maximum seasonal value is {round(float(np.max(result.seasonal)), 2)}, the minimum seasonal value is {round(float(np.min(result.seasonal)), 2)}. "
    else:
        prompt += f"From the seasonal perspective, no significant seasonal fluctuation is found. "
    prompt += f"From the noise perspective, the standard deviation is {round(float(np.std(result.resid)), 3)}. "

    return prompt

def anomaly_detection_tool(timeseries: np.ndarray) -> str:
    # Do anomaly detection with adtk
    autoregression_ad = AutoregressionAD()
    s = pd.Series(timeseries, index=pd.date_range('2021-01-01', periods=len(timeseries), freq='s'))
    anomalies = autoregression_ad.fit_detect(s)

    # Find start points of each anomaly blocks
    anomalous_points = []
    for i in range(len(anomalies.values)):
        if np.isnan(anomalies.values[i]):
            continue
        if anomalies.values[i] == True and (i == 0 or anomalies.values[i - 1] == False):
            anomalous_points.append(i)

    # Generate prompt
    if len(anomalous_points) > 0:
        prompt = f"The anomalous points are {anomalous_points}."
    else:
        prompt = "No anomaly is detected."
    return prompt

def classification_tool(timeseries: np.ndarray) -> str:
    timeseries = np.array(timeseries)
    clf = RocketClassifier.load_from_path('result/rocket.zip')
    assert len(timeseries) == 64, f"Time series length should be 64 for classification, but got {len(timeseries)}"
    
    print(f"[CLASSIFICATION] {timeseries.shape=}")
    result = clf.predict(timeseries[np.newaxis, np.newaxis, :])[0]
    prompt = "The classification result is " + change_type_list[result]
    
    return prompt

def correlation_tool(timeseries: list, src_idx: int, cols: List[str]) -> str:
    min_ts_len = min([len(i) for i in timeseries])
    timeseries = [np.array(i)[:min_ts_len] for i in timeseries]
    timeseries = np.stack(timeseries, axis=0)

    prompt = f'The pearson correlation between {cols[src_idx]} and other metrics are as follows: '

    for target_idx in range(len(cols)):
        if target_idx == src_idx:
            continue
        correlation = np.corrcoef(timeseries[src_idx], timeseries[target_idx])[0, 1]
        prompt += f"The correlation between {cols[src_idx]} and {cols[target_idx]} is {correlation:.3f}. "
    
    return prompt

def anomaly_detection_multi_tool(timeseries: list, cols: List[str]) -> str:
    results = []
    for i, col in enumerate(cols):
        ts = np.array(timeseries[i])
        s = pd.Series(ts, index=pd.date_range('2021-01-01', periods=len(ts), freq='s'))
        detector = AutoregressionAD()
        anomalies = detector.fit_detect(s)
        
        anomalous_points = []
        for idx in range(1, len(anomalies)):
            if anomalies[idx] and not anomalies[idx-1]:
                anomalous_points.append(idx)

        # Generate prompt
        if len(anomalous_points) > 0:
            prompt = f"- For {col}, the anomalous points are {anomalous_points}."
        else:
            prompt = f"- For {col}, no anomaly is detected."

        results.append(prompt)

    return "The detection results are: \n" + "\n".join(results)

def classification_multi_tool(timeseries: list, position: int, cols: List[str]) -> str:
    """Batch process multivariate time series classification with optimized model loading"""
    # Load model once at the beginning
    clf = RocketClassifier.load_from_path('result/rocket.zip')
    
    # Prepare batch input
    batch_windows = []
    valid_indices = []
    results = [""] * len(cols)
    
    # Preprocess all windows first
    for i, (col, ts) in enumerate(zip(cols, timeseries)):
        ts = np.array(ts)
        start_pos = max(0, position - 32)
        end_pos = start_pos + 64
        
        if end_pos > len(ts):
            end_pos = len(ts)
            start_pos = max(0, end_pos - 64)
        
        window = ts[start_pos:end_pos]
        
        if len(window) != 64:
            results[i] = f"- {col}: Invalid window position"
            continue
            
        batch_windows.append(window[np.newaxis, np.newaxis, :])  # Add channel and sample dimensions
        valid_indices.append(i)

    # Batch predict for all valid windows
    if batch_windows:
        X = np.vstack(batch_windows)
        predictions = clf.predict(X)
        
        # Map predictions to results
        for idx, pred in zip(valid_indices, predictions):
            results[idx] = f"- {cols[idx]}: {change_type_list[pred]}"

    return "Classification results:\n" + "\n".join(results)

def trend_correlation_tool(timeseries: list, src_idx: int, cols: List[str]) -> str:
    def find_period(signal, min_period=10, amplitude_threshold=0.1):
        acf = np.correlate(signal, signal, 'full')[-len(signal):]
        inflection = np.diff(np.sign(np.diff(acf)))
        peaks = (inflection < 0).nonzero()[0] + 1

        if len(peaks) == 0:
            return None

        max_acf_value = acf[peaks].max()
        valid_peaks = [p for p in peaks if acf[p] >= amplitude_threshold * max_acf_value]

        valid_peaks = [p for p in valid_peaks if p >= min_period]

        if len(valid_peaks) == 0:
            return None

        return valid_peaks[acf[valid_peaks].argmax()]

    min_len = min(len(ts) for ts in timeseries)
    aligned_series = [np.array(ts)[:min_len] for ts in timeseries]
    
    trends = []
    for ts in aligned_series:
        period = find_period(ts)
        if period is None:
            stl_period = 20
        else:
            stl_period = period
        stl = STL(ts, period=stl_period, robust=True)
        res = stl.fit()
        trends.append(res.trend)
    
    prompt = f"Trend correlations for {cols[src_idx]}:\n"
    src_trend = trends[src_idx]

    high_corr_cols = []

    for i, trend in enumerate(trends):
        corr = np.corrcoef(src_trend, trend)[0,1]
        prompt += f"- {cols[i]}: {corr:.3f}\n"

        if corr > 0.7:
            high_corr_cols.append(cols[i])

    prompt += f"Therefore, time series with high correlation (>0.7) are: " + ', '.join(high_corr_cols)

    return prompt

def fluctuation_correlation_tool(timeseries: list, src_idx: int, cols: List[str]) -> str:
    """
    Calculate fluctuation correlation based on anomaly position matching with 5% sequence length tolerance.
    """
    # Align time series length
    seq_len = min(len(ts) for ts in timeseries)
    aligned_series = [np.array(ts)[:seq_len] for ts in timeseries]
    tolerance = int(seq_len * 0.05)  # 5% tolerance window
    
    # Get anomaly positions for all series
    anomaly_positions = []
    for ts in aligned_series:
        s = pd.Series(ts, index=pd.date_range('2021-01-01', periods=seq_len, freq='s'))
        detector = AutoregressionAD()
        anomalies = detector.fit_detect(s)
        anomaly_positions.append(np.where(anomalies)[0].tolist())
    
    # Find matching anomalies between src and target series
    src_anomalies = set(anomaly_positions[src_idx])
    matches = {}
    
    for target_idx, target_anomalies in enumerate(anomaly_positions):
        if target_idx == src_idx:
            continue
            
        matched = 0
        target_anomaly_set = set(target_anomalies)
        
        # Check for matches within tolerance window
        for pos in src_anomalies:
            # Create tolerance range [pos-tolerance, pos+tolerance]
            lower = max(0, pos - tolerance)
            upper = min(seq_len-1, pos + tolerance)
            
            # Check if any target anomaly falls in this range
            if any(lower <= t_pos <= upper for t_pos in target_anomaly_set):
                matched += 1
                
        # Calculate match ratio
        total_src = len(src_anomalies) if len(src_anomalies) > 0 else 1
        match_ratio = matched / total_src
        matches[cols[target_idx]] = match_ratio
    
    # Generate report
    prompt = f"Fluctuation correlation for {cols[src_idx]}:\n"
    for col, ratio in sorted(matches.items(), key=lambda x: -x[1]):
        if ratio > 0:
            prompt += f"- {col}: has matching fluctuations, may be correlated with {cols[src_idx]}\n"
        else:
            prompt += f"- {col}: no matching fluctuations found, may be not correlated with {cols[src_idx]}\n"
    
    return prompt

def col_idx(name: str, cols: List[str]) -> int:
    for i, col in enumerate(cols):
        if match_metric_name(col, name):
            return i
    raise RuntimeError(f"Metric: {name} not found")

def call_plugin(plugin_name: str, plugin_args: str, timeseries: list, cols: List[str]) -> str:
    try:
        if plugin_name == 'datapoint_value':
            name = parse_json(plugin_args)['name']
            position = parse_json(plugin_args)['position']
            return f"The value of datapoint {position} in {name} is {float(timeseries[col_idx(name, cols)][position]):.2f}"
        elif plugin_name == 'datarange_value':
            name = parse_json(plugin_args)['name']
            cur_idx = col_idx(name, cols)
            position_start = parse_json(plugin_args)['position_start']
            position_end = parse_json(plugin_args)['position_end']
            return f"The value between datapoint {position_start} and datapoint {position_end} in {name} is {[round(float(timeseries[cur_idx][pos]), 3) for pos in range(position_start, position_end)]}"
        elif plugin_name == 'stl_decomposition':
            name = parse_json(plugin_args)['name']
            return stl_tool(np.array(timeseries[col_idx(name, cols)]))
        elif plugin_name == 'anomaly_detection':
            name = parse_json(plugin_args)['name']
            return anomaly_detection_tool(np.array(timeseries[col_idx(name, cols)]))
        elif plugin_name == 'anomaly_detection_multi':
            return anomaly_detection_multi_tool(timeseries, cols)
        elif plugin_name == 'classification':
            name = parse_json(plugin_args)['name']
            position = parse_json(plugin_args)['position']
            cur_idx = col_idx(name, cols)
            # Set start and end position
            start_pos = min(max(0, position - 32), len(timeseries[cur_idx]) - 64)
            end_pos = start_pos + 64
            # print(f"{start_pos=}, {end_pos=}")
            return classification_tool(np.array(timeseries[cur_idx])[start_pos:end_pos])
        elif plugin_name == 'classification_multi':
            position = parse_json(plugin_args)['position']
            return classification_multi_tool(timeseries, position, cols)
        elif plugin_name == 'similarity':
            name = parse_json(plugin_args)['name']
            return correlation_tool(timeseries, col_idx(name, cols), cols)
        elif plugin_name == 'trend_correlation':
            name = parse_json(plugin_args)['name']
            return trend_correlation_tool(timeseries, col_idx(name, cols), cols)
        elif plugin_name == 'fluctuation_correlation':
            name = parse_json(plugin_args)['name']
            return fluctuation_correlation_tool(timeseries, col_idx(name, cols), cols)
        else:
            return f"plugin: {plugin_name} not found! You have to strictly format your Action (just put tool name here) and Action Input (just put the json format input here) to use the tools. If you want to output the final answer, strictly it format to: Final Answer: the final answer to the original input question."
    except Exception as err:
        return f"Error when calling {plugin_name}: {str(err)}"


def answer_question_react(question, timeseries, cols, model, client):
    TOOLS = [
        {
            'name_for_human':
                'Datapoint Value',
            'name_for_model':
                'datapoint_value',
            'description_for_model':
                'Output the value of a time series datapoint according to the input position.',
            'parameters': [{
                'name': 'position',
                'description': 'The position of the point to query (0 to seq_len - 1).',
                'required': True,
                'schema': {
                    'type': 'int'
                },
            }, {
                'name': 'name',
                'description': 'The name of the time series to query.',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            }],
        },
        {
            'name_for_human':
                'Datarange Value',
            'name_for_model':
                'datarange_value',
            'description_for_model':
                'Output the values (in list format) of the datapoints between the range [position_start, position_end).',
            'parameters': [{
                'name': 'position_start',
                'description': 'The start position of the data range',
                'required': True,
                'schema': {
                    'type': 'int'
                },
            }, {
                'name': 'position_end',
                'description': 'The end position of the data range',
                'required': True,
                'schema': {
                    'type': 'int'
                },
            }, {
                'name': 'name',
                'description': 'The name of the time series to query.',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            }],
        },
        {
            'name_for_human': 'STL Decomposition',
            'name_for_model': 'stl_decomposition',
            'description_for_model': 'Output the trend values, seasonal (period and max/min values), and residual (std) values after stl decomposition.',
            'parameters': [{
                'name': 'name',
                'description': 'The name of the time series to do the stl decomposition.',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            }],
        },
        {
            'name_for_human': 'Anomaly Detection',
            'name_for_model': 'anomaly_detection',
            'description_for_model': 'Output the anomaly detection result (which points are anomalous) of the given time series. When determin the local fluctuations, always use this tool first to get the accurate position of local fluctuations. ',
            'parameters': [{
                'name': 'name',
                'description': 'The name of the time series to do the anomaly detection.',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            }],
        },
        {
            'name_for_human': 'Anomaly Detection for Multivariate Timeseries',
            'name_for_model': 'anomaly_detection_multi',
            'description_for_model': 'Output the anomaly detection result (which points are anomalous) for all the time series (MTS version of anomaly_detection)',
            'parameters': [],
        },
        {
            'name_for_human': 'Local Fluctuation Classification',
            'name_for_model': 'classification',
            'description_for_model': 'Output the classification result (including type and amplitude) of the local fluctuations in a given time series window (the window size if 64).',
            'parameters': [{
                'name': 'name',
                'description': 'The name of the time series to do the anomaly detection.',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            }, {
                'name': 'position',
                'description': 'The middle point of the window for time series classification (the window size is 64).',
                'required': True,
                'schema': {
                    'type': 'int'
                },
            }],
        },
        {
            'name_for_human': 'Local Fluctuation Classification for Multivariate Timeseries',
            'name_for_model': 'classification_multi',
            'description_for_model': 'Output the classification result (including type and amplitude) of the local fluctuations in a given time series window (the window size if 64) for all the time series (MTS version of classification).',
            'parameters': [{
                'name': 'position',
                'description': 'The middle point of the window for time series classification (the window size is 64).',
                'required': True,
                'schema': {
                    'type': 'int'
                },
            }],
        },
        {
            'name_for_human': 'Trend Correlation',
            'name_for_model': 'trend_correlation',
            'description_for_model': 'Output the correlated time series of the given time series in terms of trend similarity.',
            'parameters': [{
                'name': 'name',
                'description': 'The name of the given time series.',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            }],
        },
        {
            'name_for_human': 'Fluctuation Correlation',
            'name_for_model': 'fluctuation_correlation',
            'description_for_model': 'Output the correlated time series of the given time series in terms of the similarity of local fluctuations.',
            'parameters': [{
                'name': 'name',
                'description': 'The name of the given time series.',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            }],
        }
    ]

    TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. The {name_for_human} API is useful for: {description_for_model} Parameters: {parameters}. Format the arguments as a JSON object."""

    REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

    {tool_descs}

    You can use the tools provided to access the timeseries. When outputing Final Answer, please strictly format your answer according to the requirement and use English. Your output should be in plain text **without** MarkDown. Note that if you find the final answer, you should strictly format it as "Final Answer: the final answer to the original input question".
    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can be repeated zero or more times, as needed but no more than 10 times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {query}
    """

    import json
    tool_descs = []
    tool_names = []
    for info in TOOLS:
        tool_descs.append(
            TOOL_DESC.format(
                name_for_model=info['name_for_model'],
                name_for_human=info['name_for_human'],
                description_for_model=info['description_for_model'],
                parameters=json.dumps(
                    info['parameters'], ensure_ascii=False),
            )
        )
        tool_names.append(info['name_for_model'])
    tool_descs = '\n\n'.join(tool_descs)
    tool_names = ','.join(tool_names)

    # Reformat question
    question = question.replace(': <ts><ts/>', '. ')
    question += f' The names of the give time series are (you can use those for query): {cols}'
    prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names, query=question)
    # prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
    total_prompt_tokens = 0
    cot_length = 0

    while True:
        timeout_cnt = 0
        while True:
            if timeout_cnt > 10:
                print("Too many timeout!")
                raise RuntimeError("Too many timeout!")
            try:
                out_put = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                            ]
                        }
                    ],
                    stop="\nObservation"
                )
                break
            except Exception as err:
                print(err)
                print("API timeout, trying again...")
                timeout_cnt += 1

        total_prompt_tokens += out_put.usage.prompt_tokens
        out_put = out_put.choices[0].message.content
        out_put = out_put.replace('**', '')
        # print("out_put:", out_put)
        import re
        if re.search('final answer:', out_put, re.IGNORECASE):
            final_answer = re.search('final answer:(.*)', out_put, re.DOTALL | re.IGNORECASE).group(1).strip()
            return final_answer, total_prompt_tokens, prompt
        plugin_name, plugin_args = parse_latest_plugin_call(out_put)
        print("[PLUGIN] Plugin Name: ", plugin_name, "Plugin Args: ", plugin_args)
        plugin_res = call_plugin(plugin_name, plugin_args, timeseries, cols)

        prompt += "\n"
        prompt += out_put
        Observation_res = "\nObservation:" + json.dumps(plugin_res, ensure_ascii=False)
        prompt += Observation_res
        print("==========================")
        print(prompt)

        cot_length += 1
        if cot_length > 20:
            return "ReAct failed: too much interaction", total_prompt_tokens, prompt

if __name__ == '__main__':
    # idx = 1
    # dataset = json.load(open(f"result/evaluation_100_1012.json"))
    # sample = dataset[idx]
    # timeseries = np.array(sample['timeseries'])
    # cols = sample['cols']
    # question_text = sample['question']
    # label = sample['answer']

    # print("===ReAct Begin===")
    # output, total_tokens = answer_question_react(question_text, timeseries, cols, 'gpt-4o-mini')
    # print("===ReAct End===")
    # print(output)
    # print(f"{total_tokens=}")

    # print(call_plugin('classification', '{"name": "' + cols[0] + '", "position": 100}', timeseries, cols))

    timeseries1 = [0.01 * i for i in range(256)]
    timeseries2 = [0.02 * i for i in range(300)]

    timeseries1[100] = 100.0
    timeseries2[105] = -50.0

    timeseries = [timeseries1, timeseries2]
    cols = ['ts1', 'ts2']
    # plugin = 'classification_multi'
    # plugin_args = '{"position": 100}'
    # plugin = 'anomaly_detection_multi'
    # plugin_args = '{}'
    # plugin = 'trend_correlation'
    # plugin_args = '{"name": "ts1"}'
    plugin = 'fluctuation_correlation'
    plugin_args = '{"name": "ts1"}'
    print(call_plugin(plugin, plugin_args, timeseries, cols))
