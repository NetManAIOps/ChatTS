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

"""
Example code to demonstrate how to use the ChatTS vLLM API to analyze time series data. Make sure that the vLLM server is running before executing this code.
"""

import openai
import numpy as np

# Load Time Series Data
SEQ_LEN_1 = 256
SEQ_LEN_2 = 1000

x1 = np.arange(SEQ_LEN_1)
x2 = np.arange(SEQ_LEN_2)

# TS1: A simple sin signal with a sudden decrease
ts1 = np.sin(x1 / 10) * 5.0
ts1[103:] -= 10.0

# TS2: A increasing trend with a upward spike
ts2 = x2 * 0.01
ts2[100] += 10.0
prompt = f"I have 2 time series. TS1 is of length {SEQ_LEN_1}: <ts><ts/>; TS2 if of length {SEQ_LEN_2}: <ts><ts/>. Please analyze the local changes in these time series first and then conclude if these time series showing local changes near the same time?"
prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n"

# Create ts_list
ts_list = [ts1.tolist(), ts2.tolist()]

# Send message
client = openai.OpenAI(base_url="http://127.0.0.1:12345/v1", api_key="test")

response = client.chat.completions.create(
    model="chatts",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ] + [{"timeseries": ts} for ts in ts_list],
        }
    ]
)

print(response.choices[0].message.content)
