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

# Note: You have to install `vllm==0.8.5`.
# Note: Usage: `VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ALLOW_INSECURE_SERIALIZATION=1 python3 -m demo.demo_vllm`
# Note: `chatts.vllm.chatts_vllm` has to be imported here first as it will rigister the custom ChatTS module and the multimodal processor.
import chatts.vllm.chatts_vllm

from vllm import LLM, SamplingParams
import numpy as np


# CONFIG
MODEL_PATH = "./ckpt"
ctx_length = 6000


# Load Model with vLLM
language_model = LLM(model=MODEL_PATH, trust_remote_code=True, max_model_len=ctx_length, tensor_parallel_size=1, gpu_memory_utilization=0.95, limit_mm_per_prompt={"timeseries": 50})

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
prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

mm_data = {"timeseries": [ts1.tolist(), ts2.tolist()]}
inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data
}

# TODO: Test batch inference speed
inputs = [inputs] * 100

# Inference
outputs = language_model.generate(inputs, sampling_params=SamplingParams(max_tokens=300))

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
