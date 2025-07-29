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

# Note: You have to install `vllm==0.8.5`.
# Note: This is a beta version, which may change in the future.
# Note: `chatts.vllm.chatts_vllm` has to be imported here first as it will register the custom ChatTS module and the multimodal processor.
# Note: Usage: `python3 -m chatts.utils.inference_tsmllm_vllm`

import asyncio
import time
import re
import json
import numpy as np
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from loguru import logger

# CONFIG
MODEL_PATH = "[CHATTS_MODEL_PATH]"  # Replace with your model path
ENABLE_TS = True
MAX_MODEL_LEN = 12000
TENSOR_PARALLEL = 2

if ENABLE_TS:
    import chatts.vllm.chatts_vllm
    engine_args = AsyncEngineArgs(model=MODEL_PATH, enforce_eager=True, gpu_memory_utilization=0.95, max_model_len=MAX_MODEL_LEN, tensor_parallel_size=TENSOR_PARALLEL,  limit_mm_per_prompt={"timeseries": 15}, trust_remote_code=True)
else:
    engine_args = AsyncEngineArgs(model=MODEL_PATH, enforce_eager=True, gpu_memory_utilization=0.95, max_model_len=MAX_MODEL_LEN, tensor_parallel_size=TENSOR_PARALLEL, trust_remote_code=True)
model = AsyncLLMEngine.from_engine_args(engine_args)

def extract_and_remove_ts(s):
    pattern = r'(<ts>)(.*?)(<ts/>)'
    logger.warning(f"[extract_and_remove_ts] {s}")
    matches = re.findall(pattern, s)
    extracted_lists = [json.loads(match[1]) for match in matches]
    modified_s = re.sub(pattern, r'\1\3', s)
    
    if len(extracted_lists) == 0:
        extracted_lists = None
    return modified_s, extracted_lists

async def generate_streaming(prompt):
    results_generator = model.generate(prompt, SamplingParams(max_tokens=MAX_MODEL_LEN), request_id=time.monotonic())
    previous_text = ""
    async for request_output in results_generator:
        text = request_output.outputs[0].text
        print(text[len(previous_text):], end="", flush=True)
        previous_text = text
    return text

async def main():
    history = []
    all_timeseries = []
    while True:
        print("=" * 80)

        # Get user input
        question = input("Question: ")
        if question == 'exit':
            break
        elif question == 'clear':
            history = []
            all_timeseries = []
            print("History cleared.")
            continue
        elif question == 'file':
            with open('vllm_input.txt', 'rt') as f:
                question = f.read().rstrip()
                logger.warning(f"Loaded input from file.")

        input_timeseries = None
        if ENABLE_TS:
            question, timeseries = extract_and_remove_ts(question)
            if timeseries is not None and len(timeseries) > 0:
                logger.warning(f"[Timeseries]: {len(timeseries)}")
                all_timeseries.extend([np.array(ts) for ts in timeseries])
            if len(all_timeseries) > 0:
                input_timeseries = all_timeseries.copy()

        # Apply chat template
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        for q, a in history:
            prompt += f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

        inputs = {
            "prompt": prompt
        }
        if input_timeseries is not None:
            inputs['multi_modal_data'] = {
                "timeseries": input_timeseries
            }

        print("=" * 80)
        response = await generate_streaming(inputs)
        print(f"\n[Answer]: {response}")
        history.append((question, response))

if __name__ == "__main__":
    asyncio.run(main())
