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

# [Important Note] This script is still under development and may not work as expected.

import chatts.vllm.chatts_vllm
from vllm import LLM, SamplingParams
from chatts.utils.llm_utils import LLMClient
import torch
import os
import json
from loguru import logger
import numpy as np


# CONFIG
EXP = 'chatts_dataset_a'
MODEL_PATH = os.path.abspath('ckpt')
DATASET = f'./evaluation/dataset/dataset_a.json'
WORKDIR = os.path.abspath('./')
NUM_GPUS = 8
NUM_GPUS_PER_PROCESS = 2
MAX_MM_PER_PROMPT = 50  # Maximum number of time series per prompt


# Sampling parameters
sampling_params = SamplingParams(
    max_tokens=512,
    temperature=0.2
)

def answer_question_list(question_list, ts_list):
    answer_dict = {}
    llm_client = LLMClient(model_path=MODEL_PATH, engine='vllm-ts', num_gpus=NUM_GPUS, gpus_per_model=NUM_GPUS_PER_PROCESS)
    llm_client.wait_for_ready()
    answer_list = llm_client.llm_batch_generate(question_list, ts_list, sampling_params=sampling_params)
    llm_client.kill()

    for idx, answer in enumerate(answer_list):
        answer_dict[idx] = {
            "response": answer
        }

    return answer_dict


if __name__ == '__main__':
    dataset = json.load(open(DATASET))

    generated_answer = []
    exp_dir = os.path.join(WORKDIR, f"exp/{EXP}")
    logger.info(f"Experiment directory: {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)

    # Generation
    logger.info("Start Generation...")
    question_list = []
    ts_list = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        question_list.append(sample['question'])
        ts_list.append([np.array(item) for item in sample['timeseries']])
    
    # Generate answers
    answer = answer_question_list(question_list, ts_list)
    
    # Prepare results
    for idx, ans in answer.items():
        generated_answer.append({
            'idx': idx,
            'question_text': question_list[idx],
            'response': ans['response']
        })

    # Save results
    output_file = os.path.join(exp_dir, f"generated_answer.json")
    json.dump(generated_answer, open(output_file, "wt"), ensure_ascii=False, indent=4)
    logger.info(f"Results saved to {output_file}")
