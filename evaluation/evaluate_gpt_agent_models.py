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
import base64
import multiprocessing
import matplotlib.pyplot as plt
from sktime.classification.kernel_based import RocketClassifier
from loguru import logger
import numpy as np
import json
import os
from tqdm import tqdm
import traceback
from typing import *
import io
plt.rcParams["font.sans-serif"] = ["Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

from evaluation.evaluate_qa import evaluate_batch_qa
from evaluation.react_agent import answer_question_react

# CONFIG
MODEL = 'gpt-4o-mini'
EXP = 'gpt-4o-mini-agent-dataset-a'
DATASET = 'evaluation/dataset/dataset_a.json'
OPENAI_API_KEY = "[Your API Key]"
OPENAI_BASE_URL = "[Your Base URL]"
NUM_WORKERS = 32

def worker_generation(input_queue, output_list):
    while not input_queue.empty():
        idx, sample = input_queue.get()
        try:
            timeseries = sample['timeseries']
            question_text = sample['question']
            label = sample['answer']
            cols = sample['cols']

            # CONFIG
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            openai.api_key = OPENAI_API_KEY
            openai.base_url = OPENAI_BASE_URL
            client = openai.OpenAI(api_key=openai.api_key, base_url=openai.base_url)

            answer, num_tokens, cot_prompt = answer_question_react(question_text, timeseries, cols, MODEL, client)
            output_list.append({
                'idx': idx,
                'question_text': question_text,
                'response': answer,
                'num_tokens': num_tokens,
                'cot_prompt': cot_prompt
            })
        except Exception as err:
            logger.error(err)
            traceback.print_exc()


if __name__ == '__main__':
    dataset = json.load(open(DATASET))

    generated_answer = []
    os.makedirs(f'exp/{EXP}', exist_ok=True)
    if os.path.exists(f"exp/{EXP}/generated_answer.json"):
        generated_answer = json.load(open(f"exp/{EXP}/generated_answer.json"))
    generated_idx = set([i['idx'] for i in generated_answer])

    # Generation
    logger.info("Start Generation...")
    manager = multiprocessing.Manager()
    input_queue = manager.Queue()
    output_list = manager.list()

    total_cnt = 0
    processes = []
    for idx in range(len(dataset)):
        if idx in generated_idx:
            continue
        input_queue.put((idx, dataset[idx]))
        total_cnt += 1
    for _ in range(NUM_WORKERS):
        p = multiprocessing.Process(target=worker_generation, args=(input_queue, output_list))
        processes.append(p)
        p.start()

    with tqdm(total=total_cnt) as pbar:
        previous_len = 0
        while any(p.is_alive() for p in processes):
            current_len = len(output_list)
            pbar.update(current_len - previous_len)

            for output in output_list[previous_len:current_len]:
                generated_answer.append(output)
                json.dump(generated_answer, open(f"exp/{EXP}/generated_answer.json", "wt"), ensure_ascii=False, indent=4)

            previous_len = current_len

    for p in processes:
        p.join()

    # Evaluation
    evaluate_batch_qa(dataset, generated_answer, EXP, num_workers=16)
