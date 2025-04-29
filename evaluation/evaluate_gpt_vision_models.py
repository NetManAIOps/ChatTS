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
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
import json
import os
import time
from tqdm import tqdm
import traceback
from typing import *
import io

from evaluation.evaluate_qa import evaluate_batch_qa
from multiprocessing import Pool

# CONFIG
MODEL = 'gpt-4o-mini'
EXP = 'gpt-4o-mini-vision-dataset-a'
DATASET = 'evaluation/dataset/dataset_a.json'
OPENAI_API_KEY = "[Your API Key]"
OPENAI_BASE_URL = "[Your Base URL]"


def generate_image_from_timeseries(case_idx: int, timeseries: np.ndarray, cols: List[str]) -> str:
    if len(timeseries) == 1:
        plt.figure(figsize=(6, 2))
        plt.plot(timeseries[0], linewidth=2, color='blue')
        plt.grid(True)
        plt.title(cols[0])
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(len(timeseries), 1, figsize=(6, len(timeseries) * 1.3))
        for i in range(len(timeseries)):
            ax[i].plot(timeseries[i])
            ax[i].set_title(cols[i])
        fig.tight_layout()
    os.makedirs(f'exp/{EXP}/fig', exist_ok=True)
    plt.savefig(f"exp/{EXP}/fig/{case_idx}.jpg", format='JPG')
    plt.close()

    img_b64_str = base64.b64encode(open(f"exp/{EXP}/fig/{case_idx}.jpg", 'rb').read()).decode('utf-8')
    return img_b64_str

def ask_gpt4o_with_timeseries(case_idx: int, timeseries: np.ndarray, cols: List[str], question: str) -> str:
    openai.api_key = OPENAI_API_KEY
    openai.base_url = OPENAI_BASE_URL

    client = openai.OpenAI(api_key=openai.api_key, base_url=openai.base_url)

    img_b64_str = generate_image_from_timeseries(case_idx, timeseries, cols)
    img_type = "image/jpeg"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"}}
            ]
        }
    ]

    timeout_cnt = 0
    while True:
        if timeout_cnt > 10:
            logger.error("Too many timeout!")
            raise RuntimeError("Too many timeout!")
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                timeout=20
            )
            # Quick check
            res_head = response.choices[0].message.content.lower()[:20]
            if any([i in res_head for i in ["i'm sorry", "unable to", "sorry", "there was an error", "i cannot", "i can't"]]):
                logger.error("API Error: " + res_head)
                logger.error("API timeout, trying again...")
                time.sleep(5.0)
                timeout_cnt += 1
                continue
            break
        except Exception as err:
            logger.error(err)
            logger.error("API timeout, trying again...")
            timeout_cnt += 1

    answer = response.choices[0].message.content
    total_tokens = response.usage.prompt_tokens
    return answer, total_tokens

def process_sample(args):
    sample, idx = args
    try:
        timeseries = sample['timeseries']
        cols = sample['cols']
        question_text = sample['question']
        label = sample['answer']

        answer, total_tokens = ask_gpt4o_with_timeseries(idx, timeseries, cols, question_text)

        return {
            'idx': idx,
            'question_text': question_text,
            'response': answer,
            'num_tokens': total_tokens
        }
    except Exception as err:
        logger.error(err)
        traceback.print_exc()
        return None


if __name__ == '__main__':
    dataset = json.load(open(DATASET))

    generated_answer = []
    os.makedirs(f'exp/{EXP}', exist_ok=True)
    if os.path.exists(f"exp/{EXP}/generated_answer.json"):
        generated_answer = json.load(open(f"exp/{EXP}/generated_answer.json"))
    generated_idx = set([i['idx'] for i in generated_answer])

    # Generation
    logger.info("Start Generation...")
    idx_to_generate = [i for i in range(len(dataset)) if i not in generated_idx]
    with Pool(processes=32) as pool:
        results = list(tqdm(pool.imap(process_sample, [(dataset[idx], idx) for idx in idx_to_generate]), total=len(idx_to_generate)))

    # Filter out None results and update generated_answer
    generated_answer.extend([res for res in results if res is not None])
    json.dump(generated_answer, open(f"exp/{EXP}/generated_answer.json", "wt"), ensure_ascii=False, indent=4)

    # Evaluation
    evaluate_batch_qa(dataset, generated_answer, EXP, num_workers=16)
