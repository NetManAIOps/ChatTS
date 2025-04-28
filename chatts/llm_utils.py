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

import multiprocessing
from tqdm import tqdm
import json
import os
from loguru import logger
import re
import numpy as np
import random
import time
import traceback
from typing import *


# Config
MODEL_PATH = json.load(open("config/datagen_config.json"))["local_llm_path"]
ctx_length = 5000
num_gpus = 8
gpu_per_model = 1
batch_size = 32
ENGINE = 'vllm'


def worker_llama_cpp(input_queue, output_list, gpu_id, batch_size, model_path=MODEL_PATH):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, # Uncomment to use GPU acceleration
            n_ctx=ctx_length, # Uncomment to increase the context window,
            chat_format='qwen'
        )
        
        while not input_queue.empty():
            batch_prompts = []
            batch_args = []
            for _ in range(batch_size):
                if not input_queue.empty():
                    cur_items = input_queue.get()
                    batch_prompts.append(cur_items[0])
                    batch_args.append(cur_items[1:])
                else:
                    break
            
            if batch_prompts:
                batch_generates = []
                for prompt in batch_prompts:
                    logger.debug(f"[INPUT] {prompt}")
                    cur_generate = llm(
                        prompt, 
                        stop='<|im_end|>',
                        temperature=0.1,
                        top_k=10,
                        max_tokens=ctx_length
                    )
                    logger.debug(f"[OUTPUT] {cur_generate['choices'][0]['text']}")
                    batch_generates.append(cur_generate['choices'][0]['text'])
                output_list.extend(list(zip(batch_generates, *zip(*batch_args))))
    except Exception as err:
        logger.error(f"[worker {gpu_id}] {err}")
        time.sleep(5)

def worker_vllm(input_queue, output_list, gpu_id, batch_size, model_path=MODEL_PATH):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        from vllm import LLM, SamplingParams
        sampling_params = SamplingParams(temperature=0.5, top_p=0.95, max_tokens=ctx_length, stop_token_ids=[151643, 151645], stop=['<|endoftext|>', '<|im_end|>'])
        llm = LLM(model=model_path, trust_remote_code=True, max_model_len=ctx_length, tensor_parallel_size=gpu_per_model, gpu_memory_utilization=0.95)
        
        while not input_queue.empty():
            batch_prompts = []
            batch_args = []
            for _ in range(batch_size):
                if not input_queue.empty():
                    cur_items = input_queue.get()
                    batch_prompts.append(cur_items[0])
                    batch_args.append(cur_items[1:])
                else:
                    break
            
            if batch_prompts:
                answers = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
                answers = [i.outputs[0].text for i in answers]
                output_list.extend(list(zip(answers, *zip(*batch_args))))
    except Exception as err:
        logger.error(f"[worker {gpu_id}] {err}")
        traceback.print_exc()
        time.sleep(5)

def worker_lmdeploy(input_queue, output_list, gpu_id, batch_size, model_path=MODEL_PATH):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        from lmdeploy import TurbomindEngineConfig, pipeline, GenerationConfig
        pipe = pipeline(
            model_path=model_path,
            backend_config=TurbomindEngineConfig(
                tp=gpu_per_model,
                session_len=ctx_length * 2
            )
        )
        generation_config = GenerationConfig(top_p=0.8, top_k=20, temperature=0.5, max_new_tokens=ctx_length, stop_words=['<|endoftext|>', '<|im_end|>'])
        
        while not input_queue.empty():
            batch_prompts = []
            batch_args = []
            for _ in range(batch_size):
                if not input_queue.empty():
                    cur_items = input_queue.get()
                    batch_prompts.append(cur_items[0])
                    batch_args.append(cur_items[1:])
                else:
                    break
            
            if batch_prompts:
                answers = pipe(batch_prompts, gen_config=generation_config)
                answers = [i.text for i in answers]
                output_list.extend(list(zip(answers, *zip(*batch_args))))
    except Exception as err:
        logger.error(f"[worker {gpu_id}] {err}")
        time.sleep(5)

def llm_batch_generate(batch_prompts: List[str], use_chat_template=True, num_gpus=num_gpus, batch_size=batch_size, model_path=MODEL_PATH, engine=ENGINE):
    manager = multiprocessing.Manager()
    input_queue = manager.Queue()
    output_list = manager.list()
    total_cnt = 0
    for i, item in enumerate(batch_prompts):
        if use_chat_template:
            prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{item}<|im_end|><|im_start|>assistant\n"
        else:
            prompt = item
        input_queue.put((prompt, i, item))
        total_cnt += 1

    answer_dict = {}

    processes = []
    for gpu_id in range(0, num_gpus, gpu_per_model):
        gpu_id_str = ",".join(map(str, range(gpu_id, gpu_id + gpu_per_model)))
        if engine == 'llama':
            p = multiprocessing.Process(target=worker_llama_cpp, args=(input_queue, output_list, gpu_id_str, batch_size, model_path))
        elif engine == 'vllm':
            p = multiprocessing.Process(target=worker_vllm, args=(input_queue, output_list, gpu_id_str, batch_size, model_path))
        elif engine == 'lmdeploy':
            p = multiprocessing.Process(target=worker_lmdeploy, args=(input_queue, output_list, gpu_id_str, batch_size, model_path))
        else:
            raise NotImplementedError(f"Unrecognized inference engine: {engine}")
        processes.append(p)
        p.start()

    with tqdm(total=total_cnt) as pbar:
        previous_len = 0
        while any(p.is_alive() for p in processes):
            current_len = len(output_list)
            pbar.update(current_len - previous_len)

            # Append to answer
            for line in output_list[previous_len:current_len]:
                answer_dict[line[1]] = line[0]

            previous_len = current_len

    for p in processes:
        p.join()
    
    answer_list = []
    for i in range(len(batch_prompts)):
        if i not in answer_dict:
            answer_list.append(None)
        else:
            answer_list.append(answer_dict[i])

    return answer_list


def try_fix_json(json_string, special_words=['question', 'answer', 'success', 'reference', ',', ':', '\n', '}', '{']):
    # Fix unmatched quots
    quotes_indices = [m.start() for m in re.finditer(r'"', json_string)]
    fixed_json = list(json_string)
    for i in quotes_indices:
        for special in special_words:
            if json_string[i + 1:].startswith(special) or json_string[:i].endswith(': '):
                break
        else:
            fixed_json[i] = r'\"'

    # Fix special words
    result = ''.join(fixed_json)
    result = result.replace('True', 'true').replace('False', 'false')

    # Fix delimeters
    result = re.sub(r'"\s*\n\s*"', '",\n"', result)

    return result

def escape_newlines_in_quotes(json_string):
    matches = list(re.finditer(r'(?<!\\)"([^"\\]*(?:\\.[^"\\]*)*)"', json_string, re.DOTALL))
    fixed_json = []
    last_end = 0
    
    for match in matches:
        start, end = match.span()
        text_between_quotes = json_string[start:end]
        escaped_text = text_between_quotes.replace('\n', '\\n')
        fixed_json.append(json_string[last_end:start])
        fixed_json.append(escaped_text)
        last_end = end
    
    fixed_json.append(json_string[last_end:])
    
    return ''.join(fixed_json)

def parse_llm_json(json_string, special_words=['question', 'answer', 'success', 'reference', ',', ':', '\n', '}', '{']):
    json_string = json_string.replace('```json', '').replace('```', '')
    try:
        json.loads(json_string)
    except Exception as err:
        json_string = try_fix_json(json_string, special_words)
        json_string = escape_newlines_in_quotes(json_string)
    
    return json.loads(json_string)


if __name__ == '__main__':
    # Load questions
    batch_questions = [
        'How are you?',
        'Who are you?',
        '1+1=?'
    ]
    batch_answers = llm_batch_generate(batch_questions)
    print(batch_answers)
