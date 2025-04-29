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

"""
    Main function for time series evol-instruct.
    Usage:
        1. Set MODEL_PATH to the path of the model and set num_gpus, gpu_per_model accordingly.
        2. Run `python3 -m chatts.evol.evol_instruct_gen`.
        3. The output will be saved to the file specified in OUTPUT_FILE.
"""

import multiprocessing
from tqdm import tqdm
import json
import os
from loguru import logger
import re
import numpy as np
import random
import time
from typing import *
from chatts.evol.prompt import EvolPrompt
from chatts.generate_template_qa import univariate_seed_qa, multivariate_seed_qa
import copy


# Config
MODEL_PATH = json.load(open("config/datagen_config.json"))["local_llm_path"]
ctx_length = 4096
num_gpus = 8
gpu_per_model = 1
batch_size = 32
ENGINE = 'vllm'
MULTIPROCESS = True
DFS_K = 3
ENCODING_METHOD = 'sp'
TOTAL_CNT = 100

INPUT_FILES = [
    (f'data/template_qa_1000_{ENCODING_METHOD}.jsonl', f'data/labels/template_qa_1000_{ENCODING_METHOD}.json'),
    (f'data/llm_qa_1000_{ENCODING_METHOD}.jsonl', f'data/labels/llm_qa_1000_{ENCODING_METHOD}.json')
]
OUTPUT_BASE_DIR = json.load(open("config/datagen_config.json"))["data_output_dir"]
OUTPUT_FILE = f'{OUTPUT_BASE_DIR}/evol_{TOTAL_CNT}_{ENCODING_METHOD}.jsonl'


def worker_vllm(input_queue, validation_queue, input_response, validation_response, gpu_id, batch_size, model_path=MODEL_PATH):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        from vllm import LLM, SamplingParams
        sampling_params = SamplingParams(temperature=0.5, top_p=0.95, max_tokens=ctx_length, stop_token_ids=[151643, 151645], stop=['<|endoftext|>', '<|im_end|>'])
        llm = LLM(model=model_path, trust_remote_code=True, max_model_len=ctx_length, tensor_parallel_size=gpu_per_model, gpu_memory_utilization=0.96)
        
        while True:
            batch_prompts = []
            batch_args = []
            batch_flag = []

            for _ in range(batch_size):
                if not validation_queue.empty():
                    cur_items = validation_queue.get()
                    batch_prompts.append(cur_items[0])
                    batch_args.append(cur_items[1:])
                    batch_flag.append('validation')
                elif not input_queue.empty():
                    cur_items = input_queue.get()
                    batch_prompts.append(cur_items[0])
                    batch_args.append(cur_items[1:])
                    batch_flag.append('input')
                else:
                    break
            
            if batch_prompts:
                answers = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
                answers = [i.outputs[0].text for i in answers]

                for i in range(len(answers)):
                    if batch_flag[i] == 'input':
                        input_response.put((answers[i], *batch_args[i]))
                    else:
                        validation_response.append((answers[i], *batch_args[i]))
            else:
                time.sleep(1.0)
    except Exception as err:
        logger.error(f"[worker {gpu_id}] {err}")
        time.sleep(5)

def worker_dryrun(input_queue, validation_queue, input_response, validation_response, gpu_id, batch_size, model_path=MODEL_PATH):
    try:
        while True:
            batch_prompts = []
            batch_args = []
            batch_flag = []

            for _ in range(batch_size):
                if not validation_queue.empty():
                    cur_items = validation_queue.get()
                    batch_prompts.append(cur_items[0])
                    batch_args.append(cur_items[1:])
                    batch_flag.append('validation')
                elif not input_queue.empty():
                    cur_items = input_queue.get()
                    batch_prompts.append(cur_items[0])
                    batch_args.append(cur_items[1:])
                    batch_flag.append('input')
                else:
                    break
            
            if batch_prompts:
                for i in range(len(batch_prompts)):
                    time.sleep(0.05)
                    print('==================================================================================')
                    print(f"[INPUT] {batch_prompts[i]}")
                    if batch_flag[i] == 'input':
                        input_response.put((json.dumps({
                            'question': 'This is a test question.',
                            'answer': 'This is a test answer.'
                        }), *batch_args[i]))
                    else:
                        validation_response.append(('valid', *batch_args[i]))
            else:
                time.sleep(1.0)
    except Exception as err:
        logger.error(f"[worker {gpu_id}] {err}")
        time.sleep(5)

def llm_batch_generate(seed_prompts: List[EvolPrompt], use_chat_template=True, num_gpus=num_gpus, batch_size=batch_size, model_path=MODEL_PATH, engine=ENGINE):
    manager = multiprocessing.Manager()
    input_queue = manager.Queue()
    input_response = manager.Queue()
    validation_queue = manager.Queue()
    output_list = manager.list()
    for i, item in enumerate(seed_prompts):
        for _ in range(DFS_K):
            cur_item = copy.deepcopy(item)
            cur_item.evol()
            cur_prompt = cur_item.generate_prompt()
            if use_chat_template:
                prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{cur_prompt}<|im_end|><|im_start|>assistant\n"
            else:
                prompt = cur_prompt
            input_queue.put((prompt, cur_item))

    if MULTIPROCESS:
        processes: List[multiprocessing.Process] = []
        for gpu_id in range(0, num_gpus, gpu_per_model):
            gpu_id_str = ",".join(map(str, range(gpu_id, gpu_id + gpu_per_model)))
            if engine == 'vllm':
                p = multiprocessing.Process(target=worker_vllm, args=(input_queue, validation_queue, input_response, output_list, gpu_id_str, batch_size, model_path))
            elif engine == 'dryrun':
                p = multiprocessing.Process(target=worker_dryrun, args=(input_queue, validation_queue, input_response, output_list, gpu_id_str, batch_size, model_path))
            else:
                raise NotImplementedError(f"Unrecognized inference engine: {engine}")
            processes.append(p)
            p.start()

        os.makedirs('result', exist_ok=True)
        with tqdm(total=TOTAL_CNT) as pbar_input, tqdm(total=TOTAL_CNT) as pbar_validation, open(OUTPUT_FILE, 'wt') as fo:
            previous_output_len = 0
            parse_failed = 0
            validation_failed = 0
            success_cnt = 0
            while success_cnt < TOTAL_CNT:
                # Append to validation
                while not input_response.empty():
                    cur_items = input_response.get()
                    try:
                        cur_qa = parse_llm_json(cur_items[0])
                        cur_seed_prompt: EvolPrompt = copy.deepcopy(cur_items[-1])
                        cur_validation_prompt = cur_seed_prompt.generate_comparison_prompt(cur_qa['question'], cur_qa['answer'])
                        cur_seed_prompt.push(cur_qa['question'], cur_qa['answer'])

                        if use_chat_template:
                            cur_validation_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{cur_validation_prompt}<|im_end|><|im_start|>assistant\n"
                        validation_queue.put((cur_validation_prompt, cur_seed_prompt))
                        pbar_input.update()
                    except Exception as err:
                        parse_failed += 1
                        continue

                current_output_len = len(output_list)
                for line in output_list[previous_output_len:current_output_len]:
                    # Check validation result
                    if 'valid' in line[0].lower() and 'invalid' not in line[0].lower():
                        pbar_validation.update()
                        fo.write(json.dumps(line[-1].to_dataset()) + '\n')
                        success_cnt += 1

                        # Push to input queue
                        for _ in range(DFS_K):
                            item: EvolPrompt = copy.deepcopy(line[-1])
                            item.evol()
                            cur_prompt = item.generate_prompt()
                            if use_chat_template:
                                prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{cur_prompt}<|im_end|><|im_start|>assistant\n"
                            else:
                                prompt = cur_prompt
                            input_queue.put((prompt, item))
                    else:
                        validation_failed += 1
                previous_output_len = current_output_len

                pbar_input.set_description(f'[EVOL]fail={parse_failed}')
                pbar_validation.set_description(f'[VALID]fail={validation_failed}')

        for p in processes:
            p.terminate()
            p.kill()
    else:
        pass

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
    

def evol_instruct():
    # Load files
    input_list: List[EvolPrompt] = []

    print("Loading seed QA...")
    for input_file, label_file in tqdm(INPUT_FILES, desc='Loading files'):
        qa_dataset = [json.loads(line.rstrip()) for line in open(input_file)]
        labels = json.load(open(label_file))

        for data, label in zip(qa_dataset, labels):
            input_list.append(EvolPrompt(
                ts_idx=label['ts_idx'],
                seed_q=label['question'],
                seed_a=data['output'],
                seed_fields=label['fields'],
                instruction=label['instruction'],
                timeseries=np.array(data['timeseries']),
                attribute_pool=label['attribute_pool'],
                corr_pool=label['corr_pool'],
                metrics=label['metrics']
            ))

    # Randomly shuffle input_list
    random.shuffle(input_list)

    print(f"{len(input_list)} seed QAs loaded from file.")

    # Run llm inference
    llm_batch_generate(input_list)

    print(f"-------------------------------------------")
    print(f"Finished! File saved to {OUTPUT_FILE}.")

if __name__ == '__main__':
    evol_instruct()
