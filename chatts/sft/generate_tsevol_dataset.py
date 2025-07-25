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
        2. Run `python3 -m chatts.sft.generate_tsevol_dataset`.
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
import yaml
from typing import *
from json_repair import repair_json
from chatts.sft.utils.evol_prompt import EvolPrompt
import copy


# Config
MODEL_PATH = yaml.safe_load(open("config/datagen_config.yaml"))["local_llm_path"]
num_gpus = yaml.safe_load(open("config/datagen_config.yaml"))["num_gpus"]
gpu_per_model = yaml.safe_load(open("config/datagen_config.yaml"))["gpu_per_model"]
SEQ_LEN = yaml.safe_load(open("config/datagen_config.yaml"))["seq_len"]  # Set to None to enable random sequence length selection
DATA_OUTPUT_DIR = yaml.safe_load(open("config/datagen_config.yaml"))["data_output_dir"]
ENCODING_METHOD = yaml.safe_load(open("config/datagen_config.yaml"))['encoding_method']

ctx_length = 4096
batch_size = 32
ENGINE = 'vllm'
MULTIPROCESS = True
DFS_K = 3
TOTAL_CNT = yaml.safe_load(open("config/datagen_config.yaml"))["num_data_tsevol"]
NUM_DATA_LLM = yaml.safe_load(open("config/datagen_config.yaml"))["num_data_llm_qa"]

INPUT_FILES = [
    (f'{DATA_OUTPUT_DIR}/llm_qa_{NUM_DATA_LLM}_{ENCODING_METHOD}.jsonl', f'{DATA_OUTPUT_DIR}/evol_labels/llm_qa_{NUM_DATA_LLM}_{ENCODING_METHOD}.json'),
    (f'{DATA_OUTPUT_DIR}/mts_local_llm_{SEQ_LEN}_{NUM_DATA_LLM}_{ENCODING_METHOD}.jsonl', f'{DATA_OUTPUT_DIR}/evol_labels/mts_local_llm_{SEQ_LEN}_{NUM_DATA_LLM}_{ENCODING_METHOD}.json'),
    (f'{DATA_OUTPUT_DIR}/mts_shape_llm_{SEQ_LEN}_{NUM_DATA_LLM}_{ENCODING_METHOD}.jsonl', f'{DATA_OUTPUT_DIR}/evol_labels/mts_shape_llm_{SEQ_LEN}_{NUM_DATA_LLM}_{ENCODING_METHOD}.json'),
    (f'{DATA_OUTPUT_DIR}/uts_llm_{SEQ_LEN}_{NUM_DATA_LLM}_{ENCODING_METHOD}.jsonl', f'{DATA_OUTPUT_DIR}/evol_labels/uts_llm_{SEQ_LEN}_{NUM_DATA_LLM}_{ENCODING_METHOD}.json')
]
OUTPUT_FILE = f'{DATA_OUTPUT_DIR}/evol_{TOTAL_CNT}_{ENCODING_METHOD}.jsonl'


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
                        cur_qa = json.loads(repair_json(cur_items[0]))
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


def evol_instruct():
    # Load files
    input_list: List[EvolPrompt] = []

    print("Loading files...")
    for input_file, label_file in INPUT_FILES:
        print(f"Loading {input_file} and {label_file}...")
        qa_dataset = [json.loads(line.rstrip()) for line in tqdm(open(input_file))]
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
