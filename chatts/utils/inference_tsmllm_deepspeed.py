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

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import deepspeed
from transformers.integrations import HfDeepSpeedConfig
import json
from loguru import logger
import numpy as np
from chatts.utils.encoding_utils import timeseries_encoding


# CONFIG
EXP = 'chatts_dataset_a'
MODEL_PATH = os.path.abspath('ckpt')
DATASET = f'./evaluation/dataset/dataset_a.json'
WORKDIR = os.path.abspath('./')

# Don't change the below config
BATCH_SIZE = 1
ENCODING_METHOD = 'sp'


# Initialize
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend='nccl')
deepspeed.init_distributed()

ds_config = {
    "fp16": {"enabled": True},
    "bf16": {"enabled": False},
    "zero_optimization": {
        "stage": 0,
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "steps_per_print": 2000,
    "train_batch_size": world_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}
dscfg = HfDeepSpeedConfig(ds_config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
ds_engine = deepspeed.initialize(model=model, config=ds_config)[0]
model_engine = ds_engine.module
model_engine.eval()

def answer_question_list(question_list, ts_list, batch_size=BATCH_SIZE):
    answer_dict = {}
    total_cnt = len([i for i in range(len(question_list)) if i % world_size == local_rank])
    
    local_indices = [i for i in range(len(question_list)) if i % world_size == local_rank]
    
    for batch_start in range(0, len(local_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(local_indices))
        batch_indices = local_indices[batch_start:batch_end]
        
        batch_question_list = [question_list[i] for i in batch_indices]
        batch_ts_list = []
        for i in batch_indices:
            if ts_list[i] is None:
                continue
            for ts in ts_list[i]:
                batch_ts_list.append(np.array([ts]))
        ts_num_tokens = []
        for i in batch_indices:
            if ts_list[i] is None:
                ts_num_tokens.append(0)
            else:
                ts_num_tokens.append(sum([len(t) for t in ts_list[i]]) // model.config.ts['patch_size'])

        print(f"[worker {local_rank}] {batch_question_list=}")
        inputs = tokenizer(batch_question_list, return_tensors="pt", padding=True, truncation=True).to(device=local_rank)
        
        if len(batch_ts_list) > 0:
            max_length = max(arr.shape[1] for arr in batch_ts_list)
            padded_time_series_attributes = [
                np.pad(arr, ((0, 0), (0, max_length - arr.shape[1]), (0, 0)), mode='constant', constant_values=0)
                for arr in batch_ts_list
            ]
            concatenated_time_series = np.concatenate(padded_time_series_attributes, axis=0)
            ts_tensors = torch.tensor(concatenated_time_series, dtype=torch.float16, device=local_rank)
            print(f"[worker {local_rank}] {ts_tensors.shape=}")
        else:
            ts_tensors = None
        print(f"[worker {local_rank}] {inputs['input_ids'].shape=}, {inputs['attention_mask'].shape=}")

        with torch.no_grad():
            outputs = model_engine.generate(
                inputs['input_ids'], 
                attention_mask=inputs['attention_mask'], 
                timeseries=ts_tensors, 
                synced_gpus=False, 
                max_length=inputs['input_ids'].shape[-1] + 512,
                temperature=0.2
            )

        for i, idx in enumerate(batch_indices):
            output = outputs[i]
            input_len = inputs['attention_mask'][i].sum().item()
            generated_output = output[input_len:]
            text_out = tokenizer.decode(generated_output, skip_special_tokens=True)
            print(f"[worker {local_rank}] {idx=}, {text_out=}")
            answer_dict[idx] = {
                'response': text_out,
                'num_tokens': int(ts_num_tokens[i] + input_len)
            }

        print(f"[worker {local_rank}] {len(answer_dict)}/{total_cnt} finished.")
    
    return answer_dict


if __name__ == '__main__':
    dataset = json.load(open(DATASET))

    generated_answer = []
    exp_dir = os.path.join(WORKDIR, f"exp/{EXP}")
    print(f"Experiment directory: {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)

    # Generation
    logger.info("Start Generation...")
    question_list = []
    ts_list = []
    for idx in range(len(dataset)):
        sample = dataset[idx]

        # Scaler
        prompt_list = sample['question'].split('<ts><ts/>')
        prompt = prompt_list[0]
        for ts in range(len(sample['timeseries'])):
            scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(sample['timeseries'][ts], ENCODING_METHOD)
            sample['timeseries'][ts] = scaled_timeseries
            prompt += cur_ts_prompt + prompt_list[ts + 1]
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n"

        question_list.append(prompt)
        ts_list.append(sample['timeseries'])
    answer = answer_question_list(question_list, ts_list)
    for idx, ans in answer.items():
        generated_answer.append({
            'idx': idx,
            'question_text': question_list[idx],
            'response': ans['response'],
            'num_tokens': ans['num_tokens']
        })

    # Save label
    json.dump(generated_answer, open(os.path.join(WORKDIR, f"exp/{EXP}/generated_answer_{world_size}_{local_rank}.json"), "wt"), ensure_ascii=False, indent=4)
