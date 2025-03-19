<div align="center">

# ChatTS: Understanding, Chat, Reasoning about Time Series with TS-MLLM

[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/bytedance-research/ChatTS-14B)
[![DOI](https://zenodo.org/badge/DOI/Datasets.svg)](https://doi.org/10.5281/zenodo.14349206)
[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2412.03104&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2412.03104)
![Visits Badge](https://badges.pufler.dev/visits/NetManAIOps/ChatTS)
![Stars](https://img.shields.io/github/stars/NetManAIOps/ChatTS)
![Forks](https://img.shields.io/github/forks/NetManAIOps/ChatTS)

</div>

`ChatTS` is a Time Series Multimodal LLM focuses on **Understanding and Reasoning** about time series, much like what vision/video/audio-MLLMs do.
This repo provides code, datasets and model for `ChatTS`: [ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning](https://arxiv.org/pdf/2412.03104).

Here is an example of a ChatTS application, which allows users to interact with a LLM to understand and reason about time series data:
![Chat](figures/chat_example.png)

`ChatTS` features native support for multi-variate time series data with any length and range of values. With `ChatTS`, you can easily understand and reason about both the **shape** features and **value** features in the time series. `ChatTS` can also be integrated into existing LLM pipelines for more time series-related applications, leveraging existing inference frameworks such as `vLLMs`. Check out the [Case Studies](#case-studies) section for more examples.

<!-- We also provide the evaluation datasets collected by us. You can download the evaluation datasets from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14349206.svg)](https://doi.org/10.5281/zenodo.14349206). The training scripts can be found in [ChatTS-Training](https://github.com/xiezhe-24/ChatTS-Training).
A fine-tuned `ChatTS` model (based on a modified version of QWen2.5-14B-Instruct) have been open-sourced at [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/bytedance-research/ChatTS-14B). You can download and try it! -->

## News
- **2025/01/01**: We have released a new version of `ChatTS` model, with enhanced CoT and question answering capability. Check [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/bytedance-research/ChatTS-14B) for more information.
- **2024/12/30**: A experimental version of `vLLM` support for ChatTS is available! Check [demo_vllm.py](demo_vllm.py) for more information. (**Note**: This version is still under development and may not be stable.) We have also updated the ChatTS model implementation, which supports `kv_cache` and `AutoProcessor` now.

## Introduction
This repository provides several toolkits for generating synthetic data with the approaches introduced in `ChatTS`, as well as the evaluation code and evaluation datasets for reproduction:
- Toolkits for generating synthetic time series data and the corresponding attribues: `chatts/ts_generator.py`.
- Example code for generating a training dataset with pre-defined templates: `chatts/generate_template_qa.py`, which can be further used as seed QAs for TSEvol.
- Example code for generating a training dataset with LLMs: `chatts/generate_llm_qa`, which can be further used as seed QAs for TSEvol.
- Code implementation for `TSEvol` with the generated seed QAs: `chatts/evol/evol_instruct.py`.
- Code implementation for evaluation: `evaluation/`.
- Simple demos for inference: `demo_hf.ipynb` and `demo_vllm.py`.
- A trained `ChatTS` model (fine-tuned based on a modified version of QWen2.5-14B-Instruct) at [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/bytedance-research/ChatTS-14B).
- Evaluations datasets: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14349206.svg)](https://doi.org/10.5281/zenodo.14349206).
- Training scripts for training your own model: [ChatTS-Training](https://github.com/xiezhe-24/ChatTS-Training).

## How To Use
### Installation
- Basic requirements for model inference: `python>=3.11`, `deepspeed`, `vllm==0.6.6.post1`, `torch==2.5.1`, `flash-attn` (refer to `requirements.txt`).
- Download the evaluation datasets from [Zenodo](https://doi.org/10.5281/zenodo.14349206) and put them under `evaluation/dataset` (`evaluation/dataset/dataset_a.json` and `evaluation/dataset/dataset_b.json`).
- Download the trained model weights from [HuggingFace](https://huggingface.co/bytedance-research/ChatTS-14B), extract it and put all the extracted files under `ckpt/` (`ckpt/config.json`, etc).
- **Note:** `ChatTS` is trained based on a 14B-sized base model, so you need to ensure that you have a GPU with sufficient memory for inference. Additionally, due to the model's requirements, `Flash-Attention` (https://github.com/Dao-AILab/flash-attention) is essential, so you need to ensure that your GPU meets the installation requirements for Flash-Attention. Recommended GPUs: A100/A800.

### Try the ChatTS Model
- Following the steps in `Installation` to download the trained `ChatTS` model and place it under `ckpt`. 
- The ChatTS model can be loaded directly using the `transformers` library. **Refer to `demo_hf.ipynb` for more information.**
- **About `sp` Encoding.** To facilitate the input of variable-length batch time series, we adopted a method named `sp` encoding when encoding the time series. For each time series data point, an additional numerical value of 1.0 is added as a mask. For convenience, we have a Processor which can be loaded with `AutoProcessor` in `transformers` to normalize and convert the time series and text (Value-Preserved Time Series Encoding). Please refer to `demo_hf.ipynb` for more information about their usage. 
- An example usage of ChatTS (with `HuggingFace`):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
import numpy as np

# Load the model, tokenizer and processor
model = AutoModelForCausalLM.from_pretrained("./ckpt", trust_remote_code=True, device_map=0, torch_dtype='float16')
tokenizer = AutoTokenizer.from_pretrained("./ckpt", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("./ckpt", trust_remote_code=True, tokenizer=tokenizer)
# Create time series and prompts
timeseries = np.sin(np.arange(256) / 10) * 5.0
timeseries[100:] -= 10.0
prompt = f"I have a time series length of 256: <ts><ts/>. Please analyze the local changes in this time series."
# Apply Chat Template
prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n"
# Convert to tensor
inputs = processor(text=[prompt], timeseries=[timeseries], padding=True, return_tensors="pt")
# Move to GPU
inputs = {k: v.to(0) for k, v in inputs.items()}
# Model Generate
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True))
```
- **Note:** Due to training data limitations, the recommended time series length for ChatTS ranges from **64 to 1024**, with a maximum of **30** series in one input. Currently, time series that are too short (i.e., shorter than 64) may not be correctly recognized by ChatTS. We will improve this in future updates.

### vLLM Inference (Experimental)
Since [vLLM](https://github.com/vllm-project/vllm) lacks native support for the `ChatTS` model, we have provided a [patch](chatts/vllm/chatts_vllm.py) to enable vLLM to support inference. Therefore, before using vLLM to load the model, please make sure that the code includes: `import chatts.vllm.chatts_vllm` to register the ChatTS model in vLLM. Please refer to the following steps to use vLLM to load ChatTS:

1. Install `vllm==0.6.6.post1` (please ensure that you have installed the exact version as vLLM's multimodal APIs change frequently).
2. Please refer to `demo_vllm.py` for detailed usage methods.

A simple example of using vLLM to load ChatTS: 
```python
import chatts.vllm.chatts_vllm
from vllm import LLM, SamplingParams
# Load the model
language_model = LLM(model="./ckpt", trust_remote_code=True, max_model_len=ctx_length, tensor_parallel_size=1, gpu_memory_utilization=0.95, limit_mm_per_prompt={"timeseries": 50})
# Create time series (np.ndarray) and prompts (chat_templated applied)
ts1, ts2 = ...
prompt = ...
# Model Inference
outputs = language_model.generate([{
      "prompt": prompt,
      "multi_modal_data": {"timeseries": [ts1, ts2]}
  }], sampling_params=SamplingParams(max_tokens=300))
```

### Training Data Generation
1. **QA Generation with Templates (Stage-1)**. Use `python3 -m chatts.generate_template_qa` to generate a training dataset with pre-defined templates.
2. **QA Generation with LLMs (Stage-2)**. You need a downloaded LLM that can be loaded with `vLLM` to perform this step. Set `[LOCAL_LLM_PATH]` in `chatts/generate_llm_qa.py` to a local LLM model (e.g., QWen2.5-32B-Instruct, **NOT ChatTS Model**) and set num_gpus, gpu_per_model accordingly. Use `python3 -m chatts.generate_llm_qa` to generate a training dataset with LLMs.
3. **TSEvol**. You need a downloaded LLM that can be loaded with `vLLM` to perform this step. The datasets generated in Step 1 and Step 2 will be used as seed QAs in TSEvol, so please make sure that you have successfully generated the previous datasets before running TSEvol. Then, refer to the steps in `chatts/evol/evol_instruct.py`:
    1. Set `[LOCAL_LLM_PATH]` in `evol_instruct.py` to the path of a local LLM model (e.g., QWen2.5-32B-Instruct. **NOT ChatTS Model**) for QA generation and set num_gpus, gpu_per_model accordingly in `chatts/evol/evol_instruct.py`.
    2. Run `python3 -m chatts.evol.evol_instruct`.
    3. The output will be saved to the file specified in `OUTPUT_FILE`.
  
**Note:** These codes for data generation are provided only as references rather than complete implementations. The complete code and training datasets will be released soon after the acceptance of this paper.

### Deepspeed Model Inference for Evaluation
- We provide a simple script for inference of ChatTS (`chatts/inference_tsmllm_deepspeed.py`) with `deepspeed`. After installing `deepspeed`, please set the `WORKDIR` (the absolute path of the current directory) and the evaluation dataset in the script. Then, run the following command to do the model inference:
```sh
deepspeed --num_gpus [YOUR_NUM_GPUS] --master_port 12345 chatts/inference_tsmllm_deepspeed.py
```
You should find the inference results under `exp/` folder, which will be further used for evaluation.

### Evaluation
- Install `ragas==0.1.9` (https://github.com/explodinggradients/ragas), which is used for evaluating the inductive reasoning results.
- Set the `API_KEY` and `OPENAI_URL` in `evaluation/ragas/config/config.toml` (Refer to https://platform.openai.com/docs/api-reference).
- Run `python3 -m evaluation.evaluate_tsmllm_models` to evaluate `ChatTS` (make sure you have done the model inference before).
- We also provide a simple demo to evaluate the performance of text-based GPT models. After setting your `API_KEY` and `OPENAI_URL` in `evaluation/evaluate_gpt_text_models.py`, run the command `python3 -m evaluation.evaluate_gpt_text_models` to obtain the evaluation results of the text-based GPT model.

### Fine-Tuning Your Own Model
- We provide a simple script for fine-tuning your own TS-MLLM models: https://github.com/xiezhe-24/ChatTS-Training (modified based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)). Refer to this repository for more details.

## Evaluation Datasets
- We've provided the two evaluation datasets we gathered, as stated in the paper. You can find them in the `evaluation/dataset` folder. Each sample in these datasets has several parts: `timeseries`, which is the time series data itself; `question`, the query related to the time series; `answer`, the text-form standard answers provided for your reference only; `attributes`, the structured labels used to evaluate results; and `ability_types`, which indicates the types of tasks the question involves.
**Please pay special attention to this**: To cut down on evaluation costs, we've combined different questions that pertain to the same time series into one `question`. We use numbering to tell these different questions apart. So, when you look at the evaluation dataset, the actual count of questions might be more than the number of `timeseries` entries. Another thing to note is that some tasks in inductive reasoning and alignment are grouped together in one question. This is because inductive reasoning tasks often require explaining the physical meanings of time series attributes. 
- The `MCQ2` dataset is sourced from a third-party and is open-source. However, due to licensing restrictions, we are unable to provide it within this repository. You can directly download it via https://github.com/behavioral-data/TSandLanguage.

## Case Studies
![image](figures/case_studies.png)
In `ChatTS`, we mainly focus on **Understanding and Reasoning** about time series, just like what vision/video/audio-MLLMs do, rather than conducting time series prediction, anomaly detection and classification tasks.
You can try more application scenarios of ChatTS by modifying the time series and the text of questions in `demo_hf.ipynb`! 

## Third-Party Dependencies
- QWen (https://github.com/QwenLM/Qwen2.5)
- DeepSpeed (https://www.deepspeed.ai/)
- RAGAS (https://github.com/explodinggradients/ragas)
- vLLM (https://github.com/vllm-project/vllm)
- Flash Attention (https://github.com/Dao-AILab/flash-attention)

## Security
If you discover a potential security issue in this project, or think you may
have discovered a security issue, we ask that you notify Bytedance Security via our [security center](https://security.bytedance.com/src) or [vulnerability reporting email](sec@bytedance.com).

Please do **not** create a public GitHub issue for a security vulnerability.

## License
This project is licensed under the [MIT License](LICENSE).

## Cite
```bibtex
@article{xie2024chatts,
  title={ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning},
  author={Xie, Zhe and Li, Zeyan and He, Xiao and Xu, Longlong and Wen, Xidao and Zhang, Tieying and Chen, Jianjun and Shi, Rui and Pei, Dan},
  journal={arXiv preprint arXiv:2412.03104},
  year={2024}
}
```
