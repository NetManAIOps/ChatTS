<div align="center">

# ChatTS: Time Series LLM for Understanding and Reasoning

[[🌐 **Website**]](https://netmanaiops.github.io/ChatTS/) ·
[[🤗 **Model**]](https://huggingface.co/bytedance-research/ChatTS-14B) ·
[[🤗 **Web Demo**]](https://huggingface.co/spaces/xiezhe22/ChatTS) ·
[[📄 **Paper**]](https://arxiv.org/abs/2412.03104)

[[🧰 **Training Scripts**]](https://github.com/xiezhe-24/ChatTS-Training) ·
[[📚 **Training Datasets**]](https://huggingface.co/datasets/ChatTSRepo/ChatTS-Training-Dataset) ·
[[🧪 **Evaluation Datasets**]](https://doi.org/10.5281/zenodo.14349206)

</div>

`ChatTS` is a Time Series Multimodal LLM (TS-MLLM) focused on **understanding** and **reasoning** over time series—similar in spirit to vision/video/audio MLLMs, but **natively built for time series**. This repository provides code, datasets, and the **ChatTS-14B-0801** model (VLDB’25): *ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning*.

![Chat](figures/chat_example.png)

Check out the [Case Studies](#case-studies) section for more real-world applications and sample conversations.

---

## What ChatTS Does

* ✅ **Native multivariate support**
  Works with multiple time series **of different lengths** and **flexible dimensionality** in one prompt.

* ✅ **Conversational understanding + reasoning**
  Interactively explore structure, changes, and relationships across series.

* ✅ **Value-preserving encoding**
  Retains **raw numerical values** so the model can answer questions like
  *“How large is the spike at timestamp *t*?”*

* ✅ **Easy integration**
  Plug into standard LLM pipelines; supports Transformers and vLLM (with the registration patch).

**Example application**
See `figures/chat_example.png` and the **Case Studies** section for real-world scenarios and conversation examples.

---

## News

* **2025/10/23**: The [official website of ChatTS](https://netmanaiops.github.io/ChatTS/) launched.
* **2025/08/01**: `ChatTS-14B-0801` model launched with enhanced reasoning compability and Chinese support.
* **2025/07/24**: Web Demo launched on Hugging Face Spaces.
* **2025/07/03**: Released `ChatTS-14B-GPTQ-Int4` on Hugging Face.
* **2025/04/29**: Updated data generation code and released baseline implementations; see **Evaluation**.
* **2025/04/16**: ChatTS accepted by VLDB’25 and training datasets released.
* **2025/01/01**: New ChatTS release with enhanced CoT & QA capability.
* **2024/12/30**: Experimental vLLM support (KV cache & `AutoProcessor` enabled).

---

## Quick Start

### 1) Try It in the Browser (no install)
Use the Web Demo on Hugging Face Spaces:  
**👉 [ChatTS Web Demo](https://huggingface.co/spaces/xiezhe22/ChatTS)**  
You can upload your `.csv` file (example: `demo/ts_example.csv`) and chat with ChatTS-14B-0801 about your data.

### 2) Install & Prepare
**Requirements (for inference):** `python>=3.11`, `torch==2.6.0`, `vllm==0.8.5`, `deepspeed`, `flash-attn` (see `requirements.txt`).  
**Hardware:** ChatTS-14B-0801 is a 14B model. Use a GPU with sufficient memory and ensure your GPU supports Flash-Attention (e.g., A100/A800).

**Steps**
1. **Download model weights**:  
   - [ChatTS-14B-0801 on Hugging Face](https://huggingface.co/bytedance-research/ChatTS-14B)  
   - Unzip into `./ckpt/` (files like `ckpt/config.json`, etc.)
2. (**Optional for local evaluation**) Download evaluation datasets to `evaluation/dataset/`:  
   - `dataset_a.json`, `dataset_b.json` from [Zenodo](https://doi.org/10.5281/zenodo.14349206).

### 3) Deploy with Transformers
ChatTS-14B-0801 supports **Value-Preserved Time Series Encoding** via `AutoProcessor` and an `sp` mask for variable-length batches.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch, numpy as np

# 1) Load model, tokenizer, processor
model = AutoModelForCausalLM.from_pretrained("./ckpt", trust_remote_code=True, device_map=0, torch_dtype='float16')
tokenizer = AutoTokenizer.from_pretrained("./ckpt", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("./ckpt", trust_remote_code=True, tokenizer=tokenizer)

# 2) Prepare a timeseries and prompt
timeseries = np.sin(np.arange(256) / 10) * 5.0
timeseries[100:] -= 10.0
prompt = "I have a time series length of 256: <ts><ts/>. Please analyze the local changes in this time series."
prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# 3) Preprocess & generate
inputs = processor(text=[prompt], timeseries=[timeseries], padding=True, return_tensors="pt")
inputs = {k: v.to(0) for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True))
```

**Notes**

* Recommended time series length: **64–1024**; up to **30** series per input. Very short series (<64) may be less reliable (to be improved).
* See `demo/demo_hf.ipynb` for more examples.

### 4) vLLM Inference (Experimental)

vLLM does not natively ship ChatTS support; we provide a registration patch.
Before loading with vLLM, **register ChatTS**:

```python
# in your script, before constructing LLM()
import chatts.vllm.chatts_vllm

from vllm import LLM, SamplingParams

ctx_length = 6000
language_model = LLM(
    model="./ckpt",
    trust_remote_code=True,
    max_model_len=ctx_length,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
    limit_mm_per_prompt={"timeseries": 50}
)

ts_list = [...]  # list of np.ndarray time series
prompt = "...(chat template applied)..."

outputs = language_model.generate([{
  "prompt": prompt,
  "multi_modal_data": {"timeseries": ts_list}
}], sampling_params=SamplingParams(max_tokens=300))
```

### 5) OpenAI-Compatible Server (vLLM)

You can deploy an OpenAI API–compatible server using vLLM. The following **text replaces the original Issue #11 link** and summarizes the necessary steps:

* Install a vLLM build that includes ChatTS registration and multimodal timeseries support.
* Start the server, passing trust/override flags so vLLM recognizes the ChatTS model type, and set a per-prompt limit for timeseries inputs.

Example command:

```bash
vllm serve ./ckpt \
  --served-model-name chatts \
  --trust-remote-code \
  --hf-overrides '{"model_type":"chatts"}' \
  --max-model-len 6000 \
  --gpu-memory-utilization 0.97 \
  --limit-mm-per-prompt timeseries=15 \
  --allowed-local-media-path $(pwd) \
  --host 0.0.0.0 \
  --port 12345 \
  --uvicorn-log-level debug
```

If client calls fail with a 500 error during serialization, set:

```bash
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
```

---

## Training (Build Your Own)

We provide a full pipeline and **two modes** for generating data:

* **Local LLM mode (vLLM)** — traditional local GPU inference.
* **Remote API mode** — call OpenAI/Claude, etc., with parallel workers.

Configure in `config/datagen_config.yaml`:

```yaml
# Local LLM mode
local_llm_path: "/path/to/your/model"

# Remote API mode
remote_api:
  enabled: true
  base_url: "https://api.openai.com"
  api_key: "your-api-key"
  model: "gpt-4o"
  max_workers: 8
```

### Data Generation Steps

1. **Alignment datasets**

   * Configure the backend, then run:

   ```bash
   bash scripts/generate_align_datasets.sh
   ```

   * (Optional) Template-based seed QAs:

   ```bash
   python3 -m demo.generate_template_qa
   ```
2. **SFT datasets**

   * LLM-generated seed QAs:

   ```bash
   python3 -m chatts.sft.generate_llm_qa
   ```

   * **TSEvol** to expand datasets (supports local LLM or remote APIs):

   ```bash
   python3 -m chatts.sft.generate_tsevol_dataset
   ```

   * **IFT (Instruction-Following)** datasets (requires alignment labels first):

   ```bash
   python3 -m chatts.sft.generate_ift_dataset
   ```

**Sequence length**: configure `SEQ_LEN` in `config/datagen_config.yaml`. ChatTS commonly uses length **256** (we also mix other lengths via `seq_len=null`).

**Enhanced SFT (used by ChatTS-14B-0801)**
Scripts for diversified SFT data:

* `generate_uts_reason.py` (English univariate reasoning)
* `generate_uts_reason_cn.py` (Chinese univariate reasoning w/ consistency checks)
* `generate_mts_reason.py` (multivariate reasoning)
* `generate_rewrite_dataset.py` (augmentation by rewriting)

Run all at once:

```bash
bash scripts/generate_enhanced_sft_datasets.sh
```

**Fine-tuning**
Use **ChatTS-Training** (modified from LLaMA-Factory):
👉 [https://github.com/xiezhe-24/ChatTS-Training](https://github.com/xiezhe-24/ChatTS-Training)

---

## Evaluation

### Step 1: Generate Inference Outputs

* **vLLM Inference**:

```bash
python3 -m chatts.utils.inference_tsmllm_vllm
```

Outputs are written under `exp/`.

### Step 2: Compute Metrics

* Install `ragas==0.1.9`.
* Set `API_KEY` and `OPENAI_URL` in `evaluation/ragas/config/config.toml`.
* Run:

```bash
python3 -m evaluation.evaluate_tsmllm_models
```

(We also include code to evaluate text/vision/agent LLM baselines; see `evaluation/` for details.)

---

## Time Series Generator

A user-friendly generator produces attribute pools and corresponding series.
See `demo/demo_ts_generator.ipynb`.

---

## Resource Links

| Resource                    | Link                                                                                                                                     | Description                   |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| **Paper (VLDB’25)**         | [https://arxiv.org/abs/2412.03104](https://arxiv.org/abs/2412.03104)                                                                     | ChatTS paper                  |
| **ChatTS-14B-0801 Model** | [https://huggingface.co/bytedance-research/ChatTS-14B](https://huggingface.co/bytedance-research/ChatTS-14B)                             | Model weights                 |
| **ChatTS-14B-GPTQ-Int4**    | [https://huggingface.co/xiezhe24/ChatTS-14B-GPTQ-Int4](https://huggingface.co/xiezhe24/ChatTS-14B-GPTQ-Int4)                             | Quantized model               |
| **Training Datasets**       | [https://huggingface.co/datasets/ChatTSRepo/ChatTS-Training-Dataset](https://huggingface.co/datasets/ChatTSRepo/ChatTS-Training-Dataset) | Synthetic training set        |
| **Evaluation Datasets**     | [https://doi.org/10.5281/zenodo.14349206](https://doi.org/10.5281/zenodo.14349206)                                                       | Real & synthetic eval data    |
| **Training Scripts**        | [https://github.com/xiezhe-24/ChatTS-Training](https://github.com/xiezhe-24/ChatTS-Training)                                             | Fine-tuning & data generation |

---

## Third-Party Dependencies

* Qwen ([https://github.com/QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5))
* DeepSpeed ([https://www.deepspeed.ai/](https://www.deepspeed.ai/))
* RAGAS ([https://github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas))
* vLLM ([https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm))
* Flash-Attention ([https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention))

---

## Case Studies
![image](figures/case_studies.png)
In `ChatTS`, we mainly focus on **Understanding and Reasoning** about time series, just like what vision/video/audio-MLLMs do, rather than conducting time series prediction, anomaly detection and classification tasks.
You can try more application scenarios of ChatTS by modifying the time series and the text of questions in `demo_hf.ipynb`! 

---

## Security

If you discover a potential security issue, please contact ByteDance Security via the [security center](https://security.bytedance.com/src) or email **[sec@bytedance.com](mailto:sec@bytedance.com)**.
**Do not** open public GitHub issues for vulnerabilities.

---

## License

This project is licensed under the **MIT License** (see `LICENSE`).

---

## Cite

```bibtex
@inproceedings{xie2025chatts,
  title={ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning},
  author={Xie, Zhe and Li, Zeyan and He, Xiao and Xu, Longlong and Wen, Xidao and Zhang, Tieying and Chen, Jianjun and Shi, Rui and Pei, Dan},
  booktitle={Proceedings of the VLDB Endowment, 2025},
  year={2025}
}
```
