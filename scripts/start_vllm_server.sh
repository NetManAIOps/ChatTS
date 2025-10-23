python3 -c "import vllm"
VLLM_ALLOW_INSECURE_SERIALIZATION=1 vllm serve ./ckpt \
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
