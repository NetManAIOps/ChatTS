pip3 uninstall -y vllm flash-attn

echo ">>> Reinstalling modified vLLM build..."
ORIGINAL_DIR="$(pwd)"
echo "Please enter the absolute path where the vLLM repository should be cloned." \
     "Press Enter to accept the default location under your home directory." \
     "The directory will be created if it does not exist."
read -r -p "Clone destination [default: $HOME]: " USER_CLONE_ROOT
CLONE_ROOT="${USER_CLONE_ROOT:-$HOME}"
if [[ "$CLONE_ROOT" == ~* ]]; then
    CLONE_ROOT="${CLONE_ROOT/#\~/$HOME}"
fi
echo ">>> Cloning vLLM into: $CLONE_ROOT/vllm"
mkdir -p "$CLONE_ROOT"
cd "$CLONE_ROOT" || exit 1
if [[ -d vllm ]]; then
    echo ">>> Existing directory detected at $CLONE_ROOT/vllm; removing it before cloning."
    rm -rf vllm
fi
git clone https://github.com/xiez22/vllm.git
cd vllm || exit 1
git checkout timeseries

# Check the python version (3.11 or 3.12) and install the corresponding flash-attn wheel
if python3 -c "import sys; exit(0) if sys.version_info[:2] == (3, 11) else exit(1)"; then
    FLASH_ATTN_WHL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
elif python3 -c "import sys; exit(0) if sys.version_info[:2] == (3, 12) else exit(1)"; then
    FLASH_ATTN_WHL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
else
    echo "Unsupported Python version. Please use Python 3.11 or 3.12."
    exit 1
fi

echo ">>> Installing Flash Attention from: $FLASH_ATTN_WHL_URL"
pip3 install "$FLASH_ATTN_WHL_URL"
VLLM_USE_PRECOMPILED=1 pip install -vvv -e "$CLONE_ROOT/vllm" --no-build-isolation
cd "$ORIGINAL_DIR" || exit 1
pip install transformers==4.52.4 "numpy<2.0.0"

echo ">>> Verifying vLLM installation..."
python3 -c "import vllm" || {
    echo ">>> vLLM installation failed! Please check the error messages above or open an issue at GitHub."
    exit 1
}

echo ">>> vLLM installation completed!"
