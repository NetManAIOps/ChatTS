echo "Make sure to run this script from the root of the repository."
echo "Make sure that you have set up the correct local_llm_path in config/datagen_config.yaml!!!"

echo "------------------------------"
echo "[1/4] UTS Reasoning (English)"
python3 -m chatts.sft.generate_uts_reason

echo "------------------------------"
echo "[2/4] UTS Reasoning (Chinese)"
python3 -m chatts.sft.generate_uts_reason_cn

echo "------------------------------"
echo "[3/4] MTS Reasoning (Multivariate)"
python3 -m chatts.sft.generate_mts_reason

echo "------------------------------"
echo "[4/4] TSRewrite Dataset"
python3 -m chatts.sft.generate_rewrite_dataset

echo "==============================="
echo "All enhanced SFT datasets have been generated."
echo "You can find them in the data folder."
echo "==============================="
