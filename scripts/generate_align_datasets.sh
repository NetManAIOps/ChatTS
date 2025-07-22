echo "Make sure to run this script from the root of the repository."
echo "Make sure that you have set up the correct local_llm_path in config/datagen_config.json!!!"

echo "------------------------------"
echo "[1/6] UTS Template"
python3 -m chatts.align.uts_template_qa

echo "------------------------------"
echo "[2/6] MTS-Shape Template"
python3 -m chatts.align.mts_shape_template_qa

echo "------------------------------"
echo "[3/6] MTS-Local Template"
python3 -m chatts.align.mts_local_template_qa

echo "------------------------------"
echo "[4/6] UTS LLM"
python3 -m chatts.align.uts_llm_qa

echo "------------------------------"
echo "[5/6] MTS-Shape LLM"
python3 -m chatts.align.mts_shape_llm_qa

echo "------------------------------"
echo "[6/6] MTS-Local LLM"
python3 -m chatts.align.mts_local_llm_qa

echo "==============================="
echo "All datasets have been generated."
echo "You can find them in the data folder."
echo "==============================="
