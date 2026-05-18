#!/bin/bash
# Usage: bash scripts/vastai_train.sh [radiology|cardiology|oncology]
# Run this inside the VastAI Docker container after mounting your data volume.

set -euo pipefail

AGENT=${1:-radiology}
CONFIG="configs/training/${AGENT}.yaml"

echo "=== AI Medical Dept — Training: ${AGENT} ==="
echo "Config: ${CONFIG}"
echo "Data root: ${DATA_ROOT}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"

# Login to HuggingFace (token from env)
huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential

# Run fine-tuning
python "training/finetune_${AGENT}.py" --config "${CONFIG}"

echo "=== Done. Adapter saved to ${CHECKPOINT_DIR}/lora_${AGENT}/ ==="
