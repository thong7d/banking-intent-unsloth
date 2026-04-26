#!/usr/bin/env bash
# =============================================================================
# train.sh — Banking Intent Classification: Preprocess + Fine-tune
#
# Usage (local):  bash train.sh
# Usage (colab):  bash train.sh --env colab
#
# This script calls main.py sequentially for the two mandatory pipeline steps:
#   1. preprocess  — stratified sampling, text cleaning, train/val/test split
#   2. train       — QLoRA fine-tuning of Llama-3.1-8B-Instruct via Unsloth
# =============================================================================

set -e  # Exit immediately if any command returns a non-zero status

# ---- Parse optional --env argument (default: local) ----
ENV="local"
for arg in "$@"; do
    case $arg in
        --env=*)  ENV="${arg#*=}" ;;
        --env)    shift; ENV="$1" ;;
    esac
done

echo "============================================="
echo " Banking Intent Pipeline — train.sh"
echo " Environment : ${ENV}"
echo "============================================="

# ---- Step 1: Preprocess ----
echo ""
echo "[1/2] Running data preprocessing (sampling + cleaning + splitting)..."
python main.py --step preprocess --env "${ENV}"

# ---- Step 2: Train ----
echo ""
echo "[2/2] Starting QLoRA fine-tuning (Llama-3.1-8B-Instruct)..."
python main.py --step train --env "${ENV}"

echo ""
echo "============================================="
echo " Training complete. Checkpoint saved to:"
if [ "${ENV}" = "colab" ]; then
    echo "  /content/drive/MyDrive/banking-intent-unsloth/outputs/banking-intent-llama31-8b/"
else
    echo "  outputs/banking-intent-llama31-8b/"
fi
echo "============================================="
