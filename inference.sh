#!/usr/bin/env bash
# =============================================================================
# inference.sh — Banking Intent Classification: Standalone Inference Demo
#
# Usage (local):  bash inference.sh
# Usage (colab):  bash inference.sh --env colab
#
# This script invokes main.py --step infer, which in turn calls:
#   scripts/inference.py
#       --config   configs/inference.yaml
#       --checkpoint_dir <resolved outputs dir>/banking-intent-llama31-8b
#       --data_dir       <resolved data dir>/sample_data
#
# The IntentClassification class will:
#   1. Load the LoRA adapter from the checkpoint directory.
#   2. Run inference on 3 built-in banking example queries.
#   3. Print the predicted intent label for each query.
#
# Requirements: The LoRA adapter and id2label.json must already exist.
#   - Local: copy from Google Drive or run train.sh first.
#   - Colab:  mount Drive; weights are loaded from MyDrive automatically.
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
echo " Banking Intent Pipeline — inference.sh"
echo " Environment : ${ENV}"
echo "============================================="
echo ""
echo "Loading model and running inference demo..."
echo ""

python main.py --step infer --env "${ENV}"

echo ""
echo "============================================="
echo " Inference demo complete."
echo "============================================="
