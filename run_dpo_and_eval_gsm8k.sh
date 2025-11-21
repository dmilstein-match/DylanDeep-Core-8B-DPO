#!/bin/bash
# Complete pipeline for DPO training and 8-shot evaluation on GSM8K
# Designed for Lambda 8x H100 instances

set -e  # Exit on error

echo "========================================="
echo "DPO Training + 8-Shot Eval Pipeline"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
BASE_MODEL="GAIR/Abel-7B-002"
SFT_CHECKPOINT="checkpoints/abel_sft_lora"
MERGED_MODEL="checkpoints/abel_sft_merged"
ROLLOUTS_FILE="data/abel_regime_w_rollouts.jsonl"
PREFS_FILE="data/abel_coherence_preferences.jsonl"
DPO_CHECKPOINT="checkpoints/abel_coherence_lora"
EVAL_OUTPUT="outputs/abel_coherence_8shot_maudlin0.jsonl"

echo -e "${BLUE}Configuration:${NC}"
echo "  Base Model: $BASE_MODEL"
echo "  SFT Checkpoint: $SFT_CHECKPOINT"
echo "  DPO Output: $DPO_CHECKPOINT"
echo "  8-Shot Eval Output: $EVAL_OUTPUT"
echo ""

# Phase 1: Check if SFT checkpoint exists
echo -e "${BLUE}[Phase 1] Checking SFT Checkpoint...${NC}"
if [ ! -d "$SFT_CHECKPOINT" ]; then
    echo -e "${RED}ERROR: SFT checkpoint not found at $SFT_CHECKPOINT${NC}"
    echo "Please run SFT training first:"
    echo "  accelerate launch src/baseline_sft/train_sft_abel.py"
    exit 1
fi
echo -e "${GREEN}✓ SFT checkpoint found${NC}"
echo ""

# Phase 2: Merge LoRA (if not already merged)
echo -e "${BLUE}[Phase 2] Merging LoRA with Base Model...${NC}"
if [ ! -d "$MERGED_MODEL" ]; then
    echo "Merging SFT LoRA adapter with base model..."
    python src/merge_lora.py \
        --base_model "$BASE_MODEL" \
        --lora_path "$SFT_CHECKPOINT" \
        --output_path "$MERGED_MODEL"
    echo -e "${GREEN}✓ LoRA merge complete${NC}"
else
    echo -e "${GREEN}✓ Merged model already exists, skipping${NC}"
fi
echo ""

# Phase 3: Collect rollouts with vLLM (if not already done)
echo -e "${BLUE}[Phase 3] Collecting Rollouts with vLLM...${NC}"
if [ ! -f "$ROLLOUTS_FILE" ] || [ $(wc -l < "$ROLLOUTS_FILE") -lt 500 ]; then
    echo "Installing vLLM (if not installed)..."
    pip install -q vllm==0.6.3.post1

    echo "Generating 500 rollouts with 8-GPU tensor parallelism..."
    python src/rl_training/collect_rollouts_abel_vllm.py \
        --model_path "$MERGED_MODEL" \
        --data_path data/gsm8k_train.jsonl \
        --out_path "$ROLLOUTS_FILE" \
        --n_samples 500

    # Verify output
    ROLLOUT_COUNT=$(wc -l < "$ROLLOUTS_FILE")
    echo -e "${GREEN}✓ Generated $ROLLOUT_COUNT rollouts${NC}"
else
    echo -e "${GREEN}✓ Rollouts already exist, skipping${NC}"
fi
echo ""

# Phase 4: Build preference pairs
echo -e "${BLUE}[Phase 4] Building Preference Pairs...${NC}"
if [ ! -f "$PREFS_FILE" ]; then
    echo "Creating correctness-first preference pairs..."
    python -m src.rl_training.build_preferences_abel

    PREFS_COUNT=$(wc -l < "$PREFS_FILE")
    echo -e "${GREEN}✓ Generated $PREFS_COUNT preference pairs${NC}"
else
    echo -e "${GREEN}✓ Preference pairs already exist, skipping${NC}"
fi
echo ""

# Phase 5: Train DPO
echo -e "${BLUE}[Phase 5] Training DPO (3 epochs, 5e-5 LR)...${NC}"
echo "This will use 8x H100 GPUs with distributed training"
echo "Estimated time: 15-30 minutes"
echo ""
accelerate launch src/rl_training/train_dpo_coherence.py
echo -e "${GREEN}✓ DPO training complete${NC}"
echo ""

# Phase 6: Run 8-shot evaluation on GSM8K
echo -e "${BLUE}[Phase 6] Running 8-Shot Evaluation (Maudlin_0)...${NC}"
echo "Configuration: temp=0.2, 8 samples per question, majority voting"
echo "Dataset: GSM8K test set (1,319 examples)"
echo "Estimated time: 40-60 minutes"
echo ""
python eval_8shot_maudlin0.py
echo -e "${GREEN}✓ 8-shot evaluation complete${NC}"
echo ""

# Final results
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo ""
echo "Results:"
echo "  DPO Checkpoint: $DPO_CHECKPOINT"
echo "  8-Shot Eval Results: $EVAL_OUTPUT"
echo ""
echo "To view accuracy:"
echo "  python -c 'import json; results = [json.loads(l) for l in open(\"$EVAL_OUTPUT\")]; print(f\"Accuracy: {100 * sum(r[\"correct\"] for r in results) / len(results):.2f}%\")'"
echo ""
echo "To submit to HuggingFace leaderboard:"
echo "  1. Review results in $EVAL_OUTPUT"
echo "  2. Ensure accuracy is 80%+"
echo "  3. Follow HF submission guidelines"
echo ""
