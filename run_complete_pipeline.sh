#!/bin/bash
# Complete Pipeline Runner - Abel Regime W Coherence Training
# Runs the full pipeline from data download to final evaluation

set -e  # Exit on any error

echo "================================================================================"
echo "Abel Regime W Complete Pipeline"
echo "================================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Track start time
START_TIME=$(date +%s)

# Step 1: Download GSM8K dataset
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 1/6: Download GSM8K Dataset${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ -f "data/gsm8k_train.jsonl" ] && [ -f "data/gsm8k_platinum_test.jsonl" ]; then
    echo -e "${GREEN}✓ GSM8K dataset already exists, skipping download${NC}"
else
    echo "Downloading GSM8K dataset..."
    python download_gsm8k.py
    echo -e "${GREEN}✓ Dataset downloaded successfully${NC}"
fi
echo ""

# Step 2: Check for merged SFT model
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 2/6: Verify Merged SFT Model${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ -d "checkpoints/abel_sft_merged" ] && [ -f "checkpoints/abel_sft_merged/config.json" ]; then
    echo -e "${GREEN}✓ Merged SFT model found at checkpoints/abel_sft_merged${NC}"
else
    echo -e "${RED}✗ Merged SFT model not found${NC}"
    echo "You need to either:"
    echo "  1. Train SFT: accelerate launch src/baseline_sft/train_sft_abel.py"
    echo "  2. Merge LoRA: python src/merge_lora.py --base_model GAIR/Abel-7B-002 --lora_path checkpoints/abel_sft_lora --output_path checkpoints/abel_sft_merged"
    exit 1
fi
echo ""

# Step 3: Generate Regime W rollouts (OPTIMIZED vLLM)
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 3/6: Generate Regime W Rollouts (Optimized vLLM)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

ROLLOUT_START=$(date +%s)

if [ -f "data/abel_regime_w_rollouts.jsonl" ]; then
    echo -e "${YELLOW}⚠ Rollouts already exist. Delete data/abel_regime_w_rollouts.jsonl to regenerate.${NC}"
    read -p "Use existing rollouts? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        rm data/abel_regime_w_rollouts.jsonl
        echo "Generating rollouts with optimized vLLM (16 questions × 11 arms batched)..."
        python src/rl_training/collect_rollouts_abel_vllm_optimized.py \
            --model_path checkpoints/abel_sft_merged \
            --n_samples 500 \
            --question_batch_size 16
    fi
else
    echo "Generating rollouts with optimized vLLM (16 questions × 11 arms batched)..."
    python src/rl_training/collect_rollouts_abel_vllm_optimized.py \
        --model_path checkpoints/abel_sft_merged \
        --n_samples 500 \
        --question_batch_size 16
fi

ROLLOUT_END=$(date +%s)
ROLLOUT_TIME=$((ROLLOUT_END - ROLLOUT_START))
echo -e "${GREEN}✓ Rollouts generated in ${ROLLOUT_TIME}s${NC}"
echo ""

# Step 4: Build preference pairs
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 4/6: Build Preference Pairs${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo "Building correctness-first preference pairs..."
python src/rl_training/build_preferences_abel.py

echo -e "${GREEN}✓ Preference pairs built${NC}"
echo ""

# Validate pipeline before DPO training
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Pipeline Validation${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

./debug_pipeline.sh

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Pipeline validation failed. Fix errors before continuing.${NC}"
    exit 1
fi
echo ""

# Step 5: Train DPO
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 5/6: Train DPO Coherence Model${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

DPO_START=$(date +%s)
echo "Training DPO with coherence rewards (8× H100, 5e-5 LR, 3 epochs)..."

accelerate launch src/rl_training/train_dpo_coherence.py

DPO_END=$(date +%s)
DPO_TIME=$((DPO_END - DPO_START))
echo -e "${GREEN}✓ DPO training completed in ${DPO_TIME}s${NC}"
echo ""

# Step 6: Evaluate on GSM8K Platinum
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 6/6: Evaluate on GSM8K Platinum${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

EVAL_START=$(date +%s)
echo "Evaluating coherence-optimized model on GSM8K Platinum (1,210 examples)..."

python src/eval/eval_abel_coherence_platinum.py

EVAL_END=$(date +%s)
EVAL_TIME=$((EVAL_END - EVAL_START))
echo -e "${GREEN}✓ Evaluation completed in ${EVAL_TIME}s${NC}"
echo ""

# Summary
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "================================================================================"
echo -e "${GREEN}✓ PIPELINE COMPLETE!${NC}"
echo "================================================================================"
echo ""
echo "Timing Summary:"
echo "  Rollout collection: ${ROLLOUT_TIME}s (~$((ROLLOUT_TIME / 60)) minutes)"
echo "  DPO training:       ${DPO_TIME}s (~$((DPO_TIME / 60)) minutes)"
echo "  Evaluation:         ${EVAL_TIME}s (~$((EVAL_TIME / 60)) minutes)"
echo "  Total time:         ${TOTAL_TIME}s (~$((TOTAL_TIME / 60)) minutes)"
echo ""
echo "Results:"
echo "  Final checkpoint:   checkpoints/abel_coherence_lora/"
echo "  Evaluation output:  outputs/abel_coherence_platinum_eval.jsonl"
echo ""
echo "To check accuracy:"
echo "  cat outputs/abel_coherence_platinum_eval.jsonl | jq .correct | grep true | wc -l"
echo "  # Divide by 1210 for accuracy percentage"
echo ""
echo "================================================================================"
