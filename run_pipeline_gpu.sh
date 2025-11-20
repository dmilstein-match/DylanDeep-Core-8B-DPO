#!/bin/bash
# GPU Pipeline Execution Script
# Run this on a machine with GPU access and internet connectivity

set -e

echo "================================================================================"
echo "Looper-Math-Platinum Pipeline - GPU Execution"
echo "================================================================================"

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. This script requires GPU access."
    exit 1
fi

GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "✓ Detected $GPU_COUNT GPUs"

# Step 1: Install dependencies
echo ""
echo "[Step 1/6] Installing Python dependencies..."
pip install -q transformers datasets accelerate peft trl vllm sentencepiece torch bitsandbytes
echo "✓ Dependencies installed"

# Step 2: Use base Abel model (skip SFT training for now)
echo ""
echo "[Step 2/6] Using GAIR/Abel-7B-002 base model..."
echo "  (Skipping SFT training - using base model directly)"
MODEL_PATH="GAIR/Abel-7B-002"

# Step 3: Collect rollouts with Regime W (11 arms)
echo ""
echo "[Step 3/6] Collecting rollouts with Regime W..."
python src/rl_training/collect_rollouts_abel_vllm.py \
  --model_path "$MODEL_PATH" \
  --data_path data/gsm8k_train.jsonl \
  --out_path data/abel_regime_w_rollouts.jsonl \
  --n_samples 100

if [ ! -f "data/abel_regime_w_rollouts.jsonl" ]; then
    echo "ERROR: Rollout collection failed"
    exit 1
fi
echo "✓ Rollouts collected"

# Step 4: Build preference pairs
echo ""
echo "[Step 4/6] Building preference pairs..."
python src/rl_training/build_preferences_abel.py

if [ ! -f "data/abel_coherence_preferences.jsonl" ]; then
    echo "ERROR: Preference pair generation failed"
    exit 1
fi

# Verify we have coherence pairs
COHERENCE_COUNT=$(python3 -c "
import json
with open('data/abel_coherence_preferences.jsonl') as f:
    prefs = [json.loads(line) for line in f]
coherence = sum(1 for p in prefs if p.get('pair_type') == 'coherence')
print(coherence)
")

echo "✓ Generated $COHERENCE_COUNT coherence pairs"

if [ "$COHERENCE_COUNT" -eq 0 ]; then
    echo "WARNING: No coherence pairs generated - may need more rollouts"
fi

# Step 5: Train DPO
echo ""
echo "[Step 5/6] Training DPO with coherence pairs..."
accelerate launch src/rl_training/train_dpo_coherence.py

if [ ! -d "checkpoints/abel_dpo_coherence" ]; then
    echo "ERROR: DPO training failed - no checkpoint created"
    exit 1
fi
echo "✓ DPO training complete"

# Step 6: Evaluate
echo ""
echo "[Step 6/6] Evaluating on GSM8K Platinum..."
python src/eval/eval_abel_coherence_platinum.py

echo ""
echo "================================================================================"
echo "Pipeline Complete!"
echo "================================================================================"
echo "Results:"
echo "  - Rollouts: data/abel_regime_w_rollouts.jsonl"
echo "  - Preferences: data/abel_coherence_preferences.jsonl"
echo "  - Model checkpoint: checkpoints/abel_dpo_coherence/"
echo "  - Evaluation: Check output above"
echo ""
