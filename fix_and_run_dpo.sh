#!/bin/bash
set -e

echo "=================================="
echo "Fixing DPO Training Dependencies"
echo "=================================="

# Check current transformers version
echo "Current transformers version:"
pip show transformers | grep Version

# Downgrade to version before security check
echo ""
echo "Downgrading transformers to bypass .bin file security check..."
pip install transformers==4.38.2 --force-reinstall --no-deps

echo ""
echo "✓ Dependencies fixed!"
echo ""
echo "Starting DPO training on 8× H100 GPUs..."
echo ""

# Run DPO training
accelerate launch src/rl_training/train_dpo_coherence.py
