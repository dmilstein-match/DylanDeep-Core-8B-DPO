#!/usr/bin/env python3
"""
Download GSM8K datasets for training and evaluation.

Datasets:
1. GSM8K training set (for rollout collection and SFT)
2. GSM8K Platinum test set (for evaluation)
"""
import os
import json
from datasets import load_dataset


def main():
    print("=" * 80)
    print("GSM8K Dataset Download")
    print("=" * 80)

    # Create data directory
    os.makedirs("data", exist_ok=True)

    # 1. Download GSM8K training set
    print("\n[1/2] Downloading GSM8K training set...")
    print("  Source: gsm8k (main split)")
    gsm8k_train = load_dataset("gsm8k", "main", split="train")
    print(f"  Loaded {len(gsm8k_train)} training examples")

    # Save to JSONL
    train_path = "data/gsm8k_train.jsonl"
    print(f"  Saving to {train_path}...")
    with open(train_path, "w", encoding="utf-8") as f:
        for example in gsm8k_train:
            f.write(json.dumps(example) + "\n")
    print(f"  ✓ Saved {len(gsm8k_train)} examples to {train_path}")

    # 2. Download GSM8K Platinum test set
    print("\n[2/2] Downloading GSM8K Platinum test set...")
    print("  Source: madrylab/gsm8k-platinum (test split)")
    platinum_test = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
    print(f"  Loaded {len(platinum_test)} test examples")

    # Save to JSONL
    test_path = "data/gsm8k_platinum_test.jsonl"
    print(f"  Saving to {test_path}...")
    with open(test_path, "w", encoding="utf-8") as f:
        for example in platinum_test:
            f.write(json.dumps(example) + "\n")
    print(f"  ✓ Saved {len(platinum_test)} examples to {test_path}")

    # Summary
    print("\n" + "=" * 80)
    print("Download complete!")
    print("=" * 80)
    print(f"\nDatasets saved:")
    print(f"  Training:   {train_path} ({len(gsm8k_train)} examples)")
    print(f"  Evaluation: {test_path} ({len(platinum_test)} examples)")
    print(f"\nNext steps:")
    print(f"  1. Collect rollouts: python src/rl_training/collect_rollouts_abel_batched.py")
    print(f"  2. Build preferences: python src/rl_training/build_preferences_abel.py")
    print(f"  3. Train DPO: accelerate launch src/rl_training/train_dpo_coherence.py")
    print()


if __name__ == "__main__":
    main()
