# Looper-Math-Platinum

Private research project for training Abel-7B-002 with coherence-aware mathematical reasoning using:
- Baseline supervised fine-tuning (SFT) with LoRA
- **Regime W coherence module** (proprietary 11-armed framework with diagnostic probes)
- Correctness-first coherence DPO reinforcement learning
- Evaluation on GSM8K Platinum benchmark

## 🚀 Quick Start

**For full deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)**

## Overview

This project trains a high-performance mathematical reasoning model by:

1. **Fine-tuning Abel-7B-002** (~80% GSM8K baseline) with LoRA on full-precision bf16
2. **Generating multi-trajectory rollouts** using Regime W's **11-armed framework** (8 variance + 3 diagnostic probes) to evaluate reasoning coherence
3. **Building preference pairs** that prioritize correctness first, then coherence quality
4. **Training with DPO** to optimize for both correct answers and clear explanations (5e-5 LR, 3 epochs)
5. **Evaluating on GSM8K Platinum** (1,210 cleaned test examples)

### Regime W Framework (Three-Tier Architecture)

**Tier 1: Training Mechanics**
- Learning rate: 5e-5 (10x higher for actual weight updates)
- Training epochs: 3 (up from 1)

**Tier 2: Structured Probe Arms (11 total)**
- 8 variance arms: `wolfram_0-3`, `maudlin_0-3` (temps 0.2-0.5)
- 3 diagnostic probes:
  - `wolfram_standard` (A): Canonical clear reasoning
  - `wolfram_rephrase` (A′): Paraphrase stability check
  - `maudlin_cf1` (CF1): Counterfactual robustness probe

**Tier 3: Upgraded Coherence Metrics**
- **s_end**: Final answer agreement (pairwise)
- **s_path**: Reasoning coherence (40% length + 60% token overlap via Jaccard similarity)
- **s_cf**: Counterfactual robustness (answer consensus across arms, replaces constant 0.5)
- **s_wm**: Weighted combined score (0.4×s_end + 0.3×s_path + 0.3×s_cf)

## Project Structure

```
looper-math-platinum/
  data/                           # GSM8K data and generated rollouts
  checkpoints/
    abel_sft_lora/                # Phase 1: Abel SFT checkpoint
    abel_coherence_lora/          # Phase 4: Coherence-optimized checkpoint
  outputs/                        # Evaluation results (JSONL)
  src/
    baseline_sft/
      train_sft_abel.py           # SFT training on Abel-7B-002
    rl_training/
      collect_rollouts_abel.py    # Multi-trajectory rollout generation
      build_preferences_abel.py   # Correctness-first preference construction
      train_dpo_coherence.py      # Coherence DPO training
    regime_w/                     # PRIVATE: Multi-path coherence engine
      arms.py                     # 11-arm framework (8 variance + 3 diagnostic probes)
      scoring.py                  # s_end, s_path, s_cf, s_wm metrics (upgraded)
      reward.py                   # Combined correctness + coherence rewards
    eval/
      eval_abel_sft_platinum.py   # Evaluate SFT checkpoint on Platinum
      eval_abel_coherence_platinum.py  # Evaluate coherence checkpoint on Platinum
    legacy/                       # Archived DeepSeek-R1 experiments
```

## H100 Setup (Required for Multi-GPU Training)

This pipeline is optimized for **8× H100 SXM5** instances with:
- BF16/FP16 automatic dtype detection
- TF32 matmul acceleration
- Flash attention kernels
- Memory-efficient attention
- Multi-GPU distributed training via Accelerate

### Initial Setup on Lambda
```bash
# Clone repo and install dependencies
git clone <your-private-repo>
cd looper-math-platinum
pip install -r requirements.txt

# Configure Accelerate for 8× H100 GPUs
accelerate config --config_file .accelerate_config.yaml
```

The `.accelerate_config.yaml` file is pre-configured for 8 GPUs with bf16 mixed precision.

## Main Pipeline Commands

**Training scripts** (SFT, DPO) use `accelerate launch` for multi-GPU distribution.  
**Inference scripts** (rollout, eval) use single-process `python -m` to prevent JSONL corruption while still benefiting from H100 optimizations.

### Phase 1: Supervised Fine-Tuning
```bash
# Train Abel-7B-002 with LoRA on GSM8K training set (distributed across 8× H100)
accelerate launch src/baseline_sft/train_sft_abel.py

# Output: checkpoints/abel_sft_lora/
# Batch size: 2 per device, gradient accumulation: 2 (effective batch=32 globally)
```

### Phase 2: Regime W Rollout Collection

**RECOMMENDED: vLLM-Optimized Approach (5-10× faster)**

```bash
# Step 2a: Merge LoRA adapter with base model (takes ~60 seconds)
python src/merge_lora.py \
  --base_model GAIR/Abel-7B-002 \
  --lora_path checkpoints/abel_sft_lora \
  --output_path checkpoints/abel_sft_merged

# Step 2b: Install vLLM (one-time setup)
pip install vllm==0.6.3.post1

# Step 2c: Generate rollouts with 8-GPU tensor parallelism (15-40 minutes)
python src/rl_training/collect_rollouts_abel_vllm.py \
  --model_path checkpoints/abel_sft_merged \
  --data_path data/gsm8k_train.jsonl \
  --out_path data/abel_regime_w_rollouts.jsonl \
  --n_samples 500

# Output: data/abel_regime_w_rollouts.jsonl
# Each trajectory includes: full_text, answer, reasoning, num_tokens, reward, correct flag
# 90%+ GPU utilization, 5-10× faster than single-GPU approach
```

**ALTERNATIVE: Single-GPU Approach (slower, deprecated)**

```bash
# Generate 8 trajectories per question using Regime W arms
# Computes per-trajectory rewards (correctness + coherence + length penalty)
# Note: Single-process to avoid JSONL output corruption
python -m src.rl_training.collect_rollouts_abel

# Configurable parameters:
python -m src.rl_training.collect_rollouts_abel \
  --n_samples 5000 \
  --data_path data/gsm8k_train.jsonl \
  --sft_path checkpoints/abel_sft_lora \
  --out_path data/abel_regime_w_rollouts.jsonl

# Warning: This approach has low GPU utilization (5-20%) and takes 2-4 hours.
# Use the vLLM-optimized approach above for production workloads.
```

### Phase 3: Preference Pair Construction
```bash
# Build preference pairs with correctness-first strategy:
#   1. Correct vs incorrect trajectories
#   2. High-coherence vs low-coherence among correct trajectories
python -m src.rl_training.build_preferences_abel

# Output: data/abel_coherence_preferences.jsonl
# Each pair includes: question, chosen, rejected, pair_type (correctness/coherence)
```

### Phase 4: Coherence DPO Training
```bash
# Train DPO on preference pairs to optimize coherence (distributed across 8× H100)
accelerate launch src/rl_training/train_dpo_coherence.py

# Output: checkpoints/abel_coherence_lora/
# Batch size: 2 per device, gradient accumulation: 2 (effective batch=32 globally)
```

### Phase 5: Evaluation on GSM8K Platinum (Batched, H100-Optimized)
```bash
# Evaluate SFT checkpoint with batched generation (batch_size=32, ~15 min for 1,210 examples)
# Note: Single-process to avoid JSONL output corruption
python -m src.eval.eval_abel_sft_platinum

# Evaluate coherence-optimized checkpoint
python -m src.eval.eval_abel_coherence_platinum

# Outputs:
#   outputs/abel_sft_platinum_eval.jsonl
#   outputs/abel_coherence_platinum_eval.jsonl
# (Per-example logs with predictions, correctness flags, and metadata)
# Both scripts use deterministic generation (do_sample=False) for reproducibility
# H100 optimizations (TF32, flash attention, batched inference) still apply
```

## Base Model

**GAIR/Abel-7B-002**
- Strong mathematical reasoning foundation (~80% GSM8K baseline)
- 7B parameters (H100-optimized for full bf16 training)
- Pre-trained on mathematical datasets
- Fully private (hosted on Lambda infrastructure)

## Key Design Decisions

### Correctness-First Preference Strategy
Unlike traditional DPO that only optimizes style, this pipeline teaches:
1. **First: Get the answer right** (correct vs incorrect pairs)
2. **Then: Explain clearly** (high vs low coherence among correct answers)

This ensures the model prioritizes correctness before optimizing explanation quality.

### Regime W Coherence Scoring (Three-Tier Upgrade)

The proprietary Regime W module evaluates each trajectory across multiple dimensions:

**Updated Metrics (Tier 3):**
- **s_end**: Final answer agreement (pairwise comparison across all arms)
- **s_path**: Reasoning path coherence
  - 40% length variance (low variance = high coherence)
  - 60% token overlap (Jaccard similarity across reasoning paths)
  - Explicitly rewards A ↔ A′ stability (paraphrase robustness)
- **s_cf**: Counterfactual robustness (NEW: real implementation)
  - Measures answer consensus across all arms including CF1 probe
  - 1.0 = perfect agreement, 0.0 = complete disagreement
  - Replaces previous constant 0.5 placeholder
- **s_wm**: Weighted coherence metric
  - Formula: `0.4 * s_end + 0.3 * s_path + 0.3 * s_cf`

**Structured Probe Arms (Tier 2):**
- **A (wolfram_standard)**: Canonical clear reasoning baseline
- **A′ (wolfram_rephrase)**: Tests stability under problem restatement
- **CF1 (maudlin_cf1)**: Probes for edge cases and reasoning traps

Per-trajectory rewards combine: `ALPHA_CORRECT * correct + BETA_COHERENCE * s_wm - length_penalty`

### Full-Precision Training with H100 Optimizations
Abel training uses bf16/fp16 (no quantization) on H100 GPUs for maximum quality, with:
- **Automatic dtype detection**: BF16 on H100/A100, FP16 fallback for older GPUs
- **TensorFloat-32 (TF32) matmul**: ~2× speedup on matrix operations
- **Flash Attention**: Memory-efficient attention kernels
- **Batched evaluation**: 32 examples per batch for ~30× faster inference vs sequential
- **Multi-GPU training**: Automatic distribution across 8× H100 via Accelerate

This is a significant upgrade from legacy DeepSeek experiments that used 4-bit quantization.

## Workflow

### Development Environment
- **Code editing**: Replit (synced with private GitHub repo)
- **Training**: Lambda H100 GPU instances
- **Version control**: GitHub (private repository)

### Deployment Flow
1. Develop and test code in Replit
2. Push to GitHub
3. Pull on Lambda GPU instance
4. Execute training pipeline (Phases 1-4)
5. Run evaluation (Phase 5)
6. Analyze results in `outputs/` JSONL files

## IP Protection

The `src/regime_w/` module contains proprietary coherence scoring logic. Keep this directory private and reference it publicly only as "proprietary multi-path coherence signal."

## Legacy Experiments

The `src/legacy/` directory contains archived scripts from DeepSeek-R1-Distill-Llama-8B experiments (November 2025). These used:
- 4-bit quantization (vs full bf16 for Abel)
- Simple reward-based preference pairs (vs correctness-first strategy)
- Lower baseline performance (~50% vs ~80%)

Legacy scripts are preserved for reference but are not part of the active pipeline.

## Requirements

See `requirements.txt` for dependencies:
- `transformers`, `datasets`, `trl`, `peft` (Hugging Face ecosystem)
- `torch`, `accelerate` (PyTorch training)
- `bitsandbytes` (only used in legacy DeepSeek experiments)

Install on Lambda:
```bash
pip install -r requirements.txt
```

## Quick Start on Lambda H100

**See [DEPLOYMENT.md](DEPLOYMENT.md) for complete step-by-step instructions.**

```bash
# 1. Launch 8× H100 SXM5 instance and SSH in
ssh -i ~/path/to/key.pem ubuntu@<H100-IP>

# 2. Clone and setup
git clone <your-private-repo>
cd looper-math-platinum
pip install -r requirements.txt
accelerate config --config_file .accelerate_config.yaml

# 3. Download datasets (NEW)
python download_gsm8k.py
# Creates: data/gsm8k_train.jsonl, data/gsm8k_platinum_test.jsonl

# 4. Run complete pipeline
# Training scripts use accelerate for multi-GPU distribution (8× H100)
accelerate launch src/baseline_sft/train_sft_abel.py

# Rollout collection with 11-arm framework (NEW)
python src/rl_training/collect_rollouts_abel_batched.py \
  --model_path checkpoints/abel_sft_merged_fixed \
  --n_samples 500

# Build preferences and train DPO with upgraded config
python src/rl_training/build_preferences_abel.py
accelerate launch src/rl_training/train_dpo_coherence.py  # 5e-5 LR, 3 epochs

# Evaluate
python src/eval/eval_abel_coherence_platinum.py

# 5. Analyze results
cat outputs/abel_coherence_platinum_eval.jsonl | jq .correct | grep true | wc -l
```

## Performance Benchmarks (8× H100 SXM5)

- **SFT Training**: ~25-30 minutes (full GSM8K train set, effective batch=32 globally, 8 GPUs)
- **Rollout Collection (vLLM)**: ~15-40 minutes (8 trajectories × 500 examples, 8-GPU tensor parallel)
- **Rollout Collection (legacy)**: ~2-4 hours (single GPU, deprecated)
- **DPO Training**: ~15-30 minutes (preference pairs, effective batch=32 globally, 8 GPUs)
- **Platinum Evaluation**: ~15 minutes (1,210 examples, batched generation, single GPU)
