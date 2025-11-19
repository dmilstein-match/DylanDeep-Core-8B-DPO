# Looper-Math-Platinum

Private research project for training Abel-7B-002 with coherence-aware mathematical reasoning using:
- Baseline supervised fine-tuning (SFT) with LoRA
- Regime W coherence module (proprietary multi-armed bandit approach)
- Correctness-first coherence DPO reinforcement learning
- Evaluation on GSM8K Platinum benchmark

## Overview

This project trains a high-performance mathematical reasoning model by:

1. **Fine-tuning Abel-7B-002** (~80% GSM8K baseline) with LoRA on full-precision bf16
2. **Generating multi-trajectory rollouts** using Regime W's 8-armed bandit to evaluate reasoning coherence
3. **Building preference pairs** that prioritize correctness first, then coherence quality
4. **Training with DPO** to optimize for both correct answers and clear explanations
5. **Evaluating on GSM8K Platinum** (1,210 cleaned test examples)

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
      arms.py                     # 8-arm bandit (Wolfram + Maudlin strategies)
      scoring.py                  # s_end, s_path, s_cf, s_wm metrics
      reward.py                   # Combined correctness + coherence rewards
    eval/
      eval_abel_sft_platinum.py   # Evaluate SFT checkpoint on Platinum
      eval_abel_coherence_platinum.py  # Evaluate coherence checkpoint on Platinum
    legacy/                       # Archived DeepSeek-R1 experiments
```

## Main Pipeline Commands

All training executes on Lambda H100 GPU instances with full bf16 precision.

### Phase 1: Supervised Fine-Tuning
```bash
# Train Abel-7B-002 with LoRA on GSM8K training set
python -m src.baseline_sft.train_sft_abel

# Output: checkpoints/abel_sft_lora/
```

### Phase 2: Regime W Rollout Collection
```bash
# Generate 8 trajectories per question using Regime W arms
# Computes per-trajectory rewards (correctness + coherence + length penalty)
python -m src.rl_training.collect_rollouts_abel

# Output: data/abel_regime_w_rollouts.jsonl
# Each trajectory includes: full_text, answer, reasoning, num_tokens, reward, correct flag
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
# Train DPO on preference pairs to optimize coherence
# Loads Abel + SFT LoRA as policy model
python -m src.rl_training.train_dpo_coherence

# Output: checkpoints/abel_coherence_lora/
```

### Phase 5: Evaluation on GSM8K Platinum
```bash
# Evaluate SFT checkpoint
python -m src.eval.eval_abel_sft_platinum

# Evaluate coherence-optimized checkpoint
python -m src.eval.eval_abel_coherence_platinum

# Outputs:
#   outputs/abel_sft_platinum_eval.jsonl
#   outputs/abel_coherence_platinum_eval.jsonl
# (Per-example logs with predictions, correctness flags, and metadata)
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

### Regime W Coherence Scoring
The proprietary Regime W module evaluates each trajectory across multiple dimensions:
- **s_end**: Final answer agreement across arms
- **s_path**: Reasoning path consistency
- **s_cf**: Counterfactual robustness
- **s_wm**: Weighted coherence metric

Per-trajectory rewards combine: `ALPHA_CORRECT * correct + BETA_COHERENCE * s_wm - length_penalty`

### Full-Precision Training
Abel training uses bf16 (no quantization) on H100 GPUs for maximum quality, unlike legacy DeepSeek experiments that used 4-bit quantization.

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

## Next Steps

1. Clone repo on Lambda H100 instance
2. Install dependencies
3. Run Phase 1 (SFT) to create base checkpoint
4. Execute Phases 2-4 to collect rollouts and train coherence
5. Evaluate on GSM8K Platinum (Phase 5)
6. Analyze per-example results in `outputs/` for further improvements
