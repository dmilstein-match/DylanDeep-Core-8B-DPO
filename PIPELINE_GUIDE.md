# Training Pipeline Guide

## Pipeline Overview

This guide describes the complete training pipeline for GAIR/Abel-7B-002, optimized for 8× H100 GPUs with BF16 precision.

### Architecture
```
Base: GAIR/Abel-7B-002 (80.44% GSM8K baseline)
  ↓
Phase 1: Supervised Fine-Tuning (SFT) - baseline correctness
  ↓
Phase 2: Rollout Collection (diverse prompt variants)
  ↓
Phase 3: Preference Pair Construction (correctness-first strategy)
  ↓
Phase 4: DPO Training (optimize for robust reasoning)
  ↓
Phase 5: Evaluation on GSM8K
```

### Key Design Principles
1. **SFT establishes baseline** - Train on GSM8K with tutoring prompts
2. **Multi-variant sampling** - Generate diverse reasoning trajectories
3. **Correctness-first preferences** - DPO teaches correctness, then reasoning quality
4. **LoRA throughout** - Efficient fine-tuning, frozen base weights

---

## Phase 1: Supervised Fine-Tuning

**Goal:** Train Abel base model with LoRA on GSM8K dataset using tutoring-style prompts.

### Commands
```bash
cd DylanDeep-Core-8B-DPO

# Run SFT training
python src/baseline_sft/train_sft_abel.py

# Output: checkpoints/abel_sft_lora/
```

### Expected Results
- Training time: ~2-3 hours on single H100
- Checkpoint: `checkpoints/abel_sft_lora/`
- Expected accuracy: ~82-85% on GSM8K (vs 80.44% baseline)

---

## Phase 2: Rollout Collection

**Goal:** Generate diverse reasoning trajectories using SFT policy with prompt perturbation variants.

### Commands
```bash
# Multi-GPU parallel rollout
python src/rl_training/collect_rollouts_abel_vllm_optimized.py \
  --model_path checkpoints/abel_sft_merged \
  --n_samples 500 \
  --question_batch_size 16

# Output: data/rollouts.jsonl
```

### Trajectory Generation

Each question generates multiple trajectories using different prompt variants that encourage different reasoning styles (direct, reflective, deliberative).

**Per-trajectory metrics:**
- Answer correctness
- Reasoning quality score
- Length penalty

### Expected Output
```
Loaded 500 rollout records
Each record has multiple trajectories with:
  - full_text (complete solution)
  - answer (extracted answer)
  - num_tokens (length)
  - reward (composite quality score)
  - correct (boolean flag)
```

---

## Phase 3: Preference Pair Construction

**Goal:** Build correctness + quality preference pairs from rollouts.

### Commands
```bash
python src/rl_training/build_preferences_abel.py

# Reads:  data/rollouts.jsonl
# Writes: data/preferences.jsonl
```

### Preference Strategy

1. **Correctness pairs** (correct vs incorrect trajectories)
   - Teaches model to get the right answer
2. **Quality pairs** (high vs low quality among correct trajectories)
   - Teaches model to reason robustly

### Expected Output
```
Created preference pairs for DPO training
```

---

## Phase 4: DPO Training

**Goal:** Fine-tune SFT checkpoint with preference pairs via DPO.

### Commands
```bash
# 8× H100 DPO training
accelerate launch src/rl_training/train_dpo_coherence.py

# Output: checkpoints/abel_dpo_lora/
```

### Training Configuration

- **Policy model:** Abel + SFT LoRA (trainable)
- **Reference model:** Abel + SFT LoRA (frozen)
- **Per-device batch:** 2
- **Gradient accumulation:** 2
- **Global batch:** 2 × 2 × 8 = 32
- **Learning rate:** 5e-6
- **Precision:** BF16

### Expected Results
- Training time: ~30-45 minutes on 8× H100
- Checkpoint: `checkpoints/abel_dpo_lora/`
- Expected improvement: Better reasoning quality while maintaining correctness

---

## Phase 5: Evaluation on GSM8K

**Goal:** Compare Base → SFT → DPO performance on GSM8K benchmark.

### Commands
```bash
# Evaluate final checkpoint
python src/eval/eval_abel_coherence_platinum.py

# Output: outputs/eval.jsonl
```

### Analysis
```bash
# Compute accuracy
CORRECT=$(cat outputs/eval.jsonl | jq .correct | grep true | wc -l)
TOTAL=$(cat outputs/eval.jsonl | wc -l)
echo "Accuracy: $CORRECT / $TOTAL = $(echo "scale=2; 100*$CORRECT/$TOTAL" | bc)%"
```

### Expected Performance
| Model | GSM8K Accuracy |
|-------|----------------|
| Base Abel-7B-002 | 80.44% |
| Abel + SFT | 82-85% |
| Abel + SFT + DPO | 83-87% |

---

## Complete Pipeline Execution

### Full Workflow
```bash
# 1. SFT baseline training (~2-3 hours)
python src/baseline_sft/train_sft_abel.py

# 2. Rollout collection (10-15 minutes on 8× H100)
python src/rl_training/collect_rollouts_abel_vllm_optimized.py \
  --model_path checkpoints/abel_sft_merged \
  --n_samples 500

# 3. Build preference pairs (<1 minute)
python src/rl_training/build_preferences_abel.py

# 4. DPO training (30-45 minutes on 8× H100)
accelerate launch src/rl_training/train_dpo_coherence.py

# 5. Evaluate checkpoint (~10 minutes)
python src/eval/eval_abel_coherence_platinum.py
```

**Total runtime:** ~4-5 hours end-to-end

---

## Checkpoint Structure

```
checkpoints/
├── abel_sft_lora/              # Phase 1: SFT baseline
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
│
└── abel_dpo_lora/              # Phase 4: DPO optimized
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files

data/
├── gsm8k_train.jsonl          # GSM8K training data
├── rollouts.jsonl             # Phase 2: Trajectory rollouts
└── preferences.jsonl          # Phase 3: Preference pairs

outputs/
└── eval.jsonl                 # Phase 5: Evaluation results
```

---

## Troubleshooting

### Issue: No quality pairs generated
**Cause:** All correct trajectories have identical scores
**Fix:** Threshold configured in `build_preferences_abel.py`

### Issue: OOM during multi-GPU rollout
**Cause:** Loading multiple models simultaneously
**Fix:** Reduce `--question_batch_size` parameter

### Issue: SFT training diverges
**Cause:** Learning rate too high
**Fix:** Lower learning rate in `train_sft_abel.py`

### Issue: DPO requires merged model
**Cause:** Old scripts expected merged checkpoints
**Fix:** Updated scripts now load base + LoRA directly (no merge needed)

---

## References

- **Base Model:** [GAIR/Abel-7B-002](https://huggingface.co/GAIR/Abel-7B-002)
- **Dataset:** [GSM8K](https://huggingface.co/datasets/gsm8k)
- **Architecture:** SFT → Prompt Perturbations → DPO (preference-based optimization)
