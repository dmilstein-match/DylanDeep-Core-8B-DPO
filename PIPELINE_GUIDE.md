# Abel SFT → Regime W → DPO Pipeline Guide

## Pipeline Overview

This guide describes the complete **SFT → Regime W → DPO** training pipeline for GAIR/Abel-7B-002, optimized for 8× H100 GPUs with BF16 precision.

### Architecture
```
Base: GAIR/Abel-7B-002 (80.44% GSM8K baseline)
  ↓
Phase 1: Supervised Fine-Tuning (SFT) - baseline correctness
  ↓
Phase 2: Regime W Rollout Collection (8-armed bandit coherence scoring)
  ↓
Phase 3: Preference Pair Construction (correctness-first, then coherence)
  ↓
Phase 4: DPO Coherence Training (shapes reasoning clarity)
  ↓
Phase 5: Evaluation on GSM8K Platinum
```

### Key Design Principles (from Regime W spec)
1. **SFT establishes baseline** - Train on GSM8K with tutoring prompts
2. **Regime W multi-path coherence** - Sample multiple trajectories from SFT policy, measure agreement/consistency
3. **Correctness-first preferences** - DPO teaches correctness, then coherence
4. **Threshold lowered to 0.01** - Captures coherence variance from length differences
5. **LoRA throughout** - Efficient fine-tuning, frozen base weights

---

## Phase 1: Supervised Fine-Tuning

**Goal:** Train Abel base model with LoRA on GSM8K dataset using tutoring-style prompts.

### Lambda Commands
```bash
# On Lambda H100 instance (single GPU or multi-GPU via accelerate)
cd Looper-Math-Platinum

# Run SFT training
python src/baseline_sft/train_sft_abel.py

# Output: checkpoints/abel_sft_lora/
```

### Expected Results
- Training time: ~2-3 hours on single H100
- Checkpoint: `checkpoints/abel_sft_lora/`
- Expected accuracy: ~82-85% on GSM8K (vs 80.44% baseline)

---

## Phase 2: Regime W Rollout Collection

**Goal:** Generate 8 trajectories per question using SFT policy with Regime W coherence scoring.

### Lambda Commands
```bash
# Multi-GPU parallel rollout (19-minute runtime!)
python src/rl_training/collect_rollouts_abel_multigpu.py \
  --base_model GAIR/Abel-7B-002 \
  --lora_path checkpoints/abel_sft_lora \
  --data_path data/gsm8k_train.jsonl \
  --out_path data/abel_regime_w_rollouts.jsonl \
  --n_samples 500 \
  --n_gpus 8

# Output: data/abel_regime_w_rollouts.jsonl
```

### Regime W Scoring Details (Per Your Spec)

Each question gets 8 trajectories (8-armed bandit):

**Question-level coherence metrics:**
- **s_end**: End-state agreement (fraction of arms agreeing on final answer)
- **s_path**: Path consistency (1 - normalized length variance)
- **s_cf**: Counterfactual coherence (internal consistency checks)
- **s_wm**: Weighted coherence = 0.5×s_end + 0.3×s_path + 0.2×s_cf

**Per-trajectory reward formula:**
```python
base = ALPHA_CORRECT × correctness  # 1.0 if correct, 0.0 if wrong
coherence_term = BETA_COHERENCE × s_wm  # same for all trajectories on this question
length_penalty = LENGTH_PENALTY_PER_TOKEN × max(0, num_tokens - 512)
agreement_bonus = FULL_AGREEMENT_BONUS (if all correct trajectories agree)

reward = base + coherence_term + agreement_bonus - length_penalty
```

**Key insight from your spec:** 
> "Regime W analyzes this set of trajectories and produces agreement scores over final answers, similarity scores over intermediate reasoning paths, and internal consistency / counterfactual checks across trajectories."

Coherence (s_wm) is computed **per-question** (measures group alignment), but variance comes from:
- Correctness differences between trajectories
- Length differences (creates small reward deltas)
- Agreement bonus differences

### Expected Output
```
Loaded 500 rollout records
Each record has 8 trajectories with:
  - full_text (complete solution)
  - answer (extracted answer)
  - num_tokens (length)
  - reward (correctness + coherence + length penalty)
  - correct (boolean flag)
```

---

## Phase 3: Preference Pair Construction

**Goal:** Build correctness + coherence preference pairs from Regime W rollouts.

### Lambda Commands
```bash
python src/rl_training/build_preferences_abel.py

# Reads:  data/abel_regime_w_rollouts.jsonl
# Writes: data/abel_coherence_preferences.jsonl
```

### Preference Strategy (From Your Spec)
> "Construct preference pairs by selecting, for each question, a 'better' and 'worse' trajectory based on reward."

1. **Correctness pairs** (correct vs incorrect trajectories)
   - Teaches model to get the right answer
2. **Coherence pairs** (high vs low coherence among correct trajectories)
   - Teaches model to explain clearly
   - Threshold: 0.01 (captures length-based reward differences)

### Expected Output
```
Created 2,535 correctness pairs
Created 150-300 coherence pairs (depending on length variance)
```

---

## Phase 4: DPO Coherence Training

**Goal:** Fine-tune SFT checkpoint with coherence preferences via DPO.

### Lambda Commands
```bash
# 8× H100 DPO training
torchrun --nproc_per_node=8 src/rl_training/train_dpo_coherence.py

# Output: checkpoints/abel_dpo_coherence_lora/
```

### Training Configuration (From Your Spec)
> "We perform a preference-based RL fine-tuning stage (DPO/GRPO-style) on top of the LoRA-SFT policy"

- **Policy model:** Abel + SFT LoRA (trainable)
- **Reference model:** Abel + SFT LoRA (frozen)
- **Per-device batch:** 2
- **Gradient accumulation:** 2
- **Global batch:** 2 × 2 × 8 = 32
- **Learning rate:** 5e-6
- **Precision:** BF16

### Expected Results
- Training time: ~30-45 minutes on 8× H100
- Checkpoint: `checkpoints/abel_dpo_coherence_lora/`
- Expected improvement: Better coherence while maintaining correctness

---

## Phase 5: Evaluation on GSM8K Platinum

**Goal:** Compare Base → SFT → DPO performance on GSM8K Platinum benchmark.

### Lambda Commands
```bash
# Evaluate SFT checkpoint
python src/eval/eval_abel_sft_platinum.py

# Evaluate DPO checkpoint
python src/eval/eval_abel_dpo_platinum.py

# Outputs:
#   outputs/abel_sft_platinum_eval.jsonl
#   outputs/abel_dpo_coherence_platinum_eval.jsonl
```

### Analysis
```bash
# Compute accuracy
python -c "
import json
correct = sum(1 for line in open('outputs/abel_sft_platinum_eval.jsonl') 
              if json.loads(line)['correct'])
total = sum(1 for _ in open('outputs/abel_sft_platinum_eval.jsonl'))
print(f'SFT Accuracy: {100*correct/total:.2f}%')
"

# Same for DPO checkpoint
python -c "
import json
correct = sum(1 for line in open('outputs/abel_dpo_coherence_platinum_eval.jsonl') 
              if json.loads(line)['correct'])
total = sum(1 for _ in open('outputs/abel_dpo_coherence_platinum_eval.jsonl'))
print(f'DPO Accuracy: {100*correct/total:.2f}%')
"
```

### Expected Performance
| Model | GSM8K Platinum Accuracy |
|-------|------------------------|
| Base Abel-7B-002 | 80.44% |
| Abel + SFT | 82-85% |
| Abel + SFT + DPO | 83-87% |

---

## Complete Pipeline Execution

### Full Lambda Workflow
```bash
# 1. SFT baseline training (~2-3 hours)
python src/baseline_sft/train_sft_abel.py

# 2. Regime W rollouts (19 minutes on 8× H100)
python src/rl_training/collect_rollouts_abel_multigpu.py \
  --base_model GAIR/Abel-7B-002 \
  --lora_path checkpoints/abel_sft_lora \
  --n_samples 500 \
  --n_gpus 8

# 3. Build preference pairs (<1 minute)
python src/rl_training/build_preferences_abel.py

# 4. DPO coherence training (30-45 minutes on 8× H100)
torchrun --nproc_per_node=8 src/rl_training/train_dpo_coherence.py

# 5. Evaluate both checkpoints (~10 minutes each)
python src/eval/eval_abel_sft_platinum.py
python src/eval/eval_abel_dpo_platinum.py
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
└── abel_dpo_coherence_lora/    # Phase 4: DPO coherence
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files

data/
├── gsm8k_train.jsonl              # GSM8K training data
├── abel_regime_w_rollouts.jsonl   # Phase 2: Regime W rollouts
└── abel_coherence_preferences.jsonl  # Phase 3: Preference pairs

outputs/
├── abel_sft_platinum_eval.jsonl        # Phase 5: SFT evaluation
└── abel_dpo_coherence_platinum_eval.jsonl  # Phase 5: DPO evaluation
```

---

## Troubleshooting

### Issue: No coherence pairs generated
**Cause:** All correct trajectories have identical rewards  
**Fix:** Threshold lowered to 0.01 in `build_preferences_abel.py` (line 82)

### Issue: OOM during multi-GPU rollout
**Cause:** Loading 8 models simultaneously  
**Fix:** Reduce `--n_gpus` or use smaller batch processing

### Issue: SFT training diverges
**Cause:** Learning rate too high  
**Fix:** Lower learning rate in `train_sft_abel.py`

### Issue: DPO requires merged model
**Cause:** Old scripts expected merged checkpoints  
**Fix:** Updated scripts now load base + LoRA directly (no merge needed)

---

## Regime W Privacy Notice

From your spec:
> "Regime W is model-agnostic and used only during training; its implementation details and exact aggregation scheme are kept private."

The reward formula is exposed in `src/regime_w/reward.py`, but the detailed coherence metric implementations in `src/regime_w/scoring.py` remain proprietary.

At inference time (from your spec):
> "At inference time, no Regime W calls are made. Looper-Math-Platinum is used as a standard single-call model: question → model → chain-of-thought + final answer"

---

## References

- **Base Model:** [GAIR/Abel-7B-002](https://huggingface.co/GAIR/Abel-7B-002)
- **Dataset:** [GSM8K](https://huggingface.co/datasets/gsm8k)
- **Regime W:** Proprietary multi-path coherence evaluation framework
- **Architecture:** SFT → Regime W → DPO (preference-based RL)
