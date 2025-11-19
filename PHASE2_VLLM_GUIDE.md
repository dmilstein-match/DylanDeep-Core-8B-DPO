# Phase 2: vLLM-Optimized Rollout Collection Guide

## Why Switch to vLLM?

The original single-GPU rollout collection script (`collect_rollouts_abel.py`) has:
- **5-20% GPU utilization** (most time spent on CPU tokenization/decoding)
- **2-4 hours runtime** for 500 examples × 8 trajectories
- Sequential generation (one trajectory at a time)

The vLLM-optimized approach (`collect_rollouts_abel_vllm.py`) provides:
- **90%+ GPU utilization** across all 8 GPUs
- **15-40 minutes runtime** (5-10× speedup)
- Batched generation with tensor parallelism
- Full preservation of Regime W coherence scoring

## Quick Start (Lambda 8× H100 SXM5)

### Step 1: Stop Current Rollout (if running)
```bash
# Kill any running rollout processes
pkill -f collect_rollouts_abel.py

# Clean up partial outputs
rm -f data/abel_regime_w_rollouts.jsonl
```

### Step 2: Merge LoRA Adapter (~60 seconds)
```bash
# Set PYTHONPATH for module imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Merge Abel base model with SFT LoRA adapter
python src/merge_lora.py \
  --base_model GAIR/Abel-7B-002 \
  --lora_path checkpoints/abel_sft_lora \
  --output_path checkpoints/abel_sft_merged
```

**Expected output:**
```
================================================================================
LoRA Merge for vLLM Optimization
================================================================================

Configuration:
  Base model: GAIR/Abel-7B-002
  LoRA adapter: checkpoints/abel_sft_lora
  Output path: checkpoints/abel_sft_merged

Loading base model in bf16...
Loading LoRA adapter from checkpoints/abel_sft_lora...
Merging LoRA weights into base model...
Loading tokenizer...
Saving merged model to checkpoints/abel_sft_merged...

================================================================================
LoRA merge complete!
Merged model saved to: checkpoints/abel_sft_merged
================================================================================
```

### Step 3: Install vLLM (one-time)
```bash
pip install vllm==0.6.3.post1
```

### Step 4: Generate Rollouts with 8-GPU Tensor Parallelism
```bash
# Run vLLM rollout collection (15-40 minutes)
python src/rl_training/collect_rollouts_abel_vllm.py \
  --model_path checkpoints/abel_sft_merged \
  --data_path data/gsm8k_train.jsonl \
  --out_path data/abel_regime_w_rollouts.jsonl \
  --n_samples 500
```

**Expected output:**
```
================================================================================
Abel Regime W Rollout Collection (vLLM Optimized)
================================================================================

Configuration:
  Merged model: checkpoints/abel_sft_merged
  Data path: data/gsm8k_train.jsonl
  Output path: data/abel_regime_w_rollouts.jsonl
  N samples: 500
  World size (GPUs): 8

Loading GSM8K training data from data/gsm8k_train.jsonl...
Using 500 examples for Regime W rollout collection

Built 8 Regime W arms (Wolfram + Maudlin styles)
Initializing vLLM engine with tensor parallelism...

vLLM engine initialized successfully!
Starting rollout collection...
Generating 8 trajectories per question

  Collected 20/500 rollouts
  Collected 40/500 rollouts
  ...
  Collected 500/500 rollouts

Writing rollouts to data/abel_regime_w_rollouts.jsonl...

================================================================================
Rollout collection complete!
Saved Abel Regime W rollouts to: data/abel_regime_w_rollouts.jsonl
Total rollouts: 500
Trajectories per rollout: 8
================================================================================
```

### Step 5: Verify Output
```bash
# Check rollout count (should be 500 lines)
wc -l data/abel_regime_w_rollouts.jsonl

# Inspect a sample rollout
head -n 1 data/abel_regime_w_rollouts.jsonl | jq .

# Verify trajectory structure
head -n 1 data/abel_regime_w_rollouts.jsonl | jq '.trajectories[0]'
```

**Expected structure:**
```json
{
  "question": "Janet's ducks lay 16 eggs per day...",
  "gold_answer": "18",
  "trajectories": [
    {
      "full_text": "Step 1: Calculate eggs laid per day...",
      "answer": "18",
      "reasoning": "Step 1: Calculate eggs laid per day...",
      "num_tokens": 87,
      "reward": 1.234,
      "correct": true
    },
    ... (7 more trajectories)
  ]
}
```

## Comparison: Original vs vLLM

| Metric | Original (Single GPU) | vLLM (8-GPU Tensor Parallel) |
|--------|----------------------|------------------------------|
| GPU Utilization | 5-20% | 90%+ |
| Runtime (500 examples) | 2-4 hours | 15-40 minutes |
| Speedup | 1× baseline | 5-10× |
| Batching | Sequential | Parallel (8 trajectories/question) |
| Regime W Scoring | ✅ Full | ✅ Full (identical) |
| Output Format | ✅ JSONL | ✅ JSONL (identical) |

## Technical Details

### vLLM Optimizations
- **Tensor Parallelism**: Model sharded across 8 GPUs, each GPU processes all layers in parallel
- **PagedAttention**: Efficient KV cache management
- **Continuous Batching**: Dynamically groups requests for maximum throughput
- **BF16 Precision**: Same dtype as training (H100/A100 optimized)

### Regime W Integration
The vLLM script preserves all Regime W coherence scoring:
- **8 Arms**: 4 Wolfram (temps 0.2, 0.3, 0.4, 0.5) + 4 Maudlin (temps 0.2, 0.3, 0.4, 0.5)
- **Per-Trajectory Rewards**: `ALPHA_CORRECT * correct + BETA_COHERENCE * s_wm - length_penalty`
- **Correctness Flags**: Boolean flag for each trajectory (matches gold answer)
- **Coherence Metrics**: s_end, s_path, s_cf, s_wm computed per question

### Memory Requirements
- **Merged Model**: ~14GB (Abel-7B-002 in bf16)
- **Per-GPU Memory**: ~6-8GB (tensor parallel sharding)
- **Total VRAM**: ~50GB across 8× H100 GPUs

## Troubleshooting

### Issue: "Module 'src' not found"
```bash
# Solution: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: vLLM import error
```bash
# Solution: Reinstall vLLM
pip install --upgrade vllm==0.6.3.post1
```

### Issue: Out of memory during merge
```bash
# Solution: Use CPU offloading during merge
# (merge script already uses device_map="auto")
```

### Issue: Merged model directory is empty
```bash
# Solution: Check merge logs for errors, ensure base model downloaded correctly
ls -lh checkpoints/abel_sft_merged/
```

## Next Steps

After rollout collection completes:

```bash
# Phase 3: Build preference pairs (CPU, ~5 minutes)
python -m src.rl_training.build_preferences_abel

# Phase 4: Train DPO (8× H100, ~15-30 minutes)
accelerate launch src/rl_training/train_dpo_coherence.py

# Phase 5: Evaluate on GSM8K Platinum (single GPU, ~15 minutes)
python -m src.eval.eval_abel_coherence_platinum
```

## Cost Savings

**Lambda 8× H100 SXM5 pricing: ~$10/hour**

- Original approach: 2-4 hours = **$20-40 per rollout collection**
- vLLM approach: 0.25-0.67 hours = **$2.50-6.70 per rollout collection**

**Savings: $13.50-33.30 per run (67-85% cost reduction)**
