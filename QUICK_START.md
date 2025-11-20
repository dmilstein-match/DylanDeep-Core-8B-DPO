# Quick Start: Complete Pipeline

## Maximum Safe H100 Speed Configuration ✓

All scripts are already optimized for **maximum safe throughput** on 8× H100 GPUs:

### vLLM Rollout Collection
- **Multi-question batching**: 16 questions × 11 arms = 176 sequences/batch
- **Optimized context**: max_model_len=2048 (vs 8192 default)
- **CUDA graphs**: Enabled for 20-30% speedup
- **Memory utilization**: 95% (76GB/80GB per GPU)
- **Chunked prefill**: Efficient variable-length batching
- **Expected time**: 10-15 min for 500 questions (vs 40 min original)

### DPO Training
- **Batch size**: 2 per device × 2 accum × 8 GPUs = 32 global
- **Precision**: BF16 (native H100 support)
- **TF32 matmul**: Enabled
- **Flash attention**: Enabled
- **Learning rate**: 5e-5 (optimized for 3 epochs)
- **Expected time**: 15-30 min

### Evaluation
- **Batch size**: 32 examples
- **Deterministic**: do_sample=False for reproducibility
- **H100 optimizations**: TF32, flash attention, batched inference
- **Expected time**: ~15 min for 1,210 examples

---

## Run Complete Pipeline

### Option 1: Automated Script (Recommended)
```bash
# Make script executable
chmod +x run_complete_pipeline.sh

# Run entire pipeline
./run_complete_pipeline.sh
```

### Option 2: Manual Steps
```bash
# 1. Download data
python download_gsm8k.py

# 2. Verify you have merged SFT model
ls -lh checkpoints/abel_sft_merged/

# 3. Generate rollouts (OPTIMIZED - 16 question batches)
python src/rl_training/collect_rollouts_abel_vllm_optimized.py \
  --model_path checkpoints/abel_sft_merged \
  --n_samples 500 \
  --question_batch_size 16

# 4. Build preferences
python src/rl_training/build_preferences_abel.py

# 5. Validate before DPO
./debug_pipeline.sh

# 6. Train DPO (8× H100 distributed)
accelerate launch src/rl_training/train_dpo_coherence.py

# 7. Evaluate
python src/eval/eval_abel_coherence_platinum.py

# 8. Check results
cat outputs/abel_coherence_platinum_eval.jsonl | jq .correct | grep true | wc -l
```

---

## Performance Tuning (If Needed)

### If You Want Even More Aggressive Batching
Edit `run_complete_pipeline.sh` line 69:
```bash
# Change from 16 to 24 or 32 (may approach memory limits)
--question_batch_size 32  # 32 questions × 11 arms = 352 sequences
```

### If You Hit OOM (Out of Memory)
Reduce batch size in vLLM script:
```bash
--question_batch_size 8  # Conservative: 8 × 11 = 88 sequences
```

Or reduce DPO batch size in `src/rl_training/train_dpo_coherence.py`:
```python
per_device_train_batch_size=1  # Line 133 (default is 2)
```

---

## Expected Timeline

| Step | Time (8× H100) |
|------|----------------|
| Download data | 1-2 min |
| Rollout collection (500Q) | 10-15 min |
| Build preferences | <1 min |
| DPO training | 15-30 min |
| Evaluation | 15 min |
| **TOTAL** | **~45-60 min** |

Full dataset (7,473 questions):
- Rollout collection: 2.5-4 hours
- DPO training: 30-45 min
- **Total: ~3-5 hours**

---

## Monitoring GPU Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Look for:
# - GPU Utilization: 90-98% during rollout/training
# - Memory Used: 70-76GB per GPU
# - Temperature: <85°C
```

---

## Current Optimizations Applied

✅ **Regime W Bug Fixes**
- Fixed comma-separated number extraction
- Rebalanced coherence weights (disabled placeholder s_cf)

✅ **vLLM Optimizations**
- Multi-question batching (16× larger batches)
- Reduced context size (8192 → 2048)
- CUDA graphs enabled
- 95% GPU memory utilization

✅ **Training Optimizations**
- BF16 precision (H100 native)
- TF32 matmul acceleration
- Flash attention kernels
- Non-reentrant gradient checkpointing

---

## After Completion

Results will be in:
- **Final checkpoint**: `checkpoints/abel_coherence_lora/`
- **Evaluation log**: `outputs/abel_coherence_platinum_eval.jsonl`

Calculate accuracy:
```bash
CORRECT=$(cat outputs/abel_coherence_platinum_eval.jsonl | jq .correct | grep true | wc -l)
TOTAL=$(cat outputs/abel_coherence_platinum_eval.jsonl | wc -l)
echo "Accuracy: $CORRECT / $TOTAL = $(echo "scale=2; 100*$CORRECT/$TOTAL" | bc)%"
```
