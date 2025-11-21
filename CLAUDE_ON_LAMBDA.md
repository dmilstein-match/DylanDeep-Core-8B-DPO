# Getting Claude on Lambda for DPO Training & Evaluation

## Quick Answer

You have 3 options to run DPO training and 8-shot GSM8K evaluation on your Lambda instance:

### Option 1: Automated Script (Fastest)

```bash
# SSH into Lambda
ssh -i ~/path/to/key.pem ubuntu@192.222.54.90

# Navigate to project
cd looper-math-platinum

# Pull latest code
git pull origin claude/lambda-dpo-setup-01YMc384M9YPuHBnQucMPddk

# Run complete pipeline
./run_dpo_and_eval_gsm8k.sh
```

This script automatically handles:
- ✅ LoRA merging
- ✅ vLLM rollout collection (15-40 min)
- ✅ Preference pair building
- ✅ DPO training with 8x H100 (15-30 min)
- ✅ 8-shot evaluation with majority voting (40-60 min)

**Total runtime: ~90-130 minutes**

### Option 2: Install Claude Code CLI on Lambda

```bash
# SSH into Lambda
ssh -i ~/path/to/key.pem ubuntu@192.222.54.90

# Install Node.js (if not already installed)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Navigate to project
cd looper-math-platinum

# Start Claude session
claude-code
```

Then inside Claude Code, you can ask:
> "Run DPO training and 8-shot evaluation on GSM8K for the coherence checkpoint"

### Option 3: Manual Step-by-Step Commands

```bash
# SSH into Lambda
ssh -i ~/path/to/key.pem ubuntu@192.222.54.90
cd looper-math-platinum
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Step 1: Merge LoRA (if needed)
python src/merge_lora.py \
  --base_model GAIR/Abel-7B-002 \
  --lora_path checkpoints/abel_sft_lora \
  --output_path checkpoints/abel_sft_merged

# Step 2: Install vLLM (one-time)
pip install vllm==0.6.3.post1

# Step 3: Collect rollouts (15-40 min)
python src/rl_training/collect_rollouts_abel_vllm.py \
  --model_path checkpoints/abel_sft_merged \
  --data_path data/gsm8k_train.jsonl \
  --out_path data/abel_regime_w_rollouts.jsonl \
  --n_samples 500

# Step 4: Build preference pairs
python -m src.rl_training.build_preferences_abel

# Step 5: Train DPO (15-30 min)
accelerate launch src/rl_training/train_dpo_coherence.py

# Step 6: Run 8-shot evaluation (40-60 min)
python eval_8shot_maudlin0.py
```

## Understanding the 8-Shot Evaluation

The `eval_8shot_maudlin0.py` script:
- Uses **maudlin_0** configuration (temp=0.2, optimal for GSM8K)
- Generates **8 answers per question** with sampling
- Takes **majority vote** across the 8 samples
- Evaluates on full **GSM8K test set** (1,319 examples)
- Expected accuracy: **80%+** (official HuggingFace leaderboard quality)

## For HuggingFace Leaderboard Submission

After evaluation completes:

1. **Check results:**
   ```bash
   # View final accuracy
   python -c 'import json; results = [json.loads(l) for l in open("outputs/abel_coherence_8shot_maudlin0.jsonl")]; print(f"Accuracy: {100 * sum(r[\"correct\"] for r in results) / len(results):.2f}%")'

   # View sample predictions
   head -n 5 outputs/abel_coherence_8shot_maudlin0.jsonl | jq .
   ```

2. **Verify checkpoint:**
   - Model: `checkpoints/abel_coherence_lora` (LoRA adapter)
   - Base: `GAIR/Abel-7B-002`

3. **Submit to HuggingFace:**
   - Upload merged model to HF Hub
   - Report 8-shot accuracy from evaluation
   - Include configuration: temp=0.2, majority voting across 8 samples

## Monitoring Progress

### View GPU utilization:
```bash
watch -n 1 nvidia-smi
```

### Monitor training logs:
```bash
# For DPO training
tail -f <output-from-accelerate-command>

# For evaluation
tail -f outputs/abel_coherence_8shot_maudlin0.jsonl
```

### Check intermediate results:
```bash
# Count rollouts generated
wc -l data/abel_regime_w_rollouts.jsonl

# Count preference pairs
wc -l data/abel_coherence_preferences.jsonl

# Check evaluation progress
wc -l outputs/abel_coherence_8shot_maudlin0.jsonl
```

## Troubleshooting

### "No module named 'src'"
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### vLLM import error
```bash
pip install --upgrade vllm==0.6.3.post1
```

### Out of memory during training
```bash
# Already handled by device_map="auto" in scripts
# If issues persist, reduce batch size in train_dpo_coherence.py
```

### CUDA out of memory during 8-shot eval
```bash
# Reduce num_return_sequences in eval_8shot_maudlin0.py from 8 to 4
# Run eval twice with different seeds and combine results
```

## Cost Estimate (Lambda H100 at ~$10/hour)

| Phase | Time | Cost |
|-------|------|------|
| LoRA Merge | 1 min | $0.17 |
| vLLM Rollouts | 15-40 min | $2.50-6.70 |
| Preference Build | 2 min | $0.33 |
| DPO Training | 15-30 min | $2.50-5.00 |
| 8-Shot Eval | 40-60 min | $6.70-10.00 |
| **Total** | **~90-130 min** | **~$12-22** |

## Remote Development with Claude Code (Advanced)

If you want to use Claude Code from your local machine but execute on Lambda:

1. **Set up SSH config** (`~/.ssh/config`):
   ```
   Host lambda-h100
       HostName 192.222.54.90
       User ubuntu
       IdentityFile ~/path/to/key.pem
       ForwardAgent yes
   ```

2. **Use VS Code Remote SSH:**
   - Install "Remote - SSH" extension
   - Connect to `lambda-h100`
   - Install Claude Code extension in remote environment

3. **Start Claude Code:**
   - Open terminal in VS Code (connected to Lambda)
   - Run `claude-code` in project directory

## Next Steps

After successful evaluation:
1. ✅ Verify 80%+ accuracy on GSM8K
2. ✅ Upload checkpoint to HuggingFace Hub
3. ✅ Submit to official leaderboard
4. ✅ Document Regime W coherence methodology

## Questions?

If you need Claude to help with any of these steps:
- Install Claude Code CLI on Lambda (Option 2)
- Or run the automated script (Option 1) and report any errors
- Or ask specific questions about the pipeline
