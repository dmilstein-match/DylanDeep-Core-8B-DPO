# Regime W Framework Deployment Guide (Lambda H100)

## 🎯 What's New (Three-Tier Framework)

This deployment includes all three tiers of the Regime W framework upgrade:

### **Tier 1: Fix DPO Training Mechanics**
- ✅ Learning rate: `5e-6` → `5e-5` (10x higher for actual weight updates)
- ✅ Training epochs: `1` → `3` (sufficient gradient signal)
- **Result**: Model weights will actually change during DPO training

### **Tier 2: Add Structured Probe Arms**
- ✅ **11 total arms** (8 variance + 3 diagnostic probes):
  - `wolfram_0-3` (temps 0.2-0.5, seeds 42-45)
  - `maudlin_0-3` (temps 0.2-0.5, seeds 100-103)
  - `wolfram_standard` (A probe: canonical reasoning, seed=1000)
  - `wolfram_rephrase` (A′ probe: paraphrase stability, seed=1001)
  - `maudlin_cf1` (CF1 probe: counterfactual robustness, seed=2000)
- **Result**: Interpretable variance with diagnostic visibility

### **Tier 3: Upgrade Metrics**
- ✅ **Real `s_cf`**: Measures answer agreement (replaces constant 0.5)
- ✅ **Improved `s_path`**: 40% length coherence + 60% token overlap (Jaccard)
- ✅ **Trajectory-specific quality scoring**: Creates reward differentiation among correct trajectories
  - 40% conciseness (shorter is better - Occam's razor)
  - 60% alignment with canonical probes (A, A')
  - Enables coherence pair generation
- ✅ **Standardized format**: `Answer: <number>` (backward compatible with `####`)
- **Result**: Metrics aligned with dimensional reasoning + coherence pairs now possible

---

## 🚀 Full Pipeline Execution (Lambda H100)

### **Prerequisites**
- Lambda 8× H100 SXM5 instance
- Checkpoint exists: `checkpoints/abel_sft_merged_fixed` (or `abel_sft_merged`)
- Python environment with dependencies installed

---

### **Step 1: Deploy Latest Code**

```bash
# SSH into Lambda instance
ssh -i ~/path/to/key.pem ubuntu@<H100-IP>

# Navigate to project
cd looper-math-platinum-8b

# Pull latest framework improvements
git fetch origin
git checkout claude/framework-dimensional-labeling-01VygfyHmFPgaBqkKmHdETaB
git pull

# Verify you're on the right commit
git log --oneline -3
# Should show:
#   d78974a Add GSM8K dataset download script
#   3557bf7 Implement all three tiers of Regime W framework improvements
#   ad408ef Transitioned from Plan to Build mode
```

---

### **Step 2: Download Datasets**

```bash
# Activate virtual environment
source venv/bin/activate

# Download GSM8K training + Platinum test sets
python download_gsm8k.py

# Expected output:
#   ✓ data/gsm8k_train.jsonl (7,473 examples)
#   ✓ data/gsm8k_platinum_test.jsonl (1,210 examples)

# Verify downloads
ls -lh data/
```

---

### **Step 3: Collect Rollouts with 11-Arm Setup**

**Option A: Batched (faster, single GPU)**

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python src/rl_training/collect_rollouts_abel_batched.py \
  --model_path checkpoints/abel_sft_merged_fixed \
  --data_path data/gsm8k_train.jsonl \
  --out_path data/abel_regime_w_rollouts.jsonl \
  --n_samples 500 \
  --batch_size 11

# Runtime: ~30-60 minutes
# Output: data/abel_regime_w_rollouts.jsonl
# Each question generates 11 trajectories (8 variance + 3 probe arms)
```

**Option B: vLLM (fastest, 8-GPU tensor parallel)**

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python src/rl_training/collect_rollouts_abel_vllm.py \
  --model_path checkpoints/abel_sft_merged \
  --data_path data/gsm8k_train.jsonl \
  --out_path data/abel_regime_w_rollouts.jsonl \
  --n_samples 500

# Runtime: ~15-40 minutes
# Output: data/abel_regime_w_rollouts.jsonl
# 90%+ GPU utilization across all 8 GPUs
```

**Verify Rollouts:**

```bash
# Check file size and record count
wc -l data/abel_regime_w_rollouts.jsonl
# Should show: 500 lines (one per question)

# Inspect first rollout
head -1 data/abel_regime_w_rollouts.jsonl | python3 -m json.tool | head -30

# Verify 11 trajectories per question
python3 -c "
import json
with open('data/abel_regime_w_rollouts.jsonl') as f:
    sample = json.loads(f.readline())
    print(f'Trajectories per question: {len(sample[\"trajectories\"])}')
    print(f'Arms: {[t.get(\"arm_name\", \"?\") for t in sample[\"trajectories\"]]}')
"
# Expected: 11 trajectories with arm names including wolfram_standard, wolfram_rephrase, maudlin_cf1
```

---

### **Step 4: Build Preference Pairs**

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python src/rl_training/build_preferences_abel.py

# Output: data/abel_coherence_preferences.jsonl
# Strategy: Correctness-first (correct vs incorrect, then high vs low coherence)

# Verify preferences
wc -l data/abel_coherence_preferences.jsonl

# Check pair type distribution
python3 -c "
import json
correctness = coherence = 0
with open('data/abel_coherence_preferences.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        if rec.get('pair_type') == 'correctness':
            correctness += 1
        elif rec.get('pair_type') == 'coherence':
            coherence += 1
print(f'Correctness pairs: {correctness}')
print(f'Coherence pairs: {coherence}')
print(f'Total: {correctness + coherence}')
"
```

---

### **Step 5: Train DPO with New Config**

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Launch DPO training with 5e-5 LR + 3 epochs
accelerate launch src/rl_training/train_dpo_coherence.py

# Config details:
#   Learning rate: 5e-5 (10x higher than before)
#   Epochs: 3 (up from 1)
#   Per-device batch: 2
#   Gradient accumulation: 2
#   Global effective batch: 32 (across 8 GPUs)

# Runtime: ~15-30 minutes
# Output: checkpoints/abel_dpo_coherence_lora/

# Monitor training progress
# Look for:
#   1. Loss decreasing across 3 epochs
#   2. Gradient norms non-zero
#   3. Checkpoint saves after each epoch
```

**What to Watch For:**

```
# GOOD SIGNS:
Step 10/XX | Loss: 0.XXX | LR: 5e-05 | Grad Norm: 0.XXX  ← Non-zero gradients!
Step 20/XX | Loss: 0.YYY | LR: 5e-05 | Grad Norm: 0.YYY  ← Loss decreasing
...
Epoch 1/3 complete | Saving checkpoint...
Epoch 2/3 complete | Saving checkpoint...
Epoch 3/3 complete | Saving checkpoint...

# BAD SIGNS:
Grad Norm: 0.000  ← Zero gradients = no learning
Loss staying constant across steps
```

---

### **Step 6: Evaluate Coherence Model**

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python src/eval/eval_abel_coherence_platinum.py

# Runtime: ~15 minutes
# Output: outputs/abel_coherence_platinum_eval.jsonl

# Calculate accuracy
cat outputs/abel_coherence_platinum_eval.jsonl | jq .correct | grep true | wc -l

# Expected: Accuracy improvement vs baseline SFT
# Baseline SFT: ~80% (Abel-7B-002 is already strong)
# Target coherence boost: +2-5% from DPO optimization
```

---

### **Step 7: Compare Baseline vs Coherence**

```bash
# Eval baseline SFT (if not already done)
python src/eval/eval_abel_sft_platinum.py

# Compare results
echo "=== Baseline SFT ==="
cat outputs/abel_sft_platinum_eval.jsonl | jq .correct | grep true | wc -l
echo "/1210 examples"

echo "=== Coherence DPO ==="
cat outputs/abel_coherence_platinum_eval.jsonl | jq .correct | grep true | wc -l
echo "/1210 examples"
```

---

## 🔍 Validation Checklist

### **After Rollout Collection (Step 3):**
- [ ] `data/abel_regime_w_rollouts.jsonl` exists
- [ ] File has 500 lines (one per question)
- [ ] Each question has **11 trajectories** (not 8!)
- [ ] Trajectories include new probe arms: `wolfram_standard`, `wolfram_rephrase`, `maudlin_cf1`
- [ ] All answers use new `Answer: <number>` format

### **After Preference Building (Step 4):**
- [ ] `data/abel_coherence_preferences.jsonl` exists
- [ ] File has 3,000-5,000 preference pairs
- [ ] Mix of correctness pairs (majority) and coherence pairs
- [ ] Each pair has `chosen`, `rejected`, `question`, `pair_type`

### **After DPO Training (Step 5):**
- [ ] Training completed 3 full epochs
- [ ] Loss decreased across epochs (not flat!)
- [ ] Gradient norms were non-zero throughout
- [ ] Checkpoint saved: `checkpoints/abel_dpo_coherence_lora/`
- [ ] Adapter files present: `adapter_config.json`, `adapter_model.safetensors`

### **After Evaluation (Step 6):**
- [ ] `outputs/abel_coherence_platinum_eval.jsonl` exists
- [ ] File has 1,210 lines (one per test example)
- [ ] Accuracy > baseline SFT (target: +2-5% improvement)
- [ ] Predictions logged for manual inspection

---

## 📊 Expected Metrics

### **Rollout Metrics (from Step 3):**
```
Per question:
- 11 trajectories
- s_end (answer agreement): 0.5-1.0 (higher = more consensus)
- s_path (reasoning coherence): 0.3-0.9 (higher = more similar paths)
- s_cf (counterfactual robustness): 0.0-1.0 (higher = CF1 agrees with A/A')
- Rewards: ~0.5-1.2 per trajectory
```

### **Preference Pair Distribution:**
```
Correctness pairs: ~70-80% (correct answer wins)
Coherence pairs: ~20-30% (trajectory quality differentiation among correct)
  - Quality score based on: conciseness (40%) + probe alignment (60%)
  - Requires >0.01 reward difference threshold
Total pairs: ~3,000-5,000
```

### **DPO Training Loss:**
```
Epoch 1: Loss ~0.6-0.7 → ~0.5
Epoch 2: Loss ~0.5 → ~0.4
Epoch 3: Loss ~0.4 → ~0.35-0.40
```

### **Platinum Accuracy:**
```
Baseline SFT: ~80% (970/1210)
Coherence DPO: ~82-85% (990-1030/1210)
Target improvement: +2-5% absolute
```

---

## 🐛 Troubleshooting

### **Issue: "No module named 'datasets'"**
```bash
pip install datasets transformers peft trl accelerate
```

### **Issue: Rollouts only have 8 trajectories (not 11)**
```bash
# Check arms.py
python3 -c "from src.regime_w.arms import build_all_arms; print(len(build_all_arms()))"
# Should output: 11

# If output is 8, you're on old code - pull latest:
git pull origin claude/framework-dimensional-labeling-01VygfyHmFPgaBqkKmHdETaB
```

### **Issue: DPO training loss stays flat**
```bash
# Check learning rate in logs
# Should see: "LR: 5e-05" (not 5e-06!)

# If LR is wrong, verify train_dpo_coherence.py line 124:
grep "learning_rate" src/rl_training/train_dpo_coherence.py
# Should show: learning_rate=5e-5
```

### **Issue: s_cf always 0.5 in rollouts**
```bash
# Verify scoring.py has real s_cf (not constant)
head -20 src/regime_w/scoring.py
# Should NOT see: "return 0.5" as only line in s_cf_for_question

# Check for updated s_cf:
grep -A 5 "def s_cf_for_question" src/regime_w/scoring.py
# Should see logic for answer agreement, not just "return 0.5"
```

### **Issue: Zero coherence pairs despite having correct trajectories**
```bash
# This was fixed by adding trajectory-specific quality scoring
# Old rollouts (before fix) will still produce 0 coherence pairs

# Check if rollouts have arm_name field:
head -1 data/abel_regime_w_rollouts.jsonl | python3 -c "
import json, sys
rec = json.loads(sys.stdin.read())
has_arm_name = 'arm_name' in rec['trajectories'][0]
print(f'Has arm_name field: {has_arm_name}')
if not has_arm_name:
    print('⚠️  Old rollouts detected - must regenerate with updated scripts')
"

# If arm_name is missing, regenerate rollouts:
# Backup old rollouts first
mv data/abel_regime_w_rollouts.jsonl data/abel_regime_w_rollouts_OLD.jsonl

# Regenerate with trajectory quality scoring
python src/rl_training/collect_rollouts_abel_vllm.py --n_samples 500
```

---

## 📦 What Gets Created

```
data/
  gsm8k_train.jsonl                    # 7,473 training examples
  gsm8k_platinum_test.jsonl            # 1,210 test examples
  abel_regime_w_rollouts.jsonl         # 500 questions × 11 trajectories
  abel_coherence_preferences.jsonl     # 3,000-5,000 preference pairs

checkpoints/
  abel_dpo_coherence_lora/             # Coherence-optimized LoRA adapter
    adapter_config.json
    adapter_model.safetensors
    checkpoint-{epoch}/                # Intermediate checkpoints

outputs/
  abel_coherence_platinum_eval.jsonl   # Coherence model evaluation results
  abel_sft_platinum_eval.jsonl         # Baseline SFT evaluation results
```

---

## 🎓 What Changed from Previous Runs

### **Old Stack (what didn't work):**
- 8 arms only
- `s_cf = 0.5` (constant)
- Simple `s_path` (length variance only)
- LR `5e-6`, 1 epoch
- Result: ~3.8k pairs but no model movement

### **New Stack (what we have now):**
- **11 arms** (8 variance + 3 probes)
- **Real `s_cf`** (answer agreement)
- **Improved `s_path`** (length + token overlap)
- **LR `5e-5`**, **3 epochs**
- **Reproducible seeds** per arm
- **Answer: format** standardization

**Expected result:** Actual model weight updates + diagnostic visibility into which coherence dimensions changed.

---

## ✅ Success Criteria

1. **Rollout collection**: 500 questions × 11 trajectories = 5,500 total trajectories
2. **Preference pairs**: 3,000-5,000 pairs (70% correctness, 30% coherence)
3. **DPO training**: Loss decreases across 3 epochs, non-zero gradients
4. **Evaluation**: Accuracy improves +2-5% over baseline SFT
5. **Metrics variance**: `s_cf` and `s_path` show non-constant values across rollouts

If all five criteria pass, the three-tier framework is working correctly! 🎉
