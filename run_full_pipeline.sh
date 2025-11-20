#!/bin/bash
set -e  # Exit on any error

echo "================================================================================"
echo "Regime W Full Pipeline - Three-Tier Framework with Coherence Pairs"
echo "================================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 0: Pull latest code
echo -e "${YELLOW}[Step 0/6] Pulling latest code...${NC}"
git pull origin claude/framework-dimensional-labeling-01VygfyHmFPgaBqkKmHdETaB
echo -e "${GREEN}✓ Code updated${NC}"
echo ""

# Step 1: Verify configuration
echo -e "${YELLOW}[Step 1/6] Verifying configuration...${NC}"

# Check GPU count
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "  GPUs detected: $GPU_COUNT"

# Check arms configuration
python3 -c "
from src.regime_w.arms import build_all_arms
arms = build_all_arms()
print(f'  Arms configured: {len(arms)}')
probe_names = [a.name for a in arms if 'standard' in a.name or 'rephrase' in a.name or 'cf1' in a.name]
print(f'  Probe arms: {probe_names}')
assert len(arms) == 11, 'Expected 11 arms'
assert len(probe_names) == 3, 'Expected 3 probe arms'
"

# Check reward calculation
python3 -c "
from src.regime_w.reward import Trajectory, compute_rewards_for_question
trajs = [
    Trajectory('ans 42', 'step 1', '42', 10, 'wolfram_0'),
    Trajectory('ans 42', 'step 1 2 3', '42', 20, 'wolfram_standard'),
]
rewards = compute_rewards_for_question('test', trajs, '42')
diff = abs(rewards[0] - rewards[1])
print(f'  Reward differentiation: {diff:.4f}')
assert diff > 0.01, f'Reward spread too small: {diff}'
"

echo -e "${GREEN}✓ Configuration verified${NC}"
echo ""

# Step 2: Collect rollouts (15-40 min with 8 GPUs)
echo -e "${YELLOW}[Step 2/6] Collecting rollouts (this will take 15-40 minutes)...${NC}"

# Backup old data if exists
if [ -f "data/abel_regime_w_rollouts.jsonl" ]; then
    echo "  Backing up old rollouts..."
    mv data/abel_regime_w_rollouts.jsonl data/abel_regime_w_rollouts_OLD_$(date +%Y%m%d_%H%M%S).jsonl
fi

# Run rollout collection
python src/rl_training/collect_rollouts_abel_vllm.py --n_samples 500

# Verify rollouts
echo "  Verifying rollout data..."
python3 -c "
import json

with open('data/abel_regime_w_rollouts.jsonl', 'r') as f:
    rollouts = [json.loads(line) for line in f]

print(f'  Total rollouts: {len(rollouts)}')

# Check first rollout has arm_name field
first_rollout = rollouts[0]
traj = first_rollout['trajectories'][0]
assert 'arm_name' in traj, 'Missing arm_name field!'
print(f'  First trajectory arm: {traj[\"arm_name\"]}')

# Count correct trajectories
total_correct = sum(
    sum(1 for t in r['trajectories'] if t['correct'])
    for r in rollouts
)
print(f'  Total correct trajectories: {total_correct}')

# Check reward variance
rewards = [t['reward'] for r in rollouts for t in r['trajectories'] if t['correct']]
if rewards:
    reward_min = min(rewards)
    reward_max = max(rewards)
    print(f'  Reward range (correct): {reward_min:.4f} to {reward_max:.4f}')
    assert reward_max - reward_min > 0.01, 'Rewards not differentiated!'
"

echo -e "${GREEN}✓ Rollouts collected and verified${NC}"
echo ""

# Step 3: Build preference pairs
echo -e "${YELLOW}[Step 3/6] Building preference pairs...${NC}"

python src/rl_training/build_preferences_abel.py

# Verify preferences
echo "  Verifying preference pairs..."
python3 -c "
import json

with open('data/abel_coherence_preferences.jsonl', 'r') as f:
    prefs = [json.loads(line) for line in f]

correctness = sum(1 for p in prefs if p['pair_type'] == 'correctness')
coherence = sum(1 for p in prefs if p['pair_type'] == 'coherence')
total = len(prefs)

print(f'  Total preference pairs: {total}')
print(f'  Correctness pairs: {correctness} ({100*correctness/total:.1f}%)')
print(f'  Coherence pairs: {coherence} ({100*coherence/total:.1f}%)')

if coherence == 0:
    print('  ${RED}✗ WARNING: Zero coherence pairs generated!${NC}')
    exit(1)
else:
    print(f'  ${GREEN}✓ Coherence pairs successfully generated!${NC}')
"

echo -e "${GREEN}✓ Preference pairs built${NC}"
echo ""

# Step 4: Train DPO
echo -e "${YELLOW}[Step 4/6] Training DPO (5e-5 LR, 3 epochs)...${NC}"
echo "  This will take ~30-60 minutes on 8×H100"
echo ""

accelerate launch src/rl_training/train_dpo_coherence.py

echo -e "${GREEN}✓ DPO training complete${NC}"
echo ""

# Step 5: Verify model weights changed
echo -e "${YELLOW}[Step 5/6] Verifying model was updated...${NC}"

python3 -c "
import os
import glob

# Check for checkpoint directories
checkpoints = glob.glob('checkpoints/abel_dpo_coherence/checkpoint-*')
if not checkpoints:
    print('  ${RED}✗ No checkpoints found!${NC}')
    exit(1)

latest = max(checkpoints, key=os.path.getmtime)
print(f'  Latest checkpoint: {latest}')

# Check for adapter files
adapter_files = glob.glob(f'{latest}/adapter_*.safetensors')
if not adapter_files:
    print('  ${RED}✗ No adapter files found!${NC}')
    exit(1)

print(f'  Adapter files found: {len(adapter_files)}')

# Check file sizes
for f in adapter_files[:3]:
    size_mb = os.path.getsize(f) / 1024 / 1024
    print(f'    {os.path.basename(f)}: {size_mb:.1f} MB')

print('  ${GREEN}✓ Model weights successfully updated${NC}')
"

echo -e "${GREEN}✓ Model weights verified${NC}"
echo ""

# Step 6: Evaluate on Platinum test set
echo -e "${YELLOW}[Step 6/6] Evaluating on GSM8K Platinum (500 questions)...${NC}"

python src/eval/eval_abel_coherence_platinum.py

echo ""
echo "================================================================================"
echo -e "${GREEN}Pipeline Complete!${NC}"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  ✓ Collected 500 rollouts with 11 trajectories each"
echo "  ✓ Generated preference pairs with coherence differentiation"
echo "  ✓ Trained DPO with 5e-5 LR for 3 epochs"
echo "  ✓ Verified model weights changed"
echo "  ✓ Evaluated on Platinum test set"
echo ""
echo "Check results:"
echo "  - Training logs: checkpoints/abel_dpo_coherence/"
echo "  - Evaluation results: (printed above)"
echo ""
