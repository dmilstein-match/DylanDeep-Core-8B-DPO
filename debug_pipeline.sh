#!/bin/bash
set +e  # Don't exit on error, we want to see all issues

echo "================================================================================"
echo "Pipeline Diagnostics - Checking Each Step"
echo "================================================================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0

# Check 1: Data directory exists
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Check data directory"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "data" ]; then
    echo -e "${GREEN}✓ data/ directory exists${NC}"
    ls -lh data/ 2>/dev/null || echo "  (empty)"
else
    echo -e "${RED}✗ data/ directory missing${NC}"
    echo "  Creating data/ directory..."
    mkdir -p data
    ERRORS=$((ERRORS+1))
fi
echo ""

# Check 2: GSM8K training data
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Check GSM8K training data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "data/gsm8k_train.jsonl" ]; then
    COUNT=$(wc -l < data/gsm8k_train.jsonl)
    echo -e "${GREEN}✓ data/gsm8k_train.jsonl exists (${COUNT} examples)${NC}"
else
    echo -e "${RED}✗ data/gsm8k_train.jsonl missing${NC}"
    echo "  Run: python download_gsm8k.py"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Check 3: Model checkpoint
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Check base model checkpoint"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "checkpoints/abel_sft_merged" ]; then
    echo -e "${GREEN}✓ checkpoints/abel_sft_merged/ exists${NC}"

    # Check for config.json
    if [ -f "checkpoints/abel_sft_merged/config.json" ]; then
        echo -e "${GREEN}  ✓ config.json found${NC}"

        # Check for head_dim
        HAS_HEAD_DIM=$(python3 -c "
import json
with open('checkpoints/abel_sft_merged/config.json') as f:
    config = json.load(f)
print('head_dim' in config)
" 2>/dev/null)

        if [ "$HAS_HEAD_DIM" = "True" ]; then
            echo -e "${GREEN}  ✓ head_dim configured for vLLM${NC}"
        else
            echo -e "${YELLOW}  ⚠ head_dim missing (vLLM may fail)${NC}"
            echo "  Fix: Add head_dim to config.json"
        fi
    else
        echo -e "${RED}  ✗ config.json missing${NC}"
        ERRORS=$((ERRORS+1))
    fi
else
    echo -e "${RED}✗ checkpoints/abel_sft_merged/ missing${NC}"
    echo "  Need to create merged SFT checkpoint first"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Check 4: Rollout data
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Check rollout data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "data/abel_regime_w_rollouts.jsonl" ]; then
    COUNT=$(wc -l < data/abel_regime_w_rollouts.jsonl)
    echo -e "${GREEN}✓ data/abel_regime_w_rollouts.jsonl exists (${COUNT} rollouts)${NC}"

    # Validate rollout structure
    python3 << 'PYEOF'
import json
import sys

try:
    with open('data/abel_regime_w_rollouts.jsonl', 'r') as f:
        first_line = f.readline()
        if not first_line:
            print("\033[0;31m  ✗ Rollout file is empty\033[0m")
            sys.exit(1)

        rollout = json.loads(first_line)

        # Check structure
        required = ['question', 'gold_answer', 'trajectories']
        missing = [k for k in required if k not in rollout]
        if missing:
            print(f"\033[0;31m  ✗ Missing keys: {missing}\033[0m")
            sys.exit(1)

        # Check trajectories
        if not rollout['trajectories']:
            print("\033[0;31m  ✗ No trajectories in first rollout\033[0m")
            sys.exit(1)

        traj = rollout['trajectories'][0]
        traj_required = ['full_text', 'answer', 'reward', 'correct', 'arm_name']
        traj_missing = [k for k in traj_required if k not in traj]

        if traj_missing:
            print(f"\033[0;31m  ✗ Trajectories missing: {traj_missing}\033[0m")
            if 'arm_name' in traj_missing:
                print("\033[0;31m  ✗ CRITICAL: No arm_name field - rollouts are OLD format!\033[0m")
                print("\033[0;31m  ✗ Must regenerate rollouts with updated script\033[0m")
            sys.exit(1)

        print(f"\033[0;32m  ✓ Rollout structure valid\033[0m")
        print(f"\033[0;32m  ✓ First trajectory arm: {traj['arm_name']}\033[0m")
        print(f"\033[0;32m  ✓ Trajectories per rollout: {len(rollout['trajectories'])}\033[0m")

        # Check reward differentiation
        rewards = [t['reward'] for t in rollout['trajectories'] if t['correct']]
        if rewards:
            if len(set(rewards)) == 1:
                print(f"\033[1;33m  ⚠ All rewards identical: {rewards[0]:.4f}\033[0m")
                print(f"\033[1;33m  ⚠ May not generate coherence pairs\033[0m")
            else:
                min_r = min(rewards)
                max_r = max(rewards)
                print(f"\033[0;32m  ✓ Reward range: {min_r:.4f} to {max_r:.4f}\033[0m")

except Exception as e:
    print(f"\033[0;31m  ✗ Error reading rollouts: {e}\033[0m")
    sys.exit(1)
PYEOF

    if [ $? -ne 0 ]; then
        ERRORS=$((ERRORS+1))
    fi
else
    echo -e "${RED}✗ data/abel_regime_w_rollouts.jsonl missing${NC}"
    echo "  Run: python src/rl_training/collect_rollouts_abel_vllm.py --n_samples 500"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Check 5: Preference pairs
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Check preference pairs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "data/abel_coherence_preferences.jsonl" ]; then
    COUNT=$(wc -l < data/abel_coherence_preferences.jsonl)
    echo -e "${GREEN}✓ data/abel_coherence_preferences.jsonl exists (${COUNT} pairs)${NC}"

    # Count pair types
    python3 << 'PYEOF'
import json

try:
    with open('data/abel_coherence_preferences.jsonl', 'r') as f:
        prefs = [json.loads(line) for line in f]

    correctness = sum(1 for p in prefs if p.get('pair_type') == 'correctness')
    coherence = sum(1 for p in prefs if p.get('pair_type') == 'coherence')
    total = len(prefs)

    print(f"\033[0;32m  ✓ Total pairs: {total}\033[0m")
    print(f"\033[0;32m  ✓ Correctness pairs: {correctness} ({100*correctness/total:.1f}%)\033[0m")

    if coherence == 0:
        print(f"\033[0;31m  ✗ Coherence pairs: {coherence} (0%)\033[0m")
        print(f"\033[0;31m  ✗ CRITICAL: No coherence pairs generated!\033[0m")
        print(f"\033[0;31m  ✗ Check if rollouts have reward differentiation\033[0m")
        import sys
        sys.exit(1)
    else:
        print(f"\033[0;32m  ✓ Coherence pairs: {coherence} ({100*coherence/total:.1f}%)\033[0m")

except Exception as e:
    print(f"\033[0;31m  ✗ Error reading preferences: {e}\033[0m")
    import sys
    sys.exit(1)
PYEOF

    if [ $? -ne 0 ]; then
        ERRORS=$((ERRORS+1))
    fi
else
    echo -e "${RED}✗ data/abel_coherence_preferences.jsonl missing${NC}"
    echo "  Run: python src/rl_training/build_preferences_abel.py"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Summary
echo "================================================================================"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Ready for DPO training.${NC}"
    echo ""
    echo "Next step:"
    echo "  accelerate launch src/rl_training/train_dpo_coherence.py"
else
    echo -e "${RED}✗ Found $ERRORS issue(s) - fix them before training${NC}"
    echo ""
    echo "Quick fix commands:"
    echo ""
    echo "# If missing training data:"
    echo "  python download_gsm8k.py"
    echo ""
    echo "# If missing/old rollouts:"
    echo "  python src/rl_training/collect_rollouts_abel_vllm.py --n_samples 500"
    echo ""
    echo "# If missing preferences:"
    echo "  python src/rl_training/build_preferences_abel.py"
fi
echo "================================================================================"
echo ""

exit $ERRORS
