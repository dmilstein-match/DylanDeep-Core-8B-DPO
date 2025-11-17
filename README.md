# Looper-Math-Platinum-8B

Private project for fine-tuning DeepSeek-R1-Distill-Llama-8B on GSM8K with:
- Baseline supervised fine-tuning (SFT)
- Regime W coherence module (proprietary multi-armed bandit approach)
- PPO reinforcement learning optimization

## Project Structure

```
looper-math-platinum-8b/
  data/                     # GSM8K train/dev/test splits (generated on Lambda)
  checkpoints/              # SFT and PPO model checkpoints
  src/
    baseline_sft/           # Phase 1: Supervised fine-tuning
      prepare_data.py       # Download and split GSM8K dataset
      train_sft.py          # SFT training on DeepSeek-R1-Distill-Llama-8B
    rl_training/            # Phase 3-4: Rollouts and PPO training
      collect_rollouts.py   # Generate trajectories with Regime W rewards
      train_ppo.py          # PPO RL training with coherence signals
    regime_w/               # PRIVATE: Multi-path coherence engine
      arms.py               # 8-arm bandit (Wolfram + Maudlin strategies)
      scoring.py            # s_end, s_path, s_cf, s_wm metrics
      reward.py             # Combined correctness + coherence rewards
      demo.py               # Sanity check demo
    eval/                   # Phase 5: Evaluation
      eval_platinum.py      # Test on GSM8K-Platinum
  requirements.txt
  README.md
```

## Workflow

### Development Environment
- **Code editing**: Replit (this repo synced with GitHub)
- **Training**: Lambda GPU instance (A10/A100 with 24-40GB VRAM)
- **Version control**: GitHub (private repo)

### Training Pipeline

**Phase 1: Baseline SFT**
```bash
# On Lambda GPU:
python -m src.baseline_sft.prepare_data
python -m src.baseline_sft.train_sft
```

**Phase 2: Regime W Setup** (private coherence module - implement after SFT)

**Phase 3: Rollout Collection** (generate trajectories with Regime W rewards)
```bash
python -m src.rl_training.collect_rollouts
```

**Phase 4: PPO RL Training** (optimize with coherence signals)
```bash
python -m src.rl_training.train_ppo
```

**Phase 5: Evaluation**
```bash
python -m src.eval.eval_platinum
```

## Base Model

**DeepSeek-R1-Distill-Llama-8B**
- Already RL-trained for reasoning & math
- Strong chain-of-thought capabilities
- 8B parameters (fits on single A10/A100)
- MIT-style permissive license
- Fully private (hosted on your Lambda box)

## IP Protection

The `regime_w/` module contains proprietary coherence scoring logic. Keep this directory private and reference it publicly only as "proprietary multi-path coherence signal."

## Next Steps

1. Push this repo to GitHub
2. Clone on Lambda GPU instance
3. Install dependencies: `pip install -r requirements.txt`
4. Run Phase 1: SFT training
5. Implement Regime W modules after SFT baseline is working
