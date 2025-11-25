# DylanDeep-Core-8B-DPO

A research framework for improving mathematical reasoning in language models through deliberative prompt perturbations and preference optimization.

## Key Finding

**Deliberative reasoning improves generalization.** Using prompt perturbation variants that encourage careful verification leads to more robust reasoning that transfers better to held-out test sets.

## Results

| Configuration | GSM8K Accuracy |
|---------------|----------------|
| Base (greedy) | 79.08% |
| + SFT | 84.46% |
| + DPO | **84.84%** |

Evaluation uses multi-sample majority voting.

## Method Overview

1. **Prompt Perturbation Variants**: Generate diverse reasoning trajectories using different prompt strategies
2. **Trajectory Selection**: Rank solutions using quality metrics
3. **Preference Pair Construction**: Create training pairs that prioritize correctness and reasoning quality
4. **DPO Training**: Optimize for robust mathematical reasoning

## Repository Structure

```
src/
├── baseline_sft/      # Supervised fine-tuning
├── regime_w/          # Multi-arm coherence framework
│   ├── arms.py        # Prompt strategy definitions
│   ├── scoring.py     # Coherence metrics
│   └── reward.py      # Reward computation
├── rl_training/       # DPO training pipeline
└── eval/              # Evaluation scripts
```

## Models

- [DylanDeep-Core-8B-DPO](https://huggingface.co/dylxnmyl/DylanDeep-Core-8B-DPO) - Final model (84.84%)
- [DylanDeep-Core-8B](https://huggingface.co/dylxnmyl/DylanDeep-Core-8B) - SFT baseline (84.46%)

## Quick Start

```bash
pip install -r requirements.txt

# SFT training
accelerate launch src/baseline_sft/train_sft_abel.py

# Collect rollouts with multi-arm prompts
python src/rl_training/collect_rollouts_abel_vllm.py

# Build preference pairs
python src/rl_training/build_preferences_abel.py

# DPO training
accelerate launch src/rl_training/train_dpo_coherence.py
```

## Citation

```bibtex
@misc{dylandeep2024deliberative,
  title={Deliberative Reasoning Improves Mathematical Generalization in Language Models},
  author={DylanDeep},
  year={2024},
  url={https://github.com/dmilstein-match/DylanDeep-Core-8B-DPO}
}
```

## License

CC BY-NC-ND 4.0 + LLaMA 2 Community License
