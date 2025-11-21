# Counterfactual Probing for Math Reasoning

A research framework for improving mathematical reasoning in language models through multi-arm prompt ensembles and preference optimization.

## Key Finding

**Counterfactual self-skepticism improves generalization.** Prompting a model to "assume your first instinct might be wrong" before solving leads to more robust reasoning that transfers better to held-out test sets.

## Results

| Configuration | GSM8K Accuracy |
|---------------|----------------|
| Base (greedy) | 79.08% |
| + SFT | 84.46% |
| + DPO | **84.84%** |

Evaluation uses 8-shot majority voting across prompt variants.

## Method Overview

1. **Multi-Arm Rollout Collection**: Generate solutions using diverse prompt strategies (precise reasoning, reflective reasoning, counterfactual probing)
2. **Coherence Scoring**: Measure agreement across reasoning paths
3. **Preference Pair Construction**: Create training pairs from correct vs incorrect solutions, plus coherence-based pairs among correct solutions
4. **DPO Training**: Optimize for both correctness and reasoning quality

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
@misc{dylandeep2024counterfactual,
  title={Counterfactual Probing for Robust Mathematical Reasoning},
  author={DylanDeep},
  year={2024},
  url={https://github.com/dmilstein-match/DylanDeep-Core-8B-DPO}
}
```

## License

CC BY-NC-ND 4.0 + LLaMA 2 Community License
