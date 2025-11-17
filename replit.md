# Looper-Math-Platinum-8B

## Overview

Private research project for fine-tuning DeepSeek-R1-Distill-Llama-8B on mathematical reasoning tasks using GSM8K dataset. The system implements a three-phase training pipeline: baseline supervised fine-tuning (SFT), proprietary multi-armed bandit coherence evaluation (Regime W), and reinforcement learning optimization via PPO. The goal is to create a high-performance, privately-hosted math reasoning model with enhanced coherence scoring.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Development Workflow
- **Code Repository**: GitHub private repository synced with Replit for version control
- **Development Environment**: Replit for code editing and version control
- **Training Infrastructure**: Lambda GPU instances (A10/A100 with 24-40GB VRAM) for model training
- **Data Flow**: Code developed in Replit → pushed to GitHub → pulled to Lambda for training execution

### Base Model Selection
- **Model**: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- **Rationale**: Pre-trained reasoning model with strong chain-of-thought capabilities, 8B parameter size balances performance with trainability on single GPU, MIT-style licensing permits commercial use and private hosting
- **Privacy**: Self-hosted weights on Lambda infrastructure, no external API dependencies

### Training Pipeline Architecture

**Phase 1 - Supervised Fine-Tuning (SFT)**
- Dataset: GSM8K mathematical reasoning dataset
- Split: 80% train, 20% validation from training set, separate test set retained
- Approach: Standard supervised fine-tuning on question-solution pairs
- Format: Prompts structured as tutoring scenarios with step-by-step solutions ending in `#### [answer]` format
- Framework: Hugging Face TRL's SFTTrainer with TRL 0.9+ API (SFTConfig + formatting_func)
- Configuration: 1 epoch, batch size 1 with 8 gradient accumulation steps, learning rate 5e-5
- Output: Base fine-tuned checkpoint in `checkpoints/sft/`

**Phase 2 - Regime W Coherence Module (Proprietary)**
- Architecture: Multi-armed bandit with 8 arms testing different reasoning strategies
- Arms: 4 Wolfram-style computational variants + 4 Maudlin-style counterfactual variants
- Scoring Metrics:
  - `s_end`: End-state correctness evaluation
  - `s_path`: Path consistency through reasoning steps
  - `s_cf`: Counterfactual robustness scoring
  - `s_wm`: Wolfram/Maudlin strategy effectiveness
- Output: Combined coherence + correctness reward signal

**Phase 3 - Rollout Collection**
- Process: Generate multiple solution trajectories per training question
- Reward Assignment: Each trajectory scored via Regime W metrics
- Data Format: Question-trajectory-reward tuples for PPO training
- Purpose: Create preference dataset combining correctness with coherence signals

**Phase 4 - PPO Reinforcement Learning**
- Framework: TRL's PPOTrainer
- Reward Function: Regime W combined scores (correctness + coherence)
- Optimization: Policy gradient updates to maximize coherent reasoning
- Output: Final PPO-optimized checkpoint in `checkpoints/ppo/`

**Phase 5 - Evaluation**
- Test Set: GSM8K test split (reserved, never seen during training)
- Comparison: Base model vs SFT checkpoint vs PPO checkpoint
- Metrics: Accuracy and coherence scores across model variants

### Data Management
- **Storage Location**: `data/` directory (populated on Lambda during training)
- **Format**: JSONL files for train/dev/test splits
- **Preprocessing**: Handled by `prepare_data.py` with automated 80/20 splitting
- **Dataset Source**: Hugging Face `datasets` library for GSM8K access

### Model Checkpointing
- **Location**: `checkpoints/` directory with subdirectories per training phase
- **SFT Checkpoints**: Saved in `checkpoints/sft/`
- **PPO Checkpoints**: Saved in `checkpoints/ppo/`
- **Strategy**: Incremental saves during training for recovery and comparison

### Module Organization
- **baseline_sft/**: Data preparation and supervised fine-tuning scripts
- **regime_w/**: Private coherence evaluation engine (multi-armed bandit implementation)
- **rl_training/**: Rollout generation and PPO training orchestration
- **eval/**: Model evaluation and comparison utilities

## External Dependencies

### Machine Learning Frameworks
- **transformers**: Hugging Face library for model loading, tokenization, and inference
- **datasets**: Hugging Face library for GSM8K dataset access and preprocessing
- **trl**: Transformer Reinforcement Learning library (SFTTrainer, PPOTrainer)
- **peft**: Parameter-efficient fine-tuning techniques (if needed for memory optimization)
- **accelerate**: Distributed training and mixed-precision support
- **bitsandbytes**: 8-bit quantization for memory efficiency

### Core Libraries
- **torch**: PyTorch deep learning framework (included via transformers[torch])
- **sentencepiece**: Tokenization library for model compatibility
- **numpy**: Numerical computing for data manipulation

### Data Sources
- **GSM8K Dataset**: Mathematical reasoning dataset from Hugging Face (`gsm8k`, `main` config)
- **Access Method**: `datasets.load_dataset()` with automatic downloading and caching

### Infrastructure
- **Lambda Labs**: GPU cloud provider for training (A10/A100 instances)
- **GitHub**: Version control and code repository hosting (private repository)
- **Replit**: Cloud-based development environment with GitHub integration

### Model Hosting
- **Base Model Source**: Hugging Face Hub (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`)
- **Hosting Strategy**: Self-hosted weights on Lambda infrastructure for privacy
- **No External APIs**: All inference and training performed on private infrastructure