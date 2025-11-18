# Looper-Math-Platinum-8B

## Overview

Private research project for fine-tuning DeepSeek-R1-Distill-Llama-8B on mathematical reasoning tasks using GSM8K dataset. The system implements a three-phase training pipeline: baseline supervised fine-tuning (SFT), proprietary multi-armed bandit coherence evaluation (Regime W), and reinforcement learning optimization via PPO. The goal is to create a high-performance, privately-hosted math reasoning model with enhanced coherence scoring.

## Current Status (November 18, 2025)

**Completed:**
- ✅ Phase 1: SFT with QLoRA - Successfully trained on Lambda GPU, LoRA adapters saved to `checkpoints/sft_lora/`
- ✅ Phase 2: Regime W Coherence Module - Fully implemented with 8-arm bandit, scoring metrics (s_end, s_path, s_cf, s_wm), and reward computation
- ✅ Phase 3: Multi-arm rollout collection script - Generates 8 solutions per question and computes Regime W rewards
- ✅ Phase 4: PPO training script - Uses TRL PPOTrainer with offline Regime W rollouts
- ✅ Phase 5: Final evaluation script - Compares Base vs SFT vs PPO on GSM8K test set
- ✅ Dev evaluation script - Quick validation on 50 dev examples for Base vs SFT comparison

**Ready for Execution:**
All code is implemented and ready to run on Lambda GPU. The complete training pipeline can now be executed:
1. Run `python -m src.regime_w.demo` to verify Regime W module
2. Run `python -m src.rl_training.collect_rollouts` to generate rollouts (Phase 3)
3. Run `python -m src.rl_training.train_ppo` to perform PPO training (Phase 4)
4. Run `python -m src.eval.eval_platinum` to evaluate all models (Phase 5)

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

**Phase 1 - Supervised Fine-Tuning (SFT) with QLoRA**
- Dataset: GSM8K mathematical reasoning dataset
- Split: 80% train, 20% validation from training set, separate test set retained
- Approach: QLoRA (Quantized Low-Rank Adaptation) for memory-efficient fine-tuning
- Format: Prompts structured as tutoring scenarios with step-by-step solutions ending in `#### [answer]` format
- Memory Optimization Strategy:
  - **4-bit Quantization**: Base model loaded in 4-bit using `BitsAndBytesConfig` (NF4 quant type, double quantization, bfloat16 compute)
  - **LoRA Adapters**: Only train small adapter weights (r=32, alpha=16, dropout=0.05) instead of full 8B parameters
  - **8-bit Optimizer**: Uses `paged_adamw_8bit` to reduce optimizer state memory
  - **Gradient Checkpointing**: Trades compute for memory by recomputing activations during backward pass
  - **Rationale**: Full fine-tuning 8B model requires ~40GB+ for model + optimizer states; QLoRA fits comfortably in 40GB GPU
- Framework: Hugging Face TRL's SFTTrainer + PEFT for LoRA
  - Uses `processing_class=tokenizer` (TRL 0.12+ standard for both regular and PEFT-enabled training)
  - Uses `peft_config=LoraConfig(...)` to enable adapter training
  - `formatting_func` returns single concatenated string (prompt + answer)
- Configuration: 1 epoch, batch size 1 with 8 gradient accumulation steps, learning rate 5e-5
- Output: LoRA adapter checkpoint in `checkpoints/sft_lora/` (can be merged with base model later if needed)

**Phase 2 - Regime W Coherence Module (Proprietary)**
- **Implementation**: Fully implemented in `src/regime_w/`
- **Architecture**: Multi-armed bandit with 8 arms testing different reasoning strategies
- **Arms** (`arms.py`):
  - 4 Wolfram-style arms (precise, linear reasoning) with temps 0.2, 0.3, 0.4, 0.5
  - 4 Maudlin-style arms (reflective reasoning) with temps 0.2, 0.3, 0.4, 0.5
- **Scoring Metrics** (`scoring.py`):
  - `s_end`: Pairwise agreement on final answers across arms
  - `s_path`: Path consistency measured by reasoning length variance
  - `s_cf`: Counterfactual robustness (placeholder for future CHSH logic)
  - `s_wm`: Combined coherence score (0.4*s_end + 0.3*s_path + 0.3*s_cf)
- **Reward Function** (`reward.py`):
  - Base: +1.0 if any arm correct, -0.5 otherwise
  - Coherence bonus: +0.7 * s_wm
  - Length penalty: -0.001 * max(0, avg_length - 400)
- **Demo**: `demo.py` provides sanity test with sample questions
- **Output**: Combined coherence + correctness reward signal for PPO training

**Phase 3 - Rollout Collection**
- **Implementation**: `src/rl_training/collect_rollouts.py`
- **Process**: Loads SFT LoRA model, generates 8 solutions (one per arm) for each training question
- **Scoring**: Each question's 8-arm output set scored together via Regime W reward function
- **Data Format**: JSONL file `data/regime_w_rollouts.jsonl` with fields:
  - `question`: Training question text
  - `gold_answer`: Ground truth answer
  - `arm_outputs`: Array of 8 solution attempts (one per arm) with full_text, reasoning, answer
  - `reward`: Single Regime W reward score for that question's arm set
- **Default**: Processes first 500 training examples (configurable)
- **Purpose**: Create offline rollout dataset with coherence-aware rewards for PPO training

**Phase 4 - PPO Reinforcement Learning**
- **Implementation**: `src/rl_training/train_ppo.py`
- **Framework**: TRL's PPOTrainer with offline rollout training
- **Input**: Loads pre-computed Regime W rollouts from `data/regime_w_rollouts.jsonl`
- **Strategy**: For each batch, randomly selects one arm's response per question as training sample
- **Reward**: Uses pre-computed Regime W reward (correctness + coherence) for that question
- **Configuration**:
  - Learning rate: 5e-6
  - Batch size: 8, mini-batch: 4
  - Target KL: 0.1 with adaptive KL control
  - Max grad norm: 1.0
- **Optimization**: Policy gradient updates on LoRA parameters to maximize coherent reasoning
- **Output**: PPO-optimized LoRA checkpoint in `checkpoints/ppo_regime_w/`

**Phase 5 - Evaluation**
- **Implementation**: 
  - `src/eval/eval_gsm8k_dev.py`: Quick dev validation (50 examples, Base vs SFT)
  - `src/eval/eval_platinum.py`: Full test evaluation (Base vs SFT vs PPO)
- **Test Set**: GSM8K test split (reserved, never seen during training)
- **Models Compared**:
  - Base: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (vanilla)
  - SFT: LoRA checkpoint from `checkpoints/sft_lora/`
  - PPO: Regime W optimized checkpoint from `checkpoints/ppo_regime_w/`
- **Metrics**: Accuracy on mathematical reasoning tasks with robust answer extraction
- **Answer Extraction**: Regex-based, prefers `####` marker, falls back to last number in text

### Data Management
- **Storage Location**: `data/` directory (populated on Lambda during training)
- **Format**: JSONL files for train/dev/test splits
- **Preprocessing**: Handled by `prepare_data.py` with automated 80/20 splitting
- **Dataset Source**: Hugging Face `datasets` library for GSM8K access

### Model Checkpointing
- **Location**: `checkpoints/` directory with subdirectories per training phase
- **SFT Checkpoints**: LoRA adapters saved in `checkpoints/sft_lora/`
- **PPO Checkpoints**: Saved in `checkpoints/ppo/`
- **Strategy**: Incremental saves during training for recovery and comparison
- **Note**: SFT produces LoRA adapter weights that can be merged with base model or loaded separately for inference

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
- **peft**: Parameter-efficient fine-tuning library for LoRA adapter training
- **accelerate**: Distributed training and mixed-precision support
- **bitsandbytes**: 4-bit/8-bit quantization for memory-efficient training

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