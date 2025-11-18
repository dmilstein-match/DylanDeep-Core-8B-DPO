# Looper-Math-Platinum-8B

## Overview

Private research project for fine-tuning DeepSeek-R1-Distill-Llama-8B on mathematical reasoning tasks using GSM8K dataset. The system implements a three-phase training pipeline: baseline supervised fine-tuning (SFT), proprietary multi-armed bandit coherence evaluation (Regime W), and preference-based reinforcement learning optimization via DPO. The goal is to create a high-performance, privately-hosted math reasoning model with enhanced coherence scoring.

## Current Status (November 18, 2025)

**Completed:**
- ✅ Phase 1: SFT with QLoRA - Successfully trained on Lambda GPU, LoRA adapters saved to `checkpoints/sft_lora/`
- ✅ Phase 2: Regime W Coherence Module - Fully implemented with 8-arm bandit, scoring metrics (s_end, s_path, s_cf, s_wm), and per-trajectory reward computation
- ✅ Phase 3: Multi-trajectory rollout collection script - Generates 8 solutions per question and computes Regime W rewards per trajectory
- ✅ Phase 3.5: Preference pair builder - Converts rollouts to better/worse trajectory pairs
- ✅ Phase 4: DPO training script - Uses TRL DPOTrainer with offline Regime W preference pairs
- ✅ Phase 5: Final evaluation script - Compares Base vs SFT vs DPO RL on GSM8K test set
- ✅ Dev evaluation script - Quick validation on 50 dev examples for Base vs SFT comparison

**Ready for Execution:**
All code is implemented and ready to run on Lambda GPU. The complete training pipeline can now be executed:
1. Run `python -m src.regime_w.demo` to verify Regime W module
2. Run `python -m src.rl_training.collect_rollouts` to generate multi-trajectory rollouts (Phase 3)
3. Run `python -m src.rl_training.build_preferences` to create preference pairs from rollouts
4. Run `python -m src.rl_training.train_dpo` to perform DPO training (Phase 4)
5. Run `python -m src.eval.eval_platinum` to evaluate all models (Phase 5)

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
  - Trajectory dataclass: text, reasoning, answer, num_tokens
  - Hyperparameters: ALPHA_CORRECT=1.0, BETA_COHERENCE=0.5, LENGTH_PENALTY_PER_TOKEN=0.001, threshold=512
  - Per-trajectory reward formula:
    - Base: ALPHA_CORRECT * correctness (1.0 if correct, 0.0 otherwise)
    - Coherence term: BETA_COHERENCE * s_wm (question-level coherence score)
    - Length penalty: -0.001 * max(0, num_tokens - 512)
    - Agreement bonus: +0.2 if correct and all correct trajectories agree (s_end > 0.9)
  - Returns list of rewards (one per trajectory) for preference pair creation
- **Demo**: `demo.py` provides sanity test with sample questions
- **Output**: Per-trajectory rewards combining correctness + coherence for preference-based RL

**Phase 3 - Multi-Trajectory Rollout Collection**
- **Implementation**: `src/rl_training/collect_rollouts.py`
- **Model Loading**: Loads base model in 8-bit + SFT LoRA adapter using PEFT pattern
- **Process**: Generates 8 trajectories (one per arm) for each training question
- **Trajectory Sampling**: Each arm uses different system prompt and temperature settings
- **Scoring**: Computes per-trajectory Regime W rewards via `compute_rewards_for_question`
- **Answer Extraction Fix**: Slices response_ids from generation to exclude prompt before extraction
- **Data Format**: JSONL file `data/regime_w_rollouts.jsonl` with fields:
  - `question`: Training question text
  - `gold_answer`: Ground truth answer
  - `trajectories`: Array of 8 trajectories, each with text, reasoning, answer, num_tokens, reward
- **Default**: Processes first 500 training examples (configurable)
- **Purpose**: Create trajectory dataset with per-sample coherence-aware rewards for preference pair creation

**Phase 3.5 - Preference Pair Construction**
- **Implementation**: `src/rl_training/build_preferences.py`
- **Input**: Loads rollouts from `data/regime_w_rollouts.jsonl`
- **Selection Strategy**: For each question, selects highest reward vs lowest reward trajectory
- **Output Format**: JSONL file `data/preferences.jsonl` with fields:
  - `question`: Training question
  - `gold_answer`: Ground truth
  - `chosen`: Higher-reward trajectory text
  - `rejected`: Lower-reward trajectory text
  - `chosen_reward`, `rejected_reward`: For debugging/analysis
- **Purpose**: Convert rollout rewards into preference pairs for DPO training

**Phase 4 - DPO Reinforcement Learning**
- **Implementation**: `src/rl_training/train_dpo.py`
- **Framework**: TRL's DPOTrainer for preference-based optimization
- **Model Loading**:
  - Policy model: Base (8-bit) + SFT LoRA adapter (with gradient checkpointing)
  - Reference model: Separate frozen copy of base (8-bit) + SFT LoRA adapter
- **Input**: Preference pairs from `data/preferences.jsonl`
- **Dataset Format**: Maps to (prompt, chosen, rejected) tuples for DPO
- **Prompt Format**: Same tutoring-style prompt used in SFT
- **Configuration**:
  - Learning rate: 5e-6
  - Batch size: 1 with 8 gradient accumulation steps
  - Training epochs: 1
  - Beta: 0.1 (KL regularization strength)
  - BF16 precision for efficiency
- **Optimization**: Updates only LoRA parameters to prefer high-coherence trajectories
- **Output**: DPO-optimized LoRA checkpoint in `checkpoints/lora_rl/`

**Phase 5 - Evaluation**
- **Implementation**: 
  - `src/eval/eval_gsm8k_dev.py`: Quick dev validation (50 examples, Base vs SFT)
  - `src/eval/eval_platinum.py`: Full test evaluation (Base vs SFT vs DPO RL)
  - `src/eval/eval_sft_platinum.py`: Production-grade SFT evaluation on GSM8K Platinum with resume support
  - `src/eval/eval_rl_platinum.py`: Production-grade RL evaluation on GSM8K Platinum with resume support
  - `src/eval/eval_with_logging.py`: Saves predictions to JSONL for manual inspection
- **Test Set**: GSM8K test split (reserved, never seen during training)
- **GSM8K Platinum Benchmark**: Official cleaned test set (`madrylab/gsm8k-platinum`, 1,210 examples)
- **Models Compared**:
  - Base: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (vanilla)
  - SFT: LoRA checkpoint from `checkpoints/sft_lora/`
  - DPO RL: Regime W optimized checkpoint from `checkpoints/lora_rl/`
- **Model Loading**: Uses base model (8-bit) + LoRA adapter pattern for SFT and RL models
- **Answer Extraction**: Regex-based, prefers `####` marker, falls back to last number in text
- **Answer Normalization**: Critical for accuracy - handles "42.0" → "42", removes commas, strips whitespace
- **Production Features** (both `eval_sft_platinum.py` and `eval_rl_platinum.py`):
  - Per-example JSONL logging to `outputs/{sft,rl}_platinum_eval.jsonl` with all predictions and metadata
  - Automatic resume support: reads existing results, skips completed examples, continues from last index
  - Robust corruption handling: gracefully handles truncated JSON lines from crashes, automatically repairs file
  - Progress tracking: shows running accuracy every 50 examples
  - Periodic flushing: saves progress every 10 examples to prevent data loss on GPU timeout/disconnect
  - Analysis-ready output: each record includes question, gold answer (raw/extracted/normalized), model output, prediction (raw/extracted/normalized), correctness flag

### Data Management
- **Storage Location**: `data/` directory (populated on Lambda during training)
- **Format**: JSONL files for train/dev/test splits
- **Preprocessing**: Handled by `prepare_data.py` with automated 80/20 splitting
- **Dataset Source**: Hugging Face `datasets` library for GSM8K access

### Model Checkpointing
- **Location**: `checkpoints/` directory with subdirectories per training phase
- **SFT Checkpoints**: LoRA adapters saved in `checkpoints/sft_lora/`
- **DPO RL Checkpoints**: LoRA adapters saved in `checkpoints/lora_rl/`
- **Strategy**: Incremental saves during training for recovery and comparison
- **Loading Pattern**: Base model (8-bit) + LoRA adapter using PEFT's `PeftModel.from_pretrained()`
- **Note**: All checkpoints are LoRA adapter weights that must be loaded with base model for inference

### Module Organization
- **baseline_sft/**: Data preparation and supervised fine-tuning scripts with QLoRA
- **regime_w/**: Private coherence evaluation engine (8-armed bandit, scoring metrics, reward computation)
- **rl_training/**: Multi-trajectory rollout generation, preference pair construction, and DPO training
- **eval/**: Model evaluation and comparison utilities with answer extraction fixes

## External Dependencies

### Machine Learning Frameworks
- **transformers**: Hugging Face library for model loading, tokenization, and inference
- **datasets**: Hugging Face library for GSM8K dataset access and preprocessing
- **trl**: Transformer Reinforcement Learning library (SFTTrainer, DPOTrainer)
- **peft**: Parameter-efficient fine-tuning library for LoRA adapter training and loading
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