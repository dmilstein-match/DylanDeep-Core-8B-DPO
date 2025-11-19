# Looper-Math-Platinum

## Overview

Looper-Math-Platinum is a private research project focused on advancing mathematical reasoning capabilities using a proprietary coherence-aware training methodology. The project leverages the Abel-7B-002 model as its foundation, applying a **SFT → Regime W → DPO** training pipeline: Supervised Fine-Tuning (SFT) for baseline correctness, followed by Regime W coherence evaluation (8-armed bandit multi-path analysis), and preference-based DPO for coherence shaping. The primary objective is to develop a high-performance, privately-hosted math reasoning model that first establishes correctness via SFT and then optimizes for the coherence and clarity of its solutions via DPO.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### UI/UX Decisions
Not applicable as this is a backend model training and evaluation project with no direct user interface.

### Technical Implementations
The core system is built around a **SFT → Regime W → DPO** training pipeline for the `GAIR/Abel-7B-002` model:

1.  **Supervised Fine-Tuning (SFT)**: Trains Abel base model with LoRA using the GSM8K dataset and tutoring-style prompts. Establishes baseline mathematical reasoning capability.
2.  **Regime W Rollout Collection**: Generates 8 trajectories per question using SFT policy. An 8-armed bandit system evaluates group coherence based on final answer agreement (`s_end`), path consistency (`s_path`), counterfactual analysis (`s_cf`), and weighted coherence score (`s_wm`). Computes per-trajectory rewards integrating correctness, per-question coherence, and length penalties.
3.  **Preference Pair Construction**: Builds two types of preference pairs: (a) correctness pairs (correct vs incorrect), (b) coherence pairs (high vs low coherence among correct trajectories). Threshold lowered to 0.01 to capture variance from length differences.
4.  **DPO Coherence Training**: Fine-tunes SFT LoRA checkpoint using coherence preferences. Uses SFT checkpoint as both policy (trainable) and reference (frozen) models, optimizing for coherence while maintaining correctness.

### Feature Specifications
-   **Model Base**: `GAIR/Abel-7B-002` (7B parameters, ~80% GSM8K baseline).
-   **Training Precision**: Full bf16 precision on H100 GPUs, avoiding quantization.
-   **Evaluation**: Production-grade evaluation scripts on the GSM8K Platinum benchmark, featuring resume support, robust corruption handling, and detailed per-example logging.
-   **Data Management**: All data stored in JSONL format within the `data/` directory, managed via Hugging Face `datasets` library.
-   **Model Checkpointing**: LoRA adapters are saved incrementally in `checkpoints/` for each training phase.

### System Design Choices
-   **Development Environment**: Replit for coding and version control, integrated with a private GitHub repository.
-   **Training Infrastructure**: Lambda GPU instances (8× H100 SXM5) for model training, ensuring private hosting of model weights.
-   **Architectural Pattern**: LoRA (Low-Rank Adaptation) for efficient fine-tuning across all stages (SFT and DPO).
-   **Regime W Privacy**: Coherence scoring implementation details remain proprietary; only the reward formula is exposed in training code.
-   **Privacy**: Emphasis on self-hosted weights and no reliance on external APIs for inference or training.
-   **Workflow**: Code developed in Replit, pushed to GitHub, then pulled to Lambda for execution.

## Abel SFT → Regime W → DPO Pipeline Execution

### Phase 1: Supervised Fine-Tuning
```bash
# On Lambda 8× H100 GPU instance
python src/baseline_sft/train_sft_abel.py

# Output: checkpoints/abel_sft_lora/
```

### Phase 2: Regime W Rollout Collection (Multi-GPU)
```bash
# Generate 8 trajectories per question with Regime W scoring (19-min runtime!)
python src/rl_training/collect_rollouts_abel_multigpu.py \
  --base_model GAIR/Abel-7B-002 \
  --lora_path checkpoints/abel_sft_lora \
  --n_samples 500 \
  --n_gpus 8

# Output: data/abel_regime_w_rollouts.jsonl
# Contains per-trajectory rewards and correctness flags
```

### Phase 3: Preference Pair Construction
```bash
# Build correctness-first preference pairs (threshold=0.01 for coherence pairs)
python src/rl_training/build_preferences_abel.py

# Output: data/abel_coherence_preferences.jsonl
# Contains chosen/rejected pairs with correctness and coherence ranking
```

### Phase 4: DPO Coherence Training
```bash
# Train on preference pairs to optimize coherence (SFT checkpoint as base)
torchrun --nproc_per_node=8 src/rl_training/train_dpo_coherence.py

# Output: checkpoints/abel_dpo_coherence_lora/
```

### Phase 5: Evaluation on GSM8K Platinum
```bash
# Evaluate SFT checkpoint
python src/eval/eval_abel_sft_platinum.py

# Evaluate DPO checkpoint
python src/eval/eval_abel_dpo_platinum.py

# Outputs: 
#   outputs/abel_sft_platinum_eval.jsonl
#   outputs/abel_dpo_coherence_platinum_eval.jsonl
```

## External Dependencies

### Machine Learning Frameworks
-   **Hugging Face Ecosystem**:
    -   `transformers`: Model loading, tokenization, inference.
    -   `datasets`: Dataset access (GSM8K).
    -   `trl`: Transformer Reinforcement Learning (SFTTrainer, DPOTrainer).
    -   `peft`: Parameter-Efficient Fine-Tuning (LoRA via PeftModel and get_peft_model).
-   `accelerate`: Distributed training and mixed-precision.
-   `torch`: PyTorch deep learning framework.

### Core Libraries
-   `sentencepiece`: Tokenization.
-   `numpy`: Numerical computing.

### Data Sources
-   `GSM8K Dataset`: Mathematical reasoning dataset accessed via `datasets.load_dataset()`.

### Infrastructure
-   **Lambda Labs**: GPU cloud provider for model training.
-   **GitHub**: Version control and code hosting.
-   **Replit**: Cloud-based development environment.

### Model Hosting
-   **Active Model**: `GAIR/Abel-7B-002` from Hugging Face Hub.
-   **Legacy Model**: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` from Hugging Face Hub (archived).
-   **Hosting Strategy**: All models are self-hosted on Lambda infrastructure to maintain privacy and control.