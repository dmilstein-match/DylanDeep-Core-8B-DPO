# Looper-Math-Platinum

## Overview

Looper-Math-Platinum is a private research project focused on advancing mathematical reasoning capabilities using a proprietary coherence-aware training methodology. The project leverages the Abel-7B-002 model as its foundation, applying a three-phase training pipeline: Supervised Fine-Tuning (SFT), a proprietary multi-armed bandit coherence evaluation (Regime W), and preference-based Reinforcement Learning via DPO. The primary objective is to develop a high-performance, privately-hosted math reasoning model that first ensures correctness and then optimizes for the coherence and clarity of its solutions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### UI/UX Decisions
Not applicable as this is a backend model training and evaluation project with no direct user interface.

### Technical Implementations
The core system is built around a three-phase training pipeline for the `GAIR/Abel-7B-002` model:

1.  **Supervised Fine-Tuning (SFT)**: Utilizes LoRA on the full-precision Abel model with the GSM8K dataset, employing a tutoring-style prompt format.
2.  **Regime W Coherence Module (Proprietary)**: An 8-armed bandit system evaluates reasoning strategies based on final answer agreement (`s_end`), path consistency (`s_path`), and a combined coherence score (`s_wm`). It calculates per-trajectory rewards, integrating correctness, coherence, and length penalties.
3.  **Preference-Based Reinforcement Learning (DPO)**: Constructs preference pairs from Regime W rollouts, prioritizing correctness first (correct vs. incorrect trajectories) and then coherence (high vs. low coherence among correct trajectories). These pairs are used to fine-tune the LoRA adapter of the SFT model, optimizing for coherence.

### Feature Specifications
-   **Model Base**: `GAIR/Abel-7B-002` (7B parameters, ~80% GSM8K baseline).
-   **Training Precision**: Full bf16 precision on H100 GPUs, avoiding quantization.
-   **Evaluation**: Production-grade evaluation scripts on the GSM8K Platinum benchmark, featuring resume support, robust corruption handling, and detailed per-example logging.
-   **Data Management**: All data stored in JSONL format within the `data/` directory, managed via Hugging Face `datasets` library.
-   **Model Checkpointing**: LoRA adapters are saved incrementally in `checkpoints/` for each training phase.

### System Design Choices
-   **Development Environment**: Replit for coding and version control, integrated with a private GitHub repository.
-   **Training Infrastructure**: Lambda GPU instances (A10/A100) for model training, ensuring private hosting of model weights.
-   **Architectural Pattern**: LoRA (Low-Rank Adaptation) for efficient fine-tuning, applied to both SFT and DPO phases.
-   **Privacy**: Emphasis on self-hosted weights and no reliance on external APIs for inference or training.
-   **Workflow**: Code developed in Replit, pushed to GitHub, then pulled to Lambda for execution.

## Abel Pipeline Execution

### Phase 1: Supervised Fine-Tuning
```bash
# On Lambda H100 GPU instance
python src/baseline_sft/train_sft_abel.py

# Output: checkpoints/abel_sft_lora/
```

### Phase 2: Regime W Rollout Collection
```bash
# Generate 8 trajectories per question with Regime W scoring
python src/rl_training/collect_rollouts_abel.py

# Output: data/abel_regime_w_rollouts.jsonl
# Contains per-trajectory rewards and correctness flags
```

### Phase 3: Preference Pair Construction
```bash
# Build correctness-first preference pairs
python src/rl_training/build_preferences_abel.py

# Output: data/abel_coherence_preferences.jsonl
# Contains chosen/rejected pairs with correctness and coherence ranking
```

### Phase 4: Coherence DPO Training
```bash
# Train on preference pairs to optimize coherence
python src/rl_training/train_dpo_coherence.py

# Output: checkpoints/abel_coherence_lora/
```

### Phase 5: Evaluation on GSM8K Platinum
```bash
# Evaluate SFT checkpoint
python src/eval/eval_abel_sft_platinum.py

# Evaluate Coherence checkpoint
python src/eval/eval_abel_coherence_platinum.py

# Outputs: 
#   outputs/abel_sft_platinum_eval.jsonl
#   outputs/abel_coherence_platinum_eval.jsonl
```

## External Dependencies

### Machine Learning Frameworks
-   **Hugging Face Ecosystem**:
    -   `transformers`: Model loading, tokenization, inference.
    -   `datasets`: Dataset access (GSM8K).
    -   `trl`: Transformer Reinforcement Learning (SFTTrainer, DPOTrainer).
    -   `peft`: Parameter-Efficient Fine-Tuning (LoRA).
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