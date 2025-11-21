import os
import json
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# H100 optimization flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

BASE_MODEL = "GAIR/Abel-7B-002"
TRAIN_PATH = "data/gsm8k_train.jsonl"
OUTPUT_DIR = "checkpoints/abel_sft_lora"


@dataclass
class TrainExample:
    question: str
    answer: str


def load_gsm8k(path: str) -> List[TrainExample]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            data.append(TrainExample(
                question=raw["question"],
                answer=raw["answer"],
            ))
    return data


def formatting_func(example):
    """Format single example as tutoring-style prompt with step-by-step solution."""
    q = example["question"]
    a = example["answer"]
    
    prompt = (
        "You are a careful math tutor. Solve the problem step-by-step, "
        "then give the final answer in the format '#### 42'.\n\n"
        f"Problem:\n{q}\n\nSolution:\n{a}"
    )
    return [prompt]  # TRL 0.11.4 requires list return


def main():
    print("=" * 80)
    print("Abel-7B-002 SFT Training with LoRA")
    print("=" * 80)
    
    # Load dataset
    print(f"\nLoading GSM8K training data from {TRAIN_PATH}...")
    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_PATH},
        split="train",
    )
    print(f"Loaded {len(dataset)} training examples\n")

    # Load tokenizer
    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Important for causal LM training
    
    # Detect dtype support (bf16 for H100/A100, fp16 fallback)
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    dtype_name = "bf16" if use_bf16 else "fp16"
    
    print(f"Loading Abel base model in {dtype_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    
    # Enable non-reentrant gradient checkpointing (fixes LoRA + DDP conflict)
    print("Enabling non-reentrant gradient checkpointing for LoRA + DDP compatibility...")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # LoRA configuration (attention + MLP layers)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training configuration (8× H100-optimized batch sizes)
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        warmup_steps=100,
        max_seq_length=1024,  # Explicit to avoid warning
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        optim="adamw_torch_fused",
        seed=42,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Create trainer (TRL 0.11.4 compatibility)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=lora_config,
        train_dataset=dataset,
        formatting_func=formatting_func,
        args=sft_config,
    )

    print("\nStarting Abel SFT (LoRA) training...\n")
    per_device_batch = sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() if torch.cuda.is_available() else 1))
    global_batch = per_device_batch * world_size
    print(f"Per-device effective batch size: {per_device_batch}")
    print(f"World size (GPUs): {world_size}")
    print(f"Global effective batch size: {global_batch}")
    print(f"Total training steps: {len(dataset) // global_batch}")
    print()

    # Train
    trainer.train()

    # Save LoRA adapter and tokenizer
    print(f"\nSaving LoRA adapter and tokenizer to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("=" * 80)
    print("Abel SFT training complete!")
    print(f"LoRA adapter saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
