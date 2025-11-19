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


def formatting_func(examples):
    """Format examples as tutoring-style prompts with step-by-step solutions."""
    texts = []
    for q, a in zip(examples["question"], examples["answer"]):
        prompt = (
            "You are a careful math tutor. Solve the problem step-by-step, "
            "then give the final answer in the format '#### 42'.\n\n"
            f"Problem:\n{q}\n\nSolution:\n{a}"
        )
        texts.append(prompt)
    return texts


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
    
    # Detect dtype support (bf16 for H100/A100, fp16 fallback)
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    dtype_name = "bf16" if use_bf16 else "fp16"
    
    print(f"Loading Abel base model in {dtype_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=dtype,
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training configuration (H100-optimized batch sizes)
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        max_seq_length=1024,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        peft_config=lora_config,
        train_dataset=dataset,
        formatting_func=formatting_func,
        args=sft_config,
    )

    print("\nStarting Abel SFT (LoRA) training...\n")
    print(f"Effective batch size: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
    print(f"Total training steps: {len(dataset) // (sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps)}")
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
