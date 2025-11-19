import os
import json
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TRAIN_PATH = "data/gsm8k_train.jsonl"
OUTPUT_DIR = "checkpoints/sft_lora"  # change to "checkpoints/sft" if you prefer


def formatting_func(example):
    """
    Format each GSM8K example as a single prompt+answer string.
    TRL will automatically add EOS token.
    """
    question = example["question"]
    answer = example["answer"]  # already '#### 42' style

    prompt = (
        "You are a careful math tutor. Solve the problem step-by-step, "
        "then give the final answer in the format '#### 42'.\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )
    return prompt + answer


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading training data from JSONL...")
    train_ds = load_dataset("json", data_files=TRAIN_PATH)["train"]

    print("Loading tokenizer & 4-bit base model:", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization config (QLoRA-style)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # LoRA config – r=64 for expressivity, targeting all projection layers
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # SFT training config – tiny batch, 8-bit optimizer
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    print("Creating SFTTrainer (QLoRA)...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        peft_config=peft_config,
        args=sft_config,
    )

    print("\nStarting SFT training...\n")
    trainer.train()

    print("Saving LoRA adapter + tokenizer to:", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone! LoRA SFT model saved.")


if __name__ == "__main__":
    main()
