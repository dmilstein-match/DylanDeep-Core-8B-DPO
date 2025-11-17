import os
import json
from dataclasses import dataclass
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TRAIN_PATH = "data/gsm8k_train.jsonl"
OUTPUT_DIR = "checkpoints/sft"


def formatting_func(example):
    """
    Format the example as a single combined prompt+answer string.
    TRL will automatically add EOS token.
    """

    question = example["question"]
    answer = example["answer"]  # GSM8K answer already includes #### 42 format

    prompt = (
        "You are a careful math tutor. Solve the problem step-by-step, "
        "then give the final answer in the format '#### 42'.\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )

    full_text = prompt + answer  # Combine into ONE string
    return full_text             # Return string directly, not a list


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading training data from JSONL...")
    train_ds = load_dataset("json", data_files=TRAIN_PATH)["train"]

    print("Loading tokenizer & model:", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype="auto",
    )

    # TRL 0.9+ SFTConfig
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy="epoch",
    )

    print("Creating SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        formatting_func=formatting_func,
        args=sft_config,
    )

    print("\nStarting SFT training...\n")
    trainer.train()

    print("Saving SFT checkpoint to:", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone! SFT model saved.")


if __name__ == "__main__":
    main()
