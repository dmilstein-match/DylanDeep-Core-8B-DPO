import os
import json
from dataclasses import dataclass
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # per our plan

DATA_PATH = "data/gsm8k_train.jsonl"
OUTPUT_DIR = "checkpoints/sft"

@dataclass
class Example:
    prompt: str
    completion: str

def load_examples(path: str) -> List[Example]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            q = j["question"]
            # GSM8K "answer" field contains full solution with final "#### 42"
            a = j["answer"]

            prompt = (
                "You are a careful math tutor. Solve the problem step by step, "
                "then give the final numeric answer in the form '#### 42'.\n\n"
                f"Problem:\n{q}\n\nSolution:"
            )

            completion = a  # we'll train to emit the GSM8K-style solution

            examples.append(Example(prompt=prompt, completion=completion))
    return examples

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading training data...")
    data = load_examples(DATA_PATH)

    # Convert to dicts for SFTTrainer
    train_dataset = [
        {"text": ex.prompt + "\n" + ex.completion}
        for ex in data
    ]

    print("Loading base model:", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        output_dir=OUTPUT_DIR,
        packing=True,
        num_train_epochs=1,
        learning_rate=1e-5,
    )

    print("Starting SFT training...")
    trainer.train()
    print("Saving SFT checkpoint to", OUTPUT_DIR)
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
