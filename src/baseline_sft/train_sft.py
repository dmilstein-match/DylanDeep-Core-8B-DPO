import os
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    # Load your dataset here
    ds = load_dataset("json", data_files="data/gsm8k_train.jsonl")["train"]

    # Format function (ensure input_ids + labels created)
    def formatting_func(example):
        prompt = (
            "You are a careful math tutor. Solve step by step.\n\n"
            f"Problem:\n{example['question']}\n\nSolution:"
        )
        target = example["answer"]
        return prompt, target

    # TRL v0.9+ config
    sft_config = SFTConfig(
        output_dir="checkpoints/sft",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        formatting_func=formatting_func,
        args=sft_config,
    )

    trainer.train()
    trainer.save_model("checkpoints/sft")
    tokenizer.save_pretrained("checkpoints/sft")


if __name__ == "__main__":
    main()
