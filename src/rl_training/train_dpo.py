from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
SFT_PATH = "checkpoints/sft_lora"
PREFERENCES_PATH = "data/preferences.jsonl"
RL_OUTPUT_DIR = "checkpoints/lora_rl"


def build_prompt(question: str) -> str:
    return (
        "You are a careful math tutor. Solve the problem step-by-step, "
        "then give the final answer in the format '#### 42'.\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )


def main():
    print("Loading tokenizer from base model", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in 8-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        load_in_8bit=True,
    )
    base_model.gradient_checkpointing_enable()

    print("Loading LoRA SFT adapter as policy model from", SFT_PATH)
    policy_model = PeftModel.from_pretrained(base_model, SFT_PATH)
    policy_model.tokenizer = tokenizer

    print("Loading separate reference model (frozen SFT)")
    ref_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        load_in_8bit=True,
    )
    ref_model = PeftModel.from_pretrained(ref_base, SFT_PATH)
    ref_model.eval()
    ref_model.tokenizer = tokenizer
    for param in ref_model.parameters():
        param.requires_grad = False

    print("Loading preferences dataset from", PREFERENCES_PATH)
    dataset = load_dataset("json", data_files=PREFERENCES_PATH)["train"]

    def format_dataset(examples):
        prompts = [build_prompt(q) for q in examples["question"]]
        return {
            "prompt": prompts,
            "chosen": examples["chosen"],
            "rejected": examples["rejected"],
        }

    dataset = dataset.map(format_dataset, batched=True, remove_columns=["question", "gold_answer", "chosen_reward", "rejected_reward"])

    print(f"Loaded {len(dataset)} preference pairs.")

    import torch
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    
    dpo_config = DPOConfig(
        output_dir=RL_OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
    )

    print("Starting DPO training...")
    trainer.train()

    print("Saving LoRA RL adapter to", RL_OUTPUT_DIR)
    policy_model.save_pretrained(RL_OUTPUT_DIR)
    tokenizer.save_pretrained(RL_OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
