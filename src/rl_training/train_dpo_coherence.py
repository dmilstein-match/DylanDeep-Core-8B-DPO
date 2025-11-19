import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig


BASE_MODEL = "GAIR/Abel-7B-002"
SFT_PATH = "checkpoints/abel_sft_lora"
PREFERENCES_PATH = "data/abel_coherence_preferences.jsonl"
RL_OUTPUT_DIR = "checkpoints/abel_coherence_lora"


def build_prompt(question: str) -> str:
    return (
        "You are a careful math tutor. Solve the problem step-by-step, "
        "then give the final answer in the format '#### 42'.\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )


def main():
    print("=" * 80)
    print("Abel Coherence DPO Training")
    print("=" * 80)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect dtype support (bf16 for H100/A100, fp16 fallback)
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    dtype_name = "bf16" if use_bf16 else "fp16"

    # Load base model with detected dtype
    print(f"Loading Abel base model in {dtype_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=dtype,
    )
    base_model.gradient_checkpointing_enable()

    # Load SFT LoRA as starting point for policy model
    print(f"Loading Abel SFT LoRA adapter from {SFT_PATH}...")
    policy_model = PeftModel.from_pretrained(base_model, SFT_PATH)
    policy_model.tokenizer = tokenizer
    
    # Re-enable gradient checkpointing for PEFT model (required for H100 memory efficiency)
    policy_model.enable_input_require_grads()
    policy_model.gradient_checkpointing_enable()

    # Create separate reference model (frozen SFT) with same dtype
    print(f"Loading separate reference model (frozen Abel + SFT LoRA) in {dtype_name}...")
    ref_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=dtype,
    )
    ref_model = PeftModel.from_pretrained(ref_base, SFT_PATH)
    ref_model.eval()
    ref_model.tokenizer = tokenizer
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load preferences dataset
    print(f"\nLoading coherence preferences from {PREFERENCES_PATH}...")
    dataset = load_dataset("json", data_files=PREFERENCES_PATH)["train"]

    def format_dataset(examples):
        prompts = [build_prompt(q) for q in examples["question"]]
        return {
            "prompt": prompts,
            "chosen": examples["chosen"],
            "rejected": examples["rejected"],
        }

    # Ultra-explicit column removal: keep ONLY prompt, chosen, rejected
    # Remove all other columns to ensure clean DPO dataset
    dataset = dataset.map(
        format_dataset,
        batched=True,
        remove_columns=[
            c for c in dataset.column_names
            if c not in ["prompt", "chosen", "rejected"]
        ]
    )

    print(f"Loaded {len(dataset)} preference pairs")
    
    # Count pair types from raw file for logging
    try:
        import json
        correctness_pairs = 0
        coherence_pairs = 0
        with open(PREFERENCES_PATH, "r") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("pair_type") == "correctness":
                    correctness_pairs += 1
                elif rec.get("pair_type") == "coherence":
                    coherence_pairs += 1
        print(f"  - {correctness_pairs} correctness pairs")
        print(f"  - {coherence_pairs} coherence pairs\n")
    except:
        pass

    # DPO training configuration (Lambda-compatible, uses detected dtype from above)
    dpo_config = DPOConfig(
        output_dir=RL_OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16,
        remove_unused_columns=False,
    )

    # Create DPO trainer (no tokenizer= or beta= for Lambda TRL compatibility)
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
    )

    print("Starting Abel Coherence DPO training...")
    print(f"Effective batch size: {dpo_config.per_device_train_batch_size * dpo_config.gradient_accumulation_steps}")
    print(f"Training epochs: {dpo_config.num_train_epochs}\n")

    # Train
    trainer.train()

    # Save coherence LoRA adapter
    print(f"\nSaving coherence LoRA adapter to {RL_OUTPUT_DIR}...")
    policy_model.save_pretrained(RL_OUTPUT_DIR)
    tokenizer.save_pretrained(RL_OUTPUT_DIR)
    
    print(f"\n{'=' * 80}")
    print("Abel Coherence DPO training complete!")
    print(f"Coherence LoRA adapter saved to: {RL_OUTPUT_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
