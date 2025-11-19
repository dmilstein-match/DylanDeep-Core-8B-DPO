#!/usr/bin/env python3
"""
PPO Training for Correctness-Only RL (DeepSeek-style)
Uses GAIR/Abel-7B-002 as base model with LoRA adapters.
Optimized for 8× H100 GPUs with BF16 precision.
"""
import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# H100 optimization flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

from src.common.answer_utils import extract_answer, normalize_answer


BASE_MODEL = "GAIR/Abel-7B-002"
OUTPUT_DIR = "checkpoints/abel_ppo_lora"
DATASET_NAME = "gsm8k"


def build_prompt(question: str) -> str:
    """Build tutoring-style prompt for math problems."""
    return (
        "You are a careful math tutor. Solve the problem step-by-step, "
        "then give the final answer in the format '#### 42'.\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="PPO Correctness Training for Abel"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of training examples (default: 2000)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate (default: 5e-6)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Global batch size across all GPUs (default: 4)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Abel PPO Correctness Training (DeepSeek-style)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  N samples: {args.n_samples}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}\n")

    # Load tokenizer
    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect dtype support
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    dtype_name = "bf16" if use_bf16 else "fp16"

    # Create value head model first (PPO requires this)
    print("Creating PPO model with value head...")
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Enable gradient checkpointing on the pretrained model
    ppo_model.pretrained_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    ppo_model.pretrained_model.config.use_cache = False
    
    # Apply LoRA to the pretrained_model inside the value head wrapper
    print("Applying LoRA adapters to PPO model...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    ppo_model.pretrained_model = get_peft_model(ppo_model.pretrained_model, lora_config)

    # PPO configuration
    ppo_config = PPOConfig(
        model_name=BASE_MODEL,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        target_kl=0.1,
        log_with=None,  # Disable wandb
    )

    # Initialize PPO trainer
    print("Initializing PPO trainer...")
    trainer = PPOTrainer(
        config=ppo_config,
        model=ppo_model,
        tokenizer=tokenizer,
    )

    # Load training data
    print(f"Loading GSM8K training data...")
    dataset = load_dataset(DATASET_NAME, "main", split="train")
    dataset = dataset.shuffle(seed=42).select(range(min(args.n_samples, len(dataset))))
    print(f"Using {len(dataset)} examples for PPO training\n")

    print("Starting PPO training with correctness reward...")
    print("(Reward = 1.0 if correct, 0.0 if incorrect)\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for step, example in enumerate(dataset):
        question = example["question"]
        gold_answer = example["answer"]

        prompt = build_prompt(question)
        
        # Tokenize prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(ppo_model.pretrained_model.device)

        # Generate response
        with torch.no_grad():
            gen_ids = trainer.generate(input_ids, max_new_tokens=256, do_sample=True, temperature=0.7)
        
        # Decode generated text
        generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        response = generated_text[len(prompt):]  # Remove prompt

        # Compute correctness reward
        pred_answer = normalize_answer(extract_answer(response))
        gold_norm = normalize_answer(extract_answer(gold_answer))
        reward = 1.0 if pred_answer == gold_norm and gold_norm != "" else 0.0

        # Convert reward to tensor
        reward_tensor = torch.tensor([reward], dtype=torch.float32).to(ppo_model.pretrained_model.device)

        # PPO step
        query_tensors = [input_ids[0]]
        response_tensors = [gen_ids[0][len(input_ids[0]):]]
        rewards = [reward_tensor]
        
        stats = trainer.step(query_tensors, response_tensors, rewards)

        # Logging
        if step % 50 == 0:
            print(f"[Step {step}/{len(dataset)}] Reward: {reward:.2f} | Answer: {pred_answer[:50]}")

        # Save checkpoint
        if step % args.save_every == 0 and step > 0:
            print(f"\nSaving checkpoint at step {step}...")
            ppo_model.pretrained_model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"Checkpoint saved to {OUTPUT_DIR}\n")

    # Final save
    print(f"\nTraining complete! Saving final PPO LoRA...")
    ppo_model.pretrained_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n{'=' * 80}")
    print(f"PPO training finished!")
    print(f"Saved to: {OUTPUT_DIR}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
