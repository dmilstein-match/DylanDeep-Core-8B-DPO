#!/usr/bin/env python3
"""
Fast batched Regime W rollout collection using merged Abel SFT model.
Generates 8 trajectories per question with proper Regime W scoring.
"""
import argparse
import json
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# H100 optimization flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

from src.regime_w.arms import build_all_arms
from src.regime_w.reward import Trajectory, compute_rewards_for_question
from src.common.answer_utils import extract_answer, normalize_answer


def build_prompt(question: str, system_prompt: str) -> str:
    """Build prompt with system message and question."""
    return (
        f"{system_prompt}\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fast batched Regime W rollout collection"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/gsm8k_train.jsonl",
        help="Path to GSM8K training data JSONL file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/abel_sft_merged_fixed",
        help="Path to merged Abel SFT model",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/abel_regime_w_rollouts.jsonl",
        help="Output path for rollout JSONL file",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of training examples to process (default: 500)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8 = all arms at once)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Abel Regime W Rollout Collection (Batched)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Merged model: {args.model_path}")
    print(f"  Data path: {args.data_path}")
    print(f"  Output path: {args.out_path}")
    print(f"  N samples: {args.n_samples}")
    print(f"  Batch size: {args.batch_size}\n")
    
    # Load dataset
    print(f"Loading GSM8K training data from {args.data_path}...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    if args.n_samples > 0:
        dataset = dataset.select(range(min(args.n_samples, len(dataset))))
    print(f"Using {len(dataset)} examples for Regime W rollout collection\n")

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect dtype support
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    dtype_name = "bf16" if use_bf16 else "fp16"
    
    # Load merged model
    print(f"Loading merged Abel model in {dtype_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Build Regime W arms
    arms = build_all_arms()
    print(f"Built {len(arms)} Regime W arms (Wolfram + Maudlin styles)\n")

    print("Starting batched rollout collection...")
    print(f"Generating {len(arms)} trajectories per question\n")

    # Clear output file
    open(args.out_path, "w").close()
    
    outputs = []
    
    with tqdm(total=len(dataset), desc="Processing questions") as pbar:
        for idx in range(len(dataset)):
            question = dataset[idx]["question"]
            gold_answer = dataset[idx]["answer"]

            # Build prompts for all 8 arms
            prompts = [build_prompt(question, arm.system_prompt) for arm in arms]
            
            # Tokenize all prompts
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_length = inputs["input_ids"].shape[1]

            # Generate all 8 trajectories in parallel (one per arm)
            with torch.no_grad():
                # Use arm-specific temperatures by generating one at a time
                # but batching the tokenization for efficiency
                trajectories = []
                
                for i, arm in enumerate(arms):
                    # Get single prompt for this arm
                    arm_input = {k: v[i:i+1] for k, v in inputs.items()}

                    # Set seed for reproducibility if provided in extra_cfg
                    if arm.extra_cfg and "seed" in arm.extra_cfg:
                        torch.manual_seed(arm.extra_cfg["seed"])
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(arm.extra_cfg["seed"])

                    # Generate with arm-specific temperature
                    gen = model.generate(
                        **arm_input,
                        max_new_tokens=512,
                        temperature=arm.temp,
                        top_p=arm.top_p,
                        do_sample=True,
                    )
                    
                    # Decode response only (exclude prompt)
                    response_ids = gen[0][arm_input["input_ids"].shape[1]:]
                    text = tokenizer.decode(response_ids, skip_special_tokens=True)

                    # Create Trajectory object (answer=full text, not extracted)
                    trajectories.append(Trajectory(
                        text=text,
                        reasoning=text,
                        answer=text,  # Store full text for downstream extract_answer calls
                        num_tokens=len(text.split()),
                        arm_name=arm.name,  # Track which arm generated this
                    ))

            # Compute Regime W rewards for all trajectories
            rewards = compute_rewards_for_question(question, trajectories, gold_answer)
            
            # Compute correctness flags for each trajectory
            gold_normalized = normalize_answer(extract_answer(gold_answer))
            
            # Build trajectory records with rewards and correctness flags
            trajectory_records = []
            for t, r in zip(trajectories, rewards):
                # Extract answer from full text for correctness check
                pred_normalized = normalize_answer(extract_answer(t.answer))
                is_correct = (pred_normalized == gold_normalized) and (gold_normalized != "")

                # Extract answer for JSONL output
                extracted_answer = extract_answer(t.answer)

                trajectory_records.append({
                    "full_text": t.text,
                    "answer": extracted_answer,
                    "reasoning": t.reasoning,
                    "num_tokens": t.num_tokens,
                    "reward": float(r),
                    "correct": bool(is_correct),
                    "arm_name": t.arm_name,  # Track which arm generated this
                })

            # Build rollout record
            record = {
                "question": question,
                "gold_answer": gold_answer,
                "trajectories": trajectory_records,
            }
            outputs.append(record)

            # Write every 50 questions to survive crashes
            if len(outputs) >= 50:
                with open(args.out_path, "a", encoding="utf-8") as out_f:
                    for rec in outputs:
                        out_f.write(json.dumps(rec) + "\n")
                outputs = []
            
            pbar.update(1)

    # Final write
    if outputs:
        with open(args.out_path, "a", encoding="utf-8") as out_f:
            for rec in outputs:
                out_f.write(json.dumps(rec) + "\n")

    print(f"\n{'=' * 80}")
    print(f"Rollout collection complete!")
    print(f"Saved Abel Regime W rollouts to: {args.out_path}")
    print(f"Total rollouts: {len(dataset)}")
    print(f"Trajectories per rollout: {len(arms)}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
