#!/usr/bin/env python3
"""
Fast Regime W rollout collection using vLLM with 8-GPU tensor parallelism.
Generates 8 trajectories per question (Wolfram + Maudlin styles across 4 temperatures).
"""
import argparse
import json
import os
from typing import List, Dict

from vllm import LLM, SamplingParams

from src.regime_w.arms import build_all_arms
from src.regime_w.reward import Trajectory, compute_rewards_for_question
from src.common.answer_utils import extract_answer, normalize_answer


def load_gsm8k(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_prompt(question: str, system_prompt: str) -> str:
    """Build prompt with system message and question."""
    # System prompt already contains answer format instruction
    return f"{system_prompt}\n\nProblem:\n{question}\n\nSolution:\n"


def main():
    parser = argparse.ArgumentParser(
        description="Fast Regime W rollout collection with vLLM (8-GPU tensor parallel)"
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
        default="checkpoints/abel_sft_merged",
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
    args = parser.parse_args()

    # Auto-detect GPU count
    import subprocess
    try:
        gpu_count = len(subprocess.check_output(
            ["nvidia-smi", "-L"], encoding="utf-8"
        ).strip().split("\n"))
    except:
        gpu_count = 1

    # Override with environment variable if set
    world_size = int(os.environ.get("WORLD_SIZE", gpu_count))
    rank = int(os.environ.get("RANK", 0))

    if rank == 0:
        print("=" * 80)
        print("Abel Regime W Rollout Collection (vLLM Optimized)")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Merged model: {args.model_path}")
        print(f"  Data path: {args.data_path}")
        print(f"  Output path: {args.out_path}")
        print(f"  N samples: {args.n_samples}")
        print(f"  GPUs detected: {gpu_count}")
        print(f"  Tensor parallel size: {world_size}\n")
    
    # Load training data
    if rank == 0:
        print(f"Loading GSM8K training data from {args.data_path}...")
    dataset = load_gsm8k(args.data_path)
    dataset = dataset[:args.n_samples]
    
    if rank == 0:
        print(f"Using {len(dataset)} examples for Regime W rollout collection\n")

    # Build Regime W arms (8 total: 4 Wolfram + 4 Maudlin with varying temps)
    arms = build_all_arms()
    if rank == 0:
        print(f"Built {len(arms)} Regime W arms (Wolfram + Maudlin styles)")
        print("Initializing vLLM engine with tensor parallelism...\n")

    # Initialize vLLM with tensor parallelism across all GPUs
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=world_size,
        max_model_len=8192,
        dtype="bfloat16" if world_size > 1 else "auto",  # bf16 for multi-GPU
        enforce_eager=True,  # Helps with some 7B models
    )

    if rank == 0:
        print("vLLM engine initialized successfully!")
        print(f"Starting rollout collection...")
        print(f"Generating {len(arms)} trajectories per question\n")

    # Each question generates all 8 arms at once (batched)
    all_rollouts = []
    
    for idx, ex in enumerate(dataset):
        q = ex["question"]
        gold = ex["answer"]

        # Build prompts for all arms
        prompts = []
        sampling_params_list = []
        arm_names = []  # Track arm names for trajectory labeling

        for arm in arms:
            prompt = build_prompt(q, arm.system_prompt)
            prompts.append(prompt)
            arm_names.append(arm.name)

            # Create sampling params for this arm's temperature + seed
            seed = arm.extra_cfg.get("seed", 42) if arm.extra_cfg else 42
            sampling_params_list.append(
                SamplingParams(
                    temperature=arm.temp,
                    top_p=arm.top_p,
                    max_tokens=512,
                    seed=seed,  # Reproducible generation
                    stop_token_ids=[],  # Abel uses default EOS
                )
            )

        # Generate all 8 trajectories in parallel for this question
        # vLLM will batch these efficiently across all GPUs
        outputs = llm.generate(prompts, sampling_params_list, use_tqdm=False)
        
        # Extract generated texts
        # IMPORTANT: Store full text in Trajectory.answer (not extracted)
        # Downstream functions (compute_rewards_for_question, s_end_for_question)
        # expect to call extract_answer on the full text
        trajectories = []
        for output, arm_name in zip(outputs, arm_names):
            text = output.outputs[0].text.strip()

            trajectories.append(Trajectory(
                text=text,
                reasoning=text,
                answer=text,  # Store full text, not extract_answer(text)
                num_tokens=len(text.split()),
                arm_name=arm_name,  # Track which arm generated this
            ))

        # Compute Regime W rewards for all trajectories
        rewards = compute_rewards_for_question(q, trajectories, gold)
        
        # Compute correctness flags for each trajectory
        gold_normalized = normalize_answer(extract_answer(gold))
        
        # Build trajectory records with rewards and correctness flags
        trajectory_records = []
        for t, r in zip(trajectories, rewards):
            # Extract and normalize answer for correctness check
            pred_normalized = normalize_answer(extract_answer(t.answer))
            is_correct = (pred_normalized == gold_normalized) and (gold_normalized != "")

            # Extract answer for JSONL output (not full text)
            extracted_answer = extract_answer(t.answer)

            trajectory_records.append({
                "full_text": t.text,
                "answer": extracted_answer,  # Store extracted answer in output
                "reasoning": t.reasoning,
                "num_tokens": t.num_tokens,
                "reward": float(r),
                "correct": bool(is_correct),
                "arm_name": t.arm_name,  # Track which arm generated this
            })

        # Build rollout record
        record = {
            "question": q,
            "gold_answer": gold,
            "trajectories": trajectory_records,
        }
        all_rollouts.append(record)

        # Progress logging
        if rank == 0 and (idx + 1) % 20 == 0:
            print(f"  Collected {idx+1}/{len(dataset)} rollouts")

    # Write output file (all ranks write to same file since tensor parallel means same data)
    if rank == 0:
        print(f"\nWriting rollouts to {args.out_path}...")
        with open(args.out_path, "w", encoding="utf-8") as out_f:
            for record in all_rollouts:
                out_f.write(json.dumps(record) + "\n")

        print(f"\n{'=' * 80}")
        print(f"Rollout collection complete!")
        print(f"Saved Abel Regime W rollouts to: {args.out_path}")
        print(f"Total rollouts: {len(all_rollouts)}")
        print(f"Trajectories per rollout: {len(arms)}")
        print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
