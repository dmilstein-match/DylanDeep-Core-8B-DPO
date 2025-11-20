#!/usr/bin/env python3
"""
OPTIMIZED Regime W rollout collection using vLLM with multi-question batching.
Expected speedup: 2-4x faster than sequential version.

Key optimizations:
1. Batch multiple questions together (not just 8 arms per question)
2. Reduced max_model_len from 8192 to 2048 (saves 4-6GB per GPU)
3. Disabled enforce_eager for CUDA graph optimization (20-30% faster)
4. Increased gpu_memory_utilization to 0.95 (safe on H100)
5. Added chunked prefill for better batching
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
    return f"{system_prompt}\n\nProblem:\n{question}\n\nSolution:\n"


def main():
    parser = argparse.ArgumentParser(
        description="OPTIMIZED Regime W rollout collection with vLLM (multi-question batching)"
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
    parser.add_argument(
        "--question_batch_size",
        type=int,
        default=16,
        help="Number of questions to batch together (default: 16, total batch=16×8=128)",
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
        print("Abel Regime W Rollout Collection (vLLM OPTIMIZED)")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Merged model: {args.model_path}")
        print(f"  Data path: {args.data_path}")
        print(f"  Output path: {args.out_path}")
        print(f"  N samples: {args.n_samples}")
        print(f"  GPUs detected: {gpu_count}")
        print(f"  Tensor parallel size: {world_size}")
        print(f"  Question batch size: {args.question_batch_size}")
        print(f"  Total batch size: {args.question_batch_size * 8} (questions × arms)")
        print("\nOptimizations enabled:")
        print("  ✓ Multi-question batching")
        print("  ✓ Reduced max_model_len (8192 → 2048)")
        print("  ✓ CUDA graphs enabled (enforce_eager=False)")
        print("  ✓ High GPU memory utilization (95%)")
        print("  ✓ Chunked prefill for better batching\n")

    # Load training data
    if rank == 0:
        print(f"Loading GSM8K training data from {args.data_path}...")
    dataset = load_gsm8k(args.data_path)
    dataset = dataset[:args.n_samples]

    if rank == 0:
        print(f"Using {len(dataset)} examples for Regime W rollout collection\n")

    # Build Regime W arms (11 total: 8 variance + 3 diagnostic probes)
    arms = build_all_arms()
    if rank == 0:
        print(f"Built {len(arms)} Regime W arms")
        print("Initializing vLLM engine with OPTIMIZED settings...\n")

    # Initialize vLLM with OPTIMIZED settings for H100
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=world_size,
        # OPTIMIZATION 1: Reduce max_model_len from 8192 to 2048
        # GSM8K prompts: 200-400 tokens, solutions: 200-500 tokens
        # Total: ~800 max, so 2048 gives 2.5× headroom while saving 4-6GB/GPU
        max_model_len=2048,

        # OPTIMIZATION 2: Use BF16 for H100 precision
        dtype="bfloat16",

        # OPTIMIZATION 3: Disable enforce_eager to enable CUDA graphs (20-30% faster)
        # enforce_eager=True was needed for stability, but H100 + vLLM 0.6+ handles this well
        enforce_eager=False,

        # OPTIMIZATION 4: Increase GPU memory utilization from 90% to 95% (safe on H100)
        gpu_memory_utilization=0.95,

        # OPTIMIZATION 5: Enable chunked prefill for better batching
        # Allows vLLM to batch prompts of different lengths more efficiently
        enable_chunked_prefill=True,

        # Keep KV cache efficient
        max_num_seqs=256,  # Allow up to 256 sequences in flight
        swap_space=4,      # 4GB CPU offloading if needed
    )

    if rank == 0:
        print("vLLM engine initialized successfully!")
        print(f"Starting OPTIMIZED rollout collection...")
        print(f"Generating {len(arms)} trajectories per question\n")

    # Process dataset in batches of questions (not just arms)
    all_rollouts = []

    # Batch questions together for massive parallelism
    question_batch_size = args.question_batch_size
    n_batches = (len(dataset) + question_batch_size - 1) // question_batch_size

    for batch_idx in range(n_batches):
        # Get batch of questions
        start_idx = batch_idx * question_batch_size
        end_idx = min(start_idx + question_batch_size, len(dataset))
        batch_examples = dataset[start_idx:end_idx]

        # Build prompts for ALL questions × ALL arms in this batch
        # Total: question_batch_size × 8 arms = up to 128 prompts at once
        all_prompts = []
        all_sampling_params = []
        prompt_metadata = []  # Track which question/arm each prompt belongs to

        for ex_idx, ex in enumerate(batch_examples):
            q = ex["question"]

            for arm in arms:
                prompt = build_prompt(q, arm.system_prompt)
                all_prompts.append(prompt)

                # Create sampling params for this arm
                seed = arm.extra_cfg.get("seed", 42) if arm.extra_cfg else 42
                all_sampling_params.append(
                    SamplingParams(
                        temperature=arm.temp,
                        top_p=arm.top_p,
                        max_tokens=512,
                        seed=seed,
                        stop_token_ids=[],
                    )
                )

                # Track metadata
                prompt_metadata.append({
                    "ex_idx": ex_idx,
                    "question": q,
                    "gold": ex["answer"],
                    "arm_name": arm.name,
                })

        # Generate ALL prompts in this batch at once
        # vLLM will distribute across 8 GPUs with tensor parallelism
        # Total parallelism: 128 sequences across 8 GPUs = ~16 per GPU
        outputs = llm.generate(all_prompts, all_sampling_params, use_tqdm=False)

        # Group outputs back by question
        question_outputs = {}  # ex_idx -> list of (text, arm_name) tuples

        for output, meta in zip(outputs, prompt_metadata):
            text = output.outputs[0].text.strip()
            ex_idx = meta["ex_idx"]

            if ex_idx not in question_outputs:
                question_outputs[ex_idx] = {
                    "question": meta["question"],
                    "gold": meta["gold"],
                    "trajectories": []
                }

            question_outputs[ex_idx]["trajectories"].append({
                "text": text,
                "arm_name": meta["arm_name"],
            })

        # Process each question's trajectories
        for ex_idx in sorted(question_outputs.keys()):
            q_data = question_outputs[ex_idx]
            q = q_data["question"]
            gold = q_data["gold"]

            # Build Trajectory objects for reward computation
            trajectories = []
            for traj_data in q_data["trajectories"]:
                text = traj_data["text"]
                trajectories.append(Trajectory(
                    text=text,
                    reasoning=text,
                    answer=text,  # Store full text for extract_answer
                    num_tokens=len(text.split()),
                    arm_name=traj_data["arm_name"],
                ))

            # Compute Regime W rewards
            rewards = compute_rewards_for_question(q, trajectories, gold)

            # Build trajectory records with correctness flags
            gold_normalized = normalize_answer(extract_answer(gold))
            trajectory_records = []

            for t, r in zip(trajectories, rewards):
                pred_normalized = normalize_answer(extract_answer(t.answer))
                is_correct = (pred_normalized == gold_normalized) and (gold_normalized != "")
                extracted_answer = extract_answer(t.answer)

                trajectory_records.append({
                    "full_text": t.text,
                    "answer": extracted_answer,
                    "reasoning": t.reasoning,
                    "num_tokens": t.num_tokens,
                    "reward": float(r),
                    "correct": bool(is_correct),
                    "arm_name": t.arm_name,
                })

            # Build rollout record
            record = {
                "question": q,
                "gold_answer": gold,
                "trajectories": trajectory_records,
            }
            all_rollouts.append(record)

        # Progress logging
        if rank == 0:
            print(f"  Batch {batch_idx+1}/{n_batches}: Collected {len(all_rollouts)}/{len(dataset)} rollouts")

    # Write output file
    if rank == 0:
        print(f"\nWriting rollouts to {args.out_path}...")
        with open(args.out_path, "w", encoding="utf-8") as out_f:
            for record in all_rollouts:
                out_f.write(json.dumps(record) + "\n")

        print(f"\n{'=' * 80}")
        print(f"✓ Rollout collection complete!")
        print(f"Saved Abel Regime W rollouts to: {args.out_path}")
        print(f"Total rollouts: {len(all_rollouts)}")
        print(f"Trajectories per rollout: {len(arms)}")
        print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
