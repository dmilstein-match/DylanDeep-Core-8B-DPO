#!/usr/bin/env python3
"""
Multi-GPU Regime W rollout collection using 8 H100 GPUs in parallel.
Processes 8 questions simultaneously for ~8x speedup (30-45 min vs 3-4 hours).
"""
import argparse
import json
import os
from typing import List, Dict
from multiprocessing import Process, Queue

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
        "You are a careful math tutor. Solve the problem step-by-step, "
        "then give the final answer in the format '#### 42'.\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )


def gpu_worker(
    gpu_id: int,
    model_path: str,
    questions_queue: Queue,
    results_queue: Queue,
):
    """
    Worker process that loads model on a specific GPU and processes questions.
    
    Args:
        gpu_id: GPU device ID (0-7)
        model_path: Path to merged model
        questions_queue: Queue of (idx, question, gold_answer) tuples
        results_queue: Queue for results
    """
    # Set this process to use specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")  # Will be the only visible GPU
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Detect dtype
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    # Load model on this GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    # Build arms
    arms = build_all_arms()
    
    # Process questions from queue
    while True:
        item = questions_queue.get()
        if item is None:  # Poison pill to stop worker
            break
        
        idx, question, gold_answer = item
        
        # Build prompts for all 8 arms
        prompts = [build_prompt(question, arm.system_prompt) for arm in arms]
        
        trajectories = []
        
        # Generate one trajectory per arm with arm-specific temperature
        for i, arm in enumerate(arms):
            # Tokenize single prompt
            inputs = tokenizer([prompts[i]], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_length = inputs["input_ids"].shape[1]
            
            # Generate with arm-specific temperature
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=arm.temp,
                    top_p=arm.top_p,
                    do_sample=True,
                )
            
            # Decode response only (exclude prompt)
            response_ids = gen[0][input_length:]
            text = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Create Trajectory object
            trajectories.append(Trajectory(
                text=text,
                reasoning=text,
                answer=text,  # Store full text for downstream extract_answer calls
                num_tokens=len(text.split()),
            ))
        
        # Compute Regime W rewards
        rewards = compute_rewards_for_question(question, trajectories, gold_answer)
        
        # Compute correctness flags
        gold_normalized = normalize_answer(extract_answer(gold_answer))
        
        # Build trajectory records
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
            })
        
        # Build result record
        record = {
            "idx": idx,
            "question": question,
            "gold_answer": gold_answer,
            "trajectories": trajectory_records,
        }
        
        results_queue.put(record)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Regime W rollout collection (8x speedup)"
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
        "--n_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use (default: 8)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Abel Regime W Rollout Collection (Multi-GPU)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Merged model: {args.model_path}")
    print(f"  Data path: {args.data_path}")
    print(f"  Output path: {args.out_path}")
    print(f"  N samples: {args.n_samples}")
    print(f"  N GPUs: {args.n_gpus}\n")
    
    # Load dataset
    print(f"Loading GSM8K training data from {args.data_path}...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    if args.n_samples > 0:
        dataset = dataset.select(range(min(args.n_samples, len(dataset))))
    print(f"Using {len(dataset)} examples for Regime W rollout collection\n")

    # Build Regime W arms
    arms = build_all_arms()
    print(f"Built {len(arms)} Regime W arms (Wolfram + Maudlin styles)\n")

    print(f"Starting multi-GPU rollout collection with {args.n_gpus} GPUs...")
    print(f"Generating {len(arms)} trajectories per question\n")
    print(f"Expected speedup: ~{args.n_gpus}x faster than single-GPU\n")
    print(f"Estimated time: ~{3.5 * 60 / args.n_gpus:.0f} minutes\n")

    # Create queues for work distribution
    questions_queue = Queue(maxsize=args.n_gpus * 2)
    results_queue = Queue()

    # Start worker processes
    workers = []
    for gpu_id in range(args.n_gpus):
        p = Process(
            target=gpu_worker,
            args=(gpu_id, args.model_path, questions_queue, results_queue),
        )
        p.start()
        workers.append(p)
    
    print(f"Started {args.n_gpus} GPU workers\n")

    # Clear output file
    open(args.out_path, "w").close()
    
    # Feed questions to workers
    def producer():
        for idx in range(len(dataset)):
            question = dataset[idx]["question"]
            gold_answer = dataset[idx]["answer"]
            questions_queue.put((idx, question, gold_answer))
        
        # Send poison pills to stop workers
        for _ in range(args.n_gpus):
            questions_queue.put(None)
    
    # Start producer in separate thread
    from threading import Thread
    producer_thread = Thread(target=producer)
    producer_thread.start()
    
    # Collect results and write to file
    results = []
    completed = 0
    
    with tqdm(total=len(dataset), desc="Collecting rollouts") as pbar:
        while completed < len(dataset):
            record = results_queue.get()
            results.append(record)
            completed += 1
            pbar.update(1)
            
            # Write every 50 results
            if len(results) >= 50:
                # Sort by index to maintain order
                results.sort(key=lambda x: x["idx"])
                with open(args.out_path, "a", encoding="utf-8") as out_f:
                    for rec in results:
                        # Remove idx from output
                        output_rec = {k: v for k, v in rec.items() if k != "idx"}
                        out_f.write(json.dumps(output_rec) + "\n")
                results = []
    
    # Final write
    if results:
        results.sort(key=lambda x: x["idx"])
        with open(args.out_path, "a", encoding="utf-8") as out_f:
            for rec in results:
                output_rec = {k: v for k, v in rec.items() if k != "idx"}
                out_f.write(json.dumps(output_rec) + "\n")
    
    # Wait for all workers to finish
    producer_thread.join()
    for w in workers:
        w.join()

    print(f"\n{'=' * 80}")
    print(f"Rollout collection complete!")
    print(f"Saved Abel Regime W rollouts to: {args.out_path}")
    print(f"Total rollouts: {len(dataset)}")
    print(f"Trajectories per rollout: {len(arms)}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
