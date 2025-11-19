import json
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.regime_w.arms import build_all_arms
from src.regime_w.reward import Trajectory, compute_rewards_for_question, extract_answer, normalize_answer


DATA_PATH = "data/gsm8k_train.jsonl"
BASE_MODEL = "GAIR/Abel-7B-002"
SFT_PATH = "checkpoints/abel_sft_lora"
OUT_PATH = "data/abel_regime_w_rollouts.jsonl"
N_SAMPLES = 500


def load_gsm8k(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_prompt(question: str, system_prompt: str) -> str:
    """Build prompt with system message and question."""
    return (
        f"{system_prompt}\n\n"
        "You are a careful math tutor. Solve the problem step-by-step, "
        "then give the final answer in the format '#### 42'.\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )


def main():
    print("=" * 80)
    print("Abel Regime W Rollout Collection")
    print("=" * 80)
    
    # Load training data
    print(f"\nLoading GSM8K training data from {DATA_PATH}...")
    dataset = load_gsm8k(DATA_PATH)
    dataset = dataset[:N_SAMPLES]
    print(f"Using {len(dataset)} examples for Regime W rollout collection\n")

    # Load tokenizer
    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Abel base model
    print(f"Loading Abel base model in bf16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Load SFT LoRA adapter
    print(f"Loading Abel SFT LoRA adapter from {SFT_PATH}...")
    model = PeftModel.from_pretrained(base_model, SFT_PATH)
    model.eval()

    # Build Regime W arms
    arms = build_all_arms()
    print(f"Built {len(arms)} Regime W arms (Wolfram + Maudlin styles)\n")

    print("Starting rollout collection...")
    print(f"Generating {len(arms)} trajectories per question\n")

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for idx, ex in enumerate(dataset):
            q = ex["question"]
            gold = ex["answer"]

            # Generate trajectories for all arms
            trajectories = []
            for arm in arms:
                prompt = build_prompt(q, arm.system_prompt)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                input_length = inputs["input_ids"].shape[1]
                
                # Generate with arm-specific temperature
                gen = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=arm.temp,
                    top_p=arm.top_p,
                    do_sample=True,
                )
                
                # Decode response only (exclude prompt)
                response_ids = gen[0][input_length:]
                text = tokenizer.decode(response_ids, skip_special_tokens=True)
                ans = extract_answer(text)
                
                trajectories.append(Trajectory(
                    text=text,
                    reasoning=text,
                    answer=ans,
                    num_tokens=len(text.split()),
                ))

            # Compute Regime W rewards for all trajectories
            rewards = compute_rewards_for_question(q, trajectories, gold)
            
            # Compute correctness flags for each trajectory
            gold_normalized = normalize_answer(extract_answer(gold))
            
            # Build trajectory records with rewards and correctness flags
            trajectory_records = []
            for t, r in zip(trajectories, rewards):
                pred_normalized = normalize_answer(extract_answer(t.answer))
                is_correct = (pred_normalized == gold_normalized) and (gold_normalized != "")
                
                trajectory_records.append({
                    "full_text": t.text,
                    "answer": t.answer,
                    "reasoning": t.reasoning,
                    "num_tokens": t.num_tokens,
                    "reward": float(r),
                    "correct": bool(is_correct),
                })

            # Write rollout to JSONL
            record = {
                "question": q,
                "gold_answer": gold,
                "trajectories": trajectory_records,
            }
            out_f.write(json.dumps(record) + "\n")

            # Progress logging
            if (idx + 1) % 20 == 0:
                print(f"  Collected {idx+1}/{len(dataset)} rollouts")

    print(f"\n{'=' * 80}")
    print(f"Rollout collection complete!")
    print(f"Saved Abel Regime W rollouts to: {OUT_PATH}")
    print(f"Total rollouts: {len(dataset)}")
    print(f"Trajectories per rollout: {len(arms)}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
