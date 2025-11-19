import os
import json
import re
from typing import List, Dict

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


OUT_DIR = "outputs"
OUT_PATH = os.path.join(OUT_DIR, "abel_coherence_platinum_eval.jsonl")


def normalize_answer(ans: str) -> str:
    """
    Normalize answer for comparison.
    - Strip whitespace
    - Lowercase
    - Remove trailing .0 from decimals (42.0 -> 42)
    - Handle commas in numbers (1,000 -> 1000)
    """
    ans = ans.strip().lower()
    ans = ans.replace(",", "")

    # Convert "42.0" to "42" when it's a pure number
    try:
        if "." in ans:
            num = float(ans)
            if num == int(num):
                ans = str(int(num))
    except ValueError:
        pass

    return ans


def extract_answer(text: str) -> str:
    """
    Extract the final numeric answer from the model's output.
    - Prefer the number after '####' if present.
    - Otherwise, take the last integer/decimal in the text.
    """
    marker = "####"
    if marker in text:
        tail = text.split(marker)[-1]
        m = re.search(r"-?\d+\.?\d*", tail)
        if m:
            return m.group(0).strip()

    nums = re.findall(r"-?\d+\.?\d*", text)
    if nums:
        return nums[-1].strip()

    return text.strip()


def build_prompt(question: str) -> str:
    return (
        "You are a careful math tutor. Solve the problem step-by-step, "
        "then give the final answer in the format '#### 42'.\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )


def eval_abel_coherence_on_platinum():
    """
    Evaluate Abel Coherence model on GSM8K Platinum test set with:
      - per-example JSONL logging
      - resume support (skip examples already logged)
    """
    print("=" * 80)
    print("GSM8K PLATINUM Evaluation: Abel Coherence Model (with resume + logging)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1) Load dataset
    # ------------------------------------------------------------------
    print("\nLoading GSM8K Platinum test set from Hugging Face...")
    platinum_test = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
    total_examples = len(platinum_test)
    print(f"Loaded {total_examples} test examples from GSM8K Platinum\n")

    # ------------------------------------------------------------------
    # 2) Prepare output directory and check for existing results
    # ------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)

    start_index = 0
    correct_so_far = 0

    if os.path.exists(OUT_PATH):
        print(f"Found existing eval file at {OUT_PATH}, loading to resume...")
        valid_lines = []
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    valid_lines.append(rec)
                    start_index += 1
                    if rec.get("correct"):
                        correct_so_far += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping corrupted line {line_num} in {OUT_PATH}: {e}")
                    print("  (This can happen if the script crashed mid-write)")
                    continue

        # Always rewrite file with only valid lines to clean up any corruption
        # (Even if empty, this ensures corrupted tails are removed)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            for rec in valid_lines:
                f.write(json.dumps(rec) + "\n")
            f.flush()  # Ensure clean state before appending

        print(
            f"Already evaluated {start_index} examples "
            f"({correct_so_far} correct, acc={correct_so_far / max(1, start_index):.3f})"
        )
    else:
        print(f"No existing eval file found. Will write new results to {OUT_PATH}")

    # If everything was already done, exit early
    if start_index >= total_examples:
        final_acc = correct_so_far / total_examples
        print("\nAll examples already evaluated.")
        print("=" * 80)
        print(
            f"Abel Coherence Model on GSM8K Platinum: {final_acc:.3f} "
            f"({correct_so_far}/{total_examples})"
        )
        print("=" * 80)
        return final_acc

    # Open file in append mode for new results
    out_f = open(OUT_PATH, "a", encoding="utf-8")

    # ------------------------------------------------------------------
    # 3) Load model + tokenizer
    # ------------------------------------------------------------------
    base_model_name = "GAIR/Abel-7B-002"
    lora_path = "checkpoints/abel_coherence_lora"

    print(f"\nLoading Abel Coherence model from {lora_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # Optional safety: ensure pad_token exists
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    import torch
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    # ------------------------------------------------------------------
    # 4) Main evaluation loop (with resume)
    # ------------------------------------------------------------------
    correct = correct_so_far

    print(
        f"\nStarting / resuming evaluation at index {start_index} "
        f"out of {total_examples} total examples.\n"
    )

    for i, ex in enumerate(platinum_test):
        if i < start_index:
            continue  # skip examples we've already logged

        q = ex["question"]
        gold = ex["answer"]

        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        gen = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.6,
            top_p=0.9,
        )
        response_ids = gen[0][input_length:]
        text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # Extract and normalize both answers
        pred_raw = extract_answer(text)
        gold_raw = extract_answer(gold)

        pred_norm = normalize_answer(pred_raw)
        gold_norm = normalize_answer(gold_raw)

        is_correct = (pred_norm == gold_norm) and (pred_norm != "")
        if is_correct:
            correct += 1

        # Write per-example record to JSONL
        record = {
            "index": i,
            "question": q,
            "gold_answer_raw": gold,
            "gold_answer_extracted": gold_raw,
            "gold_answer_normalized": gold_norm,
            "model_output": text,
            "pred_answer_extracted": pred_raw,
            "pred_answer_normalized": pred_norm,
            "correct": bool(is_correct),
        }
        out_f.write(json.dumps(record) + "\n")

        # Flush occasionally so progress is not lost if the job dies
        if (i + 1) % 10 == 0:
            out_f.flush()

        # Progress logging
        if (i + 1) % 50 == 0 or (i + 1) == total_examples:
            current_n = i + 1
            current_acc = correct / current_n
            print(
                f"  Progress: {current_n}/{total_examples}, "
                f"Current accuracy (over seen examples): {current_acc:.3f}"
            )

    out_f.close()

    # ------------------------------------------------------------------
    # 5) Final summary
    # ------------------------------------------------------------------
    acc = correct / total_examples
    print("\n" + "=" * 80)
    print(f"Abel Coherence Model on GSM8K Platinum: {acc:.3f} ({correct}/{total_examples})")
    print("=" * 80)
    print("Evaluation complete!")

    return acc


if __name__ == "__main__":
    eval_abel_coherence_on_platinum()
