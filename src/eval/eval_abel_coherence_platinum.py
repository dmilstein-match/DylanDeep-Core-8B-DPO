import os
import json
from typing import List, Dict
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# H100 optimization flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

from src.common.answer_utils import extract_answer, normalize_answer


OUT_DIR = "outputs"
OUT_PATH = os.path.join(OUT_DIR, "abel_coherence_platinum_eval.jsonl")


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
    # Ensure pad_token exists for batched generation
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For decoder-only models

    # Detect dtype support
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    # ------------------------------------------------------------------
    # 4) Main evaluation loop (batched, H100-optimized)
    # ------------------------------------------------------------------
    correct = correct_so_far
    BATCH_SIZE = 32  # H100-optimized batch size for fast evaluation
    
    print(
        f"\nStarting / resuming evaluation at index {start_index} "
        f"out of {total_examples} total examples (batch_size={BATCH_SIZE}).\n"
    )

    # Prepare remaining examples
    remaining_examples = [
        (i, ex) for i, ex in enumerate(platinum_test) if i >= start_index
    ]
    
    # Process in batches
    for batch_start in range(0, len(remaining_examples), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(remaining_examples))
        batch = remaining_examples[batch_start:batch_end]
        
        # Prepare batch inputs
        batch_prompts = [build_prompt(ex["question"]) for _, ex in batch]
        batch_questions = [ex["question"] for _, ex in batch]
        batch_golds = [ex["answer"] for _, ex in batch]
        batch_indices = [i for i, _ in batch]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)
        input_lengths = inputs["attention_mask"].sum(dim=1)
        
        # Generate batch (deterministic for reproducibility)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Deterministic generation
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode and process each example in batch
        for j, (idx, question, gold) in enumerate(zip(batch_indices, batch_questions, batch_golds)):
            # Extract generated response (skip input tokens)
            input_len = input_lengths[j].item()
            response_ids = gen[j][input_len:]
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
                "index": idx,
                "question": question,
                "gold_answer_raw": gold,
                "gold_answer_extracted": gold_raw,
                "gold_answer_normalized": gold_norm,
                "model_output": text,
                "pred_answer_extracted": pred_raw,
                "pred_answer_normalized": pred_norm,
                "correct": bool(is_correct),
            }
            out_f.write(json.dumps(record) + "\n")
        
        # Flush after each batch
        out_f.flush()
        
        # Progress logging
        processed_count = start_index + batch_end
        current_acc = correct / processed_count
        print(
            f"  Progress: {processed_count}/{total_examples}, "
            f"Current accuracy: {current_acc:.3f}"
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
