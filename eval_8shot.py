#!/usr/bin/env python3
"""
8-Shot Evaluation Script
Generates 8 answers per question with optimized temperature, takes majority vote
Expected accuracy: 80%+
"""
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from collections import Counter
from src.common.answer_utils import extract_answer, normalize_answer

# H100 optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

BASE_MODEL = "GAIR/Abel-7B-002"
LORA_PATH = "checkpoints/abel_coherence_lora_v3"
OUTPUT_FILE = "outputs/abel_coherence_8shot_eval.jsonl"
BATCH_SIZE = 4  # Process 4 questions at a time (each generates 8 answers)

def build_prompt(question: str) -> str:
    """Build evaluation prompt - concise and clear"""
    return (
        "Solve this math problem step-by-step. Show your work clearly and concisely.\n"
        "End with 'Answer: [number]'.\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )

def majority_vote(answers):
    """Take majority vote from 8 generated answers"""
    normalized = [normalize_answer(extract_answer(ans)) for ans in answers]
    # Filter out empty answers
    normalized = [a for a in normalized if a]
    if not normalized:
        return None
    # Return most common answer
    counter = Counter(normalized)
    return counter.most_common(1)[0][0]

def main():
    print("=" * 80)
    print("8-Shot Evaluation")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Temperature: 0.2")
    print(f"  Shots per question: 8")
    print(f"  Strategy: Majority voting")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Output: {OUTPUT_FILE}\n")

    # Load tokenizer
    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect dtype
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Load model
    print(f"Loading model from {LORA_PATH} in {dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
        use_safetensors=False
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    # Load test dataset
    print("\nLoading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")
    print(f"Loaded {len(dataset)} test examples\n")

    # Check for existing results
    existing_results = []
    try:
        with open(OUTPUT_FILE, 'r') as f:
            existing_results = [json.loads(line) for line in f]
        print(f"Found {len(existing_results)} existing results, resuming...\n")
    except FileNotFoundError:
        print("Starting fresh evaluation...\n")

    # Evaluate
    correct = sum(1 for r in existing_results if r.get('correct', False))

    with open(OUTPUT_FILE, 'a') as out_f:
        for i in tqdm(range(len(existing_results), len(dataset)), desc="Evaluating"):
            example = dataset[i]
            question = example['question']
            gold_answer = example['answer'].split("####")[1].strip()

            # Build prompt
            prompt = build_prompt(question)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Generate 8 answers with temp=0.2
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.95,
                    num_return_sequences=8,  # 8-shot
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Decode all 8 answers
            generated_texts = []
            for output in outputs:
                text = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                generated_texts.append(text.strip())

            # Majority vote
            predicted = majority_vote(generated_texts)
            gold_norm = normalize_answer(gold_answer)
            is_correct = (predicted == gold_norm) and (gold_norm != "")

            if is_correct:
                correct += 1

            # Log result
            result = {
                "question": question,
                "gold_answer": gold_answer,
                "generated_answers": generated_texts,
                "majority_vote": predicted,
                "correct": is_correct,
                "accuracy_so_far": correct / (i + 1)
            }
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    # Final results
    total = len(dataset)
    accuracy = 100 * correct / total

    print("\n" + "=" * 80)
    print("8-Shot Evaluation Complete!")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  Correct: {correct} / {total}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
