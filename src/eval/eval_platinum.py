import json
import re
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

TEST_PATH = "data/gsm8k_test.jsonl"


def load_test(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_answer(text: str) -> str:
    """
    Extract the final numeric answer from the model's output.
    - Prefer the number after '####' if present.
    - Otherwise, take the last integer/decimal in the text.
    Returns it as a plain string (e.g., '42' or '3.5').
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


def eval_base_model(test_data: List[Dict]):
    label = "Base DeepSeek R1 8B"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    print(f"\nEvaluating {label} from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    correct = 0

    for i, ex in enumerate(test_data):
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
        pred = extract_answer(text)
        gold_num = extract_answer(gold)

        if pred == gold_num:
            correct += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(test_data)}")

    acc = correct / len(test_data)
    print(f"{label} accuracy on test ({len(test_data)} examples): {acc:.3f} ({correct}/{len(test_data)})\n")
    return acc


def eval_lora_sft_model(test_data: List[Dict]):
    label = "LoRA SFT model"
    base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    lora_path = "checkpoints/sft_lora"

    print(f"Evaluating {label} from {lora_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        load_in_8bit=True,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    correct = 0

    for i, ex in enumerate(test_data):
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
        pred = extract_answer(text)
        gold_num = extract_answer(gold)

        if pred == gold_num:
            correct += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(test_data)}")

    acc = correct / len(test_data)
    print(f"{label} accuracy on test ({len(test_data)} examples): {acc:.3f} ({correct}/{len(test_data)})\n")
    return acc


def eval_dpo_rl_model(test_data: List[Dict]):
    label = "DPO RL model"
    base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    lora_path = "checkpoints/lora_rl"

    print(f"Evaluating {label} from {lora_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        load_in_8bit=True,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    correct = 0

    for i, ex in enumerate(test_data):
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
        pred = extract_answer(text)
        gold_num = extract_answer(gold)

        if pred == gold_num:
            correct += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(test_data)}")

    acc = correct / len(test_data)
    print(f"{label} accuracy on test ({len(test_data)} examples): {acc:.3f} ({correct}/{len(test_data)})\n")
    return acc


def main():
    print("="*80)
    print("GSM8K Test Set Evaluation: Base vs SFT vs DPO RL")
    print("="*80)
    
    print("\nLoading GSM8K test data...")
    test_data = load_test(TEST_PATH)
    print(f"Loaded {len(test_data)} test examples\n")
    
    base_acc = eval_base_model(test_data)
    sft_acc = eval_lora_sft_model(test_data)
    rl_acc = eval_dpo_rl_model(test_data)
    
    print("="*80)
    print("Final Results:")
    print(f"  Base Model:      {base_acc:.3f}")
    print(f"  SFT LoRA:        {sft_acc:.3f}")
    print(f"  DPO RL:          {rl_acc:.3f}")
    print("="*80)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
