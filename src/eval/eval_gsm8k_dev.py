import json
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM

DEV_PATH = "data/gsm8k_dev.jsonl"


def load_dev(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_answer(text: str) -> str:
    marker = "####"
    if marker in text:
        return text.split(marker)[-1].strip().split()[0].strip()
    return text.strip()


def build_prompt(question: str) -> str:
    return (
        "You are a careful math tutor. Solve the problem step-by-step, "
        "then give the final answer in the format '#### 42'.\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )


def eval_base_model():
    label = "Base DeepSeek R1 8B"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    print(f"\nEvaluating {label} from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    dev = load_dev(DEV_PATH)
    dev = dev[:200]  # small subset for speed

    correct = 0

    for i, ex in enumerate(dev):
        q = ex["question"]
        gold = ex["answer"]

        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.6,
            top_p=0.9,
        )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        pred = extract_answer(text)
        gold_num = extract_answer(gold)

        if pred == gold_num:
            correct += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(dev)}")

    acc = correct / len(dev)
    print(f"{label} accuracy on dev ({len(dev)} examples): {acc:.3f} ({correct}/{len(dev)})\n")


def eval_lora_sft_model():
    label = "LoRA SFT model"
    model_path = "checkpoints/sft_lora"

    print(f"Evaluating {label} from {model_path}")
    # For PEFT models, load with AutoPeftModelForCausalLM
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dev = load_dev(DEV_PATH)
    dev = dev[:200]  # same subset for fair comparison

    correct = 0

    for i, ex in enumerate(dev):
        q = ex["question"]
        gold = ex["answer"]

        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.6,
            top_p=0.9,
        )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        pred = extract_answer(text)
        gold_num = extract_answer(gold)

        if pred == gold_num:
            correct += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(dev)}")

    acc = correct / len(dev)
    print(f"{label} accuracy on dev ({len(dev)} examples): {acc:.3f} ({correct}/{len(dev)})\n")


def main():
    print("="*60)
    print("GSM8K Dev Set Evaluation: Base vs LoRA SFT")
    print("="*60)
    
    eval_base_model()
    eval_lora_sft_model()
    
    print("="*60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
