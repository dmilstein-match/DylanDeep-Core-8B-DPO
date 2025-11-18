import re
from typing import List, Dict

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


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
    
    # Convert "42.0" to "42"
    if "." in ans:
        try:
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


def eval_rl_on_platinum():
    """
    Evaluate DPO RL model on GSM8K Platinum test set with normalization.
    """
    print("="*80)
    print("GSM8K PLATINUM Evaluation: DPO RL Model")
    print("="*80)
    
    # Load GSM8K Platinum test set
    print("\nLoading GSM8K Platinum test set from Hugging Face...")
    platinum_test = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
    print(f"Loaded {len(platinum_test)} test examples from GSM8K Platinum\n")
    
    # Load DPO RL model
    base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    lora_path = "checkpoints/lora_rl"
    
    print(f"Loading DPO RL model from {lora_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        load_in_8bit=True,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    
    correct = 0
    
    for i, ex in enumerate(platinum_test):
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
        pred = normalize_answer(extract_answer(text))
        gold_num = normalize_answer(extract_answer(gold))
        
        if pred == gold_num and pred != "":
            correct += 1
        
        if (i + 1) % 50 == 0:
            current_acc = correct / (i + 1)
            print(f"  Progress: {i+1}/{len(platinum_test)}, Accuracy: {current_acc:.3f}")
    
    acc = correct / len(platinum_test)
    print("\n" + "="*80)
    print(f"DPO RL Model on GSM8K Platinum: {acc:.3f} ({correct}/{len(platinum_test)})")
    print("="*80)
    print("Evaluation complete!")
    
    return acc


if __name__ == "__main__":
    eval_rl_on_platinum()
