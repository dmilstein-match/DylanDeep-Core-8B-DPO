import json
import re
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

TEST_PATH = "data/gsm8k_test.jsonl"
OUTPUT_LOG = "predictions_log.jsonl"


def load_test(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_answer(text: str) -> str:
    """
    Extract the final numeric answer from the model's output.
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


def eval_model_with_logging(model_name: str, lora_path: str = None):
    """
    Evaluate a model and log all predictions to file.
    """
    print(f"\nEvaluating: {model_name}")
    print(f"LoRA path: {lora_path if lora_path else 'None (base model)'}")
    
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    
    if lora_path:
        base_model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            device_map="auto",
            load_in_8bit=True,
        )
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            device_map="auto"
        )

    test_data = load_test(TEST_PATH)
    correct = 0
    logs = []

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

        match = (pred == gold_num)
        if match:
            correct += 1
        
        # Log this prediction
        logs.append({
            "model": model_name,
            "question": q,
            "gold_answer": gold,
            "gold_extracted": gold_num,
            "predicted_text": text[:200],  # First 200 chars
            "predicted_extracted": pred,
            "match": match
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(test_data)}, Accuracy so far: {correct/(i+1):.3f}")

    acc = correct / len(test_data)
    print(f"\n{model_name} accuracy: {acc:.3f} ({correct}/{len(test_data)})\n")
    
    return logs, acc


def main():
    print("="*80)
    print("GSM8K Test Set Evaluation WITH LOGGING")
    print("="*80)
    
    all_logs = []
    
    # Eval Base Model
    logs_base, _ = eval_model_with_logging("Base DeepSeek R1 8B")
    all_logs.extend(logs_base)
    
    # Eval SFT LoRA
    logs_sft, _ = eval_model_with_logging("SFT LoRA", "checkpoints/sft_lora")
    all_logs.extend(logs_sft)
    
    # Eval DPO RL
    logs_dpo, _ = eval_model_with_logging("DPO RL", "checkpoints/lora_rl")
    all_logs.extend(logs_dpo)
    
    # Save all logs
    print(f"\nSaving all predictions to {OUTPUT_LOG}")
    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")
    
    print(f"Saved {len(all_logs)} predictions to {OUTPUT_LOG}")
    print("You can now manually inspect predictions to verify normalization would help.")


if __name__ == "__main__":
    main()
