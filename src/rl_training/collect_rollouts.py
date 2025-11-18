import json
from typing import List, Dict

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

from src.regime_w.arms import build_all_arms
from src.regime_w.reward import compute_reward, extract_answer

DATA_PATH = "data/gsm8k_train.jsonl"
SFT_PATH = "checkpoints/sft_lora"
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUT_PATH = "data/regime_w_rollouts.jsonl"


def load_gsm8k(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_prompt(system_prompt: str, question: str) -> str:
    return (
        system_prompt
        + "\n\n"
        "Solve the problem step by step and end with the final answer "
        "in the format '#### 42'.\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )


def main():
    print("Loading GSM8K train data...")
    dataset = load_gsm8k(DATA_PATH)
    dataset = dataset[:500]
    print(f"Using {len(dataset)} examples for Regime W rollout collection.")

    print("Loading tokenizer from base model", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading LoRA SFT model from", SFT_PATH)
    model = AutoPeftModelForCausalLM.from_pretrained(
        SFT_PATH,
        device_map="auto",
    )

    arms = build_all_arms()
    print(f"Built {len(arms)} arms.")

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for idx, ex in enumerate(dataset):
            q = ex["question"]
            gold = ex["answer"]

            arm_outputs = []

            for arm in arms:
                prompt = build_prompt(arm.system_prompt, q)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                input_length = inputs["input_ids"].shape[1]
                gen = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=arm.temp,
                    top_p=arm.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response_ids = gen[0][input_length:]
                text = tokenizer.decode(response_ids, skip_special_tokens=True)
                ans = extract_answer(text)

                arm_outputs.append({
                    "arm_name": arm.name,
                    "full_text": text,
                    "reasoning": text,
                    "answer": ans,
                })

            reward = compute_reward(q, arm_outputs, gold)

            record = {
                "question": q,
                "gold_answer": gold,
                "arm_outputs": arm_outputs,
                "reward": reward,
            }
            out_f.write(json.dumps(record) + "\n")

            if (idx + 1) % 20 == 0:
                print(f"  Collected {idx+1}/{len(dataset)} rollouts")

    print("Saved Regime W rollouts to", OUT_PATH)


if __name__ == "__main__":
    main()
