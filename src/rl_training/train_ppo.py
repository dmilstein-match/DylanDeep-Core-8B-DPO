import json
import random
from typing import List, Dict

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from trl import PPOTrainer, PPOConfig

SFT_PATH = "checkpoints/sft_lora"
ROLLOUT_PATH = "data/regime_w_rollouts.jsonl"
RL_OUTPUT_DIR = "checkpoints/ppo_regime_w"


def load_rollouts(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_prompt(question: str) -> str:
    return (
        "You are a careful math tutor. Solve the problem step-by-step, "
        "then give the final answer in the format '#### 42'.\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )


def main():
    print("Loading Regime W rollouts from", ROLLOUT_PATH)
    rollouts = load_rollouts(ROLLOUT_PATH)
    random.seed(42)
    random.shuffle(rollouts)

    print(f"Loaded {len(rollouts)} rollout records.")

    print("Loading SFT LoRA model from", SFT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(SFT_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        SFT_PATH,
        device_map="auto",
    )

    config = PPOConfig(
        model_name=SFT_PATH,
        learning_rate=5e-6,
        batch_size=8,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
        target_kl=0.1,
        adap_kl_ctrl=True,
        max_grad_norm=1.0,
        output_dir=RL_OUTPUT_DIR,
        log_with=None,
    )

    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
    )

    print("Starting PPO training loop using offline Regime W rollouts...")
    num_epochs = 1

    for epoch in range(num_epochs):
        print(f"\n=== PPO Epoch {epoch+1}/{num_epochs} ===")
        for start in range(0, len(rollouts), config.batch_size):
            batch = rollouts[start : start + config.batch_size]
            if not batch:
                break

            queries = []
            responses = []
            rewards = []

            for rec in batch:
                q = rec["question"]
                reward = rec["reward"]
                arm_outputs = rec["arm_outputs"]

                chosen = random.choice(arm_outputs)
                resp_text = chosen["full_text"]

                queries.append(build_prompt(q))
                responses.append(resp_text)
                rewards.append(reward)

            stats = ppo_trainer.step(queries, responses, rewards)

            step_idx = start // config.batch_size
            if step_idx % 10 == 0:
                avg_reward = sum(rewards) / len(rewards)
                print(
                    f"  Step {step_idx}, "
                    f"batch_size={len(batch)}, "
                    f"avg_reward={avg_reward:.3f}, "
                    f"kl={stats.get('kl', 0):.4f}"
                )

    print("Saving PPO + Regime W model to", RL_OUTPUT_DIR)
    ppo_trainer.model.save_pretrained(RL_OUTPUT_DIR)
    tokenizer.save_pretrained(RL_OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
