import json
from typing import List, Dict

ROLLOUTS_PATH = "data/regime_w_rollouts.jsonl"
PREFERENCES_PATH = "data/preferences.jsonl"


def load_rollouts(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    print(f"Loading rollouts from {ROLLOUTS_PATH}...")
    rollouts = load_rollouts(ROLLOUTS_PATH)
    print(f"Loaded {len(rollouts)} rollout records.")

    preferences = []
    skipped = 0

    for record in rollouts:
        question = record["question"]
        gold_answer = record["gold_answer"]
        trajectories = record["trajectories"]

        if len(trajectories) < 2:
            skipped += 1
            continue

        trajectories_sorted = sorted(trajectories, key=lambda t: t["reward"], reverse=True)
        
        best = trajectories_sorted[0]
        worst = trajectories_sorted[-1]

        if best["reward"] <= worst["reward"]:
            skipped += 1
            continue

        preference = {
            "question": question,
            "gold_answer": gold_answer,
            "chosen": best["text"],
            "rejected": worst["text"],
            "chosen_reward": best["reward"],
            "rejected_reward": worst["reward"],
        }
        preferences.append(preference)

    print(f"Created {len(preferences)} preference pairs (skipped {skipped} questions).")

    with open(PREFERENCES_PATH, "w", encoding="utf-8") as f:
        for pref in preferences:
            f.write(json.dumps(pref) + "\n")

    print(f"Saved preferences to {PREFERENCES_PATH}")


if __name__ == "__main__":
    main()
