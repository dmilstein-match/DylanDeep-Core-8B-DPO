import json
from typing import List, Dict


ROLLOUTS_PATH = "data/abel_regime_w_rollouts.jsonl"
PREFERENCES_PATH = "data/abel_coherence_preferences.jsonl"


def load_rollouts(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    print("=" * 80)
    print("Abel Coherence Preference Pair Builder")
    print("=" * 80)
    
    print(f"\nLoading rollouts from {ROLLOUTS_PATH}...")
    rollouts = load_rollouts(ROLLOUTS_PATH)
    print(f"Loaded {len(rollouts)} rollout records\n")
    
    preferences = []
    skipped = 0

    print("Building preference pairs with correctness-first strategy:")
    print("  1. Correct vs incorrect trajectories")
    print("  2. High-coherence vs low-coherence among correct trajectories\n")

    for rec in rollouts:
        trajs = rec["trajectories"]
        if len(trajs) < 2:
            skipped += 1
            continue

        # Trajectories already have correct flag from rollout collection
        # (computed by comparing normalized predicted answer to normalized gold answer)

        # Strategy 1: Correctness pairs (correct vs wrong)
        # These pairs teach the model to get the answer right
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                a, b = trajs[i], trajs[j]
                if a["correct"] and not b["correct"]:
                    preferences.append({
                        "question": rec["question"],
                        "gold_answer": rec["gold_answer"],
                        "chosen": a["full_text"],
                        "rejected": b["full_text"],
                        "chosen_reward": a["reward"],
                        "rejected_reward": b["reward"],
                        "pair_type": "correctness",
                    })
                elif b["correct"] and not a["correct"]:
                    preferences.append({
                        "question": rec["question"],
                        "gold_answer": rec["gold_answer"],
                        "chosen": b["full_text"],
                        "rejected": a["full_text"],
                        "chosen_reward": b["reward"],
                        "rejected_reward": a["reward"],
                        "pair_type": "correctness",
                    })

        # Strategy 2: Coherence pairs among correct trajectories
        # These pairs teach the model to explain clearly
        correct_trajs = [t for t in trajs if t["correct"]]
        if len(correct_trajs) >= 2:
            # Sort by reward (which includes coherence component)
            correct_trajs = sorted(correct_trajs, key=lambda t: t["reward"], reverse=True)
            best = correct_trajs[0]
            worst = correct_trajs[-1]
            
            # Only create pair if there's meaningful difference
            if best["reward"] > worst["reward"] + 0.1:
                preferences.append({
                    "question": rec["question"],
                    "gold_answer": rec["gold_answer"],
                    "chosen": best["full_text"],
                    "rejected": worst["full_text"],
                    "chosen_reward": best["reward"],
                    "rejected_reward": worst["reward"],
                    "pair_type": "coherence",
                })

    correctness_pairs = sum(1 for p in preferences if p["pair_type"] == "correctness")
    coherence_pairs = sum(1 for p in preferences if p["pair_type"] == "coherence")

    print(f"Created {len(preferences)} total preference pairs:")
    print(f"  - {correctness_pairs} correctness pairs (correct vs incorrect)")
    print(f"  - {coherence_pairs} coherence pairs (high vs low coherence among correct)")
    print(f"  - Skipped {skipped} rollouts (insufficient trajectories)\n")

    print(f"Saving preferences to {PREFERENCES_PATH}...")
    with open(PREFERENCES_PATH, "w", encoding="utf-8") as f:
        for pref in preferences:
            f.write(json.dumps(pref) + "\n")

    print(f"\n{'=' * 80}")
    print(f"Preference pair building complete!")
    print(f"Saved to: {PREFERENCES_PATH}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
