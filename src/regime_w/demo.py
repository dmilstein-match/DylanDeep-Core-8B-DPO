from .reward import compute_reward


def main():
    question = "If you have 3 apples and buy 4 more, how many apples?"
    gold = "#### 7"
    arm_outputs = [
        {
            "arm_name": "wolfram_0",
            "full_text": "3 + 4 = 7\n#### 7",
            "reasoning": "3 + 4 = 7",
            "answer": "#### 7",
        },
        {
            "arm_name": "maudlin_0",
            "full_text": "Start with 3 apples, add 4, total 7.\n#### 7",
            "reasoning": "Start with 3 apples, add 4",
            "answer": "#### 7",
        },
    ]

    r = compute_reward(question, arm_outputs, gold)
    print("Regime W demo reward:", r)


if __name__ == "__main__":
    main()
