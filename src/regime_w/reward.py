from typing import List, Dict, Optional
import re

from .scoring import (
    s_end_for_question,
    s_path_for_question,
    s_cf_for_question,
    s_wm_for_question,
)


def extract_answer(text: str) -> str:
    """
    Prefer the number after '####'; otherwise last number in text.
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


def compute_regime_w_scores(
    question: str,
    arm_outputs: List[Dict],
    gold_answer: Optional[str] = None,
) -> Dict:
    """
    arm_outputs: list of dicts with keys:
      - "answer"
      - "reasoning"
      - "full_text"
      - "arm_name" (optional, for logging)
    """
    answers = [a["answer"] for a in arm_outputs]
    reasonings = [a["reasoning"] for a in arm_outputs]

    s_end = s_end_for_question(answers)
    s_path = s_path_for_question(reasonings)
    s_cf = s_cf_for_question(arm_outputs)
    s_wm = s_wm_for_question(s_end, s_path, s_cf)

    correct_flags: List[bool] = []
    if gold_answer is not None:
        gold = extract_answer(gold_answer)
        for ans in answers:
            correct_flags.append(extract_answer(ans) == gold)

    return {
        "s_end": s_end,
        "s_path": s_path,
        "s_cf": s_cf,
        "s_wm": s_wm,
        "correct_flags": correct_flags,
    }


def compute_reward(
    question: str,
    arm_outputs: List[Dict],
    gold_answer: Optional[str] = None,
) -> float:
    """
    Regime W scalar reward:
      - base: correctness across arms
      - + coherence bonus
      - + mild length penalty
    """
    scores = compute_regime_w_scores(question, arm_outputs, gold_answer)
    s_wm = scores["s_wm"]
    correct_flags = scores["correct_flags"]

    correct_any = any(correct_flags) if correct_flags else False

    base = 1.0 if correct_any else -0.5
    coherence_bonus = 0.7 * s_wm

    avg_len = sum(
        len(a["full_text"].split()) for a in arm_outputs
    ) / max(1, len(arm_outputs))
    len_penalty = -0.001 * max(0, avg_len - 400)

    reward = base + coherence_bonus + len_penalty
    return reward
