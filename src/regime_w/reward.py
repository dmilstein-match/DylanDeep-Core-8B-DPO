from dataclasses import dataclass
from typing import List, Dict, Optional

from .scoring import (
    s_end_for_question,
    s_path_for_question,
    s_cf_for_question,
    s_wm_for_question,
)
from src.common.answer_utils import extract_answer, normalize_answer


ALPHA_CORRECT = 1.0
BETA_COHERENCE = 0.5
LENGTH_PENALTY_PER_TOKEN = 0.001
LENGTH_PENALTY_THRESHOLD = 512
FULL_AGREEMENT_BONUS = 0.2


@dataclass
class Trajectory:
    text: str
    reasoning: str
    answer: str
    num_tokens: int


def compute_regime_w_metrics(question: str, trajectories: List[Trajectory]) -> Dict:
    """
    Compute question-level coherence metrics from all trajectories.
    """
    answers = [t.answer for t in trajectories]
    reasonings = [t.reasoning for t in trajectories]
    
    arm_outputs = [
        {"answer": t.answer, "reasoning": t.reasoning, "full_text": t.text}
        for t in trajectories
    ]

    s_end = s_end_for_question(answers)
    s_path = s_path_for_question(reasonings)
    s_cf = s_cf_for_question(arm_outputs)
    s_wm = s_wm_for_question(s_end, s_path, s_cf)

    return {
        "s_end": s_end,
        "s_path": s_path,
        "s_cf": s_cf,
        "s_wm": s_wm,
    }


def compute_rewards_for_question(
    question: str,
    trajectories: List[Trajectory],
    gold_answer: str,
) -> List[float]:
    """
    Given one question, a list of trajectories from the policy,
    and the gold answer, return a reward per trajectory.
    
    This is the core Regime W -> scalar reward mapping.
    """
    gold = normalize_answer(extract_answer(gold_answer))
    correct_flags = []
    for t in trajectories:
        pred = normalize_answer(extract_answer(t.answer))
        correct_flags.append(1.0 if pred == gold and gold != "" else 0.0)

    metrics = compute_regime_w_metrics(question, trajectories)
    s_end = metrics.get("s_end", 0.0)
    s_path = metrics.get("s_path", 0.0)
    s_cf = metrics.get("s_cf", 0.0)
    s_wm = metrics.get("s_wm", 0.0)

    any_correct = any(c > 0.5 for c in correct_flags)
    all_correct_same_answer = False
    if any_correct:
        correct_answers = {
            normalize_answer(extract_answer(t.answer))
            for t, c in zip(trajectories, correct_flags)
            if c > 0.5
        }
        all_correct_same_answer = len(correct_answers) == 1

    rewards: List[float] = []
    for t, correct in zip(trajectories, correct_flags):
        base = ALPHA_CORRECT * correct

        coherence_term = BETA_COHERENCE * s_wm

        extra_tokens = max(0, t.num_tokens - LENGTH_PENALTY_THRESHOLD)
        length_penalty = LENGTH_PENALTY_PER_TOKEN * float(extra_tokens)

        agreement_bonus = 0.0
        if correct > 0.5 and all_correct_same_answer and s_end > 0.9:
            agreement_bonus = FULL_AGREEMENT_BONUS

        reward = base + coherence_term + agreement_bonus - length_penalty
        rewards.append(reward)

    return rewards


def compute_reward(
    question: str,
    arm_outputs: List[Dict],
    gold_answer: Optional[str] = None,
) -> float:
    """
    Legacy function for backward compatibility.
    Converts arm_outputs to Trajectory objects and returns single reward.
    """
    trajectories = [
        Trajectory(
            text=a.get("full_text", ""),
            reasoning=a.get("reasoning", ""),
            answer=a.get("answer", ""),
            num_tokens=len(a.get("full_text", "").split())
        )
        for a in arm_outputs
    ]
    
    if gold_answer is None:
        gold_answer = ""
    
    rewards = compute_rewards_for_question(question, trajectories, gold_answer)
    return sum(rewards) / len(rewards) if rewards else 0.0
