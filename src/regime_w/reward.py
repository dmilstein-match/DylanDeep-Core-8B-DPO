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
GAMMA_TRAJECTORY_QUALITY = 0.3  # NEW: Trajectory-specific quality bonus
LENGTH_PENALTY_PER_TOKEN = 0.001
LENGTH_PENALTY_THRESHOLD = 512
FULL_AGREEMENT_BONUS = 0.2


@dataclass
class Trajectory:
    text: str
    reasoning: str
    answer: str
    num_tokens: int
    arm_name: Optional[str] = None  # NEW: Track which arm generated this


def compute_trajectory_quality_score_optimized(
    idx: int,
    correct: float,
    trajectory: Trajectory,
    min_len: int,
    max_len: int,
    canonical_probe_indices: List[int],
    trajectory_token_sets: List[set],
) -> float:
    """
    Optimized trajectory-specific quality score (uses pre-computed values).

    Among correct trajectories, this measures:
    1. Conciseness (shorter is better - Occam's razor)
    2. Alignment with canonical probes (A, A')

    Returns: Score in [0.0, 1.0]
    """
    # Only differentiate among correct trajectories
    if correct < 0.5:
        return 0.0

    # Component 1: Conciseness score (shorter is better)
    # Uses pre-computed min/max lengths
    if max_len == min_len:
        conciseness = 0.5
    else:
        # Invert: shorter gets higher score
        conciseness = 1.0 - (trajectory.num_tokens - min_len) / (max_len - min_len)

    # Component 2: Alignment with canonical probes (if available)
    if not canonical_probe_indices:
        # No probes available, use conciseness only
        return conciseness

    # Compute token overlap (Jaccard similarity) with canonical probes
    trajectory_tokens = trajectory_token_sets[idx]

    if not trajectory_tokens:
        return conciseness * 0.5

    probe_overlaps = []
    for probe_idx in canonical_probe_indices:
        probe_tokens = trajectory_token_sets[probe_idx]
        if probe_tokens:
            intersection = len(trajectory_tokens & probe_tokens)
            union = len(trajectory_tokens | probe_tokens)
            if union > 0:
                probe_overlaps.append(intersection / union)

    if not probe_overlaps:
        probe_alignment = 0.5
    else:
        probe_alignment = max(probe_overlaps)  # Best alignment with any probe

    # Combine: 40% conciseness + 60% probe alignment
    quality_score = 0.4 * conciseness + 0.6 * probe_alignment

    return max(0.0, min(1.0, quality_score))


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

    # Pre-compute token sets for efficiency (avoid redundant tokenization)
    trajectory_token_sets = [
        set(t.reasoning.lower().split()) for t in trajectories
    ]

    # Pre-identify correct trajectories and canonical probes
    correct_indices = [
        i for i, c in enumerate(correct_flags) if c > 0.5
    ]

    # Pre-compute min/max lengths for conciseness scoring
    if len(correct_indices) >= 2:
        correct_lengths = [trajectories[i].num_tokens for i in correct_indices]
        min_len = min(correct_lengths)
        max_len = max(correct_lengths)
    else:
        min_len = max_len = 0  # Not used when <2 correct trajectories

    # Find canonical probe indices once
    canonical_probe_indices = [
        i for i, t in enumerate(trajectories)
        if t.arm_name in ["wolfram_standard", "wolfram_rephrase"]
        and correct_flags[i] > 0.5
    ]

    rewards: List[float] = []
    for idx, (t, correct) in enumerate(zip(trajectories, correct_flags)):
        base = ALPHA_CORRECT * correct

        # Question-level coherence (same for all trajectories)
        coherence_term = BETA_COHERENCE * s_wm

        # NEW: Trajectory-specific quality score (creates differentiation)
        trajectory_quality = compute_trajectory_quality_score_optimized(
            idx, correct, t, min_len, max_len,
            canonical_probe_indices, trajectory_token_sets
        )
        quality_bonus = GAMMA_TRAJECTORY_QUALITY * trajectory_quality

        extra_tokens = max(0, t.num_tokens - LENGTH_PENALTY_THRESHOLD)
        length_penalty = LENGTH_PENALTY_PER_TOKEN * float(extra_tokens)

        agreement_bonus = 0.0
        if correct > 0.5 and all_correct_same_answer and s_end > 0.9:
            agreement_bonus = FULL_AGREEMENT_BONUS

        reward = base + coherence_term + quality_bonus + agreement_bonus - length_penalty
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
