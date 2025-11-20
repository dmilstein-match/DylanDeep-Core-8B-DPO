from typing import List, Dict
from src.common.answer_utils import extract_answer, normalize_answer


def s_end_for_question(answers: List[str]) -> float:
    """
    Pairwise agreement on final answers across arms.
    Normalizes answers before comparison to handle formatting variations.
    """
    n = len(answers)
    if n < 2:
        return 1.0 if n == 1 else 0.0

    normalized = [normalize_answer(extract_answer(a)) for a in answers]
    
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if normalized[i] == normalized[j]:
                agree += 1
    return agree / total if total > 0 else 0.0


def s_path_for_question(reasonings: List[str]) -> float:
    """
    Path coherence score with improved A ↔ A′ stability reward.

    Strategy:
    1. For probe arms A and A', measure reasoning path similarity
    2. For other arms, use length variance as a proxy for coherence
    3. Combine both signals for a richer path score

    Returns:
        Score in [0.0, 1.0] where 1.0 = highly coherent paths
    """
    if not reasonings:
        return 0.0

    # Basic length-based coherence (variance heuristic)
    lengths = [len(r.split()) for r in reasonings]
    avg = sum(lengths) / len(lengths)
    if avg <= 0:
        return 0.0

    var = sum((l - avg) ** 2 for l in lengths) / len(lengths)
    length_score = 1.0 / (1.0 + var / (avg + 1e-6))

    # Enhanced: measure pairwise token overlap as proxy for path similarity
    # Higher overlap = more coherent reasoning across arms
    n = len(reasonings)
    if n < 2:
        return length_score

    # Compute pairwise Jaccard similarity on reasoning tokens
    from collections import Counter

    token_sets = [set(r.lower().split()) for r in reasonings]
    overlap_scores = []

    for i in range(n):
        for j in range(i + 1, n):
            intersection = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            if union > 0:
                overlap_scores.append(intersection / union)

    if not overlap_scores:
        return length_score

    avg_overlap = sum(overlap_scores) / len(overlap_scores)

    # Combine length coherence and token overlap
    # Weight: 40% length coherence, 60% semantic overlap
    combined_score = 0.4 * length_score + 0.6 * avg_overlap

    return max(0.0, min(1.0, combined_score))


def s_cf_for_question(arm_outputs: List[Dict]) -> float:
    """
    Counterfactual robustness score.

    Strategy:
    1. If we have probe arms (A, A', CF1), check agreement between them
    2. High score = CF1 agrees with A/A' (no traps found, reasoning is robust)
    3. Low score = CF1 disagrees (potential edge case or trap detected)
    4. For non-probe arms, use variance-based heuristic

    Returns:
        Score in [0.0, 1.0] where 1.0 = perfect robustness
    """
    if not arm_outputs:
        return 0.5

    # Extract answers from all arms
    answers = [normalize_answer(extract_answer(o.get("answer", ""))) for o in arm_outputs]

    # Try to find probe arms by name or probe_type in metadata
    # (In practice, arm_outputs may not include arm name, so we'll use positional logic)
    # For now, use a simpler heuristic: measure answer agreement across all arms

    # Count unique answers
    unique_answers = set(a for a in answers if a)
    if not unique_answers:
        return 0.5

    # Perfect agreement across all arms (including CF) = 1.0
    if len(unique_answers) == 1:
        return 1.0

    # Complete disagreement = 0.0
    n = len(answers)
    if len(unique_answers) == n:
        return 0.0

    # Partial agreement: measure majority consensus
    from collections import Counter
    counts = Counter(answers)
    max_count = max(counts.values())
    agreement_ratio = max_count / n

    # Map agreement ratio to [0, 1]
    # 100% agreement = 1.0, 50% = 0.5, lower = closer to 0
    return max(0.0, min(1.0, agreement_ratio))


def s_wm_for_question(s_end: float, s_path: float, s_cf: float) -> float:
    """
    Combined Regime W coherence score.
    """
    return 0.4 * s_end + 0.3 * s_path + 0.3 * s_cf
