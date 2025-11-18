from typing import List, Dict
import re


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


def normalize_answer(ans: str) -> str:
    """
    Simple normalization for answer comparison.
    """
    return ans.strip().lower()


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
    Coarse path coherence: similar lengths => higher score.
    """
    if not reasonings:
        return 0.0

    lengths = [len(r.split()) for r in reasonings]
    avg = sum(lengths) / len(lengths)
    if avg <= 0:
        return 0.0

    var = sum((l - avg) ** 2 for l in lengths) / len(lengths)
    score = 1.0 / (1.0 + var / (avg + 1e-6))
    return max(0.0, min(1.0, score))


def s_cf_for_question(arm_outputs: List[Dict]) -> float:
    """
    Placeholder counterfactual/cross-check score.
    You can later plug in your real CHSH-ish logic here.
    """
    return 0.5


def s_wm_for_question(s_end: float, s_path: float, s_cf: float) -> float:
    """
    Combined Regime W coherence score.
    """
    return 0.4 * s_end + 0.3 * s_path + 0.3 * s_cf
