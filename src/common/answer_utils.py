"""
Centralized answer extraction and normalization utilities.

Used across:
- Regime W reward computation
- Rollout collection
- Preference building
- Evaluation scripts

This ensures consistent correctness detection across the entire pipeline.
"""

import re


def extract_answer(text: str) -> str:
    """
    Extract the final numeric answer from text.

    Strategy:
    1. Prefer the number after 'Answer:' marker if present (new format)
    2. Fall back to '####' marker for backward compatibility
    3. Otherwise, take the last number in the text
    4. If no numbers found, return the text as-is

    Args:
        text: Model output or gold answer string

    Returns:
        Extracted answer string (may still need normalization)
    """
    # Try new "Answer:" format first
    if "Answer:" in text:
        tail = text.split("Answer:")[-1]
        m = re.search(r"-?[\d,]+\.?\d*", tail)
        if m:
            return m.group(0).strip()

    # Backward compatibility: try "####" format
    if "####" in text:
        tail = text.split("####")[-1]
        m = re.search(r"-?[\d,]+\.?\d*", tail)
        if m:
            return m.group(0).strip()

    # Fallback: last number in text
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    if nums:
        return nums[-1].strip()

    return text.strip()


def normalize_answer(ans: str) -> str:
    """
    Normalize answer for comparison.
    
    Normalization steps:
    1. Strip whitespace
    2. Lowercase
    3. Remove commas from numbers (1,000 -> 1000)
    4. Convert whole-number decimals (42.0 -> 42)
    
    This handles common formatting variations while preserving
    numeric precision where needed.
    
    Args:
        ans: Extracted answer string
        
    Returns:
        Normalized answer for equality comparison
    """
    ans = ans.strip().lower()
    ans = ans.replace(",", "")

    # Convert "42.0" to "42" when it's a pure whole number
    try:
        if "." in ans:
            num = float(ans)
            if num == int(num):
                ans = str(int(num))
    except ValueError:
        pass

    return ans
