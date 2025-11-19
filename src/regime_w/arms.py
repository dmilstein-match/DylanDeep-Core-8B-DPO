from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ArmSpec:
    name: str
    system_prompt: str
    temp: float
    top_p: float
    extra_cfg: Optional[Dict] = field(default_factory=dict)


def build_all_arms() -> List[ArmSpec]:
    """
    Build all Regime W arms:
    - 8 original variance arms (wolfram_0-3, maudlin_0-3)
    - 3 structured probe arms (wolfram_standard=A, wolfram_rephrase=A', maudlin_cf1)
    """
    arms: List[ArmSpec] = []

    # ===== Original 8 variance arms (keep for backward compat) =====
    wolfram_system = (
        "You are a precise math reasoner. "
        "Solve the problem step by step, keep reasoning linear and explicit. "
        "Avoid extra commentary. End with 'Answer: <number>'."
    )
    for i, t in enumerate([0.2, 0.3, 0.4, 0.5]):
        arms.append(ArmSpec(
            name=f"wolfram_{i}",
            system_prompt=wolfram_system,
            temp=t,
            top_p=0.95,
            extra_cfg={"seed": 42 + i},  # Reproducible variance
        ))

    maudlin_system = (
        "You are a reflective math reasoner. "
        "Explain your reasoning and highlight any steps that might be fragile. "
        "End with 'Answer: <number>'."
    )
    for i, t in enumerate([0.2, 0.3, 0.4, 0.5]):
        arms.append(ArmSpec(
            name=f"maudlin_{i}",
            system_prompt=maudlin_system,
            temp=t,
            top_p=0.9,
            extra_cfg={"seed": 100 + i},  # Reproducible variance
        ))

    # ===== Tier 2: Structured probe arms =====
    # A: wolfram_standard (canonical clear reasoning)
    arms.append(ArmSpec(
        name="wolfram_standard",
        system_prompt=(
            "You are a precise math reasoner. "
            "Solve the problem step by step with clear, linear reasoning. "
            "End with 'Answer: <number>'."
        ),
        temp=0.3,
        top_p=0.95,
        extra_cfg={"seed": 1000, "probe_type": "A"},
    ))

    # A': wolfram_rephrase (paraphrase stability check)
    arms.append(ArmSpec(
        name="wolfram_rephrase",
        system_prompt=(
            "You are a precise math reasoner. "
            "First, restate the problem in your own words to confirm understanding. "
            "Then solve step by step with clear reasoning. "
            "End with 'Answer: <number>'."
        ),
        temp=0.3,
        top_p=0.95,
        extra_cfg={"seed": 1001, "probe_type": "A_prime"},
    ))

    # CF1: maudlin_cf1 (counterfactual robustness probe)
    arms.append(ArmSpec(
        name="maudlin_cf1",
        system_prompt=(
            "You are a reflective math reasoner. "
            "Assume your first instinct might be wrong. Look for traps or edge cases. "
            "Then solve carefully, highlighting any fragile steps. "
            "End with 'Answer: <number>'."
        ),
        temp=0.5,
        top_p=0.9,
        extra_cfg={"seed": 2000, "probe_type": "CF1"},
    ))

    return arms
