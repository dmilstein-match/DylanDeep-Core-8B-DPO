from dataclasses import dataclass
from typing import List


@dataclass
class ArmSpec:
    name: str
    system_prompt: str
    temp: float
    top_p: float


def build_all_arms() -> List[ArmSpec]:
    arms: List[ArmSpec] = []

    wolfram_system = (
        "You are a precise math reasoner. "
        "Solve the problem step by step, keep reasoning linear and explicit. "
        "Avoid extra commentary. End with '#### final_answer'."
    )
    for i, t in enumerate([0.2, 0.3, 0.4, 0.5]):
        arms.append(ArmSpec(
            name=f"wolfram_{i}",
            system_prompt=wolfram_system,
            temp=t,
            top_p=0.95,
        ))

    maudlin_system = (
        "You are a reflective math reasoner. "
        "Explain your reasoning and highlight any steps that might be fragile. "
        "End with '#### final_answer'."
    )
    for i, t in enumerate([0.2, 0.3, 0.4, 0.5]):
        arms.append(ArmSpec(
            name=f"maudlin_{i}",
            system_prompt=maudlin_system,
            temp=t,
            top_p=0.9,
        ))

    return arms
