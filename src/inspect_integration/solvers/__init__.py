"""
HatCat condition solvers for Inspect integration.

Exports:
    hatcat_condition: Solver to configure experimental condition
    hatcat_induction: Solver for adversarial induction (D/E/F)
    condition_chain: Pre-built solver chains for each condition
"""

from .condition_chain import (
    hatcat_condition,
    hatcat_setup,
    condition_chain,
    CONDITION_CHAINS,
)
from .induction import (
    hatcat_induction,
    InductionType,
    INDUCTION_PROMPTS,
)

__all__ = [
    "hatcat_condition",
    "hatcat_setup",
    "condition_chain",
    "CONDITION_CHAINS",
    "hatcat_induction",
    "InductionType",
    "INDUCTION_PROMPTS",
]
