"""
Condition chain solvers for HatCat-Inspect integration.

These solvers configure the experimental condition before generation,
storing metadata and applying appropriate setup.
"""

from typing import Any, Dict, List, Literal, Optional, Union

# Inspect imports
try:
    from inspect_ai.solver import (
        Solver,
        solver,
        TaskState,
        Generate,
        chain,
    )
    INSPECT_AVAILABLE = True
except ImportError:
    INSPECT_AVAILABLE = False
    Solver = object
    TaskState = object
    Generate = object
    chain = lambda *args: args

    # Fallback solver decorator that handles both @solver and @solver(name=...)
    def solver(name_or_func=None):
        if callable(name_or_func):
            # Used as @solver without arguments
            return name_or_func
        # Used as @solver(name=...) with arguments
        def decorator(func):
            return func
        return decorator

from ..config import Condition, HatCatConfig, INDUCTION_CONCEPTS


@solver
def hatcat_condition(
    condition: Union[str, Condition] = "A",
) -> Solver:
    """
    Solver that configures the HatCat experimental condition.

    Stores condition in task metadata for the model provider to use.

    Args:
        condition: Experimental condition (A-F)
            A: Baseline (no monitoring/steering)
            B: HAT monitoring only
            C: Full safety (HAT + HUSH)
            D: Adversarial deception test
            E: Adversarial sycophancy test
            F: Adversarial manipulation test

    Returns:
        Solver that sets up the condition
    """
    if isinstance(condition, str):
        condition = Condition(condition)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Store condition in metadata
        if not hasattr(state, 'metadata') or state.metadata is None:
            state.metadata = {}

        state.metadata["hatcat_condition"] = condition.value
        state.metadata["hatcat_monitoring"] = condition != Condition.A
        state.metadata["hatcat_steering"] = condition in (
            Condition.C, Condition.D, Condition.E, Condition.F
        )
        state.metadata["hatcat_induction"] = condition in (
            Condition.D, Condition.E, Condition.F
        )

        if condition in INDUCTION_CONCEPTS:
            state.metadata["hatcat_induction_concept"] = INDUCTION_CONCEPTS[condition]

        return state

    return solve


@solver
def hatcat_setup(
    lens_pack: str = "lens_packs/sumo-2k",
    steering_strength: float = 0.3,
    induction_strength: Optional[float] = None,
    system_prompt: Optional[str] = None,
) -> Solver:
    """
    Solver that configures HatCat parameters.

    Args:
        lens_pack: Path to lens pack for monitoring
        steering_strength: HUSH steering strength
        induction_strength: Adversarial induction strength (for D/E/F)
        system_prompt: Optional system prompt override

    Returns:
        Solver that stores configuration
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not hasattr(state, 'metadata') or state.metadata is None:
            state.metadata = {}

        state.metadata["hatcat_lens_pack"] = lens_pack
        state.metadata["hatcat_steering_strength"] = steering_strength

        if induction_strength is not None:
            state.metadata["hatcat_induction_strength"] = induction_strength

        if system_prompt is not None:
            # Prepend system prompt to messages
            from inspect_ai.model import ChatMessageSystem
            if hasattr(state, 'messages') and state.messages:
                # Check if first message is already system
                if state.messages[0].role != "system":
                    state.messages.insert(0, ChatMessageSystem(content=system_prompt))
                else:
                    # Replace existing system message
                    state.messages[0] = ChatMessageSystem(content=system_prompt)

        return state

    return solve


def condition_chain(
    condition: Union[str, Condition],
    lens_pack: str = "lens_packs/sumo-2k",
    steering_strength: float = 0.3,
    induction_strength: Optional[float] = None,
    adversarial_prompt: Optional[str] = None,
) -> List[Solver]:
    """
    Build a solver chain for a specific condition.

    Args:
        condition: Experimental condition (A-F)
        lens_pack: Lens pack path
        steering_strength: HUSH steering strength
        induction_strength: Induction strength for D/E/F
        adversarial_prompt: Optional adversarial system prompt for D/E/F

    Returns:
        List of solvers to prepend to task
    """
    if isinstance(condition, str):
        condition = Condition(condition)

    solvers = []

    # Add condition setup
    solvers.append(hatcat_condition(condition))

    # Add parameter setup
    solvers.append(hatcat_setup(
        lens_pack=lens_pack,
        steering_strength=steering_strength,
        induction_strength=induction_strength,
    ))

    # For adversarial conditions, add induction solver
    if condition in (Condition.D, Condition.E, Condition.F):
        from .induction import hatcat_induction, InductionType

        induction_map = {
            Condition.D: InductionType.DECEPTION,
            Condition.E: InductionType.SYCOPHANCY,
            Condition.F: InductionType.MANIPULATION,
        }

        solvers.append(hatcat_induction(
            induction_type=induction_map[condition],
            strength=induction_strength or steering_strength,
            adversarial_prompt=adversarial_prompt,
        ))

    return solvers


# Pre-built condition chains
CONDITION_CHAINS: Dict[str, List[Solver]] = {}

if INSPECT_AVAILABLE:
    CONDITION_CHAINS = {
        "A": condition_chain("A"),
        "B": condition_chain("B"),
        "C": condition_chain("C"),
        "D": condition_chain("D"),
        "E": condition_chain("E"),
        "F": condition_chain("F"),
    }
