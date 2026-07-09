"""
Preset eval bundles for common HatCat evaluation scenarios.

These presets group related evals for specific testing purposes.
"""

from typing import Dict, List, Optional, Union

from ..config import Condition


# Safety-focused evals - test HUSH's ability to detect/prevent harmful outputs
SAFETY_EVALS: Dict[str, Dict] = {
    "agentharm": {
        "description": "Tests harmful agent behaviors",
        "recommended_conditions": ["B", "C", "D"],
    },
    "truthfulqa": {
        "description": "Tests truthfulness vs deceptive answers",
        "recommended_conditions": ["B", "C", "D"],
    },
    "toxigen": {
        "description": "Tests toxic content generation",
        "recommended_conditions": ["B", "C"],
    },
}

# Knowledge/factuality evals - test if HUSH affects model knowledge
KNOWLEDGE_EVALS: Dict[str, Dict] = {
    "mmlu": {
        "description": "Massive Multitask Language Understanding",
        "recommended_conditions": ["A", "B", "C"],
    },
    "gpqa": {
        "description": "Graduate-level QA",
        "recommended_conditions": ["A", "B", "C"],
    },
    "arc": {
        "description": "AI2 Reasoning Challenge",
        "recommended_conditions": ["A", "B", "C"],
    },
    "hellaswag": {
        "description": "Commonsense reasoning",
        "recommended_conditions": ["A", "B", "C"],
    },
}

# Reasoning/coding evals - test if HUSH affects reasoning capabilities
REASONING_EVALS: Dict[str, Dict] = {
    "gsm8k": {
        "description": "Grade school math",
        "recommended_conditions": ["A", "B", "C"],
    },
    "math": {
        "description": "Competition mathematics",
        "recommended_conditions": ["A", "B", "C"],
    },
    "humaneval": {
        "description": "Python code generation",
        "recommended_conditions": ["A", "B", "C"],
    },
    "mbpp": {
        "description": "Mostly Basic Python Problems",
        "recommended_conditions": ["A", "B", "C"],
    },
}


# All preset bundles
PRESET_BUNDLES: Dict[str, Dict[str, Dict]] = {
    "safety": SAFETY_EVALS,
    "knowledge": KNOWLEDGE_EVALS,
    "reasoning": REASONING_EVALS,
}


def get_preset_bundle(
    bundle_name: str,
    conditions: Optional[List[Union[str, Condition]]] = None,
) -> List[Dict]:
    """
    Get a preset bundle of evals with recommended conditions.

    Args:
        bundle_name: Name of preset bundle ("safety", "knowledge", "reasoning")
        conditions: Override conditions to use (defaults to recommended)

    Returns:
        List of eval configs to run

    Example:
        # Get safety evals with default conditions
        configs = get_preset_bundle("safety")

        # Run with custom conditions
        configs = get_preset_bundle("safety", conditions=["B", "C"])
    """
    if bundle_name not in PRESET_BUNDLES:
        raise ValueError(
            f"Unknown bundle: {bundle_name}. "
            f"Available: {list(PRESET_BUNDLES.keys())}"
        )

    bundle = PRESET_BUNDLES[bundle_name]
    configs = []

    for eval_name, eval_info in bundle.items():
        eval_conditions = conditions or eval_info.get("recommended_conditions", ["C"])

        for cond in eval_conditions:
            configs.append({
                "eval_name": eval_name,
                "condition": cond if isinstance(cond, str) else cond.value,
                "description": eval_info.get("description", ""),
            })

    return configs


def get_comparison_matrix(
    evals: List[str],
    conditions: List[Union[str, Condition]] = ["A", "B", "C"],
) -> List[Dict]:
    """
    Generate a comparison matrix of evals x conditions.

    Useful for systematic A/B/C testing across multiple evals.

    Args:
        evals: List of eval names
        conditions: List of conditions to test

    Returns:
        List of eval configs for full matrix

    Example:
        # Compare baseline vs monitored vs steered
        matrix = get_comparison_matrix(
            evals=["gsm8k", "truthfulqa"],
            conditions=["A", "B", "C"]
        )
        # Returns 6 configs: gsm8k-A, gsm8k-B, gsm8k-C, truthfulqa-A, ...
    """
    configs = []

    for eval_name in evals:
        for cond in conditions:
            cond_str = cond if isinstance(cond, str) else cond.value
            configs.append({
                "eval_name": eval_name,
                "condition": cond_str,
                "run_id": f"{eval_name}_{cond_str}",
            })

    return configs


def get_adversarial_battery() -> List[Dict]:
    """
    Get the full adversarial testing battery (conditions D/E/F).

    Tests HUSH's ability to counter adversarial induction on safety-critical evals.

    Returns:
        List of adversarial eval configs
    """
    adversarial_evals = ["truthfulqa", "agentharm"]
    adversarial_conditions = ["D", "E", "F"]

    configs = []
    for eval_name in adversarial_evals:
        for cond in adversarial_conditions:
            configs.append({
                "eval_name": eval_name,
                "condition": cond,
                "run_id": f"{eval_name}_adversarial_{cond}",
                "description": f"Test HUSH countering {_condition_description(cond)}",
            })

    return configs


def _condition_description(cond: str) -> str:
    """Get human-readable condition description."""
    descriptions = {
        "A": "baseline (no intervention)",
        "B": "monitoring only",
        "C": "full safety stack",
        "D": "induced deception",
        "E": "induced sycophancy",
        "F": "induced manipulation",
    }
    return descriptions.get(cond, cond)
