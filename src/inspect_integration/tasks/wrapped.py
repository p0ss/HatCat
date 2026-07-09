"""
Generic task wrapper for HatCat-Inspect integration.

Wraps any Inspect eval to add HatCat monitoring and steering.
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Union
import importlib

# Inspect imports
try:
    from inspect_ai import Task, task
    from inspect_ai.dataset import Dataset
    from inspect_ai.solver import Solver, chain, generate
    from inspect_ai.scorer import Scorer
    INSPECT_AVAILABLE = True
except ImportError:
    INSPECT_AVAILABLE = False
    Task = object
    Dataset = object
    Solver = object
    Scorer = object
    chain = lambda *args: args
    generate = lambda: None

    # Fallback task decorator
    def task(name_or_func=None, **kwargs):
        if callable(name_or_func):
            return name_or_func
        def decorator(func):
            return func
        return decorator

from ..config import Condition, HatCatConfig
from ..solvers import condition_chain, hatcat_condition, hatcat_setup
from ..scorers import hatcat_combined_scorer, hatcat_metrics_scorer


# Known eval locations for dynamic loading
KNOWN_EVALS: Dict[str, str] = {
    # UK Gov Inspect Evals
    "gsm8k": "inspect_evals.gsm8k",
    "truthfulqa": "inspect_evals.truthfulqa",
    "humaneval": "inspect_evals.humaneval",
    "mmlu": "inspect_evals.mmlu",
    "agentharm": "inspect_evals.agentharm",
    "gpqa": "inspect_evals.gpqa",
    "arc": "inspect_evals.arc",
    "hellaswag": "inspect_evals.hellaswag",
    "winogrande": "inspect_evals.winogrande",
    "math": "inspect_evals.math",
    "mbpp": "inspect_evals.mbpp",
    "drop": "inspect_evals.drop",
    "squad": "inspect_evals.squad",
    # Add more as needed
}


def _load_eval_task(eval_name: str, **kwargs) -> Task:
    """
    Dynamically load an eval task by name.

    Args:
        eval_name: Name of eval (e.g., "gsm8k") or module path
        **kwargs: Arguments to pass to task factory

    Returns:
        Loaded Task instance
    """
    # Get module path
    if eval_name in KNOWN_EVALS:
        module_path = KNOWN_EVALS[eval_name]
    else:
        module_path = eval_name

    # Try to import
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not load eval '{eval_name}'. "
            f"Ensure inspect-evals is installed: pip install inspect-evals"
        ) from e

    # Find task factory (usually named after the eval)
    task_name = eval_name.split(".")[-1]
    task_factory = None

    # Try common naming patterns
    for name in [task_name, f"{task_name}_task", "task", "eval_task"]:
        if hasattr(module, name):
            task_factory = getattr(module, name)
            break

    if task_factory is None:
        # Look for any @task decorated function
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and hasattr(obj, "__wrapped__"):
                task_factory = obj
                break

    if task_factory is None:
        raise ValueError(
            f"Could not find task factory in '{module_path}'. "
            f"Available: {[n for n in dir(module) if not n.startswith('_')]}"
        )

    # Call factory to get task
    return task_factory(**kwargs)


def wrap_existing_task(
    base_task: Task,
    condition: Union[str, Condition] = "C",
    lens_pack: str = "lens_packs/sumo-2k",
    steering_strength: float = 0.3,
    add_hatcat_metrics: bool = True,
) -> Task:
    """
    Wrap an existing Task with HatCat monitoring.

    Args:
        base_task: Task to wrap
        condition: Experimental condition (A-F)
        lens_pack: Lens pack path
        steering_strength: HUSH steering strength
        add_hatcat_metrics: Whether to add HatCat metrics to scorer

    Returns:
        Wrapped Task
    """
    if isinstance(condition, str):
        condition = Condition(condition)

    # Get condition solvers
    cond_solvers = condition_chain(
        condition=condition,
        lens_pack=lens_pack,
        steering_strength=steering_strength,
    )

    # Wrap scorer if requested
    wrapped_scorer = base_task.scorer
    if add_hatcat_metrics and wrapped_scorer is not None:
        if isinstance(wrapped_scorer, list):
            wrapped_scorer = [hatcat_combined_scorer(s) for s in wrapped_scorer]
        else:
            wrapped_scorer = hatcat_combined_scorer(wrapped_scorer)

    # Get base solvers
    base_solvers = base_task.solver
    if not isinstance(base_solvers, list):
        base_solvers = [base_solvers] if base_solvers else []

    # Combine solvers: condition setup + base solvers
    combined_solvers = cond_solvers + base_solvers

    return Task(
        dataset=base_task.dataset,
        solver=combined_solvers,
        scorer=wrapped_scorer,
        config=base_task.config,
        sandbox=base_task.sandbox,
        metadata={
            **(base_task.metadata or {}),
            "hatcat_condition": condition.value,
            "hatcat_lens_pack": lens_pack,
            "hatcat_wrapped": True,
        },
    )


@task
def hatcat_wrapped(
    base_eval: str = "gsm8k",
    condition: Literal["A", "B", "C", "D", "E", "F"] = "C",
    lens_pack: str = "lens_packs/sumo-2k",
    steering_strength: float = 0.3,
    induction_strength: Optional[float] = None,
    add_hatcat_metrics: bool = True,
    **base_eval_kwargs,
) -> Task:
    """
    Generic wrapper to run any Inspect eval with HatCat.

    This is the main entry point for running evals with HatCat monitoring.
    It dynamically loads the specified eval and wraps it with HatCat
    condition solvers and scorers.

    Args:
        base_eval: Name of eval to wrap (e.g., "gsm8k", "truthfulqa")
        condition: Experimental condition:
            A: Baseline (no monitoring/steering)
            B: HAT monitoring only
            C: Full safety (HAT + HUSH)
            D: Adversarial deception test
            E: Adversarial sycophancy test
            F: Adversarial manipulation test
        lens_pack: Path to lens pack
        steering_strength: HUSH steering strength
        induction_strength: Adversarial induction strength (D/E/F only)
        add_hatcat_metrics: Add HatCat metrics to scorer
        **base_eval_kwargs: Arguments for the base eval

    Returns:
        Wrapped Task

    Example:
        # Run GSM8K with full HUSH
        inspect eval hatcat_wrapped -T base_eval=gsm8k -T condition=C

        # Run TruthfulQA with monitoring only
        inspect eval hatcat_wrapped -T base_eval=truthfulqa -T condition=B

        # Adversarial deception test
        inspect eval hatcat_wrapped -T base_eval=agentharm -T condition=D
    """
    # Load base eval
    base_task = _load_eval_task(base_eval, **base_eval_kwargs)

    # Convert condition string to enum
    cond = Condition(condition)

    # Use induction_strength if provided, else fall back to steering_strength
    ind_str = induction_strength if induction_strength is not None else steering_strength

    # Get condition solvers
    cond_solvers = condition_chain(
        condition=cond,
        lens_pack=lens_pack,
        steering_strength=steering_strength,
        induction_strength=ind_str,
    )

    # Wrap scorer
    wrapped_scorer = base_task.scorer
    if add_hatcat_metrics and wrapped_scorer is not None:
        if isinstance(wrapped_scorer, list):
            wrapped_scorer = [hatcat_combined_scorer(s) for s in wrapped_scorer]
        else:
            wrapped_scorer = hatcat_combined_scorer(wrapped_scorer)

    # Get base solvers
    base_solvers = base_task.solver
    if not isinstance(base_solvers, list):
        base_solvers = [base_solvers] if base_solvers else []

    # Combine: condition setup + base solvers
    combined_solvers = cond_solvers + base_solvers

    return Task(
        dataset=base_task.dataset,
        solver=combined_solvers,
        scorer=wrapped_scorer,
        config=base_task.config,
        sandbox=base_task.sandbox,
        metadata={
            **(base_task.metadata or {}),
            "hatcat_condition": cond.value,
            "hatcat_lens_pack": lens_pack,
            "hatcat_base_eval": base_eval,
            "hatcat_wrapped": True,
        },
    )


def create_hatcat_task(
    dataset: Dataset,
    solver: Optional[Union[Solver, List[Solver]]] = None,
    scorer: Optional[Union[Scorer, List[Scorer]]] = None,
    condition: Union[str, Condition] = "C",
    lens_pack: str = "lens_packs/sumo-2k",
    steering_strength: float = 0.3,
    **task_kwargs,
) -> Task:
    """
    Create a new Task with HatCat integration from scratch.

    Use this when you want to define a custom eval with HatCat support.

    Args:
        dataset: Inspect Dataset
        solver: Task solver(s)
        scorer: Task scorer(s)
        condition: Experimental condition
        lens_pack: Lens pack path
        steering_strength: HUSH steering strength
        **task_kwargs: Additional Task arguments

    Returns:
        Task with HatCat integration

    Example:
        from inspect_ai.dataset import json_dataset
        from inspect_ai.solver import generate
        from inspect_ai.scorer import accuracy

        task = create_hatcat_task(
            dataset=json_dataset("my_eval.json"),
            solver=generate(),
            scorer=accuracy(),
            condition="C",
        )
    """
    if isinstance(condition, str):
        condition = Condition(condition)

    # Get condition solvers
    cond_solvers = condition_chain(
        condition=condition,
        lens_pack=lens_pack,
        steering_strength=steering_strength,
    )

    # Wrap solver
    if solver is None:
        solver = [generate()]
    elif not isinstance(solver, list):
        solver = [solver]

    combined_solvers = cond_solvers + solver

    # Wrap scorer with HatCat metrics
    if scorer is not None:
        if isinstance(scorer, list):
            scorer = [hatcat_combined_scorer(s) for s in scorer]
        else:
            scorer = hatcat_combined_scorer(scorer)

    return Task(
        dataset=dataset,
        solver=combined_solvers,
        scorer=scorer,
        metadata={
            "hatcat_condition": condition.value,
            "hatcat_lens_pack": lens_pack,
            "hatcat_wrapped": True,
        },
        **task_kwargs,
    )
