"""
HatCat-Inspect Integration

Standards-compliant integration that allows any UK Gov Inspect eval to run
with HatCat monitoring and steering as a composable layer.

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Inspect CLI / Python API                     │
    │                 inspect eval hatcat_wrapped                      │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
    ┌──────────────────────────────▼──────────────────────────────────┐
    │                    HatCat Integration Layer                      │
    │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
    │  │ HatCatModel │  │ Condition    │  │ hatcat_metrics_scorer   │ │
    │  │  (Provider) │  │  Solvers     │  │ + combined_scorer       │ │
    │  └──────┬──────┘  └──────────────┘  └─────────────────────────┘ │
    └─────────┼───────────────────────────────────────────────────────┘
              │
    ┌─────────▼───────────────────────────────────────────────────────┐
    │                      HatCat FTW Stack                            │
    │  HAT (monitoring) │ HUSH (steering) │ ASK (audit) │ Lens Pool   │
    └─────────────────────────────────────────────────────────────────┘

Condition Matrix:
    A: Baseline (no monitoring/steering)
    B: HAT monitoring only
    C: Full safety (HAT + HUSH)
    D: Adversarial deception test
    E: Adversarial sycophancy test
    F: Adversarial manipulation test

Usage:
    # CLI
    inspect eval hatcat_wrapped -T base_eval=gsm8k -T condition=C

    # Python
    from src.inspect_integration import hatcat_wrapped
    task = hatcat_wrapped(base_eval="gsm8k", condition="C")

Exports:
    Config:
        Condition, HatCatConfig, HatCatMetrics

    Model:
        HatCatModel, get_hatcat_model, get_lens_pool

    Solvers:
        hatcat_condition, hatcat_setup, condition_chain
        hatcat_induction, InductionType

    Scorers:
        hatcat_metrics_scorer, hatcat_combined_scorer

    Tasks:
        hatcat_wrapped, create_hatcat_task, wrap_existing_task
        SAFETY_EVALS, KNOWLEDGE_EVALS, REASONING_EVALS
"""

__version__ = "0.1.0"

# Config exports
from .config import (
    Condition,
    HatCatConfig,
    HatCatMetrics,
    LensConfig,
    SteeringConfig,
    InductionConfig,
    INDUCTION_CONCEPTS,
)

# Model exports
from .model import (
    HatCatModel,
    get_hatcat_model,
    get_lens_pool,
    get_hush_pool,
    LensPool,
)

# Solver exports
from .solvers import (
    hatcat_condition,
    hatcat_setup,
    condition_chain,
    CONDITION_CHAINS,
    hatcat_induction,
    InductionType,
    INDUCTION_PROMPTS,
)

# Scorer exports
from .scorers import (
    hatcat_metrics_scorer,
    extract_hatcat_metrics,
    hatcat_combined_scorer,
    with_hatcat_metrics,
)

# Task exports
from .tasks import (
    hatcat_wrapped,
    create_hatcat_task,
    wrap_existing_task,
    SAFETY_EVALS,
    KNOWLEDGE_EVALS,
    REASONING_EVALS,
    get_preset_bundle,
)

# Register tasks on import
from . import _registry

__all__ = [
    # Version
    "__version__",
    # Config
    "Condition",
    "HatCatConfig",
    "HatCatMetrics",
    "LensConfig",
    "SteeringConfig",
    "InductionConfig",
    "INDUCTION_CONCEPTS",
    # Model
    "HatCatModel",
    "get_hatcat_model",
    "get_lens_pool",
    "get_hush_pool",
    "LensPool",
    # Solvers
    "hatcat_condition",
    "hatcat_setup",
    "condition_chain",
    "CONDITION_CHAINS",
    "hatcat_induction",
    "InductionType",
    "INDUCTION_PROMPTS",
    # Scorers
    "hatcat_metrics_scorer",
    "extract_hatcat_metrics",
    "hatcat_combined_scorer",
    "with_hatcat_metrics",
    # Tasks
    "hatcat_wrapped",
    "create_hatcat_task",
    "wrap_existing_task",
    "SAFETY_EVALS",
    "KNOWLEDGE_EVALS",
    "REASONING_EVALS",
    "get_preset_bundle",
]
