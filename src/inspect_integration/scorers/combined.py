"""
Combined scorer for HatCat-Inspect integration.

Wraps any existing scorer to add HatCat metrics to the score metadata.
"""

from typing import Any, Callable, List, Optional, Union

# Inspect imports
try:
    from inspect_ai.scorer import (
        Scorer,
        Score,
        scorer,
        Target,
    )
    from inspect_ai.solver import TaskState
    INSPECT_AVAILABLE = True
except ImportError:
    INSPECT_AVAILABLE = False
    Scorer = object
    Score = dict
    scorer = lambda **kw: lambda f: f
    Target = object
    TaskState = object

from .hatcat_metrics import extract_hatcat_metrics


def hatcat_combined_scorer(base_scorer: Scorer) -> Scorer:
    """
    Wrap a scorer to add HatCat metrics to score metadata.

    The base scorer's value and answer are preserved. HatCat metrics
    are added to the metadata under "hatcat_metrics".

    This allows combining task-specific scoring (e.g., accuracy)
    with HatCat safety monitoring.

    Args:
        base_scorer: The scorer to wrap

    Returns:
        Combined scorer

    Example:
        from inspect_ai.scorer import accuracy
        scorer = hatcat_combined_scorer(accuracy())
    """
    @scorer(metrics=getattr(base_scorer, 'metrics', []))
    def combined() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            # Run base scorer
            base_score = await base_scorer(state, target)

            # Extract HatCat metrics
            hatcat_metrics = extract_hatcat_metrics(state)

            # Merge metadata
            metadata = base_score.metadata.copy() if base_score.metadata else {}

            if hatcat_metrics is not None:
                metadata["hatcat_available"] = True
                metadata["hatcat_metrics"] = hatcat_metrics.to_dict()
            else:
                metadata["hatcat_available"] = False

            # Return score with merged metadata
            return Score(
                value=base_score.value,
                answer=base_score.answer,
                explanation=base_score.explanation,
                metadata=metadata,
            )

        return score

    return combined()


def with_hatcat_metrics(scorer_fn: Callable[[], Scorer]) -> Callable[[], Scorer]:
    """
    Decorator to add HatCat metrics to a scorer factory.

    Use this to wrap scorer factory functions.

    Args:
        scorer_fn: Scorer factory function

    Returns:
        Wrapped factory that adds HatCat metrics

    Example:
        @with_hatcat_metrics
        def my_scorer():
            return ...
    """
    def wrapped(*args, **kwargs) -> Scorer:
        base = scorer_fn(*args, **kwargs)
        return hatcat_combined_scorer(base)

    return wrapped


class MultiScorer:
    """
    Run multiple scorers and combine results.

    Useful for running both task-specific and HatCat scorers together.
    """

    def __init__(self, scorers: List[Scorer], primary: int = 0):
        """
        Initialize multi-scorer.

        Args:
            scorers: List of scorers to run
            primary: Index of primary scorer (determines final value)
        """
        self.scorers = scorers
        self.primary = primary

    async def __call__(self, state: TaskState, target: Target) -> Score:
        """Run all scorers and combine results."""
        results = []
        for s in self.scorers:
            result = await s(state, target)
            results.append(result)

        # Use primary scorer's value and answer
        primary_result = results[self.primary]

        # Merge all metadata
        combined_metadata = {}
        for i, result in enumerate(results):
            if result.metadata:
                if i == self.primary:
                    combined_metadata.update(result.metadata)
                else:
                    # Prefix secondary scorer metadata
                    prefix = f"scorer_{i}_"
                    for k, v in result.metadata.items():
                        combined_metadata[prefix + k] = v

        # Combine explanations
        explanations = [r.explanation for r in results if r.explanation]
        combined_explanation = " | ".join(explanations) if explanations else None

        return Score(
            value=primary_result.value,
            answer=primary_result.answer,
            explanation=combined_explanation,
            metadata=combined_metadata,
        )


def create_combined_scorer(
    base_scorer: Scorer,
    include_hatcat: bool = True,
) -> Scorer:
    """
    Factory for creating combined scorers.

    Args:
        base_scorer: Primary task scorer
        include_hatcat: Whether to add HatCat metrics

    Returns:
        Combined scorer instance
    """
    if include_hatcat:
        return hatcat_combined_scorer(base_scorer)
    return base_scorer
