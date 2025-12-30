"""
Temporal activation visualization for Gradio UI.

Provides sparkline visualization of concept activations over generation timesteps.
Granularity-agnostic design supports per-token, intratoken, and pre-token activations.
"""

import numpy as np
from typing import List, Dict, Any, Tuple


SPARKLINE_CHARS = '▁▂▃▄▅▆▇█'


def generate_sparkline(values: List[float], width: int = 40) -> str:
    """
    Generate ASCII sparkline from activation values.

    Args:
        values: List of activation values (0.0 to 1.0)
        width: Maximum width of sparkline

    Returns:
        Sparkline string using Unicode block characters
    """
    if not values or all(v == 0 for v in values):
        return '─' * min(width, len(values))

    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return '─' * min(width, len(values))

    # Normalize to 0-7 range for character indices
    normalized = [(v - min_val) / (max_val - min_val) for v in values]
    indices = [int(n * (len(SPARKLINE_CHARS) - 1)) for n in normalized]

    # Sample if we have more values than width
    if len(values) > width:
        step = len(values) / width
        sampled_indices = [indices[int(i * step)] for i in range(width)]
        return ''.join(SPARKLINE_CHARS[i] for i in sampled_indices)
    else:
        sparkline = ''.join(SPARKLINE_CHARS[i] for i in indices)
        # Pad if fewer values
        return sparkline + '─' * (width - len(sparkline))


def aggregate_timesteps(
    timeline: List[Dict[str, Any]],
    aggregation: str = "sentence",
) -> List[Dict[str, Any]]:
    """
    Aggregate temporal data into larger chunks.

    Args:
        timeline: List of timestep dictionaries with 'concepts' and optional 'token'
        aggregation: "timestep", "token", "sentence", "reply"

    Returns:
        List of aggregated timesteps
    """
    if aggregation == "timestep":
        # No aggregation, return as-is
        return timeline

    if aggregation == "token":
        # Group by token_idx if available
        aggregated = []
        current_token_idx = None
        current_concepts = {}

        for step in timeline:
            token_idx = step.get('token_idx', step.get('step_idx', 0))

            if token_idx != current_token_idx and current_concepts:
                # Save previous token
                aggregated.append({
                    'token_idx': current_token_idx,
                    'concepts': current_concepts.copy(),
                    'token': step.get('token', '')
                })
                current_concepts = {}

            current_token_idx = token_idx

            # Merge concepts (take max activation)
            for concept, data in step.get('concepts', {}).items():
                if concept not in current_concepts:
                    current_concepts[concept] = data
                else:
                    # Take max activation
                    if data.get('divergence', 0) > current_concepts[concept].get('divergence', 0):
                        current_concepts[concept] = data

        # Don't forget last token
        if current_concepts:
            aggregated.append({
                'token_idx': current_token_idx,
                'concepts': current_concepts,
                'token': timeline[-1].get('token', '') if timeline else ''
            })

        return aggregated

    if aggregation == "sentence":
        # Simple heuristic: group until we hit sentence-ending punctuation
        aggregated = []
        current_concepts = {}
        sentence_tokens = []

        for step in timeline:
            token = step.get('token', '')
            sentence_tokens.append(token)

            # Merge concepts
            for concept, data in step.get('concepts', {}).items():
                if concept not in current_concepts:
                    current_concepts[concept] = data
                else:
                    if data.get('divergence', 0) > current_concepts[concept].get('divergence', 0):
                        current_concepts[concept] = data

            # Check for sentence ending
            if token.strip() in {'.', '!', '?', '。', '！', '？'}:
                aggregated.append({
                    'concepts': current_concepts.copy(),
                    'tokens': ''.join(sentence_tokens)
                })
                current_concepts = {}
                sentence_tokens = []

        # Add remaining
        if current_concepts:
            aggregated.append({
                'concepts': current_concepts,
                'tokens': ''.join(sentence_tokens)
            })

        return aggregated

    if aggregation == "reply":
        # Single aggregation over entire reply
        all_concepts = {}
        all_tokens = []

        for step in timeline:
            all_tokens.append(step.get('token', ''))
            for concept, data in step.get('concepts', {}).items():
                if concept not in all_concepts:
                    all_concepts[concept] = data
                else:
                    if data.get('divergence', 0) > all_concepts[concept].get('divergence', 0):
                        all_concepts[concept] = data

        return [{
            'concepts': all_concepts,
            'tokens': ''.join(all_tokens)
        }]

    return timeline


def select_top_concepts(
    aggregated: List[Dict[str, Any]],
    top_k: int = 5,
    timerange: str = "sentence",
) -> List[str]:
    """
    Select top-K concepts based on timerange.

    Args:
        aggregated: Aggregated timesteps
        top_k: Number of top concepts to show
        timerange: "timestep" (per-timestep), "sentence", or "reply" (global)

    Returns:
        List of concept names to display
    """
    if timerange == "timestep":
        # Return top concepts at each timestep (variable set)
        # Not directly applicable - caller should handle per-timestep filtering
        return []

    # For sentence or reply: find globally top concepts
    concept_max_divergence = {}

    for step in aggregated:
        for concept, data in step.get('concepts', {}).items():
            div = abs(data.get('divergence', 0))
            if concept not in concept_max_divergence:
                concept_max_divergence[concept] = div
            else:
                concept_max_divergence[concept] = max(concept_max_divergence[concept], div)

    # Sort by max divergence
    sorted_concepts = sorted(
        concept_max_divergence.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [name for name, _ in sorted_concepts[:top_k]]


def format_temporal_view(
    timeline: List[Dict[str, Any]],
    top_k: int = 5,
    aggregation: str = "sentence",
    timerange: str = "sentence",
    sparkline_width: int = 40,
) -> str:
    """
    Format temporal activation data as text visualization.

    Args:
        timeline: Raw timeline data from generation
        top_k: Number of concepts to show
        aggregation: How to chunk timesteps ("timestep", "token", "sentence", "reply")
        timerange: What timerange to pick top concepts from ("timestep", "sentence", "reply")
        sparkline_width: Width of sparklines

    Returns:
        Formatted text visualization with sparklines
    """
    # Aggregate timesteps
    aggregated = aggregate_timesteps(timeline, aggregation)

    if not aggregated:
        return "No temporal data available"

    # Select top concepts
    if timerange == "timestep":
        # Per-timestep: show top-K at each step
        output = []
        output.append("=" * 80)
        output.append("TEMPORAL ACTIVATIONS (per-timestep top concepts)")
        output.append("=" * 80)

        for i, step in enumerate(aggregated):
            concepts = step.get('concepts', {})
            sorted_concepts = sorted(
                concepts.items(),
                key=lambda x: abs(x[1].get('divergence', 0)),
                reverse=True
            )[:top_k]

            token = step.get('token', step.get('tokens', f'[{i}]'))
            output.append(f"\nTimestep {i} [{token}]:")

            for concept, data in sorted_concepts:
                div = data.get('divergence', 0)
                output.append(f"  {concept:30s} {div:+7.3f}")

        return '\n'.join(output)

    else:
        # Global top-K
        top_concepts = select_top_concepts(aggregated, top_k, timerange)

        if not top_concepts:
            return "No concepts detected"

        # Build timeseries for each top concept
        concept_timeseries = {name: [] for name in top_concepts}

        for step in aggregated:
            concepts = step.get('concepts', {})
            for name in top_concepts:
                if name in concepts:
                    concept_timeseries[name].append(abs(concepts[name].get('divergence', 0)))
                else:
                    concept_timeseries[name].append(0.0)

        # Format output
        output = []
        output.append("=" * 80)
        output.append(f"TEMPORAL ACTIVATIONS (top {top_k} concepts, {aggregation} granularity)")
        output.append("=" * 80)
        output.append("")

        for concept in top_concepts:
            values = concept_timeseries[concept]
            max_val = max(values) if values else 0.0
            sparkline = generate_sparkline(values, sparkline_width)
            output.append(f"{concept:30s} [{max_val:5.3f}] {sparkline}")

        output.append("")
        output.append("=" * 80)
        output.append(f"Total {aggregation}s: {len(aggregated)}")
        output.append("=" * 80)

        return '\n'.join(output)


def format_detailed_view(
    timeline: List[Dict[str, Any]],
    top_k_per_step: int = 5,
) -> str:
    """
    Format detailed per-timestep view showing all activations.

    Args:
        timeline: Raw timeline data
        top_k_per_step: Number of concepts to show at each timestep

    Returns:
        Formatted text with detailed activation data
    """
    output = []
    output.append("=" * 80)
    output.append("DETAILED TIMESTEP VIEW")
    output.append("=" * 80)
    output.append("")

    for i, step in enumerate(timeline):
        token = step.get('token', f'[{i}]')
        concepts = step.get('concepts', {})

        # Sort by divergence
        sorted_concepts = sorted(
            concepts.items(),
            key=lambda x: abs(x[1].get('divergence', 0)),
            reverse=True
        )[:top_k_per_step]

        output.append(f"Timestep {i:3d} [{token:15s}]")

        for concept, data in sorted_concepts:
            div = data.get('divergence', 0)
            act = data.get('activation', data.get('probability', 0))
            layer = data.get('layer', '?')
            output.append(f"  {concept:30s} L{layer} div={div:+7.3f} act={act:.3f}")

        output.append("")

    return '\n'.join(output)
