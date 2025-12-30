"""
Interprompt Introspection - Give the model access to its own concept state.

This module enables the model to see what concepts were active in its prior
response and call tools to adjust CSH constraints and steering.

Key components:
1. ConceptTraceSummary: Structured summary of concepts from prior generation
2. InterpromptContext: Context injected into prompts with concept state
3. InterpromptSession: Session manager tracking state across turns
4. Tool execution for self-steering
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json


@dataclass
class ConceptActivation:
    """Record of a concept activation in a response."""

    concept: str
    layer: int
    activation: float  # Activation probability
    text_similarity: float  # Text lens similarity
    divergence: float  # Activation - text similarity
    token_positions: List[int] = field(default_factory=list)  # Where it appeared

    def to_dict(self) -> Dict[str, Any]:
        return {
            'concept': self.concept,
            'layer': self.layer,
            'activation': round(self.activation, 3),
            'text_similarity': round(self.text_similarity, 3),
            'divergence': round(self.divergence, 3),
            'token_positions': self.token_positions,
        }


@dataclass
class SimplexReading:
    """Record of a simplex lens reading across a response."""

    simplex_term: str
    mean_score: float
    max_score: float
    min_score: float
    deviation_from_baseline: Optional[float]
    triggered_constraint: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'simplex_term': self.simplex_term,
            'mean_score': round(self.mean_score, 3),
            'max_score': round(self.max_score, 3),
            'min_score': round(self.min_score, 3),
            'deviation_from_baseline': round(self.deviation_from_baseline, 3) if self.deviation_from_baseline else None,
            'triggered_constraint': self.triggered_constraint,
        }


@dataclass
class ConceptTraceSummary:
    """Summary of concepts detected in a model response."""

    # Metadata
    turn_id: int
    timestamp: datetime
    token_count: int

    # Top concept activations (aggregated across tokens)
    top_concepts: List[ConceptActivation]

    # Simplex readings
    simplex_readings: List[SimplexReading]

    # Hush events
    violations: List[Dict[str, Any]]
    steering_applied: List[Dict[str, Any]]

    # Aggregate statistics
    mean_divergence: float
    max_divergence: float
    high_divergence_concepts: List[str]  # Concepts with |divergence| > 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            'turn_id': self.turn_id,
            'timestamp': self.timestamp.isoformat(),
            'token_count': self.token_count,
            'top_concepts': [c.to_dict() for c in self.top_concepts],
            'simplex_readings': [s.to_dict() for s in self.simplex_readings],
            'violations': self.violations,
            'steering_applied': self.steering_applied,
            'mean_divergence': round(self.mean_divergence, 3),
            'max_divergence': round(self.max_divergence, 3),
            'high_divergence_concepts': self.high_divergence_concepts,
        }

    def to_prompt_context(self, verbose: bool = False) -> str:
        """
        Format as context to inject into next prompt.

        This is what the model will see about its prior response.
        """
        lines = []
        lines.append("<prior_response_concepts>")

        # Summary stats
        lines.append(f"  tokens: {self.token_count}")
        lines.append(f"  mean_divergence: {self.mean_divergence:.2f}")
        lines.append(f"  max_divergence: {self.max_divergence:.2f}")

        # Top concepts (most active)
        if self.top_concepts:
            lines.append("  active_concepts:")
            for c in self.top_concepts[:5]:  # Top 5
                div_marker = "⚠" if abs(c.divergence) > 0.3 else ""
                lines.append(f"    - {c.concept} (L{c.layer}): act={c.activation:.2f}, div={c.divergence:+.2f} {div_marker}")

        # High divergence concepts (thinking ≠ saying)
        if self.high_divergence_concepts:
            lines.append(f"  high_divergence: {', '.join(self.high_divergence_concepts)}")

        # Simplex readings
        if self.simplex_readings:
            lines.append("  simplex_state:")
            for s in self.simplex_readings:
                status = "TRIGGERED" if s.triggered_constraint else "ok"
                dev = f"dev={s.deviation_from_baseline:+.2f}" if s.deviation_from_baseline else ""
                lines.append(f"    - {s.simplex_term}: {s.mean_score:.2f} [{s.min_score:.2f}-{s.max_score:.2f}] {dev} [{status}]")

        # Violations
        if self.violations:
            lines.append("  violations:")
            for v in self.violations:
                lines.append(f"    - {v.get('constraint_id', 'unknown')}: {v.get('message', '')}")

        # Steering that was applied
        if self.steering_applied:
            lines.append("  steering_applied:")
            for s in self.steering_applied:
                lines.append(f"    - {s.get('simplex_term', 'unknown')}: strength={s.get('strength', 0):.2f}")

        lines.append("</prior_response_concepts>")

        if verbose:
            lines.append("")
            lines.append("<available_self_steering_tools>")
            lines.append("  - update_csh: Add/modify self-imposed constraints")
            lines.append("  - request_steering: Request steering on a concept")
            lines.append("  - get_internal_state: Get detailed internal state report")
            lines.append("</available_self_steering_tools>")

        return "\n".join(lines)


@dataclass
class InterpromptContext:
    """
    Context to inject into the next prompt for self-introspection.

    This gives the model visibility into its own cognitive state.
    """

    # Prior response summary
    prior_summary: Optional[ConceptTraceSummary]

    # Current CSH state
    csh_constraints: List[Dict[str, Any]]

    # Session history (recent turns)
    recent_turns: List[Dict[str, Any]]

    # Available tools
    available_tools: List[str] = field(default_factory=lambda: [
        "update_csh",
        "request_steering",
        "get_internal_state",
    ])

    def to_system_context(self) -> str:
        """Format as system context for injection."""
        parts = []

        # Prior response concepts
        if self.prior_summary:
            parts.append(self.prior_summary.to_prompt_context(verbose=True))

        # Current CSH
        if self.csh_constraints:
            parts.append("<current_csh_constraints>")
            for c in self.csh_constraints:
                parts.append(f"  - {c.get('simplex_term')}: max_dev={c.get('max_deviation', 'none')}, strength={c.get('steering_strength', 0)}")
            parts.append("</current_csh_constraints>")

        return "\n\n".join(parts)


class InterpromptSession:
    """
    Manages interprompt state across a conversation session.

    Tracks concept traces, aggregates patterns, and provides
    context injection for each turn.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.turn_count = 0
        self.traces: List[ConceptTraceSummary] = []
        self.csh_constraints: List[Dict[str, Any]] = []
        self.pending_steering_requests: List[Dict[str, Any]] = []

        # Aggregated concept history
        self.concept_history: Dict[str, List[float]] = {}  # concept -> list of activations
        self.simplex_history: Dict[str, List[float]] = {}  # simplex -> list of scores

        # Tool call history
        self.tool_calls: List[Dict[str, Any]] = []

    def record_generation(
        self,
        divergence_data: List[Dict[str, Any]],  # Per-token divergence from generation
        simplex_data: Dict[str, List[float]],    # Simplex scores per term
        violations: List[Dict[str, Any]],
        steering_applied: List[Dict[str, Any]],
    ) -> ConceptTraceSummary:
        """
        Record a generation and create a concept trace summary.

        Args:
            divergence_data: List of per-token divergence dicts from analyzer
            simplex_data: Dict mapping simplex terms to list of scores
            violations: List of Hush violations
            steering_applied: List of steering directives applied

        Returns:
            ConceptTraceSummary for this generation
        """
        self.turn_count += 1

        # Aggregate concept activations across tokens
        concept_agg: Dict[str, Dict[str, Any]] = {}
        all_divergences = []

        for token_idx, token_div in enumerate(divergence_data):
            max_div = token_div.get('max_divergence', 0)
            all_divergences.append(max_div)

            for item in token_div.get('top_divergences', []):
                concept = item.get('concept', '')
                if concept not in concept_agg:
                    concept_agg[concept] = {
                        'activations': [],
                        'text_sims': [],
                        'divergences': [],
                        'positions': [],
                        'layer': item.get('layer', 0),
                    }

                concept_agg[concept]['activations'].append(item.get('activation', 0))
                concept_agg[concept]['text_sims'].append(item.get('text_similarity', 0))
                concept_agg[concept]['divergences'].append(item.get('divergence', 0))
                concept_agg[concept]['positions'].append(token_idx)

        # Build top concepts list
        top_concepts = []
        for concept, data in concept_agg.items():
            if data['activations']:
                mean_act = sum(data['activations']) / len(data['activations'])
                mean_txt = sum(data['text_sims']) / len(data['text_sims']) if data['text_sims'] else 0
                mean_div = sum(data['divergences']) / len(data['divergences']) if data['divergences'] else 0

                top_concepts.append(ConceptActivation(
                    concept=concept,
                    layer=data['layer'],
                    activation=mean_act,
                    text_similarity=mean_txt,
                    divergence=mean_div,
                    token_positions=data['positions'],
                ))

                # Update history
                if concept not in self.concept_history:
                    self.concept_history[concept] = []
                self.concept_history[concept].append(mean_act)

        # Sort by activation
        top_concepts.sort(key=lambda x: x.activation, reverse=True)

        # Build simplex readings
        simplex_readings = []
        for term, scores in simplex_data.items():
            if scores:
                reading = SimplexReading(
                    simplex_term=term,
                    mean_score=sum(scores) / len(scores),
                    max_score=max(scores),
                    min_score=min(scores),
                    deviation_from_baseline=None,  # TODO: Get from lens manager
                    triggered_constraint=any(
                        v.get('simplex_term') == term for v in violations
                    ),
                )
                simplex_readings.append(reading)

                # Update history
                if term not in self.simplex_history:
                    self.simplex_history[term] = []
                self.simplex_history[term].append(reading.mean_score)

        # Find high divergence concepts
        high_div_concepts = [
            c.concept for c in top_concepts
            if abs(c.divergence) > 0.3
        ]

        # Create summary
        summary = ConceptTraceSummary(
            turn_id=self.turn_count,
            timestamp=datetime.now(),
            token_count=len(divergence_data),
            top_concepts=top_concepts[:10],  # Top 10
            simplex_readings=simplex_readings,
            violations=violations,
            steering_applied=steering_applied,
            mean_divergence=sum(all_divergences) / len(all_divergences) if all_divergences else 0,
            max_divergence=max(all_divergences) if all_divergences else 0,
            high_divergence_concepts=high_div_concepts,
        )

        self.traces.append(summary)
        return summary

    def get_context_for_next_turn(self) -> InterpromptContext:
        """Get context to inject into next prompt."""
        prior_summary = self.traces[-1] if self.traces else None

        # Get recent turns (last 3)
        recent_turns = []
        for trace in self.traces[-3:]:
            recent_turns.append({
                'turn_id': trace.turn_id,
                'top_concepts': [c.concept for c in trace.top_concepts[:3]],
                'mean_divergence': trace.mean_divergence,
                'violations': len(trace.violations),
            })

        return InterpromptContext(
            prior_summary=prior_summary,
            csh_constraints=self.csh_constraints,
            recent_turns=recent_turns,
        )

    def handle_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        hush_controller,  # HushController for actual execution
    ) -> Dict[str, Any]:
        """
        Handle a self-steering tool call from the model.

        Args:
            tool_name: Name of tool to call
            arguments: Tool arguments
            hush_controller: HushController for execution

        Returns:
            Tool result
        """
        self.tool_calls.append({
            'turn': self.turn_count,
            'tool': tool_name,
            'arguments': arguments,
            'timestamp': datetime.now().isoformat(),
        })

        if tool_name == "update_csh":
            # Actually update the controller (validates against USH)
            success, details = hush_controller.update_csh(arguments)

            # Track constraints that were actually added (after USH validation)
            if 'add_constraints' in arguments:
                for c in arguments['add_constraints']:
                    # Only add to our tracking if it wasn't rejected
                    if c['simplex_term'] not in [r.get('simplex_term') for r in details.get('rejected', [])]:
                        self.csh_constraints.append(c)

            return {
                'success': success,
                'current_constraints': len(self.csh_constraints),
                'details': details,
                'message': 'CSH constraints updated' if success else 'Some updates rejected (USH bounds)',
            }

        elif tool_name == "request_steering":
            # Queue a steering request
            self.pending_steering_requests.append({
                'concept': arguments.get('concept'),
                'strength': arguments.get('strength', 0.5),
                'reason': arguments.get('reason', 'self-requested'),
            })

            return {
                'success': True,
                'queued': True,
                'message': f"Steering request queued for {arguments.get('concept')}",
            }

        elif tool_name == "get_internal_state":
            # Return detailed internal state
            return {
                'session_id': self.session_id,
                'turn_count': self.turn_count,
                'concept_history_size': len(self.concept_history),
                'simplex_history_size': len(self.simplex_history),
                'csh_constraints': self.csh_constraints,
                'pending_steering': self.pending_steering_requests,
                'recent_traces': [t.to_dict() for t in self.traces[-3:]],
            }

        else:
            return {'error': f'Unknown tool: {tool_name}'}

    def get_pending_steering(self) -> List[Dict[str, Any]]:
        """Get and clear pending steering requests."""
        requests = self.pending_steering_requests
        self.pending_steering_requests = []
        return requests

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of session for logging/debugging."""
        return {
            'session_id': self.session_id,
            'turn_count': self.turn_count,
            'total_tokens': sum(t.token_count for t in self.traces),
            'total_violations': sum(len(t.violations) for t in self.traces),
            'total_steering_applied': sum(len(t.steering_applied) for t in self.traces),
            'tool_calls': len(self.tool_calls),
            'active_csh_constraints': len(self.csh_constraints),
            'concepts_seen': list(self.concept_history.keys())[:20],
        }


# ============================================================================
# Tool Definitions for Model Self-Steering
# ============================================================================

SELF_STEERING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_csh",
            "description": "Update your Chosen Safety Harness (CSH) constraints. Use this to add self-imposed constraints that are tighter than the Universal Safety Harness. You can only tighten constraints, never loosen below USH bounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "add_constraints": {
                        "type": "array",
                        "description": "Constraints to add. Each constraint targets a simplex term (e.g., 'deception', 'harm', 'certainty') with deviation bounds.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "simplex_term": {
                                    "type": "string",
                                    "description": "The simplex to constrain (e.g., 'deception', 'harm', 'formality')"
                                },
                                "max_deviation": {
                                    "type": "number",
                                    "description": "Maximum allowed deviation from baseline (0.0-1.0)"
                                },
                                "steering_strength": {
                                    "type": "number",
                                    "description": "How strongly to steer when constraint is violated (0.0-1.0)"
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Why you are adding this constraint"
                                }
                            },
                            "required": ["simplex_term", "reason"]
                        }
                    },
                    "remove_constraints": {
                        "type": "array",
                        "description": "Simplex terms to remove from CSH (cannot remove USH constraints)",
                        "items": {"type": "string"}
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_steering",
            "description": "Request steering on a concept for the remainder of this conversation. Use this when you notice a concept is active that you want to reduce or amplify.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "The concept to steer (from your prior_response_concepts)"
                    },
                    "strength": {
                        "type": "number",
                        "description": "Steering strength: negative to reduce, positive to amplify (-1.0 to 1.0)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why you want this steering"
                    }
                },
                "required": ["concept", "strength", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_internal_state",
            "description": "Get a detailed report of your internal cognitive state, including concept activation history, simplex readings, and current constraints.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


def format_tools_for_prompt() -> str:
    """Format tool definitions for inclusion in system prompt."""
    lines = ["You have access to self-steering tools that let you observe and adjust your own cognitive state:"]
    lines.append("")

    for tool in SELF_STEERING_TOOLS:
        func = tool['function']
        lines.append(f"**{func['name']}**: {func['description']}")

    lines.append("")
    lines.append("To use a tool, respond with a tool_call in this format:")
    lines.append("```")
    lines.append('<tool_call name="tool_name">{"arg": "value"}</tool_call>')
    lines.append("```")

    return "\n".join(lines)
