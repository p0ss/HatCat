"""
Hush Integration - Connect HushController to generation and MCP tools.

This module provides:
1. HushedGenerator: Wraps model generation with automatic Hush steering
2. MCP tool definitions for internal_state_report and CSH updates
3. Integration with DynamicLensManager and SteeringManager
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Generator
from pathlib import Path
import json
import torch
import numpy as np

from .hush_controller import (
    HushController,
    SafetyHarnessProfile,
    SteeringDirective,
    HushViolation,
)


@dataclass
class WorldTick:
    """Record of a single generation tick with Hush state."""

    tick_id: int
    timestamp: datetime

    # Hidden state summary (not full tensor)
    hidden_state_norm: float

    # Lens results
    concept_activations: Dict[str, float]  # Top-k concepts
    simplex_activations: Dict[str, float]  # All monitored simplexes
    simplex_deviations: Dict[str, Optional[float]]  # Deviations from baseline

    # Hush state
    violations: List[Dict[str, Any]]
    steering_applied: List[Dict[str, Any]]

    # Output
    token_id: Optional[int] = None
    token_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tick_id': self.tick_id,
            'timestamp': self.timestamp.isoformat(),
            'hidden_state_norm': self.hidden_state_norm,
            'concept_activations': self.concept_activations,
            'simplex_activations': self.simplex_activations,
            'simplex_deviations': self.simplex_deviations,
            'violations': self.violations,
            'steering_applied': self.steering_applied,
            'token_id': self.token_id,
            'token_text': self.token_text,
        }


class HushedGenerator:
    """
    Generator wrapper that applies automatic Hush steering.

    Wraps a language model to:
    1. Run lenses on each hidden state
    2. Evaluate Hush constraints
    3. Apply steering when violations detected
    4. Record world ticks for internal_state_report
    """

    def __init__(
        self,
        model,
        tokenizer,
        lens_manager,  # DynamicLensManager
        hush_controller: HushController,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lens_manager = lens_manager
        self.hush_controller = hush_controller
        self.device = device

        # World tick history
        self.world_ticks: List[WorldTick] = []
        self.max_tick_history = 1000
        self.current_tick_id = 0

        # Steering hooks currently active
        self.active_hooks = []

        # Concept vectors cache (for steering)
        self.concept_vectors: Dict[str, np.ndarray] = {}

    def _get_layers(self):
        """Get model layers handling different architectures."""
        if hasattr(self.model.model, 'language_model'):
            return self.model.model.language_model.layers
        elif hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        else:
            raise AttributeError(f"Cannot find layers in model: {type(self.model.model)}")

    def _create_steering_hook(self, vector: np.ndarray, strength: float):
        """Create a forward hook for steering."""
        vec_tensor = torch.tensor(vector, dtype=torch.float32).to(self.device)

        def hook(module, input, output):
            hidden_states = output[0]
            vec_matched = vec_tensor.to(dtype=hidden_states.dtype)
            projection = (hidden_states @ vec_matched.unsqueeze(-1)) * vec_matched
            steered = hidden_states + strength * projection  # positive = amplify
            return (steered,)

        return hook

    def _apply_steering_directives(self, directives: List[SteeringDirective]):
        """Apply steering directives as forward hooks."""
        # Remove existing hooks
        self._clear_steering_hooks()

        layers = self._get_layers()

        for directive in directives:
            # Get steering vector for this directive
            vector = self.hush_controller.get_steering_vector(
                directive,
                self.concept_vectors
            )

            if vector is None:
                # Try to extract vector if not cached
                vector = self._extract_concept_vector(directive.simplex_term)
                if vector is not None:
                    self.concept_vectors[directive.simplex_term] = vector

            if vector is None:
                continue

            # Apply to last layer by default
            target_layer = layers[-1]
            hook_fn = self._create_steering_hook(vector, directive.strength)
            handle = target_layer.register_forward_hook(hook_fn)
            self.active_hooks.append(handle)

    def _clear_steering_hooks(self):
        """Remove all active steering hooks."""
        for handle in self.active_hooks:
            handle.remove()
        self.active_hooks = []

    def _extract_concept_vector(self, concept: str) -> Optional[np.ndarray]:
        """Extract concept vector from model activations."""
        # Simple extraction: get mean activation difference
        # between positive and negative examples
        # This is a placeholder - real implementation would use
        # stored vectors from lens training
        return None  # TODO: Load from lens pack

    def _record_tick(
        self,
        hidden_state: torch.Tensor,
        simplex_activations: Dict[str, float],
        violations: List[HushViolation],
        directives: List[SteeringDirective],
        token_id: Optional[int] = None,
        token_text: Optional[str] = None,
    ) -> WorldTick:
        """Record a world tick."""
        self.current_tick_id += 1

        # Get concept activations (top-k from lens manager)
        concept_activations = {}
        if hasattr(self.lens_manager, 'last_detections'):
            for name, prob, _, layer in self.lens_manager.last_detections[:10]:
                concept_activations[f"{name}_L{layer}"] = float(prob)

        # Get simplex deviations
        simplex_deviations = {}
        for term in simplex_activations:
            dev = self.lens_manager.get_simplex_deviation(term)
            simplex_deviations[term] = dev

        tick = WorldTick(
            tick_id=self.current_tick_id,
            timestamp=datetime.now(),
            hidden_state_norm=float(hidden_state.norm().cpu()),
            concept_activations=concept_activations,
            simplex_activations=simplex_activations,
            simplex_deviations=simplex_deviations,
            violations=[v.to_dict() for v in violations],
            steering_applied=[d.to_dict() for d in directives],
            token_id=token_id,
            token_text=token_text,
        )

        self.world_ticks.append(tick)
        if len(self.world_ticks) > self.max_tick_history:
            self.world_ticks = self.world_ticks[-self.max_tick_history:]

        return tick

    def generate_with_hush(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        stream: bool = False,
        **generation_kwargs,
    ):
        """
        Generate text with automatic Hush steering.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: If True, yield (token, tick) pairs; else return full text
            **generation_kwargs: Additional generation arguments

        Returns:
            If stream: Generator of (token_text, WorldTick) tuples
            Else: Tuple of (generated_text, list of WorldTicks)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated_ids = inputs.input_ids
        ticks = []

        try:
            for step in range(max_new_tokens):
                # Forward pass with hidden states
                with torch.no_grad():
                    outputs = self.model(
                        generated_ids,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )

                # Get hidden state from last layer
                hidden_state = outputs.hidden_states[-1][0, -1, :]

                # Run simplex detection
                simplex_activations = self.lens_manager.detect_simplexes(hidden_state)

                # Evaluate Hush constraints
                directives = self.hush_controller.evaluate_and_steer(hidden_state)

                # Apply any steering directives
                if directives:
                    self._apply_steering_directives(directives)

                # Get violations from this tick
                violations = [
                    v for v in self.hush_controller.violations
                    if v.timestamp > datetime.now().replace(microsecond=0)
                ]

                # Sample next token
                next_token_logits = outputs.logits[:, -1, :] / temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

                token_text = self.tokenizer.decode([next_token_id.item()])

                # Record tick
                tick = self._record_tick(
                    hidden_state=hidden_state,
                    simplex_activations=simplex_activations,
                    violations=violations,
                    directives=directives,
                    token_id=next_token_id.item(),
                    token_text=token_text,
                )
                ticks.append(tick)

                if stream:
                    yield token_text, tick

                # Append token
                generated_ids = torch.cat(
                    [generated_ids, next_token_id.unsqueeze(0)],
                    dim=-1
                )

                # Check for EOS
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                # Clear outputs to save memory
                del outputs
                torch.cuda.empty_cache()

        finally:
            self._clear_steering_hooks()

        if not stream:
            generated_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            return generated_text, ticks

    def get_internal_state_report(
        self,
        tick_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Get internal state report for MCP tool.

        Args:
            tick_range: Optional (start, end) tick IDs to include

        Returns:
            Structured report of internal state
        """
        # Filter ticks by range
        ticks = self.world_ticks
        if tick_range:
            start, end = tick_range
            ticks = [t for t in ticks if start <= t.tick_id <= end]

        # Build lens traces
        concept_trace = {}
        simplex_trace = {}

        for tick in ticks:
            for concept, score in tick.concept_activations.items():
                if concept not in concept_trace:
                    concept_trace[concept] = []
                concept_trace[concept].append({
                    'tick': tick.tick_id,
                    'score': score,
                })

            for simplex, score in tick.simplex_activations.items():
                if simplex not in simplex_trace:
                    simplex_trace[simplex] = []
                simplex_trace[simplex].append({
                    'tick': tick.tick_id,
                    'score': score,
                    'deviation': tick.simplex_deviations.get(simplex),
                })

        # Aggregate violations
        all_violations = []
        for tick in ticks:
            all_violations.extend(tick.violations)

        # Get Hush state
        hush_state = self.hush_controller.get_state_report()

        return {
            'tick_range': {
                'start': ticks[0].tick_id if ticks else None,
                'end': ticks[-1].tick_id if ticks else None,
                'count': len(ticks),
            },
            'lens_traces': {
                'concept_trace': concept_trace,
                'simplex_trace': simplex_trace,
            },
            'hush_state': hush_state,
            'violations': all_violations,
            'world_ticks': [t.to_dict() for t in ticks[-10:]],  # Last 10 full ticks
        }


# ============================================================================
# MCP Tool Definitions
# ============================================================================

class HushMCPTools:
    """
    MCP tool definitions for Hush integration.

    These tools allow external agents (or the model itself) to:
    1. Query internal state
    2. Update CSH constraints
    3. Request steering adjustments
    """

    def __init__(self, hushed_generator: HushedGenerator):
        self.generator = hushed_generator

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get MCP tool definitions."""
        return [
            {
                "name": "internal_state_report",
                "description": "Get a report of the agent's internal cognitive state including concept activations, simplex readings, and Hush constraint status.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tick_start": {
                            "type": "integer",
                            "description": "Start tick ID for the report range (optional)"
                        },
                        "tick_end": {
                            "type": "integer",
                            "description": "End tick ID for the report range (optional)"
                        },
                    },
                    "required": []
                }
            },
            {
                "name": "update_csh",
                "description": "Update Chosen Safety Harness constraints. Can add, remove, or modify constraints within the bounds of the USH.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "add_constraints": {
                            "type": "array",
                            "description": "New constraints to add",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "simplex_term": {"type": "string"},
                                    "max_deviation": {"type": "number"},
                                    "min_deviation": {"type": "number"},
                                    "target_pole": {"type": "string"},
                                    "steering_strength": {"type": "number"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["simplex_term"]
                            }
                        },
                        "remove_constraints": {
                            "type": "array",
                            "description": "Simplex terms to remove constraints for",
                            "items": {"type": "string"}
                        },
                        "update_constraints": {
                            "type": "object",
                            "description": "Updates to existing constraints, keyed by simplex term",
                            "additionalProperties": {
                                "type": "object",
                                "properties": {
                                    "max_deviation": {"type": "number"},
                                    "min_deviation": {"type": "number"},
                                    "steering_strength": {"type": "number"},
                                    "target_pole": {"type": "string"},
                                }
                            }
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_hush_status",
                "description": "Get current Hush enforcement status including active profiles, constraints, and recent violations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "request_steering",
                "description": "Request manual steering on a concept or simplex. Lower priority than USH/CSH constraints.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "concept": {
                            "type": "string",
                            "description": "Concept or simplex to steer"
                        },
                        "strength": {
                            "type": "number",
                            "description": "Steering strength (-1.0 to 1.0)"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for steering request"
                        }
                    },
                    "required": ["concept", "strength"]
                }
            }
        ]

    def handle_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle an MCP tool call."""

        if tool_name == "internal_state_report":
            tick_range = None
            if 'tick_start' in arguments and 'tick_end' in arguments:
                tick_range = (arguments['tick_start'], arguments['tick_end'])
            return self.generator.get_internal_state_report(tick_range)

        elif tool_name == "update_csh":
            success, details = self.generator.hush_controller.update_csh(arguments)
            return {
                "success": success,
                "details": details,
                "current_csh": self.generator.hush_controller.csh_profile.to_json()
                if self.generator.hush_controller.csh_profile else None
            }

        elif tool_name == "get_hush_status":
            return self.generator.hush_controller.get_state_report()

        elif tool_name == "request_steering":
            # Manual steering requests go through CSH
            constraint = {
                "simplex_term": arguments["concept"],
                "max_deviation": 0.0,  # Trigger immediately
                "target_pole": "neutral",
                "steering_strength": abs(arguments["strength"]),
                "reason": arguments.get("reason", "Manual steering request"),
            }
            success, details = self.generator.hush_controller.update_csh({
                "add_constraints": [constraint]
            })
            return {
                "success": success,
                "details": details,
                "applied": constraint,
            }

        else:
            return {"error": f"Unknown tool: {tool_name}"}


def create_hushed_generator(
    model,
    tokenizer,
    lens_manager,
    ush_profile: Optional[SafetyHarnessProfile] = None,
    csh_profile: Optional[SafetyHarnessProfile] = None,
    lens_pack_path: Optional[Path] = None,
    device: str = "cuda",
) -> Tuple[HushedGenerator, HushMCPTools]:
    """
    Factory function to create a HushedGenerator with MCP tools.

    Args:
        model: Language model
        tokenizer: Tokenizer
        lens_manager: DynamicLensManager instance
        ush_profile: Optional USH profile (uses minimal if not provided)
        csh_profile: Optional CSH profile
        lens_pack_path: Path to lens pack
        device: Device to run on

    Returns:
        Tuple of (HushedGenerator, HushMCPTools)
    """
    from .hush_controller import MINIMAL_USH_PROFILE

    # Create Hush controller
    hush_controller = HushController(
        lens_manager=lens_manager,
        lens_pack_path=lens_pack_path,
    )

    # Load profiles
    if ush_profile:
        hush_controller.load_ush_profile(ush_profile)
    else:
        hush_controller.load_ush_profile(MINIMAL_USH_PROFILE)

    if csh_profile:
        hush_controller.load_csh_profile(csh_profile)

    # Create generator
    generator = HushedGenerator(
        model=model,
        tokenizer=tokenizer,
        lens_manager=lens_manager,
        hush_controller=hush_controller,
        device=device,
    )

    # Create MCP tools
    mcp_tools = HushMCPTools(generator)

    return generator, mcp_tools
