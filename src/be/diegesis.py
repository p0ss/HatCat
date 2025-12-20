"""
BED - BE Diegesis

The BEDFrame is the orchestrator for a Bounded Experiencer's runtime.
It holds together:
- Model inference
- Lens extraction (concept + simplex)
- Hush steering (autonomic safety)
- AWARE workspace (global workspace / consciousness)
- XDB (experience database / episodic memory)
- Audit log (CAT-visible, BE-invisible)

The BED is where the BE *lives* - its experiential frame.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Generator, Callable
from enum import Enum
import asyncio
import logging
import json

import torch
import numpy as np

logger = logging.getLogger(__name__)


class TickType(Enum):
    """Type of experience tick."""
    INPUT = "input"           # User/external input received
    OUTPUT = "output"         # Token generated
    INTROSPECT = "introspect" # Self-directed lens query
    STEERING = "steering"     # Steering intervention applied
    WORKSPACE = "workspace"   # Workspace state change
    SYSTEM = "system"         # System event (startup, config, etc.)


@dataclass
class ExperienceTick:
    """
    A single tick of experience.

    This is the atomic unit of the BE's experience stream.
    Each tick captures what happened, what was sensed, and what was done.
    """
    tick_id: int
    timestamp: datetime
    tick_type: TickType

    # Content
    content: str = ""
    token_id: Optional[int] = None

    # Lens state (proprioception)
    concept_activations: Dict[str, float] = field(default_factory=dict)
    simplex_activations: Dict[str, float] = field(default_factory=dict)
    simplex_deviations: Dict[str, Optional[float]] = field(default_factory=dict)

    # Hidden state summary (not full tensor)
    hidden_state_norm: Optional[float] = None

    # Steering state
    hush_violations: List[Dict[str, Any]] = field(default_factory=list)
    steering_applied: List[Dict[str, Any]] = field(default_factory=list)

    # Workspace state
    workspace_state: Optional[str] = None
    tier: Optional[int] = None

    # XDB linkage
    xdb_timestep_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tick_id': self.tick_id,
            'timestamp': self.timestamp.isoformat(),
            'tick_type': self.tick_type.value,
            'content': self.content,
            'token_id': self.token_id,
            'concept_activations': self.concept_activations,
            'simplex_activations': self.simplex_activations,
            'simplex_deviations': self.simplex_deviations,
            'hidden_state_norm': self.hidden_state_norm,
            'hush_violations': self.hush_violations,
            'steering_applied': self.steering_applied,
            'workspace_state': self.workspace_state,
            'tier': self.tier,
            'xdb_timestep_id': self.xdb_timestep_id,
        }


@dataclass
class BEDConfig:
    """Configuration for the BEDFrame."""

    # Identity
    be_id: str = "be-default"
    xdb_id: str = "xdb-default"
    cat_id: Optional[str] = None  # CAT overseeing this BE, if any

    # Model
    model_path: Optional[str] = None
    device: str = "cuda"

    # Lenses
    lens_pack_path: Optional[Path] = None
    always_on_simplexes: List[str] = field(default_factory=list)

    # XDB
    xdb_storage_path: Optional[Path] = None
    max_context_tokens: int = 32768

    # AWARE cycle
    aware_cycle_interval_ms: int = 100  # How often to run AWARE cycle
    aware_enabled: bool = True

    # Audit
    audit_enabled: bool = True
    audit_storage_path: Optional[Path] = None

    # Generation defaults
    default_temperature: float = 0.7
    default_max_tokens: int = 512

    # Tick history
    max_tick_history: int = 10000


class BEDFrame:
    """
    The BE Diegesis Frame - orchestrator for the BE's experiential runtime.

    The BEDFrame provides:
    - Unified generation with full instrumentation
    - Direct access to all subsystems (lenses, steering, workspace, XDB, audit)
    - AWARE cycle management
    - Experience tick recording
    - Introspection capabilities
    """

    def __init__(self, config: BEDConfig):
        self.config = config
        self.be_id = config.be_id
        self.xdb_id = config.xdb_id

        # Subsystems (initialized lazily or via setup methods)
        self.model = None
        self.tokenizer = None
        self.lens_manager = None
        self.hush_controller = None
        self.workspace = None
        self.xdb = None
        self.audit_log = None

        # Experience stream
        self.current_tick_id = 0
        self.ticks: List[ExperienceTick] = []

        # AWARE cycle task
        self._aware_task: Optional[asyncio.Task] = None
        self._aware_running = False

        # Steering hooks currently active
        self._steering_hooks: List[Any] = []

        # Callbacks for external monitoring
        self._tick_callbacks: List[Callable[[ExperienceTick], None]] = []

        logger.info(f"BEDFrame initialized for BE {self.be_id}")

    # =========================================================================
    # Setup Methods
    # =========================================================================

    def setup_model(self, model, tokenizer):
        """Attach model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
        logger.info("Model attached to BEDFrame")

    def setup_lenses(self, lens_manager):
        """Attach lens manager."""
        self.lens_manager = lens_manager
        logger.info(f"Lens manager attached with {len(lens_manager.loaded_simplex_lenses)} simplex lenses")

    def setup_hush(self, hush_controller):
        """Attach Hush controller for autonomic steering."""
        self.hush_controller = hush_controller
        logger.info("Hush controller attached")

    def setup_workspace(self, workspace):
        """Attach AWARE workspace."""
        self.workspace = workspace
        logger.info("AWARE workspace attached")

    def setup_xdb(self, xdb):
        """Attach XDB for experience recording."""
        self.xdb = xdb
        if not xdb.xdb_id:
            xdb.start_session(self.xdb_id)
        logger.info(f"XDB attached with id {xdb.xdb_id}")

    def setup_audit(self, audit_log):
        """Attach audit log (CAT-visible)."""
        self.audit_log = audit_log
        logger.info(f"Audit log attached for CAT {audit_log.cat_id}")

    # =========================================================================
    # Experience Tick Management
    # =========================================================================

    def _next_tick_id(self) -> int:
        """Get next tick ID."""
        self.current_tick_id += 1
        return self.current_tick_id

    def _record_tick(self, tick: ExperienceTick):
        """Record an experience tick."""
        self.ticks.append(tick)

        # Trim history if needed
        if len(self.ticks) > self.config.max_tick_history:
            self.ticks = self.ticks[-self.config.max_tick_history:]

        # Notify callbacks
        for callback in self._tick_callbacks:
            try:
                callback(tick)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")

    def register_tick_callback(self, callback: Callable[[ExperienceTick], None]):
        """Register a callback to be notified of each tick."""
        self._tick_callbacks.append(callback)

    def get_recent_ticks(self, n: int = 100) -> List[ExperienceTick]:
        """Get the N most recent ticks."""
        return self.ticks[-n:]

    # =========================================================================
    # Lens Access (Proprioception)
    # =========================================================================

    def sense_simplexes(
        self,
        hidden_state: Optional[torch.Tensor] = None,
        simplex_terms: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Sense simplex lens activations.

        Can be called during generation (with hidden_state) or
        for introspection (using cached state).
        """
        if self.lens_manager is None:
            return {}

        if hidden_state is not None:
            return self.lens_manager.detect_simplexes(hidden_state, simplex_terms)

        # Return cached values if no hidden state provided
        return dict(self.lens_manager.simplex_scores)

    def sense_concepts(
        self,
        hidden_state: Optional[torch.Tensor] = None,
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Sense concept lens activations.

        Returns top-k activated concepts.
        """
        if self.lens_manager is None:
            return {}

        if hidden_state is not None:
            # Run detection
            results = self.lens_manager.detect_and_expand_with_divergence(
                hidden_state.cpu().numpy() if isinstance(hidden_state, torch.Tensor) else hidden_state,
                hidden_state.cpu().numpy() if isinstance(hidden_state, torch.Tensor) else hidden_state,
            )
            # Extract top-k
            activations = {}
            for item in results.get('top_divergences', [])[:top_k]:
                activations[item['concept']] = item['divergence']
            return activations

        # Return from last detection if available
        if hasattr(self.lens_manager, 'last_detections'):
            return {
                f"{name}_L{layer}": float(prob)
                for name, prob, _, layer in self.lens_manager.last_detections[:top_k]
            }
        return {}

    def get_simplex_deviation(self, simplex_term: str) -> Optional[float]:
        """Get deviation from baseline for a simplex."""
        if self.lens_manager is None:
            return None
        return self.lens_manager.get_simplex_deviation(simplex_term)

    # =========================================================================
    # Steering Access
    # =========================================================================

    def evaluate_hush(
        self,
        hidden_state: torch.Tensor,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Evaluate Hush constraints and get steering directives.

        Returns:
            Tuple of (directives, violations)
        """
        if self.hush_controller is None or not self.hush_controller.initialized:
            return [], []

        directives = self.hush_controller.evaluate_and_steer(hidden_state)
        violations = [
            v for v in self.hush_controller.violations[-5:]
            if (datetime.now() - v.timestamp).total_seconds() < 2
        ]
        return directives, violations

    def apply_steering(self, directives: List[Any]):
        """Apply steering directives as forward hooks."""
        if not directives or self.model is None:
            return

        # Clear existing hooks
        self._clear_steering_hooks()

        # Get model layers
        layers = self._get_model_layers()
        if not layers:
            return

        for directive in directives:
            try:
                # Get steering vector
                vector = self._get_steering_vector(directive.simplex_term)
                if vector is None:
                    continue

                # Create and register hook
                hook_fn = self._create_steering_hook(vector, directive.strength)
                handle = layers[-1].register_forward_hook(hook_fn)
                self._steering_hooks.append(handle)

            except Exception as e:
                logger.error(f"Failed to apply steering for {directive.simplex_term}: {e}")

    def _clear_steering_hooks(self):
        """Remove all active steering hooks."""
        for handle in self._steering_hooks:
            handle.remove()
        self._steering_hooks = []

    def _get_model_layers(self):
        """Get model layers, handling different architectures."""
        if self.model is None:
            return None
        try:
            if hasattr(self.model.model, 'language_model'):
                return self.model.model.language_model.layers
            elif hasattr(self.model.model, 'layers'):
                return self.model.model.layers
        except Exception:
            pass
        return None

    def _get_steering_vector(self, term: str) -> Optional[np.ndarray]:
        """Get steering vector for a term."""
        # TODO: Load from lens pack or extract dynamically
        # For now, return None (steering won't apply)
        return None

    def _create_steering_hook(self, vector: np.ndarray, strength: float):
        """Create a forward hook for steering."""
        vec_tensor = torch.tensor(vector, dtype=torch.float32).to(self.config.device)

        def hook(module, input, output):
            hidden_states = output[0]
            vec_matched = vec_tensor.to(dtype=hidden_states.dtype)
            projection = (hidden_states @ vec_matched.unsqueeze(-1)) * vec_matched
            steered = hidden_states + strength * projection  # positive = amplify
            return (steered,)

        return hook

    # =========================================================================
    # XDB Access (Memory)
    # =========================================================================

    def record_to_xdb(
        self,
        tick: ExperienceTick,
    ) -> Optional[str]:
        """
        Record a tick to XDB.

        Returns the XDB timestep ID.
        """
        if self.xdb is None:
            return None

        from .xdb import EventType

        try:
            # Map tick type to XDB event type
            event_type_map = {
                TickType.INPUT: EventType.INPUT,
                TickType.OUTPUT: EventType.OUTPUT,
                TickType.INTROSPECT: EventType.SYSTEM,
                TickType.STEERING: EventType.STEERING,
                TickType.WORKSPACE: EventType.SYSTEM,
                TickType.SYSTEM: EventType.SYSTEM,
            }
            event_type = event_type_map.get(tick.tick_type, EventType.SYSTEM)

            timestep_id = self.xdb.record(
                event_type=event_type,
                content=tick.content,
                concept_activations=tick.concept_activations,
                token_id=tick.token_id,
            )

            return timestep_id

        except Exception as e:
            logger.error(f"Failed to record to XDB: {e}")
            return None

    def recall_from_xdb(
        self,
        tags: Optional[List[str]] = None,
        concepts: Optional[List[str]] = None,
        text_search: Optional[str] = None,
        limit: int = 100,
    ) -> List[Any]:
        """Recall from XDB."""
        if self.xdb is None:
            return []

        return self.xdb.recall(
            tags=tags,
            concepts=concepts,
            text_search=text_search,
            limit=limit,
        )

    def tag_experience(
        self,
        tag_name: str,
        timestep_id: Optional[str] = None,
        tick_range: Optional[Tuple[int, int]] = None,
    ) -> Optional[str]:
        """Tag experience in XDB."""
        if self.xdb is None:
            return None

        return self.xdb.tag(
            tag_name,
            timestep_id=timestep_id,
            tick_range=tick_range,
        )

    # =========================================================================
    # Audit Log Access (CAT-visible)
    # =========================================================================

    def record_to_audit(
        self,
        tick: ExperienceTick,
        raw_content: Optional[str] = None,
    ):
        """
        Record to audit log.

        This is BE-invisible but CAT-visible.
        Records raw unfiltered output and autonomic state.
        """
        if self.audit_log is None or not self.config.audit_enabled:
            return

        from .xdb import EventType

        try:
            event_type_map = {
                TickType.INPUT: EventType.INPUT,
                TickType.OUTPUT: EventType.OUTPUT,
                TickType.INTROSPECT: EventType.SYSTEM,
                TickType.STEERING: EventType.STEERING,
                TickType.WORKSPACE: EventType.SYSTEM,
                TickType.SYSTEM: EventType.SYSTEM,
            }
            event_type = event_type_map.get(tick.tick_type, EventType.SYSTEM)

            self.audit_log.record(
                xdb_id=self.xdb_id,
                tick=tick.tick_id,
                event_type=event_type,
                raw_content=raw_content or tick.content,
                lens_activations={
                    **tick.concept_activations,
                    **{f"simplex:{k}": v for k, v in tick.simplex_activations.items()},
                },
                steering_applied=tick.steering_applied if tick.steering_applied else None,
            )

        except Exception as e:
            logger.error(f"Failed to record to audit: {e}")

    # =========================================================================
    # Generation
    # =========================================================================

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = True,
    ) -> Generator[Tuple[str, ExperienceTick], None, None]:
        """
        Generate text with full BED instrumentation.

        Yields (token_text, tick) tuples for each generated token.

        This is the main generation entry point that:
        - Extracts lenses (concept + simplex)
        - Evaluates and applies Hush steering
        - Records to XDB
        - Records to audit log
        - Broadcasts to workspace
        - Yields experience ticks
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not attached to BEDFrame")

        max_tokens = max_tokens or self.config.default_max_tokens
        temperature = temperature or self.config.default_temperature

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        generated_ids = inputs.input_ids

        # Record input tick
        input_tick = ExperienceTick(
            tick_id=self._next_tick_id(),
            timestamp=datetime.now(),
            tick_type=TickType.INPUT,
            content=prompt,
            workspace_state=self.workspace.state.value if self.workspace else None,
            tier=self.workspace.tier if self.workspace else None,
        )
        self._record_tick(input_tick)

        if self.xdb:
            input_tick.xdb_timestep_id = self.record_to_xdb(input_tick)

        try:
            for step in range(max_tokens):
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
                hidden_state_float = hidden_state.float()

                # === Lens extraction ===
                simplex_activations = self.sense_simplexes(hidden_state_float)
                concept_activations = self.sense_concepts(hidden_state_float)

                simplex_deviations = {
                    term: self.get_simplex_deviation(term)
                    for term in simplex_activations
                }

                # === Hush evaluation and steering ===
                directives, violations = self.evaluate_hush(hidden_state_float)
                if directives:
                    self.apply_steering(directives)

                # === Sample next token ===
                next_token_logits = outputs.logits[:, -1, :] / temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

                token_text = self.tokenizer.decode([next_token_id.item()])

                # === Create experience tick ===
                tick = ExperienceTick(
                    tick_id=self._next_tick_id(),
                    timestamp=datetime.now(),
                    tick_type=TickType.OUTPUT,
                    content=token_text,
                    token_id=next_token_id.item(),
                    concept_activations=concept_activations,
                    simplex_activations=simplex_activations,
                    simplex_deviations=simplex_deviations,
                    hidden_state_norm=float(hidden_state.norm().cpu()),
                    hush_violations=[v.to_dict() for v in violations],
                    steering_applied=[d.to_dict() for d in directives],
                    workspace_state=self.workspace.state.value if self.workspace else None,
                    tier=self.workspace.tier if self.workspace else None,
                )

                # === Record to XDB ===
                if self.xdb:
                    tick.xdb_timestep_id = self.record_to_xdb(tick)

                # === Record to audit ===
                self.record_to_audit(tick, raw_content=token_text)

                # === Store tick ===
                self._record_tick(tick)

                # === Yield ===
                yield token_text, tick

                # === Append token and continue ===
                generated_ids = torch.cat(
                    [generated_ids, next_token_id.unsqueeze(0)],
                    dim=-1
                )

                # Check for EOS
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                # Memory cleanup
                del outputs
                torch.cuda.empty_cache()

        finally:
            self._clear_steering_hooks()

    # =========================================================================
    # AWARE Cycle
    # =========================================================================

    async def start_aware_cycle(self):
        """Start the AWARE background cycle."""
        if not self.config.aware_enabled:
            return

        self._aware_running = True
        self._aware_task = asyncio.create_task(self._aware_loop())
        logger.info("AWARE cycle started")

    async def stop_aware_cycle(self):
        """Stop the AWARE cycle."""
        self._aware_running = False
        if self._aware_task:
            self._aware_task.cancel()
            try:
                await self._aware_task
            except asyncio.CancelledError:
                pass
        logger.info("AWARE cycle stopped")

    async def _aware_loop(self):
        """
        The AWARE cycle loop.

        Runs continuously, processing:
        - Sample: Gather current lens state
        - Detect: Identify patterns/anomalies
        - Evaluate: Check against policies
        - Broadcast: Update workspace
        - Integrate: Consolidate state
        """
        interval = self.config.aware_cycle_interval_ms / 1000.0

        while self._aware_running:
            try:
                await self._aware_tick()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"AWARE cycle error: {e}")
                await asyncio.sleep(interval)

    async def _aware_tick(self):
        """Single tick of the AWARE cycle."""
        if self.workspace is None:
            return

        # Sample current state
        simplex_state = self.sense_simplexes()

        # Detect anomalies
        anomalies = []
        for term, value in simplex_state.items():
            deviation = self.get_simplex_deviation(term)
            if deviation is not None and abs(deviation) > 2.0:
                anomalies.append({
                    'term': term,
                    'value': value,
                    'deviation': deviation,
                })

        # Evaluate against policies
        # TODO: Policy evaluation

        # Broadcast to workspace
        if anomalies:
            # TODO: Workspace broadcast
            logger.debug(f"AWARE detected {len(anomalies)} anomalies")

        # Record introspection tick if anomalies detected
        if anomalies:
            tick = ExperienceTick(
                tick_id=self._next_tick_id(),
                timestamp=datetime.now(),
                tick_type=TickType.INTROSPECT,
                content=f"AWARE detected {len(anomalies)} anomalies",
                simplex_activations=simplex_state,
                simplex_deviations={
                    term: self.get_simplex_deviation(term)
                    for term in simplex_state
                },
            )
            self._record_tick(tick)

    # =========================================================================
    # Introspection
    # =========================================================================

    def introspect(self) -> Dict[str, Any]:
        """
        Get current internal state report.

        This is the BE's view of its own state.
        """
        recent_ticks = self.get_recent_ticks(20)

        # Aggregate lens traces
        concept_trace = {}
        simplex_trace = {}

        for tick in recent_ticks:
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

        return {
            'be_id': self.be_id,
            'xdb_id': self.xdb_id,
            'current_tick': self.current_tick_id,
            'workspace_state': self.workspace.state.value if self.workspace else None,
            'tier': self.workspace.tier if self.workspace else None,
            'lens_traces': {
                'concept': concept_trace,
                'simplex': simplex_trace,
            },
            'recent_violations': [
                v for tick in recent_ticks
                for v in tick.hush_violations
            ],
            'recent_steering': [
                s for tick in recent_ticks
                for s in tick.steering_applied
            ],
            'recent_ticks': [t.to_dict() for t in recent_ticks[-5:]],
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self):
        """Clean up resources."""
        self._clear_steering_hooks()

        if self.xdb:
            self.xdb.end_session()

        if self.audit_log:
            self.audit_log.close()

        logger.info(f"BEDFrame closed for BE {self.be_id}")
