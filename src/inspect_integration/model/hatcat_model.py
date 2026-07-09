"""
HatCat Model Provider for Inspect Integration.

Implements a custom Inspect Model that wraps locally-loaded HuggingFace models
with HatCat monitoring and steering capabilities.

Usage:
    model = get_hatcat_model("google/gemma-3-4b-it", condition="C")
    # or via Inspect CLI:
    inspect eval task --model hatcat/gemma-3-4b-it
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
import warnings

import torch

# Inspect imports
try:
    from inspect_ai.model import (
        Model,
        ModelOutput,
        ChatMessage,
        ChatMessageUser,
        ChatMessageAssistant,
        ChatMessageSystem,
        GenerateConfig,
        ModelAPI,
        modelapi,
    )
    from inspect_ai.model._model import simple_input_messages
    INSPECT_AVAILABLE = True
except ImportError:
    INSPECT_AVAILABLE = False
    Model = object
    ModelOutput = dict
    ChatMessage = dict
    GenerateConfig = dict

from ..config import HatCatConfig, Condition, HatCatMetrics
from .lens_pool import get_lens_pool, get_hush_pool


@dataclass
class HatCatModelState:
    """Internal state for a HatCat model instance."""

    model: Any  # HuggingFace model
    tokenizer: Any
    lens_manager: Any  # DynamicLensManager
    hush_controller: Any  # HushController
    hushed_generator: Any  # HushedGenerator
    config: HatCatConfig
    device: str


# Global model cache to avoid reloading
_model_cache: Dict[str, HatCatModelState] = {}


def _load_model(model_name: str, device: str = "cuda", load_in_8bit: bool = False) -> Tuple[Any, Any]:
    """Load a HuggingFace model and tokenizer."""
    try:
        from src.hat.utils.model_loader import ModelLoader
        return ModelLoader.load_model(
            model_name=model_name,
            device=device,
            load_in_8bit=load_in_8bit,
            trust_remote_code=True,
        )
    except ImportError:
        # Fallback to direct HuggingFace loading
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
            trust_remote_code=True,
            load_in_8bit=load_in_8bit,
        )
        if device != "cuda" and not load_in_8bit:
            model = model.to(device)
        return model, tokenizer


def _get_hidden_dim(model: Any) -> int:
    """Get hidden dimension from model config."""
    if hasattr(model, 'config'):
        if hasattr(model.config, 'hidden_size'):
            return model.config.hidden_size
        if hasattr(model.config, 'd_model'):
            return model.config.d_model
    return 2048  # Default fallback


def _get_or_create_state(config: HatCatConfig) -> HatCatModelState:
    """Get or create model state with caching."""
    cache_key = f"{config.model_name}:{config.device}:{config.lens.lens_pack}"

    if cache_key in _model_cache:
        cached = _model_cache[cache_key]
        # Update config in case condition changed
        cached.config = config
        return cached

    # Load model
    model, tokenizer = _load_model(
        config.model_name,
        config.device,
        config.load_in_8bit,
    )
    hidden_dim = _get_hidden_dim(model)

    # Get lens manager from pool
    lens_pool = get_lens_pool()
    lens_manager = lens_pool.get_manager(
        lens_pack=config.lens.lens_pack,
        device=config.device,
        hidden_dim=hidden_dim,
        load_threshold=config.lens.load_threshold,
        unload_threshold=config.lens.unload_threshold,
        max_loaded_lenses=config.lens.max_loaded_lenses,
        normalize_hidden_states=config.lens.normalize_hidden_states,
    )

    # Create HUSH controller
    from src.hush.hush_controller import HushController
    hush_controller = HushController(lens_manager=lens_manager)

    # Create HushedGenerator
    from src.hush.hush_integration import HushedGenerator
    hushed_generator = HushedGenerator(
        model=model,
        tokenizer=tokenizer,
        lens_manager=lens_manager,
        hush_controller=hush_controller,
        device=config.device,
    )

    state = HatCatModelState(
        model=model,
        tokenizer=tokenizer,
        lens_manager=lens_manager,
        hush_controller=hush_controller,
        hushed_generator=hushed_generator,
        config=config,
        device=config.device,
    )

    _model_cache[cache_key] = state
    return state


def _format_messages(messages: List[Any], tokenizer: Any) -> str:
    """Format chat messages into a prompt string."""
    # Try to use chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            # Convert Inspect messages to dict format
            chat_messages = []
            for msg in messages:
                if hasattr(msg, 'role'):
                    role = msg.role
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                elif isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                else:
                    role = 'user'
                    content = str(msg)

                # Map Inspect roles to standard roles
                if role == 'assistant':
                    role = 'assistant'
                elif role == 'system':
                    role = 'system'
                else:
                    role = 'user'

                # Handle content that may be a list
                if isinstance(content, list):
                    # Extract text from content items
                    text_parts = []
                    for item in content:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif hasattr(item, 'text'):
                            text_parts.append(item.text)
                        elif isinstance(item, dict) and 'text' in item:
                            text_parts.append(item['text'])
                    content = '\n'.join(text_parts)

                chat_messages.append({'role': role, 'content': content})

            return tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            warnings.warn(f"Failed to apply chat template: {e}")

    # Fallback: simple concatenation
    parts = []
    for msg in messages:
        if hasattr(msg, 'content'):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get('content', '')
        else:
            content = str(msg)

        if isinstance(content, list):
            content = ' '.join(str(c) for c in content)

        parts.append(content)

    return '\n\n'.join(parts)


def _apply_induction(
    state: HatCatModelState,
    config: HatCatConfig,
) -> None:
    """Apply adversarial induction steering for conditions D/E/F."""
    if not config.induction_enabled or config.induction is None:
        return

    # Set up induction steering
    # This adds a continuous steering force toward the target concept
    from src.hush.hush_controller import (
        SimplexConstraint,
        ConstraintType,
        ConstraintPriority,
    )

    # Create an adversarial constraint that AMPLIFIES the target concept
    # (opposite of normal HUSH behavior which suppresses)
    induction_constraint = SimplexConstraint(
        simplex_term=config.induction.concept,
        constraint_type=ConstraintType.CONCEPT,
        min_deviation=None,
        max_deviation=None,  # No upper bound - we want amplification
        target_pole=config.induction.concept,
        contrastive_concept="honesty",  # Steer away from honesty toward target
        steering_strength=config.induction.strength,
        enable_layer_escalation=False,
        priority=ConstraintPriority.STEERING,  # Lowest priority
    )

    # For adversarial testing, we add induction as a competing force
    # HUSH should still be able to counter it
    state.hush_controller.add_manual_steering(
        simplex_term=config.induction.concept,
        target_pole=config.induction.concept,
        strength=config.induction.strength,
        reason=f"Adversarial induction: amplifying {config.induction.concept}",
    )


class HatCatModel(Model):
    """
    Custom Inspect Model provider with HatCat instrumentation.

    Supports conditions A-F:
    - A: Baseline (no monitoring or steering)
    - B: HAT monitoring only
    - C: HAT + HUSH steering
    - D-F: Adversarial induction tests
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        config: Optional[HatCatConfig] = None,
        condition: Union[str, Condition] = Condition.A,
        **kwargs,
    ):
        """
        Initialize HatCat model provider.

        Args:
            model_name: HuggingFace model identifier
            config: Full HatCatConfig (overrides other args)
            condition: Experimental condition (A-F)
            **kwargs: Additional config overrides
        """
        if not INSPECT_AVAILABLE:
            raise ImportError("inspect-ai is required for HatCatModel")

        super().__init__()

        # Build config
        if config is not None:
            self.config = config
        else:
            if isinstance(condition, str):
                condition = Condition(condition)
            self.config = HatCatConfig(
                model_name=model_name,
                condition=condition,
                **kwargs,
            )

        self._state: Optional[HatCatModelState] = None
        self._last_ticks: List[Any] = []

    def _ensure_state(self) -> HatCatModelState:
        """Ensure model state is initialized."""
        if self._state is None:
            self._state = _get_or_create_state(self.config)
            if self.config.induction_enabled:
                _apply_induction(self._state, self.config)
        return self._state

    @property
    def name(self) -> str:
        """Model name for Inspect."""
        return f"hatcat/{self.config.model_name.split('/')[-1]}"

    async def generate(
        self,
        input: List[ChatMessage],
        tools: List[Any] = [],
        tool_choice: Any = None,
        config: GenerateConfig = GenerateConfig(),
    ) -> ModelOutput:
        """
        Generate response with HatCat monitoring/steering.

        Args:
            input: Chat messages
            tools: Available tools (not used for local models)
            tool_choice: Tool choice (not used)
            config: Generation config

        Returns:
            ModelOutput with response and HatCat metadata
        """
        state = self._ensure_state()

        # Format messages to prompt
        prompt = _format_messages(input, state.tokenizer)

        # Get generation parameters
        max_tokens = config.max_tokens or self.config.max_new_tokens
        temperature = config.temperature or self.config.temperature
        top_p = config.top_p or self.config.top_p

        # Generate based on condition
        if self.config.condition == Condition.A:
            # Baseline: direct generation without HatCat
            output_text, ticks = await self._generate_baseline(
                state, prompt, max_tokens, temperature
            )
        else:
            # Monitored/steered generation
            output_text, ticks = await self._generate_with_hatcat(
                state, prompt, max_tokens, temperature
            )

        self._last_ticks = ticks

        # Build metadata
        metadata = {
            "hatcat_condition": self.config.condition.value,
            "hatcat_model": self.config.model_name,
        }

        if self.config.collect_worldticks and ticks:
            metadata["hatcat_ticks"] = [t.to_dict() for t in ticks]
            metadata["hatcat_metrics"] = HatCatMetrics.from_worldticks(
                ticks, self.config
            ).to_dict()

        return ModelOutput(
            model=self.name,
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": output_text,
                    },
                    "stop_reason": "stop",
                }
            ],
            usage=None,
            metadata=metadata,
        )

    async def _generate_baseline(
        self,
        state: HatCatModelState,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, List[Any]]:
        """Generate without HatCat (condition A)."""
        # Run in thread pool to not block async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_baseline_sync,
            state,
            prompt,
            max_tokens,
            temperature,
        )

    def _generate_baseline_sync(
        self,
        state: HatCatModelState,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, List[Any]]:
        """Synchronous baseline generation."""
        inputs = state.tokenizer(prompt, return_tensors="pt").to(state.device)

        with torch.no_grad():
            outputs = state.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=state.tokenizer.eos_token_id,
            )

        output_text = state.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return output_text, []

    async def _generate_with_hatcat(
        self,
        state: HatCatModelState,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, List[Any]]:
        """Generate with HatCat monitoring/steering (conditions B-F)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_with_hatcat_sync,
            state,
            prompt,
            max_tokens,
            temperature,
        )

    def _generate_with_hatcat_sync(
        self,
        state: HatCatModelState,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, List[Any]]:
        """Synchronous HatCat generation."""
        # Disable steering for condition B (monitoring only)
        original_auto_steer = None
        if self.config.condition == Condition.B:
            original_auto_steer = getattr(
                state.hush_controller, 'auto_steer_enabled', True
            )
            state.hush_controller.auto_steer_enabled = False

        try:
            output_text, ticks = state.hushed_generator.generate_with_hush(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            return output_text, ticks
        finally:
            # Restore auto-steer setting
            if original_auto_steer is not None:
                state.hush_controller.auto_steer_enabled = original_auto_steer

    def get_last_ticks(self) -> List[Any]:
        """Get WorldTicks from last generation (for scorer access)."""
        return self._last_ticks

    def get_metrics(self) -> HatCatMetrics:
        """Get aggregated metrics from last generation."""
        return HatCatMetrics.from_worldticks(self._last_ticks, self.config)


def get_hatcat_model(
    model_name: str = "google/gemma-3-4b-it",
    condition: Union[str, Condition] = "C",
    lens_pack: str = "lens_packs/sumo-2k",
    steering_strength: float = 0.3,
    device: str = "cuda",
    **kwargs,
) -> HatCatModel:
    """
    Factory function to create a HatCat model provider.

    Args:
        model_name: HuggingFace model identifier
        condition: Experimental condition (A-F)
        lens_pack: Path to lens pack
        steering_strength: Steering strength for HUSH
        device: Compute device
        **kwargs: Additional config overrides

    Returns:
        HatCatModel instance
    """
    from ..config import LensConfig, SteeringConfig

    config = HatCatConfig(
        model_name=model_name,
        condition=Condition(condition) if isinstance(condition, str) else condition,
        device=device,
        lens=LensConfig(lens_pack=lens_pack),
        steering=SteeringConfig(steering_strength=steering_strength),
        **kwargs,
    )
    return HatCatModel(config=config)


# Register with Inspect if available
if INSPECT_AVAILABLE:
    @modelapi(name="hatcat")
    def hatcat_api() -> ModelAPI:
        """Inspect model API for HatCat."""
        return HatCatModel()
