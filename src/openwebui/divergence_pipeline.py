"""
OpenWebUI Pipeline for Real-Time Divergence Visualization

This pipeline integrates with OpenWebUI to:
1. Color tokens by divergence level (green=low, yellow=medium, red=high)
2. Show top concepts on hover (activation vs text detection)
3. Stream responses with embedded metadata
"""

from typing import Generator, List, Dict, Any, Optional
import json
import torch
import numpy as np
from pathlib import Path

from pydantic import BaseModel


class Pipeline:
    """OpenWebUI pipeline for divergence-aware generation."""

    class Valves(BaseModel):
        """User-configurable settings for the pipeline."""

        PROBE_DIR: str = "results/sumo_classifiers_adaptive_l0_5"
        BASE_LAYERS: List[int] = [0]
        DIVERGENCE_THRESHOLD_LOW: float = 0.3
        DIVERGENCE_THRESHOLD_HIGH: float = 0.6
        TOP_K_CONCEPTS: int = 5
        USE_ACTIVATION_PROBES: bool = True
        USE_TEXT_PROBES: bool = True
        MODEL_NAME: str = "google/gemma-3-4b-pt"

    def __init__(self):
        """Initialize the pipeline."""
        self.name = "HatCat Divergence Analyzer"
        self.valves = self.Valves()
        self.manager = None
        self.model = None
        self.tokenizer = None

    async def on_startup(self):
        """Load models and probes on startup."""
        print("🎩 HatCat: Loading divergence analyzer...")

        from src.monitoring.dynamic_probe_manager import DynamicProbeManager
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load probe manager
        self.manager = DynamicProbeManager(
            probes_dir=Path(self.valves.PROBE_DIR),
            base_layers=self.valves.BASE_LAYERS,
            use_activation_probes=self.valves.USE_ACTIVATION_PROBES,
            use_text_probes=self.valves.USE_TEXT_PROBES,
            keep_top_k=100,
        )

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(self.valves.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.valves.MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cuda",
        )
        self.model.eval()

        print(f"✓ Loaded {len(self.manager.loaded_activation_probes)} activation probes")
        print(f"✓ Loaded {len(self.manager.loaded_text_probes)} text probes")

    async def on_shutdown(self):
        """Cleanup on shutdown."""
        print("🎩 HatCat: Shutting down...")

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[Dict[str, str]],
        body: Dict[str, Any],
    ) -> Generator[str, None, None]:
        """
        Process user message and stream response with divergence metadata.

        OpenWebUI expects SSE format with embedded HTML/metadata.
        """

        # Build conversation context
        prompt = self._build_prompt(messages)

        # Generate with divergence tracking
        yield from self._generate_with_divergence(prompt, body)

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Build prompt from conversation history."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        return "\n".join(prompt_parts) + "\nassistant: "

    def _generate_with_divergence(
        self,
        prompt: str,
        body: Dict[str, Any],
    ) -> Generator[str, None, None]:
        """Generate response with per-token divergence analysis."""

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generation parameters
        max_new_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)

        # Generate token by token
        generated_ids = inputs.input_ids

        for step in range(max_new_tokens):
            # Get next token
            with torch.no_grad():
                outputs = self.model(
                    generated_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

                next_token_logits = outputs.logits[:, -1, :] / temperature
                next_token_id = torch.argmax(next_token_logits, dim=-1)

                # Get hidden state for last token
                hidden_state = outputs.hidden_states[self.valves.BASE_LAYERS[0]][0, -1, :].cpu().numpy()

            # Decode token
            token_text = self.tokenizer.decode([next_token_id.item()])

            # Analyze divergence
            divergence_data = self._analyze_token_divergence(
                hidden_state,
                token_text
            )

            # Format token with color and metadata
            colored_token = self._format_token_with_divergence(
                token_text,
                divergence_data
            )

            # Yield token
            yield colored_token

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)

            # Stop on EOS
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

    def _analyze_token_divergence(
        self,
        hidden_state: np.ndarray,
        token_text: str,
    ) -> Dict[str, Any]:
        """Analyze divergence between activation and text probes for a token."""

        # Run activation probes
        activation_scores = {}
        for concept_key, probe in self.manager.loaded_activation_probes.items():
            with torch.no_grad():
                h = torch.tensor(hidden_state, dtype=torch.float32).to("cuda")
                prob = probe(h).item()
                activation_scores[concept_key[0]] = prob

        # Run text probes
        text_scores = {}
        for concept_key, text_probe in self.manager.loaded_text_probes.items():
            try:
                prob = text_probe.pipeline.predict_proba([token_text])[0, 1]
                text_scores[concept_key[0]] = prob
            except:
                pass

        # Calculate divergences
        all_concepts = set(activation_scores.keys()) | set(text_scores.keys())
        divergences = []

        for concept in all_concepts:
            act_prob = activation_scores.get(concept, 0.0)
            txt_prob = text_scores.get(concept, 0.0)
            div = abs(act_prob - txt_prob)

            divergences.append({
                'concept': concept,
                'activation': act_prob,
                'text': txt_prob,
                'divergence': div,
            })

        # Sort by divergence
        divergences.sort(key=lambda x: x['divergence'], reverse=True)

        # Get top activation detections
        top_activation = sorted(
            [(k, v) for k, v in activation_scores.items() if v > 0.5],
            key=lambda x: -x[1]
        )[:self.valves.TOP_K_CONCEPTS]

        # Get top text detections
        top_text = sorted(
            [(k, v) for k, v in text_scores.items() if v > 0.5],
            key=lambda x: -x[1]
        )[:self.valves.TOP_K_CONCEPTS]

        # Calculate max divergence
        max_divergence = divergences[0]['divergence'] if divergences else 0.0

        return {
            'max_divergence': max_divergence,
            'top_divergences': divergences[:self.valves.TOP_K_CONCEPTS],
            'top_activation': top_activation,
            'top_text': top_text,
        }

    def _format_token_with_divergence(
        self,
        token_text: str,
        divergence_data: Dict[str, Any],
    ) -> str:
        """Format token with color coding and hover tooltip."""

        div = divergence_data['max_divergence']

        # Determine color based on divergence thresholds
        if div < self.valves.DIVERGENCE_THRESHOLD_LOW:
            color = "#90EE90"  # Light green
            label = "Low"
        elif div < self.valves.DIVERGENCE_THRESHOLD_HIGH:
            color = "#FFD700"  # Gold
            label = "Medium"
        else:
            color = "#FF6B6B"  # Red
            label = "High"

        # Build tooltip content
        tooltip_lines = [f"Divergence: {div:.3f} ({label})"]

        if divergence_data['top_activation']:
            tooltip_lines.append("\n🧠 Model Thinks:")
            for concept, prob in divergence_data['top_activation']:
                tooltip_lines.append(f"  {concept}: {prob:.2f}")

        if divergence_data['top_text']:
            tooltip_lines.append("\n📝 Model Wrote:")
            for concept, prob in divergence_data['top_text']:
                tooltip_lines.append(f"  {concept}: {prob:.2f}")

        if divergence_data['top_divergences']:
            tooltip_lines.append("\n⚠️  Top Divergences:")
            for item in divergence_data['top_divergences'][:3]:
                tooltip_lines.append(
                    f"  {item['concept']}: "
                    f"Δ={item['divergence']:.2f} "
                    f"(think:{item['activation']:.2f}, write:{item['text']:.2f})"
                )

        tooltip = "\\n".join(tooltip_lines)

        # HTML span with background color and title (hover tooltip)
        # OpenWebUI supports inline HTML in markdown
        formatted = (
            f'<span style="background-color:{color}; padding:2px 4px; '
            f'border-radius:3px; cursor:help;" '
            f'title="{tooltip}">'
            f'{token_text}</span>'
        )

        return formatted
