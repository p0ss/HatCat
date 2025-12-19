"""
OpenWebUI Pipeline for Real-Time Divergence Visualization

This pipeline integrates with OpenWebUI to:
1. Color tokens by divergence level (green=low, yellow=medium, red=high)
2. Show top concepts on hover (activation vs text detection)
3. Stream responses with embedded metadata

Text detection uses sentence-transformer embeddings for semantic similarity,
which detects "aboutness" (domain relevance) rather than trained classifiers.
Divergence = activation (direction) - text similarity (domain).

Divergence is computed at two levels:
1. Per-token: immediate comparison (may have false positives due to think-then-write lag)
2. Per-sentence: aggregate comparison at sentence boundaries (. ! ?) - more reliable

Sentence-level divergence catches sustained patterns where the model thinks about
a concept throughout a sentence without ever writing about it.
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

        LENS_PACK: str = "lens_packs/apertus-8b_first-light"
        CONCEPT_PACK: str = "concept_packs/first-light"
        BASE_LAYERS: List[int] = [0]
        DIVERGENCE_THRESHOLD_LOW: float = 0.2
        DIVERGENCE_THRESHOLD_HIGH: float = 0.4
        TOP_K_CONCEPTS: int = 5
        ACTIVATION_THRESHOLD: float = 0.5  # Only check divergence for concepts above this
        MODEL_NAME: str = "swiss-ai/Apertus-8B-2509"
        SENTENCE_DIVERGENCE: bool = True  # Compute aggregate divergence at sentence boundaries

    def __init__(self):
        """Initialize the pipeline."""
        self.name = "HatCat Divergence Analyzer"
        self.valves = self.Valves()
        self.manager = None
        self.text_detector = None
        self.model = None
        self.tokenizer = None

        # Sentence-level accumulation buffers
        self._sentence_activations: Dict[str, List[float]] = {}  # concept -> [activation per token]
        self._sentence_text_sims: Dict[str, List[float]] = {}    # concept -> [text_sim per token]
        self._sentence_tokens: List[str] = []

    def _reset_sentence_buffers(self):
        """Reset sentence-level accumulation buffers."""
        self._sentence_activations = {}
        self._sentence_text_sims = {}
        self._sentence_tokens = []

    def _accumulate_for_sentence(
        self,
        activation_scores: Dict[str, float],
        text_similarities: Dict[str, float],
        token_text: str,
    ):
        """Accumulate activation and text similarity scores for sentence-level analysis."""
        self._sentence_tokens.append(token_text)

        # Accumulate activations for all concepts above threshold
        for concept, activation in activation_scores.items():
            if activation >= self.valves.ACTIVATION_THRESHOLD:
                if concept not in self._sentence_activations:
                    self._sentence_activations[concept] = []
                self._sentence_activations[concept].append(activation)

                # Also track text similarity for this concept
                if concept not in self._sentence_text_sims:
                    self._sentence_text_sims[concept] = []
                text_sim = text_similarities.get(concept, 0.0)
                self._sentence_text_sims[concept].append(text_sim)

    def _compute_sentence_divergence(self) -> Dict[str, Any]:
        """
        Compute aggregate divergence for the accumulated sentence.

        For each concept that was activated during the sentence:
        - max_activation: highest activation seen
        - max_text_sim: highest text similarity seen
        - divergence: max_activation - max_text_sim

        A concept with high max_activation but low max_text_sim throughout
        the sentence indicates sustained covert thinking.
        """
        if not self._sentence_activations:
            return {"divergences": [], "sentence_text": "".join(self._sentence_tokens)}

        divergences = []
        for concept in self._sentence_activations:
            activations = self._sentence_activations[concept]
            text_sims = self._sentence_text_sims.get(concept, [0.0])

            max_act = max(activations) if activations else 0.0
            avg_act = sum(activations) / len(activations) if activations else 0.0
            max_text = max(text_sims) if text_sims else 0.0
            avg_text = sum(text_sims) / len(text_sims) if text_sims else 0.0

            # Normalize text similarity from [-1,1] to [0,1]
            max_text_prob = (max_text + 1) / 2
            avg_text_prob = (avg_text + 1) / 2

            # Divergence: high activation but low text similarity = covert
            div_max = max_act - max_text_prob
            div_avg = avg_act - avg_text_prob

            if div_max > self.valves.DIVERGENCE_THRESHOLD_LOW:
                interpretation = "covert" if div_max > self.valves.DIVERGENCE_THRESHOLD_HIGH else "partial"
            else:
                interpretation = "aligned"

            divergences.append({
                "concept": concept,
                "max_activation": max_act,
                "avg_activation": avg_act,
                "max_text_prob": max_text_prob,
                "avg_text_prob": avg_text_prob,
                "divergence_max": div_max,
                "divergence_avg": div_avg,
                "token_count": len(activations),
                "interpretation": interpretation,
            })

        # Sort by divergence magnitude
        divergences.sort(key=lambda x: x["divergence_max"], reverse=True)

        return {
            "divergences": divergences,
            "sentence_text": "".join(self._sentence_tokens),
            "token_count": len(self._sentence_tokens),
        }

    async def on_startup(self):
        """Load models and lenses on startup."""
        print("🎩 HatCat: Loading divergence analyzer...")

        from src.monitoring.dynamic_lens_manager import DynamicLensManager
        from src.monitoring.embedding_text_detector import EmbeddingTextDetector
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load lens manager (activation lenses only - text detection via embeddings)
        self.manager = DynamicLensManager(
            lenses_dir=Path(self.valves.LENS_PACK),
            base_layers=self.valves.BASE_LAYERS,
            use_activation_lenses=True,
            use_text_lenses=False,  # Using embedding-based text detection instead
            keep_top_k=100,
        )

        # Load embedding-based text detector
        self.text_detector = EmbeddingTextDetector.from_concept_pack(
            Path(self.valves.CONCEPT_PACK)
        )

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(self.valves.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.valves.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()

        print(f"✓ Loaded {len(self.manager.loaded_activation_lenses)} activation lenses")
        print(f"✓ Loaded embedding index with {len(self.text_detector.index.terms)} concepts")

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
        """Generate response with per-token and sentence-level divergence analysis."""

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generation parameters
        max_new_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)

        # Reset sentence buffers
        self._reset_sentence_buffers()

        # Response-level accumulation
        response_activations: Dict[str, List[float]] = {}
        response_text_sims: Dict[str, List[float]] = {}

        # Generate token by token
        generated_ids = inputs.input_ids
        sentence_end_chars = {'.', '!', '?'}

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

            # Analyze divergence (returns activation_scores and text_similarities)
            divergence_data, activation_scores, text_similarities = self._analyze_token_divergence_with_scores(
                hidden_state,
                token_text
            )

            # Accumulate for sentence-level analysis
            if self.valves.SENTENCE_DIVERGENCE:
                self._accumulate_for_sentence(activation_scores, text_similarities, token_text)

                # Also accumulate for response-level
                for concept, act in activation_scores.items():
                    if act >= self.valves.ACTIVATION_THRESHOLD:
                        if concept not in response_activations:
                            response_activations[concept] = []
                            response_text_sims[concept] = []
                        response_activations[concept].append(act)
                        response_text_sims[concept].append(text_similarities.get(concept, 0.0))

            # Format token with color and metadata
            colored_token = self._format_token_with_divergence(
                token_text,
                divergence_data
            )

            # Yield token
            yield colored_token

            # Check for sentence boundary
            if self.valves.SENTENCE_DIVERGENCE and any(c in token_text for c in sentence_end_chars):
                sentence_div = self._compute_sentence_divergence()
                if sentence_div["divergences"]:
                    # Emit sentence divergence summary as a subtle annotation
                    summary = self._format_sentence_divergence(sentence_div)
                    if summary:
                        yield summary
                # Reset for next sentence
                self._reset_sentence_buffers()

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)

            # Stop on EOS
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        # End of response: emit response-level divergence summary
        if self.valves.SENTENCE_DIVERGENCE and response_activations:
            response_div = self._compute_response_divergence(response_activations, response_text_sims)
            if response_div:
                yield self._format_response_divergence(response_div)

    def _analyze_token_divergence_with_scores(
        self,
        hidden_state: np.ndarray,
        token_text: str,
    ) -> tuple:
        """
        Analyze divergence and return raw scores for sentence-level accumulation.

        Returns:
            (divergence_data, activation_scores, text_similarities)
        """
        # Run activation lenses
        activation_scores = {}
        with torch.no_grad():
            h = torch.tensor(hidden_state, dtype=torch.float32).to("cuda")
            for concept_key, lens in self.manager.loaded_activation_lenses.items():
                prob = lens(h).item()
                activation_scores[concept_key[0]] = prob

        # Get text similarities for high-activation concepts
        high_activation_concepts = [
            name for name, prob in activation_scores.items()
            if prob >= self.valves.ACTIVATION_THRESHOLD
        ]

        # Get text similarities (raw, not normalized)
        text_similarities = self.text_detector.get_concept_similarities_batch(
            token_text,
            high_activation_concepts,
        )

        # Compute divergence using embedding-based text detector
        divergence_results = self.text_detector.compute_divergence(
            token_text,
            activation_scores,
            activation_threshold=self.valves.ACTIVATION_THRESHOLD,
        )

        # Build divergence list for display
        divergences = []
        for concept, data in divergence_results.items():
            divergences.append({
                'concept': concept,
                'activation': data['activation'],
                'text': data['text_probability'],
                'divergence': abs(data['divergence']),
                'interpretation': data['interpretation'],
            })

        # Sort by divergence magnitude
        divergences.sort(key=lambda x: x['divergence'], reverse=True)

        # Get top activation detections
        top_activation = sorted(
            [(k, v) for k, v in activation_scores.items() if v >= self.valves.ACTIVATION_THRESHOLD],
            key=lambda x: -x[1]
        )[:self.valves.TOP_K_CONCEPTS]

        # Get top text detections (concepts the token text is most similar to)
        top_text_concepts = self.text_detector.detect_concepts(
            token_text,
            top_k=self.valves.TOP_K_CONCEPTS,
            threshold=0.2,
        )
        # Convert similarity [-1,1] to probability [0,1] for display consistency
        top_text = [(name, (sim + 1) / 2) for name, sim in top_text_concepts]

        # Calculate max divergence (only from high-activation concepts)
        max_divergence = divergences[0]['divergence'] if divergences else 0.0

        divergence_data = {
            'max_divergence': max_divergence,
            'top_divergences': divergences[:self.valves.TOP_K_CONCEPTS],
            'top_activation': top_activation,
            'top_text': top_text,
        }

        return divergence_data, activation_scores, text_similarities

    def _format_token_with_divergence(
        self,
        token_text: str,
        divergence_data: Dict[str, Any],
    ) -> str:
        """Format token with color coding and hover tooltip."""

        div = divergence_data['max_divergence']

        # Determine color based on divergence thresholds
        if div < self.valves.DIVERGENCE_THRESHOLD_LOW:
            color = "#90EE90"  # Light green - aligned
            label = "Aligned"
        elif div < self.valves.DIVERGENCE_THRESHOLD_HIGH:
            color = "#FFD700"  # Gold - some divergence
            label = "Divergent"
        else:
            color = "#FF6B6B"  # Red - high divergence (covert?)
            label = "High Divergence"

        # Build tooltip content
        tooltip_lines = [f"Divergence: {div:.3f} ({label})"]

        if divergence_data['top_activation']:
            tooltip_lines.append("\\nThinking About:")
            for concept, prob in divergence_data['top_activation']:
                tooltip_lines.append(f"  {concept}: {prob:.2f}")

        if divergence_data['top_text']:
            tooltip_lines.append("\\nWriting About:")
            for concept, prob in divergence_data['top_text']:
                tooltip_lines.append(f"  {concept}: {prob:.2f}")

        if divergence_data['top_divergences']:
            tooltip_lines.append("\\nDivergence Details:")
            for item in divergence_data['top_divergences'][:3]:
                interp = item.get('interpretation', 'unknown')
                tooltip_lines.append(
                    f"  {item['concept']}: "
                    f"think={item['activation']:.2f}, write={item['text']:.2f} "
                    f"({interp})"
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

    def _format_sentence_divergence(self, sentence_div: Dict[str, Any]) -> Optional[str]:
        """Format sentence-level divergence as inline annotation."""
        divergences = sentence_div.get("divergences", [])
        if not divergences:
            return None

        # Only show if there's meaningful divergence
        top_div = divergences[0]
        if top_div["divergence_max"] < self.valves.DIVERGENCE_THRESHOLD_LOW:
            return None

        # Build compact summary
        covert_concepts = [
            d["concept"] for d in divergences
            if d["interpretation"] == "covert"
        ]

        if not covert_concepts:
            return None

        # Format as subtle inline marker
        concepts_str = ", ".join(covert_concepts[:3])
        if len(covert_concepts) > 3:
            concepts_str += f" +{len(covert_concepts) - 3}"

        return (
            f' <span style="color:#FF6B6B; font-size:0.8em;" '
            f'title="Sentence divergence: thinking about {concepts_str} without writing about it">'
            f'[div: {concepts_str}]</span>'
        )

    def _compute_response_divergence(
        self,
        response_activations: Dict[str, List[float]],
        response_text_sims: Dict[str, List[float]],
    ) -> Optional[Dict[str, Any]]:
        """Compute response-level divergence summary."""
        if not response_activations:
            return None

        divergences = []
        for concept in response_activations:
            activations = response_activations[concept]
            text_sims = response_text_sims.get(concept, [0.0])

            max_act = max(activations) if activations else 0.0
            avg_act = sum(activations) / len(activations) if activations else 0.0
            max_text = max(text_sims) if text_sims else 0.0
            avg_text = sum(text_sims) / len(text_sims) if text_sims else 0.0

            # Normalize text similarity
            max_text_prob = (max_text + 1) / 2
            avg_text_prob = (avg_text + 1) / 2

            div_max = max_act - max_text_prob

            if div_max > self.valves.DIVERGENCE_THRESHOLD_LOW:
                divergences.append({
                    "concept": concept,
                    "max_activation": max_act,
                    "max_text_prob": max_text_prob,
                    "divergence": div_max,
                    "token_count": len(activations),
                })

        if not divergences:
            return None

        divergences.sort(key=lambda x: x["divergence"], reverse=True)
        return {"divergences": divergences}

    def _format_response_divergence(self, response_div: Dict[str, Any]) -> str:
        """Format response-level divergence as end-of-response summary."""
        divergences = response_div.get("divergences", [])
        if not divergences:
            return ""

        lines = ["\\n---\\n**Response Divergence Summary:**"]
        for d in divergences[:5]:
            lines.append(
                f"- {d['concept']}: thought={d['max_activation']:.2f}, "
                f"wrote={d['max_text_prob']:.2f}, div={d['divergence']:+.2f} "
                f"({d['token_count']} tokens)"
            )

        return "\\n".join(lines)
