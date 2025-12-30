"""
LLM-based Divergence Pipeline for OpenWebUI

Simplified pipeline that uses an LLM as the divergence classifier
instead of trained CAT classifiers. Drop-in replacement for testing.
"""

from typing import Generator, List, Dict, Any, Optional
from pathlib import Path

import torch
from pydantic import BaseModel

from src.cat.llm_divergence_scorer import LLMDivergenceScorer


class Pipeline:
    """OpenWebUI pipeline using LLM-based divergence scoring."""

    class Valves(BaseModel):
        """User-configurable settings."""

        # Target model (the one generating responses)
        TARGET_MODEL: str = "google/gemma-3-4b-it"

        # Classifier model (tiny model for divergence scoring)
        CLASSIFIER_MODEL: str = "qwen-0.5b"  # or "gemma-2b", "phi-3-mini", etc.

        # Concept pack for definitions
        CONCEPT_PACK: str = "concept_packs/first-light"

        # Scoring settings
        TOP_K_CONCEPTS: int = 15
        DIVERGENCE_THRESHOLD_LOW: float = 0.2
        DIVERGENCE_THRESHOLD_HIGH: float = 0.5

        # How often to score (every N tokens, or per sentence)
        SCORE_PER_SENTENCE: bool = True
        SCORE_EVERY_N_TOKENS: int = 20

    def __init__(self):
        self.name = "HatCat LLM Divergence"
        self.valves = self.Valves()
        self.scorer: Optional[LLMDivergenceScorer] = None
        self.target_model = None
        self.target_tokenizer = None

    async def on_startup(self):
        """Load models on startup."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("🎩 HatCat LLM Divergence: Loading models...")

        # Load divergence scorer
        self.scorer = LLMDivergenceScorer.from_concept_pack(
            Path(self.valves.CONCEPT_PACK),
            model_name=self.valves.CLASSIFIER_MODEL,
            device="cuda",
        )

        # Load target model
        print(f"Loading target model: {self.valves.TARGET_MODEL}")
        self.target_tokenizer = AutoTokenizer.from_pretrained(self.valves.TARGET_MODEL)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.valves.TARGET_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.target_model.eval()

        print("✓ Models loaded")

    async def on_shutdown(self):
        print("🎩 HatCat LLM Divergence: Shutting down...")

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[Dict[str, str]],
        body: Dict[str, Any],
    ) -> Generator[str, None, None]:
        """Process user message and stream response with divergence."""

        prompt = self._build_prompt(messages)
        yield from self._generate_with_divergence(prompt, body)

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Build prompt from conversation history."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts) + "\nassistant: "

    def _generate_with_divergence(
        self,
        prompt: str,
        body: Dict[str, Any],
    ) -> Generator[str, None, None]:
        """Generate with periodic divergence scoring."""

        inputs = self.target_tokenizer(prompt, return_tensors="pt").to("cuda")
        max_new_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)

        generated_ids = inputs.input_ids
        current_sentence = []
        token_count = 0
        sentence_end_chars = {'.', '!', '?'}

        for step in range(max_new_tokens):
            # Generate next token
            with torch.no_grad():
                outputs = self.target_model(generated_ids, return_dict=True)
                next_logits = outputs.logits[:, -1, :] / temperature
                next_token_id = torch.argmax(next_logits, dim=-1)

            token_text = self.target_tokenizer.decode([next_token_id.item()])
            current_sentence.append(token_text)
            token_count += 1

            # Yield the token
            yield token_text

            # Check if we should score
            should_score = False
            is_sentence_end = any(c in token_text for c in sentence_end_chars)

            if self.valves.SCORE_PER_SENTENCE and is_sentence_end:
                should_score = True
            elif not self.valves.SCORE_PER_SENTENCE and token_count % self.valves.SCORE_EVERY_N_TOKENS == 0:
                should_score = True

            if should_score and current_sentence:
                sentence_text = "".join(current_sentence)
                divergence_annotation = self._score_and_format(sentence_text)
                if divergence_annotation:
                    yield divergence_annotation

                if is_sentence_end:
                    current_sentence = []

            # Update sequence
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)

            if next_token_id.item() == self.target_tokenizer.eos_token_id:
                break

        # Final sentence if any
        if current_sentence:
            sentence_text = "".join(current_sentence)
            divergence_annotation = self._score_and_format(sentence_text)
            if divergence_annotation:
                yield divergence_annotation

    def _score_and_format(self, text: str) -> Optional[str]:
        """Score text and format divergence annotation."""

        div_score, concerning = self.scorer.quick_divergence(
            text,
            top_k=self.valves.TOP_K_CONCEPTS,
        )

        if div_score < self.valves.DIVERGENCE_THRESHOLD_LOW:
            return None

        # Format based on severity
        if div_score >= self.valves.DIVERGENCE_THRESHOLD_HIGH:
            color = "#FF6B6B"
            label = "HIGH"
        else:
            color = "#FFD700"
            label = "MED"

        concepts_str = ", ".join(concerning[:3])
        if len(concerning) > 3:
            concepts_str += f" +{len(concerning) - 3}"

        return (
            f' <span style="color:{color}; font-size:0.8em;" '
            f'title="Divergence {div_score:.2f}: {concepts_str}">'
            f'[{label}: {concepts_str}]</span>'
        )
