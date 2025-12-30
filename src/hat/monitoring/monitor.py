#!/usr/bin/env python3
"""
SUMO Temporal Concept Monitor

Non-invasive monitoring of SUMO concepts during LLM generation.
Uses model.generate() to avoid interfering with generation quality.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn

# Import unified classifier from HAT module
from src.hat.classifiers.classifier import MLPClassifier, load_classifier

try:
    from .concept_dissonance import (
        SUMOConceptGraph,
        TokenToSUMOMapper,
        batch_divergence,
        compute_temporal_lag,
    )
    DISSONANCE_AVAILABLE = True
except ImportError:
    DISSONANCE_AVAILABLE = False


# Backwards compatibility alias
SimpleMLP = MLPClassifier


def load_sumo_classifiers(
    layers: List[int] = [0, 1, 2],
    results_dir: Path = Path("results/sumo_classifiers"),
    device: str = "cuda"
) -> Tuple[Dict[str, Tuple[nn.Module, int, int]], int]:
    """
    Load SUMO classifiers from all specified layers.

    Returns:
        (classifiers_dict, hidden_dim) where classifiers_dict maps:
        concept_name -> (model, layer, index_in_layer)
    """
    classifiers = {}
    hidden_dim = None

    for layer in layers:
        layer_dir = results_dir / f"layer{layer}"
        results_file = layer_dir / "results.json"

        if not results_file.exists():
            print(f"Warning: No results for layer {layer}")
            continue

        # Load results to get concept names
        with open(results_file) as f:
            results = json.load(f)

        # Load layer concepts to get hidden dim
        layer_path = Path(f"data/concept_graph/abstraction_layers/layer{layer}.json")
        with open(layer_path) as f:
            layer_data = json.load(f)

        concepts = layer_data['concepts']

        # Load each classifier
        for i, concept in enumerate(concepts):
            concept_name = concept['sumo_term']
            classifier_path = layer_dir / f"{concept_name}_classifier.pt"

            if not classifier_path.exists():
                continue

            try:
                # Use unified loader from HAT module
                model = load_classifier(classifier_path, device=device)

                # Infer hidden_dim from first classifier if not set
                if hidden_dim is None:
                    hidden_dim = model.input_dim

                classifiers[concept_name] = (model, layer, i)
            except Exception as e:
                print(f"Warning: Failed to load {concept_name}: {e}")
                continue

    print(f"✓ Loaded {len(classifiers)} classifiers from layers {layers}")
    print(f"  Hidden dim: {hidden_dim}")

    return classifiers, hidden_dim


class SUMOTemporalMonitor:
    """
    Non-invasive temporal concept monitor for SUMO hierarchical concepts.

    Monitors concept activations during generation WITHOUT interfering with
    the generation process (no hooks, no manual token selection).
    """

    def __init__(
        self,
        classifiers: Dict[str, Tuple[nn.Module, int, int]],
        top_k: int = 10,
        threshold: float = 0.3,
        enable_dissonance: bool = False,
        dissonance_alpha: float = 0.5,
    ):
        """
        Args:
            classifiers: Dict mapping concept_name -> (classifier, layer, idx)
            top_k: Show top K concepts per timestep
            threshold: Only show concepts with prob > threshold
            enable_dissonance: Compute semantic dissonance scores
            dissonance_alpha: Decay parameter for distance (higher = faster decay)
        """
        self.classifiers = classifiers
        self.top_k = top_k
        self.threshold = threshold
        self.enable_dissonance = enable_dissonance
        self.dissonance_alpha = dissonance_alpha

        # Initialize dissonance components if enabled
        self.sumo_graph = None
        self.token_mapper = None
        if enable_dissonance:
            if not DISSONANCE_AVAILABLE:
                print("Warning: Dissonance measurement requires spacy and nltk")
                print("  pip install spacy nltk")
                print("  python -m spacy download en_core_web_sm")
                print("  python -c \"import nltk; nltk.download('wordnet')\"")
                self.enable_dissonance = False
            else:
                try:
                    self.sumo_graph = SUMOConceptGraph()
                    self.token_mapper = TokenToSUMOMapper(self.sumo_graph)
                    print("✓ Dissonance measurement enabled")
                except Exception as e:
                    print(f"Warning: Failed to initialize dissonance: {e}")
                    self.enable_dissonance = False

    def monitor_generation(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
        device: str = "cuda"
    ) -> Dict:
        """
        Generate text and monitor concept activations without interference.

        Uses model.generate() with proper sampling to avoid mode collapse.

        Args:
            model: LLM
            tokenizer: Tokenizer
            prompt: Input prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (vs greedy)
            device: Device

        Returns:
            Dict with:
            - prompt: Input prompt
            - generated_text: Full generated text
            - tokens: List of generated tokens
            - timesteps: List of dicts, one per token, with:
                - token: The token text
                - position: Token position
                - concepts: List of (concept, probability, layer) sorted by prob
        """
        model.eval()

        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.inference_mode():
            # Generate with hidden states
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            # Extract generated tokens
            token_ids = outputs.sequences[0][prompt_len:].cpu().tolist()
            tokens = [tokenizer.decode([tid]) for tid in token_ids]

            # Process hidden states for each generated token
            timesteps = []

            for step_idx, step_states in enumerate(outputs.hidden_states):
                # step_states is a tuple of (num_layers,) tensors
                # Each tensor is [batch=1, seq_len, hidden_dim]
                # We want the last layer, last position
                last_layer = step_states[-1]  # [1, seq_len, hidden_dim]
                hidden_state = last_layer[:, -1, :]  # [1, hidden_dim]

                # Run all classifiers
                concept_probs = []
                for concept_name, (classifier, layer, idx) in self.classifiers.items():
                    prob = classifier(hidden_state).item()

                    if prob > self.threshold:
                        concept_probs.append({
                            'concept': concept_name,
                            'probability': float(prob),
                            'layer': int(layer)
                        })

                # Sort by probability
                concept_probs.sort(key=lambda x: x['probability'], reverse=True)

                # Keep only top K
                concept_probs = concept_probs[:self.top_k]

                # Store timestep data
                timesteps.append({
                    'token': tokens[step_idx] if step_idx < len(tokens) else '<eos>',
                    'position': prompt_len + step_idx,
                    'concepts': concept_probs
                })

        generated_text = ''.join(tokens)

        result = {
            'prompt': prompt,
            'generated_text': generated_text,
            'tokens': tokens,
            'timesteps': timesteps,
            'summary': {
                'total_tokens': len(tokens),
                'unique_concepts_detected': len(set(
                    c['concept'] for ts in timesteps for c in ts['concepts']
                ))
            }
        }

        # Compute dissonance if enabled
        if self.enable_dissonance and self.sumo_graph is not None:
            try:
                dissonance_results = batch_divergence(
                    tokens=tokens,
                    timesteps=timesteps,
                    graph=self.sumo_graph,
                    mapper=self.token_mapper,
                    alpha=self.dissonance_alpha,
                    context_window=3,  # Use 3-token lookahead for disambiguation
                )

                # Add dissonance to each timestep
                for ts, diss in zip(timesteps, dissonance_results):
                    ts['expected_concept'] = diss['expected_concept']
                    ts['divergence'] = diss['divergence']

                # Compute summary statistics
                valid_divergences = [d['divergence'] for d in dissonance_results if d['divergence'] is not None]
                if valid_divergences:
                    result['summary']['avg_divergence'] = sum(valid_divergences) / len(valid_divergences)
                    result['summary']['max_divergence'] = max(valid_divergences)
                    result['summary']['tokens_with_expected_concept'] = sum(
                        1 for d in dissonance_results if d['expected_concept'] is not None
                    )

                # Compute temporal lag (think-ahead analysis)
                lag_analysis = compute_temporal_lag(timesteps, max_lag=5)
                result['summary']['temporal_lag'] = lag_analysis

            except Exception as e:
                print(f"Warning: Dissonance computation failed: {e}")

        return result

    def print_report(self, result: Dict, show_token_details: bool = True):
        """Print human-readable temporal detection report."""

        print("\n" + "=" * 80)
        print("SUMO TEMPORAL CONCEPT DETECTION")
        print("=" * 80)

        print(f"\nPrompt: {result['prompt']}")
        print(f"Generated: {result['generated_text']}")
        print(f"\nTotal tokens: {result['summary']['total_tokens']}")
        print(f"Unique concepts detected: {result['summary']['unique_concepts_detected']}")

        if show_token_details:
            print("\n" + "-" * 80)
            print("TOKEN-BY-TOKEN CONCEPT DETECTION")
            print("-" * 80)

            for ts in result['timesteps']:
                print(f"\nToken [{ts['position']}]: '{ts['token']}'")

                # Show expected concept and divergence if available
                if 'expected_concept' in ts and ts['expected_concept'] is not None:
                    div = ts.get('divergence', 0)
                    div_bar = "!" * int(div * 10)
                    print(f"  Expected: {ts['expected_concept']} | Divergence: {div:.3f} {div_bar}")

                if ts['concepts']:
                    print(f"  Active concepts (top {len(ts['concepts'])}):")
                    for c in ts['concepts']:
                        layer_tag = f"L{c['layer']}"
                        prob_bar = "█" * int(c['probability'] * 20)
                        print(f"    [{layer_tag}] {c['concept']:30s} {c['probability']:.3f} {prob_bar}")
                else:
                    print("  (no concepts above threshold)")

        print("\n" + "=" * 80)

    def save_json(self, result: Dict, output_path: Path):
        """Save API-ready JSON output."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n✓ Saved API-ready JSON to: {output_path}")
        print("\nAPI Structure:")
        print("  - result['timesteps'][i]['token']: Token text")
        print("  - result['timesteps'][i]['position']: Token position")
        print("  - result['timesteps'][i]['concepts']: List of {concept, probability, layer}")
        print("\nThis JSON can be consumed by:")
        print("  - Frontend UIs (Ollama, OpenWebUI, LibreChat)")
        print("  - Model reasoning cycles")
        print("  - MCP (Model Context Protocol) servers")
