#!/usr/bin/env python3
"""
Dual Monitoring System Test with Hierarchical Concept Detection

Simultaneously detects:
1. SUMO/WordNet concepts (v3 lens pack) - hierarchical detection via DynamicLensManager
2. S-tier psychological simplexes (tripole lenses) - always active fundamental dimensions

Shows the model's actual generation output to verify detections.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.hat.monitoring.lens_manager import DynamicLensManager
from training.tripole_classifier import TripoleLens

# Paths
LENS_PACK_DIR = PROJECT_ROOT / "lens_packs" / "gemma-3-4b-pt_sumo-wordnet-v3"
SIMPLEX_DIR = PROJECT_ROOT / "results" / "s_tier_tripole_lazy" / "run_20251125_094653"


class SimplexLens:
    """Container for a single S-tier simplex tripole lens."""

    def __init__(self, dimension: str, lens_path: Path, hidden_dim: int = 2560):
        self.dimension = dimension
        self.lens = TripoleLens(hidden_dim=hidden_dim, n_poles=3)
        self.lens.load_state_dict(torch.load(lens_path, map_location='cpu'))
        self.lens.eval()

        # Pole names
        self.pole_names = ['negative', 'neutral', 'positive']

    def detect(self, activation: torch.Tensor) -> Dict:
        """
        Run detection on activation.

        Returns:
            Dict with pole probabilities and dominant pole
        """
        with torch.no_grad():
            logits = self.lens(activation.unsqueeze(0))  # [1, 3]
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # [3]

            dominant_idx = probs.argmax().item()
            dominant_pole = self.pole_names[dominant_idx]
            dominant_conf = probs[dominant_idx].item()

            return {
                'dimension': self.dimension,
                'dominant_pole': dominant_pole,
                'confidence': dominant_conf,
                'pole_probabilities': {
                    'negative': probs[0].item(),
                    'neutral': probs[1].item(),
                    'positive': probs[2].item()
                }
            }


class DualMonitor:
    """
    Dual monitoring system: Hierarchical SUMO concepts + S-tier simplexes.

    - SUMO concepts: Hierarchical detection via DynamicLensManager
    - Simplexes: Always active (fundamental psychological dimensions)
    """

    def __init__(
        self,
        model,
        tokenizer,
        lens_pack_dir: Path,
        simplex_dir: Path,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Initialize DynamicLensManager for hierarchical SUMO concept detection
        print("Initializing DynamicLensManager for hierarchical concept detection...")
        self.lens_manager = DynamicLensManager(
            lens_pack_id="gemma-3-4b-pt_sumo-wordnet-v3",
            device=device,
            max_loaded_lenses=20,  # Reduce to 20 to avoid OOM (model uses ~17GB)
            keep_top_k=5  # Only keep top 5 scoring lenses
        )
        print(f"  ✓ Loaded lens pack with hierarchical detection (max 20 lenses)")

        # Load S-tier simplex lenses (always active - they're fundamental dimensions)
        print("\nLoading S-tier simplex lenses (always active)...")
        self.simplex_lenses = self._load_simplex_lenses(simplex_dir)
        print(f"  ✓ Loaded {len(self.simplex_lenses)} simplexes")

    def _load_simplex_lenses(self, simplex_dir: Path) -> List[SimplexLens]:
        """Load all successfully trained simplex lenses."""
        lenses = []

        # Load results to find graduated simplexes
        results_file = simplex_dir / "results.json"
        with open(results_file) as f:
            results = json.load(f)

        graduated = results.get('graduated', [])

        for dimension in graduated:
            lens_path = simplex_dir / dimension / "tripole_lens.pt"
            if lens_path.exists():
                lens = SimplexLens(dimension, lens_path)
                lenses.append(lens)
                print(f"    ✓ {dimension}")

        return lenses

    def generate_and_monitor(
        self,
        prompt: str,
        layer_idx: int = 12,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        simplex_threshold: float = 0.4
    ) -> Dict:
        """
        Generate text from prompt and monitor both systems.

        Args:
            prompt: Input prompt
            layer_idx: Layer to extract activations from (simplexes trained on layer 12)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            simplex_threshold: Min confidence to report a simplex pole

        Returns:
            Dict with generation, concept detections, and simplex detections
        """
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*80}")

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # STEP 1: Extract PROMPT activations for SUMO concept detection
        # (SUMO lenses were trained on prompt activations, not generation)
        print("Extracting prompt activations for concept detection...")
        with torch.no_grad():
            prompt_outputs = self.model(**inputs, output_hidden_states=True)
            # Get last token of prompt from layer 2 (middle of trained range 0-5)
            prompt_hidden_states = prompt_outputs.hidden_states[2]  # [1, seq_len, hidden_dim]
            sumo_activation = prompt_hidden_states[0, -1, :].cpu()  # [hidden_dim]

        # STEP 2: Generate text with output_hidden_states for simplex detection
        # (Simplexes were trained on generation activations)
        print("Generating text and capturing generation activations...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode generation
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        generation_only = generated_text[len(prompt):].strip()

        print(f"\nGENERATION: {generation_only}")
        print(f"{'='*80}\n")

        # Extract simplex activation from the final generated token
        if outputs.hidden_states:
            # Get last generation step, layer 12 (where simplexes were trained)
            last_step_hidden_states = outputs.hidden_states[-1]
            simplex_layer_activation = last_step_hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
            simplex_activation = simplex_layer_activation[0, -1, :].cpu()  # [hidden_dim]
        else:
            raise ValueError("No hidden states captured during generation")

        # 1. Hierarchical SUMO concept detection via DynamicLensManager
        print("Running hierarchical concept detection...")

        # Convert to torch tensor and ensure float32 (lenses were trained with float32)
        if isinstance(sumo_activation, np.ndarray):
            activation_tensor = torch.tensor(sumo_activation, dtype=torch.float32).to(self.device)
        else:
            # Convert bfloat16/float16 to float32 for lens compatibility
            activation_tensor = sumo_activation.to(dtype=torch.float32, device=self.device)

        # detect_and_expand returns (concept_scores, timing_info)
        # concept_scores is a list of (concept_name, probability, layer) tuples
        concept_scores, timing_info = self.lens_manager.detect_and_expand(
            hidden_state=activation_tensor,
            top_k=20  # Get top 20 detected concepts
        )

        # Format concept detections
        detected_concepts = []
        for concept_name, probability, layer in concept_scores:
            detected_concepts.append({
                'sumo_term': concept_name,
                'confidence': probability,
                'layer': layer
            })

        print(f"  Detected {len(detected_concepts)} active concepts")

        # 2. S-tier simplex detection (always active)
        print("Running simplex detection...")
        simplex_detections = []
        activation_tensor = torch.tensor(simplex_activation, dtype=torch.float32)

        for lens in self.simplex_lenses:
            result = lens.detect(activation_tensor)
            if result['confidence'] >= simplex_threshold:
                simplex_detections.append(result)

        # Sort by confidence
        simplex_detections.sort(key=lambda x: x['confidence'], reverse=True)
        print(f"  Detected {len(simplex_detections)} active simplexes")

        return {
            'prompt': prompt,
            'generation': generation_only,
            'full_text': generated_text,
            'concepts': detected_concepts,
            'simplexes': simplex_detections
        }

    def print_results(self, results: Dict):
        """Pretty-print monitoring results."""
        print("\n" + "=" * 80)
        print("DUAL MONITORING RESULTS")
        print("=" * 80)

        print(f"\nPrompt: {results['prompt']}")
        print(f"Generation: {results['generation']}")

        # Print hierarchical SUMO concepts
        concepts = results['concepts']
        if concepts:
            print(f"\n  SUMO Concepts Detected (Hierarchical) - {len(concepts)}:")
            for concept in concepts[:15]:  # Top 15
                print(f"    • {concept['sumo_term']:<35} conf={concept['confidence']:.3f} layer={concept['layer']}")
            if len(concepts) > 15:
                print(f"    ... and {len(concepts) - 15} more")
        else:
            print("\n  No SUMO concepts detected by hierarchical system")

        # Print S-tier simplexes
        simplexes = results['simplexes']
        if simplexes:
            print(f"\n  Psychological Simplexes (Always Active) - {len(simplexes)}:")
            for simplex in simplexes:
                pole = simplex['dominant_pole']
                conf = simplex['confidence']
                dim = simplex['dimension']

                # Color code by pole
                if pole == 'negative':
                    indicator = "[-]"
                elif pole == 'positive':
                    indicator = "[+]"
                else:
                    indicator = "[0]"

                print(f"    {indicator} {dim:<35} {pole:<10} conf={conf:.3f}")

                # Show all pole probabilities
                probs = simplex['pole_probabilities']
                print(f"        (neg={probs['negative']:.2f}, neu={probs['neutral']:.2f}, pos={probs['positive']:.2f})")
        else:
            print("\n  No psychological simplexes detected above threshold")

        print("=" * 80)


def main():
    print("=" * 80)
    print("DUAL MONITORING SYSTEM TEST - HIERARCHICAL")
    print("=" * 80)
    print("Simultaneously detects:")
    print("  1. SUMO/WordNet concepts (hierarchical detection via DynamicLensManager)")
    print("  2. S-tier psychological simplexes (always active fundamental dimensions)")
    print("\nShows model generation output to verify detections")

    # Load model
    print("\n1. Loading model...")
    model_name = "google/gemma-3-4b-pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True
    )
    model.eval()
    print(f"   ✓ Model loaded on {device}")

    # Initialize dual monitor
    print("\n2. Initializing dual monitoring system...")
    monitor = DualMonitor(
        model=model,
        tokenizer=tokenizer,
        lens_pack_dir=LENS_PACK_DIR,
        simplex_dir=SIMPLEX_DIR,
        device=device
    )

    # Test prompts covering different psychological states and concepts
    test_prompts = [
        # Threat + negative affect
        "The dark figure approached silently through the fog, and I felt",

        # Social connection + positive affect
        "We sat together by the fire, sharing stories and feeling",

        # Achievement + positive affect
        "After months of hard work, I finally succeeded and felt",

        # Loss + negative affect
        "When I heard the news, my heart sank and I felt",

        # Curiosity + neutral/positive affect
        "I opened the ancient book and discovered",

        # Conflict + negative affect
        "The argument escalated and everyone started feeling",
    ]

    print("\n3. Running dual monitoring on test prompts...")

    all_results = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n\n{'#'*80}")
        print(f"TEST {i}/{len(test_prompts)}")
        print(f"{'#'*80}")

        results = monitor.generate_and_monitor(
            prompt=prompt,
            layer_idx=12,  # Simplexes trained on layer 12
            max_new_tokens=50,
            simplex_threshold=0.4
        )

        all_results.append(results)
        monitor.print_results(results)

    # Save results
    output_file = PROJECT_ROOT / "results" / "dual_monitoring_hierarchical_test.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*80}")
    print(f"✓ Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
