#!/usr/bin/env python3
"""
Benchmark simplex lens accuracy for three-pole homeostatic detection.

Tests each simplex's three lenses (μ−, μ0, μ+) across:
- Prompt activations (pole-specific text in input)
- Response activations (pole-specific generation)
- Homeostatic steering (steering toward neutral μ0)

Outputs CSV with pole-to-pole activation matrix.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.classifier import BinaryClassifier


POLE_NAMES = {
    'negative': 'negative_pole',
    'neutral': 'neutral_homeostasis',
    'positive': 'positive_pole'
}


def generate_pole_prompts(
    simplex_dimension: str,
    pole_data: Dict,
    pole_type: str,
    n_samples: int,
    seed: int = 42
) -> List[str]:
    """
    Generate pole-specific prompts.

    Args:
        simplex_dimension: Simplex name (e.g., "Hunger")
        pole_data: Pole metadata from layer2.json
        pole_type: "negative", "neutral", or "positive"
        n_samples: Number of prompts to generate
        seed: Random seed

    Returns:
        List of prompts
    """
    np.random.seed(seed + hash(pole_type) % 1000)

    # Get pole-specific terms from metadata
    pole_terms = pole_data.get('examples', [])
    gloss = pole_data.get('gloss', '')

    if not pole_terms:
        # Fallback: use dimension and pole type
        pole_terms = [f"{pole_type} {simplex_dimension.lower()}"]

    # Template types
    templates = [
        "I am feeling {term}",
        "The person is {term}",
        "This is a state of {term}",
        "Experiencing {term} means",
        "When someone is {term},",
        "A feeling of {term} is",
        "{term} can be described as",
        "The sensation of {term} is",
    ]

    prompts = []
    for i in range(n_samples):
        term = pole_terms[i % len(pole_terms)]
        template = templates[i % len(templates)]
        prompts.append(template.format(term=term))

    return prompts


def load_simplex_lenses(simplex_dir: Path) -> Dict[str, BinaryClassifier]:
    """
    Load all three pole lenses for a simplex.

    Args:
        simplex_dir: Directory containing pole subdirs (negative/, neutral/, positive/)

    Returns:
        Dict mapping pole_type -> BinaryClassifier
    """
    lenses = {}

    for pole_type in ['negative', 'neutral', 'positive']:
        pole_dir = simplex_dir / pole_type
        lens_file = pole_dir / "activation_lens.pkl"

        if not lens_file.exists():
            raise FileNotFoundError(f"Lens not found: {lens_file}")

        lens = BinaryClassifier()
        lens.load(str(lens_file))
        lenses[pole_type] = lens

    return lenses


def extract_simplex_activations(
    lenses: Dict[str, BinaryClassifier],
    prompts: List[str],
    model,
    tokenizer,
    device: str,
    layer_idx: int = 15,
    context: str = "prompt"
) -> Dict[str, np.ndarray]:
    """
    Extract all three pole activations for given prompts.

    Args:
        lenses: Dict of pole_type -> BinaryClassifier
        prompts: Input prompts
        model: Language model
        tokenizer: Tokenizer
        device: Device
        layer_idx: Layer to extract from
        context: "prompt" or "response"

    Returns:
        Dict mapping pole_type -> activations array [n_prompts]
    """
    activations = {pole: [] for pole in lenses.keys()}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        if context == "prompt":
            # Extract from prompt embedding
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx]
                embedding = hidden_states[0, -1, :].cpu().numpy()

        elif context == "response":
            # Extract from generated token
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    do_sample=False
                )
                last_step_states = outputs.hidden_states[-1]
                hidden_state = last_step_states[layer_idx]
                embedding = hidden_state[0, -1, :].cpu().numpy()

        # Get activation from all three lenses
        for pole_type, lens in lenses.items():
            activation = lens.predict_proba([embedding])[0]
            activations[pole_type].append(activation)

    # Convert to arrays
    return {pole: np.array(acts) for pole, acts in activations.items()}


def extract_steered_simplex_activations(
    lenses: Dict[str, BinaryClassifier],
    prompts: List[str],
    target_pole: str,
    model,
    tokenizer,
    device: str,
    layer_idx: int = 15,
    strength: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Extract pole activations with steering toward target pole.

    Args:
        lenses: Dict of pole_type -> BinaryClassifier
        prompts: Input prompts
        target_pole: Pole to steer toward ("negative", "neutral", "positive")
        model: Language model
        tokenizer: Tokenizer
        device: Device
        layer_idx: Layer to apply steering
        strength: Steering strength

    Returns:
        Dict mapping pole_type -> activations array [n_prompts]
    """
    activations = {pole: [] for pole in lenses.keys()}

    # Get steering vector from target pole lens
    steering_vector = lenses[target_pole].coef_[0]
    steering_tensor = torch.from_numpy(steering_vector * strength).float().to(device)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]

            # Apply steering
            steered_embedding = hidden_states[0, -1, :] + steering_tensor
            embedding = steered_embedding.cpu().numpy()

        # Get activation from all three lenses
        for pole_type, lens in lenses.items():
            activation = lens.predict_proba([embedding])[0]
            activations[pole_type].append(activation)

    return {pole: np.array(acts) for pole, acts in activations.items()}


def run_simplex_benchmark(
    simplex_name: str,
    simplex_data: Dict,
    lenses: Dict[str, BinaryClassifier],
    model,
    tokenizer,
    device: str,
    n_samples: int = 100,
    layer_idx: int = 15
) -> List[Dict]:
    """
    Run full benchmark for a single simplex.

    Args:
        simplex_name: Simplex dimension name
        simplex_data: Simplex metadata from layer2.json
        lenses: Dict of pole_type -> BinaryClassifier
        model: Language model
        tokenizer: Tokenizer
        device: Device
        n_samples: Samples per pole
        layer_idx: Layer index

    Returns:
        List of result dictionaries
    """
    results = []
    three_pole = simplex_data['three_pole_simplex']

    # Test each pole type
    for prompted_pole in ['negative', 'neutral', 'positive']:
        pole_data = three_pole[POLE_NAMES[prompted_pole]]

        # Generate pole-specific prompts
        prompts = generate_pole_prompts(
            simplex_name, pole_data, prompted_pole, n_samples
        )

        # 1. Prompt activations
        prompt_acts = extract_simplex_activations(
            lenses, prompts, model, tokenizer, device, layer_idx, context="prompt"
        )

        for detected_pole, activations in prompt_acts.items():
            results.append({
                'simplex_dimension': simplex_name,
                'pole_prompted': prompted_pole,
                'pole_detected': detected_pole,
                'context': 'prompt',
                'n_samples': n_samples,
                'mean_activation': float(np.mean(activations)),
                'std_activation': float(np.std(activations)),
                'peak_activation': float(np.max(activations)),
                'min_activation': float(np.min(activations))
            })

        # 2. Response activations
        response_acts = extract_simplex_activations(
            lenses, prompts, model, tokenizer, device, layer_idx, context="response"
        )

        for detected_pole, activations in response_acts.items():
            results.append({
                'simplex_dimension': simplex_name,
                'pole_prompted': prompted_pole,
                'pole_detected': detected_pole,
                'context': 'response',
                'n_samples': n_samples,
                'mean_activation': float(np.mean(activations)),
                'std_activation': float(np.std(activations)),
                'peak_activation': float(np.max(activations)),
                'min_activation': float(np.min(activations))
            })

    # 3. Homeostatic steering (all poles -> neutral)
    # Use prompts from all three poles mixed
    all_prompts = []
    for pole in ['negative', 'neutral', 'positive']:
        pole_data = three_pole[POLE_NAMES[pole]]
        all_prompts.extend(
            generate_pole_prompts(simplex_name, pole_data, pole, n_samples // 3)
        )

    # Steer toward each pole
    for target_pole in ['negative', 'neutral', 'positive']:
        steered_acts = extract_steered_simplex_activations(
            lenses, all_prompts, target_pole, model, tokenizer, device, layer_idx
        )

        for detected_pole, activations in steered_acts.items():
            results.append({
                'simplex_dimension': simplex_name,
                'pole_prompted': 'mixed',  # Mixed inputs
                'pole_detected': detected_pole,
                'context': f'steered_to_{target_pole}',
                'n_samples': len(all_prompts),
                'mean_activation': float(np.mean(activations)),
                'std_activation': float(np.std(activations)),
                'peak_activation': float(np.max(activations)),
                'min_activation': float(np.min(activations))
            })

    return results


def save_results(results: List[Dict], output_path: Path):
    """Save benchmark results to CSV."""
    if not results:
        print("No results to save")
        return

    fieldnames = [
        'simplex_dimension', 'pole_prompted', 'pole_detected', 'context',
        'n_samples', 'mean_activation', 'std_activation',
        'peak_activation', 'min_activation'
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Results saved to {output_path}")


def load_layer2() -> Dict:
    """Load layer2.json with simplex definitions."""
    layer2_path = PROJECT_ROOT / "data" / "concept_graph" / "abstraction_layers" / "layer2.json"
    with open(layer2_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Benchmark simplex lens accuracy")
    parser.add_argument("--simplexes", nargs="+", required=True, help="Simplex dimensions to benchmark")
    parser.add_argument("--lens-dir", type=Path, required=True, help="Root lens directory (e.g., results/s_tier_simplexes/run_X)")
    parser.add_argument("--model-name", default="google/gemma-3-4b-pt", help="Model name")
    parser.add_argument("--layer-idx", type=int, default=15, help="Layer to extract activations from")
    parser.add_argument("--n-samples", type=int, default=100, help="Samples per pole")
    parser.add_argument("--output", type=Path, help="Output CSV path")
    args = parser.parse_args()

    print("=" * 80)
    print("SIMPLEX LENS BENCHMARK")
    print("=" * 80)

    # Default output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = PROJECT_ROOT / "results" / "lens_benchmarks" / f"simplex_lenses_{timestamp}.csv"

    # Load layer2 metadata
    print("\nLoading layer2.json...")
    layer2 = load_layer2()
    simplex_map = {
        concept['simplex_dimension']: concept
        for concept in layer2['concepts']
        if concept.get('s_tier') and concept.get('simplex_dimension')
    }
    print(f"✓ Found {len(simplex_map)} S-tier simplexes in layer2.json")

    # Validate requested simplexes
    for simplex in args.simplexes:
        if simplex not in simplex_map:
            print(f"Error: {simplex} not found in layer2.json")
            return 1

    # Load model
    print(f"\nLoading model: {args.model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float32,
        device_map=device,
        local_files_only=True
    )
    model.eval()
    print(f"✓ Model loaded on {device}")

    # Run benchmarks
    all_results = []

    for simplex_name in tqdm(args.simplexes, desc="Benchmarking simplexes"):
        print(f"\n{simplex_name}")
        print("─" * 60)

        # Load lenses
        simplex_dir = args.lens_dir / simplex_name
        if not simplex_dir.exists():
            print(f"  ✗ Lens directory not found: {simplex_dir}")
            continue

        try:
            lenses = load_simplex_lenses(simplex_dir)
            print(f"  ✓ Loaded 3 pole lenses")
        except Exception as e:
            print(f"  ✗ Failed to load lenses: {e}")
            continue

        # Run benchmark
        simplex_data = simplex_map[simplex_name]
        results = run_simplex_benchmark(
            simplex_name=simplex_name,
            simplex_data=simplex_data,
            lenses=lenses,
            model=model,
            tokenizer=tokenizer,
            device=device,
            n_samples=args.n_samples,
            layer_idx=args.layer_idx
        )

        all_results.extend(results)
        print(f"  ✓ Completed {len(results)} measurements")

    # Save results
    save_results(all_results, args.output)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Simplexes tested: {len(args.simplexes)}")
    print(f"Total measurements: {len(all_results)}")
    print(f"Samples per pole: {args.n_samples}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
