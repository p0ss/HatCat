"""
Script to capture activation patterns for diverse concepts.
Tests activation capture with 10 sample concepts.
"""

import torch
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hat.classifiers.classifiers.capture import ActivationCapture, ActivationConfig, BaselineGenerator
from src.hat.utils.model_loader import ModelLoader
from src.hat.utils import ActivationStorage


# 10 diverse sample concepts for initial testing
SAMPLE_CONCEPTS = [
    {
        "name": "justice",
        "category": "abstract_concept",
        "prompts": [
            "The concept of justice",
            "Justice is important for society",
            "A fair trial ensures justice",
            "Justice demands equality",
            "Without justice, there is chaos"
        ]
    },
    {
        "name": "dog",
        "category": "animal",
        "prompts": [
            "A dog",
            "The dog barked loudly",
            "Dogs are loyal companions",
            "My dog loves to play fetch",
            "A friendly dog wagged its tail"
        ]
    },
    {
        "name": "democracy",
        "category": "political_concept",
        "prompts": [
            "Democracy",
            "Democracy allows citizens to vote",
            "A democratic government",
            "The principles of democracy",
            "Democracy requires participation"
        ]
    },
    {
        "name": "happy",
        "category": "emotion",
        "prompts": [
            "Happy",
            "She felt happy",
            "A happy child laughed",
            "The happy couple celebrated",
            "Happy memories filled her mind"
        ]
    },
    {
        "name": "computer",
        "category": "technology",
        "prompts": [
            "A computer",
            "The computer processed data quickly",
            "Modern computers are powerful",
            "She used her computer for work",
            "Computer technology advances rapidly"
        ]
    },
    {
        "name": "love",
        "category": "emotion",
        "prompts": [
            "Love",
            "They fell in love",
            "Love brings people together",
            "A mother's love for her child",
            "Love conquers all"
        ]
    },
    {
        "name": "mountain",
        "category": "geography",
        "prompts": [
            "A mountain",
            "The tall mountain peaks",
            "Mountains covered in snow",
            "Climbing the mountain was difficult",
            "The mountain range stretched far"
        ]
    },
    {
        "name": "learning",
        "category": "cognitive_process",
        "prompts": [
            "Learning",
            "Children are learning new skills",
            "The process of learning",
            "Learning requires practice",
            "She enjoyed learning languages"
        ]
    },
    {
        "name": "money",
        "category": "economics",
        "prompts": [
            "Money",
            "Saving money is important",
            "They needed more money",
            "Money enables trade",
            "The value of money fluctuates"
        ]
    },
    {
        "name": "freedom",
        "category": "abstract_concept",
        "prompts": [
            "Freedom",
            "The freedom to choose",
            "Freedom is a fundamental right",
            "They fought for freedom",
            "Personal freedom matters"
        ]
    }
]


def capture_concept_activations(
    model,
    tokenizer,
    baseline,
    concept,
    config
):
    """
    Capture activation patterns for a single concept.

    Returns:
        Dictionary with averaged activation diffs across prompts
    """
    all_diffs = []

    with ActivationCapture(model, config) as capturer:
        capturer.register_hooks()

        for prompt in concept['prompts']:
            capturer.clear_activations()

            # Tokenize and run
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                _ = model(**inputs)

            # Compute difference from baseline
            activations = capturer.get_activations()
            diff = capturer.compute_activation_diff(baseline, activations)

            all_diffs.append(diff)

    # Average across prompts
    averaged_diff = {}
    layer_names = all_diffs[0].keys()

    for layer_name in layer_names:
        layer_diffs = [d[layer_name] for d in all_diffs if layer_name in d]
        averaged_diff[layer_name] = torch.stack(layer_diffs).mean(dim=0)

    return averaged_diff


def analyze_activation_stability(all_concept_diffs):
    """
    Analyze stability of activation patterns across concepts.

    Returns:
        Dictionary with stability metrics
    """
    print("\n" + "=" * 80)
    print("ACTIVATION PATTERN STABILITY ANALYSIS")
    print("=" * 80)

    stability_metrics = {}

    # For each concept, look at variance across prompts
    for concept_name, diffs_list in all_concept_diffs.items():
        # Get first layer as example
        sample_layer = list(diffs_list[0].keys())[0]

        # Stack activations across different prompts
        layer_acts = torch.stack([d[sample_layer] for d in diffs_list])

        # Calculate coefficient of variation (std / mean)
        mean = layer_acts.mean(dim=0)
        std = layer_acts.std(dim=0)

        # Only compute CV for non-zero activations
        non_zero_mask = mean.abs() > 1e-6
        cv = torch.zeros_like(mean)
        cv[non_zero_mask] = std[non_zero_mask] / mean[non_zero_mask].abs()

        stability_metrics[concept_name] = {
            'mean_cv': cv.mean().item(),
            'median_cv': cv.median().item(),
            'max_activation': layer_acts.abs().max().item(),
            'sparsity': (layer_acts == 0).float().mean().item()
        }

    # Print summary
    print(f"\nStability Metrics (Coefficient of Variation - lower is better):")
    print(f"{'Concept':<20} {'Mean CV':<12} {'Median CV':<12} {'Max Act':<12} {'Sparsity'}")
    print("-" * 80)

    for concept, metrics in stability_metrics.items():
        print(f"{concept:<20} {metrics['mean_cv']:<12.4f} {metrics['median_cv']:<12.4f} "
              f"{metrics['max_activation']:<12.6f} {metrics['sparsity']:.2%}")

    return stability_metrics


def main():
    """Main execution."""
    print("=" * 80)
    print("CAPTURING ACTIVATION PATTERNS FOR 10 DIVERSE CONCEPTS")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model, tokenizer = ModelLoader.load_gemma_270m()

    # Configuration
    config = ActivationConfig(
        top_k=100,
        storage_dtype=torch.float16
    )

    # Generate baseline
    print("\nGenerating baseline activations...")
    baseline_gen = BaselineGenerator(model, tokenizer, config)
    baseline = baseline_gen.generate_baseline(num_samples=10)
    print(f"✓ Baseline generated from {len(baseline)} layers")

    # Capture concepts
    print("\nCapturing concept activation patterns...")

    storage_path = Path("data/concept_activations.h5")
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    all_concept_diffs = {}

    with ActivationStorage(storage_path, mode='w') as storage:
        # Store baseline
        baseline_np = {k: v.cpu().numpy() for k, v in baseline.items()}
        storage.store_baseline(baseline_np)

        # Capture each concept
        for concept in tqdm(SAMPLE_CONCEPTS, desc="Processing concepts"):
            print(f"\n  Processing '{concept['name']}' ({concept['category']})...")

            # Capture for multiple prompts
            individual_diffs = []
            with ActivationCapture(model, config) as capturer:
                capturer.register_hooks()

                for prompt in concept['prompts']:
                    capturer.clear_activations()

                    inputs = tokenizer(prompt, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}

                    with torch.no_grad():
                        _ = model(**inputs)

                    activations = capturer.get_activations()
                    diff = capturer.compute_activation_diff(baseline, activations)
                    individual_diffs.append(diff)

            # Store individual diffs for stability analysis
            all_concept_diffs[concept['name']] = individual_diffs

            # Average across prompts
            averaged_diff = {}
            layer_names = individual_diffs[0].keys()
            for layer_name in layer_names:
                layer_diffs = [d[layer_name] for d in individual_diffs if layer_name in d]
                averaged_diff[layer_name] = torch.stack(layer_diffs).mean(dim=0)

            # Convert to numpy and store
            diff_np = {k: v.cpu().numpy() for k, v in averaged_diff.items()}

            storage.store_concept_activations(
                concept['name'],
                diff_np,
                metadata={
                    'category': concept['category'],
                    'num_prompts': len(concept['prompts']),
                    'prompts': concept['prompts']
                }
            )

            # Print sample stats
            sample_layer = list(averaged_diff.keys())[0]
            sample_diff = averaged_diff[sample_layer]
            print(f"    Avg activation: {sample_diff.abs().mean():.6f}, "
                  f"Max: {sample_diff.abs().max():.6f}, "
                  f"Sparsity: {(sample_diff == 0).float().mean():.2%}")

        # Get storage stats
        stats = storage.get_storage_stats()
        print(f"\n" + "=" * 80)
        print("STORAGE SUMMARY")
        print("=" * 80)
        print(f"File: {storage_path}")
        print(f"Size: {stats['file_size_mb']:.2f} MB")
        print(f"Concepts stored: {stats['num_concepts']}")
        print(f"Concepts: {', '.join(stats['concepts'].keys())}")

    # Analyze stability
    stability = analyze_activation_stability(all_concept_diffs)

    # Summary
    avg_cv = np.mean([m['mean_cv'] for m in stability.values()])
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Successfully captured {len(SAMPLE_CONCEPTS)} concepts")
    print(f"✓ Average coefficient of variation: {avg_cv:.4f}")
    print(f"✓ Activation patterns {'STABLE' if avg_cv < 0.5 else 'VARIABLE'} across contexts")
    print(f"✓ Data saved to: {storage_path}")

    return 0


if __name__ == "__main__":
    exit(main())
