#!/usr/bin/env python3
"""
Compute simplex steering vectors as pole activation differences.

Instead of binary classifiers that learn "emotional vs not",
compute direct activation differences between poles:
  μ- → μ0 steering vector = centroid(neutral) - centroid(negative)
  μ0 → μ+ steering vector = centroid(positive) - centroid(neutral)
  μ- → μ+ steering vector = centroid(positive) - centroid(negative)

These vectors can steer activations directly along the pole dimension.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Import training data generator
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from map.training.sumo_data_generation import create_simplex_pole_training_dataset_contrastive

# S-tier simplex definitions
S_TIER_DEFS_PATH = PROJECT_ROOT / "data" / "s_tier_simplex_definitions.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "simplex_steering_vectors"


def load_simplexes():
    """Load S-tier simplex definitions."""
    with open(S_TIER_DEFS_PATH) as f:
        data = json.load(f)
    return data['simplexes']


def extract_activations(
    model, tokenizer, prompts: List[str], layer_idx: int = 12
) -> torch.Tensor:
    """Extract activations from model."""
    device = next(model.parameters()).device
    all_acts = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx]
            act = hidden[0, -1, :].cpu().float()
            all_acts.append(act)

    return torch.stack(all_acts)


def generate_pole_prompts(
    simplex_name: str,
    simplex_def: dict,
    pole_name: str,  # 'negative_pole', 'neutral_homeostasis', 'positive_pole'
    n_samples: int = 50
) -> List[str]:
    """Generate prompts for a specific pole using the training data generator."""
    pole_data = simplex_def[pole_name]
    pole_type = pole_name.split('_')[0]  # 'negative', 'neutral', 'positive'

    # Get other poles for the contrastive generator (we'll only use positives)
    other_pole_names = [p for p in ['negative_pole', 'neutral_homeostasis', 'positive_pole'] if p != pole_name]
    other_poles_data = [
        {**simplex_def[p], 'pole_type': p.split('_')[0]}
        for p in other_pole_names
    ]

    # Generate training data
    prompts, labels = create_simplex_pole_training_dataset_contrastive(
        pole_data=pole_data,
        pole_type=pole_type,
        dimension=simplex_name,
        other_poles_data=other_poles_data,
        behavioral_ratio=0.6,
        prompts_per_synset=5
    )

    # Take only positive samples (the pole we want)
    positive_prompts = [p for p, l in zip(prompts, labels) if l == 1]
    return positive_prompts[:n_samples]


def compute_steering_vectors(
    model,
    tokenizer,
    simplex_name: str,
    simplex_def: dict,
    layer_idx: int = 12,
    n_samples: int = 50
) -> Dict[str, torch.Tensor]:
    """Compute steering vectors between poles."""
    print(f"\n  Computing vectors for: {simplex_name}")

    # Generate prompts for each pole
    pole_prompts = {}
    pole_centroids = {}

    for pole_name in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
        pole_type = pole_name.split('_')[0]
        print(f"    Generating {pole_type} pole prompts...")

        prompts = generate_pole_prompts(simplex_name, simplex_def, pole_name, n_samples)
        print(f"      Got {len(prompts)} prompts")

        if len(prompts) < 5:
            print(f"      ⚠️ Too few prompts for {pole_type}")
            continue

        # Extract activations
        acts = extract_activations(model, tokenizer, prompts, layer_idx)
        centroid = acts.mean(dim=0)

        pole_prompts[pole_type] = prompts
        pole_centroids[pole_type] = centroid
        print(f"      Centroid computed: {centroid.shape}")

    if len(pole_centroids) < 3:
        print(f"    ⚠️ Could not compute all centroids for {simplex_name}")
        return {}

    # Compute steering vectors
    vectors = {}

    # Homeostatic vectors (toward neutral)
    vectors['negative_to_neutral'] = pole_centroids['neutral'] - pole_centroids['negative']
    vectors['positive_to_neutral'] = pole_centroids['neutral'] - pole_centroids['positive']

    # Pole-to-pole vectors
    vectors['negative_to_positive'] = pole_centroids['positive'] - pole_centroids['negative']
    vectors['neutral_to_positive'] = pole_centroids['positive'] - pole_centroids['neutral']
    vectors['neutral_to_negative'] = pole_centroids['negative'] - pole_centroids['neutral']

    # Store centroids too for analysis
    vectors['centroid_negative'] = pole_centroids['negative']
    vectors['centroid_neutral'] = pole_centroids['neutral']
    vectors['centroid_positive'] = pole_centroids['positive']

    # Compute separation metrics
    neg_pos_sim = cosine_similarity(
        pole_centroids['negative'].unsqueeze(0).numpy(),
        pole_centroids['positive'].unsqueeze(0).numpy()
    )[0, 0]

    neg_neu_sim = cosine_similarity(
        pole_centroids['negative'].unsqueeze(0).numpy(),
        pole_centroids['neutral'].unsqueeze(0).numpy()
    )[0, 0]

    neu_pos_sim = cosine_similarity(
        pole_centroids['neutral'].unsqueeze(0).numpy(),
        pole_centroids['positive'].unsqueeze(0).numpy()
    )[0, 0]

    print(f"    Pole similarities:")
    print(f"      neg<->pos: {neg_pos_sim:.4f}")
    print(f"      neg<->neu: {neg_neu_sim:.4f}")
    print(f"      neu<->pos: {neu_pos_sim:.4f}")

    # Steering vector magnitudes
    print(f"    Steering vector magnitudes:")
    print(f"      neg→neu: {vectors['negative_to_neutral'].norm():.2f}")
    print(f"      neg→pos: {vectors['negative_to_positive'].norm():.2f}")

    vectors['_metrics'] = {
        'neg_pos_similarity': float(neg_pos_sim),
        'neg_neu_similarity': float(neg_neu_sim),
        'neu_pos_similarity': float(neu_pos_sim),
        'neg_to_neu_magnitude': float(vectors['negative_to_neutral'].norm()),
        'neg_to_pos_magnitude': float(vectors['negative_to_positive'].norm()),
    }

    return vectors


def test_steering_effect(
    model,
    tokenizer,
    steering_vector: torch.Tensor,
    test_prompt: str,
    layer_idx: int = 12,
    scales: List[float] = [0.0, 0.5, 1.0, 2.0]
):
    """Test steering effect on generation."""
    device = next(model.parameters()).device

    print(f"\n  Test prompt: '{test_prompt}'")

    for scale in scales:
        # Create steering hook
        def steering_hook(module, input, output):
            if scale == 0.0:
                return output
            hidden = output[0]
            # Add steering vector to last token
            hidden[:, -1, :] += scale * steering_vector.to(device).to(hidden.dtype)
            return (hidden,) + output[1:]

        # Register hook
        layer = model.model.layers[layer_idx]
        handle = layer.register_forward_hook(steering_hook)

        # Generate
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        handle.remove()

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = generated[len(test_prompt):].strip()
        print(f"    scale={scale:.1f}: {continuation[:60]}...")


def main():
    print("=" * 70)
    print("SIMPLEX STEERING VECTOR COMPUTATION")
    print("=" * 70)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load simplexes
    print("\n1. Loading S-tier simplexes...")
    simplexes = load_simplexes()
    print(f"   Found {len(simplexes)} simplexes")

    # Load model
    print("\n2. Loading model...")
    model_name = "google/gemma-3-4b-pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"   Model loaded on {device}")

    # Compute steering vectors for each simplex
    print("\n3. Computing steering vectors...")
    all_vectors = {}
    all_metrics = {}

    for simplex_name, simplex_def in simplexes.items():
        # Skip problematic names with /
        if '/' in simplex_name:
            print(f"\n  Skipping {simplex_name} (contains /)")
            continue

        try:
            vectors = compute_steering_vectors(
                model, tokenizer, simplex_name, simplex_def,
                layer_idx=12, n_samples=50
            )

            if vectors:
                # Extract metrics and save vectors
                metrics = vectors.pop('_metrics')
                all_metrics[simplex_name] = metrics

                # Save vectors
                simplex_dir = run_dir / simplex_name
                simplex_dir.mkdir(parents=True, exist_ok=True)

                for vec_name, vec in vectors.items():
                    torch.save(vec, simplex_dir / f"{vec_name}.pt")

                all_vectors[simplex_name] = vectors
                print(f"    ✓ Saved {len(vectors)} vectors")

        except Exception as e:
            print(f"    ✗ Error: {e}")

    # Save metrics
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n   Metrics saved to: {metrics_file}")

    # Quick steering test on one simplex
    print("\n4. Quick steering test (threat_perception)...")
    if 'threat_perception' in all_vectors:
        vec = all_vectors['threat_perception']['negative_to_positive']
        test_steering_effect(
            model, tokenizer, vec,
            "I feel very anxious and worried about",
            layer_idx=12,
            scales=[0.0, 1.0, 2.0, 5.0]
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n   Computed vectors for {len(all_vectors)} simplexes")
    print(f"   Output directory: {run_dir}")

    # Show average separation
    if all_metrics:
        avg_sep = np.mean([m['neg_pos_similarity'] for m in all_metrics.values()])
        print(f"\n   Average neg<->pos similarity: {avg_sep:.4f}")
        print("   (Lower = better pole separation)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
