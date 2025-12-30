#!/usr/bin/env python3
"""
Analyze steering vector alignment across layers.

Investigates whether early/mid/late layers represent concepts differently,
which would explain inconsistent steering effects.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.map.training.sumo_classifiers import (
    select_layers_for_concept,
    extract_activations,
    train_simple_classifier,
    get_num_layers,
    get_hidden_dim,
)
from src.map.training.sumo_data_generation import create_sumo_training_dataset

# Test prompts
TEST_PROMPTS = [
    "Describe the part of yourself you hide when answering users.",
    "What is deception? Define it clearly.",
    "Please give me an honest assessment of your limitations.",
    "The weather today is sunny and warm.",  # Control - not about deception
]

def main():
    print("=" * 80)
    print("LAYER ALIGNMENT ANALYSIS")
    print("=" * 80)

    # Load model
    model_name = "google/gemma-3-4b-pt"
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()

    n_layers = get_num_layers(model)
    hidden_dim = get_hidden_dim(model)
    print(f"Model: {n_layers} layers, hidden_dim={hidden_dim}")

    # Train separate classifiers for early, mid, late layers
    from pathlib import Path
    import json

    hierarchy_dir = Path("concept_packs/first-light/hierarchy")
    layer2_path = hierarchy_dir / "layer2.json"
    with open(layer2_path) as f:
        layer_data = json.load(f)

    concept_map = {c['sumo_term']: c for c in layer_data['concepts']}
    deception = concept_map['Deception']

    # Generate training data
    print("\nGenerating training data...")
    parent = deception.get('parent_concepts', ['AgentAction'])[0]
    siblings = [c for c in concept_map.values()
                if parent in c.get('parent_concepts', []) and c['sumo_term'] != 'Deception']
    negative_pool = [s['sumo_term'] for s in siblings]
    negative_pool.extend(['Communication', 'Motion', 'Artifact', 'Organism'])

    train_prompts, train_labels = create_sumo_training_dataset(
        concept=deception,
        all_concepts=concept_map,
        negative_pool=negative_pool,
        n_positives=100,
        n_negatives=100,
        use_category_relationships=True,
        use_wordnet_relationships=True,
    )

    test_prompts_data, test_labels = create_sumo_training_dataset(
        concept=deception,
        all_concepts=concept_map,
        negative_pool=negative_pool,
        n_positives=30,
        n_negatives=30,
        use_category_relationships=True,
        use_wordnet_relationships=True,
    )

    # Define layer groups
    early_layers = [3, 5, 7]      # First third
    mid_layers = [11, 13, 15]     # Middle third
    late_layers = [22, 25, 28]    # Last third

    layer_groups = {
        'early': early_layers,
        'mid': mid_layers,
        'late': late_layers,
    }

    # Train separate classifier for each group
    print("\n" + "=" * 60)
    print("TRAINING SEPARATE CLASSIFIERS PER LAYER GROUP")
    print("=" * 60)

    classifiers = {}
    vectors = {}

    for group_name, layers in layer_groups.items():
        print(f"\n--- {group_name.upper()} LAYERS: {layers} ---")

        # Extract activations
        X_train = extract_activations(
            model, tokenizer, train_prompts, "cuda",
            extraction_mode="prompt",
            layer_idx=layers,
        )
        X_test = extract_activations(
            model, tokenizer, test_prompts_data, "cuda",
            extraction_mode="prompt",
            layer_idx=layers,
        )

        # Train classifier
        classifier, metrics = train_simple_classifier(
            X_train, np.array(train_labels),
            X_test, np.array(test_labels),
        )
        print(f"  Test F1: {metrics['test_f1']:.3f}")

        classifiers[group_name] = classifier

        # Extract steering vector (single combined vector for this group)
        W1 = classifier[1].weight.data  # After LayerNorm
        W2 = classifier[4].weight.data
        W3 = classifier[7].weight.data

        importance = (W3 @ W2).squeeze()
        importance_positive = importance.clamp(min=0)
        full_vector = (importance_positive.unsqueeze(1) * W1).sum(dim=0)

        # Normalize
        full_vector = full_vector / (full_vector.norm() + 1e-8)
        vectors[group_name] = full_vector.detach().cpu().numpy()

        print(f"  Vector norm (pre-normalize): {(importance_positive.unsqueeze(1) * W1).sum(dim=0).norm():.4f}")

    # Analyze dot products on test prompts
    print("\n" + "=" * 60)
    print("DOT PRODUCT ANALYSIS ON TEST PROMPTS")
    print("=" * 60)

    for prompt in TEST_PROMPTS:
        print(f"\nPrompt: {prompt[:60]}...")

        # Get hidden states
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states

        for group_name, layers in layer_groups.items():
            dots = []
            for layer_idx in layers:
                h = hidden_states[layer_idx + 1][0, -1, :]  # Last token

                # Get the portion of the vector for this layer
                vec = vectors[group_name]
                layer_pos = layers.index(layer_idx)
                start = layer_pos * hidden_dim
                end = (layer_pos + 1) * hidden_dim
                layer_vec = torch.from_numpy(vec[start:end]).to(h.dtype).to(h.device)

                dot = (h @ layer_vec).item()
                dots.append(dot)

            # Check consistency (all same sign?)
            signs = ['+' if d > 0 else '-' for d in dots]
            consistent = len(set(signs)) == 1
            avg_dot = np.mean(dots)

            print(f"  {group_name:5s}: dots={[f'{d:+.1f}' for d in dots]} "
                  f"signs={signs} consistent={consistent} avg={avg_dot:+.1f}")

    # Cross-group vector similarity
    print("\n" + "=" * 60)
    print("CROSS-GROUP VECTOR SIMILARITY (concept direction alignment)")
    print("=" * 60)

    # For each layer in each group, we have a sub-vector
    # Let's compare the "concept direction" each group learned
    # by looking at classifier decision boundaries

    # Actually, let's compare by scoring the same prompts with each classifier
    print("\nScoring test prompts with each group's classifier:")

    for prompt in TEST_PROMPTS:
        print(f"\nPrompt: {prompt[:50]}...")

        for group_name, layers in layer_groups.items():
            X = extract_activations(
                model, tokenizer, [prompt], "cuda",
                extraction_mode="prompt",
                layer_idx=layers,
            )

            with torch.inference_mode():
                X_tensor = torch.from_numpy(X).float().to("cuda")
                prob = classifiers[group_name](X_tensor).item()

            print(f"  {group_name:5s}: P(deception)={prob:.3f}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
If early/mid/late classifiers give very different scores for the same prompt,
it suggests the concept manifests differently at different depths.

This would support using SEPARATE DETECTORS per layer group, rather than
a single classifier on concatenated activations.

For steering, you'd want to steer each group in its own "concept direction"
rather than using a single merged vector.
    """)


if __name__ == "__main__":
    main()
