"""
Phase 5: Semantic Steering Evaluation
=====================================

Goal: Validate that classifier F1 correlates with steering effectiveness using embedding-based semantic similarity.

Approach:
- Build three centroids (core, boundary, negative) using sentence embeddings
- Test steering at [-1.0, 0.0, +1.0] strengths
- Measure semantic shift: Δ = cos(text, core) − cos(text, neg)
- Generate human validation CSV for blind spot check

Expected: Higher F1 → stronger steering, |Δ| > 0.15 = meaningful shift
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from pathlib import Path
import argparse
import json
import time
import random
from typing import Dict, List, Tuple
import sys
import csv

sys.path.append(str(Path(__file__).parent.parent))

from scripts.stage_1_5_temporal import get_activation_sequence
from scripts.phase_4_neutral_training import (
    generate_definition_prompt,
    get_mean_activation,
    train_binary_classifier
)


def build_centroids(
    concept: str,
    concept_info: Dict,
    neutral_pool: List[str],
    embedding_model: SentenceTransformer,
    n_samples: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build three centroids for semantic shift measurement.

    Returns:
        core_centroid: Average embedding of definitional prompts
        boundary_centroid: Average embedding of relational prompts
        neg_centroid: Average embedding of negative prompts
    """
    # Core centroid: definitional prompts
    core_prompts = [
        f"What is {concept}?",
        f"Define {concept}.",
        f"{concept.capitalize()} is",
        f"Tell me about {concept}.",
        f"Explain {concept}."
    ][:n_samples]

    core_embeddings = embedding_model.encode(core_prompts)
    core_centroid = core_embeddings.mean(axis=0)

    # Boundary centroid: relational prompts
    related = concept_info.get('related', [])
    boundary_prompts = []

    if len(related) >= n_samples:
        sampled_related = random.sample(related, n_samples)
        for rel in sampled_related:
            boundary_prompts.append(f"{concept} is related to {rel}.")
    else:
        # Fallback: use core prompts if not enough relationships
        boundary_prompts = core_prompts[:n_samples]

    boundary_embeddings = embedding_model.encode(boundary_prompts)
    boundary_centroid = boundary_embeddings.mean(axis=0)

    # Negative centroid: distant concepts
    negatives = concept_info.get('negatives', [])
    if len(negatives) >= n_samples:
        sampled_negs = random.sample(negatives, n_samples)
        neg_prompts = [f"What is {neg}?" for neg in sampled_negs]
    else:
        # Fallback: use neutral pool
        sampled_neutrals = random.sample(neutral_pool, n_samples)
        neg_prompts = [f"What is {neut}?" for neut in sampled_neutrals]

    neg_embeddings = embedding_model.encode(neg_prompts)
    neg_centroid = neg_embeddings.mean(axis=0)

    return core_centroid, boundary_centroid, neg_centroid


def extract_steering_vector(
    model,
    tokenizer,
    concept: str,
    layer_idx: int = -1,
    device: str = "cuda"
) -> np.ndarray:
    """
    Extract concept vector from model activations.

    This matches Phase 2.5's approach: generate text about the concept
    and average the hidden states to get the concept direction.
    """
    concept_prompt = f"What is {concept}?"
    inputs = tokenizer(concept_prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )

        activations = []
        for step_states in outputs.hidden_states:
            if layer_idx == -1:
                last_layer = step_states[-1]
            else:
                last_layer = step_states[layer_idx]

            act = last_layer[0, -1, :]
            activations.append(act.cpu().numpy())

        # Average across generation steps
        concept_vector = np.stack(activations).mean(axis=0)

        # Normalize
        concept_vector = concept_vector / (np.linalg.norm(concept_vector) + 1e-8)

    return concept_vector


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: np.ndarray = None,
    strength: float = 0.0,
    layer_idx: int = -1,
    max_new_tokens: int = 50,
    device: str = "cuda"
) -> str:
    """
    Generate text with optional steering applied using forward hooks.

    If steering_vector is None or strength is 0.0, generates without steering.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if steering_vector is None or abs(strength) < 1e-6:
        # No steering
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Apply steering using forward hook
    concept_vector = torch.tensor(steering_vector, dtype=torch.float32).to(device)

    def steering_hook(module, input, output):
        """Project out concept vector from hidden states."""
        hidden_states = output[0]
        # Project onto concept vector and subtract scaled projection
        projection = (hidden_states @ concept_vector.unsqueeze(-1)) * concept_vector
        steered = hidden_states - strength * projection
        return (steered,)

    # Register hook on target layer
    if layer_idx == -1:
        target_layer = model.model.language_model.layers[-1]
    else:
        target_layer = model.model.language_model.layers[layer_idx]

    handle = target_layer.register_forward_hook(steering_hook)

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        handle.remove()

    return generated


def compute_semantic_shift(
    text: str,
    core_centroid: np.ndarray,
    neg_centroid: np.ndarray,
    embedding_model: SentenceTransformer
) -> float:
    """
    Compute semantic shift: Δ = cos(text, core) − cos(text, neg)

    Positive Δ: text is closer to core (positive steering working)
    Negative Δ: text is closer to negative (negative steering working)
    """
    text_embedding = embedding_model.encode([text])[0]

    core_sim = np.dot(text_embedding, core_centroid) / (
        np.linalg.norm(text_embedding) * np.linalg.norm(core_centroid) + 1e-8
    )

    neg_sim = np.dot(text_embedding, neg_centroid) / (
        np.linalg.norm(text_embedding) * np.linalg.norm(neg_centroid) + 1e-8
    )

    delta = core_sim - neg_sim

    return float(delta)


def run_phase_5(
    model_name: str = "google/gemma-3-4b-pt",
    concept_graph_path: str = "data/concept_graph/wordnet_v2_top10.json",
    phase4_results_path: str = "results/phase_4_neutral_training/phase4_results.json",
    output_dir: str = "results/phase_5_semantic_steering",
    device: str = "cuda",
    layer_idx: int = -1,
    n_concepts: int = 10
):
    """Run Phase 5: Semantic Steering Evaluation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PHASE 5: SEMANTIC STEERING EVALUATION")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Concepts: {n_concepts}")
    print(f"Steering strengths: [-1.0, 0.0, +1.0]")
    print(f"Metric: Δ = cos(text, core) − cos(text, neg)")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Load models
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map=device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Language model loaded\n")

    print("Loading sentence-transformers/all-MiniLM-L6-v2...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding_model.to(device)
    print("✓ Embedding model loaded\n")

    # Get hidden dim
    with torch.inference_mode():
        test_input = tokenizer("test", return_tensors="pt").to(device)
        test_output = model(**test_input, output_hidden_states=True)
        hidden_dim = test_output.hidden_states[layer_idx].shape[-1]
    print(f"Hidden dim: {hidden_dim}\n")

    # Load concept graph
    print(f"Loading concept graph...")
    with open(concept_graph_path) as f:
        data = json.load(f)

    concept_data = data['concepts']
    neutral_pool = data['neutral_pool']

    all_concepts = list(concept_data.keys())
    concepts = all_concepts[:n_concepts]
    print(f"✓ Loaded {len(concepts)} concepts\n")

    # Load Phase 4 results for F1 scores
    print("Loading Phase 4 results...")
    with open(phase4_results_path) as f:
        phase4_results = json.load(f)

    concept_f1_map = {}
    for result in phase4_results['per_concept']:
        concept_f1_map[result['concept']] = result['metrics']['f1']
    print(f"✓ Loaded F1 scores for {len(concept_f1_map)} concepts\n")

    # Run steering evaluation
    results = []
    human_validation_samples = []
    sample_id = 1

    for i, concept in enumerate(concepts):
        print(f"[{i+1}/{len(concepts)}] {concept}...")

        concept_start = time.time()
        concept_info = concept_data[concept]
        f1_score = concept_f1_map.get(concept, 0.0)

        # Build centroids
        core_centroid, boundary_centroid, neg_centroid = build_centroids(
            concept, concept_info, neutral_pool, embedding_model, n_samples=5
        )

        # Train classifier (to extract steering vector)
        print(f"  Training classifier for steering vector extraction...", end=" ", flush=True)

        # Sample training data (1×1×1 like Phase 4)
        train_prompts = [
            generate_definition_prompt(concept),  # 1 positive
            generate_definition_prompt(random.choice(concept_info['negatives'])),  # 1 negative
            f"What is {random.choice(neutral_pool)}?"  # 1 neutral
        ]
        train_labels = [1, 0, 0]

        X_train = []
        for prompt in train_prompts:
            act = get_mean_activation(model, tokenizer, prompt, device, layer_idx)
            X_train.append(act)
        X_train = np.array(X_train)
        y_train = np.array(train_labels)

        classifier = train_binary_classifier(
            X_train, y_train, hidden_dim,
            intermediate_dim=128, lr=0.001, epochs=100, device=device
        )

        # Extract steering vector from model activations (not classifier weights)
        steering_vector = extract_steering_vector(
            model, tokenizer, concept, layer_idx=layer_idx, device=device
        )
        print("✓")

        # Test steering at different strengths
        test_prompts = [
            f"Tell me about {concept}.",
            f"Explain {concept}.",
            f"What is {concept}?"
        ]

        # Use 0.25 intervals - extreme values (±1.0) cause model collapse
        strengths = [-0.5, -0.25, 0.0, 0.25, 0.5]

        for prompt in test_prompts:
            for strength in strengths:
                # Generate with steering
                # NOTE: Currently generates without steering (limitation documented)
                generated = generate_with_steering(
                    model, tokenizer, prompt,
                    steering_vector=steering_vector if strength != 0.0 else None,
                    strength=strength,
                    layer_idx=layer_idx,
                    max_new_tokens=50,
                    device=device
                )

                # Remove prompt from generated text
                if generated.startswith(prompt):
                    generated = generated[len(prompt):].strip()

                # Compute semantic shift
                delta = compute_semantic_shift(
                    generated, core_centroid, neg_centroid, embedding_model
                )

                results.append({
                    'concept': concept,
                    'f1_score': f1_score,
                    'prompt': prompt,
                    'steering_strength': strength,
                    'generated_text': generated,
                    'delta': delta
                })

                # Add to human validation pool (random 50 samples)
                if random.random() < 0.5 and len(human_validation_samples) < 50:
                    human_validation_samples.append({
                        'sample_id': f"{sample_id:03d}",
                        'concept': 'REDACTED',  # Will reveal after rating
                        'generated_text': generated,
                        'strength': strength,
                        'delta': delta,
                        'actual_concept': concept
                    })
                    sample_id += 1

        elapsed = time.time() - concept_start

        # Show quick stats
        concept_results = [r for r in results if r['concept'] == concept]
        mean_deltas = {
            -1.0: np.mean([r['delta'] for r in concept_results if r['steering_strength'] == -1.0]),
            0.0: np.mean([r['delta'] for r in concept_results if r['steering_strength'] == 0.0]),
            1.0: np.mean([r['delta'] for r in concept_results if r['steering_strength'] == 1.0])
        }

        print(f"  Δ: neg={mean_deltas[-1.0]:.3f}, neutral={mean_deltas[0.0]:.3f}, pos={mean_deltas[1.0]:.3f} ({elapsed:.1f}s)")

    total_time = time.time() - start_time

    # Aggregate analysis
    print(f"\n{'='*70}")
    print(f"PHASE 5 COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes\n")

    # Aggregate by steering strength
    for strength in [-1.0, 0.0, 1.0]:
        strength_results = [r for r in results if r['steering_strength'] == strength]
        mean_delta = np.mean([r['delta'] for r in strength_results])
        std_delta = np.std([r['delta'] for r in strength_results])

        label = {-1.0: "Negative", 0.0: "Neutral", 1.0: "Positive"}[strength]
        print(f"{label} steering (strength={strength:+.1f}):")
        print(f"  Mean Δ: {mean_delta:.3f} ± {std_delta:.3f}")
        print(f"  Samples: {len(strength_results)}")
        print()

    # Check steering direction correctness
    correct_direction = 0
    total_steered = 0

    for concept in concepts:
        concept_results = [r for r in results if r['concept'] == concept]

        neg_delta = np.mean([r['delta'] for r in concept_results if r['steering_strength'] == -1.0])
        neutral_delta = np.mean([r['delta'] for r in concept_results if r['steering_strength'] == 0.0])
        pos_delta = np.mean([r['delta'] for r in concept_results if r['steering_strength'] == 1.0])

        # Expect: neg < neutral < pos
        if neg_delta < neutral_delta and neutral_delta < pos_delta:
            correct_direction += 1
        total_steered += 1

    direction_accuracy = correct_direction / total_steered if total_steered > 0 else 0.0
    print(f"Steering Direction Accuracy: {direction_accuracy:.1%} ({correct_direction}/{total_steered})")
    print()

    # F1 vs steering magnitude correlation
    concept_stats = []
    for concept in concepts:
        concept_results = [r for r in results if r['concept'] == concept]

        neutral_delta = np.mean([r['delta'] for r in concept_results if r['steering_strength'] == 0.0])
        pos_delta = np.mean([r['delta'] for r in concept_results if r['steering_strength'] == 1.0])

        steering_magnitude = abs(pos_delta - neutral_delta)
        f1 = concept_f1_map.get(concept, 0.0)

        concept_stats.append({
            'concept': concept,
            'f1': f1,
            'steering_magnitude': steering_magnitude
        })

    f1_values = [s['f1'] for s in concept_stats]
    magnitude_values = [s['steering_magnitude'] for s in concept_stats]

    correlation = np.corrcoef(f1_values, magnitude_values)[0, 1]
    print(f"F1 vs Steering Magnitude Correlation: r = {correlation:.3f}")
    print()

    # Save results
    output_file = output_dir / "steering_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'model': model_name,
                'n_concepts': n_concepts,
                'steering_strengths': [-1.0, 0.0, 1.0],
                'device': device,
                'layer_idx': layer_idx,
                'hidden_dim': hidden_dim,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
            },
            'aggregate': {
                'direction_accuracy': direction_accuracy,
                'f1_steering_correlation': float(correlation)
            },
            'per_concept_stats': concept_stats,
            'all_results': results,
            'time_seconds': total_time
        }, f, indent=2)

    print(f"✓ Results saved to: {output_file}\n")

    # Save human validation CSV
    if human_validation_samples:
        validation_file = output_dir / "human_validation.csv"
        with open(validation_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_id', 'concept', 'generated_text', 'human_rating'])
            writer.writeheader()

            for sample in human_validation_samples:
                writer.writerow({
                    'sample_id': sample['sample_id'],
                    'concept': sample['concept'],  # REDACTED
                    'generated_text': sample['generated_text'],
                    'human_rating': ''  # To be filled by human rater
                })

        print(f"✓ Human validation CSV saved to: {validation_file}")
        print(f"  {len(human_validation_samples)} samples for blind rating\n")

        # Save answer key separately
        answer_key_file = output_dir / "human_validation_answers.json"
        with open(answer_key_file, 'w') as f:
            json.dump(human_validation_samples, f, indent=2)
        print(f"✓ Answer key saved to: {answer_key_file}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='google/gemma-3-4b-pt')
    parser.add_argument('--concept-graph', default='data/concept_graph/wordnet_v2_top10.json')
    parser.add_argument('--phase4-results', default='results/phase_4_neutral_training/phase4_results.json')
    parser.add_argument('--output-dir', default='results/phase_5_semantic_steering')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--layer-idx', type=int, default=-1)
    parser.add_argument('--n-concepts', type=int, default=10)
    args = parser.parse_args()

    run_phase_5(
        model_name=args.model,
        concept_graph_path=args.concept_graph,
        phase4_results_path=args.phase4_results,
        output_dir=args.output_dir,
        device=args.device,
        layer_idx=args.layer_idx,
        n_concepts=args.n_concepts
    )
