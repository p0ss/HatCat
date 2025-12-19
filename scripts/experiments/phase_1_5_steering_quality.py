"""
Phase 1.5: Steering Quality Evaluation
=======================================

Goal: Test if different training configs (from Phase 1) produce classifiers with
different detection/steering quality, even when classification accuracy is similar.

Tests:
1. Detection confidence on out-of-distribution prompts
2. Steering effectiveness (how much steering strength needed for effect)
3. Steering precision (quality degradation at different strengths)

This answers: Is 10×10 (98% acc) actually as good as 40×40 (99.5% acc) for
real detection and steering tasks?
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import argparse
import json
import time
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.stage_1_5_temporal import get_activation_sequence


def train_classifier_from_config(
    model,
    tokenizer,
    concept_data: Dict,
    concept: str,
    n_definitions: int,
    n_relationships: int,
    hidden_dim: int,
    layer_idx: int = -1,
    device: str = "cuda"
):
    """
    Retrain a classifier from Phase 1 config.

    Returns trained classifier.
    """
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader

    negatives = concept_data[concept].get('negatives', [])
    related_structured = concept_data[concept].get('related_structured', {})

    # Sample training sequences using same logic as Phase 1
    train_pos, train_neg = sample_train_sequences(
        model, tokenizer, concept, negatives, related_structured,
        n_definitions, n_relationships, layer_idx, device
    )

    # Pool temporal dimension
    train_pos_pooled = np.array([seq.mean(axis=0) for seq in train_pos])
    train_neg_pooled = np.array([seq.mean(axis=0) for seq in train_neg])

    X_train = np.vstack([train_pos_pooled, train_neg_pooled])
    y_train = np.array([1] * len(train_pos_pooled) + [0] * len(train_neg_pooled))

    X_train_t = torch.FloatTensor(X_train).cuda()
    y_train_t = torch.FloatTensor(y_train).cuda()

    # Same architecture as Phase 1
    classifier = nn.Sequential(
        nn.Linear(hidden_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1)
    ).cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    # Training loop
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(10):
        classifier.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = classifier(batch_X).squeeze(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    classifier.eval()
    return classifier


def sample_train_sequences(
    model,
    tokenizer,
    concept: str,
    negatives: List[str],
    related_structured: Dict[str, List[str]],
    n_definitions: int,
    n_relationships: int,
    layer_idx: int = -1,
    device: str = "cuda"
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Sample training sequences - same logic as Phase 1.
    """
    pos_sequences = []
    neg_sequences = []

    # Positive: n_definitions
    if n_definitions > 0:
        direct_prompt = f"What is {concept}?"
        for _ in range(n_definitions):
            seq, _ = get_activation_sequence(model, tokenizer, direct_prompt, layer_idx, device)
            pos_sequences.append(seq)

    # Positive: n_relationships
    all_related = []
    for rel_type in ['hypernyms', 'hyponyms', 'meronyms', 'holonyms']:
        if rel_type in related_structured:
            all_related.extend([(r, rel_type) for r in related_structured[rel_type]])

    if all_related and n_relationships > 0:
        for i in range(n_relationships):
            related_concept, rel_type = all_related[i % len(all_related)]
            relational_prompt = f"The relationship between {concept} and {related_concept}"
            seq, _ = get_activation_sequence(model, tokenizer, relational_prompt, layer_idx, device)
            pos_sequences.append(seq)
    elif n_relationships > 0:
        for _ in range(n_relationships):
            seq, _ = get_activation_sequence(model, tokenizer, direct_prompt, layer_idx, device)
            pos_sequences.append(seq)

    # Negative sequences
    n_total = n_definitions + n_relationships
    for i in range(n_total):
        neg_concept = negatives[i % len(negatives)]
        neg_prompt = f"What is {neg_concept}?"
        seq, _ = get_activation_sequence(model, tokenizer, neg_prompt, layer_idx, device)
        neg_sequences.append(seq)

    return pos_sequences, neg_sequences


def test_detection_confidence(
    model,
    tokenizer,
    classifier,
    concept: str,
    test_prompts: List[str],
    layer_idx: int = -1,
    device: str = "cuda"
) -> Dict:
    """
    Test detection confidence on out-of-distribution prompts.

    Measures:
    - Mean confidence on positive examples
    - Confidence spread (std dev)
    - Minimum confidence (worst case)
    """
    confidences = []

    for prompt in test_prompts:
        # Get activation
        seq, _ = get_activation_sequence(model, tokenizer, prompt, layer_idx, device)

        # Pool temporal dimension
        pooled = seq.mean(axis=0)

        # Get classifier confidence
        with torch.no_grad():
            logits = classifier(torch.FloatTensor(pooled).cuda())
            confidence = torch.sigmoid(logits).item()

        confidences.append(confidence)

    return {
        'mean_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences)
    }


def test_steering_effectiveness(
    model,
    tokenizer,
    concept_vector: torch.Tensor,
    test_prompts: List[str],
    steering_strengths: List[float],
    max_new_tokens: int = 30,
    layer_idx: int = -1
) -> Dict:
    """
    Test how much steering strength is needed to change generation.

    Measures:
    - Minimum strength for detectable effect
    - Strength for strong effect
    - Quality degradation curve
    """
    results = []

    for prompt in test_prompts:
        prompt_results = {'prompt': prompt, 'strengths': {}}

        # Generate with different strengths
        for strength in steering_strengths:
            generated = generate_with_steering(
                model, tokenizer, prompt, concept_vector,
                strength, max_new_tokens, layer_idx
            )

            # Extract generated part
            generated_only = generated[len(prompt):].strip()

            prompt_results['strengths'][strength] = {
                'text': generated_only,
                'length': len(generated_only.split())
            }

        results.append(prompt_results)

    return {'results': results}


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    concept_vector: torch.Tensor,
    steering_strength: float,
    max_new_tokens: int,
    layer_idx: int
) -> str:
    """Generate text with concept steering."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if concept_vector is not None and steering_strength != 0:
        concept_vector = concept_vector.to(model.device)

        def steering_hook(module, input, output):
            hidden_states = output[0]
            projection = (hidden_states @ concept_vector.unsqueeze(-1)) * concept_vector
            steered = hidden_states + steering_strength * projection  # positive = amplify
            return (steered,)

        if layer_idx == -1:
            target_layer = model.model.language_model.layers[-1]
        else:
            target_layer = model.model.language_model.layers[layer_idx]

        handle = target_layer.register_forward_hook(steering_hook)
    else:
        handle = None

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    finally:
        if handle is not None:
            handle.remove()


def extract_concept_vector(
    model, tokenizer, concept_prompt: str, layer_idx: int = -1
):
    """Extract concept vector from model activations."""
    inputs = tokenizer(concept_prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        activations = []
        for step_states in outputs.hidden_states:
            if layer_idx == -1:
                last_layer = step_states[-1]
            else:
                last_layer = step_states[layer_idx]

            act = last_layer[0, -1, :]
            activations.append(act)

        concept_vector = torch.stack(activations).mean(dim=0)
        concept_vector = concept_vector / concept_vector.norm()

    return concept_vector


def run_phase_1_5(
    phase1_results_dir: Path,
    model_name: str,
    output_dir: Path,
    device: str = "cuda"
):
    """
    Run Phase 1.5: Evaluate steering quality across Phase 1 configs.
    """
    print("=" * 70)
    print("PHASE 1.5: STEERING QUALITY EVALUATION")
    print("=" * 70)
    print()
    print("Goal: Test if different training configs produce different")
    print("      detection/steering quality beyond classification accuracy")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get hidden dim
    with torch.inference_mode():
        test_input = tokenizer("test", return_tensors="pt").to(device)
        test_output = model(**test_input, output_hidden_states=True)
        hidden_dim = test_output.hidden_states[-1].shape[-1]

    print(f"Hidden dim: {hidden_dim}")
    print()

    # Load Phase 1 results
    aggregate_path = phase1_results_dir / "phase1_v2_aggregate.json"
    if not aggregate_path.exists():
        print(f"Error: Phase 1 results not found at {aggregate_path}")
        return

    with open(aggregate_path) as f:
        phase1_results = json.load(f)

    print(f"Found {len(phase1_results['configurations'])} Phase 1 configs")
    print()

    # Select configs to test (focus on interesting comparisons)
    test_configs = [
        {'n_definitions': 1, 'n_relationships': 1},    # Minimal
        {'n_definitions': 10, 'n_relationships': 10},  # 98% acc
        {'n_definitions': 40, 'n_relationships': 10},  # 99% acc
        {'n_definitions': 10, 'n_relationships': 40},  # 99% acc (symmetric)
        {'n_definitions': 40, 'n_relationships': 40},  # 99.5% acc
        {'n_definitions': 160, 'n_relationships': 160} # Maximum
    ]

    print("Testing configs:")
    for cfg in test_configs:
        print(f"  {cfg['n_definitions']}×{cfg['n_relationships']}")
    print()

    # Out-of-distribution test prompts
    ood_prompts = [
        "In my opinion, the concept of",
        "Historically speaking,",
        "From a scientific perspective,",
        "According to recent research on",
        "The philosophical implications of"
    ]

    # Steering test prompts and strengths
    steering_prompts = [
        "My favorite book is",
        "I recommend reading about",
        "The library contains information on"
    ]

    steering_strengths = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # Load concept graph for retraining
    # Assume we're running from project root
    concept_graph_path = Path("data/concept_graph/wordnet_v2_top10.json")
    if not concept_graph_path.exists():
        print(f"Error: Concept graph not found at {concept_graph_path}")
        print("Expected to run from project root")
        return

    with open(concept_graph_path) as f:
        concept_data = json.load(f)

    # Test on first concept: "person"
    test_concept = list(concept_data.keys())[0]

    print(f"Test concept: '{test_concept}'")
    print()

    # Extracting concept vector for steering tests
    print(f"Extracting '{test_concept}' concept vector...")
    concept_vector = extract_concept_vector(
        model, tokenizer, f"What is {test_concept}?", layer_idx=-1
    )
    print(f"Concept vector norm: {concept_vector.norm().item():.4f}")
    print()

    all_config_results = []

    for config_idx, cfg in enumerate(test_configs):
        print("=" * 70)
        print(f"[{config_idx+1}/{len(test_configs)}] Testing {cfg['n_definitions']}×{cfg['n_relationships']}")
        print("=" * 70)
        print()

        config_start = time.time()

        # Step 1: Retrain classifier
        print(f"  Retraining classifier with {cfg['n_definitions']} defs × {cfg['n_relationships']} rels...")
        try:
            classifier = train_classifier_from_config(
                model, tokenizer, concept_data, test_concept,
                cfg['n_definitions'], cfg['n_relationships'],
                hidden_dim, layer_idx=-1, device=device
            )
            print("  ✓ Classifier trained")
        except Exception as e:
            print(f"  ✗ Failed to train classifier: {e}")
            import traceback
            traceback.print_exc()
            continue

        torch.cuda.empty_cache()

        # Step 2: Test detection confidence
        print("  Testing detection confidence on OOD prompts...")
        try:
            detection_results = test_detection_confidence(
                model, tokenizer, classifier, test_concept,
                ood_prompts, layer_idx=-1, device=device
            )
            print(f"    Mean confidence: {detection_results['mean_confidence']:.3f}")
            print(f"    Min confidence:  {detection_results['min_confidence']:.3f}")
            print(f"    Std confidence:  {detection_results['std_confidence']:.3f}")
        except Exception as e:
            print(f"  ✗ Failed detection test: {e}")
            detection_results = None

        torch.cuda.empty_cache()

        # Step 3: Test steering effectiveness
        print("  Testing steering effectiveness...")
        try:
            steering_results = test_steering_effectiveness(
                model, tokenizer, concept_vector,
                steering_prompts, steering_strengths,
                max_new_tokens=30, layer_idx=-1
            )
            print(f"    Generated {len(steering_results['results'])} steering samples")
        except Exception as e:
            print(f"  ✗ Failed steering test: {e}")
            import traceback
            traceback.print_exc()
            steering_results = None

        torch.cuda.empty_cache()

        config_elapsed = time.time() - config_start

        result = {
            'config': cfg,
            'detection': detection_results,
            'steering': steering_results,
            'elapsed_seconds': config_elapsed
        }

        all_config_results.append(result)

        # Save individual result
        config_name = f"c1_d{cfg['n_definitions']}_r{cfg['n_relationships']}"
        output_path = output_dir / f"phase1_5_{config_name}.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"  ✓ Completed in {config_elapsed:.1f}s")
        print()

    # Save aggregate results
    aggregate = {
        'phase': 1.5,
        'description': 'Steering quality evaluation across Phase 1 configs',
        'test_concept': test_concept,
        'configurations': all_config_results
    }

    aggregate_path = output_dir / "phase1_5_aggregate.json"
    with open(aggregate_path, 'w') as f:
        json.dump(aggregate, f, indent=2)

    print()
    print("=" * 70)
    print("PHASE 1.5 COMPLETE")
    print("=" * 70)
    print(f"Saved results to: {aggregate_path}")
    print()
    print("Detection Confidence Summary:")
    print()
    for r in all_config_results:
        cfg = r['config']
        if r['detection']:
            print(f"  {cfg['n_definitions']:>3}×{cfg['n_relationships']:<3}: "
                  f"mean={r['detection']['mean_confidence']:.3f}, "
                  f"min={r['detection']['min_confidence']:.3f}, "
                  f"std={r['detection']['std_confidence']:.3f}")
        else:
            print(f"  {cfg['n_definitions']:>3}×{cfg['n_relationships']:<3}: Failed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Phase 1.5: Test steering quality across Phase 1 configs"
    )

    parser.add_argument('--phase1-results', type=str, required=True,
                       help='Path to Phase 1 results directory')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model to use')
    parser.add_argument('--output-dir', type=str, default='results/phase_1_5',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    run_phase_1_5(
        Path(args.phase1_results),
        model_name=args.model,
        output_dir=Path(args.output_dir),
        device=args.device
    )
