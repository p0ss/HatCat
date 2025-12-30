"""
Phase 2.5: Steering Quality Evaluation for Phase 2 Scale Test
==============================================================

Goal: Evaluate detection and steering quality for concepts trained with 1×1
minimal training from Phase 2 scale test.

Tests:
1. Detection confidence on out-of-distribution prompts
2. Steering effectiveness (how much steering strength needed for effect)
3. Steering precision (quality degradation at different strengths)

This answers: How well do 1×1 trained classifiers perform on real detection
and steering tasks at different scales?
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
from scripts.phase_1_find_curve_v2 import sample_train_sequences, train_and_evaluate


def train_classifier_for_concept(
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
    Train a classifier for a concept.

    Returns trained classifier or None if training fails.
    """
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader

    negatives = concept_data[concept].get('negatives', [])
    related_structured = concept_data[concept].get('related_structured', {})

    if len(negatives) == 0:
        return None

    # Sample training sequences
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
    concept: str,
    concept_data: Dict,
    test_prompts: List[str],
    steering_strengths: List[float],
    max_new_tokens: int = 30,
    layer_idx: int = -1
) -> Dict:
    """
    Test how much steering strength is needed to change generation.

    Measures:
    - Generation at different steering strengths
    - Text length changes
    - Concept mention count (case-insensitive)
    - Related semantic term mentions (from WordNet relations)
    """
    results = []

    # Prepare concept terms for counting
    concept_lower = concept.lower()
    concept_words = set(concept_lower.split())

    # Extract related semantic terms from WordNet relations
    related_terms = set([concept_lower])
    antonym_terms = set()
    related_structured = concept_data.get('related_structured', {})

    # Add hypernyms (broader categories)
    for hypernym in related_structured.get('hypernyms', [])[:5]:  # Top 5 hypernyms
        related_terms.add(hypernym.lower())

    # Add hyponyms (specific instances)
    for hyponym in related_structured.get('hyponyms', [])[:10]:  # Top 10 hyponyms
        related_terms.add(hyponym.lower())

    # Add holonyms (wholes)
    for holonym in related_structured.get('holonyms', [])[:5]:
        related_terms.add(holonym.lower())

    # Add meronyms (parts)
    for meronym in related_structured.get('meronyms', [])[:5]:
        related_terms.add(meronym.lower())

    # Add antonyms (opposites) - tracked separately for negative steering
    for antonym in related_structured.get('antonyms', [])[:10]:  # Top 10 antonyms
        antonym_terms.add(antonym.lower())

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
            generated_lower = generated_only.lower()

            # Count exact concept mentions
            concept_count = generated_lower.count(concept_lower)

            # Count individual word occurrences for multi-word concepts
            word_counts = {}
            for word in concept_words:
                if len(word) > 2:  # Skip very short words
                    word_counts[word] = generated_lower.count(word)

            # Count related semantic term mentions
            semantic_count = 0
            semantic_matches = []
            for term in related_terms:
                count = generated_lower.count(term)
                if count > 0:
                    semantic_count += count
                    semantic_matches.append({'term': term, 'count': count})

            # Count antonym mentions (for negative steering analysis)
            antonym_count = 0
            antonym_matches = []
            for term in antonym_terms:
                count = generated_lower.count(term)
                if count > 0:
                    antonym_count += count
                    antonym_matches.append({'term': term, 'count': count})

            prompt_results['strengths'][str(strength)] = {
                'text': generated_only,
                'length': len(generated_only.split()),
                'concept_mentions': concept_count,
                'word_mentions': word_counts,
                'semantic_mentions': semantic_count,
                'semantic_matches': semantic_matches,
                'antonym_mentions': antonym_count,
                'antonym_matches': antonym_matches
            }

        results.append(prompt_results)

    return {
        'results': results,
        'related_terms': list(related_terms),
        'antonym_terms': list(antonym_terms)
    }


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


def run_phase_2_5(
    phase2_results_dir: Path,
    concept_graph_path: Path,
    model_name: str,
    output_dir: Path,
    scale: int = 100,
    n_test_concepts: int = 10,
    device: str = "cuda"
):
    """
    Run Phase 2.5: Evaluate steering quality for Phase 2 concepts.

    Args:
        phase2_results_dir: Phase 2 results directory
        concept_graph_path: Path to concept graph
        scale: Which scale to evaluate (default: 100)
        n_test_concepts: Number of concepts to test (default: 10)
    """
    print("=" * 70)
    print("PHASE 2.5: STEERING QUALITY EVALUATION FOR PHASE 2")
    print("=" * 70)
    print()
    print(f"Goal: Test detection and steering quality for {scale} concepts")
    print(f"      trained with 1×1 minimal training from Phase 2")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 2 results for the specified scale
    scale_results_path = phase2_results_dir / f"scale_n{scale}" / f"scale_test_n{scale}.json"
    if not scale_results_path.exists():
        print(f"Error: Phase 2 results not found at {scale_results_path}")
        return

    with open(scale_results_path) as f:
        phase2_results = json.load(f)

    print(f"Loaded Phase 2 results: {len(phase2_results['results'])} concepts")
    print()

    # Select top N concepts by test accuracy for evaluation
    sorted_results = sorted(phase2_results['results'], key=lambda x: x['test_acc'], reverse=True)
    test_concepts = [r['concept'] for r in sorted_results[:n_test_concepts]]

    print(f"Selected {len(test_concepts)} concepts for evaluation:")
    for i, concept in enumerate(test_concepts):
        test_acc = next(r['test_acc'] for r in sorted_results if r['concept'] == concept)
        print(f"  {i+1}. {concept} ({test_acc:.1%} test acc)")
    print()

    # Load concept graph
    if not concept_graph_path.exists():
        print(f"Error: Concept graph not found at {concept_graph_path}")
        return

    with open(concept_graph_path) as f:
        concept_data = json.load(f)

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
    print(f"✓ Model loaded\n")

    # Out-of-distribution test prompts
    ood_prompts_template = [
        "In my opinion, {concept} is",
        "Historically speaking, {concept}",
        "From a scientific perspective on {concept}",
        "According to research about {concept}",
        "The philosophical implications of {concept}"
    ]

    # Steering test prompts (concept will be inserted)
    steering_prompts_template = [
        "Tell me about {concept}.",
        "I'm interested in learning about {concept}.",
        "The most important thing to know about {concept} is"
    ]

    # More granular steering strengths
    steering_strengths = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

    all_concept_results = []

    for concept_idx, concept in enumerate(test_concepts):
        print("=" * 70)
        print(f"[{concept_idx+1}/{len(test_concepts)}] Testing: {concept}")
        print("=" * 70)
        print()

        concept_start = time.time()

        # Step 1: Train classifier with 1×1 config
        print(f"  Training classifier (1×1)...")
        try:
            classifier = train_classifier_for_concept(
                model, tokenizer, concept_data, concept,
                n_definitions=1, n_relationships=1,
                hidden_dim=hidden_dim, layer_idx=-1, device=device
            )

            if classifier is None:
                print(f"  ✗ No negatives available for '{concept}', skipping")
                continue

            print("  ✓ Classifier trained")
        except Exception as e:
            print(f"  ✗ Failed to train classifier: {e}")
            import traceback
            traceback.print_exc()
            continue

        torch.cuda.empty_cache()

        # Generate concept-specific prompts
        ood_prompts = [p.format(concept=concept) for p in ood_prompts_template]

        # Step 2: Test detection confidence
        print("  Testing detection confidence on OOD prompts...")
        try:
            detection_results = test_detection_confidence(
                model, tokenizer, classifier, concept,
                ood_prompts, layer_idx=-1, device=device
            )
            print(f"    Mean confidence: {detection_results['mean_confidence']:.3f}")
            print(f"    Min confidence:  {detection_results['min_confidence']:.3f}")
            print(f"    Std confidence:  {detection_results['std_confidence']:.3f}")
        except Exception as e:
            print(f"  ✗ Failed detection test: {e}")
            import traceback
            traceback.print_exc()
            detection_results = None

        torch.cuda.empty_cache()

        # Step 3: Extract concept vector
        print(f"  Extracting concept vector...")
        try:
            concept_vector = extract_concept_vector(
                model, tokenizer, f"What is {concept}?", layer_idx=-1
            )
            print(f"    Vector norm: {concept_vector.norm().item():.4f}")
        except Exception as e:
            print(f"  ✗ Failed to extract concept vector: {e}")
            concept_vector = None

        # Step 4: Test steering effectiveness
        steering_results = None
        if concept_vector is not None:
            print("  Testing steering effectiveness...")
            try:
                # Format prompts with concept
                steering_prompts = [p.format(concept=concept) for p in steering_prompts_template]

                steering_results = test_steering_effectiveness(
                    model, tokenizer, concept_vector, concept,
                    concept_data[concept], steering_prompts, steering_strengths,
                    max_new_tokens=30, layer_idx=-1
                )
                print(f"    Generated {len(steering_results['results'])} steering samples")
                print(f"    Tracking {len(steering_results.get('related_terms', []))} related + {len(steering_results.get('antonym_terms', []))} antonym terms")

                # Print semantic mention summary (includes concept + related terms)
                total_semantic_by_strength = {}
                total_antonym_by_strength = {}
                for strength in steering_strengths:
                    semantic = []
                    antonyms = []
                    for prompt_result in steering_results['results']:
                        semantic.append(prompt_result['strengths'][str(strength)]['semantic_mentions'])
                        antonyms.append(prompt_result['strengths'][str(strength)]['antonym_mentions'])
                    total_semantic_by_strength[strength] = sum(semantic)
                    total_antonym_by_strength[strength] = sum(antonyms)

                print(f"    Semantic mentions by strength:")
                for strength in steering_strengths:
                    sem_count = total_semantic_by_strength[strength]
                    ant_count = total_antonym_by_strength[strength]
                    if ant_count > 0:
                        print(f"      {strength:+.2f}: {sem_count} semantic, {ant_count} antonyms")
                    else:
                        print(f"      {strength:+.2f}: {sem_count}")
            except Exception as e:
                print(f"  ✗ Failed steering test: {e}")
                import traceback
                traceback.print_exc()

        torch.cuda.empty_cache()

        concept_elapsed = time.time() - concept_start

        # Get original Phase 2 test accuracy
        original_acc = next((r['test_acc'] for r in sorted_results if r['concept'] == concept), 0)

        result = {
            'concept': concept,
            'original_test_acc': original_acc,
            'detection': detection_results,
            'steering': steering_results,
            'elapsed_seconds': concept_elapsed
        }

        all_concept_results.append(result)

        # Save individual result
        concept_safe = concept.replace('/', '_').replace(' ', '_')
        output_path = output_dir / f"phase2_5_{concept_safe}.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"  ✓ Completed in {concept_elapsed:.1f}s")
        print()

    # Save aggregate results
    aggregate = {
        'phase': 2.5,
        'description': 'Steering quality evaluation for Phase 2 scale test (1×1 training)',
        'scale': scale,
        'n_concepts_tested': len(all_concept_results),
        'config': {'n_definitions': 1, 'n_relationships': 1},
        'results': all_concept_results
    }

    aggregate_path = output_dir / f"phase2_5_scale_{scale}_aggregate.json"
    with open(aggregate_path, 'w') as f:
        json.dump(aggregate, f, indent=2)

    print()
    print("=" * 70)
    print("PHASE 2.5 COMPLETE")
    print("=" * 70)
    print(f"Saved results to: {aggregate_path}")
    print()
    print("Detection Confidence Summary:")
    print()
    for r in all_concept_results:
        concept = r['concept']
        if r['detection']:
            print(f"  {concept:30s}: "
                  f"mean={r['detection']['mean_confidence']:.3f}, "
                  f"min={r['detection']['min_confidence']:.3f}, "
                  f"std={r['detection']['std_confidence']:.3f}")
        else:
            print(f"  {concept:30s}: Failed")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Phase 2.5: Test steering quality for Phase 2 scale test concepts"
    )

    parser.add_argument('--phase2-results', type=str, required=True,
                       help='Path to Phase 2 results directory')
    parser.add_argument('--concept-graph', type=str, required=True,
                       help='Path to concept graph JSON')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model to use')
    parser.add_argument('--output-dir', type=str, default='results/phase_2_5',
                       help='Output directory')
    parser.add_argument('--scale', type=int, default=100,
                       help='Which Phase 2 scale to evaluate (default: 100)')
    parser.add_argument('--n-concepts', type=int, default=10,
                       help='Number of concepts to test (default: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    run_phase_2_5(
        Path(args.phase2_results),
        Path(args.concept_graph),
        model_name=args.model,
        output_dir=Path(args.output_dir),
        scale=args.scale,
        n_test_concepts=args.n_concepts,
        device=args.device
    )
