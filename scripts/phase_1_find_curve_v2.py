"""
Phase 1: Find the Curve (v2 - Fixed Test Set)
===============================================

Goal: Identify where diminishing returns kick in for definitions vs relationships.

Approach:
- Fixed test set: 10 concepts × 10 samples each = 200 test samples (reused across all configs)
- Training data: Varying amounts from same concepts
- Evaluation: All configs tested on same 200 samples

Configuration:
- Concepts: 10 (from WordNet top 10)
- Test samples: 10 per concept (fixed)
- Train definitions: [1, 10, 40, 160]
- Train relationships: [1, 10, 40, 160]
- Total: 4×4 = 16 configurations
- Estimated time: ~50 minutes

Output: Performance curves to determine optimal allocation for Phase 2.
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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.stage_1_5_temporal import get_activation_sequence


def sample_test_set(
    model,
    tokenizer,
    concept: str,
    negatives: List[str],
    related_structured: Dict[str, List[str]],
    n_samples: int = 10,
    layer_idx: int = -1,
    device: str = "cuda"
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate fixed test set for a concept.

    Returns n_samples positive and n_samples negative sequences.
    Uses diverse prompts to get varied samples.
    """
    pos_sequences = []
    neg_sequences = []

    # Diverse prompt templates for test set
    templates = [
        "What is {concept}?",
        "Define {concept}.",
        "Explain {concept}.",
        "Describe {concept}.",
        "{concept} is defined as",
        "The meaning of {concept} is",
        "Tell me about {concept}.",
        "{concept} refers to",
        "In summary, {concept} is",
        "The concept of {concept} involves"
    ]

    # Generate positive samples
    for i in range(n_samples):
        prompt = templates[i % len(templates)].format(concept=concept)
        seq, _ = get_activation_sequence(model, tokenizer, prompt, layer_idx, device)
        pos_sequences.append(seq)

    # Generate negative samples
    if len(negatives) == 0:
        raise ValueError(f"No negatives for concept '{concept}'")

    for i in range(n_samples):
        neg_concept = negatives[i % len(negatives)]
        prompt = templates[i % len(templates)].format(concept=neg_concept)
        seq, _ = get_activation_sequence(model, tokenizer, prompt, layer_idx, device)
        neg_sequences.append(seq)

    return pos_sequences, neg_sequences


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
    Sample training sequences with explicit definition/relationship split.

    Uses DIFFERENT prompts/data than test set.
    """
    pos_sequences = []
    neg_sequences = []

    # Positive: n_definitions
    if n_definitions > 0:
        direct_prompt = f"What is {concept}?"
        for _ in range(n_definitions):
            seq, _ = get_activation_sequence(model, tokenizer, direct_prompt, layer_idx, device)
            pos_sequences.append(seq)

    # Positive: n_relationships from structured related concepts
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
        # Fallback: more definitions if no related concepts
        for _ in range(n_relationships):
            seq, _ = get_activation_sequence(model, tokenizer, direct_prompt, layer_idx, device)
            pos_sequences.append(seq)

    # Negative sequences (match total positive count)
    n_total = n_definitions + n_relationships
    if len(negatives) == 0:
        raise ValueError(f"No negatives for concept '{concept}'")

    for i in range(n_total):
        neg_concept = negatives[i % len(negatives)]
        neg_prompt = f"What is {neg_concept}?"
        seq, _ = get_activation_sequence(model, tokenizer, neg_prompt, layer_idx, device)
        neg_sequences.append(seq)

    return pos_sequences, neg_sequences


def train_and_evaluate(
    train_pos: List[np.ndarray],
    train_neg: List[np.ndarray],
    test_pos: List[np.ndarray],
    test_neg: List[np.ndarray],
    hidden_dim: int,
    epochs: int = 10,
    lr: float = 1e-3
) -> Tuple[float, float]:
    """
    Train binary classifier and evaluate on fixed test set.

    Returns train_acc, test_acc
    """
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader

    # Prepare training data: pool temporal sequences (mean over time)
    train_pos_pooled = np.array([seq.mean(axis=0) for seq in train_pos])
    train_neg_pooled = np.array([seq.mean(axis=0) for seq in train_neg])

    X_train = np.vstack([train_pos_pooled, train_neg_pooled])
    y_train = np.array([1] * len(train_pos_pooled) + [0] * len(train_neg_pooled))

    # Prepare test data
    test_pos_pooled = np.array([seq.mean(axis=0) for seq in test_pos])
    test_neg_pooled = np.array([seq.mean(axis=0) for seq in test_neg])

    X_test = np.vstack([test_pos_pooled, test_neg_pooled])
    y_test = np.array([1] * len(test_pos_pooled) + [0] * len(test_neg_pooled))

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    # Simple MLP classifier
    model = nn.Sequential(
        nn.Linear(hidden_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    ).cuda()

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

            optimizer.zero_grad()
            pred = model(batch_X).squeeze(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_pred = (model(X_train_t.cuda()).squeeze(-1) > 0.5).cpu().float()
        train_acc = (train_pred == y_train_t).float().mean().item()

        test_pred = (model(X_test_t.cuda()).squeeze(-1) > 0.5).cpu().float()
        test_acc = (test_pred == y_test_t).float().mean().item()

    return train_acc, test_acc


def run_phase_1_v2(
    concept_graph_path: Path,
    model_name: str,
    output_dir: Path,
    device: str = "cuda"
):
    """
    Run Phase 1 v2: Fixed test set per concept.
    """
    print("=" * 70)
    print("PHASE 1: FIND THE CURVE (v2 - Fixed Test Set)")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  - Concepts: 10 (fixed)")
    print("  - Test samples: 10 per concept (fixed, 200 total)")
    print("  - Train definitions: [1, 10, 40, 160]")
    print("  - Train relationships: [1, 10, 40, 160]")
    print("  - Total: 16 configurations")
    print("  - Estimated time: ~50 minutes")
    print()
    print("Goal: Find diminishing returns using fixed test set")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model ONCE for all configs
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

    # Load concept graph
    with open(concept_graph_path) as f:
        concept_data = json.load(f)

    concepts = list(concept_data.keys())[:10]

    # Step 1: Generate fixed test set (ONCE for all configs)
    print("Step 1: Generating fixed test set...")
    print()

    test_sets = {}  # {concept: (pos_seqs, neg_seqs)}

    for concept in concepts:
        negatives = concept_data[concept].get('negatives', [])
        related_structured = concept_data[concept].get('related_structured', {})

        if len(negatives) == 0:
            print(f"  ⚠ Skipping {concept}: no negatives")
            continue

        print(f"  Generating test set for '{concept}'...", flush=True)

        try:
            pos_seqs, neg_seqs = sample_test_set(
                model, tokenizer, concept, negatives, related_structured,
                n_samples=10, layer_idx=-1, device=device
            )
            test_sets[concept] = (pos_seqs, neg_seqs)
            print(f"    ✓ {len(pos_seqs)} pos + {len(neg_seqs)} neg", flush=True)
        except Exception as e:
            print(f"    ✗ Failed: {e}", flush=True)
            continue

        # Clear cache after each concept
        torch.cuda.empty_cache()

    print()
    print(f"Generated test sets for {len(test_sets)} concepts")
    print(f"Total test samples: {len(test_sets) * 20}")
    print()

    # Step 2: Run 16 configurations with varying training data
    definition_scales = [1, 10, 40, 160]
    relationship_scales = [1, 10, 40, 160]

    all_results = []
    start_time = time.time()

    config_num = 0
    total_configs = len(definition_scales) * len(relationship_scales)

    for n_defs in definition_scales:
        for n_rels in relationship_scales:
            config_num += 1
            config_name = f"c10_d{n_defs}_r{n_rels}"

            print(f"[{config_num}/{total_configs}] Config: {n_defs} defs × {n_rels} rels", flush=True)

            config_start = time.time()

            # Train on each concept and aggregate test accuracy
            concept_results = []

            for concept in test_sets.keys():
                negatives = concept_data[concept].get('negatives', [])
                related_structured = concept_data[concept].get('related_structured', {})

                try:
                    # Sample training data
                    train_pos, train_neg = sample_train_sequences(
                        model, tokenizer, concept, negatives, related_structured,
                        n_defs, n_rels, layer_idx=-1, device=device
                    )

                    # Get test data
                    test_pos, test_neg = test_sets[concept]

                    # Train and evaluate
                    train_acc, test_acc = train_and_evaluate(
                        train_pos, train_neg, test_pos, test_neg, hidden_dim
                    )

                    concept_results.append({
                        'concept': concept,
                        'train_acc': train_acc,
                        'test_acc': test_acc
                    })

                except Exception as e:
                    import traceback
                    print(f"  ⚠ Error on '{concept}': {e}")
                    traceback.print_exc()
                    continue

                # Clear cache
                torch.cuda.empty_cache()

            # Aggregate results
            if concept_results:
                mean_train = np.mean([r['train_acc'] for r in concept_results])
                mean_test = np.mean([r['test_acc'] for r in concept_results])
            else:
                mean_train = mean_test = 0.0

            config_elapsed = time.time() - config_start

            result = {
                'config': {
                    'n_concepts': len(test_sets),
                    'n_definitions': n_defs,
                    'n_relationships': n_rels,
                    'total_train_samples': n_defs + n_rels
                },
                'results': {
                    'mean_train_acc': mean_train,
                    'mean_test_acc': mean_test,
                    'n_successful': len(concept_results)
                },
                'per_concept': concept_results,
                'elapsed_seconds': config_elapsed
            }

            all_results.append(result)

            # Save individual result
            output_path = output_dir / f"phase1_v2_{config_name}.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"  ✓ Test: {mean_test:.1%}, Train: {mean_train:.1%}, Time: {config_elapsed:.1f}s", flush=True)
            print()

    # Save aggregate results
    total_elapsed = time.time() - start_time

    aggregate = {
        'phase': 1,
        'version': 2,
        'description': 'Find the curve - fixed test set per concept',
        'test_set_size': len(test_sets) * 20,
        'total_configs': len(all_results),
        'total_time_seconds': total_elapsed,
        'configurations': all_results
    }

    aggregate_path = output_dir / "phase1_v2_aggregate.json"
    with open(aggregate_path, 'w') as f:
        json.dump(aggregate, f, indent=2)

    print()
    print("=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Completed: {len(all_results)}/{total_configs} configurations")
    print()
    print("Results Matrix (Test Accuracy on Fixed 200 Samples):")
    print()
    print("         Relationships")
    print("       ", end="")
    for n_rels in relationship_scales:
        print(f"{n_rels:>7}", end="")
    print()
    print("      " + "-" * (7 * len(relationship_scales)))

    for n_defs in definition_scales:
        print(f"D {n_defs:>3} |", end="")
        for n_rels in relationship_scales:
            # Find result
            found = False
            for r in all_results:
                if r['config']['n_definitions'] == n_defs and r['config']['n_relationships'] == n_rels:
                    test_acc = r['results']['mean_test_acc']
                    print(f"{test_acc:>6.1%}", end=" ")
                    found = True
                    break
            if not found:
                print("    -- ", end=" ")
        print()

    print()
    print(f"Saved aggregate results to: {aggregate_path}")
    print()
    print("Next: Analyze results to determine optimal allocation for Phase 2")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phase 1 v2: Find the curve with fixed test set")

    parser.add_argument('--concept-graph', type=str, required=True,
                       help='Path to concept graph JSON (WordNet)')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model to use')
    parser.add_argument('--output-dir', type=str, default='results/phase_1_v2',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    concept_graph_path = Path(args.concept_graph)
    output_dir = Path(args.output_dir)

    run_phase_1_v2(
        concept_graph_path,
        model_name=args.model,
        output_dir=output_dir,
        device=args.device
    )
