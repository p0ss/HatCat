"""
Scaling Study: Test samples × relationships × concepts matrix

Goal: Determine optimal allocation before large-scale compute investment.

Test matrix:
- Concepts: 1, 10, 100
- Samples per concept: 1, 10, 100
- Relationships per concept: 1, 10, 100

Key comparison:
- 10 concepts × (1 definition + 9 relationships)
  vs
- 10 concepts × 10 definitions

For each configuration:
1. Extract temporal sequences with specified sample/relationship mix
2. Train binary classifiers
3. Evaluate validation accuracy
4. Measure training time and compute
"""

import torch
import numpy as np
import h5py
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


def sample_sequences_with_config(
    model,
    tokenizer,
    concept: str,
    negatives: List[str],
    related_structured: Dict[str, List[str]],
    n_definitions: int = 5,
    n_relationships: int = 5,
    layer_idx: int = -1,
    device: str = "cuda"
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Sample sequences with explicit definition/relationship split.

    Args:
        n_definitions: Number of "What is X?" samples
        n_relationships: Number of "Relationship between X and Y" samples

    Returns:
        pos_sequences, neg_sequences
    """
    pos_sequences = []
    neg_sequences = []

    # Positive: n_definitions of "What is X?"
    direct_prompt = f"What is {concept}?"
    for _ in range(n_definitions):
        seq, _ = get_activation_sequence(model, tokenizer, direct_prompt, layer_idx, device)
        pos_sequences.append(seq)

    # Positive: n_relationships from structured related concepts
    # Priority order: hypernyms > hyponyms > meronyms > holonyms
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
    else:
        # Fallback: more definitions if no related concepts
        for _ in range(n_relationships):
            seq, _ = get_activation_sequence(model, tokenizer, direct_prompt, layer_idx, device)
            pos_sequences.append(seq)

    # Negative sequences
    n_total = n_definitions + n_relationships
    if len(negatives) == 0:
        raise ValueError(f"No negatives for concept '{concept}'")

    for i in range(n_total):
        neg_concept = negatives[i % len(negatives)]
        neg_prompt = f"What is {neg_concept}?"
        seq, _ = get_activation_sequence(model, tokenizer, neg_prompt, layer_idx, device)
        neg_sequences.append(seq)

    return pos_sequences, neg_sequences


def sample_sequences_relationship_first(
    model,
    tokenizer,
    concepts_data: Dict[str, Dict],  # {concept: {negatives: [], related_structured: {}}}
    n_definitions: int = 5,
    n_relationships: int = 5,
    layer_idx: int = -1,
    device: str = "cuda"
) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Relationship-first mode: Generate each edge once, reuse for both concepts.

    Returns:
        {concept: (pos_sequences, neg_sequences)}
    """
    # Phase 1: Collect all unique edges
    edges = set()
    for concept, data in concepts_data.items():
        related_structured = data['related_structured']
        all_related = []
        for rel_type in ['hypernyms', 'hyponyms', 'meronyms', 'holonyms']:
            if rel_type in related_structured:
                all_related.extend(related_structured[rel_type])

        for rel in all_related[:n_relationships]:
            edge = tuple(sorted([concept, rel]))
            edges.add(edge)

    print(f"  Relationship-first: {len(edges)} unique edges across {len(concepts_data)} concepts")

    # Phase 2: Generate activations for all unique edges once
    edge_activations = {}
    for edge in edges:
        a, b = edge
        rel_prompt = f"The relationship between {a} and {b}"
        seq, _ = get_activation_sequence(model, tokenizer, rel_prompt, layer_idx, device)
        edge_activations[edge] = seq

    # Phase 3: Distribute to concepts
    result = {}
    for concept, data in concepts_data.items():
        pos_sequences = []
        neg_sequences = []

        # Definitions
        direct_prompt = f"What is {concept}?"
        for _ in range(n_definitions):
            seq, _ = get_activation_sequence(model, tokenizer, direct_prompt, layer_idx, device)
            pos_sequences.append(seq)

        # Relationships - reuse edge activations
        related_structured = data['related_structured']
        all_related = []
        for rel_type in ['hypernyms', 'hyponyms', 'meronyms', 'holonyms']:
            if rel_type in related_structured:
                all_related.extend(related_structured[rel_type])

        for rel in all_related[:n_relationships]:
            edge = tuple(sorted([concept, rel]))
            if edge in edge_activations:
                pos_sequences.append(edge_activations[edge])

        # Fallback if no relationships
        if len(pos_sequences) == n_definitions:
            for _ in range(n_relationships):
                seq, _ = get_activation_sequence(model, tokenizer, direct_prompt, layer_idx, device)
                pos_sequences.append(seq)

        # Negatives
        negatives = data['negatives']
        n_total = len(pos_sequences)
        for i in range(n_total):
            neg_concept = negatives[i % len(negatives)]
            neg_prompt = f"What is {neg_concept}?"
            seq, _ = get_activation_sequence(model, tokenizer, neg_prompt, layer_idx, device)
            neg_sequences.append(seq)

        result[concept] = (pos_sequences, neg_sequences)

    return result


def train_binary_classifier(
    pos_sequences: List[np.ndarray],
    neg_sequences: List[np.ndarray],
    hidden_dim: int,
    epochs: int = 10,
    lr: float = 1e-3
) -> Tuple[float, float]:
    """
    Train simple binary classifier and return train/val accuracy.

    Uses 80/20 train/val split.
    """
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader

    # Prepare data: pool temporal sequences (mean over time)
    pos_pooled = np.array([seq.mean(axis=0) for seq in pos_sequences])
    neg_pooled = np.array([seq.mean(axis=0) for seq in neg_sequences])

    X = np.vstack([pos_pooled, neg_pooled])
    y = np.array([1] * len(pos_pooled) + [0] * len(neg_pooled))

    # 80/20 split
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    X_train, y_train = torch.FloatTensor(X[train_idx]), torch.FloatTensor(y[train_idx])
    X_val, y_val = torch.FloatTensor(X[val_idx]), torch.FloatTensor(y[val_idx])

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
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

            optimizer.zero_grad()
            pred = model(batch_X).squeeze()
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_pred = (model(X_train.cuda()).squeeze() > 0.5).cpu().float()
        train_acc = (train_pred == y_train).float().mean().item()

        val_pred = (model(X_val.cuda()).squeeze() > 0.5).cpu().float()
        val_acc = (val_pred == y_val).float().mean().item()

    return train_acc, val_acc


def run_scaling_experiment(
    concept_graph_path: Path,
    model_name: str,
    n_concepts: int,
    n_definitions: int,
    n_relationships: int,
    output_path: Path,
    device: str = "cuda",
    use_relationship_first: bool = False
):
    """
    Run single scaling experiment configuration.
    """
    print(f"\n{'='*70}")
    print(f"SCALING EXPERIMENT: {n_concepts} concepts × {n_definitions} defs × {n_relationships} rels")
    print(f"{'='*70}")

    start_time = time.time()

    # Load concept graph
    with open(concept_graph_path) as f:
        concept_data = json.load(f)

    concepts = list(concept_data.keys())[:n_concepts]
    print(f"Testing {len(concepts)} concepts: {concepts[:3]}...")

    # Load model
    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
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

    # Extract sequences and train classifiers
    results = []

    if use_relationship_first and n_relationships > 0:
        print(f"\nUsing relationship-first mode...")
        # Build concepts_data dict for batch processing
        concepts_dict = {}
        for concept in concepts:
            negatives = concept_data[concept].get('negatives', [])
            related_structured = concept_data[concept].get('related_structured', {})
            if len(negatives) > 0:
                concepts_dict[concept] = {
                    'negatives': negatives,
                    'related_structured': related_structured
                }

        # Batch generate all sequences with relationship-first
        try:
            all_sequences = sample_sequences_relationship_first(
                model, tokenizer, concepts_dict,
                n_definitions, n_relationships, layer_idx=-1, device=device
            )
        except Exception as e:
            print(f"  ⚠ Error in batch sampling: {e}")
            all_sequences = {}

        # Train classifiers for each concept
        for concept_idx, concept in enumerate(concepts):
            if concept not in all_sequences:
                continue

            print(f"\n[{concept_idx+1}/{len(concepts)}] Training '{concept}'...")
            pos_seqs, neg_seqs = all_sequences[concept]
            print(f"  {len(pos_seqs)} pos, {len(neg_seqs)} neg sequences")

            try:
                train_acc, val_acc = train_binary_classifier(
                    pos_seqs, neg_seqs, hidden_dim
                )
                results.append({
                    'concept': concept,
                    'train_acc': train_acc,
                    'val_acc': val_acc
                })
                print(f"  Train: {train_acc:.1%}, Val: {val_acc:.1%}")
            except Exception as e:
                print(f"  ⚠ Error training: {e}")

    else:
        print(f"\nUsing per-concept mode...")
        for concept_idx, concept in enumerate(concepts):
            print(f"\n[{concept_idx+1}/{len(concepts)}] Processing '{concept}'...")

            # Get negatives and related
            negatives = concept_data[concept].get('negatives', [])
            related_structured = concept_data[concept].get('related_structured', {})

            if len(negatives) == 0:
                print(f"  ⚠ Skipping - no negatives")
                continue

            # Sample sequences
            try:
                pos_seqs, neg_seqs = sample_sequences_with_config(
                    model, tokenizer, concept, negatives, related_structured,
                    n_definitions, n_relationships, layer_idx=-1, device=device
                )
            except Exception as e:
                print(f"  ⚠ Error sampling: {e}")
                continue

            print(f"  Sampled {len(pos_seqs)} pos, {len(neg_seqs)} neg sequences")

            # Train classifier
            try:
                train_acc, val_acc = train_binary_classifier(
                    pos_seqs, neg_seqs, hidden_dim
                )
                print(f"  Train: {train_acc:.1%}, Val: {val_acc:.1%}")

                results.append({
                    'concept': concept,
                    'train_acc': train_acc,
                    'val_acc': val_acc
                })
            except Exception as e:
                print(f"  ⚠ Error training: {e}")
                continue

    # Aggregate results
    elapsed = time.time() - start_time

    if results:
        mean_train = np.mean([r['train_acc'] for r in results])
        mean_val = np.mean([r['val_acc'] for r in results])
    else:
        mean_train = mean_val = 0.0

    summary = {
        'config': {
            'n_concepts': n_concepts,
            'n_definitions': n_definitions,
            'n_relationships': n_relationships,
            'total_samples_per_concept': n_definitions + n_relationships
        },
        'results': {
            'mean_train_acc': mean_train,
            'mean_val_acc': mean_val,
            'n_successful': len(results),
            'elapsed_seconds': elapsed
        },
        'per_concept': results
    }

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Mean train acc: {mean_train:.1%}")
    print(f"Mean val acc: {mean_val:.1%}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Saved to: {output_path}")

    # Clean up model to free GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    print(f"✓ Model cleaned up, GPU memory freed")

    return summary


def run_full_scaling_study(
    concept_graph_path: Path,
    model_name: str,
    output_dir: Path,
    device: str = "cuda"
):
    """
    Run full 3×3×3 scaling study matrix.
    """
    print("="*70)
    print("FULL SCALING STUDY: 3×3×3 MATRIX")
    print("="*70)
    print()
    print("Testing dimensions:")
    print("  - Concepts: 10, 20, 40")
    print("  - Definitions: 10, 20, 40")
    print("  - Relationships: 10, 20, 40")
    print()
    print("Rationale:")
    print("  - 10-20-40 provides better granularity than 1-10-100")
    print("  - Tests realistic production ranges")
    print("  - More useful data for smaller compute budget")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Test matrix: 10-20-40 (more practical than 1-10-100)
    concept_scales = [10, 20, 40]
    definition_scales = [10, 20, 40]
    relationship_scales = [10, 20, 40]

    all_results = []

    for n_concepts in concept_scales:
        for n_defs in definition_scales:
            for n_rels in relationship_scales:
                config_name = f"c{n_concepts}_d{n_defs}_r{n_rels}"
                output_path = output_dir / f"scaling_{config_name}.json"

                try:
                    result = run_scaling_experiment(
                        concept_graph_path,
                        model_name,
                        n_concepts,
                        n_defs,
                        n_rels,
                        output_path,
                        device
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"\n❌ Failed {config_name}: {e}")
                    continue

    # Save aggregate results
    aggregate_path = output_dir / "scaling_aggregate.json"
    with open(aggregate_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("SCALING STUDY COMPLETE")
    print(f"{'='*70}")
    print(f"Saved aggregate results to: {aggregate_path}")

    # Print summary matrix
    print("\nValidation Accuracy Matrix:")
    print("(Concepts × Definitions × Relationships)")
    print()

    for result in all_results:
        cfg = result['config']
        res = result['results']
        print(f"  {cfg['n_concepts']:3d} × {cfg['n_definitions']:3d} × {cfg['n_relationships']:3d} = {res['mean_val_acc']:5.1%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scaling study for temporal sequences")

    parser.add_argument('--concept-graph', type=str, required=True,
                       help='Path to concept graph JSON (WordNet V2)')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model to use')
    parser.add_argument('--output-dir', type=str, default='results/scaling_study',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    # For single experiment
    parser.add_argument('--single', action='store_true',
                       help='Run single experiment (not full matrix)')
    parser.add_argument('--n-concepts', type=int, default=10)
    parser.add_argument('--n-definitions', type=int, default=5)
    parser.add_argument('--n-relationships', type=int, default=5)
    parser.add_argument('--relationship-first', action='store_true',
                       help='Use relationship-first mode: generate each edge once, reuse for concepts')

    args = parser.parse_args()

    concept_graph_path = Path(args.concept_graph)
    output_dir = Path(args.output_dir)

    if args.single:
        # Run single experiment
        output_path = output_dir / f"scaling_c{args.n_concepts}_d{args.n_definitions}_r{args.n_relationships}.json"
        run_scaling_experiment(
            concept_graph_path,
            args.model,
            args.n_concepts,
            args.n_definitions,
            args.n_relationships,
            output_path,
            args.device,
            use_relationship_first=args.relationship_first
        )
    else:
        # Run full matrix
        run_full_scaling_study(
            concept_graph_path,
            args.model,
            output_dir,
            args.device
        )
