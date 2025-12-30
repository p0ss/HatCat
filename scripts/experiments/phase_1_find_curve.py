"""
Phase 1: Find the Curve
=======================

Goal: Identify where diminishing returns kick in for definitions vs relationships.

Configuration:
- Concepts: 10 (fixed - from WordNet top 10)
- Definitions: [1, 10, 40, 160]
- Relationships: [1, 10, 40, 160]
- Total: 4×4 = 16 configurations
- Estimated time: ~40 minutes

Output: Performance curves to determine optimal allocation for Phase 2.
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

    Relationships are ordered by strength: hypernyms > hyponyms > meronyms > holonyms
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
            pred = model(batch_X).squeeze(-1)  # Only squeeze last dim
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_pred = (model(X_train.cuda()).squeeze(-1) > 0.5).cpu().float()
        train_acc = (train_pred == y_train).float().mean().item()

        val_pred = (model(X_val.cuda()).squeeze(-1) > 0.5).cpu().float()
        val_acc = (val_pred == y_val).float().mean().item()

    return train_acc, val_acc


def run_single_config(
    concept_graph_path: Path,
    model,
    tokenizer,
    hidden_dim: int,
    n_concepts: int,
    n_definitions: int,
    n_relationships: int,
    device: str = "cuda"
) -> Dict:
    """
    Run single configuration and return results.

    Args:
        model: Pre-loaded model (reused across configs)
        tokenizer: Pre-loaded tokenizer
        hidden_dim: Hidden dimension of model
    """
    # Load concept graph
    with open(concept_graph_path) as f:
        concept_data = json.load(f)

    concepts = list(concept_data.keys())[:n_concepts]

    # Extract sequences and train classifiers
    results = []

    for concept_idx, concept in enumerate(concepts):
        # Clear CUDA cache to avoid fragmentation
        torch.cuda.empty_cache()

        # Get negatives and related
        negatives = concept_data[concept].get('negatives', [])
        related_structured = concept_data[concept].get('related_structured', {})

        if len(negatives) == 0:
            continue

        # Sample sequences
        try:
            pos_seqs, neg_seqs = sample_sequences_with_config(
                model, tokenizer, concept, negatives, related_structured,
                n_definitions, n_relationships, layer_idx=-1, device=device
            )
        except Exception as e:
            continue

        # Train classifier
        try:
            train_acc, val_acc = train_binary_classifier(
                pos_seqs, neg_seqs, hidden_dim
            )

            results.append({
                'concept': concept,
                'train_acc': train_acc,
                'val_acc': val_acc
            })
        except Exception as e:
            import traceback
            print(f"  ⚠ Error training classifier: {e}")
            traceback.print_exc()
            continue

    # Aggregate
    if results:
        mean_train = np.mean([r['train_acc'] for r in results])
        mean_val = np.mean([r['val_acc'] for r in results])
    else:
        mean_train = mean_val = 0.0

    return {
        'config': {
            'n_concepts': n_concepts,
            'n_definitions': n_definitions,
            'n_relationships': n_relationships,
            'total_samples': n_definitions + n_relationships
        },
        'results': {
            'mean_train_acc': mean_train,
            'mean_val_acc': mean_val,
            'n_successful': len(results)
        },
        'per_concept': results
    }


def run_phase_1(
    concept_graph_path: Path,
    model_name: str,
    output_dir: Path,
    device: str = "cuda"
):
    """
    Run Phase 1: Find the curve with Option B configuration.

    Tests 4×4 = 16 configurations to identify diminishing returns.
    """
    print("=" * 70)
    print("PHASE 1: FIND THE CURVE")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  - Concepts: 10 (fixed)")
    print("  - Definitions: [1, 10, 40, 160]")
    print("  - Relationships: [1, 10, 40, 160]")
    print("  - Total: 16 configurations")
    print("  - Estimated time: ~40 minutes")
    print()
    print("Goal: Identify where diminishing returns kick in")
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

    definition_scales = [1, 10, 40, 160]
    relationship_scales = [1, 10, 40, 160]
    n_concepts = 10

    all_results = []
    start_time = time.time()

    config_num = 0
    total_configs = len(definition_scales) * len(relationship_scales)

    for n_defs in definition_scales:
        for n_rels in relationship_scales:
            config_num += 1
            config_name = f"c{n_concepts}_d{n_defs}_r{n_rels}"

            print(f"[{config_num}/{total_configs}] Running: {n_concepts} concepts × {n_defs} defs × {n_rels} rels", flush=True)

            config_start = time.time()

            try:
                result = run_single_config(
                    concept_graph_path,
                    model,
                    tokenizer,
                    hidden_dim,
                    n_concepts,
                    n_defs,
                    n_rels,
                    device
                )

                config_elapsed = time.time() - config_start
                result['elapsed_seconds'] = config_elapsed

                all_results.append(result)

                # Save individual result
                output_path = output_dir / f"phase1_{config_name}.json"
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)

                print(f"  ✓ Val: {result['results']['mean_val_acc']:.1%}, Time: {config_elapsed:.1f}s")

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue

    # Save aggregate results
    total_elapsed = time.time() - start_time

    aggregate = {
        'phase': 1,
        'description': 'Find the curve - asymmetric 4×4 matrix',
        'total_configs': len(all_results),
        'total_time_seconds': total_elapsed,
        'configurations': all_results
    }

    aggregate_path = output_dir / "phase1_aggregate.json"
    with open(aggregate_path, 'w') as f:
        json.dump(aggregate, f, indent=2)

    print()
    print("=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Completed: {len(all_results)}/{total_configs} configurations")
    print()
    print("Results Matrix (Validation Accuracy):")
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
                    val_acc = r['results']['mean_val_acc']
                    print(f"{val_acc:>6.1%}", end=" ")
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
    parser = argparse.ArgumentParser(description="Phase 1: Find the curve")

    parser.add_argument('--concept-graph', type=str, required=True,
                       help='Path to concept graph JSON (WordNet V2)')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model to use')
    parser.add_argument('--output-dir', type=str, default='results/phase_1',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    concept_graph_path = Path(args.concept_graph)
    output_dir = Path(args.output_dir)

    run_phase_1(
        concept_graph_path,
        model_name=args.model,
        output_dir=output_dir,
        device=args.device
    )
