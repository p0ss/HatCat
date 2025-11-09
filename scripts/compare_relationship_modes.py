"""
Compare Per-Concept vs Relationship-First Training

Tests whether generating relationship edges once and reusing them
maintains accuracy while reducing compute.

Per-concept mode:  Generate "A→B" separately for each concept
Relationship-first: Generate "A→B" once, reuse for both A and B
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import argparse
import json
import time
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.stage_1_5_temporal import get_activation_sequence


def load_concept_graph(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


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


def sample_per_concept_mode(
    model,
    tokenizer,
    concepts: List[str],
    concept_graph: Dict,
    n_defs: int,
    n_rels: int,
    layer_idx: int,
    device: str
) -> Dict[str, Tuple[List, List]]:
    """
    Current mode: Generate each relationship separately per concept.

    For concept A with relationship to B:
    - Generates "relationship between A and B" for A's training
    - Generates "relationship between B and A" for B's training (if B also has edge to A)
    """
    concept_data = {}

    for concept in concepts:
        pos_seqs = []
        neg_seqs = []

        # Definitions
        def_prompt = f"What is {concept}?"
        for _ in range(n_defs):
            seq, _ = get_activation_sequence(model, tokenizer, def_prompt, layer_idx, device)
            pos_seqs.append(seq)

        # Relationships - per concept generation
        related = concept_graph[concept].get('related', [])
        for i in range(min(n_rels, len(related))):
            rel = related[i % len(related)]
            rel_prompt = f"The relationship between {concept} and {rel}"
            seq, _ = get_activation_sequence(model, tokenizer, rel_prompt, layer_idx, device)
            pos_seqs.append(seq)

        # Negatives
        negatives = concept_graph[concept].get('negatives', [])
        n_total = n_defs + min(n_rels, len(related))
        for i in range(n_total):
            neg = negatives[i % len(negatives)]
            neg_prompt = f"What is {neg}?"
            seq, _ = get_activation_sequence(model, tokenizer, neg_prompt, layer_idx, device)
            neg_seqs.append(seq)

        concept_data[concept] = (pos_seqs, neg_seqs)

    return concept_data


def sample_relationship_first_mode(
    model,
    tokenizer,
    concepts: List[str],
    concept_graph: Dict,
    n_defs: int,
    n_rels: int,
    layer_idx: int,
    device: str
) -> Dict[str, Tuple[List, List]]:
    """
    Relationship-first mode: Generate each edge once, reuse for both concepts.

    1. Collect all unique edges in the graph
    2. Generate activation for each edge once
    3. Distribute edge activations to participating concepts
    """
    # Phase 1: Collect all unique edges
    edges = set()
    for concept in concepts:
        related = concept_graph[concept].get('related', [])
        for rel in related[:n_rels]:  # Limit to n_rels per concept
            # Store as sorted tuple to ensure uniqueness
            edge = tuple(sorted([concept, rel]))
            edges.add(edge)

    print(f"  Phase 1: Found {len(edges)} unique relationship edges")

    # Phase 2: Generate activations for all unique edges once
    edge_activations = {}
    for edge in edges:
        a, b = edge
        rel_prompt = f"The relationship between {a} and {b}"
        seq, _ = get_activation_sequence(model, tokenizer, rel_prompt, layer_idx, device)
        edge_activations[edge] = seq

    print(f"  Phase 2: Generated activations for {len(edge_activations)} edges")

    # Phase 3: Distribute to concepts
    concept_data = {}
    for concept in concepts:
        pos_seqs = []
        neg_seqs = []

        # Definitions
        def_prompt = f"What is {concept}?"
        for _ in range(n_defs):
            seq, _ = get_activation_sequence(model, tokenizer, def_prompt, layer_idx, device)
            pos_seqs.append(seq)

        # Relationships - reuse edge activations
        related = concept_graph[concept].get('related', [])
        for rel in related[:n_rels]:
            edge = tuple(sorted([concept, rel]))
            if edge in edge_activations:
                pos_seqs.append(edge_activations[edge])

        # Negatives
        negatives = concept_graph[concept].get('negatives', [])
        n_total = len(pos_seqs)
        for i in range(n_total):
            neg = negatives[i % len(negatives)]
            neg_prompt = f"What is {neg}?"
            seq, _ = get_activation_sequence(model, tokenizer, neg_prompt, layer_idx, device)
            neg_seqs.append(seq)

        concept_data[concept] = (pos_seqs, neg_seqs)

    return concept_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concept-graph', type=Path, required=True)
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--n-concepts', type=int, default=10)
    parser.add_argument('--n-defs', type=int, default=1)
    parser.add_argument('--n-rels', type=int, default=9)
    parser.add_argument('--layer-idx', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("RELATIONSHIP MODE COMPARISON")
    print("="*70)
    print(f"Testing: {args.n_concepts} concepts × ({args.n_defs} defs + {args.n_rels} rels)")
    print()

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.eval()

    hidden_dim = model.get_input_embeddings().embedding_dim
    print(f"Hidden dim: {hidden_dim}\n")

    # Load concepts
    concept_graph = load_concept_graph(args.concept_graph)
    concepts = list(concept_graph.keys())[:args.n_concepts]

    results = {}

    # Mode 1: Per-concept
    print("="*70)
    print("MODE 1: PER-CONCEPT (Current)")
    print("="*70)
    start = time.time()
    data_per_concept = sample_per_concept_mode(
        model, tokenizer, concepts, concept_graph,
        args.n_defs, args.n_rels, args.layer_idx, args.device
    )

    per_concept_results = []
    for concept in concepts:
        pos_seqs, neg_seqs = data_per_concept[concept]
        train_acc, val_acc = train_binary_classifier(
            pos_seqs, neg_seqs, hidden_dim
        )
        per_concept_results.append({'concept': concept, 'train_acc': train_acc, 'val_acc': val_acc})
        print(f"  {concept:25s} Train: {train_acc:.1%}, Val: {val_acc:.1%}")

    per_concept_time = time.time() - start
    per_concept_mean = np.mean([r['val_acc'] for r in per_concept_results])

    print(f"\nMean val acc: {per_concept_mean:.1%}")
    print(f"Time: {per_concept_time:.1f}s\n")

    results['per_concept'] = {
        'mean_val_acc': per_concept_mean,
        'time_seconds': per_concept_time,
        'results': per_concept_results
    }

    # Mode 2: Relationship-first
    print("="*70)
    print("MODE 2: RELATIONSHIP-FIRST (New)")
    print("="*70)
    start = time.time()
    data_rel_first = sample_relationship_first_mode(
        model, tokenizer, concepts, concept_graph,
        args.n_defs, args.n_rels, args.layer_idx, args.device
    )

    rel_first_results = []
    for concept in concepts:
        pos_seqs, neg_seqs = data_rel_first[concept]
        train_acc, val_acc = train_binary_classifier(
            pos_seqs, neg_seqs, hidden_dim
        )
        rel_first_results.append({'concept': concept, 'train_acc': train_acc, 'val_acc': val_acc})
        print(f"  {concept:25s} Train: {train_acc:.1%}, Val: {val_acc:.1%}")

    rel_first_time = time.time() - start
    rel_first_mean = np.mean([r['val_acc'] for r in rel_first_results])

    print(f"\nMean val acc: {rel_first_mean:.1%}")
    print(f"Time: {rel_first_time:.1f}s\n")

    results['relationship_first'] = {
        'mean_val_acc': rel_first_mean,
        'time_seconds': rel_first_time,
        'results': rel_first_results
    }

    # Comparison
    print("="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Per-concept mode:       {per_concept_mean:.1%} in {per_concept_time:.1f}s")
    print(f"Relationship-first mode: {rel_first_mean:.1%} in {rel_first_time:.1f}s")
    print(f"Accuracy delta:         {rel_first_mean - per_concept_mean:+.1%}")
    print(f"Speedup:                {per_concept_time / rel_first_time:.2f}x")

    # Save results
    output_path = args.output_dir / "relationship_mode_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
