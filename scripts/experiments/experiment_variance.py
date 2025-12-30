#!/usr/bin/env python3
"""
Inter-run variance experiment for lens training.

Trains the same lenses multiple times with identical setup but different random
seeds to measure how much variance exists in the training process itself.

This helps establish error bars for comparing different training approaches.
"""

import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
from datetime import datetime
import random

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# Test concepts - same as noise baseline for comparability
TEST_CONCEPTS = [
    ('AlignmentProcess', 1),
    ('Agent', 1),
    ('Projectile', 2),
    ('AIAlignmentProcess', 2),
    ('AAV', 3),
    ('ACPowerSource', 3),
    ('AH1', 3),
    ('ADHD', 4),
]


class LensClassifier(nn.Module):
    """MLP classifier for concept detection."""
    def __init__(self, input_dim: int = 4096):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def load_hierarchy(layers_dir: Path) -> Dict:
    """Load concept hierarchy."""
    hierarchy = {}
    for layer in range(5):
        layer_file = layers_dir / f'layer{layer}.json'
        if layer_file.exists():
            with open(layer_file) as f:
                raw_data = json.load(f)

            if isinstance(raw_data, dict) and 'concepts' in raw_data:
                concepts_list = raw_data['concepts']
            elif isinstance(raw_data, list):
                concepts_list = raw_data
            else:
                concepts_list = [{'sumo_term': k, **v} for k, v in raw_data.items()]

            layer_data = {}
            for concept in concepts_list:
                name = concept.get('sumo_term', concept.get('name', ''))
                if name:
                    layer_data[name] = concept
            hierarchy[layer] = layer_data
    return hierarchy


def get_concept_definitions(concept_data: Dict, concept_name: str, max_samples: int = 50) -> List[str]:
    """Get definitions for a concept."""
    definitions = []

    if concept_data.get('definitions'):
        definitions = concept_data['definitions']
    elif concept_data.get('definition'):
        definitions = [concept_data['definition']]
    elif concept_data.get('sumo_definition'):
        definitions = [concept_data['sumo_definition']]

    lemmas = concept_data.get('lemmas', [])
    for lemma in lemmas[:10]:
        definitions.append(f"{lemma}")
        definitions.append(f"This is a {lemma}.")
        definitions.append(f"The {lemma} is an example of {concept_name}.")

    if not definitions:
        definitions = [
            f"{concept_name}",
            f"This is {concept_name}.",
            f"An example of {concept_name}.",
        ]

    return definitions[:max_samples]


def get_distant_concepts(hierarchy: Dict, concept: str, layer: int, count: int = 10) -> List[Tuple[str, Dict]]:
    """Get concepts from distant parts of the hierarchy."""
    distant = []
    current_data = hierarchy.get(layer, {}).get(concept, {})
    current_parents = set(current_data.get('parent_concepts', []))

    for other_layer in range(5):
        if other_layer == layer:
            continue
        for other_concept, other_data in hierarchy.get(other_layer, {}).items():
            other_parents = set(other_data.get('parent_concepts', []))
            if not current_parents.intersection(other_parents):
                distant.append((other_concept, other_data))

    random.shuffle(distant)
    return distant[:count]


def capture_activation(text: str, model, tokenizer, target_layer: int = 15) -> torch.Tensor:
    """Capture activation for a text sample."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[target_layer]
        activation = hidden_states[:, -1, :]

    return activation.float().cpu()


def get_embeddings(model, tokenizer, texts: List[str], target_layer: int = 15) -> torch.Tensor:
    """Get embeddings for a list of texts."""
    embeddings = []
    for text in texts:
        try:
            emb = capture_activation(text, model, tokenizer, target_layer)
            embeddings.append(emb.squeeze(0))
        except Exception:
            continue

    if embeddings:
        return torch.stack(embeddings)
    return torch.empty(0, 4096)


def train_lens(positive_emb: torch.Tensor, negative_emb: torch.Tensor,
                seed: int = None, max_epochs: int = 100, patience: int = 10) -> Tuple[LensClassifier, Dict]:
    """Train a lens with optional seed for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    pos_labels = torch.ones(len(positive_emb), 1)
    neg_labels = torch.zeros(len(negative_emb), 1)

    X = torch.cat([positive_emb, negative_emb], dim=0)
    y = torch.cat([pos_labels, neg_labels], dim=0)

    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    if len(X_val) == 0:
        X_val, y_val = X_train[-2:], y_train[-2:]

    lens = LensClassifier(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(lens.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):
        lens.train()
        optimizer.zero_grad()
        pred = lens(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        lens.eval()
        with torch.no_grad():
            val_pred = lens(X_val)
            val_loss = criterion(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = lens.model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        lens.model.load_state_dict(best_state)

    lens.eval()
    with torch.no_grad():
        train_pred = lens(X_train)
        train_acc = ((train_pred > 0.5) == y_train).float().mean().item()
        val_pred = lens(X_val)
        val_acc = ((val_pred > 0.5) == y_val).float().mean().item()

    return lens, {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'epochs': epoch + 1,
        'seed': seed,
    }


def evaluate_lens(lens: LensClassifier, positive_emb: torch.Tensor, negative_emb: torch.Tensor) -> Dict:
    """Evaluate lens discrimination."""
    lens.eval()
    with torch.no_grad():
        pos_scores = lens(positive_emb).squeeze().numpy() if len(positive_emb) > 0 else np.array([])
        neg_scores = lens(negative_emb).squeeze().numpy() if len(negative_emb) > 0 else np.array([])

    if pos_scores.ndim == 0:
        pos_scores = np.array([pos_scores.item()])
    if neg_scores.ndim == 0:
        neg_scores = np.array([neg_scores.item()])

    return {
        'avg_positive': float(np.mean(pos_scores)) if len(pos_scores) > 0 else None,
        'avg_negative': float(np.mean(neg_scores)) if len(neg_scores) > 0 else None,
        'positive_negative_gap': float(np.mean(pos_scores) - np.mean(neg_scores)) if len(pos_scores) > 0 and len(neg_scores) > 0 else None,
        'n_positive': len(pos_scores),
        'n_negative': len(neg_scores),
    }


def main():
    parser = argparse.ArgumentParser(description='Inter-run variance experiment')
    parser.add_argument('--model', default='swiss-ai/Apertus-8B-2509')
    parser.add_argument('--layers-dir', default='data/concept_graph/v4')
    parser.add_argument('--output-dir', default='results/variance_experiment')
    parser.add_argument('--n-runs', type=int, default=5, help='Number of training runs per concept')
    parser.add_argument('--min-definitions', type=int, default=5)
    parser.add_argument('--target-layer', type=int, default=15)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.eval()

    print("Loading hierarchy...")
    hierarchy = load_hierarchy(Path(args.layers_dir))

    results = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'n_runs': args.n_runs,
        'lenses': {},
    }

    for concept, layer in TEST_CONCEPTS:
        print(f"\n{'='*60}")
        print(f"Testing {concept} (layer {layer}) - {args.n_runs} runs")
        print('='*60)

        concept_data = hierarchy.get(layer, {}).get(concept)
        if not concept_data:
            print(f"  ERROR: Concept not found in layer {layer}")
            results['lenses'][f'{concept}_L{layer}'] = {'error': f'Concept not found'}
            continue

        pos_defs = get_concept_definitions(concept_data, concept, max_samples=50)
        if len(pos_defs) < args.min_definitions:
            print(f"  ERROR: Not enough definitions: {len(pos_defs)}")
            results['lenses'][f'{concept}_L{layer}'] = {'error': f'Not enough definitions'}
            continue

        pos_emb = get_embeddings(model, tokenizer, pos_defs, args.target_layer)
        print(f"  Positive embeddings: {len(pos_emb)}")

        distant = get_distant_concepts(hierarchy, concept, layer, count=10)
        distant_defs = []
        for dist_concept, dist_data in distant:
            defs = get_concept_definitions(dist_data, dist_concept, max_samples=10)
            distant_defs.extend(defs[:5])

        neg_emb = get_embeddings(model, tokenizer, distant_defs, args.target_layer)
        print(f"  Negative embeddings: {len(neg_emb)}")

        if len(neg_emb) < 5:
            print(f"  ERROR: Not enough negatives")
            results['lenses'][f'{concept}_L{layer}'] = {'error': 'Not enough negatives'}
            continue

        # Split for consistent evaluation
        eval_pos = pos_emb[-5:]
        train_pos = pos_emb[:-5]
        eval_neg = neg_emb[-5:]
        train_neg = neg_emb[:-5]

        runs = []
        for run_idx in range(args.n_runs):
            seed = 42 + run_idx * 100
            print(f"\n  Run {run_idx + 1}/{args.n_runs} (seed={seed})...")

            lens, train_metrics = train_lens(train_pos, train_neg, seed=seed)
            eval_metrics = evaluate_lens(lens, eval_pos, eval_neg)

            run_result = {
                'run': run_idx + 1,
                'training': train_metrics,
                'evaluation': eval_metrics,
            }
            runs.append(run_result)

            print(f"    Val acc: {train_metrics['val_acc']:.3f}, "
                  f"Pos-Neg gap: {eval_metrics['positive_negative_gap']:.3f}")

        # Compute variance statistics
        gaps = [r['evaluation']['positive_negative_gap'] for r in runs if r['evaluation']['positive_negative_gap'] is not None]
        val_accs = [r['training']['val_acc'] for r in runs]

        stats = {
            'n_runs': len(runs),
            'gap_mean': float(np.mean(gaps)),
            'gap_std': float(np.std(gaps)),
            'gap_min': float(np.min(gaps)),
            'gap_max': float(np.max(gaps)),
            'val_acc_mean': float(np.mean(val_accs)),
            'val_acc_std': float(np.std(val_accs)),
        }

        print(f"\n  Summary: gap = {stats['gap_mean']:.3f} ± {stats['gap_std']:.3f}")

        results['lenses'][f'{concept}_L{layer}'] = {
            'concept': concept,
            'layer': layer,
            'runs': runs,
            'statistics': stats,
        }

    # Overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)

    all_gaps = []
    all_stds = []
    for lens_key, lens_data in results['lenses'].items():
        if 'statistics' in lens_data:
            all_gaps.append(lens_data['statistics']['gap_mean'])
            all_stds.append(lens_data['statistics']['gap_std'])
            print(f"  {lens_key}: {lens_data['statistics']['gap_mean']:.3f} ± {lens_data['statistics']['gap_std']:.3f}")

    if all_stds:
        avg_std = np.mean(all_stds)
        print(f"\n  Average inter-run std dev: {avg_std:.3f}")
        print(f"  This suggests ~{avg_std * 2:.3f} as a reasonable error margin for comparisons")

    results['overall_statistics'] = {
        'avg_gap_std': float(np.mean(all_stds)) if all_stds else None,
        'suggested_error_margin': float(np.mean(all_stds) * 2) if all_stds else None,
    }

    output_file = output_dir / 'variance_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
