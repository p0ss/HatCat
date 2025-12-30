#!/usr/bin/env python3
"""
Generalist Baseline Experiment

The hypothesis: There's a "concept-like" direction in embedding space that
all trained lenses learn. By first training on a diverse set of unrelated
concepts, we can identify this common direction and subtract it from
future lens training.

Approach:
1. Train lenses on N diverse, unrelated concepts (the "generalist set")
2. Extract the common direction from their first-layer weights
3. Train new lenses on target concepts WITH the common direction subtracted
4. Compare calibration before/after

This is the PROACTIVE version - we compute the bias BEFORE training.
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


# Generalist concepts - deliberately diverse to capture "general concept-ness"
GENERALIST_CONCEPTS = [
    # Physical objects
    ('Artifact', 1),
    ('Device', 2),
    ('Furniture', 2),
    # Living things
    ('Organism', 1),
    ('Plant', 2),
    # Abstract
    ('Quantity', 1),
    ('Process', 1),
    ('Attribute', 1),
    # Actions
    ('Motion', 2),
    ('Communication', 2),
]

# Test concepts to evaluate improvement
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


def get_siblings(hierarchy: Dict, concept: str, layer: int) -> List[Tuple[str, Dict]]:
    """Get sibling concepts (share same parent)."""
    concept_data = hierarchy.get(layer, {}).get(concept, {})
    parents = set(concept_data.get('parent_concepts', []))

    siblings = []
    for other_concept, other_data in hierarchy.get(layer, {}).items():
        if other_concept == concept:
            continue
        other_parents = set(other_data.get('parent_concepts', []))
        if parents.intersection(other_parents):
            siblings.append((other_concept, other_data))

    return siblings


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
                max_epochs: int = 100, patience: int = 10) -> Tuple[LensClassifier, Dict]:
    """Train a lens."""
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

    return lens, {'train_acc': train_acc, 'val_acc': val_acc, 'epochs': epoch + 1}


def get_first_layer_weights(lens: LensClassifier) -> torch.Tensor:
    """Extract the first layer weights (shape: 128 x 4096)."""
    return lens.model[0].weight.data.clone()


def compute_common_direction(lenses: List[LensClassifier]) -> torch.Tensor:
    """Compute the mean first-layer weight direction across all lenses."""
    all_weights = []
    for lens in lenses:
        weights = get_first_layer_weights(lens)  # 128 x 4096
        # Normalize each row
        normalized = weights / (weights.norm(dim=1, keepdim=True) + 1e-8)
        all_weights.append(normalized)

    stacked = torch.stack(all_weights, dim=0)  # N x 128 x 4096
    mean_direction = stacked.mean(dim=0)  # 128 x 4096
    return mean_direction


def train_lens_with_offset(positive_emb: torch.Tensor, negative_emb: torch.Tensor,
                            common_direction: torch.Tensor, offset_strength: float = 1.0,
                            max_epochs: int = 100, patience: int = 10) -> Tuple[LensClassifier, Dict]:
    """
    Train a lens and subtract the common direction from its first layer weights.

    We do this DURING training by modifying the weights after each epoch.
    """
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

        # Apply offset: subtract common direction from first layer weights
        with torch.no_grad():
            weights = lens.model[0].weight.data
            for i in range(weights.shape[0]):
                w = weights[i]
                c = common_direction[i]
                c_norm = c / (c.norm() + 1e-8)
                projection = (w @ c_norm) * c_norm
                weights[i] = w - offset_strength * projection

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

    return lens, {'train_acc': train_acc, 'val_acc': val_acc, 'epochs': epoch + 1}


def evaluate_sibling_discrimination(lens: LensClassifier,
                                     positive_emb: torch.Tensor,
                                     sibling_emb: torch.Tensor,
                                     distant_emb: torch.Tensor) -> Dict:
    """Evaluate lens on positives, siblings, and distant concepts."""
    lens.eval()
    with torch.no_grad():
        pos_scores = lens(positive_emb).squeeze().numpy() if len(positive_emb) > 0 else np.array([])
        sib_scores = lens(sibling_emb).squeeze().numpy() if len(sibling_emb) > 0 else np.array([])
        dist_scores = lens(distant_emb).squeeze().numpy() if len(distant_emb) > 0 else np.array([])

    if pos_scores.ndim == 0:
        pos_scores = np.array([pos_scores.item()])
    if sib_scores.ndim == 0:
        sib_scores = np.array([sib_scores.item()])
    if dist_scores.ndim == 0:
        dist_scores = np.array([dist_scores.item()])

    avg_pos = float(np.mean(pos_scores)) if len(pos_scores) > 0 else None
    avg_sib = float(np.mean(sib_scores)) if len(sib_scores) > 0 else None
    avg_dist = float(np.mean(dist_scores)) if len(dist_scores) > 0 else None

    return {
        'avg_positive': avg_pos,
        'avg_sibling': avg_sib,
        'avg_distant': avg_dist,
        'positive_sibling_gap': float(avg_pos - avg_sib) if avg_pos and avg_sib else None,
        'positive_distant_gap': float(avg_pos - avg_dist) if avg_pos and avg_dist else None,
    }


def main():
    parser = argparse.ArgumentParser(description='Generalist baseline experiment')
    parser.add_argument('--model', default='swiss-ai/Apertus-8B-2509')
    parser.add_argument('--layers-dir', default='data/concept_graph/v4')
    parser.add_argument('--output-dir', default='results/generalist_baseline_experiment')
    parser.add_argument('--offset-strengths', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0])
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

    # Phase 1: Train generalist lenses to find common direction
    print("\n" + "="*60)
    print("PHASE 1: Training generalist lenses")
    print("="*60)

    generalist_lenses = []
    for concept, layer in GENERALIST_CONCEPTS:
        concept_data = hierarchy.get(layer, {}).get(concept)
        if not concept_data:
            print(f"  Skipping {concept} (not found)")
            continue

        pos_defs = get_concept_definitions(concept_data, concept)
        if len(pos_defs) < 5:
            print(f"  Skipping {concept} (insufficient definitions)")
            continue

        pos_emb = get_embeddings(model, tokenizer, pos_defs, args.target_layer)

        distant = get_distant_concepts(hierarchy, concept, layer, count=10)
        distant_defs = []
        for dist_concept, dist_data in distant:
            defs = get_concept_definitions(dist_data, dist_concept, max_samples=5)
            distant_defs.extend(defs)

        neg_emb = get_embeddings(model, tokenizer, distant_defs, args.target_layer)

        if len(neg_emb) < 5:
            print(f"  Skipping {concept} (insufficient negatives)")
            continue

        print(f"  Training generalist lens: {concept}...")
        lens, metrics = train_lens(pos_emb, neg_emb)
        generalist_lenses.append(lens)
        print(f"    Val acc: {metrics['val_acc']:.3f}")

    print(f"\nTrained {len(generalist_lenses)} generalist lenses")

    # Compute common direction
    print("\nComputing common direction from generalist lenses...")
    common_direction = compute_common_direction(generalist_lenses)
    print(f"  Common direction shape: {common_direction.shape}")
    print(f"  Common direction norm: {common_direction.norm():.4f}")

    # Save common direction
    torch.save(common_direction, output_dir / 'generalist_common_direction.pt')

    # Phase 2: Train test lenses with and without offset
    print("\n" + "="*60)
    print("PHASE 2: Training test lenses with offset")
    print("="*60)

    results = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'n_generalist_lenses': len(generalist_lenses),
        'offset_strengths': args.offset_strengths,
        'lenses': {},
    }

    for concept, layer in TEST_CONCEPTS:
        print(f"\n{'='*60}")
        print(f"Testing {concept} (layer {layer})")
        print('='*60)

        concept_data = hierarchy.get(layer, {}).get(concept)
        if not concept_data:
            print(f"  ERROR: Concept not found")
            results['lenses'][f'{concept}_L{layer}'] = {'error': 'Concept not found'}
            continue

        pos_defs = get_concept_definitions(concept_data, concept)
        pos_emb = get_embeddings(model, tokenizer, pos_defs, args.target_layer)

        if len(pos_emb) < 5:
            print(f"  ERROR: Insufficient positive samples")
            results['lenses'][f'{concept}_L{layer}'] = {'error': 'Insufficient samples'}
            continue

        # Get siblings
        siblings = get_siblings(hierarchy, concept, layer)
        sibling_defs = []
        for sib_concept, sib_data in siblings[:5]:
            defs = get_concept_definitions(sib_data, sib_concept, max_samples=5)
            sibling_defs.extend(defs)
        sibling_emb = get_embeddings(model, tokenizer, sibling_defs, args.target_layer)

        # Get distant
        distant = get_distant_concepts(hierarchy, concept, layer, count=10)
        distant_defs = []
        for dist_concept, dist_data in distant:
            defs = get_concept_definitions(dist_data, dist_concept, max_samples=5)
            distant_defs.extend(defs)
        distant_emb = get_embeddings(model, tokenizer, distant_defs, args.target_layer)

        if len(distant_emb) < 5:
            print(f"  ERROR: Insufficient distant samples")
            results['lenses'][f'{concept}_L{layer}'] = {'error': 'Insufficient distant'}
            continue

        # Split for training/eval
        eval_pos = pos_emb[-3:]
        train_pos = pos_emb[:-3]
        eval_sib = sibling_emb[-3:] if len(sibling_emb) >= 3 else sibling_emb
        eval_dist = distant_emb[-5:]
        train_neg = distant_emb[:-5]

        if len(train_pos) < 3 or len(train_neg) < 3:
            print(f"  ERROR: Insufficient training samples after split")
            results['lenses'][f'{concept}_L{layer}'] = {'error': 'Insufficient after split'}
            continue

        print(f"  Train pos: {len(train_pos)}, Train neg: {len(train_neg)}")
        print(f"  Eval pos: {len(eval_pos)}, Eval sib: {len(eval_sib)}, Eval dist: {len(eval_dist)}")

        lens_results = {
            'concept': concept,
            'layer': layer,
            'modes': {},
        }

        # Baseline (no offset)
        print(f"\n  Training BASELINE (no offset)...")
        lens_base, train_metrics_base = train_lens(train_pos, train_neg)
        eval_base = evaluate_sibling_discrimination(lens_base, eval_pos, eval_sib, eval_dist)
        lens_results['modes']['baseline'] = {
            'training': train_metrics_base,
            'evaluation': eval_base,
        }
        print(f"    Val acc: {train_metrics_base['val_acc']:.3f}")
        pos_sib_gap = eval_base['positive_sibling_gap']
        pos_dist_gap = eval_base['positive_distant_gap']
        print(f"    Pos-Sib gap: {pos_sib_gap:.3f}" if pos_sib_gap is not None else "    Pos-Sib gap: N/A")
        print(f"    Pos-Dist gap: {pos_dist_gap:.3f}" if pos_dist_gap is not None else "    Pos-Dist gap: N/A")

        # With offset at different strengths
        for strength in args.offset_strengths:
            print(f"\n  Training with OFFSET (strength={strength})...")
            lens_offset, train_metrics_offset = train_lens_with_offset(
                train_pos, train_neg, common_direction, offset_strength=strength
            )
            eval_offset = evaluate_sibling_discrimination(lens_offset, eval_pos, eval_sib, eval_dist)
            lens_results['modes'][f'offset_{strength}'] = {
                'training': train_metrics_offset,
                'evaluation': eval_offset,
            }
            print(f"    Val acc: {train_metrics_offset['val_acc']:.3f}")
            off_sib_gap = eval_offset['positive_sibling_gap']
            off_dist_gap = eval_offset['positive_distant_gap']
            print(f"    Pos-Sib gap: {off_sib_gap:.3f}" if off_sib_gap is not None else "    Pos-Sib gap: N/A")
            print(f"    Pos-Dist gap: {off_dist_gap:.3f}" if off_dist_gap is not None else "    Pos-Dist gap: N/A")

        results['lenses'][f'{concept}_L{layer}'] = lens_results

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for mode in ['baseline'] + [f'offset_{s}' for s in args.offset_strengths]:
        sib_gaps = []
        dist_gaps = []
        for lens_key, lens_data in results['lenses'].items():
            if 'error' in lens_data:
                continue
            if mode in lens_data.get('modes', {}):
                eval_data = lens_data['modes'][mode].get('evaluation', {})
                if eval_data.get('positive_sibling_gap') is not None:
                    sib_gaps.append(eval_data['positive_sibling_gap'])
                if eval_data.get('positive_distant_gap') is not None:
                    dist_gaps.append(eval_data['positive_distant_gap'])

        if sib_gaps:
            print(f"\n{mode}: ({len(sib_gaps)} lenses)")
            print(f"  Avg pos-sib gap:  {np.mean(sib_gaps):.3f} ± {np.std(sib_gaps):.3f}")
            print(f"  Avg pos-dist gap: {np.mean(dist_gaps):.3f} ± {np.std(dist_gaps):.3f}")

    # Save results
    output_file = output_dir / 'generalist_baseline_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
