#!/usr/bin/env python3
"""
Contrastive Training Experiment - Sibling Hard Negatives

Hypothesis: Lenses fail at sibling discrimination because training negatives
are from distant concepts, not siblings. Siblings produce similar activations
in the model, so lenses never learn to distinguish them.

This experiment trains lenses with:
1. Standard negatives (distant concepts) - baseline
2. Hard negatives (siblings only) - contrastive
3. Mixed negatives (50% sibling, 50% distant) - balanced

We test on a subset of benchmark lenses to measure impact.
"""

import argparse
import json
import sys
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Benchmark lenses - concepts with known siblings in v4 hierarchy
# Layer 1, 2, 3, 4 have concepts with parents (and thus siblings)
BENCHMARK_LENSS = {
    # Layer 1 - abstract categories
    ('Agent', 1, 'test'),
    ('AlignmentProcess', 1, 'test'),
    ('AnatomicalStructure', 1, 'test'),
    # Layer 2 - more specific concepts
    ('AIAlignmentProcess', 2, 'test'),
    ('Projectile', 2, 'test'),
    ('Animal', 2, 'test'),
    # Layer 3 - concrete concepts
    ('AAV', 3, 'test'),
    ('ACPowerSource', 3, 'test'),
    ('AH1', 3, 'test'),
    # Layer 4 - most specific
    ('AIAlignment', 4, 'test'),
    ('ADHD', 4, 'test'),
}


class LensClassifier(nn.Module):
    """MLP lens classifier."""
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


def capture_activation(
    text: str,
    model,
    tokenizer,
    target_layer: int = 15
) -> torch.Tensor:
    """Capture activation for a text sample at the target layer."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[target_layer]
        activation = hidden_states[:, -1, :]

    return activation.float().cpu()


def get_concept_definitions(concept_data: Dict, max_samples: int = 50) -> List[str]:
    """Get definitions for a concept from hierarchy data."""
    definitions = []

    # Try different definition sources
    if concept_data.get('definitions'):
        definitions = concept_data['definitions']
    elif concept_data.get('definition'):
        definitions = [concept_data['definition']]
    elif concept_data.get('sumo_definition'):
        definitions = [concept_data['sumo_definition']]

    # Add lemma-based samples if we have them
    if concept_data.get('lemmas'):
        for lemma in concept_data['lemmas'][:5]:
            definitions.append(f"The concept of {lemma}")
            definitions.append(f"This relates to {lemma}")

    # Fallback: use concept name
    if not definitions:
        name = concept_data.get('sumo_term', concept_data.get('name', ''))
        if name:
            definitions = [
                f"The concept of {name}",
                f"This relates to {name}",
                f"An instance of {name}",
            ]

    return definitions[:max_samples]


def load_hierarchy_with_siblings(layers_dir: Path) -> Dict:
    """Load hierarchy and compute sibling relationships."""
    hierarchy = {}
    parent_to_children = defaultdict(list)

    for layer in range(5):
        layer_file = layers_dir / f'layer{layer}.json'
        if layer_file.exists():
            with open(layer_file) as f:
                raw_data = json.load(f)

            # Handle both formats: {concepts: [...]} or direct dict
            if isinstance(raw_data, dict) and 'concepts' in raw_data:
                concepts_list = raw_data['concepts']
            elif isinstance(raw_data, list):
                concepts_list = raw_data
            else:
                # Assume it's already a dict mapping names to data
                concepts_list = [{'sumo_term': k, **v} for k, v in raw_data.items()]

            # Convert list to dict keyed by sumo_term
            layer_data = {}
            for concept in concepts_list:
                name = concept.get('sumo_term', concept.get('name', ''))
                if name:
                    layer_data[name] = concept

            hierarchy[layer] = layer_data

            # Build parent-child mapping for sibling lookup
            for concept_name, concept_data in layer_data.items():
                parents = concept_data.get('parent_concepts', [])
                for parent in parents:
                    parent_to_children[parent].append((concept_name, layer))

    # Add siblings to each concept
    for layer, layer_data in hierarchy.items():
        for concept_name, concept_data in layer_data.items():
            siblings = set()
            parents = concept_data.get('parent_concepts', [])
            for parent in parents:
                for sibling_name, sibling_layer in parent_to_children[parent]:
                    if sibling_name != concept_name:
                        siblings.add((sibling_name, sibling_layer))
            concept_data['siblings'] = list(siblings)

    return hierarchy


def generate_training_data(
    concept_name: str,
    layer: int,
    hierarchy: Dict,
    model,
    tokenizer,
    n_pos: int = 30,
    n_neg_sibling: int = 15,
    n_neg_distant: int = 15,
    negative_mode: str = 'mixed'  # 'standard', 'sibling', 'mixed'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate training data with configurable negative sampling.

    negative_mode:
    - 'standard': all negatives from distant concepts (baseline)
    - 'sibling': all negatives from sibling concepts (contrastive)
    - 'mixed': 50% sibling, 50% distant (balanced)
    """
    concept_data = hierarchy[layer].get(concept_name)
    if not concept_data:
        raise ValueError(f"Concept {concept_name} not found in layer {layer}")

    # Get positive samples
    pos_definitions = get_concept_definitions(concept_data, n_pos * 2)
    if len(pos_definitions) < 5:
        raise ValueError(f"Not enough definitions for {concept_name}: {len(pos_definitions)}")

    pos_activations = []
    for defn in pos_definitions[:n_pos]:
        try:
            act = capture_activation(defn, model, tokenizer)
            pos_activations.append(act)
        except Exception as e:
            continue

    if len(pos_activations) < 5:
        raise ValueError(f"Could not generate enough positive activations for {concept_name}")

    # Get negative samples based on mode
    neg_activations = []

    if negative_mode in ['sibling', 'mixed']:
        # Get sibling negatives
        siblings = concept_data.get('siblings', [])
        n_sibling_needed = n_neg_sibling if negative_mode == 'sibling' else n_neg_sibling

        if negative_mode == 'sibling':
            n_sibling_needed = n_neg_sibling + n_neg_distant  # All negatives from siblings

        sibling_defs = []
        for sib_name, sib_layer in siblings:
            if sib_layer in hierarchy:
                sib_data = hierarchy[sib_layer].get(sib_name)
                if sib_data:
                    sib_defs = get_concept_definitions(sib_data, 5)
                    sibling_defs.extend(sib_defs)

        random.shuffle(sibling_defs)
        for defn in sibling_defs[:n_sibling_needed]:
            try:
                act = capture_activation(defn, model, tokenizer)
                neg_activations.append(act)
            except:
                continue

    if negative_mode in ['standard', 'mixed']:
        # Get distant negatives
        n_distant_needed = n_neg_distant if negative_mode == 'mixed' else n_neg_sibling + n_neg_distant

        if negative_mode == 'standard':
            n_distant_needed = n_neg_sibling + n_neg_distant  # All negatives from distant

        # Collect distant concepts (different layer or unrelated)
        distant_defs = []
        for other_layer, other_data in hierarchy.items():
            if abs(other_layer - layer) >= 2:  # At least 2 layers away
                for other_name, other_concept in other_data.items():
                    if other_name != concept_name:
                        defs = get_concept_definitions(other_concept, 3)
                        distant_defs.extend(defs)

        random.shuffle(distant_defs)
        for defn in distant_defs[:n_distant_needed]:
            try:
                act = capture_activation(defn, model, tokenizer)
                neg_activations.append(act)
            except:
                continue

    if len(neg_activations) < 5:
        raise ValueError(f"Could not generate enough negative activations for {concept_name}")

    # Create tensors
    X_pos = torch.cat(pos_activations, dim=0)
    X_neg = torch.cat(neg_activations, dim=0)

    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat([
        torch.ones(len(pos_activations)),
        torch.zeros(len(neg_activations))
    ]).unsqueeze(1)

    return X, y


def train_lens(
    X: torch.Tensor,
    y: torch.Tensor,
    input_dim: int = 4096,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = 'cuda'
) -> Tuple[LensClassifier, Dict]:
    """Train a lens classifier."""
    lens = LensClassifier(input_dim).to(device)
    optimizer = torch.optim.Adam(lens.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Split into train/val
    n = len(X)
    indices = torch.randperm(n)
    train_idx = indices[:int(0.8 * n)]
    val_idx = indices[int(0.8 * n):]

    X_train, y_train = X[train_idx].to(device), y[train_idx].to(device)
    X_val, y_val = X[val_idx].to(device), y[val_idx].to(device)

    best_val_loss = float('inf')
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        lens.train()
        optimizer.zero_grad()
        pred = lens(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        lens.eval()
        with torch.no_grad():
            val_pred = lens(X_val)
            val_loss = criterion(val_pred, y_val).item()

            # Compute metrics
            val_pred_binary = (val_pred > 0.5).float()
            accuracy = (val_pred_binary == y_val).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = lens.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        lens.load_state_dict(best_state)

    # Final evaluation
    lens.eval()
    with torch.no_grad():
        train_pred = (lens(X_train) > 0.5).float()
        train_acc = (train_pred == y_train).float().mean().item()

        val_pred = (lens(X_val) > 0.5).float()
        val_acc = (val_pred == y_val).float().mean().item()

        # F1 score
        tp = ((val_pred == 1) & (y_val == 1)).sum().item()
        fp = ((val_pred == 1) & (y_val == 0)).sum().item()
        fn = ((val_pred == 0) & (y_val == 1)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'val_f1': f1,
        'epochs': epoch + 1,
    }

    return lens, metrics


def evaluate_sibling_discrimination(
    lens: LensClassifier,
    concept_name: str,
    layer: int,
    hierarchy: Dict,
    model,
    tokenizer,
    device: str = 'cuda'
) -> Dict:
    """Evaluate how well the lens discriminates against siblings AND distant concepts."""
    concept_data = hierarchy[layer].get(concept_name)
    if not concept_data:
        return {}

    lens.eval()
    lens.to(device)

    # Score positive samples
    pos_defs = get_concept_definitions(concept_data, 10)
    pos_scores = []
    for defn in pos_defs[:5]:
        try:
            act = capture_activation(defn, model, tokenizer).to(device)
            with torch.no_grad():
                score = lens(act).item()
            pos_scores.append(score)
        except:
            continue

    # Score sibling samples
    siblings = concept_data.get('siblings', [])
    sibling_scores = []
    for sib_name, sib_layer in siblings[:5]:
        if sib_layer in hierarchy:
            sib_data = hierarchy[sib_layer].get(sib_name)
            if sib_data:
                sib_defs = get_concept_definitions(sib_data, 3)
                for defn in sib_defs[:2]:
                    try:
                        act = capture_activation(defn, model, tokenizer).to(device)
                        with torch.no_grad():
                            score = lens(act).item()
                        sibling_scores.append(score)
                    except:
                        continue

    # Score distant/neutral concepts (from different layers, unrelated)
    distant_scores = []
    for other_layer in hierarchy:
        if abs(other_layer - layer) >= 2:  # At least 2 layers away
            other_concepts = list(hierarchy[other_layer].items())[:10]
            for other_name, other_data in other_concepts:
                if other_name != concept_name:
                    other_defs = get_concept_definitions(other_data, 2)
                    for defn in other_defs[:1]:
                        try:
                            act = capture_activation(defn, model, tokenizer).to(device)
                            with torch.no_grad():
                                score = lens(act).item()
                            distant_scores.append(score)
                        except:
                            continue
                if len(distant_scores) >= 10:
                    break
            if len(distant_scores) >= 10:
                break

    avg_pos = np.mean(pos_scores) if pos_scores else None
    avg_sib = np.mean(sibling_scores) if sibling_scores else None
    avg_dist = np.mean(distant_scores) if distant_scores else None

    return {
        'avg_positive': avg_pos,
        'avg_sibling': avg_sib,
        'avg_distant': avg_dist,
        'positive_sibling_gap': (avg_pos - avg_sib) if avg_pos is not None and avg_sib is not None else None,
        'positive_distant_gap': (avg_pos - avg_dist) if avg_pos is not None and avg_dist is not None else None,
        'sibling_distant_gap': (avg_sib - avg_dist) if avg_sib is not None and avg_dist is not None else None,
        'n_positive_samples': len(pos_scores),
        'n_sibling_samples': len(sibling_scores),
        'n_distant_samples': len(distant_scores),
    }


def main():
    parser = argparse.ArgumentParser(description='Contrastive training experiment')
    parser.add_argument('--model', default='swiss-ai/Apertus-8B-2509',
                        help='Model name')
    parser.add_argument('--layers-dir', default='data/concept_graph/abstraction_layers',
                        help='Directory containing layer hierarchy files')
    parser.add_argument('--output-dir', default='results/contrastive_experiment',
                        help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--max-concepts', type=int, default=None,
                        help='Max concepts to test (default: all benchmark)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.eval()

    print("Loading hierarchy with siblings...")
    hierarchy = load_hierarchy_with_siblings(Path(args.layers_dir))

    # Count siblings per concept
    for layer, layer_data in hierarchy.items():
        sibling_counts = [len(c.get('siblings', [])) for c in layer_data.values()]
        print(f"  Layer {layer}: {len(layer_data)} concepts, avg {np.mean(sibling_counts):.1f} siblings")

    results = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'modes': ['standard', 'sibling', 'mixed'],
        'lenses': {}
    }

    test_lenses = list(BENCHMARK_LENSS)
    if args.max_concepts:
        test_lenses = test_lenses[:args.max_concepts]

    print(f"\nTesting {len(test_lenses)} benchmark lenses...")

    for concept_name, layer, category in tqdm(test_lenses):
        print(f"\n{'='*60}")
        print(f"Concept: {concept_name} (Layer {layer}, {category})")
        print(f"{'='*60}")

        lens_results = {
            'concept': concept_name,
            'layer': layer,
            'category': category,
            'modes': {}
        }

        for mode in ['standard', 'sibling', 'mixed']:
            print(f"\n  Training with {mode} negatives...")
            try:
                X, y = generate_training_data(
                    concept_name, layer, hierarchy, model, tokenizer,
                    n_pos=30, n_neg_sibling=15, n_neg_distant=15,
                    negative_mode=mode
                )
                print(f"    Generated {len(X)} samples ({(y==1).sum().item()} pos, {(y==0).sum().item()} neg)")

                lens, train_metrics = train_lens(X, y, device=args.device)
                print(f"    Train acc: {train_metrics['train_acc']:.3f}, Val F1: {train_metrics['val_f1']:.3f}")

                eval_metrics = evaluate_sibling_discrimination(
                    lens, concept_name, layer, hierarchy, model, tokenizer, args.device
                )

                if eval_metrics.get('positive_sibling_gap') is not None:
                    print(f"    Pos-Sibling gap: {eval_metrics['positive_sibling_gap']:.4f}")
                    print(f"    Pos-Distant gap: {eval_metrics.get('positive_distant_gap', 0):.4f}")
                    print(f"    (pos={eval_metrics['avg_positive']:.4f}, sib={eval_metrics['avg_sibling']:.4f}, dist={eval_metrics.get('avg_distant', 0):.4f})")

                lens_results['modes'][mode] = {
                    'training': train_metrics,
                    'evaluation': eval_metrics,
                }

            except Exception as e:
                print(f"    ERROR: {e}")
                lens_results['modes'][mode] = {'error': str(e)}

        results['lenses'][f"{concept_name}_L{layer}"] = lens_results

        # Save intermediate results
        with open(output_dir / 'contrastive_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for mode in ['standard', 'sibling', 'mixed']:
        sib_gaps = []
        dist_gaps = []
        for lens_key, lens_data in results['lenses'].items():
            mode_data = lens_data['modes'].get(mode, {})
            eval_data = mode_data.get('evaluation', {})
            sib_gap = eval_data.get('positive_sibling_gap')
            dist_gap = eval_data.get('positive_distant_gap')
            if sib_gap is not None:
                sib_gaps.append(sib_gap)
            if dist_gap is not None:
                dist_gaps.append(dist_gap)

        if sib_gaps:
            print(f"\n{mode.upper()} negatives:")
            print(f"  Pos-Sibling gap: {np.mean(sib_gaps):.4f} (+/- {np.std(sib_gaps):.4f})")
            if dist_gaps:
                print(f"  Pos-Distant gap: {np.mean(dist_gaps):.4f} (+/- {np.std(dist_gaps):.4f})")
            print(f"  Sibling min: {np.min(sib_gaps):.4f}, max: {np.max(sib_gaps):.4f}")

    print(f"\nResults saved to {output_dir / 'contrastive_results.json'}")


if __name__ == '__main__':
    main()
