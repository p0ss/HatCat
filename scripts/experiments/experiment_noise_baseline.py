#!/usr/bin/env python3
"""
Noise baseline experiment for lens training.

Tests two hypotheses:
1. How do lenses trained with random/noise negatives compare to structured negatives?
2. Do existing lenses fire on random noise inputs (testing specificity)?

Training modes:
- 'standard': negatives from distant concepts (existing baseline)
- 'noise': negatives are random gaussian noise in embedding space
- 'random_text': negatives are embeddings from random unrelated text

This establishes whether our lenses are learning concept-specific features
or just "anything that looks like text vs noise".
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
from datetime import datetime
import random

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# Test concepts across layers - selected for having sufficient data
TEST_CONCEPTS = [
    # Layer 1 - broad concepts
    ('AlignmentProcess', 1),
    ('Agent', 1),
    # Layer 2 - mid-level
    ('Projectile', 2),
    ('AIAlignmentProcess', 2),
    # Layer 3 - specific
    ('AAV', 3),
    ('ACPowerSource', 3),
    ('AH1', 3),
    # Layer 4 - very specific
    ('ADHD', 4),
]

# Random text samples for generating "random text" negatives
RANDOM_TEXT_SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Lorem ipsum dolor sit amet consectetur adipiscing elit.",
    "To be or not to be that is the question.",
    "Four score and seven years ago our fathers brought forth.",
    "It was the best of times it was the worst of times.",
    "Call me Ishmael some years ago never mind how long.",
    "All happy families are alike each unhappy family is unhappy.",
    "It is a truth universally acknowledged that a single man.",
    "In the beginning God created the heavens and the earth.",
    "The only thing we have to fear is fear itself.",
    "Ask not what your country can do for you.",
    "I have a dream that one day this nation will rise.",
    "That's one small step for man one giant leap for mankind.",
    "Elementary my dear Watson the game is afoot.",
    "Here's looking at you kid.",
    "May the Force be with you always.",
    "I'll be back.",
    "You can't handle the truth.",
    "There's no place like home.",
    "Life is like a box of chocolates.",
    "Houston we have a problem.",
    "Winter is coming.",
    "The night is dark and full of terrors.",
    "Veni vidi vici.",
    "Et tu Brute?",
    "Cogito ergo sum.",
    "E pluribus unum.",
    "Carpe diem seize the day.",
    "All that glitters is not gold.",
    "A rose by any other name would smell as sweet.",
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

    # Add lemma-based samples
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

    # Look in other layers
    for other_layer in range(5):
        if other_layer == layer:
            continue
        for other_concept, other_data in hierarchy.get(other_layer, {}).items():
            other_parents = set(other_data.get('parent_concepts', []))
            # No overlap in parents = distant
            if not current_parents.intersection(other_parents):
                distant.append((other_concept, other_data))

    random.shuffle(distant)
    return distant[:count]


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


def get_embeddings(model, tokenizer, texts: List[str], target_layer: int = 15) -> torch.Tensor:
    """Get embeddings for a list of texts."""
    embeddings = []
    for text in texts:
        try:
            emb = capture_activation(text, model, tokenizer, target_layer)
            embeddings.append(emb.squeeze(0))
        except Exception as e:
            print(f"  Warning: Failed to embed text: {e}")
            continue

    if embeddings:
        return torch.stack(embeddings)
    return torch.empty(0, 4096)


def generate_noise_embeddings(count: int, dim: int = 4096,
                              mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Generate random Gaussian noise embeddings."""
    return torch.randn(count, dim) * std + mean


def train_lens(positive_emb: torch.Tensor, negative_emb: torch.Tensor,
                max_epochs: int = 100, patience: int = 10) -> Tuple[LensClassifier, Dict]:
    """Train a lens with given positive and negative embeddings."""

    # Create labels
    pos_labels = torch.ones(len(positive_emb), 1)
    neg_labels = torch.zeros(len(negative_emb), 1)

    # Combine
    X = torch.cat([positive_emb, negative_emb], dim=0)
    y = torch.cat([pos_labels, neg_labels], dim=0)

    # Shuffle
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    # Split train/val
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    if len(X_val) == 0:
        X_val, y_val = X_train[-2:], y_train[-2:]

    # Create model
    lens = LensClassifier(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(lens.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):
        # Train
        lens.train()
        optimizer.zero_grad()
        pred = lens(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        # Validate
        lens.eval()
        with torch.no_grad():
            val_pred = lens(X_val)
            val_loss = criterion(val_pred, y_val).item()
            val_acc = ((val_pred > 0.5) == y_val).float().mean().item()

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

    # Final metrics
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
    }


def evaluate_lens(lens: LensClassifier,
                   positive_emb: torch.Tensor,
                   negative_emb: torch.Tensor,
                   noise_emb: torch.Tensor) -> Dict:
    """Evaluate lens on positives, structured negatives, and noise."""
    lens.eval()
    with torch.no_grad():
        pos_scores = lens(positive_emb).squeeze().numpy() if len(positive_emb) > 0 else np.array([])
        neg_scores = lens(negative_emb).squeeze().numpy() if len(negative_emb) > 0 else np.array([])
        noise_scores = lens(noise_emb).squeeze().numpy() if len(noise_emb) > 0 else np.array([])

    # Handle scalar case
    if pos_scores.ndim == 0:
        pos_scores = np.array([pos_scores.item()])
    if neg_scores.ndim == 0:
        neg_scores = np.array([neg_scores.item()])
    if noise_scores.ndim == 0:
        noise_scores = np.array([noise_scores.item()])

    return {
        'avg_positive': float(np.mean(pos_scores)) if len(pos_scores) > 0 else None,
        'avg_negative': float(np.mean(neg_scores)) if len(neg_scores) > 0 else None,
        'avg_noise': float(np.mean(noise_scores)) if len(noise_scores) > 0 else None,
        'positive_negative_gap': float(np.mean(pos_scores) - np.mean(neg_scores)) if len(pos_scores) > 0 and len(neg_scores) > 0 else None,
        'positive_noise_gap': float(np.mean(pos_scores) - np.mean(noise_scores)) if len(pos_scores) > 0 and len(noise_scores) > 0 else None,
        'negative_noise_gap': float(np.mean(neg_scores) - np.mean(noise_scores)) if len(neg_scores) > 0 and len(noise_scores) > 0 else None,
        'n_positive': len(pos_scores),
        'n_negative': len(neg_scores),
        'n_noise': len(noise_scores),
    }


def main():
    parser = argparse.ArgumentParser(description='Noise baseline experiment')
    parser.add_argument('--model', default='swiss-ai/Apertus-8B-2509')
    parser.add_argument('--layers-dir', default='data/concept_graph/v4')
    parser.add_argument('--output-dir', default='results/noise_baseline_experiment')
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

    # Get random text embeddings for 'random_text' mode
    print("Generating random text embeddings...")
    random_text_emb = get_embeddings(model, tokenizer, RANDOM_TEXT_SAMPLES, args.target_layer)
    print(f"  Got {len(random_text_emb)} random text embeddings")

    # Compute embedding statistics for calibrated noise
    print("Computing embedding statistics...")
    sample_texts = []
    for layer in range(5):
        for concept, data in list(hierarchy.get(layer, {}).items())[:10]:
            defs = get_concept_definitions(data, concept, max_samples=3)
            sample_texts.extend(defs[:3])

    sample_emb = get_embeddings(model, tokenizer, sample_texts[:100], args.target_layer)
    emb_mean = sample_emb.mean().item()
    emb_std = sample_emb.std().item()
    print(f"  Embedding stats: mean={emb_mean:.4f}, std={emb_std:.4f}")

    results = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'embedding_stats': {'mean': emb_mean, 'std': emb_std},
        'modes': ['standard', 'noise', 'random_text'],
        'lenses': {},
    }

    for concept, layer in TEST_CONCEPTS:
        print(f"\n{'='*60}")
        print(f"Testing {concept} (layer {layer})")
        print('='*60)

        # Get concept data
        concept_data = hierarchy.get(layer, {}).get(concept)
        if not concept_data:
            print(f"  ERROR: Concept not found in layer {layer}")
            results['lenses'][f'{concept}_L{layer}'] = {'error': f'Concept not found in layer {layer}'}
            continue

        # Get positive definitions
        pos_defs = get_concept_definitions(concept_data, concept, max_samples=50)
        if len(pos_defs) < args.min_definitions:
            print(f"  ERROR: Not enough definitions: {len(pos_defs)}")
            results['lenses'][f'{concept}_L{layer}'] = {'error': f'Not enough definitions: {len(pos_defs)}'}
            continue

        print(f"  Positive samples: {len(pos_defs)}")
        pos_emb = get_embeddings(model, tokenizer, pos_defs, args.target_layer)
        print(f"  Positive embeddings: {len(pos_emb)}")

        # Get distant concept definitions for structured negatives
        distant = get_distant_concepts(hierarchy, concept, layer, count=10)
        distant_defs = []
        for dist_concept, dist_data in distant:
            defs = get_concept_definitions(dist_data, dist_concept, max_samples=10)
            distant_defs.extend(defs[:5])

        print(f"  Distant negative samples: {len(distant_defs)}")
        distant_emb = get_embeddings(model, tokenizer, distant_defs, args.target_layer)
        print(f"  Distant negative embeddings: {len(distant_emb)}")

        if len(distant_emb) < 5:
            print(f"  ERROR: Not enough distant negatives")
            results['lenses'][f'{concept}_L{layer}'] = {'error': 'Not enough distant negatives'}
            continue

        # Generate noise embeddings (calibrated to embedding distribution)
        noise_emb = generate_noise_embeddings(len(distant_emb), dim=pos_emb.shape[1],
                                               mean=emb_mean, std=emb_std)

        # Prepare eval sets (held out)
        eval_pos = pos_emb[-5:]
        train_pos = pos_emb[:-5]
        eval_neg = distant_emb[-5:]
        train_neg = distant_emb[:-5]
        eval_noise = noise_emb[-5:]
        train_noise = noise_emb[:-5]
        eval_random_text = random_text_emb[-5:] if len(random_text_emb) >= 5 else random_text_emb
        train_random_text = random_text_emb[:-5] if len(random_text_emb) >= 5 else random_text_emb

        lens_results = {
            'concept': concept,
            'layer': layer,
            'modes': {},
        }

        # Mode 1: Standard (distant negatives)
        print(f"\n  Training STANDARD mode (distant negatives)...")
        try:
            lens_std, train_metrics_std = train_lens(train_pos, train_neg)
            eval_metrics_std = evaluate_lens(lens_std, eval_pos, eval_neg, eval_noise)
            lens_results['modes']['standard'] = {
                'training': train_metrics_std,
                'evaluation': eval_metrics_std,
            }
            print(f"    Train acc: {train_metrics_std['train_acc']:.3f}, Val acc: {train_metrics_std['val_acc']:.3f}")
            print(f"    Pos: {eval_metrics_std['avg_positive']:.3f}, Neg: {eval_metrics_std['avg_negative']:.3f}, Noise: {eval_metrics_std['avg_noise']:.3f}")
            print(f"    Pos-Neg gap: {eval_metrics_std['positive_negative_gap']:.3f}, Pos-Noise gap: {eval_metrics_std['positive_noise_gap']:.3f}")
        except Exception as e:
            lens_results['modes']['standard'] = {'error': str(e)}
            print(f"    ERROR: {e}")

        # Mode 2: Noise negatives
        print(f"\n  Training NOISE mode (gaussian noise negatives)...")
        try:
            lens_noise, train_metrics_noise = train_lens(train_pos, train_noise)
            eval_metrics_noise = evaluate_lens(lens_noise, eval_pos, eval_neg, eval_noise)
            lens_results['modes']['noise'] = {
                'training': train_metrics_noise,
                'evaluation': eval_metrics_noise,
            }
            print(f"    Train acc: {train_metrics_noise['train_acc']:.3f}, Val acc: {train_metrics_noise['val_acc']:.3f}")
            print(f"    Pos: {eval_metrics_noise['avg_positive']:.3f}, Neg: {eval_metrics_noise['avg_negative']:.3f}, Noise: {eval_metrics_noise['avg_noise']:.3f}")
            print(f"    Pos-Neg gap: {eval_metrics_noise['positive_negative_gap']:.3f}, Pos-Noise gap: {eval_metrics_noise['positive_noise_gap']:.3f}")
        except Exception as e:
            lens_results['modes']['noise'] = {'error': str(e)}
            print(f"    ERROR: {e}")

        # Mode 3: Random text negatives
        print(f"\n  Training RANDOM_TEXT mode (random text negatives)...")
        try:
            lens_rand, train_metrics_rand = train_lens(train_pos, train_random_text)
            eval_metrics_rand = evaluate_lens(lens_rand, eval_pos, eval_neg, eval_noise)
            lens_results['modes']['random_text'] = {
                'training': train_metrics_rand,
                'evaluation': eval_metrics_rand,
            }
            print(f"    Train acc: {train_metrics_rand['train_acc']:.3f}, Val acc: {train_metrics_rand['val_acc']:.3f}")
            print(f"    Pos: {eval_metrics_rand['avg_positive']:.3f}, Neg: {eval_metrics_rand['avg_negative']:.3f}, Noise: {eval_metrics_rand['avg_noise']:.3f}")
            print(f"    Pos-Neg gap: {eval_metrics_rand['positive_negative_gap']:.3f}, Pos-Noise gap: {eval_metrics_rand['positive_noise_gap']:.3f}")
        except Exception as e:
            lens_results['modes']['random_text'] = {'error': str(e)}
            print(f"    ERROR: {e}")

        results['lenses'][f'{concept}_L{layer}'] = lens_results

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    summary = {'standard': [], 'noise': [], 'random_text': []}

    for lens_key, lens_data in results['lenses'].items():
        if 'error' in lens_data:
            continue
        for mode in ['standard', 'noise', 'random_text']:
            if mode in lens_data.get('modes', {}) and 'evaluation' in lens_data['modes'][mode]:
                eval_data = lens_data['modes'][mode]['evaluation']
                if eval_data.get('positive_negative_gap') is not None:
                    summary[mode].append({
                        'lens': lens_key,
                        'pos_neg_gap': eval_data['positive_negative_gap'],
                        'pos_noise_gap': eval_data['positive_noise_gap'],
                        'avg_negative': eval_data['avg_negative'],
                        'avg_noise': eval_data['avg_noise'],
                    })

    for mode in ['standard', 'noise', 'random_text']:
        if summary[mode]:
            avg_pos_neg = np.mean([s['pos_neg_gap'] for s in summary[mode]])
            avg_pos_noise = np.mean([s['pos_noise_gap'] for s in summary[mode]])
            avg_neg = np.mean([s['avg_negative'] for s in summary[mode]])
            avg_noise = np.mean([s['avg_noise'] for s in summary[mode]])
            print(f"\n{mode.upper()} mode ({len(summary[mode])} lenses):")
            print(f"  Avg pos-neg gap:   {avg_pos_neg:.3f}")
            print(f"  Avg pos-noise gap: {avg_pos_noise:.3f}")
            print(f"  Avg negative score: {avg_neg:.3f}")
            print(f"  Avg noise score:    {avg_noise:.3f}")

    results['summary'] = {
        mode: {
            'n_lenses': len(summary[mode]),
            'avg_pos_neg_gap': float(np.mean([s['pos_neg_gap'] for s in summary[mode]])) if summary[mode] else None,
            'avg_pos_noise_gap': float(np.mean([s['pos_noise_gap'] for s in summary[mode]])) if summary[mode] else None,
            'avg_negative': float(np.mean([s['avg_negative'] for s in summary[mode]])) if summary[mode] else None,
            'avg_noise': float(np.mean([s['avg_noise'] for s in summary[mode]])) if summary[mode] else None,
        }
        for mode in ['standard', 'noise', 'random_text']
    }

    # Save results
    output_file = output_dir / 'noise_baseline_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
