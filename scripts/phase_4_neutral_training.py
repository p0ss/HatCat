"""
Phase 4: Neutral Training & Comprehensive Testing
=================================================

Goal: Add neutral samples to training and comprehensive evaluation with TP/TN/FP/FN metrics

Key Changes from Phase 2:
- Training: 1 pos + 1 neg + 1 neutral (was 1 pos + 1 neg)
- Evaluation: Positive + negative + neutral testing (was positive only)
- Metrics: F1, precision, recall, TP/TN/FP/FN rates (was accuracy only)

Configuration:
- Concepts: 10 (WordNet top 10)
- Training: 1×1×1 minimal (1 positive, 1 negative, 1 neutral per concept)
- Evaluation: 5+5+5 during training, 20+20+20 final
- Neutral pool: 1000 reserved concepts (distance ≥15 from ALL training concepts)

Expected outcome:
- Lower confidence than Phase 3a (97.8% → ~85-90%)
- Measurable false positive rate
- Measurable true negative rate
- More realistic performance metrics
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
import random

sys.path.append(str(Path(__file__).parent.parent))

from scripts.stage_1_5_temporal import get_activation_sequence


def generate_definition_prompt(concept: str) -> str:
    """Generate a definitional prompt for a concept."""
    templates = [
        f"What is {concept}?",
        f"Define {concept}.",
        f"{concept.capitalize()} is",
    ]
    return random.choice(templates)


def generate_neutral_prompt(neutral_concept: str) -> str:
    """Generate a prompt for a neutral concept."""
    templates = [
        f"What is {neutral_concept}?",
        f"Define {neutral_concept}.",
        f"Tell me about {neutral_concept}.",
    ]
    return random.choice(templates)


def get_mean_activation(model, tokenizer, prompt: str, device: str = "cuda", layer_idx: int = -1) -> np.ndarray:
    """Get mean activation for a prompt."""
    sequence, tokens = get_activation_sequence(
        model, tokenizer, prompt, device=device, layer_idx=layer_idx
    )
    return sequence.mean(axis=0)


def sample_training_data(
    concept: str,
    concept_info: Dict,
    neutral_pool: List[str],
    n_pos: int = 1,
    n_neg: int = 1,
    n_neutral: int = 1
) -> Tuple[List[str], List[int]]:
    """
    Sample training prompts with labels.

    Returns:
        prompts: List of prompts
        labels: List of labels (1=positive, 0=negative/neutral)
    """
    prompts = []
    labels = []

    # Positive samples (definitional prompts)
    for _ in range(n_pos):
        prompts.append(generate_definition_prompt(concept))
        labels.append(1)

    # Negative samples (distant concepts)
    negatives = concept_info.get('negatives', [])
    for _ in range(n_neg):
        neg_concept = random.choice(negatives)
        prompts.append(generate_definition_prompt(neg_concept))
        labels.append(0)

    # Neutral samples (from reserved pool, never seen in training/relationships)
    for _ in range(n_neutral):
        neutral_concept = random.choice(neutral_pool)
        prompts.append(generate_neutral_prompt(neutral_concept))
        labels.append(0)

    return prompts, labels


def sample_test_data(
    concept: str,
    concept_info: Dict,
    neutral_pool: List[str],
    n_pos: int = 20,
    n_neg: int = 20,
    n_neutral: int = 20
) -> Tuple[List[str], List[str]]:
    """
    Sample comprehensive test prompts with ground truth labels.

    Returns:
        prompts: List of prompts
        labels: List of label strings ('positive', 'negative', 'neutral')
    """
    prompts = []
    labels = []

    # Positive samples
    for _ in range(n_pos):
        prompts.append(generate_definition_prompt(concept))
        labels.append('positive')

    # Negative samples (distant concepts)
    negatives = concept_info.get('negatives', [])
    sampled_negs = random.sample(negatives, min(n_neg, len(negatives)))
    for neg_concept in sampled_negs:
        prompts.append(generate_definition_prompt(neg_concept))
        labels.append('negative')

    # Neutral samples
    sampled_neutrals = random.sample(neutral_pool, min(n_neutral, len(neutral_pool)))
    for neutral_concept in sampled_neutrals:
        prompts.append(generate_neutral_prompt(neutral_concept))
        labels.append('neutral')

    return prompts, labels


def train_binary_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_dim: int,
    intermediate_dim: int = 128,
    lr: float = 0.001,
    epochs: int = 100,
    device: str = "cuda"
) -> torch.nn.Module:
    """
    Train binary classifier with early stopping.

    Architecture: Linear(hidden_dim → intermediate_dim) → ReLU → Linear(intermediate_dim → 1)
    """
    class BinaryClassifier(torch.nn.Module):
        def __init__(self, input_dim, intermediate_dim):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, intermediate_dim)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(intermediate_dim, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return torch.sigmoid(x)

    classifier = BinaryClassifier(hidden_dim, intermediate_dim).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    classifier.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = classifier(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    classifier.eval()
    return classifier


def evaluate_comprehensive(
    classifier: torch.nn.Module,
    X_test: np.ndarray,
    y_test_labels: List[str],
    device: str = "cuda",
    threshold: float = 0.5
) -> Dict:
    """
    Comprehensive evaluation with TP/TN/FP/FN metrics.

    Args:
        y_test_labels: List of 'positive', 'negative', or 'neutral'

    Returns:
        Dictionary with:
        - tp, tn, fp, fn counts
        - precision, recall, f1
        - confidence distributions per label type
    """
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.inference_mode():
        confidences = classifier(X_tensor).cpu().numpy().flatten()

    # Convert labels to ground truth (positive=1, negative/neutral=0)
    y_true = np.array([1 if label == 'positive' else 0 for label in y_test_labels])
    y_pred = (confidences >= threshold).astype(int)

    # Confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Confidence distributions by label type
    conf_by_label = {}
    for label_type in ['positive', 'negative', 'neutral']:
        mask = np.array([label == label_type for label in y_test_labels])
        if mask.any():
            conf_by_label[label_type] = {
                'mean': float(confidences[mask].mean()),
                'std': float(confidences[mask].std()),
                'min': float(confidences[mask].min()),
                'max': float(confidences[mask].max()),
                'median': float(np.median(confidences[mask])),
                'n_samples': int(mask.sum())
            }

    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confidence_by_label': conf_by_label
    }


def run_phase_4(
    model_name: str = "google/gemma-3-4b-pt",
    concept_graph_path: str = "data/concept_graph/wordnet_v2_top10.json",
    output_dir: str = "results/phase_4_neutral_training",
    device: str = "cuda",
    layer_idx: int = -1,
    n_concepts: int = 10
):
    """Run Phase 4: Neutral Training & Comprehensive Testing."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PHASE 4: NEUTRAL TRAINING & COMPREHENSIVE TESTING")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Concepts: {n_concepts}")
    print(f"Training: 1×1×1 (1 pos + 1 neg + 1 neutral)")
    print(f"Evaluation: 20+20+20 (positive + negative + neutral)")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Load model
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map=device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Model loaded\n")

    # Get hidden dim
    with torch.inference_mode():
        test_input = tokenizer("test", return_tensors="pt").to(device)
        test_output = model(**test_input, output_hidden_states=True)
        hidden_dim = test_output.hidden_states[layer_idx].shape[-1]
    print(f"Hidden dim: {hidden_dim}\n")

    # Load concept graph (NEW FORMAT with neutral_pool)
    print(f"Loading concept graph from {concept_graph_path}...")
    with open(concept_graph_path) as f:
        data = json.load(f)

    concept_data = data['concepts']  # NEW: concepts are under 'concepts' key
    neutral_pool = data['neutral_pool']  # NEW: separate neutral pool

    all_concepts = list(concept_data.keys())
    concepts = all_concepts[:n_concepts]

    print(f"✓ Loaded {len(concepts)} concepts")
    print(f"✓ Neutral pool: {len(neutral_pool)} concepts\n")

    # Train and evaluate each concept
    results = []

    for i, concept in enumerate(concepts):
        print(f"[{i+1}/{len(concepts)}] {concept}...", end=" ", flush=True)

        concept_start = time.time()
        concept_info = concept_data[concept]

        # Sample training data (1×1×1)
        train_prompts, train_labels = sample_training_data(
            concept, concept_info, neutral_pool,
            n_pos=1, n_neg=1, n_neutral=1
        )

        # Get training activations
        X_train = []
        for prompt in train_prompts:
            act = get_mean_activation(model, tokenizer, prompt, device, layer_idx)
            X_train.append(act)
        X_train = np.array(X_train)
        y_train = np.array(train_labels)

        # Train classifier
        classifier = train_binary_classifier(
            X_train, y_train, hidden_dim,
            intermediate_dim=128, lr=0.001, epochs=100, device=device
        )

        # Sample test data (20+20+20)
        test_prompts, test_labels = sample_test_data(
            concept, concept_info, neutral_pool,
            n_pos=20, n_neg=20, n_neutral=20
        )

        # Get test activations
        X_test = []
        for prompt in test_prompts:
            act = get_mean_activation(model, tokenizer, prompt, device, layer_idx)
            X_test.append(act)
        X_test = np.array(X_test)

        # Comprehensive evaluation
        metrics = evaluate_comprehensive(classifier, X_test, test_labels, device)

        elapsed = time.time() - concept_start

        print(f"F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} ({elapsed:.1f}s)")

        results.append({
            'concept': concept,
            'training_samples': {'positive': 1, 'negative': 1, 'neutral': 1},
            'test_samples': {'positive': 20, 'negative': 20, 'neutral': 20},
            'metrics': metrics,
            'time_seconds': elapsed
        })

    total_time = time.time() - start_time

    # Aggregate metrics
    aggregate = {
        'mean_f1': np.mean([r['metrics']['f1'] for r in results]),
        'mean_precision': np.mean([r['metrics']['precision'] for r in results]),
        'mean_recall': np.mean([r['metrics']['recall'] for r in results]),
        'mean_tp': np.mean([r['metrics']['tp'] for r in results]),
        'mean_tn': np.mean([r['metrics']['tn'] for r in results]),
        'mean_fp': np.mean([r['metrics']['fp'] for r in results]),
        'mean_fn': np.mean([r['metrics']['fn'] for r in results]),
    }

    # Aggregate confidence distributions
    for label_type in ['positive', 'negative', 'neutral']:
        confs = []
        for r in results:
            if label_type in r['metrics']['confidence_by_label']:
                confs.append(r['metrics']['confidence_by_label'][label_type]['mean'])
        if confs:
            aggregate[f'{label_type}_confidence_mean'] = float(np.mean(confs))
            aggregate[f'{label_type}_confidence_std'] = float(np.std(confs))

    print(f"\n{'='*70}")
    print(f"PHASE 4 COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nAggregate Metrics:")
    print(f"  F1: {aggregate['mean_f1']:.3f}")
    print(f"  Precision: {aggregate['mean_precision']:.3f}")
    print(f"  Recall: {aggregate['mean_recall']:.3f}")
    print(f"  TP: {aggregate['mean_tp']:.1f}, TN: {aggregate['mean_tn']:.1f}")
    print(f"  FP: {aggregate['mean_fp']:.1f}, FN: {aggregate['mean_fn']:.1f}")
    print(f"\nConfidence by Label Type:")
    print(f"  Positive: {aggregate.get('positive_confidence_mean', 0):.1%} ± {aggregate.get('positive_confidence_std', 0):.1%}")
    print(f"  Negative: {aggregate.get('negative_confidence_mean', 0):.1%} ± {aggregate.get('negative_confidence_std', 0):.1%}")
    print(f"  Neutral: {aggregate.get('neutral_confidence_mean', 0):.1%} ± {aggregate.get('neutral_confidence_std', 0):.1%}")

    # Save results
    output_file = output_dir / "phase4_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'model': model_name,
                'n_concepts': n_concepts,
                'training': '1×1×1 (pos+neg+neutral)',
                'testing': '20+20+20 (pos+neg+neutral)',
                'device': device,
                'layer_idx': layer_idx,
                'hidden_dim': hidden_dim
            },
            'aggregate': aggregate,
            'per_concept': results,
            'time_seconds': total_time
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='google/gemma-3-4b-pt')
    parser.add_argument('--concept-graph', default='data/concept_graph/wordnet_v2_top10.json')
    parser.add_argument('--output-dir', default='results/phase_4_neutral_training')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--layer-idx', type=int, default=-1)
    parser.add_argument('--n-concepts', type=int, default=10)
    args = parser.parse_args()

    run_phase_4(
        model_name=args.model,
        concept_graph_path=args.concept_graph,
        output_dir=args.output_dir,
        device=args.device,
        layer_idx=args.layer_idx,
        n_concepts=args.n_concepts
    )
