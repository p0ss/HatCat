"""
Phase 3: Relationship-First Training Comparison

Compare two approaches:
A) Standard 1×1: 1 definition + 1 relationship per concept
B) Relationship-First: 1 definition + N relationships (adaptive by connectivity)

Key Innovation in Approach B:
- Generate relationship activations once, reuse for multiple concepts
- High-connectivity concepts get more training data automatically
- Semantically-principled negatives: antonyms first, then graph-distant fallback
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForCausalLM
import h5py
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
import random
from collections import defaultdict


def load_concept_graph(path: Path) -> Dict:
    """Load WordNet concept graph."""
    with open(path) as f:
        return json.load(f)


def extract_temporal_activations(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int = -1,
    max_seq_len: int = 20
) -> Tuple[np.ndarray, int]:
    """
    Extract temporal activation sequence for a prompt.

    Returns:
        activations: [timesteps, hidden_dim]
        length: Actual sequence length
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_seq_len,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        # Extract activation sequence from generated tokens
        activation_sequence = []
        for step_states in outputs.hidden_states:
            last_layer = step_states[layer_idx]
            act = last_layer[0, -1, :].float().cpu().numpy()
            activation_sequence.append(act)

        activations = np.array(activation_sequence)  # [timesteps, hidden]
        length = len(activations)

        # Pad if needed
        if length < max_seq_len:
            padding = np.zeros((max_seq_len - length, activations.shape[1]))
            activations = np.concatenate([activations, padding], axis=0)
        else:
            activations = activations[:max_seq_len]

    return activations.astype(np.float32), length


def generate_standard_1x1_data(
    model,
    tokenizer,
    concept_graph: Dict,
    layer_idx: int = -1
) -> Dict:
    """
    Approach A: Standard 1×1 training.

    For each concept:
    - 1 definition positive
    - 1 relationship positive
    - 2 negatives (balanced)

    Returns:
        concept_data: {concept: {positives: [...], negatives: [...], lengths_pos: [...], lengths_neg: [...]}}
    """
    print("\n=== Generating Standard 1×1 Training Data ===")
    concept_data = {}

    for concept_name, concept_info in tqdm(concept_graph.items(), desc="Standard 1×1"):
        positives = []
        positives_lens = []
        negatives = []
        negatives_lens = []

        # 1. Definition positive
        def_prompt = f"What is {concept_name}?"
        def_act, def_len = extract_temporal_activations(model, tokenizer, def_prompt, layer_idx)
        positives.append(def_act)
        positives_lens.append(def_len)

        # 2. One relationship positive (first related concept)
        if concept_info['related']:
            related = concept_info['related'][0]
            rel_prompt = f"How does {concept_name} relate to {related}?"
            rel_act, rel_len = extract_temporal_activations(model, tokenizer, rel_prompt, layer_idx)
            positives.append(rel_act)
            positives_lens.append(rel_len)

        # 3. Negatives: Antonym-first strategy
        antonyms = concept_info.get('related_structured', {}).get('antonyms', [])

        if antonyms:
            # Use first antonym
            antonym = antonyms[0]

            # Antonym definition
            ant_def_prompt = f"What is {antonym}?"
            ant_def_act, ant_def_len = extract_temporal_activations(model, tokenizer, ant_def_prompt, layer_idx)
            negatives.append(ant_def_act)
            negatives_lens.append(ant_def_len)

            # Antonym relationship (if it has relationships)
            # For now, use graph-distant as second negative
            if concept_info['negatives']:
                distant = random.choice(concept_info['negatives'])
                dist_prompt = f"What is {distant}?"
                dist_act, dist_len = extract_temporal_activations(model, tokenizer, dist_prompt, layer_idx)
                negatives.append(dist_act)
                negatives_lens.append(dist_len)
        else:
            # Fallback: 2 graph-distant negatives
            for _ in range(2):
                if concept_info['negatives']:
                    distant = random.choice(concept_info['negatives'])
                    dist_prompt = f"What is {distant}?"
                    dist_act, dist_len = extract_temporal_activations(model, tokenizer, dist_prompt, layer_idx)
                    negatives.append(dist_act)
                    negatives_lens.append(dist_len)

        concept_data[concept_name] = {
            'positives': np.array(positives),
            'negatives': np.array(negatives),
            'lengths_pos': np.array(positives_lens),
            'lengths_neg': np.array(negatives_lens)
        }

    return concept_data


def generate_relationship_first_data(
    model,
    tokenizer,
    concept_graph: Dict,
    layer_idx: int = -1
) -> Tuple[Dict, Dict]:
    """
    Approach B: Relationship-First training.

    Phase 1: Generate all relationship activations once
    Phase 2: Aggregate for concept training (1 definition + N relationships)

    Returns:
        relationship_activations: {(concept1, concept2): (activation, length)}
        concept_data: {concept: {positives: [...], negatives: [...], lengths_pos: [...], lengths_neg: [...]}}
    """
    print("\n=== Generating Relationship-First Training Data ===")

    # Phase 1: Generate relationship activations
    print("\nPhase 1: Generating relationship activations...")
    relationship_activations = {}
    all_edges = set()

    # Collect all unique edges
    for concept_name, concept_info in concept_graph.items():
        for related in concept_info['related']:
            # FIXED: Reverse edge so prompt is "How does SPECIFIC relate to GENERAL?"
            # e.g., "How does Rickettsiales relate to animal order?" (makes semantic sense)
            # instead of "How does animal order relate to Rickettsiales?" (backwards/weird)
            edge = (related, concept_name)
            all_edges.add(edge)

    # Generate activations for each edge
    for source, target in tqdm(all_edges, desc="Relationships"):
        prompt = f"How does {source} relate to {target}?"
        act, length = extract_temporal_activations(model, tokenizer, prompt, layer_idx)
        relationship_activations[(source, target)] = (act, length)

    # Phase 2: Aggregate for concepts
    print("\nPhase 2: Aggregating to concepts...")
    concept_data = {}

    for concept_name, concept_info in tqdm(concept_graph.items(), desc="Concepts"):
        positives = []
        positives_lens = []
        negatives = []
        negatives_lens = []

        # 1. Definition positive
        def_prompt = f"What is {concept_name}?"
        def_act, def_len = extract_temporal_activations(model, tokenizer, def_prompt, layer_idx)
        positives.append(def_act)
        positives_lens.append(def_len)

        # 2. All relationship activations where this concept appears
        for related in concept_info['related']:
            # FIXED: Match reversed edge from above
            edge = (related, concept_name)
            if edge in relationship_activations:
                rel_act, rel_len = relationship_activations[edge]
                positives.append(rel_act)
                positives_lens.append(rel_len)

        n_positives = len(positives)

        # 3. Negatives: Antonym-first strategy (balanced with positives)
        antonyms = concept_info.get('related_structured', {}).get('antonyms', [])

        if antonyms:
            # Use antonym and its relationships
            antonym = antonyms[0]

            # Antonym definition
            ant_def_prompt = f"What is {antonym}?"
            ant_def_act, ant_def_len = extract_temporal_activations(model, tokenizer, ant_def_prompt, layer_idx)
            negatives.append(ant_def_act)
            negatives_lens.append(ant_def_len)

            # Antonym relationships (up to n_positives - 1)
            # Look up antonym in graph to get its relationships
            if antonym in concept_graph:
                antonym_rels = concept_graph[antonym]['related']
                for ant_related in antonym_rels[:n_positives - 1]:
                    # FIXED: Match reversed edge
                    edge = (ant_related, antonym)
                    if edge in relationship_activations:
                        ant_rel_act, ant_rel_len = relationship_activations[edge]
                        negatives.append(ant_rel_act)
                        negatives_lens.append(ant_rel_len)

            # Fill remaining with graph-distant if needed
            while len(negatives) < n_positives:
                if concept_info['negatives']:
                    distant = random.choice(concept_info['negatives'])
                    dist_prompt = f"What is {distant}?"
                    dist_act, dist_len = extract_temporal_activations(model, tokenizer, dist_prompt, layer_idx)
                    negatives.append(dist_act)
                    negatives_lens.append(dist_len)
                else:
                    break
        else:
            # Fallback: n_positives graph-distant negatives
            for _ in range(n_positives):
                if concept_info['negatives']:
                    distant = random.choice(concept_info['negatives'])
                    dist_prompt = f"What is {distant}?"
                    dist_act, dist_len = extract_temporal_activations(model, tokenizer, dist_prompt, layer_idx)
                    negatives.append(dist_act)
                    negatives_lens.append(dist_len)

        concept_data[concept_name] = {
            'positives': np.array(positives),
            'negatives': np.array(negatives),
            'lengths_pos': np.array(positives_lens),
            'lengths_neg': np.array(negatives_lens),
            'n_relationships': len(concept_info['related'])
        }

    return relationship_activations, concept_data


class BinaryConceptClassifier(nn.Module):
    """Binary classifier for single concept detection."""

    def __init__(self, hidden_dim: int, lstm_dim: int = 256):
        super().__init__()

        self.lstm = nn.LSTM(
            hidden_dim,
            lstm_dim,
            batch_first=True,
            bidirectional=True
        )

        # Output logits (no sigmoid - we'll use BCEWithLogitsLoss)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x, lengths):
        # Pack for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        return self.classifier(hidden).squeeze()


def train_classifier(
    positives: np.ndarray,
    negatives: np.ndarray,
    lengths_pos: np.ndarray,
    lengths_neg: np.ndarray,
    test_positives: np.ndarray,
    test_negatives: np.ndarray,
    test_lengths_pos: np.ndarray,
    test_lengths_neg: np.ndarray,
    hidden_dim: int,
    epochs: int = 20,
    lr: float = 1e-3
) -> Tuple[BinaryConceptClassifier, float]:
    """
    Train a single binary concept classifier with separate OOD test set.

    Returns:
        model: Trained classifier
        test_acc: Test accuracy on OOD test set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare training data (all positives and negatives become training data)
    X_train = np.concatenate([positives, negatives], axis=0)
    lengths_train = np.concatenate([lengths_pos, lengths_neg], axis=0)
    y_train = np.concatenate([
        np.ones(len(positives)),
        np.zeros(len(negatives))
    ])

    # Shuffle training data
    indices = np.random.permutation(len(y_train))
    X_train = X_train[indices]
    lengths_train = lengths_train[indices]
    y_train = y_train[indices]

    # Prepare separate OOD test data
    X_test = np.concatenate([test_positives, test_negatives], axis=0)
    lengths_test = np.concatenate([test_lengths_pos, test_lengths_neg], axis=0)
    y_test = np.concatenate([
        np.ones(len(test_positives)),
        np.zeros(len(test_negatives))
    ])

    # Convert to tensors
    X_train = torch.from_numpy(X_train).to(device)
    X_test = torch.from_numpy(X_test).to(device)
    lengths_train = torch.from_numpy(lengths_train).to(device)
    lengths_test = torch.from_numpy(lengths_test).to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    # Create model
    model = BinaryConceptClassifier(hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid + BCE for numerical stability

    # Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train, lengths_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Test
    model.eval()
    with torch.no_grad():
        logits = model(X_test, lengths_test)
        probs = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
        predictions = (probs > 0.5).float()
        test_acc = (predictions == y_test).float().mean().item()

    return model, test_acc


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Relationship-First Comparison")
    parser.add_argument("--concept-graph", type=Path, required=True,
                       help="Path to concept graph JSON")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-pt",
                       help="Model to use for activation extraction")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for results")
    parser.add_argument("--layer-idx", type=int, default=-1,
                       help="Layer index to extract activations from")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Training epochs per classifier")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # Get hidden dimension from embedding layer
    hidden_dim = model.get_input_embeddings().embedding_dim

    # Load concept graph
    concept_graph = load_concept_graph(args.concept_graph)
    print(f"Loaded {len(concept_graph)} concepts")

    # ===== Approach A: Standard 1×1 =====
    print("\n" + "="*60)
    print("APPROACH A: Standard 1×1 Training")
    print("="*60)

    data_1x1 = generate_standard_1x1_data(
        model, tokenizer, concept_graph, args.layer_idx
    )

    print("\nTraining classifiers (1×1)...")
    results_1x1 = {}
    for concept_name, data in tqdm(data_1x1.items(), desc="Training 1×1"):
        test_data = ood_test_sets[concept_name]
        classifier, test_acc = train_classifier(
            data['positives'],
            data['negatives'],
            data['lengths_pos'],
            data['lengths_neg'],
            test_data['test_positives'],
            test_data['test_negatives'],
            test_data['test_lengths_pos'],
            test_data['test_lengths_neg'],
            hidden_dim,
            epochs=args.epochs,
            lr=args.lr
        )
        results_1x1[concept_name] = {
            'test_accuracy': test_acc,
            'n_train_samples': len(data['positives']) + len(data['negatives'])
        }

    # ===== Approach B: Relationship-First =====
    print("\n" + "="*60)
    print("APPROACH B: Relationship-First Training")
    print("="*60)

    relationship_acts, data_relfirst = generate_relationship_first_data(
        model, tokenizer, concept_graph, args.layer_idx
    )

    print(f"\nGenerated {len(relationship_acts)} unique relationship activations")
    print("These will be reused for all concepts that participate in them.")

    print("\nTraining classifiers (Relationship-First)...")
    results_relfirst = {}
    for concept_name, data in tqdm(data_relfirst.items(), desc="Training RelFirst"):
        test_data = ood_test_sets[concept_name]
        classifier, test_acc = train_classifier(
            data['positives'],
            data['negatives'],
            data['lengths_pos'],
            data['lengths_neg'],
            test_data['test_positives'],
            test_data['test_negatives'],
            test_data['test_lengths_pos'],
            test_data['test_lengths_neg'],
            hidden_dim,
            epochs=args.epochs,
            lr=args.lr
        )
        results_relfirst[concept_name] = {
            'test_accuracy': test_acc,
            'n_train_samples': len(data['positives']) + len(data['negatives']),
            'n_relationships': data['n_relationships']
        }

    # ===== Compare Results =====
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    # Calculate summary statistics
    acc_1x1 = [r['test_accuracy'] for r in results_1x1.values()]
    acc_relfirst = [r['test_accuracy'] for r in results_relfirst.values()]

    perfect_1x1 = sum(1 for a in acc_1x1 if a == 1.0)
    perfect_relfirst = sum(1 for a in acc_relfirst if a == 1.0)

    print(f"\nApproach A (Standard 1×1):")
    print(f"  Perfect accuracy: {perfect_1x1}/{len(acc_1x1)} ({100*perfect_1x1/len(acc_1x1):.1f}%)")
    print(f"  Mean accuracy: {np.mean(acc_1x1):.4f}")
    print(f"  Std accuracy: {np.std(acc_1x1):.4f}")

    print(f"\nApproach B (Relationship-First):")
    print(f"  Perfect accuracy: {perfect_relfirst}/{len(acc_relfirst)} ({100*perfect_relfirst/len(acc_relfirst):.1f}%)")
    print(f"  Mean accuracy: {np.mean(acc_relfirst):.4f}")
    print(f"  Std accuracy: {np.std(acc_relfirst):.4f}")

    # Compute efficiency metrics
    total_generations_1x1 = len(concept_graph) * 4  # 1 def + 1 rel + 2 negs per concept
    total_generations_relfirst = (
        len(relationship_acts) * 2 +  # Each relationship: 1 pos + 1 neg (reused)
        len(concept_graph) * 2  # Each concept: 1 def + ~1 avg negative def
    )

    print(f"\nCompute Efficiency:")
    print(f"  1×1 total generations: {total_generations_1x1}")
    print(f"  RelFirst total generations: {total_generations_relfirst}")
    print(f"  Efficiency gain: {total_generations_1x1 / total_generations_relfirst:.2f}x")

    # Per-concept comparison
    print(f"\nPer-Concept Detailed Comparison:")
    print(f"{'Concept':<25} {'1×1 Acc':<10} {'RelFirst Acc':<15} {'N Relationships':<20}")
    print("-" * 70)
    for concept_name in concept_graph.keys():
        acc_a = results_1x1[concept_name]['test_accuracy']
        acc_b = results_relfirst[concept_name]['test_accuracy']
        n_rels = results_relfirst[concept_name]['n_relationships']
        print(f"{concept_name:<25} {acc_a:<10.4f} {acc_b:<15.4f} {n_rels:<20}")

    # Save results
    output = {
        'approach_a_1x1': results_1x1,
        'approach_b_relationship_first': results_relfirst,
        'summary': {
            '1x1': {
                'perfect_accuracy_count': perfect_1x1,
                'perfect_accuracy_pct': 100 * perfect_1x1 / len(acc_1x1),
                'mean_accuracy': float(np.mean(acc_1x1)),
                'std_accuracy': float(np.std(acc_1x1))
            },
            'relationship_first': {
                'perfect_accuracy_count': perfect_relfirst,
                'perfect_accuracy_pct': 100 * perfect_relfirst / len(acc_relfirst),
                'mean_accuracy': float(np.mean(acc_relfirst)),
                'std_accuracy': float(np.std(acc_relfirst))
            },
            'compute_efficiency': {
                'total_generations_1x1': total_generations_1x1,
                'total_generations_relfirst': total_generations_relfirst,
                'efficiency_gain': total_generations_1x1 / total_generations_relfirst
            }
        }
    }

    output_path = args.output_dir / "phase3_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
