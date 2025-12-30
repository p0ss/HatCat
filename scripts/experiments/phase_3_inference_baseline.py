"""
Phase 3: Inference Baseline Evaluation
========================================

Goal: Establish runtime performance baselines and detection quality metrics
before making training changes (Phases 4-6).

Metrics:
- Runtime Performance:
  - Latency per concept (single vs batched)
  - Memory scaling (10, 100, 1000 concepts loaded)
  - Throughput (tokens/sec with detection enabled)
- Detection Quality:
  - Confidence score distributions
  - Detection timing (when in generation does concept activate?)
  - Baseline false positive rate

Usage:
    python scripts/phase_3_inference_baseline.py \
        --model google/gemma-3-4b-pt \
        --classifier-dir results/phase_2_scale/scale_n100 \
        --concept-graph data/concept_graph/wordnet_v2_top100.json \
        --output-dir results/phase_3_inference_baseline \
        --n-concepts 100
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import argparse
import json
import time
import psutil
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.stage_1_5_temporal import get_activation_sequence
from torch import nn


class ConceptDetector:
    """Real-time concept detection during generation."""

    def __init__(
        self,
        classifiers: Dict[str, nn.Module],
        concepts: List[str],
        device: str = "cuda"
    ):
        self.classifiers = classifiers
        self.concepts = concepts
        self.device = device

    def detect_batch(
        self,
        activations: torch.Tensor  # (batch_size, hidden_dim) or (batch_size, seq_len, hidden_dim)
    ) -> Dict[str, np.ndarray]:
        """
        Run all classifiers on activation batch.

        Returns:
            Dict mapping concept -> confidence scores (array of shape (batch_size,))
        """
        # Pool temporal dimension if needed
        if len(activations.shape) == 3:
            activations = activations.mean(dim=1)  # (batch_size, hidden_dim)

        results = {}

        with torch.no_grad():
            for concept in self.concepts:
                classifier = self.classifiers[concept]
                scores = classifier(activations).squeeze(-1).cpu().numpy()
                results[concept] = scores

        return results


def load_trained_classifiers(
    checkpoint_path: Path,
    concepts: List[str],
    hidden_dim: int,
    device: str = "cuda"
) -> Dict[str, nn.Module]:
    """Load trained classifiers from Phase 2 checkpoint."""

    with open(checkpoint_path) as f:
        data = json.load(f)

    classifiers = {}

    for result in data['results']:
        concept = result['concept']

        if concept not in concepts:
            continue

        # Reconstruct classifier architecture (must match phase_2_scale_test.py)
        classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)

        # Load weights if saved (Phase 2 doesn't save weights currently, so we'll skip this)
        # This means we'll need to retrain for now, or update Phase 2 to save weights

        classifiers[concept] = classifier

    return classifiers


def measure_latency(
    detector: ConceptDetector,
    activation: torch.Tensor,
    n_runs: int = 100
) -> Dict[str, float]:
    """Measure inference latency."""

    # Warmup
    for _ in range(10):
        detector.detect_batch(activation)

    # Measure
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        detector.detect_batch(activation)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'p50_ms': np.percentile(times, 50) * 1000,
        'p95_ms': np.percentile(times, 95) * 1000,
        'p99_ms': np.percentile(times, 99) * 1000,
    }


def measure_memory(detector: ConceptDetector) -> Dict[str, float]:
    """Measure memory usage."""

    process = psutil.Process()

    # Get GPU memory if available
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    else:
        gpu_mem_allocated = 0
        gpu_mem_reserved = 0

    # Get CPU memory
    cpu_mem_rss = process.memory_info().rss / 1024**3  # GB

    return {
        'gpu_allocated_gb': gpu_mem_allocated,
        'gpu_reserved_gb': gpu_mem_reserved,
        'cpu_rss_gb': cpu_mem_rss,
        'n_classifiers': len(detector.classifiers)
    }


def measure_confidence_distribution(
    detector: ConceptDetector,
    model,
    tokenizer,
    concept_data: Dict,
    n_prompts: int = 20,
    device: str = "cuda",
    layer_idx: int = -1
) -> Dict[str, Dict]:
    """Measure confidence score distributions on OOD prompts."""

    results = {}

    for concept in detector.concepts:
        # Generate OOD prompts for this concept
        test_prompts = [
            f"Tell me about {concept}.",
            f"What is {concept}?",
            f"Explain {concept} to me.",
            f"Describe {concept}.",
            f"How would you define {concept}?",
        ]

        # Get related terms for additional prompts
        related_structured = concept_data[concept].get('related_structured', {})
        for rel_type in ['hypernyms', 'hyponyms', 'holonyms', 'meronyms']:
            if rel_type in related_structured:
                for term in related_structured[rel_type][:3]:
                    test_prompts.append(f"What is the relationship between {concept} and {term}?")

        # Limit to n_prompts
        test_prompts = test_prompts[:n_prompts]

        # Get activations and confidences
        confidences = []

        for prompt in test_prompts:
            seq, _ = get_activation_sequence(model, tokenizer, prompt, layer_idx, device)

            # Pool temporal sequence
            pooled = torch.FloatTensor(seq.mean(axis=0)).unsqueeze(0).to(device)

            # Detect
            scores = detector.detect_batch(pooled)
            confidences.append(scores[concept][0])

        results[concept] = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'median': float(np.median(confidences)),
            'p25': float(np.percentile(confidences, 25)),
            'p75': float(np.percentile(confidences, 75)),
            'n_prompts': len(confidences),
            'confidences': [float(c) for c in confidences]
        }

    return results


def measure_detection_timing(
    detector: ConceptDetector,
    model,
    tokenizer,
    concept: str,
    prompt: str,
    device: str = "cuda",
    layer_idx: int = -1,
    max_new_tokens: int = 50
) -> Dict:
    """Measure when concept activates during generation."""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate token by token, detecting at each step
    detection_scores = []
    generated_tokens = []

    with torch.no_grad():
        past_key_values = None
        current_input_ids = inputs['input_ids']

        for i in range(max_new_tokens):
            # Forward pass
            outputs = model(
                input_ids=current_input_ids if past_key_values is None else current_input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values

            # Get last hidden state
            hidden_state = outputs.hidden_states[layer_idx][:, -1, :]  # (1, hidden_dim)

            # Detect
            scores = detector.detect_batch(hidden_state)
            detection_scores.append(float(scores[concept][0]))

            # Sample next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

            generated_tokens.append(tokenizer.decode(next_token[0]))
            current_input_ids = next_token

    generated_text = ''.join(generated_tokens)

    return {
        'prompt': prompt,
        'generated_text': generated_text,
        'detection_scores': detection_scores,
        'mean_score': float(np.mean(detection_scores)),
        'max_score': float(np.max(detection_scores)),
        'max_score_position': int(np.argmax(detection_scores)),
        'n_tokens': len(detection_scores)
    }


def run_baseline_evaluation(
    model,
    tokenizer,
    concept_data: Dict,
    concepts: List[str],
    output_dir: Path,
    device: str = "cuda",
    layer_idx: int = -1
):
    """Run complete baseline evaluation."""

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PHASE 3: INFERENCE BASELINE EVALUATION")
    print(f"{'='*70}")
    print(f"Concepts: {len(concepts)}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # Get hidden dim
    with torch.inference_mode():
        test_input = tokenizer("test", return_tensors="pt").to(device)
        test_output = model(**test_input, output_hidden_states=True)
        hidden_dim = test_output.hidden_states[layer_idx].shape[-1]

    print(f"Hidden dim: {hidden_dim}\n")

    # For now, we'll train mini classifiers on-the-fly since Phase 2 doesn't save weights
    # This is a limitation we'll note in the results

    print("NOTE: Phase 2 didn't save classifier weights, so we'll train minimal")
    print("      classifiers on-the-fly for baseline measurement.\n")

    classifiers = {}

    print("Training mini classifiers...")
    for i, concept in enumerate(concepts):
        print(f"  [{i+1}/{len(concepts)}] {concept}...", end=" ", flush=True)

        # Generate 1×1 training data
        negatives = concept_data[concept].get('negatives', [])

        # Positive
        pos_seq, _ = get_activation_sequence(
            model, tokenizer, f"What is {concept}?", layer_idx, device
        )
        pos_pooled = torch.FloatTensor(pos_seq.mean(axis=0)).unsqueeze(0).to(device)

        # Negative
        neg_seq, _ = get_activation_sequence(
            model, tokenizer, f"What is {negatives[0]}?", layer_idx, device
        )
        neg_pooled = torch.FloatTensor(neg_seq.mean(axis=0)).unsqueeze(0).to(device)

        # Train
        X = torch.cat([pos_pooled, neg_pooled], dim=0)
        y = torch.FloatTensor([1.0, 0.0]).to(device)

        classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

        classifier.train()
        for _ in range(50):  # Quick training
            optimizer.zero_grad()
            pred = classifier(X).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        classifier.eval()
        classifiers[concept] = classifier
        print("✓")

    print("\nClassifiers trained.\n")

    # Create detector
    detector = ConceptDetector(classifiers, concepts, device)

    # 1. Measure latency
    print("="*70)
    print("1. LATENCY MEASUREMENT")
    print("="*70 + "\n")

    test_activation = torch.randn(1, hidden_dim).to(device)
    latency_stats = measure_latency(detector, test_activation, n_runs=100)

    print(f"Latency (n={len(concepts)} concepts):")
    print(f"  Mean:   {latency_stats['mean_ms']:.3f} ms")
    print(f"  Std:    {latency_stats['std_ms']:.3f} ms")
    print(f"  Median: {latency_stats['p50_ms']:.3f} ms")
    print(f"  P95:    {latency_stats['p95_ms']:.3f} ms")
    print(f"  P99:    {latency_stats['p99_ms']:.3f} ms")
    print(f"\nPer-concept latency: {latency_stats['mean_ms'] / len(concepts):.4f} ms\n")

    # 2. Measure memory
    print("="*70)
    print("2. MEMORY MEASUREMENT")
    print("="*70 + "\n")

    memory_stats = measure_memory(detector)

    print(f"Memory usage:")
    print(f"  GPU allocated: {memory_stats['gpu_allocated_gb']:.3f} GB")
    print(f"  GPU reserved:  {memory_stats['gpu_reserved_gb']:.3f} GB")
    print(f"  CPU RSS:       {memory_stats['cpu_rss_gb']:.3f} GB")
    print(f"  Classifiers:   {memory_stats['n_classifiers']}")
    print(f"\nPer-classifier GPU: {memory_stats['gpu_allocated_gb'] / len(concepts) * 1000:.2f} MB\n")

    # 3. Measure confidence distributions
    print("="*70)
    print("3. CONFIDENCE DISTRIBUTION")
    print("="*70 + "\n")

    print("Measuring confidence on OOD prompts...")
    confidence_stats = measure_confidence_distribution(
        detector, model, tokenizer, concept_data,
        n_prompts=20, device=device, layer_idx=layer_idx
    )

    # Aggregate stats
    all_means = [s['mean'] for s in confidence_stats.values()]
    all_stds = [s['std'] for s in confidence_stats.values()]

    print(f"\nAggregate confidence statistics:")
    print(f"  Mean (across concepts): {np.mean(all_means):.3f}")
    print(f"  Std (across concepts):  {np.mean(all_stds):.3f}")
    print(f"  Min mean:  {np.min(all_means):.3f}")
    print(f"  Max mean:  {np.max(all_means):.3f}\n")

    # 4. Measure detection timing (sample a few concepts)
    print("="*70)
    print("4. DETECTION TIMING")
    print("="*70 + "\n")

    timing_results = []
    sample_concepts = concepts[:3]  # Just sample a few

    for concept in sample_concepts:
        print(f"Testing {concept}...")
        timing = measure_detection_timing(
            detector, model, tokenizer, concept,
            prompt=f"Tell me about {concept}.",
            device=device, layer_idx=layer_idx,
            max_new_tokens=30
        )
        timing_results.append({
            'concept': concept,
            **timing
        })
        print(f"  Generated: {timing['generated_text'][:50]}...")
        print(f"  Mean score: {timing['mean_score']:.3f}")
        print(f"  Max score: {timing['max_score']:.3f} @ token {timing['max_score_position']}\n")

    # Save results
    results = {
        'config': {
            'n_concepts': len(concepts),
            'model': model.config._name_or_path,
            'device': device,
            'layer_idx': layer_idx,
            'hidden_dim': hidden_dim
        },
        'latency': latency_stats,
        'memory': memory_stats,
        'confidence_distributions': confidence_stats,
        'detection_timing': timing_results,
        'notes': [
            'Phase 2 did not save classifier weights',
            'Classifiers retrained on-the-fly with 1×1 minimal training',
            'Results may differ from Phase 2 due to retraining'
        ]
    }

    output_file = output_dir / 'baseline_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("="*70)
    print(f"✓ Baseline evaluation complete: {output_file}")
    print("="*70 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Inference baseline evaluation")
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt')
    parser.add_argument('--concept-graph', type=str, required=True,
                       help='Path to concept graph JSON')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--n-concepts', type=int, default=100,
                       help='Number of concepts to evaluate')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--layer-idx', type=int, default=-1,
                       help='Which layer to extract activations from')

    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        device_map=args.device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded\n")

    # Load concept graph
    with open(args.concept_graph) as f:
        concept_data = json.load(f)

    concepts = list(concept_data.keys())[:args.n_concepts]
    print(f"Loaded {len(concepts)} concepts\n")

    # Run evaluation
    results = run_baseline_evaluation(
        model=model,
        tokenizer=tokenizer,
        concept_data=concept_data,
        concepts=concepts,
        output_dir=Path(args.output_dir),
        device=args.device,
        layer_idx=args.layer_idx
    )


if __name__ == '__main__':
    main()
