"""
Benchmark Temporal Detection Inference

Measures the cost of running temporal concept detection at scale:
- Latency per concept classifier
- Memory overhead (RAM/VRAM)
- Scaling behavior (10/100/1000 concepts)
- Throughput (sequences/second)
- Batch efficiency

Usage:
    python scripts/benchmark_temporal_inference.py \
        --model google/gemma-3-4b-pt \
        --train-data results/adaptive_relfirst_100 \
        --n-concepts 10 \
        --n-sequences 100 \
        --device cuda
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import argparse
import json
import time
import psutil
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.stage_1_5_temporal import get_activation_sequence


class TemporalInferenceBenchmark:
    """Benchmark temporal concept detection inference."""

    def __init__(
        self,
        model,
        tokenizer,
        train_data_dir: Path,
        device: str = "cuda",
        layer_idx: int = -1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data_dir = train_data_dir
        self.device = device
        self.layer_idx = layer_idx

        # Load available concepts from training data
        self.available_concepts = self._discover_concepts()

        # Metrics storage
        self.metrics = {
            'loading': {},
            'inference': {},
            'memory': {},
            'throughput': {}
        }

    def _discover_concepts(self) -> List[str]:
        """Discover which concepts have trained classifiers."""
        train_data_path = self.train_data_dir / "train_data"
        if not train_data_path.exists():
            raise ValueError(f"Training data directory not found: {train_data_path}")

        concepts = []
        for data_file in train_data_path.glob("*.npz"):
            # Convert filename back to concept name
            concept = data_file.stem.replace('_', ' ')
            concepts.append(concept)

        print(f"Found {len(concepts)} trained concepts")
        return sorted(concepts)

    def load_classifier(self, concept: str) -> torch.nn.Module:
        """Load a trained classifier for a concept."""
        from torch import nn

        # Load training data to get dimensions
        data_path = self.train_data_dir / "train_data" / f"{concept.replace('/', '_')}.npz"
        data = np.load(data_path)

        # Get hidden dimension from first sample
        hidden_dim = data['pos'][0].shape[-1]

        # Create classifier (same architecture as training)
        classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(self.device)

        return classifier, hidden_dim

    def train_classifiers(self, concepts: List[str]) -> Dict[str, torch.nn.Module]:
        """Train/load classifiers for given concepts."""
        print(f"\nLoading {len(concepts)} classifiers...")
        start_time = time.time()

        classifiers = {}
        for i, concept in enumerate(concepts):
            try:
                # Load training data
                data_path = self.train_data_dir / "train_data" / f"{concept.replace('/', '_')}.npz"
                data = np.load(data_path)

                train_pos = [data['pos'][i] for i in range(len(data['pos']))]
                train_neg = [data['neg'][i] for i in range(len(data['neg']))]

                # Get hidden dim
                hidden_dim = train_pos[0].shape[-1]

                # Pool temporal sequences (mean over time)
                pos_pooled = np.array([seq.mean(axis=0) for seq in train_pos])
                neg_pooled = np.array([seq.mean(axis=0) for seq in train_neg])

                X_train = np.vstack([pos_pooled, neg_pooled])
                y_train = np.array([1] * len(pos_pooled) + [0] * len(neg_pooled))

                # Convert to tensors
                X_train = torch.FloatTensor(X_train).to(self.device)
                y_train = torch.FloatTensor(y_train).to(self.device)

                # Create and train classifier
                from torch import nn
                classifier = nn.Sequential(
                    nn.Linear(hidden_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                ).to(self.device)

                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

                # Quick training (fewer epochs for benchmark)
                from torch.utils.data import TensorDataset, DataLoader
                train_dataset = TensorDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

                classifier.train()
                for epoch in range(5):  # Reduced from 10 for speed
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        pred = classifier(batch_X).squeeze()
                        loss = criterion(pred, batch_y)
                        loss.backward()
                        optimizer.step()

                classifier.eval()
                classifiers[concept] = classifier

                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(concepts)}] loaded")

            except Exception as e:
                print(f"  ✗ Failed to load {concept}: {e}")
                continue

        elapsed = time.time() - start_time
        self.metrics['loading']['time'] = elapsed
        self.metrics['loading']['n_concepts'] = len(classifiers)

        print(f"✓ Loaded {len(classifiers)} classifiers in {elapsed:.1f}s ({elapsed/len(classifiers):.2f}s each)")

        return classifiers

    def generate_test_sequences(self, n_sequences: int) -> List[Tuple[str, np.ndarray]]:
        """Generate random test sequences."""
        print(f"\nGenerating {n_sequences} test sequences...")

        # Use simple prompts to get diverse sequences
        templates = [
            "What is {}?",
            "The concept of {}",
            "Understanding {}",
            "In the context of {}",
            "An example of {} is"
        ]

        # Sample random words/concepts
        test_words = ["dog", "tree", "democracy", "running", "happiness",
                      "science", "computer", "music", "art", "philosophy",
                      "biology", "physics", "chemistry", "mathematics", "history"]

        sequences = []
        for i in range(n_sequences):
            word = test_words[i % len(test_words)]
            template = templates[i % len(templates)]
            prompt = template.format(word)

            seq, _ = get_activation_sequence(
                self.model, self.tokenizer, prompt,
                self.layer_idx, self.device
            )
            sequences.append((prompt, seq))

        print(f"✓ Generated {len(sequences)} sequences")
        return sequences

    def benchmark_sequential_inference(
        self,
        classifiers: Dict[str, torch.nn.Module],
        sequences: List[Tuple[str, np.ndarray]]
    ) -> Dict:
        """Benchmark running all classifiers sequentially on all sequences."""
        print(f"\nBenchmarking sequential inference...")
        print(f"  {len(classifiers)} classifiers × {len(sequences)} sequences = {len(classifiers) * len(sequences)} operations")

        start_time = time.time()
        total_operations = 0

        results = {}

        for concept, classifier in classifiers.items():
            concept_results = []

            for prompt, seq in sequences:
                # Pool sequence (mean over time)
                pooled = seq.mean(axis=0)

                # Convert to tensor
                x = torch.FloatTensor(pooled).unsqueeze(0).to(self.device)

                # Inference
                with torch.no_grad():
                    pred = classifier(x).squeeze().item()

                concept_results.append(pred)
                total_operations += 1

            results[concept] = concept_results

        elapsed = time.time() - start_time

        metrics = {
            'total_time': elapsed,
            'n_classifiers': len(classifiers),
            'n_sequences': len(sequences),
            'total_operations': total_operations,
            'ops_per_second': total_operations / elapsed,
            'time_per_operation': elapsed / total_operations,
            'time_per_sequence': elapsed / len(sequences),
        }

        print(f"✓ Sequential inference complete:")
        print(f"    Total time: {elapsed:.2f}s")
        print(f"    Throughput: {metrics['ops_per_second']:.1f} ops/sec")
        print(f"    Per operation: {metrics['time_per_operation']*1000:.2f}ms")
        print(f"    Per sequence (all concepts): {metrics['time_per_sequence']:.2f}s")

        return metrics

    def benchmark_batch_inference(
        self,
        classifiers: Dict[str, torch.nn.Module],
        sequences: List[Tuple[str, np.ndarray]],
        batch_size: int = 32
    ) -> Dict:
        """Benchmark running classifiers on batched sequences."""
        print(f"\nBenchmarking batch inference (batch_size={batch_size})...")

        start_time = time.time()

        # Pool all sequences
        pooled_seqs = np.array([seq.mean(axis=0) for _, seq in sequences])
        X = torch.FloatTensor(pooled_seqs).to(self.device)

        results = {}
        total_operations = 0

        for concept, classifier in classifiers.items():
            # Run in batches
            all_preds = []
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                with torch.no_grad():
                    preds = classifier(batch).squeeze()
                all_preds.append(preds.cpu().numpy())
                total_operations += len(batch)

            results[concept] = np.concatenate(all_preds)

        elapsed = time.time() - start_time

        metrics = {
            'total_time': elapsed,
            'batch_size': batch_size,
            'n_classifiers': len(classifiers),
            'n_sequences': len(sequences),
            'total_operations': total_operations,
            'ops_per_second': total_operations / elapsed,
            'time_per_operation': elapsed / total_operations,
            'speedup_vs_sequential': None  # Will be filled in later
        }

        print(f"✓ Batch inference complete:")
        print(f"    Total time: {elapsed:.2f}s")
        print(f"    Throughput: {metrics['ops_per_second']:.1f} ops/sec")
        print(f"    Per operation: {metrics['time_per_operation']*1000:.2f}ms")

        return metrics

    def measure_memory(self, n_classifiers: int) -> Dict:
        """Measure memory usage with loaded classifiers."""
        # System memory
        process = psutil.Process()
        mem_info = process.memory_info()
        sys_mem_mb = mem_info.rss / (1024 ** 2)

        # GPU memory
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        else:
            gpu_mem_allocated = 0
            gpu_mem_reserved = 0

        metrics = {
            'n_classifiers': n_classifiers,
            'system_memory_mb': sys_mem_mb,
            'gpu_memory_allocated_mb': gpu_mem_allocated,
            'gpu_memory_reserved_mb': gpu_mem_reserved,
            'memory_per_classifier_mb': sys_mem_mb / n_classifiers if n_classifiers > 0 else 0
        }

        print(f"\nMemory usage ({n_classifiers} classifiers):")
        print(f"  System: {sys_mem_mb:.1f} MB ({metrics['memory_per_classifier_mb']:.2f} MB/classifier)")
        print(f"  GPU allocated: {gpu_mem_allocated:.1f} MB")
        print(f"  GPU reserved: {gpu_mem_reserved:.1f} MB")

        return metrics

    def run_benchmark(
        self,
        n_concepts: int,
        n_sequences: int,
        batch_size: int = 32
    ) -> Dict:
        """Run complete benchmark suite."""
        print("="*70)
        print("TEMPORAL INFERENCE BENCHMARK")
        print("="*70)
        print(f"Concepts: {n_concepts}")
        print(f"Sequences: {n_sequences}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {self.device}")

        # Select concepts
        if n_concepts > len(self.available_concepts):
            print(f"Warning: Only {len(self.available_concepts)} concepts available, using all")
            n_concepts = len(self.available_concepts)

        concepts = self.available_concepts[:n_concepts]

        # Load classifiers
        classifiers = self.train_classifiers(concepts)

        # Measure memory after loading
        mem_metrics = self.measure_memory(len(classifiers))
        self.metrics['memory'] = mem_metrics

        # Generate test sequences
        sequences = self.generate_test_sequences(n_sequences)

        # Benchmark sequential
        seq_metrics = self.benchmark_sequential_inference(classifiers, sequences)
        self.metrics['inference']['sequential'] = seq_metrics

        # Benchmark batched
        batch_metrics = self.benchmark_batch_inference(classifiers, sequences, batch_size)
        batch_metrics['speedup_vs_sequential'] = seq_metrics['ops_per_second'] / batch_metrics['ops_per_second']
        self.metrics['inference']['batched'] = batch_metrics

        # Summary
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        print(f"\nThroughput comparison:")
        print(f"  Sequential: {seq_metrics['ops_per_second']:.1f} ops/sec")
        print(f"  Batched (size={batch_size}): {batch_metrics['ops_per_second']:.1f} ops/sec")
        print(f"  Speedup: {batch_metrics['ops_per_second']/seq_metrics['ops_per_second']:.2f}x")

        print(f"\nMemory efficiency:")
        print(f"  {mem_metrics['memory_per_classifier_mb']:.2f} MB per classifier")
        print(f"  Projected for 1000 concepts: {mem_metrics['memory_per_classifier_mb'] * 1000:.1f} MB")

        print(f"\nLatency per full analysis ({n_concepts} concepts):")
        print(f"  Sequential: {seq_metrics['time_per_sequence']:.2f}s")
        print(f"  Batched: {batch_metrics['total_time']/n_sequences:.2f}s")

        return self.metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark temporal inference")
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model to use')
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training data directory (e.g., results/adaptive_relfirst_100)')
    parser.add_argument('--n-concepts', type=int, default=10,
                       help='Number of concepts to test')
    parser.add_argument('--n-sequences', type=int, default=100,
                       help='Number of test sequences to generate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for batched inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for metrics')

    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map=args.device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded")

    # Run benchmark
    benchmark = TemporalInferenceBenchmark(
        model=model,
        tokenizer=tokenizer,
        train_data_dir=Path(args.train_data),
        device=args.device
    )

    metrics = benchmark.run_benchmark(
        n_concepts=args.n_concepts,
        n_sequences=args.n_sequences,
        batch_size=args.batch_size
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Metrics saved to {output_path}")


if __name__ == '__main__':
    main()
