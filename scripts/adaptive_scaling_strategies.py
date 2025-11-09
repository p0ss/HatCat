"""
Adaptive Scaling Strategies - Controlled Experiment

Tests three different scaling strategies to reach 95% accuracy:

1. SYMMETRIC: X(C+R)
   - Each iteration adds 1 definition + 1 relationship sample
   - Ignores relationship count N

2. HALF-SCALED: X(C(N/2))
   - Each iteration adds max(1, N/2) definition samples
   - Scales with relationship count

3. RELFIRST-PURE: X(C*N)
   - Each iteration adds N definition samples (one per relationship)
   - Fully scales with relationship count

All strategies:
- Use relationship-first: each edge generated once per iteration, reused for both concepts
- Adaptively iterate until 95% accuracy
- Graduate concepts or hit OOM ceiling (no minimum sample floor)
- Test on fixed holdout set
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import argparse
import json
import time
from typing import Dict, List, Set, Tuple
import sys
import psutil

sys.path.append(str(Path(__file__).parent.parent))

from scripts.stage_1_5_temporal import get_activation_sequence


class AdaptiveScalingStrategies:
    """Manages adaptive scaling with different strategies."""

    def __init__(
        self,
        model,
        tokenizer,
        concept_data: Dict,
        output_dir: Path,
        strategy: str = "symmetric",  # symmetric, half-scaled, relfirst-pure
        target_accuracy: float = 0.95,
        layer_idx: int = -1,
        device: str = "cuda",
        max_data_size: int = 200,
        memory_threshold_gb: float = 2.0
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.concept_data = concept_data
        self.output_dir = output_dir
        self.strategy = strategy
        self.target_accuracy = target_accuracy
        self.layer_idx = layer_idx
        self.device = device
        self.max_data_size = max_data_size
        self.memory_threshold_gb = memory_threshold_gb

        # State
        self.active_concepts: Set[str] = set()
        self.concept_training_level: Dict[str, Tuple[int, int]] = {}  # {concept: (defs, rels)}
        self.concept_accuracy: Dict[str, float] = {}
        self.test_sets: Dict[str, Tuple[List, List]] = {}  # {concept: (pos_seqs, neg_seqs)}
        self.iteration: int = 0

        # Relationship cache: {(concept_a, concept_b): activation}
        self.relationship_cache: Dict[Tuple[str, str], np.ndarray] = {}

        # Track which relationships each concept uses
        self.concept_relationships: Dict[str, List[Tuple[str, str]]] = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.train_data_dir = self.output_dir / "train_data"
        self.train_data_dir.mkdir(exist_ok=True)
        self.relationship_cache_file = self.output_dir / "relationship_cache.npz"

    def save_relationship_cache(self):
        """Save relationship cache to disk."""
        if not self.relationship_cache:
            return

        # Convert dict to arrays for npz format
        keys = list(self.relationship_cache.keys())
        key_strs = [f"{a}|||{b}" for a, b in keys]
        values = np.array([self.relationship_cache[k] for k in keys])

        np.savez_compressed(
            self.relationship_cache_file,
            keys=key_strs,
            values=values
        )

    def load_relationship_cache(self) -> bool:
        """Load relationship cache from disk. Returns True if loaded."""
        if not self.relationship_cache_file.exists():
            return False

        data = np.load(self.relationship_cache_file, allow_pickle=True)
        keys = data['keys']
        values = data['values']

        for key_str, value in zip(keys, values):
            a, b = key_str.split('|||')
            self.relationship_cache[(a, b)] = value

        print(f"  ✓ Loaded {len(self.relationship_cache)} relationships from cache")
        return True

    def save_train_data(self, concept: str, train_pos: List, train_neg: List):
        """Save accumulated training data to disk."""
        data_path = self.train_data_dir / f"{concept.replace('/', '_')}.npz"

        pos_arrays = [seq.cpu().numpy() if isinstance(seq, torch.Tensor) else seq for seq in train_pos]
        neg_arrays = [seq.cpu().numpy() if isinstance(seq, torch.Tensor) else seq for seq in train_neg]

        np.savez_compressed(
            data_path,
            pos=np.array(pos_arrays),
            neg=np.array(neg_arrays)
        )

    def load_train_data(self, concept: str) -> Tuple[List, List]:
        """Load accumulated training data from disk."""
        data_path = self.train_data_dir / f"{concept.replace('/', '_')}.npz"

        if not data_path.exists():
            raise ValueError(f"No saved data for concept '{concept}'")

        data = np.load(data_path)
        train_pos = [data['pos'][i] for i in range(len(data['pos']))]
        train_neg = [data['neg'][i] for i in range(len(data['neg']))]

        return train_pos, train_neg

    def save_checkpoint(self):
        """Save current state to disk."""
        checkpoint = {
            'iteration': self.iteration,
            'active_concepts': list(self.active_concepts),
            'concept_training_level': {k: list(v) for k, v in self.concept_training_level.items()},
            'concept_accuracy': self.concept_accuracy,
            'concept_relationships': {k: [list(edge) for edge in v] for k, v in self.concept_relationships.items()},
            'n_done': len(self.concept_data) - len(self.active_concepts),
            'n_total': len(self.concept_data)
        }

        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        # Also save relationship cache
        self.save_relationship_cache()

        print(f"  💾 Checkpoint saved: {self.checkpoint_path}")

    def load_checkpoint(self) -> bool:
        """Load state from disk if exists. Returns True if loaded."""
        if not self.checkpoint_path.exists():
            return False

        with open(self.checkpoint_path) as f:
            checkpoint = json.load(f)

        self.iteration = checkpoint['iteration']
        self.active_concepts = set(checkpoint['active_concepts'])
        self.concept_training_level = {k: tuple(v) for k, v in checkpoint['concept_training_level'].items()}
        self.concept_accuracy = checkpoint['concept_accuracy']
        self.concept_relationships = {k: [tuple(edge) for edge in v] for k, v in checkpoint['concept_relationships'].items()}

        # Load relationship cache
        self.load_relationship_cache()

        print(f"  ✓ Loaded checkpoint from iteration {self.iteration}")
        print(f"    Active concepts: {len(self.active_concepts)}/{checkpoint['n_total']}")

        return True

    def initialize_concepts(self, concepts: List[str]):
        """Initialize all concepts (start at 0×0, first iteration will add data)."""
        self.active_concepts = set(concepts)

        for concept in concepts:
            self.concept_training_level[concept] = (0, 0)
            self.concept_accuracy[concept] = 0.0
            self.concept_relationships[concept] = []

        print(f"\nInitialized {len(concepts)} concepts (strategy: {self.strategy})")

    def generate_test_set(self, concept: str, n_samples: int = 10):
        """Generate fixed test set for a concept (NOT using relationship cache)."""
        negatives = self.concept_data[concept].get('negatives', [])
        related_structured = self.concept_data[concept].get('related_structured', {})

        if len(negatives) == 0:
            raise ValueError(f"No negatives for concept '{concept}'")

        pos_seqs = []
        neg_seqs = []

        # Positive: definitions
        for _ in range(n_samples // 2):
            prompt = f"What is {concept}?"
            seq, _ = get_activation_sequence(self.model, self.tokenizer, prompt, self.layer_idx, self.device)
            pos_seqs.append(seq)

        # Positive: relationships
        all_related = []
        for rel_type in ['hypernyms', 'hyponyms', 'meronyms', 'holonyms']:
            if rel_type in related_structured:
                all_related.extend(related_structured[rel_type])

        if all_related:
            for i in range(n_samples // 2):
                related = all_related[i % len(all_related)]
                prompt = f"The relationship between {concept} and {related}"
                seq, _ = get_activation_sequence(self.model, self.tokenizer, prompt, self.layer_idx, self.device)
                pos_seqs.append(seq)
        else:
            # Fallback: more definitions
            for _ in range(n_samples // 2):
                prompt = f"What is {concept}?"
                seq, _ = get_activation_sequence(self.model, self.tokenizer, prompt, self.layer_idx, self.device)
                pos_seqs.append(seq)

        # Negatives
        for i in range(n_samples):
            neg = negatives[i % len(negatives)]
            prompt = f"What is {neg}?"
            seq, _ = get_activation_sequence(self.model, self.tokenizer, prompt, self.layer_idx, self.device)
            neg_seqs.append(seq)

        self.test_sets[concept] = (pos_seqs, neg_seqs)
        return pos_seqs, neg_seqs

    def get_or_generate_relationship(self, concept_a: str, concept_b: str) -> np.ndarray:
        """Get relationship from cache or generate if not exists."""
        # Normalize edge (always store in sorted order)
        edge = tuple(sorted([concept_a, concept_b]))

        if edge not in self.relationship_cache:
            # Generate relationship activation
            prompt = f"The relationship between {concept_a} and {concept_b}"
            seq, _ = get_activation_sequence(self.model, self.tokenizer, prompt, self.layer_idx, self.device)
            self.relationship_cache[edge] = seq

        return self.relationship_cache[edge]

    def check_memory_available(self) -> Tuple[bool, str]:
        """Check if sufficient memory is available for next iteration."""
        # Check system memory
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)

        # Check GPU memory if using CUDA
        gpu_mem_free = None
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_mem_free = torch.cuda.mem_get_info()[0] / (1024 ** 3)

            if gpu_mem_free < self.memory_threshold_gb:
                return False, f"GPU memory low: {gpu_mem_free:.2f}GB free (need {self.memory_threshold_gb}GB)"

        if available_gb < self.memory_threshold_gb:
            return False, f"System memory low: {available_gb:.2f}GB free (need {self.memory_threshold_gb}GB)"

        if gpu_mem_free is not None:
            return True, f"Memory OK: {available_gb:.2f}GB system, {gpu_mem_free:.2f}GB GPU"
        else:
            return True, f"Memory OK: {available_gb:.2f}GB system"

    def check_training_size_limit(self, concept: str) -> Tuple[bool, str]:
        """Check if concept has reached maximum training size."""
        defs, rels = self.concept_training_level[concept]

        if defs >= self.max_data_size or rels >= self.max_data_size:
            return False, f"Max training size reached: {defs}×{rels} (limit: {self.max_data_size})"

        return True, f"Training size OK: {defs}×{rels}"

    def get_increment_sizes(self, concept: str) -> Tuple[int, int]:
        """Calculate increment sizes based on strategy and relationship count."""
        # Count relationships for this concept
        related_structured = self.concept_data[concept].get('related_structured', {})
        n_relationships = sum(
            len(related_structured.get(rel_type, []))
            for rel_type in ['hypernyms', 'hyponyms', 'meronyms', 'holonyms']
        )

        if n_relationships == 0:
            # Fallback: treat as 1 to avoid zero samples
            n_relationships = 1

        if self.strategy == "symmetric":
            # X(C+R): 1 definition + 1 relationship per iteration
            return (1, 1)
        elif self.strategy == "half-scaled":
            # X(C(N/2)): max(1, N/2) definitions per iteration
            n_defs = max(1, n_relationships // 2)
            return (n_defs, 1)
        elif self.strategy == "relfirst-pure":
            # X(C*N): N definitions (one per relationship) per iteration
            return (n_relationships, 1)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def add_training_data_relationship_first(self, concept: str, n_defs: int, n_rels: int):
        """Add incremental training data using relationship-first approach."""
        negatives = self.concept_data[concept].get('negatives', [])
        related_structured = self.concept_data[concept].get('related_structured', {})

        # NEW training data
        new_pos = []
        new_neg = []

        # 1. Sample definitions
        for _ in range(n_defs):
            prompt = f"What is {concept}?"
            seq, _ = get_activation_sequence(self.model, self.tokenizer, prompt, self.layer_idx, self.device)
            new_pos.append(seq)

        # 2. Sample relationships (relationship-first: reuse from cache)
        all_related = []
        for rel_type in ['hypernyms', 'hyponyms', 'meronyms', 'holonyms']:
            if rel_type in related_structured:
                all_related.extend(related_structured[rel_type])

        if all_related and n_rels > 0:
            # Track which new relationships we're adding
            new_rels = []
            for i in range(n_rels):
                related = all_related[i % len(all_related)]
                seq = self.get_or_generate_relationship(concept, related)
                new_pos.append(seq)
                new_rels.append((concept, related))

            # Update concept's relationship tracking
            self.concept_relationships[concept].extend(new_rels)
        else:
            # Fallback: more definitions
            for _ in range(n_rels):
                prompt = f"What is {concept}?"
                seq, _ = get_activation_sequence(self.model, self.tokenizer, prompt, self.layer_idx, self.device)
                new_pos.append(seq)

        # 3. Negatives
        n_total = n_defs + n_rels
        for i in range(n_total):
            neg = negatives[i % len(negatives)]
            prompt = f"What is {neg}?"
            seq, _ = get_activation_sequence(self.model, self.tokenizer, prompt, self.layer_idx, self.device)
            new_neg.append(seq)

        # Accumulate with existing data
        data_path = self.train_data_dir / f"{concept.replace('/', '_')}.npz"
        if data_path.exists():
            existing_pos, existing_neg = self.load_train_data(concept)
            train_pos = existing_pos + new_pos
            train_neg = existing_neg + new_neg
        else:
            train_pos = new_pos
            train_neg = new_neg

        # Save accumulated data
        self.save_train_data(concept, train_pos, train_neg)

        return train_pos, train_neg

    def train_concept(self, concept: str):
        """Train classifier for a concept using accumulated training data."""
        from torch import nn
        from torch.utils.data import TensorDataset, DataLoader

        # Load accumulated training data
        train_pos, train_neg = self.load_train_data(concept)

        # Get test data
        if concept not in self.test_sets:
            self.generate_test_set(concept)

        test_pos, test_neg = self.test_sets[concept]

        # Get hidden dim
        hidden_dim = test_pos[0].shape[-1]

        # Pool temporal sequences (mean over time)
        pos_pooled = np.array([seq.mean(axis=0) for seq in train_pos])
        neg_pooled = np.array([seq.mean(axis=0) for seq in train_neg])

        X_train = np.vstack([pos_pooled, neg_pooled])
        y_train = np.array([1] * len(pos_pooled) + [0] * len(neg_pooled))

        # Test data
        test_pos_pooled = np.array([seq.mean(axis=0) for seq in test_pos])
        test_neg_pooled = np.array([seq.mean(axis=0) for seq in test_neg])
        X_test = np.vstack([test_pos_pooled, test_neg_pooled])
        y_test = np.array([1] * len(test_pos_pooled) + [0] * len(test_neg_pooled))

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).cuda()
        y_train = torch.FloatTensor(y_train).cuda()
        X_test = torch.FloatTensor(X_test).cuda()
        y_test = torch.FloatTensor(y_test).cuda()

        # Simple MLP classifier
        model = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).cuda()

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        for epoch in range(10):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_X).squeeze()
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        # Test
        model.eval()
        with torch.no_grad():
            test_pred = (model(X_test).squeeze() > 0.5).float()
            test_acc = (test_pred == y_test).float().mean().item()

        return test_acc

    def run_iteration(self):
        """Run one iteration: test all active concepts, add data to those below target."""
        print(f"\n{'='*70}")
        print(f"ITERATION {self.iteration}")
        print(f"{'='*70}")
        print(f"Active concepts: {len(self.active_concepts)}")
        print(f"Relationship cache size: {len(self.relationship_cache)}")

        # Check memory before proceeding
        mem_ok, mem_msg = self.check_memory_available()
        print(f"Memory status: {mem_msg}")
        if not mem_ok:
            print(f"⚠️  WARNING: {mem_msg}")
            print(f"⚠️  Stopping to avoid crash. Checkpoint saved at iteration {self.iteration - 1}")
            return False
        print()

        start_time = time.time()

        # If iteration 0, add first data increment for all concepts
        if self.iteration == 0:
            print(f"  Generating first training data ({self.strategy} strategy)...")
            for i, concept in enumerate(sorted(self.active_concepts)):
                try:
                    n_defs, n_rels = self.get_increment_sizes(concept)
                    self.add_training_data_relationship_first(concept, n_defs, n_rels)
                    defs, rels = self.concept_training_level[concept]
                    print(f"    [{i+1}/{len(self.active_concepts)}] ✓ {concept}: {defs}×{rels} (N={n_defs+n_rels})", flush=True)
                except Exception as e:
                    print(f"    [{i+1}/{len(self.active_concepts)}] ✗ {concept}: Failed - {e}", flush=True)
                    self.active_concepts.remove(concept)
            print()

        # Test all active concepts
        concepts_to_graduate = []

        for i, concept in enumerate(sorted(self.active_concepts)):
            defs, rels = self.concept_training_level[concept]

            print(f"  [{i+1}/{len(self.active_concepts)}] {concept} ({defs}×{rels})...", end=" ", flush=True)

            try:
                # Train on accumulated data
                accuracy = self.train_concept(concept)
                self.concept_accuracy[concept] = accuracy

                if accuracy >= self.target_accuracy:
                    concepts_to_graduate.append(concept)
                    print(f"✓ {accuracy:.1%} DONE", flush=True)
                else:
                    print(f"{accuracy:.1%}", flush=True)

            except Exception as e:
                print(f"✗ Failed: {e}", flush=True)
                continue

            # Clear cache
            torch.cuda.empty_cache()

        # Graduate concepts that reached target
        for concept in concepts_to_graduate:
            self.active_concepts.remove(concept)

        # Check which concepts hit size limits
        concepts_at_limit = []
        for concept in list(self.active_concepts):
            size_ok, size_msg = self.check_training_size_limit(concept)
            if not size_ok:
                print(f"  ⚠️  {concept}: {size_msg} - graduating with {self.concept_accuracy[concept]:.1%} accuracy")
                concepts_at_limit.append(concept)

        # Remove concepts at size limit
        for concept in concepts_at_limit:
            self.active_concepts.remove(concept)

        # Add incremental data for remaining concepts
        if self.active_concepts:
            print()
            print(f"  Adding more data to {len(self.active_concepts)} concepts ({self.strategy} strategy)...")
            for concept in self.active_concepts:
                try:
                    # Calculate increment sizes based on strategy
                    n_defs, n_rels = self.get_increment_sizes(concept)

                    # Add increment to accumulated data
                    self.add_training_data_relationship_first(concept, n_defs, n_rels)

                    # Update training level counter
                    defs, rels = self.concept_training_level[concept]
                    self.concept_training_level[concept] = (
                        defs + n_defs,
                        rels + n_rels
                    )
                except Exception as e:
                    print(f"    ✗ {concept}: Failed to add data - {e}", flush=True)

        elapsed = time.time() - start_time

        print()
        print(f"Iteration {self.iteration} complete:")
        print(f"  Graduated: {len(concepts_to_graduate)} concepts")
        if concepts_at_limit:
            print(f"  At size limit: {len(concepts_at_limit)} concepts")
        print(f"  Remaining: {len(self.active_concepts)} concepts")
        print(f"  Relationship cache: {len(self.relationship_cache)} edges")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # Save checkpoint
        self.save_checkpoint()

        self.iteration += 1
        return True

    def run(self, max_iterations: int = 100):
        """Run adaptive scaling until convergence or max iterations."""
        print(f"\n{'='*70}")
        print("ADAPTIVE RELATIONSHIP-FIRST TRAINING")
        print(f"{'='*70}")
        print(f"Target accuracy: {self.target_accuracy:.1%}")
        print(f"Strategy: {self.strategy}")
        print(f"Max iterations: {max_iterations}")
        print(f"Max training size: {self.max_data_size}×{self.max_data_size}")
        print(f"Memory threshold: {self.memory_threshold_gb:.1f}GB")
        print()

        while self.active_concepts and self.iteration < max_iterations:
            should_continue = self.run_iteration()
            if not should_continue:
                print("\n⚠️  Stopping due to insufficient memory")
                break

        print(f"\n{'='*70}")
        print("ADAPTIVE TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total iterations: {self.iteration}")
        print(f"Concepts completed: {len(self.concept_data) - len(self.active_concepts)}/{len(self.concept_data)}")
        print(f"Relationship cache: {len(self.relationship_cache)} unique edges")

        if self.active_concepts:
            print(f"Remaining concepts: {len(self.active_concepts)}")
            for concept in sorted(self.active_concepts):
                defs, rels = self.concept_training_level[concept]
                acc = self.concept_accuracy.get(concept, 0.0)
                print(f"  - {concept}: {defs}×{rels}, {acc:.1%} accuracy")
            print("  (stopped at max iterations or memory limit)")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Scaling Strategies - Controlled Experiment"
    )

    parser.add_argument('--concept-graph', type=str, required=True,
                       help='Path to concept graph JSON')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model to use')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for checkpoints and results')
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['symmetric', 'half-scaled', 'relfirst-pure'],
                       help='Scaling strategy: symmetric (1+1), half-scaled (N/2), relfirst-pure (N)')
    parser.add_argument('--n-concepts', type=int, default=None,
                       help='Number of concepts to test (default: all)')
    parser.add_argument('--target-accuracy', type=float, default=0.95,
                       help='Target accuracy for graduation')
    parser.add_argument('--max-data-size', type=int, default=200,
                       help='Maximum training size (defs or rels) before forcing graduation')
    parser.add_argument('--memory-threshold', type=float, default=2.0,
                       help='Minimum free memory in GB before stopping')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if exists')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Load concept graph
    with open(args.concept_graph) as f:
        concept_data = json.load(f)

    all_concepts = list(concept_data.keys())

    if args.n_concepts:
        concepts = all_concepts[:args.n_concepts]
        print(f"Testing with {len(concepts)} concepts (subset)")
    else:
        concepts = all_concepts
        print(f"Testing with all {len(concepts)} concepts")

    # Initialize scaler
    scaler = AdaptiveScalingStrategies(
        model=model,
        tokenizer=tokenizer,
        concept_data=concept_data,
        output_dir=output_dir,
        strategy=args.strategy,
        target_accuracy=args.target_accuracy,
        layer_idx=-1,
        device=args.device,
        max_data_size=args.max_data_size,
        memory_threshold_gb=args.memory_threshold
    )

    # Resume or initialize
    if args.resume and scaler.load_checkpoint():
        print("Resuming from checkpoint...")
    else:
        print("Starting fresh...")
        scaler.initialize_concepts(concepts)

    # Run adaptive scaling
    scaler.run(max_iterations=100)


if __name__ == '__main__':
    main()
