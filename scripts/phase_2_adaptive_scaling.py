"""
Phase 2: Adaptive Scaling to 100k Concepts
===========================================

Goal: Scale to 100k concepts using adaptive training allocation.

Strategy:
1. Start all concepts at 10×10 baseline (Phase 1 sweet spot)
2. Test accuracy on fixed test set
3. Concepts above 95% accuracy are "done" - stop training them
4. Add 1×1 more samples to remaining concepts
5. Repeat until all concepts reach 95%

This allows:
- Fast initial coverage (10×10 for all concepts)
- Efficient resource allocation (only train what needs it)
- Checkpointing and resume capability
- Test at multiple scales (1, 10, 100, 1k, 10k, 100k concepts)
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

sys.path.append(str(Path(__file__).parent.parent))

from scripts.stage_1_5_temporal import get_activation_sequence


class AdaptiveScaler:
    """Manages adaptive scaling with checkpointing."""

    def __init__(
        self,
        model,
        tokenizer,
        concept_data: Dict,
        output_dir: Path,
        target_accuracy: float = 0.95,
        baseline_defs: int = 10,
        baseline_rels: int = 10,
        increment: int = 1,
        layer_idx: int = -1,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.concept_data = concept_data
        self.output_dir = output_dir
        self.target_accuracy = target_accuracy
        self.baseline_defs = baseline_defs
        self.baseline_rels = baseline_rels
        self.increment = increment
        self.layer_idx = layer_idx
        self.device = device

        # State
        self.active_concepts: Set[str] = set()
        self.concept_training_level: Dict[str, Tuple[int, int]] = {}  # {concept: (defs, rels)}
        self.concept_accuracy: Dict[str, float] = {}
        self.test_sets: Dict[str, Tuple[List, List]] = {}  # {concept: (pos_seqs, neg_seqs)}
        self.classifiers: Dict[str, torch.nn.Module] = {}
        self.iteration: int = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.train_data_dir = self.output_dir / "train_data"
        self.train_data_dir.mkdir(exist_ok=True)

    def save_train_data(self, concept: str, train_pos: List, train_neg: List):
        """Save accumulated training data to disk."""
        data_path = self.train_data_dir / f"{concept.replace('/', '_')}.npz"

        # Convert lists of tensors to numpy arrays
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

        # Convert back to list of arrays (keeping as numpy for compatibility)
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
            'n_done': len(self.concept_data) - len(self.active_concepts),
            'n_total': len(self.concept_data)
        }

        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

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

        print(f"  ✓ Loaded checkpoint from iteration {self.iteration}")
        print(f"    Active concepts: {len(self.active_concepts)}/{checkpoint['n_total']}")

        return True

    def initialize_concepts(self, concepts: List[str]):
        """Initialize all concepts at baseline training level."""
        self.active_concepts = set(concepts)

        for concept in concepts:
            self.concept_training_level[concept] = (self.baseline_defs, self.baseline_rels)
            self.concept_accuracy[concept] = 0.0

        print(f"\nInitialized {len(concepts)} concepts at {self.baseline_defs}×{self.baseline_rels}")

    def generate_test_set(self, concept: str, n_samples: int = 10):
        """Generate fixed test set for a concept."""
        from scripts.phase_1_find_curve_v2 import sample_test_set

        negatives = self.concept_data[concept].get('negatives', [])
        related_structured = self.concept_data[concept].get('related_structured', {})

        if len(negatives) == 0:
            raise ValueError(f"No negatives for concept '{concept}'")

        pos_seqs, neg_seqs = sample_test_set(
            self.model, self.tokenizer, concept, negatives, related_structured,
            n_samples, self.layer_idx, self.device
        )

        self.test_sets[concept] = (pos_seqs, neg_seqs)
        return pos_seqs, neg_seqs

    def add_training_data(self, concept: str, n_defs: int, n_rels: int):
        """Add incremental training data for a concept."""
        from scripts.phase_1_find_curve_v2 import sample_train_sequences

        negatives = self.concept_data[concept].get('negatives', [])
        related_structured = self.concept_data[concept].get('related_structured', {})

        # Sample NEW training data
        new_pos, new_neg = sample_train_sequences(
            self.model, self.tokenizer, concept, negatives, related_structured,
            n_defs, n_rels, self.layer_idx, self.device
        )

        # Accumulate with existing data from disk
        data_path = self.train_data_dir / f"{concept.replace('/', '_')}.npz"
        if data_path.exists():
            existing_pos, existing_neg = self.load_train_data(concept)
            train_pos = existing_pos + new_pos
            train_neg = existing_neg + new_neg
        else:
            train_pos = new_pos
            train_neg = new_neg

        # Save accumulated data to disk
        self.save_train_data(concept, train_pos, train_neg)

        return train_pos, train_neg

    def train_concept(self, concept: str):
        """Train classifier for a concept using accumulated training data."""
        from scripts.phase_1_find_curve_v2 import train_and_evaluate

        # Load accumulated training data from disk
        train_pos, train_neg = self.load_train_data(concept)

        # Get test data
        if concept not in self.test_sets:
            self.generate_test_set(concept)

        test_pos, test_neg = self.test_sets[concept]

        # Get hidden dim
        hidden_dim = test_pos[0].shape[-1]

        # Train and evaluate on ALL accumulated data
        train_acc, test_acc = train_and_evaluate(
            train_pos, train_neg, test_pos, test_neg, hidden_dim
        )

        return test_acc

    def run_iteration(self):
        """Run one iteration: test all active concepts, add data to those below target."""
        print(f"\n{'='*70}")
        print(f"ITERATION {self.iteration}")
        print(f"{'='*70}")
        print(f"Active concepts: {len(self.active_concepts)}")
        print()

        start_time = time.time()

        # If iteration 0, add baseline data for all concepts
        if self.iteration == 0:
            print("  Generating baseline training data...")
            for concept in self.active_concepts:
                try:
                    self.add_training_data(concept, self.baseline_defs, self.baseline_rels)
                    defs, rels = self.concept_training_level[concept]
                    print(f"    ✓ {concept}: {defs}×{rels}", flush=True)
                except Exception as e:
                    print(f"    ✗ {concept}: Failed - {e}", flush=True)
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

        # Add incremental data for remaining concepts
        if self.active_concepts:
            print()
            print(f"  Adding {self.increment}×{self.increment} more data to {len(self.active_concepts)} concepts...")
            for concept in self.active_concepts:
                try:
                    # Add increment to accumulated data
                    self.add_training_data(concept, self.increment, self.increment)

                    # Update training level counter
                    defs, rels = self.concept_training_level[concept]
                    self.concept_training_level[concept] = (
                        defs + self.increment,
                        rels + self.increment
                    )
                except Exception as e:
                    print(f"    ✗ {concept}: Failed to add data - {e}", flush=True)

        elapsed = time.time() - start_time

        print()
        print(f"Iteration {self.iteration} complete:")
        print(f"  Graduated: {len(concepts_to_graduate)} concepts")
        print(f"  Remaining: {len(self.active_concepts)} concepts")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # Save checkpoint
        self.save_checkpoint()

        self.iteration += 1

    def run(self, max_iterations: int = 100):
        """Run adaptive scaling until convergence or max iterations."""
        print(f"\n{'='*70}")
        print("PHASE 2: ADAPTIVE SCALING")
        print(f"{'='*70}")
        print(f"Target accuracy: {self.target_accuracy:.1%}")
        print(f"Baseline: {self.baseline_defs}×{self.baseline_rels}")
        print(f"Increment: {self.increment}×{self.increment}")
        print(f"Max iterations: {max_iterations}")
        print()

        while self.active_concepts and self.iteration < max_iterations:
            self.run_iteration()

        print(f"\n{'='*70}")
        print("ADAPTIVE SCALING COMPLETE")
        print(f"{'='*70}")
        print(f"Total iterations: {self.iteration}")
        print(f"Concepts completed: {len(self.concept_data) - len(self.active_concepts)}/{len(self.concept_data)}")

        if self.active_concepts:
            print(f"Remaining concepts: {len(self.active_concepts)}")
            print("  (stopped at max iterations)")


def test_at_scale(
    concept_graph_path: Path,
    model_name: str,
    output_dir: Path,
    n_concepts: int = None,
    resume: bool = False,
    device: str = "cuda"
):
    """
    Test scaling at different concept counts.

    Can be run at 1, 10, 100, 1000, 10000, 100000 concepts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded")

    # Load concept graph
    with open(concept_graph_path) as f:
        concept_data = json.load(f)

    all_concepts = list(concept_data.keys())

    if n_concepts:
        concepts = all_concepts[:n_concepts]
        print(f"Testing with {len(concepts)} concepts (subset)")
    else:
        concepts = all_concepts
        print(f"Testing with all {len(concepts)} concepts")

    # Initialize scaler
    scaler = AdaptiveScaler(
        model=model,
        tokenizer=tokenizer,
        concept_data=concept_data,
        output_dir=output_dir,
        target_accuracy=0.95,
        baseline_defs=10,
        baseline_rels=10,
        increment=1,
        layer_idx=-1,
        device=device
    )

    # Resume or initialize
    if resume and scaler.load_checkpoint():
        print("Resuming from checkpoint...")
    else:
        print("Starting fresh...")
        scaler.initialize_concepts(concepts)

    # Run adaptive scaling
    scaler.run(max_iterations=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Phase 2: Adaptive scaling to 100k concepts"
    )

    parser.add_argument('--concept-graph', type=str, required=True,
                       help='Path to concept graph JSON')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Model to use')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for checkpoints and results')
    parser.add_argument('--n-concepts', type=int, default=None,
                       help='Number of concepts to test (default: all)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if exists')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    test_at_scale(
        concept_graph_path=Path(args.concept_graph),
        model_name=args.model,
        output_dir=Path(args.output_dir),
        n_concepts=args.n_concepts,
        resume=args.resume,
        device=args.device
    )
