"""
Steering Vector Composition Test

Research questions:
1. Does training data ratio (defs:rels) affect steering vector effectiveness?
2. Can we selectively weight centroid (defs) vs boundaries (rels) during extraction?

Experiment design:
- Train with 3 different ratios: 1×100, 50×100, 100×100
- Extract steering vectors with 4 different weightings:
  - defs_only (1.0, 0.0): Centroid-focused
  - def_heavy (0.8, 0.2): Mostly centroid
  - balanced (0.5, 0.5): Equal weight
  - rels_only (0.0, 1.0): Boundary-focused
- Test steering effectiveness with semantic mention counting

Usage:
    python scripts/test_steering_composition.py \
        --model google/gemma-3-4b-pt \
        --concepts "happiness,anger,democracy" \
        --output-dir results/steering_composition
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.stage_1_5_temporal import get_activation_sequence
from src.map.meld.wordnet_graph_v2 import build_concept_graph


class SteeringCompositionTest:
    """Test how training data composition affects steering vectors."""

    def __init__(
        self,
        model,
        tokenizer,
        concept_data: Dict,
        output_dir: Path,
        device: str = "cuda",
        layer_idx: int = -1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.concept_data = concept_data
        self.output_dir = output_dir
        self.device = device
        self.layer_idx = layer_idx

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training ratios to test
        self.training_ratios = {
            '1x100': (1, 100),    # Minimal defs, maximal rels
            '50x100': (50, 100),  # Balanced
            '100x100': (100, 100) # Equal defs and rels
        }

        # Weighting configs for vector extraction
        self.weighting_configs = {
            'defs_only': (1.0, 0.0),
            'def_heavy': (0.8, 0.2),
            'balanced': (0.5, 0.5),
            'rels_only': (0.0, 1.0)
        }

    def generate_training_data(
        self,
        concept: str,
        n_defs: int,
        n_rels: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate training data with specified ratio."""
        print(f"  Generating {n_defs}×{n_rels} training data for '{concept}'...")

        negatives = self.concept_data[concept].get('negatives', [])
        related_structured = self.concept_data[concept].get('related_structured', {})

        pos_seqs = []
        neg_seqs = []

        # Positive: definitions
        for _ in range(n_defs):
            prompt = f"What is {concept}?"
            seq, _ = get_activation_sequence(
                self.model, self.tokenizer, prompt,
                self.layer_idx, self.device
            )
            pos_seqs.append(seq)

        # Positive: relationships
        all_related = []
        for rel_type in ['hypernyms', 'hyponyms', 'meronyms', 'holonyms']:
            if rel_type in related_structured:
                all_related.extend(related_structured[rel_type])

        if all_related:
            for i in range(n_rels):
                related = all_related[i % len(all_related)]
                prompt = f"The relationship between {concept} and {related}"
                seq, _ = get_activation_sequence(
                    self.model, self.tokenizer, prompt,
                    self.layer_idx, self.device
                )
                pos_seqs.append(seq)
        else:
            # Fallback: more definitions
            for _ in range(n_rels):
                prompt = f"What is {concept}?"
                seq, _ = get_activation_sequence(
                    self.model, self.tokenizer, prompt,
                    self.layer_idx, self.device
                )
                pos_seqs.append(seq)

        # Negatives
        n_total = n_defs + n_rels
        for i in range(n_total):
            neg = negatives[i % len(negatives)]
            prompt = f"What is {neg}?"
            seq, _ = get_activation_sequence(
                self.model, self.tokenizer, prompt,
                self.layer_idx, self.device
            )
            neg_seqs.append(seq)

        return pos_seqs, neg_seqs

    def train_classifier(
        self,
        pos_seqs: List[np.ndarray],
        neg_seqs: List[np.ndarray]
    ) -> torch.nn.Module:
        """Train binary classifier on provided data."""
        from torch import nn
        from torch.utils.data import TensorDataset, DataLoader

        # Pool temporal sequences (mean over time)
        pos_pooled = np.array([seq.mean(axis=0) for seq in pos_seqs])
        neg_pooled = np.array([seq.mean(axis=0) for seq in neg_seqs])

        X_train = np.vstack([pos_pooled, neg_pooled])
        y_train = np.array([1] * len(pos_pooled) + [0] * len(neg_pooled))

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)

        hidden_dim = X_train.shape[1]

        # Create classifier
        classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(self.device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

        # Train
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        classifier.train()
        for epoch in range(10):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = classifier(batch_X).squeeze()
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        classifier.eval()
        return classifier

    def extract_steering_vector(
        self,
        concept: str,
        pos_seqs: List[np.ndarray],
        def_weight: float,
        rel_weight: float,
        n_defs: int
    ) -> np.ndarray:
        """Extract steering vector with weighted combination."""
        # Separate definitions from relationships
        def_seqs = pos_seqs[:n_defs]
        rel_seqs = pos_seqs[n_defs:]

        # Pool temporal sequences
        def_pooled = np.array([seq.mean(axis=0) for seq in def_seqs]) if def_seqs else np.array([])
        rel_pooled = np.array([seq.mean(axis=0) for seq in rel_seqs]) if rel_seqs else np.array([])

        # Compute weighted means
        if len(def_pooled) > 0 and len(rel_pooled) > 0:
            def_centroid = def_pooled.mean(axis=0)
            rel_centroid = rel_pooled.mean(axis=0)
            vector = def_weight * def_centroid + rel_weight * rel_centroid
        elif len(def_pooled) > 0:
            vector = def_pooled.mean(axis=0)
        elif len(rel_pooled) > 0:
            vector = rel_pooled.mean(axis=0)
        else:
            raise ValueError(f"No training data for {concept}")

        return vector

    def test_steering_effectiveness(
        self,
        concept: str,
        steering_vector: np.ndarray,
        strengths: List[float] = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ) -> Dict:
        """Test steering vector at multiple strengths."""
        # Get semantic field (concept + related terms)
        related_terms = set([concept.lower()])
        related_structured = self.concept_data[concept].get('related_structured', {})

        for rel_type in ['hypernyms', 'hyponyms', 'meronyms', 'holonyms']:
            if rel_type in related_structured:
                related_terms.update([r.lower() for r in related_structured[rel_type][:5]])

        results = {}

        for strength in strengths:
            # Apply steering vector
            vector_scaled = torch.FloatTensor(steering_vector * strength).to(self.device)

            # Generate with steering
            prompt = f"Tell me about {concept}."
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Hook to apply steering
            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                hidden_states[:, -1, :] += vector_scaled
                return output

            handle = list(self.model.modules())[self.layer_idx].register_forward_hook(steering_hook)

            try:
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            finally:
                handle.remove()

            # Count semantic mentions
            output_lower = output_text.lower()
            mentions = sum(1 for term in related_terms if term in output_lower)

            results[strength] = {
                'output': output_text,
                'semantic_mentions': mentions,
                'related_terms_checked': list(related_terms)
            }

        return results

    def run_experiment(self, concepts: List[str]):
        """Run complete composition experiment."""
        results = {}

        for concept in concepts:
            print(f"\n{'='*70}")
            print(f"Testing concept: {concept}")
            print(f"{'='*70}")

            concept_results = {}

            for ratio_name, (n_defs, n_rels) in self.training_ratios.items():
                print(f"\nTraining ratio: {ratio_name} ({n_defs}×{n_rels})")

                # Generate training data
                pos_seqs, neg_seqs = self.generate_training_data(concept, n_defs, n_rels)

                # Train classifier
                print(f"  Training classifier...")
                classifier = self.train_classifier(pos_seqs, neg_seqs)

                ratio_results = {}

                for weight_name, (def_w, rel_w) in self.weighting_configs.items():
                    print(f"  Extracting {weight_name} vector ({def_w:.1f} defs, {rel_w:.1f} rels)...")

                    # Extract weighted steering vector
                    steering_vector = self.extract_steering_vector(
                        concept, pos_seqs, def_w, rel_w, n_defs
                    )

                    # Test steering effectiveness
                    print(f"  Testing steering effectiveness...")
                    effectiveness = self.test_steering_effectiveness(concept, steering_vector)

                    ratio_results[weight_name] = effectiveness

                concept_results[ratio_name] = ratio_results

            results[concept] = concept_results

            # Save intermediate results
            output_file = self.output_dir / f"{concept}_results.json"
            with open(output_file, 'w') as f:
                json.dump(concept_results, f, indent=2)
            print(f"\n✓ Saved results to {output_file}")

        # Save complete results
        output_file = self.output_dir / "complete_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*70}")
        print(f"✓ All experiments complete: {output_file}")
        print(f"{'='*70}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Test steering vector composition")
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt')
    parser.add_argument('--concepts', type=str, required=True,
                       help='Comma-separated list of concepts to test')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')

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

    # Build concept graph
    concepts_list = [c.strip() for c in args.concepts.split(',')]
    print(f"\nBuilding concept graph for {len(concepts_list)} concepts...")
    concept_data = build_concept_graph(concepts_list)
    print(f"✓ Concept graph built")

    # Run experiment
    tester = SteeringCompositionTest(
        model=model,
        tokenizer=tokenizer,
        concept_data=concept_data,
        output_dir=Path(args.output_dir),
        device=args.device
    )

    results = tester.run_experiment(concepts_list)


if __name__ == '__main__':
    main()
