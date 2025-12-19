#!/usr/bin/env python3
"""
Concept Activation Mapper using learned classifier weights.

Instead of computing centroids from activation distributions (which didn't work),
we extract the learned feature importance from trained binary classifiers.

The key insight: each classifier's first layer weights encode what features
it learned to look for. This is the discriminative signal that IDF missed.

The mapping is:
    concept_representation[concept] = L2_norm(classifier.layer1.weights, axis=0)

This gives us a [4096] vector per concept showing which features that concept cares about.
To classify a new activation, we compute similarity between the activation and these
learned importance vectors.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class ClassifierMapActivation:
    """Activation strength for a single concept."""
    concept: str
    layer: int
    score: float
    normalized: float


@dataclass
class ClassifierActivationMap:
    """Complete activation map from classifier-based mapping."""
    activations: List[ClassifierMapActivation]
    by_concept: Dict[str, ClassifierMapActivation]
    top_concepts: List[str]
    n_concepts: int


def extract_classifier_signature(state_dict: Dict) -> np.ndarray:
    """
    Extract the learned "signature" from a classifier.

    We use the L2 norm of first layer weights across hidden units.
    This tells us which input features the classifier learned to attend to.
    """
    # Find first layer weights [hidden_dim, input_dim] = [128, 4096]
    for key in state_dict:
        if 'weight' in key and state_dict[key].shape[1] == 4096:
            weights = state_dict[key].numpy()
            # L2 norm across hidden dimension -> [4096]
            return np.linalg.norm(weights, axis=0)
    raise ValueError("Could not find first layer weights")


class ClassifierBasedMapper:
    """
    Maps activations to concepts using learned classifier signatures.

    Unlike the centroid approach which failed, this uses what the classifiers
    actually learned - their first layer weights encode which features matter.
    """

    def __init__(self):
        self.signatures: Dict[str, np.ndarray] = {}
        self.concept_layers: Dict[str, int] = {}
        self._signature_matrix: Optional[np.ndarray] = None
        self._concept_list: Optional[List[Tuple[str, int]]] = None

    def load_from_lens_pack(
        self,
        lens_pack_dir: Path,
        layers: Optional[List[int]] = None,
    ):
        """Load classifier signatures from a lens pack."""
        lens_pack_dir = Path(lens_pack_dir)

        if layers is None:
            layers = []
            for layer_dir in lens_pack_dir.iterdir():
                if layer_dir.is_dir() and layer_dir.name.startswith('layer'):
                    try:
                        layers.append(int(layer_dir.name.replace('layer', '')))
                    except ValueError:
                        pass
            layers.sort()

        print(f"Loading classifier signatures from layers {layers}...")

        for layer in layers:
            layer_dir = lens_pack_dir / f"layer{layer}"
            if not layer_dir.exists():
                continue

            for lens_path in tqdm(list(layer_dir.glob("*_classifier.pt")), desc=f"Layer {layer}"):
                concept = lens_path.stem.replace("_classifier", "")
                try:
                    state_dict = torch.load(lens_path, map_location='cpu')
                    signature = extract_classifier_signature(state_dict)
                    self.signatures[concept] = signature
                    self.concept_layers[concept] = layer
                except Exception as e:
                    print(f"  Error loading {concept}: {e}")

        print(f"Loaded {len(self.signatures)} classifier signatures")

        # Build matrix for batch operations
        self._concept_list = [(c, self.concept_layers[c]) for c in self.signatures.keys()]
        self._signature_matrix = np.stack([self.signatures[c] for c, _ in self._concept_list])

        # L2 normalize each row for cosine similarity
        norms = np.linalg.norm(self._signature_matrix, axis=1, keepdims=True)
        self._signature_matrix = self._signature_matrix / (norms + 1e-8)

    def compute_activations(
        self,
        activation: np.ndarray,
        top_k: int = 10,
        layer_filter: Optional[List[int]] = None,
    ) -> ClassifierActivationMap:
        """
        Compute concept activations for a model hidden state.

        The key insight: we compute similarity between the activation and
        each classifier's learned importance pattern. A high score means
        the activation is strong in the features that classifier cares about.
        """
        if self._signature_matrix is None:
            raise ValueError("Must call load_from_lens_pack first")

        # Weight activation by each signature and sum
        # This is like asking: "how much does this activation match what each classifier looks for?"
        # Method 1: Dot product with normalized signatures (cosine-like)
        activation_normalized = activation / (np.linalg.norm(activation) + 1e-8)
        scores = self._signature_matrix @ activation_normalized

        # Method 2 (alternative): Weight activation by signature, sum
        # scores = (self._signature_matrix * activation[np.newaxis, :]).sum(axis=1)

        # Normalize scores
        min_score = scores.min()
        max_score = scores.max()
        score_range = max_score - min_score + 1e-8
        normalized = (scores - min_score) / score_range

        # Build activation list
        activations = []
        for i, (concept, layer) in enumerate(self._concept_list):
            if layer_filter and layer not in layer_filter:
                continue
            activations.append(ClassifierMapActivation(
                concept=concept,
                layer=layer,
                score=float(scores[i]),
                normalized=float(normalized[i]),
            ))

        # Sort by score
        activations.sort(key=lambda x: x.score, reverse=True)
        activations = activations[:top_k]

        return ClassifierActivationMap(
            activations=activations,
            by_concept={a.concept: a for a in activations},
            top_concepts=[a.concept for a in activations],
            n_concepts=len(self._concept_list),
        )

    def save(self, path: Path):
        """Save mapper to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "signature_matrix.npy", self._signature_matrix)

        metadata = {
            'concept_list': self._concept_list,
            'n_concepts': len(self._concept_list),
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ClassifierBasedMapper':
        """Load mapper from disk."""
        path = Path(path)

        mapper = cls()
        mapper._signature_matrix = np.load(path / "signature_matrix.npy")

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        mapper._concept_list = [tuple(x) for x in metadata['concept_list']]
        mapper.concept_layers = {c: l for c, l in mapper._concept_list}

        return mapper


def run_temporal_test(
    mapper: ClassifierBasedMapper,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    device: str = "cuda",
    layer_idx: int = -1,
):
    """Run temporal concept detection using classifier-based mapping."""
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        token_ids = outputs.sequences[0][prompt_len:].cpu().tolist()
        tokens = [tokenizer.decode([tid]) for tid in token_ids]

        print(f"\nPrompt: {prompt}")
        print(f"Generated: {''.join(tokens)}")
        print("\n" + "-" * 80)

        for step_idx, step_states in enumerate(outputs.hidden_states):
            last_layer = step_states[layer_idx]
            hidden_state = last_layer[:, -1, :].float().cpu().numpy()[0]

            act_map = mapper.compute_activations(hidden_state, top_k=5)

            token = tokens[step_idx] if step_idx < len(tokens) else '<eos>'
            print(f"\nToken [{prompt_len + step_idx}]: '{token}'")
            for act in act_map.activations:
                print(f"  [L{act.layer}] {act.concept:30s} {act.normalized:.3f} (raw: {act.score:.3f})")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Classifier-based concept mapping')
    parser.add_argument('--lens-pack', required=True, help='Path to lens pack')
    parser.add_argument('--model', type=str, default=None, help='Model for temporal test')
    parser.add_argument('--layers', nargs='+', type=int, default=None, help='Layers to load')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt for temporal test')
    parser.add_argument('--save', type=str, default=None, help='Save mapper to path')

    args = parser.parse_args()

    mapper = ClassifierBasedMapper()
    mapper.load_from_lens_pack(Path(args.lens_pack), args.layers)

    if args.save:
        mapper.save(Path(args.save))
        print(f"Saved mapper to {args.save}")

    if args.model and args.prompt:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"\nLoading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        run_temporal_test(mapper, model, tokenizer, args.prompt)


if __name__ == '__main__':
    main()
