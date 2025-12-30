#!/usr/bin/env python3
"""
Matrix-based concept comparison using classifier weight structure.

Instead of running N classifiers sequentially (slow) or extracting simple vectors (loses info),
we batch the matrix operations and compare activation-weight interactions directly.

Key insight: The classifier's hidden layer activations ARE the "fingerprint" of how
the input matches that classifier's learned pattern. A 192-dim fingerprint (128+64)
preserves much more discrimination than the final 1-dim probability.

Approaches:
1. Batched full inference - stack all classifiers, run in parallel
2. First-layer fingerprints - h1 = W1 @ activation gives 128-dim interaction pattern
3. Full hidden fingerprints - concatenate h1 and h2 for 192-dim pattern
4. Weight-activation correlation - more sophisticated similarity measures
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class ConceptMatch:
    """Match result for a single concept."""
    concept: str
    layer: int
    probability: float  # From full classifier
    fingerprint_norm: float  # Norm of hidden activation fingerprint
    first_layer_response: float  # Mean activation in first hidden layer


class BatchedClassifierBank:
    """
    All classifiers stacked for batch inference.

    Instead of running N separate forward passes, we stack all weights
    into tensors and do batch matrix multiplies.

    Classifier architecture:
        Linear(4096 -> 128) -> ReLU -> Dropout(0.2)
        Linear(128 -> 64) -> ReLU -> Dropout(0.2)
        Linear(64 -> 1) -> Sigmoid
    """

    def __init__(self):
        self.concepts: List[Tuple[str, int]] = []  # (name, layer)

        # Stacked weight matrices [N, out_dim, in_dim]
        self.W1: Optional[torch.Tensor] = None  # [N, 128, 4096]
        self.b1: Optional[torch.Tensor] = None  # [N, 128]
        self.W2: Optional[torch.Tensor] = None  # [N, 64, 128]
        self.b2: Optional[torch.Tensor] = None  # [N, 64]
        self.W3: Optional[torch.Tensor] = None  # [N, 1, 64]
        self.b3: Optional[torch.Tensor] = None  # [N, 1]

        self.device = 'cpu'

    def load_from_lens_pack(
        self,
        lens_pack_dir: Path,
        layers: Optional[List[int]] = None,
        device: str = 'cuda',
    ):
        """Load all classifiers and stack their weights."""
        self.device = device
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

        # Collect all weights
        W1_list, b1_list = [], []
        W2_list, b2_list = [], []
        W3_list, b3_list = [], []

        print(f"Loading classifiers from layers {layers}...")

        for layer in layers:
            layer_dir = lens_pack_dir / f"layer{layer}"
            if not layer_dir.exists():
                continue

            for lens_path in tqdm(list(layer_dir.glob("*_classifier.pt")), desc=f"Layer {layer}"):
                concept = lens_path.stem.replace("_classifier", "")

                try:
                    state_dict = torch.load(lens_path, map_location='cpu')

                    # Extract weights (handle different key formats)
                    w1 = self._get_weight(state_dict, 0, 'weight')
                    b1 = self._get_weight(state_dict, 0, 'bias')
                    w2 = self._get_weight(state_dict, 3, 'weight')
                    b2 = self._get_weight(state_dict, 3, 'bias')
                    w3 = self._get_weight(state_dict, 6, 'weight')
                    b3 = self._get_weight(state_dict, 6, 'bias')

                    W1_list.append(w1)
                    b1_list.append(b1)
                    W2_list.append(w2)
                    b2_list.append(b2)
                    W3_list.append(w3)
                    b3_list.append(b3)

                    self.concepts.append((concept, layer))

                except Exception as e:
                    print(f"  Error loading {concept}: {e}")

        # Stack into batch tensors
        self.W1 = torch.stack(W1_list).to(device)  # [N, 128, 4096]
        self.b1 = torch.stack(b1_list).to(device)  # [N, 128]
        self.W2 = torch.stack(W2_list).to(device)  # [N, 64, 128]
        self.b2 = torch.stack(b2_list).to(device)  # [N, 64]
        self.W3 = torch.stack(W3_list).to(device)  # [N, 1, 64]
        self.b3 = torch.stack(b3_list).to(device)  # [N, 1]

        print(f"Loaded {len(self.concepts)} classifiers")
        print(f"  W1 shape: {self.W1.shape}")
        print(f"  Total parameters: {sum(w.numel() for w in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]):,}")

    def _get_weight(self, state_dict: Dict, layer_idx: int, weight_type: str) -> torch.Tensor:
        """Extract weight from state dict handling different key formats."""
        # Try different key patterns
        patterns = [
            f'{layer_idx}.{weight_type}',
            f'net.{layer_idx}.{weight_type}',
        ]
        for pattern in patterns:
            if pattern in state_dict:
                return state_dict[pattern]
        raise KeyError(f"Could not find {weight_type} for layer {layer_idx}")

    def forward_batch(
        self,
        activation: torch.Tensor,  # [4096] or [batch, 4096]
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Run all classifiers in parallel on an activation.

        Args:
            activation: Input activation [4096] or [batch, 4096]
            return_hidden: If True, also return hidden layer activations

        Returns:
            probabilities: [N] or [batch, N] classifier outputs
            hidden: Optional tuple of (h1, h2) hidden activations
        """
        if activation.dim() == 1:
            activation = activation.unsqueeze(0)  # [1, 4096]

        batch_size = activation.shape[0]
        N = len(self.concepts)

        # Expand activation for batch matmul with all classifiers
        # activation: [batch, 4096] -> [batch, 1, 4096]
        # W1: [N, 128, 4096]
        # We want: [batch, N, 128]

        a = activation.unsqueeze(1)  # [batch, 1, 4096]

        # First layer: h1 = ReLU(W1 @ a + b1)
        # einsum: batch i k, N j k -> batch N j
        h1_pre = torch.einsum('bik,njk->bnj', a, self.W1) + self.b1.unsqueeze(0)  # [batch, N, 128]
        h1 = F.relu(h1_pre)

        # Second layer: h2 = ReLU(W2 @ h1 + b2)
        h2_pre = torch.einsum('bni,nji->bnj', h1, self.W2) + self.b2.unsqueeze(0)  # [batch, N, 64]
        h2 = F.relu(h2_pre)

        # Output layer: p = sigmoid(W3 @ h2 + b3)
        out_pre = torch.einsum('bni,nji->bnj', h2, self.W3) + self.b3.unsqueeze(0)  # [batch, N, 1]
        probs = torch.sigmoid(out_pre).squeeze(-1)  # [batch, N]

        if batch_size == 1:
            probs = probs.squeeze(0)  # [N]
            if return_hidden:
                h1 = h1.squeeze(0)  # [N, 128]
                h2 = h2.squeeze(0)  # [N, 64]

        if return_hidden:
            return probs, (h1, h2)
        return probs, None

    def get_top_concepts(
        self,
        activation: torch.Tensor,
        top_k: int = 10,
        threshold: float = 0.3,
        return_fingerprints: bool = False,
    ) -> List[ConceptMatch]:
        """
        Get top matching concepts with optional fingerprints.

        Fingerprints are the hidden layer activations - they encode HOW
        the activation matched the classifier's pattern, not just IF it matched.
        """
        with torch.no_grad():
            probs, hidden = self.forward_batch(activation, return_hidden=return_fingerprints)

        # Get top-k indices
        if threshold > 0:
            mask = probs > threshold
            valid_probs = probs.clone()
            valid_probs[~mask] = -1
            top_indices = torch.argsort(valid_probs, descending=True)[:top_k]
        else:
            top_indices = torch.argsort(probs, descending=True)[:top_k]

        results = []
        for idx in top_indices:
            idx = idx.item()
            if probs[idx] < threshold and threshold > 0:
                continue

            concept, layer = self.concepts[idx]

            match = ConceptMatch(
                concept=concept,
                layer=layer,
                probability=float(probs[idx]),
                fingerprint_norm=0.0,
                first_layer_response=0.0,
            )

            if return_fingerprints and hidden is not None:
                h1, h2 = hidden
                # Fingerprint is concatenation of hidden activations
                fingerprint = torch.cat([h1[idx], h2[idx]])
                match.fingerprint_norm = float(fingerprint.norm())
                match.first_layer_response = float(h1[idx].mean())

            results.append(match)

        return results

    def compute_fingerprints(
        self,
        activation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute hidden layer fingerprints for all classifiers.

        Returns:
            h1: [N, 128] first hidden layer activations
            h2: [N, 64] second hidden layer activations

        The concatenated [N, 192] matrix can be compared across activations
        to find similar concept patterns.
        """
        with torch.no_grad():
            _, (h1, h2) = self.forward_batch(activation, return_hidden=True)
        return h1, h2

    def fingerprint_similarity(
        self,
        activation1: torch.Tensor,
        activation2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compare how two activations interact with the classifier bank.

        Returns [N] similarity scores - how similarly each classifier
        responds to the two activations.
        """
        h1_a, h2_a = self.compute_fingerprints(activation1)
        h1_b, h2_b = self.compute_fingerprints(activation2)

        # Concatenate for full fingerprint
        fp_a = torch.cat([h1_a, h2_a], dim=1)  # [N, 192]
        fp_b = torch.cat([h1_b, h2_b], dim=1)  # [N, 192]

        # Cosine similarity per classifier
        fp_a_norm = F.normalize(fp_a, dim=1)
        fp_b_norm = F.normalize(fp_b, dim=1)
        similarity = (fp_a_norm * fp_b_norm).sum(dim=1)  # [N]

        return similarity


def run_temporal_test(
    bank: BatchedClassifierBank,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    model_device: str = "cuda",
):
    """Run temporal concept detection using batched classifiers."""
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
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
            last_layer = step_states[-1]
            # Move to CPU for classifier bank (which is on CPU)
            hidden_state = last_layer[:, -1, :].float().cpu()  # [1, 4096]

            matches = bank.get_top_concepts(
                hidden_state.squeeze(0),
                top_k=5,
                threshold=0.3,
                return_fingerprints=True,
            )

            token = tokens[step_idx] if step_idx < len(tokens) else '<eos>'
            print(f"\nToken [{prompt_len + step_idx}]: '{token}'")
            for m in matches:
                bar = "█" * int(m.probability * 20)
                print(f"  [L{m.layer}] {m.concept:30s} {m.probability:.3f} {bar}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batched classifier inference')
    parser.add_argument('--lens-pack', required=True, help='Path to lens pack')
    parser.add_argument('--model', type=str, default=None, help='Model for temporal test')
    parser.add_argument('--layers', nargs='+', type=int, default=None, help='Layers to load')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt for temporal test')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    # Load classifier bank on CPU to save GPU memory for model
    bank = BatchedClassifierBank()
    bank.load_from_lens_pack(Path(args.lens_pack), args.layers, device='cpu')

    if args.model and args.prompt:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\nLoading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
        )

        run_temporal_test(bank, model, tokenizer, args.prompt, model_device=args.device)


if __name__ == '__main__':
    main()
