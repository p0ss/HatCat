"""
Progressive bootstrap using existing activation capture infrastructure.
Implements Stage 0-3 refinement strategy.
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import time
from typing import List, Optional, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hat.classifiers.classifiers.capture import ActivationCapture, ActivationConfig
from src.hat.utils.model_loader import ModelLoader


class ProgressiveBootstrap:
    """
    Progressive refinement bootstrap using existing activation capture.
    Bridges hooks.py with the Stage 0-3 pipeline.
    """

    def __init__(self, model_name: str = "google/gemma-3-270m", device: Optional[str] = None):
        """
        Initialize with existing model loader and activation capture.

        Args:
            model_name: Hugging Face model identifier
            device: Device to use (None = auto-detect)
        """
        print(f"Loading {model_name}...")

        # Use existing model loader
        self.model, self.tokenizer = ModelLoader.load_gemma_270m(
            model_name=model_name,
            device=device
        )

        self.device = next(self.model.parameters()).device

        # Get activation dimension
        with torch.no_grad():
            dummy = self.tokenizer("test", return_tensors="pt").to(self.device)
            outputs = self.model(**dummy, output_hidden_states=True)
            self.activation_dim = outputs.hidden_states[-1].shape[-1]

        print(f"Model loaded. Activation dimension: {self.activation_dim}")

    def get_activation(self, text: str, layer_idx: int = -1) -> np.ndarray:
        """
        Extract activation vector with attention-masked pooling.

        Args:
            text: Input text
            layer_idx: Layer index to extract from

        Returns:
            Activation vector as numpy array
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

            # Get hidden states
            hs = outputs.hidden_states[layer_idx]  # [B, T, D]

            # Attention-masked pooling
            mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        return pooled.squeeze(0).float().cpu().numpy()

    def batch_process(
        self,
        concepts: List[str],
        batch_size: int = 32,
        layer_idx: int = -1
    ) -> np.ndarray:
        """
        Process concepts in batches for efficiency.

        Args:
            concepts: List of concept strings
            batch_size: Batch size for processing
            layer_idx: Layer index to extract

        Returns:
            Stacked activation matrix [n_concepts, hidden_dim]
        """
        activations = []

        for i in tqdm(range(0, len(concepts), batch_size), desc=f"Layer {layer_idx}"):
            batch = concepts[i:i+batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hs = outputs.hidden_states[layer_idx]  # [B, T, D]

                # Attention-masked pooling
                mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

            activations.append(pooled.float().cpu().numpy())

        return np.vstack(activations)

    def bootstrap_stage0(
        self,
        concepts: List[str],
        output_path: Path,
        layer_indices: List[int] = [-1],
        batch_size: int = 32
    ):
        """
        Stage 0 bootstrap: Single-pass raw concept processing.

        Args:
            concepts: List of concept strings
            output_path: Path to save HDF5 file
            layer_indices: Which layers to capture
            batch_size: Batch size for processing
        """
        start_time = time.time()
        n_concepts = len(concepts)

        print(f"\n{'='*70}")
        print("STAGE 0 BOOTSTRAP")
        print(f"{'='*70}")
        print(f"Concepts: {n_concepts:,}")
        print(f"Layers: {layer_indices}")
        print(f"Output: {output_path}")
        print(f"Batch size: {batch_size}")
        print()

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create HDF5 file
        with h5py.File(output_path, 'w') as f:
            # Metadata
            f.attrs['n_concepts'] = n_concepts
            f.attrs['activation_dim'] = self.activation_dim
            f.attrs['model'] = str(self.model.config._name_or_path)
            f.attrs['stage'] = 0
            f.attrs['samples_per_concept'] = 1
            f.attrs['timestamp'] = time.time()

            # Store concepts
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('concepts', data=np.array(concepts, dtype=object), dtype=dt)

            # Process each layer
            for layer_idx in layer_indices:
                print(f"\nProcessing layer {layer_idx}...")

                # Batch process for efficiency
                activations = self.batch_process(concepts, batch_size, layer_idx)

                # Store activations (float16 for space efficiency)
                f.create_dataset(
                    f'layer_{layer_idx}/activations',
                    data=activations.astype(np.float16),
                    compression='gzip',
                    compression_opts=4
                )

                # Initialize variance as unknown (computed in later stages)
                f.create_dataset(
                    f'layer_{layer_idx}/variance',
                    data=np.full(n_concepts, np.nan, dtype=np.float16)
                )

                # Store metadata
                f[f'layer_{layer_idx}'].attrs['samples_per_concept'] = 1
                f[f'layer_{layer_idx}'].attrs['confidence'] = 'low'
                f[f'layer_{layer_idx}'].attrs['stage'] = 0

        elapsed = time.time() - start_time
        throughput = n_concepts / elapsed
        file_size_mb = Path(output_path).stat().st_size / 1024**2

        print(f"\n{'='*70}")
        print("✓ BOOTSTRAP COMPLETE")
        print(f"{'='*70}")
        print(f"Time elapsed:  {elapsed:.1f}s")
        print(f"Throughput:    {throughput:.1f} concepts/sec")
        print(f"File size:     {file_size_mb:.1f} MB")
        print(f"Storage/concept: {file_size_mb*1024/n_concepts:.2f} KB")
        print()
        print(f"Extrapolated times:")
        print(f"  10K concepts:  {10000/throughput:.0f}s ({10000/throughput/60:.1f} min)")
        print(f"  50K concepts:  {50000/throughput:.0f}s ({50000/throughput/60:.1f} min)")
        print(f"  100K concepts: {100000/throughput:.0f}s ({100000/throughput/60:.1f} min)")

    def bootstrap_stage1(
        self,
        encyclopedia_path: Path,
        uncertain_concepts: Optional[List[str]] = None,
        templates: Optional[List[str]] = None,
        layer_indices: List[int] = [-1]
    ):
        """
        Stage 1 refinement: Add simple context templates for uncertain concepts.

        Args:
            encyclopedia_path: Path to existing Stage 0 HDF5
            uncertain_concepts: List of concepts needing refinement (None = all)
            templates: Context templates (None = use defaults)
            layer_indices: Layers to update
        """
        if templates is None:
            templates = [
                "The concept of {concept}",
                "An example of {concept}",
                "{concept} refers to",
                "Understanding {concept}",
                "The meaning of {concept}"
            ]

        print(f"\n{'='*70}")
        print("STAGE 1 REFINEMENT")
        print(f"{'='*70}")

        with h5py.File(encyclopedia_path, 'r+') as f:
            all_concepts = f['concepts'][:].astype(str)

            if uncertain_concepts is None:
                uncertain_concepts = all_concepts

            n_refine = len(uncertain_concepts)
            print(f"Refining {n_refine} concepts with {len(templates)} templates")

            for layer_idx in layer_indices:
                print(f"\nProcessing layer {layer_idx}...")

                for concept in tqdm(uncertain_concepts, desc="Refining"):
                    # Generate contexts
                    contexts = [t.format(concept=concept) for t in templates]

                    # Capture activations for each context
                    acts = []
                    for ctx in contexts:
                        act = self.get_activation(ctx, layer_idx)
                        acts.append(act)

                    # Average activations
                    avg_act = np.mean(acts, axis=0)
                    variance = np.var(acts, axis=0).mean()

                    # Update in HDF5
                    concept_idx = np.where(all_concepts == concept)[0][0]
                    f[f'layer_{layer_idx}/activations'][concept_idx] = avg_act.astype(np.float16)
                    f[f'layer_{layer_idx}/variance'][concept_idx] = variance

                # Update metadata
                f[f'layer_{layer_idx}'].attrs['stage'] = 1
                f[f'layer_{layer_idx}'].attrs['confidence'] = 'medium'
                f[f'layer_{layer_idx}'].attrs['samples_per_concept'] = len(templates)

        print(f"✓ Stage 1 refinement complete")
