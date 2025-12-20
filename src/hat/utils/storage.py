"""
HDF5 storage utilities for activation patterns.
Handles compression, sparse storage, and hierarchical organization.
"""

import h5py
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


class ActivationStorage:
    """
    Manages HDF5 storage for activation patterns.
    Supports hierarchical organization and compression.
    """

    def __init__(self, filepath: Path, mode: str = 'a'):
        """
        Initialize activation storage.

        Args:
            filepath: Path to HDF5 file
            mode: File mode ('r', 'w', 'a')
        """
        self.filepath = Path(filepath)
        self.mode = mode
        self.file: Optional[h5py.File] = None

    def __enter__(self):
        """Open HDF5 file."""
        self.file = h5py.File(self.filepath, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HDF5 file."""
        if self.file is not None:
            self.file.close()

    def store_concept_activations(
        self,
        concept_name: str,
        layer_activations: Dict[str, np.ndarray],
        metadata: Optional[Dict] = None,
        compression: str = 'gzip'
    ):
        """
        Store activation patterns for a concept.

        Args:
            concept_name: Name of the concept
            layer_activations: Dictionary of {layer_name: activation_array}
            metadata: Optional metadata (prompts, contexts, etc.)
            compression: HDF5 compression method
        """
        # Create concept group
        if concept_name in self.file:
            concept_group = self.file[concept_name]
        else:
            concept_group = self.file.create_group(concept_name)

        # Store metadata
        if metadata is not None:
            concept_group.attrs['metadata'] = json.dumps(metadata)

        # Store activations for each layer
        for layer_name, activation in layer_activations.items():
            # Sanitize layer name for HDF5
            safe_layer_name = layer_name.replace('.', '_')

            if safe_layer_name in concept_group:
                del concept_group[safe_layer_name]

            # Store with compression
            concept_group.create_dataset(
                safe_layer_name,
                data=activation,
                compression=compression,
                compression_opts=9 if compression == 'gzip' else None
            )

    def store_baseline(
        self,
        layer_activations: Dict[str, np.ndarray],
        compression: str = 'gzip'
    ):
        """
        Store baseline activations.

        Args:
            layer_activations: Dictionary of {layer_name: activation_array}
            compression: HDF5 compression method
        """
        # Create or get baseline group
        if 'baseline' in self.file:
            baseline_group = self.file['baseline']
        else:
            baseline_group = self.file.create_group('baseline')

        # Store activations
        for layer_name, activation in layer_activations.items():
            safe_layer_name = layer_name.replace('.', '_')

            if safe_layer_name in baseline_group:
                del baseline_group[safe_layer_name]

            baseline_group.create_dataset(
                safe_layer_name,
                data=activation,
                compression=compression,
                compression_opts=9 if compression == 'gzip' else None
            )

    def load_concept_activations(
        self,
        concept_name: str
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict]]:
        """
        Load activation patterns for a concept.

        Args:
            concept_name: Name of the concept

        Returns:
            Tuple of (layer_activations, metadata)
        """
        if concept_name not in self.file:
            raise KeyError(f"Concept '{concept_name}' not found in storage")

        concept_group = self.file[concept_name]

        # Load metadata
        metadata = None
        if 'metadata' in concept_group.attrs:
            metadata = json.loads(concept_group.attrs['metadata'])

        # Load activations
        layer_activations = {}
        for layer_name in concept_group.keys():
            # Restore original layer name
            original_name = layer_name.replace('_', '.')
            layer_activations[original_name] = concept_group[layer_name][:]

        return layer_activations, metadata

    def load_baseline(self) -> Dict[str, np.ndarray]:
        """
        Load baseline activations.

        Returns:
            Dictionary of {layer_name: activation_array}
        """
        if 'baseline' not in self.file:
            raise KeyError("Baseline not found in storage")

        baseline_group = self.file['baseline']

        # Load activations
        layer_activations = {}
        for layer_name in baseline_group.keys():
            original_name = layer_name.replace('_', '.')
            layer_activations[original_name] = baseline_group[layer_name][:]

        return layer_activations

    def list_concepts(self) -> List[str]:
        """
        List all stored concepts.

        Returns:
            List of concept names
        """
        concepts = []
        for key in self.file.keys():
            if key != 'baseline':
                concepts.append(key)
        return concepts

    def get_storage_stats(self) -> Dict:
        """
        Get statistics about stored data.

        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'num_concepts': len(self.list_concepts()),
            'has_baseline': 'baseline' in self.file,
            'file_size_mb': self.filepath.stat().st_size / (1024 * 1024),
            'concepts': {}
        }

        for concept in self.list_concepts():
            concept_group = self.file[concept]
            stats['concepts'][concept] = {
                'num_layers': len(concept_group.keys()),
                'layers': list(concept_group.keys())
            }

        return stats


class SparseActivationStorage:
    """
    Specialized storage for sparse activation patterns.
    Stores indices and values separately for better compression.
    """

    def __init__(self, filepath: Path, mode: str = 'a'):
        self.filepath = Path(filepath)
        self.mode = mode
        self.file: Optional[h5py.File] = None

    def __enter__(self):
        self.file = h5py.File(self.filepath, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()

    def store_sparse_activations(
        self,
        concept_name: str,
        layer_name: str,
        indices: np.ndarray,
        values: np.ndarray,
        shape: Tuple[int, ...],
        metadata: Optional[Dict] = None
    ):
        """
        Store sparse activations as indices + values.

        Args:
            concept_name: Name of concept
            layer_name: Name of layer
            indices: Indices of non-zero activations
            values: Values of non-zero activations
            shape: Original shape of activation tensor
            metadata: Optional metadata
        """
        # Create concept group
        if concept_name not in self.file:
            concept_group = self.file.create_group(concept_name)
        else:
            concept_group = self.file[concept_name]

        # Create layer subgroup
        safe_layer_name = layer_name.replace('.', '_')
        if safe_layer_name in concept_group:
            del concept_group[safe_layer_name]

        layer_group = concept_group.create_group(safe_layer_name)

        # Store indices and values
        layer_group.create_dataset('indices', data=indices, compression='gzip')
        layer_group.create_dataset('values', data=values, compression='gzip')
        layer_group.attrs['shape'] = shape

        if metadata is not None:
            layer_group.attrs['metadata'] = json.dumps(metadata)

    def load_sparse_activations(
        self,
        concept_name: str,
        layer_name: str,
        as_dense: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        """
        Load sparse activations.

        Args:
            concept_name: Name of concept
            layer_name: Name of layer
            as_dense: Whether to return dense array

        Returns:
            If as_dense: dense array
            Otherwise: (indices, values, shape)
        """
        concept_group = self.file[concept_name]
        safe_layer_name = layer_name.replace('.', '_')
        layer_group = concept_group[safe_layer_name]

        indices = layer_group['indices'][:]
        values = layer_group['values'][:]
        shape = tuple(layer_group.attrs['shape'])

        if as_dense:
            # Reconstruct dense array
            dense = np.zeros(shape, dtype=values.dtype)
            dense.flat[indices] = values
            return dense

        return indices, values, shape
