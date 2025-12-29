#!/usr/bin/env python3
"""
Cross-Activation Calibration

Measures per-concept noise floors using DynamicLensManager's hierarchical loading.
For each concept:
- self_mean: Average activation on its OWN prompts
- cross_mean: Average activation on OTHER concepts' prompts where it was loaded and fired

This produces calibration data enabling normalized scores at inference:
- 1.0 = firing at self_mean level (genuine signal)
- 0.5 = firing at cross_mean level (noise floor for this concept)
- 0.0 = floor

Uses hierarchical loading (same as production) so only tests cross-activation
for lenses that would actually be loaded during normal operation.
"""

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from .activation_cache import ActivationCache


def run_cross_activation_calibration(
    lens_pack_dir: Path,
    concept_pack_dir: Path,
    model,
    tokenizer,
    device: str,
    layers: List[int],
    n_samples_per_concept: int = 3,
    firing_threshold: float = 0.5,
    layer_idx: int = 15,
    max_concepts: Optional[int] = None,
    activation_cache: Optional["ActivationCache"] = None,
) -> Dict:
    """
    Measure cross-activation using DynamicLensManager's hierarchical loading.

    For each concept's prompt:
    1. Run DynamicLensManager.detect_and_expand()
    2. Get scores for ALL loaded lenses (not just top-k)
    3. Record:
       - self_score: target concept's score on its own prompt
       - cross_scores: other concepts' scores when loaded during this prompt

    Returns dict with format:
    {
        "timestamp": "...",
        "calibration": {
            "ConceptName_L0": {
                "concept": "ConceptName",
                "layer": 0,
                "self_mean": 0.92,
                "self_std": 0.05,
                "cross_mean": 0.31,
                "cross_std": 0.15,
                "cross_fire_count": 847,
                "times_loaded": 1200,
                "cross_fire_rate": 0.71,
            },
            ...
        }
    }
    """
    from src.hat.monitoring.lens_manager import DynamicLensManager

    print(f"\n{'='*60}")
    print("CROSS-ACTIVATION CALIBRATION")
    print(f"{'='*60}")
    print(f"  Lens pack: {lens_pack_dir}")
    print(f"  Using DynamicLensManager (hierarchical loading)")
    print(f"  Samples per concept: {n_samples_per_concept}")
    print(f"  Firing threshold: {firing_threshold}")
    print(f"  Using activation cache: {activation_cache is not None}")

    # Initialize DynamicLensManager
    print(f"\nInitializing DynamicLensManager...")
    manager = DynamicLensManager(
        lenses_dir=lens_pack_dir,
        layers_data_dir=concept_pack_dir / "hierarchy",
        base_layers=layers[:3] if len(layers) >= 3 else layers,
        device=device,
        max_loaded_lenses=1000,
        normalize_hidden_states=True,
    )
    print(f"  Base lenses loaded: {len(manager.cache.loaded_lenses)}")

    # Load concept definitions
    print(f"\nLoading concept definitions...")
    definitions = {}
    all_concepts = {}  # concept -> layer

    for layer in layers:
        layer_file = concept_pack_dir / "hierarchy" / f"layer{layer}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        for concept in layer_data.get('concepts', []):
            term = concept.get('sumo_term') or concept.get('term')
            if not term:
                continue

            all_concepts[term] = layer

            # Collect definitions
            defs = []
            definition = concept.get('definition', '')
            if definition and len(definition) > 10:
                defs.append(definition)
            sumo_def = concept.get('sumo_definition', '')
            if sumo_def and len(sumo_def) > 10 and sumo_def != definition:
                defs.append(sumo_def)
            for lemma in concept.get('lemmas', [])[:2]:
                if lemma and len(lemma) > 3:
                    defs.append(f"A type of {lemma}")

            if defs:
                definitions[(term, layer)] = defs[:n_samples_per_concept]

    print(f"  Loaded definitions for {len(definitions)} concepts")

    # Filter to concepts that have lenses
    concepts_to_test = []
    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if not layer_dir.exists():
            continue
        for lens_file in layer_dir.glob("*.pt"):
            if lens_file.name == "results.json":
                continue
            concept_name = lens_file.stem
            # Handle both naming conventions: {concept}.pt and {concept}_classifier.pt
            if concept_name.endswith("_classifier"):
                concept_name = concept_name[:-11]  # Remove "_classifier" suffix
            if (concept_name, layer) in definitions:
                concepts_to_test.append((concept_name, layer))

    print(f"  Concepts with lenses: {len(concepts_to_test)}")

    if max_concepts and len(concepts_to_test) > max_concepts:
        random.shuffle(concepts_to_test)
        concepts_to_test = concepts_to_test[:max_concepts]
        print(f"  Limited to {max_concepts} concepts")

    # Tracking structures
    self_scores = defaultdict(list)   # concept_key -> [scores on own prompts]
    cross_scores = defaultdict(list)  # concept_key -> [scores on other prompts]
    times_loaded = defaultdict(int)   # concept_key -> how many times loaded

    # Run calibration
    print(f"\nRunning cross-activation measurement...")
    model.eval()

    for target_name, target_layer in tqdm(concepts_to_test, desc="Concepts"):
        target_key = (target_name, target_layer)

        # Get activations from cache or compute
        if activation_cache and activation_cache.has(target_name, target_layer):
            cached_acts = activation_cache.get(target_name, target_layer)
            activations = [act.to(device) for act in cached_acts]
        else:
            # Compute activations from prompts
            prompts = definitions.get(target_key, [target_name])
            activations = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[layer_idx]
                    activation = hidden_states[0, -1, :].float()
                    activations.append(activation)

        for activation in activations:
            with torch.no_grad():
                # Run DynamicLensManager - this loads relevant lenses hierarchically
                detected, _ = manager.detect_and_expand(
                    activation,
                    top_k=20,
                    use_calibration=False,  # Don't use calibration during calibration!
                )

            # Ensure target lens is loaded (it may not be if not in base layers or expansion)
            if target_key not in manager.cache.loaded_lenses:
                manager._load_concepts([target_key], reason="calibration_target")

            # Get scores for ALL loaded lenses
            scores = {}
            for concept_key, lens in manager.cache.loaded_lenses.items():
                with torch.no_grad():
                    # Need to normalize activation like the manager does
                    act = activation.unsqueeze(0)
                    if manager._layer_norm is not None:
                        act = manager._layer_norm(act)
                    lens_dtype = next(lens.parameters()).dtype
                    act = act.to(dtype=lens_dtype)
                    score = lens(act).item()
                    scores[concept_key] = score

            # Record scores
            for concept_key, score in scores.items():
                times_loaded[concept_key] += 1

                if concept_key == target_key:
                    # Self-activation
                    self_scores[concept_key].append(score)
                else:
                    # Cross-activation
                    if score >= firing_threshold:
                        cross_scores[concept_key].append(score)

    # Compute calibration stats
    print(f"\nComputing calibration statistics...")
    calibration = {}

    for concept_key in tqdm(concepts_to_test, desc="Computing stats"):
        concept_name, layer = concept_key
        key_str = f"{concept_name}_L{layer}"

        self_vals = self_scores.get(concept_key, [])
        cross_vals = cross_scores.get(concept_key, [])
        n_loaded = times_loaded.get(concept_key, 0)

        if not self_vals:
            continue

        self_mean = float(np.mean(self_vals))
        self_std = float(np.std(self_vals)) if len(self_vals) > 1 else 0.0

        if cross_vals:
            cross_mean = float(np.mean(cross_vals))
            cross_std = float(np.std(cross_vals)) if len(cross_vals) > 1 else 0.0
        else:
            cross_mean = 0.0
            cross_std = 0.0

        # Cross-fire rate = times it fired on others / times it was loaded for others
        n_cross_opportunities = n_loaded - len(self_vals)
        cross_fire_rate = len(cross_vals) / max(1, n_cross_opportunities)

        calibration[key_str] = {
            "concept": concept_name,
            "layer": layer,
            "self_mean": self_mean,
            "self_std": self_std,
            "cross_mean": cross_mean,
            "cross_std": cross_std,
            "cross_fire_count": len(cross_vals),
            "times_loaded": n_loaded,
            "n_self_samples": len(self_vals),
            "n_cross_samples": len(cross_vals),
            "cross_fire_rate": cross_fire_rate,
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lens_pack": lens_pack_dir.name,
        "n_samples_per_concept": n_samples_per_concept,
        "firing_threshold": firing_threshold,
        "total_concepts_calibrated": len(calibration),
        "calibration": calibration,
    }
