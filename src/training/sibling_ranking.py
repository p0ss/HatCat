"""
Sibling ranking refinement for lens training.

After initial binary training, this module refines lenses within sibling groups
using margin ranking loss to ensure each lens ranks highest on its own prompts.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .sumo_classifiers import extract_activations
from .sumo_data_generation import create_sumo_training_dataset


def load_lens(
    lens_path: Path,
    hidden_dim: int = 4096,
    device: str = "cuda",
) -> nn.Sequential:
    """Load a trained lens from disk."""
    state = torch.load(lens_path, map_location=device)
    # Infer input dimension from checkpoint (supports both single-layer and all-layers)
    input_dim = state["0.weight"].shape[1]
    lens = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 1)
    ).to(device)
    lens.load_state_dict(state)
    return lens


def save_lens(lens: nn.Module, lens_path: Path):
    """Save a lens to disk."""
    torch.save(lens.state_dict(), lens_path)


def load_refinement_manifest(lens_dir: Path) -> Dict:
    """Load the sibling refinement manifest for a layer."""
    manifest_path = lens_dir / "sibling_refinement.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"refined_groups": [], "version": "1.0"}


def needs_sibling_refinement(
    lens_dir: Path,
    hierarchy_dir: Path,
    layer: int,
    min_siblings: int = 2,
) -> bool:
    """
    Check if a layer needs sibling refinement.

    Returns True if:
    1. The layer directory exists with trained lenses
    2. No sibling_refinement.json exists OR it has incomplete groups
    3. OR any sibling group has expanded since last refinement

    Args:
        lens_dir: Directory containing trained lenses (e.g., lens_packs/xxx/layer2)
        hierarchy_dir: Directory containing hierarchy JSON files
        layer: Layer number
        min_siblings: Minimum siblings to form a group

    Returns:
        True if refinement is needed, False otherwise
    """
    if not lens_dir.exists():
        return False

    # Check if there are any trained lenses
    lens_files = list(lens_dir.glob("*_classifier.pt"))
    if len(lens_files) < min_siblings:
        return False  # Not enough lenses to form sibling groups

    # Load layer concepts
    layer_path = hierarchy_dir / f"layer{layer}.json"
    if not layer_path.exists():
        return False

    with open(layer_path) as f:
        layer_data = json.load(f)
    concepts = layer_data['concepts']

    # Get potential sibling groups (don't skip already refined to get full picture)
    sibling_groups = get_sibling_groups(concepts, lens_dir, min_siblings, skip_already_refined=False)

    if not sibling_groups:
        return False  # No sibling groups possible

    # Check manifest for completion
    manifest = load_refinement_manifest(lens_dir)
    refined_parents = set(manifest.get("refined_groups", []))
    group_stats = manifest.get("group_stats", {})

    # If no manifest or empty, definitely needs refinement
    if not refined_parents:
        return True

    # Check if all groups have been refined AND haven't expanded
    for parent, current_siblings in sibling_groups.items():
        if parent not in refined_parents:
            return True  # New group not yet refined

        # Check if group has expanded since last refinement
        prev_stats = group_stats.get(parent, {})
        prev_siblings = set(prev_stats.get("siblings", []))
        current_siblings_set = set(current_siblings)

        if current_siblings_set - prev_siblings:
            return True  # Group has new siblings

    return False


def get_layers_needing_refinement(
    output_dir: Path,
    hierarchy_dir: Path,
    layers: List[int],
    min_siblings: int = 2,
) -> List[int]:
    """
    Get list of layers that need sibling refinement.

    Args:
        output_dir: Base output directory containing layerN subdirectories
        hierarchy_dir: Directory containing hierarchy JSON files
        layers: List of layers to check
        min_siblings: Minimum siblings to form a group

    Returns:
        List of layer numbers that need refinement
    """
    needs_refinement = []
    for layer in layers:
        lens_dir = output_dir / f"layer{layer}"
        if needs_sibling_refinement(lens_dir, hierarchy_dir, layer, min_siblings):
            needs_refinement.append(layer)
    return needs_refinement


def save_refinement_manifest(lens_dir: Path, manifest: Dict):
    """Save the sibling refinement manifest."""
    manifest_path = lens_dir / "sibling_refinement.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def get_sibling_groups(
    concepts: List[Dict],
    lens_dir: Path,
    min_siblings: int = 2,
    skip_already_refined: bool = True,
) -> Dict[str, List[str]]:
    """
    Group concepts by their parent (siblings).

    Only returns groups where all siblings have trained lenses.
    Detects when sibling groups have expanded since last refinement.

    Args:
        concepts: List of concept dicts with 'sumo_term' and 'parent_concepts'
        lens_dir: Directory containing trained lenses
        min_siblings: Minimum siblings required to form a group
        skip_already_refined: If True, skip groups that haven't changed since refinement

    Returns:
        Dict mapping parent name to list of sibling concept names
    """
    # Load refinement manifest for resume capability
    manifest = load_refinement_manifest(lens_dir) if skip_already_refined else {"refined_groups": [], "group_stats": {}}
    refined_parents = set(manifest.get("refined_groups", []))
    group_stats = manifest.get("group_stats", {})

    # Group by parent - concepts can have multiple parents (multi-inheritance)
    # We use the first parent as the primary grouping for simplicity
    parent_to_children: Dict[str, List[str]] = {}
    for c in concepts:
        # Try new field name first, fall back to legacy field
        parents = c.get('parent_concepts', [])
        if not parents:
            # Legacy fallback
            legacy_parent = c.get('category_parent')
            if legacy_parent:
                parents = [legacy_parent]

        if parents:
            # Use first parent for primary grouping
            parent = parents[0]
            if parent not in parent_to_children:
                parent_to_children[parent] = []
            parent_to_children[parent].append(c['sumo_term'])

    # Filter to groups where all siblings have trained lenses
    valid_groups = {}
    skipped_refined = 0
    expanded_groups = 0
    for parent, children in parent_to_children.items():
        if len(children) < min_siblings:
            continue

        # Check all have lenses
        trained_siblings = []
        for child in children:
            lens_path = lens_dir / f"{child}_classifier.pt"
            if lens_path.exists():
                trained_siblings.append(child)

        if len(trained_siblings) < min_siblings:
            continue

        # Check if already refined AND sibling set unchanged
        if skip_already_refined and parent in refined_parents:
            # Get the siblings that were refined last time
            prev_stats = group_stats.get(parent, {})
            prev_siblings = set(prev_stats.get("siblings", []))
            current_siblings = set(trained_siblings)

            # Detect expanded groups (new siblings added)
            new_siblings = current_siblings - prev_siblings
            if new_siblings:
                # Group has expanded - needs re-refinement
                expanded_groups += 1
                print(f"    🔄 {parent}: {len(new_siblings)} new sibling(s) added since last refinement")
                valid_groups[parent] = trained_siblings
            else:
                # No change, skip
                skipped_refined += 1
        else:
            valid_groups[parent] = trained_siblings

    if skipped_refined > 0:
        print(f"    ⏭️  Skipped {skipped_refined} unchanged sibling groups")
    if expanded_groups > 0:
        print(f"    📈 {expanded_groups} sibling groups expanded and need re-refinement")

    return valid_groups


def refine_sibling_group(
    siblings: List[str],
    lens_dir: Path,
    concept_map: Dict[str, Dict],
    model,
    tokenizer,
    device: str = "cuda",
    n_prompts_per_sibling: int = 15,
    epochs: int = 20,
    lr: float = 0.001,
    margin: float = 1.0,
    hidden_dim: int = 4096,
    save_refined: bool = True,
) -> Dict:
    """
    Refine lenses for a sibling group using margin ranking loss.

    Each lens should rank highest on prompts specifically about its concept.

    Args:
        siblings: List of sibling concept names
        lens_dir: Directory containing trained lenses
        concept_map: Mapping of concept names to concept dicts
        model: Language model for activation extraction
        tokenizer: Tokenizer
        device: Device for computation
        n_prompts_per_sibling: Number of prompts to generate per sibling
        epochs: Training epochs
        lr: Learning rate
        margin: Margin for ranking loss
        hidden_dim: Model hidden dimension
        save_refined: Whether to save refined lenses (saves to _refined.pt suffix)

    Returns:
        Dict with refinement results
    """
    start_time = time.time()

    # Load lenses
    lenses = {}
    for sibling in siblings:
        lens_path = lens_dir / f"{sibling}_classifier.pt"
        lenses[sibling] = load_lens(lens_path, hidden_dim, device)
        lenses[sibling].train()

    # Detect all-layers mode from first lens input dimension
    first_lens = next(iter(lenses.values()))
    input_dim = first_lens[0].in_features
    all_layers_mode = input_dim > hidden_dim
    layer_idx = None if all_layers_mode else 15  # None = all layers

    # Generate prompts for each sibling
    prompts_by_sibling = {}
    for sibling in siblings:
        if sibling not in concept_map:
            continue
        concept = concept_map[sibling]
        prompts, _ = create_sumo_training_dataset(
            concept=concept,
            all_concepts=concept_map,
            negative_pool=[],  # No negatives needed
            n_positives=n_prompts_per_sibling,
            n_negatives=0,
            use_category_relationships=True,
            use_wordnet_relationships=True,
        )
        prompts_by_sibling[sibling] = prompts

    # Pre-extract activations
    activations_by_sibling = {}
    for sibling, prompts in prompts_by_sibling.items():
        acts = extract_activations(model, tokenizer, prompts, device, layer_idx=layer_idx)
        activations_by_sibling[sibling] = torch.tensor(acts, dtype=torch.float32).to(device)

    # Evaluate pre-refinement accuracy
    pre_accuracy = evaluate_sibling_ranking_accuracy(lenses, activations_by_sibling, device)

    # Set up optimizer and loss
    all_params = []
    for lens in lenses.values():
        all_params.extend(lens.parameters())
    optimizer = Adam(all_params, lr=lr)
    margin_loss = nn.MarginRankingLoss(margin=margin)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        n_pairs = 0

        optimizer.zero_grad()

        for target_sibling, acts in activations_by_sibling.items():
            target_lens = lenses[target_sibling]

            for other_sibling in siblings:
                if other_sibling == target_sibling:
                    continue
                if other_sibling not in lenses:
                    continue

                other_lens = lenses[other_sibling]

                # Compute scores
                target_scores = target_lens(acts)
                other_scores = other_lens(acts)

                # Target should rank higher
                target_indicator = torch.ones(acts.shape[0], device=device)
                loss = margin_loss(
                    target_scores.squeeze(),
                    other_scores.squeeze(),
                    target_indicator
                )

                epoch_loss += loss.item()
                n_pairs += 1
                loss.backward()

        optimizer.step()

    # Set back to eval mode
    for lens in lenses.values():
        lens.eval()

    # Evaluate post-refinement accuracy
    post_accuracy = evaluate_sibling_ranking_accuracy(lenses, activations_by_sibling, device)

    # Save refined lenses (overwrites originals)
    if save_refined:
        for sibling, lens in lenses.items():
            lens_path = lens_dir / f"{sibling}_classifier.pt"
            save_lens(lens, lens_path)

    elapsed = time.time() - start_time

    # Explicitly free GPU memory
    for sibling in list(lenses.keys()):
        del lenses[sibling]
    for sibling in list(activations_by_sibling.keys()):
        del activations_by_sibling[sibling]
    del all_params
    del optimizer

    # Force CUDA to release memory
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        'siblings': siblings,
        'pre_accuracy': pre_accuracy,
        'post_accuracy': post_accuracy,
        'improvement': post_accuracy - pre_accuracy,
        'epochs': epochs,
        'time': elapsed,
    }


def evaluate_sibling_ranking_accuracy(
    lenses: Dict[str, nn.Module],
    activations_by_sibling: Dict[str, torch.Tensor],
    device: str = "cuda"
) -> float:
    """
    Evaluate what fraction of prompts have the correct lens ranking first.
    """
    total_correct = 0
    total_prompts = 0

    siblings = list(lenses.keys())

    for target_sibling, acts in activations_by_sibling.items():
        if target_sibling not in lenses:
            continue

        for i in range(acts.shape[0]):
            act = acts[i:i+1]

            # Get scores from all lenses
            scores = {}
            with torch.no_grad():
                for sib_name, lens in lenses.items():
                    scores[sib_name] = lens(act).item()

            # Check if target wins
            winner = max(scores, key=scores.get)
            if winner == target_sibling:
                total_correct += 1
            total_prompts += 1

    return total_correct / total_prompts if total_prompts > 0 else 0


def refine_all_sibling_groups(
    layer: int,
    lens_dir: Path,
    hierarchy_dir: Path,
    model,
    tokenizer,
    device: str = "cuda",
    n_prompts_per_sibling: int = 15,
    epochs: int = 20,
    hidden_dim: int = 4096,
    min_siblings: int = 2,
) -> List[Dict]:
    """
    Refine all sibling groups in a layer.

    Args:
        layer: Layer number
        lens_dir: Directory containing trained lenses (e.g., lens_packs/xxx/layer3)
        hierarchy_dir: Directory containing hierarchy JSON files
        model: Language model
        tokenizer: Tokenizer
        device: Device
        n_prompts_per_sibling: Prompts per sibling for refinement
        epochs: Training epochs per group
        hidden_dim: Model hidden dimension
        min_siblings: Minimum siblings to form a group

    Returns:
        List of refinement results per group
    """
    # Load layer concepts
    layer_path = hierarchy_dir / f"layer{layer}.json"
    with open(layer_path) as f:
        layer_data = json.load(f)
    concepts = layer_data['concepts']
    concept_map = {c['sumo_term']: c for c in concepts}

    # Get sibling groups (skips already-refined from manifest)
    sibling_groups = get_sibling_groups(concepts, lens_dir, min_siblings)

    # Load manifest for updating after each group
    manifest = load_refinement_manifest(lens_dir)

    print(f"\n{'='*70}")
    print(f"SIBLING RANKING REFINEMENT - Layer {layer}")
    print(f"{'='*70}")
    print(f"Found {len(sibling_groups)} sibling groups to refine")
    print()

    results = []
    for i, (parent, siblings) in enumerate(sibling_groups.items()):
        print(f"[{i+1}/{len(sibling_groups)}] Refining {parent} ({len(siblings)} siblings)")

        result = refine_sibling_group(
            siblings=siblings,
            lens_dir=lens_dir,
            concept_map=concept_map,
            model=model,
            tokenizer=tokenizer,
            device=device,
            n_prompts_per_sibling=n_prompts_per_sibling,
            epochs=epochs,
            hidden_dim=hidden_dim,
        )

        print(f"    Accuracy: {result['pre_accuracy']:.1%} → {result['post_accuracy']:.1%} "
              f"({result['improvement']:+.1%}) [{result['time']:.1f}s]")

        results.append(result)

        # Update manifest after each group (crash-safe resume)
        manifest["refined_groups"].append(parent)
        if "group_stats" not in manifest:
            manifest["group_stats"] = {}
        manifest["group_stats"][parent] = {
            "siblings": siblings,
            "pre_accuracy": result['pre_accuracy'],
            "post_accuracy": result['post_accuracy'],
            "improvement": result['improvement'],
            "epochs": result['epochs'],
            "time": result['time'],
        }
        save_refinement_manifest(lens_dir, manifest)

    # Summary
    if results:
        avg_pre = np.mean([r['pre_accuracy'] for r in results])
        avg_post = np.mean([r['post_accuracy'] for r in results])
        avg_improvement = np.mean([r['improvement'] for r in results])

        print(f"\n{'='*70}")
        print(f"SUMMARY - Layer {layer}")
        print(f"{'='*70}")
        print(f"Groups refined: {len(results)}")
        print(f"Avg accuracy: {avg_pre:.1%} → {avg_post:.1%} ({avg_improvement:+.1%})")

    return results


__all__ = [
    'load_lens',
    'save_lens',
    'get_sibling_groups',
    'refine_sibling_group',
    'refine_all_sibling_groups',
    'evaluate_sibling_ranking_accuracy',
    'needs_sibling_refinement',
    'get_layers_needing_refinement',
]
