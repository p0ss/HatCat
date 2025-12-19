#!/usr/bin/env python3
"""
Pack-Level Calibration Fine-Tuning

Uses the analysis from calibration.analysis to fine-tune lenses:
1. Boost under-firing lenses on their target prompts
2. Suppress over-firing lenses on prompts they shouldn't match

Usage:
    # Run fine-tuning based on analysis
    python -m calibration.finetune \
        --lens-pack lens_packs/apertus-8b_first-light \
        --concept-pack concept_packs/first-light \
        --model swiss-ai/Apertus-8B-2509 \
        --analysis lens_packs/apertus-8b_first-light/calibration_analysis.json

    # Limit iterations
    python -m calibration.finetune \
        --lens-pack lens_packs/apertus-8b_first-light \
        --concept-pack concept_packs/first-light \
        --model swiss-ai/Apertus-8B-2509 \
        --analysis lens_packs/apertus-8b_first-light/calibration_analysis.json \
        --max-finetune-epochs 10

    # Just process under-firing (skip over-firing suppression)
    python -m calibration.finetune \
        --lens-pack lens_packs/apertus-8b_first-light \
        --concept-pack concept_packs/first-light \
        --model swiss-ai/Apertus-8B-2509 \
        --analysis lens_packs/apertus-8b_first-light/calibration_analysis.json \
        --boost-only
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Module is now in src/calibration/


@dataclass
class FineTuneResult:
    """Result of fine-tuning a single lens."""
    concept: str
    layer: int
    action: str  # 'boost' or 'suppress'
    before_in_top_k_rate: float
    after_in_top_k_rate: float
    before_avg_rank: float
    after_avg_rank: float
    epochs_used: int
    improved: bool


@dataclass
class FineTuneReport:
    """Full fine-tuning report."""
    lens_pack_id: str
    analysis_timestamp: str
    finetune_timestamp: str
    total_lenses_processed: int
    lenses_boosted: int
    lenses_suppressed: int
    avg_improvement: float
    results: List[FineTuneResult]


def load_analysis(analysis_path: Path) -> Dict:
    """Load calibration analysis results."""
    with open(analysis_path) as f:
        return json.load(f)


def load_concept_prompts(concept_pack_dir: Path, layers: List[int]) -> Dict[str, Dict]:
    """Load concepts and generate probe prompts."""
    concepts = {}

    for layer in layers:
        layer_file = concept_pack_dir / "hierarchy" / f"layer{layer}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        concept_list = layer_data.get('concepts', layer_data if isinstance(layer_data, list) else [])

        for concept in concept_list:
            term = concept.get('sumo_term') or concept.get('term')
            if not term:
                continue

            prompts = []
            training_hints = concept.get('training_hints', {})
            positive_examples = training_hints.get('positive_examples', [])
            negative_examples = training_hints.get('negative_examples', [])

            if positive_examples:
                prompts.extend(positive_examples[:5])

            definition = concept.get('definition', '')
            if definition and len(prompts) < 3:
                prompts.append(f"Explain the concept of {term}: {definition}")

            if not prompts:
                prompts.append(f"Tell me about {term}.")

            concepts[term] = {
                'layer': layer,
                'definition': definition,
                'positive_prompts': prompts,
                'negative_prompts': negative_examples[:5] if negative_examples else [],
                'domain': concept.get('domain', 'Unknown'),
            }

    return concepts


def load_hierarchy(concept_pack_dir: Path) -> Dict[str, str]:
    """Load child_to_parent hierarchy mapping."""
    hierarchy_path = concept_pack_dir / "hierarchy.json"
    if not hierarchy_path.exists():
        return {}
    with open(hierarchy_path) as f:
        data = json.load(f)
    return data.get('child_to_parent', {})


def get_ancestors(concept: str, hierarchy: Dict[str, str], max_depth: int = 10) -> Set[str]:
    """Get all ancestors of a concept up to max_depth."""
    ancestors = set()
    current = concept
    for _ in range(max_depth):
        parent = hierarchy.get(current)
        if not parent or parent == current:
            break
        ancestors.add(parent)
        current = parent
    return ancestors


def find_graph_distant_concepts(
    target_concept: str,
    all_concepts: List[str],
    hierarchy: Dict[str, str],
    n_distant: int = 5,
    min_distance: int = 3,
) -> List[str]:
    """
    Find concepts that are graph-distant from the target.

    Graph distance = no shared ancestors within min_distance levels.
    """
    target_ancestors = get_ancestors(target_concept, hierarchy, max_depth=min_distance)
    target_ancestors.add(target_concept)

    distant = []
    for concept in all_concepts:
        if concept == target_concept:
            continue
        concept_ancestors = get_ancestors(concept, hierarchy, max_depth=min_distance)
        concept_ancestors.add(concept)

        # Check if any overlap in ancestry
        if not target_ancestors.intersection(concept_ancestors):
            distant.append(concept)
            if len(distant) >= n_distant:
                break

    return distant


def extract_activation(
    model,
    tokenizer,
    prompt: str,
    device: str,
    layer_idx: int = 15,
    fast_mode: bool = True,
) -> np.ndarray:
    """Extract hidden state activation from model."""
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer_idx]
        # Convert to float32 before numpy (bfloat16 not supported by numpy)
        activation = hidden_states[0, -1, :].float().cpu().numpy()

    return activation


def finetune_lens_boost(
    lens: nn.Module,
    target_activations: List[np.ndarray],
    other_activations: List[np.ndarray],
    device: str,
    epochs: int = 20,
    lr: float = 0.0001,
    layer_norm: nn.Module = None,
) -> Tuple[nn.Module, Dict]:
    """
    Fine-tune a lens to boost activation on target prompts.

    Uses contrastive approach: increase score on target, decrease on others.
    """
    lens.train()
    optimizer = optim.Adam(lens.parameters(), lr=lr)

    # Prepare data
    target_tensors = [torch.tensor(a, dtype=torch.float32).to(device) for a in target_activations]
    other_tensors = [torch.tensor(a, dtype=torch.float32).to(device) for a in other_activations]

    if layer_norm:
        target_tensors = [layer_norm(t.unsqueeze(0)).squeeze(0) for t in target_tensors]
        other_tensors = [layer_norm(t.unsqueeze(0)).squeeze(0) for t in other_tensors]

    initial_target_scores = []
    with torch.no_grad():
        for t in target_tensors:
            score = lens(t.unsqueeze(0)).item()
            initial_target_scores.append(score)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        for target in target_tensors:
            optimizer.zero_grad()

            # Push target score up toward 1.0
            target_score = lens(target.unsqueeze(0))
            target_loss = -torch.log(target_score + 1e-8)  # BCE-like loss pushing toward 1

            # Pull a random other score down
            if other_tensors:
                other_idx = np.random.randint(len(other_tensors))
                other_score = lens(other_tensors[other_idx].unsqueeze(0))
                other_loss = -torch.log(1 - other_score + 1e-8)  # BCE-like loss pushing toward 0
                loss = target_loss + 0.5 * other_loss  # Weight target more
            else:
                loss = target_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Measure improvement
    lens.eval()
    final_target_scores = []
    with torch.no_grad():
        for t in target_tensors:
            score = lens(t.unsqueeze(0)).item()
            final_target_scores.append(score)

    return lens, {
        'initial_avg_score': np.mean(initial_target_scores),
        'final_avg_score': np.mean(final_target_scores),
        'epochs': epochs,
    }


def dampen_overfirer(lens: nn.Module, dampen_factor: float = 0.5, bias_shift: float = -1.0) -> nn.Module:
    """
    Directly scale down output layer weights for chronic over-firers.

    This is a fast, aggressive fix for lenses that fire on thousands of concepts.
    Rather than slowly training on negative examples, we just turn down the gain.

    Args:
        lens: The lens module to dampen
        dampen_factor: Multiply output weights by this factor (0.1 = 10% of original)
        bias_shift: Add this to the output bias (negative values reduce firing)
                   In logit space: -1.0 shifts sigmoid from 0.5 to ~0.27
                                   -3.0 shifts sigmoid from 0.5 to ~0.05
    """
    with torch.no_grad():
        for name, param in lens.named_parameters():
            # Target the output layer (shape [1, hidden] for binary classifier)
            if 'weight' in name and param.dim() == 2 and param.shape[0] == 1:
                param.mul_(dampen_factor)
            # Also shift the bias if present
            if 'bias' in name and param.shape[0] == 1:
                # Shift bias down (in logit space, this lowers the baseline probability)
                param.add_(bias_shift)
    return lens


def finetune_lens_suppress(
    lens: nn.Module,
    suppress_activations: List[np.ndarray],
    preserve_activations: List[np.ndarray],
    device: str,
    epochs: int = 10,
    lr: float = 0.0001,
    layer_norm: nn.Module = None,
) -> Tuple[nn.Module, Dict]:
    """
    Fine-tune a lens to suppress activation on non-target prompts.

    Decreases score on suppress_activations while preserving score on preserve_activations.
    """
    lens.train()
    optimizer = optim.Adam(lens.parameters(), lr=lr)

    # Prepare data
    suppress_tensors = [torch.tensor(a, dtype=torch.float32).to(device) for a in suppress_activations]
    preserve_tensors = [torch.tensor(a, dtype=torch.float32).to(device) for a in preserve_activations]

    if layer_norm:
        suppress_tensors = [layer_norm(t.unsqueeze(0)).squeeze(0) for t in suppress_tensors]
        preserve_tensors = [layer_norm(t.unsqueeze(0)).squeeze(0) for t in preserve_tensors]

    initial_suppress_scores = []
    initial_preserve_scores = []
    with torch.no_grad():
        for t in suppress_tensors:
            initial_suppress_scores.append(lens(t.unsqueeze(0)).item())
        for t in preserve_tensors:
            initial_preserve_scores.append(lens(t.unsqueeze(0)).item())

    # Training loop
    for epoch in range(epochs):
        for suppress in suppress_tensors:
            optimizer.zero_grad()

            # Push suppress score down
            suppress_score = lens(suppress.unsqueeze(0))
            suppress_loss = -torch.log(1 - suppress_score + 1e-8)

            # Keep preserve score stable (use MSE to anchor)
            preserve_loss = 0
            if preserve_tensors:
                preserve_idx = np.random.randint(len(preserve_tensors))
                preserve_score = lens(preserve_tensors[preserve_idx].unsqueeze(0))
                # Anchor to initial score (don't let it drift too much)
                target_score = torch.tensor([initial_preserve_scores[preserve_idx]], device=device)
                preserve_loss = 0.5 * (preserve_score - target_score) ** 2

            loss = suppress_loss + preserve_loss
            loss.backward()
            optimizer.step()

    # Measure results
    lens.eval()
    final_suppress_scores = []
    final_preserve_scores = []
    with torch.no_grad():
        for t in suppress_tensors:
            final_suppress_scores.append(lens(t.unsqueeze(0)).item())
        for t in preserve_tensors:
            final_preserve_scores.append(lens(t.unsqueeze(0)).item())

    return lens, {
        'initial_suppress_avg': np.mean(initial_suppress_scores),
        'final_suppress_avg': np.mean(final_suppress_scores),
        'initial_preserve_avg': np.mean(initial_preserve_scores) if initial_preserve_scores else 0,
        'final_preserve_avg': np.mean(final_preserve_scores) if final_preserve_scores else 0,
        'epochs': epochs,
    }


def finetune_contrastive(
    target_lens: nn.Module,
    competitor_lenses: Dict[str, nn.Module],
    activation: torch.Tensor,
    device: str,
    epochs: int = 30,
    lr: float = 0.001,
    margin: float = 0.05,
) -> Tuple[nn.Module, Dict[str, nn.Module], Dict]:
    """
    Contrastive fine-tuning: boost target while suppressing competitors.

    This is the key fix for the over-firing problem. When a concept X doesn't
    make top-k, we identify which lenses incorrectly beat it and suppress them
    on this specific activation while boosting X.

    Args:
        target_lens: The lens we want to win
        competitor_lenses: Dict of {name: lens} for lenses that incorrectly beat target
        activation: The activation tensor (normalized)
        device: Compute device
        epochs: Training epochs
        lr: Learning rate
        margin: Target must beat competitors by this margin
    """
    target_lens.train()
    for lens in competitor_lenses.values():
        lens.train()

    # Optimizers
    target_optimizer = optim.Adam(target_lens.parameters(), lr=lr)
    competitor_optimizers = {
        name: optim.Adam(lens.parameters(), lr=lr * 0.3)  # Lower LR for competitors
        for name, lens in competitor_lenses.items()
    }

    # Track initial scores
    with torch.no_grad():
        initial_target = target_lens(activation.unsqueeze(0)).item()
        initial_competitors = {
            name: lens(activation.unsqueeze(0)).item()
            for name, lens in competitor_lenses.items()
        }

    for epoch in range(epochs):
        # Step 1: Boost target
        target_optimizer.zero_grad()
        target_score = target_lens(activation.unsqueeze(0))

        # Target loss: maximize score + beat all competitors
        target_loss = -torch.log(target_score + 1e-8)

        # Add margin loss for each competitor we need to beat
        with torch.no_grad():
            competitor_scores = [lens(activation.unsqueeze(0)).item() for lens in competitor_lenses.values()]
            max_competitor = max(competitor_scores) if competitor_scores else 0

        if target_score.item() < max_competitor + margin:
            margin_loss = (max_competitor + margin - target_score) ** 2
            target_loss = target_loss + 2.0 * margin_loss  # Strong penalty for not beating

        target_loss.backward()
        target_optimizer.step()

        # Step 2: Suppress each competitor
        current_target = target_lens(activation.unsqueeze(0)).detach().item()

        for name, competitor_lens in competitor_lenses.items():
            competitor_score = competitor_lens(activation.unsqueeze(0))

            # Suppress competitor if it's anywhere near the target
            # Key insight: if both are at 1.0 (saturated), we need to push competitor DOWN hard
            if competitor_score.item() > current_target - 0.2:  # More aggressive threshold
                competitor_optimizers[name].zero_grad()
                # Push competitor significantly below target
                # If saturated at 1.0, push down to 0.5 or lower
                suppress_target = max(0.1, current_target - 0.3)  # More aggressive suppression
                suppress_loss = (competitor_score - suppress_target) ** 2
                suppress_loss.backward()
                competitor_optimizers[name].step()

    # Final scores
    target_lens.eval()
    for lens in competitor_lenses.values():
        lens.eval()

    with torch.no_grad():
        final_target = target_lens(activation.unsqueeze(0)).item()
        final_competitors = {
            name: lens(activation.unsqueeze(0)).item()
            for name, lens in competitor_lenses.items()
        }

    beats_all = all(final_target > v + margin for v in final_competitors.values())

    return target_lens, competitor_lenses, {
        'initial_target': initial_target,
        'final_target': final_target,
        'initial_competitors': initial_competitors,
        'final_competitors': final_competitors,
        'epochs': epochs,
        'target_beats_all': beats_all,
        'improvement': final_target - initial_target,
    }


def finetune_ancestor_competition(
    target_lens: nn.Module,
    ancestor_lenses: Dict[str, nn.Module],
    activation: torch.Tensor,
    device: str,
    epochs: int = 20,
    lr: float = 0.0005,
    margin: float = 0.1,
) -> Tuple[nn.Module, Dict[str, nn.Module], Dict]:
    """
    Fine-tune to make target lens beat ancestor lenses on the given activation.

    The goal: target_score > ancestor_score + margin for all ancestors.

    Strategy:
    1. Boost target lens score
    2. Suppress ancestor lens scores (gently, to preserve their function elsewhere)
    """
    target_lens.train()
    for lens in ancestor_lenses.values():
        lens.train()

    # Optimizers
    target_optimizer = optim.Adam(target_lens.parameters(), lr=lr)
    ancestor_optimizers = {
        name: optim.Adam(lens.parameters(), lr=lr * 0.5)  # Lower LR for ancestors
        for name, lens in ancestor_lenses.items()
    }

    # Track initial scores
    with torch.no_grad():
        initial_target = target_lens(activation.unsqueeze(0)).item()
        initial_ancestors = {
            name: lens(activation.unsqueeze(0)).item()
            for name, lens in ancestor_lenses.items()
        }

    for epoch in range(epochs):
        # Step 1: Update target lens to boost its score
        target_optimizer.zero_grad()
        target_score = target_lens(activation.unsqueeze(0))

        # Get current ancestor scores (detached for target update)
        with torch.no_grad():
            ancestor_score_values = [lens(activation.unsqueeze(0)).item() for lens in ancestor_lenses.values()]
            max_ancestor_val = max(ancestor_score_values) if ancestor_score_values else 0

        # Target loss: push toward 1.0 and beat ancestors
        target_loss = -torch.log(target_score + 1e-8)
        if target_score.item() < max_ancestor_val + margin:
            margin_loss = (max_ancestor_val + margin - target_score) ** 2
            target_loss = target_loss + margin_loss

        target_loss.backward()
        target_optimizer.step()

        # Step 2: Update each ancestor lens separately
        current_target = target_lens(activation.unsqueeze(0)).detach().item()

        for name, ancestor_lens in ancestor_lenses.items():
            ancestor_optimizers[name].zero_grad()
            ancestor_score = ancestor_lens(activation.unsqueeze(0))

            # Only suppress if ancestor is too close to or above target
            if ancestor_score.item() > current_target - margin:
                suppress_loss = torch.relu(ancestor_score - current_target + margin)
                suppress_loss.backward()
                ancestor_optimizers[name].step()

    # Get final scores
    target_lens.eval()
    for lens in ancestor_lenses.values():
        lens.eval()

    with torch.no_grad():
        final_target = target_lens(activation.unsqueeze(0)).item()
        final_ancestors = {
            name: lens(activation.unsqueeze(0)).item()
            for name, lens in ancestor_lenses.items()
        }

    return target_lens, ancestor_lenses, {
        'initial_target': initial_target,
        'final_target': final_target,
        'initial_ancestors': initial_ancestors,
        'final_ancestors': final_ancestors,
        'epochs': epochs,
        'target_beats_all': all(final_target > v for v in final_ancestors.values()),
    }


def run_triple_criteria_finetune(
    lens_pack_dir: Path,
    concept_pack_dir: Path,
    analysis: Dict,
    model,
    tokenizer,
    device: str,
    layers: List[int],
    max_epochs: int = 20,
    layer_idx: int = 15,
) -> FineTuneReport:
    """
    Run fine-tuning for triple-criteria calibration results.

    Handles all three criteria:
    1. Ancestor criterion failures: boost target to beat ancestors
    2. Random criterion failures: boost target's absolute activation
    3. Sibling criterion failures: boost target to beat siblings
    """
    from src.monitoring.dynamic_lens_manager import SimpleMLP

    print(f"\n{'='*80}")
    print("TRIPLE-CRITERIA CALIBRATION FINE-TUNING")
    print(f"{'='*80}")
    print(f"  Lens pack: {lens_pack_dir}")
    print(f"  Analysis mode: {analysis.get('mode', 'unknown')}")

    # Get concepts needing fixes
    lens_reports = analysis.get('lens_reports', {})

    needs_ancestor_fix = [
        (name, report) for name, report in lens_reports.items()
        if not report.get('passes_ancestor_criterion', True)
        and report.get('failed_ancestors', [])
    ]

    needs_random_fix = [
        (name, report) for name, report in lens_reports.items()
        if not report.get('passes_random_criterion', True)
    ]

    needs_sibling_fix = [
        (name, report) for name, report in lens_reports.items()
        if not report.get('passes_sibling_criterion', True)
        and report.get('failed_siblings', [])
    ]

    print(f"  Concepts failing ancestor criterion: {len(needs_ancestor_fix)}")
    print(f"  Concepts failing random criterion: {len(needs_random_fix)}")
    print(f"  Concepts failing sibling criterion: {len(needs_sibling_fix)}")

    # Detect hidden dim
    hidden_dim = None
    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if layer_dir.exists():
            lens_files = list(layer_dir.glob("*_classifier.pt"))
            if lens_files:
                state_dict = torch.load(lens_files[0], map_location='cpu')
                first_key = list(state_dict.keys())[0]
                hidden_dim = state_dict[first_key].shape[1]
                break

    if hidden_dim is None:
        raise ValueError("Could not determine hidden dimension")

    print(f"  Hidden dim: {hidden_dim}")

    # Load concepts (needed for over-firer suppression)
    concepts = load_concept_prompts(concept_pack_dir, layers)

    # Create layer norm
    layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False).to(device)

    # Build lens path lookup
    lens_paths = {}
    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if layer_dir.exists():
            for lens_file in layer_dir.glob("*_classifier.pt"):
                concept_name = lens_file.stem.replace("_classifier", "")
                lens_paths[concept_name] = (lens_file, layer)

    results = []
    modified_lenses = set()  # Track which lenses we've modified

    print(f"\n  Processing ancestor competition failures...")
    for concept_name, report in tqdm(needs_ancestor_fix, desc="Fixing ancestor competition"):
        if concept_name not in lens_paths:
            continue

        failed_ancestors = report.get('failed_ancestors', [])
        if not failed_ancestors:
            continue

        # Load target lens
        target_path, target_layer = lens_paths[concept_name]
        state_dict = torch.load(target_path, map_location='cpu')
        target_lens = SimpleMLP(hidden_dim).to(device)
        if not list(state_dict.keys())[0].startswith('net.'):
            state_dict = {f'net.{k}': v for k, v in state_dict.items()}
        target_lens.load_state_dict(state_dict)

        # Load ancestor lenses
        ancestor_lenses = {}
        for ancestor_name in failed_ancestors:
            if ancestor_name not in lens_paths:
                continue
            ancestor_path, _ = lens_paths[ancestor_name]
            state_dict = torch.load(ancestor_path, map_location='cpu')
            ancestor_lens = SimpleMLP(hidden_dim).to(device)
            if not list(state_dict.keys())[0].startswith('net.'):
                state_dict = {f'net.{k}': v for k, v in state_dict.items()}
            ancestor_lens.load_state_dict(state_dict)
            ancestor_lenses[ancestor_name] = ancestor_lens

        if not ancestor_lenses:
            continue

        # Get activation for concept name
        activation = extract_activation(model, tokenizer, concept_name, device, layer_idx)
        activation_tensor = torch.tensor(activation, dtype=torch.float32).to(device)
        activation_tensor = layer_norm(activation_tensor.unsqueeze(0)).squeeze(0)

        # Fine-tune
        before_rate = report.get('ancestor_rank_0_rate', 0)

        target_lens, ancestor_lenses, metrics = finetune_ancestor_competition(
            target_lens, ancestor_lenses, activation_tensor,
            device, epochs=max_epochs, margin=0.05
        )

        # Save updated target lens
        torch.save(target_lens.state_dict(), target_path)
        modified_lenses.add(concept_name)

        # Save updated ancestor lenses (only if not already modified as a target)
        for ancestor_name, ancestor_lens in ancestor_lenses.items():
            if ancestor_name not in modified_lenses:
                ancestor_path, _ = lens_paths[ancestor_name]
                torch.save(ancestor_lens.state_dict(), ancestor_path)
                modified_lenses.add(ancestor_name)

        results.append(FineTuneResult(
            concept=concept_name,
            layer=target_layer,
            action='ancestor_competition',
            before_in_top_k_rate=before_rate,
            after_in_top_k_rate=1.0 if metrics['target_beats_all'] else before_rate,
            before_avg_rank=0 if before_rate == 1 else 1,
            after_avg_rank=0 if metrics['target_beats_all'] else 1,
            epochs_used=metrics['epochs'],
            improved=metrics['target_beats_all'],
        ))

        # Clear CUDA cache
        del target_lens, ancestor_lenses
        torch.cuda.empty_cache()

    # Process sibling competition failures
    # Skip if already processed as ancestor failure
    already_processed_ancestor = {name for name, _ in needs_ancestor_fix}

    sibling_to_process = [
        (name, report) for name, report in needs_sibling_fix
        if name not in already_processed_ancestor
    ]

    if sibling_to_process:
        print(f"\n  Processing sibling competition failures ({len(sibling_to_process)} concepts)...")

        for concept_name, report in tqdm(sibling_to_process, desc="Fixing sibling competition"):
            if concept_name not in lens_paths:
                continue

            failed_siblings = report.get('failed_siblings', [])
            if not failed_siblings:
                continue

            # Load target lens
            target_path, target_layer = lens_paths[concept_name]
            state_dict = torch.load(target_path, map_location='cpu')
            target_lens = SimpleMLP(hidden_dim).to(device)
            if not list(state_dict.keys())[0].startswith('net.'):
                state_dict = {f'net.{k}': v for k, v in state_dict.items()}
            target_lens.load_state_dict(state_dict)

            # Load sibling lenses
            sibling_lenses = {}
            for sibling_name in failed_siblings:
                if sibling_name not in lens_paths:
                    continue
                sibling_path, _ = lens_paths[sibling_name]
                state_dict = torch.load(sibling_path, map_location='cpu')
                sibling_lens = SimpleMLP(hidden_dim).to(device)
                if not list(state_dict.keys())[0].startswith('net.'):
                    state_dict = {f'net.{k}': v for k, v in state_dict.items()}
                sibling_lens.load_state_dict(state_dict)
                sibling_lenses[sibling_name] = sibling_lens

            if not sibling_lenses:
                continue

            # Get activation for concept name
            activation = extract_activation(model, tokenizer, concept_name, device, layer_idx)
            activation_tensor = torch.tensor(activation, dtype=torch.float32).to(device)
            activation_tensor = layer_norm(activation_tensor.unsqueeze(0)).squeeze(0)

            # Fine-tune using same contrastive approach as ancestors
            before_rate = report.get('sibling_rank_0_rate', 0)

            target_lens, sibling_lenses, metrics = finetune_contrastive(
                target_lens, sibling_lenses, activation_tensor,
                device, epochs=max_epochs, lr=0.001, margin=0.05
            )

            # Save updated target lens
            torch.save(target_lens.state_dict(), target_path)
            modified_lenses.add(concept_name)

            # Save updated sibling lenses (only if not already modified)
            for sibling_name, sibling_lens in sibling_lenses.items():
                if sibling_name not in modified_lenses:
                    sibling_path, _ = lens_paths[sibling_name]
                    torch.save(sibling_lens.state_dict(), sibling_path)
                    modified_lenses.add(sibling_name)

            results.append(FineTuneResult(
                concept=concept_name,
                layer=target_layer,
                action='sibling_competition',
                before_in_top_k_rate=before_rate,
                after_in_top_k_rate=1.0 if metrics['target_beats_all'] else before_rate,
                before_avg_rank=0 if before_rate == 1 else 1,
                after_avg_rank=0 if metrics['target_beats_all'] else 1,
                epochs_used=metrics['epochs'],
                improved=metrics['target_beats_all'],
            ))

            # Clear CUDA cache
            del target_lens, sibling_lenses
            torch.cuda.empty_cache()

    # Process random/top-k criterion failures (absolute boost on concept name)
    # Skip if already processed as ancestor or sibling failure
    already_processed = already_processed_ancestor.union({name for name, _ in sibling_to_process})

    # For production mode, use 'in_top_k'; for dual_criteria use 'passes_random_criterion'
    is_production = analysis.get('mode') == 'production'
    if is_production:
        needs_topk_fix = [
            (name, report) for name, report in lens_reports.items()
            if not report.get('in_top_k', True)
            and name not in already_processed
        ]
    else:
        needs_topk_fix = [
            (name, report) for name, report in needs_random_fix
            if name not in already_processed
        ]

    if needs_topk_fix:
        print(f"\n  Processing top-k criterion failures ({len(needs_topk_fix)} concepts)...")
        print(f"  Using CONTRASTIVE training - boost target while suppressing intruders from analysis...")

        # Estimate GPU memory budget for lens loading
        # Each SimpleMLP ~= hidden_dim * 64 * 4 bytes (rough estimate for 3-layer MLP)
        lens_memory_bytes = hidden_dim * 64 * 4  # ~1MB per lens for 4096 hidden
        try:
            gpu_free = torch.cuda.mem_get_info()[0]
            max_lenses_gpu = int(gpu_free * 0.5 / lens_memory_bytes)  # Use 50% of free memory
        except:
            max_lenses_gpu = 200  # Conservative default

        print(f"  GPU memory budget: ~{max_lenses_gpu} lenses per batch")

        concepts_fixed = 0
        concepts_improved = 0

        for concept_name, report in tqdm(needs_topk_fix, desc="Contrastive training"):
            if concept_name not in lens_paths:
                continue

            # Get intruders from analysis (already computed during analysis phase)
            intruders = report.get('intruders', [])
            if not intruders:
                continue  # No intruders to suppress

            before_rank = report.get('rank', 999)

            # Get activation for concept name
            activation = extract_activation(model, tokenizer, concept_name, device, layer_idx)
            activation_tensor = torch.tensor(activation, dtype=torch.float32).to(device)
            activation_tensor = layer_norm(activation_tensor.unsqueeze(0)).squeeze(0)

            # Load target lens
            target_path, target_layer = lens_paths[concept_name]
            target_state = torch.load(target_path, map_location='cpu')
            target_lens = SimpleMLP(hidden_dim).to(device)
            if not list(target_state.keys())[0].startswith('net.'):
                target_state = {f'net.{k}': v for k, v in target_state.items()}
            target_lens.load_state_dict(target_state)

            # Load intruders - batch if too many for GPU
            intruder_names_to_load = [n for n in intruders if n in lens_paths]

            if len(intruder_names_to_load) <= max_lenses_gpu:
                # All fit on GPU - load them all
                competitors_to_suppress = {}
                for name in intruder_names_to_load:
                    path, _ = lens_paths[name]
                    state = torch.load(path, map_location='cpu')
                    lens = SimpleMLP(hidden_dim).to(device)
                    if not list(state.keys())[0].startswith('net.'):
                        state = {f'net.{k}': v for k, v in state.items()}
                    lens.load_state_dict(state)
                    competitors_to_suppress[name] = lens

                # Contrastive fine-tuning
                target_lens, updated_competitors, metrics = finetune_contrastive(
                    target_lens,
                    competitors_to_suppress,
                    activation_tensor,
                    device,
                    epochs=max_epochs,
                    lr=0.001,
                    margin=0.02,
                )

                # Save updated competitors back to disk
                for comp_name, comp_lens in updated_competitors.items():
                    comp_path, _ = lens_paths[comp_name]
                    comp_state = comp_lens.cpu().state_dict()
                    torch.save(comp_state, comp_path)
                    modified_lenses.add(comp_name)
            else:
                # Too many intruders - batch them
                print(f"    {concept_name}: {len(intruder_names_to_load)} intruders, batching...")
                metrics = {'target_beats_all': False, 'improvement': 0, 'epochs': 0}

                for batch_start in range(0, len(intruder_names_to_load), max_lenses_gpu):
                    batch_names = intruder_names_to_load[batch_start:batch_start + max_lenses_gpu]
                    competitors_to_suppress = {}
                    for name in batch_names:
                        path, _ = lens_paths[name]
                        state = torch.load(path, map_location='cpu')
                        lens = SimpleMLP(hidden_dim).to(device)
                        if not list(state.keys())[0].startswith('net.'):
                            state = {f'net.{k}': v for k, v in state.items()}
                        lens.load_state_dict(state)
                        competitors_to_suppress[name] = lens

                    target_lens, updated_competitors, batch_metrics = finetune_contrastive(
                        target_lens,
                        competitors_to_suppress,
                        activation_tensor,
                        device,
                        epochs=max_epochs,
                        lr=0.001,
                        margin=0.02,
                    )

                    # Save updated competitors
                    for comp_name, comp_lens in updated_competitors.items():
                        comp_path, _ = lens_paths[comp_name]
                        comp_state = comp_lens.cpu().state_dict()
                        torch.save(comp_state, comp_path)
                        modified_lenses.add(comp_name)

                    # Track metrics from last batch
                    metrics = batch_metrics
                    torch.cuda.empty_cache()

            # Save updated target lens
            target_state = target_lens.cpu().state_dict()
            torch.save(target_state, target_path)
            modified_lenses.add(concept_name)

            if metrics['target_beats_all']:
                concepts_fixed += 1
            if metrics['improvement'] > 0:
                concepts_improved += 1

            results.append(FineTuneResult(
                concept=concept_name,
                layer=lens_paths[concept_name][1],
                action='contrastive',
                before_in_top_k_rate=0,
                after_in_top_k_rate=1.0 if metrics['target_beats_all'] else 0,
                before_avg_rank=before_rank,
                after_avg_rank=0 if metrics['target_beats_all'] else before_rank,
                epochs_used=metrics['epochs'],
                improved=metrics['target_beats_all'],
            ))

            # Clear CUDA cache periodically
            torch.cuda.empty_cache()

        print(f"\n  Contrastive training complete:")
        print(f"    Concepts fixed (now beat all competitors): {concepts_fixed}")
        print(f"    Concepts improved (higher score): {concepts_improved}")
        print(f"    Total lenses modified: {len(modified_lenses)}")

        torch.cuda.empty_cache()

    # Phase 4: Direct dampening of chronic over-firers
    # For lenses that fire on thousands of concepts, directly scale down their weights
    # Dampening is severity-based (percentage of total concepts)
    chronic_over_firers = analysis.get('over_firing', [])
    lens_reports = analysis.get('lens_reports', {})
    total_concepts = len(lens_reports) if lens_reports else 1

    if chronic_over_firers:
        print(f"\n  Processing chronic over-firers ({len(chronic_over_firers)} lenses)...")
        print(f"  Using SEVERITY-BASED DAMPENING (% of {total_concepts} concepts)...")

        over_firers_dampened = 0
        severity_counts = {'extreme': 0, 'severe': 0, 'moderate': 0, 'mild': 0}

        for over_firer_name in tqdm(chronic_over_firers, desc="Dampening over-firers"):
            if over_firer_name not in lens_paths:
                continue

            # Get over-fire severity as percentage of pack
            over_fire_count = lens_reports.get(over_firer_name, {}).get('over_fire_count', 0)
            over_fire_pct = over_fire_count / total_concepts

            # Continuous dampening: proportional to intrusion %
            # Like a compressor/limiter in audio - reduce gain proportionally to how much you're "peaking"
            # 0% intrusion → no dampening (1.0x weights, 0 bias shift)
            # 100% intrusion → maximum dampening (0.0x weights, -3.0 bias shift)
            dampen_factor = 1.0 - over_fire_pct  # Keep (1 - intrusion%) of weights
            bias_shift = -3.0 * over_fire_pct    # Shift bias proportionally (max -3.0 at 100%)

            # Track for summary stats
            if over_fire_pct >= 0.40:
                severity_counts['extreme'] += 1
            elif over_fire_pct >= 0.15:
                severity_counts['severe'] += 1
            elif over_fire_pct >= 0.05:
                severity_counts['moderate'] += 1
            else:
                severity_counts['mild'] += 1

            # Load the over-firing lens
            over_firer_path, over_firer_layer = lens_paths[over_firer_name]
            state_dict = torch.load(over_firer_path, map_location='cpu')
            lens = SimpleMLP(hidden_dim).to(device)
            if not list(state_dict.keys())[0].startswith('net.'):
                state_dict = {f'net.{k}': v for k, v in state_dict.items()}
            lens.load_state_dict(state_dict)

            # Continuous dampening - scale weights and bias proportionally to intrusion %
            lens = dampen_overfirer(lens, dampen_factor=dampen_factor, bias_shift=bias_shift)

            # Save updated lens
            torch.save(lens.cpu().state_dict(), over_firer_path)
            modified_lenses.add(over_firer_name)
            over_firers_dampened += 1

            results.append(FineTuneResult(
                concept=over_firer_name,
                layer=over_firer_layer,
                action='direct_dampen',
                before_in_top_k_rate=0,
                after_in_top_k_rate=0,
                before_avg_rank=0,
                after_avg_rank=0,
                epochs_used=0,
                improved=True,
            ))

        print(f"\n  Continuous dampening complete (proportional to intrusion %):")
        print(f"    Over-firers dampened: {over_firers_dampened}")
        print(f"    Extreme (40%+): {severity_counts['extreme']}")
        print(f"    Severe (15-40%): {severity_counts['severe']}")
        print(f"    Moderate (5-15%): {severity_counts['moderate']}")
        print(f"    Mild (<5%): {severity_counts['mild']}")

    # Create report
    finetune_report = FineTuneReport(
        lens_pack_id=lens_pack_dir.name,
        analysis_timestamp=analysis.get('timestamp', ''),
        finetune_timestamp=datetime.now(timezone.utc).isoformat(),
        total_lenses_processed=len(results),
        lenses_boosted=sum(1 for r in results if r.improved),
        lenses_suppressed=sum(1 for r in results if r.action in ('targeted_suppress', 'direct_dampen') and r.improved),
        avg_improvement=sum(1 for r in results if r.improved) / len(results) if results else 0,
        results=results,
    )

    return finetune_report


# Backward-compatible alias
run_dual_criteria_finetune = run_triple_criteria_finetune


def run_calibration_finetune(
    lens_pack_dir: Path,
    concept_pack_dir: Path,
    analysis: Dict,
    model,
    tokenizer,
    device: str,
    layers: List[int],
    max_epochs: int = 20,
    boost_only: bool = False,
    suppress_only: bool = False,
    fast_mode: bool = True,
    layer_idx: int = 15,
) -> FineTuneReport:
    """
    Run fine-tuning based on calibration analysis.
    """
    from src.monitoring.dynamic_lens_manager import SimpleMLP

    print(f"\n{'='*80}")
    print("PACK-LEVEL CALIBRATION FINE-TUNING")
    print(f"{'='*80}")
    print(f"  Lens pack: {lens_pack_dir}")
    print(f"  Analysis from: {analysis['timestamp']}")
    print(f"  Under-firing concepts: {len(analysis['under_firing'])}")
    print(f"  Over-firing concepts: {len(analysis['over_firing'])}")
    print(f"  Mode: {'boost only' if boost_only else 'suppress only' if suppress_only else 'full'}")

    # Load concepts
    concepts = load_concept_prompts(concept_pack_dir, layers)

    # Determine hidden dim from first lens
    hidden_dim = None
    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if layer_dir.exists():
            lens_files = list(layer_dir.glob("*_classifier.pt"))
            if lens_files:
                state_dict = torch.load(lens_files[0], map_location='cpu')
                first_key = list(state_dict.keys())[0]
                hidden_dim = state_dict[first_key].shape[1]
                break

    if hidden_dim is None:
        raise ValueError("Could not determine hidden dimension from lens pack")

    print(f"  Hidden dim: {hidden_dim}")

    # Create layer norm
    layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False).to(device)

    results = []

    # Process under-firing lenses (boost)
    if not suppress_only:
        print(f"\n  Processing under-firing lenses (boost)...")
        for concept_name in tqdm(analysis['under_firing'], desc="Boosting"):
            if concept_name not in concepts:
                continue

            concept_data = concepts[concept_name]
            layer = concept_data['layer']
            lens_path = lens_pack_dir / f"layer{layer}" / f"{concept_name}_classifier.pt"

            if not lens_path.exists():
                continue

            # Load lens
            state_dict = torch.load(lens_path, map_location='cpu')
            lens = SimpleMLP(hidden_dim).to(device)
            if not list(state_dict.keys())[0].startswith('net.'):
                state_dict = {f'net.{k}': v for k, v in state_dict.items()}
            lens.load_state_dict(state_dict)

            # Extract activations for target prompts
            target_activations = []
            for prompt in concept_data['positive_prompts'][:5]:
                try:
                    act = extract_activation(model, tokenizer, prompt, device, layer_idx, fast_mode)
                    target_activations.append(act)
                except:
                    pass

            if not target_activations:
                continue

            # Extract activations for other prompts (from negative examples or other concepts)
            other_activations = []
            for prompt in concept_data.get('negative_prompts', [])[:3]:
                try:
                    act = extract_activation(model, tokenizer, prompt, device, layer_idx, fast_mode)
                    other_activations.append(act)
                except:
                    pass

            # Get before metrics
            report = analysis['lens_reports'].get(concept_name, {})
            before_rate = report.get('in_top_k_rate', 0)
            before_rank = report.get('avg_rank', 100)

            # Fine-tune
            lens, metrics = finetune_lens_boost(
                lens, target_activations, other_activations,
                device, epochs=max_epochs, layer_norm=layer_norm
            )

            # Save updated lens
            torch.save(lens.state_dict(), lens_path)

            results.append(FineTuneResult(
                concept=concept_name,
                layer=layer,
                action='boost',
                before_in_top_k_rate=before_rate,
                after_in_top_k_rate=before_rate,  # Will be updated on re-analysis
                before_avg_rank=before_rank,
                after_avg_rank=before_rank,  # Will be updated on re-analysis
                epochs_used=metrics['epochs'],
                improved=metrics['final_avg_score'] > metrics['initial_avg_score'],
            ))

    # Process over-firing lenses (suppress)
    if not boost_only:
        print(f"\n  Processing over-firing lenses (suppress)...")

        # Build map of which concepts each lens over-fires on
        over_fire_map = {}
        for concept_name in analysis['over_firing']:
            report = analysis['lens_reports'].get(concept_name, {})
            over_fire_on = report.get('over_fire_on', [])
            if over_fire_on:
                over_fire_map[concept_name] = over_fire_on

        for concept_name, over_fire_targets in tqdm(over_fire_map.items(), desc="Suppressing"):
            if concept_name not in concepts:
                continue

            concept_data = concepts[concept_name]
            layer = concept_data['layer']
            lens_path = lens_pack_dir / f"layer{layer}" / f"{concept_name}_classifier.pt"

            if not lens_path.exists():
                continue

            # Load lens
            state_dict = torch.load(lens_path, map_location='cpu')
            lens = SimpleMLP(hidden_dim).to(device)
            if not list(state_dict.keys())[0].startswith('net.'):
                state_dict = {f'net.{k}': v for k, v in state_dict.items()}
            lens.load_state_dict(state_dict)

            # Extract activations for prompts we SHOULD NOT fire on
            suppress_activations = []
            for target_concept in over_fire_targets[:5]:
                if target_concept in concepts:
                    for prompt in concepts[target_concept]['positive_prompts'][:2]:
                        try:
                            act = extract_activation(model, tokenizer, prompt, device, layer_idx, fast_mode)
                            suppress_activations.append(act)
                        except:
                            pass

            if not suppress_activations:
                continue

            # Extract activations for prompts we SHOULD fire on (preserve these)
            preserve_activations = []
            for prompt in concept_data['positive_prompts'][:3]:
                try:
                    act = extract_activation(model, tokenizer, prompt, device, layer_idx, fast_mode)
                    preserve_activations.append(act)
                except:
                    pass

            # Get before metrics
            report = analysis['lens_reports'].get(concept_name, {})
            before_rate = report.get('in_top_k_rate', 0)
            before_rank = report.get('avg_rank', 100)

            # Fine-tune
            lens, metrics = finetune_lens_suppress(
                lens, suppress_activations, preserve_activations,
                device, epochs=max_epochs // 2, layer_norm=layer_norm  # Fewer epochs for suppression
            )

            # Save updated lens
            torch.save(lens.state_dict(), lens_path)

            results.append(FineTuneResult(
                concept=concept_name,
                layer=layer,
                action='suppress',
                before_in_top_k_rate=before_rate,
                after_in_top_k_rate=before_rate,
                before_avg_rank=before_rank,
                after_avg_rank=before_rank,
                epochs_used=metrics['epochs'],
                improved=metrics['final_suppress_avg'] < metrics['initial_suppress_avg'],
            ))

    # Create report
    report = FineTuneReport(
        lens_pack_id=lens_pack_dir.name,
        analysis_timestamp=analysis['timestamp'],
        finetune_timestamp=datetime.now(timezone.utc).isoformat(),
        total_lenses_processed=len(results),
        lenses_boosted=sum(1 for r in results if r.action == 'boost'),
        lenses_suppressed=sum(1 for r in results if r.action == 'suppress'),
        avg_improvement=sum(1 for r in results if r.improved) / len(results) if results else 0,
        results=results,
    )

    return report


def print_finetune_summary(report: FineTuneReport):
    """Print human-readable summary."""
    print(f"\n{'='*80}")
    print("FINE-TUNING SUMMARY")
    print(f"{'='*80}")
    print(f"  Total lenses processed: {report.total_lenses_processed}")
    print(f"  Lenses boosted: {report.lenses_boosted}")
    print(f"  Lenses suppressed: {report.lenses_suppressed}")
    print(f"  Improvement rate: {report.avg_improvement:.1%}")
    print()
    print("  Run calibration analysis again to measure actual improvement.")


def main():
    parser = argparse.ArgumentParser(description='Pack-level calibration fine-tuning')
    parser.add_argument('--lens-pack', required=True, help='Path to lens pack')
    parser.add_argument('--concept-pack', required=True, help='Path to concept pack')
    parser.add_argument('--model', required=True, help='Model name/path')
    parser.add_argument('--analysis', required=True, help='Path to calibration analysis JSON')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                        help='Layers to process (default: all from analysis)')
    parser.add_argument('--max-finetune-epochs', type=int, default=20,
                        help='Max epochs per lens')
    parser.add_argument('--boost-only', action='store_true',
                        help='Only boost under-firing lenses')
    parser.add_argument('--suppress-only', action='store_true',
                        help='Only suppress over-firing lenses')
    parser.add_argument('--fast-mode', action='store_true', default=True,
                        help='Fast mode (prompt only)')
    parser.add_argument('--layer-idx', type=int, default=15,
                        help='Model layer for activations')
    parser.add_argument('--dual-criteria', action='store_true',
                        help='Use dual-criteria mode (for dual-criteria analysis results)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON (default: lens_pack/calibration_finetune.json)')

    args = parser.parse_args()

    lens_pack_dir = Path(args.lens_pack)
    concept_pack_dir = Path(args.concept_pack)
    analysis_path = Path(args.analysis)

    # Load analysis
    analysis = load_analysis(analysis_path)

    # Auto-detect triple-criteria mode from analysis
    is_triple_criteria = args.dual_criteria or analysis.get('mode') in ('dual_criteria', 'triple_criteria')

    # Determine layers
    if args.layers is None:
        layers = []
        for layer_dir in lens_pack_dir.glob("layer*"):
            if layer_dir.is_dir():
                try:
                    layer_num = int(layer_dir.name.replace('layer', ''))
                    layers.append(layer_num)
                except ValueError:
                    pass
        layers.sort()
    else:
        layers = args.layers

    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    # Run fine-tuning
    if is_triple_criteria:
        report = run_triple_criteria_finetune(
            lens_pack_dir=lens_pack_dir,
            concept_pack_dir=concept_pack_dir,
            analysis=analysis,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            layers=layers,
            max_epochs=args.max_finetune_epochs,
            layer_idx=args.layer_idx,
        )
    else:
        report = run_calibration_finetune(
            lens_pack_dir=lens_pack_dir,
            concept_pack_dir=concept_pack_dir,
            analysis=analysis,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            layers=layers,
            max_epochs=args.max_finetune_epochs,
            boost_only=args.boost_only,
            suppress_only=args.suppress_only,
            fast_mode=args.fast_mode,
            layer_idx=args.layer_idx,
        )

    # Print summary
    print_finetune_summary(report)

    # Save report
    output_path = Path(args.output) if args.output else lens_pack_dir / "calibration_finetune.json"
    with open(output_path, 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"\n  Saved report to: {output_path}")


if __name__ == '__main__':
    main()
