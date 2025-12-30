#!/usr/bin/env python3
"""
Cross-Activation Calibration

For each concept lens, measures:
1. self_mean: Average activation on its OWN prompts
2. cross_mean: Average activation across all OTHER concepts' prompts where it fired

This produces calibration data that allows normalizing detections at inference:
- 1.0 = firing at self_mean level (genuine signal)
- 0.5 = firing at cross_mean level (noise floor for this concept)
- 0.0 = floor

Usage:
    python scripts/tools/calibrate_cross_activation.py \
        --lens-pack apertus-8b_first-light \
        --model google/gemma-3-4b-pt \
        --output lens_packs/apertus-8b_first-light/calibration.json
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_concept_definitions(layers_dir: Path) -> Dict[Tuple[str, int], List[str]]:
    """Load definitions for all concepts from layer files."""
    definitions = {}

    for layer_file in sorted(layers_dir.glob("layer*.json")):
        with open(layer_file) as f:
            layer_data = json.load(f)

        if 'metadata' in layer_data:
            layer = layer_data['metadata']['layer']
        else:
            layer = layer_data.get('layer', int(layer_file.stem.replace('layer', '')))

        for concept in layer_data.get('concepts', []):
            sumo_term = concept['sumo_term']
            key = (sumo_term, layer)

            defs = []

            # Main definition
            definition = concept.get('definition', '')
            if definition and len(definition) > 10:
                defs.append(definition)

            # SUMO definition as secondary
            sumo_def = concept.get('sumo_definition', '')
            if sumo_def and len(sumo_def) > 10 and sumo_def != definition:
                defs.append(sumo_def)

            # Lemmas as short samples
            for lemma in concept.get('lemmas', [])[:3]:
                if lemma and len(lemma) > 3:
                    defs.append(f"A type of {lemma}")

            if defs:
                definitions[key] = defs

    return definitions


def capture_activation(
    text: str,
    model,
    tokenizer,
    target_layer: int = -1
) -> torch.Tensor:
    """Capture activation for a text sample at the target layer."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[target_layer]
        activation = hidden_states[:, -1, :]

    return activation.float()


def infer_hidden_dim_from_state_dict(state_dict: dict) -> int:
    """Infer the hidden dim (input size) from the first linear layer."""
    # Look for the first linear layer weight
    for key in ["net.0.weight", "0.weight", "net.1.weight", "1.weight"]:
        if key in state_dict:
            weight = state_dict[key]
            # If it's 2D, it's a Linear layer: [out_features, in_features]
            if len(weight.shape) == 2:
                return weight.shape[1]
            # If it's 1D, it's LayerNorm, skip to next
    # Fallback - try to find any 2D weight
    for key, val in state_dict.items():
        if "weight" in key and len(val.shape) == 2:
            return val.shape[1]
    return 2560  # Default for gemma-3-4b


def load_all_lenses_to_cpu(
    lens_pack_path: Path,
    concept_keys: List[Tuple[str, int]],
) -> Dict[Tuple[str, int], torch.nn.Module]:
    """Load all lenses to CPU memory."""
    from src.hat.monitoring.lens_types import create_lens_from_state_dict

    lenses = {}
    for concept_name, layer in tqdm(concept_keys, desc="Loading lenses to CPU"):
        # Try different naming conventions
        lens_file = lens_pack_path / f"layer{layer}" / f"{concept_name}.pt"
        if not lens_file.exists():
            lens_file = lens_pack_path / f"layer{layer}" / f"{concept_name}_classifier.pt"
        if not lens_file.exists():
            continue

        try:
            state_dict = torch.load(lens_file, map_location='cpu', weights_only=True)
            hidden_dim = infer_hidden_dim_from_state_dict(state_dict)
            lens = create_lens_from_state_dict(state_dict, hidden_dim=hidden_dim, device='cpu')
            lens.eval()
            lenses[(concept_name, layer)] = lens
        except Exception as e:
            continue

    return lenses


def run_cross_activation_calibration(
    lens_pack_path: Path,
    model,
    tokenizer,
    definitions: Dict[Tuple[str, int], List[str]],
    n_samples_per_concept: int = 5,
    firing_threshold: float = 0.5,
    model_layer: int = -1,
    device: str = "cuda",
    batch_size: int = 100,
) -> Dict[str, Dict]:
    """
    Run all lenses on all concepts' prompts to measure cross-activation.

    Lenses are kept on CPU and moved to GPU in batches to avoid OOM.

    Returns per-concept calibration data:
    {
        "ConceptName_L0": {
            "concept": "ConceptName",
            "layer": 0,
            "self_mean": 0.92,      # Mean activation on own prompts
            "self_std": 0.05,       # Std on own prompts
            "cross_mean": 0.31,     # Mean activation on other concepts' prompts
            "cross_std": 0.15,      # Std on cross-fires
            "cross_fire_count": 847, # How many other-concept prompts it fired on
            "total_prompts": 4000,  # Total prompts tested
            "cross_fire_rate": 0.21, # cross_fire_count / total_prompts
        }
    }
    """
    # Load all lenses to CPU
    print(f"Loading all lenses to CPU...")
    all_concept_keys = list(definitions.keys())
    cpu_lenses = load_all_lenses_to_cpu(lens_pack_path, all_concept_keys)
    available_keys = list(cpu_lenses.keys())
    print(f"  {len(available_keys)} lenses loaded to CPU")

    # Prepare sample prompts for each concept
    concept_prompts = {}
    for key in available_keys:
        defs = definitions.get(key, [])[:n_samples_per_concept]
        if defs:
            concept_prompts[key] = defs

    total_prompts = sum(len(p) for p in concept_prompts.values())
    print(f"  {total_prompts} total prompts across {len(concept_prompts)} concepts")

    # Data structures for accumulating scores
    # self_scores[concept_key] = [scores on own prompts]
    # cross_scores[concept_key] = [scores on other concepts' prompts where it fired]
    self_scores = defaultdict(list)
    cross_scores = defaultdict(list)
    cross_fire_counts = defaultdict(int)

    # Create layer norm for normalization
    layer_norm = None

    # Process all prompts - lenses stay on CPU, we batch-move them to GPU
    print("\nRunning cross-activation measurement...")
    print(f"  Processing in batches of {batch_size} lenses")

    lens_keys = list(cpu_lenses.keys())

    for source_key, prompts in tqdm(concept_prompts.items(), desc="Concepts"):
        source_name, source_layer = source_key

        for prompt in prompts:
            # Capture activation
            try:
                activation = capture_activation(prompt, model, tokenizer, model_layer)
            except Exception as e:
                continue

            # Normalize
            hidden_dim = activation.shape[-1]
            if layer_norm is None or layer_norm.normalized_shape[0] != hidden_dim:
                layer_norm = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False).to(activation.device)
            activation = layer_norm(activation)

            # Process lenses in batches
            with torch.inference_mode():
                for batch_start in range(0, len(lens_keys), batch_size):
                    batch_keys = lens_keys[batch_start:batch_start + batch_size]

                    # Move batch to GPU
                    gpu_lenses = {}
                    for key in batch_keys:
                        gpu_lenses[key] = cpu_lenses[key].to(device)

                    # Run batch
                    for target_key, lens in gpu_lenses.items():
                        # Match dtype
                        lens_dtype = next(lens.parameters()).dtype
                        act = activation.to(dtype=lens_dtype)

                        prob = lens(act).item()

                        if target_key == source_key:
                            self_scores[target_key].append(prob)
                        else:
                            if prob >= firing_threshold:
                                cross_scores[target_key].append(prob)
                                cross_fire_counts[target_key] += 1

                    # Move back to CPU to free GPU memory
                    for key in batch_keys:
                        cpu_lenses[key] = gpu_lenses[key].to('cpu')
                    del gpu_lenses
                    torch.cuda.empty_cache()

    # Compute calibration stats
    print("\nComputing calibration statistics...")
    calibration = {}

    for key in tqdm(available_keys, desc="Computing stats"):
        concept_name, layer = key
        key_str = f"{concept_name}_L{layer}"

        self_vals = self_scores.get(key, [])
        cross_vals = cross_scores.get(key, [])

        # Need at least some self-scores to calibrate
        if not self_vals:
            continue

        self_mean = float(np.mean(self_vals))
        self_std = float(np.std(self_vals)) if len(self_vals) > 1 else 0.0

        # For cross, use 0 if never fired on other concepts
        if cross_vals:
            cross_mean = float(np.mean(cross_vals))
            cross_std = float(np.std(cross_vals)) if len(cross_vals) > 1 else 0.0
        else:
            cross_mean = 0.0
            cross_std = 0.0

        calibration[key_str] = {
            "concept": concept_name,
            "layer": layer,
            "self_mean": self_mean,
            "self_std": self_std,
            "cross_mean": cross_mean,
            "cross_std": cross_std,
            "cross_fire_count": cross_fire_counts.get(key, 0),
            "total_other_prompts": total_prompts - len(self_vals),
            "cross_fire_rate": cross_fire_counts.get(key, 0) / max(1, total_prompts - len(self_vals)),
            "n_self_samples": len(self_vals),
            "n_cross_samples": len(cross_vals),
        }

    return calibration


def get_activation_cache_path(
    lens_pack_path: Path,
    prompt_source: str,
    n_prompts: int,
    n_tokens: int,
    model_layer: int,
) -> Path:
    """Get the path for cached activations."""
    cache_dir = lens_pack_path / "activation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"{prompt_source}_p{n_prompts}_t{n_tokens}_L{model_layer}.pt"
    return cache_dir / cache_name


def load_activation_cache(cache_path: Path) -> Optional[torch.Tensor]:
    """Load cached activations if they exist and are valid."""
    if not cache_path.exists():
        return None

    meta_path = cache_path.with_suffix('.json')
    if not meta_path.exists():
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)

        activation_tensor = torch.load(cache_path, map_location='cpu', weights_only=True)
        print(f"  ✓ Loaded cached activations: {activation_tensor.shape}")
        print(f"    Created: {meta.get('created_at', 'unknown')}")
        print(f"    Prompts: {meta.get('n_prompts', 'unknown')}")
        print(f"    Tokens: {meta.get('total_tokens', 'unknown')}")
        return activation_tensor
    except Exception as e:
        print(f"  ✗ Failed to load cache: {e}")
        return None


def save_activation_cache(
    cache_path: Path,
    activation_tensor: torch.Tensor,
    prompt_source: str,
    n_prompts: int,
    n_tokens_per_prompt: int,
    model_layer: int,
    model_name: str,
) -> None:
    """Save activations and metadata to cache."""
    # Save tensor
    torch.save(activation_tensor, cache_path)

    # Save metadata
    meta = {
        "created_at": datetime.now().isoformat(),
        "prompt_source": prompt_source,
        "n_prompts": n_prompts,
        "n_tokens_per_prompt": n_tokens_per_prompt,
        "total_tokens": activation_tensor.shape[0],
        "hidden_dim": activation_tensor.shape[1],
        "model_layer": model_layer,
        "model": model_name,
    }
    meta_path = cache_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ Saved activation cache: {cache_path}")
    print(f"    Shape: {activation_tensor.shape}")


def run_generation_calibration(
    lens_pack_path: Path,
    model,
    tokenizer,
    lenses: Dict[Tuple[str, int], torch.nn.Module],
    generation_prompts: List[str],
    n_tokens_per_prompt: int = 100,
    firing_threshold: float = 0.5,
    model_layer: int = 15,
    device: str = "cuda",
    batch_size: int = 100,
    prompt_source: str = "default",
    model_name: str = "unknown",
    use_cache: bool = True,
) -> Dict[str, Dict]:
    """
    Generation-based calibration: measure how often each concept fires during text generation.

    Memory-efficient approach:
    1. Check for cached activations (default: use if available)
    2. If no cache: generate text, cache hidden states to CPU, save to disk
    3. Run lenses in batches on cached states (model unloaded)

    Returns per-concept calibration data with gen_fire_rate, gen_mean, etc.
    """
    print(f"\n{'='*60}")
    print("GENERATION-BASED CALIBRATION")
    print(f"{'='*60}")
    print(f"  Generation prompts: {len(generation_prompts)}")
    print(f"  Tokens per prompt: {n_tokens_per_prompt}")
    print(f"  Total tokens expected: {len(generation_prompts) * n_tokens_per_prompt}")
    print(f"  Lenses to evaluate: {len(lenses)}")
    print(f"  Prompt source: {prompt_source}")
    print(f"  Cache enabled: {use_cache}")

    # ========== Check for cached activations ==========
    activation_tensor = None
    cache_path = get_activation_cache_path(
        lens_pack_path, prompt_source, len(generation_prompts),
        n_tokens_per_prompt, model_layer
    )

    if use_cache:
        print(f"\nChecking for cached activations...")
        print(f"  Cache path: {cache_path}")
        activation_tensor = load_activation_cache(cache_path)

    # ========== PHASE 1: Generate and cache hidden states (if needed) ==========
    if activation_tensor is None:
        print(f"\nPhase 1: Generating text and caching hidden states...")

        cached_activations = []  # List of tensors on CPU
        layer_norm = None

        # Save progress incrementally every N prompts
        checkpoint_interval = 100
        checkpoint_path = cache_path.with_suffix('.checkpoint.pt')

        for prompt_idx, prompt in enumerate(tqdm(generation_prompts, desc="Generating")):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            for step in range(n_tokens_per_prompt):
                with torch.inference_mode():
                    outputs = model(input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[model_layer]
                    activation = hidden_states[0, -1, :].float()

                    # Normalize
                    if layer_norm is None:
                        layer_norm = torch.nn.LayerNorm(
                            activation.shape[-1], elementwise_affine=False
                        ).to(device)
                    activation = layer_norm(activation.unsqueeze(0)).squeeze(0)

                    # Cache to CPU
                    cached_activations.append(activation.cpu())

                    # Sample next token
                    logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    if next_token.item() == tokenizer.eos_token_id:
                        break

                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    if input_ids.shape[1] > 512:
                        input_ids = input_ids[:, -400:]

            # Periodic checkpoint save
            if use_cache and (prompt_idx + 1) % checkpoint_interval == 0:
                checkpoint_tensor = torch.stack(cached_activations)
                torch.save(checkpoint_tensor, checkpoint_path)
                print(f"  Checkpoint saved: {len(cached_activations)} activations")

            if prompt_idx % 20 == 0:
                torch.cuda.empty_cache()

        total_tokens = len(cached_activations)
        print(f"  Generated {total_tokens} activation vectors")

        # Stack into tensor
        activation_tensor = torch.stack(cached_activations)
        del cached_activations
        torch.cuda.empty_cache()

        # Save final cache
        if use_cache:
            save_activation_cache(
                cache_path, activation_tensor, prompt_source,
                len(generation_prompts), n_tokens_per_prompt,
                model_layer, model_name
            )
            # Remove checkpoint
            if checkpoint_path.exists():
                checkpoint_path.unlink()
    else:
        print(f"\n  Using cached activations (skipping generation)")

    total_tokens = activation_tensor.shape[0]

    # ========== PHASE 2: Unload model, run lenses ==========
    print(f"\nPhase 2: Running lenses on cached activations...")
    print(f"  Freeing model memory...")

    # Move model to CPU to free GPU for lenses
    model.cpu()
    torch.cuda.empty_cache()

    # Tracking
    gen_fire_counts = defaultdict(int)
    gen_scores = defaultdict(list)
    lens_keys = list(lenses.keys())

    # Process lenses in batches
    # Move activation tensor to GPU in chunks
    act_batch_size = 500  # Process 500 activations at a time

    for lens_batch_start in tqdm(range(0, len(lens_keys), batch_size), desc="Lens batches"):
        lens_batch_keys = lens_keys[lens_batch_start:lens_batch_start + batch_size]

        # Move this batch of lenses to GPU
        gpu_lenses = {}
        for key in lens_batch_keys:
            gpu_lenses[key] = lenses[key].to(device)

        # Process activations in chunks
        for act_start in range(0, total_tokens, act_batch_size):
            act_end = min(act_start + act_batch_size, total_tokens)
            act_batch = activation_tensor[act_start:act_end].to(device)

            with torch.inference_mode():
                for key in lens_batch_keys:
                    lens = gpu_lenses[key]
                    lens_dtype = next(lens.parameters()).dtype
                    act = act_batch.to(dtype=lens_dtype)

                    # Run lens on batch
                    probs = lens(act)  # (batch, 1) or (batch,)
                    if probs.dim() > 1:
                        probs = probs.squeeze(-1)

                    # Count fires
                    fired_mask = probs >= firing_threshold
                    fire_count = fired_mask.sum().item()
                    gen_fire_counts[key] += fire_count

                    if fire_count > 0:
                        fired_probs = probs[fired_mask].cpu().tolist()
                        gen_scores[key].extend(fired_probs)

        # Move lenses back to CPU
        for key in lens_batch_keys:
            lenses[key] = gpu_lenses[key].cpu()
        del gpu_lenses
        torch.cuda.empty_cache()

    # ========== PHASE 3: Compute stats ==========
    print(f"\nPhase 3: Computing statistics...")
    calibration = {}

    for key in tqdm(lens_keys, desc="Computing stats"):
        concept_name, layer = key
        key_str = f"{concept_name}_L{layer}"

        fire_count = gen_fire_counts.get(key, 0)
        scores = gen_scores.get(key, [])

        gen_fire_rate = fire_count / max(1, total_tokens)

        if scores:
            gen_mean = float(np.mean(scores))
            gen_std = float(np.std(scores)) if len(scores) > 1 else 0.0
        else:
            gen_mean = 0.0
            gen_std = 0.0

        calibration[key_str] = {
            "concept": concept_name,
            "layer": layer,
            "gen_fire_rate": gen_fire_rate,
            "gen_fire_count": fire_count,
            "total_gen_tokens": total_tokens,
            "gen_mean": gen_mean,
            "gen_std": gen_std,
        }

    # Reload model to GPU for any subsequent operations
    model.to(device)

    return calibration


# Diverse prompts for generation calibration - covers many topics
GENERATION_CALIBRATION_PROMPTS = [
    # Everyday conversation
    "Tell me about your day so far.",
    "What's the weather like today?",
    "Can you recommend a good restaurant?",
    "How do I get to the nearest train station?",

    # Technical/coding
    "Write a Python function that sorts a list.",
    "Explain how a hash table works.",
    "What's the difference between TCP and UDP?",
    "Debug this code: for i in range(10) print(i)",

    # Creative writing
    "Write a short story about a dragon.",
    "Compose a poem about the ocean.",
    "Describe a sunset in vivid detail.",
    "Create a dialogue between two old friends.",

    # Science/education
    "Explain photosynthesis to a child.",
    "What causes earthquakes?",
    "How do vaccines work?",
    "Describe the water cycle.",

    # Philosophy/abstract
    "What is the meaning of life?",
    "Is free will an illusion?",
    "What makes something beautiful?",
    "Can machines truly think?",

    # Practical/how-to
    "How do I bake a chocolate cake?",
    "What's the best way to learn a new language?",
    "How do I change a car tire?",
    "Tips for public speaking?",

    # News/current events style
    "Summarize the latest developments in AI.",
    "What are the main challenges facing climate policy?",
    "Discuss the future of remote work.",
    "Analyze the global economy.",

    # Emotional/personal
    "I'm feeling anxious about a job interview.",
    "How do I deal with a difficult coworker?",
    "What should I do if I'm feeling lonely?",
    "Advice for maintaining long-distance friendships?",

    # Random/varied
    "Compare cats and dogs as pets.",
    "What's your favorite book and why?",
    "Explain the rules of chess.",
    "Describe the history of the internet.",
    "What makes a good leader?",
    "How do airplanes fly?",
    "Tell me about ancient Egypt.",
    "What's the future of space exploration?",
]


def load_hierarchy(concept_pack_path: Path) -> Dict:
    """Load hierarchy.json for family-based generation."""
    hier_path = concept_pack_path / "hierarchy.json"
    if hier_path.exists():
        with open(hier_path) as f:
            return json.load(f)
    return {}


def generate_concept_prompts(definitions: Dict[Tuple[str, int], List[str]]) -> List[str]:
    """Generate a prompt for each concept using its definition."""
    prompts = []
    for (concept_name, layer), defs in definitions.items():
        if defs:
            # Use the main definition as the prompt
            main_def = defs[0]
            # Create a generation-friendly prompt
            prompts.append(f"Write about {concept_name.lower()}: {main_def}")
    return prompts


def generate_family_prompts(hierarchy: Dict, definitions: Dict[Tuple[str, int], List[str]]) -> List[str]:
    """Generate prompts for each parent concept (family) in the hierarchy."""
    prompts = []
    parent_to_children = hierarchy.get("parent_to_children", {})

    # Build a lookup for definitions by concept name (without layer)
    def_by_name = {}
    for (concept_name, layer), defs in definitions.items():
        if defs and concept_name not in def_by_name:
            def_by_name[concept_name] = defs[0]

    for parent in parent_to_children.keys():
        # Extract base name (remove :layer suffix if present)
        base_name = parent.split(":")[0] if ":" in parent else parent

        if base_name in def_by_name:
            prompts.append(f"Write about {base_name.lower()}: {def_by_name[base_name]}")
        else:
            # Fallback: just use the concept name
            prompts.append(f"Write about the concept of {base_name.lower()}.")

    return prompts


def estimate_runtime(n_prompts: int, tokens_per_prompt: int, reference_prompts: int = 2000,
                     reference_tokens: int = 100, reference_hours: float = 10.0) -> str:
    """Estimate runtime based on reference run."""
    reference_total = reference_prompts * reference_tokens
    target_total = n_prompts * tokens_per_prompt

    estimated_hours = (target_total / reference_total) * reference_hours

    if estimated_hours < 1:
        return f"{estimated_hours * 60:.0f} minutes"
    elif estimated_hours < 24:
        return f"{estimated_hours:.1f} hours"
    else:
        days = estimated_hours / 24
        return f"{days:.1f} days ({estimated_hours:.0f} hours)"


def main():
    parser = argparse.ArgumentParser(description="Cross-activation calibration")
    parser.add_argument("--lens-pack", required=True, help="Lens pack ID")
    parser.add_argument("--model", default="google/gemma-3-4b-pt", help="Model for activations")
    parser.add_argument("--layers-dir", default=None, help="Concept layers directory")
    parser.add_argument("--n-samples", type=int, default=5, help="Samples per concept")
    parser.add_argument("--firing-threshold", type=float, default=0.5, help="Threshold for counting cross-fires")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-concepts", type=int, default=None, help="Limit concepts (for testing)")
    parser.add_argument("--mode", choices=["cross", "generation", "both"], default="cross",
                       help="Calibration mode: cross (concept prompts), generation (free generation), both")
    parser.add_argument("--gen-prompts", type=int, default=40, help="Number of generation prompts")
    parser.add_argument("--gen-tokens", type=int, default=100, help="Tokens per generation prompt")
    parser.add_argument("--model-layer", type=int, default=15, help="Model layer for hidden states")
    parser.add_argument("--all-concepts", action="store_true",
                       help="Generate using each concept's definition as a prompt (overrides --gen-prompts)")
    parser.add_argument("--all-families", action="store_true",
                       help="Generate using each parent/family concept as a prompt (overrides --gen-prompts)")
    parser.add_argument("--estimate-only", action="store_true",
                       help="Only estimate runtime, don't run calibration")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable activation caching (regenerate even if cache exists)")

    args = parser.parse_args()

    # Setup paths early (needed for estimate)
    lens_pack_path = Path(f"lens_packs/{args.lens_pack}")

    if args.layers_dir:
        layers_dir = Path(args.layers_dir)
    else:
        # Try to find from pack_info
        pack_info_path = lens_pack_path / "pack_info.json"
        if pack_info_path.exists():
            with open(pack_info_path) as f:
                pack_info = json.load(f)
            source_pack = pack_info.get("source_pack", "first-light")
            layers_dir = Path(f"concept_packs/{source_pack}/hierarchy")
            concept_pack_path = Path(f"concept_packs/{source_pack}")
        else:
            layers_dir = Path("concept_packs/first-light/hierarchy")
            concept_pack_path = Path("concept_packs/first-light")

    # Load definitions early (needed for prompt generation and estimate)
    definitions = load_concept_definitions(layers_dir)

    if args.max_concepts:
        definitions = dict(list(definitions.items())[:args.max_concepts])

    # Determine generation prompts based on mode
    gen_prompt_source = "default"
    if args.all_concepts:
        gen_prompts = generate_concept_prompts(definitions)
        gen_prompt_source = "all_concepts"
    elif args.all_families:
        hierarchy = load_hierarchy(concept_pack_path)
        gen_prompts = generate_family_prompts(hierarchy, definitions)
        gen_prompt_source = "all_families"
    else:
        gen_prompts = GENERATION_CALIBRATION_PROMPTS[:args.gen_prompts]
        if args.gen_prompts > len(GENERATION_CALIBRATION_PROMPTS):
            gen_prompts = gen_prompts * (args.gen_prompts // len(gen_prompts) + 1)
            gen_prompts = gen_prompts[:args.gen_prompts]

    n_gen_prompts = len(gen_prompts)
    total_tokens = n_gen_prompts * args.gen_tokens

    print("=" * 80)
    print("CONCEPT LENS CALIBRATION")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Lens pack: {args.lens_pack}")
    print(f"Model: {args.model}")
    print(f"Firing threshold: {args.firing_threshold}")
    print(f"Total concepts: {len(definitions)}")
    if args.mode in ["cross", "both"]:
        print(f"Cross-activation samples per concept: {args.n_samples}")
    if args.mode in ["generation", "both"]:
        print(f"Generation prompt source: {gen_prompt_source}")
        print(f"Generation prompts: {n_gen_prompts}")
        print(f"Tokens per prompt: {args.gen_tokens}")
        print(f"Total generation tokens: {total_tokens:,}")
        runtime_estimate = estimate_runtime(n_gen_prompts, args.gen_tokens)
        print(f"Estimated runtime: {runtime_estimate}")

    if args.estimate_only:
        print("\n" + "=" * 80)
        print("ESTIMATE ONLY - Not running calibration")
        print("=" * 80)
        if args.mode in ["generation", "both"]:
            print(f"\nGeneration calibration estimate:")
            print(f"  Prompts: {n_gen_prompts}")
            print(f"  Tokens per prompt: {args.gen_tokens}")
            print(f"  Total tokens: {total_tokens:,}")
            print(f"  Estimated runtime (first run): {runtime_estimate}")

            # Cache info
            cache_path = get_activation_cache_path(
                lens_pack_path, gen_prompt_source, n_gen_prompts,
                args.gen_tokens, args.model_layer
            )
            # Estimate cache size: tokens * hidden_dim * 4 bytes (float32)
            hidden_dim = 2560  # gemma-3-4b default, adjust for other models
            cache_size_mb = (total_tokens * hidden_dim * 4) / (1024 * 1024)

            print(f"\nCache info:")
            print(f"  Cache path: {cache_path}")
            print(f"  Cache exists: {cache_path.exists()}")
            print(f"  Estimated cache size: {cache_size_mb:.0f} MB")
            if cache_path.exists():
                print(f"  → Will use cached activations (instant)")
            else:
                print(f"  → Will generate and cache for reuse")

            print(f"\nSample prompts (first 5):")
            for i, p in enumerate(gen_prompts[:5]):
                print(f"  {i+1}. {p[:80]}{'...' if len(p) > 80 else ''}")
        return

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    print("✓ Model loaded")

    print(f"Layers dir: {layers_dir}")
    print(f"✓ Loaded definitions for {len(definitions)} concepts")

    # Initialize result structure
    result = {
        "timestamp": datetime.now().isoformat(),
        "lens_pack": args.lens_pack,
        "model": args.model,
        "mode": args.mode,
        "firing_threshold": args.firing_threshold,
        "calibration": {},
    }

    # Run cross-activation calibration
    cross_calibration = {}
    if args.mode in ["cross", "both"]:
        cross_calibration = run_cross_activation_calibration(
            lens_pack_path=lens_pack_path,
            model=model,
            tokenizer=tokenizer,
            definitions=definitions,
            n_samples_per_concept=args.n_samples,
            firing_threshold=args.firing_threshold,
            device=args.device,
        )
        result["n_samples_per_concept"] = args.n_samples

    # Run generation calibration
    gen_calibration = {}
    if args.mode in ["generation", "both"]:
        # Load lenses for generation calibration
        print("\nLoading lenses for generation calibration...")
        all_concept_keys = list(definitions.keys())
        lenses = load_all_lenses_to_cpu(lens_pack_path, all_concept_keys)
        print(f"  Loaded {len(lenses)} lenses")

        # Use pre-computed prompts (from --all-concepts, --all-families, or default)
        gen_calibration = run_generation_calibration(
            lens_pack_path=lens_pack_path,
            model=model,
            tokenizer=tokenizer,
            lenses=lenses,
            generation_prompts=gen_prompts,
            n_tokens_per_prompt=args.gen_tokens,
            firing_threshold=args.firing_threshold,
            model_layer=args.model_layer,
            device=args.device,
            prompt_source=gen_prompt_source,
            model_name=args.model,
            use_cache=not args.no_cache,
        )
        result["gen_prompt_source"] = gen_prompt_source
        result["gen_prompts"] = n_gen_prompts
        result["gen_tokens"] = args.gen_tokens
        result["cache_enabled"] = not args.no_cache

    # Merge calibration data
    # For each concept, combine cross and generation stats
    all_concepts = set(cross_calibration.keys()) | set(gen_calibration.keys())

    for key in all_concepts:
        cross_data = cross_calibration.get(key, {})
        gen_data = gen_calibration.get(key, {})

        merged = {}
        # Copy cross-activation fields
        for field in ["concept", "layer", "self_mean", "self_std", "cross_mean", "cross_std",
                      "cross_fire_count", "total_other_prompts", "cross_fire_rate",
                      "n_self_samples", "n_cross_samples"]:
            if field in cross_data:
                merged[field] = cross_data[field]

        # Copy generation fields
        for field in ["gen_fire_rate", "gen_fire_count", "total_gen_tokens", "gen_mean", "gen_std"]:
            if field in gen_data:
                merged[field] = gen_data[field]

        # Ensure concept and layer are set
        if "concept" not in merged:
            merged["concept"] = gen_data.get("concept", key.rsplit("_L", 1)[0])
        if "layer" not in merged:
            merged["layer"] = gen_data.get("layer", int(key.rsplit("_L", 1)[1]))

        result["calibration"][key] = merged

    result["total_concepts_calibrated"] = len(result["calibration"])

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Summary stats
    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Total concepts calibrated: {len(result['calibration'])}")

    calibration = result["calibration"]

    # Cross-activation stats
    if args.mode in ["cross", "both"]:
        by_cross_rate = sorted(
            [c for c in calibration.values() if "cross_fire_rate" in c],
            key=lambda x: x["cross_fire_rate"],
            reverse=True
        )
        print("\nTop 20 over-firers (by cross-fire rate):")
        for i, c in enumerate(by_cross_rate[:20]):
            print(f"  {i+1:2d}. {c['concept']:40s} L{c['layer']} "
                  f"cross_rate={c['cross_fire_rate']:.3f} "
                  f"self={c.get('self_mean', 0):.2f} cross={c.get('cross_mean', 0):.2f}")

    # Generation stats
    if args.mode in ["generation", "both"]:
        by_gen_rate = sorted(
            [c for c in calibration.values() if "gen_fire_rate" in c],
            key=lambda x: x["gen_fire_rate"],
            reverse=True
        )
        print("\nTop 20 over-firers (by generation fire rate):")
        for i, c in enumerate(by_gen_rate[:20]):
            print(f"  {i+1:2d}. {c['concept']:40s} L{c['layer']} "
                  f"gen_rate={c['gen_fire_rate']:.3f} "
                  f"gen_mean={c.get('gen_mean', 0):.2f}")

        # Distribution of gen_fire_rate
        rates = [c["gen_fire_rate"] for c in calibration.values() if "gen_fire_rate" in c]
        if rates:
            print(f"\nGeneration fire rate distribution:")
            thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
            for thresh in thresholds:
                count = sum(1 for r in rates if r > thresh)
                print(f"  >{thresh:.0%}: {count} concepts ({100*count/len(rates):.1f}%)")

    print(f"\n✓ Saved calibration to {output_path}")


if __name__ == "__main__":
    main()
