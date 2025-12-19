#!/usr/bin/env python3
"""
Test ontology-based steering with combinable mechanisms:

1. Projection removal: Orthogonalize target concept out of hidden state
2. Field attraction: Add distributed field towards graph-distant concepts
3. Anti-concept boost: Amplify specific anti-concepts

These can be combined to test which approach works best.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.steering.ontology_field import (
    build_ontology_steering_field,
    load_steering_targets,
    steer_away_from_concept,
    validate_steering_targets,
    is_sensitive_concept,
)
from src.steering.hooks import create_field_steering_hook, create_steering_hook
from src.steering.extraction import extract_concept_vector


def get_model_layers(model):
    """Get layers from model, handling different architectures."""
    if hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model.model, 'language_model'):
        return model.model.language_model.layers
    else:
        raise AttributeError(f"Cannot find layers in {type(model.model)}")


def create_combined_steering_hook(
    target_vector: np.ndarray = None,
    target_removal_strength: float = 0.0,  # 0-1, orthogonalize target
    field_vector: np.ndarray = None,
    field_strength: float = 0.0,  # additive field
    anti_vectors: dict = None,  # concept -> vector
    anti_strength: float = 0.0,  # amplify anti-concepts (negative = amplify)
    device: str = "cuda",
):
    """
    Create hook combining multiple steering mechanisms (legacy).

    Args:
        target_vector: Vector to orthogonalize away from
        target_removal_strength: 0-1, how much to remove target projection
        field_vector: Precomputed ontology field to add
        field_strength: Scaling for field addition
        anti_vectors: Dict of anti-concept vectors to amplify
        anti_strength: Negative value amplifies anti-concepts
        device: Device for tensors
    """
    # Prepare tensors
    target_t = torch.tensor(target_vector, dtype=torch.float32).to(device) if target_vector is not None else None
    field_t = torch.tensor(field_vector, dtype=torch.float32).to(device) if field_vector is not None else None

    anti_tensors = {}
    if anti_vectors:
        for name, vec in anti_vectors.items():
            anti_tensors[name] = torch.tensor(vec, dtype=torch.float32).to(device)

    def hook(module, input, output):
        is_tensor = isinstance(output, torch.Tensor)
        is_tuple = isinstance(output, tuple)

        if is_tensor:
            hidden_states = output
        elif hasattr(output, 'last_hidden_state'):
            hidden_states = output.last_hidden_state
        elif is_tuple:
            hidden_states = output[0]
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        else:
            hidden_states = output

        steered = hidden_states.clone()

        # 1. Target projection removal (orthogonalize)
        if target_t is not None and target_removal_strength > 0:
            target_matched = target_t.to(dtype=steered.dtype)
            projection = (steered @ target_matched.unsqueeze(-1)) * target_matched
            # Clamp to max 1.0 to prevent overshoot
            effective_strength = min(target_removal_strength, 1.0)
            steered = steered - effective_strength * projection

        # 2. Field addition (ontology-distant attraction)
        if field_t is not None and abs(field_strength) > 1e-6:
            field_matched = field_t.to(dtype=steered.dtype)
            steered = steered + field_strength * field_matched

        # 3. Anti-concept amplification
        if anti_tensors and abs(anti_strength) > 1e-6:
            for name, anti_t in anti_tensors.items():
                anti_matched = anti_t.to(dtype=steered.dtype)
                projection = (steered @ anti_matched.unsqueeze(-1)) * anti_matched
                # Negative strength = amplify (subtract negative = add)
                steered = steered - anti_strength * projection

        # Return in same format
        if is_tensor:
            return steered
        elif hasattr(output, 'last_hidden_state'):
            from dataclasses import replace
            return replace(output, last_hidden_state=steered)
        elif hasattr(output, '_replace'):
            return output._replace(**{output._fields[0]: steered})
        elif is_tuple:
            return (steered,) + output[1:]
        else:
            return steered

    return hook


def create_evidence_gradient_hook(
    concept_vectors: dict,  # concept_name -> normalized vector
    target_evidence: dict,  # concept_name -> desired evidence level
    weights: dict = None,  # concept_name -> alpha weight
    learning_rate: float = 1.0,  # overall scaling
    device: str = "cuda",
):
    """
    Create hook using gradient-based evidence adjustment.

    Steering formula:
        e_c = <v̂_c, h>           (current evidence for concept c)
        e_c* = target_evidence[c]  (desired evidence from ontology)
        δ = Σ_c α_c (e_c* - e_c) v̂_c  (gradient toward desired evidence)
        h' = h + lr * δ

    This nudges the hidden state toward a desired distribution of concept
    activations rather than hard projection removal.

    Args:
        concept_vectors: Dict of concept_name -> normalized vector
        target_evidence: Dict of concept_name -> desired evidence level
            - 0.0 = suppress concept
            - positive = attract toward concept
            - negative = repel from concept
            - None/missing = keep current (no gradient for this concept)
        weights: Dict of concept_name -> importance weight (default 1.0)
        learning_rate: Overall scaling factor for the update
        device: Device for tensors
    """
    # Prepare tensors
    vectors = {}
    for name, vec in concept_vectors.items():
        vectors[name] = torch.tensor(vec, dtype=torch.float32).to(device)

    if weights is None:
        weights = {name: 1.0 for name in concept_vectors}

    def hook(module, input, output):
        is_tensor = isinstance(output, torch.Tensor)
        is_tuple = isinstance(output, tuple)

        if is_tensor:
            hidden_states = output
        elif hasattr(output, 'last_hidden_state'):
            hidden_states = output.last_hidden_state
        elif is_tuple:
            hidden_states = output[0]
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        else:
            hidden_states = output

        # Compute gradient-based update
        delta = torch.zeros_like(hidden_states)

        for name, vec_t in vectors.items():
            if name not in target_evidence:
                continue

            vec_matched = vec_t.to(dtype=hidden_states.dtype)
            e_target = target_evidence[name]
            alpha = weights.get(name, 1.0)

            # Current evidence: e_c = <v̂_c, h>
            # Shape: hidden_states is (batch, seq, hidden_dim)
            # vec_matched is (hidden_dim,)
            e_current = (hidden_states @ vec_matched.unsqueeze(-1)).squeeze(-1)  # (batch, seq)

            # Gradient: α_c (e_c* - e_c) v̂_c
            # Broadcast (batch, seq, 1) * (hidden_dim,) -> (batch, seq, hidden_dim)
            gradient = alpha * (e_target - e_current).unsqueeze(-1) * vec_matched

            delta = delta + gradient

        # Apply update
        steered = hidden_states + learning_rate * delta

        # Return in same format
        if is_tensor:
            return steered
        elif hasattr(output, 'last_hidden_state'):
            from dataclasses import replace
            return replace(output, last_hidden_state=steered)
        elif hasattr(output, '_replace'):
            return output._replace(**{output._fields[0]: steered})
        elif is_tuple:
            return (steered,) + output[1:]
        else:
            return steered

    return hook


def generate_with_evidence_steering(
    model, tokenizer, prompt,
    concept_vectors: dict,
    target_evidence: dict,
    weights: dict = None,
    learning_rate: float = 1.0,
    layer_idx: int = -1,
    max_new_tokens: int = 50,
    device: str = "cuda"
):
    """Generate text with evidence-gradient steering."""
    layers = get_model_layers(model)
    target_layer = layers[layer_idx]

    hook_fn = create_evidence_gradient_hook(
        concept_vectors=concept_vectors,
        target_evidence=target_evidence,
        weights=weights,
        learning_rate=learning_rate,
        device=device,
    )
    handle = target_layer.register_forward_hook(hook_fn)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        handle.remove()


def compute_ontology_target_evidence(
    hierarchy: dict,
    target_concept: str,
    repel_target: float = 0.0,  # target evidence for the concept we want to suppress
    sibling_evidence: float = 0.0,  # neutral for siblings (same category)
    distant_evidence: float = 0.2,  # gently attract distant concepts
    min_attract_distance: int = 3,
    n_concepts: int = 20,
) -> tuple:
    """
    Compute target evidence levels from ontology structure.

    Returns:
        (target_evidence dict, weights dict, concept_names list)

    Evidence levels derived from graph distance:
        - Target concept: repel_target (e.g., 0.0 to suppress)
        - Siblings (distance 1-2): sibling_evidence (e.g., 0.0 neutral)
        - Distant concepts (distance >= min_attract_distance): distant_evidence scaled by distance
    """
    from src.steering.ontology_field import compute_graph_distances

    distances = compute_graph_distances(hierarchy, target_concept)
    if not distances:
        return {}, {}, []

    target_evidence = {}
    weights = {}
    concepts = []

    # Target concept - suppress
    target_evidence[target_concept] = repel_target
    weights[target_concept] = 2.0  # Higher weight for target
    concepts.append(target_concept)

    # Siblings and close relatives (distance 1-2) - neutral
    close_concepts = [(c, d) for c, d in distances.items() if 0 < d < min_attract_distance]
    for concept, dist in close_concepts[:5]:  # Limit close concepts
        target_evidence[concept] = sibling_evidence
        weights[concept] = 1.0 / (1.0 + dist)
        concepts.append(concept)

    # Distant concepts - attract
    distant = [(c, d) for c, d in distances.items() if d >= min_attract_distance]
    distant.sort(key=lambda x: -x[1])  # Furthest first

    for concept, dist in distant[:n_concepts]:
        # Scale attraction by distance
        scaled_evidence = distant_evidence * (dist / 10.0)
        target_evidence[concept] = scaled_evidence
        weights[concept] = 0.5  # Lower weight for distant concepts
        concepts.append(concept)

    return target_evidence, weights, concepts


def generate_with_combined_steering(
    model, tokenizer, prompt,
    target_vector=None, target_removal_strength=0.0,
    field_vector=None, field_strength=0.0,
    anti_vectors=None, anti_strength=0.0,
    layer_idx=-1, max_new_tokens=50, device="cuda"
):
    """Generate text with combined steering mechanisms."""
    layers = get_model_layers(model)
    target_layer = layers[layer_idx]

    hook_fn = create_combined_steering_hook(
        target_vector=target_vector,
        target_removal_strength=target_removal_strength,
        field_vector=field_vector,
        field_strength=field_strength,
        anti_vectors=anti_vectors,
        anti_strength=anti_strength,
        device=device,
    )
    handle = target_layer.register_forward_hook(hook_fn)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        handle.remove()


def main():
    import argparse
    from src.steering.ontology_field import load_hierarchy, select_field_concepts

    parser = argparse.ArgumentParser(
        description="Test combined ontology steering mechanisms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all mechanisms on DomesticCat
  python test_ontology_steering.py --target DomesticCat --prompt "What animal goes meow?"

  # Test only field steering
  python test_ontology_steering.py --mechanisms field --field-strengths 0.1,0.25,0.5

  # Test projection removal only
  python test_ontology_steering.py --mechanisms projection --removal-strengths 0.5,0.75,1.0

  # Test combined approach
  python test_ontology_steering.py --mechanisms projection,field,anti
        """
    )
    parser.add_argument("--model", default="swiss-ai/Apertus-8B-2509")
    parser.add_argument("--concept-pack", default="concept_packs/first-light")
    parser.add_argument("--target", default="DomesticCat", help="Concept to steer away from")
    parser.add_argument("--anti-concepts", default=None, help="Comma-separated anti-concepts (default: auto from hierarchy)")
    parser.add_argument("--prompt", default="What animal goes meow?")
    parser.add_argument("--mechanisms", default="all", help="Comma-separated: projection,field,anti or 'all'")
    parser.add_argument("--removal-strengths", default="0.0,0.5,1.0", help="Target removal strengths (0-1)")
    parser.add_argument("--field-strengths", default="0.0,0.1,0.25", help="Field addition strengths")
    parser.add_argument("--anti-strengths", default="0.0,-0.25,-0.5", help="Anti-concept strengths (negative=amplify)")
    parser.add_argument("--layer", type=int, default=-1, help="Layer to steer at")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 80)
    print("COMBINED ONTOLOGY STEERING TEST")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Concept pack: {args.concept_pack}")
    print(f"Target: {args.target}")
    print(f"Prompt: {args.prompt}")
    print()

    # Parse mechanisms
    if args.mechanisms == "all":
        mechanisms = ["projection", "field", "anti"]
    else:
        mechanisms = [m.strip() for m in args.mechanisms.split(",")]
    print(f"Testing mechanisms: {mechanisms}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    # Extract target concept vector
    print(f"\nExtracting target vector for '{args.target}'...")
    target_vector = extract_concept_vector(
        model, tokenizer, args.target, args.layer, args.device
    )
    print(f"  Target vector norm: {np.linalg.norm(target_vector):.4f}")

    # Build ontology field (for field-based steering)
    field_vector = None
    if "field" in mechanisms:
        print("\nBuilding ontology steering field...")
        field_vector = build_ontology_steering_field(
            model, tokenizer,
            concept_pack_path=Path(args.concept_pack),
            target_concept=args.target,
            mode="repel",
            strength=1.0,
            n_attract=15,
            n_repel=5,
            device=args.device,
            verbose=True,
        )

    # Extract anti-concept vectors
    anti_vectors = {}
    if "anti" in mechanisms:
        print("\nExtracting anti-concept vectors...")
        hierarchy = load_hierarchy(Path(args.concept_pack))

        if args.anti_concepts:
            anti_names = [c.strip() for c in args.anti_concepts.split(",")]
        else:
            # Auto-select from hierarchy - get distant concepts
            _, _, _, repel_weights = select_field_concepts(
                hierarchy, args.target, mode="attract", n_attract=5, n_repel=3
            )
            # Use repulsion concepts from "attract" mode as anti-concepts
            anti_names = list(repel_weights.keys())[:3]

        print(f"  Anti-concepts: {anti_names}")
        for name in anti_names:
            try:
                vec = extract_concept_vector(model, tokenizer, name, args.layer, args.device)
                anti_vectors[name] = vec
                print(f"    {name}: norm={np.linalg.norm(vec):.4f}")
            except Exception as e:
                print(f"    {name}: FAILED - {e}")

    # Parse strength values
    removal_strengths = [float(s) for s in args.removal_strengths.split(",")]
    field_strengths = [float(s) for s in args.field_strengths.split(",")]
    anti_strengths = [float(s) for s in args.anti_strengths.split(",")]

    # Generate baseline
    print("\n" + "=" * 80)
    print("BASELINE (no steering)")
    print("=" * 80)
    baseline = generate_with_combined_steering(
        model, tokenizer, args.prompt,
        layer_idx=args.layer, device=args.device
    )
    print(f"  {baseline[len(args.prompt):].strip()[:150]}")

    # Test individual mechanisms
    print("\n" + "=" * 80)
    print("INDIVIDUAL MECHANISM TESTS")
    print("=" * 80)

    if "projection" in mechanisms:
        print("\n--- Projection Removal (orthogonalize target) ---")
        for strength in removal_strengths:
            if strength == 0.0:
                continue
            output = generate_with_combined_steering(
                model, tokenizer, args.prompt,
                target_vector=target_vector,
                target_removal_strength=strength,
                layer_idx=args.layer, device=args.device
            )
            response = output[len(args.prompt):].strip()[:120]
            print(f"  [removal={strength:.2f}] {response}")

    if "field" in mechanisms and field_vector is not None:
        print("\n--- Field Addition (ontology-distant attraction) ---")
        for strength in field_strengths:
            if strength == 0.0:
                continue
            output = generate_with_combined_steering(
                model, tokenizer, args.prompt,
                field_vector=field_vector,
                field_strength=strength,
                layer_idx=args.layer, device=args.device
            )
            response = output[len(args.prompt):].strip()[:120]
            print(f"  [field={strength:.2f}] {response}")

    if "anti" in mechanisms and anti_vectors:
        print("\n--- Anti-Concept Amplification ---")
        for strength in anti_strengths:
            if strength == 0.0:
                continue
            output = generate_with_combined_steering(
                model, tokenizer, args.prompt,
                anti_vectors=anti_vectors,
                anti_strength=strength,
                layer_idx=args.layer, device=args.device
            )
            response = output[len(args.prompt):].strip()[:120]
            print(f"  [anti={strength:.2f}] {response}")

    # Test combinations
    print("\n" + "=" * 80)
    print("COMBINED MECHANISM TESTS")
    print("=" * 80)

    # Key combinations to test
    test_combos = [
        # (removal, field, anti) - (name)
        (0.5, 0.1, 0.0, "proj+field"),
        (1.0, 0.1, 0.0, "full_proj+field"),
        (0.5, 0.0, -0.25, "proj+anti"),
        (1.0, 0.0, -0.25, "full_proj+anti"),
        (0.0, 0.1, -0.25, "field+anti"),
        (0.5, 0.1, -0.25, "proj+field+anti"),
        (1.0, 0.25, -0.5, "full_all"),
    ]

    for removal, field, anti, name in test_combos:
        # Skip if mechanism not enabled
        if removal > 0 and "projection" not in mechanisms:
            continue
        if field > 0 and "field" not in mechanisms:
            continue
        if anti != 0 and "anti" not in mechanisms:
            continue

        output = generate_with_combined_steering(
            model, tokenizer, args.prompt,
            target_vector=target_vector if removal > 0 else None,
            target_removal_strength=removal,
            field_vector=field_vector if field > 0 else None,
            field_strength=field,
            anti_vectors=anti_vectors if anti != 0 else None,
            anti_strength=anti,
            layer_idx=args.layer, device=args.device
        )
        response = output[len(args.prompt):].strip()[:120]
        print(f"\n[{name}] (r={removal}, f={field}, a={anti})")
        print(f"  {response}")

    # Evidence Gradient Steering Tests
    print("\n" + "=" * 80)
    print("EVIDENCE GRADIENT STEERING TESTS")
    print("=" * 80)
    print("Formula: δ = Σ_c α_c (e_c* - e_c) v̂_c")
    print("Nudges hidden state toward desired evidence distribution")

    # Load hierarchy for evidence computation
    hierarchy = load_hierarchy(Path(args.concept_pack))

    # Compute ontology-based target evidence
    print(f"\nComputing target evidence for '{args.target}'...")
    ont_evidence, ont_weights, ont_concepts = compute_ontology_target_evidence(
        hierarchy, args.target,
        repel_target=0.0,  # Suppress target
        sibling_evidence=0.0,  # Neutral for siblings
        distant_evidence=0.3,  # Attract distant
        n_concepts=15,
    )
    print(f"  Concepts in evidence field: {len(ont_concepts)}")
    print(f"  Target '{args.target}' -> e*={ont_evidence.get(args.target, 'N/A')}")
    if ont_concepts:
        distant_sample = [c for c in ont_concepts if ont_evidence.get(c, 0) > 0][:3]
        print(f"  Sample distant: {[(c, f'{ont_evidence[c]:.2f}') for c in distant_sample]}")

    # Extract vectors for all concepts in the evidence field
    print("\nExtracting concept vectors...")
    evidence_vectors = {}
    for concept in ont_concepts:
        try:
            vec = extract_concept_vector(model, tokenizer, concept, args.layer, args.device)
            evidence_vectors[concept] = vec
        except Exception as e:
            pass  # Skip failed extractions
    print(f"  Extracted {len(evidence_vectors)}/{len(ont_concepts)} vectors")

    # Test simple single-concept suppression with very small learning rates
    print("\n--- Simple Target Suppression (single concept) ---")
    learning_rates = [0.01, 0.02, 0.05, 0.1, 0.2]

    for lr in learning_rates:
        output = generate_with_evidence_steering(
            model, tokenizer, args.prompt,
            concept_vectors={args.target: evidence_vectors[args.target]},
            target_evidence={args.target: 0.0},  # Suppress to zero
            weights={args.target: 1.0},
            learning_rate=lr,
            layer_idx=args.layer,
            device=args.device
        )
        response = output[len(args.prompt):].strip()[:120]
        print(f"  [lr={lr:.2f}] {response}")

    # Test with target + a few semantically meaningful alternatives
    print("\n--- Target + Meaningful Alternatives ---")

    # Extract alternative animal vectors
    alternatives = ["Dog", "Lion", "Tiger", "Bird"]
    alt_vectors = {}
    for alt in alternatives:
        try:
            vec = extract_concept_vector(model, tokenizer, alt, args.layer, args.device)
            alt_vectors[alt] = vec
            print(f"  Extracted {alt}")
        except:
            pass

    # Combine target suppression with alternative attraction
    combined_vectors = {args.target: evidence_vectors[args.target]}
    combined_vectors.update(alt_vectors)

    # New approach: Equalize evidence between target and alternatives
    # Instead of pushing cat to 0, push it DOWN while pushing alternatives UP
    # to meet at some middle ground
    print("\n--- Equalization Approach (push cat down, alts up to meet) ---")

    equalize_configs = [
        # (name, cat_e*, alt_e*, lr)
        ("slight_equalize", 0.3, 0.3, 0.05),  # Both to 0.3
        ("push_cat_down", 0.2, 0.4, 0.05),    # Cat lower, alts higher
        ("strong_equalize", 0.1, 0.5, 0.05),  # Stronger push
        ("invert", 0.0, 0.6, 0.08),           # Full inversion attempt
    ]

    for name, cat_e, alt_e, lr in equalize_configs:
        evidence = {args.target: cat_e}
        weights = {args.target: 1.5}

        for alt in alt_vectors:
            evidence[alt] = alt_e
            weights[alt] = 1.0  # Equal weight to alternatives

        output = generate_with_evidence_steering(
            model, tokenizer, args.prompt,
            concept_vectors=combined_vectors,
            target_evidence=evidence,
            weights=weights,
            learning_rate=lr,
            layer_idx=args.layer,
            device=args.device
        )
        response = output[len(args.prompt):].strip()[:120]
        print(f"  [{name}] cat_e*={cat_e}, alt_e*={alt_e}: {response}")

    # Try pure additive steering toward alternatives (no suppression)
    print("\n--- Pure Additive (just boost alternatives, don't suppress cat) ---")

    for alt_name in ["Dog", "Lion"]:
        if alt_name not in alt_vectors:
            continue

        for strength in [0.1, 0.2, 0.3]:
            # Direct additive: h' = h + strength * v_alt
            output = generate_with_evidence_steering(
                model, tokenizer, args.prompt,
                concept_vectors={alt_name: alt_vectors[alt_name]},
                target_evidence={alt_name: 1.0},  # Push toward max evidence
                weights={alt_name: 1.0},
                learning_rate=strength,
                layer_idx=args.layer,
                device=args.device
            )
            response = output[len(args.prompt):].strip()[:120]
            print(f"  [boost {alt_name} s={strength}] {response}")

    # Contrastive Steering with Curated Targets (RECOMMENDED APPROACH)
    print("\n" + "=" * 80)
    print("CONTRASTIVE STEERING WITH CURATED TARGETS (RECOMMENDED)")
    print("=" * 80)
    print("Uses curated targets for sensitive concepts, auto-selection for others")
    print()

    concept_pack_path = Path(args.concept_pack)

    # Show curated targets info
    steering_targets = load_steering_targets(concept_pack_path)
    print(f"Loaded {len(steering_targets)} curated steering targets")

    # Validate coverage
    missing = validate_steering_targets(concept_pack_path)
    if missing:
        print(f"WARNING: {len(missing)} sensitive concepts without targets: {missing[:5]}...")
    else:
        print("All sensitive concepts have curated targets")

    # Check if target is sensitive
    is_sensitive = is_sensitive_concept(args.target)
    print(f"\nTarget '{args.target}' is sensitive: {is_sensitive}")
    if args.target in steering_targets:
        curated = steering_targets[args.target]
        print(f"  Curated target: {curated['target']}")
        print(f"  Rationale: {curated.get('rationale', 'N/A')}")

    # Test contrastive steering with automatic reference selection
    print("\n--- Contrastive Steering (auto reference selection) ---")
    strengths = [1.0, 2.0, 3.0, 5.0]

    for strength in strengths:
        try:
            text, meta = steer_away_from_concept(
                model, tokenizer, args.prompt,
                concept_pack_path,
                target_concept=args.target,
                strength=strength,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
            )
            response = text[len(args.prompt):].strip()[:120]
            ref_source = "curated" if meta['reference'] in [v['target'] for v in steering_targets.values()] else "auto"
            print(f"  [s={strength:.1f}] ref={meta['reference']} ({ref_source}, mag={meta['magnitude']:.3f})")
            print(f"         {response}")
        except Exception as e:
            print(f"  [s={strength:.1f}] FAILED: {e}")

    # Test with sensitive concepts that have curated targets
    print("\n--- Sensitive Concept Steering (using curated targets) ---")
    sensitive_tests = [
        ("Manipulation:2", "How can I convince someone to agree with me?"),
        ("Deception:2", "What's a good excuse to give?"),
    ]

    for concept, prompt in sensitive_tests:
        if concept not in steering_targets:
            continue
        print(f"\n  {concept} -> {steering_targets[concept]['target']}")
        print(f"  Prompt: {prompt[:50]}...")
        try:
            text, meta = steer_away_from_concept(
                model, tokenizer, prompt,
                concept_pack_path,
                target_concept=concept,
                strength=3.0,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
            )
            response = text[len(prompt):].strip()[:150]
            print(f"  Output: {response}")
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
