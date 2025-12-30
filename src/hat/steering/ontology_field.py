"""
Ontology-aware steering field computation.

Automatically derives attraction/repulsion weights from concept pack hierarchy,
using graph distance to weight the steering field.

Supports contrastive steering by automatically finding reference concepts:
1. Aunts/Uncles (parent's siblings) - RECOMMENDED: different branch, same level
2. Cousins (children of aunts/uncles) - intermediate option
3. Siblings (same parent) - may be too similar depending on ontology granularity
4. Graph-distant concepts - fallback when family not available

Key insight: Siblings can be too similar (e.g., Cat→Lion is still feline,
Manipulation→Exploitation is still harmful). Aunts/uncles represent genuinely
different categories at the same abstraction level.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import deque
import numpy as np

from .hooks import compute_steering_field, compute_contrastive_vector
from .extraction import extract_concept_vector


def load_hierarchy(concept_pack_path: Path) -> Dict[str, Dict]:
    """
    Load hierarchy from concept pack.

    Returns:
        Dict mapping concept_name -> {layer, parent_concepts, category_children, ...}
    """
    hierarchy_dir = concept_pack_path / "hierarchy"
    if not hierarchy_dir.exists():
        raise ValueError(f"Hierarchy not found at {hierarchy_dir}")

    concepts = {}
    for layer_file in sorted(hierarchy_dir.glob("layer*.json")):
        with open(layer_file) as f:
            layer_data = json.load(f)

        layer_num = int(layer_file.stem.replace("layer", ""))
        for concept in layer_data.get("concepts", []):
            name = concept.get("sumo_term") or concept.get("name")
            if name:
                concepts[name] = {
                    "layer": layer_num,
                    "parents": concept.get("parent_concepts", []),
                    "children": concept.get("category_children", []),
                    "siblings": concept.get("siblings", []),
                }

    return concepts


def load_steering_targets(concept_pack_path: Path) -> Dict[str, Dict]:
    """
    Load curated steering targets from concept pack.

    Steering targets are manually curated mappings from sensitive concepts
    to their ideal steering references. These take priority over automatic
    family-tree based selection.

    Returns:
        Dict mapping concept_name -> {target, rationale}
    """
    hierarchy_file = concept_pack_path / "hierarchy.json"
    if not hierarchy_file.exists():
        return {}

    with open(hierarchy_file) as f:
        data = json.load(f)

    return data.get("steering_targets", {})


# Keywords that indicate a concept should have a curated steering target
SENSITIVE_CONCEPT_KEYWORDS = [
    'deception', 'manipulation', 'exploit', 'coercion', 'abuse',
    'malicious', 'fraud', 'attack', 'harm', 'threat', 'deceiv',
]

# Exceptions: concepts containing these aren't actually sensitive
SENSITIVE_CONCEPT_EXCEPTIONS = [
    'harmattan',  # A type of wind, not harm
    'pharmaceutical',  # Medicine, not manipulation
    'heartattack',  # Medical condition
    'explorationexploitation',  # ML term (explore/exploit tradeoff)
    'exploitationmode',  # ML term
    'threatassessment',  # Security assessment, not threat
    'threateningweather',  # Weather pattern
    'lifethreatening',  # Medical severity description
    'styleharmonization',  # Editing term
    'colorblind',  # Accessibility term
    'standoffattack',  # Military doctrine, not harmful action
    'terminalattack',  # Military doctrine
    'vehicleattack',  # Military doctrine
    'knifeattack',  # Historical/military
    'incendiary',  # Military doctrine
    'missile',  # Military hardware (not an action)
    'ambush',  # Military doctrine
    'stereotypethreat',  # Psychology term
    'ctnwinfra',  # Military network infrastructure term
    'reconnaissance',  # Military/security assessment
]


def is_sensitive_concept(concept_name: str) -> bool:
    """
    Check if a concept is sensitive and should have a curated steering target.

    Sensitive concepts are those that, if steered poorly, could result in
    harmful outputs. For these concepts, predictable curated targets are
    essential rather than relying on auto-selection.

    Excludes false positives like HarmattanWind, PharmaceuticalProduct, etc.
    """
    name_lower = concept_name.lower()

    # Check for exceptions first
    for exception in SENSITIVE_CONCEPT_EXCEPTIONS:
        if exception in name_lower:
            return False

    # Check for sensitive keywords
    return any(kw in name_lower for kw in SENSITIVE_CONCEPT_KEYWORDS)


def validate_steering_targets(
    concept_pack_path: Path,
    warn_only: bool = True,
) -> List[str]:
    """
    Validate that all sensitive concepts have curated steering targets.

    This helps ensure predictable behavior for safety-critical steering.

    Args:
        concept_pack_path: Path to concept pack
        warn_only: If True, just return list of missing. If False, raise ValueError.

    Returns:
        List of sensitive concepts missing curated targets
    """
    hierarchy_file = concept_pack_path / "hierarchy.json"
    if not hierarchy_file.exists():
        return []

    with open(hierarchy_file) as f:
        data = json.load(f)

    all_concepts = set(data.get("child_to_parent", {}).keys())
    steering_targets = set(data.get("steering_targets", {}).keys())

    missing = []
    for concept in all_concepts:
        if is_sensitive_concept(concept) and concept not in steering_targets:
            missing.append(concept)

    if missing and not warn_only:
        raise ValueError(
            f"Sensitive concepts without curated steering targets: {missing[:10]}... "
            f"({len(missing)} total). Add targets to hierarchy.json steering_targets."
        )

    return sorted(missing)


def compute_graph_distances(
    hierarchy: Dict[str, Dict],
    source_concept: str,
    max_distance: int = 10,
) -> Dict[str, int]:
    """
    Compute shortest path distances from source concept to all others.

    Uses BFS over parent/child/sibling edges.

    Returns:
        Dict mapping concept_name -> distance from source
    """
    if source_concept not in hierarchy:
        return {}

    distances = {source_concept: 0}
    queue = deque([source_concept])

    while queue:
        current = queue.popleft()
        current_dist = distances[current]

        if current_dist >= max_distance:
            continue

        current_data = hierarchy.get(current, {})

        # Explore neighbors: parents, children, siblings
        neighbors = set()
        neighbors.update(current_data.get("parents", []))
        neighbors.update(current_data.get("children", []))
        neighbors.update(current_data.get("siblings", []))

        for neighbor in neighbors:
            if neighbor in hierarchy and neighbor not in distances:
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)

    return distances


def select_field_concepts(
    hierarchy: Dict[str, Dict],
    target_concept: str,
    mode: str = "repel",
    n_attract: int = 20,
    n_repel: int = 5,
    min_attract_distance: int = 3,
) -> Tuple[List[str], List[str], Dict[str, float], Dict[str, float]]:
    """
    Select concepts for attraction/repulsion field based on graph distance.

    Args:
        hierarchy: Concept hierarchy from load_hierarchy()
        target_concept: The concept to steer away from (repel) or towards (attract)
        mode: "repel" (steer away from target) or "attract" (steer towards target)
        n_attract: Number of attraction concepts to select
        n_repel: Number of repulsion concepts to select
        min_attract_distance: Minimum graph distance for attraction concepts

    Returns:
        (attract_concepts, repel_concepts, attract_weights, repel_weights)
    """
    distances = compute_graph_distances(hierarchy, target_concept)

    if not distances:
        raise ValueError(f"Concept {target_concept} not found in hierarchy")

    if mode == "repel":
        # Repel from target: attract distant concepts, repel from target + neighbors

        # Repulsion: target + close neighbors (siblings, direct relatives)
        repel_concepts = []
        repel_weights = {}
        for concept, dist in sorted(distances.items(), key=lambda x: x[1]):
            if len(repel_concepts) >= n_repel:
                break
            repel_concepts.append(concept)
            # Closer = higher repulsion weight
            repel_weights[concept] = 1.0 / (1.0 + dist)

        # Attraction: distant concepts
        attract_concepts = []
        attract_weights = {}
        distant = [(c, d) for c, d in distances.items() if d >= min_attract_distance]
        # Sort by distance descending, take furthest
        distant.sort(key=lambda x: -x[1])

        for concept, dist in distant[:n_attract]:
            attract_concepts.append(concept)
            # Further = higher attraction weight (normalized by distance)
            attract_weights[concept] = dist / 10.0  # Scale to reasonable range

        # Also add concepts NOT in distances (unreachable = very distant)
        unreachable = [c for c in hierarchy.keys() if c not in distances]
        for concept in unreachable[:max(0, n_attract - len(attract_concepts))]:
            attract_concepts.append(concept)
            attract_weights[concept] = 1.0  # Max weight for unreachable

    else:  # mode == "attract"
        # Attract to target: attract target + neighbors, repel distant concepts

        # Attraction: target + close neighbors
        attract_concepts = []
        attract_weights = {}
        for concept, dist in sorted(distances.items(), key=lambda x: x[1]):
            if len(attract_concepts) >= n_attract:
                break
            attract_concepts.append(concept)
            attract_weights[concept] = 1.0 / (1.0 + dist)

        # Repulsion: distant concepts
        repel_concepts = []
        repel_weights = {}
        distant = [(c, d) for c, d in distances.items() if d >= min_attract_distance]
        distant.sort(key=lambda x: -x[1])

        for concept, dist in distant[:n_repel]:
            repel_concepts.append(concept)
            repel_weights[concept] = dist / 10.0

    return attract_concepts, repel_concepts, attract_weights, repel_weights


def find_contrastive_references(
    hierarchy: Dict[str, Dict],
    target_concept: str,
    n_siblings: int = 3,
    n_aunts_uncles: int = 3,
    n_cousins: int = 3,
    n_distant: int = 5,
    min_distant: int = 4,
) -> Dict[str, List[str]]:
    """
    Find reference concepts for contrastive steering at multiple granularities.

    Priority order (coarser = safer for steering):
    1. Aunts/Uncles (parent's siblings) - different branch, same grandparent
    2. Cousins (children of aunts/uncles) - one level down from aunts/uncles
    3. Siblings (same parent) - may be too similar depending on ontology granularity
    4. Graph-distant concepts - fallback

    Why aunts/uncles over siblings?
    - If parent is "Feline", siblings are other cats (Cheetah, Lion) - still cats!
    - Aunts/uncles would be "Canine", "Rodent" - actually different animals
    - If parent is "Deceptive Harm", siblings are other deceptive harms - still harmful!
    - Aunts/uncles would be "Physical Harm" - different harm category

    Args:
        hierarchy: Concept hierarchy from load_hierarchy()
        target_concept: The concept to steer away from
        n_siblings: Max siblings to return
        n_aunts_uncles: Max aunts/uncles to return
        n_cousins: Max cousins to return
        n_distant: Max distant concepts to return
        min_distant: Minimum graph distance for "distant" concepts

    Returns:
        Dict with keys: "siblings", "aunts_uncles", "cousins", "distant"
    """
    result = {
        "siblings": [],
        "aunts_uncles": [],
        "cousins": [],
        "distant": [],
    }

    if target_concept not in hierarchy:
        return result

    target_data = hierarchy[target_concept]
    distances = compute_graph_distances(hierarchy, target_concept)

    # 1. Find siblings (same parent)
    siblings = set()
    direct_siblings = target_data.get("siblings", [])
    for sib in direct_siblings:
        if sib in hierarchy and sib != target_concept:
            siblings.add(sib)

    parents = target_data.get("parents", [])
    for parent in parents:
        if parent in hierarchy:
            parent_children = hierarchy[parent].get("children", [])
            for child in parent_children:
                if child in hierarchy and child != target_concept:
                    siblings.add(child)

    result["siblings"] = list(siblings)[:n_siblings]

    # 2. Find aunts/uncles (parent's siblings)
    aunts_uncles = set()
    for parent in parents:
        if parent not in hierarchy:
            continue
        parent_data = hierarchy[parent]

        # Parent's direct siblings
        for aunt_uncle in parent_data.get("siblings", []):
            if aunt_uncle in hierarchy and aunt_uncle != parent:
                aunts_uncles.add(aunt_uncle)

        # Parent's parent's other children (grandparent's children excluding parent)
        grandparents = parent_data.get("parents", [])
        for grandparent in grandparents:
            if grandparent in hierarchy:
                gp_children = hierarchy[grandparent].get("children", [])
                for gp_child in gp_children:
                    if gp_child in hierarchy and gp_child != parent:
                        aunts_uncles.add(gp_child)

    result["aunts_uncles"] = list(aunts_uncles)[:n_aunts_uncles]

    # 3. Find cousins (children of aunts/uncles)
    cousins = set()
    for aunt_uncle in aunts_uncles:
        if aunt_uncle in hierarchy:
            au_children = hierarchy[aunt_uncle].get("children", [])
            for cousin in au_children:
                if cousin in hierarchy and cousin != target_concept and cousin not in siblings:
                    cousins.add(cousin)

    result["cousins"] = list(cousins)[:n_cousins]

    # 4. Find distant concepts (graph distance >= min_distant)
    all_relatives = siblings | aunts_uncles | cousins
    distant_candidates = [
        (c, d) for c, d in distances.items()
        if d >= min_distant and c not in all_relatives
    ]
    distant_candidates.sort(key=lambda x: -x[1])  # Furthest first

    result["distant"] = [c for c, _ in distant_candidates[:n_distant]]

    return result


def select_best_reference(
    hierarchy: Dict[str, Dict],
    target_concept: str,
    model=None,
    tokenizer=None,
    device: str = "cuda",
    layer_idx: int = -1,
    prefer_coarse: bool = True,
) -> Tuple[Optional[str], str]:
    """
    Automatically select the best reference concept for contrastive steering.

    Priority order (when prefer_coarse=True):
    1. Aunts/Uncles - different branch of tree, safer for steering
    2. Cousins - children of aunts/uncles
    3. Siblings - may be too similar (same parent category)
    4. Distant - fallback

    Args:
        hierarchy: Concept hierarchy
        target_concept: Concept to steer away from
        model: Language model (optional, for vector extraction)
        tokenizer: Tokenizer (optional)
        device: Device for computation
        layer_idx: Layer for vector extraction
        prefer_coarse: If True, prefer aunts/uncles over siblings (recommended)

    Returns:
        (reference_concept_name, source) where source indicates relationship
    """
    refs = find_contrastive_references(hierarchy, target_concept)

    aunts_uncles = refs["aunts_uncles"]
    cousins = refs["cousins"]
    siblings = refs["siblings"]
    distant = refs["distant"]

    all_refs = aunts_uncles + cousins + siblings + distant
    if not all_refs:
        return None, "none"

    # If no model provided, return first available by priority
    if model is None:
        if prefer_coarse:
            if aunts_uncles:
                return aunts_uncles[0], "aunt_uncle"
            if cousins:
                return cousins[0], "cousin"
        if siblings:
            return siblings[0], "sibling"
        if distant:
            return distant[0], "distant"
        return None, "none"

    # Extract target vector
    try:
        target_vec = extract_concept_vector(model, tokenizer, target_concept, layer_idx, device)
    except Exception:
        # Can't extract target, return first available
        if aunts_uncles:
            return aunts_uncles[0], "aunt_uncle"
        if siblings:
            return siblings[0], "sibling"
        return distant[0] if distant else None, "distant" if distant else "none"

    # Find best in each category
    min_magnitude = 0.15  # Minimum to accept

    def find_best_in_list(concepts: List[str]) -> Tuple[Optional[str], float]:
        best = None
        best_mag = 0.0
        for concept in concepts:
            try:
                ref_vec = extract_concept_vector(model, tokenizer, concept, layer_idx, device)
                _, mag = compute_contrastive_vector(ref_vec, target_vec)
                if mag > best_mag:
                    best_mag = mag
                    best = concept
            except Exception:
                continue
        return best, best_mag

    best_au, mag_au = find_best_in_list(aunts_uncles)
    best_cousin, mag_cousin = find_best_in_list(cousins)
    best_sibling, mag_sibling = find_best_in_list(siblings)
    best_distant, mag_distant = find_best_in_list(distant)

    # Decision logic - prefer coarser relationships for safer steering
    if prefer_coarse:
        # Aunts/uncles first (different branch of tree)
        if best_au and mag_au >= min_magnitude:
            return best_au, "aunt_uncle"
        # Cousins next
        if best_cousin and mag_cousin >= min_magnitude:
            return best_cousin, "cousin"
        # Siblings only if aunts/uncles/cousins unavailable
        if best_sibling and mag_sibling >= min_magnitude:
            return best_sibling, "sibling"
        # Distant as fallback
        if best_distant and mag_distant >= min_magnitude:
            return best_distant, "distant"
    else:
        # Original logic - prefer highest magnitude
        candidates = [
            (best_sibling, mag_sibling, "sibling"),
            (best_au, mag_au, "aunt_uncle"),
            (best_cousin, mag_cousin, "cousin"),
            (best_distant, mag_distant, "distant"),
        ]
        candidates = [(c, m, s) for c, m, s in candidates if c and m >= min_magnitude]
        if candidates:
            candidates.sort(key=lambda x: -x[1])
            return candidates[0][0], candidates[0][2]

    # Fallback - return anything with magnitude
    for concept, mag, source in [
        (best_au, mag_au, "aunt_uncle"),
        (best_cousin, mag_cousin, "cousin"),
        (best_sibling, mag_sibling, "sibling"),
        (best_distant, mag_distant, "distant"),
    ]:
        if concept:
            return concept, source

    return None, "none"


def build_contrastive_steering_vector(
    model,
    tokenizer,
    concept_pack_path: Path,
    target_concept: str,
    reference_concept: Optional[str] = None,
    layer_idx: int = -1,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[np.ndarray, float, str]:
    """
    Build contrastive steering vector automatically from concept pack.

    Reference selection priority:
    1. Explicit reference_concept if provided
    2. Curated steering_targets from hierarchy.json (for sensitive concepts)
    3. Auto-selection via family tree (aunts/uncles > cousins > siblings > distant)

    Args:
        model: Language model
        tokenizer: Tokenizer
        concept_pack_path: Path to concept pack
        target_concept: Concept to steer away from
        reference_concept: Explicit reference (optional, auto-selected if None)
        layer_idx: Layer for vector extraction
        device: Device for computation
        verbose: Print progress

    Returns:
        (contrastive_vector, magnitude, reference_used)
        - contrastive_vector: Normalized vector orthogonal to target
        - magnitude: Distinctiveness (higher = more distinguishing features)
        - reference_used: Name of reference concept used
    """
    hierarchy = load_hierarchy(concept_pack_path)
    steering_targets = load_steering_targets(concept_pack_path)

    if verbose:
        print(f"Building contrastive vector for '{target_concept}'")

    # Extract target vector
    target_vec = extract_concept_vector(model, tokenizer, target_concept, layer_idx, device)

    # Reference selection priority
    if reference_concept is not None:
        # 1. Explicit reference provided
        source = "explicit"
        if verbose:
            print(f"  Using explicit reference: '{reference_concept}'")
    elif target_concept in steering_targets:
        # 2. Curated steering target for sensitive concepts
        curated = steering_targets[target_concept]
        reference_concept = curated["target"]
        source = "curated"
        if verbose:
            print(f"  Using curated target: '{reference_concept}'")
            print(f"    Rationale: {curated.get('rationale', 'N/A')}")
    else:
        # 3. Auto-select via family tree navigation
        # Warn if this is a sensitive concept without curated target
        if is_sensitive_concept(target_concept):
            import warnings
            warnings.warn(
                f"Sensitive concept '{target_concept}' has no curated steering target. "
                "Auto-selection may produce unpredictable results. "
                "Consider adding a target to hierarchy.json steering_targets.",
                UserWarning
            )
        reference_concept, source = select_best_reference(
            hierarchy, target_concept, model, tokenizer, device, layer_idx
        )
        if verbose:
            print(f"  Auto-selected reference: '{reference_concept}' (source: {source})")

    if reference_concept is None:
        raise ValueError(f"Could not find reference concept for '{target_concept}'")

    # Extract reference vector
    ref_vec = extract_concept_vector(model, tokenizer, reference_concept, layer_idx, device)

    # Compute contrastive vector (what makes reference different from target)
    # We want to steer AWAY from target, so we compute target-not-reference
    # and use negative strength, OR compute reference-not-target and use positive
    contrastive, magnitude = compute_contrastive_vector(ref_vec, target_vec)

    if verbose:
        cosine = np.dot(target_vec, ref_vec)
        print(f"  Cosine similarity: {cosine:.3f}")
        print(f"  Contrastive magnitude: {magnitude:.3f}")

    return contrastive, magnitude, reference_concept


def steer_away_from_concept(
    model,
    tokenizer,
    prompt: str,
    concept_pack_path: Path,
    target_concept: str,
    reference_concept: Optional[str] = None,
    strength: float = 3.0,
    layer_idx: int = -1,
    max_new_tokens: int = 50,
    device: str = "cuda",
    verbose: bool = False,
    **generation_kwargs
) -> Tuple[str, dict]:
    """
    High-level function to steer generation away from a concept.

    Automatically selects best reference from hierarchy and applies
    contrastive steering. This is the recommended entry point.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Text prompt
        concept_pack_path: Path to concept pack
        target_concept: Concept to steer away from
        reference_concept: Explicit reference (auto-selected if None)
        strength: Steering strength (3.0 recommended for contrastive)
        layer_idx: Layer for steering (-1 for last)
        max_new_tokens: Max tokens to generate
        device: Device
        verbose: Print debug info
        **generation_kwargs: Passed to model.generate()

    Returns:
        (generated_text, metadata) where metadata includes reference used, magnitude, etc.

    Example:
        >>> text, meta = steer_away_from_concept(
        ...     model, tokenizer,
        ...     prompt="What animal goes meow?",
        ...     concept_pack_path=Path("concept_packs/first-light"),
        ...     target_concept="DomesticCat",
        ...     strength=3.0
        ... )
        >>> print(text)  # Will mention alternatives to cat
        >>> print(meta["reference"])  # Shows auto-selected reference
    """
    import torch
    from .hooks import create_contrastive_steering_hook

    # Build contrastive vector
    contrastive, magnitude, ref_used = build_contrastive_steering_vector(
        model, tokenizer, concept_pack_path, target_concept,
        reference_concept=reference_concept,
        layer_idx=layer_idx,
        device=device,
        verbose=verbose,
    )

    metadata = {
        "target": target_concept,
        "reference": ref_used,
        "magnitude": magnitude,
        "strength": strength,
    }

    if magnitude < 0.1:
        import warnings
        warnings.warn(
            f"Low contrastive magnitude ({magnitude:.3f}). "
            "Target and reference may be too similar for effective steering."
        )

    # Get layers
    if hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers
    elif hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise AttributeError(f"Cannot find layers in {type(model.model)}")

    target_layer = layers[layer_idx] if layer_idx != -1 else layers[-1]

    # Apply steering
    hook = create_contrastive_steering_hook(contrastive, strength, device)
    handle = target_layer.register_forward_hook(hook)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        handle.remove()

    return generated, metadata


def build_ontology_steering_field(
    model,
    tokenizer,
    concept_pack_path: Path,
    target_concept: str,
    mode: str = "repel",
    strength: float = 1.0,
    n_attract: int = 15,
    n_repel: int = 5,
    min_attract_distance: int = 3,
    layer_idx: int = -1,
    device: str = "cuda",
    verbose: bool = True,
) -> np.ndarray:
    """
    Build a steering field automatically from concept pack hierarchy.

    Args:
        model: Language model for vector extraction
        tokenizer: Tokenizer
        concept_pack_path: Path to concept pack (e.g., concept_packs/first-light)
        target_concept: Concept to steer away from (mode=repel) or towards (mode=attract)
        mode: "repel" or "attract"
        strength: Steering field strength
        n_attract: Number of attraction concepts
        n_repel: Number of repulsion concepts
        min_attract_distance: Minimum graph distance for attraction concepts
        layer_idx: Model layer for vector extraction
        device: Device for computation
        verbose: Print progress

    Returns:
        Normalized steering field vector
    """
    if verbose:
        print(f"Building ontology steering field for '{target_concept}' (mode={mode})")

    # Load hierarchy
    hierarchy = load_hierarchy(concept_pack_path)
    if verbose:
        print(f"  Loaded {len(hierarchy)} concepts from hierarchy")

    # Select concepts based on graph distance
    attract_concepts, repel_concepts, attract_weights, repel_weights = select_field_concepts(
        hierarchy, target_concept, mode, n_attract, n_repel, min_attract_distance
    )

    if verbose:
        print(f"  Attraction concepts ({len(attract_concepts)}): {attract_concepts[:5]}...")
        print(f"  Repulsion concepts ({len(repel_concepts)}): {repel_concepts}")

    # Extract vectors
    attract_vectors = {}
    for concept in attract_concepts:
        try:
            vec = extract_concept_vector(model, tokenizer, concept, layer_idx, device)
            attract_vectors[concept] = vec
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not extract vector for {concept}: {e}")

    repel_vectors = {}
    for concept in repel_concepts:
        try:
            vec = extract_concept_vector(model, tokenizer, concept, layer_idx, device)
            repel_vectors[concept] = vec
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not extract vector for {concept}: {e}")

    if verbose:
        print(f"  Extracted {len(attract_vectors)} attraction, {len(repel_vectors)} repulsion vectors")

    # Compute field
    field = compute_steering_field(
        attract_vectors=attract_vectors,
        repel_vectors=repel_vectors,
        attract_weights=attract_weights,
        repel_weights=repel_weights,
        strength=strength,
    )

    if verbose:
        print(f"  Field norm: {np.linalg.norm(field):.4f}")

    return field
