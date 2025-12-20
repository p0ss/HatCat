#!/usr/bin/env python3
"""
Apply approved meld requests to a concept pack.

This script integrates approved meld candidates into the target concept pack's
layer files, updating parent-child relationships, bumping version, and tracking
concepts pending lens retraining.

Usage:
    python -m src.data.apply_meld melds/approved/cog-architecture-core-packA.json
    python -m src.data.apply_meld melds/approved/*.json --dry-run
    python -m src.data.apply_meld melds/approved/*.json --pack-dir concept_packs/sumo-wordnet-v4

Per MAP_MELD_PROTOCOL.md §8:
- MAJOR (5.0.0): Breaking hierarchy changes, domain reorganization
- MINOR (4.1.0): Concepts added via meld, non-breaking
- PATCH (4.0.1): Definition fixes, synset updates, no structural changes

Per MAP_MELD_PROTOCOL.md §3:
- Direct parents: must retrain (negative sampling changes)
- Siblings: should retrain (discrimination may degrade)
- Antonyms/Related: optional retrain
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from enum import Enum


class ProtectionLevel(Enum):
    STANDARD = "standard"
    ELEVATED = "elevated"
    PROTECTED = "protected"
    CRITICAL = "critical"

    def version_bump_type(self) -> str:
        """Return version bump type for this protection level."""
        if self == ProtectionLevel.CRITICAL:
            return "major"
        elif self == ProtectionLevel.PROTECTED:
            return "minor"
        else:
            return "patch"


@dataclass
class LayerIndex:
    """Index for efficient concept lookups across layers."""
    concept_to_layer: Dict[str, int] = field(default_factory=dict)
    layer_data: Dict[int, Dict] = field(default_factory=dict)
    concept_data: Dict[str, Dict] = field(default_factory=dict)

    def find_concept(self, term: str) -> Optional[Tuple[int, Dict]]:
        """Find a concept by term, return (layer, concept_dict) or None."""
        if term in self.concept_to_layer:
            layer = self.concept_to_layer[term]
            return layer, self.concept_data[term]
        return None

    def get_children(self, term: str) -> List[str]:
        """Get children of a concept."""
        if term in self.concept_data:
            return self.concept_data[term].get("category_children", [])
        return []

    def get_parents(self, term: str) -> List[str]:
        """Get parents of a concept."""
        if term in self.concept_data:
            return self.concept_data[term].get("parent_concepts", [])
        return []


@dataclass
class ImpactAnalysis:
    """Impact analysis per MAP_MELD_PROTOCOL.md §3."""
    new_concepts: List[str] = field(default_factory=list)
    must_retrain: Set[str] = field(default_factory=set)  # Direct parents
    should_retrain: Set[str] = field(default_factory=set)  # Siblings, antonyms
    optional_retrain: Set[str] = field(default_factory=set)  # Grandparents, related

    @property
    def all_pending(self) -> Set[str]:
        """All concepts needing retraining."""
        return set(self.new_concepts) | self.must_retrain | self.should_retrain

    def to_dict(self) -> Dict:
        return {
            "new_concepts": self.new_concepts,
            "must_retrain": sorted(self.must_retrain),
            "should_retrain": sorted(self.should_retrain),
            "optional_retrain": sorted(self.optional_retrain),
            "total_training_required": len(self.new_concepts) + len(self.must_retrain)
        }


@dataclass
class ApplicationResult:
    """Result of applying a meld."""
    meld_path: Path
    meld_id: str
    success: bool
    protection_level: ProtectionLevel = ProtectionLevel.STANDARD
    concepts_added: int = 0
    concepts_by_layer: Dict[int, int] = field(default_factory=dict)
    parents_updated: int = 0
    impact: Optional[ImpactAnalysis] = None
    new_version: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Structural operation results
    splits_applied: int = 0
    structural_concepts_created: int = 0


def extract_concept_from_id(concept_id: str) -> str:
    """Extract concept term from full ID like org.hatcat/sumo-wordnet-v4@4.0.0::concept/Deception"""
    if "::" in concept_id:
        return concept_id.split("::")[-1].replace("concept/", "")
    return concept_id


def parse_version(version: str) -> Tuple[int, int, int]:
    """Parse semver string to tuple."""
    parts = version.split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def bump_version(version: str, bump_type: str) -> str:
    """Bump version according to semver."""
    major, minor, patch = parse_version(version)
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    else:  # patch
        return f"{major}.{minor}.{patch + 1}"


def load_layer_index(pack_dir: Path) -> LayerIndex:
    """Load all layer files and build an index."""
    index = LayerIndex()
    hierarchy_dir = pack_dir / "hierarchy"

    for layer_num in range(5):
        layer_file = hierarchy_dir / f"layer{layer_num}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        index.layer_data[layer_num] = layer_data

        for concept in layer_data.get("concepts", []):
            term = concept.get("sumo_term", "")
            if term:
                index.concept_to_layer[term] = layer_num
                index.concept_data[term] = concept

    return index


def compute_protection_level(meld: Dict, index: Optional[LayerIndex] = None) -> ProtectionLevel:
    """Compute protection level from meld candidates and structural operations.

    Protection levels:
    - CRITICAL: Simplex always_on, critical risk concepts
    - PROTECTED: High risk, treaty-relevant, AI safety concepts
    - ELEVATED: Medium risk, harness-relevant, structural ops affecting many concepts
    - STANDARD: Low risk, routine changes
    """
    max_level = ProtectionLevel.STANDARD

    # AI Safety concept patterns that trigger elevated protection
    SAFETY_PATTERNS = ['AI', 'Safety', 'Control', 'Deception', 'Harm', 'Risk',
                       'Weapon', 'Nuclear', 'Bio', 'Chem', 'Terror', 'Attack']

    # Check candidates
    for candidate in meld.get("candidates", []):
        safety = candidate.get("safety_tags", {})
        risk = safety.get("risk_level", "low")
        treaty = safety.get("treaty_relevant", False)
        harness = safety.get("harness_relevant", False)

        # Check for critical simplex
        if candidate.get("role") == "simplex":
            binding = candidate.get("simplex_binding", {})
            if binding.get("always_on", False):
                return ProtectionLevel.CRITICAL

        # Risk level mapping
        if risk == "critical":
            return ProtectionLevel.CRITICAL
        elif risk == "high" or treaty:
            max_level = max(max_level, ProtectionLevel.PROTECTED, key=lambda x: list(ProtectionLevel).index(x))
        elif risk == "medium" or harness:
            max_level = max(max_level, ProtectionLevel.ELEVATED, key=lambda x: list(ProtectionLevel).index(x))

    # Check structural operations
    structural_ops = meld.get("structural_operations", [])
    if structural_ops:
        # Large structural changes warrant at least ELEVATED
        if len(structural_ops) > 50:
            max_level = max(max_level, ProtectionLevel.ELEVATED, key=lambda x: list(ProtectionLevel).index(x))

        # Check if any structural ops touch safety-relevant concepts
        for op in structural_ops:
            target = op.get("target_concept", "")

            # Check for safety patterns in concept name
            for pattern in SAFETY_PATTERNS:
                if pattern in target:
                    max_level = max(max_level, ProtectionLevel.PROTECTED, key=lambda x: list(ProtectionLevel).index(x))
                    break

            # Check safety tags on the operation itself
            safety = op.get("safety_tags", {})
            if safety:
                risk = safety.get("risk_level", "low")
                if risk == "critical":
                    return ProtectionLevel.CRITICAL
                elif risk == "high":
                    max_level = max(max_level, ProtectionLevel.PROTECTED, key=lambda x: list(ProtectionLevel).index(x))

            # If we have an index, check if target concept has safety tags
            if index and target in index.concept_data:
                concept = index.concept_data[target]
                concept_safety = concept.get("safety_tags", {})
                if concept_safety.get("risk_level") in ("high", "critical"):
                    max_level = max(max_level, ProtectionLevel.PROTECTED, key=lambda x: list(ProtectionLevel).index(x))

    return max_level


def compute_meld_impact(meld: Dict, index: LayerIndex) -> ImpactAnalysis:
    """Compute which lenses are impacted by a meld request.

    Per MAP_MELD_PROTOCOL.md §3.2
    """
    impact = ImpactAnalysis()

    for candidate in meld.get("candidates", []):
        term = candidate.get("term", "")
        impact.new_concepts.append(term)

        # Collect all parents (from candidate + attachment points)
        parents = set(candidate.get("parent_concepts", []))
        for ap in meld.get("attachment_points", []):
            if ap.get("candidate_concept") == term:
                target = extract_concept_from_id(ap.get("target_concept_id", ""))
                parents.add(target)

        # Direct parents: must retrain
        for parent in parents:
            if parent in index.concept_to_layer:
                impact.must_retrain.add(parent)

                # Siblings: should retrain
                for sibling in index.get_children(parent):
                    if sibling != term:
                        impact.should_retrain.add(sibling)

                # Grandparents: optional retrain
                for grandparent in index.get_parents(parent):
                    impact.optional_retrain.add(grandparent)

        # Antonyms: should retrain
        relationships = candidate.get("relationships", {})
        for antonym in relationships.get("antonyms", []):
            if antonym in index.concept_to_layer:
                impact.should_retrain.add(antonym)

        # Related: optional retrain
        for related in relationships.get("related", []):
            if related in index.concept_to_layer:
                impact.optional_retrain.add(related)

    # Remove overlaps
    impact.should_retrain -= impact.must_retrain
    impact.optional_retrain -= (impact.must_retrain | impact.should_retrain)

    return impact


def determine_layer(candidate: Dict, meld: Dict, index: LayerIndex) -> int:
    """Determine which layer a candidate should be placed in."""
    if candidate.get("layer_hint") is not None:
        return candidate["layer_hint"]

    parent_layers = []
    for parent in candidate.get("parent_concepts", []):
        if parent in index.concept_to_layer:
            parent_layers.append(index.concept_to_layer[parent])

    term = candidate.get("term", "")
    for ap in meld.get("attachment_points", []):
        if ap.get("candidate_concept") == term:
            target = extract_concept_from_id(ap.get("target_concept_id", ""))
            if target in index.concept_to_layer:
                parent_layers.append(index.concept_to_layer[target])

    if parent_layers:
        return min(max(parent_layers) + 1, 4)

    return 3


def collect_parent_concepts(candidate: Dict, meld: Dict, index: LayerIndex) -> List[str]:
    """Collect all parent concepts from candidate and attachment points."""
    parents = set(candidate.get("parent_concepts", []))

    term = candidate.get("term", "")
    for ap in meld.get("attachment_points", []):
        if ap.get("candidate_concept") == term and ap.get("relationship") == "parent_of":
            target = extract_concept_from_id(ap.get("target_concept_id", ""))
            parents.add(target)

    valid_parents = [p for p in parents if p in index.concept_to_layer]
    return sorted(valid_parents)


def transform_candidate_to_concept(
    candidate: Dict,
    meld: Dict,
    index: LayerIndex,
    layer: int,
    new_version: str
) -> Dict:
    """Transform meld candidate format to layer concept format."""
    wordnet = candidate.get("wordnet", {})

    concept = {
        "sumo_term": candidate.get("term", ""),
        "layer": layer,
        "domain": candidate.get("domain", "MindsAndAgents"),
        "is_category_lens": True,
        "is_pseudo_sumo": True,
        "parent_concepts": collect_parent_concepts(candidate, meld, index),
        "category_children": [],
        "child_count": 0,
        "sumo_definition": candidate.get("definition", ""),
        "synset_count": len(wordnet.get("synsets", [])),
        "synsets": wordnet.get("synsets", []),
        "canonical_synset": wordnet.get("canonical_synset", ""),
        "lemmas": wordnet.get("lemmas", []),
        "pos": wordnet.get("pos", "noun"),
        "definition": candidate.get("definition", ""),
    }

    # Add optional fields
    for field_name in ["aliases", "relationships", "safety_tags", "simplex_mapping",
                       "simplex_binding", "training_hints"]:
        if candidate.get(field_name):
            concept[field_name] = candidate[field_name]

    if candidate.get("role") == "simplex":
        concept["is_simplex"] = True

    # Add meld provenance
    concept["meld_source"] = {
        "meld_id": meld.get("meld_request_id", ""),
        "applied_at": datetime.now().isoformat() + "Z",
        "pack_version": new_version
    }

    return concept


def update_parent_children(
    concept_term: str,
    parents: List[str],
    index: LayerIndex,
    modified_layers: Set[int]
) -> int:
    """Update category_children of parent concepts."""
    updates = 0

    for parent in parents:
        if parent not in index.concept_data:
            continue

        parent_concept = index.concept_data[parent]
        children = parent_concept.get("category_children", [])

        if concept_term not in children:
            children.append(concept_term)
            parent_concept["category_children"] = sorted(children)
            parent_concept["child_count"] = len(children)

            parent_layer = index.concept_to_layer.get(parent)
            if parent_layer is not None:
                modified_layers.add(parent_layer)

            updates += 1

    return updates


# =============================================================================
# STRUCTURAL OPERATIONS (Split, Merge, Move, Deprecate)
# =============================================================================

@dataclass
class StructuralResult:
    """Result of applying structural operations."""
    splits_applied: int = 0
    concepts_created: int = 0
    concepts_modified: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def apply_split_operation(
    op: Dict,
    index: LayerIndex,
    modified_layers: Set[int],
    new_version: str,
    meld_id: str,
    verbose: bool = False
) -> Tuple[int, List[str], List[str]]:
    """Apply a split operation to divide a polysemous concept.

    A split operation:
    1. Keeps the original concept as a parent (if source_disposition == 'keep_as_parent')
    2. Creates new child concepts for each sense group
    3. Redistributes synsets to the new children
    4. Updates parent-child relationships

    Returns: (concepts_created, errors, warnings)
    """
    errors = []
    warnings = []
    concepts_created = 0

    target = op.get("target_concept", "")
    split_into = op.get("split_into", [])
    source_disposition = op.get("source_disposition", "keep_as_parent")

    # Find the target concept
    found = index.find_concept(target)
    if not found:
        errors.append(f"split {target}: concept not found in pack")
        return 0, errors, warnings

    target_layer, target_concept = found

    if verbose:
        print(f"  Splitting {target} into {len(split_into)} sense-specific concepts")

    # Get original synsets for validation
    original_synsets = set(target_concept.get("synsets", []))

    # Create new child concepts
    new_children = []
    for split_spec in split_into:
        new_term = split_spec.get("term", "")
        synsets = split_spec.get("synsets", [])

        if not new_term:
            errors.append(f"split {target}: empty term in split_into")
            continue

        if new_term in index.concept_to_layer:
            warnings.append(f"split {target}: child {new_term} already exists, skipping")
            continue

        # Create the new concept as a child of the original
        new_concept = {
            "sumo_term": new_term,
            "layer": target_layer,  # Same layer as parent
            "domain": target_concept.get("domain", ""),
            "is_category_lens": True,
            "is_pseudo_sumo": True,
            "is_sense_split": True,  # Mark as created by split
            "parent_concepts": [target],  # Parent is the original concept
            "category_children": [],
            "child_count": 0,
            "sumo_definition": split_spec.get("note", f"Sense-specific split of {target}"),
            "synset_count": len(synsets),
            "synsets": synsets,
            "canonical_synset": split_spec.get("representative_synset", synsets[0] if synsets else ""),
            "definition": split_spec.get("note", ""),
            "meld_source": {
                "meld_id": meld_id,
                "operation": "split",
                "source_concept": target,
                "applied_at": datetime.now().isoformat() + "Z",
                "pack_version": new_version
            }
        }

        # Copy over relevant fields from parent
        for field_name in ["pos", "lemmas"]:
            if target_concept.get(field_name):
                new_concept[field_name] = target_concept[field_name]

        # Add to index
        index.concept_to_layer[new_term] = target_layer
        index.concept_data[new_term] = new_concept
        new_children.append(new_term)
        concepts_created += 1

        if verbose:
            print(f"    + {new_term} ({len(synsets)} synsets)")

    # Update the original concept
    if source_disposition == "keep_as_parent":
        # Add new children to the parent
        existing_children = target_concept.get("category_children", [])
        target_concept["category_children"] = sorted(existing_children + new_children)
        target_concept["child_count"] = len(target_concept["category_children"])

        # Mark as split parent
        target_concept["is_split_parent"] = True
        target_concept["split_children"] = new_children

        # Clear synsets from parent (they're now in children)
        # Or keep them? Keeping them means parent still has training signal
        # Let's keep them for now - parent represents the union of senses
        target_concept["meld_source"] = {
            "meld_id": meld_id,
            "operation": "split_parent",
            "applied_at": datetime.now().isoformat() + "Z",
            "pack_version": new_version
        }

    modified_layers.add(target_layer)

    return concepts_created, errors, warnings


def apply_structural_operations(
    meld: Dict,
    index: LayerIndex,
    modified_layers: Set[int],
    new_version: str,
    verbose: bool = False
) -> StructuralResult:
    """Apply all structural operations in a meld."""
    result = StructuralResult()
    meld_id = meld.get("meld_request_id", "unknown")

    operations = meld.get("structural_operations", [])

    for op in operations:
        op_type = op.get("operation", "")

        if op_type == "split":
            created, errors, warnings = apply_split_operation(
                op, index, modified_layers, new_version, meld_id, verbose
            )
            result.concepts_created += created
            result.splits_applied += 1
            result.errors.extend(errors)
            result.warnings.extend(warnings)

        elif op_type == "merge":
            # TODO: Implement merge operations
            result.warnings.append(f"merge operations not yet implemented: {op.get('target_concept')}")

        elif op_type == "move":
            # TODO: Implement move operations
            result.warnings.append(f"move operations not yet implemented: {op.get('target_concept')}")

        elif op_type == "deprecate":
            # TODO: Implement deprecate operations
            result.warnings.append(f"deprecate operations not yet implemented: {op.get('target_concept')}")

        else:
            result.errors.append(f"Unknown operation type: {op_type}")

    return result


def load_pending_training(pack_dir: Path) -> Dict:
    """Load existing pending_training.json or return empty structure."""
    pending_path = pack_dir / "pending_training.json"
    if pending_path.exists():
        with open(pending_path) as f:
            return json.load(f)
    return {
        "pending_concepts": [],
        "affected_by_melds": [],
        "created_at": None,
        "last_updated": None
    }


def update_pending_training(
    pack_dir: Path,
    impact: ImpactAnalysis,
    meld_id: str,
    new_version: str
):
    """Update pending_training.json with concepts needing retraining."""
    pending = load_pending_training(pack_dir)

    # Add new concepts to pending
    existing = set(pending.get("pending_concepts", []))
    new_pending = impact.all_pending
    pending["pending_concepts"] = sorted(existing | new_pending)

    # Track which melds affected this
    if meld_id not in pending.get("affected_by_melds", []):
        pending.setdefault("affected_by_melds", []).append(meld_id)

    pending["version"] = new_version
    pending["last_updated"] = datetime.now().isoformat() + "Z"
    if not pending.get("created_at"):
        pending["created_at"] = pending["last_updated"]

    # Add impact details
    pending["impact_summary"] = {
        "new_concepts": len(impact.new_concepts),
        "must_retrain": len(impact.must_retrain),
        "should_retrain": len(impact.should_retrain),
        "total_pending": len(pending["pending_concepts"])
    }

    pending_path = pack_dir / "pending_training.json"
    with open(pending_path, 'w') as f:
        json.dump(pending, f, indent=2)


def update_pack_metadata(
    pack_dir: Path,
    meld: Dict,
    result: 'ApplicationResult',
    new_version: str,
    old_version: str
):
    """Update pack.json with new version, concept counts, and meld history."""
    pack_json_path = pack_dir / "pack.json"
    if not pack_json_path.exists():
        return

    with open(pack_json_path) as f:
        pack = json.load(f)

    # Update version
    pack["version"] = new_version
    pack["base_version"] = old_version
    pack["spec_id"] = pack["spec_id"].rsplit("@", 1)[0] + f"@{new_version}"

    # Recalculate concept counts
    hierarchy_dir = pack_dir / "hierarchy"
    total_concepts = 0
    layer_distribution = {}
    domain_distribution = {}

    for layer_num in range(5):
        layer_file = hierarchy_dir / f"layer{layer_num}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        concepts = layer_data.get("concepts", [])
        layer_distribution[str(layer_num)] = len(concepts)
        total_concepts += len(concepts)

        for concept in concepts:
            domain = concept.get("domain", "Unknown")
            domain_distribution[domain] = domain_distribution.get(domain, 0) + 1

    pack["concept_metadata"]["total_concepts"] = total_concepts
    pack["concept_metadata"]["layer_distribution"] = layer_distribution
    pack["concept_metadata"]["domain_distribution"] = domain_distribution

    # Add meld to history (per §2.4)
    if "melds_applied" not in pack:
        pack["melds_applied"] = []

    pack["melds_applied"].append({
        "meld_result_id": f"{pack['spec_id']}::meld/{meld.get('meld_request_id', '')}",
        "meld_request_id": meld.get("meld_request_id", ""),
        "melded_at": datetime.now().isoformat() + "Z",
        "concepts_added": result.concepts_added,
        "source": meld.get("metadata", {}).get("source", "manual"),
        "protection_level": result.protection_level.value
    })

    with open(pack_json_path, 'w') as f:
        json.dump(pack, f, indent=2)

    print(f"Updated pack.json: v{old_version} → v{new_version} ({total_concepts} concepts)")


def apply_meld(
    meld_path: Path,
    pack_dir: Path,
    dry_run: bool = False,
    verbose: bool = False
) -> ApplicationResult:
    """Apply a single meld to the concept pack."""
    result = ApplicationResult(meld_path=meld_path, meld_id="", success=False)

    # Load meld
    try:
        with open(meld_path) as f:
            content = f.read().rstrip()
            if content.endswith("```"):
                content = content[:-3].rstrip()
            meld = json.loads(content)
    except Exception as e:
        result.errors.append(f"Failed to load meld: {e}")
        return result

    result.meld_id = meld.get("meld_request_id", "unknown")

    # Check validation status
    validation = meld.get("validation", {})
    if validation.get("status") == "rejected":
        result.errors.append("Meld has been rejected - cannot apply")
        return result

    # Load pack metadata for version
    pack_json_path = pack_dir / "pack.json"
    if not pack_json_path.exists():
        result.errors.append(f"pack.json not found in {pack_dir}")
        return result

    with open(pack_json_path) as f:
        pack = json.load(f)

    old_version = pack.get("version", "4.0.0")

    # Load layer index
    index = load_layer_index(pack_dir)
    if not index.layer_data:
        result.errors.append(f"No layer files found in {pack_dir / 'hierarchy'}")
        return result

    # Compute protection level and version bump
    result.protection_level = compute_protection_level(meld, index)
    bump_type = result.protection_level.version_bump_type()
    new_version = bump_version(old_version, bump_type)
    result.new_version = new_version

    # Compute impact analysis
    result.impact = compute_meld_impact(meld, index)

    if verbose:
        print(f"  Protection: {result.protection_level.value}")
        print(f"  Version: {old_version} → {new_version} ({bump_type})")
        print(f"  Impact: {len(result.impact.new_concepts)} new, {len(result.impact.must_retrain)} must retrain")

    # Track modifications
    modified_layers: Set[int] = set()
    concepts_by_layer: Dict[int, List[Dict]] = defaultdict(list)

    # Process each candidate
    candidates = meld.get("candidates", [])
    for candidate in candidates:
        term = candidate.get("term", "")

        if term in index.concept_to_layer:
            result.warnings.append(f"{term}: already exists in layer {index.concept_to_layer[term]}, skipping")
            continue

        layer = determine_layer(candidate, meld, index)
        concept = transform_candidate_to_concept(candidate, meld, index, layer, new_version)

        concepts_by_layer[layer].append(concept)
        modified_layers.add(layer)

        index.concept_to_layer[term] = layer
        index.concept_data[term] = concept

        result.concepts_added += 1
        result.concepts_by_layer[layer] = result.concepts_by_layer.get(layer, 0) + 1

        if verbose:
            print(f"  + {term} -> layer {layer}")

    # Update parent-child relationships
    for layer, concepts in concepts_by_layer.items():
        for concept in concepts:
            updates = update_parent_children(
                concept["sumo_term"],
                concept["parent_concepts"],
                index,
                modified_layers
            )
            result.parents_updated += updates

    # Process structural operations (split, merge, move, deprecate)
    structural_ops = meld.get("structural_operations", [])
    if structural_ops:
        if verbose:
            print(f"  Processing {len(structural_ops)} structural operations...")

        struct_result = apply_structural_operations(
            meld, index, modified_layers, new_version, verbose
        )

        result.splits_applied = struct_result.splits_applied
        result.structural_concepts_created = struct_result.concepts_created
        result.errors.extend(struct_result.errors)
        result.warnings.extend(struct_result.warnings)

        if verbose:
            print(f"  Structural: {struct_result.splits_applied} splits, {struct_result.concepts_created} concepts created")

    if dry_run:
        result.success = True
        return result

    # Write modified layers
    hierarchy_dir = pack_dir / "hierarchy"

    for layer_num in modified_layers:
        if layer_num not in index.layer_data:
            continue

        layer_data = index.layer_data[layer_num]

        # Add new concepts from candidates
        for concept in concepts_by_layer.get(layer_num, []):
            layer_data["concepts"].append(concept)

        # Rebuild concepts list from index to include structural changes
        # This ensures split children and modified parents are included
        existing_terms = {c["sumo_term"] for c in layer_data["concepts"]}
        for term, concept in index.concept_data.items():
            if index.concept_to_layer.get(term) == layer_num and term not in existing_terms:
                layer_data["concepts"].append(concept)

        # Update any modified concepts (like split parents)
        for i, concept in enumerate(layer_data["concepts"]):
            term = concept.get("sumo_term", "")
            if term in index.concept_data:
                # Replace with updated version from index
                layer_data["concepts"][i] = index.concept_data[term]

        layer_data["concepts"].sort(key=lambda c: c.get("sumo_term", "").lower())

        layer_data["summary"] = {
            "total_concepts": len(layer_data["concepts"]),
            "last_modified": datetime.now().isoformat() + "Z",
            "pack_version": new_version
        }

        layer_file = hierarchy_dir / f"layer{layer_num}.json"
        with open(layer_file, 'w') as f:
            json.dump(layer_data, f, indent=2)

    # Update pending training
    update_pending_training(pack_dir, result.impact, result.meld_id, new_version)

    # Update pack metadata with version bump and meld history
    update_pack_metadata(pack_dir, meld, result, new_version, old_version)

    # Update meld validation status
    meld["validation"] = {
        "status": "applied",
        "applied_at": datetime.now().isoformat() + "Z",
        "result_pack_version": new_version,
        "concepts_added": result.concepts_added,
        "concepts_by_layer": result.concepts_by_layer,
        "parents_updated": result.parents_updated,
        "splits_applied": result.splits_applied,
        "structural_concepts_created": result.structural_concepts_created,
        "protection_level": result.protection_level.value,
        "impact": result.impact.to_dict() if result.impact else None,
        "errors": result.errors,
        "warnings": result.warnings
    }

    # Move to applied folder
    applied_dir = meld_path.parent.parent / "applied"
    applied_dir.mkdir(exist_ok=True)
    applied_path = applied_dir / meld_path.name

    with open(applied_path, 'w') as f:
        json.dump(meld, f, indent=2)

    meld_path.unlink()

    result.success = True
    return result


def print_result(result: ApplicationResult, verbose: bool = False):
    """Print application result."""
    status = "SUCCESS" if result.success else "FAILED"
    icon = "✓" if result.success else "✗"

    print(f"\n{'=' * 60}")
    print(f"{icon} {result.meld_path.name} - {status}")
    print(f"{'=' * 60}")
    print(f"  Meld ID: {result.meld_id}")
    print(f"  Protection: {result.protection_level.value}")
    if result.new_version:
        print(f"  Version: → {result.new_version}")

    # Candidate concepts
    if result.concepts_added:
        print(f"  Concepts added: {result.concepts_added}")
        if result.concepts_by_layer:
            print(f"  By layer:")
            for layer, count in sorted(result.concepts_by_layer.items()):
                print(f"    Layer {layer}: {count}")

    # Structural operations
    if result.splits_applied:
        print(f"  Splits applied: {result.splits_applied}")
        print(f"  Concepts created from splits: {result.structural_concepts_created}")

    print(f"  Parent updates: {result.parents_updated}")

    if result.impact:
        print(f"  Training impact:")
        print(f"    Must retrain: {len(result.impact.must_retrain)}")
        print(f"    Should retrain: {len(result.impact.should_retrain)}")

    if result.errors:
        print(f"\n  ERRORS:")
        for e in result.errors:
            print(f"    - {e}")

    if result.warnings and verbose:
        print(f"\n  WARNINGS:")
        for w in result.warnings:
            print(f"    - {w}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply approved meld requests to a concept pack"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Meld files to apply"
    )
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=Path("concept_packs/sumo-wordnet-v4"),
        help="Target concept pack directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    from glob import glob
    files = []
    for pattern in args.files:
        expanded = glob(pattern)
        if expanded:
            files.extend(expanded)
        elif Path(pattern).exists():
            files.append(pattern)

    if not files:
        print("No meld files found")
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN - no changes will be made\n")

    print(f"Target pack: {args.pack_dir}")
    print(f"Melds to apply: {len(files)}")

    results = []
    total_added = 0
    final_version = None

    for fname in sorted(files):
        path = Path(fname)
        result = apply_meld(
            path,
            args.pack_dir,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        results.append(result)
        print_result(result, verbose=args.verbose)
        total_added += result.concepts_added
        if result.new_version:
            final_version = result.new_version

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    successful = sum(1 for r in results if r.success)
    print(f"Melds applied: {successful}/{len(results)}")
    print(f"Total concepts added: {total_added}")
    if final_version and not args.dry_run:
        print(f"Final pack version: {final_version}")

        # Show pending training status
        pending_path = args.pack_dir / "pending_training.json"
        if pending_path.exists():
            with open(pending_path) as f:
                pending = json.load(f)
            print(f"Concepts pending training: {len(pending.get('pending_concepts', []))}")

    if args.dry_run:
        print("\n(dry run - no changes made)")


if __name__ == "__main__":
    main()
