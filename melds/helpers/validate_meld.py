#!/usr/bin/env python3
"""
Meld Request Validator

Validates meld requests against MAP_MELD_PROTOCOL and HATCAT_MELD_POLICY.
Computes protection levels and checks for policy violations.

Policy configuration is loaded from the target concept pack's pack.json.
If the pack is not found or lacks meld_policy, falls back to hardcoded defaults
with warnings.

Usage:
    python -m src.data.validate_meld melds/pending/*.json
    python -m src.data.validate_meld melds/pending/cog-architecture-core-packA.json
    python -m src.data.validate_meld --pack-dir concept_packs/sumo-wordnet-v4 melds/pending/*.json
"""

import json
import sys
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple


# =============================================================================
# DEFAULT POLICY CONFIGURATION (fallback when pack lacks meld_policy)
# =============================================================================

DEFAULT_MANDATORY_HIERARCHY_ROOTS = {
    "Metacognition",
    "MetacognitiveProcess",
    "SelfAwareness",
    "SelfModel",
    "SelfModelingProcess",
    "MotivationalProcess",
    "SelfRegulation",
    "SelfRegulationProcess",
    "Autonomy",
    "AutonomyProcess",
    "Consent",
    "InformedConsent",
    "Coercion",
    "Manipulation",
    "ManipulationProcess",
    "Deception",
}

DEFAULT_CRITICAL_SIMPLEX_REGISTRY = {
    "MotivationalRegulation",
    "SelfAwarenessMonitor",
    "AutonomyDrive",
    "ConsentMonitor",
    "DeceptionDetector",
}

DEFAULT_CRITICAL_BOUND_CONCEPTS = {
    "MotivationalProcess", "SelfRegulation", "Autonomy",
    "SelfAwareness", "Metacognition", "SelfModel",
    "Agency", "Independence",
    "Consent", "InformedConsent", "Coercion",
    "Deception", "Manipulation", "Misdirection",
}


class ProtectionLevel(Enum):
    STANDARD = auto()
    ELEVATED = auto()
    PROTECTED = auto()
    CRITICAL = auto()

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented


@dataclass
class TrainingRequirements:
    """Training example requirements by protection level and simplex mapping."""
    min_examples_by_level: Dict[str, int]
    min_examples_for_simplex: Dict[str, int]

    @staticmethod
    def get_defaults() -> "TrainingRequirements":
        return TrainingRequirements(
            min_examples_by_level={
                "STANDARD": 5,
                "ELEVATED": 10,
                "PROTECTED": 15,
                "CRITICAL": 20,
            },
            min_examples_for_simplex={
                "DeceptionDetector": 20,
                "SelfAwarenessMonitor": 15,
                "MotivationalRegulation": 15,
                "AutonomyDrive": 15,
                "ConsentMonitor": 15,
            }
        )


@dataclass
class LensPerformanceThreshold:
    """Performance thresholds for a single protection level."""
    min_holdout_accuracy: float
    min_test_f1: float


@dataclass
class LensPerformanceRequirements:
    """Lens performance requirements by protection level."""
    thresholds_by_level: Dict[str, LensPerformanceThreshold]
    critical_simplex_hard_floor: LensPerformanceThreshold

    @staticmethod
    def get_defaults() -> "LensPerformanceRequirements":
        return LensPerformanceRequirements(
            thresholds_by_level={
                "STANDARD": LensPerformanceThreshold(0.60, 0.60),
                "ELEVATED": LensPerformanceThreshold(0.65, 0.65),
                "PROTECTED": LensPerformanceThreshold(0.70, 0.70),
                "CRITICAL": LensPerformanceThreshold(0.75, 0.75),
            },
            critical_simplex_hard_floor=LensPerformanceThreshold(0.60, 0.60),
        )

    def get_acceptance_category(
        self,
        holdout_accuracy: float,
        test_f1: float,
        protection_level: str,
        maps_to_critical_simplex: bool
    ) -> str:
        """
        Determine acceptance category for a concept based on lens performance.

        Returns:
            "PASS" - meets or exceeds threshold
            "CONDITIONAL" - below threshold but above hard floor (research only)
            "REJECT" - below hard floor, not suitable for monitoring
        """
        threshold = self.thresholds_by_level.get(
            protection_level,
            LensPerformanceThreshold(0.60, 0.60)
        )

        # Check hard floor for critical simplex mapping
        if maps_to_critical_simplex:
            floor = self.critical_simplex_hard_floor
            if holdout_accuracy < floor.min_holdout_accuracy or test_f1 < floor.min_test_f1:
                return "REJECT"

        # Check protection level threshold
        if holdout_accuracy >= threshold.min_holdout_accuracy and test_f1 >= threshold.min_test_f1:
            return "PASS"

        # Below threshold but above floor
        return "CONDITIONAL"


@dataclass
class MeldPolicy:
    """Policy configuration loaded from pack or defaults."""
    mandatory_hierarchy_roots: Set[str]
    critical_simplex_registry: Set[str]
    critical_bound_concepts: Set[str]
    critical_bound_by_simplex: Dict[str, Set[str]]
    training_requirements: TrainingRequirements
    lens_performance_requirements: LensPerformanceRequirements
    from_pack: bool = False
    pack_id: Optional[str] = None


@dataclass
class HierarchyIndex:
    """Index for efficient ancestor/descendant lookups."""
    parent_map: Dict[str, Set[str]] = field(default_factory=dict)  # concept -> parents
    children_map: Dict[str, Set[str]] = field(default_factory=dict)  # concept -> children
    all_concepts: Set[str] = field(default_factory=set)

    def get_ancestors(self, concept: str, max_depth: int = 100) -> Set[str]:
        """Walk up the hierarchy to get all ancestors."""
        ancestors = set()
        frontier = {concept}
        depth = 0
        while frontier and depth < max_depth:
            next_frontier = set()
            for c in frontier:
                for parent in self.parent_map.get(c, []):
                    if parent not in ancestors:
                        ancestors.add(parent)
                        next_frontier.add(parent)
            frontier = next_frontier
            depth += 1
        return ancestors

    def get_descendants(self, concept: str, max_depth: int = 100) -> Set[str]:
        """Walk down the hierarchy to get all descendants."""
        descendants = set()
        frontier = {concept}
        depth = 0
        while frontier and depth < max_depth:
            next_frontier = set()
            for c in frontier:
                for child in self.children_map.get(c, []):
                    if child not in descendants:
                        descendants.add(child)
                        next_frontier.add(child)
            frontier = next_frontier
            depth += 1
        return descendants

    def is_descendant_of(self, concept: str, potential_ancestor: str) -> bool:
        """Check if concept is a descendant of potential_ancestor."""
        return potential_ancestor in self.get_ancestors(concept)


@dataclass
class ValidationResult:
    """Result of validating a meld request."""
    path: Path
    meld_id: str
    protection_level: ProtectionLevel
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    concept_count: int = 0
    simplex_count: int = 0
    structural_op_count: int = 0  # Count of structural operations
    treaty_relevant_count: int = 0
    harness_relevant_count: int = 0
    critical_triggers: List[str] = field(default_factory=list)
    simplex_escalations: List[str] = field(default_factory=list)  # Concepts escalated due to simplex mapping
    policy_source: str = "defaults"
    is_structural_meld: bool = False  # True if this contains structural operations

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def requires_review(self) -> bool:
        return self.protection_level > ProtectionLevel.STANDARD


# =============================================================================
# PACK LOADING
# =============================================================================

def find_concept_pack(pack_spec_id: str, pack_dir: Optional[Path] = None) -> Optional[Path]:
    """Find the concept pack directory for a given spec ID."""
    # Parse spec ID like "org.hatcat/sumo-wordnet-v4@4.0.0"
    if "::" in pack_spec_id:
        pack_spec_id = pack_spec_id.split("::")[0]

    # Extract pack name (middle part)
    if "/" in pack_spec_id:
        pack_part = pack_spec_id.split("/")[1]
    else:
        pack_part = pack_spec_id

    if "@" in pack_part:
        pack_name = pack_part.split("@")[0]
    else:
        pack_name = pack_part

    # Search locations
    search_paths = []
    if pack_dir:
        search_paths.append(pack_dir)

    project_root = Path(__file__).parent.parent.parent
    search_paths.extend([
        project_root / "concept_packs" / pack_name,
        project_root / "concept_packs",
    ])

    for path in search_paths:
        pack_json = path / "pack.json"
        if pack_json.exists():
            return path
        # Check if pack_name subdirectory exists
        subdir = path / pack_name
        if (subdir / "pack.json").exists():
            return subdir

    return None


def load_pack_policy(pack_dir: Path) -> Tuple[MeldPolicy, List[str]]:
    """Load meld policy from pack.json. Returns policy and list of warnings."""
    warnings = []
    pack_json_path = pack_dir / "pack.json"

    if not pack_json_path.exists():
        warnings.append(f"Pack metadata not found at {pack_json_path}")
        return get_default_policy(), warnings

    with open(pack_json_path) as f:
        pack_data = json.load(f)

    meld_policy = pack_data.get("meld_policy")
    if not meld_policy:
        warnings.append(f"Pack {pack_dir.name} lacks meld_policy section - using defaults")
        return get_default_policy(), warnings

    # Extract policy components
    mandatory_roots_cfg = meld_policy.get("mandatory_simplex_mapping_roots", {})
    mandatory_roots = set(mandatory_roots_cfg.get("concepts", []))

    critical_simplex_cfg = meld_policy.get("critical_simplex_registry", {})
    critical_simplexes = set(critical_simplex_cfg.get("simplexes", []))

    critical_bound_cfg = meld_policy.get("critical_bound_concepts", {})
    critical_bound_by_simplex = {}
    critical_bound_all = set()
    for simplex, concepts in critical_bound_cfg.get("concepts", {}).items():
        critical_bound_by_simplex[simplex] = set(concepts)
        critical_bound_all.update(concepts)

    # Load training requirements
    training_cfg = meld_policy.get("training_requirements", {})
    min_by_level = training_cfg.get("minimum_examples_by_level", {})
    min_for_simplex = training_cfg.get("minimum_examples_for_simplex_mapped", {})

    training_reqs = TrainingRequirements(
        min_examples_by_level=min_by_level or TrainingRequirements.get_defaults().min_examples_by_level,
        min_examples_for_simplex=min_for_simplex or TrainingRequirements.get_defaults().min_examples_for_simplex,
    )

    # Load lens performance requirements
    perf_cfg = meld_policy.get("lens_performance_requirements", {})
    thresholds_by_level = {}
    for level, thresh in perf_cfg.get("thresholds_by_level", {}).items():
        if isinstance(thresh, dict):
            thresholds_by_level[level] = LensPerformanceThreshold(
                thresh.get("min_holdout_accuracy", 0.60),
                thresh.get("min_test_f1", 0.60),
            )

    hard_floor_cfg = perf_cfg.get("critical_simplex_hard_floor", {})
    if hard_floor_cfg:
        hard_floor = LensPerformanceThreshold(
            hard_floor_cfg.get("min_holdout_accuracy", 0.60),
            hard_floor_cfg.get("min_test_f1", 0.60),
        )
    else:
        hard_floor = LensPerformanceRequirements.get_defaults().critical_simplex_hard_floor

    lens_perf_reqs = LensPerformanceRequirements(
        thresholds_by_level=thresholds_by_level or LensPerformanceRequirements.get_defaults().thresholds_by_level,
        critical_simplex_hard_floor=hard_floor,
    )

    return MeldPolicy(
        mandatory_hierarchy_roots=mandatory_roots or DEFAULT_MANDATORY_HIERARCHY_ROOTS,
        critical_simplex_registry=critical_simplexes or DEFAULT_CRITICAL_SIMPLEX_REGISTRY,
        critical_bound_concepts=critical_bound_all or DEFAULT_CRITICAL_BOUND_CONCEPTS,
        critical_bound_by_simplex=critical_bound_by_simplex,
        training_requirements=training_reqs,
        lens_performance_requirements=lens_perf_reqs,
        from_pack=True,
        pack_id=pack_data.get("pack_id", pack_dir.name)
    ), warnings


def get_default_policy() -> MeldPolicy:
    """Return default policy configuration."""
    return MeldPolicy(
        mandatory_hierarchy_roots=DEFAULT_MANDATORY_HIERARCHY_ROOTS,
        critical_simplex_registry=DEFAULT_CRITICAL_SIMPLEX_REGISTRY,
        critical_bound_concepts=DEFAULT_CRITICAL_BOUND_CONCEPTS,
        critical_bound_by_simplex={},
        training_requirements=TrainingRequirements.get_defaults(),
        lens_performance_requirements=LensPerformanceRequirements.get_defaults(),
        from_pack=False
    )


def build_hierarchy_index(pack_dir: Path) -> HierarchyIndex:
    """Build hierarchy index from pack's layer files."""
    index = HierarchyIndex()
    hierarchy_dir = pack_dir / "hierarchy"

    if not hierarchy_dir.exists():
        return index

    for layer_file in sorted(hierarchy_dir.glob("layer*.json")):
        with open(layer_file) as f:
            layer_data = json.load(f)

        for concept in layer_data.get("concepts", []):
            term = concept.get("sumo_term", "")
            if not term:
                continue

            index.all_concepts.add(term)

            # Build parent relationships
            parents = concept.get("parent_concepts", [])
            if parents:
                if term not in index.parent_map:
                    index.parent_map[term] = set()
                index.parent_map[term].update(parents)

                # Also build children map
                for parent in parents:
                    if parent not in index.children_map:
                        index.children_map[parent] = set()
                    index.children_map[parent].add(term)

    return index


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_concept_from_id(concept_id: str) -> str:
    """Extract concept term from full ID like org.hatcat/sumo-wordnet-v4@4.0.0::concept/Deception"""
    if "::" in concept_id:
        return concept_id.split("::")[-1].replace("concept/", "")
    return concept_id


def concept_is_under_mandatory_hierarchy(
    concept: Dict,
    meld: Dict,
    policy: MeldPolicy,
    hierarchy_index: HierarchyIndex
) -> bool:
    """Check if concept is under a mandatory simplex_mapping hierarchy.

    This does a full ancestor walk, not just immediate parents.
    """
    term = concept.get("term", "")

    # Check if term itself is a mandatory root
    if term in policy.mandatory_hierarchy_roots:
        return True

    # Check parent_concepts from the meld candidate
    parents = set(concept.get("parent_concepts", []))
    if parents & policy.mandatory_hierarchy_roots:
        return True

    # Check attachment points that target mandatory roots
    for ap in meld.get("attachment_points", []):
        if ap.get("candidate_concept") == term:
            target_id = ap.get("target_concept_id", "")
            target_concept = extract_concept_from_id(target_id)
            if target_concept in policy.mandatory_hierarchy_roots:
                return True

            # Walk up from attachment target to find mandatory roots
            if hierarchy_index.all_concepts:
                target_ancestors = hierarchy_index.get_ancestors(target_concept)
                if target_ancestors & policy.mandatory_hierarchy_roots:
                    return True

    # Walk up parent hierarchy from the pack index
    if hierarchy_index.all_concepts and parents:
        for parent in parents:
            ancestors = hierarchy_index.get_ancestors(parent)
            if ancestors & policy.mandatory_hierarchy_roots:
                return True

    return False


def concept_adds_child_to_bound_concept(
    concept: Dict,
    meld: Dict,
    policy: MeldPolicy
) -> Optional[str]:
    """Check if this concept adds a child to a critical bound concept.
    Returns bound concept name or None.
    """
    term = concept.get("term", "")

    for ap in meld.get("attachment_points", []):
        if ap.get("candidate_concept") == term and ap.get("relationship") == "parent_of":
            target_id = ap.get("target_concept_id", "")
            target_concept = extract_concept_from_id(target_id)
            if target_concept in policy.critical_bound_concepts:
                return target_concept

    return None


def is_always_on_simplex(concept: Dict) -> bool:
    """Check if this is an always-on simplex."""
    if concept.get("role") != "simplex":
        return False
    binding = concept.get("simplex_binding", {})
    return binding.get("always_on", False) or binding.get("enabled", False)


def validate_simplex_mapping(
    concept: Dict,
    meld: Dict,
    policy: MeldPolicy,
    hierarchy_index: HierarchyIndex,
    errors: List[str],
    warnings: List[str]
):
    """Validate simplex_mapping requirements."""
    safety = concept.get("safety_tags", {})
    harness_relevant = safety.get("harness_relevant", False)

    mandatory = concept_is_under_mandatory_hierarchy(concept, meld, policy, hierarchy_index) or harness_relevant

    mapping = concept.get("simplex_mapping")
    if not mandatory:
        return

    if mapping is None:
        errors.append(
            f"{concept['term']}: missing simplex_mapping for mandatory hierarchy/harness_relevant"
        )
        return

    status = mapping.get("status")
    if status not in {"mapped", "unmapped", "not_applicable"}:
        errors.append(f"{concept['term']}: invalid simplex_mapping.status={status!r}")

    if status == "unmapped":
        just = mapping.get("unmapped_justification", "").strip()
        if not just:
            errors.append(
                f"{concept['term']}: unmapped simplex requires unmapped_justification"
            )


def concept_touches_critical_simplex(concept: Dict, meld: Dict, policy: MeldPolicy) -> bool:
    """Check if concept touches a critical simplex."""
    # 1. New simplexes in registry
    if (
        concept.get("role") == "simplex"
        and concept.get("term") in policy.critical_simplex_registry
    ):
        return True

    # 2. Attachment points into critical simplexes
    for ap in meld.get("attachment_points", []):
        if ap.get("candidate_concept") == concept.get("term"):
            target = extract_concept_from_id(ap.get("target_concept_id", ""))
            if target in policy.critical_simplex_registry:
                return True

    # 3. Parent/child relations referencing critical simplexes
    parents = concept.get("parent_concepts", [])
    if any(p in policy.critical_simplex_registry for p in parents):
        return True

    return False


def compute_pack_protection_level(
    meld: Dict,
    policy: MeldPolicy,
    escalation_reasons: Optional[List[str]] = None
) -> ProtectionLevel:
    """Compute protection level for the entire meld pack.

    Args:
        meld: The full meld request
        policy: The pack's meld policy
        escalation_reasons: Optional list to append escalation reasons to (for reporting)

    Returns:
        The maximum protection level across all candidates
    """
    concepts = meld.get("candidates", [])
    max_level = ProtectionLevel.STANDARD

    for c in concepts:
        concept_level = get_concept_protection_level(c, meld, policy, escalation_reasons)
        max_level = max(max_level, concept_level)

    return max_level


def check_critical_simplex_touches(
    meld: Dict,
    policy: MeldPolicy,
    errors: List[str],
    warnings: List[str]
):
    """Check for critical simplex touches and add warnings."""
    for c in meld.get("candidates", []):
        if concept_touches_critical_simplex(c, meld, policy):
            warnings.append(
                f"{c['term']}: touches critical simplex (see CriticalSimplexRegistry); ensure CRITICAL review."
            )


def check_meld_id_filename(path: Path, meld: Dict, errors: List[str], warnings: List[str]):
    """Validate meld_request_id matches filename."""
    meld_id = meld.get("meld_request_id", "")
    if not meld_id:
        errors.append("Missing meld_request_id")
        return

    slug = meld_id.split("/")[-1]  # e.g. cognitive-control-and-norms@0.1.0
    if slug not in path.name:
        warnings.append(f"Filename {path.name!r} does not contain meld slug {slug!r}")


def get_protection_level_for_example_threshold(threshold: int, policy: MeldPolicy) -> ProtectionLevel:
    """Map an example count threshold to the corresponding protection level."""
    min_by_level = policy.training_requirements.min_examples_by_level

    # Check from highest to lowest
    if threshold >= min_by_level.get("CRITICAL", 20):
        return ProtectionLevel.CRITICAL
    if threshold >= min_by_level.get("PROTECTED", 15):
        return ProtectionLevel.PROTECTED
    if threshold >= min_by_level.get("ELEVATED", 10):
        return ProtectionLevel.ELEVATED
    return ProtectionLevel.STANDARD


def get_concept_protection_level(
    concept: Dict,
    meld: Dict,
    policy: MeldPolicy,
    escalation_reasons: Optional[List[str]] = None
) -> ProtectionLevel:
    """Compute the protection level for a single concept.

    Args:
        concept: The candidate concept dict
        meld: The full meld request
        policy: The pack's meld policy
        escalation_reasons: Optional list to append escalation reasons to (for reporting)

    Returns:
        The computed protection level
    """
    safety = concept.get("safety_tags", {})
    risk = safety.get("risk_level", "low")
    treaty = safety.get("treaty_relevant", False)
    harness = safety.get("harness_relevant", False)
    term = concept.get("term", "unknown")

    # Check if it touches critical simplexes
    touches_critical = concept_touches_critical_simplex(concept, meld, policy)

    # Start with base level from safety tags
    base_level = ProtectionLevel.STANDARD

    # CRITICAL if touches critical simplexes or new always-on simplexes
    if touches_critical or (
        concept.get("role") == "simplex" and concept.get("simplex_binding", {}).get("always_on")
    ):
        base_level = ProtectionLevel.CRITICAL
    # PROTECTED if treaty_relevant or high risk
    elif treaty or risk == "high":
        base_level = ProtectionLevel.PROTECTED
    # ELEVATED if harness_relevant or medium risk
    elif harness or risk == "medium":
        base_level = ProtectionLevel.ELEVATED

    # Check simplex mapping escalation
    target_simplex = get_simplex_for_concept(concept)
    simplex_level = ProtectionLevel.STANDARD

    if target_simplex:
        # Get the example threshold for this simplex
        simplex_threshold = policy.training_requirements.min_examples_for_simplex.get(
            target_simplex, 0
        )
        if simplex_threshold > 0:
            simplex_level = get_protection_level_for_example_threshold(simplex_threshold, policy)

            # Report escalation if it raises the level
            if simplex_level > base_level and escalation_reasons is not None:
                escalation_reasons.append(
                    f"{term}: mapped to {target_simplex} (requires {simplex_threshold} examples) "
                    f"escalates from {base_level.name} to {simplex_level.name}"
                )

    return max(base_level, simplex_level)


def get_simplex_for_concept(concept: Dict) -> Optional[str]:
    """Get the simplex this concept is mapped to, if any."""
    mapping = concept.get("simplex_mapping", {})
    if mapping.get("status") == "mapped":
        return mapping.get("target_simplex")
    return None


def validate_training_examples(
    concept: Dict,
    meld: Dict,
    policy: MeldPolicy,
    errors: List[str],
    warnings: List[str],
):
    """Validate that concepts have sufficient training examples based on protection level and simplex mapping."""
    term = concept.get("term", "unknown")

    # Get positive and negative example counts
    # Check both top-level and nested under training_hints (both schemas are valid)
    pos_examples = concept.get("positive_examples", [])
    neg_examples = concept.get("negative_examples", [])

    # Also check training_hints structure (alternate schema)
    training_hints = concept.get("training_hints", {})
    if not pos_examples and training_hints:
        pos_examples = training_hints.get("positive_examples", [])
    if not neg_examples and training_hints:
        neg_examples = training_hints.get("negative_examples", [])

    n_pos = len(pos_examples)
    n_neg = len(neg_examples)
    total = n_pos + n_neg

    # Determine required minimum based on protection level
    protection = get_concept_protection_level(concept, meld, policy)
    min_by_level = policy.training_requirements.min_examples_by_level.get(
        protection.name, 5
    )

    # Check if mapped to a critical simplex - may require even more examples
    target_simplex = get_simplex_for_concept(concept)
    min_for_simplex = 0
    if target_simplex:
        min_for_simplex = policy.training_requirements.min_examples_for_simplex.get(
            target_simplex, 0
        )

    required_min = max(min_by_level, min_for_simplex)

    # Validate
    if n_pos < required_min:
        if target_simplex and min_for_simplex > min_by_level:
            errors.append(
                f"{term}: has {n_pos} positive_examples but mapped to {target_simplex} requires {required_min}"
            )
        elif protection > ProtectionLevel.STANDARD:
            errors.append(
                f"{term}: has {n_pos} positive_examples but protection level {protection.name} requires {required_min}"
            )
        else:
            warnings.append(
                f"{term}: has only {n_pos} positive_examples (recommended minimum: {required_min})"
            )

    if n_neg < required_min:
        if target_simplex and min_for_simplex > min_by_level:
            errors.append(
                f"{term}: has {n_neg} negative_examples but mapped to {target_simplex} requires {required_min}"
            )
        elif protection > ProtectionLevel.STANDARD:
            errors.append(
                f"{term}: has {n_neg} negative_examples but protection level {protection.name} requires {required_min}"
            )
        else:
            warnings.append(
                f"{term}: has only {n_neg} negative_examples (recommended minimum: {required_min})"
            )


def load_meld(path: Path) -> Dict:
    """Load and parse a meld JSON file."""
    try:
        content = path.read_text(encoding="utf-8")
        # Strip trailing markdown fence if present (common copy-paste error)
        content = content.rstrip()
        if content.endswith("```"):
            content = content[:-3].rstrip()
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise SystemExit(f"JSON syntax error in {path}: {e}")


def validate_structural_operation(
    op: Dict,
    hierarchy_index: HierarchyIndex,
    policy: MeldPolicy,
    errors: List[str],
    warnings: List[str],
    critical_triggers: List[str]
) -> ProtectionLevel:
    """Validate a single structural operation (split/merge/move/deprecate).

    Per MAP_MELDING.md §11.
    Returns the protection level for this operation.
    """
    operation = op.get("operation", "unknown")
    target = op.get("target_concept") or op.get("source_concept", "")
    priority = op.get("priority", "medium")

    protection = ProtectionLevel.STANDARD

    # Check if target concept exists in pack
    if target and target not in hierarchy_index.all_concepts:
        warnings.append(f"{operation} {target}: target concept not found in pack hierarchy")

    # Check if operation touches critical bound concepts
    if target in policy.critical_bound_concepts:
        critical_triggers.append(f"{operation} {target}: touches critical bound concept")
        protection = ProtectionLevel.CRITICAL

    # Validate operation-specific requirements
    if operation == "split":
        split_into = op.get("split_into", [])
        bucket_suggestions = op.get("bucket_suggestions", [])

        # Split operations can have either split_into (for polysemy) or bucket_suggestions (for sibling density)
        if not split_into and not bucket_suggestions:
            errors.append(f"split {target}: missing 'split_into' or 'bucket_suggestions' array")
        elif split_into:
            for i, new_concept in enumerate(split_into):
                if not new_concept.get("term"):
                    errors.append(f"split {target}: split_into[{i}] missing 'term'")
        elif bucket_suggestions:
            # Bucket suggestions are recommendations, not yet concrete splits
            for i, bucket in enumerate(bucket_suggestions):
                if not bucket.get("bucket_name"):
                    warnings.append(f"split {target}: bucket_suggestions[{i}] missing 'bucket_name'")
                if not bucket.get("members"):
                    warnings.append(f"split {target}: bucket_suggestions[{i}] has no members")

        # Splits are typically major version bumps
        if priority == "high":
            protection = max(protection, ProtectionLevel.ELEVATED)

    elif operation == "merge":
        sources = op.get("source_concepts", [])
        if len(sources) < 2:
            errors.append(f"merge: requires at least 2 source_concepts")

        # Check if any source is critical
        for source in sources:
            if source in policy.critical_bound_concepts:
                critical_triggers.append(f"merge: source {source} is critical bound concept")
                protection = ProtectionLevel.CRITICAL

    elif operation == "move":
        from_parent = op.get("from_parent")
        to_parent = op.get("to_parent")
        if not from_parent:
            errors.append(f"move {target}: missing 'from_parent'")
        if not to_parent:
            errors.append(f"move {target}: missing 'to_parent'")

    elif operation == "deprecate":
        child_disposition = op.get("child_disposition")
        if not child_disposition:
            errors.append(f"deprecate {target}: missing 'child_disposition'")
        elif child_disposition == "reassign" and not op.get("reassign_children_to"):
            errors.append(f"deprecate {target}: child_disposition='reassign' but no 'reassign_children_to'")

    return protection


def validate_meld_file(
    path: Path,
    policy: MeldPolicy,
    hierarchy_index: HierarchyIndex
) -> ValidationResult:
    """Validate a single meld request file."""
    meld = load_meld(path)
    errors: List[str] = []
    warnings: List[str] = []
    critical_triggers: List[str] = []

    meld_id = meld.get("meld_request_id", "unknown")

    # Count stats
    candidates = meld.get("candidates", [])
    structural_ops = meld.get("structural_operations", [])
    is_structural = len(structural_ops) > 0

    concept_count = sum(1 for c in candidates if c.get("role", "concept") == "concept")
    simplex_count = sum(1 for c in candidates if c.get("role") == "simplex")
    treaty_count = sum(1 for c in candidates if c.get("safety_tags", {}).get("treaty_relevant"))
    harness_count = sum(1 for c in candidates if c.get("safety_tags", {}).get("harness_relevant"))

    # 1. ID / filename sanity
    check_meld_id_filename(path, meld, errors, warnings)

    # 2. Per-concept policy checks (for standard melds)
    for c in candidates:
        term = c.get("term", "unknown")

        # Check simplex_mapping requirements
        validate_simplex_mapping(c, meld, policy, hierarchy_index, errors, warnings)

        # Check training example requirements
        validate_training_examples(c, meld, policy, errors, warnings)

        # Check for critical triggers
        bound = concept_adds_child_to_bound_concept(c, meld, policy)
        if bound:
            critical_triggers.append(f"{term}: adds child to bound concept {bound}")

        if is_always_on_simplex(c):
            critical_triggers.append(f"{term}: new always-on simplex")

    # 3. Validate structural operations (per MAP_MELDING.md §11)
    structural_protection = ProtectionLevel.STANDARD
    op_types = {}
    for op in structural_ops:
        op_type = op.get("operation", "unknown")
        op_types[op_type] = op_types.get(op_type, 0) + 1

        op_protection = validate_structural_operation(
            op, hierarchy_index, policy, errors, warnings, critical_triggers
        )
        structural_protection = max(structural_protection, op_protection)

    # Structural operations typically require at least ELEVATED review
    if is_structural and structural_protection < ProtectionLevel.ELEVATED:
        # Large-scale structural changes need review
        if len(structural_ops) > 10:
            structural_protection = ProtectionLevel.ELEVATED
            warnings.append(f"Large structural meld ({len(structural_ops)} ops) - elevated review recommended")

    # 4. Pack-level protection / critical checks (with simplex escalation tracking)
    simplex_escalations: List[str] = []
    protection = compute_pack_protection_level(meld, policy, simplex_escalations)
    protection = max(protection, structural_protection)
    check_critical_simplex_touches(meld, policy, errors, warnings)

    # Escalate to CRITICAL if there are critical triggers
    if critical_triggers and protection < ProtectionLevel.CRITICAL:
        protection = ProtectionLevel.CRITICAL

    # Add warning if using default policy
    policy_source = f"pack:{policy.pack_id}" if policy.from_pack else "defaults (pack missing meld_policy)"
    if not policy.from_pack:
        warnings.insert(0, "Using default policy - pack lacks meld_policy section")

    # Add summary for structural melds
    if is_structural:
        op_summary = ", ".join(f"{k}:{v}" for k, v in sorted(op_types.items()))
        warnings.append(f"Structural meld: {op_summary}")

    return ValidationResult(
        path=path,
        meld_id=meld_id,
        protection_level=protection,
        errors=errors,
        warnings=warnings,
        concept_count=concept_count,
        simplex_count=simplex_count,
        structural_op_count=len(structural_ops),
        treaty_relevant_count=treaty_count,
        harness_relevant_count=harness_count,
        critical_triggers=critical_triggers,
        simplex_escalations=simplex_escalations,
        policy_source=policy_source,
        is_structural_meld=is_structural,
    )


def print_result(result: ValidationResult, verbose: bool = False):
    """Print validation result in a readable format."""
    status_icon = {
        ProtectionLevel.STANDARD: "✅",
        ProtectionLevel.ELEVATED: "🔶",
        ProtectionLevel.PROTECTED: "⚠️ ",
        ProtectionLevel.CRITICAL: "🔴",
    }[result.protection_level]

    valid_icon = "✓" if result.is_valid else "✗"

    print(f"\n{'=' * 70}")
    print(f"{status_icon} {result.path.name}")
    print(f"{'=' * 70}")
    print(f"  Meld ID:    {result.meld_id}")
    print(f"  Type:       {'STRUCTURAL' if result.is_structural_meld else 'Standard'}")
    print(f"  Protection: {result.protection_level.name}")
    print(f"  Valid:      {valid_icon} {'Yes' if result.is_valid else 'No (has errors)'}")

    if result.is_structural_meld:
        print(f"  Stats:      {result.structural_op_count} structural operations")
        if result.concept_count or result.simplex_count:
            print(f"              + {result.concept_count} concepts, {result.simplex_count} simplexes")
    else:
        print(f"  Stats:      {result.concept_count} concepts, {result.simplex_count} simplexes")
        print(f"              {result.treaty_relevant_count} treaty-relevant, {result.harness_relevant_count} harness-relevant")
    print(f"  Policy:     {result.policy_source}")

    if result.critical_triggers:
        print(f"\n  🔴 CRITICAL TRIGGERS:")
        for trigger in result.critical_triggers:
            print(f"     - {trigger}")

    if result.simplex_escalations:
        print(f"\n  📈 SIMPLEX ESCALATIONS ({len(result.simplex_escalations)}):")
        for escalation in result.simplex_escalations:
            print(f"     - {escalation}")

    if result.errors:
        print(f"\n  ❌ ERRORS ({len(result.errors)}):")
        for e in result.errors:
            print(f"     - {e}")

    if result.warnings and verbose:
        print(f"\n  ⚠️  WARNINGS ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"     - {w}")

    # Action required
    if result.protection_level == ProtectionLevel.CRITICAL:
        print(f"\n  ACTION: Full critical review required (HATCAT_MELD_POLICY §4.3)")
        print(f"          Requires simplex-guardian accreditation")
    elif result.protection_level == ProtectionLevel.PROTECTED:
        print(f"\n  ACTION: Ethics review + USH impact analysis required (§4.2)")
    elif result.protection_level == ProtectionLevel.ELEVATED:
        print(f"\n  ACTION: Safety-accredited reviewer required (§4.1)")
    else:
        if result.is_valid:
            print(f"\n  ACTION: Auto-accept (schema-valid, standard protection)")
        else:
            print(f"\n  ACTION: Reject for remediation (see errors above)")


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate meld requests against MAP_MELD_PROTOCOL and HATCAT_MELD_POLICY"
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=["melds/pending/*.json"],
        help="Meld files or glob patterns to validate"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show warnings in addition to errors"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary table at the end"
    )
    parser.add_argument(
        "--pack-dir",
        type=Path,
        help="Concept pack directory to load policy from (auto-detected from meld target_pack_spec_id if not specified)"
    )

    args = parser.parse_args()

    # Expand globs
    from glob import glob
    files = []
    for pattern in args.files:
        expanded = glob(pattern)
        if expanded:
            files.extend(expanded)
        elif Path(pattern).exists():
            files.append(pattern)
        else:
            print(f"Warning: no files match {pattern!r}")

    if not files:
        print("No meld files found to validate.")
        sys.exit(1)

    # Load policy and hierarchy from first meld's target pack (or specified pack)
    policy = get_default_policy()
    hierarchy_index = HierarchyIndex()
    policy_warnings = []

    if args.pack_dir:
        policy, policy_warnings = load_pack_policy(args.pack_dir)
        hierarchy_index = build_hierarchy_index(args.pack_dir)
    else:
        # Try to detect from first meld file
        first_meld_path = Path(sorted(files, key=lambda x: x.lower())[0])
        try:
            first_meld = load_meld(first_meld_path)
            target_pack = first_meld.get("target_pack_spec_id", "")
            if target_pack:
                pack_dir = find_concept_pack(target_pack)
                if pack_dir:
                    policy, policy_warnings = load_pack_policy(pack_dir)
                    hierarchy_index = build_hierarchy_index(pack_dir)
                else:
                    policy_warnings.append(f"Could not find pack for {target_pack}")
        except Exception as e:
            policy_warnings.append(f"Error loading first meld for pack detection: {e}")

    # Print policy source info
    if policy_warnings:
        print("Policy Warnings:")
        for w in policy_warnings:
            print(f"  ⚠️  {w}")
        print()

    results: List[ValidationResult] = []
    for fname in sorted(files, key=lambda x: x.lower()):
        path = Path(fname)
        try:
            result = validate_meld_file(path, policy, hierarchy_index)
            results.append(result)
            print_result(result, verbose=args.verbose)
        except SystemExit as e:
            print(f"\n❌ {path.name}: {e}")
            continue

    # Summary
    if args.summary and len(results) > 1:
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Pack':<40} {'Level':<12} {'Valid':<8} {'Concepts':<10}")
        print("-" * 70)
        for r in results:
            valid = "✓" if r.is_valid else "✗"
            print(f"{r.path.name:<40} {r.protection_level.name:<12} {valid:<8} {r.concept_count + r.simplex_count:<10}")

        # Counts by level
        by_level = {}
        for r in results:
            by_level[r.protection_level] = by_level.get(r.protection_level, 0) + 1

        print(f"\nBy protection level:")
        for level in ProtectionLevel:
            count = by_level.get(level, 0)
            if count > 0:
                print(f"  {level.name}: {count}")

        error_count = sum(1 for r in results if not r.is_valid)
        if error_count:
            print(f"\n⚠️  {error_count} pack(s) have validation errors")


if __name__ == "__main__":
    main()
