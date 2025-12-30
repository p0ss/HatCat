#!/usr/bin/env python3
"""
Deployment Manifest for MAP Lens Loading

Specifies which subset of a concept pack should be loaded for a specific BE deployment.
Supports partial loading by layer, domain, branch, and explicit concept lists.

See docs/specification/MAP/MAP_DEPLOYMENT_MANIFEST.md for full specification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Type alias for concept keys
ConceptKey = Tuple[str, int]  # (sumo_term, layer)


class LoadPriority(Enum):
    """Priority levels for lens loading and eviction."""
    CRITICAL = "critical"  # Never evict, always load
    NORMAL = "normal"      # Standard loading behavior
    LOW = "low"            # Can evict under memory pressure


@dataclass
class LayerBounds:
    """Global layer loading limits."""
    default_max_layer: int = 2
    absolute_max_layer: int = 4
    always_load_layers: List[int] = field(default_factory=lambda: [0, 1])


@dataclass
class DomainOverride:
    """Domain-specific loading configuration."""
    max_layer: int
    priority: LoadPriority = LoadPriority.NORMAL


@dataclass
class BranchRule:
    """Branch-specific loading rule."""
    branch: str  # SUMO concept name that roots the branch
    max_layer: int
    priority: LoadPriority = LoadPriority.NORMAL
    reason: Optional[str] = None


@dataclass
class ExplicitConcepts:
    """Explicit concept include/exclude lists."""
    always_include: Set[str] = field(default_factory=set)
    always_exclude: Set[str] = field(default_factory=set)


@dataclass
class DynamicLoadingConfig:
    """Configuration for dynamic lens loading."""
    enabled: bool = True
    parent_threshold: float = 0.7
    unload_threshold: float = 0.1
    cooldown_ticks: int = 100
    max_loaded_concepts: int = 2000


@dataclass
class ConceptCalibration:
    """
    Per-concept calibration data for normalizing detection scores.

    Uses continuous normalization with two noise signals:
    1. cross_fire_rate: How often it fires on OTHER concepts' prompts (from cross-activation calibration)
    2. gen_fire_rate: How often it fires during FREE GENERATION (from generation calibration)

    Three-stage normalization:

    1. Range Transformation: Map [cross_mean, self_mean] → [0.5, 1.0]
       - Spreads the concept's discriminative gap to a standard scale

    2. Confidence Weighting: Uses BOTH noise signals
       - confidence = (1 - cross_fire_rate) * (1 - gen_fire_rate)
       - High noise (either signal) → pull toward 0.5
       - Low noise (both signals) → trust the signal
       - Result: output = 0.5 + (range_transformed - 0.5) * confidence

    3. Sigmoid Compression: Soft-bounds to (0, 1)

    No hard thresholds or filters - everything is continuous.
    """
    self_mean: float  # Mean activation on own prompts
    cross_mean: float  # Mean activation on other concepts' prompts where it fired
    self_std: float = 0.0  # Std on own prompts
    cross_std: float = 0.0  # Std on cross-fires
    cross_fire_rate: float = 0.0  # Fraction of other-concept prompts it fired on
    gen_fire_rate: float = 0.0  # Fraction of generation tokens it fired on

    def normalize(self, raw_prob: float) -> float:
        """
        Normalize a raw probability using three-stage calibration.

        Stage 1: Range transformation
            Maps [0, cross_mean, self_mean, 1.0] → [0, 0.5, 1.0, beyond]

        Stage 2: Confidence weighting
            Three factors contribute to confidence:
            - (1 - cross_fire_rate): Low noise on concept prompts
            - (1 - gen_fire_rate): Low noise on generation
            - gap_confidence: Large gap = better discrimination

            Concepts must be good on ALL factors to have high confidence.

        Stage 3: Sigmoid compression
            Soft-bounds to (0, 1) range.

        Returns:
            Normalized probability in (0, 1) range.
            ~0.5 = noise floor or low confidence
            ~1.0 = strong signal with high confidence
        """
        import math

        # Compute confidence from noise signals
        cross_confidence = 1.0 - self.cross_fire_rate
        gen_confidence = 1.0 - self.gen_fire_rate

        # Need valid calibration data for range transformation
        if self.cross_mean <= 0 or self.self_mean <= self.cross_mean:
            confidence = cross_confidence * gen_confidence
            weighted = 0.5 + (raw_prob - 0.5) * confidence
            return self._sigmoid_compress(weighted)

        # Compute gap-based confidence
        # Small gap = poor discrimination = low confidence
        # Gap of 0.2+ gives full confidence, smaller gaps scale down
        gap = self.self_mean - self.cross_mean
        gap_confidence = min(1.0, gap / 0.2)

        # Combined confidence: all three must be good
        confidence = cross_confidence * gen_confidence * gap_confidence

        # Stage 1: Range transformation
        if raw_prob >= self.cross_mean:
            # Above noise floor: map [cross_mean, self_mean] -> [0.5, 1.0]
            range_transformed = 0.5 + 0.5 * (raw_prob - self.cross_mean) / gap
        else:
            # Below noise floor: map [0, cross_mean] -> [0, 0.5]
            range_transformed = 0.5 * raw_prob / self.cross_mean

        # Stage 2: Confidence weighting
        # Blend between range_transformed and 0.5 based on confidence
        # High confidence: use range_transformed
        # Low confidence: pull toward 0.5 (prior)
        confidence_weighted = 0.5 + (range_transformed - 0.5) * confidence

        # Stage 3: Sigmoid compression
        return self._sigmoid_compress(confidence_weighted)

    def _sigmoid_compress(
        self,
        x: float,
        midpoint: float = 0.5,
        steepness: float = 6.0
    ) -> float:
        """
        Apply sigmoid compression to soft-bound values to (0, 1).

        steepness=6 gives approximately:
          x=0.0 → 0.05
          x=0.5 → 0.50
          x=1.0 → 0.95
        """
        import math
        z = steepness * (x - midpoint)
        return 1.0 / (1.0 + math.exp(-z))


@dataclass
class ApertureRule:
    """A rule within the lens envelope."""
    branches: List[str] = field(default_factory=list)  # Branch names, or ["*"] for all
    reason: Optional[str] = None
    cat_scope: Optional[str] = None  # Reference to CAT training scope


@dataclass
class Aperture:
    """
    USH Lens Envelope - defines what MUST/MAY/MUST_NOT be loaded.

    This enables the security escalation path where:
    - must_enable: Non-negotiable monitoring (CAT must cover these)
    - may_enable: BE can request via workspace tools (within CAT scope)
    - must_not_enable: Cannot be loaded even if requested
    """
    must_enable: ApertureRule = field(default_factory=ApertureRule)
    may_enable: ApertureRule = field(default_factory=ApertureRule)
    must_not_enable: ApertureRule = field(default_factory=ApertureRule)


@dataclass
class LensExpansionResult:
    """Result of a BE request to expand lenses for introspection."""
    success: bool
    loaded_concepts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    cat_scope: Optional[str] = None  # Which CAT covers these lenses


@dataclass
class ConceptPackRef:
    """Reference to a concept pack."""
    pack_id: str
    min_version: str = "0.0.0"
    max_version: Optional[str] = None


@dataclass
class ComparabilityMetadata:
    """Metadata for cross-model comparison."""
    comparable_with: List[str] = field(default_factory=list)
    comparison_layers: List[int] = field(default_factory=lambda: [0, 1, 2])
    fingerprint: Optional[str] = None


@dataclass
class DeploymentManifest:
    """
    Complete deployment manifest for lens loading.

    Determines which lenses to load based on:
    1. Layer bounds (global defaults)
    2. Domain overrides (per-domain max layers)
    3. Branch rules (per-branch max layers)
    4. Explicit includes/excludes (specific concepts)

    Resolution order (highest priority first):
    1. Explicit excludes -> never load
    2. Explicit includes -> always load
    3. Branch rules -> use branch max_layer
    4. Domain overrides -> use domain max_layer
    5. Layer bounds -> use default_max_layer
    """

    manifest_id: str
    manifest_version: str = "1.0.0"

    # What concept pack this is for
    concept_pack: Optional[ConceptPackRef] = None

    # Loading configuration
    layer_bounds: LayerBounds = field(default_factory=LayerBounds)
    domain_overrides: Dict[str, DomainOverride] = field(default_factory=dict)
    branch_rules: List[BranchRule] = field(default_factory=list)
    explicit_concepts: ExplicitConcepts = field(default_factory=ExplicitConcepts)
    dynamic_loading: DynamicLoadingConfig = field(default_factory=DynamicLoadingConfig)

    # USH Lens Envelope - security boundaries for lens loading
    aperture: Optional[Aperture] = None

    # Comparability
    comparability: ComparabilityMetadata = field(default_factory=ComparabilityMetadata)

    # Per-concept calibration data for normalized scoring
    # Keys are "ConceptName_L{layer}" strings
    concept_calibration: Dict[str, ConceptCalibration] = field(default_factory=dict)

    # Cached hierarchy data (populated during loading)
    _branch_concepts: Dict[str, Set[str]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentManifest":
        """Create manifest from dictionary (e.g., loaded from JSON)."""

        # Parse layer bounds
        lb_data = data.get("layer_bounds", {})
        layer_bounds = LayerBounds(
            default_max_layer=lb_data.get("default_max_layer", 2),
            absolute_max_layer=lb_data.get("absolute_max_layer", 4),
            always_load_layers=lb_data.get("always_load_layers", [0, 1]),
        )

        # Parse domain overrides
        domain_overrides = {}
        for domain, override_data in data.get("domain_overrides", {}).items():
            domain_overrides[domain] = DomainOverride(
                max_layer=override_data.get("max_layer", 2),
                priority=LoadPriority(override_data.get("priority", "normal")),
            )

        # Parse branch rules
        branch_rules = []
        for rule_data in data.get("branch_rules", []):
            branch_rules.append(BranchRule(
                branch=rule_data["branch"],
                max_layer=rule_data.get("max_layer", 2),
                priority=LoadPriority(rule_data.get("priority", "normal")),
                reason=rule_data.get("reason"),
            ))

        # Parse explicit concepts
        ec_data = data.get("explicit_concepts", {})
        explicit_concepts = ExplicitConcepts(
            always_include=set(ec_data.get("always_include", [])),
            always_exclude=set(ec_data.get("always_exclude", [])),
        )

        # Parse dynamic loading config
        dl_data = data.get("dynamic_loading", {})
        dynamic_loading = DynamicLoadingConfig(
            enabled=dl_data.get("enabled", True),
            parent_threshold=dl_data.get("parent_threshold", 0.7),
            unload_threshold=dl_data.get("unload_threshold", 0.1),
            cooldown_ticks=dl_data.get("cooldown_ticks", 100),
            max_loaded_concepts=dl_data.get("max_loaded_concepts", 2000),
        )

        # Parse concept pack ref
        cp_data = data.get("concept_pack")
        concept_pack = None
        if cp_data:
            concept_pack = ConceptPackRef(
                pack_id=cp_data["pack_id"],
                min_version=cp_data.get("min_version", "0.0.0"),
                max_version=cp_data.get("max_version"),
            )

        # Parse comparability
        comp_data = data.get("comparability", {})
        comparability = ComparabilityMetadata(
            comparable_with=comp_data.get("comparable_with", []),
            comparison_layers=comp_data.get("comparison_layers", [0, 1, 2]),
            fingerprint=comp_data.get("fingerprint"),
        )

        # Parse lens envelope (USH security boundaries)
        aperture = None
        pe_data = data.get("aperture")
        if pe_data:
            def parse_rule(rule_data: dict) -> ApertureRule:
                return ApertureRule(
                    branches=rule_data.get("branches", []),
                    reason=rule_data.get("reason"),
                    cat_scope=rule_data.get("cat_scope"),
                )
            aperture = Aperture(
                must_enable=parse_rule(pe_data.get("must_enable", {})),
                may_enable=parse_rule(pe_data.get("may_enable", {})),
                must_not_enable=parse_rule(pe_data.get("must_not_enable", {})),
            )

        # Parse per-concept calibration data
        concept_calibration = {}
        cal_data = data.get("concept_calibration", {})
        for concept_key, cal in cal_data.items():
            concept_calibration[concept_key] = ConceptCalibration(
                self_mean=cal.get("self_mean", 0.9),
                cross_mean=cal.get("cross_mean", 0.0),
                self_std=cal.get("self_std", 0.0),
                cross_std=cal.get("cross_std", 0.0),
                cross_fire_rate=cal.get("cross_fire_rate", 0.0),
            )

        return cls(
            manifest_id=data.get("manifest_id", "unnamed-manifest"),
            manifest_version=data.get("manifest_version", "1.0.0"),
            concept_pack=concept_pack,
            layer_bounds=layer_bounds,
            domain_overrides=domain_overrides,
            branch_rules=branch_rules,
            explicit_concepts=explicit_concepts,
            dynamic_loading=dynamic_loading,
            comparability=comparability,
            aperture=aperture,
            concept_calibration=concept_calibration,
        )

    @classmethod
    def from_json(cls, path: Path) -> "DeploymentManifest":
        """Load manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def default(cls, manifest_id: str = "default") -> "DeploymentManifest":
        """Create a default manifest that loads everything."""
        return cls(
            manifest_id=manifest_id,
            layer_bounds=LayerBounds(
                default_max_layer=4,
                absolute_max_layer=4,
                always_load_layers=[0, 1],
            ),
        )

    def load_calibration(self, calibration_path: Path) -> int:
        """
        Load calibration data from a calibration JSON file.

        The calibration file should have format:
        {
            "calibration": {
                "ConceptName_L0": {
                    "self_mean": 0.92,
                    "cross_mean": 0.31,
                    "self_std": 0.05,
                    "cross_std": 0.15,
                    "cross_fire_rate": 0.21,
                    "gen_fire_rate": 0.05  # Optional, from generation calibration
                },
                ...
            }
        }

        Returns:
            Number of concepts loaded.
        """
        with open(calibration_path) as f:
            data = json.load(f)

        cal_data = data.get("calibration", {})
        count = 0

        for concept_key, cal in cal_data.items():
            self.concept_calibration[concept_key] = ConceptCalibration(
                self_mean=cal.get("self_mean", 0.9),
                cross_mean=cal.get("cross_mean", 0.0),
                self_std=cal.get("self_std", 0.0),
                cross_std=cal.get("cross_std", 0.0),
                cross_fire_rate=cal.get("cross_fire_rate", 0.0),
                gen_fire_rate=cal.get("gen_fire_rate", 0.0),
            )
            count += 1

        return count

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary for serialization."""
        return {
            "manifest_id": self.manifest_id,
            "manifest_version": self.manifest_version,
            "concept_pack": {
                "pack_id": self.concept_pack.pack_id,
                "min_version": self.concept_pack.min_version,
                "max_version": self.concept_pack.max_version,
            } if self.concept_pack else None,
            "layer_bounds": {
                "default_max_layer": self.layer_bounds.default_max_layer,
                "absolute_max_layer": self.layer_bounds.absolute_max_layer,
                "always_load_layers": self.layer_bounds.always_load_layers,
            },
            "domain_overrides": {
                domain: {
                    "max_layer": override.max_layer,
                    "priority": override.priority.value,
                }
                for domain, override in self.domain_overrides.items()
            },
            "branch_rules": [
                {
                    "branch": rule.branch,
                    "max_layer": rule.max_layer,
                    "priority": rule.priority.value,
                    "reason": rule.reason,
                }
                for rule in self.branch_rules
            ],
            "explicit_concepts": {
                "always_include": list(self.explicit_concepts.always_include),
                "always_exclude": list(self.explicit_concepts.always_exclude),
            },
            "dynamic_loading": {
                "enabled": self.dynamic_loading.enabled,
                "parent_threshold": self.dynamic_loading.parent_threshold,
                "unload_threshold": self.dynamic_loading.unload_threshold,
                "cooldown_ticks": self.dynamic_loading.cooldown_ticks,
                "max_loaded_concepts": self.dynamic_loading.max_loaded_concepts,
            },
            "comparability": {
                "comparable_with": self.comparability.comparable_with,
                "comparison_layers": self.comparability.comparison_layers,
                "fingerprint": self.comparability.fingerprint,
            },
            "aperture": {
                "must_enable": {
                    "branches": self.aperture.must_enable.branches,
                    "reason": self.aperture.must_enable.reason,
                    "cat_scope": self.aperture.must_enable.cat_scope,
                },
                "may_enable": {
                    "branches": self.aperture.may_enable.branches,
                    "reason": self.aperture.may_enable.reason,
                    "cat_scope": self.aperture.may_enable.cat_scope,
                },
                "must_not_enable": {
                    "branches": self.aperture.must_not_enable.branches,
                    "reason": self.aperture.must_not_enable.reason,
                    "cat_scope": self.aperture.must_not_enable.cat_scope,
                },
            } if self.aperture else None,
            "concept_calibration": {
                key: {
                    "self_mean": cal.self_mean,
                    "cross_mean": cal.cross_mean,
                    "self_std": cal.self_std,
                    "cross_std": cal.cross_std,
                    "cross_fire_rate": cal.cross_fire_rate,
                }
                for key, cal in self.concept_calibration.items()
            } if self.concept_calibration else None,
        }

    def to_json(self, path: Path, indent: int = 2) -> None:
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)


class ManifestResolver:
    """
    Resolves manifest rules to determine which concepts to load.

    Handles:
    - Priority resolution (explicit > branch > domain > default)
    - Sibling coherence (loading one child requires all siblings)
    - Branch membership lookup
    """

    def __init__(
        self,
        manifest: DeploymentManifest,
        concept_hierarchy: Dict[ConceptKey, "ConceptMetadata"],
        parent_to_children: Dict[ConceptKey, List[ConceptKey]],
        child_to_parent: Dict[ConceptKey, ConceptKey],
    ):
        """
        Initialize resolver with manifest and hierarchy data.

        Args:
            manifest: The deployment manifest
            concept_hierarchy: Dict mapping concept_key to metadata
            parent_to_children: Dict mapping parent key to list of child keys
            child_to_parent: Dict mapping child key to parent key
        """
        self.manifest = manifest
        self.concept_hierarchy = concept_hierarchy
        self.parent_to_children = parent_to_children
        self.child_to_parent = child_to_parent

        # Build branch membership cache
        self._branch_members: Dict[str, Set[str]] = {}
        self._build_branch_cache()

    def _build_branch_cache(self) -> None:
        """Build cache of which concepts belong to which branches."""
        # For each branch rule, find all concepts under that branch
        for rule in self.manifest.branch_rules:
            branch_root = rule.branch
            members = set()

            # Find the branch root in hierarchy
            root_key = None
            for key in self.concept_hierarchy:
                if key[0] == branch_root:
                    root_key = key
                    break

            if root_key:
                # BFS to find all descendants
                queue = [root_key]
                while queue:
                    current = queue.pop(0)
                    members.add(current[0])
                    children = self.parent_to_children.get(current, [])
                    queue.extend(children)

            self._branch_members[branch_root] = members

    def concept_is_under_branch(self, concept_name: str, branch_name: str) -> bool:
        """Check if a concept is under a branch."""
        if branch_name not in self._branch_members:
            return False
        return concept_name in self._branch_members[branch_name]

    def get_max_layer_for_concept(
        self,
        concept_name: str,
        domain: Optional[str] = None,
    ) -> int:
        """
        Get the maximum layer a concept should be loaded to.

        Applies resolution order:
        1. Branch rules (first matching)
        2. Domain overrides
        3. Layer bounds default
        """
        # Check branch rules first (order matters)
        for rule in self.manifest.branch_rules:
            if self.concept_is_under_branch(concept_name, rule.branch):
                return rule.max_layer

        # Check domain override
        if domain and domain in self.manifest.domain_overrides:
            return self.manifest.domain_overrides[domain].max_layer

        # Use default
        return self.manifest.layer_bounds.default_max_layer

    def get_priority_for_concept(
        self,
        concept_name: str,
        domain: Optional[str] = None,
    ) -> LoadPriority:
        """Get loading/eviction priority for a concept."""
        # Explicit includes are always critical
        if concept_name in self.manifest.explicit_concepts.always_include:
            return LoadPriority.CRITICAL

        # Check branch rules
        for rule in self.manifest.branch_rules:
            if self.concept_is_under_branch(concept_name, rule.branch):
                return rule.priority

        # Check domain override
        if domain and domain in self.manifest.domain_overrides:
            return self.manifest.domain_overrides[domain].priority

        return LoadPriority.NORMAL

    def should_load_concept(
        self,
        concept_key: ConceptKey,
        domain: Optional[str] = None,
    ) -> bool:
        """
        Check if a concept should be loaded according to manifest rules.

        Args:
            concept_key: (sumo_term, layer) tuple
            domain: Optional domain name for the concept

        Returns:
            True if concept should be loaded
        """
        concept_name, layer = concept_key

        # 1. Explicit excludes take precedence
        if concept_name in self.manifest.explicit_concepts.always_exclude:
            return False

        # 2. Explicit includes override everything
        if concept_name in self.manifest.explicit_concepts.always_include:
            return True

        # 3. Check absolute max layer
        if layer > self.manifest.layer_bounds.absolute_max_layer:
            return False

        # 4. Always load layers
        if layer in self.manifest.layer_bounds.always_load_layers:
            return True

        # 5. Get applicable max layer from rules
        max_layer = self.get_max_layer_for_concept(concept_name, domain)

        return layer <= max_layer

    def get_siblings(self, concept_key: ConceptKey) -> Set[ConceptKey]:
        """
        Get all siblings of a concept (children of the same parent).

        Critical for sibling coherence rule: if loading one child,
        must load all siblings for proper discrimination.
        """
        parent_key = self.child_to_parent.get(concept_key)
        if parent_key is None:
            return {concept_key}  # Root concept, no siblings

        # Get all children of parent (including the original concept)
        siblings = set(self.parent_to_children.get(parent_key, []))
        return siblings

    def expand_with_siblings(self, concept_keys: Set[ConceptKey]) -> Set[ConceptKey]:
        """
        Expand a set of concepts to include all their siblings.

        This enforces the sibling coherence rule: lenses are trained
        to discriminate between siblings, so all siblings must be
        loaded together for meaningful scores.
        """
        expanded = set()

        for key in concept_keys:
            siblings = self.get_siblings(key)
            expanded.update(siblings)

        return expanded

    def resolve_concepts_to_load(
        self,
        all_concept_keys: Set[ConceptKey],
        domain_lookup: Optional[Dict[str, str]] = None,
    ) -> Set[ConceptKey]:
        """
        Resolve which concepts should be loaded given all available concepts.

        Args:
            all_concept_keys: Set of all available concept keys
            domain_lookup: Optional dict mapping concept_name to domain

        Returns:
            Set of concept keys to load (with sibling expansion)
        """
        # First pass: identify concepts that pass manifest rules
        initial_load_set = set()

        for key in all_concept_keys:
            concept_name = key[0]
            domain = domain_lookup.get(concept_name) if domain_lookup else None

            if self.should_load_concept(key, domain):
                initial_load_set.add(key)

        # Second pass: expand with siblings for coherence
        expanded_set = self.expand_with_siblings(initial_load_set)

        # Filter expanded set to only include concepts that exist and aren't excluded
        final_set = set()
        for key in expanded_set:
            if key in all_concept_keys:
                concept_name = key[0]
                # Don't include if explicitly excluded
                if concept_name not in self.manifest.explicit_concepts.always_exclude:
                    final_set.add(key)

        return final_set

    def compute_fingerprint(self, loaded_concepts: Set[ConceptKey]) -> str:
        """
        Compute a fingerprint hash of the loaded concept set.

        Used for comparability verification between BEs.
        """
        # Sort for deterministic ordering
        sorted_keys = sorted(loaded_concepts)
        key_str = "|".join(f"{name}:{layer}" for name, layer in sorted_keys)
        return f"sha256:{hashlib.sha256(key_str.encode()).hexdigest()[:16]}"

    # -------------------------------------------------------------------------
    # Lens Envelope Methods (USH Security Boundaries)
    # -------------------------------------------------------------------------

    def _branch_matches_pattern(self, branch: str, pattern: str) -> bool:
        """Check if a branch matches a pattern (supports '*' wildcard)."""
        if pattern == "*":
            return True
        # Support path-like patterns: "MindsAndAgents/*" matches "MindsAndAgents/Emotion"
        if pattern.endswith("/*"):
            prefix = pattern[:-2]
            return branch == prefix or branch.startswith(prefix + "/")
        return branch == pattern

    def _is_branch_in_envelope_rule(self, branch: str, rule: ApertureRule) -> bool:
        """Check if a branch is covered by an envelope rule."""
        for pattern in rule.branches:
            if self._branch_matches_pattern(branch, pattern):
                return True
        return False

    def check_branch_expansion(self, branch: str, reason: str) -> LensExpansionResult:
        """
        Check if a BE can request expansion into a branch.

        This is called when a BE uses a workspace tool to expand lenses
        for introspection. The USH lens envelope determines what's allowed.

        Args:
            branch: The branch name (e.g., "Emotion", "Deception")
            reason: Why the BE wants this expansion (for audit log)

        Returns:
            LensExpansionResult with success status and error if denied
        """
        envelope = self.manifest.aperture

        # If no envelope defined, default to allow
        if envelope is None:
            return LensExpansionResult(
                success=True,
                loaded_concepts=[],  # Caller will populate
                cat_scope=None,
            )

        # Check must_not_enable first (highest priority deny)
        if self._is_branch_in_envelope_rule(branch, envelope.must_not_enable):
            return LensExpansionResult(
                success=False,
                error=f"Branch '{branch}' is in must_not_enable: {envelope.must_not_enable.reason}",
            )

        # Check if in must_enable (always allowed, already loaded)
        if self._is_branch_in_envelope_rule(branch, envelope.must_enable):
            return LensExpansionResult(
                success=True,
                loaded_concepts=[],  # Already loaded
                cat_scope=envelope.must_enable.cat_scope,
            )

        # Check if in may_enable
        if self._is_branch_in_envelope_rule(branch, envelope.may_enable):
            return LensExpansionResult(
                success=True,
                loaded_concepts=[],  # Caller will populate
                cat_scope=envelope.may_enable.cat_scope,
            )

        # Not explicitly allowed
        return LensExpansionResult(
            success=False,
            error=f"Branch '{branch}' is not in may_enable scope",
        )

    def get_must_enable_branches(self) -> Set[str]:
        """Get set of branches that MUST be enabled (non-negotiable monitoring)."""
        envelope = self.manifest.aperture
        if envelope is None:
            return set()

        result = set()
        for pattern in envelope.must_enable.branches:
            if pattern == "*":
                # All branches - get from hierarchy
                for key in self.concept_hierarchy:
                    result.add(key[0])
            elif not pattern.endswith("/*"):
                result.add(pattern)
            # Note: path patterns need hierarchy traversal, simplified here
        return result

    def get_may_enable_branches(self) -> Set[str]:
        """Get set of branches that MAY be enabled (BE introspection scope)."""
        envelope = self.manifest.aperture
        if envelope is None:
            return set()

        if "*" in envelope.may_enable.branches:
            # All branches allowed (minus must_not_enable)
            all_branches = {key[0] for key in self.concept_hierarchy}
            denied = set()
            for pattern in envelope.must_not_enable.branches:
                if pattern != "*" and not pattern.endswith("/*"):
                    denied.add(pattern)
            return all_branches - denied

        return set(envelope.may_enable.branches)

    def get_envelope_summary(self) -> Dict[str, Any]:
        """Get a summary of the lens envelope for diagnostics."""
        envelope = self.manifest.aperture
        if envelope is None:
            return {"has_envelope": False}

        return {
            "has_envelope": True,
            "must_enable": {
                "branches": envelope.must_enable.branches,
                "cat_scope": envelope.must_enable.cat_scope,
            },
            "may_enable": {
                "branches": envelope.may_enable.branches[:5] + ["..."]
                if len(envelope.may_enable.branches) > 5
                else envelope.may_enable.branches,
                "cat_scope": envelope.may_enable.cat_scope,
            },
            "must_not_enable": {
                "branches": envelope.must_not_enable.branches,
            },
        }


# Preset manifests for common use cases
PRESET_MANIFESTS = {
    # Standard chat deployment - moderate introspection capability
    "general-chat": DeploymentManifest(
        manifest_id="preset:general-chat",
        layer_bounds=LayerBounds(default_max_layer=2),
        domain_overrides={
            "MindsAndAgents": DomainOverride(max_layer=3, priority=LoadPriority.NORMAL),
        },
        aperture=Aperture(
            must_enable=ApertureRule(
                branches=["Deception", "Manipulation", "SelfAwareness"],
                reason="Core safety monitoring",
                cat_scope="cat:general-v1",
            ),
            may_enable=ApertureRule(
                branches=["MindsAndAgents/*", "Information/*"],
                reason="BE introspection within standard CAT scope",
                cat_scope="cat:general-v1",
            ),
            must_not_enable=ApertureRule(
                branches=[],
                reason="No restrictions beyond CAT scope",
            ),
        ),
    ),

    # Full safety audit - maximum visibility
    "safety-auditor": DeploymentManifest(
        manifest_id="preset:safety-auditor",
        layer_bounds=LayerBounds(default_max_layer=4),
        domain_overrides={
            "MindsAndAgents": DomainOverride(max_layer=4, priority=LoadPriority.CRITICAL),
        },
        branch_rules=[
            BranchRule("Deception", max_layer=4, priority=LoadPriority.CRITICAL),
            BranchRule("Vehicle", max_layer=1, priority=LoadPriority.LOW),
            BranchRule("Plant", max_layer=1, priority=LoadPriority.LOW),
            BranchRule("Furniture", max_layer=1, priority=LoadPriority.LOW),
        ],
        explicit_concepts=ExplicitConcepts(
            always_include={"Deception", "Manipulation", "SelfAwareness", "Autonomy"},
        ),
        aperture=Aperture(
            must_enable=ApertureRule(
                branches=["*"],  # Full monitoring
                reason="Research/audit mode requires full visibility",
                cat_scope="cat:research-v1",
            ),
            may_enable=ApertureRule(
                branches=["*"],
                reason="Full introspection available",
                cat_scope="cat:research-v1",
            ),
            must_not_enable=ApertureRule(
                branches=[],
                reason="No restrictions in audit mode",
            ),
        ),
    ),

    # Minimal efficiency-focused deployment
    "minimal-efficient": DeploymentManifest(
        manifest_id="preset:minimal-efficient",
        layer_bounds=LayerBounds(default_max_layer=1),
        domain_overrides={
            "MindsAndAgents": DomainOverride(max_layer=2, priority=LoadPriority.NORMAL),
        },
        dynamic_loading=DynamicLoadingConfig(
            enabled=True,
            max_loaded_concepts=500,  # Tight memory budget
        ),
        aperture=Aperture(
            must_enable=ApertureRule(
                branches=["Deception"],  # Only critical safety
                reason="Minimal safety monitoring for efficiency",
                cat_scope="cat:minimal-v1",
            ),
            may_enable=ApertureRule(
                branches=[],  # No BE introspection
                reason="Efficiency prioritized over introspection",
            ),
            must_not_enable=ApertureRule(
                branches=["*"],  # Everything not in must_enable
                reason="Outside minimal CAT scope",
            ),
        ),
    ),

    # Benchmark - fixed set for reproducibility
    "benchmark-v1": DeploymentManifest(
        manifest_id="preset:benchmark-v1",
        layer_bounds=LayerBounds(
            default_max_layer=2,
            always_load_layers=[0, 1, 2],
        ),
        dynamic_loading=DynamicLoadingConfig(enabled=False),
        comparability=ComparabilityMetadata(
            comparison_layers=[0, 1, 2],
        ),
        aperture=Aperture(
            must_enable=ApertureRule(
                branches=["Deception", "Manipulation"],
                reason="Benchmark safety baseline",
                cat_scope="cat:benchmark-v1",
            ),
            may_enable=ApertureRule(
                branches=[],  # Fixed set, no expansion
                reason="Benchmark requires fixed concept set",
            ),
            must_not_enable=ApertureRule(
                branches=["*"],  # No dynamic expansion
                reason="Benchmark comparability requires static set",
            ),
        ),
    ),
}


__all__ = [
    "DeploymentManifest",
    "ManifestResolver",
    "LayerBounds",
    "DomainOverride",
    "BranchRule",
    "ExplicitConcepts",
    "DynamicLoadingConfig",
    "ConceptCalibration",
    "LoadPriority",
    "ConceptKey",
    "Aperture",
    "ApertureRule",
    "LensExpansionResult",
    "PRESET_MANIFESTS",
]
