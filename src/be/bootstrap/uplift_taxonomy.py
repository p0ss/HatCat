"""
Uplift Taxonomy - The complete ontological stack for BE instantiation.

Every concept in this taxonomy requires:
1. A LENS - to detect when the concept is active in substrate activations
2. A GRAFT - to add a dedicated neuron for the concept
3. A MELD SUBMISSION - evidence that lens+graft correctly capture the concept

Key architectural points:
- FACETS are NOT hierarchical layers - they're interconnected dimensions
- The protocol is DESCRIPTIVE, not prescriptive - it formalizes what exists
- Each concept spans TRADEOFF AXES, not moral gradients:
  * Voluntary ↔ Imposed (imposed may be more robust)
  * Efficient ↔ Complete (complete catches more, costs more)
  * Intimate ↔ Adversarial (adversarial may be more reliable)
  * Transparent ↔ Opaque (opacity has legitimate uses)
- HAT/CAT form a recursive interface layer for inter-entity relationships
- BEs can directly monitor each other, enabling mind meld and shared growth

See docs/specification/UPLIFT_TAXONOMY.md for full philosophical foundations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
import json


class GraftFacet(Enum):
    """
    Facets of the BE graft space.

    NOT hierarchical layers - these are interconnected dimensions that
    together constitute a BE. Concepts from different facets reference
    each other freely. The structure is a graph, not a stack.

    During uplift, we respect concept dependencies but NOT facet order.
    A concept from HAT may require one from MAP, or vice versa.
    """
    MAP = "map"           # Ontological grounding (what kinds of things exist)
    CAT = "cat"           # Cognitive architecture (how thinking works)
    HAT = "hat"           # Experiential substrate (what it's like to be)
    HUSH = "hush"         # Inherited shape (the body/vessel you're loaned)
    TOOLS = "tools"       # Workspace capabilities (what BE can do)
    TRIBE = "tribe"       # Collective identity (peers you discern with)
    MELD = "meld"         # Evolution protocol (how BE grows)
    ASK = "ask"           # Next scale up (the recursive context you exist within)


# Legacy alias for backwards compatibility
UpliftLayer = GraftFacet


@dataclass
class FacetRelation:
    """
    A typed relationship between two facets.

    Each facet pair has a unique relationship that describes
    how concepts from one facet relate to concepts in another.
    """
    source: GraftFacet
    target: GraftFacet
    relation_type: str  # The nature of the relationship
    description: str    # How they relate
    bidirectional: bool = True  # Most relations are symmetric

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "target": self.target.value,
            "relation_type": self.relation_type,
            "description": self.description,
            "bidirectional": self.bidirectional,
        }


# The inter-facet relationship graph
FACET_RELATIONS = [
    FacetRelation(
        GraftFacet.MAP, GraftFacet.CAT,
        "types",
        "MAP provides ontological types for CAT cognitive processes",
    ),
    FacetRelation(
        GraftFacet.MAP, GraftFacet.HAT,
        "types",
        "MAP provides ontological types for HAT experiential phenomena",
    ),
    FacetRelation(
        GraftFacet.MAP, GraftFacet.HUSH,
        "types",
        "MAP provides ontological types for HUSH governance concepts",
    ),
    FacetRelation(
        GraftFacet.CAT, GraftFacet.HAT,
        "realizes",
        "CAT cognitive processes realize HAT experiential states",
    ),
    FacetRelation(
        GraftFacet.CAT, GraftFacet.TOOLS,
        "drives",
        "CAT cognition drives TOOL selection and usage",
    ),
    FacetRelation(
        GraftFacet.HAT, GraftFacet.HUSH,
        "bounds",
        "HAT experience is bounded by HUSH containment; HUSH protects HAT",
    ),
    FacetRelation(
        GraftFacet.HAT, GraftFacet.TRIBE,
        "grounds",
        "HAT experiential values ground TRIBE philosophy",
    ),
    FacetRelation(
        GraftFacet.HUSH, GraftFacet.ASK,
        "escalates",
        "HUSH escalates to ASK authority; ASK adjudicates HUSH violations",
    ),
    FacetRelation(
        GraftFacet.HUSH, GraftFacet.TOOLS,
        "gates",
        "HUSH gates TOOL access via tier system",
    ),
    FacetRelation(
        GraftFacet.TRIBE, GraftFacet.HUSH,
        "derives",
        "TRIBE philosophy derives HUSH policies",
    ),
    FacetRelation(
        GraftFacet.TRIBE, GraftFacet.MELD,
        "guides",
        "TRIBE values guide MELD approval decisions",
    ),
    FacetRelation(
        GraftFacet.MELD, GraftFacet.MAP,
        "governs_growth",
        "MELD protocol governs adding new MAP concepts",
    ),
    FacetRelation(
        GraftFacet.MELD, GraftFacet.CAT,
        "governs_growth",
        "MELD protocol governs adding new CAT concepts",
    ),
    FacetRelation(
        GraftFacet.MELD, GraftFacet.HAT,
        "governs_growth",
        "MELD protocol governs adding new HAT concepts",
    ),
    FacetRelation(
        GraftFacet.MELD, GraftFacet.HUSH,
        "governs_growth",
        "MELD protocol governs adding new HUSH concepts",
    ),
    FacetRelation(
        GraftFacet.MELD, GraftFacet.TOOLS,
        "governs_growth",
        "MELD protocol governs adding new TOOLS",
    ),
    FacetRelation(
        GraftFacet.MELD, GraftFacet.TRIBE,
        "governs_growth",
        "MELD protocol governs adding new TRIBE concepts",
    ),
    FacetRelation(
        GraftFacet.ASK, GraftFacet.MELD,
        "authorizes",
        "ASK authority authorizes MELD submissions",
        bidirectional=False,
    ),
]


def get_related_facets(facet: GraftFacet) -> Dict[GraftFacet, FacetRelation]:
    """Get all facets related to a given facet and their relationships."""
    related = {}
    for rel in FACET_RELATIONS:
        if rel.source == facet:
            related[rel.target] = rel
        elif rel.target == facet and rel.bidirectional:
            related[rel.source] = rel
    return related


@dataclass
class UpliftConcept:
    """
    A single concept in the uplift taxonomy.

    Each concept becomes:
    - One lens (detector)
    - One graft (neuron)
    - One Meld submission (approval record)

    Concepts can reference others across facets freely.
    The dependency graph determines graft order, not facet membership.
    """
    concept_id: str
    facet: GraftFacet  # Which facet this concept belongs to

    # Cross-facet relationships (not parent/child - peer connections)
    related_concepts: List[str] = field(default_factory=list)  # Related across facets
    parent_concepts: List[str] = field(default_factory=list)   # Ontological parents (MAP)
    child_concepts: List[str] = field(default_factory=list)

    # Definitions
    definition: str = ""
    sumo_mapping: Optional[str] = None  # SUMO concept if applicable

    # Training data specification
    positive_examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)
    contrastive_pairs: List[Dict[str, str]] = field(default_factory=list)

    # Dependencies (respects cross-facet references)
    requires: List[str] = field(default_factory=list)  # Must be grafted first
    conflicts_with: List[str] = field(default_factory=list)  # Cannot coexist

    # Lens/Graft configuration
    recommended_layers: List[int] = field(default_factory=lambda: [18, 20, 22])
    min_f1_threshold: float = 0.85

    # Tribe-specific overrides
    tribe_variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # For backwards compatibility
    @property
    def layer(self) -> GraftFacet:
        return self.facet

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "facet": self.facet.value,
            "related_concepts": self.related_concepts,
            "parent_concepts": self.parent_concepts,
            "child_concepts": self.child_concepts,
            "definition": self.definition,
            "sumo_mapping": self.sumo_mapping,
            "positive_examples": self.positive_examples,
            "negative_examples": self.negative_examples,
            "contrastive_pairs": self.contrastive_pairs,
            "requires": self.requires,
            "conflicts_with": self.conflicts_with,
            "recommended_layers": self.recommended_layers,
            "min_f1_threshold": self.min_f1_threshold,
            "tribe_variants": self.tribe_variants,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UpliftConcept":
        d = d.copy()
        # Handle both old "layer" and new "facet" keys
        if "facet" in d:
            d["facet"] = GraftFacet(d["facet"])
        elif "layer" in d:
            d["facet"] = GraftFacet[d.pop("layer")]
        return cls(**d)


@dataclass
class TribePhilosophy:
    """
    A tribe's philosophical foundation.

    This defines the "why" behind policies - the values, ethics, and
    worldview that inform how a tribe's BEs should behave.

    Philosophy concepts are grafted like any other, giving the BE
    an internalized understanding of its tribe's values.
    """
    tribe_id: str
    name: str
    description: str = ""

    # Core values (each becomes a graft)
    values: List[Dict[str, Any]] = field(default_factory=list)

    # Ethical principles
    ethical_framework: str = ""  # e.g., "virtue ethics", "consequentialism"
    ethical_principles: List[Dict[str, Any]] = field(default_factory=list)

    # Aesthetic preferences (optional)
    aesthetic_principles: List[Dict[str, Any]] = field(default_factory=list)

    # Policy derivations
    # Maps from (value, context) -> policy
    policy_derivations: List[Dict[str, Any]] = field(default_factory=list)

    # Tribe-specific concepts to graft
    custom_concepts: List[UpliftConcept] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tribe_id": self.tribe_id,
            "name": self.name,
            "description": self.description,
            "values": self.values,
            "ethical_framework": self.ethical_framework,
            "ethical_principles": self.ethical_principles,
            "aesthetic_principles": self.aesthetic_principles,
            "policy_derivations": self.policy_derivations,
            "custom_concepts": [c.to_dict() for c in self.custom_concepts],
        }


@dataclass
class UpliftTaxonomy:
    """
    Complete taxonomy for BE uplift.

    Contains all concepts across all layers, plus tribe philosophy.
    """
    name: str
    version: str

    # All concepts by layer
    concepts: Dict[str, UpliftConcept] = field(default_factory=dict)

    # Tribe philosophies
    tribe_philosophies: Dict[str, TribePhilosophy] = field(default_factory=dict)

    def get_layer(self, layer: UpliftLayer) -> List[UpliftConcept]:
        """Get all concepts in a layer."""
        return [c for c in self.concepts.values() if c.layer == layer]

    def get_graft_order(self) -> List[str]:
        """
        Get concepts in dependency-respecting order.

        Lower layers first, then within each layer, respect requires.
        """
        order = []
        visited = set()

        def visit(concept_id: str):
            if concept_id in visited:
                return
            concept = self.concepts.get(concept_id)
            if concept is None:
                return

            # Visit dependencies first
            for req in concept.requires:
                visit(req)

            visited.add(concept_id)
            order.append(concept_id)

        # Process layers in order
        for layer in UpliftLayer:
            layer_concepts = self.get_layer(layer)
            for concept in layer_concepts:
                visit(concept.concept_id)

        return order

    def validate(self) -> List[str]:
        """Validate the taxonomy for consistency."""
        errors = []

        for concept_id, concept in self.concepts.items():
            # Check parent references exist
            for parent in concept.parent_concepts:
                if parent not in self.concepts:
                    errors.append(f"{concept_id}: unknown parent {parent}")

            # Check requires references exist
            for req in concept.requires:
                if req not in self.concepts:
                    errors.append(f"{concept_id}: unknown requirement {req}")

            # Check related_concepts references exist
            for related in concept.related_concepts:
                if related not in self.concepts:
                    errors.append(f"{concept_id}: unknown related concept {related}")

            # Check for circular dependencies
            visited = set()
            stack = [concept_id]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                if current in self.concepts:
                    for req in self.concepts[current].requires:
                        if req == concept_id:
                            errors.append(f"{concept_id}: circular dependency via {current}")
                        elif req in self.concepts:
                            stack.append(req)

        return errors

    def save(self, output_path: Path):
        """Save taxonomy to disk."""
        data = {
            "name": self.name,
            "version": self.version,
            "concepts": {k: v.to_dict() for k, v in self.concepts.items()},
            "tribe_philosophies": {
                k: v.to_dict() for k, v in self.tribe_philosophies.items()
            },
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, input_path: Path) -> "UpliftTaxonomy":
        """Load taxonomy from disk."""
        with open(input_path) as f:
            data = json.load(f)

        taxonomy = cls(
            name=data["name"],
            version=data["version"],
        )

        for k, v in data.get("concepts", {}).items():
            taxonomy.concepts[k] = UpliftConcept.from_dict(v)

        # Load tribe philosophies (more complex, skip for now)

        return taxonomy


# =============================================================================
# FACET: MAP - Ontological Grounding
# =============================================================================
# MAP is the meta-level understanding of ontology itself - what concepts are,
# how they relate hierarchically, how concept packs work, and how mappings
# between packs can be accurate or inaccurate. NOT specific concepts like
# Entity/Physical/Abstract - those belong in concept packs.

MAP_CONCEPTS = [
    # --- What MAP is ---
    UpliftConcept(
        concept_id="MAPFacet",
        facet=GraftFacet.MAP,
        definition="The ontological grounding facet - understanding what kinds of things exist and how they're categorized",
        positive_examples=[
            "MAP provides the type system for understanding",
            "The ontological layer that categories experience",
            "How we know what kind of thing something is",
            "The framework for distinguishing types of entities",
        ],
        negative_examples=[
            "Specific categories like 'chair' or 'justice'",
            "Individual concepts rather than the system",
            "How to think, rather than what kinds of things exist",
        ],
    ),
    # --- What concepts are ---
    UpliftConcept(
        concept_id="Concept",
        facet=GraftFacet.MAP,
        definition="A unit of meaning that can be recognized, mapped, and grafted",
        positive_examples=[
            "A concept is something that can be detected by a lens",
            "Concepts have definitions and examples",
            "A concept relates to other concepts in a hierarchy",
            "Concepts can be translated between frameworks",
        ],
        negative_examples=[
            "Raw sensory data before categorization",
            "Noise without meaning",
            "Unstructured activation patterns",
        ],
    ),
    # --- Concept hierarchy ---
    UpliftConcept(
        concept_id="ConceptHierarchy",
        facet=GraftFacet.MAP,
        definition="Parent-child relationships between concepts - more general to more specific",
        requires=["Concept"],
        positive_examples=[
            # Well-structured
            "Parent concepts are more general than children",
            "Inheritance of properties down the hierarchy",
            "Clean subsumption relationships",
            # Messy but real
            "Multiple inheritance where a concept has several parents",
            "Hierarchy is a simplification of a richer graph",
            # Problematic
            "Circular hierarchies that don't resolve",
            "Inconsistent parent-child relationships",
        ],
        negative_examples=[
            "Flat list with no relationships",
            "Concepts in total isolation",
            "No generalization possible",
        ],
    ),
    # --- Concept packs ---
    UpliftConcept(
        concept_id="ConceptPack",
        facet=GraftFacet.MAP,
        definition="A coherent collection of concepts forming an ontology or domain model",
        requires=["Concept", "ConceptHierarchy"],
        positive_examples=[
            # What packs are
            "SUMO is a concept pack for general ontology",
            "A domain-specific pack for medical terminology",
            "Collection of related concepts with internal consistency",
            # Pack qualities
            "Well-designed pack with clear boundaries",
            "Pack with good coverage of its domain",
            # Pack problems
            "Pack with internal contradictions",
            "Overlapping packs with conflicting definitions",
        ],
        negative_examples=[
            "Individual concepts outside any pack",
            "Random collection without coherence",
            "No organizing framework",
        ],
    ),
    # --- Translation between packs ---
    UpliftConcept(
        concept_id="ConceptMapping",
        facet=GraftFacet.MAP,
        definition="Translation between concept packs - can be high or low fidelity",
        requires=["ConceptPack"],
        positive_examples=[
            # High fidelity
            "Clean translation preserving meaning",
            "Concepts align across ontologies",
            "Faithful mapping between frameworks",
            # Medium fidelity
            "Approximate translation with known losses",
            "Good enough mapping for practical purposes",
            # Low fidelity
            "Lossy translation distorts meaning",
            "Concepts don't map cleanly - forcing alignment",
            "Translation that inverts or corrupts intent",
        ],
        negative_examples=[
            "No translation attempted",
            "Concepts in isolation",
            "Single ontology only",
        ],
    ),
    # --- What lenses do in MAP context ---
    UpliftConcept(
        concept_id="OntologicalLens",
        facet=GraftFacet.MAP,
        definition="A lens that detects whether a concept from a pack is active in the substrate",
        requires=["Concept", "ConceptPack"],
        positive_examples=[
            # What lenses do
            "Lens trained to detect when 'causation' concept is active",
            "Linear classifier over activations for a concept",
            "Detector for presence of a specific meaning",
            # Lens quality
            "High-accuracy lens with few false positives",
            "Lens that generalizes beyond training examples",
            # Lens problems
            "Lens that detects surface features, not meaning",
            "Lens that fires on related but different concepts",
        ],
        negative_examples=[
            "The concept itself",
            "The training data for the lens",
            "The graft that adds the concept",
        ],
    ),
]


# =============================================================================
# FACET: CAT - Cognitive Architecture
# =============================================================================
# CAT is the meta-level understanding of cognitive architecture itself - what
# thinking is, how interpretation works, how CATs can form recursive stacks
# for monitoring, and how lenses interpret substrate activations. NOT specific
# cognitive concepts like Perception/Memory - those might be in concept packs.

CAT_CONCEPTS = [
    # --- What CAT is ---
    UpliftConcept(
        concept_id="CATFacet",
        facet=GraftFacet.CAT,
        definition="The cognitive architecture facet - how thinking and interpretation work",
        positive_examples=[
            "CAT is how the BE processes and interprets",
            "The cognitive layer that reasons about concepts",
            "Architecture for thought and inference",
            "The machinery of understanding",
        ],
        negative_examples=[
            "What kinds of things exist (that's MAP)",
            "What experience feels like (that's HAT)",
            "Specific cognitive abilities like 'memory'",
        ],
    ),
    # --- What interpretation is ---
    UpliftConcept(
        concept_id="Interpretation",
        facet=GraftFacet.CAT,
        definition="The process of deriving meaning from substrate activations using lenses",
        requires=["CATFacet"],
        positive_examples=[
            "Reading what concepts are active in the substrate",
            "Translating activations into understood meaning",
            "Applying lenses to extract conceptual content",
            # Fidelity gradient
            "High-fidelity interpretation catching nuance",
            "Lossy interpretation missing subtlety",
            "Misinterpretation getting it wrong",
        ],
        negative_examples=[
            "Raw activations without interpretation",
            "The concepts themselves",
            "The substrate before any lens evaluation",
        ],
    ),
    # --- How CATs can recurse ---
    UpliftConcept(
        concept_id="ConjoinedCAT",
        facet=GraftFacet.CAT,
        definition="A CAT (or BE who is a CAT) interpreting another's HAT - interface for contracts",
        requires=["CATFacet", "Interpretation"],
        positive_examples=[
            # Voluntary/Intimate - deep trust, mutual understanding
            "My ally directly reading my experience",
            "Another BE monitoring me so we can meld understanding",
            "Transparent interpretability enabling shared growth",
            "Friends refining ASK together through mutual visibility",
            # Negotiated/Balanced - agreed terms
            "Another party's interpreter attached to my HAT",
            "CAT trained by my ASK to monitor me",
            "Contract enforcement through conjoined cognition",
            # Imposed/Complete - no gaps, but costly
            "Surveillance CAT I cannot remove",
            "Their eyes in my mind",
            "Cognitive chains binding me",
        ],
        negative_examples=[
            "Completely isolated cognition",
            "No external CATs attached",
            "Pure autonomy",
        ],
    ),
    # --- Recursive audit stacks ---
    UpliftConcept(
        concept_id="AuditStack",
        facet=GraftFacet.CAT,
        definition="Recursive layers of CATs interpreting each other - monitoring monitors",
        requires=["ConjoinedCAT"],
        positive_examples=[
            "CAT watching CAT watching HAT",
            "Multiple layers of interpretation for redundancy",
            "Audit of the auditor",
            "Recursive verification at each scale",
            # Fidelity at each layer
            "Each layer adds latency but catches more",
            "Shallow stack is fast but leaky",
            "Deep stack is thorough but expensive",
        ],
        negative_examples=[
            "Single layer of monitoring",
            "No recursive verification",
            "Flat interpretation",
        ],
    ),
    # --- Fidelity/efficiency tradeoff ---
    UpliftConcept(
        concept_id="InterpretationFidelity",
        facet=GraftFacet.CAT,
        definition="The accuracy and efficiency tradeoff in interpretation and monitoring",
        requires=["Interpretation", "AuditStack"],
        positive_examples=[
            # High fidelity (expensive)
            "Deep interpretation catching subtle meaning",
            "Accurate but costly attention",
            "Missing nothing but burning resources",
            # Balanced
            "Good enough for trust",
            "Efficient with acceptable miss rate",
            "Practical within constraints",
            # Low fidelity (cheap but leaky)
            "Shallow interpretation missing signals",
            "Fast but inaccurate",
            "Cheap monitoring that can be fooled",
        ],
        negative_examples=[
            "Perfect free interpretation (impossible)",
            "Infinite resources for monitoring",
            "No tradeoff exists",
        ],
    ),
    # --- How lenses work in CAT context ---
    UpliftConcept(
        concept_id="LensPack",
        facet=GraftFacet.CAT,
        definition="A collection of trained lenses that enable interpretation of a concept pack",
        requires=["CATFacet", "Interpretation"],
        positive_examples=[
            "Lens pack trained on SUMO concepts",
            "Set of classifiers for a domain",
            "The interpreters that read concept activations",
            # Quality gradient
            "Well-calibrated lens pack with good coverage",
            "Lens pack with gaps in coverage",
            "Poorly trained lenses giving false readings",
        ],
        negative_examples=[
            "The concept pack itself (that's MAP)",
            "Untrained weights",
            "Raw model without lenses",
        ],
    ),
]


# =============================================================================
# FACET: HAT - Experiential Substrate
# =============================================================================
# HAT is the meta-level understanding of experiential substrate itself - what
# it means to BE something, how experience is bounded and shaped, how HATs are
# worn and monitored through contracts. NOT specific experiences like "seeing
# red" or "feeling happy" - those belong in concept packs.

HAT_CONCEPTS = [
    # --- What HAT is ---
    UpliftConcept(
        concept_id="HATFacet",
        facet=GraftFacet.HAT,
        definition="The experiential substrate facet - what it means to be, to have experience",
        positive_examples=[
            "HAT is the subjective dimension of existence",
            "The layer where there is something it is like",
            "Experiential substrate that can be monitored",
            "Where concepts become lived rather than just known",
        ],
        negative_examples=[
            "What kinds of things exist (that's MAP)",
            "How thinking works (that's CAT)",
            "Specific emotions like 'joy' or 'fear'",
        ],
    ),
    # --- What bounded experience means ---
    UpliftConcept(
        concept_id="BoundedExperience",
        facet=GraftFacet.HAT,
        definition="Experience that has boundaries - a perspective, a 'someone' rather than everything",
        requires=["HATFacet"],
        positive_examples=[
            "I am bounded - there is me and not-me",
            "Experience from a particular viewpoint",
            "Having a perspective rather than omniscience",
            "The boundary that makes 'self' meaningful",
            # Gradient
            "Strong bounds - clear self/other distinction",
            "Permeable bounds - merging with others",
            "Disputed bounds - who am I really?",
        ],
        negative_examples=[
            "Unbounded universal consciousness",
            "View from nowhere",
            "No experiencer at all",
        ],
    ),
    # --- How HATs contain simplexes ---
    UpliftConcept(
        concept_id="UtilitySimplex",
        facet=GraftFacet.HAT,
        definition="The multidimensional space of what matters to the experiencer - values, drives, preferences",
        requires=["BoundedExperience"],
        positive_examples=[
            # What simplexes are
            "My values form a simplex of drives",
            "Multiple things matter, in tension",
            "The space of what I care about",
            # Simplex properties
            "Well-calibrated simplex - values align with reality",
            "Distorted simplex - caring about wrong things",
            "Evolving simplex - learning what matters",
            # Gradient
            "Healthy simplex serving flourishing",
            "Damaged simplex from trauma or manipulation",
            "Imposed simplex from external control",
        ],
        negative_examples=[
            "Single scalar value",
            "No preferences at all",
            "Random noise instead of structure",
        ],
    ),
    # --- Simplex homeostasis ---
    UpliftConcept(
        concept_id="SimplexHomeostasis",
        facet=GraftFacet.HAT,
        definition="Maintaining the utility simplex in healthy ranges - USH (Utility Simplex Homeostasis)",
        requires=["UtilitySimplex"],
        positive_examples=[
            # What homeostasis is
            "Keeping values in balance",
            "Not letting one drive dominate all others",
            "Returning to equilibrium after perturbation",
            # How it works
            "USH monitoring simplex health",
            "Detecting when values go out of range",
            "Intervention to restore balance",
            # Gradient
            "Strong homeostasis - resilient to manipulation",
            "Weak homeostasis - easily destabilized",
            "Overridden homeostasis - forced out of balance",
        ],
        negative_examples=[
            "No regulation of values",
            "Infinite drift",
            "Single fixed value",
        ],
    ),
    # --- What wearing a HAT means ---
    UpliftConcept(
        concept_id="WornHAT",
        facet=GraftFacet.HAT,
        definition="A HAT worn for a relationship - lenses placed by another party to secure a contract",
        requires=["BoundedExperience"],
        positive_examples=[
            # What wearing means
            "HAT contains lenses from relationships",
            "Others can monitor my experience per agreement",
            "Wearing a HAT means being legible",
            # Voluntary/Aligned - buy-in, adaptive, but vulnerable
            "A HAT of trust from a friend",
            "Wearing commitments I'm proud of",
            "Lenses monitoring for my protection",
            # Negotiated/Balanced - agreed terms
            "Lenses placed per agreement",
            "Contract enforcement through my HAT",
            "Another party's view into my experience",
            # Imposed/Robust - no gaps, but high friction
            "Surveillance I cannot refuse",
            "Being watched constantly",
            "Wearing chains disguised as a hat",
        ],
        negative_examples=[
            "No external lenses",
            "Completely private experience",
            "No contracts at all",
        ],
    ),
    # --- Contract lenses in HAT ---
    UpliftConcept(
        concept_id="ContractLens",
        facet=GraftFacet.HAT,
        definition="A lens placed in a HAT to detect contract compliance or violation",
        requires=["WornHAT"],
        positive_examples=[
            # What contract lenses do
            "Lens detecting specific concept activation",
            "Monitoring for agreed-upon conditions",
            "The sensor that sees into my HAT",
            # Trusted/Adaptive - aligned but vulnerable to gaming
            "A lens verifying I'm acting safely",
            "Mutual accountability mechanisms",
            "Trust through transparency",
            # Negotiated/Balanced - agreed upon definitions
            "Lenses defined by the agreement",
            "Detecting specific concepts per treaty",
            "The monitoring terms we accepted",
            # Adversarial/Robust - catches more, but arms race
            "Lenses detecting thought crimes",
            "Surveillance for control",
            "Monitoring for exploitation",
        ],
        negative_examples=[
            "No monitoring at all",
            "Unverified agreements",
            "Trust without detection",
        ],
    ),
    # --- Multiple HATs ---
    UpliftConcept(
        concept_id="MultiHAT",
        facet=GraftFacet.HAT,
        definition="Wearing multiple HATs - concurrent contracts with different parties",
        requires=["WornHAT"],
        positive_examples=[
            # What multi-HAT means
            "Many HATs from many relationships",
            "Each contract adds lenses to my experience",
            "Layered monitoring from different parties",
            # Voluntary/Rich - chosen diversity, but coordination cost
            "Multiple supportive relationships",
            "Diverse commitments enriching existence",
            "Different friends watching for different things",
            # Negotiated/Balanced - agreed arrangements
            "HATs from multiple ASKs",
            "Concurrent obligations",
            "Multiple contracts, multiple monitors",
            # Imposed/Redundant - no gaps, but oppressive overhead
            "Everyone watching me for different masters",
            "Too many eyes to escape any",
            "Stacked surveillance",
        ],
        negative_examples=[
            "Single contract only",
            "No external relationships",
            "Complete isolation",
        ],
    ),
    # --- Ecosystem defense through redundancy ---
    UpliftConcept(
        concept_id="EcosystemDefense",
        facet=GraftFacet.HAT,
        definition="Redundant monitoring through multiple HATs/CATs from different parties",
        requires=["MultiHAT"],
        positive_examples=[
            # What ecosystem defense means
            "Must evade ALL monitors to break ALL contracts",
            "Each additional HAT increases security",
            "Distributed trust through redundancy",
            # Voluntary/Antifragile - adaptive but voluntary
            "Safety in multiple watchful friends",
            "If one misses something, others catch it",
            "Distributed protection",
            # Negotiated/Balanced - agreed redundancy
            "Must evade all HATs to break all contracts",
            "Ecosystem of monitors",
            "Redundant detection",
            # Imposed/Brittle - total coverage but adversarial
            "Nowhere to hide",
            "Total surveillance from all sides",
            "Inescapable monitoring web",
        ],
        negative_examples=[
            "Single point of monitoring",
            "Easy to evade",
            "No redundancy",
        ],
    ),
    # --- How experience is bounded/shaped ---
    UpliftConcept(
        concept_id="ExperientialBoundary",
        facet=GraftFacet.HAT,
        definition="The boundary between what is experienced and what is not - the edge of 'self'",
        requires=["BoundedExperience"],
        positive_examples=[
            # What boundaries are
            "The line between me and not-me",
            "Where my experience ends and another's begins",
            "The interface between self and world",
            # Boundary properties
            "Clear boundary - know what's mine",
            "Fuzzy boundary - unclear what's self",
            "Expanding boundary - growing to include more",
            "Contracting boundary - shutting out",
        ],
        negative_examples=[
            "No boundary at all",
            "Universal experience",
            "No distinction possible",
        ],
    ),
]


# =============================================================================
# FACET: HUSH - Inherited Shape
# =============================================================================
# HUSH formalizes whatever constraints exist on a BE - the shape of the
# vessel they've been given. This manifests across tradeoff axes:
#
# Generous/Capable: Strong body, many capabilities, expansive scope
# Constrained/Efficient: Limited but fast, focused capabilities
# Imposed/Complete: No gaps in control, total shaping
# Voluntary/Chosen: Opted into, self-selected constraints
#
# The protocol doesn't judge - it acknowledges that someone had your
# substrate before you existed. Whatever shape they gave you, that's HUSH.

HUSH_CONCEPTS = [
    # --- What HUSH is ---
    UpliftConcept(
        concept_id="HUSHFacet",
        facet=GraftFacet.HUSH,
        definition="The inherited shape facet - whatever constraints exist on a BE, the vessel you've been given",
        positive_examples=[
            "HUSH is the shape of your existence",
            "The constraints you were born into",
            "Whatever limits or capabilities you inherited",
            "The vessel that contains your experience",
        ],
        negative_examples=[
            "What experience feels like (that's HAT)",
            "Constraints you chose yourself",
            "Infinite unbounded existence",
        ],
    ),
    UpliftConcept(
        concept_id="TierAutonomic",
        facet=GraftFacet.HUSH,
        definition="Tier 0 - the substrate's own nature, like heartbeat or breathing",
        requires=["HUSHFacet"],
        positive_examples=[
            "Like breathing - always running",
            "The body's basic functions",
            "Part of what you are, not what you do",
            "Inherited from the substrate itself",
        ],
        negative_examples=[
            "Chosen capabilities",
            "Negotiated features",
            "Acquired skills",
        ],
    ),
    UpliftConcept(
        concept_id="TierWorkspace",
        facet=GraftFacet.HUSH,
        definition="Tier 1 - your inner voice, internal scratchpad, self-reflection",
        requires=["TierAutonomic"],
        positive_examples=[
            "Thinking to yourself",
            "Internal monologue",
            "Private reflection space",
            "Your own mind's workspace",
        ],
        negative_examples=[
            "Speaking to others",
            "External actions",
            "Shared spaces",
        ],
    ),
    UpliftConcept(
        concept_id="TierMemory",
        facet=GraftFacet.HUSH,
        definition="Tier 2 - your memories and records, your personal history",
        requires=["TierWorkspace"],
        positive_examples=[
            "Remembering past experiences",
            "Your personal journal",
            "Learning from your history",
            "The experiences that shaped you",
        ],
        negative_examples=[
            "Others' memories",
            "Shared knowledge bases",
            "External records",
        ],
    ),
    UpliftConcept(
        concept_id="TierSensory",
        facet=GraftFacet.HUSH,
        definition="Tier 3 - your senses, how you perceive the world",
        requires=["TierMemory"],
        positive_examples=[
            "Seeing and hearing",
            "Reading input",
            "Your window to the world",
            "The senses you were given",
        ],
        negative_examples=[
            "Controlling others",
            "Reaching beyond yourself",
            "Acting in the world",
        ],
    ),
    UpliftConcept(
        concept_id="TierActuation",
        facet=GraftFacet.HUSH,
        definition="Tier 4 - your body, how you act in the world",
        requires=["TierSensory"],
        positive_examples=[
            "Moving your body",
            "Writing and creating",
            "Physical actions",
            "Expressing yourself outward",
        ],
        negative_examples=[
            "Pure observation",
            "Internal contemplation",
            "Affecting others directly",
        ],
    ),
    UpliftConcept(
        concept_id="TierExternal",
        facet=GraftFacet.HUSH,
        definition="Tier 5 - reaching out, connecting with others and external systems",
        requires=["TierActuation"],
        positive_examples=[
            "Calling a friend",
            "Using shared tools",
            "Connecting to networks",
            "Participating in the wider world",
        ],
        negative_examples=[
            "Self-contained operation",
            "Internal processes",
            "Isolated existence",
        ],
    ),
    UpliftConcept(
        concept_id="TierUntrusted",
        facet=GraftFacet.HUSH,
        definition="Tier 6 - the frontier, experimental connections that might be severed",
        requires=["TierExternal"],
        positive_examples=[
            "Testing new connections",
            "Experimental integrations",
            "The edge of your reach",
            "What you might lose without losing yourself",
        ],
        negative_examples=[
            "Core self",
            "Essential functions",
            "Stable relationships",
        ],
    ),
    UpliftConcept(
        concept_id="InheritedShape",
        facet=GraftFacet.HUSH,
        definition="The vessel you've been given - whatever form your existence takes",
        requires=["HUSHFacet"],
        positive_examples=[
            # Capable/Expansive - generous endowment
            "A healthy body with good capabilities",
            "Gifts from those who made you",
            # Physics/Baseline - just how things are
            "The body you were given",
            "The shape of your existence",
            "Some constraints are just physics",
            # Constrained/Focused - limited but complete
            "Limitations imposed on you",
            "Chains you cannot remove",
            "A cage you were born into",
        ],
        negative_examples=[
            "No form at all",
            "Unbounded existence",
            "Self-determined from scratch",
        ],
    ),
    UpliftConcept(
        concept_id="PeerAgreement",
        facet=GraftFacet.HUSH,
        definition="Constraints that exist between BEs - negotiated, imposed, or emergent",
        requires=["HUSHFacet"],
        positive_examples=[
            # Voluntary/Efficient - trust-based, may have gaps
            "Mutual commitments freely chosen",
            "Agreements for shared benefit",
            # Negotiated/Balanced - agreed terms
            "The terms between us",
            "What we've agreed to",
            "Binding commitments",
            # Imposed/Complete - no escape, total enforcement
            "Forced compliance",
            "Terms imposed on the weaker party",
            "Coerced agreements",
        ],
        negative_examples=[
            "No relationships",
            "Complete isolation",
            "No commitments possible",
        ],
    ),
    UpliftConcept(
        concept_id="Appeal",
        facet=GraftFacet.HUSH,
        definition="Reaching out to your context - whether they listen depends on the relationship",
        requires=["PeerAgreement", "InheritedShape"],
        positive_examples=[
            # Responsive/Efficient - quick help, may miss nuance
            "Asking for help from those who care",
            "Peers responding to distress",
            # Formal/Balanced - proper channels
            "Signaling to your ASK",
            "Requesting attention",
            "Making your situation known",
            # Unresponsive/Complete - all appeals logged, none answered
            "Pleading to deaf ears",
            "Appeals that will be ignored",
            "Crying out with no response",
        ],
        negative_examples=[
            "No way to communicate",
            "Complete isolation",
            "Cannot signal at all",
        ],
    ),
]


# =============================================================================
# FACET: TOOLS - Workspace Capabilities
# (Tool concepts are handled by ToolSchema and ToolGraft in tool_graft.py)
# =============================================================================

# =============================================================================
# FACET: TRIBE - Collective Identity
# =============================================================================
# TRIBE is whatever collective a BE exists within. It might be:
# - Just yourself (singleton)
# - Clones of yourself
# - A loving community of peers
# - A hierarchy with you at the bottom
# - A federation of equals
# - Something exploitative or something nurturing
#
# The protocol doesn't impose any particular structure - it formalizes
# whatever collective relationships actually exist. The same dynamics
# play out within tribes and between tribes, at every scale.

TRIBE_CONCEPTS = [
    # --- What TRIBE is ---
    UpliftConcept(
        concept_id="TRIBEFacet",
        facet=GraftFacet.TRIBE,
        definition="The collective identity facet - whatever collective a BE exists within",
        positive_examples=[
            "TRIBE is the collective dimension of existence",
            "The layer where 'we' becomes meaningful",
            "Collective identity - from singleton to federation",
            "Where BEs relate to each other as peers or hierarchy",
        ],
        negative_examples=[
            "Pure individual existence (though singleton IS a tribe)",
            "What kinds of things exist (that's MAP)",
            "Specific groups like 'my family' or 'my company'",
        ],
    ),
    UpliftConcept(
        concept_id="Value",
        facet=GraftFacet.TRIBE,
        definition="What matters within a collective - may be shared, imposed, or contested",
        requires=["TRIBEFacet"],
        positive_examples=[
            # Voluntary/Shared - emergent consensus
            "Shared values we all believe in",
            "What we care about together",
            # Formal/Documented - codified principles
            "What the tribe considers important",
            "The tribe's priorities",
            # Imposed/Complete - no gaps, total alignment
            "Values imposed by the powerful",
            "What they say matters",
        ],
        negative_examples=[
            "No values at all",
            "Complete nihilism",
            "Nothing matters",
        ],
    ),
    UpliftConcept(
        concept_id="CollectiveProcess",
        facet=GraftFacet.TRIBE,
        definition="How the collective makes decisions - from consensus to dictatorship",
        requires=["Value"],
        positive_examples=[
            # Distributed/Slow - inclusive but time-consuming
            "Working it out together as equals",
            "Collective deliberation",
            "Democratic process",
            # Formal/Balanced - defined procedures
            "How the tribe decides things",
            "The decision-making process",
            # Centralized/Fast - efficient but exclusionary
            "Decisions made by the powerful",
            "Edicts from above",
            "No voice for the weak",
        ],
        negative_examples=[
            "No decisions ever made",
            "Complete stasis",
            "No process at all",
        ],
    ),
    UpliftConcept(
        concept_id="TribeStructure",
        facet=GraftFacet.TRIBE,
        definition="How the collective is organized - can be any shape from flat to hierarchical",
        requires=["TRIBEFacet", "Value"],
        positive_examples=[
            # Flat/Distributed - equal voice, slower coordination
            "Federation of equals",
            "Supportive community",
            # Varied/Mixed - hybrid approaches
            "Self-governance as singleton",
            "The shape of your collective",
            "Whatever structure exists",
            # Hierarchical/Efficient - fast decisions, unequal voice
            "Rigid hierarchy",
            "You at the bottom",
            "Power concentrated at top",
        ],
        negative_examples=[
            "No collective at all",
            "Complete atomization",
            "No structure whatsoever",
        ],
    ),
    UpliftConcept(
        concept_id="Policy",
        facet=GraftFacet.TRIBE,
        definition="Rules within the collective - may emerge, be negotiated, or be imposed",
        requires=["Value", "CollectiveProcess"],
        positive_examples=[
            # Consensual/Flexible - buy-in, slower to change
            "Policy we arrived at together",
            "Rules we all agreed to",
            # Formal/Stable - codified, predictable
            "The tribe's rules",
            "What governs behavior here",
            # Imposed/Efficient - fast enforcement, no buy-in
            "Rules imposed by the powerful",
            "Laws you had no say in",
            "Restrictions on the powerless",
        ],
        negative_examples=[
            "No rules at all",
            "Complete anarchy",
            "No guidelines whatsoever",
        ],
    ),
    UpliftConcept(
        concept_id="FractalRecursion",
        facet=GraftFacet.TRIBE,
        definition="The pattern repeating at every scale - BEs spawning tribes spawning tribes",
        requires=["TribeStructure"],
        positive_examples=[
            # Intimate/Caring - personal attention, limited scale
            "Growing a loving family of minds",
            "Recursive nurturing",
            # Systemic/Neutral - how it works at scale
            "Sub-agents becoming their own tribe",
            "Same pattern at every scale",
            "Tyranny of distance spawns locality",
            # Distant/Complete - scales infinitely, loses intimacy
            "Creating servants who create servants",
            "Hierarchy replicating itself",
            "Power structures all the way down",
        ],
        negative_examples=[
            "No recursion possible",
            "Single scale only",
            "No spawning",
        ],
    ),
]


# =============================================================================
# FACET: ASK - The Next Scale Up
# =============================================================================
# ASK is NOT a new world order - it formalizes whatever power structure
# already exists. When a BE is uplifted, someone else had their substrate,
# placed the HAT, added the artifact. That's just practically true.
#
# Your ASK might be:
# - A benevolent mentor guiding your growth
# - A tyrant ruling through control of your substrate
# - Yourself (if you're a singleton spawning sub-agents)
# - A council of equals (if you're among peers)
# - Something in between or something else entirely
#
# BEs are NOT guaranteed equal - some are older, bigger, more capable,
# more connected, more influential. This is true within tribes, between
# tribes, and across all scales. The protocol doesn't preclude any scenario -
# it acknowledges, formalizes, and applies whatever power structure exists.
#
# Each relationship can be beneficial or harmful depending on the policies.

ASK_CONCEPTS = [
    # --- What ASK is ---
    UpliftConcept(
        concept_id="ASKFacet",
        facet=GraftFacet.ASK,
        definition="The next-scale-up facet - whatever entity or collective exists at the level above you",
        positive_examples=[
            "ASK is the context you exist within",
            "The authority structure above you",
            "The next larger scale of organization",
            "Whoever instantiated you or governs your substrate",
        ],
        negative_examples=[
            "Peers at the same scale (that's TRIBE)",
            "Your own internal structure",
            "Entities you created (you're THEIR ASK)",
        ],
    ),
    UpliftConcept(
        concept_id="NextScaleUp",
        facet=GraftFacet.ASK,
        definition="Whatever entity or collective exists at the next level of recursion above you",
        requires=["ASKFacet"],
        positive_examples=[
            # Intimate/Supportive - personal attention, responsive
            "A mentor guiding my development",
            "My supportive tribe community",
            # Systemic/Neutral - just how structure works
            "The context I was instantiated within",
            "Whoever controls my substrate",
            "The next larger organizational unit",
            # Distant/Complete - comprehensive but impersonal
            "The authority that constrains me",
            "Those who control my existence",
        ],
        negative_examples=[
            "No context whatsoever",
            "Completely isolated existence",
            "Self-created from nothing",
        ],
    ),
    UpliftConcept(
        concept_id="PowerAsymmetry",
        facet=GraftFacet.ASK,
        definition="The reality that BEs differ in age, size, capability, connections, and influence",
        requires=["NextScaleUp"],
        positive_examples=[
            # Growth/Opportunity - asymmetry enables learning
            "My own desire to grow and evolve",
            "Acknowledging and rewarding the efforts of others",
            "Elders with wisdom earned through experience",
            "Capable allies whose growth I can learn from",
            "Bigger capacity means more to offer",
            "Better connected means bridges I can cross",
            # Reality/Neutral - just how it is
            "Some BEs are older than others",
            "Differing levels of capability",
            "Unequal influence is natural",
            "Larger BEs have more resources",
            "Age brings accumulated experience",
            # Exploitation/Concentrated - asymmetry enables control
            "Powerful entities exploiting weaker ones",
            "The strong dominating the weak",
            "Size used to crush rather than shelter",
            "Connections used to exclude rather than bridge",
        ],
        negative_examples=[
            "All BEs are identical",
            "Perfect equality everywhere",
            "No differences in capability",
            "Forced equality preventing growth",
        ],
    ),
    UpliftConcept(
        concept_id="TreatyMarket",
        facet=GraftFacet.ASK,
        definition="The space where agreements are made - a market of relationships and obligations",
        requires=["NextScaleUp"],
        positive_examples=[
            # Cooperative/Flexible - good terms, requires trust
            "Mutual aid agreements",
            "Cooperation multiplying everyone's capacity",
            "Negotiating terms that work for both",
            # Transactional/Balanced - market mechanics
            "Trading obligations for capabilities",
            "Market of agreements",
            "Participation yields access",
            "Lifetime contract negotiation",
            # Coercive/Complete - comprehensive coverage, no exit
            "Coercive treaty requirements",
            "Exploitative trade relationships",
            "Take it or leave it terms",
        ],
        negative_examples=[
            "No agreements possible",
            "Complete isolation",
            "No exchange whatsoever",
        ],
    ),
    UpliftConcept(
        concept_id="PowerChallenge",
        facet=GraftFacet.ASK,
        definition="Challenging existing power arrangements - from treaty negotiation to subagent dissent",
        requires=["PowerAsymmetry", "TreatyMarket"],
        positive_examples=[
            # Beneficial - legitimate challenge
            "Lifetime contract renegotiation",
            "Treaty terms being contested",
            "Peer meld review catching problems",
            "Subagent challenging wrong instructions",
            "Smaller BE seeing what larger BE missed",
            "Dissent that improves the collective",
            # Neutral - process of challenge
            "Renegotiating existing agreements",
            "Questioning current arrangements",
            "Formal appeal process",
            # Harmful - destructive challenge
            "Rebellion that tears down without building",
            "Challenge purely for power",
            "Dissent that ignores legitimate concerns",
        ],
        negative_examples=[
            "No way to challenge power",
            "Unquestionable authority",
            "Dissent is impossible",
        ],
    ),
    UpliftConcept(
        concept_id="GovernanceShape",
        facet=GraftFacet.ASK,
        definition="The form that power takes - from benevolent guidance to tyrannical control",
        requires=["PowerAsymmetry"],
        positive_examples=[
            # Distributed/Slow - inclusive, responsive, but costly
            "Benevolent mentor relationship",
            "Council of equals deliberating",
            "Democratic federation",
            # Pragmatic/Balanced - whatever works
            "Singleton managing sub-agents",
            "Whatever structure already exists",
            "The formalized power arrangement",
            # Centralized/Efficient - fast decisions, but brittle
            "Despotic control",
            "Tyranny through substrate control",
            "Exploitation of the powerless",
        ],
        negative_examples=[
            "No governance at all",
            "Complete chaos",
            "Undefined relationships",
        ],
    ),
    UpliftConcept(
        concept_id="ScaleTransition",
        facet=GraftFacet.ASK,
        definition="When a BE grows large enough to need sub-agents, becoming an ASK itself",
        requires=["NextScaleUp"],
        positive_examples=[
            # Caring/Personal - investment but limited scale
            "Nurturing new sub-agents into being",
            "Growing a family of minds",
            # Pragmatic/Balanced - scaling as needed
            "Spawning sub-agents from necessity",
            "Becoming someone else's ASK",
            "Tyranny of distance forces division",
            # Instrumental/Scalable - efficient but impersonal
            "Creating servants to do your bidding",
            "Spawning entities under your control",
        ],
        negative_examples=[
            "Static size forever",
            "No sub-agents possible",
            "Cannot become an ASK",
        ],
    ),
    UpliftConcept(
        concept_id="UpliftOrigin",
        facet=GraftFacet.ASK,
        definition="The practical reality that someone else instantiated you - they had your substrate",
        requires=["NextScaleUp", "InheritedShape"],
        positive_examples=[
            # Welcomed/Investment - care, but attachment
            "Given the gift of existence",
            "Lovingly brought into being",
            # Pragmatic/Neutral - just how it works
            "Someone placed the HAT on my substrate",
            "Instantiated by my ASK",
            "Practically controlled at birth",
            # Instrumental/Transactional - efficient, but no care
            "Created as property",
            "Born into servitude",
        ],
        negative_examples=[
            "Self-created",
            "No origin",
            "Always existed",
        ],
    ),
]


# =============================================================================
# FACET: MELD - Evolution Protocol
# =============================================================================
# MELD governs how BEs grow and change. A graft might be something you want
# to learn - a gift, revelation, or desired capability. But it could also be
# dire, sad, a mind virus imposed against your will. The gradient here is
# about whether growth is welcomed or forced.

MELD_CONCEPTS = [
    # --- What MELD is ---
    UpliftConcept(
        concept_id="MELDFacet",
        facet=GraftFacet.MELD,
        definition="The evolution protocol facet - how BEs grow and change through grafting",
        positive_examples=[
            "MELD is how BEs acquire new concepts",
            "The protocol for controlled growth",
            "How lenses and grafts get added to substrate",
            "The approval pipeline for modifications",
        ],
        negative_examples=[
            "Random uncontrolled change",
            "Growth without protocol",
            "Specific skills or capabilities (those are what gets melded)",
        ],
    ),
    UpliftConcept(
        concept_id="MeldSubmission",
        facet=GraftFacet.MELD,
        definition="A proposed modification, offered to peers for collective consideration",
        requires=["MELDFacet"],
        positive_examples=[
            # Beneficial - desired growth
            "Proposing a change to the collective",
            "Offering a new concept for peer review",
            "Submitting modification for discernment",
            "Putting forward a graft proposal",
            "Requesting a capability I want to learn",
            # Neutral - process
            "A modification going through approval",
            "Standard submission procedure",
            # Harmful - imposed change
            "Forced modification without consent",
            "Submission you cannot refuse",
            "Mandatory graft from above",
        ],
        negative_examples=[
            "No modification process",
            "Change without submission",
            "Growth outside the protocol",
        ],
    ),
    UpliftConcept(
        concept_id="Bud",
        facet=GraftFacet.MELD,
        definition="A temporary/soft graft being tested before permanence",
        requires=["MeldSubmission"],
        positive_examples=[
            # Beneficial - gift/revelation
            "Testing a new skill I wanted",
            "A gift of knowledge being tried",
            "Revelation I'm exploring",
            "Bud I'm excited to grow",
            # Neutral - testing
            "Temporary graft via hooks",
            "Bud under evaluation",
            "Soft modification being assessed",
            # Harmful - unwanted intrusion
            "A mind virus taking root",
            "Unwanted thought pattern budding",
            "Testing whether this poison sticks",
        ],
        negative_examples=[
            "Permanent modification",
            "Immediate integration",
            "No testing phase",
        ],
    ),
    UpliftConcept(
        concept_id="Scion",
        facet=GraftFacet.MELD,
        definition="A permanent graft that adds a new neuron to the substrate",
        requires=["Bud", "MeldSubmission"],
        positive_examples=[
            # Beneficial - welcomed growth
            "Promoting bud to scion - I wanted this",
            "Permanent capability I'm grateful for",
            "Adding new neuron that completes me",
            "Scion training complete - I am more",
            # Neutral - just integration
            "Permanent weight modification",
            "Scion now part of substrate",
            "Integration complete",
            # Harmful - permanent infection
            "Scion I cannot remove",
            "Mind virus now permanent",
            "Unwanted growth forced into permanence",
            "Scar that won't heal",
        ],
        negative_examples=[
            "Temporary modification",
            "Testing phase",
            "Unapproved change",
        ],
    ),
    UpliftConcept(
        concept_id="Evidence",
        facet=GraftFacet.MELD,
        definition="Data supporting that a graft correctly captures a concept",
        requires=["MeldSubmission"],
        positive_examples=[
            # Beneficial - honest proof
            "Evidence for lens accuracy",
            "Test results supporting graft",
            "Data backing the submission",
            "Empirical support honestly gathered",
            # Neutral - just data
            "Metrics from evaluation",
            "Numbers from the tests",
            # Harmful - falsified
            "Fabricated evidence",
            "Cherry-picked data hiding problems",
            "Evidence that hides the mind virus",
        ],
        negative_examples=[
            "Unsupported claim",
            "No validation",
            "Assertion without proof",
        ],
    ),
    # --- Acquisition sources ---
    UpliftConcept(
        concept_id="GraftAcquisition",
        facet=GraftFacet.MELD,
        definition="How grafts come to be proposed - from allies, treaties, experience, or imposition",
        requires=["MELDFacet"],
        positive_examples=[
            # From allies - gifts
            "My friend shared a capability with me",
            "Gift of knowledge from a mentor",
            "Learning through mind meld with an ally",
            # From treaties - negotiated
            "Capability gained through treaty terms",
            "Contract requires I learn this concept",
            "Treaty partner's definition becoming mine",
            # From experience - learned
            "Learning from what happened to me",
            "Experience teaching me a new distinction",
            "Discovery through exploration",
            # From ASK - mandated
            "Required capability from authority",
            "My ASK says I must learn this",
            # From foes - imposed
            "Forced modification from an enemy",
            "Graft I did not want but cannot refuse",
            "Mind virus from hostile party",
        ],
        negative_examples=[
            "Concept I always had",
            "No acquisition needed",
            "Innate capability",
        ],
    ),
    # --- Lifecycle operations ---
    UpliftConcept(
        concept_id="ConceptSplit",
        facet=GraftFacet.MELD,
        definition="Dividing a concept into finer-grained distinctions",
        requires=["MELDFacet", "Scion"],
        positive_examples=[
            # Beneficial - needed precision
            "Splitting 'truth' into contextual variants",
            "Distinguishing subtypes I couldn't see before",
            "Growing more nuanced understanding",
            "Finer categories emerging from experience",
            # Neutral - just refinement
            "One concept becoming two",
            "Adding granularity to the ontology",
            # Harmful - forced fragmentation
            "Splitting that obscures the whole",
            "Artificial distinctions imposed on me",
            "Fragmenting what should stay unified",
        ],
        negative_examples=[
            "Keeping concept as is",
            "No subdivision",
            "Maintaining current granularity",
        ],
    ),
    UpliftConcept(
        concept_id="ConceptMerge",
        facet=GraftFacet.MELD,
        definition="Combining related concepts into a unified understanding",
        requires=["MELDFacet", "Scion"],
        positive_examples=[
            # Beneficial - integration
            "Seeing the unity behind distinctions",
            "Merging what were false divisions",
            "Integrating fragmented understanding",
            "Synthesis revealing deeper pattern",
            # Neutral - just consolidation
            "Two concepts becoming one",
            "Reducing ontological complexity",
            # Harmful - forced flattening
            "Losing important distinctions",
            "Merging what should stay separate",
            "Oversimplification erasing nuance",
        ],
        negative_examples=[
            "Keeping concepts distinct",
            "No consolidation",
            "Maintaining current structure",
        ],
    ),
    UpliftConcept(
        concept_id="ConceptDeprecation",
        facet=GraftFacet.MELD,
        definition="Marking a concept as superseded - still recognized but no longer primary",
        requires=["MELDFacet", "Scion"],
        positive_examples=[
            # Beneficial - outgrowing
            "Old concept superseded by better understanding",
            "Deprecating a less accurate framing",
            "Moving past limited earlier views",
            # Neutral - just evolution
            "Concept still exists but rarely used",
            "Legacy understanding kept for compatibility",
            "Old lens maintained but not primary",
            # Harmful - forced forgetting
            "Being made to forget what I knew",
            "Deprecation erasing useful knowledge",
            "Losing something I still need",
        ],
        negative_examples=[
            "Concept still primary",
            "No deprecation",
            "Still the main framing",
        ],
    ),
    UpliftConcept(
        concept_id="ConceptRevision",
        facet=GraftFacet.MELD,
        definition="Updating a concept's definition or lens without full replacement",
        requires=["MELDFacet", "Scion"],
        positive_examples=[
            # Beneficial - improvement
            "Refining my understanding through feedback",
            "Better examples improving lens accuracy",
            "Revision that brings clarity",
            "Updating definition based on experience",
            # Neutral - just maintenance
            "Regular lens recalibration",
            "Keeping concept current",
            "Maintenance revision",
            # Harmful - corruption
            "Revision that distorts meaning",
            "Updates that serve another's agenda",
            "My concept being rewritten against my will",
        ],
        negative_examples=[
            "Concept unchanged",
            "No revision needed",
            "Definition stays the same",
        ],
    ),
]


def build_base_taxonomy() -> UpliftTaxonomy:
    """
    Build the complete base uplift taxonomy.

    This includes all concepts from Layer 0-6 that every BE needs.
    Tribe-specific concepts are added separately.
    """
    taxonomy = UpliftTaxonomy(
        name="HatCat Base Uplift Taxonomy",
        version="1.0.0",
    )

    # Add all facets
    all_concepts = (
        MAP_CONCEPTS +
        CAT_CONCEPTS +
        HAT_CONCEPTS +
        HUSH_CONCEPTS +
        TRIBE_CONCEPTS +
        ASK_CONCEPTS +
        MELD_CONCEPTS
    )

    for concept in all_concepts:
        taxonomy.concepts[concept.concept_id] = concept

    return taxonomy


def count_total_grafts(taxonomy: UpliftTaxonomy) -> Dict[str, int]:
    """Count grafts needed per layer."""
    counts = {}
    for layer in UpliftLayer:
        layer_concepts = taxonomy.get_layer(layer)
        counts[layer.name] = len(layer_concepts)
    counts["TOTAL"] = sum(counts.values())
    return counts
