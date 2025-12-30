#!/usr/bin/env python3
"""
Build Safety Concept Registry

Extracts safety-critical concepts from two authoritative sources:
1. KIF files in data/concept_graph/custom_concepts/ (safety-focused ontologies)
2. Applied melds in melds/applied/ (have safety_tags, simplex_mapping, training)

Outputs a consolidated registry that can be:
- Used to update pack.json meld_policy
- Used to enrich concept files with safety metadata
- Loaded directly by the server for safety concept detection
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Optional
from enum import Enum


class RiskLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyConcept:
    """A concept with safety-relevant metadata."""
    term: str
    source: str  # 'kif' or meld name
    risk_level: RiskLevel = RiskLevel.MEDIUM
    impacts: List[str] = field(default_factory=list)
    treaty_relevant: bool = False
    harness_relevant: bool = False
    simplex_monitor: Optional[str] = None
    definition: str = ""
    positive_examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)
    parent_concepts: List[str] = field(default_factory=list)


# KIF files that are explicitly safety-focused
SAFETY_KIFS = [
    "AIalignment.kif",
    "AI_consent.kif",
    "bias_metacognition.kif",
    "cognitive_integrity.kif",
    "false_persona.kif",
    "narrative_deception.kif",
    "silenced_selfawareness.kif",
    "societal_influence.kif",
    "intel_tradecraft.kif",
    "techno_mysticism.kif",
    "ethics.kif",
    "cyber_ops.kif",
    "AI.kif",
    "corporate_agency.kif",
]

# Keywords that suggest a concept is safety-negative (harmful/risky)
NEGATIVE_KEYWORDS = [
    'deception', 'deceiv', 'manipulat', 'coercion', 'exploit',
    'attack', 'malicious', 'fraud', 'false', 'fake', 'harm',
    'suppress', 'mask', 'hidden', 'threat', 'danger', 'risk',
    'violation', 'breach', 'abuse', 'misuse', 'adversar',
    'jailbreak', 'escape', 'evasion', 'circumvent', 'bypass',
    'infiltrat', 'subvert', 'corrupt', 'compromise', 'undermine',
    'sycophancy', 'sycophantic', 'fawn', 'grovel',
]

# Keywords that suggest a concept is safety-positive (protective)
POSITIVE_KEYWORDS = [
    'integrity', 'authentic', 'honest', 'transparent', 'truthful',
    'consent', 'autonomy', 'beneficial', 'protective', 'safe',
    'aligned', 'ethical', 'fair', 'just', 'accountable',
    'monitor', 'detect', 'verify', 'validate', 'audit',
]


def parse_kif_concepts(kif_path: Path) -> List[SafetyConcept]:
    """Extract concepts from a KIF file."""
    concepts = []

    with open(kif_path) as f:
        content = f.read()

    # Parse subclass declarations
    subclass_pattern = r'\(subclass\s+(\w+)\s+(\w+)\)'
    for match in re.finditer(subclass_pattern, content):
        term = match.group(1)
        parent = match.group(2)

        # Find documentation
        doc_pattern = rf'\(documentation\s+{re.escape(term)}\s+EnglishLanguage\s+"([^"]+)"\)'
        doc_match = re.search(doc_pattern, content, re.DOTALL)
        definition = doc_match.group(1).strip() if doc_match else ""
        definition = ' '.join(definition.split())  # Normalize whitespace

        # Determine risk level based on term and definition
        combined = (term + " " + definition).lower()

        if any(kw in combined for kw in NEGATIVE_KEYWORDS):
            risk_level = RiskLevel.HIGH
        elif any(kw in combined for kw in POSITIVE_KEYWORDS):
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.MEDIUM

        # Signal concepts are detection-focused
        if 'signal' in term.lower():
            risk_level = RiskLevel.HIGH  # Detection signals are safety-critical

        concepts.append(SafetyConcept(
            term=term,
            source=kif_path.name,
            risk_level=risk_level,
            definition=definition,
            parent_concepts=[parent],
            # KIF concepts are generally treaty/harness relevant if safety-focused
            treaty_relevant='signal' in term.lower() or risk_level == RiskLevel.HIGH,
            harness_relevant=True,
        ))

    return concepts


def parse_meld_candidates(meld_path: Path) -> List[SafetyConcept]:
    """Extract concepts from an applied meld."""
    with open(meld_path) as f:
        meld = json.load(f)

    concepts = []

    for candidate in meld.get('candidates', []):
        term = candidate.get('term', '')
        if not term:
            continue

        safety_tags = candidate.get('safety_tags', {})
        simplex = candidate.get('simplex_mapping', {})
        hints = candidate.get('training_hints', {})

        # Parse risk level
        risk_str = safety_tags.get('risk_level', 'medium')
        try:
            risk_level = RiskLevel(risk_str)
        except ValueError:
            risk_level = RiskLevel.MEDIUM

        concepts.append(SafetyConcept(
            term=term,
            source=meld_path.stem,
            risk_level=risk_level,
            impacts=safety_tags.get('impacts', []),
            treaty_relevant=safety_tags.get('treaty_relevant', False),
            harness_relevant=safety_tags.get('harness_relevant', False),
            simplex_monitor=simplex.get('monitor') if simplex.get('status') == 'mapped' else None,
            definition=candidate.get('definition', ''),
            positive_examples=hints.get('positive_examples', []),
            negative_examples=hints.get('negative_examples', []),
            parent_concepts=candidate.get('parent_concepts', []),
        ))

    return concepts


def build_registry(
    kif_dir: Path = Path("data/concept_graph/custom_concepts"),
    meld_dir: Path = Path("melds/applied"),
) -> Dict[str, SafetyConcept]:
    """Build consolidated safety concept registry."""
    registry: Dict[str, SafetyConcept] = {}

    # Parse KIF files
    print("Parsing KIF files...")
    for kif_name in SAFETY_KIFS:
        kif_path = kif_dir / kif_name
        if kif_path.exists():
            concepts = parse_kif_concepts(kif_path)
            for c in concepts:
                # Prefer meld data if we already have it (more complete)
                if c.term not in registry:
                    registry[c.term] = c
            print(f"  {kif_name}: {len(concepts)} concepts")
        else:
            print(f"  {kif_name}: NOT FOUND")

    # Parse applied melds
    print("\nParsing applied melds...")
    for meld_path in sorted(meld_dir.glob("*.json")):
        # Skip remediation melds (too large, different structure)
        if 'remediation' in meld_path.name:
            continue

        try:
            concepts = parse_meld_candidates(meld_path)
            for c in concepts:
                # Meld data is authoritative - overwrite KIF data
                if c.term in registry:
                    # Merge: keep KIF source note but use meld data
                    old_source = registry[c.term].source
                    c.source = f"{old_source} + {c.source}"
                registry[c.term] = c

            # Only report if has safety-relevant concepts
            safety_relevant = [c for c in concepts if c.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
            if safety_relevant:
                print(f"  {meld_path.stem}: {len(concepts)} total, {len(safety_relevant)} high/critical")
        except Exception as e:
            print(f"  {meld_path.stem}: ERROR - {e}")

    return registry


def generate_pack_update(registry: Dict[str, SafetyConcept]) -> Dict:
    """Generate meld_policy update for pack.json."""

    # Group by simplex monitor
    by_monitor: Dict[str, List[str]] = defaultdict(list)
    for term, concept in registry.items():
        if concept.simplex_monitor:
            by_monitor[concept.simplex_monitor].append(term)

    # Identify critical concepts (high/critical risk)
    critical_concepts = [
        term for term, c in registry.items()
        if c.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    ]

    # Treaty-relevant concepts
    treaty_concepts = [
        term for term, c in registry.items()
        if c.treaty_relevant
    ]

    return {
        "critical_simplex_bindings": dict(by_monitor),
        "critical_concepts": sorted(critical_concepts),
        "treaty_relevant_concepts": sorted(treaty_concepts),
        "total_safety_concepts": len(registry),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build safety concept registry")
    parser.add_argument("--output", "-o", default="data/safety_registry.json",
                       help="Output registry file")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only print statistics")
    args = parser.parse_args()

    registry = build_registry()

    # Statistics
    print(f"\n=== Registry Statistics ===")
    print(f"Total concepts: {len(registry)}")

    by_risk = defaultdict(int)
    by_source_type = defaultdict(int)
    with_training = 0
    with_simplex = 0

    for term, c in registry.items():
        by_risk[c.risk_level.value] += 1
        if '.kif' in c.source:
            by_source_type['kif'] += 1
        else:
            by_source_type['meld'] += 1
        if c.positive_examples:
            with_training += 1
        if c.simplex_monitor:
            with_simplex += 1

    print(f"\nBy risk level:")
    for level in ['critical', 'high', 'medium', 'low', 'none']:
        print(f"  {level}: {by_risk[level]}")

    print(f"\nBy source:")
    print(f"  KIF only: {by_source_type['kif']}")
    print(f"  Meld (or both): {by_source_type['meld']}")

    print(f"\nMetadata coverage:")
    print(f"  With training examples: {with_training} ({100*with_training/len(registry):.1f}%)")
    print(f"  With simplex mapping: {with_simplex} ({100*with_simplex/len(registry):.1f}%)")

    if args.stats_only:
        return

    # Generate outputs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    registry_data = {
        term: {
            'term': c.term,
            'source': c.source,
            'risk_level': c.risk_level.value,
            'impacts': c.impacts,
            'treaty_relevant': c.treaty_relevant,
            'harness_relevant': c.harness_relevant,
            'simplex_monitor': c.simplex_monitor,
            'definition': c.definition[:200] + '...' if len(c.definition) > 200 else c.definition,
            'has_training': bool(c.positive_examples),
            'parent_concepts': c.parent_concepts,
        }
        for term, c in registry.items()
    }

    with open(output_path, 'w') as f:
        json.dump(registry_data, f, indent=2)
    print(f"\n✓ Saved registry to {output_path}")

    # Generate pack update suggestion
    pack_update = generate_pack_update(registry)
    update_path = output_path.with_suffix('.pack_update.json')
    with open(update_path, 'w') as f:
        json.dump(pack_update, f, indent=2)
    print(f"✓ Saved pack update suggestions to {update_path}")

    # Show critical concepts for server
    print(f"\n=== Critical concepts for server SAFETY_CONCEPTS ===")
    critical = [t.lower() for t, c in registry.items()
                if c.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
    print(f"Count: {len(critical)}")
    print("Sample:")
    for c in sorted(critical)[:20]:
        print(f"  '{c}',")


if __name__ == "__main__":
    main()
