#!/usr/bin/env python3
"""
Concept Pack Quality Checker

Scans a concept pack to identify concepts that may be difficult to train lenses for,
using TRUE POLYSEMY detection and child count (sibling density indicator) as proxies
for lens quality.

Key insight: High synset count does NOT indicate polysemy. Most synsets in our
concept pack are hyponyms (subtypes) providing taxonomic variety, not different
word senses.

TRUE POLYSEMY is detected by checking if synsets have DIVERGENT hypernym paths
(e.g., "bank" as riverbank vs financial institution have no common ancestor
except root).

Based on the Training Data Quality Analysis findings:
- True polysemy → divergent meanings → muddy activation signal → needs sense-splitting
- High child count → dense siblings → hard negative discrimination → needs sibling bucketing
- Taxonomic depth (hyponyms) → actually HELPFUL for training variety

Generates remediation recommendations as Meld operations (Split, Move, Merge, Deprecate).

Usage:
    python -m src.data.concept_pack_quality_checker \
        --concept-pack sumo-wordnet-v4 \
        --child-threshold 10 \
        --output-dir results/quality_check/
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import NLTK for polysemy detection
try:
    from nltk.corpus import wordnet as wn
    HAS_WORDNET = True
except ImportError:
    HAS_WORDNET = False
    print("Warning: NLTK wordnet not available, polysemy detection disabled")


@dataclass
class PolysemyAnalysis:
    """Results of polysemy analysis for a concept."""
    has_true_polysemy: bool
    distinct_sense_groups: int  # Number of semantically distinct groups
    sense_groups: List[List[str]]  # Groups of related synsets
    pos_distribution: Dict[str, int]  # Count by part of speech
    divergent_synsets: List[str]  # Synsets that don't share common ancestry


@dataclass
class QualityIssue:
    """A quality issue detected for a concept."""
    issue_type: str  # 'true_polysemy' | 'dense_siblings' | 'combined' | 'taxonomic_depth'
    severity: str  # 'low' | 'medium' | 'high' | 'critical'
    metric_name: str
    metric_value: int
    threshold: int
    description: str
    recommended_action: str  # 'split' | 'bucket' | 'review' | 'none'
    details: Optional[Dict] = None  # Additional details for the issue


@dataclass
class ConceptQualityProfile:
    """Quality profile for a single concept."""
    sumo_term: str
    layer: int
    domain: str
    synset_count: int
    child_count: int
    sibling_count: int  # Number of siblings (children of same parent)
    parent_concepts: List[str]
    quadrant: str  # A/B/C/D from quality analysis
    estimated_f1_ceiling: float
    issues: List[QualityIssue]
    polysemy_analysis: Optional[PolysemyAnalysis] = None
    lens_grade: Optional[str] = None  # From trained lens pack if available
    lens_f1: Optional[float] = None


@dataclass
class RemediationMeld:
    """A recommended meld operation for remediation."""
    operation: str  # 'split' | 'merge' | 'move' | 'deprecate'
    target_concept: str
    priority: str  # 'low' | 'medium' | 'high' | 'critical'
    rationale: str
    details: Dict  # Operation-specific details


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check concept pack quality and generate remediation recommendations"
    )
    parser.add_argument('--concept-pack', required=True,
                        help='Concept pack ID (e.g., sumo-wordnet-v4)')
    parser.add_argument('--lens-pack', type=str, default=None,
                        help='Optional lens pack to include trained lens grades')
    parser.add_argument('--child-threshold', type=int, default=10,
                        help='Child count threshold for sibling bucketing warning (default: 10)')
    parser.add_argument('--polysemy-depth', type=int, default=3,
                        help='Max depth to check for common ancestors in polysemy detection (default: 3)')
    parser.add_argument('--layers', type=str, default='0,1,2,3,4',
                        help='Comma-separated layers to check (default: 0,1,2,3,4)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--generate-melds', action='store_true',
                        help='Generate meld request files for remediation')
    parser.add_argument('--format', choices=['json', 'csv', 'both'], default='both',
                        help='Output format (default: both)')
    parser.add_argument('--skip-polysemy', action='store_true',
                        help='Skip polysemy analysis (faster but less accurate)')
    return parser.parse_args()


# =============================================================================
# POLYSEMY DETECTION
# =============================================================================

def resolve_synset(synset_id: str):
    """Resolve a synset ID to a WordNet synset object.

    Handles both formats:
    - Standard: 'anchor.n.01'
    - Offset: '02131653.n'
    """
    if not HAS_WORDNET:
        return None

    try:
        # Try standard format first
        if not synset_id[0].isdigit():
            return wn.synset(synset_id)

        # Offset format: '02131653.n'
        parts = synset_id.split('.')
        if len(parts) >= 2:
            offset = int(parts[0])
            pos = parts[1][0]  # First char of pos (n, v, a, r, s)
            return wn.synset_from_pos_and_offset(pos, offset)
    except Exception:
        pass

    return None


def get_synset_label(synset_id: str) -> str:
    """Get a human-readable label for a synset ID.

    Returns format: "lemma (definition)" or just the ID if resolution fails.
    """
    if not HAS_WORDNET:
        return synset_id

    syn = resolve_synset(synset_id)
    if not syn:
        return synset_id

    # Get the primary lemma
    lemmas = [l.name().replace('_', ' ') for l in syn.lemmas()]
    primary_lemma = lemmas[0] if lemmas else synset_id

    # Get a truncated definition
    definition = syn.definition()
    if definition and len(definition) > 60:
        definition = definition[:57] + "..."

    if definition:
        return f"{primary_lemma} ({definition})"
    return primary_lemma


def get_synset_info(synset_id: str) -> Dict:
    """Get detailed info for a synset for human review.

    Returns dict with id, lemma, all_lemmas, definition, pos.
    """
    info = {"id": synset_id}

    if not HAS_WORDNET:
        return info

    syn = resolve_synset(synset_id)
    if not syn:
        return info

    lemmas = [l.name().replace('_', ' ') for l in syn.lemmas()]
    info["lemma"] = lemmas[0] if lemmas else None
    info["all_lemmas"] = lemmas
    info["definition"] = syn.definition()
    info["pos"] = syn.pos()

    return info


def get_hypernym_roots(synset, max_depth: int = 5) -> Set[str]:
    """Get the root hypernyms of a synset up to max_depth.

    Returns set of hypernym names at or near the root of the hierarchy.
    """
    if not synset:
        return set()

    roots = set()
    visited = set()

    def walk_up(syn, depth):
        if depth > max_depth or syn.name() in visited:
            return
        visited.add(syn.name())

        hypernyms = syn.hypernyms()
        if not hypernyms:
            # This is a root
            roots.add(syn.name())
        elif depth == max_depth:
            # Reached max depth, treat these as "effective roots"
            for h in hypernyms:
                roots.add(h.name())
        else:
            for h in hypernyms:
                walk_up(h, depth + 1)

    walk_up(synset, 0)
    return roots


def synsets_share_ancestry(syn1, syn2, max_depth: int = 5) -> bool:
    """Check if two synsets share a common ancestor within max_depth.

    Returns True if they're taxonomically related (one is hyponym of other,
    or they share a common hypernym).

    Uses multiple heuristics:
    1. Direct ancestry within max_depth
    2. Path similarity > 0.1 (indicates shared taxonomy)
    3. Same part of speech (nouns/verbs in same domain usually related)
    """
    if not syn1 or not syn2:
        return False

    # Quick check: same synset
    if syn1.name() == syn2.name():
        return True

    # Check path similarity - if > 0.1, they're taxonomically close
    try:
        sim = syn1.path_similarity(syn2)
        if sim and sim > 0.1:
            return True
    except Exception:
        pass

    # Check for lowest common hypernym that isn't just entity/thing
    try:
        lchs = syn1.lowest_common_hypernyms(syn2)
        if lchs:
            for lch in lchs:
                # If LCH is more specific than entity/physical_entity, they're related
                if lch.name() not in ('entity.n.01', 'physical_entity.n.01', 'thing.n.12', 'object.n.01'):
                    return True
    except Exception:
        pass

    # Fallback to depth-based check
    roots1 = get_hypernym_roots(syn1, max_depth)
    roots2 = get_hypernym_roots(syn2, max_depth)

    # Check for overlap
    return bool(roots1 & roots2)


def analyze_polysemy(synset_ids: List[str], max_depth: int = 3) -> PolysemyAnalysis:
    """Analyze synsets for true polysemy vs taxonomic depth.

    True polysemy: synsets have divergent hypernym paths (different conceptual domains)
    Taxonomic depth: synsets are hyponyms sharing common ancestry (same conceptual domain)

    Returns PolysemyAnalysis with details about sense groupings.
    """
    if not HAS_WORDNET or not synset_ids:
        return PolysemyAnalysis(
            has_true_polysemy=False,
            distinct_sense_groups=1,
            sense_groups=[synset_ids],
            pos_distribution={},
            divergent_synsets=[]
        )

    # Resolve all synsets
    synsets = []
    for sid in synset_ids:
        syn = resolve_synset(sid)
        if syn:
            synsets.append((sid, syn))

    if len(synsets) <= 1:
        return PolysemyAnalysis(
            has_true_polysemy=False,
            distinct_sense_groups=1,
            sense_groups=[synset_ids],
            pos_distribution={'n': len(synset_ids)} if synset_ids else {},
            divergent_synsets=[]
        )

    # Count by part of speech
    pos_dist = defaultdict(int)
    for sid, syn in synsets:
        pos_dist[syn.pos()] += 1

    # Group synsets by shared ancestry using union-find
    # Start with each synset in its own group
    groups = {sid: {sid} for sid, _ in synsets}
    parent = {sid: sid for sid, _ in synsets}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            groups[py] |= groups[px]
            del groups[px]

    # Merge synsets that share ancestry
    synset_list = [(sid, syn) for sid, syn in synsets]
    for i, (sid1, syn1) in enumerate(synset_list):
        for sid2, syn2 in synset_list[i+1:]:
            if synsets_share_ancestry(syn1, syn2, max_depth):
                union(sid1, sid2)

    # Collect final groups
    final_groups = []
    seen_roots = set()
    for sid, _ in synsets:
        root = find(sid)
        if root not in seen_roots:
            seen_roots.add(root)
            final_groups.append(sorted(groups[root]))

    # Find divergent synsets (synsets in singleton groups or different groups)
    divergent = []
    if len(final_groups) > 1:
        for group in final_groups:
            if len(group) == 1:
                divergent.extend(group)
            else:
                # Mark the first synset of each non-singleton group as representative
                divergent.append(group[0])

    has_polysemy = len(final_groups) > 1

    # Also check for cross-POS polysemy (e.g., "iron" noun vs verb)
    if len(pos_dist) > 1 and 'v' in pos_dist and 'n' in pos_dist:
        # Verbs and nouns are often distinct senses
        has_polysemy = True

    return PolysemyAnalysis(
        has_true_polysemy=has_polysemy,
        distinct_sense_groups=len(final_groups),
        sense_groups=final_groups,
        pos_distribution=dict(pos_dist),
        divergent_synsets=divergent
    )


def load_concept_pack(pack_id: str) -> Tuple[Dict, Dict[int, List[Dict]]]:
    """Load concept pack manifest and layer files."""
    pack_dir = PROJECT_ROOT / "concept_packs" / pack_id

    # Load manifest
    manifest_path = pack_dir / "pack.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load layer files
    layers = {}
    hierarchy_dir = pack_dir / "hierarchy"
    for layer_file in hierarchy_dir.glob("layer*.json"):
        layer_num = int(layer_file.stem.replace("layer", ""))
        with open(layer_file) as f:
            data = json.load(f)
            layers[layer_num] = data.get("concepts", [])

    return manifest, layers


def load_lens_grades(lens_pack_path: str) -> Dict[str, Tuple[str, float]]:
    """Load lens grades from a trained lens pack."""
    grades = {}
    lens_dir = Path(lens_pack_path)

    # Look for training results JSON files
    for results_file in lens_dir.glob("**/training_results*.json"):
        with open(results_file) as f:
            results = json.load(f)
            for concept, data in results.items():
                if isinstance(data, dict):
                    grade = data.get("grade", "unknown")
                    f1 = data.get("test_f1", data.get("f1", 0.0))
                    grades[concept] = (grade, f1)

    # Also check individual lens files
    for lens_file in lens_dir.glob("**/*.pt"):
        concept = lens_file.stem
        if concept not in grades:
            # Try to load metadata
            meta_file = lens_file.with_suffix(".json")
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                    grade = meta.get("grade", "unknown")
                    f1 = meta.get("test_f1", 0.0)
                    grades[concept] = (grade, f1)

    return grades


def calculate_sibling_count(concept: Dict, layer_concepts: List[Dict]) -> int:
    """Calculate how many siblings a concept has (children of same parent)."""
    parent_concepts = concept.get("parent_concepts", [])
    if not parent_concepts:
        return 0

    # Find all concepts that share at least one parent
    siblings = set()
    for other in layer_concepts:
        if other["sumo_term"] == concept["sumo_term"]:
            continue
        other_parents = other.get("parent_concepts", [])
        if any(p in parent_concepts for p in other_parents):
            siblings.add(other["sumo_term"])

    return len(siblings)


def classify_quadrant(synset_count: int, sibling_count: int,
                      synset_threshold: int, sibling_threshold: int) -> str:
    """Classify concept into A/B/C/D quadrant based on synset and sibling counts."""
    low_synsets = synset_count < synset_threshold
    low_siblings = sibling_count < sibling_threshold

    if low_synsets and low_siblings:
        return "A"  # Low/Low - simple, isolated
    elif low_synsets and not low_siblings:
        return "B"  # Low/High - simple but crowded
    elif not low_synsets and not low_siblings:
        return "C"  # High/High - complex and crowded
    else:
        return "D"  # High/Low - complex but isolated


def estimate_f1_ceiling(quadrant: str) -> float:
    """Estimate achievable F1 based on quadrant from quality analysis findings."""
    # Based on Training Data Quality Analysis results
    ceilings = {
        "A": 0.88,  # High output quality, activation ~90%
        "B": 0.83,  # OK output, moderate activation quality
        "C": 0.76,  # Worst - polysemy + dense siblings compound
        "D": 0.85,  # Good despite polysemy, fewer sibling confounders
    }
    return ceilings.get(quadrant, 0.80)


def analyze_concept(concept: Dict, layer_concepts: List[Dict],
                    child_threshold: int,
                    polysemy_depth: int = 3,
                    skip_polysemy: bool = False,
                    lens_grades: Optional[Dict] = None) -> ConceptQualityProfile:
    """Analyze a single concept for quality issues.

    Uses TRUE POLYSEMY detection (divergent hypernym paths) instead of raw synset count.
    """
    sumo_term = concept["sumo_term"]
    synset_count = concept.get("synset_count", 0)
    child_count = concept.get("child_count", 0)
    synsets = concept.get("synsets", [])

    # Calculate sibling count
    sibling_count = calculate_sibling_count(concept, layer_concepts)

    # Analyze polysemy (if not skipped)
    polysemy_analysis = None
    has_true_polysemy = False
    distinct_senses = 1

    if not skip_polysemy and synsets:
        polysemy_analysis = analyze_polysemy(synsets, polysemy_depth)
        has_true_polysemy = polysemy_analysis.has_true_polysemy
        distinct_senses = polysemy_analysis.distinct_sense_groups

    # Classify quadrant based on TRUE polysemy (not raw synset count) and sibling density
    # Use distinct_senses > 1 as the polysemy indicator
    quadrant = classify_quadrant(
        distinct_senses,  # Use sense groups instead of raw synset count
        sibling_count,
        2,  # Threshold: 2+ distinct sense groups = polysemy
        child_threshold
    )

    # Estimate F1 ceiling
    f1_ceiling = estimate_f1_ceiling(quadrant)

    # Detect issues
    issues = []

    # Check for TRUE polysemy (multiple distinct sense groups)
    if has_true_polysemy and distinct_senses >= 2:
        severity = "high" if distinct_senses >= 3 else "medium"
        issues.append(QualityIssue(
            issue_type="true_polysemy",
            severity=severity,
            metric_name="distinct_sense_groups",
            metric_value=distinct_senses,
            threshold=2,
            description=f"TRUE POLYSEMY: {distinct_senses} semantically distinct sense groups detected. "
                        f"These have divergent hypernym paths and may muddy activation signal.",
            recommended_action="split",
            details={
                "sense_groups": polysemy_analysis.sense_groups if polysemy_analysis else [],
                "pos_distribution": polysemy_analysis.pos_distribution if polysemy_analysis else {},
                "divergent_synsets": polysemy_analysis.divergent_synsets if polysemy_analysis else []
            }
        ))

    # Note: High synset count WITHOUT polysemy is actually good (taxonomic variety)
    # We can add an informational note but it's not an issue
    if synset_count >= 10 and not has_true_polysemy:
        # This is actually beneficial - add as info, not issue
        pass  # Could add a "taxonomic_richness" positive note if desired

    # Check for dense siblings (parent has many children)
    if child_count >= child_threshold:
        severity = "high" if child_count >= child_threshold * 3 else "medium"
        issues.append(QualityIssue(
            issue_type="dense_siblings",
            severity=severity,
            metric_name="child_count",
            metric_value=child_count,
            threshold=child_threshold,
            description=f"High child count ({child_count}) means children will have many hard negatives - consider semantic bucketing",
            recommended_action="bucket"
        ))

    # Check for combined issues (TRUE polysemy + dense siblings)
    if has_true_polysemy and sibling_count >= child_threshold:
        issues.append(QualityIssue(
            issue_type="combined",
            severity="critical",
            metric_name="polysemy_plus_siblings",
            metric_value=distinct_senses + sibling_count,
            threshold=2 + child_threshold,
            description=f"TRUE POLYSEMY ({distinct_senses} senses) AND dense siblings ({sibling_count}) - compound difficulty",
            recommended_action="review"
        ))

    # Add lens grade if available
    lens_grade = None
    lens_f1 = None
    if lens_grades and sumo_term in lens_grades:
        lens_grade, lens_f1 = lens_grades[sumo_term]

    return ConceptQualityProfile(
        sumo_term=sumo_term,
        layer=concept["layer"],
        domain=concept.get("domain", "Unknown"),
        synset_count=synset_count,
        child_count=child_count,
        sibling_count=sibling_count,
        parent_concepts=concept.get("parent_concepts", []),
        quadrant=quadrant,
        estimated_f1_ceiling=f1_ceiling,
        issues=issues,
        polysemy_analysis=polysemy_analysis,
        lens_grade=lens_grade,
        lens_f1=lens_f1
    )


def generate_split_meld(profile: ConceptQualityProfile, concept_data: Dict) -> Optional[RemediationMeld]:
    """Generate a ConceptSplit meld operation for truly polysemous concepts.

    Only generates splits for concepts with TRUE polysemy (divergent sense groups),
    not for concepts with high synset count due to taxonomic depth.
    """
    if not profile.polysemy_analysis or not profile.polysemy_analysis.has_true_polysemy:
        return None

    analysis = profile.polysemy_analysis

    # Generate splits based on actual sense groups, not raw synsets
    split_into = []
    for i, sense_group in enumerate(analysis.sense_groups, 1):
        # Get representative synset for this group
        representative_synset = sense_group[0] if sense_group else None

        # Try to get a meaningful name from the synset
        sense_name = f"Sense{i}"
        if representative_synset and HAS_WORDNET:
            syn = resolve_synset(representative_synset)
            if syn:
                # Use the first lemma name as hint
                lemmas = [l.name() for l in syn.lemmas()]
                if lemmas:
                    sense_name = lemmas[0].replace('_', ' ').title()

        # Build synset info with human-readable labels for review
        synsets_with_labels = []
        for synset_id in sense_group:
            synset_info = get_synset_info(synset_id)
            synsets_with_labels.append(synset_info)

        split_into.append({
            "term": f"{profile.sumo_term}_{sense_name}",
            "synsets": sense_group,  # Keep raw IDs for machine processing
            "synsets_detail": synsets_with_labels,  # Human-readable details
            "representative_synset": representative_synset,
            "representative_label": get_synset_label(representative_synset) if representative_synset else None,
            "note": f"Sense group {i} with {len(sense_group)} related synsets"
        })

    # Build sense_groups with labels for the evidence section
    sense_groups_labeled = []
    for sense_group in analysis.sense_groups:
        group_with_labels = [
            {"id": sid, "label": get_synset_label(sid)}
            for sid in sense_group
        ]
        sense_groups_labeled.append(group_with_labels)

    n_senses = analysis.distinct_sense_groups
    priority = "high" if n_senses >= 3 else "medium"

    return RemediationMeld(
        operation="split",
        target_concept=profile.sumo_term,
        priority=priority,
        rationale=f"TRUE POLYSEMY: Concept has {n_senses} semantically distinct sense groups "
                  f"(from {profile.synset_count} total synsets). Splitting into sense-specific "
                  f"lenses will prevent activation mudding from divergent meanings.",
        details={
            "source_concept": profile.sumo_term,
            "source_disposition": "keep_as_parent",  # Original becomes parent of senses
            "split_into": split_into,
            "split_rationale": {
                "reason": "TRUE polysemy detected - divergent hypernym paths",
                "evidence": {
                    "distinct_sense_groups": n_senses,
                    "sense_groups": analysis.sense_groups,  # Raw IDs
                    "sense_groups_labeled": sense_groups_labeled,  # Human-readable
                    "pos_distribution": analysis.pos_distribution,
                    "total_synsets": profile.synset_count
                }
            }
        }
    )


def generate_bucket_meld(parent_profile: ConceptQualityProfile,
                         children: List[Dict]) -> List[RemediationMeld]:
    """Generate bucket suggestions for a parent with too many children."""
    melds = []

    # Group children by domain or semantic similarity
    # For now, suggest domain-based bucketing
    domain_groups = defaultdict(list)
    for child in children:
        # Use lexname or domain as grouping hint
        lexname = child.get("lexname", child.get("domain", "other"))
        domain_groups[lexname].append(child["sumo_term"])

    # If we have meaningful groups, suggest intermediate category nodes
    if len(domain_groups) > 1:
        bucket_suggestions = []
        for group_name, members in domain_groups.items():
            if len(members) >= 3:  # Only bucket if enough members
                bucket_name = f"{parent_profile.sumo_term}_{group_name.replace('.', '_').title()}"
                bucket_suggestions.append({
                    "bucket_name": bucket_name,
                    "members": members,
                    "semantic_basis": group_name
                })

        if bucket_suggestions:
            melds.append(RemediationMeld(
                operation="split",  # Using split to create intermediate nodes
                target_concept=parent_profile.sumo_term,
                priority="medium",
                rationale=f"Concept has {parent_profile.child_count} children. "
                          f"Introducing intermediate category nodes will reduce sibling density "
                          f"and improve lens discrimination.",
                details={
                    "source_concept": parent_profile.sumo_term,
                    "source_disposition": "keep_as_parent",
                    "bucket_suggestions": bucket_suggestions,
                    "bucket_rationale": {
                        "reason": "High child count creates dense sibling groups",
                        "evidence": {
                            "child_count": parent_profile.child_count,
                            "domain_distribution": {k: len(v) for k, v in domain_groups.items()}
                        }
                    }
                }
            ))

    return melds


def generate_remediation_melds(profiles: List[ConceptQualityProfile],
                               layers: Dict[int, List[Dict]]) -> List[RemediationMeld]:
    """Generate meld operations for remediation.

    Only generates splits for TRUE polysemy (divergent sense groups),
    not for high synset count from taxonomic depth.
    """
    melds = []

    # Build concept lookup
    concept_lookup = {}
    for layer_num, concepts in layers.items():
        for c in concepts:
            concept_lookup[c["sumo_term"]] = c

    # Process profiles with issues
    processed_parents = set()
    processed_polysemy = set()

    for profile in profiles:
        if not profile.issues:
            continue

        concept_data = concept_lookup.get(profile.sumo_term, {})

        for issue in profile.issues:
            # Only generate split meld for TRUE polysemy
            if issue.recommended_action == "split" and issue.issue_type == "true_polysemy":
                if profile.sumo_term not in processed_polysemy:
                    meld = generate_split_meld(profile, concept_data)
                    if meld:  # Only add if we actually generated a meld
                        melds.append(meld)
                        processed_polysemy.add(profile.sumo_term)

            elif issue.recommended_action == "bucket" and profile.sumo_term not in processed_parents:
                # Find children and generate bucketing suggestions
                children = [c for c in concept_lookup.values()
                           if profile.sumo_term in c.get("parent_concepts", [])]

                if children:
                    bucket_melds = generate_bucket_meld(profile, children)
                    melds.extend(bucket_melds)
                    processed_parents.add(profile.sumo_term)

    return melds


def write_meld_request(melds: List[RemediationMeld], output_dir: Path, pack_id: str):
    """Write meld operations as a meld request JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    meld_request = {
        "meld_request_id": f"org.hatcat/quality-remediation-{timestamp}@0.1.0",
        "target_pack_spec_id": f"org.hatcat/{pack_id}@4.1.0",
        "metadata": {
            "name": "Quality-Based Remediation",
            "description": "Auto-generated remediation suggestions based on quality analysis",
            "source": "quality_checker",
            "author": "hatcat-quality-checker",
            "created": datetime.now().isoformat() + "Z"
        },
        "structural_operations": [
            {
                "operation": m.operation,
                "target_concept": m.target_concept,
                "priority": m.priority,
                "rationale": m.rationale,
                **m.details
            }
            for m in melds
        ],
        "candidates": [],
        "attachment_points": []
    }

    output_path = output_dir / f"remediation_meld_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(meld_request, f, indent=2)

    return output_path


def generate_summary_report(profiles: List[ConceptQualityProfile],
                            melds: List[RemediationMeld],
                            output_dir: Path) -> str:
    """Generate a human-readable summary report."""
    lines = [
        "# Concept Pack Quality Report",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        "",
        "## Executive Summary",
        "",
    ]

    # Count by quadrant
    quadrant_counts = defaultdict(int)
    for p in profiles:
        quadrant_counts[p.quadrant] += 1

    lines.append("### Quadrant Distribution")
    lines.append("")
    lines.append("| Quadrant | Description | Count | Est. F1 Ceiling |")
    lines.append("|----------|-------------|-------|-----------------|")
    lines.append(f"| A | Low synsets, Low siblings | {quadrant_counts['A']} | 0.88 |")
    lines.append(f"| B | Low synsets, High siblings | {quadrant_counts['B']} | 0.83 |")
    lines.append(f"| C | High synsets, High siblings | {quadrant_counts['C']} | 0.76 |")
    lines.append(f"| D | High synsets, Low siblings | {quadrant_counts['D']} | 0.85 |")
    lines.append("")

    # Count by issue type
    issue_counts = defaultdict(int)
    severity_counts = defaultdict(int)
    for p in profiles:
        for issue in p.issues:
            issue_counts[issue.issue_type] += 1
            severity_counts[issue.severity] += 1

    lines.append("### Issues Detected")
    lines.append("")
    lines.append("| Issue Type | Count | Description |")
    lines.append("|------------|-------|-------------|")
    lines.append(f"| true_polysemy | {issue_counts['true_polysemy']} | TRUE polysemy - divergent sense groups |")
    lines.append(f"| dense_siblings | {issue_counts['dense_siblings']} | High child count - crowded negative space |")
    lines.append(f"| combined | {issue_counts['combined']} | TRUE polysemy + dense siblings - critical |")
    lines.append("")
    lines.append("**Note**: High synset count alone is NOT flagged as an issue. Most high-synset concepts")
    lines.append("have taxonomic depth (hyponyms), which provides training variety. Only concepts with")
    lines.append("TRUE polysemy (divergent hypernym paths indicating distinct meanings) are flagged.")
    lines.append("")

    lines.append("### Severity Distribution")
    lines.append("")
    lines.append("| Severity | Count |")
    lines.append("|----------|-------|")
    for sev in ['critical', 'high', 'medium', 'low']:
        lines.append(f"| {sev} | {severity_counts[sev]} |")
    lines.append("")

    # Top offenders
    lines.append("## Top Concepts Needing Attention")
    lines.append("")

    # Sort by issue count and severity
    def sort_key(p):
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        max_severity = min(severity_order.get(i.severity, 4) for i in p.issues) if p.issues else 5
        return (max_severity, -len(p.issues), -p.synset_count, -p.child_count)

    sorted_profiles = sorted([p for p in profiles if p.issues], key=sort_key)[:20]

    lines.append("| Concept | Layer | Quadrant | Synsets | Children | Issues | Est. F1 |")
    lines.append("|---------|-------|----------|---------|----------|--------|---------|")
    for p in sorted_profiles:
        issue_types = ", ".join(set(i.issue_type for i in p.issues))
        lines.append(f"| {p.sumo_term} | {p.layer} | {p.quadrant} | {p.synset_count} | {p.child_count} | {issue_types} | {p.estimated_f1_ceiling:.2f} |")
    lines.append("")

    # Remediation summary
    lines.append("## Recommended Remediations")
    lines.append("")
    lines.append(f"Total meld operations generated: {len(melds)}")
    lines.append("")

    op_counts = defaultdict(int)
    for m in melds:
        op_counts[m.operation] += 1

    lines.append("| Operation | Count |")
    lines.append("|-----------|-------|")
    for op, count in op_counts.items():
        lines.append(f"| {op} | {count} |")
    lines.append("")

    # High priority melds
    high_priority = [m for m in melds if m.priority in ['critical', 'high']]
    if high_priority:
        lines.append("### High Priority Actions")
        lines.append("")
        for m in high_priority[:10]:
            lines.append(f"- **{m.operation.upper()}** `{m.target_concept}`: {m.rationale[:100]}...")
        lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()

    # Parse layers
    target_layers = [int(l.strip()) for l in args.layers.split(",")]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load concept pack
    print(f"Loading concept pack: {args.concept_pack}")
    manifest, layers = load_concept_pack(args.concept_pack)

    # Load lens grades if provided
    lens_grades = None
    if args.lens_pack:
        print(f"Loading lens grades from: {args.lens_pack}")
        lens_grades = load_lens_grades(args.lens_pack)
        print(f"  Loaded grades for {len(lens_grades)} lenses")

    # Analyze concepts
    polysemy_mode = "ENABLED" if HAS_WORDNET and not args.skip_polysemy else "DISABLED"
    print(f"Analyzing concepts (layers {target_layers})...")
    print(f"  Polysemy detection: {polysemy_mode}")
    print(f"  Child threshold: {args.child_threshold}")
    if HAS_WORDNET and not args.skip_polysemy:
        print(f"  Polysemy depth: {args.polysemy_depth}")

    all_profiles = []

    for layer_num in target_layers:
        if layer_num not in layers:
            print(f"  Warning: Layer {layer_num} not found")
            continue

        layer_concepts = layers[layer_num]
        print(f"  Layer {layer_num}: {len(layer_concepts)} concepts", end="", flush=True)

        for i, concept in enumerate(layer_concepts):
            if (i + 1) % 100 == 0:
                print(".", end="", flush=True)

            profile = analyze_concept(
                concept,
                layer_concepts,
                args.child_threshold,
                polysemy_depth=args.polysemy_depth,
                skip_polysemy=args.skip_polysemy,
                lens_grades=lens_grades
            )
            all_profiles.append(profile)
        print()  # Newline after layer

    # Count issues
    concepts_with_issues = [p for p in all_profiles if p.issues]
    print(f"\nFound {len(concepts_with_issues)} concepts with quality issues out of {len(all_profiles)} total")

    # Generate remediation melds
    melds = []
    if args.generate_melds:
        print("Generating remediation meld operations...")
        melds = generate_remediation_melds(all_profiles, layers)
        print(f"  Generated {len(melds)} meld operations")

        # Write meld request
        meld_path = write_meld_request(melds, output_dir, args.concept_pack)
        print(f"  Meld request written to: {meld_path}")

    # Write outputs
    if args.format in ['json', 'both']:
        # Full profiles JSON
        profiles_path = output_dir / "quality_profiles.json"
        with open(profiles_path, 'w') as f:
            json.dump([asdict(p) for p in all_profiles], f, indent=2)
        print(f"Wrote profiles to: {profiles_path}")

        # Issues only JSON
        issues_path = output_dir / "quality_issues.json"
        with open(issues_path, 'w') as f:
            issues_only = [asdict(p) for p in concepts_with_issues]
            json.dump(issues_only, f, indent=2)
        print(f"Wrote issues to: {issues_path}")

    if args.format in ['csv', 'both']:
        # CSV summary
        import csv
        csv_path = output_dir / "quality_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "concept", "layer", "domain", "quadrant",
                "synset_count", "child_count", "sibling_count",
                "est_f1_ceiling", "issue_count", "issues",
                "lens_grade", "lens_f1"
            ])
            for p in all_profiles:
                issue_str = "; ".join(f"{i.issue_type}({i.severity})" for i in p.issues)
                writer.writerow([
                    p.sumo_term, p.layer, p.domain, p.quadrant,
                    p.synset_count, p.child_count, p.sibling_count,
                    p.estimated_f1_ceiling, len(p.issues), issue_str,
                    p.lens_grade or "", p.lens_f1 or ""
                ])
        print(f"Wrote CSV to: {csv_path}")

    # Generate summary report
    report = generate_summary_report(all_profiles, melds, output_dir)
    report_path = output_dir / "quality_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Wrote report to: {report_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    quadrant_counts = defaultdict(int)
    for p in all_profiles:
        quadrant_counts[p.quadrant] += 1

    print(f"\nQuadrant distribution:")
    print(f"  A (easy):     {quadrant_counts['A']:4d} concepts (est F1: 0.88)")
    print(f"  B (crowded):  {quadrant_counts['B']:4d} concepts (est F1: 0.83)")
    print(f"  C (hardest):  {quadrant_counts['C']:4d} concepts (est F1: 0.76)")
    print(f"  D (complex):  {quadrant_counts['D']:4d} concepts (est F1: 0.85)")

    issue_counts = defaultdict(int)
    for p in all_profiles:
        for issue in p.issues:
            issue_counts[issue.issue_type] += 1

    print(f"\nIssues detected:")
    print(f"  TRUE polysemy (needs splitting):   {issue_counts['true_polysemy']:4d}")
    print(f"  Dense siblings (needs bucketing):  {issue_counts['dense_siblings']:4d}")
    print(f"  Combined (polysemy + siblings):    {issue_counts['combined']:4d}")

    # Count concepts with high synset count but NO polysemy (taxonomic depth - good!)
    taxonomic_depth = sum(1 for p in all_profiles
                          if p.synset_count >= 10 and
                          (not p.polysemy_analysis or not p.polysemy_analysis.has_true_polysemy))
    if taxonomic_depth > 0:
        print(f"\n  Note: {taxonomic_depth} concepts have high synset count (>=10) but NO true polysemy")
        print(f"        These benefit from taxonomic variety (hyponyms), not a quality issue.")

    if melds:
        print(f"\nRemediation melds generated: {len(melds)}")


if __name__ == "__main__":
    main()
