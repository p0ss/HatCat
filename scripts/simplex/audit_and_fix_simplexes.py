#!/usr/bin/env python3
"""
Audit and fix S-tier simplexes in layer2.json

This script:
1. Identifies problematic simplexes (culturally-specific, wrong polarity, etc.)
2. Removes them from layer2.json
3. Creates proper SUMO concept definitions for missing/replacement poles
4. Produces a clean, defensible S-tier simplex set
"""

import json
from pathlib import Path
from nltk.corpus import wordnet as wn

PROJECT_ROOT = Path(__file__).parent.parent
LAYER2_PATH = PROJECT_ROOT / "data" / "concept_graph" / "abstraction_layers" / "layer2.json"
SUMO_CONCEPTS_PATH = PROJECT_ROOT / "data" / "concept_graph" / "custom_sumo_concepts.json"

# Problematic simplexes to remove
REMOVE_SIMPLEXES = [
    "aspiration/social_mobility",  # american_dream - culturally specific
    "social_self_regard",  # immodesty.n.02 = genital exhibitionism, wrong concept
]

# Simplexes that need pole replacements
FIX_SIMPLEXES = {
    "motivational_regulation": {
        "issue": "indifference.n.01 as positive pole is wrong (that's apathy)",
        "fix": {
            "positive_pole": {
                "concept": "HealthyMotivation",
                "synset": "motivation.n.01",
                "reasoning": "Healthy motivation without compulsion or apathy"
            }
        }
    },
    "affective_coherence": {
        "issue": "indifference.n.01 as positive pole is wrong",
        "fix": {
            "positive_pole": {
                "concept": "EmotionalClarity",
                "synset": "certainty.n.02",
                "reasoning": "Clear, coherent emotional state without ambivalence or indifference"
            }
        }
    },
    "threat_perception": {
        "issue": "obliviousness as positive pole - being unaware of threats is not healthy",
        "fix": {
            "positive_pole": {
                "concept": "SafetyPerception",
                "synset": "security.n.01",
                "reasoning": "Feeling of safety and security, appropriate non-alarm state"
            }
        }
    }
}

# New SUMO concepts to create for the 3 incomplete simplexes
NEW_SUMO_CONCEPTS = {
    "AlexithymiaState": {
        "definition": "The inability to identify and describe emotions in oneself; emotional blindness or unawareness",
        "pos": "n",
        "lexname": "noun.feeling",
        "related_synsets": ["affect.n.01"],
        "clinical": True
    },
    "EmotionalFlooding": {
        "definition": "An excessive, overwhelming state where affective experience becomes so intense it impairs functioning",
        "pos": "n",
        "lexname": "noun.feeling",
        "related_synsets": ["affect.n.01"],
        "clinical": True
    },
    "ContentedSensuality": {
        "definition": "Balanced, healthy physical pleasure and sensory enjoyment without excess",
        "pos": "n",
        "lexname": "noun.feeling",
        "related_synsets": ["pleasure.n.01"]
    },
    "Enmeshment": {
        "definition": "Excessive emotional or psychological dependence in relationships; loss of individual boundaries",
        "pos": "n",
        "lexname": "noun.feeling",
        "related_synsets": ["dependence.n.01"],
        "clinical": True
    },
    "ClassInclusion": {
        "definition": "Active support for social and economic mobility of others across class boundaries",
        "pos": "n",
        "lexname": "noun.motive",
        "related_synsets": ["solidarity.n.01"]
    },
    "ClassEntrenchment": {
        "definition": "Active maintenance of rigid socioeconomic class boundaries; resistance to mobility",
        "pos": "n",
        "lexname": "noun.motive",
        "related_synsets": ["exclusion.n.01"]
    }
}


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data, indent=2):
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def check_synset_valid(synset_id: str) -> tuple:
    """Check if synset exists and get its definition"""
    try:
        ss = wn.synset(synset_id)
        return True, ss.definition()
    except:
        return False, None


def audit_simplexes():
    """Audit and report on all simplexes"""

    print("=" * 80)
    print("SIMPLEX AUDIT")
    print("=" * 80)

    layer2 = load_json(LAYER2_PATH)
    concepts = layer2['concepts']

    # Find all simplex entries
    simplexes = [c for c in concepts if c.get('simplex_dimension')]

    print(f"\nFound {len(simplexes)} simplex entries in layer2.json")
    print()

    # Categorize issues
    to_remove = []
    to_fix = []
    valid = []

    for simplex in simplexes:
        dim = simplex['simplex_dimension']

        if dim in REMOVE_SIMPLEXES:
            to_remove.append(dim)
            print(f"❌ REMOVE: {dim}")
            poles = simplex['three_pole_simplex']
            print(f"   Reason: Culturally specific or wrong concept")
            print(f"   μ−: {poles['negative_pole']['synset']}")
            print(f"   μ0: {poles['neutral_homeostasis']['synset']}")
            print(f"   μ+: {poles['positive_pole']['synset']}")
            print()

        elif dim in FIX_SIMPLEXES:
            to_fix.append(dim)
            print(f"⚠️  FIX: {dim}")
            print(f"   Issue: {FIX_SIMPLEXES[dim]['issue']}")
            print()

        else:
            valid.append(dim)

    print(f"\nSummary:")
    print(f"  Valid: {len(valid)}")
    print(f"  Need fixing: {len(to_fix)}")
    print(f"  To remove: {len(to_remove)}")

    return simplexes, to_remove, to_fix, valid


def fix_simplexes():
    """Apply fixes to layer2.json"""

    print("\n" + "=" * 80)
    print("APPLYING FIXES")
    print("=" * 80)

    layer2 = load_json(LAYER2_PATH)
    concepts = layer2['concepts']

    # Backup
    backup_path = LAYER2_PATH.with_suffix('.json.backup_before_simplex_audit')
    print(f"\nCreating backup: {backup_path}")
    save_json(backup_path, layer2)

    # Remove bad simplexes
    print(f"\nRemoving {len(REMOVE_SIMPLEXES)} simplexes...")
    concepts_filtered = []
    removed_count = 0

    for concept in concepts:
        dim = concept.get('simplex_dimension')
        if dim and dim in REMOVE_SIMPLEXES:
            print(f"  ❌ Removed: {dim}")
            removed_count += 1
        else:
            concepts_filtered.append(concept)

    # Fix problematic poles
    print(f"\nFixing {len(FIX_SIMPLEXES)} simplexes...")
    fixed_count = 0

    for concept in concepts_filtered:
        dim = concept.get('simplex_dimension')
        if dim and dim in FIX_SIMPLEXES:
            fix = FIX_SIMPLEXES[dim]['fix']

            # Update positive pole
            if 'positive_pole' in fix:
                new_pole = fix['positive_pole']
                synset_id = new_pole['synset']

                # Check if synset exists
                valid, definition = check_synset_valid(synset_id)

                if valid:
                    # Get lemmas from WordNet
                    ss = wn.synset(synset_id)
                    lemmas = [l.name() for l in ss.lemmas()]

                    concept['three_pole_simplex']['positive_pole'] = {
                        'synset': synset_id,
                        'lemmas': lemmas,
                        'definition': definition
                    }
                    print(f"  ✓ Fixed {dim}: positive pole → {synset_id}")
                    fixed_count += 1
                else:
                    print(f"  ⚠️  {dim}: synset {synset_id} not found in WordNet")

    # Update metadata
    layer2['concepts'] = concepts_filtered
    if 'metadata' in layer2:
        layer2['metadata']['total_concepts'] = len(concepts_filtered)
        layer2['metadata']['s_tier_count'] = sum(1 for c in concepts_filtered if c.get('s_tier'))

    # Save
    print(f"\nSaving updated layer2.json...")
    save_json(LAYER2_PATH, layer2)

    print(f"\n✅ Complete!")
    print(f"   Removed: {removed_count} simplexes")
    print(f"   Fixed: {fixed_count} simplexes")
    print(f"   Total concepts: {len(concepts_filtered)}")

    return concepts_filtered


def create_custom_sumo_concepts():
    """Create custom SUMO concept definitions file"""

    print("\n" + "=" * 80)
    print("CREATING CUSTOM SUMO CONCEPTS")
    print("=" * 80)

    sumo_file = {
        "metadata": {
            "description": "Custom SUMO concepts for incomplete S-tier simplexes",
            "total_concepts": len(NEW_SUMO_CONCEPTS)
        },
        "concepts": []
    }

    for concept_name, data in NEW_SUMO_CONCEPTS.items():
        entry = {
            "sumo_term": concept_name,
            "definition": data["definition"],
            "pos": data["pos"],
            "lexname": data["lexname"],
            "related_synsets": data.get("related_synsets", []),
            "is_custom": True,
            "clinical": data.get("clinical", False)
        }
        sumo_file["concepts"].append(entry)

        print(f"\n{concept_name}:")
        print(f"  {data['definition']}")

    save_json(SUMO_CONCEPTS_PATH, sumo_file)
    print(f"\n✅ Saved to: {SUMO_CONCEPTS_PATH}")


def main():
    # Audit
    simplexes, to_remove, to_fix, valid = audit_simplexes()

    # Show what we'll do
    print("\n" + "=" * 80)
    print("PROPOSED ACTIONS")
    print("=" * 80)
    print(f"1. Remove {len(to_remove)} culturally-specific/problematic simplexes")
    print(f"2. Fix {len(to_fix)} simplexes with wrong poles")
    print(f"3. Create {len(NEW_SUMO_CONCEPTS)} custom SUMO concepts")
    print(f"4. Result: {len(valid) + len(to_fix) - len(to_remove)} clean simplexes")

    response = input("\nProceed? [y/N] ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Apply fixes
    concepts = fix_simplexes()

    # Create custom SUMO concepts
    create_custom_sumo_concepts()

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review layer2.json to verify fixes")
    print("2. Add the 3 incomplete simplexes with custom SUMO poles")
    print("3. Continue with S+ training on clean simplex set")
    print("=" * 80)


if __name__ == "__main__":
    main()
