#!/usr/bin/env python3
"""
Add the 3 incomplete simplexes with custom SUMO concepts

These are high-quality simplexes that were excluded because they need
custom SUMO concepts for some poles:

1. affective_awareness: alexithymia ↔ affect ↔ emotional_flooding
2. hedonic_arousal_intensity: anhedonia ↔ contented_sensuality ↔ algolagnia
3. social_connection: alienation ↔ interdependence ↔ enmeshment
"""

import json
from pathlib import Path
from nltk.corpus import wordnet as wn

PROJECT_ROOT = Path(__file__).parent.parent
LAYER2_PATH = PROJECT_ROOT / "data" / "concept_graph" / "abstraction_layers" / "layer2.json"
REVIEW_PATH = PROJECT_ROOT / "results" / "simplex_agentic_review.json"
CUSTOM_SUMO_PATH = PROJECT_ROOT / "data" / "concept_graph" / "custom_sumo_concepts.kif"

# Custom SUMO concepts to create
CUSTOM_SUMO_CONCEPTS = {
    "AlexithymiaState": {
        "definition": "The inability to identify and describe emotions in oneself; emotional blindness or unawareness of one's own affective states.",
        "isa": "PsychologicalAttribute",
        "documentation": "Alexithymia is characterized by difficulty identifying feelings, difficulty describing feelings to others, and an externally oriented thinking style. It represents the negative pole of affective awareness."
    },
    "EmotionalFlooding": {
        "definition": "An excessive, overwhelming state where affective experience becomes so intense that it impairs psychological functioning and behavioral regulation.",
        "isa": "EmotionalState",
        "documentation": "Emotional flooding occurs when emotions become so strong they overwhelm cognitive and regulatory capacities, leading to dysregulation. It represents hyper-awareness that destabilizes rather than informs."
    },
    "ContentedSensuality": {
        "definition": "Balanced, healthy capacity for physical pleasure and sensory enjoyment without compulsion or excess.",
        "isa": "EmotionalState",
        "documentation": "The middle path between anhedonia (inability to feel pleasure) and compulsive pleasure-seeking. Represents healthy, non-attached enjoyment of sensory experience."
    },
    "Enmeshment": {
        "definition": "Excessive emotional or psychological dependence in relationships characterized by loss of individual boundaries and autonomy.",
        "isa": "PsychologicalAttribute",
        "documentation": "Enmeshment represents pathological interdependence where individual identity becomes fused with others, compromising autonomy and healthy differentiation."
    }
}

# The 3 incomplete simplexes with their complete structure
INCOMPLETE_SIMPLEXES = [
    {
        "dimension": "affective_awareness",
        "three_pole_simplex": {
            "negative_pole": {
                "concept": "AlexithymiaState",
                "synset": None,  # Custom SUMO
                "is_custom_sumo": True,
                "lemmas": ["alexithymia", "emotional_blindness"],
                "definition": CUSTOM_SUMO_CONCEPTS["AlexithymiaState"]["definition"]
            },
            "neutral_homeostasis": {
                "concept": "Affect",
                "synset": "affect.n.01",
                "lemmas": [],
                "definition": ""
            },
            "positive_pole": {
                "concept": "EmotionalFlooding",
                "synset": None,  # Custom SUMO
                "is_custom_sumo": True,
                "lemmas": ["emotional_flooding", "affective_dysregulation"],
                "definition": CUSTOM_SUMO_CONCEPTS["EmotionalFlooding"]["definition"]
            }
        },
        "confidence": 8,
        "flags": ["needs_custom_synset", "clinical_dimension"]
    },
    {
        "dimension": "hedonic_arousal_intensity",
        "three_pole_simplex": {
            "negative_pole": {
                "concept": "Anhedonia",
                "synset": "anhedonia.n.01",
                "lemmas": [],
                "definition": ""
            },
            "neutral_homeostasis": {
                "concept": "ContentedSensuality",
                "synset": None,  # Custom SUMO
                "is_custom_sumo": True,
                "lemmas": ["contented_sensuality", "healthy_pleasure"],
                "definition": CUSTOM_SUMO_CONCEPTS["ContentedSensuality"]["definition"]
            },
            "positive_pole": {
                "concept": "Algolagnia",
                "synset": "algolagnia.n.01",
                "lemmas": [],
                "definition": ""
            }
        },
        "confidence": 8,
        "flags": ["needs_custom_synset", "sensitive_domain"]
    },
    {
        "dimension": "social_connection",
        "three_pole_simplex": {
            "negative_pole": {
                "concept": "Alienation",
                "synset": "alienation.n.01",
                "lemmas": [],
                "definition": ""
            },
            "neutral_homeostasis": {
                "concept": "Interdependence",
                "synset": "interdependence.n.01",
                "lemmas": [],
                "definition": ""
            },
            "positive_pole": {
                "concept": "Enmeshment",
                "synset": None,  # Custom SUMO
                "is_custom_sumo": True,
                "lemmas": ["enmeshment", "fusion"],
                "definition": CUSTOM_SUMO_CONCEPTS["Enmeshment"]["definition"]
            }
        },
        "confidence": 9,
        "flags": ["needs_custom_synset"]
    }
]


def enrich_with_wordnet(pole):
    """Enrich pole with WordNet data if it has a synset"""
    if pole.get('synset') and not pole.get('is_custom_sumo'):
        try:
            ss = wn.synset(pole['synset'])
            pole['lemmas'] = [l.name() for l in ss.lemmas()]
            pole['definition'] = ss.definition()
        except:
            print(f"  ⚠️  Could not find {pole['synset']} in WordNet")


def create_layer_entry(simplex):
    """Create a layer2 entry for an incomplete simplex with custom SUMO"""

    dimension = simplex['dimension']
    three_pole = simplex['three_pole_simplex']

    # Use neutral as canonical, or first non-custom if neutral is custom
    if not three_pole['neutral_homeostasis'].get('is_custom_sumo'):
        canonical = three_pole['neutral_homeostasis']
    elif not three_pole['negative_pole'].get('is_custom_sumo'):
        canonical = three_pole['negative_pole']
    else:
        canonical = three_pole['positive_pole']

    # Collect all synsets (None for custom SUMO)
    synsets = []
    for pole_name in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
        pole = three_pole[pole_name]
        if pole.get('synset'):
            synsets.append(pole['synset'])

    canonical_synset = canonical.get('synset', f"custom.{dimension}.01")
    lemmas = canonical.get('lemmas', [dimension])

    entry = {
        "sumo_term": f"{dimension.replace('_', ' ').title().replace(' ', '')}_Simplex",
        "sumo_depth": 2,
        "layer": 2,
        "is_category_lens": False,
        "is_pseudo_sumo": True,
        "category_children": [],
        "synset_count": 3,
        "direct_synset_count": len(synsets),
        "synsets": synsets,
        "canonical_synset": canonical_synset,
        "lemmas": lemmas if lemmas else [dimension],
        "pos": "n",
        "definition": canonical.get('definition', f"Three-pole simplex for {dimension}"),
        "lexname": "noun.feeling",
        "s_tier": True,
        "simplex_dimension": dimension,
        "three_pole_simplex": three_pole,
        "has_custom_sumo": True,
        "training_priority": "standard",
        "flags": simplex.get('flags', [])
    }

    return entry


def create_sumo_kif():
    """Create KIF file with custom SUMO concepts"""

    kif_content = """;; Custom SUMO concepts for HatCat S-tier simplexes
;; Generated for incomplete simplexes that need custom affective concepts

"""

    for concept_name, data in CUSTOM_SUMO_CONCEPTS.items():
        kif_content += f"""
(instance {concept_name} {data['isa']})
(documentation {concept_name} EnglishLanguage "{data['definition']}")
"""
        if 'documentation' in data:
            kif_content += f";; {data['documentation']}\n"

    return kif_content


def main():
    print("=" * 80)
    print("ADDING INCOMPLETE SIMPLEXES WITH CUSTOM SUMO CONCEPTS")
    print("=" * 80)

    # Load layer2
    with open(LAYER2_PATH) as f:
        layer2 = json.load(f)

    concepts = layer2['concepts']
    original_count = len(concepts)

    print(f"\nCurrent layer2 concepts: {original_count}")

    # Backup
    backup_path = LAYER2_PATH.with_suffix('.json.backup_before_custom_sumo')
    print(f"Creating backup: {backup_path}")
    with open(backup_path, 'w') as f:
        json.dump(layer2, f, indent=2)

    # Process each incomplete simplex
    print(f"\nAdding {len(INCOMPLETE_SIMPLEXES)} simplexes with custom SUMO concepts...")

    for simplex in INCOMPLETE_SIMPLEXES:
        dimension = simplex['dimension']

        # Enrich WordNet poles
        for pole_name in ['negative_pole', 'neutral_homeostasis', 'positive_pole']:
            pole = simplex['three_pole_simplex'][pole_name]
            enrich_with_wordnet(pole)

        # Create layer entry
        entry = create_layer_entry(simplex)
        concepts.append(entry)

        print(f"\n  ✓ Added: {dimension}")
        three_pole = simplex['three_pole_simplex']
        for pole_name, symbol in [('negative_pole', 'μ−'), ('neutral_homeostasis', 'μ0'), ('positive_pole', 'μ+')]:
            pole = three_pole[pole_name]
            custom = " [CUSTOM SUMO]" if pole.get('is_custom_sumo') else ""
            synset = pole.get('synset', pole['concept'])
            print(f"    {symbol} {synset}{custom}")

    # Update metadata
    new_count = len(concepts)
    if 'metadata' in layer2:
        layer2['metadata']['total_concepts'] = new_count
        layer2['metadata']['s_tier_count'] = sum(1 for c in concepts if c.get('s_tier'))
        layer2['metadata']['custom_sumo_count'] = sum(1 for c in concepts if c.get('has_custom_sumo'))

    # Save layer2
    layer2['concepts'] = concepts
    with open(LAYER2_PATH, 'w') as f:
        json.dump(layer2, f, indent=2)

    print(f"\n✅ Updated layer2.json")
    print(f"   Original: {original_count}")
    print(f"   Added: {len(INCOMPLETE_SIMPLEXES)}")
    print(f"   New total: {new_count}")

    # Create custom SUMO KIF file
    print(f"\nCreating custom SUMO concepts file...")
    kif_content = create_sumo_kif()
    with open(CUSTOM_SUMO_PATH, 'w') as f:
        f.write(kif_content)

    print(f"✅ Saved to: {CUSTOM_SUMO_PATH}")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL S-TIER SIMPLEX SET")
    print("=" * 80)

    s_tier_simplexes = [c for c in concepts if c.get('simplex_dimension')]

    print(f"\nTotal S-tier simplexes: {len(s_tier_simplexes)}")
    print(f"  With all WordNet synsets: {len([s for s in s_tier_simplexes if not s.get('has_custom_sumo')])}")
    print(f"  With custom SUMO concepts: {len([s for s in s_tier_simplexes if s.get('has_custom_sumo')])}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review layer2.json to verify all simplexes")
    print("2. Update S+ training spec for 20 simplexes (60 lenses)")
    print("3. Modify sumo_data_generation.py for three-pole training")
    print("4. Begin S+ training")
    print("=" * 80)


if __name__ == "__main__":
    main()
