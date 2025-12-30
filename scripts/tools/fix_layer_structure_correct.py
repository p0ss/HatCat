#!/usr/bin/env python3
"""
Fix Layer Structure (Correct Approach)

1. Remove Layer 5 entirely, redistributing "_Other" synsets to parent concepts
2. Create Layer 0 cross-training specification (negative examples from other L0 concepts)
3. Document hierarchical suppression strategy

IMPORTANT: Does NOT duplicate synsets across layers!
Each synset belongs to ONE most-specific layer only.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List


def load_all_concepts(layers_dir: Path) -> Dict:
    """Load all concepts from all layers."""
    all_concepts = {}

    for layer_file in sorted(layers_dir.glob("layer*.json")):
        with open(layer_file) as f:
            data = json.load(f)
            layer = data['metadata']['layer']

            for concept in data['concepts']:
                name = concept['sumo_term']
                key = (name, layer)
                all_concepts[key] = concept

    return all_concepts


def redistribute_layer5_to_parents(layers_dir: Path, dry_run: bool = False):
    """
    Merge Layer 5 "_Other" and "_OtherAgent" concepts back into their parents.
    """
    print("\n" + "="*80)
    print("STEP 1: REDISTRIBUTING LAYER 5 TO PARENTS")
    print("="*80)

    all_concepts = load_all_concepts(layers_dir)

    # Find Layer 5 concepts to redistribute
    layer5_concepts = {k: v for k, v in all_concepts.items() if k[1] == 5}
    to_redistribute = {
        k: v for k, v in layer5_concepts.items()
        if '_Other' in k[0] or '_OtherAgent' in k[0]
    }

    print(f"\nFound {len(to_redistribute)} Layer 5 '_Other' grab-bag concepts:")
    for (name, layer), concept in sorted(to_redistribute.items()):
        parents = concept.get('parent_concepts', [])
        synset_count = len(concept.get('synsets', []))
        print(f"  {name}: {synset_count} synsets -> parent: {parents[0] if parents else 'NONE'}")

    # Build redistribution plan
    redistribution_plan = defaultdict(lambda: {'synsets': [], 'children': []})
    concepts_to_remove = []

    for (name, layer), concept in to_redistribute.items():
        parents = concept.get('parent_concepts', [])
        if not parents:
            print(f"  WARNING: {name} has no parent, skipping")
            continue

        parent_name = parents[0]
        synsets = concept.get('synsets', [])
        redistribution_plan[parent_name]['synsets'].extend(synsets)
        redistribution_plan[parent_name]['children'].append(name)
        concepts_to_remove.append((name, layer))

    print(f"\nRedistribution plan:")
    print(f"  Will remove {len(concepts_to_remove)} Layer 5 concepts")
    print(f"  Will enhance {len(redistribution_plan)} parent concepts")

    for parent, data in sorted(redistribution_plan.items()):
        print(f"    {parent}: +{len(data['synsets'])} synsets (from {len(data['children'])} children)")

    # Apply redistribution
    if not dry_run:
        # Update parent concepts with additional synsets
        for layer_file in sorted(layers_dir.glob("layer[0-4].json")):
            with open(layer_file) as f:
                data = json.load(f)

            modified = False
            for concept in data['concepts']:
                name = concept['sumo_term']
                if name in redistribution_plan:
                    # Add synsets from child "_Other" concepts
                    existing_synsets = set(concept.get('synsets', []))
                    new_synsets = redistribution_plan[name]['synsets']
                    combined = sorted(set(list(existing_synsets) + new_synsets))

                    old_count = len(existing_synsets)
                    new_count = len(combined)

                    concept['synsets'] = combined
                    concept['synset_count'] = new_count

                    # Remove the "_Other" child from category_children
                    children = concept.get('category_children', [])
                    removed_children = redistribution_plan[name]['children']
                    concept['category_children'] = [c for c in children if c not in removed_children]

                    modified = True

                    print(f"  ✓ Enhanced {name}: {old_count} -> {new_count} synsets (+{new_count-old_count})")

            if modified:
                # Update metadata
                data['metadata']['total_concepts'] = len(data['concepts'])

                with open(layer_file, 'w') as f:
                    json.dump(data, f, indent=2)

        # Remove "_Other" concepts from Layer 5
        layer5_file = layers_dir / "layer5.json"
        if layer5_file.exists():
            with open(layer5_file) as f:
                data = json.load(f)

            original_count = len(data['concepts'])
            data['concepts'] = [
                c for c in data['concepts']
                if not ('_Other' in c['sumo_term'] or '_OtherAgent' in c['sumo_term'])
            ]

            data['metadata']['total_concepts'] = len(data['concepts'])

            with open(layer5_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"\n✓ Layer 5: {original_count} -> {len(data['concepts'])} concepts (removed {original_count - len(data['concepts'])})")
    else:
        print("\n[DRY RUN] Would apply redistribution")

    return len(concepts_to_remove)


def create_layer0_training_spec(layers_dir: Path, output_file: Path, dry_run: bool = False):
    """
    Create training specification for Layer 0 cross-concept distinction.

    Each Layer 0 concept should be trained with:
    - Its own synsets ONLY (no descendants - they stay at their specific layers)
    - Synsets from ALL other Layer 0 concepts as negative examples
    - This ensures Layer 0 concepts can distinguish from each other
    """
    print("\n" + "="*80)
    print("STEP 2: CREATING LAYER 0 CROSS-TRAINING SPEC")
    print("="*80)

    layer0_file = layers_dir / "layer0.json"

    with open(layer0_file) as f:
        data = json.load(f)

    layer0_concepts = {}
    for concept in data['concepts']:
        name = concept['sumo_term']
        synsets = concept.get('synsets', [])
        layer0_concepts[name] = synsets

    # Build training spec
    training_spec = {
        'description': 'Layer 0 cross-concept distinction training specification',
        'created': datetime.now().isoformat(),
        'strategy': {
            'positive_examples': 'ONLY the concepts own direct synsets (5-10 each)',
            'negative_examples': 'ALL synsets from the other 13 Layer 0 concepts',
            'purpose': 'Teach Layer 0 concepts to distinguish from each other',
            'note': 'Does NOT include descendant synsets - hierarchy is implicit through parent_concepts'
        },
        'hierarchical_suppression': {
            'description': 'At inference time, suppress parent activations when children activate',
            'example': 'If "Bird" activates strongly, suppress "Physical" and "Object"',
            'implementation': 'If child_score > threshold AND parent_score > threshold, output only child'
        },
        'concepts': {}
    }

    print(f"\nLayer 0 Cross-Training Requirements:")
    print(f"{'Concept':<30} {'Positive':<12} {'Negative':<12} {'Status':<10}")
    print("-"*80)

    for name, synsets in sorted(layer0_concepts.items()):
        # Negative examples are all OTHER Layer 0 concepts
        negative_concepts = [n for n in layer0_concepts.keys() if n != name]
        negative_synset_count = sum(
            len(layer0_concepts[neg]) for neg in negative_concepts
        )

        training_spec['concepts'][name] = {
            'positive_synsets': len(synsets),
            'positive_examples': synsets,
            'negative_concepts': negative_concepts,
            'negative_synsets': negative_synset_count,
            'has_training_data': len(synsets) > 0
        }

        status = "✓ OK" if len(synsets) > 0 else "⚠️  NONE"
        print(f"{name:<30} {len(synsets):<12} {negative_synset_count:<12} {status:<10}")

    # Identify problems
    missing_data = [name for name, synsets in layer0_concepts.items() if len(synsets) == 0]

    if missing_data:
        print(f"\n⚠️  WARNING: {len(missing_data)} Layer 0 concepts have NO training data:")
        for name in missing_data:
            print(f"     - {name}")
        print(f"\n   These concepts cannot be trained without synsets!")
        print(f"   They should be removed or assigned appropriate WordNet synsets.")

    # Save spec
    if not dry_run:
        with open(output_file, 'w') as f:
            json.dump(training_spec, f, indent=2)
        print(f"\n✓ Saved training spec to: {output_file}")
    else:
        print(f"\n[DRY RUN] Would save training spec to: {output_file}")

    return training_spec


def create_hierarchical_suppression_doc(output_file: Path):
    """
    Create documentation for hierarchical suppression strategy.
    """
    print("\n" + "="*80)
    print("STEP 3: CREATING HIERARCHICAL SUPPRESSION DOCUMENTATION")
    print("="*80)

    doc = """# Hierarchical Suppression Strategy

## Problem

Layer 0 concepts like "Entity", "Physical", and "Object" are very abstract and will naturally activate on many inputs. This is not a bug - it's by design! A sparrow IS a physical entity.

However, when displaying results, we want to show the MOST SPECIFIC concept, not all ancestors.

## Solution: Hierarchical Suppression

When multiple lenses activate in a parent-child chain, suppress the parent activations and show only the most specific (deepest) concept.

### Algorithm

```python
def apply_hierarchical_suppression(activations, concept_metadata, threshold=0.5):
    \"\"\"
    Suppress parent concept activations when children activate.

    Args:
        activations: dict of {concept_name: activation_score}
        concept_metadata: dict with parent-child relationships
        threshold: minimum score to consider activated

    Returns:
        Filtered activations with parents suppressed
    \"\"\"
    suppressed = set()

    # For each activated concept
    for concept, score in activations.items():
        if score < threshold:
            continue

        # Check if any of its children also activated
        children = concept_metadata[concept].category_children
        for child in children:
            child_score = activations.get(child, 0.0)

            # If child activated, suppress this parent
            if child_score >= threshold:
                suppressed.add(concept)
                break

    # Return activations with suppressed parents removed
    return {
        concept: score
        for concept, score in activations.items()
        if concept not in suppressed
    }
```

### Example

Input activations:
- Entity (L0): 0.85
- Physical (L0): 0.82
- Object (L0): 0.78
- Animal (L2): 0.92
- Bird (L3): 0.95
- Sparrow (L4): 0.88

After hierarchical suppression:
- Sparrow (L4): 0.88  ← Most specific, keep this

Suppressed:
- Entity, Physical, Object (ancestors of Sparrow)
- Animal, Bird (ancestors of Sparrow)

## Implementation

This should be applied in:
1. `DynamicLensManager.detect_concepts()` - after getting raw activations
2. Streamlit UI - before displaying concept tags
3. Any analysis scripts that report concept activations

## Benefits

1. **Cleaner output**: Shows "Sparrow" instead of "Sparrow, Bird, Animal, Physical, Object, Entity"
2. **Preserves hierarchy**: Still maintains the semantic relationships
3. **No retraining needed**: Works with existing lenses
4. **Handles over-firing**: Layer 0 can activate broadly without polluting results

## Training Strategy for Layer 0

Even with hierarchical suppression, Layer 0 concepts need good training:

1. **Positive examples**: Their direct synsets only (5-10 each)
2. **Negative examples**: ALL other Layer 0 concept synsets
3. **Purpose**: Distinguish "Entity" from "Proposition", "Physical" from "Abstract", etc.
4. **Result**: Layer 0 lenses learn their UNIQUE characteristics, not just general abstractness

Example for "Physical":
- Positive: physical_entity.n.01, phenomenon.n.01, etc. (5 synsets)
- Negative: All synsets from Abstract, Proposition, Relation, etc. (50+ synsets)

This teaches "Physical" to recognize physical/tangible things and REJECT abstract concepts, even though both are Layer 0.
"""

    output_file.write_text(doc)
    print(f"✓ Created documentation: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix layer structure (correct approach - no synset duplication)"
    )
    parser.add_argument('--layers-dir', type=Path,
                       default=Path('data/concept_graph/abstraction_layers'))
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')

    args = parser.parse_args()

    print("="*80)
    print("LAYER STRUCTURE FIX (CORRECT APPROACH)")
    print("="*80)
    print(f"Layers dir: {args.layers_dir}")
    print(f"Dry run: {args.dry_run}")
    print("\nPrinciple: Each synset belongs to ONE most-specific layer only.")
    print("Hierarchy is implicit through parent_concepts, not data duplication.")

    # Create backup
    if not args.dry_run:
        backup_dir = args.layers_dir / 'backups'
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        print(f"\nCreating backup...")
        for layer_file in args.layers_dir.glob("layer*.json"):
            backup_file = backup_dir / f"{layer_file.stem}_{timestamp}.json"
            shutil.copy2(layer_file, backup_file)
        print(f"✓ Backed up to {backup_dir}")

    # Step 1: Redistribute Layer 5 "_Other" concepts
    removed_count = redistribute_layer5_to_parents(args.layers_dir, args.dry_run)

    # Step 2: Create Layer 0 cross-training spec
    spec_file = Path('docs/LAYER0_TRAINING_SPEC.json')
    training_spec = create_layer0_training_spec(args.layers_dir, spec_file, args.dry_run)

    # Step 3: Create hierarchical suppression documentation
    if not args.dry_run:
        doc_file = Path('docs/HIERARCHICAL_SUPPRESSION.md')
        create_hierarchical_suppression_doc(doc_file)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Removed {removed_count} Layer 5 '_Other' grab-bag concepts")
    print(f"✓ Merged their synsets back to parent concepts")
    print(f"✓ Created Layer 0 cross-training specification")
    print(f"✓ Documented hierarchical suppression strategy")
    print("\nNext steps:")
    print("  1. Review docs/LAYER0_TRAINING_SPEC.json")
    print("  2. Implement hierarchical suppression in DynamicLensManager")
    print("  3. Retrain Layer 0 lenses with cross-concept negative examples")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
