#!/usr/bin/env python3
"""
Assemble trained SUMO classifiers into a versioned lens pack.

This script:
1. Copies trained lenses from results/sumo_classifiers/ to concept_packs/
2. Creates a manifest with training metadata
3. Versions the pack based on training date
4. Includes quality metrics for each lens

Usage:
    python scripts/assemble_lens_pack.py \\
        --source results/sumo_classifiers/ \\
        --pack-id sumo-wordnet-lenses-v2 \\
        --model google/gemma-3-4b-pt \\
        --description "SUMO concept classifiers trained on Gemma-3-4b-pt"
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_layer_results(layer_dir: Path) -> dict:
    """Load training results for a layer."""
    results_file = layer_dir / "results.json"
    if not results_file.exists():
        return None

    try:
        with open(results_file) as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse {results_file}")
        return None


def assemble_lens_pack(
    source_dir: Path,
    pack_id: str,
    model_name: str,
    description: str,
    version: str = None,
    output_dir: Path = None
):
    """
    Assemble trained lenses into a versioned pack.

    Args:
        source_dir: Directory containing layer subdirectories with lenses
        pack_id: Pack identifier (e.g., 'sumo-wordnet-lenses-v2')
        model_name: Model used for training
        description: Pack description
        version: Version string (auto-generated if not provided)
        output_dir: Where to create the pack (default: concept_packs/{pack_id})
    """
    source_dir = Path(source_dir)

    # Auto-generate version from timestamp if not provided
    if version is None:
        version = datetime.now().strftime("2.%Y%m%d.0")

    # Determine output directory
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'concept_packs' / pack_id
    else:
        output_dir = Path(output_dir)

    print(f"Assembling lens pack: {pack_id}")
    print(f"  Version: {version}")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print()

    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    hierarchy_dir = output_dir / 'hierarchy'
    hierarchy_dir.mkdir(exist_ok=True)

    # Collect metadata
    total_lenses = 0
    layer_stats = {}
    all_concepts = []

    # Process each layer
    for layer_name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
        layer_dir = source_dir / layer_name

        if not layer_dir.exists():
            print(f"Skipping {layer_name} (not found)")
            continue

        print(f"Processing {layer_name}...")

        # Load layer results
        layer_results = load_layer_results(layer_dir)
        if layer_results is None:
            print(f"  Warning: No results.json found, skipping")
            continue

        # Copy classifier files
        classifier_files = list(layer_dir.glob("*_classifier.pt"))

        if not classifier_files:
            print(f"  Warning: No classifier files found")
            continue

        copied = 0
        for classifier_file in classifier_files:
            # Extract concept name (e.g., "AAM_classifier.pt" -> "AAM")
            concept_name = classifier_file.stem.replace('_classifier', '')

            # Copy to hierarchy directory
            dest_file = hierarchy_dir / classifier_file.name
            shutil.copy2(classifier_file, dest_file)

            all_concepts.append(concept_name)
            copied += 1

        layer_stats[layer_name] = {
            'total_concepts': layer_results.get('n_concepts', 0),
            'successful': layer_results.get('n_successful', 0),
            'failed': layer_results.get('n_failed', 0),
            'elapsed_minutes': layer_results.get('elapsed_minutes', 0),
            'copied_files': copied
        }

        total_lenses += copied
        print(f"  ✓ Copied {copied} lenses ({layer_results.get('n_successful', 0)}/{layer_results.get('n_concepts', 0)} successful)")

    print(f"\n✓ Copied {total_lenses} total lenses")

    # Create manifest
    manifest = {
        "pack_id": pack_id,
        "version": version,
        "created": datetime.now().isoformat(),
        "description": description,

        "model": {
            "name": model_name,
            "type": "causal_lm",
            "dtype": "float32"
        },

        "training": {
            "validation_mode": "falloff",
            "validation_layer": 12,
            "trainer": "DualAdaptiveTrainer",
            "data_source": "SUMO ontology with WordNet synsets",
            "training_date": datetime.now().strftime("%Y-%m-%d")
        },

        "lenses": {
            "total_count": total_lenses,
            "format": "pytorch",
            "layer_distribution": layer_stats,
            "concepts": sorted(all_concepts)
        },

        "compatibility": {
            "hatcat_version": ">=0.1.0",
            "requires": ["sumo-wordnet-v1"]
        },

        "usage": {
            "load_example": f"from pathlib import Path; lens = torch.load(Path('concept_packs/{pack_id}/hierarchy/{{concept}}_classifier.pt'))",
            "batch_loading": "Use src.registry.concept_pack_registry.ConceptPackRegistry"
        }
    }

    # Save manifest
    manifest_file = output_dir / 'lens_pack.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Manifest saved to {manifest_file}")

    # Create README
    readme_content = f"""# {pack_id}

{description}

## Overview

- **Version**: {version}
- **Model**: {model_name}
- **Total Lenses**: {total_lenses}
- **Created**: {datetime.now().strftime('%Y-%m-%d')}

## Layer Distribution

"""

    for layer_name, stats in layer_stats.items():
        readme_content += f"- **{layer_name}**: {stats['copied_files']} lenses ({stats['successful']}/{stats['total_concepts']} successful)\n"

    readme_content += f"""

## Training Details

- **Trainer**: DualAdaptiveTrainer with adaptive falloff validation
- **Validation Layer**: 12
- **Validation Mode**: falloff
- **Data Source**: SUMO ontology concepts with WordNet synsets

## Usage

### Load a Single Lens

```python
import torch
from pathlib import Path

# Load lens
lens_path = Path('concept_packs/{pack_id}/hierarchy/Animal_classifier.pt')
lens = torch.load(lens_path)

# Use for prediction
# (Assuming lens is a trained classifier)
```

### Load All Lenses

```python
from src.map.concept_pack import load_concept_pack

pack = load_concept_pack('{pack_id}')

# Access lens paths
hierarchy_dir = pack['pack_path'] / 'hierarchy'
lenses = {{
    f.stem.replace('_classifier', ''): torch.load(f)
    for f in hierarchy_dir.glob('*_classifier.pt')
}}
```

## Files

- `lens_pack.json`: Manifest with metadata
- `hierarchy/`: Directory containing {total_lenses} trained classifier files
- `README.md`: This file

## License

MIT

## References

- Base ontology: SUMO 2003
- Concept pack: sumo-wordnet-v1
- Training framework: HatCat v0.1.0
"""

    readme_file = output_dir / 'README.md'
    with open(readme_file, 'w') as f:
        f.write(readme_content)

    print(f"✓ README saved to {readme_file}")

    # Create summary
    print("\n" + "=" * 80)
    print("LENS PACK ASSEMBLY COMPLETE")
    print("=" * 80)
    print(f"\nPack: {pack_id} v{version}")
    print(f"Location: {output_dir}")
    print(f"Total lenses: {total_lenses}")
    print(f"\nFiles created:")
    print(f"  - lens_pack.json (manifest)")
    print(f"  - README.md (documentation)")
    print(f"  - hierarchy/ ({total_lenses} classifier files)")
    print("\n" + "=" * 80)

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Assemble trained SUMO classifiers into a versioned lens pack"
    )
    parser.add_argument(
        '--source',
        type=Path,
        default=PROJECT_ROOT / 'results' / 'sumo_classifiers',
        help='Source directory containing layer subdirectories (default: results/sumo_classifiers/)'
    )
    parser.add_argument(
        '--pack-id',
        default='sumo-wordnet-lenses-v2',
        help='Pack identifier (default: sumo-wordnet-lenses-v2)'
    )
    parser.add_argument(
        '--model',
        default='google/gemma-3-4b-pt',
        help='Model name used for training (default: google/gemma-3-4b-pt)'
    )
    parser.add_argument(
        '--description',
        default='SUMO concept classifiers trained on Gemma-3-4b-pt with adaptive falloff validation',
        help='Pack description'
    )
    parser.add_argument(
        '--version',
        help='Version string (auto-generated from date if not provided)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory (default: concept_packs/{pack_id})'
    )

    args = parser.parse_args()

    assemble_lens_pack(
        source_dir=args.source,
        pack_id=args.pack_id,
        model_name=args.model,
        description=args.description,
        version=args.version,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
