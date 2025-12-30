#!/usr/bin/env python3
"""
Create a new concept pack from scratch.

Usage:
    python scripts/create_concept_pack.py ai-safety \\
        --description "AI safety concepts" \\
        --author "HatCat Team" \\
        --license MIT
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def create_pack(
    pack_id: str,
    description: str,
    author: str = "HatCat Team",
    license: str = "MIT",
    output_dir: Path = None,
) -> Path:
    """
    Create a new concept pack directory structure.

    Args:
        pack_id: Unique pack identifier (e.g., 'ai-safety-v1')
        description: Brief description of the pack
        author: Pack author name
        license: License identifier (SPDX format)
        output_dir: Where to create the pack (default: concept_packs/{pack_id})

    Returns:
        Path to created pack directory
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'concept_packs' / pack_id
    else:
        output_dir = Path(output_dir)

    # Create directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'wordnet_patches').mkdir(exist_ok=True)
    (output_dir / 'layer_entries').mkdir(exist_ok=True)

    # Create pack.json
    pack_json = {
        "pack_id": pack_id,
        "version": "1.0.0",
        "created": datetime.now().isoformat() + "Z",
        "description": description,
        "authors": [author],
        "license": license,
        "repository": "",

        "ontology_stack": {
            "base_ontology": {
                "name": "SUMO",
                "version": "2003",
                "required": True
            },
            "dependencies": [
                {
                    "pack_id": "sumo-wordnet-v1",
                    "version": ">=1.0.0",
                    "required": True
                }
            ],
            "domain_extensions": []
        },

        "concept_metadata": {
            "total_concepts": 0,
            "new_concepts": 0,
            "modified_concepts": 0,
            "layers": [],
            "layer_distribution": {}
        },

        "compatibility": {
            "hatcat_version": ">=0.1.0",
            "required_dependencies": {
                "wordnet": "3.0"
            }
        },

        "installation": {
            "requires_recalculation": True,
            "backup_recommended": True,
            "conflicts_with": []
        }
    }

    pack_json_path = output_dir / 'pack.json'
    with open(pack_json_path, 'w') as f:
        json.dump(pack_json, f, indent=2)

    # Create empty concepts.kif
    concepts_kif_path = output_dir / 'concepts.kif'
    with open(concepts_kif_path, 'w') as f:
        f.write(f""";; {pack_id} Concept Pack
;; Version: 1.0.0
;; Created: {datetime.now().strftime('%Y-%m-%d')}
;; Description: {description}

;; Add SUMO concept definitions below
;; Example:
;; (subclass YourConcept ParentConcept)
;; (documentation YourConcept EnglishLanguage "Description of your concept.")

""")

    # Create README.md
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(f"""# {pack_id}

{description}

## Overview

(Add overview of the concepts in this pack)

## Concepts

(List the concepts added by this pack)

## Installation

```bash
python scripts/install_concept_pack.py concept_packs/{pack_id}/
```

## Usage

(Describe how to use the concepts)

## References

(Add references, papers, sources)

## License

{license}

## Authors

- {author}
""")

    # Create LICENSE file
    license_path = output_dir / 'LICENSE'
    if license == "MIT":
        with open(license_path, 'w') as f:
            f.write(f"""MIT License

Copyright (c) {datetime.now().year} {author}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""")

    print(f"✓ Created concept pack: {pack_id}")
    print(f"  Location: {output_dir}")
    print(f"  Files:")
    print(f"    - pack.json")
    print(f"    - concepts.kif (empty, ready to edit)")
    print(f"    - README.md")
    print(f"    - LICENSE ({license})")
    print(f"    - wordnet_patches/ (empty)")
    print(f"    - layer_entries/ (empty)")
    print()
    print("Next steps:")
    print(f"  1. Edit {output_dir}/concepts.kif to add SUMO concepts")
    print(f"  2. (Optional) Add WordNet patches to wordnet_patches/")
    print(f"  3. Run: python scripts/validate_concept_pack.py {pack_id}")
    print(f"  4. Run: python scripts/install_concept_pack.py concept_packs/{pack_id}/")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Create a new concept pack")
    parser.add_argument('pack_id', help='Unique pack identifier (e.g., ai-safety-v1)')
    parser.add_argument('--description', required=True, help='Brief pack description')
    parser.add_argument('--author', default='HatCat Team', help='Pack author')
    parser.add_argument('--license', default='MIT', help='License (default: MIT)')
    parser.add_argument('--output-dir', help='Output directory (default: concept_packs/{pack_id})')

    args = parser.parse_args()

    create_pack(
        pack_id=args.pack_id,
        description=args.description,
        author=args.author,
        license=args.license,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == '__main__':
    main()
