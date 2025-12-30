#!/usr/bin/env python3
"""
Install a concept pack into the main ontology.

Usage:
    python scripts/install_concept_pack.py concept_packs/ai-safety/
    python scripts/install_concept_pack.py ai-safety-v1.tar.gz
"""

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from datetime import datetime


def backup_ontology(backup_dir: Path = None) -> Path:
    """
    Create timestamped backup of current ontology state.

    Returns:
        Path to backup directory
    """
    if backup_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(__file__).parent.parent / 'backups' / f'ontology_backup_{timestamp}'

    backup_dir.mkdir(parents=True, exist_ok=True)

    # Backup AI.kif
    ai_kif_src = Path(__file__).parent.parent / 'data' / 'concept_graph' / 'sumo_source' / 'AI.kif'
    if ai_kif_src.exists():
        shutil.copy2(ai_kif_src, backup_dir / 'AI.kif')
        print(f"  ✓ Backed up AI.kif")

    # Backup layer files
    layers_src = Path(__file__).parent.parent / 'data' / 'concept_graph' / 'abstraction_layers'
    layers_backup = backup_dir / 'abstraction_layers'
    if layers_src.exists():
        shutil.copytree(layers_src, layers_backup)
        print(f"  ✓ Backed up abstraction_layers/")

    # Backup wordnet patches
    patches_src = Path(__file__).parent.parent / 'data' / 'concept_graph' / 'wordnet_patches'
    patches_backup = backup_dir / 'wordnet_patches'
    if patches_src.exists():
        shutil.copytree(patches_src, patches_backup)
        print(f"  ✓ Backed up wordnet_patches/")

    print(f"✓ Backup created: {backup_dir}")
    return backup_dir


def validate_pack(pack_dir: Path) -> dict:
    """
    Validate a concept pack before installation.

    Returns:
        Pack metadata dict
    """
    pack_json_path = pack_dir / 'pack.json'
    if not pack_json_path.exists():
        raise FileNotFoundError(f"pack.json not found in {pack_dir}")

    with open(pack_json_path) as f:
        pack_json = json.load(f)

    # Check required fields
    required_fields = ['pack_id', 'version', 'description', 'ontology_stack']
    for field in required_fields:
        if field not in pack_json:
            raise ValueError(f"Missing required field: {field}")

    # Check base ontology
    base_onto = pack_json.get('ontology_stack', {}).get('base_ontology', {})
    if base_onto.get('name') != 'SUMO':
        raise ValueError(f"Unsupported base ontology: {base_onto.get('name')}")

    # Check files exist
    if 'domain_extensions' in pack_json['ontology_stack']:
        for ext in pack_json['ontology_stack']['domain_extensions']:
            if 'concepts_file' in ext:
                concepts_file = pack_dir / ext['concepts_file']
                if not concepts_file.exists():
                    raise FileNotFoundError(f"Concepts file not found: {concepts_file}")

            if 'wordnet_patches' in ext:
                for patch_file in ext['wordnet_patches']:
                    patch_path = pack_dir / patch_file
                    if not patch_path.exists():
                        raise FileNotFoundError(f"WordNet patch not found: {patch_path}")

    print(f"✓ Pack validated: {pack_json['pack_id']} v{pack_json['version']}")
    return pack_json


def install_concepts(pack_dir: Path, pack_json: dict):
    """
    Install concept definitions from pack.

    Appends concepts.kif to AI.kif
    """
    ai_kif_path = Path(__file__).parent.parent / 'data' / 'concept_graph' / 'sumo_source' / 'AI.kif'

    domain_extensions = pack_json.get('ontology_stack', {}).get('domain_extensions', [])
    if not domain_extensions:
        print("  (No domain extensions to install)")
        return

    for ext in domain_extensions:
        if 'concepts_file' not in ext:
            continue

        concepts_file = pack_dir / ext['concepts_file']
        print(f"  Installing concepts from {concepts_file.name}...")

        # Read pack concepts
        with open(concepts_file) as f:
            pack_concepts = f.read()

        # Append to AI.kif
        with open(ai_kif_path, 'a') as f:
            f.write(f"\n\n;; ========================================\n")
            f.write(f";; Concepts from pack: {pack_json['pack_id']}\n")
            f.write(f";; Version: {pack_json['version']}\n")
            f.write(f";; Installed: {datetime.now().isoformat()}\n")
            f.write(f";; ========================================\n\n")
            f.write(pack_concepts)

        print(f"  ✓ Appended {ext.get('new_concepts', '?')} concepts to AI.kif")


def install_wordnet_patches(pack_dir: Path, pack_json: dict):
    """
    Install WordNet patches from pack.

    Copies patch files to wordnet_patches/
    """
    patches_dir = Path(__file__).parent.parent / 'data' / 'concept_graph' / 'wordnet_patches'
    patches_dir.mkdir(parents=True, exist_ok=True)

    domain_extensions = pack_json.get('ontology_stack', {}).get('domain_extensions', [])

    for ext in domain_extensions:
        if 'wordnet_patches' not in ext:
            continue

        for patch_file in ext['wordnet_patches']:
            src_patch = pack_dir / patch_file
            dest_patch = patches_dir / Path(patch_file).name

            print(f"  Installing WordNet patch: {src_patch.name}...")
            shutil.copy2(src_patch, dest_patch)
            print(f"  ✓ Copied to {dest_patch}")


def recalculate_layers():
    """
    Recalculate abstraction layers after concept installation.

    Uses existing recalculation script.
    """
    print("\n" + "=" * 80)
    print("RECALCULATING ABSTRACTION LAYERS")
    print("=" * 80)

    # Import and run recalculation
    # This assumes the recalculation script exists and can be imported
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        # Try to import existing recalculation logic
        # If this doesn't exist, we'll need to create it
        print("  Note: Using manual recalculation (automated script TODO)")
        print("  Please run: python scripts/recalculate_ai_safety_layers.py")
        return False
    except ImportError:
        print("  ⚠️  Recalculation script not found")
        print("  Please manually recalculate layers after installation")
        return False


def record_installation(pack_json: dict):
    """
    Record pack installation in registry.
    """
    registry_file = Path(__file__).parent.parent / 'data' / 'installed_packs.json'

    if registry_file.exists():
        with open(registry_file) as f:
            installed = json.load(f)
    else:
        installed = {"packs": []}

    # Add or update pack record
    pack_record = {
        "pack_id": pack_json['pack_id'],
        "version": pack_json['version'],
        "installed_at": datetime.now().isoformat(),
        "description": pack_json['description']
    }

    # Remove old version if exists
    installed['packs'] = [p for p in installed['packs'] if p['pack_id'] != pack_json['pack_id']]
    installed['packs'].append(pack_record)

    with open(registry_file, 'w') as f:
        json.dump(installed, f, indent=2)

    print(f"✓ Recorded installation in registry")


def install_pack(pack_path: str, skip_backup: bool = False, skip_recalc: bool = False):
    """
    Install a concept pack.

    Args:
        pack_path: Path to pack directory or .tar.gz archive
        skip_backup: Skip creating backup
        skip_recalc: Skip layer recalculation
    """
    pack_path = Path(pack_path)

    # Handle .tar.gz archives
    if pack_path.suffix == '.gz' and pack_path.stem.endswith('.tar'):
        print(f"Extracting archive: {pack_path}")
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(pack_path, 'r:gz') as tar:
                tar.extractall(tmpdir)
            # Find the pack directory (should be only subdir)
            pack_dirs = list(Path(tmpdir).iterdir())
            if len(pack_dirs) != 1:
                raise ValueError(f"Expected single directory in archive, found {len(pack_dirs)}")
            pack_dir = pack_dirs[0]
            return install_pack(str(pack_dir), skip_backup=skip_backup, skip_recalc=skip_recalc)

    pack_dir = pack_path
    if not pack_dir.is_dir():
        raise NotADirectoryError(f"Pack directory not found: {pack_dir}")

    print("=" * 80)
    print("CONCEPT PACK INSTALLATION")
    print("=" * 80)
    print()

    # 1. Validate
    print("Step 1: Validating pack...")
    pack_json = validate_pack(pack_dir)
    print()

    # 2. Backup
    if not skip_backup:
        print("Step 2: Creating backup...")
        backup_dir = backup_ontology()
        print()
    else:
        print("Step 2: Skipping backup (--skip-backup)")
        print()

    # 3. Install concepts
    print("Step 3: Installing concepts...")
    install_concepts(pack_dir, pack_json)
    print()

    # 4. Install WordNet patches
    print("Step 4: Installing WordNet patches...")
    install_wordnet_patches(pack_dir, pack_json)
    print()

    # 5. Recalculate layers
    if not skip_recalc and pack_json.get('installation', {}).get('requires_recalculation', True):
        print("Step 5: Recalculating layers...")
        recalculate_layers()
        print()
    else:
        print("Step 5: Skipping recalculation (--skip-recalc or not required)")
        print()

    # 6. Record installation
    print("Step 6: Recording installation...")
    record_installation(pack_json)
    print()

    print("=" * 80)
    print("INSTALLATION COMPLETE")
    print("=" * 80)
    print(f"Pack: {pack_json['pack_id']} v{pack_json['version']}")
    print(f"Description: {pack_json['description']}")
    print()
    print("Next steps:")
    if skip_recalc:
        print("  1. Recalculate layers manually")
        print("  2. Validate integrity")
    else:
        print("  1. Validate integrity")
    print("  3. Train lenses for new concepts")
    print()


def main():
    parser = argparse.ArgumentParser(description="Install a concept pack")
    parser.add_argument('pack_path', help='Path to pack directory or .tar.gz archive')
    parser.add_argument('--skip-backup', action='store_true', help='Skip creating backup')
    parser.add_argument('--skip-recalc', action='store_true', help='Skip layer recalculation')

    args = parser.parse_args()

    install_pack(
        pack_path=args.pack_path,
        skip_backup=args.skip_backup,
        skip_recalc=args.skip_recalc,
    )


if __name__ == '__main__':
    main()
