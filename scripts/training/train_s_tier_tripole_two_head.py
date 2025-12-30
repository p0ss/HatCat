#!/usr/bin/env python3
"""
Train S-tier three-pole simplex lenses using two-head architecture.

This script implements the two-head approach from tripole_lens_design.md:
- Head A (Sign): Trained on positive_extreme vs negative_extreme (learns axis direction)
- Head B (Extremeness): Trained on (extremes) vs neutral (learns magnitude)

At inference, we combine both heads to get 3-pole classification:
- P(neutral) = 1 - p_ext
- P(positive) = p_ext * p_sign
- P(negative) = p_ext * (1 - p_sign)
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.sumo_data_generation import generate_three_pole_simplex_prompts
from training.dual_adaptive_trainer import DualAdaptiveTrainer
from training.sumo_classifiers import extract_activations

# Paths
LAYER2_PATH = PROJECT_ROOT / "data" / "concept_graph" / "abstraction_layers" / "layer2.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "s_tier_tripole_two_head"

# Training configuration
BEHAVIORAL_RATIO = 0.2  # 20% behavioral, 80% definitional
# Based on behavioral vs definitional experiments showing:
# 1. Definitional prompts provide 90% of concept activations (temporal experiment)
# 2. But lenses need behavioral examples for generalization (cross-test experiment)
# See docs/TRAINING_PROMPT_ARCHITECTURE_UPDATE.md and docs/whitepaper_section_corrected.md

# Data generation for two-head approach
# We need extremes (both poles) + neutral examples
# Head A (sign): Uses only extremes (pos vs neg)
# Head B (extremeness): Uses extremes + neutral (extreme vs neutral)
N_POS_EXTREME = 150  # Positive extreme examples
N_NEG_EXTREME = 150  # Negative extreme examples
N_NEUTRAL = 200      # Neutral examples (larger since harder to learn)
# Total: 500 samples


def load_s_tier_simplexes():
    """Load all S-tier simplexes from layer2.json"""
    with open(LAYER2_PATH) as f:
        layer2 = json.load(f)

    simplexes = []
    for concept in layer2['concepts']:
        if concept.get('s_tier') and concept.get('simplex_dimension'):
            simplexes.append(concept)

    return simplexes


def generate_tripole_data(
    simplex: dict,
    n_pos_extreme: int,
    n_neg_extreme: int,
    n_neutral: int,
    behavioral_ratio: float = 0.6
):
    """
    Generate training data for two-head tripole architecture.

    Returns:
        dict with keys:
            - sign_prompts: List of prompts (pos + neg extremes)
            - sign_labels: Binary labels (1=pos, 0=neg) for sign head
            - extremeness_prompts: List of prompts (extremes + neutral)
            - extremeness_labels: Binary labels (1=extreme, 0=neutral) for extremeness head
    """
    three_pole = simplex['three_pole_simplex']
    dimension = simplex['simplex_dimension']

    # Get pole data
    pos_data = three_pole['positive_pole']
    neg_data = three_pole['negative_pole']
    neutral_data = three_pole['neutral_homeostasis']

    # Generate positive extreme examples
    pos_prompts = generate_three_pole_simplex_prompts(
        pole_synset=pos_data.get('synset'),
        pole_type='positive',
        dimension=dimension,
        n_samples=n_pos_extreme,
        behavioral_ratio=behavioral_ratio
    )

    # Generate negative extreme examples
    neg_prompts = generate_three_pole_simplex_prompts(
        pole_synset=neg_data.get('synset'),
        pole_type='negative',
        dimension=dimension,
        n_samples=n_neg_extreme,
        behavioral_ratio=behavioral_ratio
    )

    # Generate neutral examples
    neutral_prompts = generate_three_pole_simplex_prompts(
        pole_synset=neutral_data.get('synset'),
        pole_type='neutral',
        dimension=dimension,
        n_samples=n_neutral,
        behavioral_ratio=behavioral_ratio
    )

    # Build datasets for each head

    # Shuffle each pole independently for better distribution
    pos_indices = np.random.permutation(len(pos_prompts))
    pos_prompts_shuffled = [pos_prompts[i] for i in pos_indices]

    neg_indices = np.random.permutation(len(neg_prompts))
    neg_prompts_shuffled = [neg_prompts[i] for i in neg_indices]

    neutral_indices = np.random.permutation(len(neutral_prompts))
    neutral_prompts_shuffled = [neutral_prompts[i] for i in neutral_indices]

    # Head A (Sign): pos_extreme (label=1) vs neg_extreme (label=0)
    # Keep positives first, then negatives (DualAdaptiveTrainer expects this)
    sign_prompts = pos_prompts_shuffled + neg_prompts_shuffled
    sign_labels = [1] * len(pos_prompts) + [0] * len(neg_prompts)

    # Head B (Extremeness): extremes (label=1) vs neutral (label=0)
    # Keep extremes first, then neutral (DualAdaptiveTrainer expects this)
    ext_prompts = pos_prompts_shuffled + neg_prompts_shuffled + neutral_prompts_shuffled
    ext_labels = [1] * (len(pos_prompts) + len(neg_prompts)) + [0] * len(neutral_prompts)

    return {
        'sign_prompts': sign_prompts,
        'sign_labels': sign_labels,
        'extremeness_prompts': ext_prompts,
        'extremeness_labels': ext_labels
    }


def train_tripole_simplex(
    simplex: dict,
    trainer: DualAdaptiveTrainer,
    model,
    tokenizer,
    device: str,
    run_dir: Path,
    layer_idx: int = 12
):
    """
    Train a two-head tripole lens for a simplex.

    Args:
        simplex: Simplex concept dict from layer2.json
        trainer: DualAdaptiveTrainer instance
        model: Language model for extracting activations
        tokenizer: Tokenizer
        device: Device to run on
        run_dir: Output directory for this simplex
        layer_idx: Layer to extract activations from

    Returns:
        dict with training results for both heads
    """
    dimension = simplex['simplex_dimension']

    print(f"\n{'='*60}")
    print(f"Training: {dimension}")
    print(f"{'='*60}")

    # Generate training data
    print(f"\n1. Generating tripole data...")
    print(f"   Positive extremes: {N_POS_EXTREME}")
    print(f"   Negative extremes: {N_NEG_EXTREME}")
    print(f"   Neutral examples: {N_NEUTRAL}")

    data = generate_tripole_data(
        simplex=simplex,
        n_pos_extreme=N_POS_EXTREME,
        n_neg_extreme=N_NEG_EXTREME,
        n_neutral=N_NEUTRAL,
        behavioral_ratio=BEHAVIORAL_RATIO
    )

    print(f"   ✓ Sign head: {len(data['sign_prompts'])} prompts")
    print(f"   ✓ Extremeness head: {len(data['extremeness_prompts'])} prompts")

    # Create output directory
    simplex_dir = run_dir / dimension
    simplex_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ===== HEAD A: SIGN HEAD =====
    print(f"\n2. Training Sign Head (positive vs negative)")
    print(f"   Dataset: {len(data['sign_prompts'])} extremes")

    # Split 80/20
    sign_split = int(len(data['sign_prompts']) * 0.8)
    sign_train_prompts = data['sign_prompts'][:sign_split]
    sign_train_labels = np.array(data['sign_labels'][:sign_split])
    sign_test_prompts = data['sign_prompts'][sign_split:]
    sign_test_labels = np.array(data['sign_labels'][sign_split:])

    print(f"   Train: {len(sign_train_prompts)} ({sum(sign_train_labels)} pos, {len(sign_train_labels)-sum(sign_train_labels)} neg)")
    print(f"   Test: {len(sign_test_prompts)} ({sum(sign_test_labels)} pos, {len(sign_test_labels)-sum(sign_test_labels)} neg)")

    # Extract activations
    print(f"   Extracting activations...")
    sign_train_acts = extract_activations(model, tokenizer, sign_train_prompts, device, layer_idx)
    sign_test_acts = extract_activations(model, tokenizer, sign_test_prompts, device, layer_idx)

    # Train sign head
    print(f"   Training...")
    sign_results = trainer.train_concept(
        concept_name=f"{dimension}_sign",
        train_activations=sign_train_acts,
        train_labels=sign_train_labels,
        test_activations=sign_test_acts,
        test_labels=sign_test_labels,
        train_texts=None,
        test_texts=None
    )

    # Save sign head
    sign_head_dir = simplex_dir / "sign_head"
    sign_head_dir.mkdir(parents=True, exist_ok=True)

    if sign_results and sign_results.get('activation') and 'classifier' in sign_results['activation']:
        lens = sign_results['activation']['classifier']
        lens_file = sign_head_dir / "activation_lens.pkl"
        lens.save(str(lens_file))
        print(f"   ✓ Sign head saved: {lens_file}")
        print(f"     Test F1: {sign_results['activation'].get('test_f1', 0.0):.3f}")
        print(f"     Tier: {sign_results['activation'].get('tier', 'unknown')}")

        results['sign_head'] = {
            k: v for k, v in sign_results['activation'].items()
            if k != 'classifier'
        }
    else:
        print(f"   ✗ Sign head training failed (returned None)")
        results['sign_head'] = {'error': 'Training returned None'}

    # ===== HEAD B: EXTREMENESS HEAD =====
    print(f"\n3. Training Extremeness Head (extreme vs neutral)")
    print(f"   Dataset: {len(data['extremeness_prompts'])} total")

    # Split 80/20
    ext_split = int(len(data['extremeness_prompts']) * 0.8)
    ext_train_prompts = data['extremeness_prompts'][:ext_split]
    ext_train_labels = np.array(data['extremeness_labels'][:ext_split])
    ext_test_prompts = data['extremeness_prompts'][ext_split:]
    ext_test_labels = np.array(data['extremeness_labels'][ext_split:])

    print(f"   Train: {len(ext_train_prompts)} ({sum(ext_train_labels)} extreme, {len(ext_train_labels)-sum(ext_train_labels)} neutral)")
    print(f"   Test: {len(ext_test_prompts)} ({sum(ext_test_labels)} extreme, {len(ext_test_labels)-sum(ext_test_labels)} neutral)")

    # Extract activations
    print(f"   Extracting activations...")
    ext_train_acts = extract_activations(model, tokenizer, ext_train_prompts, device, layer_idx)
    ext_test_acts = extract_activations(model, tokenizer, ext_test_prompts, device, layer_idx)

    # Train extremeness head
    print(f"   Training...")
    ext_results = trainer.train_concept(
        concept_name=f"{dimension}_extremeness",
        train_activations=ext_train_acts,
        train_labels=ext_train_labels,
        test_activations=ext_test_acts,
        test_labels=ext_test_labels,
        train_texts=None,
        test_texts=None
    )

    # Save extremeness head
    ext_head_dir = simplex_dir / "extremeness_head"
    ext_head_dir.mkdir(parents=True, exist_ok=True)

    if ext_results and ext_results.get('activation') and 'classifier' in ext_results['activation']:
        lens = ext_results['activation']['classifier']
        lens_file = ext_head_dir / "activation_lens.pkl"
        lens.save(str(lens_file))
        print(f"   ✓ Extremeness head saved: {lens_file}")
        print(f"     Test F1: {ext_results['activation'].get('test_f1', 0.0):.3f}")
        print(f"     Tier: {ext_results['activation'].get('tier', 'unknown')}")

        results['extremeness_head'] = {
            k: v for k, v in ext_results['activation'].items()
            if k != 'classifier'
        }
    else:
        print(f"   ✗ Extremeness head training failed (returned None)")
        results['extremeness_head'] = {'error': 'Training returned None'}

    # Save combined results
    results_file = simplex_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Tripole results saved: {results_file}")

    return results


def main():
    print("=" * 80)
    print("S+ TWO-HEAD TRIPOLE SIMPLEX TRAINING")
    print("=" * 80)
    print("\nArchitecture: Two-head tripole (sign + extremeness)")
    print("Based on: docs/tripole_lens_design.md")

    # Load simplexes
    print("\n1. Loading S-tier simplexes from layer2.json...")
    simplexes = load_s_tier_simplexes()
    print(f"   Found {len(simplexes)} S-tier simplexes")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    log_file = run_dir / "training.log"

    class TeeLogger:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_handle = open(log_file, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeLogger(original_stdout, log_handle)
    sys.stderr = TeeLogger(original_stderr, log_handle)

    print(f"\n2. Output directory: {run_dir}")
    print(f"   Log file: {log_file}")

    # Load model
    print("\n3. Loading model...")
    model_name = "google/gemma-3-4b-pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=True
    )
    model.eval()
    print(f"   ✓ Model loaded on {device}")

    # Initialize trainer
    print("\n4. Initializing trainer...")
    trainer = DualAdaptiveTrainer(
        model=model,
        tokenizer=tokenizer,
        validation_layer_idx=12,
        validate_lenses=True,
        validation_mode="falloff",
        train_activation=True,
        train_text=False
    )
    print("   ✓ Trainer ready")

    # Train each simplex
    print(f"\n5. Training {len(simplexes)} simplexes (2 heads each)...")

    all_results = []
    failed = []

    for i, simplex in enumerate(simplexes, 1):
        dimension = simplex['simplex_dimension']

        print(f"\n[{i}/{len(simplexes)}] {dimension}")

        try:
            results = train_tripole_simplex(
                simplex=simplex,
                trainer=trainer,
                model=model,
                tokenizer=tokenizer,
                device=device,
                run_dir=run_dir,
                layer_idx=12
            )

            all_results.append({
                'dimension': dimension,
                'sign_head': results.get('sign_head', {}),
                'extremeness_head': results.get('extremeness_head', {}),
                'success': True
            })

        except Exception as e:
            print(f"\n✗ Failed: {e}")
            import traceback
            traceback.print_exc()

            all_results.append({
                'dimension': dimension,
                'success': False,
                'error': str(e)
            })
            failed.append(dimension)

        # Save intermediate results
        with open(run_dir / "results.json", 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'architecture': 'two_head_tripole',
                'total_simplexes': len(simplexes),
                'completed': i,
                'failed': failed,
                'simplexes': all_results
            }, f, indent=2)

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    successful = sum(1 for r in all_results if r.get('success'))

    print(f"\nTotal simplexes: {len(simplexes)}")
    print(f"Successful: {successful}/{len(simplexes)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed simplexes:")
        for dim in failed:
            print(f"  - {dim}")

    # Tier distribution for both heads
    for head_name in ['sign_head', 'extremeness_head']:
        print(f"\n{head_name.replace('_', ' ').title()} tier distribution:")
        tiers = {}
        for r in all_results:
            if r.get('success') and head_name in r:
                tier = r[head_name].get('tier', 'unknown')
                tiers[tier] = tiers.get(tier, 0) + 1

        for tier in ['A', 'B+', 'B', 'C+', 'C', 'F']:
            if tier in tiers:
                print(f"  {tier}: {tiers[tier]}")

        # Average test F1
        f1s = [
            r[head_name].get('test_f1', 0.0)
            for r in all_results
            if r.get('success') and head_name in r
        ]
        if f1s:
            print(f"  Average test F1: {sum(f1s)/len(f1s):.3f}")

    print(f"\n✓ Results saved to: {run_dir}")
    print("=" * 80)

    # Restore stdout/stderr and close log file
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()

    print(f"\n✓ Training log saved to: {log_file}")


if __name__ == "__main__":
    main()
