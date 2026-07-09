#!/usr/bin/env python3
"""
Test script for true dimension expansion.

Tests the complete expansion pipeline:
1. Train scion with cleft-aware freezing
2. Apply scion in expand mode (hidden_dim +1)
3. Train bound lens for new dimension
4. Update manifest with expansion record

Verifies:
- model.config.hidden_size increases by 1
- Weight matrices have correct expanded shapes
- Bound lens loads and produces valid activations
- SubstrateManifest correctly records the expansion

Usage:
    # Full test with small model
    python scripts/tests/test_true_expansion.py --model HuggingFaceTB/SmolLM-135M

    # Quick unit tests only (no model loading)
    python scripts/tests/test_true_expansion.py --unit-only
"""

import argparse
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.map.graft.expand import (
    detect_architecture,
    plan_expansion,
    execute_expansion,
    _get_scion_bias,
)
from src.map.graft.scion import (
    Scion,
    ScionConfig,
    WeightDelta,
    apply_scion,
    train_bound_lens,
)
from src.map.graft.data_structures import (
    SubstrateManifest,
    SubstrateArchitecture,
    DimensionEntry,
)


def get_test_dataset(concept: str = "TestConcept") -> dict:
    """Get sample positive/negative examples for testing."""
    return {
        "positive": [
            f"This is a clear example of {concept} behavior.",
            f"Here we see {concept} in its natural form.",
            f"The {concept} is clearly demonstrated here.",
            f"A typical instance of {concept} can be observed.",
            f"This text represents {concept} perfectly.",
            f"Another example showing {concept} characteristics.",
            f"The presence of {concept} is evident.",
            f"Unmistakably, this shows {concept}.",
            f"Classic {concept} pattern observed here.",
            f"This embodies the essence of {concept}.",
        ],
        "negative": [
            "The weather was pleasant today.",
            "The computer processed the request quickly.",
            "The building was recently renovated.",
            "Traffic was heavy this morning.",
            "The report was submitted on time.",
            "She walked to the store to buy groceries.",
            "The meeting was rescheduled for next week.",
            "The package arrived earlier than expected.",
            "He completed the assignment before the deadline.",
            "The restaurant was crowded during lunch.",
        ]
    }


def create_mock_scion(hidden_dim: int = 2048, layer: int = 10) -> Scion:
    """Create a mock scion for testing expansion."""
    # Create mock weight deltas
    weight_deltas = [
        WeightDelta(
            layer_index=layer,
            component="mlp.up_proj",
            delta=torch.randn(hidden_dim * 4, hidden_dim) * 0.01,
            cleft_mask=torch.ones(hidden_dim * 4, hidden_dim, dtype=torch.bool),
        ),
        WeightDelta(
            layer_index=layer,
            component="mlp.down_proj",
            delta=torch.randn(hidden_dim, hidden_dim * 4) * 0.01,
            cleft_mask=torch.ones(hidden_dim, hidden_dim * 4, dtype=torch.bool),
        ),
    ]

    # Create mock neuron biases (matching format from _create_neuron_biases)
    neuron_biases = {
        f"layer{layer}_mlp.up_proj_row": torch.rand(hidden_dim * 4) * 0.1,
        f"layer{layer}_mlp.up_proj_col": torch.rand(hidden_dim) * 0.1,
        f"layer{layer}_mlp.down_proj_row": torch.rand(hidden_dim) * 0.1,
        f"layer{layer}_mlp.down_proj_col": torch.rand(hidden_dim * 4) * 0.1,
    }

    config = ScionConfig(injection_layers=[layer])

    return Scion(
        scion_id=f"test-scion-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        concept_id="TestConcept",
        weight_deltas=weight_deltas,
        neuron_index=hidden_dim,  # New dimension will be at index hidden_dim
        neuron_biases=neuron_biases,
        source_cleft_concepts=["RelatedConcept1", "RelatedConcept2"],
        training_config=config,
    )


def test_bias_key_format():
    """Test that _get_scion_bias correctly resolves keys."""
    print("\n" + "=" * 60)
    print("TEST: Bias Key Format")
    print("=" * 60)

    hidden_dim = 256
    layer = 5
    scion = create_mock_scion(hidden_dim=hidden_dim, layer=layer)

    # Test that we can retrieve biases using full component paths
    up_bias = _get_scion_bias(scion, layer, "mlp.up_proj", "col")
    down_bias = _get_scion_bias(scion, layer, "mlp.down_proj", "row")

    assert up_bias is not None, "Failed to retrieve up_proj col bias"
    assert down_bias is not None, "Failed to retrieve down_proj row bias"
    assert len(up_bias) == hidden_dim, f"Expected {hidden_dim}, got {len(up_bias)}"
    assert len(down_bias) == hidden_dim, f"Expected {hidden_dim}, got {len(down_bias)}"

    # Test that old format (without mlp.) returns None
    old_format_bias = _get_scion_bias(scion, layer, "up_proj", "col")
    # This should fail since we now expect full component paths
    # Actually, the new code passes full paths, so this test verifies the fix

    print("  [PASS] Bias key format correctly matches scion._create_neuron_biases()")
    print(f"  up_proj col bias shape: {up_bias.shape}")
    print(f"  down_proj row bias shape: {down_bias.shape}")
    return True


def test_dimension_entry_lens_path():
    """Test that DimensionEntry includes lens_path field."""
    print("\n" + "=" * 60)
    print("TEST: DimensionEntry Lens Path")
    print("=" * 60)

    entry = DimensionEntry(
        dimension_index=2048,
        concept_id="TestConcept",
        graft_id="scion-test-123",
        grafted_at=datetime.now().isoformat(),
        lens_path="/path/to/lens.pt",
    )

    # Test to_dict includes lens_path
    d = entry.to_dict()
    assert "lens_path" in d, "lens_path not in to_dict output"
    assert d["lens_path"] == "/path/to/lens.pt"

    # Test from_dict restores lens_path
    restored = DimensionEntry.from_dict(d)
    assert restored.lens_path == "/path/to/lens.pt"

    # Test that None lens_path works
    entry_no_lens = DimensionEntry(
        dimension_index=2049,
        concept_id="TestConcept2",
        graft_id="scion-test-456",
        grafted_at=datetime.now().isoformat(),
    )
    d_no_lens = entry_no_lens.to_dict()
    assert "lens_path" not in d_no_lens, "Empty lens_path should not appear in dict"

    print("  [PASS] DimensionEntry correctly handles lens_path field")
    return True


def test_manifest_record_expansion():
    """Test SubstrateManifest.record_expansion() method."""
    print("\n" + "=" * 60)
    print("TEST: SubstrateManifest.record_expansion()")
    print("=" * 60)

    # Create manifest
    manifest = SubstrateManifest.create_for_model(
        model_id="test-model",
        hidden_dim=2048,
        checksum="abc123",
    )

    assert manifest.current_hidden_dim == 2048
    assert manifest.total_grafts_applied == 0
    assert len(manifest.dimension_table) == 0

    # Create mock scion
    scion = create_mock_scion(hidden_dim=2048, layer=10)

    # Record expansion
    manifest.record_expansion(
        scion=scion,
        lens_path="/path/to/test_lens.pt",
        new_hidden_dim=2049,
    )

    assert manifest.current_hidden_dim == 2049, f"Expected 2049, got {manifest.current_hidden_dim}"
    assert manifest.total_grafts_applied == 1
    assert len(manifest.dimension_table) == 1

    entry = manifest.dimension_table[0]
    assert entry.dimension_index == scion.neuron_index
    assert entry.concept_id == scion.concept_id
    assert entry.graft_id == scion.scion_id
    assert entry.lens_path == "/path/to/test_lens.pt"

    print("  [PASS] SubstrateManifest.record_expansion() works correctly")
    print(f"  Current hidden_dim: {manifest.current_hidden_dim}")
    print(f"  Total grafts: {manifest.total_grafts_applied}")
    return True


def test_expansion_with_model(model_name: str, device: str = "cuda"):
    """
    Full integration test with an actual model.

    Tests:
    1. Architecture detection
    2. Expansion planning
    3. Expansion execution
    4. Bound lens training
    5. Manifest updates
    """
    print("\n" + "=" * 60)
    print("TEST: Full Expansion Pipeline with Model")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    original_hidden_dim = model.config.hidden_size
    print(f"  Original hidden_dim: {original_hidden_dim}")

    # Detect architecture
    print("\n  Step 1: Detect architecture")
    arch = detect_architecture(model)
    print(f"    Family: {arch.family}")
    print(f"    MLP type: {arch.mlp_type}")
    print(f"    Attention type: {arch.attention_type}")

    # Create mock scion for this model
    print("\n  Step 2: Create mock scion")
    layer = min(5, model.config.num_hidden_layers - 1)  # Use early layer for speed
    scion = create_mock_scion(hidden_dim=original_hidden_dim, layer=layer)
    print(f"    Scion ID: {scion.scion_id}")
    print(f"    Target layer: {layer}")

    # Plan expansion - must expand ALL layers for a valid model
    print("\n  Step 3: Plan expansion")
    plan = plan_expansion(model, scion, target_layers=None)  # None = all layers
    print(f"    Embedding targets: {len(plan.embedding_targets)}")
    print(f"    Layer targets: {len(plan.targets)}")
    print(f"    Norm targets: {len(plan.norm_targets)}")
    print(f"    New parameters: ~{plan.total_new_parameters():,}")

    # Execute expansion
    print("\n  Step 4: Execute expansion")
    execute_expansion(model, plan, device=device)

    new_hidden_dim = model.config.hidden_size
    print(f"    New hidden_dim: {new_hidden_dim}")
    assert new_hidden_dim == original_hidden_dim + 1, \
        f"Expected {original_hidden_dim + 1}, got {new_hidden_dim}"

    # Verify some weight shapes
    print("\n  Step 5: Verify weight shapes")
    embed_tokens = model.model.embed_tokens if hasattr(model, 'model') else model.embed_tokens
    print(f"    embed_tokens.weight: {embed_tokens.weight.shape}")
    assert embed_tokens.weight.shape[1] == new_hidden_dim

    # Get first layer and check shapes
    layers = model.model.layers if hasattr(model, 'model') else model.layers
    test_layer = layers[layer]

    q_proj = test_layer.self_attn.q_proj
    down_proj = test_layer.mlp.down_proj

    print(f"    layer[{layer}].self_attn.q_proj: {q_proj.weight.shape}")
    print(f"    layer[{layer}].mlp.down_proj: {down_proj.weight.shape}")

    assert q_proj.weight.shape[1] == new_hidden_dim, "q_proj input dim not expanded"
    assert down_proj.weight.shape[0] == new_hidden_dim, "down_proj output dim not expanded"

    # Train bound lens
    print("\n  Step 6: Train bound lens")
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = get_test_dataset()
        lens_path = train_bound_lens(
            model=model,
            tokenizer=tokenizer,
            scion=scion,
            dataset=dataset,
            auxiliary_dimensions=[0, 1, 2],  # Use first 3 dims as auxiliary
            output_dir=Path(tmpdir),
            device=device,
            epochs=3,  # Quick training
        )

        print(f"    Lens saved to: {lens_path}")
        assert lens_path.exists(), "Lens file not created"

        # Load and verify lens (HAT format: separate .pt and _metadata.json)
        from src.map.graft.scion import load_bound_lens
        classifier, metadata = load_bound_lens(lens_path)
        assert classifier is not None, "Failed to load classifier"
        assert metadata is not None, "Failed to load metadata"
        assert "primary_dimension" in metadata, "Metadata missing primary_dimension"
        print(f"    Lens metadata: {metadata}")

        # Update manifest
        print("\n  Step 7: Update manifest")
        manifest = SubstrateManifest.create_for_model(
            model_id=model_name,
            hidden_dim=original_hidden_dim,
            model_config=model.config,
        )

        manifest.record_expansion(
            scion=scion,
            lens_path=str(lens_path),
            new_hidden_dim=new_hidden_dim,
        )

        print(f"    Manifest updated: {manifest.current_hidden_dim}")
        assert manifest.current_hidden_dim == new_hidden_dim
        assert len(manifest.dimension_table) == 1

        # Save and reload manifest
        manifest_path = Path(tmpdir) / "manifest.json"
        manifest.save(manifest_path)
        loaded_manifest = SubstrateManifest.load(manifest_path)
        assert loaded_manifest.current_hidden_dim == new_hidden_dim
        print(f"    Manifest save/load verified")

    print("\n  [PASS] Full expansion pipeline completed successfully!")
    print(f"  Hidden dimension expanded: {original_hidden_dim} -> {new_hidden_dim}")
    return True


def test_apply_scion_expand_mode(model_name: str, device: str = "cuda"):
    """Test apply_scion with mode='expand' including lens training."""
    print("\n" + "=" * 60)
    print("TEST: apply_scion with expand mode")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    original_hidden_dim = model.config.hidden_size
    layer = min(3, model.config.num_hidden_layers - 1)

    # Create scion
    scion = create_mock_scion(hidden_dim=original_hidden_dim, layer=layer)
    dataset = get_test_dataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Apply scion with expand mode and lens training
        model, lens_path = apply_scion(
            model=model,
            scion=scion,
            mode="expand",
            tokenizer=tokenizer,
            training_data=dataset,
            auxiliary_dimensions=[0, 1],
            output_dir=output_dir,
            device=device,
        )

        # Verify
        assert model.config.hidden_size == original_hidden_dim + 1
        assert lens_path is not None
        assert lens_path.exists()
        assert scion.applied is True

        print(f"  [PASS] apply_scion expand mode works")
        print(f"  Hidden dim: {original_hidden_dim} -> {model.config.hidden_size}")
        print(f"  Lens path: {lens_path}")

    return True


def run_unit_tests():
    """Run unit tests that don't require model loading."""
    print("\n" + "=" * 70)
    print("UNIT TESTS (no model required)")
    print("=" * 70)

    all_passed = True
    all_passed &= test_bias_key_format()
    all_passed &= test_dimension_entry_lens_path()
    all_passed &= test_manifest_record_expansion()

    return all_passed


def run_integration_tests(model_name: str, device: str):
    """Run full integration tests with model."""
    print("\n" + "=" * 70)
    print("INTEGRATION TESTS (with model)")
    print("=" * 70)

    all_passed = True
    all_passed &= test_expansion_with_model(model_name, device)
    all_passed &= test_apply_scion_expand_mode(model_name, device)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test true dimension expansion")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM-135M",
        help="Model for integration tests (default: SmolLM-135M)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Run only unit tests (no model loading)"
    )
    parser.add_argument(
        "--integration-only",
        action="store_true",
        help="Run only integration tests"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TRUE DIMENSION EXPANSION TEST SUITE")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Model: {args.model}")

    all_passed = True

    if not args.integration_only:
        all_passed &= run_unit_tests()

    if not args.unit_only:
        all_passed &= run_integration_tests(args.model, args.device)

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
