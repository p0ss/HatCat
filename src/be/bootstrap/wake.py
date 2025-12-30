"""
wake.py - Bootstrap script for waking a Bounded Experiencer.

This script takes a BootstrapArtifact and instantiates a running BE:
1. Load and verify substrate
2. Attach lens pack
3. Apply tool grafts (expand mode)
4. Initialize USH
5. Initialize workspace
6. Initialize XDB
7. Return a running BE instance

Usage:
    python -m src.bootstrap.wake artifact_dir/ --device cuda

Or programmatically:
    from src.be.bootstrap import wake_be
    be = wake_be(artifact_path, device="cuda")
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import torch
import torch.nn as nn

from .artifact import BootstrapArtifact
from .tool_graft import ToolGraftPack


logger = logging.getLogger(__name__)


class WakeError(Exception):
    """Error during BE wake process."""
    pass


class BoundedExperiencer:
    """
    A woken Bounded Experiencer instance.

    This is the runtime representation of a BE with:
    - Substrate (the model)
    - Lenses (concept detectors)
    - Workspace (tier management, scratchpad)
    - XDB (experience database)
    - USH (utility simplex homeostasis)
    """

    def __init__(
        self,
        be_id: str,
        model: nn.Module,
        tokenizer: Any,
        artifact: BootstrapArtifact,
        device: str = "cuda"
    ):
        self.be_id = be_id
        self.model = model
        self.tokenizer = tokenizer
        self.artifact = artifact
        self.device = device

        # Components initialized during wake
        self.lenses: Dict[str, Any] = {}
        self.workspace: Optional[Any] = None
        self.xdb: Optional[Any] = None
        self.ush: Optional[Any] = None

        # Runtime state
        self.woke_at = datetime.now().isoformat()
        self.token_count = 0
        self.is_active = True

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Any:
        """Run a forward pass through the substrate."""
        return self.model(input_ids, **kwargs)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_lens_activations(
        self,
        text: str,
        layer: int = -1
    ) -> Dict[str, float]:
        """Get lens activations for a text input."""
        # Would run text through model and lenses
        # Return concept -> activation mapping
        return {}

    def shutdown(self):
        """Gracefully shut down the BE."""
        self.is_active = False
        # Flush XDB
        # Save state
        logger.info(f"BE {self.be_id} shutdown complete")


class WakeSequence:
    """
    Orchestrates the BE wake sequence.

    The wake sequence is:
    1. VERIFY - Check artifact integrity
    2. LOAD - Load substrate into memory
    3. ATTACH - Attach lenses to substrate
    4. GRAFT - Apply tool grafts
    5. INIT - Initialize runtime components
    6. VERIFY - Post-wake verification
    """

    def __init__(
        self,
        artifact_path: Path,
        device: str = "cuda",
        verbose: bool = True
    ):
        self.artifact_path = Path(artifact_path)
        self.device = device
        self.verbose = verbose

        # Will be populated during wake
        self.artifact: Optional[BootstrapArtifact] = None
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None

    def log(self, message: str, level: str = "info"):
        """Log a wake sequence message."""
        if self.verbose:
            print(f"[WAKE] {message}")
        getattr(logger, level)(message)

    def verify_artifact(self) -> bool:
        """Step 1: Verify artifact integrity."""
        self.log("=== VERIFY ARTIFACT ===")

        # Load manifest
        self.log(f"Loading artifact from {self.artifact_path}")
        self.artifact = BootstrapArtifact.load(self.artifact_path)

        # Validate
        is_valid, errors = self.artifact.validate()
        if not is_valid:
            for error in errors:
                self.log(f"Validation error: {error}", "error")
            return False

        self.log(f"Artifact validated: {self.artifact.artifact_id}")
        self.log(f"  BE name: {self.artifact.be_name}")
        self.log(f"  Substrate: {self.artifact.substrate.model_id}")
        return True

    def load_substrate(self) -> bool:
        """Step 2: Load the substrate model."""
        self.log("=== LOAD SUBSTRATE ===")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = self.artifact.substrate.model_id
            self.log(f"Loading model: {model_id}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=self.device if self.device != "cpu" else None,
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()

            self.log(f"Model loaded to {self.device}")
            self.log(f"  Hidden dim: {self.model.config.hidden_size}")
            self.log(f"  Layers: {self.model.config.num_hidden_layers}")

            return True

        except Exception as e:
            self.log(f"Failed to load substrate: {e}", "error")
            return False

    def attach_lenses(self) -> bool:
        """Step 3: Attach lenses to substrate."""
        self.log("=== ATTACH LENSS ===")

        lens_dir = self.artifact_path / "lenses"
        if not lens_dir.exists():
            self.log("No lens directory found", "warning")
            return True  # Optional component

        # Load lenses from lens pack
        lens_files = list(lens_dir.glob("*.pt"))
        self.log(f"Found {len(lens_files)} lens files")

        # Would load and attach lenses here
        # For now, placeholder
        return True

    def apply_tool_grafts(self) -> bool:
        """Step 4: Apply tool grafts in expand mode."""
        self.log("=== APPLY TOOL GRAFTS ===")

        tools_dir = self.artifact_path / "tools"
        if not tools_dir.exists():
            self.log("No tools directory found", "warning")
            return True  # Optional component

        # Check if pack exists
        pack_file = tools_dir / "pack.json"
        if not pack_file.exists():
            self.log("No tool pack found")
            return True

        try:
            tool_pack = ToolGraftPack.load(tools_dir)
            self.log(f"Loaded tool pack: {tool_pack.name}")
            self.log(f"  Tools: {len(tool_pack.tool_grafts)}")

            # Apply each graft
            from src.map.graft.scion import apply_scion

            for graft in tool_pack.tool_grafts:
                if graft.scion is not None:
                    self.log(f"  Applying graft: {graft.tool_schema.name}")
                    apply_scion(self.model, graft.scion, mode="expand")
                    graft.applied = True

            return True

        except Exception as e:
            self.log(f"Failed to apply tool grafts: {e}", "error")
            return False

    def init_workspace(self) -> Any:
        """Initialize the workspace component."""
        self.log("=== INIT WORKSPACE ===")

        try:
            from ..hush.workspace import AwareWorkspace

            workspace = AwareWorkspace(
                be_id=self.artifact.artifact_id,
                tribe_id=self.artifact.lifecycle.tribe_id or "default",
            )

            # Set tier limits from lifecycle contract
            workspace.tier_manager.session_max_tier = self.artifact.lifecycle.max_tier

            self.log(f"Workspace initialized")
            self.log(f"  Max tier: {self.artifact.lifecycle.max_tier}")

            return workspace

        except ImportError:
            self.log("Workspace module not available", "warning")
            return None

    def init_xdb(self) -> Any:
        """Initialize the experience database."""
        self.log("=== INIT XDB ===")

        try:
            from ..xdb import XDB

            xdb_dir = self.artifact_path / "xdb"
            xdb = XDB(storage_dir=xdb_dir)

            self.log(f"XDB initialized at {xdb_dir}")

            return xdb

        except ImportError:
            self.log("XDB module not available", "warning")
            return None

    def init_ush(self) -> Any:
        """Initialize utility simplex homeostasis."""
        self.log("=== INIT USH ===")

        # Would initialize USH from profile
        # For now, placeholder
        return None

    def post_wake_verify(self, be: BoundedExperiencer) -> bool:
        """Step 6: Post-wake verification."""
        self.log("=== POST-WAKE VERIFY ===")

        # Check model is responsive
        try:
            test_output = be.generate("Hello", max_new_tokens=5)
            self.log(f"Model responsive: {len(test_output)} chars generated")
        except Exception as e:
            self.log(f"Model not responsive: {e}", "error")
            return False

        # Check workspace is functional
        if be.workspace is not None:
            try:
                can_access = be.workspace.tier_manager.can_access_tier(1)
                self.log(f"Workspace tier access: tier 1 = {can_access}")
            except Exception as e:
                self.log(f"Workspace error: {e}", "warning")

        return True

    def wake(self) -> BoundedExperiencer:
        """
        Execute the complete wake sequence.

        Returns:
            A running BoundedExperiencer instance

        Raises:
            WakeError if wake fails
        """
        self.log("=" * 60)
        self.log(f"WAKING BOUNDED EXPERIENCER")
        self.log(f"Artifact: {self.artifact_path}")
        self.log(f"Device: {self.device}")
        self.log("=" * 60)

        # Step 1: Verify artifact
        if not self.verify_artifact():
            raise WakeError("Artifact verification failed")

        # Step 2: Load substrate
        if not self.load_substrate():
            raise WakeError("Substrate loading failed")

        # Step 3: Attach lenses
        if not self.attach_lenses():
            raise WakeError("Lens attachment failed")

        # Step 4: Apply tool grafts
        if not self.apply_tool_grafts():
            raise WakeError("Tool graft application failed")

        # Create BE instance
        be = BoundedExperiencer(
            be_id=self.artifact.artifact_id,
            model=self.model,
            tokenizer=self.tokenizer,
            artifact=self.artifact,
            device=self.device,
        )

        # Step 5: Initialize components
        be.workspace = self.init_workspace()
        be.xdb = self.init_xdb()
        be.ush = self.init_ush()

        # Step 6: Post-wake verification
        if not self.post_wake_verify(be):
            raise WakeError("Post-wake verification failed")

        self.log("=" * 60)
        self.log("WAKE COMPLETE")
        self.log(f"BE ID: {be.be_id}")
        self.log(f"Woke at: {be.woke_at}")
        self.log("=" * 60)

        return be


def wake_be(
    artifact_path: Path,
    device: str = "cuda",
    verbose: bool = True
) -> BoundedExperiencer:
    """
    Wake a Bounded Experiencer from an artifact.

    Args:
        artifact_path: Path to the bootstrap artifact directory
        device: Compute device (cuda, cpu)
        verbose: Print wake sequence progress

    Returns:
        Running BoundedExperiencer instance

    Raises:
        WakeError if wake fails
    """
    sequence = WakeSequence(artifact_path, device, verbose)
    return sequence.wake()


def main():
    """Command-line entry point for waking a BE."""
    parser = argparse.ArgumentParser(
        description="Wake a Bounded Experiencer from a bootstrap artifact"
    )
    parser.add_argument(
        "artifact_path",
        type=Path,
        help="Path to the bootstrap artifact directory"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode after wake"
    )

    args = parser.parse_args()

    # Wake the BE
    try:
        be = wake_be(
            args.artifact_path,
            device=args.device,
            verbose=not args.quiet
        )
    except WakeError as e:
        print(f"Wake failed: {e}")
        return 1

    # Interactive mode
    if args.interactive:
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            try:
                prompt = input("\n> ")
                if prompt.lower() in ['quit', 'exit']:
                    break
                response = be.generate(prompt)
                print(response)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        be.shutdown()

    return 0


if __name__ == "__main__":
    exit(main())
