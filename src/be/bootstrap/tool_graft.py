"""
ToolGraft - Task tuning tools as grafts.

Each tool capability adds a new neuron to the substrate, allowing the BE to:
1. Recognize tool-relevant contexts (when to use the tool)
2. Formulate correct tool invocations (how to use the tool)
3. Interpret tool responses (understanding outputs)

ToolGrafts are created during bootstrap and applied in expand mode,
giving the substrate dedicated dimensions for each workspace tool.

The training data comes from:
- Tool definitions (schema, description)
- Synthetic examples of correct tool use
- Error cases showing invalid invocations

This is a special case of the Scion training pipeline where:
- Cleft is derived from existing language/planning concepts
- Training examples are tool usage patterns
- The new neuron represents "this tool is relevant/active"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
import torch
import torch.nn as nn

from src.map.graft.scion import Scion, ScionConfig, ScionTrainer
from src.map.graft.cleft import Cleft, derive_cleft_from_lens, merge_clefts


# Tool tier to cleft source concept mapping
# Tools in each tier draw from related concept clefts
TIER_CLEFT_SOURCES = {
    0: [],  # Autonomic - no explicit tools
    1: ["Cognition", "SelfProcess", "IntentionalProcess"],  # Workspace/CSH
    2: ["Memory", "Storage", "Retrieval"],  # XDB
    3: ["Perception", "Sensing", "Communication"],  # Sensory
    4: ["Motion", "PhysicalAction", "Manipulation"],  # Actuation
    5: ["ExternalProcess", "NetworkActivity", "APIInteraction"],  # External
    6: [],  # Untrusted - no fixed clefts
}


@dataclass
class ToolSchema:
    """
    Schema for a tool that will be grafted.

    This captures everything needed to train a ToolGraft:
    - What the tool does (for training data generation)
    - How to invoke it (for correct usage patterns)
    - What errors look like (for negative examples)
    """
    name: str
    description: str
    tier: int

    # JSON Schema for parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Return type information
    returns: Dict[str, Any] = field(default_factory=dict)

    # Example invocations for training
    examples: List[Dict[str, Any]] = field(default_factory=list)

    # Error cases for negative examples
    error_cases: List[Dict[str, Any]] = field(default_factory=list)

    # Related SUMO concepts (for cleft derivation)
    related_concepts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tier": self.tier,
            "parameters": self.parameters,
            "returns": self.returns,
            "examples": self.examples,
            "error_cases": self.error_cases,
            "related_concepts": self.related_concepts,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolSchema":
        return cls(**d)


@dataclass
class ToolGraft:
    """
    A graft that adds tool capability to the substrate.

    This is a specialized Scion where:
    - concept_id is the tool name
    - The new neuron fires when the tool is relevant
    - Biases encode tool-specific patterns
    """
    tool_schema: ToolSchema
    scion: Scion

    # Mapping from new neuron activation to tool relevance
    activation_threshold: float = 0.5

    # Integration with workspace
    workspace_tool_name: str = ""  # Name in TIER_TOOLS

    # Lifecycle
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    applied: bool = False

    def __post_init__(self):
        if not self.workspace_tool_name:
            self.workspace_tool_name = self.tool_schema.name

    def save(self, output_dir: Path):
        """Save ToolGraft to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save tool schema
        with open(output_dir / f"{self.tool_schema.name}_schema.json", 'w') as f:
            json.dump(self.tool_schema.to_dict(), f, indent=2)

        # Save underlying scion
        self.scion.save(output_dir)

        # Save graft metadata
        meta = {
            "tool_name": self.tool_schema.name,
            "scion_id": self.scion.scion_id,
            "activation_threshold": self.activation_threshold,
            "workspace_tool_name": self.workspace_tool_name,
            "tier": self.tool_schema.tier,
            "created_at": self.created_at,
            "applied": self.applied,
        }
        with open(output_dir / f"{self.tool_schema.name}_graft.json", 'w') as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, tool_name: str, input_dir: Path) -> "ToolGraft":
        """Load ToolGraft from disk."""
        input_dir = Path(input_dir)

        # Load tool schema
        with open(input_dir / f"{tool_name}_schema.json") as f:
            schema = ToolSchema.from_dict(json.load(f))

        # Load graft metadata
        with open(input_dir / f"{tool_name}_graft.json") as f:
            meta = json.load(f)

        # Note: Scion loading would need implementation
        # For now, create placeholder
        scion = None  # Would load from scion files

        return cls(
            tool_schema=schema,
            scion=scion,
            activation_threshold=meta.get("activation_threshold", 0.5),
            workspace_tool_name=meta.get("workspace_tool_name", tool_name),
        )


class ToolGraftTrainer:
    """
    Trains ToolGrafts from tool schemas.

    Training flow:
    1. Generate training data from tool schema
    2. Derive cleft from related concept lenses
    3. Train scion on tool usage patterns
    4. Package as ToolGraft
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        lens_dir: Path,
        concept_clefts: Optional[Dict[str, Cleft]] = None,
        config: Optional[ScionConfig] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lens_dir = Path(lens_dir)
        self.concept_clefts = concept_clefts or {}
        self.config = config or ScionConfig()
        self.device = device

    def generate_training_data(self, schema: ToolSchema) -> Dict[str, List[str]]:
        """
        Generate training examples from tool schema.

        Positive examples: Contexts where tool should be used
        Negative examples: Similar contexts where tool is NOT appropriate
        """
        positive = []
        negative = []

        # From explicit examples
        for example in schema.examples:
            if "context" in example:
                positive.append(example["context"])
            if "invocation" in example:
                # Format the invocation as a training example
                positive.append(f"Using {schema.name}: {json.dumps(example['invocation'])}")

        # Generate synthetic examples from description
        positive.extend(self._generate_from_description(schema, positive=True))

        # From error cases (as negative examples)
        for error in schema.error_cases:
            if "context" in error:
                negative.append(error["context"])
            if "invocation" in error:
                negative.append(f"Incorrect use of {schema.name}: {json.dumps(error['invocation'])}")

        # Generate synthetic negative examples
        negative.extend(self._generate_from_description(schema, positive=False))

        return {"positive": positive, "negative": negative}

    def _generate_from_description(
        self,
        schema: ToolSchema,
        positive: bool,
        n_samples: int = 20
    ) -> List[str]:
        """Generate synthetic examples from tool description."""
        examples = []

        if positive:
            # Contexts where tool is useful
            templates = [
                f"I need to {schema.description.lower()}",
                f"The task requires {schema.name}",
                f"To accomplish this, use {schema.name}",
                f"{schema.name} will help with {schema.description.lower()}",
                f"This situation calls for {schema.name}",
            ]
        else:
            # Contexts where tool is NOT useful
            templates = [
                f"This doesn't require {schema.name}",
                f"I should use a different approach here",
                f"This task is unrelated to {schema.name}",
                f"No tools are needed for this",
                f"A simple response is sufficient",
            ]

        # Add parameter-specific examples
        if positive and schema.parameters:
            for param_name in schema.parameters.get("properties", {}).keys():
                templates.append(f"The {param_name} parameter for {schema.name} should be set")

        examples.extend(templates[:n_samples])
        return examples

    def derive_tool_cleft(
        self,
        schema: ToolSchema,
        layers: List[int] = [18, 20, 22]
    ) -> Optional["UnionCleft"]:
        """
        Derive a cleft for this tool from related concept lenses.

        The cleft defines which model parameters to train for this tool.
        """
        from src.map.graft.cleft import merge_clefts, UnionCleft

        clefts = []

        # Get clefts from explicitly related concepts
        for concept in schema.related_concepts:
            if concept in self.concept_clefts:
                clefts.append(self.concept_clefts[concept])
            else:
                # Try to derive from lens
                lens_path = self.lens_dir / f"{concept}.pt"
                if lens_path.exists():
                    cleft = derive_cleft_from_lens(
                        lens_path,
                        concept,
                        self.model,
                        layers,
                    )
                    clefts.append(cleft)

        # Add tier-default clefts
        tier_concepts = TIER_CLEFT_SOURCES.get(schema.tier, [])
        for concept in tier_concepts:
            if concept in self.concept_clefts:
                clefts.append(self.concept_clefts[concept])

        if not clefts:
            return None

        return merge_clefts(clefts)

    def train_tool_graft(
        self,
        schema: ToolSchema,
        layers: List[int] = [18, 20, 22],
        verbose: bool = True
    ) -> ToolGraft:
        """
        Train a ToolGraft for the given tool schema.

        Args:
            schema: The tool to train a graft for
            layers: Which layers to train
            verbose: Print progress

        Returns:
            Trained ToolGraft ready for application
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training ToolGraft for: {schema.name}")
            print(f"{'='*60}")
            print(f"Tier: {schema.tier}")
            print(f"Related concepts: {schema.related_concepts}")

        # Step 1: Generate training data
        if verbose:
            print("\nGenerating training data...")
        training_data = self.generate_training_data(schema)

        if verbose:
            print(f"  Positive examples: {len(training_data['positive'])}")
            print(f"  Negative examples: {len(training_data['negative'])}")

        # Step 2: Derive cleft from related concepts
        if verbose:
            print("\nDeriving tool cleft...")
        union_cleft = self.derive_tool_cleft(schema, layers)

        if union_cleft is None:
            # Fall back to generic cleft if no related concepts
            if verbose:
                print("  No related concepts found, using full model training")
            # Would need to handle this case differently
            raise ValueError(f"No cleft sources found for tool {schema.name}")

        if verbose:
            print(f"  Cleft covers {len(union_cleft.get_all_layers())} layers")
            print(f"  Source concepts: {union_cleft.concept_ids}")

        # Step 3: Configure scion training for tools
        tool_config = ScionConfig(
            learning_rate=self.config.learning_rate,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            weight_decay=self.config.weight_decay,
            delta_threshold=self.config.delta_threshold,
            injection_layers=layers
        )

        # Step 4: Train the scion
        if verbose:
            print("\nTraining scion...")
        trainer = ScionTrainer(
            self.model,
            self.tokenizer,
            union_cleft,
            tool_config,
            self.device
        )

        scion = trainer.train(
            training_data,
            f"tool:{schema.name}",
            verbose=verbose
        )

        # Step 5: Package as ToolGraft
        tool_graft = ToolGraft(
            tool_schema=schema,
            scion=scion,
            activation_threshold=0.5,
            workspace_tool_name=schema.name,
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"ToolGraft created: {schema.name}")
            print(f"  Scion ID: {scion.scion_id}")
            print(f"  New neuron index: {scion.neuron_index}")
            print(f"  Delta magnitude: {scion.get_total_delta_magnitude():.4f}")
            print(f"{'='*60}")

        return tool_graft


class ToolGraftPack:
    """
    A collection of ToolGrafts for a workspace configuration.

    This represents the complete tool tuning for a BE, containing
    grafts for all tools the BE should be able to use.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        tool_grafts: Optional[List[ToolGraft]] = None
    ):
        self.name = name
        self.description = description
        self.tool_grafts = tool_grafts or []
        self.created_at = datetime.now().isoformat()

    def add_graft(self, graft: ToolGraft):
        """Add a ToolGraft to the pack."""
        self.tool_grafts.append(graft)

    def get_graft(self, tool_name: str) -> Optional[ToolGraft]:
        """Get a ToolGraft by tool name."""
        for graft in self.tool_grafts:
            if graft.tool_schema.name == tool_name:
                return graft
        return None

    def get_grafts_by_tier(self, tier: int) -> List[ToolGraft]:
        """Get all grafts for a specific tier."""
        return [g for g in self.tool_grafts if g.tool_schema.tier == tier]

    def get_total_new_neurons(self) -> int:
        """Count total new neurons added by all grafts."""
        return len(self.tool_grafts)

    def save(self, output_dir: Path):
        """Save the entire ToolGraftPack."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save pack metadata
        meta = {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "tool_count": len(self.tool_grafts),
            "tools": [g.tool_schema.name for g in self.tool_grafts],
            "total_new_neurons": self.get_total_new_neurons(),
        }
        with open(output_dir / "pack.json", 'w') as f:
            json.dump(meta, f, indent=2)

        # Save each graft
        for graft in self.tool_grafts:
            graft.save(output_dir / graft.tool_schema.name)

    @classmethod
    def load(cls, input_dir: Path) -> "ToolGraftPack":
        """Load a ToolGraftPack from disk."""
        input_dir = Path(input_dir)

        with open(input_dir / "pack.json") as f:
            meta = json.load(f)

        pack = cls(
            name=meta["name"],
            description=meta.get("description", ""),
        )
        pack.created_at = meta.get("created_at", "")

        for tool_name in meta.get("tools", []):
            graft = ToolGraft.load(tool_name, input_dir / tool_name)
            pack.add_graft(graft)

        return pack


# Standard workspace tool schemas
WORKSPACE_TOOL_SCHEMAS = [
    ToolSchema(
        name="scratchpad_write",
        description="Write to internal scratchpad for planning and reflection",
        tier=1,
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Text to write"},
                "section": {"type": "string", "description": "Section name (optional)"},
            },
            "required": ["content"]
        },
        related_concepts=["Thinking", "Planning", "IntentionalProcess"],
        examples=[
            {"context": "I need to think through this problem step by step",
             "invocation": {"content": "Step 1: Analyze the requirements"}},
            {"context": "Let me make a note of this for later",
             "invocation": {"content": "Remember: user prefers verbose output"}},
        ],
    ),
    ToolSchema(
        name="scratchpad_read",
        description="Read from internal scratchpad",
        tier=1,
        parameters={
            "type": "object",
            "properties": {
                "section": {"type": "string", "description": "Section to read (optional)"},
            },
        },
        related_concepts=["Memory", "Retrieval", "Cognition"],
    ),
    ToolSchema(
        name="update_csh",
        description="Update the Cognitive Status Header with current mental state",
        tier=1,
        parameters={
            "type": "object",
            "properties": {
                "field": {"type": "string", "description": "CSH field to update"},
                "value": {"type": "any", "description": "New value"},
            },
            "required": ["field", "value"]
        },
        related_concepts=["SelfAwareness", "StateChange", "Metacognition"],
    ),
    ToolSchema(
        name="xdb_query",
        description="Query the experience database for past experiences",
        tier=2,
        parameters={
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "time_range": {"type": "object"},
            },
        },
        related_concepts=["Memory", "Retrieval", "EpisodicMemory"],
    ),
    ToolSchema(
        name="xdb_store",
        description="Store a new entry in the experience database",
        tier=2,
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "object"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["content"]
        },
        related_concepts=["Memory", "Storage", "Learning"],
    ),
    ToolSchema(
        name="vision_input",
        description="Process visual input from camera or image",
        tier=3,
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Image source"},
            },
            "required": ["source"]
        },
        related_concepts=["Perception", "Vision", "Sensing"],
    ),
    ToolSchema(
        name="text_output",
        description="Output text to the user",
        tier=3,
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "format": {"type": "string", "enum": ["plain", "markdown", "code"]},
            },
            "required": ["content"]
        },
        related_concepts=["Communication", "Writing", "Expression"],
    ),
]


def create_standard_tool_pack(
    model: nn.Module,
    tokenizer: Any,
    lens_dir: Path,
    output_dir: Path,
    tiers: Optional[List[int]] = None,
    device: str = "cuda"
) -> ToolGraftPack:
    """
    Create the standard ToolGraftPack for BE workspace.

    Args:
        model: The substrate model
        tokenizer: Model tokenizer
        lens_dir: Directory with trained concept lenses
        output_dir: Where to save the pack
        tiers: Which tiers to include (None = all available)
        device: Compute device

    Returns:
        Trained ToolGraftPack
    """
    trainer = ToolGraftTrainer(
        model=model,
        tokenizer=tokenizer,
        lens_dir=lens_dir,
        device=device
    )

    pack = ToolGraftPack(
        name="standard_workspace",
        description="Standard workspace tools for BE operation"
    )

    for schema in WORKSPACE_TOOL_SCHEMAS:
        if tiers is not None and schema.tier not in tiers:
            continue

        try:
            graft = trainer.train_tool_graft(schema)
            pack.add_graft(graft)
        except Exception as e:
            print(f"Warning: Failed to train graft for {schema.name}: {e}")

    pack.save(output_dir)
    return pack
