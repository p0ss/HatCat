"""
Budding - Bridge between XDB experiences and grafting infrastructure.

This module provides the training data pipeline for scion creation:
1. Query XDB for experiences tagged with specific concepts
2. Build union clefts from related concept lenses
3. Prepare training datasets for ScionTrainer
4. Track scion training runs in XDB
5. Promote buds to scions

The flow:
    XDB experiences (tagged with concept activations)
              ↓
    BudTrainingData (positive/negative examples)
              ↓
    Cleft derivation (from lens weights)
              ↓
    ScionTrainer (trains on cleft regions)
              ↓
    Scion (ready for application)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import uuid4
import json
import logging

from .models import (
    BudStatus,
    TagType,
    TimestepRecord,
    Tag,
    Fidelity,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Training Data Structures
# ============================================================================

@dataclass
class BudTrainingData:
    """
    Training data extracted from XDB for scion training.

    Contains positive and negative examples, plus metadata about
    which concepts and experiences were used.
    """
    bud_tag_id: str
    concept_id: str

    # Training examples
    positive_texts: List[str] = field(default_factory=list)
    negative_texts: List[str] = field(default_factory=list)

    # Source tracking
    positive_timestep_ids: List[str] = field(default_factory=list)
    negative_timestep_ids: List[str] = field(default_factory=list)

    # Related concepts (for cleft derivation)
    related_concept_ids: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    min_activation_threshold: float = 0.7

    def to_trainer_format(self) -> Dict[str, List[str]]:
        """Convert to format expected by ScionTrainer."""
        return {
            "positive": self.positive_texts,
            "negative": self.negative_texts,
        }

    @property
    def is_sufficient(self) -> bool:
        """Check if we have enough examples for training."""
        return len(self.positive_texts) >= 10 and len(self.negative_texts) >= 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bud_tag_id": self.bud_tag_id,
            "concept_id": self.concept_id,
            "positive_count": len(self.positive_texts),
            "negative_count": len(self.negative_texts),
            "related_concept_ids": self.related_concept_ids,
            "min_activation_threshold": self.min_activation_threshold,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ScionTrainingRun:
    """
    Record of a scion training attempt.

    Tracks the full lifecycle from bud to scion.
    """
    id: str
    bud_tag_id: str
    concept_id: str

    # Status
    status: str = "pending"  # pending | training | completed | failed

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Training configuration
    training_config: Dict[str, Any] = field(default_factory=dict)

    # Evidence
    training_data_summary: Optional[Dict[str, Any]] = None
    training_window_ids: List[str] = field(default_factory=list)

    # Results
    scion_id: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bud_tag_id": self.bud_tag_id,
            "concept_id": self.concept_id,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "training_config": self.training_config,
            "training_data_summary": self.training_data_summary,
            "training_window_ids": self.training_window_ids,
            "scion_id": self.scion_id,
            "metrics": self.metrics,
            "error_message": self.error_message,
        }

    @staticmethod
    def generate_id() -> str:
        return f"scion-run-{uuid4().hex[:12]}"


# ============================================================================
# Budding Manager
# ============================================================================

class BuddingManager:
    """
    Manages the bud → scion promotion pipeline.

    Connects XDB (experience storage) with grafting (weight modification).
    """

    def __init__(
        self,
        xdb: 'XDB',
        lens_pack_path: Optional[Path] = None,
    ):
        """
        Initialize the budding manager.

        Args:
            xdb: The XDB instance for experience access
            lens_pack_path: Path to lens pack for cleft derivation
        """
        self.xdb = xdb
        self.lens_pack_path = Path(lens_pack_path) if lens_pack_path else None

        # Track training runs
        self.training_runs: Dict[str, ScionTrainingRun] = {}

    # =========================================================================
    # Training Data Extraction
    # =========================================================================

    def get_training_data(
        self,
        bud_tag_id: str,
        *,
        min_activation: float = 0.7,
        max_positive: int = 500,
        max_negative: int = 500,
        negative_strategy: str = "low_activation",
    ) -> BudTrainingData:
        """
        Extract training data for a bud from XDB.

        Positive examples: Timesteps where the bud's concepts fired strongly
        Negative examples: Timesteps where they didn't (various strategies)

        Args:
            bud_tag_id: The bud tag to get training data for
            min_activation: Minimum activation to count as positive
            max_positive: Maximum positive examples to collect
            max_negative: Maximum negative examples to collect
            negative_strategy: How to select negatives
                - "low_activation": Same concepts but low activation
                - "different_session": Random from different sessions
                - "sibling_concepts": High activation on sibling concepts

        Returns:
            BudTrainingData ready for scion training
        """
        # Get the bud tag
        bud = self.xdb.tag_index.get_tag(bud_tag_id)
        if not bud:
            raise ValueError(f"Bud tag not found: {bud_tag_id}")

        if bud.tag_type != TagType.BUD:
            raise ValueError(f"Tag is not a bud: {bud_tag_id}")

        concept_id = bud.concept_id or bud.name

        # Get positive examples (timesteps tagged with this bud)
        positive_records = self.xdb.get_bud_examples(bud_tag_id)

        # Also query by concept activation if we have concept_id
        if bud.concept_id:
            concept_records = self.xdb.recall_by_concept(
                bud.concept_id,
                min_activation=min_activation,
                limit=max_positive,
            )
            # Merge, avoiding duplicates
            seen_ids = {r.id for r in positive_records}
            for r in concept_records:
                if r.id not in seen_ids:
                    positive_records.append(r)
                    seen_ids.add(r.id)

        # Limit positives
        positive_records = positive_records[:max_positive]

        # Get negative examples based on strategy
        negative_records = self._get_negative_examples(
            concept_id=concept_id,
            positive_ids={r.id for r in positive_records},
            strategy=negative_strategy,
            max_count=max_negative,
        )

        # Get related concepts for cleft derivation
        related = self._get_related_concepts(concept_id)

        # Build training data
        training_data = BudTrainingData(
            bud_tag_id=bud_tag_id,
            concept_id=concept_id,
            positive_texts=[r.content for r in positive_records if r.content],
            negative_texts=[r.content for r in negative_records if r.content],
            positive_timestep_ids=[r.id for r in positive_records],
            negative_timestep_ids=[r.id for r in negative_records],
            related_concept_ids=related,
            min_activation_threshold=min_activation,
        )

        logger.info(
            f"Extracted training data for {bud_tag_id}: "
            f"{len(training_data.positive_texts)} positive, "
            f"{len(training_data.negative_texts)} negative"
        )

        return training_data

    def _get_negative_examples(
        self,
        concept_id: str,
        positive_ids: Set[str],
        strategy: str,
        max_count: int,
    ) -> List[TimestepRecord]:
        """Get negative examples using specified strategy."""
        negatives = []

        if strategy == "low_activation":
            # Get timesteps where concept activation was low
            # This requires querying with concept activation < threshold
            # For now, get recent timesteps not in positives
            recent = self.xdb.recall_recent(n=max_count * 3)
            for r in recent:
                if r.id not in positive_ids:
                    # Check if concept activation is low or absent
                    activation = r.concept_activations.get(concept_id, 0.0)
                    if activation < 0.3:  # Low threshold
                        negatives.append(r)
                        if len(negatives) >= max_count:
                            break

        elif strategy == "different_xdb":
            # Get random timesteps from different XDBs
            # For now, just get recent with different xdb_id
            current_xdb = self.xdb.xdb_id
            recent = self.xdb.recall_recent(n=max_count * 3)
            for r in recent:
                if r.id not in positive_ids and r.xdb_id != current_xdb:
                    negatives.append(r)
                    if len(negatives) >= max_count:
                        break

        elif strategy == "sibling_concepts":
            # Get timesteps with high activation on related but different concepts
            related = self._get_related_concepts(concept_id)
            if related:
                for sibling_id in related[:3]:  # Try first 3 siblings
                    sibling_records = self.xdb.recall_by_concept(
                        sibling_id,
                        min_activation=0.7,
                        limit=max_count // 3,
                    )
                    for r in sibling_records:
                        if r.id not in positive_ids:
                            negatives.append(r)
                            if len(negatives) >= max_count:
                                break
                    if len(negatives) >= max_count:
                        break

        # Fallback: just use recent non-positive timesteps
        if len(negatives) < max_count // 2:
            recent = self.xdb.recall_recent(n=max_count * 2)
            for r in recent:
                if r.id not in positive_ids and r.id not in {n.id for n in negatives}:
                    negatives.append(r)
                    if len(negatives) >= max_count:
                        break

        return negatives[:max_count]

    def _get_related_concepts(self, concept_id: str) -> List[str]:
        """Get related concept IDs from the concept hierarchy."""
        related = []

        # Try to get parents/children from tag index
        try:
            parents = self.xdb.tag_index.get_concept_parents(concept_id)
            related.extend([t.concept_id for t in parents if t.concept_id])

            children = self.xdb.tag_index.get_concept_children(concept_id)
            related.extend([t.concept_id for t in children if t.concept_id])
        except Exception as e:
            logger.debug(f"Could not get related concepts for {concept_id}: {e}")

        return related

    # =========================================================================
    # Scion Training
    # =========================================================================

    def prepare_scion_training(
        self,
        bud_tag_id: str,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> ScionTrainingRun:
        """
        Prepare a scion training run.

        This:
        1. Extracts training data from XDB
        2. Pins the training data as WARM
        3. Creates a training run record
        4. Updates bud status to TRAINING

        Returns the training run record (not yet started).
        """
        # Get training data
        training_data = self.get_training_data(bud_tag_id)

        if not training_data.is_sufficient:
            raise ValueError(
                f"Insufficient training data: "
                f"{len(training_data.positive_texts)} positive, "
                f"{len(training_data.negative_texts)} negative "
                f"(need at least 10 each)"
            )

        # Pin training data as WARM
        all_ids = training_data.positive_timestep_ids + training_data.negative_timestep_ids
        pinned = self.xdb.pin_for_training(all_ids, reason=f"bud:{bud_tag_id}")
        logger.info(f"Pinned {pinned} timesteps for training")

        # Create training run
        run = ScionTrainingRun(
            id=ScionTrainingRun.generate_id(),
            bud_tag_id=bud_tag_id,
            concept_id=training_data.concept_id,
            status="pending",
            training_config=training_config or {},
            training_data_summary=training_data.to_dict(),
            training_window_ids=all_ids,
        )

        self.training_runs[run.id] = run

        # Update bud status
        self.xdb.tag_index.update_bud_status(bud_tag_id, BudStatus.TRAINING)

        return run

    def run_scion_training(
        self,
        run_id: str,
        model: 'nn.Module',
        tokenizer: Any,
        *,
        layers: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> 'Scion':
        """
        Execute scion training for a prepared run.

        This:
        1. Derives clefts from related concept lenses
        2. Creates union cleft
        3. Runs ScionTrainer
        4. Updates run record with results

        Args:
            run_id: The training run ID
            model: The substrate model
            tokenizer: The tokenizer
            layers: Which layers to inject (default: middle layers)
            verbose: Print training progress

        Returns:
            The trained Scion
        """
        # Import grafting modules
        from ..grafting import (
            derive_cleft_from_lens,
            merge_clefts,
            ScionTrainer,
            ScionConfig,
        )

        run = self.training_runs.get(run_id)
        if not run:
            raise ValueError(f"Training run not found: {run_id}")

        if run.status != "pending":
            raise ValueError(f"Training run not pending: {run.status}")

        # Mark as training
        run.status = "training"
        run.started_at = datetime.now()

        try:
            # Get training data
            training_data = self.get_training_data(run.bud_tag_id)

            # Derive clefts from lenses
            clefts = []

            if self.lens_pack_path:
                # Try to load lenses for related concepts
                for concept_id in training_data.related_concept_ids:
                    # Try various lens naming conventions
                    for lens_name in [
                        f"{concept_id}.pt",
                        f"{concept_id.replace('::', '_')}.pt",
                        f"{concept_id.split('::')[-1]}.pt",
                    ]:
                        lens_path = self.lens_pack_path / lens_name
                        if lens_path.exists():
                            try:
                                cleft = derive_cleft_from_lens(
                                    lens_path,
                                    concept_id,
                                    model,
                                    layers=layers or [18, 20, 22],
                                )
                                clefts.append(cleft)
                                logger.info(f"Derived cleft from lens: {lens_name}")
                            except Exception as e:
                                logger.warning(f"Failed to derive cleft from {lens_name}: {e}")
                            break

            # If no lenses found, use a default cleft (all middle layers)
            if not clefts:
                logger.warning("No lenses found, using default cleft configuration")
                # Create a minimal cleft for the target concept
                # This will train without specific region constraints
                from src.map.graft.cleft import Cleft
                clefts = [Cleft(
                    concept_id=training_data.concept_id,
                    regions=[],
                    hidden_dim=model.config.hidden_size,
                )]

            # Merge clefts
            union_cleft = merge_clefts(clefts) if clefts else None

            # Configure training
            config = ScionConfig(
                learning_rate=run.training_config.get("learning_rate", 1e-4),
                epochs=run.training_config.get("epochs", 3),
                batch_size=run.training_config.get("batch_size", 8),
                injection_layers=layers or run.training_config.get("injection_layers", [18, 20, 22]),
            )

            # Train scion
            trainer = ScionTrainer(
                model=model,
                tokenizer=tokenizer,
                union_cleft=union_cleft,
                config=config,
            )

            scion = trainer.train(
                dataset=training_data.to_trainer_format(),
                concept_id=training_data.concept_id,
                verbose=verbose,
            )

            # Update run record
            run.status = "completed"
            run.completed_at = datetime.now()
            run.scion_id = scion.scion_id
            run.metrics = scion.metrics

            # Update bud status
            self.xdb.tag_index.update_bud_status(run.bud_tag_id, BudStatus.PROMOTED)

            logger.info(f"Scion training completed: {scion.scion_id}")
            return scion

        except Exception as e:
            run.status = "failed"
            run.completed_at = datetime.now()
            run.error_message = str(e)
            logger.error(f"Scion training failed: {e}")
            raise

    def promote_bud(
        self,
        bud_tag_id: str,
        model: 'nn.Module',
        tokenizer: Any,
        *,
        layers: Optional[List[int]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> 'Scion':
        """
        Full bud → scion promotion in one call.

        Convenience method that:
        1. Prepares training run
        2. Executes training
        3. Returns the scion

        Args:
            bud_tag_id: The bud tag to promote
            model: The substrate model
            tokenizer: The tokenizer
            layers: Which layers to inject
            training_config: Optional training configuration
            verbose: Print training progress

        Returns:
            The trained Scion
        """
        run = self.prepare_scion_training(bud_tag_id, training_config)
        return self.run_scion_training(
            run.id,
            model,
            tokenizer,
            layers=layers,
            verbose=verbose,
        )

    # =========================================================================
    # Training Run Management
    # =========================================================================

    def get_training_run(self, run_id: str) -> Optional[ScionTrainingRun]:
        """Get a training run by ID."""
        return self.training_runs.get(run_id)

    def list_training_runs(
        self,
        status: Optional[str] = None,
        bud_tag_id: Optional[str] = None,
    ) -> List[ScionTrainingRun]:
        """List training runs with optional filters."""
        runs = list(self.training_runs.values())

        if status:
            runs = [r for r in runs if r.status == status]
        if bud_tag_id:
            runs = [r for r in runs if r.bud_tag_id == bud_tag_id]

        return sorted(runs, key=lambda r: r.started_at or datetime.min, reverse=True)

    def cleanup_failed_run(self, run_id: str):
        """Clean up a failed training run."""
        run = self.training_runs.get(run_id)
        if not run:
            return

        # Unpin training data
        if run.training_window_ids:
            self.xdb.unpin_training_data(run.training_window_ids)

        # Reset bud status
        self.xdb.tag_index.update_bud_status(run.bud_tag_id, BudStatus.READY)

        # Remove run record
        del self.training_runs[run_id]

    # =========================================================================
    # Evidence Submission
    # =========================================================================

    def submit_for_review(
        self,
        scion_id: str,
        run_id: str,
    ) -> str:
        """
        Submit a scion for tribal review.

        Locks the training evidence as SUBMITTED fidelity.

        Args:
            scion_id: The scion to submit
            run_id: The training run that produced it

        Returns:
            Submission ID
        """
        run = self.training_runs.get(run_id)
        if not run:
            raise ValueError(f"Training run not found: {run_id}")

        if run.scion_id != scion_id:
            raise ValueError(f"Scion {scion_id} not from run {run_id}")

        # Generate submission ID
        submission_id = f"graft-submission-{uuid4().hex[:12]}"

        # Lock evidence
        self.xdb.submit_graft_evidence(run.bud_tag_id, submission_id)

        return submission_id

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_state(self, path: Path):
        """Save budding manager state to disk."""
        state = {
            "training_runs": {
                run_id: run.to_dict()
                for run_id, run in self.training_runs.items()
            },
            "lens_pack_path": str(self.lens_pack_path) if self.lens_pack_path else None,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: Path):
        """Load budding manager state from disk."""
        path = Path(path)
        if not path.exists():
            return

        with open(path) as f:
            state = json.load(f)

        # Restore training runs
        for run_id, run_data in state.get("training_runs", {}).items():
            run = ScionTrainingRun(
                id=run_data["id"],
                bud_tag_id=run_data["bud_tag_id"],
                concept_id=run_data["concept_id"],
                status=run_data["status"],
                training_config=run_data.get("training_config", {}),
                training_data_summary=run_data.get("training_data_summary"),
                training_window_ids=run_data.get("training_window_ids", []),
                scion_id=run_data.get("scion_id"),
                metrics=run_data.get("metrics", {}),
                error_message=run_data.get("error_message"),
            )
            if run_data.get("started_at"):
                run.started_at = datetime.fromisoformat(run_data["started_at"])
            if run_data.get("completed_at"):
                run.completed_at = datetime.fromisoformat(run_data["completed_at"])
            self.training_runs[run_id] = run

        # Restore lens pack path
        if state.get("lens_pack_path"):
            self.lens_pack_path = Path(state["lens_pack_path"])
