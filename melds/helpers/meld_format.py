"""
Meld Submission Format - Unified training data for lens+graft pairs.

Every concept that gets added to a BE needs:
1. A lens to detect when the concept is active
2. A graft to add a dedicated neuron for the concept
3. Evidence that both work correctly

The Meld submission format captures all of this in a single structure
that can go through the approval pipeline.

The format is designed to:
- Support both lens training and graft training
- Capture contrastive pairs (positive/negative examples)
- Include validation data separate from training data
- Track provenance and approval status
- Enable reproducibility
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
import hashlib


class SubmissionStatus(Enum):
    """Status of a Meld submission."""
    DRAFT = "draft"              # Being prepared
    SUBMITTED = "submitted"       # Awaiting review
    UNDER_REVIEW = "under_review" # Being evaluated
    APPROVED = "approved"         # Ready for grafting
    REJECTED = "rejected"         # Did not pass review
    INTEGRATED = "integrated"     # Graft applied to substrate


class EvidenceType(Enum):
    """Types of evidence that can support a submission."""
    LENS_METRICS = "lens_metrics"       # F1, accuracy, etc.
    GRAFT_METRICS = "graft_metrics"       # Training loss, delta magnitude
    ACTIVATION_SAMPLES = "activation_samples"  # Example activations
    CONTRASTIVE_PAIRS = "contrastive_pairs"    # A/B comparisons
    HUMAN_REVIEW = "human_review"          # Manual evaluation
    AUTOMATED_TEST = "automated_test"      # Test suite results


@dataclass
class TrainingExample:
    """
    A single training example for lens/graft training.

    Examples are labeled and can include metadata about source.
    """
    text: str
    label: int  # 1 for positive, 0 for negative

    # Optional metadata
    source: str = ""  # Where this example came from
    confidence: float = 1.0  # How confident are we in the label
    layer_preferences: List[int] = field(default_factory=list)  # Best layers for this

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "source": self.source,
            "confidence": self.confidence,
            "layer_preferences": self.layer_preferences,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingExample":
        return cls(**d)


@dataclass
class ContrastivePair:
    """
    A pair of examples that should be distinguished.

    Used for contrastive training - the model should learn to
    differentiate the positive from the negative.
    """
    positive: str
    negative: str

    # What makes them different
    contrast_dimension: str = ""

    # Optional: expected activation difference
    expected_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positive": self.positive,
            "negative": self.negative,
            "contrast_dimension": self.contrast_dimension,
            "expected_delta": self.expected_delta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContrastivePair":
        return cls(**d)


@dataclass
class EvidenceItem:
    """
    A piece of evidence supporting a submission.
    """
    evidence_type: EvidenceType
    description: str
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_type": self.evidence_type.value,
            "description": self.description,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceItem":
        d = d.copy()
        d["evidence_type"] = EvidenceType(d["evidence_type"])
        return cls(**d)


@dataclass
class MeldSubmission:
    """
    A complete Meld submission for a concept lens+graft pair.

    This is the unit of approval in the Meld protocol:
    - Submit this to request adding a concept to the BE
    - Goes through review by ASK authority
    - If approved, both lens and graft are trained and applied
    """
    # Identity
    submission_id: str
    concept_id: str

    # Concept definition
    definition: str
    parent_concepts: List[str] = field(default_factory=list)
    sumo_mapping: Optional[str] = None

    # Training data
    training_examples: List[TrainingExample] = field(default_factory=list)
    contrastive_pairs: List[ContrastivePair] = field(default_factory=list)

    # Validation data (held out)
    validation_examples: List[TrainingExample] = field(default_factory=list)
    validation_pairs: List[ContrastivePair] = field(default_factory=list)

    # Training configuration
    recommended_layers: List[int] = field(default_factory=lambda: [18, 20, 22])
    min_f1_threshold: float = 0.85
    scion_config: Dict[str, Any] = field(default_factory=dict)

    # Evidence
    evidence: List[EvidenceItem] = field(default_factory=list)

    # Status
    status: SubmissionStatus = SubmissionStatus.DRAFT
    submitted_at: Optional[str] = None
    reviewed_at: Optional[str] = None
    reviewer_id: Optional[str] = None
    review_notes: str = ""

    # Provenance
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = ""
    tribe_id: str = ""

    # Results (populated after training)
    lens_path: Optional[str] = None
    lens_metrics: Dict[str, float] = field(default_factory=dict)
    graft_path: Optional[str] = None
    graft_metrics: Dict[str, float] = field(default_factory=dict)

    def compute_checksum(self) -> str:
        """Compute checksum of training data."""
        h = hashlib.sha256()
        for ex in self.training_examples:
            h.update(ex.text.encode())
            h.update(str(ex.label).encode())
        for pair in self.contrastive_pairs:
            h.update(pair.positive.encode())
            h.update(pair.negative.encode())
        return h.hexdigest()[:16]

    def get_positive_examples(self) -> List[str]:
        """Get all positive training examples."""
        return [ex.text for ex in self.training_examples if ex.label == 1]

    def get_negative_examples(self) -> List[str]:
        """Get all negative training examples."""
        return [ex.text for ex in self.training_examples if ex.label == 0]

    def add_evidence(
        self,
        evidence_type: EvidenceType,
        description: str,
        data: Dict[str, Any]
    ):
        """Add evidence to the submission."""
        self.evidence.append(EvidenceItem(
            evidence_type=evidence_type,
            description=description,
            data=data,
        ))

    def submit(self) -> bool:
        """Submit for review."""
        if self.status != SubmissionStatus.DRAFT:
            return False

        # Validate minimum requirements
        if len(self.training_examples) < 20:
            return False
        if len(self.get_positive_examples()) < 10:
            return False
        if len(self.get_negative_examples()) < 10:
            return False

        self.status = SubmissionStatus.SUBMITTED
        self.submitted_at = datetime.now().isoformat()
        return True

    def approve(self, reviewer_id: str, notes: str = ""):
        """Approve the submission."""
        self.status = SubmissionStatus.APPROVED
        self.reviewed_at = datetime.now().isoformat()
        self.reviewer_id = reviewer_id
        self.review_notes = notes

    def reject(self, reviewer_id: str, notes: str):
        """Reject the submission."""
        self.status = SubmissionStatus.REJECTED
        self.reviewed_at = datetime.now().isoformat()
        self.reviewer_id = reviewer_id
        self.review_notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "concept_id": self.concept_id,
            "definition": self.definition,
            "parent_concepts": self.parent_concepts,
            "sumo_mapping": self.sumo_mapping,
            "training_examples": [ex.to_dict() for ex in self.training_examples],
            "contrastive_pairs": [p.to_dict() for p in self.contrastive_pairs],
            "validation_examples": [ex.to_dict() for ex in self.validation_examples],
            "validation_pairs": [p.to_dict() for p in self.validation_pairs],
            "recommended_layers": self.recommended_layers,
            "min_f1_threshold": self.min_f1_threshold,
            "scion_config": self.scion_config,
            "evidence": [e.to_dict() for e in self.evidence],
            "status": self.status.value,
            "submitted_at": self.submitted_at,
            "reviewed_at": self.reviewed_at,
            "reviewer_id": self.reviewer_id,
            "review_notes": self.review_notes,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "tribe_id": self.tribe_id,
            "lens_path": self.lens_path,
            "lens_metrics": self.lens_metrics,
            "graft_path": self.graft_path,
            "graft_metrics": self.graft_metrics,
            "checksum": self.compute_checksum(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MeldSubmission":
        d = d.copy()
        d["status"] = SubmissionStatus(d["status"])
        d["training_examples"] = [
            TrainingExample.from_dict(ex)
            for ex in d.get("training_examples", [])
        ]
        d["contrastive_pairs"] = [
            ContrastivePair.from_dict(p)
            for p in d.get("contrastive_pairs", [])
        ]
        d["validation_examples"] = [
            TrainingExample.from_dict(ex)
            for ex in d.get("validation_examples", [])
        ]
        d["validation_pairs"] = [
            ContrastivePair.from_dict(p)
            for p in d.get("validation_pairs", [])
        ]
        d["evidence"] = [
            EvidenceItem.from_dict(e)
            for e in d.get("evidence", [])
        ]
        # Remove checksum as it's computed
        d.pop("checksum", None)
        return cls(**d)

    def save(self, output_path: Path):
        """Save submission to disk."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, input_path: Path) -> "MeldSubmission":
        """Load submission from disk."""
        with open(input_path) as f:
            return cls.from_dict(json.load(f))


def create_submission_from_uplift_concept(
    concept: "UpliftConcept",
    tribe_id: str = "",
    created_by: str = "system",
) -> MeldSubmission:
    """
    Create a MeldSubmission from an UpliftConcept.

    This bridges the taxonomy definition to the Meld format.
    """
    from .uplift_taxonomy import UpliftConcept
    import uuid

    submission = MeldSubmission(
        submission_id=f"meld-{concept.concept_id}-{uuid.uuid4().hex[:8]}",
        concept_id=concept.concept_id,
        definition=concept.definition,
        parent_concepts=concept.parent_concepts,
        sumo_mapping=concept.sumo_mapping,
        recommended_layers=concept.recommended_layers,
        min_f1_threshold=concept.min_f1_threshold,
        tribe_id=tribe_id,
        created_by=created_by,
    )

    # Convert positive examples
    for text in concept.positive_examples:
        submission.training_examples.append(TrainingExample(
            text=text,
            label=1,
            source="uplift_taxonomy",
        ))

    # Convert negative examples
    for text in concept.negative_examples:
        submission.training_examples.append(TrainingExample(
            text=text,
            label=0,
            source="uplift_taxonomy",
        ))

    # Convert contrastive pairs
    for pair in concept.contrastive_pairs:
        submission.contrastive_pairs.append(ContrastivePair(
            positive=pair.get("positive", ""),
            negative=pair.get("negative", ""),
            contrast_dimension=pair.get("dimension", ""),
        ))

    return submission


@dataclass
class MeldBatch:
    """
    A batch of related Meld submissions.

    Used for uplift where many concepts are submitted together.
    """
    batch_id: str
    name: str
    description: str = ""

    submissions: List[MeldSubmission] = field(default_factory=list)

    # Batch-level configuration
    shared_layers: List[int] = field(default_factory=lambda: [18, 20, 22])
    shared_min_f1: float = 0.85

    # Status
    status: SubmissionStatus = SubmissionStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tribe_id: str = ""

    def add_submission(self, submission: MeldSubmission):
        """Add a submission to the batch."""
        self.submissions.append(submission)

    def get_graft_order(self) -> List[str]:
        """
        Get submission IDs in dependency-respecting order.

        Concepts that are parents of others must be grafted first.
        """
        # Build dependency graph
        concept_to_id = {s.concept_id: s.submission_id for s in self.submissions}

        order = []
        visited = set()

        def visit(submission: MeldSubmission):
            if submission.submission_id in visited:
                return
            # Visit parents first
            for parent in submission.parent_concepts:
                if parent in concept_to_id:
                    parent_sub = next(
                        s for s in self.submissions
                        if s.concept_id == parent
                    )
                    visit(parent_sub)
            visited.add(submission.submission_id)
            order.append(submission.submission_id)

        for sub in self.submissions:
            visit(sub)

        return order

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "name": self.name,
            "description": self.description,
            "submissions": [s.to_dict() for s in self.submissions],
            "shared_layers": self.shared_layers,
            "shared_min_f1": self.shared_min_f1,
            "status": self.status.value,
            "created_at": self.created_at,
            "tribe_id": self.tribe_id,
            "graft_order": self.get_graft_order(),
        }

    def save(self, output_dir: Path):
        """Save batch to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save batch metadata
        meta = self.to_dict()
        meta.pop("submissions")  # Save separately
        with open(output_dir / "batch.json", 'w') as f:
            json.dump(meta, f, indent=2)

        # Save each submission
        for sub in self.submissions:
            sub.save(output_dir / f"{sub.concept_id}.json")

    @classmethod
    def load(cls, input_dir: Path) -> "MeldBatch":
        """Load batch from disk."""
        input_dir = Path(input_dir)

        with open(input_dir / "batch.json") as f:
            meta = json.load(f)

        batch = cls(
            batch_id=meta["batch_id"],
            name=meta["name"],
            description=meta.get("description", ""),
            shared_layers=meta.get("shared_layers", [18, 20, 22]),
            shared_min_f1=meta.get("shared_min_f1", 0.85),
            status=SubmissionStatus(meta["status"]),
            tribe_id=meta.get("tribe_id", ""),
        )
        batch.created_at = meta.get("created_at", "")

        # Load submissions
        for json_file in input_dir.glob("*.json"):
            if json_file.name == "batch.json":
                continue
            batch.add_submission(MeldSubmission.load(json_file))

        return batch


def create_uplift_batch(
    taxonomy: "UpliftTaxonomy",
    tribe_id: str = "",
    layers: Optional[List["UpliftLayer"]] = None,
    created_by: str = "system",
) -> MeldBatch:
    """
    Create a MeldBatch from an UpliftTaxonomy.

    This creates submissions for all concepts in the taxonomy
    (or specified layers), ready for the approval pipeline.
    """
    from .uplift_taxonomy import UpliftTaxonomy, UpliftLayer
    import uuid

    batch = MeldBatch(
        batch_id=f"uplift-{uuid.uuid4().hex[:8]}",
        name=f"Uplift for {taxonomy.name}",
        description=f"Full uplift batch from {taxonomy.name} v{taxonomy.version}",
        tribe_id=tribe_id,
    )

    # Get concepts to include
    if layers is None:
        concepts = list(taxonomy.concepts.values())
    else:
        concepts = []
        for layer in layers:
            concepts.extend(taxonomy.get_layer(layer))

    # Create submissions in graft order
    graft_order = taxonomy.get_graft_order()

    for concept_id in graft_order:
        if concept_id not in taxonomy.concepts:
            continue
        concept = taxonomy.concepts[concept_id]

        # Skip if not in requested layers
        if layers is not None and concept.layer not in layers:
            continue

        submission = create_submission_from_uplift_concept(
            concept,
            tribe_id=tribe_id,
            created_by=created_by,
        )
        batch.add_submission(submission)

    return batch
