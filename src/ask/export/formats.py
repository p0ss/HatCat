"""
Export formats - regulatory submission format generation.

Creates standardized JSON exports of audit batches with all cryptographic
proofs bundled for regulatory submission or long-term archival.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from pathlib import Path
import json
import secrets
import hashlib
import base64

from ..secrets.hashing import hash_content
from ..storage.batches import AuditBatch
from ..storage.compaction import CompactedBatchSummary
from ..replication.authorities import AuthorityReceipt


class ExportFormat(Enum):
    """Supported export formats."""

    # Standard formats
    ASK_V1 = "ask.v1"  # Full ASK format with all proofs
    ASK_COMPACT = "ask.compact"  # Compacted format (summaries only)

    # Regulatory formats
    EU_AI_ACT = "eu.ai-act.v1"  # EU AI Act Article 12 format
    AU_AISF = "au.aisf.v1"  # Australian AI Safety Framework

    # Archive formats
    ARCHIVE = "archive.v1"  # Long-term archival format


class ExportScope(Enum):
    """What to include in export."""

    FULL = "full"  # All data including entry details
    SUMMARY = "summary"  # Batch summaries only
    PROOFS = "proofs"  # Cryptographic proofs only
    COMPLIANCE = "compliance"  # Compliance-relevant data only


@dataclass
class ExportConfig:
    """Configuration for export generation."""

    format: ExportFormat = ExportFormat.ASK_V1
    scope: ExportScope = ExportScope.FULL

    # What to include
    include_entries: bool = True
    include_merkle_proofs: bool = True
    include_timestamps: bool = True
    include_authority_receipts: bool = True
    include_chain_links: bool = True

    # Metadata
    include_deployment_context: bool = True
    include_signals_summary: bool = True

    # Formatting
    pretty_print: bool = False
    indent: int = 2

    # Validation
    validate_on_export: bool = True
    require_sealed: bool = True
    require_timestamped: bool = False
    require_authority_receipts: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format.value,
            "scope": self.scope.value,
            "include_entries": self.include_entries,
            "include_merkle_proofs": self.include_merkle_proofs,
            "include_timestamps": self.include_timestamps,
            "include_authority_receipts": self.include_authority_receipts,
        }


@dataclass
class ExportMetadata:
    """Metadata about an export."""

    export_id: str = ""
    export_format: str = ""
    export_version: str = "1.0.0"

    # Timing
    exported_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Content summary
    batch_count: int = 0
    entry_count: int = 0
    total_bytes: int = 0

    # Deployment info
    deployment_id: str = ""
    jurisdiction: str = ""

    # Time range covered
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    # Integrity
    content_hash: str = ""  # Hash of the export content

    def __post_init__(self):
        if not self.export_id:
            self.export_id = f"export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "export_id": self.export_id,
            "export_format": self.export_format,
            "export_version": self.export_version,
            "exported_at": self.exported_at.isoformat(),
            "batch_count": self.batch_count,
            "entry_count": self.entry_count,
            "total_bytes": self.total_bytes,
            "deployment_id": self.deployment_id,
            "jurisdiction": self.jurisdiction,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "content_hash": self.content_hash,
        }


@dataclass
class ExportedBatch:
    """A batch formatted for export."""

    batch_id: str
    batch_hash: str
    merkle_root: str

    # Timing
    created_at: str  # ISO format
    sealed_at: str

    # Chain
    prev_batch_id: str = ""
    prev_batch_hash: str = ""

    # Content
    entry_count: int = 0
    entries: List[Dict[str, Any]] = field(default_factory=list)

    # Proofs
    merkle_proofs: List[Dict[str, Any]] = field(default_factory=list)
    rfc3161_timestamp: Optional[Dict[str, Any]] = None
    authority_receipts: List[Dict[str, Any]] = field(default_factory=list)

    # Summary
    signals_summary: Dict[str, Any] = field(default_factory=dict)
    decision_count: int = 0
    violation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "batch_id": self.batch_id,
            "batch_hash": self.batch_hash,
            "merkle_root": self.merkle_root,
            "created_at": self.created_at,
            "sealed_at": self.sealed_at,
            "entry_count": self.entry_count,
        }

        if self.prev_batch_id:
            result["chain"] = {
                "prev_batch_id": self.prev_batch_id,
                "prev_batch_hash": self.prev_batch_hash,
            }

        if self.entries:
            result["entries"] = self.entries

        if self.merkle_proofs:
            result["merkle_proofs"] = self.merkle_proofs

        if self.rfc3161_timestamp:
            result["rfc3161_timestamp"] = self.rfc3161_timestamp

        if self.authority_receipts:
            result["authority_receipts"] = self.authority_receipts

        if self.signals_summary:
            result["signals_summary"] = self.signals_summary

        result["statistics"] = {
            "decision_count": self.decision_count,
            "violation_count": self.violation_count,
        }

        return result


@dataclass
class ExportResult:
    """Result of an export operation."""

    success: bool = False
    export_id: str = ""

    # Content
    metadata: Optional[ExportMetadata] = None
    content: str = ""  # JSON string
    content_bytes: bytes = field(default_factory=bytes, repr=False)

    # Validation
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

    # Output
    output_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "export_id": self.export_id,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "validated": self.validated,
            "validation_errors": self.validation_errors,
            "output_path": str(self.output_path) if self.output_path else None,
        }


class BatchExporter:
    """
    Exports audit batches to regulatory submission formats.

    Bundles batch data with all cryptographic proofs into a
    standardized, verifiable format.
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()

    def export_batch(
        self,
        batch: AuditBatch,
        authority_receipts: Optional[List[AuthorityReceipt]] = None,
        entries: Optional[List[Dict[str, Any]]] = None,
    ) -> ExportedBatch:
        """
        Export a single batch.

        Args:
            batch: The batch to export
            authority_receipts: Optional authority receipts
            entries: Optional entry data to include

        Returns:
            ExportedBatch ready for serialization
        """
        exported = ExportedBatch(
            batch_id=batch.batch_id,
            batch_hash=batch.batch_hash,
            merkle_root=batch.merkle_root,
            created_at=batch.created_at.isoformat(),
            sealed_at=batch.sealed_at.isoformat() if batch.sealed_at else "",
            prev_batch_id=batch.prev_batch_id,
            prev_batch_hash=batch.prev_batch_hash,
            entry_count=batch.entry_count,
            decision_count=batch.decision_count,
            violation_count=batch.violation_count,
        )

        # Include entries if configured and provided
        if self.config.include_entries and entries:
            exported.entries = entries

        # Include Merkle proofs
        if self.config.include_merkle_proofs and batch._merkle_tree:
            proofs = []
            for i, entry_hash in enumerate(batch.entry_hashes):
                proof = batch._merkle_tree.get_proof(entry_hash)
                if proof:
                    proofs.append(proof.to_dict())
            exported.merkle_proofs = proofs

        # Include RFC 3161 timestamp
        if self.config.include_timestamps and batch.rfc3161_token:
            from ..secrets.tokens import TimestampToken
            token = TimestampToken.from_bytes(batch.rfc3161_token)
            exported.rfc3161_timestamp = token.to_dict()

        # Include authority receipts
        if self.config.include_authority_receipts and authority_receipts:
            exported.authority_receipts = [r.to_dict() for r in authority_receipts]

        # Include signals summary
        if self.config.include_signals_summary:
            exported.signals_summary = batch.signals_summary

        return exported

    def export_batches(
        self,
        batches: List[AuditBatch],
        authority_receipts: Optional[Dict[str, List[AuthorityReceipt]]] = None,
        entries_by_batch: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        deployment_id: str = "",
        jurisdiction: str = "",
    ) -> ExportResult:
        """
        Export multiple batches to a complete export package.

        Args:
            batches: Batches to export
            authority_receipts: Dict of batch_id -> receipts
            entries_by_batch: Dict of batch_id -> entries
            deployment_id: Deployment identifier
            jurisdiction: Jurisdiction code

        Returns:
            ExportResult with serialized content
        """
        result = ExportResult()
        authority_receipts = authority_receipts or {}
        entries_by_batch = entries_by_batch or {}

        try:
            # Validate if configured
            if self.config.validate_on_export:
                errors = self._validate_batches(batches)
                if errors:
                    result.validation_errors = errors
                    if self.config.require_sealed:
                        result.success = False
                        return result

            # Export each batch
            exported_batches = []
            total_entries = 0

            for batch in batches:
                receipts = authority_receipts.get(batch.batch_id, [])
                entries = entries_by_batch.get(batch.batch_id, [])

                exported = self.export_batch(batch, receipts, entries)
                exported_batches.append(exported.to_dict())
                total_entries += batch.entry_count

            # Determine time range
            sealed_times = [b.sealed_at for b in batches if b.sealed_at]
            period_start = min(sealed_times) if sealed_times else None
            period_end = max(sealed_times) if sealed_times else None

            # Build export document
            export_doc = self._build_export_document(
                batches=exported_batches,
                deployment_id=deployment_id,
                jurisdiction=jurisdiction,
                period_start=period_start,
                period_end=period_end,
            )

            # Serialize
            if self.config.pretty_print:
                content = json.dumps(export_doc, indent=self.config.indent, ensure_ascii=False)
            else:
                content = json.dumps(export_doc, separators=(",", ":"), ensure_ascii=False)

            content_bytes = content.encode("utf-8")

            # Compute content hash
            content_hash = hash_content(content)

            # Build metadata
            metadata = ExportMetadata(
                export_format=self.config.format.value,
                batch_count=len(batches),
                entry_count=total_entries,
                total_bytes=len(content_bytes),
                deployment_id=deployment_id,
                jurisdiction=jurisdiction,
                period_start=period_start,
                period_end=period_end,
                content_hash=content_hash,
            )

            result.success = True
            result.export_id = metadata.export_id
            result.metadata = metadata
            result.content = content
            result.content_bytes = content_bytes
            result.validated = len(result.validation_errors) == 0

        except Exception as e:
            result.success = False
            result.validation_errors.append(f"Export failed: {str(e)}")

        return result

    def _build_export_document(
        self,
        batches: List[Dict[str, Any]],
        deployment_id: str,
        jurisdiction: str,
        period_start: Optional[datetime],
        period_end: Optional[datetime],
    ) -> Dict[str, Any]:
        """Build the complete export document."""

        doc = {
            "$schema": f"https://ask.ftw.dev/schemas/{self.config.format.value}.json",
            "format": self.config.format.value,
            "version": "1.0.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        # Deployment context
        if self.config.include_deployment_context:
            doc["deployment"] = {
                "deployment_id": deployment_id,
                "jurisdiction": jurisdiction,
            }

        # Time range
        if period_start or period_end:
            doc["period"] = {
                "start": period_start.isoformat() if period_start else None,
                "end": period_end.isoformat() if period_end else None,
            }

        # Chain integrity
        if self.config.include_chain_links and len(batches) > 1:
            doc["chain"] = {
                "batch_count": len(batches),
                "first_batch_id": batches[0]["batch_id"],
                "last_batch_id": batches[-1]["batch_id"],
                "first_batch_hash": batches[0]["batch_hash"],
                "last_batch_hash": batches[-1]["batch_hash"],
            }

        # Batches
        doc["batches"] = batches

        # Format-specific additions
        if self.config.format == ExportFormat.EU_AI_ACT:
            doc = self._add_eu_ai_act_fields(doc)
        elif self.config.format == ExportFormat.AU_AISF:
            doc = self._add_au_aisf_fields(doc)

        return doc

    def _add_eu_ai_act_fields(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Add EU AI Act specific fields."""
        doc["regulatory"] = {
            "framework": "EU AI Act",
            "article": "Article 12 - Record-keeping",
            "compliance_level": "high-risk",  # Would be determined by deployment
        }
        return doc

    def _add_au_aisf_fields(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Add Australian AI Safety Framework fields."""
        doc["regulatory"] = {
            "framework": "Australian AI Safety Framework",
            "principles": ["transparency", "accountability"],
        }
        return doc

    def _validate_batches(self, batches: List[AuditBatch]) -> List[str]:
        """Validate batches for export."""
        errors = []

        for batch in batches:
            if self.config.require_sealed and not batch._sealed:
                errors.append(f"Batch {batch.batch_id} is not sealed")

            if self.config.require_timestamped and not batch.rfc3161_timestamp:
                errors.append(f"Batch {batch.batch_id} has no RFC 3161 timestamp")

        return errors

    def export_to_file(
        self,
        batches: List[AuditBatch],
        output_path: Path,
        **kwargs,
    ) -> ExportResult:
        """
        Export batches to a file.

        Args:
            batches: Batches to export
            output_path: Where to write the export
            **kwargs: Additional arguments for export_batches

        Returns:
            ExportResult with output_path set
        """
        result = self.export_batches(batches, **kwargs)

        if result.success:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(result.content_bytes)
            result.output_path = output_path

        return result

    def export_compacted(
        self,
        summaries: List[CompactedBatchSummary],
        deployment_id: str = "",
        jurisdiction: str = "",
    ) -> ExportResult:
        """
        Export compacted batch summaries.

        For exports where full entry data is no longer available.
        """
        result = ExportResult()

        try:
            # Convert summaries to export format
            exported_summaries = []
            for summary in summaries:
                exported_summaries.append({
                    "batch_id": summary.batch_id,
                    "batch_hash": summary.batch_hash,
                    "merkle_root": summary.merkle_root,
                    "created_at": summary.created_at.isoformat(),
                    "sealed_at": summary.sealed_at.isoformat(),
                    "compacted_at": summary.compacted_at.isoformat(),
                    "original_entry_count": summary.original_entry_count,
                    "preserved_entry_count": summary.preserved_entry_count,
                    "chain": {
                        "prev_batch_id": summary.prev_batch_id,
                        "prev_batch_hash": summary.prev_batch_hash,
                    },
                    "statistics": {
                        "decision_count": summary.decision_count,
                        "violation_count": summary.violation_count,
                    },
                })

            # Determine time range
            sealed_times = [s.sealed_at for s in summaries]
            period_start = min(sealed_times) if sealed_times else None
            period_end = max(sealed_times) if sealed_times else None

            # Build document
            doc = {
                "$schema": f"https://ask.ftw.dev/schemas/{ExportFormat.ASK_COMPACT.value}.json",
                "format": ExportFormat.ASK_COMPACT.value,
                "version": "1.0.0",
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "compacted": True,
                "deployment": {
                    "deployment_id": deployment_id,
                    "jurisdiction": jurisdiction,
                },
                "period": {
                    "start": period_start.isoformat() if period_start else None,
                    "end": period_end.isoformat() if period_end else None,
                },
                "batches": exported_summaries,
            }

            # Serialize
            if self.config.pretty_print:
                content = json.dumps(doc, indent=self.config.indent, ensure_ascii=False)
            else:
                content = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)

            content_bytes = content.encode("utf-8")
            content_hash = hash_content(content)

            metadata = ExportMetadata(
                export_format=ExportFormat.ASK_COMPACT.value,
                batch_count=len(summaries),
                entry_count=sum(s.original_entry_count for s in summaries),
                total_bytes=len(content_bytes),
                deployment_id=deployment_id,
                jurisdiction=jurisdiction,
                period_start=period_start,
                period_end=period_end,
                content_hash=content_hash,
            )

            result.success = True
            result.export_id = metadata.export_id
            result.metadata = metadata
            result.content = content
            result.content_bytes = content_bytes
            result.validated = True

        except Exception as e:
            result.success = False
            result.validation_errors.append(f"Export failed: {str(e)}")

        return result


class ExportVerifier:
    """Verifies exported audit packages."""

    def verify_export(
        self,
        content: Union[str, bytes],
    ) -> Tuple[bool, List[str]]:
        """
        Verify an export package.

        Args:
            content: Export content (JSON string or bytes)

        Returns:
            (valid, errors)
        """
        errors = []

        try:
            if isinstance(content, bytes):
                content = content.decode("utf-8")

            doc = json.loads(content)

            # Check required fields
            required = ["format", "version", "exported_at", "batches"]
            for field in required:
                if field not in doc:
                    errors.append(f"Missing required field: {field}")

            if errors:
                return False, errors

            # Verify batch chain
            chain_errors = self._verify_chain(doc.get("batches", []))
            errors.extend(chain_errors)

            # Verify Merkle proofs if present
            for batch in doc.get("batches", []):
                if "merkle_proofs" in batch:
                    proof_errors = self._verify_merkle_proofs(batch)
                    errors.extend(proof_errors)

            return len(errors) == 0, errors

        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {e}"]
        except Exception as e:
            return False, [f"Verification failed: {e}"]

    def _verify_chain(self, batches: List[Dict[str, Any]]) -> List[str]:
        """Verify batch chain integrity."""
        errors = []

        for i in range(1, len(batches)):
            prev_batch = batches[i - 1]
            curr_batch = batches[i]

            chain = curr_batch.get("chain", {})
            if chain.get("prev_batch_id") and chain["prev_batch_id"] != prev_batch["batch_id"]:
                errors.append(
                    f"Chain break: {curr_batch['batch_id']} links to "
                    f"{chain['prev_batch_id']} but previous is {prev_batch['batch_id']}"
                )

            if chain.get("prev_batch_hash") and chain["prev_batch_hash"] != prev_batch["batch_hash"]:
                errors.append(
                    f"Hash mismatch in chain at batch {curr_batch['batch_id']}"
                )

        return errors

    def _verify_merkle_proofs(self, batch: Dict[str, Any]) -> List[str]:
        """Verify Merkle proofs in a batch."""
        errors = []
        merkle_root = batch.get("merkle_root", "")

        for proof in batch.get("merkle_proofs", []):
            if proof.get("merkle_root") != merkle_root:
                errors.append(
                    f"Merkle proof root mismatch in batch {batch['batch_id']}"
                )

        return errors

    def compute_content_hash(self, content: Union[str, bytes]) -> str:
        """Compute hash of export content."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hash_content(content.decode("utf-8"))


# Convenience functions

def create_full_exporter() -> BatchExporter:
    """Create exporter with full data included."""
    return BatchExporter(ExportConfig(
        format=ExportFormat.ASK_V1,
        scope=ExportScope.FULL,
        include_entries=True,
        include_merkle_proofs=True,
        include_timestamps=True,
        include_authority_receipts=True,
        pretty_print=True,
    ))


def create_compliance_exporter(jurisdiction: str = "EU") -> BatchExporter:
    """Create exporter for regulatory compliance."""
    format_map = {
        "EU": ExportFormat.EU_AI_ACT,
        "AU": ExportFormat.AU_AISF,
    }

    return BatchExporter(ExportConfig(
        format=format_map.get(jurisdiction, ExportFormat.ASK_V1),
        scope=ExportScope.COMPLIANCE,
        include_entries=False,  # Privacy - summaries only
        include_merkle_proofs=True,
        include_timestamps=True,
        include_authority_receipts=True,
        require_sealed=True,
        require_timestamped=True,
        pretty_print=True,
    ))


def create_archive_exporter() -> BatchExporter:
    """Create exporter for long-term archival."""
    return BatchExporter(ExportConfig(
        format=ExportFormat.ARCHIVE,
        scope=ExportScope.FULL,
        include_entries=True,
        include_merkle_proofs=True,
        include_timestamps=True,
        include_authority_receipts=True,
        validate_on_export=True,
        require_sealed=True,
        pretty_print=False,  # Compact for storage
    ))
