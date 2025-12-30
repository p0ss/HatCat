"""
SHA256 utilities for content addressing and pseudonymization.
"""

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Union


def hash_content(content: Union[str, bytes]) -> str:
    """
    Compute SHA256 hash of content for content-addressed storage.

    Args:
        content: String or bytes to hash

    Returns:
        Hash string in format "sha256:<hex>"
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    return f"sha256:{digest}"


def hash_operator_id(operator_id: str, salt: str = "") -> str:
    """
    Pseudonymize operator ID for audit logs.

    Uses SHA256 with optional salt for GDPR-compliant pseudonymization.
    The same operator_id + salt always produces the same hash, allowing
    correlation within a deployment while protecting identity.

    Args:
        operator_id: Raw operator identifier
        salt: Deployment-specific salt (recommended)

    Returns:
        Pseudonymized ID in format "op_sha256:<first 16 hex chars>"
    """
    combined = f"{salt}:{operator_id}" if salt else operator_id
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return f"op_sha256:{digest[:16]}"


def generate_entry_id() -> str:
    """
    Generate unique entry ID for audit log entries.

    Format: evt_YYYYMMDD_HHMMSSz_xxxx

    Where xxxx is 4 random hex characters for uniqueness within the same second.

    Returns:
        Entry ID string
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%SZ")
    suffix = secrets.token_hex(2)  # 4 hex chars
    return f"evt_{timestamp}_{suffix}"


def compute_entry_hash(entry_dict: dict, prev_hash: str) -> str:
    """
    Compute hash for an audit entry, chaining from previous entry.

    The entry_hash field itself is excluded from the hash computation.

    Args:
        entry_dict: Entry data as dictionary (entry_hash will be ignored)
        prev_hash: Hash of previous entry in chain

    Returns:
        Hash string in format "sha256:<hex>"
    """
    import json

    # Remove entry_hash if present (it's what we're computing)
    data = {k: v for k, v in entry_dict.items() if k != "entry_hash"}

    # Add prev_hash to computation
    data["prev_hash"] = prev_hash

    # Canonical JSON serialization
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hash_content(canonical)
