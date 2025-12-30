"""
Merkle tree implementation for audit batch integrity.

Provides cryptographic proof that entries belong to a sealed batch
without requiring the full batch data.
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def _hash_pair(left: str, right: str) -> str:
    """Hash two nodes together to form parent."""
    # Ensure consistent ordering for deterministic trees
    combined = f"{left}:{right}"
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _hash_leaf(entry_hash: str) -> str:
    """Hash a leaf node (entry hash) with leaf prefix for domain separation."""
    # Prefix with "leaf:" to prevent second-preimage attacks
    combined = f"leaf:{entry_hash}"
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass
class MerkleProof:
    """
    Proof that an entry belongs to a Merkle tree.

    The proof consists of sibling hashes along the path from leaf to root.
    Each step includes the sibling hash and whether it's on the left or right.
    """

    entry_hash: str  # The entry being proven
    leaf_index: int  # Position in the original list
    proof_path: List[Tuple[str, str]]  # [(hash, "left"|"right"), ...]
    merkle_root: str  # Expected root

    def verify(self) -> bool:
        """Verify this proof against the merkle_root."""
        current = _hash_leaf(self.entry_hash)

        for sibling_hash, position in self.proof_path:
            if position == "left":
                current = _hash_pair(sibling_hash, current)
            else:  # position == "right"
                current = _hash_pair(current, sibling_hash)

        return current == self.merkle_root

    def to_dict(self) -> dict:
        """Serialize proof to dict."""
        return {
            "entry_hash": self.entry_hash,
            "leaf_index": self.leaf_index,
            "proof_path": [
                {"hash": h, "position": p} for h, p in self.proof_path
            ],
            "merkle_root": self.merkle_root,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MerkleProof":
        """Deserialize proof from dict."""
        return cls(
            entry_hash=data["entry_hash"],
            leaf_index=data["leaf_index"],
            proof_path=[
                (step["hash"], step["position"])
                for step in data["proof_path"]
            ],
            merkle_root=data["merkle_root"],
        )


@dataclass
class MerkleTree:
    """
    Merkle tree for a batch of audit entries.

    Builds a binary tree where:
    - Leaves are hashes of entry hashes (with domain separation)
    - Internal nodes are hashes of their children
    - Root provides single hash representing all entries

    Supports generating inclusion proofs for any entry.
    """

    entry_hashes: List[str] = field(default_factory=list)
    _tree_levels: List[List[str]] = field(default_factory=list, repr=False)
    _built: bool = field(default=False, repr=False)

    @classmethod
    def build(cls, entry_hashes: List[str]) -> "MerkleTree":
        """
        Build a Merkle tree from a list of entry hashes.

        Args:
            entry_hashes: List of entry hashes (sha256:... format)

        Returns:
            MerkleTree with computed root
        """
        tree = cls(entry_hashes=list(entry_hashes))
        tree._build_tree()
        return tree

    def _build_tree(self) -> None:
        """Build the tree levels from leaves to root."""
        if not self.entry_hashes:
            self._tree_levels = []
            self._built = True
            return

        # Level 0: leaf hashes
        leaves = [_hash_leaf(h) for h in self.entry_hashes]
        self._tree_levels = [leaves]

        # Build up to root
        current_level = leaves
        while len(current_level) > 1:
            next_level = []

            # Process pairs
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                # If odd number, duplicate the last node
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                parent = _hash_pair(left, right)
                next_level.append(parent)

            self._tree_levels.append(next_level)
            current_level = next_level

        self._built = True

    @property
    def root(self) -> str:
        """Get the Merkle root hash."""
        if not self._built:
            self._build_tree()

        if not self._tree_levels:
            # Empty tree - return hash of empty
            return _hash_leaf("")

        return self._tree_levels[-1][0]

    @property
    def leaf_count(self) -> int:
        """Number of entries in the tree."""
        return len(self.entry_hashes)

    @property
    def height(self) -> int:
        """Height of the tree (number of levels)."""
        if not self._built:
            self._build_tree()
        return len(self._tree_levels)

    def get_proof(self, entry_hash: str) -> Optional[MerkleProof]:
        """
        Generate inclusion proof for an entry.

        Args:
            entry_hash: The entry hash to prove inclusion of

        Returns:
            MerkleProof if entry exists, None otherwise
        """
        if not self._built:
            self._build_tree()

        # Find the entry
        try:
            leaf_index = self.entry_hashes.index(entry_hash)
        except ValueError:
            return None

        return self.get_proof_by_index(leaf_index)

    def get_proof_by_index(self, leaf_index: int) -> Optional[MerkleProof]:
        """
        Generate inclusion proof for entry at given index.

        Args:
            leaf_index: Index of the entry in the original list

        Returns:
            MerkleProof if index valid, None otherwise
        """
        if not self._built:
            self._build_tree()

        if leaf_index < 0 or leaf_index >= len(self.entry_hashes):
            return None

        proof_path = []
        current_index = leaf_index

        # Walk up the tree, collecting siblings
        for level in range(len(self._tree_levels) - 1):
            level_nodes = self._tree_levels[level]

            # Determine sibling
            if current_index % 2 == 0:
                # Current is left child, sibling is right
                sibling_index = current_index + 1
                if sibling_index < len(level_nodes):
                    sibling_hash = level_nodes[sibling_index]
                else:
                    # Odd number of nodes - sibling is self (duplicated)
                    sibling_hash = level_nodes[current_index]
                position = "right"
            else:
                # Current is right child, sibling is left
                sibling_index = current_index - 1
                sibling_hash = level_nodes[sibling_index]
                position = "left"

            proof_path.append((sibling_hash, position))
            current_index = current_index // 2

        return MerkleProof(
            entry_hash=self.entry_hashes[leaf_index],
            leaf_index=leaf_index,
            proof_path=proof_path,
            merkle_root=self.root,
        )

    def verify_proof(self, proof: MerkleProof) -> bool:
        """
        Verify a proof against this tree's root.

        Args:
            proof: The proof to verify

        Returns:
            True if proof is valid for this tree
        """
        return proof.merkle_root == self.root and proof.verify()

    def to_dict(self) -> dict:
        """Serialize tree metadata (not full tree) to dict."""
        return {
            "merkle_root": self.root,
            "leaf_count": self.leaf_count,
            "height": self.height,
            "entry_hashes": self.entry_hashes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MerkleTree":
        """Reconstruct tree from dict."""
        return cls.build(data["entry_hashes"])


def compute_merkle_root(entry_hashes: List[str]) -> str:
    """
    Convenience function to compute Merkle root from entry hashes.

    Args:
        entry_hashes: List of entry hashes

    Returns:
        Merkle root hash
    """
    tree = MerkleTree.build(entry_hashes)
    return tree.root


def verify_inclusion(
    entry_hash: str,
    proof: MerkleProof,
    expected_root: str,
) -> bool:
    """
    Verify an entry is included in a batch with given Merkle root.

    Args:
        entry_hash: The entry hash to verify
        proof: The inclusion proof
        expected_root: The expected Merkle root

    Returns:
        True if entry is provably included
    """
    if proof.entry_hash != entry_hash:
        return False
    if proof.merkle_root != expected_root:
        return False
    return proof.verify()
