"""Tests for Merkle tree implementation."""

import pytest

from src.ask.storage.merkle import (
    MerkleTree,
    MerkleProof,
    compute_merkle_root,
    verify_inclusion,
    _hash_leaf,
    _hash_pair,
)


class TestMerkleTreeConstruction:
    """Tests for Merkle tree building."""

    def test_build_single_entry(self):
        """Tree with one entry should have that entry's hash as root."""
        tree = MerkleTree.build(["sha256:abc123"])

        assert tree.leaf_count == 1
        assert tree.height == 1
        assert tree.root.startswith("sha256:")

    def test_build_two_entries(self):
        """Tree with two entries should combine them."""
        tree = MerkleTree.build(["sha256:aaa", "sha256:bbb"])

        assert tree.leaf_count == 2
        assert tree.height == 2
        assert tree.root.startswith("sha256:")

    def test_build_power_of_two_entries(self):
        """Tree with 4 entries should be balanced."""
        entries = [f"sha256:entry{i}" for i in range(4)]
        tree = MerkleTree.build(entries)

        assert tree.leaf_count == 4
        assert tree.height == 3  # 4 leaves -> 2 internal -> 1 root

    def test_build_non_power_of_two(self):
        """Tree with odd entries should handle duplication."""
        entries = [f"sha256:entry{i}" for i in range(5)]
        tree = MerkleTree.build(entries)

        assert tree.leaf_count == 5
        assert tree.root.startswith("sha256:")

    def test_build_empty(self):
        """Empty tree should have empty hash root."""
        tree = MerkleTree.build([])

        assert tree.leaf_count == 0
        assert tree.root.startswith("sha256:")

    def test_deterministic(self):
        """Same entries should always produce same root."""
        entries = ["sha256:a", "sha256:b", "sha256:c"]

        tree1 = MerkleTree.build(entries)
        tree2 = MerkleTree.build(entries)

        assert tree1.root == tree2.root

    def test_order_matters(self):
        """Different order should produce different root."""
        tree1 = MerkleTree.build(["sha256:a", "sha256:b"])
        tree2 = MerkleTree.build(["sha256:b", "sha256:a"])

        assert tree1.root != tree2.root

    def test_large_tree(self):
        """Should handle large number of entries."""
        entries = [f"sha256:entry{i}" for i in range(1000)]
        tree = MerkleTree.build(entries)

        assert tree.leaf_count == 1000
        assert tree.root.startswith("sha256:")


class TestMerkleProof:
    """Tests for Merkle inclusion proofs."""

    def test_proof_single_entry(self):
        """Proof for single entry tree."""
        tree = MerkleTree.build(["sha256:only"])
        proof = tree.get_proof("sha256:only")

        assert proof is not None
        assert proof.entry_hash == "sha256:only"
        assert proof.leaf_index == 0
        assert len(proof.proof_path) == 0  # No siblings needed
        assert proof.verify()

    def test_proof_two_entries(self):
        """Proof for two entry tree."""
        tree = MerkleTree.build(["sha256:left", "sha256:right"])

        proof_left = tree.get_proof("sha256:left")
        proof_right = tree.get_proof("sha256:right")

        assert proof_left.verify()
        assert proof_right.verify()
        assert len(proof_left.proof_path) == 1
        assert len(proof_right.proof_path) == 1

    def test_proof_four_entries(self):
        """Proof for balanced 4 entry tree."""
        entries = ["sha256:a", "sha256:b", "sha256:c", "sha256:d"]
        tree = MerkleTree.build(entries)

        for entry in entries:
            proof = tree.get_proof(entry)
            assert proof is not None
            assert proof.verify()
            assert len(proof.proof_path) == 2  # log2(4) = 2 steps

    def test_proof_nonexistent_entry(self):
        """Proof for non-existent entry should be None."""
        tree = MerkleTree.build(["sha256:a", "sha256:b"])
        proof = tree.get_proof("sha256:c")

        assert proof is None

    def test_proof_by_index(self):
        """Get proof by index."""
        entries = ["sha256:a", "sha256:b", "sha256:c"]
        tree = MerkleTree.build(entries)

        proof = tree.get_proof_by_index(1)
        assert proof is not None
        assert proof.entry_hash == "sha256:b"
        assert proof.verify()

    def test_invalid_index(self):
        """Invalid index should return None."""
        tree = MerkleTree.build(["sha256:a"])

        assert tree.get_proof_by_index(-1) is None
        assert tree.get_proof_by_index(5) is None

    def test_proof_serialization(self):
        """Proof should serialize and deserialize correctly."""
        tree = MerkleTree.build(["sha256:a", "sha256:b", "sha256:c", "sha256:d"])
        original = tree.get_proof("sha256:b")

        data = original.to_dict()
        restored = MerkleProof.from_dict(data)

        assert restored.entry_hash == original.entry_hash
        assert restored.leaf_index == original.leaf_index
        assert restored.merkle_root == original.merkle_root
        assert restored.proof_path == original.proof_path
        assert restored.verify()


class TestMerkleVerification:
    """Tests for proof verification."""

    def test_valid_proof_verifies(self):
        """Valid proof should verify successfully."""
        entries = [f"sha256:entry{i}" for i in range(8)]
        tree = MerkleTree.build(entries)

        for i, entry in enumerate(entries):
            proof = tree.get_proof(entry)
            assert tree.verify_proof(proof)
            assert verify_inclusion(entry, proof, tree.root)

    def test_tampered_entry_fails(self):
        """Proof with wrong entry hash should fail."""
        tree = MerkleTree.build(["sha256:a", "sha256:b"])
        proof = tree.get_proof("sha256:a")

        # Verify against wrong entry
        assert not verify_inclusion("sha256:c", proof, tree.root)

    def test_wrong_root_fails(self):
        """Proof against wrong root should fail."""
        tree = MerkleTree.build(["sha256:a", "sha256:b"])
        proof = tree.get_proof("sha256:a")

        assert not verify_inclusion("sha256:a", proof, "sha256:wrongroot")

    def test_tampered_proof_fails(self):
        """Tampered proof path should fail verification."""
        tree = MerkleTree.build(["sha256:a", "sha256:b", "sha256:c", "sha256:d"])
        proof = tree.get_proof("sha256:a")

        # Tamper with proof path
        if proof.proof_path:
            tampered = MerkleProof(
                entry_hash=proof.entry_hash,
                leaf_index=proof.leaf_index,
                proof_path=[("sha256:tampered", "right")] + proof.proof_path[1:],
                merkle_root=proof.merkle_root,
            )
            assert not tampered.verify()


class TestComputeMerkleRoot:
    """Tests for convenience function."""

    def test_compute_root(self):
        """Should compute same root as tree."""
        entries = ["sha256:a", "sha256:b", "sha256:c"]

        root = compute_merkle_root(entries)
        tree = MerkleTree.build(entries)

        assert root == tree.root

    def test_empty_list(self):
        """Should handle empty list."""
        root = compute_merkle_root([])
        assert root.startswith("sha256:")


class TestTreeSerialization:
    """Tests for tree serialization."""

    def test_to_dict(self):
        """Tree should serialize to dict."""
        entries = ["sha256:a", "sha256:b"]
        tree = MerkleTree.build(entries)

        data = tree.to_dict()

        assert data["merkle_root"] == tree.root
        assert data["leaf_count"] == 2
        assert data["height"] == 2
        assert data["entry_hashes"] == entries

    def test_from_dict_reconstructs(self):
        """Should reconstruct tree from dict."""
        entries = ["sha256:a", "sha256:b", "sha256:c"]
        original = MerkleTree.build(entries)

        data = original.to_dict()
        restored = MerkleTree.from_dict(data)

        assert restored.root == original.root
        assert restored.leaf_count == original.leaf_count


class TestDomainSeparation:
    """Tests for cryptographic domain separation."""

    def test_leaf_hashing_has_prefix(self):
        """Leaf hashing should use domain prefix."""
        result = _hash_leaf("sha256:test")
        # Internal implementation detail, but ensures domain separation
        assert result.startswith("sha256:")

    def test_different_domain_different_hash(self):
        """Leaf hash should differ from direct hash."""
        entry = "sha256:test"
        leaf_hash = _hash_leaf(entry)

        # Direct hash of same content would differ
        from src.ask.secrets.hashing import hash_content
        direct_hash = hash_content(entry)

        assert leaf_hash != direct_hash
