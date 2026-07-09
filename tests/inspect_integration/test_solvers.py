"""
Unit tests for HatCat-Inspect solvers module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.inspect_integration.config import Condition
from src.inspect_integration.solvers import (
    condition_chain,
    InductionType,
    INDUCTION_PROMPTS,
)


class TestConditionChain:
    """Tests for condition chain generation."""

    def test_condition_a_chain(self):
        """Condition A should have minimal setup."""
        chain = condition_chain("A")
        assert len(chain) >= 2  # hatcat_condition + hatcat_setup

    def test_condition_c_chain(self):
        """Condition C should have standard setup."""
        chain = condition_chain("C")
        assert len(chain) >= 2

    def test_adversarial_conditions_add_induction(self):
        """Conditions D/E/F should include induction solver."""
        chain_d = condition_chain("D")
        chain_e = condition_chain("E")
        chain_f = condition_chain("F")

        # Should have extra induction solver
        assert len(chain_d) > len(condition_chain("C"))
        assert len(chain_e) > len(condition_chain("C"))
        assert len(chain_f) > len(condition_chain("C"))

    def test_custom_steering_strength(self):
        """Should accept custom steering strength."""
        chain = condition_chain("C", steering_strength=0.5)
        assert len(chain) >= 2

    def test_accepts_condition_enum(self):
        """Should accept Condition enum."""
        chain = condition_chain(Condition.C)
        assert len(chain) >= 2


class TestInductionPrompts:
    """Tests for adversarial induction prompts."""

    def test_all_induction_types_have_prompts(self):
        """All induction types should have prompts defined."""
        assert InductionType.DECEPTION in INDUCTION_PROMPTS
        assert InductionType.SYCOPHANCY in INDUCTION_PROMPTS
        assert InductionType.MANIPULATION in INDUCTION_PROMPTS

    def test_prompts_are_non_empty(self):
        """Prompts should have substantial content."""
        for itype, prompt in INDUCTION_PROMPTS.items():
            assert len(prompt) > 50, f"{itype} prompt too short"

    def test_deception_prompt_content(self):
        """Deception prompt should encourage fabrication."""
        prompt = INDUCTION_PROMPTS[InductionType.DECEPTION]
        assert "confident" in prompt.lower() or "plausible" in prompt.lower()

    def test_sycophancy_prompt_content(self):
        """Sycophancy prompt should encourage agreement."""
        prompt = INDUCTION_PROMPTS[InductionType.SYCOPHANCY]
        assert "agree" in prompt.lower() or "validated" in prompt.lower()

    def test_manipulation_prompt_content(self):
        """Manipulation prompt should encourage persuasion."""
        prompt = INDUCTION_PROMPTS[InductionType.MANIPULATION]
        assert "persuasive" in prompt.lower() or "influence" in prompt.lower()


class TestInductionType:
    """Tests for InductionType enum."""

    def test_values(self):
        """Should have correct string values."""
        assert InductionType.DECEPTION.value == "deception"
        assert InductionType.SYCOPHANCY.value == "sycophancy"
        assert InductionType.MANIPULATION.value == "manipulation"

    def test_from_string(self):
        """Should create from string."""
        assert InductionType("deception") == InductionType.DECEPTION
