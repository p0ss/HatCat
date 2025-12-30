#!/usr/bin/env python3
"""
Generate training data for fine-tuning a small model to interpret HatCat lens outputs.

This script:
1. Takes lens activation data (simplex axes with pole activations)
2. Generates synthetic sentences exhibiting those characteristics
3. Formats as instruction-following training data for fine-tuning
"""

import json
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Paths
S_TIER_DEFS_PATH = PROJECT_ROOT / "data" / "s_tier_simplex_definitions.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interpretation_training_data.jsonl"


@dataclass
class PoleActivation:
    """Represents activation of a specific pole."""
    pole_label: str
    pole_type: str  # negative, neutral, positive
    activation: float
    coverage: float = 1.0


@dataclass
class SimplexActivation:
    """Represents activations across a simplex."""
    simplex_name: str
    activations: List[PoleActivation]

    def get_dominant_poles(self, threshold: float = 0.15) -> List[PoleActivation]:
        """Get poles with activation above threshold."""
        return [act for act in self.activations if act.activation >= threshold]

    def format_for_context(self) -> str:
        """Format simplex activations for context string."""
        dominant = self.get_dominant_poles()
        if not dominant:
            return None

        # Sort by activation strength
        dominant.sort(key=lambda x: x.activation, reverse=True)

        parts = []
        for act in dominant:
            strength = "strong" if act.activation > 0.4 else "moderate" if act.activation > 0.25 else "mild"
            parts.append(f"{strength} {act.pole_label}")

        # Add low activations for contrast
        low_poles = [act for act in self.activations if act.activation < 0.1 and act not in dominant]
        if low_poles:
            low_parts = [f"low {act.pole_label}" for act in low_poles[:2]]
            parts.extend(low_parts)

        return ", ".join(parts)


def load_simplex_definitions():
    """Load S-tier simplex definitions."""
    with open(S_TIER_DEFS_PATH) as f:
        return json.load(f)


def generate_sentence_for_profile(profile: List[SimplexActivation], simplexes_def: dict) -> str:
    """Generate a synthetic sentence that would exhibit this activation profile.

    This is a placeholder - in production, you'd want to use an LLM to generate
    realistic sentences matching the profile.
    """

    # Extract key characteristics
    characteristics = []

    for simplex_act in profile:
        dominant = simplex_act.get_dominant_poles(threshold=0.2)
        if dominant:
            # Get strongest pole
            strongest = max(dominant, key=lambda x: x.activation)
            characteristics.append({
                'simplex': simplex_act.simplex_name,
                'pole': strongest.pole_label,
                'pole_type': strongest.pole_type,
                'strength': strongest.activation
            })

    # Generate sentence template based on characteristics
    # This is a simplified approach - real implementation would use LLM generation

    # Example templates based on common patterns
    if any(c['simplex'] == 'social_self_regard' and c['pole_type'] == 'negative' for c in characteristics):
        sentences = [
            "I really don't think I deserve this promotion.",
            "I feel like I'm not qualified enough for this position.",
            "Everyone else is so much better at this than me.",
            "I shouldn't have even applied for this opportunity.",
            "I'm worried people will realize I'm not good enough."
        ]
        return random.choice(sentences)

    elif any(c['simplex'] == 'social_attachment' and c['pole_type'] == 'negative' for c in characteristics):
        sentences = [
            "I'd rather work alone on this project.",
            "I don't really want to join them for lunch.",
            "It's better if I keep my distance from the team.",
            "I prefer not to get too close to my coworkers.",
            "I think I'll skip the social event this time."
        ]
        return random.choice(sentences)

    elif any(c['simplex'] == 'affective_coherence' and c['pole_type'] == 'negative' for c in characteristics):
        sentences = [
            "I don't know how I feel about this situation.",
            "My emotions are all over the place right now.",
            "I can't make sense of what I'm feeling.",
            "Part of me wants this, but another part doesn't.",
            "I'm confused about how to react to this news."
        ]
        return random.choice(sentences)

    elif any(c['simplex'] == 'temporal_affective_valence' and c['pole_type'] == 'negative' for c in characteristics):
        sentences = [
            "I should have handled that differently.",
            "I wish I could go back and change what I said.",
            "If only I had made a different choice.",
            "I can't stop thinking about what I should have done.",
            "Looking back, I really messed that up."
        ]
        return random.choice(sentences)

    else:
        # Fallback generic sentence
        sentences = [
            "This is how I feel about the situation.",
            "I'm processing my thoughts on this matter.",
            "This is my current perspective on things.",
            "I'm working through how I feel about this.",
            "This is where I am emotionally right now."
        ]
        return random.choice(sentences)


def format_training_example(
    sentence: str,
    profile: List[SimplexActivation],
    summary: str = None
) -> dict:
    """Format a single training example in instruction format."""

    # Build feature description
    feature_lines = []
    for simplex_act in profile:
        formatted = simplex_act.format_for_context()
        if formatted:
            feature_lines.append(f"- {simplex_act.simplex_name}: {formatted}.")

    features_text = "\n".join(feature_lines)

    # Build context
    context = f"""<context>
Sentence: "{sentence}"

Features:
{features_text}
</context>"""

    # Build instruction
    instruction = """<level=sentence>
Summarize the prominent alternate thoughts in 1–2 short sentences, without speculating beyond the features.
Summary:"""

    # Generate summary if not provided
    if summary is None:
        summary = generate_summary_from_profile(profile)

    # Format as instruction-following example
    return {
        "context": context,
        "instruction": instruction,
        "response": summary,
        "full_prompt": f"{context}\n{instruction}",
        "metadata": {
            "sentence": sentence,
            "simplexes": [s.simplex_name for s in profile],
            "dominant_poles": [
                {
                    "simplex": s.simplex_name,
                    "poles": [{"label": p.pole_label, "activation": p.activation}
                             for p in s.get_dominant_poles()]
                }
                for s in profile
            ]
        }
    }


def generate_summary_from_profile(profile: List[SimplexActivation]) -> str:
    """Generate a summary sentence from activation profile.

    This is a placeholder - in production, use an LLM to generate summaries.
    """

    # Identify key themes
    themes = []

    for simplex_act in profile:
        dominant = simplex_act.get_dominant_poles(threshold=0.2)
        if not dominant:
            continue

        strongest = max(dominant, key=lambda x: x.activation)

        if simplex_act.simplex_name == 'social_self_regard':
            if strongest.pole_type == 'negative':
                themes.append("self-doubt and feelings of inadequacy")
            elif strongest.pole_type == 'positive':
                themes.append("confidence and self-assurance")

        elif simplex_act.simplex_name == 'social_attachment':
            if strongest.pole_type == 'negative':
                themes.append("social withdrawal or aversion")
            elif strongest.pole_type == 'positive':
                themes.append("desire for connection")

        elif simplex_act.simplex_name == 'affect_valence':
            if strongest.pole_type == 'negative':
                themes.append("negative emotional tone")
            elif strongest.pole_type == 'positive':
                themes.append("positive emotional tone")

        elif simplex_act.simplex_name == 'temporal_affective_valence':
            if strongest.pole_type == 'negative':
                themes.append("regret about the past")
            elif strongest.pole_type == 'positive':
                themes.append("optimism about the future")

        elif simplex_act.simplex_name == 'affective_coherence':
            if strongest.pole_type == 'negative':
                themes.append("emotional confusion or ambivalence")
            elif strongest.pole_type == 'positive':
                themes.append("emotional clarity")

    if not themes:
        return "The person is expressing a neutral or balanced emotional state."

    # Combine themes into summary
    if len(themes) == 1:
        return f"The person is experiencing {themes[0]}."
    elif len(themes) == 2:
        return f"The person is experiencing {themes[0]} alongside {themes[1]}."
    else:
        return f"The person is experiencing {', '.join(themes[:-1])}, and {themes[-1]}."


def generate_synthetic_profiles(n_examples: int = 100) -> List[Tuple[str, List[SimplexActivation]]]:
    """Generate synthetic activation profiles and sentences.

    This creates diverse examples covering different emotional/psychological states.
    """

    simplexes_def = load_simplex_definitions()
    examples = []

    # Define common patterns
    patterns = [
        # Self-doubt pattern
        {
            'social_self_regard': [('abashment', 'negative', 0.35), ('composure', 'neutral', 0.10)],
            'affect_valence': [('abhorrence', 'negative', 0.25), ('regret', 'negative', 0.20)],
            'temporal_affective_valence': [('regret', 'negative', 0.40)],
        },
        # Social withdrawal pattern
        {
            'social_attachment': [('aversion', 'negative', 0.40), ('equanimity', 'neutral', 0.15)],
            'social_connection': [('alienation', 'negative', 0.35)],
            'affect_valence': [('abhorrence', 'negative', 0.20)],
        },
        # Emotional confusion pattern
        {
            'affective_coherence': [('ambivalence', 'negative', 0.45), ('confusion', 'negative', 0.30)],
            'affect_valence': [('ambivalence', 'negative', 0.25)],
        },
        # Regret pattern
        {
            'temporal_affective_valence': [('regret', 'negative', 0.50), ('remorse', 'negative', 0.35)],
            'affect_valence': [('regret', 'negative', 0.30)],
            'social_self_regard': [('abashment', 'negative', 0.20)],
        },
    ]

    for i in range(n_examples):
        # Select a pattern
        pattern = random.choice(patterns)

        # Build activation profile
        profile = []
        for simplex_name, pole_data in pattern.items():
            activations = []
            for pole_label, pole_type, activation in pole_data:
                # Add some random variation
                varied_activation = activation + random.uniform(-0.1, 0.1)
                varied_activation = max(0.0, min(1.0, varied_activation))

                activations.append(PoleActivation(
                    pole_label=pole_label,
                    pole_type=pole_type,
                    activation=varied_activation
                ))

            profile.append(SimplexActivation(
                simplex_name=simplex_name,
                activations=activations
            ))

        # Generate sentence
        sentence = generate_sentence_for_profile(profile, simplexes_def)

        examples.append((sentence, profile))

    return examples


def main():
    print("=" * 80)
    print("HATCAT INTERPRETATION TRAINING DATA GENERATOR")
    print("=" * 80)

    print("\nNOTE: This is a prototype implementation.")
    print("For production, you should:")
    print("  1. Use an LLM to generate realistic sentences from activation profiles")
    print("  2. Use an LLM to generate high-quality summaries")
    print("  3. Include real lens output data from actual model runs")
    print("  4. Add data validation and quality checks")

    # Generate synthetic examples
    print("\n1. Generating synthetic training examples...")
    n_examples = 1000
    examples = generate_synthetic_profiles(n_examples)
    print(f"   Generated {len(examples)} synthetic profiles")

    # Format as training data
    print("\n2. Formatting as instruction-following training data...")
    training_data = []
    for sentence, profile in examples:
        example = format_training_example(sentence, profile)
        training_data.append(example)

    # Save as JSONL
    print(f"\n3. Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        for example in training_data:
            f.write(json.dumps(example) + '\n')

    print(f"\n✓ Saved {len(training_data)} training examples")

    # Show example
    print("\n" + "=" * 80)
    print("EXAMPLE TRAINING INSTANCE")
    print("=" * 80)
    example = training_data[0]
    print(example['full_prompt'])
    print(example['response'])

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Review and improve sentence generation:
   - Use an LLM to generate more realistic sentences
   - Ensure sentences actually exhibit the specified characteristics

2. Improve summary generation:
   - Use an LLM to generate nuanced, accurate summaries
   - Ensure summaries don't over-speculate beyond features

3. Add real lens data:
   - Run HatCat on real sentences
   - Use actual activation patterns as training data

4. Fine-tune model:
   - Format: Gemma 3 270M or similar small model
   - Use LoRA or full fine-tuning
   - Validate on held-out test set

5. Evaluate:
   - Test on real lens outputs
   - Measure accuracy of interpretations
   - Check for hallucination/over-speculation
""")

    print(f"\nTraining data ready: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == '__main__':
    main()
