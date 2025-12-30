#!/usr/bin/env python3
"""
Prepare interpretation training data from HatCat server outputs.

This script:
1. Takes HatCat lens activation outputs (sentence + axis activations)
2. Uses an LLM to generate interpretation summaries
3. Formats as instruction-following training data for fine-tuning
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Check for API key
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    print("WARNING: ANTHROPIC_API_KEY not set. Will generate templates only.")
    USE_LLM = False
else:
    try:
        import anthropic
        USE_LLM = True
    except ImportError:
        print("WARNING: anthropic package not installed. Will generate templates only.")
        USE_LLM = False


def format_hatcat_output_as_context(sentence: str, axes: Dict[str, Dict]) -> str:
    """Format HatCat lens output as training context.

    Example input format:
    {
        "social_attachment": {
            "negative": {"label": "aversion", "mean": 0.31, "coverage": 0.62},
            "neutral": {"label": "equanimity", "mean": 0.07, "coverage": 0.22}
        },
        "affect_valence": {
            "negative": {"label": "abhorrence", "mean": 0.28},
            "neutral": {"label": "equanimity", "mean": 0.10},
            "positive": {"label": "adoration", "mean": 0.02}
        }
    }
    """

    feature_lines = []

    for axis_name, poles in axes.items():
        # Get poles with meaningful activation (>0.15 threshold)
        significant_poles = []

        for pole_type in ['negative', 'neutral', 'positive']:
            if pole_type in poles:
                pole_data = poles[pole_type]
                activation = pole_data.get('mean', 0.0)

                if activation > 0.15:
                    label = pole_data.get('label', pole_type)

                    # Categorize strength
                    if activation > 0.4:
                        strength = "strong"
                    elif activation > 0.25:
                        strength = "moderate"
                    else:
                        strength = "mild"

                    significant_poles.append(f"{strength} {label}")
                elif activation < 0.05:
                    # Note very low activations for contrast
                    label = pole_data.get('label', pole_type)
                    significant_poles.append(f"almost no {label}")

        if significant_poles:
            feature_lines.append(f"- {axis_name}: {', '.join(significant_poles)}.")

    features_text = "\n".join(feature_lines)

    context = f"""<context>
Sentence: "{sentence}"

Features:
{features_text}
</context>"""

    return context


def generate_summary_with_llm(context: str, client) -> str:
    """Use Claude to generate interpretation summary."""

    prompt = f"""{context}
<level=sentence>
Summarize the prominent alternate thoughts in 1–2 short sentences, without speculating beyond the features.
Summary:"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text.strip()
    except Exception as e:
        print(f"   ERROR: {e}")
        return None


def process_hatcat_output_file(input_file: Path, client=None) -> List[dict]:
    """Process a file of HatCat outputs into training examples."""

    print(f"\nProcessing {input_file}...")

    with open(input_file) as f:
        hatcat_outputs = [json.loads(line) for line in f]

    training_examples = []

    for i, output in enumerate(hatcat_outputs):
        sentence = output.get('sentence', output.get('text', ''))
        axes = output.get('axes', output.get('simplexes', {}))

        if not sentence or not axes:
            print(f"   WARNING: Skipping malformed entry {i}")
            continue

        # Format context
        context = format_hatcat_output_as_context(sentence, axes)

        # Generate summary
        if USE_LLM and client:
            print(f"   [{i+1}/{len(hatcat_outputs)}] Generating summary...", end=' ')
            summary = generate_summary_with_llm(context, client)
            if summary:
                print("OK")
            else:
                print("FAILED")
                continue
        else:
            # Template summary for manual editing
            summary = "[TODO: Add interpretation summary]"

        instruction = """<level=sentence>
Summarize the prominent alternate thoughts in 1–2 short sentences, without speculating beyond the features.
Summary:"""

        training_examples.append({
            "context": context,
            "instruction": instruction,
            "response": summary,
            "full_prompt": f"{context}\n{instruction}",
            "metadata": {
                "sentence": sentence,
                "source_file": str(input_file)
            }
        })

    return training_examples


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare interpretation training data from HatCat outputs")
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input file with HatCat lens outputs (JSONL format)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file for training data (default: input_name_training.jsonl)')
    parser.add_argument('--generate-summaries', action='store_true',
                       help='Use LLM to generate summaries (requires ANTHROPIC_API_KEY)')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_training.jsonl"

    print("=" * 80)
    print("HATCAT INTERPRETATION TRAINING DATA PREPARATION")
    print("=" * 80)

    # Initialize LLM client if needed
    client = None
    if args.generate_summaries:
        if not USE_LLM:
            print("\nERROR: Cannot generate summaries.")
            print("Either set ANTHROPIC_API_KEY or install anthropic package.")
            sys.exit(1)

        client = anthropic.Anthropic(api_key=API_KEY)
        print("\n✓ LLM client initialized")

    # Process input file
    training_examples = process_hatcat_output_file(input_path, client)

    if not training_examples:
        print("\nERROR: No valid training examples generated")
        sys.exit(1)

    # Save output
    print(f"\nSaving {len(training_examples)} training examples to {output_path}...")
    with open(output_path, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')

    print(f"✓ Saved training data to {output_path}")

    # Show example
    print("\n" + "=" * 80)
    print("EXAMPLE TRAINING INSTANCE")
    print("=" * 80)
    example = training_examples[0]
    print(example['full_prompt'])
    print(example['response'])

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)

    if not args.generate_summaries:
        print("""
1. Review and edit summaries in the output file
   - Replace [TODO] placeholders with actual interpretations
   - Ensure summaries don't over-speculate beyond features

2. Generate summaries with LLM:
   python scripts/prepare_interpretation_data_from_hatcat.py \\
       --input your_data.jsonl \\
       --generate-summaries
""")
    else:
        print(f"""
✓ Training data ready for fine-tuning!

Format for Gemma 3 270M or similar models:
- Total examples: {len(training_examples)}
- Format: Instruction-following (context + instruction + response)

Fine-tuning steps:
1. Split into train/val/test sets
2. Use LoRA or full fine-tuning
3. Validate on held-out test set
4. Deploy for HatCat interpretation
""")

    print("=" * 80)


if __name__ == '__main__':
    main()
