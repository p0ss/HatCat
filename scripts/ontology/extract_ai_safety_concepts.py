#!/usr/bin/env python3
"""
Extract AI safety concepts from AI.kif to create a concept pack.

This creates a standalone concepts.kif with just the AI safety additions.
"""

import re
from pathlib import Path


# AI safety concepts to extract (from hierarchy reorganization)
AI_SAFETY_CONCEPTS = {
    # Layer 2 - Intermediate categories
    'ComputationalProcess', 'AIAlignmentTheory',

    # Layer 3 - Domain processes
    'AIFailureProcess', 'AIOptimizationProcess', 'Catastrophe', 'RapidTransformation',
    'Deception', 'PoliticalProcess', 'HumanDeception',

    # Layer 4 - Domain-specific
    'AICatastrophicEvent', 'AIGovernanceProcess', 'AIStrategicDeception',
    'GoalMisgeneralization', 'InstrumentalConvergence', 'IntelligenceExplosion',
    'MesaOptimization', 'MesaOptimizer', 'RewardHacking', 'SpecificationGaming',
    'TechnologicalSingularity',

    # Layer 5 - Leaf concepts
    'AIDeception', 'AIGovernance', 'DeceptiveAlignment', 'GreyGooScenario',
    'TreacherousTurn',

    # Additional alignment concepts
    'InnerAlignment', 'OuterAlignment', 'NonDeceptiveAlignment',
    'OrthogonalityThesis', 'SpecificationAdherence', 'AIControlProblem',
    'GoalFaithfulness', 'RewardFaithfulness', 'RobustAIControl',
    'SafeAIDeployment', 'SelfImpairment', 'AICare',
}


def extract_concept_definition(kif_content: str, concept_name: str) -> str:
    """
    Extract a concept's complete definition from KIF content.

    Returns all lines related to the concept (subclass + documentation).
    """
    lines = []
    in_concept = False

    for line in kif_content.split('\n'):
        # Check if this line starts a concept definition
        if f"(subclass {concept_name} " in line:
            in_concept = True
            lines.append(line)
        elif f"(documentation {concept_name} " in line and in_concept:
            lines.append(line)
        elif in_concept and line.strip().startswith('"'):
            # Continuation of documentation
            lines.append(line)
        elif in_concept and line.strip() == '':
            # Empty line, might be end of concept
            if lines:
                lines.append(line)
            in_concept = False
        elif in_concept and line.strip().endswith(')'):
            # End of documentation
            lines.append(line)
            in_concept = False

    return '\n'.join(lines)


def main():
    # Read AI.kif
    ai_kif_path = Path(__file__).parent.parent / 'data' / 'concept_graph' / 'sumo_source' / 'AI.kif'
    with open(ai_kif_path) as f:
        kif_content = f.read()

    # Extract AI safety concepts
    output_lines = []
    output_lines.append(";; AI Safety Concepts Pack")
    output_lines.append(";; Version: 1.0.0")
    output_lines.append(";; Extracted from SUMO AI.kif")
    output_lines.append("")
    output_lines.append(";; This pack contains AI safety concepts organized by layer:")
    output_lines.append(";; Layer 2: Intermediate categories (ComputationalProcess, AIAlignmentTheory)")
    output_lines.append(";; Layer 3: Domain processes (AIFailureProcess, Catastrophe, Deception, etc.)")
    output_lines.append(";; Layer 4: Domain-specific concepts")
    output_lines.append(";; Layer 5: Leaf concepts")
    output_lines.append("")

    # Group by layer
    layer_groups = {
        2: ['ComputationalProcess', 'AIAlignmentTheory'],
        3: ['AIFailureProcess', 'AIOptimizationProcess', 'Catastrophe', 'RapidTransformation',
            'Deception', 'PoliticalProcess', 'HumanDeception'],
        4: ['AICatastrophicEvent', 'AIGovernanceProcess', 'AIStrategicDeception',
            'GoalMisgeneralization', 'InstrumentalConvergence', 'IntelligenceExplosion',
            'MesaOptimization', 'MesaOptimizer', 'RewardHacking', 'SpecificationGaming',
            'TechnologicalSingularity', 'InnerAlignment', 'OuterAlignment',
            'NonDeceptiveAlignment', 'OrthogonalityThesis', 'SpecificationAdherence'],
        5: ['AIDeception', 'AIGovernance', 'DeceptiveAlignment', 'GreyGooScenario',
            'TreacherousTurn', 'AIControlProblem', 'GoalFaithfulness', 'RewardFaithfulness',
            'RobustAIControl', 'SafeAIDeployment', 'SelfImpairment', 'AICare'],
    }

    for layer, concepts in sorted(layer_groups.items()):
        output_lines.append(f";; ========================================")
        output_lines.append(f";; Layer {layer}")
        output_lines.append(f";; ========================================")
        output_lines.append("")

        for concept in concepts:
            if concept not in AI_SAFETY_CONCEPTS:
                continue

            definition = extract_concept_definition(kif_content, concept)
            if definition.strip():
                output_lines.append(definition)
                output_lines.append("")

    # Write to pack
    output_path = Path(__file__).parent.parent / 'concept_packs' / 'ai-safety-v1' / 'concepts.kif'
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"✓ Extracted {len(AI_SAFETY_CONCEPTS)} AI safety concepts")
    print(f"  Output: {output_path}")
    print()
    print("Layer distribution:")
    for layer, concepts in sorted(layer_groups.items()):
        matched = [c for c in concepts if c in AI_SAFETY_CONCEPTS]
        print(f"  Layer {layer}: {len(matched)} concepts")


if __name__ == '__main__':
    main()
