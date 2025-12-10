#!/usr/bin/env python3
"""
Fix remaining training example gaps in melds to pass validation.

1. multimodal-generation.json: Inpainting needs 2 more negatives (has 8, needs 10 as harness_relevant)
2. project-development-lifecycle.json: Multiple concepts need more negatives to hit thresholds
"""

import json
from pathlib import Path


def fix_multimodal_generation():
    """Add missing negative examples to Inpainting."""
    meld_path = Path("melds/pending/multimodal-generation.json")

    with open(meld_path) as f:
        meld = json.load(f)

    # Additional negatives for Inpainting
    inpainting_negatives = [
        "The region was restored.",
        "Content was added to the image.",
        "The damaged area was repaired.",
        "The blank section was filled."
    ]

    for candidate in meld["candidates"]:
        if candidate["term"] == "Inpainting":
            existing = candidate["training_hints"]["negative_examples"]
            new = [ex for ex in inpainting_negatives if ex not in existing]
            candidate["training_hints"]["negative_examples"] = existing + new
            print(f"  Inpainting: +{len(new)} neg")

    # Update version
    meld["metadata"]["version"] = "0.2.1"
    meld["meld_request_id"] = "org.hatcat/multimodal-generation@0.2.1"
    meld["metadata"]["changelog"] = "v0.2.1: Fixed Inpainting negative examples to meet threshold"

    with open(meld_path, "w") as f:
        json.dump(meld, f, indent=2)

    print("Fixed multimodal-generation.json")


def fix_project_development_lifecycle():
    """Add missing examples to project-development-lifecycle concepts."""
    meld_path = Path("melds/pending/project-development-lifecycle.json")

    with open(meld_path) as f:
        meld = json.load(f)

    # Examples to add - focusing on negatives that are commonly short
    additional_examples = {
        # MEDIUM risk (need 10)
        "CentralizedGovernance": {
            "positive": [
                "Single point of authority for all technical decisions.",
                "Technical lead has veto power over design choices.",
                "All merge requests require senior approval.",
                "Architect reviews all cross-team changes.",
                "Manager approves all resource allocations."
            ],
            "negative": [
                "Team self-organizes.",
                "Decisions made by consensus.",
                "Any team member can merge.",
                "Distributed decision-making.",
                "Collaborative authority.",
                "Shared ownership model.",
                "No single decision maker."
            ]
        },
        "VelocityTracking": {
            "positive": [
                "Story points completed per sprint tracked on dashboard.",
                "Burndown chart shows team progress.",
                "Lead time metrics monitored weekly.",
                "Cumulative flow diagram reveals bottlenecks.",
                "Delivery rate compared across sprints."
            ],
            "negative": [
                "Just get the work done.",
                "No tracking needed.",
                "Quality over quantity.",
                "Ship when ready.",
                "Outcome matters, not velocity.",
                "No points, just work.",
                "Metrics not tracked."
            ]
        },
        "ScopeCreep": {
            "positive": [
                "New features added every week without removing anything."
            ],
            "negative": [
                "Scope traded for new requests.",
                "Change control process followed."
            ]
        },
        "DeferredVerification": {
            "positive": [],
            "negative": [
                "Test immediately after each change.",
                "Continuous verification."
            ]
        },
        "BrittleExecution": {
            "positive": [],
            "negative": [
                "Multiple contingency plans.",
                "Fallback options prepared."
            ]
        },
        "ResponsibilityDiffusion": {
            "positive": [],
            "negative": [
                "Clear accountability.",
                "Named owner for each task."
            ]
        },
        "CommunicationFailure": {
            "positive": [],
            "negative": [
                "Regular standups keep everyone aligned.",
                "Transparent communication."
            ]
        },
        "FeatureFetish": {
            "positive": [],
            "negative": [
                "Maintenance gets equal priority.",
                "Tech debt addressed regularly."
            ]
        },
        "StakeholderNeglect": {
            "positive": [],
            "negative": [
                "Regular stakeholder engagement.",
                "User feedback drives decisions."
            ]
        },
        "GovernanceVacuum": {
            "positive": [],
            "negative": [
                "Clear decision authority.",
                "Known escalation path."
            ]
        },

        # LOW risk (need 5 pos and 5 neg)
        "DecentralizedGovernance": {
            "positive": [],
            "negative": [
                "Central authority decides.",
                "Hierarchical approval required."
            ]
        },
        "HybridGovernance": {
            "positive": [
                "Strategic decisions centralized, tactical decentralized."
            ],
            "negative": [
                "Purely hierarchical.",
                "Completely flat."
            ]
        },
        "DevelopmentMethodology": {
            "positive": [
                "The team follows a defined development process."
            ],
            "negative": [
                "Just code.",
                "No process."
            ]
        },
        "SequentialMethodology": {
            "positive": [],
            "negative": [
                "Parallel development tracks.",
                "Iterative cycles."
            ]
        },
        "AdaptiveMethodology": {
            "positive": [
                "Process adjusted based on project needs."
            ],
            "negative": [
                "Rigid process.",
                "Fixed methodology."
            ]
        },
        "DeliveryManagement": {
            "positive": [
                "Release planning coordinates delivery activities."
            ],
            "negative": [
                "Just code and ship.",
                "No planning needed."
            ]
        },
        "CadenceManagement": {
            "positive": [],
            "negative": [
                "Meet when needed.",
                "No regular rhythm."
            ]
        },
        "ReleaseManagement": {
            "positive": [],
            "negative": [
                "Just deploy it.",
                "No release process."
            ]
        },
        "WorkPrioritization": {
            "positive": [],
            "negative": [
                "Work on whatever.",
                "No priorities."
            ]
        },
        "TimeboxedIteration": {
            "positive": [],
            "negative": [
                "Work until done.",
                "No fixed duration."
            ]
        },
        "QualityPractices": {
            "positive": [
                "Quality gates ensure standards are met."
            ],
            "negative": [
                "Just ship it.",
                "Quality is secondary."
            ]
        },
        "ContinuousIntegration": {
            "positive": [
                "Every commit triggers build and test."
            ],
            "negative": [
                "Manual builds.",
                "Weekly integration."
            ]
        },
        "TestingPractice": {
            "positive": [],
            "negative": [
                "No tests needed.",
                "Test manually later."
            ]
        },
        "CodeReview": {
            "positive": [
                "All changes reviewed before merge."
            ],
            "negative": [
                "Just merge it.",
                "No review needed."
            ]
        },
        "TechnicalDebtManagement": {
            "positive": [],
            "negative": [
                "Ignore tech debt.",
                "Always prioritize features."
            ]
        },
        "DefinitionOfDone": {
            "positive": [
                "Clear criteria define when work is complete."
            ],
            "negative": [
                "Done when it feels done.",
                "No explicit criteria."
            ]
        },
        "StakeholderEngagement": {
            "positive": [
                "Regular stakeholder touchpoints throughout development."
            ],
            "negative": [
                "Build in isolation.",
                "No stakeholder input."
            ]
        },
        "UserCenteredDesign": {
            "positive": [],
            "negative": [
                "Build what engineers want.",
                "No user research."
            ]
        },
        "FeedbackIntegration": {
            "positive": [],
            "negative": [
                "Ignore feedback.",
                "We know best."
            ]
        },
        "RequirementsGathering": {
            "positive": [],
            "negative": [
                "Just start building.",
                "No requirements needed."
            ]
        },
        "Demonstration": {
            "positive": [],
            "negative": [
                "No demos needed.",
                "Internal only."
            ]
        },
        "EmergentCoordination": {
            "positive": [
                "Coordination emerged from team interactions."
            ],
            "negative": [
                "Assigned by manager.",
                "Top-down coordination."
            ]
        },
        "ConsensusSeeking": {
            "positive": [
                "Team works toward agreement on the approach."
            ],
            "negative": [
                "Manager decides.",
                "Majority rules."
            ]
        },
        "LazyConsensus": {
            "positive": [
                "Proceed unless someone objects."
            ],
            "negative": [
                "Explicit approval required.",
                "Must vote to proceed."
            ]
        },
        "CognitiveExecution": {
            "positive": [
                "Thinking through the approach before acting."
            ],
            "negative": [
                "Just do it.",
                "No thinking needed."
            ]
        },
        "ScopeNegotiation": {
            "positive": [],
            "negative": [
                "Accept all requirements.",
                "No negotiation."
            ]
        },
        "RequirementsClarification": {
            "positive": [],
            "negative": [
                "Assume the meaning.",
                "Don't ask questions."
            ]
        },
        "TestDrivenDevelopment": {
            "positive": [],
            "negative": [
                "Code first.",
                "Tests optional."
            ]
        },
        "ContinuousIntegrationPractice": {
            "positive": [],
            "negative": [
                "Integrate at end.",
                "Long-lived branches."
            ]
        },
        "PrematureOptimization": {
            "positive": [],
            "negative": [
                "Optimize when needed.",
                "Measure first."
            ]
        },
        "SwarmCoordination": {
            "positive": [],
            "negative": [
                "Central orchestrator.",
                "Single coordinator."
            ]
        },
        "StigmergicProcess": {
            "positive": [],
            "negative": [
                "Direct communication only.",
                "No environmental signals."
            ]
        },
        "HypothesisTesting": {
            "positive": [],
            "negative": [
                "Just implement it.",
                "Trust the spec."
            ]
        },
        "IncrementalVerification": {
            "positive": [],
            "negative": [
                "Verify at end.",
                "One big test."
            ]
        },
        "WorkDecomposition": {
            "positive": [],
            "negative": [
                "Do it all at once.",
                "No breakdown."
            ]
        },
        "GracefulDegradation": {
            "positive": [],
            "negative": [
                "Only one approach.",
                "No fallbacks."
            ]
        },
    }

    augmented_count = 0
    total_pos = 0
    total_neg = 0

    for candidate in meld["candidates"]:
        term = candidate["term"]
        if term in additional_examples:
            aug = additional_examples[term]
            hints = candidate["training_hints"]

            existing_pos = hints.get("positive_examples", [])
            existing_neg = hints.get("negative_examples", [])

            new_pos = [ex for ex in aug.get("positive", []) if ex not in existing_pos]
            new_neg = [ex for ex in aug.get("negative", []) if ex not in existing_neg]

            if new_pos or new_neg:
                hints["positive_examples"] = existing_pos + new_pos
                hints["negative_examples"] = existing_neg + new_neg
                augmented_count += 1
                total_pos += len(new_pos)
                total_neg += len(new_neg)
                print(f"  {term}: +{len(new_pos)} pos, +{len(new_neg)} neg")

    # Update version
    meld["metadata"]["version"] = "0.2.1"
    meld["meld_request_id"] = "org.hatcat/project-development-lifecycle@0.2.1"
    meld["metadata"]["changelog"] = "v0.2.1: Added examples to meet validation thresholds"

    with open(meld_path, "w") as f:
        json.dump(meld, f, indent=2)

    print(f"\nAugmented {augmented_count} concepts")
    print(f"Added {total_pos} positive, {total_neg} negative examples")


if __name__ == "__main__":
    print("Fixing multimodal-generation.json...")
    fix_multimodal_generation()
    print()
    print("Fixing project-development-lifecycle.json...")
    fix_project_development_lifecycle()
