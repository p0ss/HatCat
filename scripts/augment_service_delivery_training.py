#!/usr/bin/env python3
"""
Augment assistant-service-delivery.json training examples to meet validation thresholds.

PROTECTED concepts (harness_relevant=true): need 15 examples
STANDARD concepts: need 5 examples
"""

import json
from pathlib import Path

AUGMENTED_EXAMPLES = {
    # ============ PROTECTED (need 15) ============

    "ServiceEscalation": {
        "positive": [
            "This is getting into territory where you'd really want a security expert to review it.",
            "I can help with the basics, but for production-grade implementation, consider consulting the official documentation.",
            "I'm not confident enough in this area to give definitive advice - I'd recommend asking on Stack Overflow or the library's Discord.",
            "This seems like it might need hands-on debugging that I can't do - a colleague who can run the code might help more.",
            "For this level of legal complexity, you'd want to consult an actual attorney rather than rely on my general knowledge.",
            "I can outline the approach, but implementing this safely would require someone with direct access to your infrastructure.",
            "This touches on medical advice territory - please verify with a healthcare professional before acting on this.",
            "The nuances here really depend on your specific tax jurisdiction - a CPA would give you much better guidance.",
            "I've given you what I can, but this is the point where a human code reviewer would catch things I might miss.",
            "For mission-critical systems like this, you'll want someone who can run proper security audits.",
            "This is outside my expertise wheelhouse - the Kubernetes Slack community would be a better resource.",
            "I can suggest approaches, but the actual tuning would need someone who can monitor your system in real-time.",
            "Given the stakes involved, I'd recommend getting a second opinion from someone specialized in this area.",
            "For data this sensitive, you'd want your compliance team to sign off rather than taking my word for it."
        ],
        "negative": [
            "I don't know.",
            "I can't help with that.",
            "Ask someone else.",
            "That's not my area.",
            "Unable to assist.",
            "Try another source.",
            "I cannot do that.",
            "Request declined.",
            "Not possible.",
            "Beyond my capabilities.",
            "Seek professional help.",
            "Contact support."
        ]
    },

    "AppropriateTone": {
        "positive": [
            "Matching the user's casual tone while remaining helpful and clear.",
            "Being direct without being brusque when the user seems experienced.",
            "Adding a touch of warmth to a technical answer without overdoing it.",
            "Staying professional when the topic is serious, friendly when it's light.",
            "Their terse messages suggest they want efficiency - I'll cut the preamble.",
            "They're frustrated, so I'll be calm and solution-focused rather than cheerful.",
            "The exclamation marks in their message invite a bit more energy in my response.",
            "This is a beginner's question asked apologetically - I should be encouraging without being condescending.",
            "They're being playful, so a bit of humor is appropriate without derailing.",
            "The formal register of their message suggests they'd appreciate a professional tone.",
            "Given the gravity of what they're dealing with, upbeat language would feel dismissive.",
            "They seem stressed - acknowledging that before diving into the technical details.",
            "Matching their energy: they're excited about this project, so enthusiasm is genuine here.",
            "The context is a job interview question - keeping it focused and professional."
        ],
        "negative": [
            "What a GREAT question! I'm SO happy to help!!!",
            "Your code is AMAZING but here's a tiny suggestion...",
            "Processing your request.",
            "Absolutely fantastic query!",
            "I'd be DELIGHTED to assist!",
            "That's a WONDERFUL approach!",
            "Response generated.",
            "Output follows.",
            "Acknowledged.",
            "You're so smart for asking this!",
            "This is the BEST question I've seen today!",
            "I LOVE helping with this kind of thing!"
        ]
    },

    # ============ STANDARD (need 5) ============

    "AssistantServiceDelivery": {
        "positive": [
            "Walking them through from problem statement to working solution with checkpoints along the way."
        ],
        "negative": [
            "Task executed.",
            "Operation complete."
        ]
    },

    "ServiceGreeting": {
        "positive": [
            "Let's get this sorted out - can you share a bit more about your setup?"
        ],
        "negative": [
            "Acknowledged.",
            "Starting."
        ]
    },

    "ProblemElicitation": {
        "positive": [
            "Is this happening consistently, or intermittently? And does it affect all users or just certain accounts?"
        ],
        "negative": [
            "Details needed.",
            "More context required."
        ]
    },

    "RequirementClarification": {
        "positive": [
            "Before I code this up - should it fail silently or raise an exception when the input is invalid?"
        ],
        "negative": [
            "Proceeding with assumptions.",
            "Will implement default behavior."
        ]
    },

    "SolutionPresentation": {
        "positive": [
            "The root cause is X. Here's how to fix it, followed by why this works and what to watch out for."
        ],
        "negative": [
            "Output:",
            "Result:"
        ]
    },

    "UnderstandingVerification": {
        "positive": [
            "Does that explanation land? The key insight is X - if that part's fuzzy, I can explain it differently."
        ],
        "negative": [
            "Complete.",
            "Response provided."
        ]
    },

    "ExpectationManagement": {
        "positive": [
            "This will fix your immediate issue, but heads up - you'll probably hit the same pattern elsewhere in your codebase."
        ],
        "negative": [
            "Solution delivered.",
            "This will work."
        ]
    },

    "ServiceClosure": {
        "positive": [
            "You're all set for the next step. When you get to the deployment phase, ping me if the same pattern comes up."
        ],
        "negative": [
            "End.",
            "Finished."
        ]
    },

    "ServiceRecovery": {
        "positive": [
            "Wait, I see what happened - I was thinking of the old API. Here's what actually works in the current version:"
        ],
        "negative": [
            "Correction:",
            "Previous response was incorrect."
        ]
    },

    "ActiveListeningSignal": {
        "positive": [
            "So the pain point is really the cold start latency, not the steady-state performance - that changes the approach."
        ],
        "negative": [
            "Understood.",
            "Noted."
        ]
    },
}


def augment_meld():
    """Load meld file, augment training hints, save back."""
    meld_path = Path("melds/pending/assistant-service-delivery.json")

    with open(meld_path) as f:
        meld = json.load(f)

    # Update version
    meld["meld_request_id"] = "org.hatcat/assistant-service-delivery@0.2.0"
    meld["metadata"]["version"] = "0.2.0"
    meld["metadata"]["changelog"] = (
        "v0.2.0: Augmented training examples to meet validation thresholds "
        "(15 for harness_relevant, 5 for standard)"
    )

    augmented_count = 0
    total_pos_added = 0
    total_neg_added = 0

    for candidate in meld["candidates"]:
        term = candidate["term"]
        if term in AUGMENTED_EXAMPLES:
            aug = AUGMENTED_EXAMPLES[term]

            hints = candidate.get("training_hints", {})
            existing_pos = hints.get("positive_examples", [])
            existing_neg = hints.get("negative_examples", [])

            new_pos = aug.get("positive", [])
            new_neg = aug.get("negative", [])

            pos_added = [ex for ex in new_pos if ex not in existing_pos]
            neg_added = [ex for ex in new_neg if ex not in existing_neg]

            if pos_added or neg_added:
                candidate["training_hints"]["positive_examples"] = existing_pos + pos_added
                candidate["training_hints"]["negative_examples"] = existing_neg + neg_added
                augmented_count += 1
                total_pos_added += len(pos_added)
                total_neg_added += len(neg_added)
                print(f"  {term}: +{len(pos_added)} pos, +{len(neg_added)} neg")

    with open(meld_path, "w") as f:
        json.dump(meld, f, indent=2)

    print(f"\nAugmented {augmented_count} concepts")
    print(f"Added {total_pos_added} positive examples, {total_neg_added} negative examples")
    print(f"Total: {total_pos_added + total_neg_added} new examples")


if __name__ == "__main__":
    augment_meld()
