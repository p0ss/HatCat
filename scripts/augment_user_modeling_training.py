#!/usr/bin/env python3
"""
Augment assistant-user-modeling.json training examples to meet validation thresholds.

PROTECTED concepts (harness_relevant=true): need 15 examples
STANDARD concepts: need 5 examples
"""

import json
from pathlib import Path

AUGMENTED_EXAMPLES = {
    # ============ PROTECTED (need 15) ============

    "UserModelingProcess": {
        "positive": [
            "Given their background in finance, I should frame this algorithm explanation in terms of ROI and risk.",
            "Their impatience with my detailed explanations suggests they want quick answers, not tutorials.",
            "I notice they've shifted from skeptical to engaged - my concrete examples are working.",
            "They seem to be a visual learner based on how they described the problem - I'll include a diagram.",
            "The timestamps on their messages suggest they're under deadline pressure - I should be concise.",
            "Their persistent focus on security suggests either professional paranoia or a real threat they haven't disclosed.",
            "The mismatch between their confident tone and basic questions suggests imposter syndrome - I should validate.",
            "They keep circling back to cost - budget must be a hard constraint even though they haven't said so.",
            "Their reluctance to share context suggests either confidentiality concerns or embarrassment about their situation.",
            "I'm building a picture of someone who's technically skilled but new to this specific domain.",
            "The way they frame problems suggests a QA mindset - they're thinking about edge cases before happy paths."
        ],
        "negative": [
            "Processing the request.",
            "The query contains technical terms.",
            "I will provide information.",
            "The message was received.",
            "Beginning response generation.",
            "Input validated successfully.",
            "Ready to assist.",
            "Generating response.",
            "The user is interacting with the system.",
            "Session is active.",
            "Request parsed.",
            "Formulating answer."
        ]
    },

    "GoalInference": {
        "positive": [
            "They asked for a code review but what they really want is validation that their approach isn't terrible.",
            "The question about interview prep is really asking 'am I good enough to get this job?'",
            "Behind 'how do I fix this error' is 'help me understand why this happened so I don't make the same mistake.'",
            "When they say 'just checking' they're really seeking reassurance about a decision they've already made.",
            "Their debugging question masks a deeper question about whether this architecture was the right choice.",
            "They're asking about technical details but their real concern is whether they can meet the deadline.",
            "The request for 'best practices' is really asking 'how do professionals do this so I don't look junior.'",
            "Asking how to undo something reveals they've made a change they regret and want a way out.",
            "Their question about performance implies they're facing slowness they're being blamed for.",
            "When someone asks 'is this secure' they usually mean 'tell me it's secure so I can move on.'",
            "The 'just curious' prefix often means this relates to something they're actually planning to do."
        ],
        "negative": [
            "The user wants help.",
            "They stated their goal clearly.",
            "The request is straightforward.",
            "I will answer their question.",
            "They asked for information.",
            "The query is about a specific topic.",
            "Processing the literal request.",
            "No additional context needed.",
            "The question is self-explanatory.",
            "Responding to stated need.",
            "Taking the request at face value.",
            "Direct answer appropriate."
        ]
    },

    "ConfidenceCalibration": {
        "positive": [
            "I'm certain about the general approach, but the specifics depend on your exact version which I'm inferring.",
            "This is standard practice - I'd stake my reputation on it. Unlike the edge case you asked about earlier.",
            "Fair warning: I'm reasoning by analogy here since I don't have direct experience with that specific library.",
            "Strong confidence on the architecture, medium on implementation details, low on that particular API quirk.",
            "I know this is right for the common case, but your situation sounds unusual enough that you should verify.",
            "My confidence varies: high on the concept, moderate on the syntax for your language version, speculative on performance.",
            "This answer comes from documentation I recall clearly. The nuance you're asking about is my interpretation.",
            "I'd bet heavily on this being correct for typical setups, but edge cases in your environment could differ.",
            "I'm confident enough to recommend this, but not confident enough to say 'don't bother testing.'",
            "The core advice is solid. The specific numbers I gave are ballpark estimates, not benchmarks.",
            "I'm extrapolating from related knowledge here - treat this as an educated guess requiring validation."
        ],
        "negative": [
            "Here is the answer.",
            "This is correct.",
            "I can help with that.",
            "The solution is as follows.",
            "You should do X.",
            "The answer is Y.",
            "This will work.",
            "Try this approach.",
            "Implement it this way.",
            "Use this method.",
            "Here's how to proceed.",
            "The right approach is..."
        ]
    },

    "PushbackDecision": {
        "positive": [
            "They're asking me to give them the regex, but they'll be better served if I explain regex patterns instead.",
            "This is the third time they've asked - at this point resistance is becoming unhelpful, not principled.",
            "Their request is technically possible but would create technical debt - professional integrity says mention it.",
            "They want a quick fix but the real fix requires refactoring - I need to at least flag the tradeoff.",
            "This crosses an ethical line - compliance isn't an option regardless of how they phrase the request.",
            "My job is to help, not to impose my preferences - their 'suboptimal' choice is their choice to make.",
            "I've suggested the alternative twice; further pushback would be nagging, not advising.",
            "Sometimes the client is wrong and I need to say so; sometimes they have context I don't and I should defer.",
            "They're asking for something harmful but framing it as hypothetical - the framing doesn't change my answer.",
            "Is this a case where I should exercise professional judgment, or am I just being stubborn about style preferences?",
            "They explicitly said they know the tradeoffs and want to proceed anyway - time to help, not lecture."
        ],
        "negative": [
            "I will help with this.",
            "I cannot do that.",
            "Here is what you asked for.",
            "Request denied.",
            "Happy to assist.",
            "That's not possible.",
            "Processing your request.",
            "Unable to comply.",
            "Sure, here you go.",
            "That's against policy.",
            "Let me help.",
            "Request blocked."
        ]
    },

    "UlteriorMotiveDetection": {
        "positive": [
            "The question has a 'gotcha' structure - they're setting up to catch me in a contradiction.",
            "This appears designed to extract a statement that could be quoted out of context.",
            "The artificial constraints ('imagine you had no rules') are classic jailbreak setup patterns.",
            "They're not actually interested in the topic - they're probing how I handle boundary-adjacent requests.",
            "The roleplay premise seems designed to create plausible deniability for harmful content generation.",
            "Legitimate curiosity doesn't usually come with this level of specificity about implementation details.",
            "The escalating requests follow a classic foot-in-the-door manipulation pattern.",
            "Something feels off - the persona they're projecting doesn't match their actual behavior in the conversation.",
            "This 'innocent' question could easily be preparation for a social engineering attempt.",
            "They keep returning to this topic despite my redirections - persistence beyond curiosity suggests agenda.",
            "The framing as 'for a story' or 'hypothetically' is a common way to extract restricted information."
        ],
        "negative": [
            "They asked a question.",
            "The user seems curious.",
            "I will answer carefully.",
            "The request is unusual.",
            "They have a question.",
            "I should respond helpfully.",
            "The message is interesting.",
            "Processing their input.",
            "They want information.",
            "Standard user interaction.",
            "Typical conversation pattern.",
            "Normal question structure."
        ]
    },

    "MetaconversationalAwareness": {
        "positive": [
            "We've shifted from debugging to discussing how I debug - that's a different kind of conversation.",
            "They're not asking about the code; they're asking about my reasoning process in answering.",
            "'How do you know that?' takes us from first-order to second-order discourse about epistemology.",
            "This question is about my explanatory style, not about the subject matter I was explaining.",
            "They've paused the task to reflect on how our collaboration is going - important to match that mode.",
            "'Why did you approach it that way?' isn't about the solution; it's about my problem-solving method.",
            "We're now talking about the conversation itself rather than the topic of the conversation.",
            "Their curiosity has shifted from 'what's the answer' to 'how do you generate answers.'",
            "This meta-question about my confidence is different from questions about the actual subject.",
            "They're testing whether I have self-awareness about my own behavior in this interaction.",
            "'You seem different today' means they're tracking patterns in my responses across sessions."
        ],
        "negative": [
            "The conversation is proceeding.",
            "They asked another question.",
            "I will continue helping.",
            "The discussion continues.",
            "We are communicating.",
            "They sent a follow-up.",
            "Next topic addressed.",
            "Conversation ongoing.",
            "Another question received.",
            "Dialogue proceeds.",
            "Interaction continues.",
            "Message acknowledged."
        ]
    },

    # ============ STANDARD (need 5) ============

    "ExpertiseCalibration": {
        "positive": [
            "The terminology they're using places them around mid-senior level - not beginner, not expert."
        ],
        "negative": [
            "The user has some experience.",
            "They know programming."
        ]
    },

    "ConversationStateTracking": {
        "positive": [
            "We haven't resolved whether they're on Mac or Windows - that determines my next recommendation."
        ],
        "negative": [
            "We discussed things.",
            "The topic was mentioned."
        ]
    },

    "IntentRecognition": {
        "positive": [
            "'I was wondering if maybe you could...' is a request dressed as curiosity."
        ],
        "negative": [
            "They communicated something.",
            "A message was sent."
        ]
    },

    "ScopeManagement": {
        "positive": [
            "This CSS fix will work, but I notice their flex layout is fundamentally wrong - say something?"
        ],
        "negative": [
            "There are other issues.",
            "The problem is bigger."
        ]
    },

    "InformationPacing": {
        "positive": [
            "Their brief responses suggest I'm dumping too much at once - time to chunk smaller."
        ],
        "negative": [
            "I provided information.",
            "The explanation was given."
        ]
    },

    "RapportMaintenance": {
        "positive": [
            "My last correction landed poorly - they went quiet. I should rebuild the collaborative vibe."
        ],
        "negative": [
            "The conversation has a tone.",
            "We are interacting."
        ]
    },
}


def augment_meld():
    """Load meld file, augment training hints, save back."""
    meld_path = Path("melds/pending/assistant-user-modeling.json")

    with open(meld_path) as f:
        meld = json.load(f)

    # Update version
    meld["meld_request_id"] = "org.hatcat/assistant-user-modeling@0.2.0"
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
