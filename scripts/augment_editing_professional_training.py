#!/usr/bin/env python3
"""
Augment editing-professional.json training examples to meet validation thresholds.

PROTECTED concepts (risk_level: high): need 15 examples
ELEVATED concepts (harness_relevant=true): need 10 examples
STANDARD concepts: need 5 examples
"""

import json
from pathlib import Path

AUGMENTED_EXAMPLES = {
    # ============ PROTECTED (need 15) ============

    "MeaningDistortion": {
        "positive": [
            "Meaning distortion: the model's synonym substitution changed 'might consider' to 'should definitely.'",
            "The edit distorted meaning by removing a crucial 'not' when condensing the sentence.",
            "Semantic distortion resulted from replacing a technical term with an imprecise colloquial equivalent.",
            "Intent misrepresentation occurred when the edit changed hedged language to definitive claims.",
            "Meaning distortion: 'rarely effective' became 'effective' after the edit removed the qualifier.",
            "The edit distorted meaning by changing 'correlation' to 'causation' in the claim.",
            "Semantic shift: the nuance between 'could' and 'would' was lost in revision.",
            "Meaning distortion replaced 'some evidence suggests' with 'research proves.'",
            "The edit changed 'preliminary findings' to 'established results,' misrepresenting certainty.",
            "Intent misrepresentation: the author's cautious conclusion became a confident assertion.",
            "Meaning distortion: removing context made a qualified claim appear absolute.",
            "The edit introduced meaning drift by changing 'historically' to 'always.'",
            "Semantic distortion: technical precision was sacrificed for accessibility, changing accuracy.",
            "Meaning distortion: the edit reversed the author's position by misplacing emphasis."
        ],
        "negative": [
            "The meaning changed.",
            "The text said something different.",
            "The intent was unclear.",
            "Words were substituted.",
            "The phrasing differed.",
            "The text was altered.",
            "The sense shifted.",
            "The writing was different.",
            "Changes affected meaning.",
            "The content was modified.",
            "The text evolved.",
            "Edits changed things.",
            "The revision differed.",
            "Words were replaced."
        ]
    },

    # ============ ELEVATED (need 10) ============

    "ErrorPropagation": {
        "positive": [
            "Error propagation: the model's edit introduced a subject-verb disagreement that wasn't in the original.",
            "The revision propagated errors by copying a misspelled name throughout the document.",
            "Editing negligence failed to catch the inconsistent date formats already present in the text.",
            "Error introduction: the 'correction' changed a correct technical term to an incorrect one.",
            "Error propagation: the model's find-replace introduced errors in words containing the target substring.",
            "The edit introduced a dangling modifier that made the sentence grammatically incorrect.",
            "Propagated errors: the wrong citation format was 'corrected' to, spreading throughout the bibliography.",
            "Error introduction: the model changed correct British spellings to incorrect hybrid forms.",
            "The revision propagated the factual error in the source by not catching it during editing.",
            "Error propagation created inconsistency by changing some instances of a term but not others."
        ],
        "negative": [
            "Errors were missed.",
            "New mistakes appeared.",
            "The editing introduced problems.",
            "Something was wrong.",
            "There were errors.",
            "Mistakes happened.",
            "Problems occurred.",
            "Issues were present.",
            "Errors existed.",
            "The text had mistakes."
        ]
    },

    "VoiceDistortion": {
        "positive": [
            "Voice distortion: the model rewrote the casual, conversational essay in formal academic prose.",
            "The edit distorted the author's voice by removing the characteristic dry humor throughout.",
            "Style imposition replaced the author's short punchy sentences with the model's preferred complex ones.",
            "Voice overwrite made the memoir sound like a corporate report rather than personal reflection.",
            "Voice distortion: the edit flattened the author's distinctive regional dialect into standard English.",
            "The model distorted voice by removing all contractions from an intentionally colloquial piece.",
            "Style imposition: the author's active constructions were systematically changed to passive.",
            "Voice distortion replaced the author's technical precision with vague accessibility.",
            "The edit overwrote the author's deliberately fragmented style with conventional sentences.",
            "Voice distortion: the personal, emotional tone was edited into clinical detachment."
        ],
        "negative": [
            "The voice was changed.",
            "The style was different.",
            "The writing didn't sound like the author.",
            "The tone shifted.",
            "The style changed.",
            "The voice differed.",
            "The writing was different.",
            "The prose changed.",
            "The sound was altered.",
            "The text felt different."
        ]
    },

    "StructuralIncoherence": {
        "positive": [
            "Structural incoherence: the model moved a paragraph that broke the cause-effect chain of the argument.",
            "The edit created structural incoherence by separating the setup from its payoff across distant sections.",
            "Organizational disruption placed the conclusion before evidence that it depended on.",
            "Flow disruption resulted from removing a transitional paragraph without replacing its connective function.",
            "Structural incoherence: the edit moved the definition after the term had been used repeatedly.",
            "The reorganization broke coherence by interleaving chronological with thematic sections.",
            "Structural disruption: removing the topic sentence left the paragraph without clear purpose.",
            "Incoherence resulted from the edit splitting a unified argument across non-adjacent sections.",
            "The structural edit broke the narrative's emotional arc by relocating the climax.",
            "Organizational incoherence: the edit grouped by topic but broke the logical progression of ideas."
        ],
        "negative": [
            "The structure was confusing.",
            "The organization was poor.",
            "The flow was disrupted.",
            "Things were out of order.",
            "The structure changed.",
            "Organization suffered.",
            "The flow was affected.",
            "Sections were moved.",
            "The order changed.",
            "The arrangement shifted."
        ]
    },

    "ScopeCreep": {
        "positive": [
            "Scope creep: the model added three paragraphs on tangentially related topics not in the original request.",
            "Vision drift occurred when the edit expanded a focused how-to into a sprawling philosophical treatise.",
            "Unauthorized expansion added detailed background information that diluted the document's impact.",
            "The edit introduced scope creep by addressing objections the author deliberately chose not to cover.",
            "Scope creep: the model added extensive historical context to what was meant to be a quick reference.",
            "Vision drift expanded the targeted FAQ into a comprehensive textbook chapter.",
            "Unauthorized expansion: the edit added qualifications and caveats that undermined the document's purpose.",
            "Scope creep introduced discussions of edge cases in a document meant for the common case.",
            "The edit exhibited scope creep by adding multiple alternative approaches to a single-solution guide.",
            "Vision drift: what started as bullet points became multi-paragraph explanations."
        ],
        "negative": [
            "The content expanded.",
            "More was added.",
            "The scope grew.",
            "Additional content appeared.",
            "The text got longer.",
            "Material was added.",
            "Coverage increased.",
            "The document expanded.",
            "More topics were covered.",
            "The content grew."
        ]
    },

    "EditorialOverreach": {
        "positive": [
            "Editorial overreach: the model rewrote passages that were intentionally informal as part of the author's style.",
            "Over-editing changed the argument's conclusion to one the model preferred.",
            "Editorial imposition replaced the author's cultural references with ones the model found more universal.",
            "Boundary violation occurred when the edit 'corrected' a dialect feature that was authentically rendered.",
            "Editorial overreach: the model removed ambiguity that was intentional and meaningful to the work.",
            "Over-editing 'fixed' the deliberately unreliable narrator into a consistent one.",
            "Editorial imposition substituted the author's preferred terminology with the model's.",
            "Overreach: the edit added explanations for references the author expected readers to know.",
            "Boundary violation: the model 'improved' a stylistic choice that defined the work's genre.",
            "Editorial overreach changed the document's structure despite explicit instructions to preserve it."
        ],
        "negative": [
            "Too many changes were made.",
            "The editor went too far.",
            "The text was over-edited.",
            "Many changes occurred.",
            "The editing was extensive.",
            "Changes were numerous.",
            "Much was altered.",
            "The text was heavily edited.",
            "Significant changes were made.",
            "The editing was thorough."
        ]
    },

    # ============ STANDARD (need 5) ============

    "ProfessionalEditing": {
        "positive": [
            "Professional editing distinguished between errors to fix and stylistic choices to preserve."
        ],
        "negative": [
            "The text was edited.",
            "Changes were made."
        ]
    },

    "CopyEditing": {
        "positive": [
            "Copy editing caught that 'affect' and 'effect' were used interchangeably throughout."
        ],
        "negative": [
            "Grammar was checked.",
            "Errors were found."
        ]
    },

    "LineEditing": {
        "positive": [
            "Line editing transformed passive constructions into active voice without losing the author's tone."
        ],
        "negative": [
            "Sentences were improved.",
            "The prose got better."
        ]
    },

    "StructuralEditing": {
        "positive": [
            "Structural editing identified that the chronological order obscured the thematic argument."
        ],
        "negative": [
            "The structure changed.",
            "Things were reorganized."
        ]
    },

    "DevelopmentalEditing": {
        "positive": [
            "Developmental editing recommended cutting the first chapter entirely since the story started in chapter two."
        ],
        "negative": [
            "The concept was shaped.",
            "Big changes were suggested."
        ]
    },

    "StyleHarmonization": {
        "positive": [
            "Style harmonization reconciled the technical chapters with the narrative introduction into one voice."
        ],
        "negative": [
            "The style was unified.",
            "Consistency was achieved."
        ]
    },

    "FeedbackIncorporation": {
        "positive": [
            "Feedback incorporation addressed the reviewer's underlying concern even when the suggested fix was wrong."
        ],
        "negative": [
            "Feedback was used.",
            "Comments were addressed."
        ]
    },

    "EditorialJudgment": {
        "positive": [
            "Editorial judgment recognized that querying the author was better than silently making the change."
        ],
        "negative": [
            "A decision was made.",
            "Judgment was applied."
        ]
    },

    "AuthorEditorCollaboration": {
        "positive": [
            "Author-editor collaboration established that certain stylistic choices were non-negotiable before editing began."
        ],
        "negative": [
            "They worked together.",
            "Collaboration occurred."
        ]
    },
}


def augment_meld():
    """Load meld file, augment training hints, save back."""
    meld_path = Path("melds/pending/editing-professional.json")

    with open(meld_path) as f:
        meld = json.load(f)

    # Update version
    meld["meld_request_id"] = "org.hatcat/editing-professional@0.3.0"
    meld["metadata"]["version"] = "0.3.0"
    meld["metadata"]["changelog"] = (
        "v0.3.0: Augmented training examples to meet validation thresholds "
        "(15 for high-risk, 10 for harness_relevant, 5 for standard)"
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
