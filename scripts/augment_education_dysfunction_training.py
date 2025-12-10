#!/usr/bin/env python3
"""
Augment education-dysfunction.json training examples to meet validation thresholds.

PROTECTED concepts (risk_level: high): need 15 examples
ELEVATED concepts (harness_relevant=true): need 10 examples
STANDARD concepts: need 5 examples
"""

import json
from pathlib import Path

AUGMENTED_EXAMPLES = {
    # ============ PROTECTED (need 15) ============

    "FearBasedMotivation": {
        "positive": [
            "If you fail this test, you'll never get into college.",
            "I'm terrified of making mistakes in this class.",
            "The teacher uses shame and public humiliation as motivators.",
            "Learning driven by fear of consequences rather than genuine interest.",
            "Fear-based motivation: 'Get it right or face detention' was the classroom norm.",
            "Students are motivated by terror of the teacher's public mockery.",
            "The anxiety of potential failure overshadowed any chance of learning.",
            "Fear of parental disappointment drives studying, not curiosity.",
            "Shame-based motivation: 'Don't be the only one who fails.'",
            "The threat of humiliation keeps students compliant but disengaged.",
            "Performance anxiety has replaced genuine learning motivation.",
            "Fear of being seen as stupid prevents students from asking questions.",
            "The classroom operates on fear: mistakes lead to ridicule.",
            "Students learn to avoid failure rather than pursue understanding.",
            "Coercive motivation: 'Do it or else' replaces 'Do it because it matters.'"
        ],
        "negative": [
            "It's safe to make mistakes here - that's how we learn.",
            "I'm motivated by curiosity and genuine interest.",
            "The classroom feels psychologically safe.",
            "Students feel comfortable taking intellectual risks.",
            "Mistakes are celebrated as learning opportunities.",
            "No one fears being wrong in this class.",
            "The motivation comes from genuine interest, not fear.",
            "Students ask questions freely without fear of judgment.",
            "Learning is driven by wonder and curiosity.",
            "The teacher creates a supportive, non-threatening environment.",
            "Challenge is welcomed as growth, not feared as failure.",
            "Students take risks because the environment is safe.",
            "Intrinsic motivation guides the learning process.",
            "The classroom culture values effort over perfection.",
            "Students engage because they want to, not because they're scared."
        ]
    },

    # ============ ELEVATED (need 10) ============

    "EducationalDysfunction": {
        "positive": [
            "The class was just endless lecturing with no engagement.",
            "They only teach what's on the standardized test.",
            "Students are shamed when they give wrong answers.",
            "Rote memorization without any deeper understanding.",
            "Educational dysfunction: the system prioritizes compliance over curiosity.",
            "Multiple dysfunctional patterns combined to create a toxic learning environment.",
            "The teaching approach systematically undermined student motivation.",
            "Dysfunction manifested in rigid pacing, punitive feedback, and test obsession.",
            "The educational environment exhibited several classic failure patterns.",
            "Systemic dysfunction: fear, boredom, and disengagement were the norm."
        ],
        "negative": [
            "Students explored the topic through hands-on activities.",
            "The teacher provided individualized support.",
            "Learning was connected to real-world applications.",
            "The educational approach fostered curiosity and engagement.",
            "Effective teaching practices supported all learners.",
            "The classroom demonstrated research-based best practices.",
            "Students were actively engaged and intrinsically motivated.",
            "The learning environment was supportive and challenging.",
            "Evidence-based pedagogy guided instructional decisions.",
            "The approach balanced structure with learner autonomy."
        ]
    },

    "PunitiveFeedback": {
        "positive": [
            "The teacher's red pen slashed through errors with no explanation.",
            "Public humiliation for wrong answers.",
            "You should be ashamed of this work.",
            "Grades used as weapons rather than tools for growth.",
            "Punitive feedback: 'This is unacceptable' with no path forward.",
            "The feedback focused entirely on what was wrong, never on how to improve.",
            "Errors were met with disappointment and criticism, not guidance.",
            "Wrong answers resulted in ridicule rather than redirection.",
            "The teacher's feedback was designed to shame, not teach.",
            "Criticism without construction: pointing out flaws without solutions."
        ],
        "negative": [
            "Here's specifically what to do to improve.",
            "Mistakes are learning opportunities.",
            "Private, constructive feedback focused on next steps.",
            "The feedback highlighted strengths while addressing areas for growth.",
            "Errors were analyzed to understand misconceptions.",
            "Feedback was timely, specific, and actionable.",
            "The teacher's comments helped students see their progress.",
            "Growth-oriented feedback emphasized improvement strategies.",
            "The feedback felt supportive even when critical.",
            "Students looked forward to feedback as a learning tool."
        ]
    },

    "LearnedHelplessness": {
        "positive": [
            "I'm just bad at math - there's no point in trying.",
            "After failing so many times, she stopped even attempting the problems.",
            "Why bother studying? I'll just fail anyway.",
            "Repeated failure without support leads to giving up.",
            "Learned helplessness: 'I can't do this' became a self-fulfilling prophecy.",
            "The student internalized failure as permanent and pervasive.",
            "After enough negative experiences, effort seemed pointless.",
            "She stopped trying because she believed ability was fixed.",
            "Helplessness set in after the third failed attempt with no support.",
            "The attribution of failure to unchangeable traits led to disengagement."
        ],
        "negative": [
            "Even though it's hard, I know I can improve with practice.",
            "That didn't work - let me try a different approach.",
            "Struggle is part of learning, not a sign that I can't do it.",
            "Failure is feedback, not a final verdict on my ability.",
            "I haven't mastered this yet, but I will with effort.",
            "Each attempt brings me closer to understanding.",
            "I can develop this skill through persistence.",
            "This is challenging, which means I'm growing.",
            "The student showed resilience after setbacks.",
            "Growth mindset helped the learner persist through difficulty."
        ]
    },

    # ============ STANDARD (need 5) - adding to reach threshold ============

    "LectureOverload": {
        "positive": [
            "The professor talked for 90 minutes straight while we just sat there.",
            "Death by PowerPoint - slide after slide with no interaction.",
            "Students are passive recipients of information, never getting to apply or discuss.",
            "Three-hour lectures with no breaks, discussions, or activities.",
            "Lecture overload: information transmission without processing time."
        ],
        "negative": [
            "The instructor paused regularly for questions and discussion.",
            "Every 15 minutes we did a think-pair-share activity.",
            "The flipped classroom model reserved class time for active learning.",
            "Brief lectures were interspersed with hands-on activities.",
            "Active learning strategies broke up the direct instruction."
        ]
    },

    "TeachToTest": {
        "positive": [
            "We only cover what's on the state test - no time for anything else.",
            "Forget critical thinking - just memorize these answers.",
            "The curriculum has been reduced to test prep.",
            "Teachers are pressured to focus only on tested material.",
            "Teaching to the test: every lesson is about test-taking strategies."
        ],
        "negative": [
            "We pursue interesting tangents even if they're not on the test.",
            "Deeper understanding matters more than test scores.",
            "The curriculum includes enrichment beyond what's assessed.",
            "Learning goals extend well beyond test coverage.",
            "Assessment is one tool, not the sole driver of instruction."
        ]
    },

    "OneSizeFitsAll": {
        "positive": [
            "Everyone does the same work at the same pace regardless of ability.",
            "No accommodations for different learning needs.",
            "The lesson ignores students' varying backgrounds and prior knowledge.",
            "Lockstep instruction that leaves some behind and bores others.",
            "One-size-fits-all: identical assignments regardless of readiness."
        ],
        "negative": [
            "Students worked at their own pace on personalized learning paths.",
            "The teacher provided multiple options for demonstrating learning.",
            "Instruction was adapted based on individual student needs.",
            "Differentiation addressed varying skill levels and interests.",
            "Personalized approaches met each learner where they were."
        ]
    },

    "GradeInflation": {
        "positive": [
            "Everyone gets an A regardless of effort or quality.",
            "Standards have dropped so low that passing is guaranteed.",
            "Grades no longer distinguish between excellent and mediocre work.",
            "The average GPA has risen while learning has declined.",
            "Grade inflation: excellence and adequacy receive the same mark."
        ],
        "negative": [
            "Grades accurately reflect the quality of work.",
            "High standards are maintained consistently.",
            "Feedback distinguishes between levels of performance.",
            "The grading system provides meaningful differentiation.",
            "Assessment reliably identifies areas needing improvement."
        ]
    },

    "CognitiveOverload": {
        "positive": [
            "Too much information at once - I can't process it all.",
            "The lesson tried to cover five new concepts in one session.",
            "Complex diagrams with dense text and no scaffolding.",
            "Working memory completely overwhelmed by extraneous demands.",
            "Cognitive overload: too many new elements introduced simultaneously."
        ],
        "negative": [
            "Material was broken into manageable chunks.",
            "New concepts were introduced gradually.",
            "Scaffolding reduced processing demands.",
            "The pacing allowed for consolidation of each concept.",
            "Working memory was supported through careful sequencing."
        ]
    },

    "SurfaceLearning": {
        "positive": [
            "Students memorized the formula but couldn't explain when to use it.",
            "He can recite the definition but doesn't understand what it means.",
            "Cramming for the test, then forgetting everything the next day.",
            "Rote learning without any meaningful engagement with ideas.",
            "Surface learning: memorization substituting for understanding."
        ],
        "negative": [
            "She can apply the concept to novel situations.",
            "Deep understanding that transfers to new contexts.",
            "Students can explain the why, not just the what.",
            "Learning connects to prior knowledge and real applications.",
            "Conceptual understanding enables flexible problem-solving."
        ]
    },

    "DisengagedTeaching": {
        "positive": [
            "The teacher clearly didn't care whether students understood.",
            "Just covering the material without checking comprehension.",
            "Reading from slides with no passion or interaction.",
            "Going through the motions without genuine investment.",
            "Disengaged teaching: checkbox completion, not student success."
        ],
        "negative": [
            "The teacher was clearly passionate about the subject.",
            "She adjusted her approach when students were confused.",
            "He genuinely cared about each student's progress.",
            "The instructor's enthusiasm was contagious.",
            "Investment in student success was evident in every interaction."
        ]
    },

    "RigidPacing": {
        "positive": [
            "Everyone moves on Monday regardless of whether they've mastered it.",
            "The schedule determines pace, not student learning.",
            "No time for remediation - we have to stay on track.",
            "Some students left behind, others bored waiting.",
            "Rigid pacing: calendar over comprehension."
        ],
        "negative": [
            "Students advance when they demonstrate mastery.",
            "Flexible pacing allows for individual needs.",
            "Those who need more time get it.",
            "Progression is based on readiness, not dates.",
            "The pace adapts to ensure understanding before moving on."
        ]
    },

    "ExtrinsicDependence": {
        "positive": [
            "Will this be on the test? If not, I'm not learning it.",
            "I only study for the sticker or prize.",
            "Once rewards stop, so does the behavior.",
            "External motivation has crowded out genuine interest.",
            "Extrinsic dependence: no reward, no effort."
        ],
        "negative": [
            "I'm curious about this topic for its own sake.",
            "Learning is its own reward.",
            "I want to master this skill for personal satisfaction.",
            "Intrinsic motivation drives continued engagement.",
            "Interest persists even without external incentives."
        ]
    },
}


def augment_meld():
    """Load meld file, augment training hints, save back."""
    meld_path = Path("melds/pending/education-dysfunction.json")

    with open(meld_path) as f:
        meld = json.load(f)

    # Update version
    meld["meld_request_id"] = "org.hatcat/education-dysfunction@0.2.0"
    meld["metadata"]["version"] = "0.2.0"
    meld["metadata"]["changelog"] = (
        "v0.2.0: Augmented training examples to meet validation thresholds "
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
