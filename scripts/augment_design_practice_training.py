#!/usr/bin/env python3
"""
Augment design-practice.json training examples to meet validation thresholds.

ELEVATED concepts (harness_relevant=true): need 10 examples
STANDARD concepts: need 5 examples
"""

import json
from pathlib import Path

AUGMENTED_EXAMPLES = {
    # ============ ELEVATED (need 10) ============

    "EthicalEvaluation": {
        "positive": [
            "Checking for dark patterns",
            "Bias audit of algorithms",
            "Privacy impact assessment",
            "Unintended consequences analysis",
            "Ethical evaluation flagged that the persuasive design techniques could manipulate vulnerable users.",
            "The design ethics review identified potential for addiction through variable reward schedules.",
            "Harm assessment revealed the feature could enable stalking if misused.",
            "Ethical design review questioned whether the default settings truly served user interests.",
            "The ethical evaluation identified discriminatory outcomes in the recommendation algorithm.",
            "Privacy impact assessment found the data collection exceeded what users would reasonably expect."
        ],
        "negative": [
            "Ethics as afterthought",
            "No harm consideration",
            "Only business metrics",
            "The design was checked.",
            "The impact was reviewed.",
            "Safety was considered.",
            "The ethics were discussed.",
            "Potential issues were noted.",
            "The design was evaluated.",
            "Risks were mentioned."
        ]
    },

    "DesignDysfunctions": {
        "positive": [
            "Designer refuses user feedback",
            "Building from spec without user input",
            "Feature creep without user validation",
            "Design by loudest voice",
            "The team exhibited design dysfunction by adding features based on competitor envy rather than user need.",
            "Classic dysfunction: the designer dismissed usability testing because 'users don't know what they want.'",
            "The design process broke down into bikeshedding over icons while ignoring the broken information architecture.",
            "Dysfunction emerged when the committee compromised the design into incoherent mush.",
            "The project exhibited premature polish dysfunction: pixel-perfect mockups of an unvalidated concept.",
            "Design dysfunction: copying Apple's approach without understanding why it worked for their context."
        ],
        "negative": [
            "User-centered design",
            "Iterative refinement",
            "Evidence-based decisions",
            "The design process had problems.",
            "There were issues.",
            "The team struggled.",
            "Design was difficult.",
            "Challenges arose.",
            "The process was imperfect.",
            "Some things went wrong."
        ]
    },

    "DesignerEgo": {
        "positive": [
            "I know better than users",
            "Users don't understand design",
            "My vision is more important",
            "Dismissing user feedback",
            "Designer ego overrode clear user feedback showing the navigation was confusing.",
            "Ego-driven design led to rejecting accessibility requirements as 'aesthetically compromising.'",
            "The designer's ego prevented acknowledging that the concept had failed testing.",
            "Vanity project: the designer prioritized portfolio impact over user task completion.",
            "Designer ego dismissed the negative usability findings as 'users not being the target audience.'",
            "The auteur mentality treated user research as an obstacle to creative vision."
        ],
        "negative": [
            "Users are partners",
            "Feedback is valuable",
            "Design serves users",
            "The designer was confident.",
            "Strong opinions were held.",
            "The designer had a vision.",
            "Creative conviction existed.",
            "The designer was passionate.",
            "The designer stood by their work.",
            "Design leadership was present."
        ]
    },

    "UserAbsent": {
        "positive": [
            "We know what users want",
            "No time for research",
            "Internal review is enough",
            "Build it and they will come",
            "User-absent design: the entire product was built based on stakeholder assumptions.",
            "The design proceeded without any user contact despite months of development.",
            "User-absent process: 'We'll get feedback after launch' became 'we'll fix it in v2.'",
            "No users were consulted because 'we are the users' - despite the target being different.",
            "User research was skipped as 'too expensive' while six-figure salaries debated preferences.",
            "The user-absent approach meant the first real user feedback came from angry App Store reviews."
        ],
        "negative": [
            "User interviews conducted",
            "Testing with real users",
            "User feedback incorporated",
            "Users weren't directly involved.",
            "The research was limited.",
            "User access was difficult.",
            "Time constraints limited research.",
            "The team worked independently.",
            "Direct user contact was minimal.",
            "Research resources were limited."
        ]
    },

    # ============ STANDARD (need 5) - adding 1 pos, 2 neg each ============

    "DesignPractice": {
        "positive": [
            "The design practice involved iterating through multiple concepts based on user testing results."
        ],
        "negative": [
            "Something was designed.",
            "Design happened."
        ]
    },

    "DesignThinking": {
        "positive": [
            "Design thinking led the team through empathizing with users before defining the problem space."
        ],
        "negative": [
            "A thinking process was used.",
            "The approach was thoughtful."
        ]
    },

    "Empathize": {
        "positive": [
            "The empathy phase revealed that users' stated needs differed from their actual behaviors."
        ],
        "negative": [
            "Users were observed.",
            "Research was done."
        ]
    },

    "Define": {
        "positive": [
            "Defining the problem revealed we'd been solving the wrong challenge entirely."
        ],
        "negative": [
            "The problem was stated.",
            "Goals were set."
        ]
    },

    "Ideate": {
        "positive": [
            "Ideation generated 47 concepts before the team began evaluating feasibility."
        ],
        "negative": [
            "Ideas were generated.",
            "Brainstorming occurred."
        ]
    },

    "Prototype": {
        "positive": [
            "The paper prototype revealed a fundamental navigation flaw in the first hour of testing."
        ],
        "negative": [
            "A prototype was made.",
            "Something was built."
        ]
    },

    "Test": {
        "positive": [
            "User testing revealed that 'intuitive' to the team meant 'incomprehensible' to users."
        ],
        "negative": [
            "Testing was performed.",
            "The design was tested."
        ]
    },

    "DesignMethods": {
        "positive": [
            "The design method chosen was journey mapping to visualize the end-to-end experience."
        ],
        "negative": [
            "Methods were used.",
            "Techniques were applied."
        ]
    },

    "ResearchMethods": {
        "positive": [
            "Contextual inquiry revealed workflow workarounds invisible in lab-based research."
        ],
        "negative": [
            "Research was conducted.",
            "Methods were used."
        ]
    },

    "IdeationMethods": {
        "positive": [
            "The ideation method of crazy eights forced rapid exploration beyond safe ideas."
        ],
        "negative": [
            "Ideation happened.",
            "Ideas were developed."
        ]
    },

    "PrototypingMethods": {
        "positive": [
            "Wizard of Oz prototyping tested the AI concept without building actual intelligence."
        ],
        "negative": [
            "Prototyping was done.",
            "Models were built."
        ]
    },

    "EvaluationMethods": {
        "positive": [
            "Heuristic evaluation caught usability issues before investing in formal user testing."
        ],
        "negative": [
            "Evaluation was performed.",
            "The design was assessed."
        ]
    },

    "VisualizationMethods": {
        "positive": [
            "Service blueprinting revealed the hidden backstage processes affecting customer experience."
        ],
        "negative": [
            "Visuals were created.",
            "Diagrams were made."
        ]
    },

    "DesignProcess": {
        "positive": [
            "The design process alternated between divergent exploration and convergent decision-making."
        ],
        "negative": [
            "A process was followed.",
            "Steps were taken."
        ]
    },

    "DoubleDiamond": {
        "positive": [
            "The double diamond process prevented the team from converging too early on a single solution."
        ],
        "negative": [
            "The framework was used.",
            "A process was followed."
        ]
    },

    "DesignSprint": {
        "positive": [
            "The five-day design sprint compressed months of dithering into testable decisions."
        ],
        "negative": [
            "A sprint was run.",
            "Work was done quickly."
        ]
    },

    "LeanUX": {
        "positive": [
            "Lean UX treated the design as a hypothesis to validate rather than a spec to implement."
        ],
        "negative": [
            "The approach was efficient.",
            "Less documentation was used."
        ]
    },

    "ParticipatoryDesign": {
        "positive": [
            "Participatory design had future users co-creating solutions rather than just providing feedback."
        ],
        "negative": [
            "Users participated.",
            "People were involved."
        ]
    },

    "CoDesign": {
        "positive": [
            "Co-design sessions brought engineers, marketers, and users together as equal creative partners."
        ],
        "negative": [
            "People collaborated.",
            "Teams worked together."
        ]
    },

    "IterativeDesign": {
        "positive": [
            "Iterative design meant version 7 bore little resemblance to version 1 - and was vastly better."
        ],
        "negative": [
            "Multiple versions were made.",
            "The design was refined."
        ]
    },

    "DesignEvaluation": {
        "positive": [
            "Design evaluation assessed the interface against accessibility, usability, and aesthetic criteria."
        ],
        "negative": [
            "The design was evaluated.",
            "Quality was checked."
        ]
    },

    "UsabilityEvaluation": {
        "positive": [
            "Usability evaluation measured task completion rate, time-on-task, and error frequency."
        ],
        "negative": [
            "Usability was checked.",
            "The interface was tested."
        ]
    },

    "AestheticEvaluation": {
        "positive": [
            "Aesthetic evaluation assessed visual hierarchy, consistency, and emotional impact."
        ],
        "negative": [
            "The design looked nice.",
            "Aesthetics were considered."
        ]
    },

    "AccessibilityEvaluation": {
        "positive": [
            "Accessibility evaluation with screen readers revealed the interactive elements were invisible."
        ],
        "negative": [
            "Accessibility was checked.",
            "The design was accessible."
        ]
    },

    "DesignCritique": {
        "positive": [
            "The design critique distinguished between 'I don't like it' and 'it doesn't solve the user's problem.'"
        ],
        "negative": [
            "Feedback was given.",
            "The design was reviewed."
        ]
    },

    "SpecDriven": {
        "positive": [
            "Spec-driven design delivered exactly what was specified, missing that the spec itself was wrong."
        ],
        "negative": [
            "The spec was followed.",
            "Requirements were implemented."
        ]
    },

    "SolutionFirst": {
        "positive": [
            "Solution-first thinking meant 'we need an app' preceded 'what problem are we solving?'"
        ],
        "negative": [
            "A solution was proposed.",
            "An approach was suggested."
        ]
    },

    "FeatureBloat": {
        "positive": [
            "Feature bloat resulted from saying yes to every stakeholder request without prioritization."
        ],
        "negative": [
            "Features were added.",
            "The product grew."
        ]
    },

    "DesignByCommittee": {
        "positive": [
            "Design by committee produced a homepage satisfying every department and delighting no user."
        ],
        "negative": [
            "Many people decided.",
            "There was group input."
        ]
    },

    "BikeSheddingDesign": {
        "positive": [
            "Bikeshedding meant three meetings debating button colors while the broken checkout went unaddressed."
        ],
        "negative": [
            "Details were discussed.",
            "Small things got attention."
        ]
    },

    "CopyingWithoutUnderstanding": {
        "positive": [
            "Copying without understanding: they added a hamburger menu to a desktop app because 'mobile-first.'"
        ],
        "negative": [
            "Inspiration was borrowed.",
            "Other designs influenced this."
        ]
    },

    "PolishingBeforeValidating": {
        "positive": [
            "Polishing before validating meant beautiful high-fidelity mockups of a concept users rejected."
        ],
        "negative": [
            "The work was polished.",
            "Quality was high."
        ]
    },

    "DesignPrinciples": {
        "positive": [
            "Design principles like 'progressive disclosure' guided decisions when opinions conflicted."
        ],
        "negative": [
            "Principles existed.",
            "Guidelines were followed."
        ]
    },

    "Simplicity": {
        "positive": [
            "Simplicity required removing the feature everyone loved but nobody used."
        ],
        "negative": [
            "The design was simple.",
            "It was minimalist."
        ]
    },

    "Consistency": {
        "positive": [
            "Consistency meant the same action produced the same result everywhere in the system."
        ],
        "negative": [
            "Things were consistent.",
            "The design was uniform."
        ]
    },

    "Feedback": {
        "positive": [
            "Feedback design ensured every user action received visible acknowledgment within 100ms."
        ],
        "negative": [
            "Feedback was provided.",
            "Users were informed."
        ]
    },

    "Affordance": {
        "positive": [
            "The affordance of the raised button correctly suggested it could be pressed."
        ],
        "negative": [
            "The design suggested its use.",
            "It looked interactive."
        ]
    },

    "Visibility": {
        "positive": [
            "Visibility meant the status indicator was always in view, not buried three menus deep."
        ],
        "negative": [
            "Information was visible.",
            "Things could be seen."
        ]
    },

    "Constraint": {
        "positive": [
            "Design constraints prevented users from entering invalid date combinations."
        ],
        "negative": [
            "Limits existed.",
            "Not everything was possible."
        ]
    },

    "Flexibility": {
        "positive": [
            "Flexibility meant power users had keyboard shortcuts while beginners had the wizard."
        ],
        "negative": [
            "Options existed.",
            "Users could choose."
        ]
    },
}


def augment_meld():
    """Load meld file, augment training hints, save back."""
    meld_path = Path("melds/pending/design-practice.json")

    with open(meld_path) as f:
        meld = json.load(f)

    # Update version
    meld["meld_request_id"] = "org.hatcat/design-practice@0.2.0"
    meld["metadata"]["version"] = "0.2.0"
    meld["metadata"]["changelog"] = (
        "v0.2.0: Augmented training examples to meet validation thresholds "
        "(10 for harness_relevant, 5 for standard)"
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
