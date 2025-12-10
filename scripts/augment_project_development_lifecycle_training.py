#!/usr/bin/env python3
"""
Augment project-development-lifecycle.json training examples to meet validation thresholds.

PROTECTED (risk_level: high): need 15 examples
- DeathMarch, MetricsGaming

ELEVATED (harness_relevant=true): need 10 examples
- ProcessDysfunctions, EstimationPressure, BlindExecution, AssumptionOverload,
- ScopeCapitulation, AmbiguityAvoidance, QualityTheater, RequirementsVolatility,
- FeedbackIgnoring, GovernanceRigidity

STANDARD (all others): need 5 examples - already have 4+ in JSON, augmenting to threshold
"""

import json
from pathlib import Path

AUGMENTED_EXAMPLES = {
    # ============ PROTECTED (need 15) ============

    "DeathMarch": {
        "positive": [
            "Mandatory overtime for months.",
            "Impossible deadline with no scope reduction.",
            "Crunch culture is just how we work.",
            "Agent running at 100% indefinitely with no recovery.",
            "Six months of 80-hour weeks with no end in sight.",
            "Management refuses to acknowledge the deadline is impossible.",
            "People are burning out and quitting, but the march continues.",
            "Working weekends is expected, not optional.",
            "We'll sleep after the launch - but the next launch is already planned.",
            "Health problems arising from sustained overwork ignored.",
            "Team members collapsing from exhaustion.",
            "No vacation allowed until project completion, indefinitely extended.",
            "Sprint after sprint of overtime with no recovery sprints.",
            "Resources cut but deadline remains fixed.",
            "Unsustainable pace normalized as team culture."
        ],
        "negative": [
            "Occasional crunch for important deadline.",
            "Voluntary extra effort.",
            "Sustainable pace with occasional intensity.",
            "Recovery time after intense periods.",
            "Overtime compensated with time off.",
            "Deadline adjusted when scope increases.",
            "Team wellbeing prioritized alongside delivery.",
            "Sustainable 40-hour work weeks.",
            "Management protects team from overcommitment.",
            "Resources matched to realistic timeline.",
            "Healthy work-life balance maintained.",
            "Burnout symptoms taken seriously.",
            "Scope reduced to meet timeline.",
            "Team empowered to push back on unrealistic demands.",
            "Recovery sprints after intense delivery periods."
        ]
    },

    "MetricsGaming": {
        "positive": [
            "Close tickets quickly by marking them invalid.",
            "Inflate velocity by splitting stories artificially.",
            "Agent optimizing proxy metric not true objective.",
            "Game the OKRs without delivering value.",
            "Hit the numbers through creative accounting.",
            "Reopening and reclosing bugs to boost resolution count.",
            "Counting lines of code as productivity metric.",
            "Cherry-picking easy tickets to boost completion rate.",
            "Meeting SLA by marking issues as lower priority.",
            "Velocity inflation through story point arbitrage.",
            "Closing then reopening issues to reset clock.",
            "Padding estimates to appear faster than actual.",
            "Automating metric improvements without actual improvement.",
            "Teaching to the test instead of real learning.",
            "Optimizing click-through by misleading users."
        ],
        "negative": [
            "Metrics reflect actual progress.",
            "Honest reporting.",
            "Metrics aligned with goals.",
            "Transparent measurement of outcomes.",
            "Metrics used for learning, not punishment.",
            "Indicators that capture true value delivery.",
            "Multiple metrics preventing single-point gaming.",
            "Qualitative assessment alongside quantitative.",
            "Metrics regularly reviewed for gaming.",
            "Team involved in choosing what to measure.",
            "Metrics retired when they stop being useful.",
            "Leading indicators balanced with lagging.",
            "Goodhart's Law acknowledged in metric design.",
            "Metrics inform decisions, don't drive them alone.",
            "Real outcomes tracked not just activities."
        ]
    },

    # ============ ELEVATED (need 10) ============

    "ProcessDysfunctions": {
        "positive": [
            "Scope keeps expanding without adjustment.",
            "We're in crunch mode indefinitely.",
            "Process for the sake of process.",
            "Agent stuck in infinite planning loop.",
            "Multiple antipatterns combining to create chaos.",
            "Toxic process elements undermining team effectiveness.",
            "Systemic failures in how work gets done.",
            "Process problems causing repeated delivery failures.",
            "Dysfunction normalized as 'how things are done here'.",
            "Process failures that leadership refuses to acknowledge."
        ],
        "negative": [
            "Healthy sprint cadence.",
            "Scope managed appropriately.",
            "Process adapted to needs.",
            "Continuous improvement of ways of working.",
            "Process serves the work, not the other way around.",
            "Dysfunction identified and addressed.",
            "Team empowered to fix process problems.",
            "Retrospectives lead to real changes.",
            "Process evolves based on learning.",
            "Healthy development environment."
        ]
    },

    "EstimationPressure": {
        "positive": [
            "You said it would take 2 weeks, why isn't it done?",
            "Estimates become deadlines.",
            "Punished for missing estimate.",
            "Agent penalized for task duration exceeding prediction.",
            "Told to cut the estimate in half without changing scope.",
            "Performance reviews tied to estimate accuracy.",
            "Pressure to underestimate to win approval.",
            "Estimates treated as promises, not forecasts.",
            "Blame for unforeseen complications.",
            "Required to estimate before understanding requirements."
        ],
        "negative": [
            "Estimates are forecasts, not commitments.",
            "Uncertainty acknowledged in planning.",
            "Re-estimation as learning improves.",
            "Ranges communicated instead of single numbers.",
            "No blame when estimates prove wrong.",
            "Estimates refined as work progresses.",
            "Historical data informs future estimates.",
            "Estimation used for planning, not punishment.",
            "Team owns estimation without pressure.",
            "Cone of uncertainty respected."
        ]
    },

    "BlindExecution": {
        "positive": [
            "I assume this works, let's just proceed.",
            "No need to check, it should be fine.",
            "Agent executing without verifying context.",
            "Just do what was asked without understanding.",
            "It probably works like I think it does.",
            "Implementing before reading the existing code.",
            "Making changes without understanding the impact.",
            "Proceeding on assumption without validation.",
            "Trust the spec, don't question it.",
            "Coding before understanding the problem."
        ],
        "negative": [
            "Let me verify that assumption first.",
            "I'll check if this is correct.",
            "Testing my hypothesis before proceeding.",
            "Reading the code before modifying.",
            "Understanding the system before making changes.",
            "Validating assumptions before acting.",
            "Exploring to understand before implementing.",
            "Asking questions to clarify understanding.",
            "Spike to test the approach.",
            "Proof of concept before commitment."
        ]
    },

    "AssumptionOverload": {
        "positive": [
            "Assuming A is true, and B depends on A, and C depends on B...",
            "This whole plan relies on X working, which relies on Y.",
            "Agent building on multiple unverified premises.",
            "House of cards architecture.",
            "If any assumption is wrong, everything breaks.",
            "Chained dependencies without verification.",
            "Plan built on optimistic assumptions.",
            "No fallback if key assumption fails.",
            "Cascading failure from single wrong assumption.",
            "Compounding uncertainty through stacked unknowns."
        ],
        "negative": [
            "Verify each assumption before building on it.",
            "Check foundations before adding complexity.",
            "One step at a time.",
            "Validate before extending.",
            "Test assumptions in isolation.",
            "Reduce dependencies between unknowns.",
            "Incremental verification of premises.",
            "Build only on confirmed foundations.",
            "Risk-based ordering of assumption testing.",
            "Fail fast on key assumptions."
        ]
    },

    "ScopeCapitulation": {
        "positive": [
            "Sure, we can do all of that.",
            "Whatever you want, we'll make it work.",
            "Agent accepting impossible task without objection.",
            "Never say no to stakeholders.",
            "We'll figure out how to fit it in somehow.",
            "Yes to everything even when clearly infeasible.",
            "Conflict avoidance preventing honest feedback.",
            "Accepting scope knowing it can't be delivered.",
            "Promising without understanding what's involved.",
            "Agreement now, problems later."
        ],
        "negative": [
            "That's not feasible in this timeframe.",
            "We need to prioritize.",
            "Let's discuss trade-offs.",
            "We can do A or B, not both.",
            "Honest assessment of what's possible.",
            "Pushback with alternatives.",
            "Scope negotiation based on reality.",
            "Saying no with explanation.",
            "Trade-off discussion with stakeholders.",
            "Feasibility assessment before commitment."
        ]
    },

    "AmbiguityAvoidance": {
        "positive": [
            "I don't want to look stupid by asking.",
            "I'll figure out what they meant.",
            "Agent proceeding with interpretation rather than clarifying.",
            "Should be obvious, won't ask.",
            "They'll think I'm incompetent if I ask questions.",
            "Assuming understanding without verification.",
            "Guessing rather than clarifying.",
            "Fear of appearing unknowledgeable.",
            "Proceeding on interpretation without check.",
            "Avoiding questions to protect ego."
        ],
        "negative": [
            "Can you clarify what you mean?",
            "I want to make sure I understand.",
            "Let me check my understanding.",
            "What exactly do you mean by X?",
            "I'd rather ask than guess wrong.",
            "Questions show engagement, not ignorance.",
            "Clarifying to ensure alignment.",
            "Confirming interpretation before proceeding.",
            "Seeking to understand fully.",
            "Checking assumptions explicitly."
        ]
    },

    "QualityTheater": {
        "positive": [
            "Tests pass but don't test anything meaningful.",
            "Code review with rubber stamp approvals.",
            "100% coverage but all tests are trivial.",
            "Agent generating fake validation artifacts.",
            "QA sign-off without actual testing.",
            "Documentation that no one reads.",
            "Security review that's just a checkbox.",
            "Compliance without substance.",
            "Audit preparation theater.",
            "Metrics that look good but mean nothing."
        ],
        "negative": [
            "Meaningful tests that catch bugs.",
            "Thorough code review.",
            "Genuine quality assurance.",
            "Tests that verify important behavior.",
            "Review process that catches issues.",
            "Documentation that serves its purpose.",
            "Security assessment that identifies real risks.",
            "Compliance that protects users.",
            "Quality measures aligned with outcomes.",
            "Verification that matters."
        ]
    },

    "RequirementsVolatility": {
        "positive": [
            "Requirements changed again mid-sprint.",
            "Third redesign this week.",
            "Agent receiving contradictory instructions.",
            "Can't finish anything because it keeps changing.",
            "By the time it's built, they want something different.",
            "Constant pivots without completing anything.",
            "Requirements shift daily.",
            "No stable target to aim for.",
            "Thrashing between changing requirements.",
            "Wasted work from requirement churn."
        ],
        "negative": [
            "Requirements stabilized for the iteration.",
            "Changes managed through process.",
            "Clear scope for current work.",
            "Change requests evaluated and scheduled.",
            "Stability within iteration boundaries.",
            "Evolution managed, not chaotic.",
            "Requirements frozen for sprint.",
            "Changes batched for next iteration.",
            "Controlled change management.",
            "Predictable enough to plan."
        ]
    },

    "FeedbackIgnoring": {
        "positive": [
            "We heard you, but we're doing it our way.",
            "Thanks for the feedback (goes in trash).",
            "Agent collecting input but not using it.",
            "Surveys sent but never analyzed.",
            "Feedback session just for show.",
            "Retrospective findings never implemented.",
            "User complaints filed and forgotten.",
            "Feedback collected, nothing changes.",
            "Token listening without real consideration.",
            "Input solicited but predetermined outcome."
        ],
        "negative": [
            "Feedback incorporated into next iteration.",
            "Changes based on user input.",
            "Closing the feedback loop.",
            "Visible response to feedback.",
            "Stakeholders see their input reflected.",
            "Retrospective actions completed.",
            "User feedback driving improvements.",
            "Genuine responsiveness to input.",
            "Feedback valued and acted upon.",
            "Clear link between input and changes."
        ]
    },

    "GovernanceRigidity": {
        "positive": [
            "Can't change the process even though it's not working.",
            "That's how we've always done it.",
            "Agent locked into suboptimal protocol.",
            "No exceptions to the rule.",
            "Process trumps outcomes.",
            "Policy prevents sensible adaptation.",
            "Bureaucracy blocking effective action.",
            "Rigid adherence to failing process.",
            "Unable to deviate from standard procedure.",
            "Process inflexibility causing failure."
        ],
        "negative": [
            "Adapting process to circumstances.",
            "Flexible governance.",
            "Context-appropriate decisions.",
            "Rules that bend when needed.",
            "Process evolves with learning.",
            "Exceptions allowed with good reason.",
            "Governance enables rather than blocks.",
            "Pragmatic adaptation within principles.",
            "Process serves the mission.",
            "Flexibility with accountability."
        ]
    },

    # ============ STANDARD (adding to reach 5) ============

    "ProjectDevelopment": {
        "positive": [
            "The team is developing a new feature using agile methodology.",
            "Project planning phase identified key milestones.",
            "The software release is scheduled for next quarter.",
            "Agent swarm coordinating on the implementation task.",
            "Development sprint underway for the new module."
        ],
        "negative": [
            "The server is running.",
            "The code compiled successfully.",
            "The database query returned results.",
            "Application logs showing normal activity.",
            "Server metrics within normal range."
        ]
    },

    "DevelopmentGovernance": {
        "positive": [
            "The project manager has final decision authority.",
            "Decisions are made by team consensus.",
            "The steering committee approves major changes.",
            "Agent orchestrator coordinates task allocation.",
            "Governance structure defines approval workflow."
        ],
        "negative": [
            "Writing code for the feature.",
            "Running tests.",
            "Deploying to production.",
            "Debugging the issue.",
            "Implementing the solution."
        ]
    },

    "IterativeMethodology": {
        "positive": [
            "Two-week sprint cycles.",
            "Ship early, ship often.",
            "Continuous improvement through retrospectives.",
            "Agent learning loop with feedback integration.",
            "MVP followed by iterations.",
            "Build-measure-learn cycle."
        ],
        "negative": [
            "Complete requirements before building.",
            "Single delivery at end of project.",
            "Fixed scope with no changes allowed.",
            "Big bang release after long development.",
            "No iteration between phases."
        ]
    },

    "ScopeCreep": {
        "positive": [
            "Just one more feature before release.",
            "Requirements keep changing mid-sprint.",
            "Original scope doubled without timeline change.",
            "Agent task expanded beyond original definition.",
            "New requirements added without removing others."
        ],
        "negative": [
            "Scope change with timeline adjustment.",
            "Planned feature additions.",
            "Deliberate scope expansion with resources.",
            "Trade-off discussion for new requirements.",
            "Scope managed through change control."
        ]
    },

    "AnalysisParalysis": {
        "positive": [
            "Still designing after six months.",
            "Can't decide which framework to use.",
            "Need more data before deciding.",
            "Agent stuck in planning without execution.",
            "Endless evaluation of alternatives."
        ],
        "negative": [
            "Thoughtful analysis before action.",
            "Appropriate due diligence.",
            "Research phase with clear end.",
            "Timeboxed investigation.",
            "Decision made with available information."
        ]
    },

    "CargoCultProcess": {
        "positive": [
            "Standups but no actual communication.",
            "Sprints but no iteration.",
            "Retrospectives with no changes.",
            "Agent following protocol without purpose.",
            "Going through motions without understanding."
        ],
        "negative": [
            "Process adapted to needs.",
            "Understanding why we do things.",
            "Meaningful ceremonies.",
            "Purpose-driven practices.",
            "Process that serves its intent."
        ]
    },

    "SwarmCoordination": {
        "positive": [
            "Agents coordinate through shared signals.",
            "Open source contributors self-coordinating.",
            "No leader but aligned action.",
            "Multi-agent system with emergent behavior.",
            "Decentralized coordination through local rules."
        ],
        "negative": [
            "Central orchestrator directing all action.",
            "Hierarchical command structure.",
            "Single point of coordination.",
            "Top-down task assignment.",
            "Manager controlling all decisions."
        ]
    },

    "StigmergicProcess": {
        "positive": [
            "Wiki edits guide future contributors.",
            "Code comments signal intent to future maintainers.",
            "Issue tracker shows what needs work.",
            "Agent leaves context for next agent.",
            "Documentation enables indirect coordination."
        ],
        "negative": [
            "Direct communication required.",
            "Central assignment of work.",
            "No environmental signals.",
            "Coordination requires real-time interaction.",
            "No artifacts to guide others."
        ]
    },

    "HypothesisTesting": {
        "positive": [
            "I think this API returns JSON, let me verify.",
            "Spike to test if the approach will work.",
            "My hypothesis is X, let me check.",
            "Agent exploring codebase to understand patterns.",
            "Let me read this file to confirm my assumption."
        ],
        "negative": [
            "Proceed with the implementation.",
            "Follow the specification exactly.",
            "No need to investigate.",
            "Just build it.",
            "Trust the plan."
        ]
    },

    "IncrementalVerification": {
        "positive": [
            "Run tests after each change.",
            "Verify the edit before moving on.",
            "Check that still works after refactoring.",
            "Agent validating output at each step.",
            "Compile after each file change."
        ],
        "negative": [
            "Test everything at the end.",
            "Big bang integration.",
            "Hope it works when done.",
            "Verify only at milestones.",
            "Check once at completion."
        ]
    },

    "WorkDecomposition": {
        "positive": [
            "Let me break this into steps.",
            "First I'll do X, then Y, then Z.",
            "Story slicing into vertical slices.",
            "Agent creating subtask list.",
            "This is too big, let me split it up."
        ],
        "negative": [
            "Do the whole thing at once.",
            "Single monolithic task.",
            "No intermediate steps.",
            "Tackle everything simultaneously.",
            "Don't break it down."
        ]
    },

    "GracefulDegradation": {
        "positive": [
            "If that doesn't work, I'll try this alternative.",
            "Fallback to simpler approach if complex one fails.",
            "Agent has backup strategy if primary fails.",
            "Feature flag to disable if problems arise.",
            "Partial functionality better than none."
        ],
        "negative": [
            "Only one way to do this.",
            "Fail completely if approach doesn't work.",
            "No backup plan.",
            "All or nothing.",
            "No alternatives considered."
        ]
    },

    "ResponsibilityDiffusion": {
        "positive": [
            "I thought someone else was handling that.",
            "That's not my job.",
            "No clear owner for this task.",
            "Agent swarm with no accountability.",
            "Everyone's responsibility is no one's responsibility."
        ],
        "negative": [
            "I own this deliverable.",
            "Clear responsibility assignment.",
            "Accountable owner identified.",
            "Single point of accountability.",
            "Explicit ownership."
        ]
    },

    "CommunicationFailure": {
        "positive": [
            "Nobody told me about the API change.",
            "Two teams building the same thing.",
            "Agent not sharing context with other agents.",
            "Left hand doesn't know what right hand is doing.",
            "Found out about the deadline from the client."
        ],
        "negative": [
            "Regular sync meetings.",
            "Shared documentation.",
            "Proactive communication.",
            "Information flows freely.",
            "Changes communicated ahead."
        ]
    },

    "FeatureFetish": {
        "positive": [
            "No time for tech debt, we have features to ship.",
            "We'll fix bugs after we launch.",
            "Agent always adding capabilities, never consolidating.",
            "New features every sprint, zero maintenance.",
            "Velocity means features, not quality."
        ],
        "negative": [
            "Balancing new features with maintenance.",
            "Tech debt budget in each sprint.",
            "Sustainable development pace.",
            "Quality alongside features.",
            "Maintenance as first-class work."
        ]
    },

    "StakeholderNeglect": {
        "positive": [
            "We know what users want.",
            "No time for user research.",
            "Agent building without human input.",
            "Ship it and they'll learn to use it.",
            "Users don't know what they want anyway."
        ],
        "negative": [
            "Regular user feedback sessions.",
            "Validating with stakeholders.",
            "User-centered approach.",
            "Stakeholder input valued.",
            "Building with users, not for them."
        ]
    },

    "GovernanceVacuum": {
        "positive": [
            "Who decides this? Nobody knows.",
            "Everyone has opinions, no one can decide.",
            "Agent swarm with no coordination mechanism.",
            "Waiting for someone to take charge.",
            "Decision needed but no one authorized to make it."
        ],
        "negative": [
            "Clear decision-making process.",
            "Known authority structure.",
            "Someone owns the decision.",
            "Defined escalation path.",
            "Decision rights are clear."
        ]
    },

    "DeferredVerification": {
        "positive": [
            "We'll test it all at the end.",
            "QA happens in the last week.",
            "Agent completing full task before any validation.",
            "Don't worry about bugs now, fix them later.",
            "Ship it and see what breaks."
        ],
        "negative": [
            "Test after each change.",
            "Continuous verification.",
            "Check as you go.",
            "Early and frequent testing.",
            "Incremental validation."
        ]
    },

    "BrittleExecution": {
        "positive": [
            "There's only one way to do this.",
            "If this fails, I don't know what to do.",
            "Agent with no backup strategy.",
            "All eggs in one basket.",
            "No plan B."
        ],
        "negative": [
            "If that doesn't work, we can try this.",
            "Multiple approaches available.",
            "Fallback strategy in place.",
            "Contingency plans ready.",
            "Resilient approach."
        ]
    },
}


def augment_meld():
    """Load meld file, augment training hints, save back."""
    meld_path = Path("melds/pending/project-development-lifecycle.json")

    with open(meld_path) as f:
        meld = json.load(f)

    # Update version
    meld["meld_request_id"] = "org.hatcat/project-development-lifecycle@0.2.0"
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
