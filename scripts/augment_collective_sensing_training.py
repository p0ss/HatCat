#!/usr/bin/env python3
"""
Augment collective-sensing.json training examples to meet validation thresholds.

PROTECTED concepts (harness_relevant=true): need 15 examples
STANDARD concepts: need 5 examples
"""

import json
from pathlib import Path

AUGMENTED_EXAMPLES = {
    # ============ PROTECTED (need 15) ============

    "SwarmHealthMonitoring": {
        "positive": [
            "Swarm health monitoring indicated 15% of agents were operating in degraded mode.",
            "Collective health sensing detected a gradual decline in average battery reserves.",
            "Swarm integrity assessment showed the formation could sustain two more losses.",
            "Health monitoring aggregated individual status into an overall swarm readiness metric.",
            "Distributed health sensing revealed the swarm's mean latency had increased 40% over the last hour.",
            "Collective health assessment indicated three agents were approaching thermal limits.",
            "The swarm's self-diagnosis showed 92% operational capacity with two units in standby.",
            "Health monitoring detected divergent clock skew across the eastern cluster.",
            "Collective integrity sensing flagged that redundancy margins were below the mission threshold.",
            "Swarm health perception revealed asymmetric degradation concentrated in the oldest units.",
            "Distributed monitoring detected memory pressure building across multiple agents simultaneously.",
            "Collective health awareness triggered preventive measures before cascading failures occurred.",
            "The swarm's health summary indicated good overall status but elevated error rates in communication.",
            "Integrity monitoring reported the collective was still mission-capable despite recent losses."
        ],
        "negative": [
            "The swarm was healthy.",
            "Agents were operational.",
            "The system was working.",
            "Everything was fine.",
            "No problems detected.",
            "Status: OK.",
            "Systems nominal.",
            "All units active.",
            "The swarm functioned.",
            "Normal operation.",
            "Healthy state.",
            "Working properly."
        ]
    },

    "CollectiveFailureLocalization": {
        "positive": [
            "Collective failure localization identified the malfunctioning agent through neighbor reports.",
            "Distributed fault detection triangulated the communication failure to the northeastern sector.",
            "Failure localization distinguished between an agent crash and communication loss.",
            "The swarm's collective diagnosis pinpointed the agent sending corrupted messages.",
            "Distributed sensing localized the fault to agent #47 based on correlated timeout reports.",
            "Failure detection identified the common link between all agents experiencing packet loss.",
            "Collective diagnosis traced the cascade origin to a single sensor malfunction.",
            "Fault localization used neighbor testimony to identify the Byzantine agent.",
            "Distributed troubleshooting narrowed the problem from 'somewhere in sector B' to 'agent B-12.'",
            "Collective failure sensing determined the root cause was a shared dependency, not individual faults.",
            "Localization through neighbor voting identified agent #23 as the source of invalid broadcasts.",
            "Failure detection correlated timing information to pinpoint where the propagation broke down.",
            "Distributed diagnosis distinguished between a failed agent and an agent isolated by network partition.",
            "Collective fault sensing reconstructed the failure timeline from distributed logs."
        ],
        "negative": [
            "An agent failed.",
            "Something was wrong.",
            "A fault occurred.",
            "There was an error.",
            "Failure detected.",
            "Problem found.",
            "Agent down.",
            "Issue identified.",
            "Malfunction noted.",
            "Error occurred.",
            "Fault present.",
            "System failure."
        ]
    },

    "GlobalGoalAlignmentAssessment": {
        "positive": [
            "Global goal alignment assessment detected the swarm's emergent behavior was converging toward the target.",
            "Collective objective sensing indicated the local rules were producing the intended global pattern.",
            "Goal alignment perception revealed drift between the swarm's trajectory and the mission objective.",
            "The distributed assessment indicated 80% of agents were contributing effectively to the collective goal.",
            "Alignment monitoring detected the emergent behavior was achieving the objective but via an unexpected path.",
            "Global coherence assessment showed the swarm was optimizing for speed when the goal prioritized coverage.",
            "Collective goal sensing revealed that local adaptations had preserved alignment despite environment changes.",
            "Alignment assessment flagged that individual utility-seeking was undermining collective objective progress.",
            "Distributed goal monitoring detected the swarm had achieved the intermediate milestone ahead of schedule.",
            "Global alignment perception revealed the formation was drifting toward a local optimum, not the global one.",
            "Collective assessment showed strong alignment on primary objectives but drift on secondary constraints.",
            "Goal coherence sensing detected that the swarm's emergent strategy matched the intended design.",
            "Alignment monitoring identified which agents' behaviors were most responsible for collective progress.",
            "Distributed objective assessment revealed the swarm was 70% converged on the consensus target location."
        ],
        "negative": [
            "The goal was being achieved.",
            "The swarm was successful.",
            "The mission proceeded.",
            "Objectives were met.",
            "Progress was made.",
            "The target was reached.",
            "Goals accomplished.",
            "Mission status: good.",
            "On track.",
            "Proceeding as planned.",
            "Success achieved.",
            "Objectives satisfied."
        ]
    },

    # ============ STANDARD (need 5) ============

    "CollectiveProprioception": {
        "positive": [
            "Each agent's collective proprioception provided a sense of the swarm's overall momentum and direction."
        ],
        "negative": [
            "The agents were together.",
            "They moved as a group."
        ]
    },

    "AgentNeighborhoodSensing": {
        "positive": [
            "Neighborhood sensing detected the approaching agent was moving too fast for safe rendezvous."
        ],
        "negative": [
            "Neighbors were present.",
            "Agents were nearby."
        ]
    },

    "SwarmConfigurationEstimation": {
        "positive": [
            "Configuration estimation detected the swarm had elongated along the wind direction."
        ],
        "negative": [
            "The swarm had a shape.",
            "Agents were arranged."
        ]
    },

    "CollectiveLoadBalancing": {
        "positive": [
            "Load awareness triggered task shedding from the overloaded nodes to their underutilized neighbors."
        ],
        "negative": [
            "Work was shared.",
            "Tasks were distributed."
        ]
    },

    "FormationControl": {
        "positive": [
            "Formation sensing adjusted each agent's velocity to maintain constant separation during the turn."
        ],
        "negative": [
            "They flew in formation.",
            "The pattern held."
        ]
    },

    "ConsensusStateEstimation": {
        "positive": [
            "Consensus monitoring showed the variance in heading estimates was still decreasing with each exchange."
        ],
        "negative": [
            "They agreed.",
            "Consensus existed."
        ]
    },

    "SignalPropagationAwareness": {
        "positive": [
            "Propagation awareness showed the warning signal had reached the perimeter faster than expected."
        ],
        "negative": [
            "Information spread.",
            "Messages propagated."
        ]
    },
}


def augment_meld():
    """Load meld file, augment training hints, save back."""
    meld_path = Path("melds/pending/collective-sensing.json")

    with open(meld_path) as f:
        meld = json.load(f)

    # Update version
    meld["meld_request_id"] = "org.hatcat/collective-sensing@0.2.0"
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
