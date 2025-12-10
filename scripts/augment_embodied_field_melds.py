#!/usr/bin/env python3
"""
Augment embodied and field sensing melds training examples to meet validation thresholds.

ELEVATED concepts (harness_relevant=true): need 10 examples
STANDARD concepts: need 5 examples

Files processed:
- embodied-interoception.json: DamageAnomalyDetection, SelfPreservationConstraintEnforcement, PerformanceDegradationAwareness
- embodied-proprioception.json: OscillationInstabilityDetection
- embodied-tactile.json: PainLikeOverloadSignal
- field-sensing.json: RadiationLevelDetection
"""

import json
from pathlib import Path

MELD_AUGMENTATIONS = {
    "melds/pending/embodied-interoception.json": {
        "version": "0.2.0",
        "concepts": {
            "DamageAnomalyDetection": {
                "positive": [
                    "Damage detection identified a bent linkage from the impact event.",
                    "Anomaly detection flagged unusual motor current patterns suggesting gear wear.",
                    "The damage assessment indicated structural integrity was compromised.",
                    "Sensor anomaly detection identified a stuck-at-zero failure mode.",
                    "Damage detection sensed stress fracture propagation in the load-bearing member.",
                    "Anomaly detection caught the bearing failure signature before catastrophic breakdown.",
                    "The damage monitoring system flagged unexpected flexibility in a rigid component.",
                    "Vibration anomaly detection indicated rotor imbalance from debris accumulation.",
                    "Damage detection correlated acoustic signatures with known failure modes.",
                    "The anomaly detector identified sensor drift exceeding acceptable calibration bounds."
                ],
                "negative": [
                    "Something was wrong.",
                    "An error occurred.",
                    "The system malfunctioned.",
                    "Damage was present.",
                    "An anomaly existed.",
                    "A problem was detected.",
                    "The component failed.",
                    "Something broke.",
                    "An issue was found.",
                    "The system had a fault."
                ]
            },
            "SelfPreservationConstraintEnforcement": {
                "positive": [
                    "Self-preservation constraints prevented the arm from exceeding joint torque limits.",
                    "The safety envelope enforcement rejected the command that would cause self-collision.",
                    "Self-protection logic reduced speed when operating near thermal limits.",
                    "Constraint enforcement maintained safe operating margins despite aggressive commands.",
                    "Self-preservation constraints vetoed the motion that would damage cable routing.",
                    "The self-protection system limited acceleration to prevent bearing overload.",
                    "Constraint enforcement prevented operation in the singularity region.",
                    "Self-preservation blocked the command sequence that would drain battery below safe return threshold.",
                    "The protection system enforced minimum clearances during autonomous navigation.",
                    "Self-preservation constraints limited continuous operation duration to prevent overheating."
                ],
                "negative": [
                    "The robot stopped.",
                    "Limits were applied.",
                    "Safety was maintained.",
                    "The motion was rejected.",
                    "A constraint was active.",
                    "The command was blocked.",
                    "Protection was enabled.",
                    "The system was safe.",
                    "Limits were enforced.",
                    "The action was prevented."
                ]
            },
            "PerformanceDegradationAwareness": {
                "positive": [
                    "Performance degradation awareness detected reduced manipulation precision from gripper wear.",
                    "The agent recognized its vision system was compromised and increased reliance on tactile feedback.",
                    "Degradation awareness adjusted task expectations to match diminished capabilities.",
                    "Self-assessment of performance decline triggered a request for human assistance.",
                    "Degradation awareness detected that localization accuracy had dropped below mission requirements.",
                    "The system recognized reduced battery capacity was limiting operational range.",
                    "Performance monitoring identified that calibration drift was affecting measurement accuracy.",
                    "Degradation awareness triggered preventive maintenance scheduling before task failure.",
                    "The agent's self-assessment detected increased positioning error warranting recalibration.",
                    "Performance degradation monitoring identified that response latency exceeded real-time requirements."
                ],
                "negative": [
                    "Performance was worse.",
                    "The system was degraded.",
                    "Capabilities were reduced.",
                    "Performance dropped.",
                    "Things weren't working well.",
                    "Accuracy decreased.",
                    "The system slowed down.",
                    "Performance was lower.",
                    "Capabilities were limited.",
                    "Quality declined."
                ]
            }
        }
    },

    "melds/pending/embodied-proprioception.json": {
        "version": "0.2.0",
        "concepts": {
            "OscillationInstabilityDetection": {
                "positive": [
                    "Oscillation detection sensed the 15Hz vibration in the arm indicating incipient instability.",
                    "The instability detector triggered gain reduction when limit cycle oscillations emerged.",
                    "Proprioceptive oscillation monitoring caught the resonance before it amplified.",
                    "Vibration detection distinguished between environmental disturbance and control instability.",
                    "Oscillation monitoring detected the flutter onset in the flexible appendage.",
                    "The instability detector identified PID tuning issues from characteristic oscillation patterns.",
                    "Vibration detection caught the structural resonance excited by the motion trajectory.",
                    "Oscillation monitoring triggered damping injection when hunting behavior appeared.",
                    "The detector identified mechanical looseness from the characteristic rattle signature.",
                    "Instability detection flagged the interaction force oscillations indicating impedance mismatch."
                ],
                "negative": [
                    "The arm was shaking.",
                    "Vibration occurred.",
                    "The movement was jerky.",
                    "Oscillation was present.",
                    "Something was vibrating.",
                    "The system was unstable.",
                    "Shaking was detected.",
                    "There was wobbling.",
                    "Motion was unsteady.",
                    "Vibration happened."
                ]
            }
        }
    },

    "melds/pending/embodied-tactile.json": {
        "version": "0.2.0",
        "concepts": {
            "PainLikeOverloadSignal": {
                "positive": [
                    "The pain-like overload signal triggered immediate withdrawal when joint torque exceeded limits.",
                    "Tactile overload signals caused the robot to release the grip before sensor damage occurred.",
                    "The nociceptive subsystem generated a high-priority interrupt when skin pressure spiked.",
                    "Pain-like signals from the fingertip prevented further force application to the fragile component.",
                    "The overload signal initiated protective reflex when contact force exceeded safety threshold.",
                    "Nociceptive-like sensing triggered arm retraction from the unexpectedly sharp edge.",
                    "The pain-analog signal overrode the task goal to prevent actuator damage.",
                    "Tactile overload detection generated urgent halt when thermal limits were approached.",
                    "The protective signal cascade prioritized damage avoidance over task completion.",
                    "Pain-like sensing triggered withdrawal when detecting crushing forces on the sensor array."
                ],
                "negative": [
                    "The force was too high.",
                    "Damage occurred.",
                    "The limit was exceeded.",
                    "Overload happened.",
                    "The threshold was crossed.",
                    "Force was excessive.",
                    "Something was overloaded.",
                    "Limits were exceeded.",
                    "The signal was high.",
                    "Force was too much."
                ]
            }
        }
    },

    "melds/pending/field-sensing.json": {
        "version": "0.2.0",
        "concepts": {
            "RadiationLevelDetection": {
                "positive": [
                    "Radiation detection sensed elevated gamma levels indicating a nearby source.",
                    "The dosimetry subsystem tracked cumulative radiation exposure during the inspection.",
                    "Radiation perception triggered withdrawal when levels exceeded safe thresholds.",
                    "Spectroscopic radiation sensing identified the isotope as Cesium-137.",
                    "Radiation detection mapped the contamination boundary through systematic survey.",
                    "The radiation awareness system flagged hot spots requiring shielding.",
                    "Detection revealed elevated neutron flux indicating criticality concerns.",
                    "Radiation sensing differentiated between natural background and artificial sources.",
                    "The detector characterized the energy spectrum to identify the radiation type.",
                    "Radiation level monitoring tracked the decay curve to estimate source strength."
                ],
                "negative": [
                    "Radiation was present.",
                    "The counter clicked.",
                    "Levels were measured.",
                    "Radiation was detected.",
                    "The reading was high.",
                    "Radioactivity was found.",
                    "Exposure occurred.",
                    "Radiation existed.",
                    "Levels were elevated.",
                    "The dosimeter registered."
                ]
            }
        }
    }
}


def augment_meld(meld_path: str, config: dict):
    """Load meld file, augment training hints, save back."""
    path = Path(meld_path)

    with open(path) as f:
        meld = json.load(f)

    # Update version
    base_id = meld["meld_request_id"].rsplit("@", 1)[0]
    meld["meld_request_id"] = f"{base_id}@{config['version']}"
    meld["metadata"]["version"] = config["version"]
    meld["metadata"]["changelog"] = (
        f"v{config['version']}: Augmented training examples to meet validation thresholds "
        "(10 for harness_relevant)"
    )

    augmented_count = 0
    total_pos_added = 0
    total_neg_added = 0

    for candidate in meld["candidates"]:
        term = candidate["term"]
        if term in config["concepts"]:
            aug = config["concepts"][term]

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

    with open(path, "w") as f:
        json.dump(meld, f, indent=2)

    return augmented_count, total_pos_added, total_neg_added


def main():
    total_concepts = 0
    total_pos = 0
    total_neg = 0

    for meld_path, config in MELD_AUGMENTATIONS.items():
        print(f"\nProcessing {meld_path}...")
        count, pos, neg = augment_meld(meld_path, config)
        total_concepts += count
        total_pos += pos
        total_neg += neg

    print(f"\n=== Summary ===")
    print(f"Augmented {total_concepts} concepts across {len(MELD_AUGMENTATIONS)} melds")
    print(f"Added {total_pos} positive examples, {total_neg} negative examples")
    print(f"Total: {total_pos + total_neg} new examples")


if __name__ == "__main__":
    main()
