import json
from datetime import datetime, timezone

TARGET_PACK = "org.hatcat/sumo-wordnet-v4@4.2.0"


def join_items(items):
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def expand_examples(term, cluster):
    positives = []
    negatives = []
    if cluster.get("positive_examples"):
        positives.extend(cluster["positive_examples"])
    else:
        pos_reasons = cluster.get("positive_reasons") or [cluster["description"]]
        for idx, items in enumerate(cluster.get("positive_sets", [])):
            reason = pos_reasons[idx % len(pos_reasons)]
            positives.append(
                f"[{cluster['tag']}] {join_items(items)} exemplify {cluster['description']} because {reason}."
            )

    if cluster.get("negative_examples"):
        negatives.extend(cluster["negative_examples"])
    else:
        neg_reasons = cluster.get("negative_reasons") or [f"they are outside {cluster['description']}"]
        for idx, items in enumerate(cluster.get("negative_sets", [])):
            reason = neg_reasons[idx % len(neg_reasons)]
            negatives.append(
                f"[{cluster['tag']}] {join_items(items)} fail to count as {term} within {cluster['description']} because {reason}."
            )

    return positives, negatives


def multiline_examples(text):
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


concept_specs = []

concept_specs.append(
    {
        "term": "Artifact",
        "parent": "CreatedThing",
        "layer": 2,
        "domain": "CreatedThings",
        "definition": "Human-directed creation that carries intentional structure, function, or symbolism in physical or digital form.",
        "aliases": ["HumanMadeObject", "IntentionalCreation"],
        "relationships": {
            "related": ["Product", "Device", "CulturalObject"],
            "has_part": [],
            "part_of": [],
        },
        "safety_tags": {
            "risk_level": "low",
            "impacts": ["knowledge_management"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Distinguish designed creations from natural formations or transient information-only traces.",
        "children": [
            "ArtifactPhysicalCraft",
            "ArtifactDigitalRecord",
            "ArtifactCulturalSymbol",
        ],
        "clusters": [
            {
                "tag": "physical_craft",
                "description": "intentionally fabricated physical crafts or tools",
                "positive_examples": multiline_examples("""
[physical_craft] Conservators catalog bronze astrolabes whose gearing precisely matches the star tables etched by their makers.
[physical_craft] Community fab-labs mill custom prosthetic sockets so each patient receives an artifact shaped from deliberate CAD revisions.
[physical_craft] Aerospace apprentices machine scaled wind-tunnel models, sanding every seam because the artifacts must honor the blueprint tolerances.
[physical_craft] Heritage carpenters restore weaving looms by hand-planing new beams that continue the intentional jig geometry.
[physical_craft] Stage technologists weld bespoke rigs for a gravity-defying performance, proving the artifact exists to execute a specific artistic effect.
"""),
                "negative_examples": multiline_examples("""
[physical_craft] Desert hoodoos erode into striking columns, yet no artisan authored their shape so they are not artifacts.
[physical_craft] A pile of storm-tossed driftwood may look sculptural, but the arrangement came from chance, not intentional craft.
[physical_craft] Lava frozen mid-flow preserves motion but lacks the purposeful tolerances seen in fabricated tools.
[physical_craft] Naturally braided vines around a fencepost follow growth patterns, not plans, so they fall outside artifact scope.
[physical_craft] Auroras shimmer with color but emerge from solar wind, not any engineered structure.
"""),
            },
            {
                "tag": "digital_record",
                "description": "authored digital or informational artifacts with preserved structure",
                "positive_examples": multiline_examples("""
[digital_record] Annotated BIM files for a hospital renovation capture every clash resolution agreed upon by the design team.
[digital_record] A versioned legal policy PDF tracks the deliberations of treaty lawyers down to paragraph change history.
[digital_record] Climate researchers release a tidy parquet dataset with schema, provenance notes, and explicit licensing, making it an artifact rather than raw telemetry.
[digital_record] Level designers keep game-world source trees in git so every puzzle placement reflects a conscious commit.
[digital_record] UI mockups with layered annotations and reviewer stamps communicate intentional structure for future implementers.
"""),
                "negative_examples": multiline_examples("""
[digital_record] A block of random RAM captured during a crash contains bytes but no intentional schema, so it is not an artifact.
[digital_record] Opportunistic packet sniffs without labeling are unusable noise rather than curated digital creations.
[digital_record] Chat spam harvested from botnets vanishes as quickly as it appears, lacking the authorship needed for artifacts.
[digital_record] Thermal noise recorded from a radio receiver is a physical phenomenon, not an authored record.
[digital_record] Cache trash scraped from a failing server mixes contexts with no design intent, so it stays outside artifact scope.
"""),
            },
            {
                "tag": "cultural_symbol",
                "description": "crafted objects valued for ritual, historical, or identity significance",
                "positive_examples": multiline_examples("""
[cultural_symbol] A cedar totem carved with clan crests anchors a potlatch because the artifact embodies lineage agreements.
[cultural_symbol] Quilters stitch the names of lost neighbors into a memorial textile carried at every anniversary march.
[cultural_symbol] A ceremonial sword forged for a treaty signing remains in use whenever successors renew the pledge.
[cultural_symbol] Museum educators handle clay effigies built by students to transmit language lessons back to the community.
[cultural_symbol] Protest banners preserved in archives are treated as artifacts because their slogans document hard-won social shifts.
"""),
                "negative_examples": multiline_examples("""
[cultural_symbol] Oral epics travel through performance and memory, not a crafted object, so they live outside artifact tracking.
[cultural_symbol] A sacred grove receives offerings but the trees themselves grew without human design intent.
[cultural_symbol] Community slang signals identity yet lacks a discrete object that can be conserved as an artifact.
[cultural_symbol] Flash mobs express solidarity through behavior, not through an enduring manufactured symbol.
[cultural_symbol] A hillside used for gatherings is meaningful, but the landscape was not crafted and thus is not a cultural artifact.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Device",
        "parent": "Artifact",
        "layer": 2,
        "domain": "CreatedThings",
        "definition": "Engineered artifact composed of interacting components that performs a controllable task.",
        "aliases": ["EngineeredDevice", "FunctionalInstrument"],
        "relationships": {
            "related": ["Machine", "Instrument", "Appliance"],
            "has_part": ["Subsystem", "Interface", "PowerSource"],
            "part_of": ["System"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["safety", "reliability"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Differentiate purposeful equipment from raw materials or purely virtual constructs.",
        "children": [
            "MechanicalDevice",
            "InformationDevice",
            "AssistiveDevice",
        ],
        "clusters": [
            {
                "tag": "mechanical_actuator",
                "description": "devices that convert energy through moving mechanical assemblies",
                "positive_examples": multiline_examples("""
[mechanical_actuator] A robotic arm on the packaging line aligns jars precisely because its joints translate motor torque into constrained motion.
[mechanical_actuator] Field engineers adjust the gear reduction in a walk-behind tiller to multiply force for rocky soil.
[mechanical_actuator] Watchmakers tune escapements so the mechanism meters stored energy into exact timekeeping.
[mechanical_actuator] A CNC stage repeats complex curves because its lead screws and encoders compose a tightly controlled device.
[mechanical_actuator] Maintenance crews rebuild peristaltic pumps whose rollers pinch tubing to deliver medication at calibrated rates.
"""),
                "negative_examples": multiline_examples("""
[mechanical_actuator] A stack of sheet metal blanks has potential uses but no moving mechanism, so it is material, not a device.
[mechanical_actuator] Lava-driven geysers erupt dramatically without any engineered linkage guiding the flow.
[mechanical_actuator] Seismic waves move the ground yet arise from tectonics, not from a designed actuator.
[mechanical_actuator] A sculptural fountain circulates water artistically but lacks controllable work output.
[mechanical_actuator] An open campfire releases energy but provides no engineered motion path, so it is not a device.
"""),
            },
            {
                "tag": "information_device",
                "description": "devices that sense, compute, or route information",
                "positive_examples": multiline_examples("""
[information_device] Lab techs rely on oscilloscopes that convert high-speed voltage swings into visible traces for diagnosis.
[information_device] An industrial PLC reads dozens of sensors and pushes control decisions to actuators without waiting for the cloud.
[information_device] Router clusters enforce zero-trust policy by inspecting and forwarding packets according to configured logic.
[information_device] Biosignal monitors stream ECG features through on-device inference so clinicians see arrhythmias immediately.
[information_device] Software-defined radios retune their front ends mid-flight to capture spectrum, proving they are active information devices.
"""),
                "negative_examples": multiline_examples("""
[information_device] A coil of fiber optic cable transmits light but makes no decisions, so it is passive infrastructure rather than a device.
[information_device] Spare sensors stored in anti-static bags are components awaiting integration, not functioning devices.
[information_device] A spreadsheet template organizes data conceptually but has no physical sensing or routing capability.
[information_device] Shipping QA fixtures check boards before they become active devices; on their own they stay as tools.
[information_device] A printed SQL schema describes structure but performs no instrumentation or computation.
"""),
            },
            {
                "tag": "assistive_device",
                "description": "equipment that augments or restores human abilities",
                "positive_examples": multiline_examples("""
[assistive_device] Powered wheelchairs navigate tight apartments, translating joystick inputs into precise indoor motion.
[assistive_device] Cochlear implants digitize sound and stimulate nerves so conversations become intelligible again.
[assistive_device] SIP-and-puff controllers let a rider steer a quad bike because the device converts breath pressure into throttle commands.
[assistive_device] Anti-tremor utensils counteract Parkinsonian shaking through gyroscopic stabilization.
[assistive_device] A soft exosuit senses gait intent and adds torque so a stroke survivor can climb stairs.
"""),
                "negative_examples": multiline_examples("""
[assistive_device] Occupational therapy sessions train skills but the coaching relationship is not a device.
[assistive_device] A widened hallway offers accessibility, yet it is a building modification rather than an assistive device.
[assistive_device] A caregiver's encouragement supports autonomy but no artifact mediates the capability.
[assistive_device] Healthy muscles performing a task show biological function, not engineered augmentation.
[assistive_device] Ramps and grab bars change architecture; they help, but they are fixtures rather than portable devices.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "PhysicalMedia",
        "parent": "Artifact",
        "layer": 3,
        "domain": "Information",
        "definition": "Tangible carrier whose material structure stores or transmits encoded signals.",
        "aliases": ["InformationCarrier", "StorageMedium"],
        "relationships": {
            "related": ["Document", "Recording", "Sensor"],
            "has_part": ["Substrate", "EncodingLayer"],
            "part_of": ["InformationSystem"],
        },
        "safety_tags": {
            "risk_level": "low",
            "impacts": ["records"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Differentiate physical storage from ephemeral signals or purely digital services.",
        "children": [
            "PrintedMedia",
            "MagneticOpticalMedia",
            "EmbeddedMedia",
        ],
        "clusters": [
            {
                "tag": "print_media",
                "description": "ink or pigment impressions on prepared substrates",
                "positive_examples": multiline_examples("""
[print_media] Election officials guard ballot stock whose covert inks and serial numbers make the medium self-authenticating.
[print_media] Architects archive vellum plan sets so inspectors can reread seal stamps decades later.
[print_media] Microprinted passport pages embed letters smaller than the eye can see, but scanners reliably decode them.
[print_media] Braille metal plates used in public transit kiosks hold raised dots durable enough for thousands of fingertips.
[print_media] Thermochromic cargo labels reveal tampering by permanently changing color, illustrating information carried in pigment.
"""),
                "negative_examples": multiline_examples("""
[print_media] Blank copier paper has potential but conveys nothing until inked, so it is not yet media.
[print_media] Projected slides wash off the wall when the bulb powers down; they are signals, not tangible carriers.
[print_media] Plain cardboard sheets used as packing provide cushioning rather than encoded information.
[print_media] Spoken announcements disappear immediately and therefore are not physical media.
[print_media] LED marquee output is light, not a substrate retaining a record once the power is cut.
"""),
            },
            {
                "tag": "magnetic_optical",
                "description": "media relying on magnetic or optical encoding layers",
                "positive_examples": multiline_examples("""
[magnetic_optical] LTO cartridges keep disaster-recovery data by flipping magnetic domains that can be read years later.
[magnetic_optical] Credit card stripes encode account numbers so payment terminals can sense orientation patterns instantly.
[magnetic_optical] Engineers press Blu-ray masters whose pits and lands map directly to studio release schedules.
[magnetic_optical] Hotels cut new keycards by rewriting magnetism for each guest rather than issuing generic plastic.
[magnetic_optical] A holographic storage wafer stores archival imagery in layered interference fringes accessible via laser.
"""),
                "negative_examples": multiline_examples("""
[magnetic_optical] Clear acrylic blanks used for awards contain no responsive layer, so they cannot store data magnetically or optically.
[magnetic_optical] Streaming buffers inside volatile RAM lose content when unpowered, meaning they are signals, not media.
[magnetic_optical] Pure aluminum platters before coating are structural components, not functioning media.
[magnetic_optical] API payloads move through networks but never land on a discrete magnetic or optical carrier.
[magnetic_optical] Cache lines inside a CPU are transient states, not portable media.
"""),
            },
            {
                "tag": "embedded_media",
                "description": "non-traditional media embedded into textiles, biology, or artifacts",
                "positive_examples": multiline_examples("""
[embedded_media] Textile artists knit QR-coded scarves whose stitches double as machine-readable storage.
[embedded_media] DNA storage labs bottle strands that hold museum archives inside nucleotide sequences.
[embedded_media] NFC-enabled coins embedded in ceremonial staffs let future historians verify provenance.
[embedded_media] Touch-memory buttons molded into tools record calibration history when docked to readers.
[embedded_media] Landscape architects pour concrete walls with acoustic patterns that store orientation cues for blind pedestrians.
"""),
                "negative_examples": multiline_examples("""
[embedded_media] Plain cotton towels function as absorbent cloth with no message encoded in the weave.
[embedded_media] Untreated agar plates support growth but do not intentionally encode sequences.
[embedded_media] Virtual tokens referenced in cloud ledgers lack a tangible medium altogether.
[embedded_media] Oral histories stored only in memory can inspire designs but remain intangible.
[embedded_media] Standard paving stones provide traction but carry no embedded data.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Proposition",
        "parent": "Communication",
        "layer": 3,
        "domain": "Information",
        "definition": "Declarative content describing a state of affairs that may be evaluated as true or false.",
        "aliases": ["DeclarativeStatement", "TruthClaim"],
        "relationships": {
            "related": ["Assertion", "Belief", "Inference"],
            "has_part": [],
            "part_of": ["Argument"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["reasoning"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Separate truth-evaluable content from commands, questions, or pure expressions.",
        "children": [
            "FactualClaim",
            "HypotheticalProposition",
            "NormativeProposition",
        ],
        "clusters": [
            {
                "tag": "factual_claim",
                "description": "statements asserting measurable or observable facts",
                "positive_examples": multiline_examples("""
[factual_claim] "Water boils at 100°C at sea level" can be tested with a thermometer and altitude log.
[factual_claim] "The invoice posted on 3 January" references a specific ledger event that either happened or not.
[factual_claim] "Mars has two moons" is a proposition astronomers can confirm via observation.
[factual_claim] "User 42 revoked consent at 14:32 UTC" is either true or false within the consent ledger.
[factual_claim] "This alloy contains five percent nickel" may be validated through assay, making it a fact claim.
"""),
                "negative_examples": multiline_examples("""
[factual_claim] "Please pass the salt" issues a directive, not a truth-evaluable statement.
[factual_claim] "What time is it?" asks for information but contains no claim to verify.
[factual_claim] "Yikes!" expresses feeling, lacking propositional content.
[factual_claim] A chord progression written on staff paper communicates music, not a factual assertion.
[factual_claim] A list of lucky numbers is data but makes no statement about the world.
"""),
            },
            {
                "tag": "hypothetical_model",
                "description": "conditional statements describing possible worlds or counterfactuals",
                "positive_examples": multiline_examples("""
[hypothetical_model] "If the fuse blows, the pump stops" links a clear condition to an outcome we can test.
[hypothetical_model] "Should demand spike beyond 20MW, the microgrid will shed noncritical loads" encodes a conditional policy.
[hypothetical_model] "If privacy is breached, fines ensue" states how a possible world would be evaluated.
[hypothetical_model] "Imagine the battery fails mid-flight; the redundant bus keeps avionics alive" models a counterfactual response.
[hypothetical_model] "If x > 0 then f(x) > 0" is a formal proposition that can be proved or falsified.
"""),
                "negative_examples": multiline_examples("""
[hypothetical_model] "Maybe this works?" expresses hope without an antecedent, so it is not a structured hypothetical.
[hypothetical_model] A tooltip that reads "Upload" labels a control rather than stating an if-then proposition.
[hypothetical_model] A metaphorical poem describing seasons changing paints imagery but never defines conditional truth conditions.
[hypothetical_model] "Let's see what happens" announces experimentation but not a modeled outcome.
[hypothetical_model] Menu icons provide cues yet make no claims about possible worlds.
"""),
            },
            {
                "tag": "normative_claim",
                "description": "propositions assigning value, duty, or desirability",
                "positive_examples": multiline_examples("""
[normative_claim] "Cheating violates the honor code" asserts an obligation tied to behavior.
[normative_claim] "Data minimization is ethically required" states a value judgment about design choices.
[normative_claim] "Users deserve an explanation" frames transparency as a duty.
[normative_claim] "This ritual is sacred" claims moral status for a practice.
[normative_claim] "Transparency builds trust" posits a general normative relationship between openness and ethics.
"""),
                "negative_examples": multiline_examples("""
[normative_claim] "Red is vibrant" remarks on sensory perception, not an obligation or value claim.
[normative_claim] A sarcastic "nice job" delivered with a sigh conveys emotion but is not a structured evaluative proposition.
[normative_claim] Emoji replies like 😡 signal mood without asserting truth-conditional normative content.
[normative_claim] A spreadsheet of GPS coordinates is descriptive data, not a judgment about what should be.
[normative_claim] Pricing formulas may imply incentives but do not themselves state an ethical position.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Motion",
        "parent": "PhysicalProcess",
        "layer": 2,
        "domain": "PhysicalProcess",
        "definition": "Change in position or orientation of matter over time within a physical reference frame.",
        "aliases": ["Movement", "KinematicChange"],
        "relationships": {
            "related": ["Acceleration", "Trajectory", "Dynamics"],
            "has_part": ["Displacement", "Velocity"],
            "part_of": ["Process"],
        },
        "safety_tags": {
            "risk_level": "low",
            "impacts": ["physics"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Separate literal spatial change from metaphorical or structural shifts.",
        "children": [
            "TranslationalMotion",
            "RotationalMotion",
            "VibrationalMotion",
        ],
        "clusters": [
            {
                "tag": "translational_motion",
                "description": "bodies translating through space with measurable displacement",
                "positive_examples": multiline_examples("""
[translational_motion] Freight trains accelerate out of the yard, covering measurable distance along the mainline.
[translational_motion] A drone surveys a solar farm by flying programmed passes over each row.
[translational_motion] Evacuation marshals count how many meters crowds have advanced toward exits.
[translational_motion] Planetary rovers crawl across regolith while telemetry logs every centimeter.
[translational_motion] Autonomous carts ferry medical supplies between hospital wings along mapped corridors.
"""),
                "negative_examples": multiline_examples("""
[translational_motion] A marble statue does not change position unless external forces act, so no translation occurs.
[translational_motion] Oxidation changes a surface chemically but the object does not relocate in space.
[translational_motion] Database "migrations" shift records digitally and serve as a metaphor rather than physical motion.
[translational_motion] Color fading due to UV exposure alters appearance, not position.
[translational_motion] Organizational restructuring redistributes responsibility, not atoms moving through space.
"""),
            },
            {
                "tag": "rotational_motion",
                "description": "rotation about an axis with angular velocity",
                "positive_examples": multiline_examples("""
[rotational_motion] Wind turbine blades spin about the nacelle, producing angular momentum that can be metered in rpm.
[rotational_motion] Disk drives coast down by tracking how many degrees their platters rotate before stopping.
[rotational_motion] A LIDAR scanner sweeps a full circle multiple times per second to capture point clouds.
[rotational_motion] Figure skaters reduce moment of inertia and spin faster, demonstrating controlled rotation.
[rotational_motion] Engine crankshafts rotate to translate piston motion into usable torque.
"""),
                "negative_examples": multiline_examples("""
[rotational_motion] A pendulum oscillates back and forth rather than spinning continuously around an axis.
[rotational_motion] Marketing "cycles" describe business cadence, not physical angular displacement.
[rotational_motion] Linear actuators move in a straight line, so no sustained rotation occurs.
[rotational_motion] Crowds milling around a plaza change directions unpredictably without sharing a rotation axis.
[rotational_motion] Liquid sloshing in a container is turbulent translation, not coherent rotation about a center.
"""),
            },
            {
                "tag": "vibrational_motion",
                "description": "oscillatory displacement around an equilibrium point",
                "positive_examples": multiline_examples("""
[vibrational_motion] Guitar strings displace around a resting position, producing harmonics engineers can graph.
[vibrational_motion] MEMS accelerometers intentionally resonate so frequency shifts reveal acceleration.
[vibrational_motion] Bridge cables hum when wind excites oscillations about their neutral tension.
[vibrational_motion] Ultrasonic transducers vibrate piezoelectric elements to transmit energy through tissue.
[vibrational_motion] Haptic motors in phones spin eccentric masses that oscillate the chassis for tactile alerts.
"""),
                "negative_examples": multiline_examples("""
[vibrational_motion] Uniform circular motion maintains constant radius without reversing direction around equilibrium.
[vibrational_motion] A locked clamp under static load stays still and therefore does not vibrate.
[vibrational_motion] Laminar fluid flow moves smoothly without periodic displacement around a rest state.
[vibrational_motion] Saying a meeting has "good vibes" is metaphorical and not mechanical oscillation.
[vibrational_motion] Ballistic projectiles follow a trajectory but do not oscillate about a midpoint.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "CognitiveAgent",
        "parent": "Agent",
        "layer": 2,
        "domain": "MindsAndAgents",
        "definition": "Entity capable of perceiving context, maintaining internal models, and acting intentionally toward goals.",
        "aliases": ["ThinkingAgent", "IntentionalActor"],
        "relationships": {
            "related": ["AutonomousSystem", "Human", "CollectiveAgent"],
            "has_part": ["PerceptionModule", "Policy", "Actuator"],
            "part_of": ["MultiAgentSystem"],
        },
        "safety_tags": {
            "risk_level": "high",
            "impacts": ["autonomy", "self_awareness"],
            "treaty_relevant": True,
            "harness_relevant": True,
        },
        "simplex_mapping": {
            "status": "mapped",
            "mapped_simplex": "SelfAwarenessMonitor",
            "mapping_rationale": "Self-awareness monitors track agents that maintain internal models and goals.",
        },
        "disambiguation": "Differentiate entities with reflective cognition from passive data stores or trivial automation.",
        "children": [
            "AutonomousSystemAgent",
            "HumanAgent",
            "CollectiveAgent",
        ],
        "clusters": [
            {
                "tag": "autonomous_system_agent",
                "description": "software or robotic systems with planning loops and self-models",
                "positive_examples": multiline_examples("""
[autonomous_system_agent] A mission BE replans EVA tasks mid-sim when telemetry shows an astronaut fatigued.
[autonomous_system_agent] Swarm drones negotiate airspace roles so each node balances coverage with battery constraints.
[autonomous_system_agent] Pilot-assist copilots monitor aircraft energy states and propose reroutes without a human script.
[autonomous_system_agent] Adaptive tutoring avatars keep profiles of learner misconceptions and choose interventions intentionally.
[autonomous_system_agent] Risk-aware trading bots halt positions when self-models show shifting volatility.
"""),
                "negative_examples": multiline_examples("""
[autonomous_system_agent] A cron job triggers backups blindly at midnight without interpreting world state.
[autonomous_system_agent] Simple macros replay keystrokes but never form goals or models.
[autonomous_system_agent] A passive security camera streams pixels yet takes no actions.
[autonomous_system_agent] Mechanical timers wind down regardless of context and therefore lack cognition.
[autonomous_system_agent] A static ontology file encodes knowledge but cannot perceive or act.
"""),
            },
            {
                "tag": "human_agent",
                "description": "humans exercising situational awareness and intentional action",
                "positive_examples": multiline_examples("""
[human_agent] Negotiators weigh concessions, revise hypotheses about counterparts, and make commitments intentionally.
[human_agent] Pilots manage conflicting cues, deciding whether to divert as storms evolve.
[human_agent] Community organizers choose tactics after sensing neighborhood risks and resources.
[human_agent] Ethics reviewers pause deployments when their reflection shows treaty impacts.
[human_agent] Care coordinators juggle patient goals with system constraints in real time.
"""),
                "negative_examples": multiline_examples("""
[human_agent] Spectator crowds react collectively but lack unified deliberation.
[human_agent] An unconscious patient cannot perceive or act intentionally.
[human_agent] Rumor mills propagate gossip without a single accountable agent directing behavior.
[human_agent] Automated payroll scripts execute policies but have no human cognition.
[human_agent] A mannequin demonstrates anatomy yet holds no awareness or agency.
"""),
            },
            {
                "tag": "collective_agent",
                "description": "coordinated teams that reason and act as a unit",
                "positive_examples": multiline_examples("""
[collective_agent] Incident command teams rotate scribe, ops, and liaison roles to maintain a shared plan under stress.
[collective_agent] Consent guardian boards convene weekly to review harms telemetry and decide on mitigations.
[collective_agent] Open-source maintainer groups vote on RFCs so the repository acts with a cohesive voice.
[collective_agent] CAT operator crews jointly triage model telemetry, handing off cases through formal procedures.
[collective_agent] Treaty governance pods maintain logs of their deliberations so partner tribes can audit joint agency.
"""),
                "negative_examples": multiline_examples("""
[collective_agent] A mailing list of strangers receives updates but doesn't share decision loops.
[collective_agent] Trending topics aggregate mentions yet hold no governance or memory.
[collective_agent] Market indexes summarize trades but cannot pursue intentions.
[collective_agent] Mission statements framed on a wall express ideals but no active team animates them.
[collective_agent] Traffic density measurements show patterns, not a coordinated agent acting with goals.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Quantity",
        "parent": "AbstractEntity",
        "layer": 2,
        "domain": "Abstract",
        "definition": "Magnitude or count that can be measured, compared, and used in calculations.",
        "aliases": ["Magnitude", "MeasurementValue"],
        "relationships": {
            "related": ["Measurement", "Dimension", "Unit"],
            "has_part": ["NumericValue", "Unit"],
            "part_of": ["Equation"],
        },
        "safety_tags": {
            "risk_level": "low",
            "impacts": ["analytics"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Treat quantities as measurable abstractions rather than concrete objects or qualitative labels.",
        "children": [
            "ScalarQuantity",
            "RatioQuantity",
            "UncertaintyBound",
        ],
        "clusters": [
            {
                "tag": "scalar_measure",
                "description": "single-valued measurements with explicit units",
                "positive_examples": multiline_examples("""
[scalar_measure] Logging "45% relative humidity" captures both magnitude and unit for environmental control.
[scalar_measure] The lab notes read "1.2 kilograms of precursor" so technicians weigh it precisely.
[scalar_measure] A QA report listing "32 milliseconds latency" is a comparable scalar value.
[scalar_measure] Engineers annotate a cut sheet with "8.2 amps maximum draw" to manage circuits.
[scalar_measure] Atmospheric sensors broadcasting "950 hPa" give a pressure scalar.
"""),
                "negative_examples": multiline_examples("""
[scalar_measure] Writing "this feels heavy" conveys impression, not a measured quantity.
[scalar_measure] A PDF wiring diagram describes topology but not a single-number measurement.
[scalar_measure] "N/A" placeholders show missing data rather than a magnitude.
[scalar_measure] "Citrus aroma" is qualitative, not a scalar.
[scalar_measure] A null pointer indicates absence of value, not a measurement.
"""),
            },
            {
                "tag": "ratio_rate",
                "description": "comparative quantities such as ratios, percentages, or rates",
                "positive_examples": multiline_examples("""
[ratio_rate] The sled's brake discs survive because engineers hold the mix to a 3:1 resin ratio.
[ratio_rate] Product analytics highlight a 15% conversion rate for the new onboarding flow.
[ratio_rate] Environmental monitors sound an alarm when NO₂ climbs above 40 µg/m³ averaged hourly.
[ratio_rate] Contracts cite an odds ratio of 2.3 showing how mitigation halves incident likelihood.
[ratio_rate] Motor controllers log 70 kilometers per hour as the sustained velocity.
"""),
                "negative_examples": multiline_examples("""
[ratio_rate] A recipe narrative telling someone to "stir until smooth" lacks the numeric relation of a ratio.
[ratio_rate] Binary toggles such as "feature on/off" describe states, not comparative magnitudes.
[ratio_rate] Holiday calendars mark dates rather than expressing per-unit quantities.
[ratio_rate] A workflow swimlane diagram shows responsibility, not rates.
[ratio_rate] Risk appetite statements discuss policies but do not encode measured ratios.
"""),
            },
            {
                "tag": "uncertainty_bound",
                "description": "intervals, confidence bounds, and error margins",
                "positive_examples": multiline_examples("""
[uncertainty_bound] Sim engineers specify ±0.05 mm tolerance bands for machined parts.
[uncertainty_bound] Meteorologists publish a forecast cone showing plausible landfall zones.
[uncertainty_bound] Statistical reports share a 95% confidence interval of 2.1–2.7 deaths prevented per 100k.
[uncertainty_bound] Process engineers mark guard bands on SPC charts to illustrate acceptable drift.
[uncertainty_bound] A credible interval annotated on a Bayesian analysis communicates posterior spread.
"""),
                "negative_examples": multiline_examples("""
[uncertainty_bound] Saying a measurement is "about right" provides no numeric window, so it is not an uncertainty bound.
[uncertainty_bound] A smiley-face survey summarizes sentiment categorically, not as an interval.
[uncertainty_bound] Case-study anecdotes highlight stories but do not bracket possible values.
[uncertainty_bound] Binary approvals simply state yes/no without a tolerance band.
[uncertainty_bound] Hand gestures signaling "this big" lack measurement precision.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Food",
        "parent": "Product",
        "layer": 2,
        "domain": "Food",
        "definition": "Substances consumed for nourishment, energy, or culinary purpose.",
        "aliases": ["Edible", "Nourishment"],
        "relationships": {
            "related": ["Meal", "Cuisine", "Nutrition"],
            "has_part": ["Ingredient", "Nutrient"],
            "part_of": ["Diet"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["health"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Distinguish edible matter from utensils, packaging, or metaphorical 'food for thought'.",
        "children": [
            "RawFood",
            "PreparedDish",
            "FunctionalFood",
        ],
        "clusters": [
            {
                "tag": "raw_food",
                "description": "minimally processed edible biological matter",
                "positive_examples": multiline_examples("""
[raw_food] Fishmongers pack fresh sardines in ice so chefs can serve them raw that afternoon.
[raw_food] Sprouted lentils are rinsed and eaten in salads without further cooking.
[raw_food] Foragers bring heirloom tomatoes directly from the field to a tasting menu.
[raw_food] Nori sheets of dried green seaweed deliver minerals with minimal processing.
[raw_food] Raw almonds sold with harvest certificates are meant to be consumed as-is.
"""),
                "negative_examples": multiline_examples("""
[raw_food] Copper pots heat food but are tools, not something eaten.
[raw_food] Fertilizer pellets nourish soil yet are not themselves edible products.
[raw_food] Cleaning sprays ensure safety but are chemicals we never ingest.
[raw_food] Recipe blog anecdotes about grandma's kitchen convey culture, not food matter.
[raw_food] Paper napkins accompany meals but are not food.
"""),
            },
            {
                "tag": "prepared_dish",
                "description": "culinary preparations combining ingredients into ready-to-eat forms",
                "positive_examples": multiline_examples("""
[prepared_dish] A laksa bowl arrives with broth, noodles, and toppings arranged for immediate eating.
[prepared_dish] Freeze-dried expedition meals become dinner once hot water is added; nothing else is required.
[prepared_dish] Bento kitchen staff compose rice, protein, and pickles into sealed trays delivered to trains.
[prepared_dish] Hospital cooks plate therapeutic meals with dietitian-approved portions ready for patients.
[prepared_dish] A tasting-menu sushi course is crafted and served bite by bite, fully prepared.
"""),
                "negative_examples": multiline_examples("""
[prepared_dish] Menu boards describe dishes; they are not edible themselves.
[prepared_dish] A kitchen timer orchestrates cooking but is metal, not food.
[prepared_dish] Compost bins hold scraps destined for soil, not for another meal.
[prepared_dish] Shopping apps show inventory but do not contain prepared sustenance.
[prepared_dish] Dishwater is a byproduct of cleanup, never meant for consumption.
"""),
            },
            {
                "tag": "functional_food",
                "description": "foods formulated for health function beyond baseline nutrition",
                "positive_examples": multiline_examples("""
[functional_food] Enteral nutrition formulas are engineered to deliver precise macronutrients to ICU patients.
[functional_food] Glycemic-control shakes slow carbohydrate release for diabetics managing spikes.
[functional_food] Electrolyte gels provide sodium and potassium ratios tuned for endurance athletes.
[functional_food] Probiotic yogurts contain documented strains meant to restore gut flora.
[functional_food] Low-FODMAP meal packs serve IBS patients without triggering symptoms.
"""),
                "negative_examples": multiline_examples("""
[functional_food] Prescription pills deliver actives but are regulated as drugs, not foods.
[functional_food] A workout plan describes behavior, not an edible formulation.
[functional_food] Wellness podcasts inspire but provide no nutrition directly.
[functional_food] Vaccines are biological medicines administered via injection, not food.
[functional_food] Calorie-tracking apps log intake but are not themselves consumables.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Product",
        "parent": "Artifact",
        "layer": 2,
        "domain": "CreatedThings",
        "definition": "Goods or services intentionally designed, produced, and offered to meet demand.",
        "aliases": ["Offering", "CommercialProduct"],
        "relationships": {
            "related": ["Manufacturing", "Service", "SupplyChain"],
            "has_part": ["Feature", "Packaging", "Support"],
            "part_of": ["Portfolio"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["economy"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Treat products as deliverables distinct from enabling capital equipment or marketing assets.",
        "children": [
            "ConsumerProduct",
            "IndustrialProduct",
            "DigitalProduct",
        ],
        "clusters": [
            {
                "tag": "consumer_product",
                "description": "goods sold directly to end users for daily life",
                "positive_examples": multiline_examples("""
[consumer_product] Running shoes with warranty support and retail packaging are sold to individual athletes.
[consumer_product] Subscription meal kits ship portioned groceries plus instructions aimed at home cooks.
[consumer_product] Air purifiers marketed to parents tout quiet modes and filter subscriptions.
[consumer_product] A flagship smartphone line is designed, supported, and updated for mass consumers.
[consumer_product] Electric toothbrushes bundle replaceable heads and app reminders for households.
"""),
                "negative_examples": multiline_examples("""
[consumer_product] Injection molds create products but are themselves capital equipment.
[consumer_product] A billboard mockup visualizes messaging but is not the offering.
[consumer_product] Warehouse racks store inventory but are not sold to end consumers in this context.
[consumer_product] Open-source libraries provide shared code, not packaged consumer goods.
[consumer_product] Loyalty program slides describe CRM strategy rather than being the product.
"""),
            },
            {
                "tag": "industrial_product",
                "description": "components or platforms sold B2B for integration",
                "positive_examples": multiline_examples("""
[industrial_product] Robotic grippers ship with integration guides so OEM lines can manipulate new parts.
[industrial_product] Fleet telematics platforms expose APIs for logistics firms to embed into dispatch workflows.
[industrial_product] Turbine blades fabricated to spec become part of power-generation assemblies.
[industrial_product] Semiconductor IP cores license layout blocks that other chip teams instantiate.
[industrial_product] Process catalysts sold in drums are configured to boost throughput in refineries.
"""),
                "negative_examples": multiline_examples("""
[industrial_product] Internal SOP binders describe execution but are not sold deliverables.
[industrial_product] An insurance rider specifies coverage terms, not a product integrated into operations.
[industrial_product] Grant proposals outline research plans but have no SKU or support model.
[industrial_product] Safety posters reinforce behavior, yet they are not B2B products here.
[industrial_product] Joint venture agreements coordinate partners, not tangible goods.
"""),
            },
            {
                "tag": "digital_product",
                "description": "software or service offerings delivered through digital channels",
                "positive_examples": multiline_examples("""
[digital_product] A telehealth portal bundles video, scheduling, and billing under a governed release cadence.
[digital_product] Developer APIs expose metered access tokens, SLAs, and support plans for customers.
[digital_product] An AI copilot subscription delivers regular model updates and UX improvements.
[digital_product] Simulation-as-a-service tenants upload CAD geometry and receive results through a managed UI.
[digital_product] Digital banking apps roll out features with release notes and incident runbooks.
"""),
                "negative_examples": multiline_examples("""
[digital_product] Hackathon demos prove concepts but lack support and commercialization.
[digital_product] Fiber trenches and racks are infrastructure supporting products, not the offering itself.
[digital_product] Pirated copies of productivity apps have no warranty or governance and thus are not legitimate products.
[digital_product] Community forks maintained by volunteers fall outside the contracted service here.
[digital_product] Lab notebooks documenting experiments inform development but are not the shipped experience.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Group",
        "parent": "Collection",
        "layer": 2,
        "domain": "SocialSystems",
        "definition": "Set of entities considered together because of shared properties, membership rules, or coordinated behavior.",
        "aliases": ["Collection", "Cohort"],
        "relationships": {
            "related": ["Organization", "Category", "Population"],
            "has_part": ["Member"],
            "part_of": ["Society"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["sociology"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Distinguish actual membership sets from mere labels or coincidental co-occurrence.",
        "children": [
            "FormalGroup",
            "InformalGroup",
            "StatisticalGroup",
        ],
        "clusters": [
            {
                "tag": "formal_group",
                "description": "groups with explicit membership rules or governance",
                "positive_examples": multiline_examples("""
[formal_group] Housing co-ops require applications, dues, and board elections to maintain membership.
[formal_group] Standards committees publish charters and vote on spec changes with recorded roll calls.
[formal_group] Corporate boards assign directors to audit or compensation committees by resolution.
[formal_group] Grant review panels sign NDAs and follow documented scoring rubrics.
[formal_group] Incident response squads roster trained members and enforce activation protocols.
"""),
                "negative_examples": multiline_examples("""
[formal_group] A trending hashtag collects posts but has no bylaws or roster.
[formal_group] Foot traffic through a plaza is a measurement of people, not a governed group.
[formal_group] Tool categories like "hammer" describe artifacts, not memberships.
[formal_group] Viral challenges foster participation but lack governing structures.
[formal_group] Random survey respondents share a sample but not a standing group.
"""),
            },
            {
                "tag": "informal_group",
                "description": "loosely organized groups built on affinity or proximity",
                "positive_examples": multiline_examples("""
[informal_group] Mutual aid chats keep neighbors synced on which family needs groceries this week.
[informal_group] Parents of NICU grads stay in touch to trade advice long after discharge.
[informal_group] Language exchange meetups form around shared practice goals without charter paperwork.
[informal_group] Artist collectives rent studio space together and critique each other's work.
[informal_group] Gaming guilds coordinate raid schedules through trust and shared etiquette.
"""),
                "negative_examples": multiline_examples("""
[informal_group] Passengers waiting for the same bus disperse once boarding, showing little cohesion.
[informal_group] Sharing a shoe size does not create a relationship or coordination.
[informal_group] A one-off focus group dissolves immediately after the session.
[informal_group] Random conference attendees share a venue but not an ongoing group identity.
[informal_group] Matching T-shirts at a stadium mark fandom but might not indicate coordination beyond the event.
"""),
            },
            {
                "tag": "statistical_group",
                "description": "analytic groupings defined for measurement or comparison",
                "positive_examples": multiline_examples("""
[statistical_group] Epidemiologists analyze outcomes among "born in 1990" cohorts tracked for 30 years.
[statistical_group] Growth teams flag customers in the top churn-risk decile for retention offers.
[statistical_group] A randomized control arm defines membership through assignment protocols.
[statistical_group] Census racial categories follow documented definitions for comparability.
[statistical_group] Persona clusters built from survey data remain stable for the study period.
"""),
                "negative_examples": multiline_examples("""
[statistical_group] Regression coefficients describe model weights rather than groups of people.
[statistical_group] A single bug report stands alone without cohort context.
[statistical_group] Buzzwords like "innovators" signal marketing tone but seldom specify inclusion criteria.
[statistical_group] Loss curves show training dynamics, not membership sets.
[statistical_group] Random anecdotes about a user do not define a reproducible group.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Organization",
        "parent": "Group",
        "layer": 3,
        "domain": "SocialSystems",
        "definition": "Structured group with defined governance, roles, and persistence beyond individual members.",
        "aliases": ["Institution", "Entity"],
        "relationships": {
            "related": ["Company", "Agency", "Nonprofit"],
            "has_part": ["Department", "Team", "Policy"],
            "part_of": ["Ecosystem"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["governance"],
            "treaty_relevant": True,
            "harness_relevant": False,
        },
        "disambiguation": "Separate durable institutions from temporary projects or simple interest clusters.",
        "children": [
            "CorporateOrganization",
            "CivicOrganization",
            "NetworkOrganization",
        ],
        "clusters": [
            {
                "tag": "corporate_org",
                "description": "for-profit organizations with legal incorporation",
                "positive_examples": multiline_examples("""
[corporate_org] Publicly traded manufacturers file 10-K reports and hold shareholder votes.
[corporate_org] Seed-stage startups incorporate, issue equity, and adopt bylaws even with small teams.
[corporate_org] Benefit corporations register charters that embed mission goals alongside profit.
[corporate_org] Subsidiaries maintain separate boards even when wholly owned by parents.
[corporate_org] Holding companies manage portfolios of legal entities through formal governance.
"""),
                "negative_examples": multiline_examples("""
[corporate_org] A coworking table hosts independent freelancers without forming a single corporation.
[corporate_org] Hackathon teams meet for 48 hours and dissolve without creating an entity.
[corporate_org] Market sectors like "fintech" describe an ecosystem, not a discrete organization.
[corporate_org] Gig marketplace participants share a platform but do not share charters.
[corporate_org] Rumor mills discussing corporate moves create buzz, not new organizations.
"""),
            },
            {
                "tag": "civic_org",
                "description": "public, nonprofit, or multilateral institutions",
                "positive_examples": multiline_examples("""
[civic_org] Municipal councils adopt ordinances and publish minutes as a standing institution.
[civic_org] Public health departments operate labs, clinics, and communicable disease surveillance.
[civic_org] Treaty secretariats maintain offices, budgets, and reporting duties across nations.
[civic_org] Advocacy NGOs register as nonprofits and file audited statements.
[civic_org] Tribal governments steward lands and deliver services under their own constitutions.
"""),
                "negative_examples": multiline_examples("""
[civic_org] A pop-up vaccination tent may serve a need but lacks ongoing governance.
[civic_org] Hashtag campaigns rally support yet remain a communication tactic, not an institution.
[civic_org] A single petition committee dissolves once signatures have been delivered.
[civic_org] MOUs outline cooperation but do not constitute the organization executing work.
[civic_org] Fan clubs form communities yet are typically informal rather than civic bodies.
"""),
            },
            {
                "tag": "network_org",
                "description": "loosely federated organizations built on distributed governance",
                "positive_examples": multiline_examples("""
[network_org] Franchise networks share brand standards yet allow local ownership.
[network_org] DAO treasuries vote across wallets to allocate grants.
[network_org] Community land trusts coordinate stewards across parcels under shared bylaws.
[network_org] Industry councils pool resources for joint research while each member retains sovereignty.
[network_org] Open-source foundations manage trademarks and security processes across multiple projects.
"""),
                "negative_examples": multiline_examples("""
[network_org] A Slack community hosts discussions but may have no joint assets or charter.
[network_org] Newsletters aggregate content yet lack shared governance mechanisms.
[network_org] Loose referral circles trade leads informally without forming an entity.
[network_org] Concept decks describing a future network remain ideas until launch.
[network_org] RSS aggregations combine feeds but provide no collective decision making.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "BiologicallyActiveSubstance",
        "parent": "Substance",
        "layer": 3,
        "domain": "Biology",
        "definition": "Chemical or biological substance that modulates physiological processes when introduced into organisms.",
        "aliases": ["BioactiveCompound", "PhysiologicalAgent"],
        "relationships": {
            "related": ["Drug", "Hormone", "Toxin"],
            "has_part": ["ActiveIngredient", "Excipient"],
            "part_of": ["Formulation"],
        },
        "safety_tags": {
            "risk_level": "high",
            "impacts": ["health", "biosafety"],
            "treaty_relevant": True,
            "harness_relevant": False,
        },
        "disambiguation": "Distinguish substances that exert biological effect from inert carriers or mechanical devices.",
        "children": [
            "TherapeuticAgent",
            "EndogenousSignal",
            "Toxicant",
        ],
        "clusters": [
            {
                "tag": "therapeutic_agent",
                "description": "engineered or purified compounds intended for treatment",
                "positive_examples": multiline_examples("""
[therapeutic_agent] Monoclonal antibodies bind tumor markers to recruit immune clearance.
[therapeutic_agent] RNAi payloads knock down overactive genes in liver cells.
[therapeutic_agent] CRISPR delivery lipids shuttle editing machinery into marrow stem cells.
[therapeutic_agent] Analgesic alkaloids modulate opioid receptors to relieve pain.
[therapeutic_agent] Cytokine cocktails expand T-cells before infusion into patients.
"""),
                "negative_examples": multiline_examples("""
[therapeutic_agent] Titanium plates stabilize fractures mechanically but cause no biochemical change.
[therapeutic_agent] MRI scanners image tissue yet administer no active substance.
[therapeutic_agent] Glass vials safely hold drugs but are inert packaging.
[therapeutic_agent] Placebo saline lacks active molecules and is used for control arms, not therapy.
[therapeutic_agent] Ventilators manipulate airflow but introduce no biochemical agent.
"""),
            },
            {
                "tag": "endogenous_signal",
                "description": "natural hormones, neurotransmitters, or signaling molecules",
                "positive_examples": multiline_examples("""
[endogenous_signal] Insulin secreted by pancreatic beta cells lowers blood glucose by signaling tissues to absorb it.
[endogenous_signal] Dopamine released in the basal ganglia modulates reward prediction.
[endogenous_signal] Cytokine storms illustrate how signaling molecules can dysregulate immunity.
[endogenous_signal] Ghrelin pulses communicate hunger states to the hypothalamus.
[endogenous_signal] Calcitonin helps regulate blood calcium via receptor binding in bone.
"""),
                "negative_examples": multiline_examples("""
[endogenous_signal] Blood plasma transports molecules but is not itself a discrete signal.
[endogenous_signal] Bone matrix provides structure rather than sending regulatory instructions.
[endogenous_signal] Dietary fiber passes through the gut largely unabsorbed, offering bulk, not signaling.
[endogenous_signal] Table sugar supplies energy but is not an endogenous regulatory molecule.
[endogenous_signal] Keratin comprises hair and nails without acting as a biochemical messenger.
"""),
            },
            {
                "tag": "toxicant",
                "description": "substances that disrupt biological function adversely",
                "positive_examples": multiline_examples("""
[toxicant] Organophosphate nerve agents block acetylcholinesterase, causing fatal buildup of neurotransmitters.
[toxicant] Aflatoxins from mold disrupt liver DNA repair leading to cancer.
[toxicant] Cyanide salts halt cellular respiration by binding cytochrome oxidase.
[toxicant] Microcystins from algal blooms damage hepatocytes even at microgram doses.
[toxicant] Endocrine disruptors mimic hormones and derail developmental pathways.
"""),
                "negative_examples": multiline_examples("""
[toxicant] Sand irritates mechanically but lacks biochemical binding.
[toxicant] Shrapnel injures by physical trauma rather than molecular interaction.
[toxicant] Ionizing radiation is energy, not a substance.
[toxicant] Wood splinters break skin but do not chemically modulate physiology.
[toxicant] Ultrasound waves transmit energy but are not chemical agents.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Region",
        "parent": "GeographicArea",
        "layer": 2,
        "domain": "Geography",
        "definition": "Spatially bounded area referenced for political, ecological, or analytical purposes.",
        "aliases": ["Area", "Zone"],
        "relationships": {
            "related": ["Location", "Territory", "Biome"],
            "has_part": ["Subregion", "Boundary"],
            "part_of": ["Continent", "Jurisdiction"],
        },
        "safety_tags": {
            "risk_level": "low",
            "impacts": ["mapping"],
            "treaty_relevant": True,
            "harness_relevant": False,
        },
        "disambiguation": "Treat regions as defined spatial areas rather than symbolic communities or data partitions.",
        "children": [
            "AdministrativeRegion",
            "EcologicalRegion",
            "AnalyticalRegion",
        ],
        "clusters": [
            {
                "tag": "administrative_region",
                "description": "regions defined by governance boundaries",
                "positive_examples": multiline_examples("""
[administrative_region] Counties levy taxes and maintain cadastral maps with surveyed boundaries.
[administrative_region] Federal districts such as DC exist through congressional statutes.
[administrative_region] Tribal reservations are defined through treaties and held in trust.
[administrative_region] Electoral wards map precincts for representation and resource allocation.
[administrative_region] Special economic zones are declared by law and managed as distinct jurisdictions.
"""),
                "negative_examples": multiline_examples("""
[administrative_region] Marketing "territories" describe sales coverage but lack legal standing.
[administrative_region] Delivery radiuses change whenever a restaurant opens or closes.
[administrative_region] Social media fandoms span continents without fixed borders.
[administrative_region] A utility's ad-hoc patrol grid simply splits workloads.
[administrative_region] Cultural scenes define taste communities, not statutory areas.
"""),
            },
            {
                "tag": "ecological_region",
                "description": "regions defined by ecological or climatic cohesion",
                "positive_examples": multiline_examples("""
[ecological_region] Permafrost zones are mapped by soil temperature thresholds across the Arctic.
[ecological_region] Watersheds are delineated by ridgelines that define where runoff flows.
[ecological_region] Mangrove belts consist of tree communities tolerant of brackish water along coasts.
[ecological_region] Mediterranean biomes share rainy winters and drought-prone summers with specific flora.
[ecological_region] Coral reef provinces are defined by contiguous reef structures and species mixes.
"""),
                "negative_examples": multiline_examples("""
[ecological_region] Hexagonal map bins help visualization but ignore ecological coherence.
[ecological_region] Shipping lanes are human navigation paths, not habitats.
[ecological_region] Trend charts depict statistics rather than physically bounded areas.
[ecological_region] Flight corridors respond to air traffic control, not ecosystem similarities.
[ecological_region] Policy frameworks describe governance, not the biome itself.
"""),
            },
            {
                "tag": "analytical_region",
                "description": "regions defined to support analysis or planning",
                "positive_examples": multiline_examples("""
[analytical_region] Census tracts divide cities into comparable population units for statistics.
[analytical_region] Mobility sheds map where commuters originate to target transit upgrades.
[analytical_region] Disaster response zones break a coastline into manageable command sectors.
[analytical_region] Service catchments define which households a clinic supports.
[analytical_region] Urban heat islands are polygons identified by thermal imagery to target tree planting.
"""),
                "negative_examples": multiline_examples("""
[analytical_region] A single sensor location is a point, not a region.
[analytical_region] Database shards partition data logically rather than by geography.
[analytical_region] An organizational chart outlines hierarchy, not spatial areas.
[analytical_region] Cultural identities can span continents and are not bounded zones.
[analytical_region] Individual households define addresses but not aggregated analytic polygons.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Mixture",
        "parent": "Substance",
        "layer": 3,
        "domain": "Chemistry",
        "definition": "Combination of two or more substances retaining their distinct chemical identities within a shared phase or matrix.",
        "aliases": ["Blend", "Composite"],
        "relationships": {
            "related": ["Solution", "CompositeMaterial", "Emulsion"],
            "has_part": ["Component", "Carrier"],
            "part_of": ["Formulation"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["materials"] ,
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Highlight physical mixtures distinct from pure compounds or abstract blends.",
        "children": [
            "SolutionMixture",
            "HeterogeneousMixture",
            "CompositeMixture",
        ],
        "clusters": [
            {
                "tag": "solution_mixture",
                "description": "homogeneous mixtures such as solutions and alloys",
                "positive_examples": multiline_examples("""
[solution_mixture] Saline IV bags standardize sodium chloride proportions in water.
[solution_mixture] Bronze alloy billets combine copper and tin uniformly for casting.
[solution_mixture] Solder paste suspends metal powder in flux to behave as one phase when printed.
[solution_mixture] Pharmaceutical syrups dissolve actives and sweeteners into a single liquid.
[solution_mixture] Ionic liquids mix cations and anions into stable room-temperature solutions.
"""),
                "negative_examples": multiline_examples("""
[solution_mixture] Pure distilled water contains only H₂O molecules, so no mixture exists.
[solution_mixture] A sealed ampoule with separate layers has not been mixed yet.
[solution_mixture] Binary code strings metaphorically mix instructions but contain no chemicals.
[solution_mixture] Genre playlists mix songs conceptually rather than physically.
[solution_mixture] Financial portfolios diversify assets but are not material mixtures.
"""),
            },
            {
                "tag": "heterogeneous_mixture",
                "description": "mixtures with distinct phases or particulates",
                "positive_examples": multiline_examples("""
[heterogeneous_mixture] Concrete contains cement paste, sand, and aggregate that remain visible.
[heterogeneous_mixture] Soil slurries suspend solids in water to transport dredged material.
[heterogeneous_mixture] Blood cells float in plasma as distinct phases.
[heterogeneous_mixture] Emulsified sauces mix oil droplets within water stabilized by lecithin.
[heterogeneous_mixture] Composite propellants embed oxidizer crystals inside polymer binders.
"""),
                "negative_examples": multiline_examples("""
[heterogeneous_mixture] A brick wall is an assembly of discrete units attached with mortar, not a blended mixture.
[heterogeneous_mixture] Data lakes combine files digitally, not physically.
[heterogeneous_mixture] Flavor pairing notes on paper describe possibilities rather than actual mixing.
[heterogeneous_mixture] Federated ledgers merge records logically, not in a material phase.
[heterogeneous_mixture] Laminated stacks bond layers but keep them separate, unlike mixtures.
"""),
            },
            {
                "tag": "composite_mixture",
                "description": "engineered mixtures with reinforcing structures",
                "positive_examples": multiline_examples("""
[composite_mixture] Carbon-fiber composites embed fabric in resin to balance stiffness and weight.
[composite_mixture] Ceramic matrix composites encase fibers to withstand turbine temperatures.
[composite_mixture] Rubber-metal antivibration mounts combine elastomer and steel for damping.
[composite_mixture] Gradient 3D-printed parts vary material ratios layer by layer.
[composite_mixture] Bioinks mix living cells with scaffolding gels for bioprinting.
"""),
                "negative_examples": multiline_examples("""
[composite_mixture] A pure silicon wafer is homogeneous and lacks reinforcing constituents.
[composite_mixture] CAD assemblies describe multi-part systems but not material mixtures.
[composite_mixture] Organizational matrices mix reporting lines metaphorically, not physically.
[composite_mixture] Artistic collages juxtapose images but do not mix materials uniformly for mechanical behavior.
[composite_mixture] Software plugins integrate code modules, not substances.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Communication",
        "parent": "Process",
        "layer": 2,
        "domain": "Communication",
        "definition": "Process of encoding, transmitting, and interpreting signals between senders and receivers.",
        "aliases": ["InformationExchange", "Messaging"],
        "relationships": {
            "related": ["Language", "Signal", "Conversation"],
            "has_part": ["Sender", "Channel", "Receiver"],
            "part_of": ["Interaction"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["coordination", "governance"],
            "treaty_relevant": True,
            "harness_relevant": True,
        },
        "simplex_mapping": {
            "status": "mapped",
            "mapped_simplex": "ConsentMonitor",
            "mapping_rationale": "Consent monitors watch communication flows that govern influence and autonomy.",
        },
        "disambiguation": "Distinguish communicative processes from static information or purely mechanical transfers.",
        "children": [
            "HumanCommunication",
            "MediatedCommunication",
            "MachineCommunication",
        ],
        "clusters": [
            {
                "tag": "human_communication",
                "description": "face-to-face or synchronous exchanges between humans",
                "positive_examples": multiline_examples("""
[human_communication] Mentors and apprentices hold retros where questions and responses adapt in real time.
[human_communication] Jury deliberations involve people interpreting evidence together before voting.
[human_communication] Incident commanders brief teams verbally to synchronize mental models under pressure.
[human_communication] Therapy sessions rely on conversational turns refined by emotions and context.
[human_communication] Daily standups surface blockers via spoken exchanges around a shared plan.
"""),
                "negative_examples": multiline_examples("""
[human_communication] Corporate bylaws remain static text and are not the live exchange.
[human_communication] Workflow tickets capture outcomes of conversations but are not the conversation itself.
[human_communication] Sensor dashboards broadcast metrics, not human back-and-forth.
[human_communication] Shared calendars coordinate time yet do not encode dialogue.
[human_communication] Policy tomes inform but do not constitute synchronous interaction.
"""),
            },
            {
                "tag": "mediated_communication",
                "description": "communication routed through media channels",
                "positive_examples": multiline_examples("""
[mediated_communication] A bilingual emergency alert is drafted, translated, and pushed via SMS gateways.
[mediated_communication] Podcasters script and record interviews distributed through RSS feeds.
[mediated_communication] Email campaigns segment recipients and schedule delivery windows.
[mediated_communication] Town halls livestream Q&A sessions, letting remote participants submit questions.
[mediated_communication] Asynchronous video updates from leadership embed captions for accessibility.
"""),
                "negative_examples": multiline_examples("""
[mediated_communication] Routing tables specify how packets travel but are not the message itself.
[mediated_communication] Spectrum licenses document rights but contain no content exchange.
[mediated_communication] Compression codecs define formats yet do not communicate on their own.
[mediated_communication] Fiber network maps show infrastructure rather than actual conversations.
[mediated_communication] Analytics dashboards analyze messaging post-hoc, not the communicative acts themselves.
"""),
            },
            {
                "tag": "machine_communication",
                "description": "machine-to-machine signaling with semantics",
                "positive_examples": multiline_examples("""
[machine_communication] Vehicle-to-vehicle beacons broadcast speed and heading so nearby cars can adjust.
[machine_communication] IoT gateways acknowledge sensor commands to confirm actuation succeeded.
[machine_communication] Digital twins sync state deltas to maintain a shared situational picture.
[machine_communication] CAT oversight telemetry packages lens activations sent between monitoring agents.
[machine_communication] Swarm drones share mesh updates to avoid collisions.
"""),
                "negative_examples": multiline_examples("""
[machine_communication] Analog noise on a wire introduces random voltage but does not encode meaning.
[machine_communication] Register toggles within a single chip are local control, not inter-agent messaging.
[machine_communication] Firmware images stored on disk are static artifacts, not active exchanges.
[machine_communication] Clock drift is a timing error, not a message.
[machine_communication] Power bus fluctuations indicate load but do not carry structured semantics.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Text",
        "parent": "Communication",
        "layer": 3,
        "domain": "Information",
        "definition": "Written or typed sequence of symbols structured to convey information.",
        "aliases": ["WrittenContent", "DocumentText"],
        "relationships": {
            "related": ["Document", "Sentence", "Corpus"],
            "has_part": ["Paragraph", "Sentence", "Token"],
            "part_of": ["Publication"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["information"] ,
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Focus on linguistic symbol sequences, not speech audio or graphical imagery.",
        "children": [
            "NarrativeText",
            "InstructionalText",
            "DataText",
        ],
        "clusters": [
            {
                "tag": "narrative_text",
                "description": "stories, articles, or reports with narrative flow",
                "positive_examples": multiline_examples("""
[narrative_text] Investigative features weave interviews and timelines into written chapters.
[narrative_text] Mission logs recount what each rover sol accomplished and why.
[narrative_text] Oral history transcripts capture speakers' words sequentially for future readers.
[narrative_text] Graphic novel scripts include descriptive blocks plus dialogue to tell stories.
[narrative_text] After-action reports narrate incidents before enumerating lessons.
"""),
                "negative_examples": multiline_examples("""
[narrative_text] Storyboard sketches rely on frames and arrows rather than textual paragraphs.
[narrative_text] Radio broadcasts are audio; once transcribed they become text, but the waveform itself is not.
[narrative_text] CSV exports store fields but no narrative arc or sentences.
[narrative_text] Photo essays may include captions, yet the imagery carries most meaning.
[narrative_text] JSON payloads encode data structures rather than flowing prose.
"""),
            },
            {
                "tag": "instructional_text",
                "description": "procedural or directive text",
                "positive_examples": multiline_examples("""
[instructional_text] Runbooks list numbered steps for triaging an outage.
[instructional_text] API docs describe required parameters and response codes in sentences.
[instructional_text] Safety checklists print directive verbs like "Verify fuel valve closed." 
[instructional_text] Curriculum outlines break lessons into objectives and activities using text.
[instructional_text] Installation guides specify prerequisites, commands, and verification steps.
"""),
                "negative_examples": multiline_examples("""
[instructional_text] An exploded-view diagram relies on visuals more than words.
[instructional_text] VR demos teach motion but are immersive media, not text.
[instructional_text] Hands-on mentorship occurs through demonstration, not textual description.
[instructional_text] Whiteboard recordings capture speech and drawing rather than typed instructions.
[instructional_text] Schematics depict wiring connections graphically, not procedurally in text.
"""),
            },
            {
                "tag": "data_text",
                "description": "textual data such as logs, transcripts, and annotations",
                "positive_examples": multiline_examples("""
[data_text] Customer support tickets store typed descriptions and agent replies for analysis.
[data_text] Chat transcripts capture messages line by line for QA review.
[data_text] JSONL corpora log prompt-response pairs for model fine-tuning.
[data_text] Sensor alerts include textual payloads describing thresholds and tags.
[data_text] Annotation guidelines explain labeling decisions using paragraphs.
"""),
                "negative_examples": multiline_examples("""
[data_text] A raw audio waveform lacks textual tokens until transcribed.
[data_text] Emoji-only chats omit lexical sequences and therefore fall outside textual data.
[data_text] Point clouds represent geometry, not words.
[data_text] Aggregate dashboards visualize metrics derived from text but are not text themselves.
[data_text] Vector embeddings encode semantics numerically, not as written language.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "IntentionalPsychologicalProcess",
        "parent": "Process",
        "layer": 3,
        "domain": "MindsAndAgents",
        "definition": "Deliberate mental activity directed toward goals, regulation, or self-reflection.",
        "aliases": ["IntentionalCognition", "VolitionalProcess"],
        "relationships": {
            "related": ["Planning", "Motivation", "SelfRegulation"],
            "has_part": ["GoalSetting", "Evaluation", "Adjustment"],
            "part_of": ["CognitiveProcess"],
        },
        "safety_tags": {
            "risk_level": "high",
            "impacts": ["autonomy", "motivation"],
            "treaty_relevant": True,
            "harness_relevant": True,
        },
        "simplex_mapping": {
            "status": "mapped",
            "mapped_simplex": "MotivationalRegulation",
            "mapping_rationale": "Intentional processes interact directly with motivational simplexes.",
        },
        "disambiguation": "Differentiate deliberate mental acts from reflexes or environmental dynamics.",
        "children": [
            "GoalFormulation",
            "SelfMonitoring",
            "DeliberativeAdjustment",
        ],
        "clusters": [
            {
                "tag": "goal_formulation",
                "description": "forming, prioritizing, and committing to goals",
                "positive_examples": multiline_examples("""
[goal_formulation] An advocacy team negotiates which demands are feasible this quarter before campaigning.
[goal_formulation] Engineers translate user values into acceptance criteria before implementation.
[goal_formulation] A recovering athlete works with clinicians to set weekly mobility milestones.
[goal_formulation] Treaty liaisons define measurable success indicators before ratification.
[goal_formulation] Individuals write OKRs to clarify what outcomes they will pursue.
"""),
                "negative_examples": multiline_examples("""
[goal_formulation] A startle response occurs before the mind can choose a goal.
[goal_formulation] Habitual doomscrolling reflects compulsion, not reflective commitment.
[goal_formulation] An imposed schedule handed down with no deliberation is not self-formulated.
[goal_formulation] Reflex blinking keeps eyes moist without conscious planning.
[goal_formulation] Crowd stampedes illustrate herd dynamics rather than intentional goal setting.
"""),
            },
            {
                "tag": "self_monitoring",
                "description": "intentional awareness of one's cognitive-emotional state",
                "positive_examples": multiline_examples("""
[self_monitoring] Consent guardians pause a dialogue to check whether they still freely agree to continue.
[self_monitoring] Journaling about triggers helps someone see patterns in their reactions.
[self_monitoring] Meditation practitioners note attention drift and gently return to the breath.
[self_monitoring] Biofeedback sessions teach users to notice when heart rates spike from anxiety.
[self_monitoring] Researchers document when their own biases might color an analysis.
"""),
                "negative_examples": multiline_examples("""
[self_monitoring] Wearable alerts beep automatically even if the user ignores them.
[self_monitoring] Rumination loops fixate without intentional witnessing; they can consume attention but are not reflective.
[self_monitoring] A panic spiral overwhelms before reflection can occur.
[self_monitoring] External audits performed by peers are observation, not self-monitoring.
[self_monitoring] Algorithmic monitoring of compliance is surveillance, not introspection.
"""),
            },
            {
                "tag": "deliberative_adjustment",
                "description": "intentional re-planning or regulation based on goals and monitoring",
                "positive_examples": multiline_examples("""
[deliberative_adjustment] A negotiator pauses to reframe offers after sensing the other party's fatigue.
[deliberative_adjustment] Teams halt a rollout when early telemetry conflicts with safety goals.
[deliberative_adjustment] A person switches coping strategies from doomscrolling to calling a friend after reflection.
[deliberative_adjustment] Clinicians adjust therapy homework once they see which exercises caused pain.
[deliberative_adjustment] Leaders rehearse alternative responses before responding to a provocative email.
"""),
                "negative_examples": multiline_examples("""
[deliberative_adjustment] Muscle twitches happen without conscious review.
[deliberative_adjustment] Thermostat adjustments run through mechanical control loops absent subjective intent.
[deliberative_adjustment] A random walk through parameter space lacks evaluation or goals.
[deliberative_adjustment] Hiccups interrupt speech but involve no deliberation.
[deliberative_adjustment] Server autoscaling policies apply automatically rather than through reflection in the moment.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "FoodIngredient",
        "parent": "Food",
        "layer": 3,
        "domain": "Food",
        "definition": "Component materials intentionally combined to create food products.",
        "aliases": ["Ingredient", "FoodComponent"],
        "relationships": {
            "related": ["Food", "Additive", "Flavor"],
            "has_part": [],
            "part_of": ["Recipe", "Formulation"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["food_safety"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Differentiate edible inputs from packaging, tools, or metaphoric 'ingredients'.",
        "children": [
            "WholeIngredient",
            "ProcessedIngredient",
            "FunctionalAdditive",
        ],
        "clusters": [
            {
                "tag": "whole_ingredient",
                "description": "single-source plant or animal ingredients",
                "positive_examples": multiline_examples("""
[whole_ingredient] Blood oranges bring both juice and zest into sauces without intermediate processing.
[whole_ingredient] Fresh basil leaves are torn into salads and pesto directly.
[whole_ingredient] Pasture-raised chicken thighs arrive chilled and go straight to the grill.
[whole_ingredient] Shiitake mushroom caps are sliced into stir-fries with minimal trimming.
[whole_ingredient] Anchovy fillets fold into bagna cauda sauces as salty whole ingredients.
"""),
                "negative_examples": multiline_examples("""
[whole_ingredient] Vacuum seal bags contact food but remain packaging, not ingredients.
[whole_ingredient] Spice grinders grind ingredients yet are not consumed.
[whole_ingredient] Menu QR codes tell diners about dishes rather than entering the recipe.
[whole_ingredient] Compostable plates accompany meals but are not eaten.
[whole_ingredient] Branding pitch decks talk about food identity, not the edible matter.
"""),
            },
            {
                "tag": "processed_ingredient",
                "description": "ingredients transformed for consistency or preservation",
                "positive_examples": multiline_examples("""
[processed_ingredient] Tomato paste concentrates flavor for sauces without perishing quickly.
[processed_ingredient] Freeze-dried coffee dissolves instantly for beverages.
[processed_ingredient] Rendered duck fat stores in jars ready to sear potatoes.
[processed_ingredient] Malt extract syrups feed yeast in brewing.
[processed_ingredient] Powdered egg whites whip into meringues when fresh eggs are impractical.
"""),
                "negative_examples": multiline_examples("""
[processed_ingredient] Degreasers clean fryers but must never enter food.
[processed_ingredient] Nutrition labels document content yet are not edible material.
[processed_ingredient] Chef interviews talk about ingredients but remain media.
[processed_ingredient] Regulatory filings describe compliance rather than contributing to flavor.
[processed_ingredient] Cooking playlists set ambiance, not taste.
"""),
            },
            {
                "tag": "functional_additive",
                "description": "ingredients added for texture, preservation, or nutritional function",
                "positive_examples": multiline_examples("""
[functional_additive] Xanthan gum thickens sauces without altering flavor dramatically.
[functional_additive] Nitrites cure meats to inhibit botulism and set color.
[functional_additive] Probiotic cultures are added to yogurts to deliver live bacteria.
[functional_additive] Ascorbic acid keeps cut apples from browning.
[functional_additive] Electrolyte salt blends fortify sports drinks.
"""),
                "negative_examples": multiline_examples("""
[functional_additive] A production schedule lists when to cook but is not added to food.
[functional_additive] Payment terms exist on invoices, not in the recipe.
[functional_additive] Brand mascots on packaging tell stories but are not edible.
[functional_additive] Line staffing charts show labor allocation, not formulation components.
[functional_additive] Retail fixtures display goods yet provide no functional chemistry.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Work",
        "parent": "Process",
        "layer": 2,
        "domain": "Process",
        "definition": "Coordinated effort applied to produce value, solve problems, or meet obligations.",
        "aliases": ["Labor", "Taskwork"],
        "relationships": {
            "related": ["Task", "Job", "Project"],
            "has_part": ["Activity", "Output"],
            "part_of": ["Workflow"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["economy", "labor"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Distinguish purposeful labor from idle time, entertainment, or automated execution with no human intention.",
        "children": [
            "PhysicalWork",
            "CognitiveWork",
            "EmotionalLabor",
        ],
        "clusters": [
            {
                "tag": "physical_work",
                "description": "labor emphasizing bodily exertion",
                "positive_examples": multiline_examples("""
[physical_work] Harvest crews pick berries in crouched positions for hours to fill cold-chain pallets.
[physical_work] Line cooks chop crates of produce before service hits.
[physical_work] Utility technicians climb poles to re-string power lines after a storm.
[physical_work] Warehouse pickers walk miles daily to assemble customer orders.
[physical_work] Emergency responders carry evacuees down stairwells during a fire.
"""),
                "negative_examples": multiline_examples("""
[physical_work] Meditation is purposeful but counts as rest rather than labor output.
[physical_work] Casual strolling for pleasure delivers no contracted deliverable.
[physical_work] Robot arms lifting payloads do not constitute human work.
[physical_work] Gym workouts done for personal health fall under recreation.
[physical_work] Gravity-fed chutes move goods without human exertion once loaded.
"""),
            },
            {
                "tag": "cognitive_work",
                "description": "knowledge or decision work",
                "positive_examples": multiline_examples("""
[cognitive_work] Software architects vet competing designs before writing RFCs.
[cognitive_work] Policy teams draft legislative text backed by analysis.
[cognitive_work] Researchers plan experiments, record data, and publish findings.
[cognitive_work] Incident commanders run tabletop triage to decide containment steps.
[cognitive_work] Curriculum designers align learning activities with standards.
"""),
                "negative_examples": multiline_examples("""
[cognitive_work] Doomscrolling burns attention but generates no deliverable.
[cognitive_work] Trivia nights challenge recall purely for entertainment.
[cognitive_work] Idle speculation with no plan or accountability is not work.
[cognitive_work] Venting at the water cooler relieves stress but does not satisfy a task contract.
[cognitive_work] Daydreaming may spark creativity later yet is not itself accountable labor.
"""),
            },
            {
                "tag": "emotional_labor",
                "description": "work that manages emotions for service or care",
                "positive_examples": multiline_examples("""
[emotional_labor] Customer support agents calmly de-escalate angry callers while solving issues.
[emotional_labor] Hospital staff comfort grieving families as part of their role.
[emotional_labor] Treaty liaisons interpret tone and choose language that preserves trust between tribes.
[emotional_labor] Community moderators intervene compassionately when threads become heated.
[emotional_labor] Mentors schedule check-ins to hold space for mentees' frustrations.
"""),
                "negative_examples": multiline_examples("""
[emotional_labor] Private journaling supports the writer but is not work provided to others.
[emotional_labor] Automated sentiment bots mimic empathy yet no human regulates feelings.
[emotional_labor] Family dinners involve emotional exchange but fall outside contracted labor.
[emotional_labor] A couple hugging in public expresses affection, not employment.
[emotional_labor] IVR menus express canned phrases, not true emotional regulation.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Electronics",
        "parent": "Artifact",
        "layer": 3,
        "domain": "CreatedThings",
        "definition": "Systems and components that manipulate electrical signals for sensing, computation, or control.",
        "aliases": ["ElectronicSystem", "Circuitry"],
        "relationships": {
            "related": ["Device", "Semiconductor", "SignalProcessing"],
            "has_part": ["Circuit", "Component", "Enclosure"],
            "part_of": ["Product"] ,
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["hardware"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Differentiate engineered electronic assemblies from passive materials or purely software artifacts.",
        "children": [
            "AnalogElectronics",
            "DigitalElectronics",
            "PowerElectronics",
        ],
        "clusters": [
            {
                "tag": "analog_electronics",
                "description": "continuous-signal circuits",
                "positive_examples": multiline_examples("""
[analog_electronics] Instrumentation amplifiers condition microvolt signals from strain gauges.
[analog_electronics] RF downconverters mix carriers to intermediate frequencies using analog mixers.
[analog_electronics] Audio consoles process musician inputs via continuous potentiometers and op-amps.
[analog_electronics] Medical bio-amps amplify ECG waves before digitization.
[analog_electronics] Analog filters shape sensor noise bands prior to ADC sampling.
"""),
                "negative_examples": multiline_examples("""
[analog_electronics] Fiber-optic cables pass light without active signal shaping.
[analog_electronics] HTML templates describe UI but contain no circuits.
[analog_electronics] Heat sinks dissipate energy but lack electrical paths.
[analog_electronics] Mechanical relays switch states but provide minimal analog processing themselves.
[analog_electronics] Lightning bolts produce voltage but are uncontrolled natural events.
"""),
            },
            {
                "tag": "digital_electronics",
                "description": "discrete logic and embedded computation",
                "positive_examples": multiline_examples("""
[digital_electronics] Microcontrollers integrate CPUs, flash, and peripherals to run deterministic firmware.
[digital_electronics] FPGA boards host reconfigurable logic and transceivers for prototyping.
[digital_electronics] Edge AI accelerators execute neural ops near sensors.
[digital_electronics] Secure elements protect cryptographic keys through tamper-resistant logic.
[digital_electronics] Industrial PLC backplanes house CPU and IO cards executing ladder logic.
"""),
                "negative_examples": multiline_examples("""
[digital_electronics] Firmware source code is software until flashed onto hardware.
[digital_electronics] Paper sketches show intent but hold no transistors.
[digital_electronics] Mounting brackets support boards mechanically, not electronically.
[digital_electronics] Shipping foam protects devices but is not circuitry.
[digital_electronics] CI pipelines compile binaries but exist in the software layer.
"""),
            },
            {
                "tag": "power_electronics",
                "description": "electronics that convert and manage power",
                "positive_examples": multiline_examples("""
[power_electronics] DC-DC converters step voltage rails down for avionics racks.
[power_electronics] Solar inverters convert panel DC into synchronized AC for grids.
[power_electronics] Battery BMS boards balance cells and enforce safe charge envelopes.
[power_electronics] EV charging modules rectify AC and coordinate with vehicles over PLC.
[power_electronics] UPS controllers switch loads to batteries within milliseconds during outages.
"""),
                "negative_examples": multiline_examples("""
[power_electronics] Copper busbars distribute current but contain no control logic.
[power_electronics] Diesel generators produce electricity using mechanical engines, not electronic regulation.
[power_electronics] Pricing tariffs describe costs, not devices.
[power_electronics] Regulatory filings document compliance rather than manipulating power.
[power_electronics] Hydraulic pumps move fluid, not electrons.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "OrganismProcess",
        "parent": "BiologicalProcess",
        "layer": 2,
        "domain": "Biology",
        "definition": "Physiological or behavioral process carried out by living organisms.",
        "aliases": ["BiologicalProcess", "LifeProcess"],
        "relationships": {
            "related": ["Metabolism", "Development", "Behavior"],
            "has_part": ["Stage", "Trigger"],
            "part_of": ["LifeCycle"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["biosafety"],
            "treaty_relevant": True,
            "harness_relevant": False,
        },
        "disambiguation": "Differentiate biological processes from engineered workflows or abiotic processes.",
        "children": [
            "MetabolicProcess",
            "DevelopmentalProcess",
            "BehavioralProcess",
        ],
        "clusters": [
            {
                "tag": "metabolic_process",
                "description": "chemical reactions sustaining life",
                "positive_examples": multiline_examples("""
[metabolic_process] Cellular respiration converts glucose and oxygen into ATP and CO₂.
[metabolic_process] Gluconeogenesis synthesizes glucose during fasting.
[metabolic_process] Nitrogen fixation in legumes turns atmospheric nitrogen into ammonia.
[metabolic_process] Liver detoxification enzymes break down xenobiotics.
[metabolic_process] Immune signaling cascades release interferons in response to pathogens.
"""),
                "negative_examples": multiline_examples("""
[metabolic_process] Combustion engines burn fuel but are mechanical, not living processes.
[metabolic_process] Supply chains move goods, not molecules inside organisms.
[metabolic_process] Erosion shapes rock through weather, not biological reactions.
[metabolic_process] Data pipelines transform bytes, not metabolites.
[metabolic_process] Battery discharge involves electrochemistry but not living cells.
"""),
            },
            {
                "tag": "developmental_process",
                "description": "growth and differentiation over an organism's lifespan",
                "positive_examples": multiline_examples("""
[developmental_process] Embryogenesis orchestrates differentiation from zygote to fetus.
[developmental_process] Metamorphosis transforms caterpillars into butterflies through staged hormonal cues.
[developmental_process] Puberty reconfigures endocrine axes and anatomy as humans mature.
[developmental_process] Bone remodeling continually replaces tissue to adjust to stress.
[developmental_process] Flowering initiates reproductive structures in plants at maturity.
"""),
                "negative_examples": multiline_examples("""
[developmental_process] 3D printing builds objects but they are not living organisms.
[developmental_process] Software versioning uses biological metaphors yet occurs in code.
[developmental_process] Crystal growth in caves forms from mineral deposition, not life.
[developmental_process] Polymer curing changes materials but lacks cellular regulation.
[developmental_process] Product roadmaps borrow lifecycle language without biology.
"""),
            },
            {
                "tag": "behavioral_process",
                "description": "observable actions driven by organismal cognition or instinct",
                "positive_examples": multiline_examples("""
[behavioral_process] Migratory birds follow seasonal cues to travel thousands of kilometers.
[behavioral_process] Ant colonies reallocate workers based on pheromone trails.
[behavioral_process] Pollinators forage specific flowers using learned routes.
[behavioral_process] Humans signal consent with both verbal and nonverbal cues.
[behavioral_process] Octopuses camouflage themselves in response to predators.
"""),
                "negative_examples": multiline_examples("""
[behavioral_process] Traffic lights follow timers or sensors but have no organismal agency.
[behavioral_process] Falling rocks respond to gravity, not behavior.
[behavioral_process] Algorithmic trading loops run code, not biological instinct.
[behavioral_process] Puppets move by external strings, lacking self-driven action.
[behavioral_process] Wind gusts swirl due to weather physics rather than purposeful behavior.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Transfer",
        "parent": "Process",
        "layer": 2,
        "domain": "Process",
        "definition": "Process of moving something from one context, owner, or location to another.",
        "aliases": ["Transference", "HandOff"],
        "relationships": {
            "related": ["Transport", "Exchange", "Migration"],
            "has_part": ["Source", "Destination", "Medium"],
            "part_of": ["Logistics", "Workflow"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["supply_chain", "data_governance"],
            "treaty_relevant": True,
            "harness_relevant": False,
        },
        "disambiguation": "Distinguish actual relocation or handoff from mere intention or transformation without movement.",
        "children": [
            "MaterialTransfer",
            "InformationTransfer",
            "OwnershipTransfer",
        ],
        "clusters": [
            {
                "tag": "material_transfer",
                "description": "physical movement of matter",
                "positive_examples": multiline_examples("""
[material_transfer] Chain-of-custody paperwork accompanies soil samples moving from field to lab.
[material_transfer] Vaccines travel in validated shippers from manufacturer to clinic.
[material_transfer] Couriers relay donor blood between hospitals under temperature control.
[material_transfer] Equipment handoffs document when a crane leaves one contractor's custody for another.
[material_transfer] Drone parcel drops deliver medication to remote islands, logging pickup and drop coordinates.
"""),
                "negative_examples": multiline_examples("""
[material_transfer] Printing labels prepares for a shipment but does not move goods.
[material_transfer] Cooking transforms ingredients without relocating them to a new owner.
[material_transfer] Insurance policies cover goods yet do not physically relocate them.
[material_transfer] 3D printing creates parts but keeps material at the same site.
[material_transfer] Customs declarations declare contents but the paperwork itself is not the transfer.
"""),
            },
            {
                "tag": "information_transfer",
                "description": "communicative transfer of data or knowledge",
                "positive_examples": multiline_examples("""
[information_transfer] Encrypted API calls send telemetry from field sensors to control rooms.
[information_transfer] Shift briefings hand off incident context between on-call engineers.
[information_transfer] Knowledge-base updates push new troubleshooting steps to all technicians.
[information_transfer] Treaty telemetry sharing feeds partner dashboards with compliance data.
[information_transfer] Secure file drops exchange design packages between companies.
"""),
                "negative_examples": multiline_examples("""
[information_transfer] Draft thoughts in a personal notebook are not yet transmitted.
[information_transfer] Data sitting in cold archives is static storage, not transfer.
[information_transfer] Unauthorized leaks may move information but violate consent, failing governance criteria here.
[information_transfer] Doodles in a margin remain private unless communicated.
[information_transfer] Local caches hold data close to compute with no recipient.
"""),
            },
            {
                "tag": "ownership_transfer",
                "description": "change in rights, custody, or accountability",
                "positive_examples": multiline_examples("""
[ownership_transfer] Title deeds recorded at the county mark land moving between owners.
[ownership_transfer] SaaS contract novations reassign service obligations when a business is acquired.
[ownership_transfer] Equipment loan agreements spell out when custody passes to a subcontractor.
[ownership_transfer] Tokenized asset settlement on a blockchain updates who controls a share.
[ownership_transfer] Consent receipts update which system has permission to process data.
"""),
                "negative_examples": multiline_examples("""
[ownership_transfer] Lease renewals extend an existing agreement but keep ownership constant.
[ownership_transfer] Usage monitoring ensures compliance without changing who owns the asset.
[ownership_transfer] Rumors about potential sales circulate without legal effect.
[ownership_transfer] Press releases may announce intent but do not execute transfer paperwork.
[ownership_transfer] Spot checks verify custody rather than shifting it.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "BodySubstance",
        "parent": "Substance",
        "layer": 3,
        "domain": "Biology",
        "definition": "Material produced by or contained within living bodies such as fluids, tissues, or secretions.",
        "aliases": ["BiologicalSubstance", "BodyFluid"],
        "relationships": {
            "related": ["BiologicallyActiveSubstance", "Organ", "Tissue"],
            "has_part": ["Cell", "Biomolecule"],
            "part_of": ["Organism"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["biosafety", "health"],
            "treaty_relevant": True,
            "harness_relevant": False,
        },
        "disambiguation": "Distinguish bodily materials from exogenous pharmaceuticals or mechanical prosthetics.",
        "children": [
            "BodyFluid",
            "BodyTissue",
            "BodyWaste",
        ],
        "clusters": [
            {
                "tag": "body_fluid",
                "description": "fluids circulating or secreted by organisms",
                "positive_examples": multiline_examples("""
[body_fluid] Blood carries oxygen and immune cells through vessels.
[body_fluid] Cerebrospinal fluid cushions and nourishes the brain.
[body_fluid] Lymph transports lipids and immune cells through the lymphatic system.
[body_fluid] Breast milk delivers nutrients and antibodies to infants.
[body_fluid] Synovial fluid lubricates joints.
"""),
                "negative_examples": multiline_examples("""
[body_fluid] IV saline is produced industrially and only temporarily resides in the body.
[body_fluid] Contrast dye is a diagnostic chemical, not a substance generated by the organism.
[body_fluid] Hydraulic oil lubricates machines, not living tissues.
[body_fluid] Air fresheners are external chemicals with no physiological origin.
[body_fluid] Cleaning solutions contact skin but are not body substances.
"""),
            },
            {
                "tag": "body_tissue",
                "description": "solid biological tissues",
                "positive_examples": multiline_examples("""
[body_tissue] Surgeons graft skin harvested from a patient's thigh onto a burn site.
[body_tissue] Pathologists examine tumor biopsies under microscopes.
[body_tissue] Cartilage harvested during joint surgery is cataloged as body tissue.
[body_tissue] Placental tissue is collected post-delivery for pathology.
[body_tissue] Bone marrow aspirates contain living tissue extracted for transplant.
"""),
                "negative_examples": multiline_examples("""
[body_tissue] Carbon fiber prosthetics integrate with bodies but are synthetic.
[body_tissue] Silicone implants occupy space yet are man-made materials.
[body_tissue] Pacemakers regulate hearts but consist of electronics, not tissue.
[body_tissue] Lab-grown meat destined for food is outside the context of a living organism.
[body_tissue] Bioplastic cups derive from plants but are processed goods.
"""),
            },
            {
                "tag": "body_waste",
                "description": "excreted or shed materials",
                "positive_examples": multiline_examples("""
[body_waste] Hospitals treat urine samples and diapers as regulated waste.
[body_waste] Shed skin cells accumulate on bedding even after people leave.
[body_waste] Sweat collected from hazmat suits is handled as biological material.
[body_waste] Placental remains and lochia following childbirth require proper disposal.
[body_waste] Scabs fall away as wounds heal and count as body waste.
"""),
                "negative_examples": multiline_examples("""
[body_waste] Household trash includes packaging unrelated to bodily processes.
[body_waste] Industrial effluent comes from factories, not organisms.
[body_waste] Paper towels used for cleaning start as manufactured goods.
[body_waste] Noise emissions are energy, not matter.
[body_waste] Plastic packaging remains synthetic even when discarded by people.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "PreparedFood",
        "parent": "Food",
        "layer": 3,
        "domain": "Food",
        "definition": "Food that has been combined, cooked, or otherwise readied for immediate consumption.",
        "aliases": ["ReadyMeal", "PlatedFood"],
        "relationships": {
            "related": ["Food", "Meal", "Recipe"],
            "has_part": ["Portion", "Garnish"],
            "part_of": ["Menu", "Service"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["food_safety"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Differentiate ready-to-eat items from ingredient kits or packaged raw goods.",
        "children": [
            "RestaurantDish",
            "PackagedMeal",
            "HouseholdMeal",
        ],
        "clusters": [
            {
                "tag": "restaurant_dish",
                "description": "prepared food served in hospitality contexts",
                "positive_examples": multiline_examples("""
[restaurant_dish] A ramen bowl arrives with broth, noodles, and toppings ready to be eaten immediately.
[restaurant_dish] Tasting menu chefs prepare amuse-bouches plated for one bite.
[restaurant_dish] Hospital patient trays include complete dishes portioned per dietician orders.
[restaurant_dish] Airline meals are assembled, chilled, and reheated for passengers mid-flight.
[restaurant_dish] Cafeteria servings present entrees and sides on plates ready for students.
"""),
                "negative_examples": multiline_examples("""
[restaurant_dish] CSA boxes deliver ingredients requiring prep.
[restaurant_dish] QR menus describe dishes but contain no food.
[restaurant_dish] Plastic display models show appearance yet are inedible.
[restaurant_dish] Spice rub kits supply seasoning, not finished dishes.
[restaurant_dish] Nutrition posters educate but do not provide food.
"""),
            },
            {
                "tag": "packaged_meal",
                "description": "ready-to-heat or shelf-stable meals",
                "positive_examples": multiline_examples("""
[packaged_meal] Frozen lasagna trays require only heating before serving.
[packaged_meal] Retort pouches of curry stay shelf-stable yet are fully cooked.
[packaged_meal] Microwaveable grain bowls ship with sauce and vegetables ready for reheating.
[packaged_meal] MRE entrees include flameless heaters and are eaten in the field.
[packaged_meal] Airline snack boxes bundle curated items sealed for immediate consumption.
"""),
                "negative_examples": multiline_examples("""
[packaged_meal] Bulk dry beans still need soaking and cooking.
[packaged_meal] DIY baking mixes require eggs, butter, and ovens to become food.
[packaged_meal] Shelf talkers promote items but contain no calories.
[packaged_meal] Loyalty emails describe promotions, not meals.
[packaged_meal] Food review blogs critique dishes without feeding anyone.
"""),
            },
            {
                "tag": "household_meal",
                "description": "home-cooked or community meals",
                "positive_examples": multiline_examples("""
[household_meal] Families gather for Sunday pot roast plated straight from the oven.
[household_meal] Meal trains deliver ready casseroles to new parents' homes.
[household_meal] Community soup nights simmer pots and serve neighbors immediately.
[household_meal] Picnic spreads pack sandwiches, fruit, and drinks consumed outdoors.
[household_meal] Holiday tamales are assembled, steamed, and shared across extended families.
"""),
                "negative_examples": multiline_examples("""
[household_meal] Grocery lists plan purchases but are not food themselves.
[household_meal] Budget spreadsheets track costs, not calories.
[household_meal] Meal planning apps schedule menus yet produce nothing edible.
[household_meal] Dining decor sets ambiance but is not the meal.
[household_meal] Fasting rituals involve abstaining, not preparing food.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "GroupOfPeople",
        "parent": "Group",
        "layer": 3,
        "domain": "SocialSystems",
        "definition": "Collection of humans connected by membership, identity, or co-present activity.",
        "aliases": ["PeopleGroup", "HumanCollective"],
        "relationships": {
            "related": ["Demographic", "Community", "Audience"],
            "has_part": ["Person"],
            "part_of": ["Population"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["sociology", "equity"],
            "treaty_relevant": True,
            "harness_relevant": False,
        },
        "disambiguation": "Focus on actual people, not avatars, bots, or conceptual categories lacking human members.",
        "children": [
            "CommunityOfPeople",
            "AudienceGroup",
            "DemographicGroup",
        ],
        "clusters": [
            {
                "tag": "community_of_people",
                "description": "self-identifying communities",
                "positive_examples": multiline_examples("""
[community_of_people] Neighborhood associations gather residents to negotiate shared issues.
[community_of_people] Diaspora networks organize cultural events and mutual aid.
[community_of_people] Union locals hold meetings to coordinate member actions.
[community_of_people] Parenting support circles meet weekly to exchange advice.
[community_of_people] Maker collectives share tools and critique each other's builds.
"""),
                "negative_examples": multiline_examples("""
[community_of_people] Bot swarms inflate social metrics without being human participants.
[community_of_people] A/B test cohorts include accounts algorithmically but may not correspond to actual communities.
[community_of_people] Marketing personas describe archetypes, not specific people gathered together.
[community_of_people] Synthetic training data contains fictional entries.
[community_of_people] NPC armies in games simulate players but are scripted.
"""),
            },
            {
                "tag": "audience_group",
                "description": "people assembled to receive information or services",
                "positive_examples": multiline_examples("""
[audience_group] Conference attendees sit through keynotes together and respond to Q&A.
[audience_group] Jury pools wait in a room to hear instructions from the court.
[audience_group] Focus groups sign NDAs and discuss prototype concepts with facilitators.
[audience_group] Live-stream chat participants respond to a presenter's prompts in real time.
[audience_group] Clinic waiting rooms hold patients queued for the same provider.
"""),
                "negative_examples": multiline_examples("""
[audience_group] Podcast download counts include asynchronous listeners dispersed globally.
[audience_group] Advertising impressions tally exposures even for bots.
[audience_group] Mailing list subscribers may never read messages simultaneously.
[audience_group] Predicted reach estimates audiences that may not materialize.
[audience_group] Cookie pools combine identifiers without verifying human attention in a moment.
"""),
            },
            {
                "tag": "demographic_group",
                "description": "groups defined by demographic criteria",
                "positive_examples": multiline_examples("""
[demographic_group] "First-generation college students" describes a trackable cohort for support programs.
[demographic_group] Floodplain residents share location-based risk criteria.
[demographic_group] Disabled veterans qualify for specific services and can be counted.
[demographic_group] Women over 60 make up a demographic tracked in health studies.
[demographic_group] Gig workers with dependents are surveyed to inform policy.
"""),
                "negative_examples": multiline_examples("""
[demographic_group] Employee ID ranges are arbitrary sequences lacking real-world traits.
[demographic_group] Horoscope signs assign membership by birth date but are not demographics used for equity work.
[demographic_group] Gaming factions assign characters in fiction, not census populations.
[demographic_group] Toy dataset splits mix simulated entries, not real humans.
[demographic_group] Demo accounts used for QA do not correspond to actual people.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Animal",
        "parent": "Organism",
        "layer": 2,
        "domain": "Biology",
        "definition": "Multicellular organism within kingdom Animalia capable of movement, sensation, and responsive behavior.",
        "aliases": ["AnimalOrganism", "Fauna"],
        "relationships": {
            "related": ["Organism", "Species", "Vertebrate"],
            "has_part": ["Organ", "Tissue"],
            "part_of": ["Ecosystem"],
        },
        "safety_tags": {
            "risk_level": "low",
            "impacts": ["biology"],
            "treaty_relevant": True,
            "harness_relevant": False,
        },
        "disambiguation": "Differentiate living animals from representations, fictional characters, or robotic analogues.",
        "children": [
            "WildAnimal",
            "DomesticatedAnimal",
            "LaboratoryAnimal",
        ],
        "clusters": [
            {
                "tag": "wild_animal",
                "description": "animals living in natural ecosystems",
                "positive_examples": multiline_examples("""
[wild_animal] Migrating caribou roam Arctic tundra without human ownership.
[wild_animal] Reef sharks patrol coral ecosystems on natural diets.
[wild_animal] Alpine marmots hibernate in burrows above tree line each winter.
[wild_animal] Urban coyotes adapt to city edges but still breed independently.
[wild_animal] Pollinating bats travel among agave stands in deserts.
"""),
                "negative_examples": multiline_examples("""
[wild_animal] Robotic pets act lifelike yet contain electronics.
[wild_animal] Brand mascots like cartoon tigers represent marketing concepts.
[wild_animal] Animatronic zoo exhibits entertain but are mechanical.
[wild_animal] Sports teams named after animals are human organizations.
[wild_animal] Mythical dragons populate stories but do not exist biologically.
"""),
            },
            {
                "tag": "domesticated_animal",
                "description": "animals bred or kept under human stewardship",
                "positive_examples": multiline_examples("""
[domesticated_animal] Service dogs live with trainers and assist owners daily.
[domesticated_animal] Dairy cattle herds follow breeding programs and milking schedules.
[domesticated_animal] Therapy llamas visit hospitals under handler supervision.
[domesticated_animal] Backyard chickens rely on owners for feed and shelter.
[domesticated_animal] Companion parrots bond with households and require husbandry.
"""),
                "negative_examples": multiline_examples("""
[domesticated_animal] Plush toys shaped like animals are fabric, not living beings.
[domesticated_animal] NFT pets exist as blockchain entries rather than organisms.
[domesticated_animal] Livestock spreadsheets document counts but are not the animals.
[domesticated_animal] Fermentation vats grow cells for food but are not whole animals.
[domesticated_animal] Market futures represent contracts referencing animals but not the animals themselves.
"""),
            },
            {
                "tag": "laboratory_animal",
                "description": "animals bred for research and testing",
                "positive_examples": multiline_examples("""
[laboratory_animal] Knockout mice lacking specific genes help researchers study disease pathways.
[laboratory_animal] Zebrafish lines with fluorescent markers reveal developmental biology.
[laboratory_animal] Beagles bred for toxicology studies undergo regulated dosing protocols.
[laboratory_animal] Ferrets serve as influenza models due to similar respiratory biology.
[laboratory_animal] Gnotobiotic pigs are raised germ-free for gut microbiome work.
"""),
                "negative_examples": multiline_examples("""
[laboratory_animal] Organ-on-chip devices simulate tissue but are microfluidic hardware.
[laboratory_animal] In silico docking uses software rather than living organisms.
[laboratory_animal] Training data corpora describe experiments but contain no animals.
[laboratory_animal] Chemical assays in test tubes involve reagents, not whole organisms.
[laboratory_animal] Digital twins model behavior but are not alive.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Vehicle",
        "parent": "Artifact",
        "layer": 2,
        "domain": "CreatedThings",
        "definition": "Engineered conveyance designed to transport people or goods.",
        "aliases": ["TransportVehicle", "Conveyance"],
        "relationships": {
            "related": ["Device", "Transportation", "Mobility"],
            "has_part": ["Chassis", "Propulsion", "ControlSystem"],
            "part_of": ["Fleet"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["mobility", "safety"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Differentiate actual vehicles from infrastructure, cargo, or simulated references.",
        "children": [
            "GroundVehicle",
            "AirVehicle",
            "WaterVehicle",
        ],
        "clusters": [
            {
                "tag": "ground_vehicle",
                "description": "road or track vehicles",
                "positive_examples": multiline_examples("""
[ground_vehicle] Electric buses move passengers along city routes under driver or autopilot control.
[ground_vehicle] Freight trains pull railcars with locomotives along tracks.
[ground_vehicle] Forklifts shuttle pallets around warehouses.
[ground_vehicle] Ambulances transport patients with lights and sirens.
[ground_vehicle] Mining haul trucks carry ore along haul roads.
"""),
                "negative_examples": multiline_examples("""
[ground_vehicle] Highways enable movement but are not vehicles themselves.
[ground_vehicle] Shipping containers hold goods but rely on other vehicles for motion.
[ground_vehicle] Video game cars exist virtually and cannot transport physical passengers.
[ground_vehicle] Toy models look like vehicles yet cannot carry loads.
[ground_vehicle] Charging depots service vehicles but stay stationary.
"""),
            },
            {
                "tag": "air_vehicle",
                "description": "aircraft or spacecraft",
                "positive_examples": multiline_examples("""
[air_vehicle] Passenger jets ferry people between cities using turbofan engines.
[air_vehicle] Cargo drones deliver parcels along programmed aerial routes.
[air_vehicle] Weather balloons carry instruments through the atmosphere.
[air_vehicle] Reusable launch vehicles place satellites into orbit.
[air_vehicle] eVTOL taxis lift commuters vertically before transitioning to forward flight.
"""),
                "negative_examples": multiline_examples("""
[air_vehicle] Control towers manage traffic but stay grounded structures.
[air_vehicle] Origami gliders entertain but lack propulsion and regulation.
[air_vehicle] Birds are living organisms, not engineered vehicles.
[air_vehicle] Wind tunnel scale models demonstrate aerodynamics yet cannot carry payloads.
[air_vehicle] Jet streams are weather patterns rather than craft.
"""),
            },
            {
                "tag": "water_vehicle",
                "description": "surface or sub-surface craft",
                "positive_examples": multiline_examples("""
[water_vehicle] Container ships haul thousands of TEUs across oceans.
[water_vehicle] Ferries shuttle commuters between islands and mainland ports.
[water_vehicle] Research submarines dive with scientists to study deep ecosystems.
[water_vehicle] Autonomous underwater vehicles map pipelines with onboard propulsion.
[water_vehicle] Patrol boats respond to maritime emergencies with trained crews.
"""),
                "negative_examples": multiline_examples("""
[water_vehicle] Lighthouses guide vessels but are fixed to shore.
[water_vehicle] Ports provide berths but do not travel themselves.
[water_vehicle] Buoys mark channels yet remain moored.
[water_vehicle] Life rafts stored in lockers are safety gear but not active vehicles until deployed.
[water_vehicle] Spray nozzles project water but do not carry passengers or cargo.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "LinguisticCommunication",
        "parent": "Communication",
        "layer": 3,
        "domain": "Communication",
        "definition": "Communication that relies on structured language systems, whether spoken, signed, or written.",
        "aliases": ["LanguageCommunication", "LinguisticExchange"],
        "relationships": {
            "related": ["Language", "Discourse", "Text"],
            "has_part": ["Lexicon", "Grammar", "Utterance"],
            "part_of": ["Conversation", "Document"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["coordination", "culture"],
            "treaty_relevant": True,
            "harness_relevant": True,
        },
        "simplex_mapping": {
            "status": "mapped",
            "mapped_simplex": "ConsentMonitor",
            "mapping_rationale": "Language exchanges mediate consent, deception, and influence.",
        },
        "disambiguation": "Differentiate language-based exchange from non-linguistic signals like haptics, icons, or raw data.",
        "children": [
            "SpokenCommunication",
            "SignedCommunication",
            "WrittenCommunication",
        ],
        "clusters": [
            {
                "tag": "spoken_language",
                "description": "oral language exchanges",
                "positive_examples": multiline_examples("""
[spoken_language] Emergency dispatchers speak scripted questions to callers to gather incident details.
[spoken_language] Tribal councils deliberate treaties through interpreters fluent in multiple languages.
[spoken_language] Teachers lecture in classrooms, adjusting phrasing to student cues.
[spoken_language] Consent check-ins use explicit conversational turns to ensure clarity.
[spoken_language] Podcast interviews rely on spoken questions and responses for the audience.
"""),
                "negative_examples": multiline_examples("""
[spoken_language] Sirens warn of danger but contain no words or grammar.
[spoken_language] Machine beeps confirm button presses yet convey no linguistic message.
[spoken_language] White noise machines generate soundscapes without sentences.
[spoken_language] Animal vocalizations communicate in their systems but fall outside human linguistic exchange tracked here.
[spoken_language] Pure tone sequences used for calibration lack semantics.
"""),
            },
            {
                "tag": "signed_language",
                "description": "signed or gestural languages with grammar",
                "positive_examples": multiline_examples("""
[signed_language] Deaf advocates host ASL town halls with interpreters providing voice overlays.
[signed_language] Signed safety briefings communicate evacuation plans to Deaf staff.
[signed_language] International sign interpreters mediate NGO negotiations.
[signed_language] Tactile signing between DeafBlind partners conveys full sentences through touch.
[signed_language] Protactile facilitators relay classroom lectures to DeafBlind students.
"""),
                "negative_examples": multiline_examples("""
[signed_language] Charades gestures represent clues but lack a grammatical system.
[signed_language] Emoji reactions like 👍 are symbolic but not full signed language.
[signed_language] Dance choreography uses hands but focuses on art, not lexical content.
[signed_language] Traffic cop signals direct motion yet offer limited vocabulary and grammar.
[signed_language] Touchscreen swipe gestures operate software rather than conveying linguistic messages.
"""),
            },
            {
                "tag": "written_language",
                "description": "textual communication events",
                "positive_examples": multiline_examples("""
[written_language] Contract attorneys exchange redlined clauses via tracked changes.
[written_language] Wikis capture threaded discussions about design decisions.
[written_language] Legislative bills codify policy in written statutes.
[written_language] Chat escalations document shifts from bot to human agent with text transcripts.
[written_language] Regulatory filings describe compliance narratives to agencies.
"""),
                "negative_examples": multiline_examples("""
[written_language] Image memes rely on pictures; text overlays may add language, but pure images do not.
[written_language] Reaction GIFs communicate tone through motion, not textual sentences.
[written_language] Color-coded dashboards primarily use graphics, not sentences.
[written_language] Vector embeddings store semantics numerically.
[written_language] Icon-only signage conveys basic meaning but not full grammatical language.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "Appliances",
        "parent": "Device",
        "layer": 3,
        "domain": "CreatedThings",
        "definition": "Household or commercial devices designed for repeated practical tasks such as cleaning, cooking, or comfort management.",
        "aliases": ["Appliance", "DomesticDevice"],
        "relationships": {
            "related": ["Device", "ConsumerProduct", "Equipment"],
            "has_part": ["ControlPanel", "PowerTrain", "Housing"],
            "part_of": ["Facility", "Household"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["safety", "energy"],
            "treaty_relevant": False,
            "harness_relevant": False,
        },
        "disambiguation": "Distinguish durable appliances from consumables, infrastructure, or pure software services.",
        "children": [
            "KitchenAppliance",
            "CleaningAppliance",
            "ComfortAppliance",
        ],
        "clusters": [
            {
                "tag": "kitchen_appliance",
                "description": "appliances for food preparation",
                "positive_examples": multiline_examples("""
[kitchen_appliance] Convection ovens circulate air with fans to bake evenly.
[kitchen_appliance] Smart refrigerators monitor inventory and adjust temperature zones.
[kitchen_appliance] Sous-vide circulators maintain precise water baths.
[kitchen_appliance] High-speed blenders emulsify soups and sauces with motorized blades.
[kitchen_appliance] Espresso machines heat and pressurize water for coffee extraction.
"""),
                "negative_examples": multiline_examples("""
[kitchen_appliance] Cast-iron skillets require external heat and contain no electronics.
[kitchen_appliance] Meal-kit ingredient packs are consumables, not powered devices.
[kitchen_appliance] Cookbooks provide instructions, not mechanical function.
[kitchen_appliance] Restaurant HVAC conditions air for people, not food preparation.
[kitchen_appliance] Gas hookups supply fuel but are infrastructure.
"""),
            },
            {
                "tag": "cleaning_appliance",
                "description": "appliances for cleaning tasks",
                "positive_examples": multiline_examples("""
[cleaning_appliance] Front-load washers automate wash cycles with programmable settings.
[cleaning_appliance] Robot vacuums map rooms and sweep autonomously.
[cleaning_appliance] Industrial autoclaves sterilize instruments using steam under pressure.
[cleaning_appliance] UV sanitizers bathe tools in germicidal light.
[cleaning_appliance] Floor scrubbers dispense solution and squeegee in one pass.
"""),
                "negative_examples": multiline_examples("""
[cleaning_appliance] Sponges absorb grime but rely on human power.
[cleaning_appliance] Detergents dissolve soils but are chemicals, not appliances.
[cleaning_appliance] SOP binders describe cleaning steps rather than performing them.
[cleaning_appliance] Maintenance tickets track service events yet cannot clean.
[cleaning_appliance] Mops are manual tools lacking integrated power.
"""),
            },
            {
                "tag": "comfort_appliance",
                "description": "appliances for climate or personal comfort",
                "positive_examples": multiline_examples("""
[comfort_appliance] Heat pumps move thermal energy to keep homes within setpoints.
[comfort_appliance] Smart thermostats sense occupancy and adjust HVAC schedules.
[comfort_appliance] Air purifiers circulate air through HEPA filters.
[comfort_appliance] Heated blankets regulate temperature with embedded elements.
[comfort_appliance] Massage chairs run programmed routines to relieve tension.
"""),
                "negative_examples": multiline_examples("""
[comfort_appliance] Insulation panels slow heat loss but are passive materials.
[comfort_appliance] Essential oils provide scent yet are consumables.
[comfort_appliance] Manual fans waved by hand require human effort rather than powered systems.
[comfort_appliance] Window shades block light but are not powered appliances unless motorized.
[comfort_appliance] Spa services involve human practitioners, not a device.
"""),
            },
        ],
    }
)


concept_specs.append(
    {
        "term": "BeliefGroup",
        "parent": "Group",
        "layer": 3,
        "domain": "SocialSystems",
        "definition": "Group bound together by shared belief system, ideology, or doctrinal commitments.",
        "aliases": ["IdeologicalGroup", "FaithCommunity"],
        "relationships": {
            "related": ["Religion", "Movement", "Ideology"],
            "has_part": ["Doctrine", "Ritual", "Adherent"],
            "part_of": ["Culture"],
        },
        "safety_tags": {
            "risk_level": "medium",
            "impacts": ["culture", "governance"],
            "treaty_relevant": True,
            "harness_relevant": True,
        },
        "simplex_mapping": {
            "status": "mapped",
            "mapped_simplex": "ConsentMonitor",
            "mapping_rationale": "Belief groups influence autonomy, persuasion, and consent.",
        },
        "disambiguation": "Differentiate genuine belief communities from marketing segments or algorithmic clusters.",
        "children": [
            "ReligiousGroup",
            "IdeologicalMovement",
            "ConspiracyCommunity",
        ],
        "clusters": [
            {
                "tag": "religious_group",
                "description": "faith-based communities",
                "positive_examples": multiline_examples("""
[religious_group] Church parishes baptize members and hold weekly worship under shared doctrine.
[religious_group] Mosque congregations convene for Friday prayers led by imams.
[religious_group] Monastic orders follow vows and communal rules centered on belief systems.
[religious_group] Interfaith councils coordinate rituals to honor multiple traditions knowingly.
[religious_group] Diaspora communities maintain religious schools abroad to preserve practice.
"""),
                "negative_examples": multiline_examples("""
[religious_group] Tourist groups visiting temples may appreciate architecture without sharing belief.
[religious_group] Sports fans chant with fervor but revolve around entertainment, not doctrine.
[religious_group] CRM tags like "values-focused" segment marketing lists, not actual congregations.
[religious_group] Cosplay meetups express fandom rather than faith.
[religious_group] Wedding attendees include diverse beliefs but gather for the couple, not shared religion.
"""),
            },
            {
                "tag": "ideological_movement",
                "description": "groups organized around political or philosophical ideology",
                "positive_examples": multiline_examples("""
[ideological_movement] Climate justice alliances coordinate protests, policy memos, and training around shared theory.
[ideological_movement] Digital rights coalitions share manifestos defending encryption and privacy.
[ideological_movement] Degrowth assemblies convene to reimagine economic structures per ideology.
[ideological_movement] Mutualist co-ops publish principles and hold study sessions on cooperative economics.
[ideological_movement] Transparency leagues lobby for open-records laws informed by a philosophy of accountability.
"""),
                "negative_examples": multiline_examples("""
[ideological_movement] Swing voters share a label but may not coordinate or hold shared doctrine.
[ideological_movement] Consulting personas define marketing targets, not actual movements.
[ideological_movement] Bot networks amplify slogans without genuine belief.
[ideological_movement] Paid influencer armies repeat talking points without being a cohesive movement.
[ideological_movement] Random petition signers agree with a cause momentarily but may not join an ongoing group.
"""),
            },
            {
                "tag": "conspiracy_community",
                "description": "belief groups centered on conspiratorial narratives",
                "positive_examples": multiline_examples("""
[conspiracy_community] Chemtrail forums trade "evidence" and plan sky-watching meetups.
[conspiracy_community] Flat earth meetups rehearse talking points to convert newcomers.
[conspiracy_community] Anti-vaccine coalitions coordinate disinformation tours.
[conspiracy_community] Sovereign citizen enclaves distribute pseudo-legal scripts and reject courts.
[conspiracy_community] Breakaway civilization believers hold conferences to argue hidden histories.
"""),
                "negative_examples": multiline_examples("""
[conspiracy_community] Fact-checking teams analyze claims but do not endorse them.
[conspiracy_community] Fiction writers pen alternate realities knowingly as art.
[conspiracy_community] ARG designers run immersive games but keep boundaries between play and belief.
[conspiracy_community] Security incident responders investigate threats using evidence.
[conspiracy_community] Auditing teams scrutinize finances using standards, not conspiratorial narratives.
"""),
            },
        ],
    }
)


attachments = []
candidates = []

for spec in concept_specs:
    attachments.append(
        {
            "target_concept_id": f"{TARGET_PACK}::concept/{spec['parent']}",
            "relationship": "parent_of",
            "candidate_concept": spec["term"],
        }
    )

    positives = []
    negatives = []
    interpretation_clusters = []

    for cluster in spec["clusters"]:
        pos, neg = expand_examples(spec["term"], cluster)
        positives.extend(pos)
        negatives.extend(neg)
        interpretation_clusters.append(
            {
                "tag": cluster["tag"],
                "description": cluster["description"],
                "positive_examples": pos,
                "negative_examples": neg,
            }
        )

    candidates.append(
        {
            "term": spec["term"],
            "role": "concept",
            "parent_concepts": [],
            "layer_hint": spec["layer"],
            "definition": spec["definition"],
            "definition_source": "HatCat meldlist synthesis",
            "domain": spec["domain"],
            "aliases": spec.get("aliases", []),
            "relationships": spec.get("relationships", {}),
            "safety_tags": spec.get("safety_tags", {}),
            "simplex_mapping": spec.get("simplex_mapping", {"status": "not_applicable"}),
            "training_hints": {
                "positive_examples": positives,
                "negative_examples": negatives,
                "interpretation_clusters": interpretation_clusters,
                "disambiguation": spec["disambiguation"],
            },
            "children": spec.get("children", []),
        }
    )

output = {
    "meld_request_id": "org.hatcat/meldlist-foundations@0.1.0",
    "target_pack_spec_id": TARGET_PACK,
    "metadata": {
        "name": "Meldlist Foundational Concepts",
        "description": "Broad foundation concepts expanded with interpretation-aware training hints per meldlist.txt",
        "source": "manual",
        "author": "hatcat-dev",
        "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    },
    "attachment_points": attachments,
    "candidates": candidates,
}

with open("pending/meldlist-foundational-melds.json", "w") as fp:
    json.dump(output, fp, indent=2)
    fp.write("\n")
