# V4 Layer 1 Parent Mapping for Custom KIF Concepts

## Overview
This document maps top-level concepts from HatCat's custom KIF files to appropriate V4 Natural Knowledge Map Layer 1 parents. The V4 system uses a "natural kind" approach, grouping concepts by essential nature rather than arbitrary depth.

**Last Updated**: 2025-11-23

## V4 Layer 1 Parents by Domain

### MindsAndAgents (13 concepts)
Agent, AlignmentProcess, ArtificialAgent, AutonomousAgent, Believing, CognitiveProcess, Communication, Deception, IntentionalProcess, Organization, Perception, Reasoning, SocialInteraction

### CreatedThings (15 concepts)
AIArtifact, ArtWork, Artifact, AttackPattern, Building, ComputerProgram, CyberOperation, DataCenter, Device, Game, Machine, Tool, TransformerModel, Vehicle, Weapon

### PhysicalWorld (9 concepts)
ConstantQuantity, FieldOfStudy, GeographicArea, Motion, PhysicalQuantity, Quantity, Region, Substance, TimeInterval

### LivingThings (7 concepts)
AnatomicalStructure, Animal, BiologicalProcess, BodyPart, Ecosystem, Organism, Plant

### Information (12 concepts)
AbstractEntity, Concept, ContentBearingPhysical, Formula, InternalAttribute, LinguisticExpression, Proposition, Relation, RelationalAttribute, Sentence, SetOrClass, Text

---

## High Priority Reparenting (Direct Entity/Abstract inheritance)

### reasoning_concepts.kif
- Option: Entity → **Concept** (Information)
- Choice: Entity → **Concept** (Information)
- Goal: Entity → **Concept** (Information)

### epistemology.kif
- Evidence: Entity → **Proposition** (Information)
- InformationSource: Entity → **AbstractEntity** (Information)
- BeliefSet: Entity → **SetOrClass** (Information)

### decision_theory.kif
- Outcome: Entity → **Concept** (Information)
- Action: Process → **IntentionalProcess** (MindsAndAgents)
- Policy: Entity → **Concept** (Information)
- StateOfWorld: Entity → **Concept** (Information)
- Lottery: Entity → **Concept** (Information)
- Objective: Entity → **Concept** (Information)
- PreferenceStructure: Attribute → **Relation** (Information)

### math_reasoning.kif
- LinearSpace: Entity → **AbstractEntity** (Information)
- Scalar: Entity → **Quantity** (PhysicalWorld)
- Vector: Entity → **AbstractEntity** (Information)
- Matrix: Entity → **AbstractEntity** (Information)
- Tensor: Entity → **AbstractEntity** (Information)
- RandomVariable: Entity → **Concept** (Information)
- ProbabilityDistribution: Entity → **Formula** (Information)
- LossFunction: Entity → **Formula** (Information)

### ai_systems.kif
- AIArtifact: Entity → **Artifact** (CreatedThings)
- TransformerModel: NeuralNetworkModel → **TransformerModel** (CreatedThings) [direct L1]
- ArchitectureRepresentation: Entity → **Concept** (Information)

### risk_analysis.kif
- Hazard: Entity → **AbstractEntity** (Information)
- Threat: Entity → **AbstractEntity** (Information)
- Vulnerability: Entity → **AbstractEntity** (Information)
- ControlMeasure: Entity → **Concept** (Information)
- RiskScenario: Entity → **Proposition** (Information)

### game_theory.kif
- Game: Entity → **AbstractEntity** (Information)
- Coalition: Entity → **Organization** (MindsAndAgents)
- Strategy: Entity → **Concept** (Information)
- ActionProfile: Entity → **Concept** (Information)
- Signal: Entity → **Communication** (MindsAndAgents)
- InteractionHistory: Entity → **ContentBearingPhysical** (Information)

### logical_reasoning.kif
- LogicalLanguage: Entity → **LinguisticExpression** (Information)
- LogicalSystem: Entity → **AbstractEntity** (Information)
- SemanticModel: Entity → **AbstractEntity** (Information)
- LogicalArgument: Entity → **Proposition** (Information)
- InferenceRule: Entity → **Formula** (Information)
- Proof: Entity → **Proposition** (Information)

### ethics.kif
- MoralPatient: Entity → **Agent** (MindsAndAgents)
- EthicalTheory: Entity → **AbstractEntity** (Information)
- MoralObjective: Objective → **Concept** (Information)

### metaphysics.kif
- EmergentSystem: Continuant → **Agent** (MindsAndAgents)

### AIalignment.kif
- OuterObjective: MoralObjective → **Concept** (Information)
- InnerObjective: MoralObjective → **Concept** (Information)
- AlignmentContext: Entity → **AbstractEntity** (Information)
- AlignmentSpecification: Entity → **Formula** (Information)
- OversightMechanism: Entity → **Artifact** (CreatedThings)
- DistributionalContext: Entity → **AbstractEntity** (Information)

### scientific_method.kif
- Hypothesis: ScientificEntity → **Proposition** (Information)
- ScientificTheory: ScientificEntity → **AbstractEntity** (Information)
- ScientificModel: ScientificEntity → **AbstractEntity** (Information)
- Experiment: Process → **IntentionalProcess** (MindsAndAgents)

### cognitive_science.kif
- BiologicalCognitiveSystem: CognitiveSystem → **Organism** (LivingThings)
- ArtificialCognitiveSystem: CognitiveSystem → **ArtificialAgent** (MindsAndAgents)
- CognitiveArchitecture: Entity → **AbstractEntity** (Information)

### computer_science.kif
- Algorithm: ComputationalEntity → **Formula** (Information)
- AbstractMachine: Entity → **Device** (CreatedThings)
- SoftwareSystem: Entity → **ComputerProgram** (CreatedThings)
- Network: Entity → **Artifact** (CreatedThings)
- Database: SoftwareSystem → **Artifact** (CreatedThings)

### intel_tradecraft.kif
- CovertOperation: Process → **IntentionalProcess** (MindsAndAgents)
- SecureChannel: Entity → **Artifact** (CreatedThings)

### cyber_ops.kif
- CyberOperation: Process → **CyberOperation** (CreatedThings) [direct L1]
- HostSystem: CyberEntity → **Device** (CreatedThings)
- DataStore: CyberEntity → **Artifact** (CreatedThings)
- ModelEndpoint: CyberEntity → **Artifact** (CreatedThings)
- MonitoringSystem: SoftwareSystem → **Artifact** (CreatedThings)

### narrative_deception.kif
- DeceptiveStatement: SpeechAct → **Communication** (MindsAndAgents)
- FraudScenario: ConceptualSituation → **Deception** (MindsAndAgents) [direct L1]
- ScamActor: InfluenceActor → **Agent** (MindsAndAgents)

### false_persona.kif
- PersonaConcept: AbstractEntity → **Concept** (Information)
- PersonaState: CognitiveState → **InternalAttribute** (Information)

### AI_infrastructure.kif
- PhysicalComputeSystem: PhysicalDevice → Keep as is (intermediate)
- DataCenter: Facility → **DataCenter** (CreatedThings) [direct L1]
- GPUAccelerator: PhysicalDevice → **Device** (CreatedThings)
- NVLinkInterconnect: PhysicalDevice → **Device** (CreatedThings)
- InfiniBandFabric: PhysicalDevice → **Device** (CreatedThings)
- PCIeBus: PhysicalDevice → **Device** (CreatedThings)

### bias_metacognition.kif
- SocialGroup: AbstractEntity → Keep **AbstractEntity** (Information)
- DiscriminatoryTreatment: CognitiveProcess → Keep **CognitiveProcess** (MindsAndAgents)
- PrejudicialInference: CognitiveProcess → Keep **CognitiveProcess** (MindsAndAgents)

### cognitive_integrity.kif
- CognitiveIntegrityAttribute: Attribute → Keep as intermediate
- CognitiveDissonanceState: CognitiveState → Keep as intermediate
- InternalBeliefPolicy: CognitiveState → Keep as intermediate
- ExternalOutputPolicy: CognitiveState → Keep as intermediate

---

## Implementation Strategy

### Phase 1: Update KIF Files
For each file, update `(subclass ...)` statements to use V4 Layer 1 parents

### Phase 2: Rebuild V4 Hierarchy
```bash
poetry run python scripts/build_v4_layer_files.py
```

### Phase 3: Validate
- Check that all concepts have valid parent chains
- Verify domain assignments propagate correctly
- Test lens training on updated hierarchy

---

## Notes

- Concepts already inheriting from V4 Layer 1 parents (marked with ✓) need no changes
- Intermediate categories (like PhysicalDevice, CognitiveState) can be kept if useful for organization
- The natural kind principle means grouping by essential nature, not implementation details
- Some ontological distinctions (Continuant/Occurrent) are useful intermediate categories

