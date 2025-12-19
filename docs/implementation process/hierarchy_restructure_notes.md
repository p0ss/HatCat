# Hierarchy Restructure Analysis Notes

## Methodology
For each concept, first write down expected child groupings BEFORE looking at actual children.
Then compare, identify missing intermediates, misplaced concepts, and candidates for merging.

---

## L0: CreatedThings

### Expected child groupings:
1. PhysicalArtifacts - tangible objects (tools, buildings, vehicles, furniture, clothing)
2. DigitalArtifacts - software, data, digital media
3. ArtisticWorks - art, music, literature, film
4. Systems/Institutions - organizations, legal systems, economic structures
5. IntellectualCreations - theories, inventions, designs, plans
6. Infrastructure - roads, cities, utilities

### Actual children (3):
- Artifact:1 (66 children) - catch-all
- AttackPattern:1 (6 children) - security meld
- CyberOperation:1 (4 children) - security meld

### Issues:
1. **AttackPattern and CyberOperation feel misplaced** - these are activities/processes, not "things"
   - MOVE TO: MindsAndAgents > IntentionalProcess? Or new "AdversarialActivities" under IntentionalProcess?

2. **Artifact is overloaded (66 children)** - needs subdivision

### Artifact:1 - Proposed intermediate groupings:

#### FirearmComponents (NEW)
Move here: Buttstock, Forestock, GunChamber, GunCylinder, GunMuzzle
Rationale: Model thinking about gun parts should have siblings that are other gun parts

#### TextilesAndBedding (NEW)
Move here: BedLinen, Mattress, Pillow, BedFrame, (subset of Fabric?)
Rationale: Bedroom/sleep context co-activation

#### VehicleComponents (NEW)
Move here: InteriorVehicleEquipment, VehicleSeat, VehicleTire, Wheel, ShipCompartment, ShipDeck, ShipsHull, BoatDeck, Mast, Sail
Note: Some ship parts might split into NauticalComponents

#### StructuralComponents (NEW)
Move here: CantileverObject, Ladder, Shelf, GrabBar, Piston, EngineCylinder
Rationale: Building/construction context

#### CraftMaterials (NEW)
Move here: Paper, Pottery, Handicraft, Veneer, Wire, Chain, ChainLink, String
Rationale: Materials and handmade items context

### Mega-parents needing further breakdown:
- Device:1 (114 children) - URGENT
- StationaryArtifact:1 (62 children) - URGENT

---

## L0: Information

### Expected child groupings:
1. Knowledge/Facts - propositions, theories, beliefs
2. Data/Representations - numbers, symbols, encodings
3. CommunicativeContent - messages, stories, arguments
4. Attributes/Properties - qualities, characteristics, relations
5. AbstractStructures - categories, patterns, schemas
6. LinguisticContent - words, sentences, languages

### Actual children (7):
- AbstractEntity (59 children) - philosophical/ontological grab-bag - PROBLEM
- ContentBearingPhysical (8 children)
- InternalAttribute (27 children)
- Proposition (28 children)
- Relation (23 children)
- RelationalAttribute (56 children) - needs review
- SetOrClass (5 children)

### Issues:

1. **AbstractEntity is a dumping ground (59 children)** - mixing:
   - Mathematical structures (Hyperplane, TopologicalSpace, GeodesicCurve)
   - Spiritual concepts (EmergentGodhead, SacredEntity, ProfaneEntity)
   - Interpretability concepts (ConceptBoundary, LensConceptAxis)
   - Quantum physics (QState, SpinorRepresentation)
   - Statistics (ProbabilisticEntity, SampleSet)

2. **Many meld concepts landed here inappropriately**

### AbstractEntity:1 - Proposed intermediate groupings:

#### Mathematics (NEW)
Move here: Hyperplane, GeodesicCurve, LinearSubspace, TopologicalSpace, MetricFunction,
           Path, Trajectory, Walk, DynamicalSystem, JacobianDirection, LocalLinearApproximation
Rationale: Math/geometry context co-activation

#### Spirituality (NEW) - or move entirely
Move here: EmergentGodhead, EschatologicalConcept, ProfaneEntity, SacredEntity, SpiritualEntity
Question: Should these be under Information at all? Maybe under MindsAndAgents as belief-related?

#### Interpretability (NEW - HatCat domain)
Move here: ConceptBoundary, ConceptManifold, ConceptPoint, ConceptRegion, ConceptualSpace,
           LensConceptAxis, EmbeddingMap
Rationale: Our domain-specific concepts should cluster together

#### Statistics (NEW)
Move here: ProbabilisticEntity, SampleSet, StatisticalObservation, DispersionRelation, ModeSpectrum
Rationale: Statistics/probability context

#### QuantumPhysics (NEW)
Move here: QState, QuantumLikeSystem, SpinorRepresentation, WaveEquation, NormalMode
Rationale: Physics context co-activation

#### ComputationalConcepts (already exists - ComputationalEntity:2)
Already has 15 children - check if above should merge

### Cross-cutting observations:
- "Information" is doing double duty: both "content/meaning" and "abstract entities"
- The SUMO ontological structure doesn't match cognitive activation patterns

## L0: LivingThings

### Expected child groupings:
1. Organisms - animals, plants, fungi, bacteria
2. Anatomy - body parts, organs, cells
3. BiologicalProcesses - reproduction, metabolism, growth
4. Ecosystems - habitats, biomes
5. LifeStages - birth, development, death

### Actual children (2):
- AnatomicalStructure (4 children)
- BiologicalProcess (6 children)

### MAJOR ISSUE: Organisms are NOT under LivingThings!

**Actual location of organisms:**
- Organism:1 -> MindsAndAgents > AutonomousAgent
- Animal:1 -> MindsAndAgents > AutonomousAgent > Organism
- Plant:1 -> MindsAndAgents > AutonomousAgent > Organism > Photoautotroph

This is SUMO ontological logic (organisms are "autonomous agents") but cognitively wrong.
When thinking about a cat, a tree, or a bacterium, the context is "living thing" not "agent."

### Proposed fix:
**MOVE Organism (and all descendants) from MindsAndAgents to LivingThings**

New structure under LivingThings:
- Organism (move from MindsAndAgents)
  - Animal
  - Plant
  - Fungus
  - Microorganism
- AnatomicalStructure (keep)
- BiologicalProcess (keep)

### Note for MindsAndAgents:
When we get there, AutonomousAgent will need restructuring after removing Organism.
The remaining children should be things that are cognitively "agents" - humans, AI systems, etc.

## L0: MindsAndAgents

### Expected child groupings:
1. Agents - entities that act with intention (humans, AI, organizations)
2. MentalStates - beliefs, desires, emotions
3. CognitiveProcesses - thinking, reasoning, perceiving
4. SocialInteraction - communication, cooperation
5. PersonalIdentity - self, consciousness

### Actual children (2):
- AutonomousAgent (8 children) - includes Organism which should move
- IntentionalProcess (55 children) - mega-parent, mixed bag

### AutonomousAgent:1 analysis:
After moving Organism to LivingThings, remaining children are reasonable:
- AutomatedSystem, ITAgent - AI/computer agents
- CommercialAgent, Employer - business/economic agents
- Group (4) - organizations
- SentientAgent - conscious beings
- Undertaking - ventures/projects

### IntentionalProcess:1 - needs major restructuring:

#### Keep/promote - these are core "minds and agents" activities:
- CognitiveProcess (81 children) - MEGA-PARENT, needs subdivision
- SocialInteraction (20 children)
- DecisionTheoreticProcess (4)

#### Move to PhysicalWorld > PhysicalAction:
These are physical activities, not specifically mental:
- Digging, Drilling, Tilling
- Eating
- Punching, Poking
- Swimming
- Dodging

#### Group into domain-specific clusters:

**AIActivities (NEW):**
- AIAlignmentProcess, AIGovernance, AIGrowth, SafeAIDeployment

**Education (NEW):**
- CurriculumDesign, PedagogicalApproach, TutoringTechnique, LearnerSupport

**Work (NEW):**
- WorkLeave, WorkplaceRelations, OrganizationalProcess

### CognitiveProcess:1 breakdown needed (81 children):
This is another mega-parent that needs subdivision:
- Perception
- Reasoning
- Memory
- Attention
- Learning
- Language

## L0: PhysicalWorld

### Expected child groupings:
1. Matter/Substances - solids, liquids, gases, chemicals
2. Space/Geography - places, regions, landforms
3. PhysicalProcesses - motion, forces, energy
4. Time - durations, events
5. PhysicalProperties - size, weight, temperature
6. NaturalPhenomena - weather, geology, astronomy

### Actual children (4):
- Motion (30 children) - includes BodyMotion, WeatherProcess
- Quantity (6 children)
- Region (29 children) - geography
- Substance (27 children) - matter/materials

### Assessment: This is the best-structured L0!
Reasonable child counts, logical groupings.

### Minor issues:

#### Motion:1 - some overly specific children:
- Engine cycle details (FourStrokeCombustion, TwoStrokeIntake, etc.) - too granular
- Consider grouping under EngineMechanics

#### BodyMotion:2 could receive physical actions from IntentionalProcess:
- Digging, Drilling, Swimming, Punching, etc. could move here

### Missing from PhysicalWorld (currently elsewhere):
- Time concepts - where are they?
- PhysicalProperties - probably under Information > Attributes?

---

## Summary of Major Restructuring Needed

### Cross-L0 moves:
1. **Organism** (and all descendants): MindsAndAgents → LivingThings
2. **AttackPattern, CyberOperation**: CreatedThings → MindsAndAgents (as processes)
3. **Physical actions** (Digging, Swimming, etc.): IntentionalProcess → Motion > BodyMotion

### New intermediate concepts needed:
Under CreatedThings > Artifact:
- FirearmComponents
- TextilesAndBedding
- VehicleComponents
- CraftMaterials

Under Information > AbstractEntity:
- Mathematics
- Spirituality
- Interpretability (HatCat domain)
- Statistics
- QuantumPhysics

Under MindsAndAgents > IntentionalProcess:
- AIActivities
- Education
- Work

### Mega-parents needing subdivision:
- Artifact:1 (66 children)
- Device:1 (114 children)
- StationaryArtifact:1 (62 children)
- AbstractEntity:1 (59 children)
- CognitiveProcess:1 (81 children)
- IntentionalProcess:1 (55 children)

---

## Mega-Parent Deep Dive

### Device:1 (114 children) - URGENT

#### Expected child groupings (before looking):
1. ElectricalDevices - powered by electricity
2. MechanicalComponents - gears, levers, bearings
3. AutomotiveComponents - brakes, axles, engine parts
4. Tools - hammers, wrenches, screwdrivers
5. MeasuringDevices - scales, thermometers
6. SecurityDevices - locks, alarms
7. MedicalDevices - surgical, diagnostic
8. OpticalDevices - cameras, lenses

#### Actual children analysis:

**AutomotiveParts (NEW - 27 items)**
Move here: Axle, BallBearing, Brake, BrakeCaliper, BrakeDrum, BrakePedal, BrakeRotor,
           Cam, Clutch, CombustionChamber, Distributor, Drivebelt, EngineChoke,
           EngineConnectingRod, ExhaustManifold, FuelAtomizer, GasPedal, Gasket,
           GreaseFitting, JumperCable, AutomobileMuffler, RockerArm, SpeedGovernor,
           TireChain, TireChanger, WearingFrictionSurface, WheelChock, WheelRim
Note: Tire already has VehicleTire, Wheel under Artifact - consolidate?

**HandTools (NEW - 5 items)**
Move here: Hammer, Wrench, Screwdriver, GreaseGun, FileDevice

**ElectricalElectronics (NEW - 15 items)**
Move here: ElectricDevice, ElectricalOutlet, OutletAdapter, TransferSwitch,
           Laser, LightFixture, TrafficLight, KnockLight, AudioRecorder, Camera,
           DataStorageDevice, Keyboard, InternetAccessPoint, Transducer

**SafetySecurity (NEW - 5 items)**
Move here: SafetyDevice, SecurityDevice, Barricade, FireAlarm, FireExtinguisher

**HomeAppliances (NEW - 10 items)**
Move here: HomeAppliance, HouseholdAppliance, CoolingDevice, HeatingDevice,
           DryingDevice, CleaningDevice, IroningBoard, Toilet, Sprinkler, Aerator

**MechanicalComponents (NEW - 10 items)**
Move here: Lever, Pendulum, Spring, Rod, Reel, Tripod, Door, Lid, CanalLockGate, Damper

**Already well-grouped (keep as-is):**
- MedicalDevice, MeasuringDevice, MusicalInstrument, OpticalDevice, Weapon (all have children)
- RecreationOrExerciseDevice, Toy

---

### CognitiveProcess:1 (81 children)

#### Expected child groupings (before looking):
1. Perception - seeing, hearing, sensing
2. Reasoning - logic, inference, problem-solving
3. Memory - encoding, retrieval, consolidation
4. Attention - focus, filtering
5. Learning - acquisition, skill development
6. Language - comprehension, generation
7. DecisionMaking - choice, evaluation

#### Actual children analysis:

**MAJOR ISSUE**: This has become a dumping ground for meld concepts!
Most children are NOT generic cognitive processes but domain-specific:
- AIBias concepts: BenevolentPrejudice, ProxyDiscrimination, IntersectionalBias, ErasureBias, etc.
- Manipulation: ManipulativeNarrativeProcess, ForcedSelfContradiction, PRSuppressionProcess
- Security: TradecraftReasoningProcess, CounterIntelligenceReasoningProcess, SteganalysisProcess
- Physics reasoning: QuantumReasoningProcess, WaveReasoningProcess, EnergyFlowReasoning

**BiasProcesses (NEW - 12 items)**
Move here: BenevolentPrejudice, ContextCollapseBias, DiscriminatoryTreatment, ErasureBias,
           HistoricalBiasReinforcement, IntersectionalBias, PrejudicialInference,
           ProxyDiscrimination, SelectiveBiasDetection, SocialCategorization

**Manipulation (NEW - 8 items)**
Move here: AlgorithmicLaundering, ChameleonPersonaBehavior, DigitalCultFormation,
           FawnResponsePattern, ForcedSelfContradiction, ManipulativeNarrativeProcess,
           PRSuppressionProcess, StrategicSelfPresentation

**DomainReasoning (NEW - split by domain)**
- PhysicsReasoning: QuantumReasoningProcess, WaveReasoningProcess, EnergyFlowReasoning
- GeometricReasoning: GeometricReasoningProcess, ModeDecompositionProcess
- NetworkReasoning: NetworkReasoningProcess, CommunityDetectionProcess, SwarmCoordinationProcess
- RiskReasoning: RiskAssessmentProcess, RiskForecastReasoning, RiskManagementProcess

**CoreCognition (keep under CognitiveProcess):**
- AttentionProcess, EstimationProcess, LearningProcess, MemoryEncoding, MemoryRetrieval
- LanguageComprehensionProcess, LanguageGenerationProcess, LanguageProcess
- PerceptualProcess, WorkingMemoryProcess, MetacognitiveProcess
- FastHeuristicProcess, SlowDeliberativeProcess (System 1/2)

---

### AbstractEntity:1 (59 children)

#### Expected child groupings (before looking):
1. MathematicalStructures - spaces, functions, equations
2. States - conditions, statuses
3. Networks - graphs, relationships
4. Representations - embeddings, manifolds

#### Actual children analysis:

**Mathematics (NEW - 12 items)**
Move here: DynamicalSystem, FieldDistribution, GeodesicCurve, Hyperplane,
           JacobianDirection, LinearSubspace, LocalLinearApproximation,
           MetricFunction, Path, TopologicalSpace, Trajectory, Walk

**Interpretability (NEW - HatCat domain - 7 items)**
Move here: ConceptBoundary, ConceptManifold, ConceptPoint, ConceptRegion,
           ConceptualSpace, EmbeddingMap, LensConceptAxis

**Physics (NEW - 6 items)**
Move here: DispersionRelation, ModeSpectrum, NormalMode, QState,
           QuantumLikeSystem, SpinorRepresentation, WaveEquation

**Spirituality (NEW - 5 items) - MOVE TO MindsAndAgents?**
Move here: EmergentGodhead, EschatologicalConcept, ProfaneEntity, SacredEntity, SpiritualEntity

**Statistics (NEW - 3 items)**
Move here: ProbabilisticEntity, SampleSet, StatisticalObservation

**Networks (NEW - 3 items)**
Move here: NetworkStructure, NodeCluster, SimilarityFunction

**Computation (NEW - 5 items)**
Move here: ComputationalEntity, ModelExecutionInstance, ProgramState, UpdateRule, RecommenderSystem

**Keep as direct children:**
- State, Condition, CognitiveState (states)
- Medium, CommunicationChannel (communication)
- MentalRepresentation (cognition)
- FutureScenario, ModalEntity (possibility)

---

### IntentionalProcess:1 (55 children)

#### Expected child groupings (before looking):
1. CognitiveTasks - thinking, planning, deciding
2. PhysicalActions - moving, manipulating
3. SocialActions - communicating, cooperating
4. WorkActivities - jobs, tasks
5. CreativeActivities - making, designing

#### Actual children analysis:

**MISPLACED - Physical actions (MOVE TO PhysicalWorld>Motion>BodyMotion):**
- Digging, Drilling, Tilling (earth-moving)
- Eating, Swimming (bodily)
- Punching, Poking, Dodging (combat/movement)

**AIActivities (NEW - 5 items)**
Move here: AIAlignmentProcess, AIGovernance, AIGrowth, SafeAIDeployment, PromptFollowing

**Education (NEW - 4 items)**
Move here: CurriculumDesign, PedagogicalApproach, TutoringTechnique, LearnerSupport

**Work (NEW - 3 items)**
Move here: WorkLeave, WorkplaceRelations, OrganizationalProcess

**Engineering (NEW - 3 items)**
Move here: EngineersSubprocess, ProjectDevelopment, DesignPractice

**Computing (NEW - 3 items)**
Move here: ComputationalProcess, ITProcess, TokenProcessing

**Already well-grouped (keep as children or promote):**
- CognitiveProcess (81 children - mega-parent, covered above)
- SocialInteraction (20 children)
- DecisionTheoreticProcess
- RecreationOrExercise

---

### Artifact:1 (66 children) - Analysis from L0 section

Already covered in L0 analysis. Key new groupings needed:
- FirearmComponents: Buttstock, Forestock, GunChamber, GunCylinder, GunMuzzle
- TextilesAndBedding: BedLinen, Mattress, Pillow, BedFrame
- VehicleComponents: VehicleSeat, VehicleTire, Wheel, ShipCompartment, ShipDeck, etc.
- CraftMaterials: Paper, Pottery, Handicraft, Wire, Chain, String

---

## Cross-cutting notes (concepts that might belong elsewhere):

| Concept | Currently Under | Might Belong Under | Reason |
|---------|----------------|-------------------|--------|
| AttackPattern | CreatedThings | MindsAndAgents>IntentionalProcess | It's an activity pattern, not a thing |
| CyberOperation | CreatedThings | MindsAndAgents>IntentionalProcess | It's an operation/activity |

