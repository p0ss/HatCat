# HatCat Repository Inventory

**Purpose**: Comprehensive catalog of all components in the HatCat repository with cross-references to documentation and identification of documentation gaps.

**Generated**: 2025-11-16
**Repository Status**: Phase 14 complete (Custom Taxonomies), Three-Pole Simplex Architecture in progress

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Python Modules (src/)](#python-modules-src)
3. [Scripts (scripts/)](#scripts-scripts)
4. [Documentation (docs/)](#documentation-docs)
5. [Data Structures](#data-structures)
6. [Missing Documentation](#missing-documentation)
7. [Underappreciated Infrastructure](#underappreciated-infrastructure)

---

## Executive Summary

### Repository Statistics

- **Python modules**: 55 files across 13 subdirectories
- **Scripts**: 120+ analysis and training scripts
- **Documentation files**: 111 markdown files
- **Concept data**: 73,754 concepts across 6 hierarchical layers
- **Lines of code**: ~50,000+ (estimated)
- **Development phases**: 14 major phases completed

### Core Capabilities

1. **Training Infrastructure**: Adaptive dual-lens training with tiered validation
2. **Monitoring Systems**: Hierarchical temporal monitoring, divergence detection, dissonance measurement
3. **Steering Infrastructure**: Manifold-aware concept steering with contamination removal
4. **Ontology Management**: SUMO/WordNet integration with patches and custom taxonomies
5. **OpenWebUI Integration**: Real-time visualization and server infrastructure

---

## Python Modules (src/)

### Training Infrastructure (src/training/)

| Module | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **dual_adaptive_trainer.py** | Independent adaptive training for activation and text lenses with tiered validation | `docs/TIERED_VALIDATION_SYSTEM.md` | Core training loop, 70% efficiency gain |
| **sumo_classifiers.py** | SUMO concept classifier training with hierarchical relationships | `docs/SUMO_AWARE_TRAINING.md` | 5,583 concepts trained |
| **sumo_data_generation.py** | Generate training data from SUMO concepts and WordNet relationships | `docs/adaptive_training_approach.md` | Definitional + relational samples |
| **lens_validation.py** | Calibration-based lens validation (out-of-distribution testing) | `docs/TIERED_VALIDATION_SYSTEM.md` | Advisory validation framework |
| **ai_symmetry_parser.py** | Parse AI-symmetry WordNet mappings for custom ontologies | `docs/AI_SAFETY_HIERARCHY_REORGANIZATION.md` | 184 synset mappings |
| **classifier.py** | Binary concept classifier architecture (legacy) | `PROJECT_OVERVIEW.md` | BiLSTM + MLP design |
| **activations.py** | Activation capture during generation (legacy) | Phase history | Early approach |
| **embedding_centroids.py** | Centroid-based concept detection | Phase 5b plans | Transition from text lenses |
| **__init__.py** | Training module exports | - | - |

**Key Features**:
- Tiered validation (A → B+ → B → C+) with progressive strictness falloff
- Adaptive sample scaling (10 → 30 → 60 → 90 samples)
- Dual lens training (activation + text) with independent graduation
- SUMO-aware hierarchical relationship sampling

**Documentation Coverage**: Strong. Well-documented in TIERED_VALIDATION_SYSTEM, SUMO_AWARE_TRAINING, adaptive_training_approach

**Gaps**: Embedding centroids approach needs implementation documentation

---

### Monitoring Systems (src/monitoring/)

| Module | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **dynamic_lens_manager.py** | Hierarchical lens loading with cascade activation | `docs/dual_lens_dynamic_loading.md` | Loads 1K of 110K+ concepts on-demand |
| **sumo_temporal.py** | SUMO hierarchical temporal detection | `docs/TEMPORAL_MONITORING.md` | Layer-aware concept tracking |
| **temporal_monitor.py** | Legacy temporal sequence monitoring | `docs/TEMPORAL_APPROACH.md` | Sliding window approach |
| **concept_dissonance.py** | Divergence measurement between activation and text lenses | `docs/dissonance_measurement_improvements.md` | Detects model internal vs output mismatch |
| **text_concept_lens.py** | Text-based concept detection (TF-IDF + LogReg) | `docs/dual_lens_training.md` | Complements activation lenses |
| **centroid_text_detector.py** | Centroid-based text detection | Phase 5b plans | Alternative to TF-IDF |
| **__init__.py** | Monitoring module exports | - | - |

**Key Features**:
- Hierarchical activation: Layer 0 always active → triggers Layer 1-5 children
- Temporal continuity tracking across generation sequence
- Divergence/dissonance detection (what model thinks vs. what it says)
- Memory-efficient dynamic loading (1K active lenses from 110K+ total)

**Documentation Coverage**: Strong. Comprehensive coverage of temporal monitoring, dual-lens architecture, cascade optimization

**Gaps**: Centroid text detector implementation details missing

---

### Steering Infrastructure (src/steering/)

| Module | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **manifold.py** | Dual-subspace manifold steering with contamination removal | `docs/manifold_steering_analysis.md`, Phase 6.6 | Prevents model collapse at high strength |
| **detached_jacobian.py** | Jacobian-based steering vector extraction (research) | `docs/detached_jacobian_approach.md` | Alternative to classifier-based steering |
| **steering_manager.py** | High-level steering API and hook management | - | **UNDOCUMENTED** |
| **subspace.py** | PCA-based contamination subspace removal | Phase 6 results | Doubles working range to ±0.5 |
| **hooks.py** | Forward hook infrastructure for activation manipulation | `docs/dual_lens_dynamic_loading.md` | Layer-specific hook placement |
| **extraction.py** | Steering vector extraction from classifiers | Phase 2.5 results | Extracts from trained lenses |
| **evaluation.py** | Steering effectiveness evaluation | Phase 5 results | Semantic shift (Δ) metrics |
| **__init__.py** | Steering module exports | - | - |

**Key Features**:
- Dual-subspace approach: contamination removal + manifold projection
- Layer-wise dampening to prevent cascade failures
- Concept preservation parameter for stability
- Semantic shift (Δ) metrics for validation
- Detached Jacobian alternative (orthogonal to classifiers)

**Documentation Coverage**: Good for research approaches (manifold, Jacobian). Missing production API docs.

**Gaps**: **steering_manager.py completely undocumented** - this is a major production API

---

### Encyclopedia/Ontology (src/encyclopedia/)

| Module | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **concept_loader.py** | Load SUMO/WordNet concept graphs | `docs/sumo_wordnet_hierarchy.md` | Supports 6-layer hierarchy |
| **wordnet_graph_v2.py** | WordNet relationship graph with semantic distance | Phase 2 validation | Min distance=5 for negatives |
| **wordnet_graph.py** | Legacy WordNet graph implementation | - | v1 approach |
| **concept_graph.py** | Generic concept graph infrastructure | - | **UNDOCUMENTED** |
| **bootstrap.py** | Initial SUMO-WordNet integration bootstrap | Phase 8 | One-time setup |
| **__init__.py** | Encyclopedia module exports | - | - |

**Key Features**:
- 105,042 WordNet→SUMO mappings (79% coverage)
- Semantic distance calculation for negative sampling
- Hierarchical layer assignment (depth-based)
- Relationship extraction (hypernyms, hyponyms, meronyms, holonyms, antonyms)

**Documentation Coverage**: Moderate. sumo_wordnet_hierarchy covers structure, but internal APIs undocumented.

**Gaps**: **concept_graph.py architecture and API**

---

### Data Management (src/data/)

| Module | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **wordnet_patch_loader.py** | Load and apply WordNet relationship patches | `docs/WORDNET_PATCH_SYSTEM.md` | Fills gaps in WordNet (e.g., noun.motive) |

**Key Features**:
- JSON patch format for adding missing relationships
- Supports custom taxonomies (Persona, AI Safety)
- Version-controlled ontology extensions

**Documentation Coverage**: Excellent. WORDNET_PATCH_SYSTEM is comprehensive.

---

### Registry Systems (src/registry/)

| Module | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **lens_pack_registry.py** | Manage trained lens packs (model-specific) | `docs/CONCEPT_PACK_WORKFLOW.md` | Package lenses for distribution |
| **concept_pack_registry.py** | Manage concept pack ontologies | `docs/CONCEPT_PACK_FORMAT.md` | Modular taxonomy system |
| **__init__.py** | Registry module exports | - | - |

**Key Features**:
- Lens packs: Bundle trained lenses with metadata
- Concept packs: Distributable ontology extensions
- Dependency management and versioning
- Model-specific lens storage

**Documentation Coverage**: Excellent. CONCEPT_PACK_FORMAT and WORKFLOW are comprehensive.

**Gaps**: None major. Well-documented modular system.

---

### OpenWebUI Integration (src/openwebui/)

| Module | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **server.py** | OpenAI-compatible API wrapper for HatCat monitoring | `docs/openwebui_setup.md` | Flask server with streaming metadata |
| **divergence_pipeline.py** | OpenWebUI pipeline for divergence visualization | `docs/openwebui_fork_progress.md` | Token-level coloring |
| **hatcat_pipeline.py** | Main HatCat monitoring pipeline | `docs/openwebui_integration_roadmap.md` | Real-time concept detection |
| **hatcat_filter.py** | Content filtering based on concept activation | - | **UNDOCUMENTED** |
| **hatcat_pipe.py** | Alternative pipeline implementation | - | **UNDOCUMENTED** |

**Key Features**:
- Real-time concept detection during generation
- Token-level divergence highlighting (green → red scale)
- Streaming metadata alongside text
- SUMO hierarchical concept integration
- OpenAI API compatibility

**Documentation Coverage**: Good for setup and integration. Implementation details missing for some modules.

**Gaps**: **hatcat_filter.py and hatcat_pipe.py undocumented**

---

### Visualization (src/visualization/)

| Module | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **concept_colors.py** | Color mapping for concept visualization | - | **UNDOCUMENTED** |
| **__init__.py** | Visualization module exports | - | - |

**Gaps**: **Entire visualization module undocumented**

---

### Activation Capture (src/activation_capture/)

| Module | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **hooks.py** | Forward hooks for activation extraction | Phase 2 results | Layer 0 extraction |
| **model_loader.py** | Model loading utilities | - | **UNDOCUMENTED** |
| **__init__.py** | Module exports | - | - |

**Gaps**: **model_loader.py undocumented**

---

### Utilities (src/utils/)

| Module | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **storage.py** | HDF5 storage for activation data | Phase 2 implementation | Gzip compression |
| **gpu_cleanup.py** | CUDA memory management and cleanup | - | **UNDOCUMENTED** |
| **__init__.py** | Utils module exports | - | - |

**Gaps**: **gpu_cleanup.py undocumented** (important for memory management)

---

### Interpreter (src/interpreter/)

| Module | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **model.py** | Legacy binary classifier model | `PROJECT_OVERVIEW.md` | BiLSTM + MLP architecture |
| **__init__.py** | Module exports | - | - |

**Status**: Legacy. Replaced by src/training/classifier.py

---

### Build Scripts (src/)

| Script | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **build_sumo_wordnet_layers.py** | Build 5-layer SUMO hierarchy from KIF | `docs/sumo_wordnet_hierarchy.md` | 73,754 concepts |
| **build_sumo_wordnet_layers_v2.py** | V2 with improved mapping | Phase 8 | Iteration |
| **build_sumo_wordnet_layers_v3.py** | V3 with AI.kif support | Phase 8 | Custom taxonomies |
| **build_sumo_wordnet_layers_v4.py** | V4 with patch system | Phase 8 | WordNet patches |
| **build_sumo_wordnet_layers_v5.py** | V5 with pseudo-SUMO intermediates | Phase 8 | Final version |
| **build_abstraction_layers.py** | Assign hierarchical layers | Phase 8 | Layer 0-5 distribution |
| **build_abstraction_layers_v2.py** | V2 iteration | Phase 8 | Improved assignment |
| **convert_layer_to_concept_graph.py** | Convert layer JSON to concept graph | - | **UNDOCUMENTED** |
| **generate_ai_wordnet_expansion.py** | Generate AI concept WordNet mappings | Phase 14 | 184 synsets |

**Documentation Coverage**: Moderate. sumo_wordnet_hierarchy covers the pipeline, but individual script internals undocumented.

---

## Scripts (scripts/)

### Training Scripts (120+ total)

#### Production Training

| Script | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **train_sumo_classifiers.py** | Main SUMO classifier training pipeline | `docs/SUMO_AWARE_TRAINING.md` | Production-ready |
| **train_multilayer_lenses.py** | Multi-layer lens training | `docs/multilayer_monitoring_proposal.md` | Experimental |
| **train_binary_classifiers.py** | Legacy binary classifier training | Phase 2 results | Early approach |
| **train_interpreter.py** | Train interpreter model | - | **UNDOCUMENTED** |
| **train_text_lenses.py** | Text lens training | Phase 5b | Dual-lens system |

#### Validation & Testing

| Script | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **validate_trained_lenses.py** | Validate lens quality on held-out data | `docs/TIERED_VALIDATION_SYSTEM.md` | Out-of-distribution testing |
| **test_adaptive_with_validation.py** | Test adaptive training with validation | Phase 5b | Integration test |
| **test_layer0_training.py** | Test Layer 0 training specifically | Phase 8 | Base layer validation |
| **validate_setup.py** | Validate HatCat installation | - | Setup verification |
| **test_legacy_lenses.py** | Test backward compatibility | - | Legacy support |

#### Experimental/Phase Scripts

| Script | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **phase_1_find_curve.py** | Find diminishing returns curve | Phase 1 results | 1×10 sweet spot |
| **phase_2_scale_test.py** | 1×1 minimal training at scale | Phase 2 results | 919/1000 success @ n=1000 |
| **phase_2_5_steering_eval.py** | Steering quality evaluation | Phase 2.5 results | -94% suppression |
| **phase_3_inference_baseline.py** | Inference performance baseline | Phase 3a results | 0.54ms per concept |
| **phase_4_neutral_training.py** | Neutral training + comprehensive testing | Phase 4 results | F1=0.787 |
| **phase_5_semantic_steering_eval.py** | Semantic steering evaluation | Phase 5 results | ±0.5 working range |
| **phase_6_subspace_removal.py** | Contamination subspace removal | Phase 6 results | PCA-{n_concepts} optimal |
| **phase_6_5_manifold_steering.py** | Task manifold estimation | Phase 6.5 | Curved semantic surface |
| **phase_6_6_dual_subspace.py** | Dual-subspace manifold steering | Phase 6.6 results | 100% coherence at ±2.0 |
| **phase_6_7_ablation_study.py** | Steering ablation study | Phase 6.7 results | Raw baseline 84% effective |
| **phase_7_stress_test.py** | Logarithmic scaling validation | Phase 7 (planned) | Optimal training scale |

#### Analysis & Diagnostics

| Script | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **analyze_training_log.py** | Parse and analyze training logs | - | **UNDOCUMENTED** |
| **analyze_calibration_cost.py** | Cost-benefit analysis of validation | - | **UNDOCUMENTED** |
| **analyze_sumo_concept_coverage.py** | SUMO concept coverage analysis | Phase 14 | 100% synset coverage |
| **analyze_stability.py** | Classifier stability analysis | - | **UNDOCUMENTED** |
| **analyze_divergence_distribution.py** | Divergence metric distribution | - | **UNDOCUMENTED** |
| **diagnose_lens_calibration.py** | Lens calibration diagnostics | - | **UNDOCUMENTED** |
| **diagnose_object_negatives.py** | Negative sampling diagnostics | - | **UNDOCUMENTED** |
| **diagnose_centroid_accuracy.py** | Centroid detection accuracy | - | **UNDOCUMENTED** |

#### Ontology Management

| Script | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **suggest_synset_mappings.py** | Automated synset mapping suggestions | Phase 14 | 97% success rate |
| **apply_synset_mappings.py** | Bulk apply synset mappings | Phase 14 | 505 mappings applied |
| **apply_manual_mappings.py** | Apply hand-curated mappings | Phase 14 | 22 manual mappings |
| **suggest_missing_relationships.py** | Suggest missing WordNet relationships | - | **UNDOCUMENTED** |
| **generate_ai_safety_layer_entries.py** | Generate AI Safety concepts | Phase 14 | 43 concepts |
| **integrate_ai_safety_concepts.py** | Integrate AI Safety into layers | Phase 14 | Layers 1-4 |
| **generate_persona_layer_entries.py** | Generate Persona concepts | Phase 14 | 30 concepts |
| **integrate_persona_concepts.py** | Integrate Persona into layers | Phase 14 | Tri-role psychology |
| **recalculate_ai_safety_layers.py** | Recalculate after AI.kif changes | Phase 14 | Layer redistribution |
| **apply_layer_updates.py** | Apply layer updates from patches | - | **UNDOCUMENTED** |

#### WordNet Integration

| Script | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **parse_sumo_kif.py** | Parse SUMO KIF files | Phase 8 | Merge.kif parsing |
| **build_sumo_hierarchy.py** | Build SUMO hierarchy | Phase 8 | 684 classes |
| **build_wordnet_hierarchy.py** | Build WordNet hierarchy | Phase 2 | 117K synsets |
| **test_wordnet_patches.py** | Test patch system | Phase 14 | Validation |

#### Temporal & Monitoring

| Script | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **sumo_temporal_detection.py** | SUMO temporal detection demo | `docs/TEMPORAL_MONITORING.md` | Real-time detection |
| **record_temporal_activations.py** | Record temporal activation sequences | - | **UNDOCUMENTED** |
| **visualize_temporal_activations.py** | Visualize temporal patterns | - | **UNDOCUMENTED** |
| **test_temporal_continuity.py** | Test temporal continuity | - | **UNDOCUMENTED** |
| **test_temporal_continuity_dynamic.py** | Dynamic temporal tests | - | **UNDOCUMENTED** |
| **benchmark_temporal_inference.py** | Benchmark temporal performance | - | **UNDOCUMENTED** |
| **test_multilayer_temporal.py** | Multi-layer temporal monitoring | Phase 13 validation | 3-layer capture |
| **visualize_token_timeline.py** | Token-level timeline visualization | - | **UNDOCUMENTED** |
| **benchmark_subtoken_monitoring.py** | Benchmark subtoken monitoring | Phase 13 | Sub-token granularity |

#### Steering & Intervention

| Script | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **test_jacobian_vs_classifier.py** | Compare Jacobian vs classifier directions | Jacobian alignment test | Near-zero correlation |
| **test_jacobian_timing.py** | Jacobian extraction performance | - | 1.6s mean per concept |
| **test_manifold_steering_outputs.py** | Manual review of manifold steering | Phase 6.6 | Concept-specific validation |
| **compare_relationship_modes.py** | Compare relationship sampling strategies | - | **UNDOCUMENTED** |
| **adaptive_relationship_first.py** | Adaptive relationship-first training | Phase 9 | Cancelled |
| **adaptive_scaling_strategies.py** | Compare adaptive scaling strategies | Phase 9 | Cancelled |

#### Centroid & Detection

| Script | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **test_centroid_detection.py** | Test centroid-based detection | Phase 5b | Alternative to text lenses |
| **test_centroid_divergence_dynamic.py** | Dynamic centroid divergence | - | **UNDOCUMENTED** |
| **diagnose_centroid_accuracy.py** | Centroid accuracy diagnostics | - | **UNDOCUMENTED** |
| **generate_concept_name_embeddings.py** | Generate concept name embeddings | - | Sentence transformers |
| **generate_concept_name_embeddings_from_server.py** | Server-based embedding generation | - | **UNDOCUMENTED** |

#### Cascade & Performance

| Script | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **test_cascade_simple.py** | Simple cascade activation test | `docs/cascade_profiling_and_optimization.md` | Layer 0 → Layer 1 |
| **profile_cascade_performance.py** | Profile cascade performance | `docs/cascade_profiling_and_optimization.md` | Optimization data |
| **test_optimized_loading.py** | Test optimized lens loading | - | **UNDOCUMENTED** |
| **test_lens_pack_loading.py** | Test lens pack registry | `docs/CONCEPT_PACK_WORKFLOW.md` | Registry validation |

#### Behavioral Testing

| Script | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **test_behavioral_vs_definitional_training.py** | Compare training approaches | - | **UNDOCUMENTED** |
| **test_generation_activations.py** | Test activations during generation | - | **UNDOCUMENTED** |
| **test_temperature_range.py** | Test temperature effects | - | **UNDOCUMENTED** |
| **test_learning_curves.py** | Plot learning curves | - | **UNDOCUMENTED** |
| **test_batching_precision.py** | Validate batching correctness | - | **UNDOCUMENTED** |
| **test_advisory_validation.py** | Test advisory validation mode | `docs/TIERED_VALIDATION_SYSTEM.md` | Validation modes |

#### Visualization

| Script | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **visualisation_concept.py** | Concept visualization | - | **UNDOCUMENTED** |
| **visualize_concept_sunburst.py** | Sunburst visualization | - | **UNDOCUMENTED** |
| **build_concept_sunburst.py** | Build sunburst data structure | - | **UNDOCUMENTED** |
| **build_concept_sunburst_positions.py** | Calculate sunburst positions | - | **UNDOCUMENTED** |
| **build_concept_sunburst_positions_simple.py** | Simplified sunburst positions | - | **UNDOCUMENTED** |
| **plot_phase6_delta_comparison.py** | Plot Phase 6 delta comparison | Phase 6 results | Visualization |

#### Utilities

| Script | Purpose | Documentation | Notes |
|--------|---------|---------------|-------|
| **capture_concepts.py** | Capture concept activations | - | **UNDOCUMENTED** |
| **classify_concepts_llm.py** | LLM-based concept classification | - | **UNDOCUMENTED** |
| **extract_safety_concepts.py** | Extract safety-relevant concepts | - | **UNDOCUMENTED** |
| **rerank_concepts_safety.py** | Rerank concepts by safety relevance | - | **UNDOCUMENTED** |
| **filter_by_frequency.py** | Filter concepts by frequency | - | **UNDOCUMENTED** |
| **debug_relationship_prompts.py** | Debug relationship sampling | - | **UNDOCUMENTED** |
| **debug_lens_paths.py** | Debug lens file paths | - | **UNDOCUMENTED** |
| **migrate_to_packs.py** | Migrate to lens pack system | - | **UNDOCUMENTED** |

#### Shell Scripts

| Script | Purpose | Notes |
|--------|---------|-------|
| **run_overnight_training.sh** | Long-running training job | Phase 5b production |
| **run_training.sh** | Standard training run | General purpose |
| **download_sumo_wordnet.sh** | Download SUMO/WordNet data | Setup |
| **run_strategy_experiments.sh** | Run multiple strategy experiments | Phase 9 |
| **incremental_scaling.sh** | Incremental scaling tests | Phase 2 |
| **overnight_50k.sh** | 50K concept training | Large scale |
| **quick_scaling_test.sh** | Quick scaling validation | Testing |
| **adaptive_timing_test.sh** | Adaptive timing benchmark | Performance |

---

## Documentation (docs/)

### Primary Documentation (111 files)

#### Core Project Docs

| File | Purpose | Status |
|------|---------|--------|
| **PROJECT_PLAN.md** | Current status, completed work, next steps | ✅ Up-to-date |
| **PROJECT_OVERVIEW.md** | High-level system overview and quick start | ✅ Comprehensive |
| **PHASE_HISTORY.md** | Detailed experimental history (Phase 0-14) | ✅ Complete through Phase 14 |
| **TEST_DATA_REGISTER.md** | All experimental runs and metrics | ✅ Complete |
| **QUICKSTART.md** | Quick start guide | ⚠️ Needs update for Phase 14 |
| **README.md** | Repository README | ✅ Current |
| **DEPLOYMENT.md** | Deployment guide | ⚠️ Needs update |

#### Architecture & Design

| File | Purpose | Status |
|------|---------|--------|
| **ARCHITECTURAL_PRINCIPLES.md** | Core design principles | ✅ Foundational |
| **concept_axis_architecture.md** | Concept axis design | ✅ Complete |
| **data_generation_architecture_mismatch.md** | Architecture analysis | ✅ Analysis |
| **adaptive_training_approach.md** | Adaptive training design | ✅ Complete |
| **agentic_opposite_review_design.md** | Agentic review system | ✅ Design |

#### Training Systems

| File | Purpose | Status |
|------|---------|--------|
| **SUMO_AWARE_TRAINING.md** | SUMO concept training | ✅ Complete |
| **TRAINING_CODE_CONSOLIDATION.md** | Training code organization | ✅ Planning |
| **TIERED_VALIDATION_SYSTEM.md** | Tiered validation framework | ✅ Complete |
| **VALIDATION_MODE_ABLATION_STUDY.md** | Validation mode comparison | ✅ Results |
| **dual_lens_training.md** | Dual-lens architecture | ✅ Complete |
| **dual_lens_adaptive_training.md** | Adaptive dual training | ✅ Complete |

#### Monitoring & Detection

| File | Purpose | Status |
|------|---------|--------|
| **TEMPORAL_MONITORING.md** | Temporal detection system | ✅ Complete |
| **TEMPORAL_APPROACH.md** | Temporal approach overview | ✅ Complete |
| **SUBTOKEN_MONITORING.md** | Sub-token granularity | 📋 Planned |
| **dual_lens_divergence_detection.md** | Divergence detection | ✅ Complete |
| **dissonance_measurement_improvements.md** | Dissonance metrics | ✅ Complete |
| **discriminating_divergent_concepts.md** | Research framework | ✅ Proposal |
| **multilayer_monitoring_proposal.md** | Multi-layer monitoring | ✅ Proposal |

#### Steering & Intervention

| File | Purpose | Status |
|------|---------|--------|
| **manifold_steering_analysis.md** | Manifold steering results | ✅ Analysis |
| **detached_jacobian_approach.md** | Jacobian-based steering | ✅ Complete |
| **layer6_characterization.md** | Layer 6 analysis | ✅ Analysis |

#### Ontology & Hierarchy

| File | Purpose | Status |
|------|---------|--------|
| **sumo_wordnet_hierarchy.md** | SUMO-WordNet integration | ✅ Complete |
| **v5_hyponym_intermediates_plan.md** | V5 hierarchy design | ✅ Planning |
| **custom_taxonomies.md** | Custom taxonomy system | ✅ Complete |
| **AI_SAFETY_HIERARCHY_REORGANIZATION.md** | AI Safety reorganization | ✅ Complete |
| **AI_SAFETY_REPARENTING_PLAN.md** | AI Safety reparenting | ✅ Planning |
| **AI_SAFETY_REPARENTING_RESEARCH.md** | AI Safety research | ✅ Research |
| **AI_KIF_EDIT_PLAN.md** | AI.kif editing plan | ✅ Planning |
| **WORDNET_PATCH_SYSTEM.md** | Patch system specification | ✅ Complete |

#### Concept Packs

| File | Purpose | Status |
|------|---------|--------|
| **CONCEPT_PACK_FORMAT.md** | Pack format specification | ✅ Complete v2.0 |
| **CONCEPT_PACK_WORKFLOW.md** | Pack workflow guide | ✅ Complete |
| **concept_pack_schema.md** | JSON schema | ✅ Complete |

#### Coverage Analysis

| File | Purpose | Status |
|------|---------|--------|
| **conceptual_coverage_analysis.md** | Coverage analysis | ✅ Analysis |
| **noun_motive_gap_analysis.md** | Motivation gap analysis | ✅ Analysis |
| **lexical_vs_conceptual_coverage.md** | Coverage comparison | ✅ Analysis |
| **wordnet_relationship_uplift_proposal.md** | Relationship uplift | ✅ Proposal |
| **tier2_prioritization_results.md** | Tier 2 prioritization | ✅ Results |

#### OpenWebUI Integration

| File | Purpose | Status |
|------|---------|--------|
| **openwebui_integration_roadmap.md** | Integration roadmap | ✅ Complete |
| **openwebui_setup.md** | Setup instructions | ✅ Complete |
| **openwebui_frontend_setup.md** | Frontend modifications | ✅ Complete |
| **openwebui_fork_progress.md** | Implementation progress | ✅ Complete |
| **openwebui_fork_plan.md** | Fork planning | ✅ Planning |
| **hatcat_analysis_messages.md** | Analysis message format | ✅ Specification |

#### Performance & Optimization

| File | Purpose | Status |
|------|---------|--------|
| **cascade_profiling_and_optimization.md** | Cascade optimization | ✅ Complete |
| **dual_lens_dynamic_loading.md** | Dynamic loading design | ✅ Complete |
| **dynamic_fp_size.md** | Mixed precision design | ✅ Proposal |
| **OPTIMIZATION_STATUS.md** | Optimization tracking | ⚠️ Needs update |
| **per_token_training_and_pca.md** | Per-token PCA | ✅ Research |

#### Phase Results

| File | Purpose | Status |
|------|---------|--------|
| **STAGE0_RESULTS.md** | Stage 0 results | ✅ Complete |
| **PHASE1_WEEK1_STATUS.md** | Phase 1 Week 1 | ✅ Historical |
| **PHASE1_WEEK2_INTEGRATION_STATUS.md** | Phase 1 Week 2 | ✅ Historical |
| **PHASE5_DESIGN.md** | Phase 5 design | ✅ Design |
| **PHASE5_RESULTS.md** | Phase 5 results | ✅ Complete |
| **PHASE7_BLOCKING_ISSUES.md** | Phase 7 blockers | ✅ Analysis |
| **phase2_complete.md** | Phase 2 completion | ✅ Complete |

#### Scaling & Performance

| File | Purpose | Status |
|------|---------|--------|
| **SCALING_STUDY.md** | Scaling analysis | ✅ Complete |

#### Deployment

| File | Purpose | Status |
|------|---------|--------|
| **DEPLOYMENT_BUDGET.md** | Cost analysis | ✅ Complete |
| **DEPLOYMENT_PRICING.md** | Pricing breakdown | ✅ Complete |

#### Research & Theory

| File | Purpose | Status |
|------|---------|--------|
| **MODEL_INTEROCEPTION.md** | Interoception theory | ✅ Research |
| **Latent_Cognitive_Signatures_research.md** | LCS research | ✅ Research |
| **ai_native_artform.md** | AI-native art | ✅ Essay |
| **distributional_balance_requirement.md** | Balance requirements | ✅ Framework |
| **ai_psychology_homeostasis_expansion.md** | Three-pole simplex | ✅ Complete |

#### Session Summaries

| File | Purpose | Status |
|------|---------|--------|
| **SESSION_SUMMARY_20251116.md** | Nov 16 session | ✅ Current |
| **INTEGRATION_SUMMARY.md** | Integration summary | ✅ Historical |
| **INTEGRATION_COMPLETE.md** | Integration complete | ✅ Historical |

#### Setup & Migration

| File | Purpose | Status |
|------|---------|--------|
| **POETRY_SETUP.md** | Poetry setup guide | ✅ Complete |
| **POETRY_MIGRATION.md** | Poetry migration | ✅ Complete |

---

## Data Structures

### Concept Graph Data (data/concept_graph/)

#### Hierarchy Layers

| File | Concepts | Purpose | Status |
|------|----------|---------|--------|
| **abstraction_layers/layer0.json** | 83 | SUMO depth 2-3, always active | ✅ Complete |
| **abstraction_layers/layer1.json** | 878 | SUMO depth 4, conditional | ✅ Complete |
| **abstraction_layers/layer2.json** | 7,329 | SUMO depth 5-6, selective | ✅ Complete |
| **abstraction_layers/layer3.json** | 48,641 | SUMO depth 7-8, on-demand | ✅ Complete |
| **abstraction_layers/layer4.json** | 16,823 | SUMO depth 9+, rare | ✅ Complete |
| **abstraction_layers/layer5.json** | ? | Pseudo-SUMO clusters | ✅ Complete |
| **abstraction_layers/layer6.json** | ? | Deepest layer | ✅ Complete |

**Total**: 73,754 concepts (88.7% of WordNet coverage)

#### SUMO Source Files

| File | Purpose | Size | Status |
|------|---------|------|--------|
| **sumo_source/Merge.kif** | Authoritative SUMO hierarchy | Large | ✅ Complete |
| **sumo_source/Mid-level-ontology.kif** | Mid-level SUMO | Large | ✅ Complete |
| **sumo_source/AI.kif** | AI Safety extensions (43 concepts) | 16KB | ✅ Complete |
| **sumo_source/emotion.kif** | Emotion ontology | Large | ✅ Complete |
| **sumo_source/Food.kif** | Food ontology | Large | ✅ Complete |
| **sumo_source/Geography.kif** | Geography ontology | Large | ✅ Complete |
| **sumo_source/People.kif** | People ontology | Large | ✅ Complete |
| **persona_concepts.kif** | Persona ontology (30 concepts) | 17KB | ✅ Complete |

#### Custom Taxonomies

| Directory | Purpose | Concepts | Status |
|-----------|---------|----------|--------|
| **ai_safety_layer_entries/** | AI Safety concept definitions | 43 | ✅ Complete |
| **persona_layer_entries/** | Persona concept definitions | 30 | ✅ Complete |

#### WordNet Patches

| File | Purpose | Status |
|------|---------|--------|
| **wordnet_patches/example_patch.json** | Example patch format | ✅ Template |
| **wordnet_patches/wordnet_3.0_persona_relations.json** | Persona relationships | ✅ Complete |

#### WordNet Mappings

| File | Purpose | Mappings | Status |
|------|---------|----------|--------|
| **WordNetMappings30-AI-symmetry.txt** | AI symmetry mappings | 184 | ✅ Complete |
| **WordNetMappings30-Persona.txt** | Persona mappings | ? | ✅ Complete |

#### Concept Sets

| File | Concepts | Purpose | Status |
|------|----------|---------|--------|
| **wordnet_v2_top10.json** | 10 | Testing | ✅ Complete |
| **wordnet_v2_top100.json** | 100 | Phase 2 validation | ✅ Complete |
| **wordnet_v2_top1000.json** | 1,000 | Phase 2 scale test | ✅ Complete |
| **wordnet_v2_top10000.json** | 10,000 | Production target | ✅ Complete |
| **safety_concepts_1k.json** | 1,000 | Safety-ranked | ✅ Complete |
| **safety_concepts_10k.json** | 10,000 | Safety-ranked | ✅ Complete |
| **safety_ranked_concepts.json** | ? | Full safety ranking | ✅ Complete |

#### Backups

| Directory | Purpose | Status |
|-----------|---------|--------|
| **abstraction_layers/backups/** | Layer backups (timestamped) | ✅ Multiple versions |

---

## Missing Documentation

### Critical Gaps (Production Systems)

1. **src/steering/steering_manager.py**
   - **Impact**: High - Main steering API
   - **Status**: Completely undocumented
   - **Needed**: API reference, usage examples, hook management details

2. **src/utils/gpu_cleanup.py**
   - **Impact**: Medium - Memory management
   - **Status**: Undocumented
   - **Needed**: When to use, how it works, integration points

3. **src/encyclopedia/concept_graph.py**
   - **Impact**: Medium - Core data structure
   - **Status**: Undocumented
   - **Needed**: API reference, data model, usage patterns

4. **src/activation_capture/model_loader.py**
   - **Impact**: Medium - Model loading utilities
   - **Status**: Undocumented
   - **Needed**: Model loading patterns, caching, multi-model support

### Moderate Gaps (OpenWebUI Integration)

5. **src/openwebui/hatcat_filter.py**
   - **Impact**: Medium - Content filtering
   - **Status**: Undocumented
   - **Needed**: Filtering logic, concept thresholds, integration

6. **src/openwebui/hatcat_pipe.py**
   - **Impact**: Medium - Pipeline alternative
   - **Status**: Undocumented
   - **Needed**: Difference from hatcat_pipeline.py, use cases

### Moderate Gaps (Visualization)

7. **src/visualization/concept_colors.py**
   - **Impact**: Low-Medium - Visualization
   - **Status**: Module undocumented
   - **Needed**: Color mapping scheme, customization

8. **All visualization scripts** (sunburst, concept viz)
   - **Impact**: Low - Not core functionality
   - **Status**: Undocumented
   - **Needed**: Usage examples, output formats

### Low Priority Gaps (Analysis Scripts)

Many analysis and diagnostic scripts lack documentation:
- analyze_training_log.py
- analyze_calibration_cost.py
- analyze_stability.py
- diagnose_* scripts (lens, centroid, object negatives)
- Various test_* scripts

**Impact**: Low - These are development tools, not production systems
**Recommendation**: Add inline docstrings and --help text, but formal docs not critical

---

## Underappreciated Infrastructure

These systems are more sophisticated and important than they appear:

### 1. Concept Pack System (src/registry/)

**What it looks like**: Simple file bundling system

**What it actually is**: Sophisticated modular ontology management framework
- Version-controlled taxonomy extensions
- Dependency management between packs
- Automated installation with validation
- Backup and rollback support
- Distribution packaging
- Lens pack integration

**Documentation**: Excellent (CONCEPT_PACK_FORMAT.md, CONCEPT_PACK_WORKFLOW.md)

**Impact**: Enables community contributions, domain-specific extensions, and modular deployment

---

### 2. Tiered Validation System (src/training/lens_validation.py)

**What it looks like**: Simple test accuracy check

**What it actually is**: Adaptive calibration-based quality assurance framework
- Progressive strictness falloff (A → B+ → B → C+)
- Advisory vs. blocking modes
- Out-of-distribution testing
- Calibration scoring
- 70% efficiency gain

**Documentation**: Excellent (TIERED_VALIDATION_SYSTEM.md)

**Impact**: Makes production-scale training (5,583 concepts) feasible in 8 hours

---

### 3. Dynamic Lens Manager (src/monitoring/dynamic_lens_manager.py)

**What it looks like**: Simple lens loading

**What it actually is**: Hierarchical adaptive compute allocation system
- Cascade activation (Layer 0 → Layer 1 → ... → Layer 5)
- Memory-efficient loading (1K of 110K+ concepts)
- Parent-triggered child loading
- Cold branch unloading
- Access pattern tracking

**Documentation**: Good (dual_lens_dynamic_loading.md, cascade_profiling_and_optimization.md)

**Impact**: Enables 110K+ concept monitoring with 1K lens budget

---

### 4. Dual-Subspace Manifold Steering (src/steering/manifold.py)

**What it looks like**: Another steering method

**What it actually is**: Research-grade geometric steering framework
- Contamination subspace removal (PCA)
- Task manifold projection (Huang et al.)
- Layer-wise dampening
- Concept preservation tuning
- 100% coherence at ±2.0 strength (vs. 33% baseline)

**Documentation**: Excellent (manifold_steering_analysis.md, detached_jacobian_approach.md)

**Impact**: Doubles working range, prevents model collapse, enables high-magnitude steering

---

### 5. WordNet Patch System (src/data/wordnet_patch_loader.py)

**What it looks like**: JSON file loading

**What it actually is**: Version-controlled ontology extension framework
- Fills gaps in WordNet (e.g., noun.motive had 0/42 synsets)
- Custom relationship types
- Bidirectional linking
- Patch validation
- Multiple patch composition

**Documentation**: Excellent (WORDNET_PATCH_SYSTEM.md)

**Impact**: Achieved 100% synset coverage (5,582/5,582 concepts), enables custom taxonomies

---

### 6. Adaptive Training with Relationship Scheduling (src/training/dual_adaptive_trainer.py)

**What it looks like**: Simple loop with sample increments

**What it actually is**: Multi-axis adaptive training framework
- Independent lens graduation (activation vs. text)
- Relationship importance ranking
- Progressive sample scaling (10 → 30 → 60 → 90)
- Tiered validation integration
- 70% efficiency gain

**Documentation**: Good (adaptive_training_approach.md, TIERED_VALIDATION_SYSTEM.md)

**Impact**: Trained 5,583 concepts in ~8 hours with 95%+ F1 scores

---

### 7. SUMO-WordNet Integration Pipeline (src/build_sumo_wordnet_layers*.py)

**What it looks like**: Data processing scripts

**What it actually is**: Multi-stage hierarchical ontology construction pipeline
- KIF parsing (authoritative Merge.kif)
- 105,042 WordNet→SUMO mappings (79% coverage)
- 6-layer hierarchical assignment
- Pseudo-SUMO intermediate generation
- Custom taxonomy integration
- Achieved 88.7% of WordNet coverage (73,754 concepts)

**Documentation**: Good (sumo_wordnet_hierarchy.md, v5_hyponym_intermediates_plan.md)

**Impact**: Created production-ready hierarchical ontology from research sources

---

### 8. Three-Pole Simplex Architecture (In Progress)

**What it looks like**: Concept organization

**What it actually is**: Novel architecture for AI interoception and homeostasis
- Stable attractor states (μ0) vs. binary poles
- Metabolically sustainable neutral states
- Distributional balance requirements
- Prevents downward spiral bias
- Spline geometry for safe traversal

**Documentation**: Excellent (ai_psychology_homeostasis_expansion.md, distributional_balance_requirement.md)

**Impact**: Enables self-referential AI systems with sustainable operation baselines

---

## Documentation Quality Summary

### Excellent Documentation (Well-Covered)

- Core training systems (SUMO, adaptive, tiered validation)
- Concept pack and lens pack infrastructure
- Manifold steering and geometric approaches
- WordNet patch system
- OpenWebUI integration (setup and roadmap)
- Three-pole simplex architecture
- SUMO-WordNet hierarchy construction

### Good Documentation (Adequate Coverage)

- Temporal monitoring
- Dual-lens architecture
- Cascade optimization
- Phase experimental results
- Ontology coverage analysis

### Moderate Documentation (Needs Improvement)

- Individual build script internals
- OpenWebUI module implementations (filter, pipe)
- Visualization infrastructure
- Model loading utilities

### Poor Documentation (Major Gaps)

- **steering_manager.py** (production API)
- **concept_graph.py** (core data structure)
- **gpu_cleanup.py** (memory management)
- Many analysis and diagnostic scripts

---

## Recommendations

### Priority 1: Production API Documentation

1. **steering_manager.py**: Create API reference with usage examples
2. **concept_graph.py**: Document data model and access patterns
3. **gpu_cleanup.py**: Document when/how to use, integration points

### Priority 2: Module Consistency

1. Add docstrings to all undocumented modules
2. Create module-level README.md files for src/ subdirectories
3. Standardize inline documentation format

### Priority 3: Script Documentation

1. Add --help text to all scripts
2. Create scripts/README.md with script categorization
3. Add usage examples to top of each script

### Priority 4: Architecture Overview

1. Create docs/ARCHITECTURE_OVERVIEW.md showing how all components fit together
2. Include data flow diagrams
3. Document key design decisions and tradeoffs

### Priority 5: Maintenance

1. Update QUICKSTART.md for Phase 14
2. Update DEPLOYMENT.md for current infrastructure
3. Update OPTIMIZATION_STATUS.md
4. Create docs/CHANGELOG.md for version tracking

---

## Summary Statistics

### Code Organization
- **13 src/ subdirectories** with clear separation of concerns
- **120+ scripts** organized by function
- **111 documentation files** with comprehensive coverage
- **73,754 concepts** in production hierarchy

### Documentation Coverage by Category
- **Training**: 95% documented
- **Monitoring**: 90% documented
- **Steering**: 85% documented (missing steering_manager.py)
- **Ontology**: 95% documented
- **OpenWebUI**: 80% documented (missing module internals)
- **Registry**: 100% documented
- **Utilities**: 60% documented (missing gpu_cleanup, model_loader)
- **Scripts**: 40% documented (development tools)

### Biggest Documentation Gaps
1. **steering_manager.py** - Main steering API
2. **Analysis scripts** - 30+ undocumented diagnostic tools
3. **Visualization system** - Module and scripts undocumented
4. **Utility modules** - gpu_cleanup, model_loader
5. **OpenWebUI modules** - filter and pipe implementations

### Most Underappreciated Systems
1. **Concept Pack System** - Sophisticated modular ontology framework
2. **Tiered Validation** - 70% efficiency gain through adaptive quality control
3. **Dynamic Lens Manager** - 110K concepts with 1K lens budget
4. **Dual-Subspace Steering** - Research-grade geometric steering
5. **WordNet Patch System** - Achieved 100% synset coverage
6. **Three-Pole Simplex** - Novel architecture for AI homeostasis

---

**Last Updated**: 2025-11-16
**Repository Version**: Phase 14 complete, Three-Pole Simplex in progress
**Maintainer**: HatCat Team
