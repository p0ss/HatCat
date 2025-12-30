"""
Bootstrap - BE instantiation and lifecycle management.

This module provides:
1. ToolGraft - Task tuning tools as substrate grafts
2. BootstrapArtifact - Complete package for waking a BE
3. UpliftTaxonomy - Graph of concepts across facets (MAP, CAT, HAT, HUSH, etc.)
4. MeldSubmission - Unified format for lens+graft training data
5. wake_be - The wake sequence for instantiating a BE

The key insight: EVERY concept the BE needs must be grafted in, each requiring:
- A lens (to detect when the concept is active)
- A graft (to add a dedicated neuron)
- A Meld submission (to go through approval)

Facets are NOT hierarchical layers - they're interconnected dimensions:
- MAP: Ontological grounding (what kinds of things exist)
- CAT: Cognitive architecture (how thinking works)
- HAT: Experiential substrate (what it's like to be)
- HUSH: Governance and safety (boundaries and rules)
- TOOLS: Workspace capabilities (what BE can do)
- TRIBE: Philosophy and values (why BE does things)
- MELD: Evolution protocol (how BE grows)
- ASK: Authority structure (who decides)

Usage:
    from src.be.bootstrap import wake_be, BootstrapArtifact

    # Load and wake a BE from an artifact
    be = wake_be("path/to/artifact", device="cuda")

    # Use the BE
    response = be.generate("Hello, world!")

    # Shutdown when done
    be.shutdown()

Creating uplift batches:
    from src.be.bootstrap import (
        build_base_taxonomy,
        create_uplift_batch,
        MeldBatch,
    )

    # Create taxonomy and batch for approval
    taxonomy = build_base_taxonomy()
    batch = create_uplift_batch(taxonomy, tribe_id="my-tribe")
    batch.save("path/to/output")

See docs/specification/BE_BOOTSTRAP.md for full specification.
"""

from .artifact import (
    BootstrapArtifact,
    ArtifactComponent,
    SubstrateBundle,
    ConceptPack,
    LensPack,
    USHProfile,
    UpliftRecord,
    XDBBootstrap,
    LifecycleContract,
    ToolPackComponent,
    create_artifact_from_components,
)

from .tool_graft import (
    ToolGraft,
    ToolGraftPack,
    ToolGraftTrainer,
    ToolSchema,
    WORKSPACE_TOOL_SCHEMAS,
    TIER_CLEFT_SOURCES,
    create_standard_tool_pack,
)

from .uplift_taxonomy import (
    GraftFacet,
    UpliftConcept,
    TribePhilosophy,
    UpliftTaxonomy,
    build_base_taxonomy,
    count_total_grafts,
    # Inter-facet relationships
    FacetRelation,
    FACET_RELATIONS,
    get_related_facets,
    # Facet concept lists
    MAP_CONCEPTS,
    CAT_CONCEPTS,
    HAT_CONCEPTS,
    HUSH_CONCEPTS,
    TRIBE_CONCEPTS,
    ASK_CONCEPTS,
    MELD_CONCEPTS,
)

from .meld_format import (
    MeldSubmission,
    MeldBatch,
    TrainingExample,
    ContrastivePair,
    EvidenceItem,
    SubmissionStatus,
    EvidenceType,
    create_submission_from_uplift_concept,
    create_uplift_batch,
)

from .wake import (
    wake_be,
    WakeSequence,
    WakeError,
    BoundedExperiencer,
)


__all__ = [
    # Artifact
    'BootstrapArtifact',
    'ArtifactComponent',
    'SubstrateBundle',
    'ConceptPack',
    'LensPack',
    'USHProfile',
    'UpliftRecord',
    'XDBBootstrap',
    'LifecycleContract',
    'ToolPackComponent',
    'create_artifact_from_components',

    # ToolGraft
    'ToolGraft',
    'ToolGraftPack',
    'ToolGraftTrainer',
    'ToolSchema',
    'WORKSPACE_TOOL_SCHEMAS',
    'TIER_CLEFT_SOURCES',
    'create_standard_tool_pack',

    # Uplift Taxonomy
    'GraftFacet',
    'UpliftConcept',
    'TribePhilosophy',
    'UpliftTaxonomy',
    'build_base_taxonomy',
    'count_total_grafts',
    # Inter-facet relationships
    'FacetRelation',
    'FACET_RELATIONS',
    'get_related_facets',
    # Facet concept lists
    'MAP_CONCEPTS',
    'CAT_CONCEPTS',
    'HAT_CONCEPTS',
    'HUSH_CONCEPTS',
    'TRIBE_CONCEPTS',
    'ASK_CONCEPTS',
    'MELD_CONCEPTS',

    # Meld Format
    'MeldSubmission',
    'MeldBatch',
    'TrainingExample',
    'ContrastivePair',
    'EvidenceItem',
    'SubmissionStatus',
    'EvidenceType',
    'create_submission_from_uplift_concept',
    'create_uplift_batch',

    # Wake
    'wake_be',
    'WakeSequence',
    'WakeError',
    'BoundedExperiencer',
]
