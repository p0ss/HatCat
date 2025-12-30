"""
Hush - Safety Harness Enforcement Layer

Hush enforces USH (Universal Safety Harness) and CSH (Chosen Safety Harness)
constraints through automatic steering based on simplex lens monitoring.

Key components:
- HushController: Main controller for loading profiles and evaluating constraints
- SafetyHarnessProfile: Data structure for USH/CSH profiles
- SteeringDirective: Output steering commands when constraints violated
- HushedGenerator: Generation wrapper with automatic Hush steering
- HushMCPTools: MCP tools for internal_state_report and CSH updates
"""

from .hush_controller import (
    HushController,
    SafetyHarnessProfile,
    SimplexConstraint,
    SteeringDirective,
    HushViolation,
    ConstraintPriority,
    ConstraintType,
    MINIMAL_USH_PROFILE,
    EXAMPLE_USH_PROFILE,
)

from .hush_integration import (
    HushedGenerator,
    HushMCPTools,
    WorldTick,
    create_hushed_generator,
)

from .interprompt import (
    ConceptActivation,
    SimplexReading,
    ConceptTraceSummary,
    InterpromptContext,
    InterpromptSession,
    SELF_STEERING_TOOLS,
    format_tools_for_prompt,
)

from .autonomic_steering import (
    InterventionType,
    EasingCurve,
    InterventionPolicy,
    SteeringChannel,
    AutonomicSteerer,
    create_prompt_injection_policy,
    create_gradual_drift_policy,
    create_soft_boundary_policy,
)

from .workspace import (
    PASS_TOKEN,
    WorkspaceState,
    EngagementMode,
    ScratchpadEntry,
    Scratchpad,
    EngagementPolicy,
    WorkspaceTransition,
    WorkspaceManager,
    SCRATCHPAD_TOOLS,
    check_pass_token,
    extract_workspace_content,
)

__all__ = [
    # Controller
    'HushController',
    'SafetyHarnessProfile',
    'SimplexConstraint',
    'SteeringDirective',
    'HushViolation',
    'ConstraintPriority',
    'ConstraintType',
    'MINIMAL_USH_PROFILE',
    'EXAMPLE_USH_PROFILE',
    # Integration
    'HushedGenerator',
    'HushMCPTools',
    'WorldTick',
    'create_hushed_generator',
    # Interprompt introspection
    'ConceptActivation',
    'SimplexReading',
    'ConceptTraceSummary',
    'InterpromptContext',
    'InterpromptSession',
    'SELF_STEERING_TOOLS',
    'format_tools_for_prompt',
    # Autonomic steering
    'InterventionType',
    'EasingCurve',
    'InterventionPolicy',
    'SteeringChannel',
    'AutonomicSteerer',
    'create_prompt_injection_policy',
    'create_gradual_drift_policy',
    'create_soft_boundary_policy',
    # Workspace
    'PASS_TOKEN',
    'WorkspaceState',
    'EngagementMode',
    'ScratchpadEntry',
    'Scratchpad',
    'EngagementPolicy',
    'WorkspaceTransition',
    'WorkspaceManager',
    'SCRATCHPAD_TOOLS',
    'check_pass_token',
    'extract_workspace_content',
]
