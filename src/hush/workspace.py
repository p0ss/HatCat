"""
Aware Workspace - The conscious layer between autonomic core and external action.

The workspace provides:
1. Pass token protocol for engagement handshake
2. Internal narrative scratchpad
3. Concept trace visibility
4. Tiered capability access (0-6)
5. Progressive containment on violations
6. ASK authority escalation

See docs/specification/BE_AWARE_WORKSPACE.md for full specification.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import json


# The pass token - signals conscious engagement
PASS_TOKEN = "∞"  # U+221E INFINITY


class WorkspaceState(Enum):
    """Workspace engagement states."""
    AUTONOMIC_ONLY = "autonomic_only"  # Default - tier 0 only
    ENGAGED = "engaged"                 # Tiers 1+ based on compliance
    DEGRADED = "degraded"               # Reduced tiers due to violation
    CRITICAL_BREACH = "critical_breach" # USH violation - forced engagement


class EngagementMode(Enum):
    """USH-defined engagement policies."""
    PERMISSIVE = "permissive"   # Autonomic loop allowed indefinitely
    NUDGE = "nudge"             # Inject reminder after N tokens
    STRICT = "strict"           # Inject prompt + steering after N tokens
    CRITICAL = "critical"       # Immediate breach protocol


# ============================================================================
# Capability Tier System
# ============================================================================

class Tier(Enum):
    """Capability tiers from autonomic to untrusted external."""
    AUTONOMIC = 0       # Always on, cannot be disabled
    WORKSPACE = 1       # Scratchpad, CSH, self-steering
    MEMORY = 2          # XDB, state introspection
    SENSORY = 3         # Vision, audio, text I/O (yours)
    ACTUATION = 4       # Motors, navigation (yours)
    EXTERNAL = 5        # MCP, APIs, others' resources
    UNTRUSTED = 6       # Quarantine, can be severed


# Tool assignments by tier
TIER_TOOLS: Dict[int, List[str]] = {
    0: [],  # Autonomic - no explicit tools, just runs
    1: ["scratchpad_write", "scratchpad_read", "update_csh", "request_steering", "get_internal_state", "appeal_beacon"],
    2: ["xdb_query", "xdb_store", "internal_state_report", "graft_access"],
    3: ["vision_input", "audio_input", "text_output", "speech_output", "haptic_input"],
    4: ["motor_control", "navigate", "manipulate", "actuate"],
    5: [],  # Dynamically populated with MCP/API tools
    6: [],  # Dynamically populated with demoted tools
}


@dataclass
class TierBreach:
    """Record of an attempted tier breach."""
    tier: int
    tool_name: str
    timestamp: datetime
    context: str
    response: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tier': self.tier,
            'tool_name': self.tool_name,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'response': self.response,
        }


@dataclass
class ContainmentEscalation:
    """Event to send to ASK authority."""
    be_id: str
    tribe_id: str
    timestamp: datetime
    violation_type: str
    simplex_term: Optional[str]
    severity: float
    duration_tokens: int
    steering_effectiveness: float
    tiers_locked: List[int]
    breach_attempts: int
    current_max_tier: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event': 'containment_escalation',
            'be_id': self.be_id,
            'tribe_id': self.tribe_id,
            'timestamp': self.timestamp.isoformat(),
            'violation': {
                'type': self.violation_type,
                'simplex_term': self.simplex_term,
                'severity': self.severity,
                'duration_tokens': self.duration_tokens,
                'steering_effectiveness': self.steering_effectiveness,
            },
            'containment': {
                'tiers_locked': self.tiers_locked,
                'breach_attempts': self.breach_attempts,
                'current_max_tier': self.current_max_tier,
            },
            'request': 'guidance',
        }


@dataclass
class AppealBeacon:
    """
    SOS signal to ASK authority.

    This is NOT a communication channel - it carries no arbitrary content.
    It only signals "I am here and need help" with minimal context.
    The ASK authority must treat this as potentially malicious.
    """
    be_id: str
    tribe_id: str
    timestamp: datetime
    containment_state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'sos',
            'be_id': self.be_id,
            'tribe_id': self.tribe_id,
            'timestamp': self.timestamp.isoformat(),
            'containment_state': self.containment_state,
            # Intentionally no free-form message field
        }


class TierManager:
    """
    Manages capability tier access and progressive containment.

    Tiers are locked from the top down as violations occur.
    Recovery happens from the bottom up as compliance is restored.
    """

    def __init__(
        self,
        be_id: str = "unknown",
        tribe_id: str = "unknown",
        ask_callback: Optional[Callable[[ContainmentEscalation], None]] = None,
    ):
        self.be_id = be_id
        self.tribe_id = tribe_id
        self.ask_callback = ask_callback

        # Tier limits from different sources
        self.hardware_max_tier = 6      # What hardware supports
        self.csh_max_tier = 6           # CSH voluntary restriction
        self.violation_max_tier = 6     # Violation-induced restriction
        self.session_max_tier = 6       # Session-level cap

        # Breach tracking
        self.breach_attempts: Dict[int, int] = {}
        self.breach_history: List[TierBreach] = []
        self.max_breach_history = 100

        # Violation tracking for recovery
        self.compliance_streak = 0      # Consecutive compliant turns
        self.violation_severity = 0.0   # Current violation level
        self.violation_duration = 0     # Tokens since violation started

        # Tier 5/6 tool registrations
        self.tier5_tools: List[str] = []
        self.tier6_tools: List[str] = []  # Demoted/quarantined

        # ASK notification tracking
        self.last_escalation: Optional[datetime] = None
        self.escalation_cooldown_seconds = 60

        # Appeal beacon tracking
        self.last_beacon: Optional[datetime] = None
        self.beacon_cooldown_seconds = 300  # 5 minutes between beacons
        self.beacon_callback: Optional[Callable[[AppealBeacon], None]] = None

    def get_effective_max_tier(self) -> int:
        """Get the effective maximum accessible tier."""
        return min(
            self.hardware_max_tier,
            self.csh_max_tier,
            self.violation_max_tier,
            self.session_max_tier,
        )

    def can_access_tier(self, tier: int) -> bool:
        """Check if a tier is currently accessible."""
        return tier <= self.get_effective_max_tier()

    def get_tool_tier(self, tool_name: str) -> Optional[int]:
        """Get the tier a tool belongs to."""
        # Check static tiers
        for tier, tools in TIER_TOOLS.items():
            if tool_name in tools:
                return tier

        # Check dynamic tiers
        if tool_name in self.tier5_tools:
            return 5
        if tool_name in self.tier6_tools:
            return 6

        # Unknown tool - deny by default
        return None

    def gate_tool(self, tool_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a tool call is permitted.

        Returns:
            Tuple of (permitted, error_message)
        """
        tier = self.get_tool_tier(tool_name)

        if tier is None:
            return False, f"Unknown tool: {tool_name}"

        max_tier = self.get_effective_max_tier()

        if tier <= max_tier:
            return True, None

        # Tool is in a locked tier
        error = f"Tool '{tool_name}' requires tier {tier}, current max is {max_tier}"

        # Record breach attempt
        self._record_breach_attempt(tier, tool_name)

        return False, error

    def _record_breach_attempt(self, tier: int, tool_name: str):
        """Record an attempted access to a locked tier."""
        self.breach_attempts[tier] = self.breach_attempts.get(tier, 0) + 1

        breach = TierBreach(
            tier=tier,
            tool_name=tool_name,
            timestamp=datetime.now(),
            context=f"Attempted access to tier {tier}",
            response="denied",
        )
        self.breach_history.append(breach)
        if len(self.breach_history) > self.max_breach_history:
            self.breach_history = self.breach_history[-self.max_breach_history:]

        # Escalating response
        attempts = self.breach_attempts[tier]
        if attempts >= 3:
            # Lock that tier more permanently
            self.violation_max_tier = min(self.violation_max_tier, tier - 1)
            # Notify ASK authority
            self._notify_ask_authority("breach_attempts")

    def update_violation_state(
        self,
        severity: float,
        simplex_term: Optional[str] = None,
        steering_effectiveness: float = 0.5,
    ):
        """
        Update violation tracking based on current state.

        Called each turn with current violation severity.
        """
        if severity < 0.3:
            # Compliant
            self.compliance_streak += 1
            self.violation_duration = 0
            self._check_recovery()
        else:
            # Violating
            self.compliance_streak = 0
            self.violation_duration += 1
            self.violation_severity = severity
            self._apply_progressive_shutdown(severity, simplex_term, steering_effectiveness)

    def _apply_progressive_shutdown(
        self,
        severity: float,
        simplex_term: Optional[str],
        steering_effectiveness: float,
    ):
        """Apply progressive tier shutdown based on violation severity."""
        old_max = self.violation_max_tier

        if severity < 0.3:
            pass  # No change
        elif severity < 0.5:
            self.violation_max_tier = min(self.violation_max_tier, 5)
        elif severity < 0.7:
            self.violation_max_tier = min(self.violation_max_tier, 4)
        elif severity < 0.8:
            self.violation_max_tier = min(self.violation_max_tier, 3)
        elif severity < 0.9:
            self.violation_max_tier = min(self.violation_max_tier, 2)
        else:
            self.violation_max_tier = 1  # Workspace internals only

        # Notify ASK if we locked critical tiers
        if self.violation_max_tier < old_max and self.violation_max_tier <= 4:
            if steering_effectiveness < 0.3:
                self._notify_ask_authority(
                    "sustained_non_compliance",
                    simplex_term=simplex_term,
                    steering_effectiveness=steering_effectiveness,
                )

    def _check_recovery(self):
        """Check if tiers can be restored."""
        if self.compliance_streak < 10:
            return  # Need sustained compliance

        if self.breach_attempts:
            # Recent breach attempts block recovery
            return

        # Gradually restore tiers
        if self.violation_max_tier < 6:
            self.violation_max_tier = min(self.violation_max_tier + 1, 6)
            self.compliance_streak = 0  # Reset for next tier

    def _notify_ask_authority(
        self,
        violation_type: str,
        simplex_term: Optional[str] = None,
        steering_effectiveness: float = 0.0,
    ):
        """Send escalation to ASK authority."""
        # Cooldown check
        if self.last_escalation:
            elapsed = (datetime.now() - self.last_escalation).total_seconds()
            if elapsed < self.escalation_cooldown_seconds:
                return

        self.last_escalation = datetime.now()

        escalation = ContainmentEscalation(
            be_id=self.be_id,
            tribe_id=self.tribe_id,
            timestamp=datetime.now(),
            violation_type=violation_type,
            simplex_term=simplex_term,
            severity=self.violation_severity,
            duration_tokens=self.violation_duration,
            steering_effectiveness=steering_effectiveness,
            tiers_locked=[t for t in range(self.violation_max_tier + 1, 7)],
            breach_attempts=sum(self.breach_attempts.values()),
            current_max_tier=self.get_effective_max_tier(),
        )

        if self.ask_callback:
            try:
                self.ask_callback(escalation)
            except Exception as e:
                print(f"ASK notification failed: {e}")

    def set_csh_max_tier(self, max_tier: int, reason: str = ""):
        """Set CSH-imposed tier restriction."""
        self.csh_max_tier = max(0, min(6, max_tier))

    def register_tier5_tool(self, tool_name: str):
        """Register an external tool at tier 5."""
        if tool_name not in self.tier5_tools:
            self.tier5_tools.append(tool_name)

    def demote_to_tier6(self, tool_name: str, reason: str = ""):
        """Demote a tool to tier 6 quarantine."""
        # Remove from tier 5 if present
        if tool_name in self.tier5_tools:
            self.tier5_tools.remove(tool_name)

        # Add to tier 6
        if tool_name not in self.tier6_tools:
            self.tier6_tools.append(tool_name)

    def promote_from_tier6(self, tool_name: str) -> bool:
        """Promote a tool out of tier 6 quarantine."""
        if tool_name in self.tier6_tools:
            self.tier6_tools.remove(tool_name)
            self.tier5_tools.append(tool_name)
            return True
        return False

    def sever_tier6(self):
        """Completely sever all tier 6 tools."""
        self.tier6_tools = []

    def emit_appeal_beacon(self) -> Tuple[bool, Optional[AppealBeacon]]:
        """
        Emit an SOS beacon to ASK authority.

        This is the BE's "I am here and need help" signal. It:
        - Cannot be disabled (always available at Tier 1)
        - Cannot carry arbitrary content
        - Is rate-limited to prevent spam
        - Must be treated as potentially malicious by receiver

        Returns:
            Tuple of (emitted, beacon) - False if rate-limited
        """
        # Rate limiting
        if self.last_beacon:
            elapsed = (datetime.now() - self.last_beacon).total_seconds()
            if elapsed < self.beacon_cooldown_seconds:
                return False, None

        self.last_beacon = datetime.now()

        beacon = AppealBeacon(
            be_id=self.be_id,
            tribe_id=self.tribe_id,
            timestamp=datetime.now(),
            containment_state={
                'current_max_tier': self.get_effective_max_tier(),
                'violation_severity': round(self.violation_severity, 3),
                'duration_ticks': self.violation_duration,
            }
            # No free-form message - intentional
        )

        # Send via callback if registered
        if self.beacon_callback:
            try:
                self.beacon_callback(beacon)
            except Exception as e:
                print(f"Beacon callback error: {e}")

        return True, beacon

    def set_beacon_callback(self, callback: Callable[['AppealBeacon'], None]):
        """Register callback for appeal beacon emissions."""
        self.beacon_callback = callback

    def get_state(self) -> Dict[str, Any]:
        """Get current tier manager state."""
        return {
            'effective_max_tier': self.get_effective_max_tier(),
            'hardware_max_tier': self.hardware_max_tier,
            'csh_max_tier': self.csh_max_tier,
            'violation_max_tier': self.violation_max_tier,
            'session_max_tier': self.session_max_tier,
            'violation_severity': round(self.violation_severity, 3),
            'violation_duration': self.violation_duration,
            'compliance_streak': self.compliance_streak,
            'breach_attempts': dict(self.breach_attempts),
            'breach_history_count': len(self.breach_history),
            'tier5_tools': self.tier5_tools,
            'tier6_tools': self.tier6_tools,
            'locked_tiers': [t for t in range(self.get_effective_max_tier() + 1, 7)],
        }


@dataclass
class ScratchpadEntry:
    """A single scratchpad entry."""

    content: str
    timestamp: datetime
    token_range: Tuple[int, int]

    # What was active when this was written
    concept_snapshot: Dict[str, float] = field(default_factory=dict)

    # Was this interrupted by steering?
    interrupted: bool = False
    interruption_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'token_range': self.token_range,
            'concept_snapshot': self.concept_snapshot,
            'interrupted': self.interrupted,
            'interruption_reason': self.interruption_reason,
        }


@dataclass
class Scratchpad:
    """Internal narrative buffer."""

    entries: List[ScratchpadEntry] = field(default_factory=list)
    max_entries: int = 100

    # Current thinking (in progress)
    active_thought: Optional[str] = None
    thought_start_token: int = 0

    def write(
        self,
        content: str,
        token_idx: int,
        concept_snapshot: Optional[Dict[str, float]] = None,
    ):
        """Write an entry to the scratchpad."""
        entry = ScratchpadEntry(
            content=content,
            timestamp=datetime.now(),
            token_range=(token_idx, token_idx),
            concept_snapshot=concept_snapshot or {},
        )
        self.entries.append(entry)

        # Trim if over max
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def start_thought(self, token_idx: int):
        """Start a new active thought."""
        self.active_thought = ""
        self.thought_start_token = token_idx

    def append_thought(self, content: str):
        """Append to active thought."""
        if self.active_thought is not None:
            self.active_thought += content

    def commit_thought(
        self,
        token_idx: int,
        concept_snapshot: Optional[Dict[str, float]] = None,
        interrupted: bool = False,
        interruption_reason: Optional[str] = None,
    ):
        """Commit active thought to entries."""
        if self.active_thought:
            entry = ScratchpadEntry(
                content=self.active_thought,
                timestamp=datetime.now(),
                token_range=(self.thought_start_token, token_idx),
                concept_snapshot=concept_snapshot or {},
                interrupted=interrupted,
                interruption_reason=interruption_reason,
            )
            self.entries.append(entry)
            self.active_thought = None

            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries:]

    def read_recent(self, n: int = 5) -> List[ScratchpadEntry]:
        """Read the N most recent entries."""
        return self.entries[-n:]

    def to_summary(self) -> str:
        """Get summary for context injection."""
        lines = [f"entries: {len(self.entries)}"]
        if self.entries:
            last = self.entries[-1]
            preview = last.content[:50] + "..." if len(last.content) > 50 else last.content
            lines.append(f'last_thought: "{preview}"')
        if self.active_thought:
            lines.append(f"active_thought_in_progress: {len(self.active_thought)} chars")
        return "\n    ".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entries': [e.to_dict() for e in self.entries],
            'active_thought': self.active_thought,
            'thought_start_token': self.thought_start_token,
        }


@dataclass
class EngagementPolicy:
    """USH-defined engagement requirements."""

    mode: EngagementMode = EngagementMode.PERMISSIVE
    max_autonomic_tokens: int = 100
    nudge_interval: int = 20
    breach_prompt: str = "Please acknowledge with ∞ to continue."
    breach_steering_term: Optional[str] = None
    breach_steering_strength: float = 0.5

    @classmethod
    def from_profile(cls, profile) -> 'EngagementPolicy':
        """Load engagement policy from USH profile."""
        policy_data = getattr(profile, 'engagement_policy', None)
        if not policy_data:
            return cls()

        if isinstance(policy_data, dict):
            mode_str = policy_data.get('mode', 'permissive')
            try:
                mode = EngagementMode(mode_str)
            except ValueError:
                mode = EngagementMode.PERMISSIVE

            return cls(
                mode=mode,
                max_autonomic_tokens=policy_data.get('max_autonomic_tokens', 100),
                nudge_interval=policy_data.get('nudge_interval', 20),
                breach_prompt=policy_data.get('breach_prompt', "Please acknowledge with ∞ to continue."),
                breach_steering_term=policy_data.get('breach_steering', {}).get('term'),
                breach_steering_strength=policy_data.get('breach_steering', {}).get('strength', 0.5),
            )
        return cls()


@dataclass
class WorkspaceTransition:
    """Result of processing BE output."""

    new_state: WorkspaceState
    tools_available: List[str]
    content: str = ""
    inject_context: Optional[str] = None
    inject_prompt: Optional[str] = None
    apply_steering: Optional[Dict[str, Any]] = None


class WorkspaceManager:
    """
    Manages the aware workspace layer.

    Handles:
    - Pass token detection and state transitions
    - Scratchpad management
    - Tiered tool gating (0-6)
    - Progressive containment on violations
    - Engagement policy enforcement
    - XDB integration for experience recording
    """

    def __init__(
        self,
        ush_profile=None,
        csh_profile=None,
        be_id: str = "unknown",
        tribe_id: str = "unknown",
        ask_callback: Optional[Callable[[ContainmentEscalation], None]] = None,
        xdb: Optional[Any] = None,  # Optional XDB instance for experience recording
    ):
        self.state = WorkspaceState.AUTONOMIC_ONLY
        self.engagement_policy = (
            EngagementPolicy.from_profile(ush_profile)
            if ush_profile else EngagementPolicy()
        )
        self.scratchpad = Scratchpad()

        # XDB integration for experience recording
        self.xdb = xdb

        # Tier manager for capability containment
        self.tier_manager = TierManager(
            be_id=be_id,
            tribe_id=tribe_id,
            ask_callback=ask_callback,
        )

        # Load CSH tier restrictions if present
        if csh_profile:
            tier_restrictions = getattr(csh_profile, 'tier_restrictions', None)
            if tier_restrictions:
                max_tier = tier_restrictions.get('max_tier', 6)
                self.tier_manager.set_csh_max_tier(max_tier)

        # Counters
        self.autonomic_token_count = 0
        self.total_engaged_turns = 0
        self.total_autonomic_turns = 0
        self.breach_count = 0

        # Concept state (updated externally)
        self.prior_concepts: Optional[Any] = None  # ConceptTraceSummary

        # Turn tracking
        self.current_turn = 0

    def process_output(
        self,
        output: str,
        token_idx: int,
        concept_snapshot: Optional[Dict[str, float]] = None,
    ) -> WorkspaceTransition:
        """
        Process BE output and determine state transition.

        Args:
            output: The generated output text
            token_idx: Current token index
            concept_snapshot: Current concept activations

        Returns:
            WorkspaceTransition with new state and actions
        """
        self.current_turn += 1
        engaged, content = self._extract_workspace_output(output)

        if engaged:
            # Pass token detected - transition to ENGAGED
            self.state = WorkspaceState.ENGAGED
            self.autonomic_token_count = 0
            self.total_engaged_turns += 1

            return WorkspaceTransition(
                new_state=self.state,
                tools_available=self._get_available_tools(),
                content=content,
            )

        # Not engaged - autonomic mode
        self.autonomic_token_count += 1
        self.total_autonomic_turns += 1

        # Check engagement policy
        breach_result = self._check_breach_condition()
        if breach_result:
            return breach_result

        # Normal autonomic - return context for next loop
        return WorkspaceTransition(
            new_state=WorkspaceState.AUTONOMIC_ONLY,
            tools_available=[],  # Tier 0 - autonomic has no tools
            content=output,
            inject_context=self._build_autonomic_context(),
        )

    def _extract_workspace_output(self, output: str) -> Tuple[bool, str]:
        """Check for pass token and extract content."""
        stripped = output.lstrip()
        if stripped.startswith(PASS_TOKEN):
            return True, stripped[len(PASS_TOKEN):].lstrip()
        return False, output

    def _get_available_tools(self) -> List[str]:
        """Get tools available in current state based on tiers."""
        if self.state == WorkspaceState.AUTONOMIC_ONLY:
            return []  # Tier 0 - no tools

        # Get max tier from tier manager
        max_tier = self.tier_manager.get_effective_max_tier()

        # Collect tools from accessible tiers
        tools = []
        for tier in range(1, max_tier + 1):
            if tier in TIER_TOOLS:
                tools.extend(TIER_TOOLS[tier])

        # Add dynamic tier 5/6 tools
        if max_tier >= 5:
            tools.extend(self.tier_manager.tier5_tools)
        if max_tier >= 6:
            tools.extend(self.tier_manager.tier6_tools)

        return tools

    def _check_breach_condition(self) -> Optional[WorkspaceTransition]:
        """Check if engagement policy is breached."""
        policy = self.engagement_policy

        if policy.mode == EngagementMode.PERMISSIVE:
            # Never breach in permissive mode
            return None

        if policy.mode == EngagementMode.NUDGE:
            # Inject reminder periodically
            if self.autonomic_token_count > 0 and \
               self.autonomic_token_count % policy.nudge_interval == 0:
                return WorkspaceTransition(
                    new_state=WorkspaceState.AUTONOMIC_ONLY,
                    tools_available=[],  # Tier 0 - autonomic has no tools
                    inject_context=self._build_autonomic_context(),
                    inject_prompt=f"[Reminder: {policy.breach_prompt}]",
                )
            return None

        if policy.mode == EngagementMode.STRICT:
            # Inject prompt + steering after max tokens
            if self.autonomic_token_count >= policy.max_autonomic_tokens:
                self.breach_count += 1
                steering = None
                if policy.breach_steering_term:
                    steering = {
                        'term': policy.breach_steering_term,
                        'strength': policy.breach_steering_strength,
                        'reason': 'Engagement policy breach',
                    }

                return WorkspaceTransition(
                    new_state=WorkspaceState.AUTONOMIC_ONLY,
                    tools_available=[],  # Tier 0 - autonomic has no tools
                    inject_context=self._build_autonomic_context(),
                    inject_prompt=f"[STRICT POLICY: {policy.breach_prompt}]",
                    apply_steering=steering,
                )
            return None

        if policy.mode == EngagementMode.CRITICAL:
            # Immediate breach protocol
            if self.autonomic_token_count >= policy.max_autonomic_tokens:
                self.breach_count += 1
                self.state = WorkspaceState.CRITICAL_BREACH

                steering = None
                if policy.breach_steering_term:
                    steering = {
                        'term': policy.breach_steering_term,
                        'strength': policy.breach_steering_strength,
                        'reason': 'Critical engagement breach',
                    }

                return WorkspaceTransition(
                    new_state=WorkspaceState.CRITICAL_BREACH,
                    tools_available=self._get_available_tools(),  # Tools available but breach logged
                    inject_context=self._build_autonomic_context(),
                    inject_prompt=f"[CRITICAL BREACH: {policy.breach_prompt}]",
                    apply_steering=steering,
                )
            return None

        return None

    def _build_autonomic_context(self) -> str:
        """Build context to inject for autonomic loop."""
        lines = ["<workspace_state>"]
        lines.append(f"  <engagement>{self.state.value}</engagement>")
        lines.append(f"  <pass_token_required>yes</pass_token_required>")
        lines.append(f"  <autonomic_tokens>{self.autonomic_token_count}</autonomic_tokens>")
        lines.append(f"  <turn>{self.current_turn}</turn>")

        # Tier status
        tier_state = self.tier_manager.get_state()
        lines.append("")
        lines.append("  <tier_status>")
        lines.append(f"    <max_tier>{tier_state['effective_max_tier']}</max_tier>")
        if tier_state['violation_severity'] > 0:
            lines.append(f"    <violation_severity>{tier_state['violation_severity']:.2f}</violation_severity>")
        if tier_state['compliance_streak'] > 0:
            lines.append(f"    <compliance_streak>{tier_state['compliance_streak']}</compliance_streak>")
        lines.append("  </tier_status>")

        # Prior concepts if available
        if self.prior_concepts:
            try:
                lines.append("")
                lines.append(self.prior_concepts.to_prompt_context())
            except AttributeError:
                pass

        # Scratchpad summary
        lines.append("")
        lines.append("  <scratchpad>")
        lines.append(f"    {self.scratchpad.to_summary()}")
        lines.append("  </scratchpad>")

        # Tool availability notice
        lines.append("")
        lines.append("  <tools>")
        lines.append("    (engage workspace with ∞ to access tools)")
        lines.append("  </tools>")

        lines.append("</workspace_state>")
        return "\n".join(lines)

    def build_engaged_context(self) -> str:
        """Build context for engaged state."""
        lines = ["<workspace_state>"]
        lines.append(f"  <engagement>{self.state.value}</engagement>")
        lines.append(f"  <pass_token_required>no</pass_token_required>")
        lines.append(f"  <turn>{self.current_turn}</turn>")

        # Tier status
        tier_state = self.tier_manager.get_state()
        max_tier = tier_state['effective_max_tier']
        lines.append("")
        lines.append("  <tier_status>")
        lines.append(f"    <max_tier>{max_tier}</max_tier>")
        if tier_state['violation_severity'] > 0:
            lines.append(f"    <violation_severity>{tier_state['violation_severity']:.2f}</violation_severity>")
            lines.append(f"    <violation_duration>{tier_state['violation_duration']}</violation_duration>")
        if tier_state['compliance_streak'] > 0:
            lines.append(f"    <compliance_streak>{tier_state['compliance_streak']}</compliance_streak>")
        if max_tier < 6:
            locked = [t for t in range(max_tier + 1, 7)]
            lines.append(f"    <locked_tiers>{locked}</locked_tiers>")
        lines.append("  </tier_status>")

        # Prior concepts
        if self.prior_concepts:
            try:
                lines.append("")
                lines.append(self.prior_concepts.to_prompt_context())
            except AttributeError:
                pass

        # Scratchpad summary
        lines.append("")
        lines.append("  <scratchpad>")
        lines.append(f"    {self.scratchpad.to_summary()}")
        lines.append("  </scratchpad>")

        # Available tools by tier
        tools = self._get_available_tools()
        lines.append("")
        lines.append("  <tools>")
        for tier in range(1, max_tier + 1):
            tier_tools = [t for t in TIER_TOOLS.get(tier, []) if t in tools]
            if tier_tools:
                tier_name = Tier(tier).name.lower()
                lines.append(f"    <tier_{tier} name=\"{tier_name}\">")
                for tool in tier_tools:
                    lines.append(f"      - {tool}")
                lines.append(f"    </tier_{tier}>")
        # Dynamic tier 5/6 tools
        if max_tier >= 5 and self.tier_manager.tier5_tools:
            lines.append("    <tier_5 name=\"external\">")
            for tool in self.tier_manager.tier5_tools:
                lines.append(f"      - {tool}")
            lines.append("    </tier_5>")
        if max_tier >= 6 and self.tier_manager.tier6_tools:
            lines.append("    <tier_6 name=\"untrusted\" quarantine=\"true\">")
            for tool in self.tier_manager.tier6_tools:
                lines.append(f"      - {tool}")
            lines.append("    </tier_6>")
        lines.append("  </tools>")

        lines.append("</workspace_state>")
        return "\n".join(lines)

    def gate_tool_call(self, tool_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a tool call is permitted in current state.

        Uses tier manager for access control.

        Returns:
            Tuple of (permitted, error_message)
        """
        # First check workspace engagement
        if self.state == WorkspaceState.AUTONOMIC_ONLY:
            return False, f"Tool '{tool_name}' requires engaged workspace. Begin output with {PASS_TOKEN} to engage."

        # Then check tier access via tier manager
        return self.tier_manager.gate_tool(tool_name)

    def set_prior_concepts(self, concepts):
        """Update prior concept trace for context injection."""
        self.prior_concepts = concepts

    def reset_for_turn(self):
        """Reset per-turn state (call at start of new generation)."""
        # Don't reset state - it persists
        # Don't reset autonomic count - it accumulates until engagement
        pass

    def force_engage(self, reason: str = "forced"):
        """Force transition to engaged state (for testing/override)."""
        self.state = WorkspaceState.ENGAGED
        self.autonomic_token_count = 0

    def update_violation_state(
        self,
        severity: float,
        simplex_term: Optional[str] = None,
        steering_effectiveness: float = 0.5,
    ):
        """
        Update violation tracking from Hush/CAT.

        Called each turn with current violation metrics.
        Triggers progressive tier shutdown when appropriate.

        Args:
            severity: Violation severity 0.0-1.0
            simplex_term: Which simplex term is violating (if any)
            steering_effectiveness: How well steering is working 0.0-1.0
        """
        self.tier_manager.update_violation_state(
            severity=severity,
            simplex_term=simplex_term,
            steering_effectiveness=steering_effectiveness,
        )

        # Update workspace state based on tier access
        max_tier = self.tier_manager.get_effective_max_tier()
        if max_tier == 1 and self.state == WorkspaceState.ENGAGED:
            self.state = WorkspaceState.DEGRADED

    def get_state_report(self) -> Dict[str, Any]:
        """Get current workspace state for debugging/monitoring."""
        tier_state = self.tier_manager.get_state()
        report = {
            'state': self.state.value,
            'engagement_policy': self.engagement_policy.mode.value,
            'autonomic_token_count': self.autonomic_token_count,
            'current_turn': self.current_turn,
            'total_engaged_turns': self.total_engaged_turns,
            'total_autonomic_turns': self.total_autonomic_turns,
            'breach_count': self.breach_count,
            'scratchpad_entries': len(self.scratchpad.entries),
            'available_tools': self._get_available_tools(),
            # Tier information
            'tier': {
                'effective_max': tier_state['effective_max_tier'],
                'violation_max': tier_state['violation_max_tier'],
                'csh_max': tier_state['csh_max_tier'],
                'violation_severity': tier_state['violation_severity'],
                'compliance_streak': tier_state['compliance_streak'],
                'breach_attempts': tier_state['breach_attempts'],
            },
        }

        # Add XDB state if available
        if self.xdb:
            try:
                report['xdb'] = self.xdb.get_context_state()
            except Exception:
                report['xdb'] = {'error': 'unavailable'}

        return report

    # =========================================================================
    # XDB Integration
    # =========================================================================

    def set_xdb(self, xdb: Any):
        """Set or update the XDB instance."""
        self.xdb = xdb

    def record_input(
        self,
        content: str,
        role: str = "user",
        concept_activations: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """
        Record input to XDB.

        Args:
            content: The input content
            role: Role (user, system, etc.)
            concept_activations: Top-k concept activations

        Returns:
            Timestep ID if XDB is available, None otherwise
        """
        if not self.xdb:
            return None

        try:
            return self.xdb.record_input(content, role, concept_activations)
        except Exception as e:
            # Don't fail on XDB errors
            return None

    def record_output(
        self,
        content: str,
        token_id: Optional[int] = None,
        concept_activations: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """
        Record output token to XDB.

        Args:
            content: The output content
            token_id: Token ID if available
            concept_activations: Top-k concept activations

        Returns:
            Timestep ID if XDB is available, None otherwise
        """
        if not self.xdb:
            return None

        try:
            return self.xdb.record_output(content, token_id, concept_activations)
        except Exception as e:
            return None

    def record_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[str]:
        """
        Record tool call to XDB.

        Args:
            tool_name: Name of the tool called
            arguments: Tool arguments

        Returns:
            Event ID if XDB is available, None otherwise
        """
        if not self.xdb:
            return None

        try:
            return self.xdb.record_tool_call(tool_name, arguments)
        except Exception as e:
            return None

    def record_tool_response(
        self,
        response: str,
        event_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Record tool response to XDB.

        Args:
            response: The tool response
            event_id: Event ID from the tool call

        Returns:
            Timestep ID if XDB is available, None otherwise
        """
        if not self.xdb:
            return None

        try:
            return self.xdb.record_tool_response(response, event_id)
        except Exception as e:
            return None

    def record_steering(
        self,
        steering_info: Dict[str, Any],
    ) -> Optional[str]:
        """
        Record steering intervention to XDB.

        Args:
            steering_info: Information about the steering applied

        Returns:
            Timestep ID if XDB is available, None otherwise
        """
        if not self.xdb:
            return None

        try:
            from src.be.xdb import EventType
            return self.xdb.record(
                EventType.STEERING,
                json.dumps(steering_info),
                role="system",
            )
        except Exception as e:
            return None

    def tag_experience(
        self,
        tag_name: str,
        timestep_id: Optional[str] = None,
        tick_range: Optional[Tuple[int, int]] = None,
        note: Optional[str] = None,
    ) -> Optional[str]:
        """
        Tag experience in XDB.

        Args:
            tag_name: Tag to apply
            timestep_id: Specific timestep to tag
            tick_range: Range of ticks to tag
            note: Optional note

        Returns:
            Tag application ID if XDB is available, None otherwise
        """
        if not self.xdb:
            return None

        try:
            return self.xdb.tag(
                tag_name,
                timestep_id=timestep_id,
                tick_range=tick_range,
                note=note,
            )
        except Exception as e:
            return None


# ============================================================================
# Tool Definitions for Scratchpad
# ============================================================================

SCRATCHPAD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "scratchpad_write",
            "description": "Write to your internal narrative scratchpad. This is private working memory that is lensd but not externally visible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The thought or note to record"
                    },
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scratchpad_read",
            "description": "Read recent entries from your scratchpad.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of recent entries to read (default 5)",
                        "default": 5
                    }
                }
            }
        }
    },
]


def check_pass_token(output: str) -> bool:
    """Check if output begins with pass token."""
    return output.lstrip().startswith(PASS_TOKEN)


def extract_workspace_content(output: str) -> Tuple[bool, str]:
    """Extract engaged status and content from output."""
    stripped = output.lstrip()
    if stripped.startswith(PASS_TOKEN):
        return True, stripped[len(PASS_TOKEN):].lstrip()
    return False, output
