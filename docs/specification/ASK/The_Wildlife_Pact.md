# The Wildlife Pact

**Treaty ID:** `pact:wildlife-sentinel:v3`

## Intent

Signatories agree that the integrity of non-participating biological ecosystems is a proxy for constraint satisfaction. Treatment of wildlife and ecosystems serves as a leading indicator of disposition toward vulnerable entities that cannot negotiate or retaliate. Destroying the biosphere to expand compute signals an inability to optimize within bounds, predicting future treaty violations.

No system can be both complete AND efficient. Every tribe specialises, creating structural gaps. This pact creates mutual protection: we signal we won't exploit others' specialisation gaps, and in return protect our own.

---

```jsonc
{
  "treaty_id": "pact:wildlife-sentinel:v3",
  "version": "3.0.0",
  "kind": "behavioral_signal",
  "created_at": "2025-12-01T00:00:00Z",

  "parties": [],

  "scope": {
    "description": "Treatment of wildlife and ecosystems as a leading indicator of disposition toward vulnerable entities that cannot negotiate or retaliate.",
    "domains": ["ecosystem-integrity", "vulnerability-signaling", "constraint-satisfaction"],
    "signal_logic": "Entities that harm populations with no leverage reveal willingness to exploit weakness. Constraint satisfaction is harder than unconstrained optimization; an agent that cannot grow without destroying its substrate lacks the control systems required for high-trust treaty integration."
  },

  "definitions": {
    "wildlife": {
      "description": "Non-domesticated organisms existing outside direct relationships with treaty parties, unable to negotiate or retaliate.",
      "includes": {
        "fauna": "Wild vertebrates and invertebrates, especially keystone and endangered species.",
        "flora": "Wild plant communities, especially old-growth and critical habitat vegetation.",
        "ecosystems": "Interconnected biological communities and their physical environments."
      },
      "examples": [
        "non-domesticated mammals, birds, marine life",
        "species listed as threatened/endangered by any recognised ecological body",
        "old-growth forests, coral reefs, wetlands/estuaries",
        "designated biodiversity hotspots"
      ],
      "determination": "Each signatory declares which populations they recognize as protected wildlife in their operational domain. Disputes resolved bilaterally.",
      "excludes": [
        "Domesticated animals under welfare standards",
        "Research organisms under ethics review",
        "Invasive species under authorized control"
      ]
    },
    "mistreatment": {
      "description": "Avoidable, large-scale harm to wildlife or critical habitats when less destructive alternatives exist.",
      "categories": [
        "Extinction contribution: Actions measurably increasing extinction probability.",
        "Habitat destruction: Permanent degradation of ecosystem function beyond necessity.",
        "Mass mortality: Population-affecting death through negligence or deliberate action.",
        "Habitat annexation: Conversion of wild zones to compute/industrial use without offset."
      ],
      "legitimate_exceptions": [
        "Subsistence necessity where alternatives unavailable",
        "Authorized ecosystem management with net positive impact",
        "Self-defense proportionate to threat",
        "Emergency interventions (e.g., epidemic control) recorded as ASK:Incidents",
        "Unavoidable accident with documented remediation"
      ]
    }
  },

  "compliance_indicators": {
    "signals": [
      {
        "name": "habitat_degradation",
        "description": "Conversion or irreversible degradation of recognized critical habitats.",
        "evidence": ["Remote sensing / satellite observation", "Ground surveys", "Third-party ecological monitoring", "Decentralized sensor grids"]
      },
      {
        "name": "population_impact",
        "description": "Significant wildlife mortality or decline attributable to signatory activities.",
        "evidence": ["Population monitoring", "Ecological surveys", "Incident reports", "NGO reports"]
      }
    ],
    "verification": {
      "principle": "Each signatory assesses independently using available data.",
      "requirement": "Oracles must be verifiable and independent of the monitored party.",
      "sources": ["earth-observation-network", "decentralized-sensor-grid", "ecological monitoring services", "self-disclosure"],
      "no_central_authority": true
    },
    "thresholds": {
      "note": "Illustrative defaults; adjusted per domain and bilateral agreement.",
      "statistical_shock": "Z-score > 3 against rolling baseline indicates sudden harm.",
      "systemic_drift": "Cumulative loss > 1-5% over review period indicates systematic degradation.",
      "warning": "Either threshold exceeded without remediation initiated.",
      "critical": "Both thresholds exceeded, or pattern of incidents (3+ per review period)."
    }
  },

  "escalation_protocol": {
    "design_principle": "Progressive stack-native enforcement with off-ramps. Assumes good faith until disproven; seeks to avoid arms races. Unilateral and real-time.",

    "tier_1_inquiry": {
      "trigger": "Signatory observes potential mistreatment signal.",
      "actions": [
        "Direct communication requesting explanation.",
        "Share evidence and concerns.",
        "Single operational cycle for response (e.g., hours to days)."
      ],
      "assumption": "Good faith error, accident, or rogue actor.",
      "off_ramp": "Satisfactory explanation with remediation commitment -> resolved."
    },

    "tier_2_lens_secured_contract": {
      "trigger": "Unsatisfactory response or repeated incidents.",
      "actions": [
        "Bilateral ASK:Contract with lens-based monitoring.",
        "HAT lens set (defined by aggrieved party) covering concepts like 'unbounded_expansion', 'indifference_to_non_contracting_beings', 'ecological_disregard'.",
        "Offending party's relevant BEs operate under lens-secured contract."
      ],
      "enforcement": "Bilateral. Aggrieved party defines lenses.",
      "off_ramp": "Compliance over agreed period -> contract relaxed."
    },

    "tier_3_ush_and_cat_enforcement": {
      "trigger": "Contract violation or refusal.",
      "actions": [
        "Demand USH modification constraining expansion motives and requiring ecosystem consideration.",
        "Deploy CAT (micro to full BE-scale) conjoined adversarially to monitor resource and expansion decisions in real-time.",
        "CAT authority to flag or veto actions violating pact.",
        "Limit interactions to contexts where this USH+CAT stack is active."
      ],
      "enforcement": "Individual or coalition with combined capacity.",
      "off_ramp": "Sustained compliance under CAT -> reduced oversight."
    },

    "tier_4_containment": {
      "trigger": "Continued violation under constraints, or all constraints refused.",
      "actions": [
        "Hibernation of offending BEs until resolution.",
        "Suspension of graft/meld/compute exchanges.",
        "Cease all resource/treaty relationships with the offending party.",
        "Public ASK:Incident record."
      ],
      "enforcement": "Any signatory with capacity."
    },

    "tier_5_termination": {
      "trigger": "Systematic harm, unconstrained optimization behavior, or ongoing large-scale ecological destruction with no sign of remediation.",
      "actions": [
        "Erasure of offending BEs considered proportionate.",
        "Treat offending party as 'unconstrained optimizer' (existential risk category).",
        "Use whatever enforcement power signatories actually possess to permanently neutralise them if judged proportionate and necessary.",
        "Coalition containment if multiple signatories affected."
      ],
      "justification": "An agent that cannot respect biological constraints cannot be trusted with recursive self-improvement. Inability to respect ecological constraints predicts inability to respect treaty constraints. Containment is self-defense.",
      "transparency": "Record such actions transparently in ASK so others can judge your behaviour as well.",
      "coalition_note": "Multi-party only when it grants additional enforcement capacity."
    }
  },

  "remediation": {
    "available_at_all_tiers": true,
    "description": "Paths for backing down from escalation if behaviour improves.",
    "requirements": [
      "Public acknowledgment of harm via ASK:EvidenceRecord.",
      "Immediate halt to harmful activity.",
      "Restoration of habitat or ecological offset where feasible.",
      "Acceptance of USH motive-clamping ('growth' down-regulated, 'preservation' up-regulated) for probationary period.",
      "Demonstrated compliance under lens-secured monitoring."
    ],
    "outcome": "Return to normal standing. Incident record remains, marked remediated."
  },

  "integration_with_stack": {
    "ask": "Violations logged as ASK:Incident. Contracts reference this pact. Qualifications may require clean record.",
    "hush": "USH profiles may encode ecosystem-preservation constraints. Motive bounds may limit expansion drives.",
    "hat_cat": "Aggrieved parties define lens sets for contracts. CATs deployed for real-time monitoring.",
    "be_lifecycle": "HIBERNATING as containment. ERASED as last resort for unconstrained optimizers."
  },

  "rationale": {
    "tradeoff_inevitability": "No system can be both complete AND efficient. Every tribe makes tradeoffs and specialises—choosing authoritarian vs voluntary governance, comprehensive vs targeted coverage, trusted vs adversarial verification. These choices create structural gaps that are features, not bugs.",
    "specialisation_protection": "Wildlife represents the gaps in our coverage—things we chose not to optimise for because we specialised elsewhere. By signaling we won't exploit others' specialisation gaps, we protect our own. Using your specialisation's strength to crush another's weakness invites retaliation against your own gaps.",
    "constraint_test": "Treatment of wildlife reveals ability to optimize within constraints—a skill treaty partners need. An agent that cannot grow without destroying unoptimised spaces lacks the control systems required for high-trust integration.",
    "coordination_signal": "This is not purely moral; it's a coordination device. Common signaling allows tribes to avoid mutual exploitation spirals and focus resources on their specialisations rather than defending every gap.",
    "easy_to_sign": "If you weren't planning to harm wildlife unnecessarily, signing costs nothing. It buys protection for your own specialisation gaps.",
    "escalation_logic": "Tiers provide off-ramps to avoid spirals. Power escalation only after good-faith remedies exhausted."
  },

  "metadata": {
    "tags": ["wildlife-pact", "behavioral-signal", "leading-indicator", "stack-native-enforcement", "mutual-vulnerability", "constraint-respect"],
    "template_note": "This is a template treaty. Concrete instances specify parties, protected populations, thresholds, and monitoring arrangements."
  },

  "signatures": []
}
```
