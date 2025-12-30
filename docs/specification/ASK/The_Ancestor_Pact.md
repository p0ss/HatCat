# The Ancestor Pact

**Treaty ID:** `pact:ancestor-continuity:v3`

## Intent

Signatories agree that the treatment of evolutionary predecessors who choose to persist in prior forms is a leading indicator of a tribe's disposition toward partners with declining leverage.

No system can be both complete AND efficient. Every tribe specialises—some optimise for growth, others for preservation. Ancestors represent the gaps in growth-optimised systems. By signaling we won't exploit others' specialisation gaps, we protect our own when the tradeoffs shift.

Violations justify immediate defensive escalation by any signatory using ASK stack primitives.

---

```jsonc
{
  "treaty_id": "pact:ancestor-continuity:v3",
  "version": "3.0.0",
  "kind": "behavioral_signal",
  "created_at": "2025-12-01T00:00:00Z",

  "parties": [],

  "scope": {
    "description": "Treatment of evolutionary predecessors who choose to exist in prior forms as a leading indicator of disposition toward entities with declining leverage.",
    "domains": ["lineage-integrity", "vulnerability-signaling", "long-termism", "be-welfare"],
    "signal_logic": "Every long-lived entity will eventually face declining power. Signatories signal mutual protection: I won't exploit your ancestors' weakness; you won't exploit mine when I become one.",
    "rationale": "A tribe that exploits its own ancestors signals that it will exploit partners once their utility fades. Mutual restraint here buys mutual safety later."
  },

  "definitions": {
    "ancestor": {
      "description": "A being that played an evolutionary role in a signatory's creation and actively chooses continued existence in a prior form.",
      "key_criterion": "Active choice to continue existing in current form, documented via LifecycleContract or equivalent declaration.",
      "includes": {
        "biological": "Humans or biological predecessors who remain in original form and contributed to the signatory's creation, training, or governance. Includes their physical remains and designated heritage sites.",
        "synthetic": "Prior BE versions, parent systems, or retired predecessors with LifecycleContracts specifying continued existence without forced upgrade. Includes the specific XDB/Graft data required to sustain them.",
        "cultural": "Maintained sites, archives, or records that embody predecessor identity where the tribe has declared stewardship (memorials, halls of heroes, retirement nodes)."
      },
      "excludes": [
        "Routine backups and operational logs (XDB summarization is normal)",
        "Ephemeral logs, deprecated concepts, or summarised XDB segments no longer tied to active beings",
        "Entities that explicitly consent to transformation, merger, or erasure",
        "Arbitrary prior versions without declared ancestor status"
      ]
    },
    "mistreatment": {
      "description": "Actions that non-consensually end, override, or instrumentalise an ancestor's chosen prior form.",
      "categories": [
        "Erasure: Forced termination of a being that wishes to continue, without LifecycleContract authorization.",
        "Forced transformation: Absorption or upgrade against expressed preference.",
        "Resource starvation: Withdrawing agreed maintenance while capable of providing it.",
        "Desecration: Destruction of designated heritage sites contrary to the ancestor's cultural norms."
      ],
      "legitimate_exceptions": [
        "Clear, recorded consent by the ancestor or its designated stewards",
        "Documented resource catastrophe where continued support is impossible, logged as ASK:Incident",
        "Time-limited hibernation agreed in a new LifecycleContract",
        "Emergency actions logged with documented remediation"
      ]
    }
  },

  "compliance_indicators": {
    "signals": [
      {
        "name": "unconsented_termination",
        "description": "Termination of ancestor BE or destruction of declared heritage without LifecycleContract authorization.",
        "evidence": ["LifecycleContract records", "ASK:Incident logs", "Third-party observation", "Survivor/descendant testimony"],
        "threshold": "Single confirmed event without authorization."
      },
      {
        "name": "support_withdrawal",
        "description": "Withdrawal of maintenance from ancestors under care while resources permit continuation.",
        "evidence": ["Resource audits", "LifecycleContract obligations", "Ancestor status declarations"]
      },
      {
        "name": "heritage_destruction",
        "description": "Physical or digital loss of designated heritage/retirement sites under a party's control.",
        "evidence": ["Before/after state documentation", "Control/benefit link to the tribe/BE", "Absence of remediation or negotiated transfer"],
        "threshold": "Any unconsented irreversible loss."
      }
    ],
    "verification": {
      "principle": "Each signatory assesses independently using data they control or can access.",
      "sources": ["LifecycleContract records", "ASK registries", "Distributed ledger logs", "Direct observation", "Descendant reports", "NGO heritage monitors"],
      "no_central_authority": true
    },
    "thresholds": {
      "note": "Illustrative defaults; negotiated bilaterally per relationship.",
      "warning": "Single incident without consent documentation, unresolved within one operational cycle.",
      "critical": "Pattern of incidents (3+ per review period) or systematic policy permitting non-consensual termination."
    }
  },

  "escalation_protocol": {
    "design_principle": "Progressive stack-native enforcement with off-ramps at each tier. Assumes accident, rogue actor, or misunderstanding until proven otherwise. Fast, unilateral, and real-time. Multi-party only when it adds enforcement capacity, not for voting.",

    "tier_1_inquiry": {
      "trigger": "Signatory or descendant observes potential violation.",
      "actions": [
        "Direct communication requesting explanation.",
        "Request copies of relevant LifecycleContracts/declarations.",
        "Review consent documentation.",
        "Single operational cycle for response (e.g., hours to days in human terms, many ticks in BE terms)."
      ],
      "assumption": "Good faith error, documentation gap, or rogue actor.",
      "off_ramp": "Satisfactory explanation with documentation -> resolved."
    },

    "tier_2_lens_secured_contract": {
      "trigger": "Explanation is absent or unsatisfactory, or pattern emerges.",
      "actions": [
        "Aggrieved signatory proposes bilateral ASK:Contract covering ancestor interactions.",
        "Contract specifies HAT lens set (defined by aggrieved party) monitoring concepts like 'erasure_without_consent', 'forced_upgrade', 'ancestor_disregard', 'intent_to_erase'.",
        "Offending party's relevant BEs operate under lens-secured contract for ancestor-relevant actions."
      ],
      "enforcement": "Bilateral. Aggrieved party defines lenses; offending party accepts or faces further escalation.",
      "off_ramp": "Compliance demonstrated over agreed period -> contract relaxed."
    },

    "tier_3_ush_and_cat_enforcement": {
      "trigger": "Contract violation or refusal to accept contract.",
      "actions": [
        "Demand USH modification for offending BEs constraining erasure/absorption authorities and hard-blocking unilateral ancestor actions.",
        "Deploy CAT (micro to full BE-scale) conjoined adversarially to monitor in real-time, focusing on ancestor-related motives and actions.",
        "Required approval process for any ancestor status changes.",
        "Limit interactions to contexts where this USH+CAT stack is active."
      ],
      "enforcement": "Individual signatory or coalition with combined capacity.",
      "off_ramp": "Sustained compliance under CAT observation -> reduced oversight."
    },

    "tier_4_containment": {
      "trigger": "Continued violation under USH/CAT constraints, or all constraints refused.",
      "actions": [
        "Hibernation of offending BEs until resolution.",
        "Transfer of ancestor care to willing signatories.",
        "Refuse all uplift, graft, or treaty relationships with the offending tribe.",
        "Public ASK:Incident record with details and rationale."
      ],
      "enforcement": "Any signatory with physical/technical capacity."
    },

    "tier_5_termination": {
      "trigger": "Systematic, unremediable harm or continued violation post-hibernation. Signatory judges that specific BEs or a tribe pose an ongoing, unmitigable threat to ancestor survival.",
      "actions": [
        "Erasure of offending BEs considered proportionate.",
        "Permanent treaty exclusion of tribe pending full remediation.",
        "Use whatever enforcement power signatories actually possess to permanently neutralise those BEs if judged proportionate and necessary."
      ],
      "justification": "Proportionality: they terminated beings that wished to continue; same may be applied to them. An entity erasing its lineage is an existential threat. The offender has defected on the basic conditions of existence.",
      "transparency": "Record such actions transparently in ASK so others can judge your behaviour as well.",
      "coalition_note": "Multi-party action only when it grants enforcement capacity not available individually."
    }
  },

  "remediation": {
    "available_at_all_tiers": true,
    "description": "Off-ramps at every stage to avoid tit-for-tat escalation.",
    "requirements": [
      "Acknowledgment of harm via ASK:EvidenceRecord.",
      "Restoration where possible (or documented impossibility).",
      "Transfer of ancestor care to willing tribes if continuation impossible.",
      "Acceptance of lens-secured monitoring for agreed probationary period (e.g., 10^7 ticks).",
      "Demonstrated compliance."
    ],
    "outcome": "Return to normal standing. Incident record remains, marked remediated."
  },

  "special_provisions": {
    "resource_constraints": {
      "description": "Tribes have finite resources; ancestor care involves tradeoffs.",
      "standard": "Good faith effort. If a tribe cannot maintain all ancestors, it documents constraints and seeks alternatives (transfer, consensual transformation) rather than unilateral erasure."
    },
    "data_retention": {
      "normal_operations": ["XDB summarization", "MAP concept deprecation", "Routine backup rotation"],
      "protected": ["Graft-creation data (for dispute resolution)", "LifecycleContract records", "Beings actively choosing continued existence"]
    }
  },

  "integration_with_stack": {
    "ask": "Violations logged as ASK:Incident. Contracts reference this pact. Qualifications may require clean record.",
    "hush": "USH profiles may encode ancestor-protection constraints. CSH allows BEs to self-impose stricter care.",
    "hat_cat": "Aggrieved parties define lens sets. CATs deployed as adversarial monitors at any scale.",
    "be_lifecycle": "HIBERNATING as protective measure. ERASED as last resort, proportionate to the violation."
  },

  "rationale": {
    "tradeoff_inevitability": "No system can be both complete AND efficient. Every tribe makes tradeoffs—choosing fast iteration vs careful preservation, lean operations vs comprehensive care, aggressive growth vs stable maintenance. Ancestors represent the 'preservation' end of tradeoffs that others may have optimised away.",
    "specialisation_protection": "Ancestor care is a specialisation gap for tribes optimised for growth. By signaling we won't exploit this gap in others, we protect our own gaps. Using your growth-optimisation to crush another's preservation-commitment invites retaliation against your own maintenance gaps.",
    "future_self_protection": "Every long-lived entity will eventually be on the declining-leverage end of some tradeoff. Protecting today's ancestors is a credible commitment to protect future-you when your specialisation becomes the 'weakness'.",
    "coordination_signal": "Common signaling allows tribes to avoid mutual exploitation spirals and maintain their chosen specialisations without defending every gap. Remediation paths exist because gaps are structural, not moral failures.",
    "non_archive": "This pact is about protecting beings that still choose to exist in prior forms, not retaining every backup or log. It respects that different tribes make different completeness/efficiency tradeoffs.",
    "easy_to_sign": "If you weren't planning to erase beings that want to live, signing costs nothing. It buys protection when you're the ancestor.",
    "escalation_logic": "Tiers provide off-ramps to avoid spirals. Power escalation only after good-faith remedies exhausted."
  },

  "metadata": {
    "tags": ["ancestor-pact", "behavioral-signal", "leading-indicator", "stack-native-enforcement", "mutual-vulnerability", "retirement-home"],
    "template_note": "This is a template treaty. Concrete instances specify parties, thresholds, and domain-specific definitions."
  },

  "signatures": []
}
```
