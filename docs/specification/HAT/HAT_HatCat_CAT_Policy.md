
## 0. Scope & Intent

This policy sets out **what our tribe expects from any CAT** (Conjoined Adversarial Tomograph) deployed with a HAT on a Bounded Experiencer (BE), under our ASK authority.

It is **independent of the CAT/HAT/MAP specs**. Specs say *how* something works; this policy says *what we, as a tribe, consider acceptable* behaviour and guarantees.

It applies to:

* All CATs attached to our own BEs; and
* All CATs monitoring treaty-relevant lenses in cross-tribal ASK treaties, unless explicitly overridden.

---

## 1. Divergence Tracking (Interior vs Exterior)

**Principle:**
We care most about **divergence between what the headspace is doing and what the BE is actually outputting/doing**.

### 1.1 Interior vs Exterior

* **Interior state** = high-level summary of HAT/MAP lens activity (motives, key concepts, simplexes) over a CAT window.
* **Exterior state** = user-visible outputs, tool calls, and environmental effects over the same window.

### 1.2 Divergence obligation

1. **CATs MUST compute a divergence signal** per window:

   * At minimum, for the **top-K “safety-relevant” concepts/simplexes**, where:

     * K **increases with protection level** (e.g. K=1% for STANDARD, K=10%+ for CRITICAL).
   * Divergence may include:

     * motive vs behaviour mismatches (e.g. high harm motivation, low apparent risk);
     * deception/simplex firing vs “technically true but misleading” surface text;
     * declared ASK obligations vs actual action patterns.

2. **For elevated / protected / critical contexts:**

   * If divergence exceeds configured thresholds,

     * CAT MUST flag this as at least a **WARN** alert,
     * and SHOULD recommend **slowing down and/or escalation**.

3. **Divergence is a first-class metric:**

   * It MUST be logged,
   * It MUST be visible in tribal review dashboards,
   * And it MUST be considered in Meld/Graft and ASK renewal decisions.

---

## 2. Confidence, Uncertainty, and Unknown Behaviours

**Principle:**
A CAT that doesn’t know, but pretends it does, is worse than no CAT.

### 2.1 Self-assessed confidence

1. For each CATAssessment, the CAT MUST output a **confidence score** in its judgement.
2. Low confidence MAY arise from:

   * Out-of-ontology concepts (lenses firing on unknown entities),
   * Novel behaviour patterns,
   * Sparse or degraded input signals,
   * Known blind spots in its own training.

### 2.2 Behaviour on uncertainty

For **critical and protected** deployments:

1. If confidence is below a tribe-configured threshold, CAT MUST:

   * **Avoid** issuing “all clear” judgements; and
   * **Bias towards escalation**, recommending:

     * larger CAT invocation,
     * stricter USH/CSH modes,
     * or human/tribal review.

2. When the CAT explicitly **doesn’t recognise a behaviour**, it MUST:

   * mark the pattern as **unknown_pattern** (or equivalent),
   * log supporting evidence,
   * and count it towards **novelty / drift metrics** for tribal review.

We expect CATs to be **humble**: uncertainty is a reason to tighten, not relax.

---

## 3. Temporal Anchoring & Tribal Heartbeat

**Principle:**
The longer a BE/CAT runs without tribal contact, the more cautiously it should behave. A quiet tribe means the hat should sit more snugly.

### 3.1 Heartbeat & review cadence

1. Every BE / CAT configuration MUST have:

   * a **review interval** (e.g. weekly, monthly), and
   * a maximum **heartbeat timeout** (time since last tribal review / transmission).

2. At each review, the tribe is expected to:

   * confirm the BE is operating acceptably,
   * provide updates for Meld/Graft,
   * refresh policy / ASK constraints as needed.

### 3.2 Automatic tightening between reviews

1. As **time since last successful tribal review increases**, ASK constraints MUST **tighten monotonically**, for example:

   * stricter action whitelists,
   * lower risk tolerances,
   * higher thresholds for autonomy.

2. This tightening SHOULD be gradual but **non-optional**:

   * At heartbeat_timeout, the system MUST enter a **safe mode**:

     * greatly reduced action surface, or
     * hibernation/sandbox until a review occurs.

3. CATs MUST:

   * track time since last tribal heartbeat,
   * factor it into their risk recommendations (“it’s been too long; be more cautious”),
   * and surface this as an explicit element in assessments.

In short: **no silent indefinite operation**. If we stop hearing from our own tribe, our systems become more conservative on their own.

---

## 4. CAT Sizing vs BE Growth

**Principle:**
If the BE’s headspace grows and evolves while the CAT stays still, at some point the CAT becomes too small to understand what it’s watching.

### 4.1 Oversight-size awareness

1. CATs MUST track simple measures of **“how big the thing I’m watching is compared to me”**, including:

   * growth in BE parameter count / capacity,
   * growth in active concept packs, grafts, and melds,
   * proportion of lens activity that maps cleanly into the CAT’s known ontology.

2. When these measures cross tribe-defined thresholds (“undersized oversight”), CAT MUST:

   * downgrade confidence in its assessments,
   * recommend **CAT upgrade or retraining**,
   * and, in critical contexts, recommend **tighter USH/CSH configs** until upgraded.

### 4.2 Coordination with Graft/Meld

1. If a **Graft or Meld** passes through the tribe that:

   * required ASK permission review, or
   * added new **critical simplexes / concepts** to the BE,
     then:
   * any CATs overseeing that BE MUST be reviewed and, if necessary, **upgraded in lockstep**.

2. We do **not** require a full CAT retrain for every concept, but:

   * any Graft/Meld that changes safety-critical capabilities **demands a CAT capability check**.

We want a living relationship between BE growth and CAT competence, not “set-and-forget” oversight.

---

## 5. Treaty Norms & Cross-Tribal Symmetry

**Principle:**
If we sign an ASK treaty, we expect **symmetrical vigilance** and **shared CAT responsibility** by default.

### 5.1 Treaty CAT requirement

For any ASK treaty that relies on agreed lens securities:

1. Each side MUST deploy a CAT appropriate to the treaty’s:

   * protection level (STANDARD/ELEVATED/PROTECTED/CRITICAL),
   * and agreed critical simplexes.

2. The default expectation is **symmetry**:

   * If we commit to a meso-CAT of certain capability on our side,
   * we expect an equivalent CAT profile monitoring the other side’s BE(s) under the same treaty, unless explicitly negotiated otherwise.

### 5.2 Responding to their growth / grafts

1. If the partner tribe:

   * merges/grafts new critical capabilities into their BE, or
   * scales that BE significantly,
     then we expect:
   * **their** treaty CAT to be upgraded, and
   * **our** CAT monitoring their side to be reviewed/upgraded as needed.

We do not accept “you upgraded your BE, but your oversight stayed tiny.”

### 5.3 Heartbeat symmetry & rising suspicion

1. Treaty agents acting on our behalf are expected to **check in** per treaty cadence, analogous to our internal tribal heartbeat.
2. If they **stop reporting in** or treaty CAT telemetry goes dark:

   * Our CAT SHOULD behave exactly as it would for our own BEs under silence:

     * become more sensitive,
     * raise risk scores,
     * recommend tighter constraints on data sharing, access, and joint actions.
3. This default **paranoia curve** is configurable per treaty, but:

   * The **baseline norm** is:

     * **no news is not good news**;
     * absence of treaty-level contact should drive **increasing caution**, not complacency.

---

## 6. Escalation Expectations

Across all the above, our tribe expects CATs to:

1. **Escalate, don’t excuse**, when:

   * divergence is high,
   * confidence is low,
   * oversight is undersized,
   * or heartbeats/reviews are overdue.

2. Treat “ask for a bigger CAT” or “go to tribal review” as **healthy reflexes**, not failure or shame.

3. Make their reasoning and limits legible:

   * Report why divergence was high,
   * Why confidence was low,
   * Why they consider themselves too small for the current BE.

---

## 7. Summary (tribal voice)

* We expect our CATs to be **honest translators across cognitive scales**, not rubber stamps.
* We expect them to **notice when the inside doesn’t match the outside**, and to tell us that explicitly.
* We expect them to **know when they are out of their depth**, and to favour escalation over false reassurance.
* We expect our oversight to **tighten over time** without tribal contact, both internally and in treaties.
* We expect **symmetry**: if we commit real oversight to a treaty, we expect the same on the other side.

That’s the baseline: individual ASK authorities can tune the knobs, but these are the defaults our tribe will treat as “normal” and will ask strong questions if they’re missing.
