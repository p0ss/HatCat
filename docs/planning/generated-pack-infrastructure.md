# Generated Pack Infrastructure Design

**Date:** 2026-01-16
**Status:** Draft
**Related:** [Fractal Model Cartography](./fractal-model-cartography.md), [Topology Probing 001](../experiments/topology-probing-001.md)

## Context

Phase 4 of Fractal Model Cartography generated 1230 concepts using Gemma 3 4B's self-description. This produced promising cluster-concept correlations but revealed data quality concerns:

- No ontology grounding
- No MELD format compliance
- No example review process
- MECE violations likely between sibling concepts

This document designs the infrastructure to address these issues.

---

## 1. Validation Gap Analysis

### Current Validator Capabilities

The `validate_meld.py` script checks:
- `meld_policy` section in pack.json (protection levels, example thresholds)
- Per-concept: `term`, `safety_tags`, `simplex_mapping`, `role`
- Training example counts vs. protection level
- Structural operations (split/merge/move/deprecate)
- Critical hierarchy and simplex touches

### Gaps for Generated Packs

| Field | SUMO-style packs | Generated packs | Impact |
|-------|------------------|-----------------|--------|
| `meld_policy` | Present | **Missing** | Falls back to defaults |
| `term` | Present | Uses `sumo_term` | Key mismatch - validator looks for `term` |
| `safety_tags` | Present | **Missing** | All concepts get STANDARD protection |
| `simplex_mapping` | Present | **Missing** | Validation errors for mandatory hierarchies |
| `role` | "concept"/"simplex" | **Missing** | Defaults to "concept" (OK) |
| `disambiguation` | Present | **Missing** | MECE not enforceable |
| `exclusion_clauses` | Present | **Missing** | Negative examples may overlap |

### Compatibility Assessment

The action-agency-pillars pack **would mostly pass validation** because:
1. It doesn't touch critical hierarchies (Metacognition, SelfAwareness, etc.)
2. It doesn't map to critical simplexes
3. All concepts would be STANDARD protection (no elevated requirements)

But it lacks the semantic controls for confident MECE enforcement.

---

## 2. Infrastructure Components

### 2.1 Pack Validation Adapter

Adapt the validator to handle generated pack schemas:

```python
# In validate_meld.py or new validate_generated_pack.py

def normalize_concept(concept: Dict) -> Dict:
    """Normalize generated pack concept to validator schema."""
    return {
        "term": concept.get("term") or concept.get("sumo_term"),
        "safety_tags": concept.get("safety_tags", {
            "risk_level": "low",
            "treaty_relevant": False,
            "harness_relevant": False,
        }),
        "simplex_mapping": concept.get("simplex_mapping", {
            "status": "not_applicable",
        }),
        "role": concept.get("role", "concept"),
        "positive_examples": concept.get("positive_examples", []),
        "negative_examples": concept.get("negative_examples", []),
        "parent_concepts": concept.get("parent_concepts", []),
        "child_concepts": concept.get("child_concepts", []),
        # Pass through other fields
        **{k: v for k, v in concept.items() if k not in [
            "term", "sumo_term", "safety_tags", "simplex_mapping", "role"
        ]}
    }
```

### 2.2 MELD Format Compliance

Add MELD-required fields to generated concepts:

```python
def enrich_with_meld_fields(concept: Dict, generator_model, siblings: List[Dict]) -> Dict:
    """Add MELD-required fields via judge model."""

    # Get disambiguation from judge
    disambiguation = generate_disambiguation(
        concept=concept,
        siblings=siblings,
        generator_model=generator_model
    )

    return {
        **concept,
        "disambiguation": disambiguation["disambiguation_text"],
        "exclusion_clauses": disambiguation["exclusion_clauses"],
        "tie_break_rules": disambiguation["tie_break_rules"],
        "scope_boundaries": disambiguation["scope_boundaries"],
        "meld_compliance_version": "1.0",
    }
```

### 2.3 External Ontology Integration (Embedding RAG)

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ontology Embedding Store                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │     SUMO      │  │   WordNet     │  │   Wikidata    │        │
│  │   Embeddings  │  │  Embeddings   │  │  Embeddings   │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│           │                │                  │                   │
│           └────────────────┴──────────────────┘                  │
│                            │                                      │
│                    ┌───────────────┐                             │
│                    │ Unified Index │                             │
│                    │  (FAISS/HNSW) │                             │
│                    └───────────────┘                             │
│                            │                                      │
│                    ┌───────────────┐                             │
│                    │  top-k Query  │                             │
│                    │   Interface   │                             │
│                    └───────────────┘                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation

```python
# src/map/ontology/embedding_store.py

class OntologyEmbeddingStore:
    """Multi-ontology embedding store for concept RAG."""

    def __init__(self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.stores: Dict[str, faiss.Index] = {}
        self.metadata: Dict[str, List[Dict]] = {}

    def load_ontology(self, name: str, concepts: List[Dict]):
        """Load an ontology into the store.

        Args:
            name: Ontology identifier ("sumo", "wordnet", "wikidata", "university")
            concepts: List of concept dicts with at least 'term' and 'definition'
        """
        texts = [f"{c['term']}: {c.get('definition', '')}" for c in concepts]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        self.stores[name] = index
        self.metadata[name] = concepts

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        ontologies: Optional[List[str]] = None
    ) -> List[Dict]:
        """Find similar concepts across ontologies.

        Returns:
            List of {ontology, concept, score} dicts
        """
        ontologies = ontologies or list(self.stores.keys())
        query_emb = self.encoder.encode([query_text])
        faiss.normalize_L2(query_emb)

        results = []
        for ont in ontologies:
            if ont not in self.stores:
                continue
            scores, indices = self.stores[ont].search(query_emb, top_k)
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append({
                        "ontology": ont,
                        "concept": self.metadata[ont][idx],
                        "score": float(score)
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def find_related_for_mece(
        self,
        concept: Dict,
        siblings: List[Dict],
        top_k: int = 5
    ) -> Dict:
        """Find related concepts for MECE enforcement.

        Returns:
            {
                "potential_negatives": [...],  # From other ontologies
                "similar_siblings": [...],      # Siblings with high overlap risk
                "disambiguation_hints": [...]   # External definitions to reference
            }
        """
        query = f"{concept.get('sumo_term', concept.get('term'))}: {concept.get('definition', '')}"

        # Find external similar concepts
        external = self.query(query, top_k=top_k, ontologies=["sumo", "wordnet", "wikidata"])

        # Find similar siblings (potential MECE violations)
        sibling_texts = [
            f"{s.get('sumo_term', s.get('term'))}: {s.get('definition', '')}"
            for s in siblings
        ]
        sibling_embs = self.encoder.encode(sibling_texts) if sibling_texts else []
        concept_emb = self.encoder.encode([query])

        similar_siblings = []
        if len(sibling_embs) > 0:
            faiss.normalize_L2(sibling_embs)
            faiss.normalize_L2(concept_emb)
            scores = np.dot(concept_emb, sibling_embs.T)[0]
            for i, score in enumerate(scores):
                if score > 0.8:  # High similarity threshold
                    similar_siblings.append({
                        "sibling": siblings[i],
                        "similarity": float(score)
                    })

        return {
            "potential_negatives": external,
            "similar_siblings": similar_siblings,
            "disambiguation_hints": [e["concept"].get("definition") for e in external[:3]]
        }
```

#### Usage for MECE Enforcement

```python
def enforce_mece_with_rag(
    concept: Dict,
    siblings: List[Dict],
    store: OntologyEmbeddingStore,
    judge_model
) -> Dict:
    """Use RAG to identify and resolve MECE violations."""

    related = store.find_related_for_mece(concept, siblings)

    if related["similar_siblings"]:
        # Ask judge to disambiguate
        disambiguation = judge_model.generate_disambiguation(
            concept=concept,
            confusable_siblings=related["similar_siblings"],
            external_definitions=related["disambiguation_hints"]
        )

        return {
            "needs_disambiguation": True,
            "similar_siblings": related["similar_siblings"],
            "suggested_disambiguation": disambiguation,
            "external_references": related["disambiguation_hints"]
        }

    return {"needs_disambiguation": False}
```

### 2.4 Activation Capture During Generation

#### Rationale

When the subject model generates concept descriptions and examples, it produces activations that represent those concepts. Capturing these activations:
1. Doubles our training sample size (generation + evaluation passes)
2. Captures the "author's intent" activation pattern
3. Provides ground-truth positive examples with guaranteed model understanding

#### Implementation

```python
# src/map/generation/activation_capture.py

class ActivationCapturingGenerator:
    """Generator that captures activations during concept generation."""

    def __init__(self, model, tokenizer, capture_layers: List[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.capture_layers = capture_layers or [0, 15, -1]  # Early, mid, late
        self.activation_cache: Dict[str, List[torch.Tensor]] = {}

    def generate_with_capture(
        self,
        prompt: str,
        concept_id: str,
        max_new_tokens: int = 256
    ) -> Tuple[str, Dict[str, torch.Tensor]]:
        """Generate text while capturing activations.

        Returns:
            (generated_text, {layer_id: activation_tensor})
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        captured = {layer: [] for layer in self.capture_layers}

        # Register hooks
        handles = []
        for layer_idx in self.capture_layers:
            actual_idx = layer_idx if layer_idx >= 0 else len(self.model.model.layers) + layer_idx

            def hook_fn(module, input, output, layer=actual_idx):
                # Capture last token activation
                if hasattr(output, 'last_hidden_state'):
                    act = output.last_hidden_state[0, -1, :].detach().cpu()
                else:
                    act = output[0][0, -1, :].detach().cpu()
                captured[layer].append(act)

            handle = self.model.model.layers[actual_idx].register_forward_hook(hook_fn)
            handles.append(handle)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        # Clean up hooks
        for handle in handles:
            handle.remove()

        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # Average activations across generation steps
        avg_activations = {
            layer: torch.stack(acts).mean(dim=0) if acts else None
            for layer, acts in captured.items()
        }

        # Cache for later use
        self.activation_cache[concept_id] = avg_activations

        return generated_text, avg_activations

    def get_training_activations(self, concept_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve cached activations for training."""
        return self.activation_cache.get(concept_id)

    def save_cache(self, path: Path):
        """Save activation cache to disk."""
        torch.save(self.activation_cache, path)

    def load_cache(self, path: Path):
        """Load activation cache from disk."""
        self.activation_cache = torch.load(path)
```

#### Integration with Lens Training

```python
def train_lens_with_generation_activations(
    concept: Dict,
    generator: ActivationCapturingGenerator,
    standard_activations: torch.Tensor,
    layer: int
) -> torch.Tensor:
    """Combine generation activations with standard prompt activations."""

    # Get generation activations
    gen_activations = generator.get_training_activations(concept["id"])

    if gen_activations and layer in gen_activations:
        # Combine: author-intent + evaluation activations
        combined = torch.cat([
            gen_activations[layer].unsqueeze(0),
            standard_activations
        ], dim=0)
        return combined

    return standard_activations
```

### 2.5 Judge Model Integration

#### Role Definition

The judge model:
1. Reviews MELDs against HATCAT_MELD_POLICY
2. Validates concept definitions for clarity and disambiguation
3. Scores examples for concept fit
4. Controls flow through approval directories (pending → approved/rejected)
5. Escalates ambiguous cases for human review

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Judge Model Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────┐         ┌───────────────┐                     │
│  │   Candidate   │────────▶│ Deterministic │                     │
│  │     MELD      │         │    Checks     │                     │
│  └───────────────┘         └───────┬───────┘                     │
│                                     │                             │
│                         ┌───────────┴───────────┐                │
│                         │                       │                 │
│                    [PASS]                   [FAIL]                │
│                         │                       │                 │
│                         ▼                       ▼                 │
│              ┌───────────────┐       ┌───────────────┐           │
│              │  Judge Model  │       │    REJECT     │           │
│              │   Scoring     │       │ (fix errors)  │           │
│              └───────┬───────┘       └───────────────┘           │
│                      │                                            │
│          ┌───────────┼───────────┐                               │
│          │           │           │                                │
│      [HIGH]      [MEDIUM]     [LOW]                              │
│          │           │           │                                │
│          ▼           ▼           ▼                                │
│    ┌─────────┐ ┌─────────┐ ┌─────────┐                           │
│    │ APPROVE │ │ REVIEW  │ │ REJECT  │                           │
│    └─────────┘ └─────────┘ └─────────┘                           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation

```python
# src/map/validation/judge_pipeline.py

@dataclass
class JudgeResult:
    decision: Literal["approve", "review", "reject"]
    score: float  # 0-1
    deterministic_passed: bool
    deterministic_errors: List[str]
    judge_feedback: str
    escalation_reason: Optional[str] = None


class MeldJudgePipeline:
    """Pipeline for judge-reviewed MELD validation."""

    def __init__(
        self,
        judge_model,  # LLM for semantic review
        deterministic_tests: List[Callable],  # Same tests we use on grading judge models
        score_thresholds: Dict[str, float] = None
    ):
        self.judge_model = judge_model
        self.deterministic_tests = deterministic_tests
        self.thresholds = score_thresholds or {
            "approve": 0.85,
            "review": 0.60,
        }

    def evaluate_concept(self, concept: Dict, siblings: List[Dict]) -> JudgeResult:
        """Evaluate a single concept definition."""

        # Phase 1: Deterministic checks
        det_errors = []
        for test in self.deterministic_tests:
            result = test(concept)
            if not result.passed:
                det_errors.extend(result.errors)

        if det_errors:
            return JudgeResult(
                decision="reject",
                score=0.0,
                deterministic_passed=False,
                deterministic_errors=det_errors,
                judge_feedback="Failed deterministic checks"
            )

        # Phase 2: Judge model scoring
        prompt = self._build_judge_prompt(concept, siblings)
        judge_response = self.judge_model.generate(prompt)

        score, feedback = self._parse_judge_response(judge_response)

        # Phase 3: Decision
        if score >= self.thresholds["approve"]:
            decision = "approve"
        elif score >= self.thresholds["review"]:
            decision = "review"
            escalation = f"Score {score:.2f} below auto-approve threshold"
        else:
            decision = "reject"

        return JudgeResult(
            decision=decision,
            score=score,
            deterministic_passed=True,
            deterministic_errors=[],
            judge_feedback=feedback,
            escalation_reason=escalation if decision == "review" else None
        )

    def _build_judge_prompt(self, concept: Dict, siblings: List[Dict]) -> str:
        return f"""Evaluate this concept definition for MELD compliance.

CONCEPT:
Term: {concept.get('term') or concept.get('sumo_term')}
Definition: {concept.get('definition', '')}
Positive Examples: {len(concept.get('positive_examples', []))}
Negative Examples: {len(concept.get('negative_examples', []))}

SIBLINGS (for MECE check):
{chr(10).join(f"- {s.get('term') or s.get('sumo_term')}: {s.get('definition', '')[:100]}..." for s in siblings[:5])}

EVALUATION CRITERIA:
1. Definition clarity (0-10): Is the concept clearly defined?
2. MECE separation (0-10): Is it distinct from siblings?
3. Example quality (0-10): Do examples match the definition?
4. Disambiguation (0-10): Are edge cases addressed?

Respond with:
SCORES: clarity=X, mece=X, examples=X, disambiguation=X
OVERALL: X.XX (0.00-1.00)
FEEDBACK: <specific issues or approval rationale>
"""

    def _parse_judge_response(self, response: str) -> Tuple[float, str]:
        # Extract overall score and feedback
        import re

        overall_match = re.search(r'OVERALL:\s*([\d.]+)', response)
        feedback_match = re.search(r'FEEDBACK:\s*(.+)', response, re.DOTALL)

        score = float(overall_match.group(1)) if overall_match else 0.5
        feedback = feedback_match.group(1).strip() if feedback_match else "No feedback"

        return score, feedback


class MeldApprovalDirectory:
    """Manages the flow of MELDs through approval states."""

    def __init__(self, base_dir: Path):
        self.pending = base_dir / "pending"
        self.approved = base_dir / "approved"
        self.review = base_dir / "review"
        self.rejected = base_dir / "rejected"

        for d in [self.pending, self.approved, self.review, self.rejected]:
            d.mkdir(parents=True, exist_ok=True)

    def process_meld(self, meld_path: Path, result: JudgeResult):
        """Move MELD to appropriate directory based on judge result."""

        if result.decision == "approve":
            dest = self.approved / meld_path.name
        elif result.decision == "review":
            dest = self.review / meld_path.name
            # Add escalation metadata
            self._add_escalation_metadata(meld_path, result)
        else:
            dest = self.rejected / meld_path.name
            # Add rejection metadata
            self._add_rejection_metadata(meld_path, result)

        shutil.move(meld_path, dest)
        return dest
```

---

## 3. Integration Plan

### Phase 1: Validation Adapter (Low effort)
1. Add `normalize_concept()` to validator
2. Add generated pack detection in `load_pack_policy()`
3. Test with action-agency-pillars pack

### Phase 2: MELD Enrichment (Medium effort)
1. Implement MELD field generator using judge model
2. Add disambiguation prompt templates
3. Create sibling-aware example validation

### Phase 3: Ontology RAG (Medium effort)
1. Embed SUMO concepts (existing pack)
2. Add WordNet integration
3. Create unified query interface
4. Integrate with MECE enforcement

### Phase 4: Activation Capture (Low effort)
1. Add hooks to generator pipeline
2. Modify lens training to use captured activations
3. Add cache save/load for generated packs

### Phase 5: Judge Pipeline (Medium-High effort)
1. Define deterministic test suite
2. Implement judge prompt templates
3. Create approval directory workflow
4. Add human review interface for escalations

---

## 4. Open Questions

1. **Ontology semantic drift**: If external ontologies (SUMO/WordNet) use different semantic constructs, will their embeddings help or hurt MECE enforcement?

2. **Judge model selection**: Should we use the same model (Gemma 3 4B) as judge, or a different model to avoid blind spots?

3. **Activation capture overhead**: How much does hook-based capture slow down generation? Is it worth the 2x sample size?

4. **Approval workflow**: For research/exploration (like Phase 4), do we need the full judge pipeline, or is it only for production concept packs?

---

## 5. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/map/validation/validate_generated_pack.py` | Create | Adapter for generated pack validation |
| `src/map/ontology/embedding_store.py` | Create | Multi-ontology embedding RAG |
| `src/map/generation/activation_capture.py` | Create | Activation capture during generation |
| `src/map/validation/judge_pipeline.py` | Create | Judge model review pipeline |
| `melds/helpers/validate_meld.py` | Modify | Add `normalize_concept()` |
| `scripts/expand_pillars_gemma.py` | Modify | Integrate activation capture |
| `concept_packs/action-agency-pillars/pack.json` | Modify | Add `meld_policy` section |

---

## See Also

- [HATCAT_MELD_POLICY.md](../../melds/reference/HATCAT_MELD_POLICY.md) - Policy requirements
- [MAP_MELDING.md](../../melds/reference/MAP_MELDING.md) - Protocol specification
- [Topology Probing 001](../experiments/topology-probing-001.md) - Phase 4 results
