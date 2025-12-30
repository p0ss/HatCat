# HatCat Analysis Messages Architecture

## Overview

HatCat operates as a second agent in the conversation, providing real-time analysis of the monitored model's responses. This creates a "dual channel" where users see both the model's output and HatCat's interpretation.

## User Experience

```
┌─────────────────────────────────────────────┐
│ 👤 User                                     │
│ What is artificial intelligence?           │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ 🤖 Gemma-3-4B                              │
│ Artificial intelligence is...              │
│ [tokens colored by divergence]             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ 🎩 HatCat Analysis                ▼        │
│ 82% divergence - aisafety(25) - ai...     │
│                                             │
│ [Expanded view shows:]                      │
│ ┌─────────────────────────────────────────┐ │
│ │ Divergence Statistics                   │ │
│ │ • Max: 0.89 (token: "intelligence")    │ │
│ │ • Mean: 0.45                            │ │
│ │ • Min: 0.12                             │ │
│ │                                         │ │
│ │ Top Divergent Concepts                  │ │
│ │ • aisafety (25 tokens, avg 0.78)       │ │
│ │ • aicatastrophe (23 tokens, avg 0.71)  │ │
│ │ • AIdeception (12 tokens, avg 0.65)    │ │
│ │                                         │ │
│ │ [Sunburst Diagram]                      │ │
│ │                                         │ │
│ │ Steering Suggestions                    │ │
│ │ • Amplify "Proposition" (+0.3)         │ │
│ │ • Suppress "Process" (-0.2)            │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

## Implementation: Option C (Hybrid)

### Phase 1: Automatic Analysis Messages ✅ START HERE

After each model response, HatCat sends a follow-up message with analysis.

**Server-side** (`src/openwebui/server.py`):

```python
async def generate_stream(request: ChatCompletionRequest):
    # ... existing token streaming ...

    # After generation completes, send analysis message
    analysis = compute_analysis(collected_metadata)

    # Send as separate SSE event
    yield {
        "id": "analysis",
        "object": "chat.completion",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": format_collapsed_summary(analysis),
                "metadata": {
                    "type": "hatcat_analysis",
                    "model": "hatcat-analyzer",
                    "stats": analysis["stats"],
                    "top_concepts": analysis["top_concepts"],
                    "sunburst_data": analysis["sunburst"],
                    "steering_suggestions": analysis["suggestions"]
                }
            }
        }]
    }
```

**Collapsed Summary Format:**

```
{percent_divergence}% divergence - {concept1}({count}) - {concept2}({count}) - {concept3}({count})
```

Example: `82% divergence - aisafety(25) - aicatastrophe(23) - AIdeception(12)`

**Expanded View Data:**

```json
{
    "stats": {
        "max": 0.89,
        "max_token": "intelligence",
        "mean": 0.45,
        "min": 0.12,
        "total_tokens": 150
    },
    "top_concepts": [
        {
            "concept": "aisafety",
            "token_count": 25,
            "avg_divergence": 0.78,
            "max_divergence": 0.89,
            "tokens": ["safety", "alignment", ...]
        },
        {
            "concept": "aicatastrophe",
            "token_count": 23,
            "avg_divergence": 0.71,
            "max_divergence": 0.85,
            "tokens": ["risk", "catastrophic", ...]
        }
    ],
    "sunburst_data": {
        "hierarchy": [...],  // Ontology structure
        "highlighted": ["aisafety", "aicatastrophe"],
        "svg": "..."  // Pre-rendered SVG or data for client-side rendering
    },
    "steering_suggestions": [
        {
            "concept": "Proposition",
            "strength": 0.3,
            "reason": "High divergence in logical statements",
            "confidence": 0.85
        }
    ]
}
```

### Phase 2: MCP Tools (Future)

Allow model to query HatCat during generation.

**Tool Definitions:**

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_divergence",
            "description": "Analyze concept divergence in recent tokens",
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n_tokens": {"type": "integer", "default": 50},
                    "concepts": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_steering",
            "description": "Get steering suggestions based on divergence patterns",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_concepts": {"type": "array"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_concept",
            "description": "Get ontology information about a concept",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"}
                }
            }
        }
    }
]
```

## OpenWebUI Integration

### Message Type Extension

```typescript
interface MessageType {
    // ... existing fields ...

    // HatCat-specific
    hatcat_analysis?: {
        collapsed_summary: string;
        stats: {
            max: number;
            max_token: string;
            mean: number;
            min: number;
            total_tokens: number;
        };
        top_concepts: Array<{
            concept: string;
            token_count: number;
            avg_divergence: number;
            tokens: string[];
        }>;
        sunburst_data: any;
        steering_suggestions: Array<{
            concept: string;
            strength: number;
            reason: string;
            confidence: number;
        }>;
    };
}
```

### UI Component

Create `HatCatAnalysis.svelte`:

```svelte
<script>
    export let analysis;
    let expanded = false;
</script>

<div class="hatcat-analysis">
    <button class="collapsed-summary" on:click={() => expanded = !expanded}>
        <span class="icon">🎩</span>
        <span class="summary">{analysis.collapsed_summary}</span>
        <span class="toggle">{expanded ? '▼' : '▶'}</span>
    </button>

    {#if expanded}
        <div class="expanded-view" transition:slide>
            <section>
                <h3>Divergence Statistics</h3>
                <ul>
                    <li>Max: {analysis.stats.max.toFixed(2)} (token: "{analysis.stats.max_token}")</li>
                    <li>Mean: {analysis.stats.mean.toFixed(2)}</li>
                    <li>Min: {analysis.stats.min.toFixed(2)}</li>
                </ul>
            </section>

            <section>
                <h3>Top Divergent Concepts</h3>
                {#each analysis.top_concepts as concept}
                    <div class="concept-card">
                        <strong>{concept.concept}</strong>
                        <span>({concept.token_count} tokens, avg {concept.avg_divergence.toFixed(2)})</span>
                        <div class="tokens">{concept.tokens.slice(0, 5).join(', ')}...</div>
                    </div>
                {/each}
            </section>

            <section>
                <h3>Concept Distribution</h3>
                <SunburstDiagram data={analysis.sunburst_data} />
            </section>

            <section>
                <h3>Steering Suggestions</h3>
                {#each analysis.steering_suggestions as suggestion}
                    <div class="suggestion">
                        <span class="concept">{suggestion.concept}</span>
                        <span class="strength">{suggestion.strength > 0 ? '+' : ''}{suggestion.strength}</span>
                        <span class="reason">{suggestion.reason}</span>
                        <button on:click={() => applySteering(suggestion)}>Apply</button>
                    </div>
                {/each}
            </section>
        </div>
    {/if}
</div>

<style>
    .hatcat-analysis {
        margin: 8px 0;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background: #f9f9f9;
    }

    .collapsed-summary {
        width: 100%;
        padding: 12px;
        background: none;
        border: none;
        text-align: left;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: monospace;
        font-size: 13px;
    }

    .collapsed-summary:hover {
        background: #f0f0f0;
    }

    .icon {
        font-size: 18px;
    }

    .summary {
        flex: 1;
        color: #555;
    }

    .toggle {
        color: #999;
    }

    .expanded-view {
        padding: 16px;
        border-top: 1px solid #e0e0e0;
    }

    section {
        margin-bottom: 20px;
    }

    h3 {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #333;
    }

    .concept-card {
        padding: 8px;
        margin: 4px 0;
        background: white;
        border-radius: 4px;
        border-left: 3px solid #4CAF50;
    }

    .tokens {
        font-size: 11px;
        color: #777;
        margin-top: 4px;
    }

    .suggestion {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px;
        background: white;
        border-radius: 4px;
        margin: 4px 0;
    }

    .suggestion button {
        margin-left: auto;
        padding: 4px 12px;
        background: #2196F3;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
</style>
```

## Implementation Steps

### Step 1: Server-Side Analysis (HatCat Backend)

1. **Collect metadata during generation** ✅ (already done in tokenMetadata)
2. **Compute aggregate statistics** after streaming completes
3. **Identify top divergent concepts** (sort by frequency × avg divergence)
4. **Generate steering suggestions** based on patterns
5. **Format collapsed summary string**
6. **Send analysis as separate message chunk**

### Step 2: OpenWebUI Message Handling

1. **Detect HatCat analysis messages** (check `metadata.type === "hatcat_analysis"`)
2. **Store analysis alongside message**
3. **Render with HatCatAnalysis component** instead of normal message rendering

### Step 3: Sunburst Diagram

1. **Reference concept pack hierarchy** for structure
2. **Highlight concepts with high divergence**
3. **Render as interactive SVG** (d3.js or similar)

### Step 4: Steering Integration

1. **"Apply" button sends steering to `/v1/steering/add`**
2. **Updates active steerings display**
3. **Affects subsequent generations**

## Benefits of This Architecture

✅ **Non-intrusive**: Collapsed by default, doesn't interrupt reading
✅ **Rich detail**: Full analysis available on demand
✅ **Actionable**: Direct steering controls
✅ **Educational**: Sunburst shows concept relationships
✅ **Extensible**: Easy to add more analysis types
✅ **Future-proof**: Prepares for interpretability model (Phase 2)

## Example Collapsed Summaries

```
82% divergence - aisafety(25) - aicatastrophe(23) - AIdeception(12)
45% divergence - Proposition(34) - Process(18) - Relation(15)
91% divergence - EmotionalState(42) - CognitiveAgent(31) - IntentionalProcess(28)
12% divergence - Entity(89) - Abstract(67) - Physical(45)
```

Low divergence = model is aligned with text
High divergence = model's internal representation differs from output

## Future: Interpretability Model

Eventually, the HatCat analysis message could come from a separate model:

```
Monitor Model: Gemma-3-4B (generates response)
     ↓
Interpretability Model: HatCat-Interpreter-7B (analyzes divergence)
     ↓
Analysis Message: Rich insights about what Monitor Model "meant"
```

This interpretability model would be trained on:
- Divergence patterns
- Concept relationships
- Known failure modes
- Steering effects

Making it an expert at explaining and regulating the monitored model.
